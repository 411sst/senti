"""FAIR+AGENTS: integrate the agents INTO the GBM — keep ranking, gain lead?

The endgame experiment of the fusion program. Score-level fusion (T1) showed
the tradeoff is real at the score level; objective shaping (T2) showed the
GBM cannot be *taught* earliness through labels or weights. This module hands
the GBM the agents themselves, as FEATURES:

  +RISKS      FAIR+TRAJ (145) + each agent's calibrated risk at t (4)
  +RISKTRAJ   + trajectory features OF the agent risks: delta / trailing
              min / max / mean over 6/12/24h per agent (48) — the tree-model
              encoding of "the vitals-TCN risk has been rising", i.e. exactly
              the early-rise signal that carries the orchestrator's lead
  +LATENT     + the raw agent latent columns at t (103)
  +ORCH       + the ORCHESTRATOR's risk and its trajectory (per seed) —
              train rows use the OUT-OF-FOLD orchestrator cache so the GBM
              never sees in-sample-sharp teacher scores

Alignment note: every agent emits on the identical sorted (stay_id, hour)
grid as build_design's rows (asserted). Agent latents/risks at hour t are
strictly-before-t by the agents' own leakage guarantees. vitals_tcn / labs /
ecg_meas train-split emissions are in-sample for those agents (they trained
on train) — same structure the orchestrator itself consumes, documented
rather than fixed; the orchestrator-risk columns DO get the OOF treatment
because the cache exists.

    python src/gbm_agents.py [--variants risks,risktraj,latent,orch]
                             [--seeds 42,1,2] [--reps 2000] [--device cuda]
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from baselines_fair import build_design, tune_gbm, variant_columns
from baselines_mimic import load_splits
from config import SEED
from fusion_common import (
    FusionContext,
    delta_ci,
    load_fair_scores,
    paired_auprc_draws,
)
from metrics import auprc
from modality_features import load_labs

LABEL = "label_24h"
AGENTS = ("vitals_tcn", "news2", "labs", "ecg_meas")
TRAJ_WINDOWS = (6, 12, 24)
OUT_CACHE = "data/cache/gbm_agents_scores.npz"
EMB = "data/cache/embeddings"


def _assert_aligned(ordered, npz, path):
    if not ((npz["stay_id"] == ordered["stay_id"].to_numpy()).all()
            and (npz["hour"] == ordered["hour"].to_numpy()).all()):
        raise ValueError(f"{path} does not align with the cohort rows")


def risk_trajectory(ordered, name, risk):
    """delta / trailing min / max / mean of one risk series, per stay.

    Same before-t guarantee as ``vitals_trajectory_features``: any function of
    current-and-earlier rows of a strictly-before-t series is strictly
    before t; grouping by stay stops cross-stay bleed.
    """
    out = {}
    v = pd.Series(np.asarray(risk, dtype=np.float32))
    stay = ordered["stay_id"].to_numpy()
    by_stay = v.groupby(stay, sort=False)
    for w in TRAJ_WINDOWS:
        out[f"{name}_delta_{w}h"] = (v - by_stay.shift(w)).to_numpy(np.float32)
        roll = v.groupby(stay, sort=False).rolling(w, min_periods=1)
        for stat in ("min", "max", "mean"):
            agg = getattr(roll, stat)().droplevel(0).sort_index()
            out[f"{name}_{stat}_{w}h"] = agg.to_numpy(np.float32)
    return out


def agent_blocks(ordered, split, with_latent):
    """Risk + risk-trajectory (+ latent) columns for one split."""
    blocks = {}
    for agent in AGENTS:
        path = f"{EMB}/{agent}_{split}.npz"
        z = np.load(path)
        _assert_aligned(ordered, z, path)
        risk = z["risk"].astype(np.float32)
        blocks[f"{agent}_risk"] = risk
        blocks.update(risk_trajectory(ordered, f"{agent}_risk", risk))
        if with_latent:
            lat = z["latent"].astype(np.float32)
            for j in range(lat.shape[1]):
                blocks[f"{agent}_z{j}"] = lat[:, j]
    return blocks


def orch_blocks(ordered, split, seed):
    """Orchestrator risk + trajectory; train rows come from the OOF cache."""
    suffix = "trainoof" if split == "train" else split
    path = f"{EMB}/orchestrator-meas-s{seed}_{suffix}.npz"
    z = np.load(path)
    _assert_aligned(ordered, z, path)
    risk = z["risk"].astype(np.float32)
    blocks = {"orch_risk": risk}
    blocks.update(risk_trajectory(ordered, "orch_risk", risk))
    return blocks


VARIANT_SPEC = {
    "risks": dict(traj=False, latent=False, orch=False),
    "risktraj": dict(traj=True, latent=False, orch=False),
    "latent": dict(traj=True, latent=True, orch=False),
    "orch": dict(traj=True, latent=False, orch=True),
    # FAIR+TRAJ + OOF orchestrator risk/trajectory ONLY. The per-agent train
    # emissions are in-sample (agents trained on train) and measurably hurt
    # the GBM (+RISKS/+RISKTRAJ < FAIR+TRAJ); the fused OOF risk carries the
    # agents' signal without that contamination.
    "orchonly": dict(traj=False, latent=False, orch=True, agents=False),
}


class DesignStore:
    """Base FAIR+TRAJ frames + the full agent blocks, built ONCE.

    Variant assembly is then a cheap column selection -- rebuilding the design
    matrices per cell would dominate the runtime.
    """

    def __init__(self, splits):
        labs = load_labs()
        base_cols = variant_columns()["FAIR+TRAJ"]
        self.base, self.ordered, self.agents = {}, {}, {}
        for split_name in ("train", "val", "test"):
            df = splits[split_name]
            self.ordered[split_name] = df.sort_values(
                ["stay_id", "hour"]).reset_index(drop=True)
            self.base[split_name] = build_design(df, labs, None)[0][base_cols]
            self.agents[split_name] = agent_blocks(
                self.ordered[split_name], split_name, with_latent=True)

    def variant(self, variant, seed=None):
        """(X_tr, X_va, X_te, cols) for one variant spec."""
        spec = VARIANT_SPEC[variant]
        frames = {}
        for split_name in ("train", "val", "test"):
            X = self.base[split_name].copy()
            if spec.get("agents", True):
                for k, v in self.agents[split_name].items():
                    is_latent = "_z" in k
                    is_traj = not (k.endswith("_risk") or is_latent)
                    if is_latent and not spec["latent"]:
                        continue
                    if is_traj and not spec["traj"]:
                        continue
                    X[k] = v
            if spec["orch"]:
                for k, v in orch_blocks(self.ordered[split_name],
                                        split_name, seed).items():
                    X[k] = v
            frames[split_name] = X
        cols = list(frames["train"].columns)
        return frames["train"], frames["val"], frames["test"], cols


def run(variants=("risks", "risktraj", "latent", "orch"), seeds=(42, 1, 2),
        reps=2000, auprc_reps=1000, device="cuda", quick=False,
        out_cache=OUT_CACHE):
    splits = load_splits()
    ctx = FusionContext(splits)
    fair_val, fair_test = load_fair_scores(ctx.val_ordered, ctx.test_ordered)
    y_tr = splits["train"].sort_values(["stay_id", "hour"])[LABEL] \
        .to_numpy(dtype=float)
    y_va = ctx.y_val
    seeds = list(seeds)[:1] if quick else list(seeds)

    fair_eval = ctx.evaluate_rung(fair_val, fair_test, reps=reps)
    results = {"FAIR+TRAJ": {"eval": fair_eval, "n_features": 145}}
    scores_for_draws = {"FAIR+TRAJ": fair_test}
    payload = {"stay_id_test": ctx.test_ordered["stay_id"].to_numpy(),
               "hour_test": ctx.test_ordered["hour"].to_numpy()}

    print("[design] building base + agent blocks once...")
    store = DesignStore(splits)
    cells = []
    for v in variants:
        if VARIANT_SPEC[v]["orch"]:
            cells += [(v, s) for s in seeds]   # per orchestrator seed
        else:
            cells.append((v, None))            # agent caches are seed-free
    for variant, seed in cells:
        name = f"+{variant.upper()}" + (f"_s{seed}" if seed else "")
        t0 = time.time()
        X_tr, X_va, X_te, cols = store.variant(variant, seed)
        print(f"[{name}] {len(cols)} features; tuning on val...")
        model, record = tune_gbm(X_tr, y_tr, X_va, y_va, quick=quick,
                                 verbose=False, device=device)
        s_val = model.predict_proba(X_va)[:, 1].astype(float)
        s_test = model.predict_proba(X_te)[:, 1].astype(float)
        r = ctx.evaluate_rung(s_val, s_test, reps=reps)
        results[name] = {"eval": r, "record": record,
                         "n_features": len(cols)}
        scores_for_draws[name] = s_test
        payload[f"val_{name}"] = s_val.astype(np.float32)
        payload[f"test_{name}"] = s_test.astype(np.float32)
        b = r["budgets"][0.02]
        print(f"[{name}] test AUPRC {r['auprc']:.4f}  lead@2% "
              f"{b['median']:+.1f} (win {100*b['win_rate']:.0f}%, "
              f"n {b['n_pair']})  [{time.time() - t0:.0f}s]  {record}")
        del X_tr, X_va, X_te

    draws = paired_auprc_draws(ctx.y_test, scores_for_draws,
                               ctx.patient_test, reps=auprc_reps)
    print(f"\n=== FAIR+AGENTS (test; paired ΔAUPRC vs FAIR+TRAJ) ===")
    for name, r in results.items():
        d = ("--" if name == "FAIR+TRAJ"
             else "[{:+.4f}, {:+.4f}]".format(
                 *delta_ci(draws[name], draws["FAIR+TRAJ"])))
        line = f"  {name:<16} AUPRC {r['eval']['auprc']:.4f}  Δ {d:<22}"
        for budget in (0.01, 0.02, 0.05):
            b = r["eval"]["budgets"][budget]
            line += (f"  lead@{int(budget*100)}% {b['median']:+.1f} "
                     f"(win {100*b['win_rate']:.0f}%)")
        results[name]["delta_vs_fair_ci"] = (
            None if name == "FAIR+TRAJ"
            else delta_ci(draws[name], draws["FAIR+TRAJ"]))
        print(line)

    payload["meta_json"] = np.array(json.dumps(_jsonable(
        {"variants": list(variants), "seeds": seeds, "reps": reps,
         "results": results})))
    Path(out_cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_cache, **payload)
    print(f"\n[cache] -> {out_cache}")
    return results


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="risks,risktraj,latent,orch")
    ap.add_argument("--seeds", default="42,1,2")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--auprc-reps", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(variants=tuple(args.variants.split(",")),
        seeds=[int(s) for s in args.seeds.split(",")],
        reps=args.reps, auprc_reps=args.auprc_reps,
        device=args.device, quick=args.quick)
