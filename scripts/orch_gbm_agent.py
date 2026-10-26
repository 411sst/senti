"""GBM as a fifth agent inside the orchestrator.

Direction A of the bridge program (results_fusion.md): raise OUR
architecture's accuracy without giving up its lead. The orchestrator's
attention already arbitrates per-hour between agents; this hands it the fair
GBM's ranking signal as one more agent and lets the causal TCN keep doing the
timing.

The pseudo-agent emission cache ``gbm_{split}.npz`` follows the standard
contract (latent = [logit(raw)], 1-dim; risk = isotonic-on-val), with one
upgrade: TRAIN rows are OUT-OF-FOLD (patient-level 2-fold, fold rng 42 shared
with every other OOF artifact, each fold model early-stopped on val) — the
orchestrator must never learn to trust in-sample-sharp GBM scores it will not
get at inference.

    python scripts/orch_gbm_agent.py [--seeds 42,1,2] [--reps 2000]
                                     [--device cuda] [--rebuild-cache]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

EMB = REPO / "data/cache/embeddings"
OUT_CACHE = REPO / "data/cache/orch_gbm_agent.npz"
AGENTS = ("vitals_tcn", "news2", "labs", "ecg_meas", "gbm")


def build_gbm_agent_cache(splits, device="cuda"):
    from agents.base import AgentOutput
    from baselines_fair import SCORE_CACHE, build_design, variant_columns
    from distill_lead import fit_gbm_student, gbm_predict
    from modality_features import load_labs
    from sklearn.isotonic import IsotonicRegression

    z = np.load(SCORE_CACHE, allow_pickle=True)
    val_ordered = splits["val"].sort_values(["stay_id", "hour"]) \
        .reset_index(drop=True)
    y_val = val_ordered["label_24h"].to_numpy(dtype=float)
    raw = {"val": z["rawval_FAIR+TRAJ"].astype(float),
           "test": z["raw_FAIR+TRAJ"].astype(float)}
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw["val"], y_val)

    print("[gbm-agent] building design matrices for the OOF train scores...")
    labs = load_labs()
    cols = variant_columns()["FAIR+TRAJ"]
    X_tr = build_design(splits["train"], labs, None)[0][cols]
    X_va = build_design(splits["val"], labs, None)[0][cols]
    train_ordered = splits["train"].sort_values(["stay_id", "hour"]) \
        .reset_index(drop=True)
    y_tr = train_ordered["label_24h"].to_numpy(dtype=float)

    patients = np.sort(train_ordered["patient_id"].unique())
    perm = np.random.default_rng(42).permutation(patients)
    half = len(patients) // 2
    pid = train_ordered["patient_id"].to_numpy()
    oof = np.full(len(train_ordered), np.nan)
    for hold in (set(perm[:half]), set(perm[half:])):
        mask = np.isin(pid, list(hold))
        booster = fit_gbm_student(X_tr[~mask], y_tr[~mask],
                                  np.ones(int((~mask).sum())),
                                  X_va, y_val, device=device)
        oof[mask] = gbm_predict(booster, X_tr[mask])
    assert not np.isnan(oof).any()
    raw["train"] = oof

    for split_name, df in splits.items():
        ordered = df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
        r = raw[split_name]
        logit = np.log(np.clip(r, 1e-6, 1 - 1e-6)
                       / (1 - np.clip(r, 1e-6, 1 - 1e-6)))
        out = AgentOutput(
            keys=ordered[["stay_id", "hour"]].copy(),
            latent=logit.astype(np.float32)[:, None],
            risk=iso.transform(r).astype(np.float32))
        out.save(EMB / f"gbm_{split_name}.npz")
    print(f"[gbm-agent] cached gbm_{{train,val,test}}.npz "
          f"(train = OOF, val/test = cached winner)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--auprc-reps", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    from baselines_mimic import load_splits
    from fusion_common import (FusionContext, delta_ci, load_orch_seeds,
                               paired_auprc_draws)
    from orchestrator.fusion import AgentBank, Orchestrator

    splits = load_splits()
    if args.rebuild_cache or not (EMB / "gbm_train.npz").exists():
        build_gbm_agent_cache(splits, device=args.device)

    ctx = FusionContext(splits)
    _, parent, _ = load_orch_seeds(ctx.val_ordered, ctx.test_ordered)
    bank = AgentBank.load(agent_names=AGENTS)
    print(f"[bank] {bank.n_features} features from {list(AGENTS)}")

    results, payload = {}, {
        "stay_id_test": ctx.test_ordered["stay_id"].to_numpy(),
        "hour_test": ctx.test_ordered["hour"].to_numpy(),
    }
    for seed in seeds:
        t0 = time.time()
        orch = Orchestrator(bank=bank, max_epochs=args.epochs, seed=seed,
                            verbose=False)
        orch.fit(splits["train"], splits["val"])
        orch.save_checkpoint(REPO / f"models/orchestrator/measgbm_s{seed}.pt")
        rv = orch.raw_risk(splits["val"])
        rt = orch.raw_risk(splits["test"])
        r = ctx.evaluate_rung(rv, rt, reps=args.reps)
        _, attn = orch.attention_weights(splits["val"])
        r["mean_attention"] = {a: float(attn[:, i].mean())
                               for i, a in enumerate(AGENTS)}
        draws = paired_auprc_draws(
            ctx.y_test,
            {"new": rt, "parent": parent[seed]["test"]},
            ctx.patient_test, reps=args.auprc_reps)
        r["delta_vs_parent_ci"] = delta_ci(draws["new"], draws["parent"])
        results[seed] = r
        payload[f"val_s{seed}"] = rv.astype(np.float32)
        payload[f"test_s{seed}"] = rt.astype(np.float32)
        b = r["budgets"][0.02]
        print(f"[seed {seed}] +gbm-agent test AUPRC {r['auprc']:.4f} "
              f"(Δ vs parent [{r['delta_vs_parent_ci'][0]:+.4f}, "
              f"{r['delta_vs_parent_ci'][1]:+.4f}])  lead@2% "
              f"{b['median']:+.1f} (win {100*b['win_rate']:.0f}%, "
              f"n {b['n_pair']})  attn {r['mean_attention']}  "
              f"[{time.time() - t0:.0f}s]")

    payload["meta_json"] = np.array(json.dumps(_jsonable(
        {"agents": list(AGENTS), "seeds": seeds, "reps": args.reps,
         "results": results})))
    np.savez_compressed(OUT_CACHE, **payload)
    print(f"[cache] -> {OUT_CACHE}")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


if __name__ == "__main__":
    main()
