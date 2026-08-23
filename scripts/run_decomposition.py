"""Run the clinician-vs-physiology decomposition (results_fusion.md T7).

Part (i): every headline model's cached scores through the subgroup race
(death-first vs treatment-first stays) + per-component AUPRC. Part (ii):
retrain the two flagship families on component-only labels and race those.

    python scripts/run_decomposition.py [--part i,ii] [--seeds 42,1,2]
                                        [--reps 2000] [--device cuda]
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

OUT_CACHE = REPO / "data" / "cache" / "decomposition.npz"
ALL7 = ("vitals_tcn", "news2", "labs", "ecg_meas", "gbm", "cadence", "fluid")


def _cached_models(ctx, seeds):
    """name -> (s_val, s_test) for every headline model, from caches."""
    from fusion_common import load_fair_scores, load_orch_seeds

    out = {"NEWS2": (ctx.news2_val, ctx.news2_test)}
    fv, ft = load_fair_scores(ctx.val_ordered, ctx.test_ordered)
    out["FAIR+TRAJ"] = (fv, ft)
    _, meas, _ = load_orch_seeds(ctx.val_ordered, ctx.test_ordered)
    zg = np.load(REPO / "data/cache/orch_gbm_agent.npz")
    zn = np.load(REPO / "data/cache/new_agents_fusion.npz")
    for s in seeds:
        out[f"ORCH-meas_s{s}"] = (meas[s]["val"], meas[s]["test"])
        out[f"ORCH+GBM_s{s}"] = (zg[f"val_s{s}"].astype(float),
                                 zg[f"test_s{s}"].astype(float))
        out[f"ALL-7_s{s}"] = (zn[f"val_gbm+both_s{s}"].astype(float),
                              zn[f"test_gbm+both_s{s}"].astype(float))
    for agent in ("cadence", "fluid"):
        emb = REPO / "data/cache/embeddings"
        out[agent.upper()] = (
            np.load(emb / f"{agent}_val.npz")["risk"].astype(float),
            np.load(emb / f"{agent}_test.npz")["risk"].astype(float))
    return out


def run_part_i(ctx, seeds, reps):
    from decompose import anticipation_share, component_auprc, component_race

    results = {}
    for name, (sv, st) in _cached_models(ctx, seeds).items():
        t0 = time.time()
        race = component_race(ctx, sv, st, reps=reps)
        results[name] = {"race": race,
                         "auprc": component_auprc(ctx, st),
                         "anticipation_share_2pct": anticipation_share(race)}
        c2 = race[0.02]
        print(f"[{name:<14}] lead@2% composite "
              f"{c2['composite']['median']:+.1f} (n {c2['composite']['n_pair']}) | "
              f"treatment {c2['treatment']['median']:+.1f} "
              f"(n {c2['treatment']['n_pair']}, win "
              f"{100*c2['treatment']['win_rate']:.0f}%) | death "
              f"{c2['death']['median']:+.1f} (n {c2['death']['n_pair']}, win "
              f"{100*c2['death']['win_rate']:.0f}%) | share "
              f"{results[name]['anticipation_share_2pct']:.2f} | AUPRC "
              f"d={results[name]['auprc']['death']:.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    return results


def run_part_ii(ctx, splits, seeds, reps, device):
    """Component-label retrains: FAIR+TRAJ (GPU) + ALL-7 (CPU, label override)."""
    from baselines_fair import build_design, variant_columns
    from decompose import component_race, component_auprc, treatment_only_label
    from distill_lead import (DistilledOrchestrator, fit_gbm_student,
                              gbm_predict, make_soft_frame)
    from modality_features import load_labs
    from orchestrator.fusion import AgentBank

    train_o = splits["train"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    val_o = ctx.val_ordered
    labels = {
        "death": (train_o["label_24h_death"].to_numpy(dtype=float),
                  val_o["label_24h_death"].to_numpy(dtype=float)),
        "treat": (treatment_only_label(train_o), treatment_only_label(val_o)),
    }

    print("[part ii] building FAIR+TRAJ design matrices...")
    labs = load_labs()
    cols = variant_columns()["FAIR+TRAJ"]
    X_tr = build_design(splits["train"], labs, None)[0][cols]
    X_va = build_design(splits["val"], labs, None)[0][cols]
    X_te = build_design(splits["test"], labs, None)[0][cols]
    bank = AgentBank.load(agent_names=ALL7)

    results = {}
    for lab_name, (y_tr, y_va) in labels.items():
        t0 = time.time()
        booster = fit_gbm_student(X_tr, y_tr, np.ones(len(y_tr)),
                                  X_va[ctx.fit_mask], y_va[ctx.fit_mask],
                                  device=device)
        sv, st = gbm_predict(booster, X_va), gbm_predict(booster, X_te)
        results[f"GBM_{lab_name}"] = {
            "race": component_race(ctx, sv, st, reps=reps),
            "auprc": component_auprc(ctx, st)}
        print(f"[GBM_{lab_name}] retrained ({booster.best_iteration + 1} "
              f"trees, {time.time() - t0:.0f}s)", flush=True)

        for seed in seeds:
            import pandas as pd
            t0 = time.time()
            # override val labels too, so early stopping selects on the
            # component target rather than the composite
            soft_all = pd.concat([make_soft_frame(train_o, y_tr),
                                  make_soft_frame(val_o, y_va)])
            student = DistilledOrchestrator(bank=bank, max_epochs=8,
                                            seed=seed, verbose=False,
                                            soft_frame=soft_all)
            student.label_column = ("label_24h_death" if lab_name == "death"
                                    else "label_24h")
            student.fit(splits["train"], splits["val"])
            sv = student.raw_risk(splits["val"])
            st = student.raw_risk(splits["test"])
            results[f"ALL7_{lab_name}_s{seed}"] = {
                "race": component_race(ctx, sv, st, reps=reps),
                "auprc": component_auprc(ctx, st)}
            r = results[f"ALL7_{lab_name}_s{seed}"]["race"][0.02]
            print(f"[ALL7_{lab_name}_s{seed}] death lead@2% "
                  f"{r['death']['median']:+.1f} (n {r['death']['n_pair']}) | "
                  f"treatment {r['treatment']['median']:+.1f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="i,ii")
    ap.add_argument("--seeds", default="42,1,2")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    from baselines_mimic import load_splits
    from fusion_common import FusionContext

    splits = load_splits()
    ctx = FusionContext(splits)
    types = ctx.test_ordered.groupby("stay_id")["event_type"].first()
    print(f"[cohort] test first-event census: "
          f"{types.value_counts(dropna=False).to_dict()}")

    results = {}
    if "i" in args.part.split(","):
        results["part_i"] = run_part_i(ctx, seeds, args.reps)
    if "ii" in args.part.split(","):
        results["part_ii"] = run_part_ii(ctx, splits, seeds, args.reps,
                                         args.device)
    OUT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_CACHE, meta_json=np.array(json.dumps(
        _jsonable({"seeds": seeds, "reps": args.reps, "results": results}))))
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
