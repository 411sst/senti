"""Fusion gates for the cadence/fluid agents, in both parent architectures.

For each config the orchestrator is re-fit at seeds {42, 1, 2} and compared
WITHIN seed to its cached parent (results_ecg.md methodology -- seed noise
exceeds typical fusion deltas):

    core-3+meas      + cadence | fluid | both     vs data/cache/orch_meas_seeds.npz
    core-3+meas+gbm  + cadence | fluid | both     vs data/cache/orch_gbm_agent.npz

Reported per cell: test AUPRC, paired ΔAUPRC vs the matching parent (patient-
clustered), and the paired first-alarm block at 1/2/5% budgets (fusion_common
protocol). The ship bar is unchanged: positive val-side ΔAUPRC sign-consistent
across seeds, or a lead improvement with win-rate CIs clear of 50% -- stated
per cell, decided by the team.

    python scripts/eval_new_agents.py [--agents cadence,fluid,both]
                                      [--parents meas,gbm] [--seeds 42,1,2]
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

OUT_CACHE = REPO / "data" / "cache" / "new_agents_fusion.npz"
PARENTS = {
    "meas": ("vitals_tcn", "news2", "labs", "ecg_meas"),
    "gbm": ("vitals_tcn", "news2", "labs", "ecg_meas", "gbm"),
    "all7": ("vitals_tcn", "news2", "labs", "ecg_meas", "gbm",
             "cadence", "fluid"),
    "all8": ("vitals_tcn", "news2", "labs", "ecg_meas", "gbm",
             "cadence", "fluid", "resp"),
}
ADDITIONS = {
    "cadence": ("cadence",),
    "fluid": ("fluid",),
    "both": ("cadence", "fluid"),
    "resp": ("resp",),
    "respcap4": ("resp_cap4",),
    "neuro": ("neuro_status",),
    "infect": ("infection",),
    "both2": ("neuro_status", "infection"),
}


def parent_scores(name, ctx):
    """Per-seed raw parent scores (val, test) from the cached runs."""
    if name == "meas":
        from fusion_common import load_orch_seeds
        seeds, sc, _ = load_orch_seeds(ctx.val_ordered, ctx.test_ordered)
        return {s: (sc[s]["val"], sc[s]["test"]) for s in seeds}
    if name == "all7":
        z = np.load(REPO / "data" / "cache" / "new_agents_fusion.npz",
                    allow_pickle=True)
        meta = json.loads(str(z["meta_json"]))
        return {s: (z[f"val_gbm+both_s{s}"].astype(float),
                    z[f"test_gbm+both_s{s}"].astype(float))
                for s in meta["seeds"]}
    if name == "all8":
        z = np.load(REPO / "data" / "cache" / "resp_fusion.npz",
                    allow_pickle=True)
        meta = json.loads(str(z["meta_json"]))
        return {s: (z[f"val_all7+resp_s{s}"].astype(float),
                    z[f"test_all7+resp_s{s}"].astype(float))
                for s in meta["seeds"]}
    z = np.load(REPO / "data" / "cache" / "orch_gbm_agent.npz",
                allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    return {s: (z[f"val_s{s}"].astype(float), z[f"test_s{s}"].astype(float))
            for s in meta["seeds"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="cadence,fluid,both")
    ap.add_argument("--parents", default="meas,gbm")
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--auprc-reps", type=int, default=1000)
    ap.add_argument("--out", default=str(OUT_CACHE),
                    help="output npz (set a NEW path when the parent's own "
                         "cache is this script's previous output)")
    args = ap.parse_args()
    out_cache = Path(args.out)
    seeds = [int(s) for s in args.seeds.split(",")]

    from baselines_mimic import load_splits
    from fusion_common import FusionContext, delta_ci, paired_auprc_draws
    from metrics import auprc
    from orchestrator.fusion import AgentBank, Orchestrator

    splits = load_splits()
    ctx = FusionContext(splits)
    results, payload = {}, {
        "stay_id_test": ctx.test_ordered["stay_id"].to_numpy(),
        "hour_test": ctx.test_ordered["hour"].to_numpy(),
    }
    for parent_name in args.parents.split(","):
        par = parent_scores(parent_name, ctx)
        for add_name in args.agents.split(","):
            agents = PARENTS[parent_name] + ADDITIONS[add_name]
            cfg = f"{parent_name}+{add_name}"
            bank = AgentBank.load(agent_names=agents)
            print(f"[{cfg}] {bank.n_features} features from {list(agents)}")
            for seed in seeds:
                t0 = time.time()
                orch = Orchestrator(bank=bank, max_epochs=args.epochs,
                                    seed=seed, verbose=False)
                orch.fit(splits["train"], splits["val"])
                rv = orch.raw_risk(splits["val"])
                rt = orch.raw_risk(splits["test"])
                r = ctx.evaluate_rung(rv, rt, reps=args.reps)
                pv, pt = par[seed]
                r["val_auprc"] = float(auprc(ctx.y_val, rv))
                r["parent_val_auprc"] = float(auprc(ctx.y_val, pv))
                draws = paired_auprc_draws(
                    ctx.y_test, {"new": rt, "parent": pt},
                    ctx.patient_test, reps=args.auprc_reps)
                r["delta_vs_parent_ci"] = delta_ci(draws["new"],
                                                   draws["parent"])
                results[f"{cfg}_s{seed}"] = r
                payload[f"val_{cfg}_s{seed}"] = rv.astype(np.float32)
                payload[f"test_{cfg}_s{seed}"] = rt.astype(np.float32)
                b = r["budgets"][0.02]
                print(f"  [seed {seed}] test AUPRC {r['auprc']:.4f} "
                      f"(val Δ vs parent "
                      f"{r['val_auprc'] - r['parent_val_auprc']:+.4f}; "
                      f"test Δ CI [{r['delta_vs_parent_ci'][0]:+.4f}, "
                      f"{r['delta_vs_parent_ci'][1]:+.4f}])  lead@2% "
                      f"{b['median']:+.1f} (win {100*b['win_rate']:.0f}%, "
                      f"n {b['n_pair']})  [{time.time() - t0:.0f}s]")

    payload["meta_json"] = np.array(json.dumps(_jsonable(
        {"seeds": seeds, "reps": args.reps, "results": results})))
    np.savez_compressed(out_cache, **payload)
    print(f"[cache] -> {out_cache}")


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
