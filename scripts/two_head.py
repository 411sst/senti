"""The two-head fix: treatment-trained timing + death-trained ranking (T9).

T7's inversion showed the composite head is a compromise: treatment-only
supervision teaches the best timing anywhere (+3-4h death-stay lead) while
death-only supervision doubles death ranking (AUPRC 0.34) but alarms late.
This script deploys BOTH on the ALL-8 flagship bank:

    T-head  ALL-8 orchestrator trained/early-stopped on vent-or-vaso labels
    D-head  ALL-8 orchestrator trained/early-stopped on label_24h_death
    score   noisy-OR of the V_fit-calibrated component probabilities:
            s = 1 - (1 - cT)(1 - cD)
            -- the probabilistic composite "any event in 24h" under
            component independence (Method Bench 1b's backward-compatible
            recovery), with isotonic plateaus strictified per head.

Graded like every rung: composite AUPRC + paired deltas vs the ALL-8 parent
and FAIR+TRAJ, the paired first-alarm race with the death/treatment subgroup
split, per-component AUPRC (the D-head's death ranking is a deliverable in
its own right), seeds {42, 1, 2, 3, 4} (5-seed hardening 2026-08-27) and
{5, ..., 9} (10-seed consistency check 2026-08-29).

The paired "vs ALL-8" delta needs the single-head ALL-8 parent re-fit at the
same seed (resp_fusion.npz); seeds without a cached parent report that column
as null rather than failing -- the parent is not the claim under test, and its
chain costs ~10 min/seed. ``--out`` lets seed batches run in parallel
processes (one thread each, bit-identity preserved) and be merged afterwards
(scripts/merge_two_head.py).

    python scripts/two_head.py [--seeds 42,1,2] [--reps 2000] [--out PATH]
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

OUT_CACHE = REPO / "data" / "cache" / "two_head.npz"
ALL8 = ("vitals_tcn", "news2", "labs", "ecg_meas", "gbm", "cadence",
        "fluid", "resp")


def fit_head(bank, splits, y_train, y_val, train_o, val_o, seed, epochs):
    """One component head: label-overridden orchestrator on the ALL-8 bank
    (train AND val overridden, so early stopping selects on the component)."""
    from distill_lead import DistilledOrchestrator, make_soft_frame

    soft = pd.concat([make_soft_frame(train_o, y_train),
                      make_soft_frame(val_o, y_val)])
    head = DistilledOrchestrator(bank=bank, max_epochs=epochs, seed=seed,
                                 verbose=False, soft_frame=soft)
    head._fit_model(splits["train"], splits["val"])   # no composite calibrator
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--auprc-reps", type=int, default=1000)
    ap.add_argument("--out", default=str(OUT_CACHE),
                    help="cache path (default data/cache/two_head.npz)")
    args = ap.parse_args()
    out_path = Path(args.out)
    seeds = [int(s) for s in args.seeds.split(",")]

    from baselines_mimic import load_splits
    from decompose import component_auprc, component_race, treatment_only_label
    from fusion_common import (FusionContext, delta_ci, load_fair_scores,
                               paired_auprc_draws)
    from fusion_ladder import _strictify
    from metrics import auprc
    from orchestrator.fusion import AgentBank
    from sklearn.isotonic import IsotonicRegression

    splits = load_splits()
    ctx = FusionContext(splits)
    train_o = splits["train"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    val_o = ctx.val_ordered
    y_death = {"train": train_o["label_24h_death"].to_numpy(dtype=float),
               "val": val_o["label_24h_death"].to_numpy(dtype=float)}
    y_treat = {"train": treatment_only_label(train_o),
               "val": treatment_only_label(val_o)}
    y_death_test = ctx.test_ordered["label_24h_death"].to_numpy(dtype=float)

    _, fair_test = load_fair_scores(ctx.val_ordered, ctx.test_ordered)
    zp = np.load(REPO / "data" / "cache" / "resp_fusion.npz")
    bank = AgentBank.load(agent_names=ALL8)
    print(f"[bank] ALL-8: {bank.n_features} features")

    results, payload = {}, {
        "stay_id_test": ctx.test_ordered["stay_id"].to_numpy(),
        "hour_test": ctx.test_ordered["hour"].to_numpy(),
    }
    for seed in seeds:
        t0 = time.time()
        heads = {}
        for name, y in (("T", y_treat), ("D", y_death)):
            head = fit_head(bank, splits, y["train"], y["val"],
                            train_o, val_o, seed, args.epochs)
            raw_val = head.raw_risk(splits["val"])
            raw_test = head.raw_risk(splits["test"])
            iso = IsotonicRegression(out_of_bounds="clip").fit(
                raw_val[ctx.fit_mask], y["val"][ctx.fit_mask])
            heads[name] = {
                "val": _strictify(iso.transform(raw_val), raw_val),
                "test": _strictify(iso.transform(raw_test), raw_test),
            }
        s_val = 1.0 - (1.0 - heads["T"]["val"]) * (1.0 - heads["D"]["val"])
        s_test = 1.0 - (1.0 - heads["T"]["test"]) * (1.0 - heads["D"]["test"])

        r = ctx.evaluate_rung(s_val, s_test, reps=args.reps)
        r["race"] = component_race(ctx, s_val, s_test, reps=args.reps)
        r["auprc_components"] = component_auprc(ctx, s_test)
        r["d_head_death_auprc"] = float(auprc(y_death_test,
                                              heads["D"]["test"]))
        parent_key = f"test_all7+resp_s{seed}"
        scored = {"two": s_test, "fair": fair_test}
        if parent_key in zp.files:
            scored["parent"] = zp[parent_key].astype(float)
        draws = paired_auprc_draws(ctx.y_test, scored, ctx.patient_test,
                                   reps=args.auprc_reps)
        r["delta_vs_all8_ci"] = (delta_ci(draws["two"], draws["parent"])
                                 if "parent" in draws else None)
        r["delta_vs_fair_ci"] = delta_ci(draws["two"], draws["fair"])
        results[seed] = r
        # float64, NOT float32: these scores carry the _strictify rank
        # tiebreak (1e-9), and float32's resolution near 0.1 is 7.5e-9 --
        # casting down silently restores the isotonic plateaus and makes the
        # cached scores differ from the ones actually evaluated here.
        payload[f"val_s{seed}"] = s_val
        payload[f"test_s{seed}"] = s_test
        payload[f"dhead_test_s{seed}"] = heads["D"]["test"]

        b = r["budgets"][0.02]
        c = r["race"][0.02]
        vs_all8 = (f"[{r['delta_vs_all8_ci'][0]:+.4f},"
                   f"{r['delta_vs_all8_ci'][1]:+.4f}]"
                   if r["delta_vs_all8_ci"] is not None else "no cached parent")
        print(f"[seed {seed}] TWO-HEAD test AUPRC {r['auprc']:.4f} "
              f"(vs ALL-8 {vs_all8}; vs FAIR "
              f"[{r['delta_vs_fair_ci'][0]:+.4f},"
              f"{r['delta_vs_fair_ci'][1]:+.4f}]) | death AUPRC "
              f"{r['auprc_components']['death']:.3f} (D-head "
              f"{r['d_head_death_auprc']:.3f}) | lead@2% {b['median']:+.1f} "
              f"(win {100*b['win_rate']:.0f}%) | death-stay lead "
              f"{c['death']['median']:+.1f} (win "
              f"{100*c['death']['win_rate']:.0f}%)  "
              f"[{time.time() - t0:.0f}s]", flush=True)

    payload["meta_json"] = np.array(json.dumps(_jsonable(
        {"seeds": seeds, "reps": args.reps, "agents": list(ALL8),
         "results": results})))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"[cache] -> {out_path}")


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
