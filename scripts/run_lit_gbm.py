"""Race the literature-anchored GBMs against NEWS2 and the flagship (T10, part 1).

src/lit_gbm.py fits three published GBM specifications on our cohort; this
scores them through the same harness every other row of the results table went
through, so the comparison is protocol-identical rather than merely adjacent.

Two report modes, kept apart on purpose:

  our protocol     our cohort, our label, our splits, thresholds frozen on
                   V_freeze. This is the row that belongs beside FAIR+TRAJ and
                   the flagship.
  author protocol  Sadanandan 2026 samples hours 6-48 of stays lasting at least
                   24h. Re-scoring our model on that row subset isolates how
                   much of the gap to their reported 0.7486 / 0.1123 is the
                   evaluation population rather than the model. It is NOT a
                   retrain on their cohort, and is labelled as such.

    python scripts/run_lit_gbm.py [--reps 2000]
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

OUT_JSON = REPO / "data" / "cache" / "lit_gbm_report.json"
RAW_PATH = REPO / "data" / "cache" / "mimic_raw_hourly.parquet"

# Sadanandan 2026's sampling window and minimum stay length.
AUTHOR_HOURS = (6, 48)
AUTHOR_MIN_LOS_HOURS = 24


def gather_models(ctx):
    """Every comparator on the board, as name -> (raw val, raw test)."""
    from fusion_common import load_fair_scores
    from lit_gbm import load_lit_scores

    models = {"NEWS2": (ctx.news2_val, ctx.news2_test)}
    fv, ft = load_fair_scores(ctx.val_ordered, ctx.test_ordered)
    models["FAIR+TRAJ (ours)"] = (fv, ft)

    th = np.load(REPO / "data/cache/two_head.npz")
    if (th["stay_id_test"] == ctx.test_ordered["stay_id"].to_numpy()).all():
        models["two-head flagship s42"] = (th["val_s42"].astype(float),
                                           th["test_s42"].astype(float))
    for name, pair in load_lit_scores(ctx.val_ordered, ctx.test_ordered).items():
        models[f"lit:{name}"] = pair
    return models


def author_protocol_mask(test_ordered):
    """Rows matching Sadanandan's sampling: hours 6-48 of stays lasting >=24h.

    Stay length comes from the RAW hourly grid, not from the cohort: the cohort
    is censored at the first event, so a long stay that deteriorated at hour 10
    has only 10 observed rows and would be wrongly excluded as short.
    """
    raw = pd.read_parquet(RAW_PATH, columns=["stay_id", "hour"])
    raw["stay_id"] = raw["stay_id"].astype("int64")
    los = raw.groupby("stay_id")["hour"].max() + 1
    long_enough = set(los[los >= AUTHOR_MIN_LOS_HOURS].index)

    hours = test_ordered["hour"].to_numpy()
    stays = test_ordered["stay_id"].to_numpy()
    return (np.isin(stays, list(long_enough))
            & (hours >= AUTHOR_HOURS[0]) & (hours <= AUTHOR_HOURS[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    from baselines_mimic import load_splits
    from fusion_common import FusionContext, delta_ci, paired_auprc_draws
    from lit_gbm import SCORE_CACHE
    from metrics import auprc, auroc

    splits = load_splits()
    ctx = FusionContext(splits)
    models = gather_models(ctx)
    records = json.loads(str(np.load(SCORE_CACHE, allow_pickle=False)["records_json"]))

    print("=" * 84)
    print("T10 -- literature-anchored GBM comparators, OUR protocol")
    print("=" * 84)
    print(f"test: {len(ctx.y_test):,} hours | "
          f"{ctx.test_ordered.stay_id.nunique():,} stays | "
          f"prevalence {ctx.y_test.mean():.2%}\n")

    rows = {}
    for name, (s_val, s_test) in models.items():
        res = ctx.evaluate_rung(s_val, s_test, reps=args.reps)
        b = res["budgets"][0.02]
        rows[name] = {
            "auprc": res["auprc"],
            "auroc": float(auroc(ctx.y_test, s_test)),
            "median_lead_h": b["median"], "win_rate": b["win_rate"],
            "n_pair": b["n_pair"], "model_alarm_rate": b["model_rate"],
        }

    hdr = (f"{'model':<26}{'AUPRC':>8}{'AUROC':>8}{'lead(h)':>9}"
           f"{'win':>7}{'n_pair':>8}{'alarm%':>8}")
    print(hdr); print("-" * len(hdr))
    for name, r in rows.items():
        print(f"{name:<26}{r['auprc']:>8.3f}{r['auroc']:>8.3f}"
              f"{r['median_lead_h']:>9.1f}{r['win_rate']:>7.0%}"
              f"{r['n_pair']:>8}{100*r['model_alarm_rate']:>8.2f}")
    print("\nlead / win / n_pair are the paired first-alarm race vs NEWS2 at a "
          "2% alarm budget,\nthresholds frozen on V_freeze, jointly-caught stays only.")

    # Paired AUPRC deltas against the two in-house comparators.
    print("\n--- paired AUPRC deltas (patient-clustered, 95% CI) ---")
    score_map = {k: v[1] for k, v in models.items()}
    d = paired_auprc_draws(ctx.y_test, score_map, ctx.patient_test,
                           reps=args.reps)
    for ref in ("FAIR+TRAJ (ours)", "two-head flagship s42"):
        if ref not in d:
            continue
        print(f"\n  vs {ref}:")
        for name in score_map:
            if name == ref:
                continue
            lo, hi = delta_ci(d[ref], d[name])
            gap = float(np.mean(d[ref] - d[name]))
            verdict = "ours higher" if lo > 0 else ("theirs higher" if hi < 0
                                                    else "not separated")
            print(f"    {name:<26} delta {gap:+.3f} [{lo:+.3f}, {hi:+.3f}]  {verdict}")

    # Author protocol for the one spec whose label is ours.
    print("\n" + "=" * 84)
    print("Sadanandan 2026, AUTHOR evaluation window (hours 6-48, stays >=24h)")
    print("=" * 84)
    mask = author_protocol_mask(ctx.test_ordered)
    ya = ctx.y_test[mask]
    print(f"  {mask.sum():,} of {len(mask):,} test hours | "
          f"{pd.unique(ctx.test_ordered.stay_id.to_numpy()[mask]).size:,} stays | "
          f"prevalence {ya.mean():.2%}  (they report 2.8%)")
    author = {}
    for name in ("lit:sadanandan", "FAIR+TRAJ (ours)", "NEWS2"):
        if name not in models:
            continue
        s = models[name][1][mask]
        author[name] = {"auprc": float(auprc(ya, s)), "auroc": float(auroc(ya, s))}
        print(f"  {name:<26} AUPRC {author[name]['auprc']:.4f} | "
              f"AUROC {author[name]['auroc']:.4f}")
    rep = records.get("sadanandan", {}).get("provenance", {}).get("reported", {})
    if rep:
        print(f"  {'as published (their cohort)':<26} AUPRC {rep['auprc']:.4f} | "
              f"AUROC {rep['auroc']:.4f}")
    print("  NOTE: our model re-scored on their row window -- not a retrain on "
          "their cohort.\n  Their stays are also not censored at the first event, "
          "so the populations still differ.")

    OUT_JSON.write_text(json.dumps(
        {"our_protocol": rows, "author_protocol": author, "specs": records},
        indent=2, default=float))
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
