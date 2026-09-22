"""Seed-stability of the eICU margin over the published specs.

    python scripts/eicu_seed_ci.py [--seeds 42 1 2]

T22 reported the comparative result at seed 42 alone, while the MIMIC claims
carry ten-seed evidence. The margins there are small (+0.011 to +0.018 AUPRC),
so the question that decides whether this is a headline or a finding is whether
they survive a change of fusion seed.

Seeds vary the FUSION only, following T9/T18: the agent bank is fitted once at
seed 42 and shared, so a seed is the two head fits and nothing else. The
comparators stay fixed -- they are published specs, not something we re-roll.

Every model goes through ONE bootstrap pass with the identical patient-level
resample, so the seed-to-seed and model-to-model deltas are all paired against
each other rather than computed from independent draws.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fusion_common import delta_ci, paired_auprc_draws  # noqa: E402
from metrics import auprc, auroc  # noqa: E402

LABEL = "label_24h"
VITALS = ["heart_rate", "resp_rate", "spo2", "sbp", "temperature"]
CACHE = REPO / "data" / "cache"
SPECS = ("yaib", "ecart", "sadanandan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--bound", type=float, default=2.0,
                    help="matched-density cut: all five vitals fresh within N h")
    args = ap.parse_args()

    o = (pd.read_parquet(CACHE / "eicu_cohort_hourly.parquet")
         .sort_values(["stay_id", "hour"]).reset_index(drop=True))
    y = o[LABEL].to_numpy(float)
    pid = o["patient_id"].to_numpy()
    keys = o.stay_id.to_numpy()

    models = {}
    for s in args.seeds:
        f = CACHE / ("eicu_twohead_all7.npz" if s == 42
                     else f"eicu_twohead_all7_s{s}.npz")
        if not f.exists():
            print(f"[skip] seed {s}: {f.name} not found")
            continue
        z = np.load(f)
        assert (z["stay_id"] == keys).all(), f"{f.name} misaligned"
        models[f"sentinel_s{s}"] = z["risk"].astype(float)
    lit = np.load(CACHE / "eicu_lit_scores.npz")
    assert (lit["stay_id"] == keys).all(), "lit scores misaligned"
    for spec in SPECS:
        models[spec] = lit[f"eicu_{spec}"].astype(float)

    m = np.ones(len(o), bool)
    for c in VITALS:
        m &= o[f"{c}_hrs_since"].to_numpy() <= args.bound
    print(f"[cut] all 5 vitals fresh <= {args.bound:g} h: {int(m.sum()):,} hours, "
          f"{len(np.unique(pid[m])):,} patients, {100*y[m].mean():.2f}% positive")

    seeds_present = [s for s in args.seeds if f"sentinel_s{s}" in models]
    print(f"\n=== point estimates on the cut ===")
    for name in [f"sentinel_s{s}" for s in seeds_present] + list(SPECS):
        a = auprc(y[m], models[name][m])
        print(f"  {name:<14} AUPRC {a:.4f}  lift {a/y[m].mean():5.2f}  "
              f"AUROC {auroc(y[m], models[name][m]):.4f}")

    print(f"\n=== paired deltas, {args.reps} reps, one shared resample ===")
    d = paired_auprc_draws(y[m], {k: v[m] for k, v in models.items()},
                           pid[m], reps=args.reps)
    clear = 0
    total = 0
    for s in seeds_present:
        print(f"  seed {s}:")
        for spec in SPECS:
            lo, hi = delta_ci(d[f"sentinel_s{s}"], d[spec])
            pt = float(np.mean(d[f"sentinel_s{s}"] - d[spec]))
            ok = lo > 0
            clear += ok
            total += 1
            print(f"    vs {spec:<11} {pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                  f"   {'clear of 0' if ok else 'SPANS 0'}")
    print(f"\n  {clear}/{total} comparisons with the interval clear of zero")

    if len(seeds_present) > 1:
        print(f"\n=== seed-to-seed spread (paired, same resample) ===")
        for i, a in enumerate(seeds_present):
            for b in seeds_present[i + 1:]:
                lo, hi = delta_ci(d[f"sentinel_s{a}"], d[f"sentinel_s{b}"])
                pt = float(np.mean(d[f"sentinel_s{a}"] - d[f"sentinel_s{b}"]))
                print(f"  seed {a} − seed {b}: {pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
