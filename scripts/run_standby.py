"""Standby operating points per ICU unit — the deployment answer to T10 (T12).

One flagship, one score; what goes on standby is a small table of per-unit
thresholds keyed by `first_careunit`, frozen on V_freeze within each unit.

  A. Burden equalization — does the five-fold alarm-load spread close?
  B. The race under standby points — does the lead survive re-anchoring
     NEWS2 within unit (it gets the same per-unit treatment, so the race
     stays volume-matched)?
  C. Agent roster — which agents earn their seat in which unit (standalone
     AUPRC and lift per unit, from the cached emissions).

    python scripts/run_standby.py [--budget 0.02] [--reps 1000]
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

OUT_JSON = REPO / "data" / "cache" / "standby_report.json"
EMB = REPO / "data" / "cache" / "embeddings"


def load_agent_risks(ctx):
    from standby import FLAGSHIP_AGENTS
    te = ctx.test_ordered
    risks = {}
    for name in FLAGSHIP_AGENTS:
        z = np.load(EMB / f"{name}_test.npz")
        if not ((z["stay_id"] == te["stay_id"].to_numpy()).all()
                and (z["hour"] == te["hour"].to_numpy()).all()):
            raise ValueError(f"{name} emissions misaligned with cohort test rows")
        risks[name] = z["risk"].astype(float)
    return risks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.02)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--view", default="unit_group")
    ap.add_argument("--seeds", default="42,1,2,3,4",
                    help="fusion seeds; full A/B tables print for the first, "
                         "the rest feed the D stability section")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    from baselines_mimic import load_splits
    from fusion_common import FusionContext
    from icu_strata import assign_strata
    from run_icu_strata import stay_frame          # scripts/ is on sys.path when run directly
    from standby import (agent_roster, burden_spread, evaluate_standby,
                         rows_stratum, standby_thresholds, threshold_stability)

    splits = load_splits()
    ctx = FusionContext(splits)
    strata = assign_strata(stay_frame(splits))

    th = np.load(REPO / "data/cache/two_head.npz")
    assert (th["stay_id_test"] == ctx.test_ordered["stay_id"].to_numpy()).all()

    strat_val = rows_stratum(ctx.val_ordered, strata, args.view)
    strat_test = rows_stratum(ctx.test_ordered, strata, args.view)

    per_seed = {}
    for seed in seeds:
        s_val = th[f"val_s{seed}"].astype(float)
        s_test = th[f"test_s{seed}"].astype(float)
        thr = standby_thresholds(ctx, strat_val, s_val, args.budget)
        r = evaluate_standby(ctx, strat_test, s_test, thr, reps=args.reps)
        per_seed[seed] = {"thresholds": thr, "units": r}

    # The A/B/C tables below report the first seed (the published s42 view);
    # D quantifies how much the operating points move across the others.
    thresholds = per_seed[seeds[0]]["thresholds"]
    rows = per_seed[seeds[0]]["units"]

    print("=" * 84)
    print(f"T12 — STANDBY OPERATING POINTS (budget {args.budget:.0%}, "
          f"view {args.view}, flagship s{seeds[0]}; "
          f"stability over seeds {{{args.seeds}}})")
    print("=" * 84)
    print(f"\nA — burden equalization "
          f"(global spread {burden_spread(rows, 'alarm_global'):.1f}x -> "
          f"standby {burden_spread(rows, 'alarm_standby'):.1f}x)\n")
    hdr = (f"{'unit':<14}{'hours':>9}{'prev':>7}{'thr src':>17}"
           f"{'alarm@global':>14}{'alarm@standby':>15}")
    print(hdr); print("-" * len(hdr))
    for unit, r in sorted(rows.items(), key=lambda kv: -kv[1]["hours"]):
        print(f"{unit:<14}{r['hours']:>9,}{r['prevalence']:>7.1%}"
              f"{r['source']:>17}{r['alarm_global']:>13.2%}"
              f"{r['alarm_standby']:>14.2%}")

    print("\nB — the paired race vs NEWS2, global vs standby operating points\n")
    hdr = (f"{'unit':<14}{'n_pair g':>9}{'lead g':>8}{'win g':>7}"
           f"{'n_pair s':>10}{'lead s':>8}{'win s':>7}")
    print(hdr); print("-" * len(hdr))
    for unit, r in sorted(rows.items(), key=lambda kv: -kv[1]["hours"]):
        g, s = r["race_global"], r["race_standby"]
        def fmt(c):
            if "median" not in c:
                return f"{c['n_pair']:>9}{'--':>8}{'--':>7}"
            return f"{c['n_pair']:>9}{c['median']:>7.1f}h{c['win_rate']:>7.0%}"
        print(f"{unit:<14}{fmt(g)} {fmt(s)}")

    risks = load_agent_risks(ctx)
    roster = agent_roster(ctx, strat_test, risks)
    print("\nC — agent roster: standalone lift (AUPRC / prevalence) per unit\n")
    agents = list(risks)
    hdr = f"{'unit':<14}" + "".join(f"{a[:9]:>11}" for a in agents)
    print(hdr); print("-" * len(hdr))
    for unit, r in sorted(roster.items(),
                          key=lambda kv: -kv[1].get("hours", 0)):
        if r.get("underpowered"):
            print(f"{unit:<14}  under-powered ({r['n_pos']} positives)")
            continue
        ranks = sorted(r["agents"].values(), key=lambda c: -c["lift"])[:3]
        top3 = {id(c) for c in ranks}
        line = f"{unit:<14}"
        for a in agents:
            c = r["agents"][a]
            mark = "*" if id(c) in top3 else " "
            line += f"{c['lift']:>10.1f}{mark}"
        print(line)
    print("  * top-3 in that unit — lift, not raw AUPRC, so the four-fold "
          "prevalence spread cannot rank the units for the agents")

    stability = threshold_stability(
        {s: per_seed[s]["thresholds"] for s in seeds})
    spread_by_seed = {s: burden_spread(per_seed[s]["units"], "alarm_standby")
                      for s in seeds}
    print(f"\nD — stability of the operating points across seeds {{{args.seeds}}}\n")
    hdr = (f"{'unit':<14}{'thr min':>10}{'thr max':>10}{'cv':>8}"
           f"{'sources':>28}")
    print(hdr); print("-" * len(hdr))
    for unit, st in sorted(stability.items()):
        src = ", ".join(f"{k}x{v}" for k, v in
                        sorted(st["source_counts"].items()))
        print(f"{unit:<14}{st['min']:>10.4f}{st['max']:>10.4f}"
              f"{st['cv']:>8.1%}{src:>28}")
    print(f"\n  standby burden spread by seed: "
          + "  ".join(f"s{s} {v:.1f}x" for s, v in spread_by_seed.items()))

    OUT_JSON.write_text(json.dumps(
        {"view": args.view, "budget": args.budget, "seeds": seeds,
         "thresholds": thresholds, "units": rows,          # first-seed view
         "per_seed": {str(s): per_seed[s] for s in seeds},
         "stability": stability,
         "standby_burden_spread_by_seed": {str(s): v for s, v
                                           in spread_by_seed.items()},
         "roster": roster}, indent=2, default=float))
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
