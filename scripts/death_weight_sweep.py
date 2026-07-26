"""Bounded experiment: can death-weighting fix the orchestrator's death deficit?

    python scripts/death_weight_sweep.py [--weights 1 3 10 30]

The unweighted orchestrator scores 0.082 AUPRC on death against the GBM's
0.187, because death is only ~16% of positive hours and the fusion optimises
for the vent/vaso majority -- leaning on the labs agent (the weakest death
predictor) over the vitals-TCN (the strongest).

This sweeps a loss weight applied to death-positive rows and reports the whole
tradeoff curve: does death improve, and what does it cost vent/vaso, overall
AUPRC, calibration, and lead-time? Everything else is held fixed -- same agents,
same cached latents, same three-way split, same isotonic-on-val calibration,
same evaluation harness.

Nothing here is tuned beyond the weight itself; the point is to characterise the
tradeoff honestly, including if the answer is "not worth it".
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from baselines_mimic import (  # noqa: E402
    _false_alarm_stay_rate, _stay_series, load_splits, paired_first_alarm,
)
from metrics import auprc, auroc, ece, lead_time  # noqa: E402
from mimic_cohort import EVENT_TYPES  # noqa: E402
from orchestrator.fusion import AgentBank, Orchestrator  # noqa: E402

LABEL = "label_24h"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=float, nargs="+", default=[1, 3, 10, 30])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--select-on", default="composite", choices=["composite", "death"])
    args = ap.parse_args()

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    test_sorted = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    y = test_sorted[LABEL].to_numpy(dtype=float)
    news2 = test_sorted["news2"].to_numpy(dtype=float)
    types = list(EVENT_TYPES)

    bank = AgentBank.load()
    n_death = int(test_sorted[f"{LABEL}_death"].sum())
    print(f"test: {len(y):,} patient-hours, {int(y.sum()):,} positive "
          f"({n_death:,} death-positive), {test_sorted.stay_id.nunique():,} stays")
    print(f"selection criterion: val {args.select_on} AUPRC\n")

    runs = {}
    for w in args.weights:
        t0 = time.time()
        orch = Orchestrator(bank=bank, max_epochs=args.epochs, verbose=False,
                            death_weight=w, select_on=args.select_on)
        orch.fit(train, val)
        raw, cal = orch.raw_risk(test), orch.emit(test).risk
        _, attn = orch.attention_weights(test)
        runs[w] = {"raw": raw, "cal": cal, "attn": attn.mean(axis=0)}
        print(f"[fit] death_weight={w:g}  {time.time()-t0:.0f}s")

    # ---- headline tradeoff table ---------------------------------------
    print(f"\n{'='*104}")
    print("DEATH-WEIGHTING TRADEOFF (all other settings identical)")
    print("="*104)
    hdr = (f"  {'weight':>7}{'AUPRC raw':>11}{'AUPRC cal':>11}{'AUROC':>8}"
           f"{'ECE raw':>9}{'ECE cal':>9}   " + "".join(f"{t:>9}" for t in types)
           + "    attention: " + " ".join(f"{a:>10}" for a in bank.agent_names))
    print(hdr)
    print("  " + "-"*(len(hdr)-2))
    for w in args.weights:
        r = runs[w]
        per_type = {t: auprc(test_sorted[f"{LABEL}_{t}"].to_numpy(), r["raw"])
                    for t in types}
        tag = f"{w:g}" + (" *" if w == 1 else "")
        print(f"  {tag:>7}{auprc(y, r['raw']):>11.3f}{auprc(y, r['cal']):>11.3f}"
              f"{auroc(y, r['raw']):>8.3f}{ece(y, r['raw']):>9.4f}"
              f"{ece(y, r['cal']):>9.4f}   "
              + "".join(f"{per_type[t]:>9.3f}" for t in types)
              + "                  " + " ".join(f"{a:>10.3f}" for a in r["attn"]))
    print("  * weight 1 = the current unweighted orchestrator (the baseline).")
    print(f"  Reference -- GBM: death 0.187, vent 0.081, vaso 0.049 | "
          f"vitals-TCN: death 0.139 | labs: death 0.035")
    print(f"  Per-type AUPRC floors: " + "  ".join(
        f"{t} {test_sorted[f'{LABEL}_{t}'].mean():.4f}" for t in types))

    # ---- calibration + lead-time must not be collateral damage ----------
    print("\n  Lead-time and false alarms at matched budgets (headline metrics):")
    lh = (f"  {'weight':>7}  {'budget':>7}{'actual':>8}{'caught':>8}{'falarm':>8}"
          f"{'n':>7}    lead p25/p50/p75 (h)")
    print(lh); print("  " + "-"*(len(lh)-2))
    for w in args.weights:
        s = runs[w]["raw"]
        series = _stay_series(test_sorted, s)
        for rate in (0.01, 0.05):
            thr = float(np.quantile(s, 1.0 - rate))
            r = lead_time(*series[:2], thr, times=series[2])
            band = ("n/a" if r["n_caught"] == 0 else
                    f"{r['lead_p25']:>4.0f} /{r['median_lead_time']:>4.0f} /"
                    f"{r['lead_p75']:>4.0f}")
            print(f"  {w:>7g}  {f'{rate*100:.0f}%':>7}"
                  f"{100*float((s>=thr).mean()):>7.1f}%{r['caught_in_time_rate']:>8.2f}"
                  f"{_false_alarm_stay_rate(series, thr):>8.2f}{r['n_caught']:>7}"
                  f"    {band}")

    print("\n  PAIRED first-alarm vs NEWS2, volume-matched (+ = orchestrator earlier):")
    ph = (f"  {'weight':>7}  {'budget':>7}{'n_pair':>8}{'median':>8}{'ORCH<':>7}"
          f"{'N2<':>6}{'tie':>6}{'ORCHwin':>9}{'sign_p':>11}")
    print(ph); print("  " + "-"*(len(ph)-2))
    for w in args.weights:
        s = runs[w]["raw"]
        for rate in (0.01, 0.05):
            thr_n2 = float(np.quantile(news2, 1.0 - rate))
            matched = float((news2 >= thr_n2).mean())
            thr_or = float(np.quantile(s, 1.0 - matched))
            r = paired_first_alarm(test_sorted, news2, s, thr_n2, thr_or)
            if r["n_paired"] == 0:
                continue
            print(f"  {w:>7g}  {f'{rate*100:.0f}%':>7}{r['n_paired']:>8}"
                  f"{r['median_diff']:>8.1f}{r['n_gbm_earlier']:>7}"
                  f"{r['n_news2_earlier']:>6}{r['n_tie']:>6}"
                  f"{100*r['frac_gbm_earlier']:>8.0f}%{r['sign_p']:>11.2g}")


if __name__ == "__main__":
    main()
