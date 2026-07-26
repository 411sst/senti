"""Modality ablation: which agents actually earn their place in the fusion?

    python scripts/modality_ablation.py

Trains the orchestrator on four agent subsets, everything else held identical
(same cached latents, same three-way split, same isotonic-on-val calibration,
death_weight=1, same evaluation harness):

    core-3          vitals-TCN + NEWS2 + labs      -- the verified result
    core-3 + ECG    does a near-null modality help or hurt?
    core-3 + notes  does a weak-but-real modality help or hurt?
    all-5           everything

Three questions this answers:

1. Do the two weak agents add anything on top of the core three, or are they
   redundant / actively harmful?
2. Does the attention downweight them appropriately -- i.e. is the fusion
   robust to being handed a mostly-uninformative input?
3. Does the headline "warns earlier than NEWS2" result survive at five agents?

This doubles as the graceful-degradation table: what the orchestrator scores
when a modality is unavailable, which is the realistic deployment condition.
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
CORE = ("vitals_tcn", "news2", "labs")

CONFIGS = {
    "core-3": CORE,
    "core-3 + ECG": CORE + ("ecg",),
    "core-3 + notes": CORE + ("notes",),
    "all-5": CORE + ("ecg", "notes"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    test_sorted = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    y = test_sorted[LABEL].to_numpy(dtype=float)
    news2 = test_sorted["news2"].to_numpy(dtype=float)
    types = list(EVENT_TYPES)

    runs = {}
    for name, agents in CONFIGS.items():
        t0 = time.time()
        bank = AgentBank.load(agent_names=agents)
        orch = Orchestrator(bank=bank, max_epochs=args.epochs, verbose=False)
        orch.fit(train, val)
        raw, cal = orch.raw_risk(test), orch.emit(test).risk
        _, attn = orch.attention_weights(test)
        runs[name] = {"agents": agents, "raw": raw, "cal": cal,
                      "attn": dict(zip(agents, attn.mean(axis=0)))}
        print(f"[fit] {name:<16} {len(agents)} agents  {time.time()-t0:.0f}s")

    # ---- headline table -------------------------------------------------
    print(f"\n{'=' * 104}")
    print(f"MODALITY ABLATION -- test: {len(y):,} patient-hours, "
          f"{test_sorted.stay_id.nunique():,} stays, prevalence {100*y.mean():.2f}%")
    print("=" * 104)
    hdr = (f"  {'config':<16}{'AUPRC raw':>11}{'AUPRC cal':>11}{'AUROC':>8}"
           f"{'ECE raw':>9}{'ECE cal':>9}   " + "".join(f"{t:>9}" for t in types))
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, r in runs.items():
        per_type = {t: auprc(test_sorted[f"{LABEL}_{t}"].to_numpy(), r["raw"])
                    for t in types}
        print(f"  {name:<16}{auprc(y, r['raw']):>11.3f}{auprc(y, r['cal']):>11.3f}"
              f"{auroc(y, r['raw']):>8.3f}{ece(y, r['raw']):>9.4f}"
              f"{ece(y, r['cal']):>9.4f}   "
              + "".join(f"{per_type[t]:>9.3f}" for t in types))
    print(f"  {'(random)':<16}{y.mean():>11.3f}{y.mean():>11.3f}{0.5:>8.3f}")

    # ---- attention: does the fusion downweight the weak agents? ---------
    print(f"\n  Mean attention over agents (does the fusion ignore what is useless?):")
    all_names = list(CONFIGS["all-5"])
    print(f"    {'config':<16}" + "".join(f"{a:>13}" for a in all_names))
    for name, r in runs.items():
        row = "".join(f"{r['attn'].get(a, float('nan')):>13.3f}"
                      if a in r["attn"] else f"{'--':>13}" for a in all_names)
        print(f"    {name:<16}{row}")
    print(f"    {'(uniform)':<16}"
          + "".join(f"{1/len(CONFIGS['all-5']):>13.3f}" for _ in all_names)
          + "   <- what 'no preference' would look like at 5 agents")

    # ---- does the lead-time headline survive? ---------------------------
    print(f"\n{'=' * 104}")
    print("PAIRED first-alarm vs NEWS2 -- volume-matched (+ = orchestrator earlier)")
    print("=" * 104)
    ph = (f"  {'config':<16}{'budget':>8}{'match%':>8}{'n_pair':>8}{'median':>8}"
          f"{'ORCH<':>7}{'N2<':>6}{'tie':>6}{'ORCHwin':>9}{'sign_p':>11}"
          f"{'fa_N2':>7}{'fa_OR':>7}")
    print(ph); print("  " + "-" * (len(ph) - 2))
    n2_series = _stay_series(test_sorted, news2)
    for name, r in runs.items():
        series = _stay_series(test_sorted, r["raw"])
        for rate in (0.01, 0.05):
            thr_n2 = float(np.quantile(news2, 1.0 - rate))
            matched = float((news2 >= thr_n2).mean())
            thr_or = float(np.quantile(r["raw"], 1.0 - matched))
            pr = paired_first_alarm(test_sorted, news2, r["raw"], thr_n2, thr_or)
            if pr["n_paired"] == 0:
                continue
            print(f"  {name:<16}{f'{rate*100:.0f}%':>8}{100*matched:>7.1f}%"
                  f"{pr['n_paired']:>8}{pr['median_diff']:>8.1f}"
                  f"{pr['n_gbm_earlier']:>7}{pr['n_news2_earlier']:>6}{pr['n_tie']:>6}"
                  f"{100*pr['frac_gbm_earlier']:>8.0f}%{pr['sign_p']:>11.2g}"
                  f"{_false_alarm_stay_rate(n2_series, thr_n2):>7.2f}"
                  f"{_false_alarm_stay_rate(series, thr_or):>7.2f}")

    # ---- is any difference between configs beyond noise? ----------------
    print(f"\n{'=' * 104}")
    print("PAIRED bootstrap vs core-3 (stay-level resample, 1000 reps)")
    print("=" * 104)
    rng = np.random.default_rng(0)
    stays = test_sorted["stay_id"].to_numpy()
    uniq = np.unique(stays)
    idx = {s: np.flatnonzero(stays == s) for s in uniq}
    base = runs["core-3"]["raw"]
    for name in ("core-3 + ECG", "core-3 + notes", "all-5"):
        d_pr, d_roc = [], []
        for _ in range(1000):
            pick = rng.choice(uniq, len(uniq), replace=True)
            ii = np.concatenate([idx[s] for s in pick])
            if y[ii].sum() < 5:
                continue
            d_pr.append(auprc(y[ii], runs[name]["raw"][ii]) - auprc(y[ii], base[ii]))
            d_roc.append(auroc(y[ii], runs[name]["raw"][ii]) - auroc(y[ii], base[ii]))
        d_pr, d_roc = np.array(d_pr), np.array(d_roc)
        lo, hi = np.percentile(d_pr, [2.5, 97.5])
        rlo, rhi = np.percentile(d_roc, [2.5, 97.5])
        print(f"  {name:<16} AUPRC {d_pr.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] "
              f"P(>0)={100*(d_pr > 0).mean():>3.0f}%   |   "
              f"AUROC {d_roc.mean():+.4f} [{rlo:+.4f}, {rhi:+.4f}] "
              f"P(>0)={100*(d_roc > 0).mean():>3.0f}%")


if __name__ == "__main__":
    main()
