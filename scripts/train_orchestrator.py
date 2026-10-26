"""Train the orchestrator and run the full evaluation harness.

    python scripts/train_orchestrator.py [--epochs 8]

Fuses the cached agent latents (vitals-TCN, NEWS2, labs), trains on train,
isotonic-calibrates on val, and reports on test:

  * AUPRC (raw + calibrated) and ECE (raw + calibrated) against every baseline
    and every agent,
  * AUPRC per event type,
  * caught-in-time and lead-time at matched alert budgets, with the stay-level
    false-alarm rate alongside,
  * the paired, volume-matched first-alarm comparison against NEWS2.
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from baselines_mimic import (  # noqa: E402
    _false_alarm_stay_rate,
    _stay_series,
    alertable_mask,
    paired_first_alarm,
)
from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, ece, lead_time  # noqa: E402
from mimic_cohort import EVENT_TYPES  # noqa: E402
from orchestrator.fusion import AgentBank, Orchestrator  # noqa: E402

LABEL = "label_24h"
EMB_DIR = REPO / "data" / "cache" / "embeddings"

# Previously reported rows (see the agent/baseline scripts) for the comparison.
PRIOR = [
    ("labs agent",   0.171, 0.162, 0.0035, 0.0012, 0.128, 0.100, 0.035),
    ("vitals-TCN",   0.161, 0.156, 0.0519, 0.0023, 0.083, 0.054, 0.132),
    ("NEWS2 agent",  0.108, 0.107, 0.1623, 0.0014, 0.059, 0.041, 0.062),
    ("GBM baseline", 0.170, 0.170, 0.0036, 0.0016, 0.081, 0.049, 0.187),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    print("[bank] loading cached agent latents...")
    bank = AgentBank.load()
    print(f"[bank] {bank.matrix.shape[0]:,} rows x {bank.n_features} features "
          f"({bank.latent_dims}, + {len(bank.agent_names)} risks)")

    orch = Orchestrator(bank=bank, max_epochs=args.epochs)
    t0 = time.time()
    orch.fit(train, val)
    print(f"[train] {time.time() - t0:.0f}s")

    for name, df in (("train", train), ("val", val), ("test", test)):
        out = orch.emit_and_cache(df, EMB_DIR / f"{orch.name}_{name}.npz")
        print(f"[cache] {name}: {out.latent.shape} fused latents")

    test_sorted = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    out = orch.emit(test)
    assert (out.keys["stay_id"].to_numpy() == test_sorted["stay_id"].to_numpy()).all()

    y = test_sorted[LABEL].to_numpy()
    fused_cal, fused_raw = out.risk, orch.raw_risk(test)
    cal = orch.calibration_summary

    # ---- headline table -------------------------------------------------
    print(f"\n{'=' * 78}\nORCHESTRATOR on test ({len(y):,} patient-hours, "
          f"{test_sorted['stay_id'].nunique():,} stays, "
          f"prevalence {100 * y.mean():.2f}%)\n{'=' * 78}")
    print(f"  fused latent dim {out.latent_dim}; isotonic calibration on val "
          f"({cal['split_rows']:,} rows, {cal['positives']:,} positive)\n")

    types = list(EVENT_TYPES)
    hdr = (f"  {'model':<15}{'AUPRC raw':>11}{'AUPRC cal':>11}{'ECE raw':>10}"
           f"{'ECE cal':>10}   " + "".join(f"{t:>9}" for t in types))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    per_type = {t: auprc(test_sorted[f"{LABEL}_{t}"].to_numpy(), fused_raw)
                for t in types}
    print(f"  {'ORCHESTRATOR':<15}{auprc(y, fused_raw):>11.3f}"
          f"{auprc(y, fused_cal):>11.3f}{ece(y, fused_raw):>10.4f}"
          f"{ece(y, fused_cal):>10.4f}   "
          + "".join(f"{per_type[t]:>9.3f}" for t in types))
    for name, raw, c, er, ec, v, s, d in PRIOR:
        print(f"  {name:<15}{raw:>11.3f}{c:>11.3f}{er:>10.4f}{ec:>10.4f}   "
              + "".join(f"{x:>9.3f}" for x in (v, s, d)))
    print(f"  {'(random)':<15}{y.mean():>11.3f}{y.mean():>11.3f}")

    # ---- attention: which agent drives the fused risk? -------------------
    _, weights = orch.attention_weights(test)
    print("\n  Mean attention over agents (what the fusion actually leans on):")
    for name, w in zip(bank.agent_names, weights.mean(axis=0)):
        print(f"    {name:<14}{w:>7.3f}")

    # ---- lead-time sweep vs NEWS2 ---------------------------------------
    # Warm-up rule: budgets, thresholds and alarm scans all run over alertable
    # hours only (see baselines_mimic.alertable_mask); _stay_series and
    # paired_first_alarm apply the same mask to the scans.
    news2 = test_sorted["news2"].to_numpy(dtype=float)
    alertable = alertable_mask(test_sorted)
    series = {"ORCH": _stay_series(test_sorted, fused_raw),
              "NEWS2": _stay_series(test_sorted, news2)}
    scores = {"ORCH": fused_raw, "NEWS2": news2}

    print("\n  Lead-time at matched alert budgets (test; warm-up hour excluded):")
    sweep_hdr = (f"  {'budget':>7}  {'model':<6}{'actual':>8}{'caught':>8}"
                 f"{'falarm':>8}{'n':>7}    lead p25/p50/p75 (h)")
    print(sweep_hdr)
    print("  " + "-" * (len(sweep_hdr) - 2))
    for rate in (0.01, 0.02, 0.05):
        for name in ("NEWS2", "ORCH"):
            sa = scores[name][alertable]
            thr = float(np.quantile(sa, 1.0 - rate))
            r = lead_time(*series[name][:2], thr, times=series[name][2])
            band = ("n/a" if r["n_caught"] == 0 else
                    f"{r['lead_p25']:>4.0f} /{r['median_lead_time']:>4.0f} /"
                    f"{r['lead_p75']:>4.0f}")
            print(f"  {f'{rate*100:.0f}%':>7}  {name:<6}"
                  f"{100*float((sa >= thr).mean()):>7.1f}%"
                  f"{r['caught_in_time_rate']:>8.2f}"
                  f"{_false_alarm_stay_rate(series[name], thr):>8.2f}"
                  f"{r['n_caught']:>7}    {band}")

    # ---- paired, volume-matched comparison vs NEWS2 ----------------------
    print("\n  PAIRED first-alarm vs NEWS2 -- stays caught by BOTH, orchestrator")
    print("  matched to NEWS2's ACHIEVED alert volume."
          "  diff = NEWS2 - ORCH (+ = ORCH earlier):")
    p_hdr = (f"  {'budget':>7}{'match%':>8}{'n_pair':>8}{'median':>8}{'p25':>6}"
             f"{'p75':>6}{'ORCH<':>7}{'N2<':>6}{'tie':>6}{'ORCHwin':>9}"
             f"{'sign_p':>10}{'fa_N2':>7}{'fa_OR':>7}")
    print(p_hdr)
    print("  " + "-" * (len(p_hdr) - 2))
    for rate in (0.01, 0.02, 0.05):
        thr_n2 = float(np.quantile(news2[alertable], 1.0 - rate))
        matched = float((news2[alertable] >= thr_n2).mean())
        thr_or = float(np.quantile(fused_raw[alertable], 1.0 - matched))
        # paired_first_alarm names its columns for the NEWS2-vs-GBM comparison;
        # here the "gbm" slot is the orchestrator.
        r = paired_first_alarm(test_sorted, news2, fused_raw, thr_n2, thr_or)
        if r["n_paired"] == 0:
            print(f"  {f'{rate*100:.0f}%':>7}{100*matched:>7.1f}%{0:>8}"
                  "   (no stays caught by both)")
            continue
        print(f"  {f'{rate*100:.0f}%':>7}{100*matched:>7.1f}%{r['n_paired']:>8}"
              f"{r['median_diff']:>8.1f}{r['diff_p25']:>6.0f}{r['diff_p75']:>6.0f}"
              f"{r['n_gbm_earlier']:>7}{r['n_news2_earlier']:>6}{r['n_tie']:>6}"
              f"{100*r['frac_gbm_earlier']:>8.0f}%{r['sign_p']:>10.2g}"
              f"{_false_alarm_stay_rate(series['NEWS2'], thr_n2):>7.2f}"
              f"{_false_alarm_stay_rate(series['ORCH'], thr_or):>7.2f}")
    print("\n  caught/falarm read together: a stay counts as caught if ANY hour "
          "crosses,\n  so at loose budgets both saturate. sign_p = two-sided sign "
          "test on\n  discordant pairs.")


if __name__ == "__main__":
    main()
