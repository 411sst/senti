"""NEWS2 fairness band: the timing claim against NEWS2's strongest form.

NEWS2 as implemented is a conservatively DEGRADED comparator: a vital that
goes unmeasured past its staleness bound scores 0, unknown consciousness
counts as Alert, uncharted oxygen as room air -- every default can only lower
the score. Since "warns earlier than NEWS2" is the headline claim, it has to
hold against the best case for NEWS2, not merely the as-charted one.

Two variants of the comparator:

    as-charted   the cohort's ``news2`` column: bounded LOCF (vitals go stale
                 after 8-24h per channel), missing-as-normal defaults
    best-LOCF    unbounded within-stay carry-forward of every NEWS2 input --
                 once observed, a value stays current until re-measured, so
                 nothing an observation ever said is forgotten. Defaults apply
                 only where NOTHING was ever charted.

Both are scored with the same vectorised NEWS2. The paired lead-time race
(val-frozen thresholds, patient-clustered CIs) runs the orchestrator against
each variant; the clinical-trigger table additionally evaluates the RCP
escalation rules -- aggregate >=5, aggregate >=7, and any single parameter
scoring 3 -- as fixed alarm rules with a volume-matched orchestrator paired
against each.

    python src/news2_band.py [--reps 2000]
"""

import argparse

import numpy as np
import pandas as pd

from baselines_mimic import _false_alarm_stay_rate, _stay_series, load_splits
from fair_capability import (
    BUDGETS,
    frozen_thresholds,
    load_orch,
    paired_first_alarm_diffs,
    summarize_paired,
)
from metrics import lead_time
from mimic_cohort import (
    RAW_PATH,
    STAB_HOURS,
    build_before_t_features,
    news2_component_scores,
)

LABEL = "label_24h"

# RCP escalation triggers, as predicates over (total score, component matrix).
CLINICAL_TRIGGERS = {
    "agg >= 5": lambda total, comp: total >= 5,
    "agg >= 7": lambda total, comp: total >= 7,
    "any param 3": lambda total, comp: (comp == 3).any(axis=1),
}


def news2_variants(raw_path=RAW_PATH):
    """Both NEWS2 variants for every prediction row, keyed by (stay_id, hour).

    Rebuilds features from the raw extract twice -- once with the production
    staleness bounds (as-charted) and once with none (best-LOCF) -- and scores
    each with the same vectorised NEWS2. History rows [0, STAB_HOURS) feed the
    carry-forward and are then dropped, exactly as in the cohort build.

    Returns a frame with ``news2_charted`` / ``news2_locf`` totals and
    ``param3_charted`` / ``param3_locf`` single-parameter-3 flags.
    """
    raw = pd.read_parquet(raw_path)
    raw = raw.rename(columns={"subject_id": "patient_id"})
    from mimic_cohort import CARRIED
    for col in CARRIED:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").astype(float)

    out = None
    for tag, staleness in (("charted", None), ("locf", {})):
        feat = build_before_t_features(raw, max_staleness=staleness)
        comp, _, _ = news2_component_scores(feat)
        frame = feat[["stay_id", "hour"]].copy()
        frame[f"news2_{tag}"] = comp.sum(axis=1).astype(np.int16)
        frame[f"param3_{tag}"] = (comp == 3).any(axis=1)
        keep = frame[feat["hour"] >= STAB_HOURS] if "is_history" not in feat.columns \
            else frame[pd.to_numeric(feat["is_history"]) == 0]
        out = keep if out is None else out.merge(keep, on=["stay_id", "hour"])
    return out.sort_values(["stay_id", "hour"]).reset_index(drop=True)


def _aligned(variants, split_frame, column):
    """A variant column in the split's sorted (stay_id, hour) row order."""
    ordered = split_frame.sort_values(["stay_id", "hour"])
    merged = ordered[["stay_id", "hour"]].merge(
        variants, on=["stay_id", "hour"], how="left", validate="one_to_one")
    values = merged[column]
    if values.isna().any():
        raise ValueError(f"{int(values.isna().sum())} cohort rows missing from "
                         "the NEWS2 variant frame")
    return values.to_numpy()


def run(reps=2000):
    splits = load_splits()
    val, test = splits["val"], splits["test"]
    val_ordered = val.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    test_ordered = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    stay_to_patient = dict(zip(test_ordered["stay_id"], test_ordered["patient_id"]))

    print("[band] building both NEWS2 variants from the raw extract...")
    variants = news2_variants()

    charted_test = _aligned(variants, test, "news2_charted").astype(float)
    sanity = test_ordered["news2"].to_numpy(dtype=float)
    if not (charted_test == sanity).all():
        raise ValueError("recomputed as-charted NEWS2 disagrees with the "
                         "cohort's news2 column -- variant build is wrong")

    orch_val, orch_test = load_orch(val, "val"), load_orch(test, "test")

    news2 = {
        "as-charted": (_aligned(variants, val, "news2_charted").astype(float),
                       charted_test),
        "best-LOCF": (_aligned(variants, val, "news2_locf").astype(float),
                      _aligned(variants, test, "news2_locf").astype(float)),
    }

    # ---- paired lead-time: orchestrator vs each variant -------------------
    print(f"\nPAIRED FIRST-ALARM: ORCHESTRATOR vs NEWS2 variants -- thresholds "
          f"frozen on val\n(diff > 0 = orchestrator earlier; {reps}-rep "
          f"patient-clustered CIs)\n")
    hdr = (f"  {'variant':<12}{'budget':>7}{'n2%':>6}{'orch%':>7}{'n_pair':>7}"
           f"{'median':>7}  {'95% CI':<14}{'win':>5}  {'95% CI':<14}{'ties':>5}"
           f"{'sign_p':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, (n2_val, n2_test) in news2.items():
        for budget in BUDGETS:
            thr_n2, thr_or, _ = frozen_thresholds(
                n2_val, orch_val, np.ones(len(n2_val), dtype=bool), budget)
            diffs = paired_first_alarm_diffs(test_ordered, n2_test, orch_test,
                                             thr_n2, thr_or)
            r = summarize_paired(diffs, stay_to_patient, reps=reps)
            print(f"  {name:<12}{f'{budget*100:.0f}%':>7}"
                  f"{100*float((n2_test >= thr_n2).mean()):>5.1f}%"
                  f"{100*float((orch_test >= thr_or).mean()):>6.1f}%"
                  f"{r['n_pair']:>7}{r['median']:>7.1f}  "
                  f"[{r['median_ci'][0]:>4.1f}, {r['median_ci'][1]:>4.1f}]"
                  f"{100*r['win_rate']:>4.0f}%  "
                  f"[{100*r['win_rate_ci'][0]:>3.0f}%, {100*r['win_rate_ci'][1]:>3.0f}%]"
                  f"{r['ties']:>5}{r['sign_p']:>10.2g}")

    # ---- clinical operating points ----------------------------------------
    # Fixed RCP rules, not quantiles: report each rule's own operating
    # characteristics, then pair the orchestrator against it at the SAME alert
    # volume (matched on val, frozen, applied to test).
    print("\nCLINICAL OPERATING POINTS -- fixed RCP triggers, orchestrator "
          "volume-matched to each\n")
    hdr = (f"  {'variant':<12}{'trigger':<13}{'vol%':>6}{'caught':>8}{'falarm':>8}"
           f"  lead p25/p50/p75{'':>3}{'n_pair':>7}{'median':>7}{'win':>5}"
           f"{'sign_p':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in news2:
        # column tags: as-charted -> charted, best-LOCF -> locf
        tag = "charted" if name == "as-charted" else "locf"
        total_val = _aligned(variants, val, f"news2_{tag}").astype(float)
        total_test = _aligned(variants, test, f"news2_{tag}").astype(float)
        comp_val = _aligned(variants, val, f"param3_{tag}").astype(bool)
        comp_test = _aligned(variants, test, f"param3_{tag}").astype(bool)
        flags = {
            "agg >= 5": ((total_val >= 5), (total_test >= 5)),
            "agg >= 7": ((total_val >= 7), (total_test >= 7)),
            "any param 3": (comp_val, comp_test),
        }
        for trig, (f_val, f_test) in flags.items():
            alarm_val = f_val.astype(float)
            alarm_test = f_test.astype(float)
            vol_val = float(alarm_val.mean())
            thr_or = float(np.quantile(orch_val, 1.0 - vol_val))
            series = _stay_series(test_ordered, alarm_test)
            lt = lead_time(*series[:2], 0.5, times=series[2])
            band = ("n/a" if lt["n_caught"] == 0 else
                    f"{lt['lead_p25']:>4.0f}/{lt['median_lead_time']:>4.0f}/"
                    f"{lt['lead_p75']:>4.0f}")
            diffs = paired_first_alarm_diffs(test_ordered, alarm_test, orch_test,
                                             0.5, thr_or)
            r = summarize_paired(diffs, stay_to_patient, reps=max(200, reps // 4))
            print(f"  {name:<12}{trig:<13}"
                  f"{100*float(alarm_test.mean()):>5.1f}%"
                  f"{lt['caught_in_time_rate']:>8.2f}"
                  f"{_false_alarm_stay_rate(series, 0.5):>8.2f}"
                  f"  {band:<16}{r['n_pair']:>7}{r['median']:>7.1f}"
                  f"{100*r['win_rate']:>4.0f}%{r['sign_p']:>10.2g}")
    print("\n  vol% = fraction of test hours the fixed rule alarms on; the "
          "orchestrator is\n  held to the rule's VAL volume. caught/falarm are "
          "stay-level; median/win are the\n  paired first-alarm result "
          "(+ = orchestrator earlier).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    run(reps=args.reps)
