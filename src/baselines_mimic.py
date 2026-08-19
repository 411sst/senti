"""NEWS2 and GBM baselines on the MIMIC-IV in-ICU hourly cohort.

This is the bar everything else has to clear, so the protocol is deliberately
strict:

* **Train on train, calibrate on val, report on test.** Both models' scores are
  isotonic-calibrated on the held-out val split -- never on data the model was
  fit to -- because ECE is a headline metric for this project and calibrating on
  train would flatter it.
* **Episode = ICU stay** for lead-time (each stay is one clinical episode),
  while the splits stay patient-level so no patient spans two splits.
* **Per-event-type reporting.** Ventilation dominates the composite label, so
  the table breaks AUPRC out by component -- a model that only predicts
  intubation should not be able to hide behind the composite number.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.isotonic import IsotonicRegression

from config import SEED
from metrics import auprc, ece, lead_time
from mimic_cohort import EVENT_TYPES, OUT_PATH, STAB_HOURS, VITALS

LABEL = "label_24h"
STAY_ID = "stay_id"
HOUR = "hour"


def alertable_mask(df):
    """Rows eligible to raise an alert: every prediction hour, since the h6 fix.

    Historical note: the hourly grid once started AT the stabilization
    boundary, so the strictly-before-t shift left every stay's first row
    (h6) with no observed data at all -- NEWS2 pinned to 0 while a learned
    model could still fire on its base-rate prior, a free head start that had
    to be masked out of all alerting evaluation (the old "warm-up rule").
    The extract now emits hours [0, STAB_HOURS) as history rows, so h6
    carries six hours of real carried-forward context and competes on data
    like every other hour. The function remains as the single seam where an
    eligibility rule would live if a future grid change needed one.
    """
    return np.ones(len(df), dtype=bool)

# Raw inputs only -- NEWS2 itself is the competing baseline, never a GBM feature.
GBM_FEATURES = (
    VITALS
    + [f"{v}_hrs_since" for v in VITALS]
    + ["gcs", "gcs_unable", "on_oxygen"]
)

# Alert budgets for the lead-time sweep (fraction of patient-hours flagged).
ALERT_RATES = [0.01, 0.02, 0.05, 0.10]


def load_splits(path=OUT_PATH):
    df = pd.read_parquet(path)
    return {s: df[df.split == s] for s in ("train", "val", "test")}


def train_gbm(train, seed=SEED):
    """XGBoost with library defaults (no tuning -- this is a baseline).

    XGBoost is imported HERE rather than at module scope on purpose: it and
    PyTorch each bundle their own OpenMP runtime, and loading both in one
    process deadlocks or segfaults on macOS (see tests/conftest.py). Several
    modules want this file's evaluation helpers -- lead_time sweeps, the paired
    comparison -- without wanting a GBM; keeping the import lazy means those
    callers never load the conflicting library at all.
    """
    import xgboost as xgb

    model = xgb.XGBClassifier(
        random_state=seed, n_jobs=-1, tree_method="hist", eval_metric="logloss"
    )
    model.fit(train[GBM_FEATURES], train[LABEL].to_numpy())
    return model


def _stay_series(df, scores):
    """Per-ICU-stay score sequences + exact event hour, for lead-time.

    Uses ``event_h`` (the true hour of the first deterioration event) rather
    than inferring it from the label, so lead times are exact.

    Applies :func:`alertable_mask` (all hours, since the h6 root-cause fix),
    kept as the one seam where an eligibility rule would apply.
    """
    tmp = df[[STAY_ID, HOUR, "event_h"]].copy()
    tmp["score"] = np.asarray(scores)
    tmp = tmp[alertable_mask(tmp)]
    risk, events, times = [], [], []
    for _, g in tmp.groupby(STAY_ID, sort=False):
        g = g.sort_values(HOUR)
        risk.append(g["score"].to_numpy())
        times.append(g[HOUR].to_numpy())
        e = g["event_h"].iloc[0]
        events.append(None if pd.isna(e) else float(e))
    return risk, events, times


def _false_alarm_stay_rate(series, threshold):
    """Fraction of NON-deteriorating stays that alarm at least once.

    This is the companion to ``caught_in_time_rate`` and must be read next to
    it. A stay counts as "caught" if ANY of its hours crosses the threshold, and
    stays here run a median of 18 hours -- so at a loose budget essentially
    every stay crosses eventually and "caught" saturates at 1.00 while carrying
    no information. Reporting the false-alarm side makes that visible instead of
    letting a perfect-looking recall stand on its own.
    """
    risk, events, _ = series
    non_event = [np.asarray(r) for r, e in zip(risk, events) if e is None]
    if not non_event:
        return float("nan")
    return float(np.mean([(r >= threshold).any() for r in non_event]))


def paired_first_alarm(test, s_news2, s_gbm, thr_news2, thr_gbm):
    """Paired per-stay comparison of WHEN each model first alarms.

    Restricted to stays that BOTH models catch in time, so the two are judged on
    identical patients. The quantity is

        diff = first_alarm_NEWS2 - first_alarm_GBM      (+ => GBM fired earlier)

    Pairing matters because the unpaired medians are dominated by *when the
    event happened*, which is a property of the patient and identical for both
    models -- on PhysioNet that swamped a real difference entirely. In a paired
    difference the event time cancels, leaving only the models' disagreement.

    Both models are evaluated at thresholds chosen to flag the same volume of
    hours (see :func:`run`), so this measures ranking, not alert appetite.
    Includes a two-sided sign test over the discordant pairs.

    Applies :func:`alertable_mask` (all hours, since the h6 root-cause fix).
    """
    tmp = test[[STAY_ID, HOUR, "event_h"]].copy()
    tmp["n2"] = np.asarray(s_news2)
    tmp["gb"] = np.asarray(s_gbm)
    tmp = tmp[alertable_mask(tmp)]

    diffs = []
    for _, g in tmp.groupby(STAY_ID, sort=False):
        event = g["event_h"].iloc[0]
        if pd.isna(event):
            continue                       # no event -> no lead time to compare
        g = g.sort_values(HOUR)
        hours = g[HOUR].to_numpy(dtype=float)
        cross_n2 = hours[g["n2"].to_numpy() >= thr_news2]
        cross_gb = hours[g["gb"].to_numpy() >= thr_gbm]
        if cross_n2.size == 0 or cross_gb.size == 0:
            continue                       # only stays BOTH models catch
        first_n2, first_gb = cross_n2[0], cross_gb[0]
        if first_n2 > event or first_gb > event:
            continue                       # alarm must precede the event
        diffs.append(first_n2 - first_gb)

    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    n_gbm_earlier = int((diffs > 0).sum())
    n_news2_earlier = int((diffs < 0).sum())
    n_discordant = n_gbm_earlier + n_news2_earlier
    sign_p = (binomtest(n_gbm_earlier, n_discordant, 0.5).pvalue
              if n_discordant else 1.0)

    if n:
        p25, p50, p75 = (float(np.percentile(diffs, q)) for q in (25, 50, 75))
    else:
        p25 = p50 = p75 = float("nan")
    return {
        "n_paired": n, "median_diff": p50, "diff_p25": p25, "diff_p75": p75,
        "n_gbm_earlier": n_gbm_earlier, "n_news2_earlier": n_news2_earlier,
        "n_tie": n - n_discordant,
        "frac_gbm_earlier": (n_gbm_earlier / n if n else float("nan")),
        "sign_p": float(sign_p),
    }


def run(path=OUT_PATH, seed=SEED, alert_rates=ALERT_RATES):
    splits = load_splits(path)
    train, val, test = splits["train"], splits["val"], splits["test"]
    y_val, y_test = val[LABEL].to_numpy(), test[LABEL].to_numpy()

    # --- scores -----------------------------------------------------------
    news2_val, news2_test = val["news2"].to_numpy(), test["news2"].to_numpy()
    model = train_gbm(train, seed=seed)
    gbm_val = model.predict_proba(val[GBM_FEATURES])[:, 1]
    gbm_test = model.predict_proba(test[GBM_FEATURES])[:, 1]

    # --- calibration fit on VAL, applied to TEST --------------------------
    cal_news2 = IsotonicRegression(out_of_bounds="clip").fit(news2_val, y_val)
    cal_gbm = IsotonicRegression(out_of_bounds="clip").fit(gbm_val, y_val)
    news2_cal = cal_news2.transform(news2_test)
    gbm_cal = cal_gbm.transform(gbm_test)

    scores = {"NEWS2": news2_test, "GBM": gbm_test}
    calibrated = {"NEWS2": news2_cal, "GBM": gbm_cal}

    results = {
        "summary": {
            "n_test_hours": int(len(test)), "n_test_stays": int(test[STAY_ID].nunique()),
            "n_test_patients": int(test["patient_id"].nunique()),
            "prevalence": float(y_test.mean()),
            "n_train_hours": int(len(train)), "n_val_hours": int(len(val)),
        },
        "discrimination": {}, "by_event_type": {}, "sweep": [],
    }

    for name, s in scores.items():
        results["discrimination"][name] = {
            "auprc": auprc(y_test, s),
            "ece_raw": ece(y_test, s / 20.0 if name == "NEWS2" else s),
            "ece_cal": ece(y_test, calibrated[name]),
        }
        # AUPRC against each component of the composite label
        results["by_event_type"][name] = {
            t: auprc(test[f"{LABEL}_{t}"].to_numpy(), s) for t in EVENT_TYPES
        }

    # --- lead-time sweep at matched alert budgets -------------------------
    alertable = alertable_mask(test)
    series = {name: _stay_series(test, s) for name, s in scores.items()}
    for rate in alert_rates:
        for name, s in scores.items():
            sa = s[alertable]
            thr = float(np.quantile(sa, 1.0 - rate))
            r = lead_time(*series[name][:2], thr, times=series[name][2])
            r.update(budget=f"{rate*100:.0f}%", model=name,
                     actual_rate=float((sa >= thr).mean()),
                     false_alarm_stay_rate=_false_alarm_stay_rate(series[name], thr))
            results["sweep"].append(r)

    # --- paired, volume-matched first-alarm comparison ---------------------
    # NEWS2's integer score cannot hit a fine budget exactly -- it overshoots.
    # Matching GBM to the TARGET rate would therefore let NEWS2 spend ~2x the
    # alerts and "win" on earliness for free. So GBM is matched to NEWS2's
    # ACHIEVED rate at each budget, making the comparison volume-fair.
    results["paired"] = []
    news2_a, gbm_a = news2_test[alertable], gbm_test[alertable]
    for rate in (0.01, 0.02, 0.05):
        thr_news2 = float(np.quantile(news2_a, 1.0 - rate))
        matched_rate = float((news2_a >= thr_news2).mean())
        thr_gbm = float(np.quantile(gbm_a, 1.0 - matched_rate))
        row = paired_first_alarm(test, news2_test, gbm_test, thr_news2, thr_gbm)
        row.update(
            budget=f"{rate*100:.0f}%", matched_rate=matched_rate,
            gbm_rate=float((gbm_a >= thr_gbm).mean()),
            falarm_news2=_false_alarm_stay_rate(series["NEWS2"], thr_news2),
            falarm_gbm=_false_alarm_stay_rate(series["GBM"], thr_gbm),
        )
        results["paired"].append(row)

    _print_report(results)
    return results


def _print_report(r):
    s = r["summary"]
    print(f"\nMIMIC-IV in-ICU deterioration (24h horizon), 6h stabilization")
    print(f"  hours [0, {STAB_HOURS}) enter as LOCF history, so hour {STAB_HOURS} "
          f"predicts from real data and every hour is alertable")
    print(f"  train {s['n_train_hours']:,}h | val {s['n_val_hours']:,}h (calibration) "
          f"| test {s['n_test_hours']:,}h")
    print(f"  test: {s['n_test_stays']:,} ICU stays, {s['n_test_patients']:,} patients, "
          f"prevalence {100*s['prevalence']:.2f}%\n")

    print("Discrimination + calibration (test; calibrators fit on val):")
    print(f"  {'model':<7}{'AUPRC':>8}{'ECE raw':>10}{'ECE cal':>10}")
    for name, d in r["discrimination"].items():
        print(f"  {name:<7}{d['auprc']:>8.3f}{d['ece_raw']:>10.4f}{d['ece_cal']:>10.4f}")
    print(f"  {'(random)':<7}{s['prevalence']:>8.3f}")

    print("\nAUPRC by event type (composite is dominated by ventilation):")
    types = list(EVENT_TYPES)
    print("  " + f"{'model':<7}" + "".join(f"{t:>10}" for t in types))
    for name, d in r["by_event_type"].items():
        print("  " + f"{name:<7}" + "".join(f"{d[t]:>10.3f}" for t in types))

    print("\nLead-time at matched alert budgets (test):")
    hdr = (f"  {'budget':>7}  {'model':<7}{'actual':>8}{'caught':>8}{'falarm':>8}"
           f"{'n':>7}    lead p25/p50/p75 (h)")
    print(hdr); print("  " + "-" * (len(hdr)))
    for row in r["sweep"]:
        band = ("n/a" if row["n_caught"] == 0 else
                f"{row['lead_p25']:>4.0f} /{row['median_lead_time']:>4.0f} /{row['lead_p75']:>4.0f}")
        print(f"  {row['budget']:>7}  {row['model']:<7}{100*row['actual_rate']:>7.1f}%"
              f"{row['caught_in_time_rate']:>8.2f}{row['false_alarm_stay_rate']:>8.2f}"
              f"{row['n_caught']:>7}    {band}")
    print("\n  caught = fraction of deteriorating stays alerted before the event."
          "\n  falarm = fraction of NON-deteriorating stays that alarm at least once."
          "\n           Read these together: a stay counts as caught if ANY hour crosses,"
          "\n           so at loose budgets both saturate and 'caught' stops being informative."
          "\n  NEWS2's integer score cannot hit fine budgets exactly, so its low-budget"
          "\n  rows collapse onto the same achievable threshold.")

    print("\nPAIRED first-alarm comparison -- stays caught by BOTH, GBM matched to")
    print("NEWS2's ACHIEVED alert volume.  diff = NEWS2 - GBM  (+ = GBM fires earlier):")
    hdr = (f"  {'budget':>7}{'match%':>8}{'n_pair':>8}{'median':>8}{'p25':>6}{'p75':>6}"
           f"{'GBM<':>7}{'N2<':>6}{'tie':>6}{'GBMwin':>8}{'sign_p':>10}"
           f"{'fa_N2':>7}{'fa_GBM':>8}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for row in r["paired"]:
        if row["n_paired"] == 0:
            print(f"  {row['budget']:>7}{100*row['matched_rate']:>7.1f}%"
                  f"{0:>8}     (no stays caught by both)")
            continue
        print(f"  {row['budget']:>7}{100*row['matched_rate']:>7.1f}%{row['n_paired']:>8}"
              f"{row['median_diff']:>8.1f}{row['diff_p25']:>6.0f}{row['diff_p75']:>6.0f}"
              f"{row['n_gbm_earlier']:>7}{row['n_news2_earlier']:>6}{row['n_tie']:>6}"
              f"{100*row['frac_gbm_earlier']:>7.0f}%{row['sign_p']:>10.2g}"
              f"{row['falarm_news2']:>7.2f}{row['falarm_gbm']:>8.2f}")
    print("  match% = alert volume BOTH models are held to (NEWS2's achievable rate).")
    print("  GBM< / N2< = stays where that model alarmed first; tie = same hour.")
    print("  sign_p = two-sided sign test on discordant pairs; fa_* = stay-level false alarm.")


if __name__ == "__main__":
    run()
