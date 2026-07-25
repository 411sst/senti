"""Baseline models for hourly deterioration prediction (Week-1 smoke test).

Two baselines, both scored per patient-hour and evaluated on a held-out,
patient-level test split with :mod:`metrics`:

* **NEWS2 (degraded)** -- the degraded NEWS2 score used directly as the risk
  score (see :func:`news2.degraded_news2_score`).
* **GBM (XGBoost)** -- gradient-boosted trees on the five available vitals plus
  a "hours since last measured" staleness feature per vital.

Leakage safety (CLAUDE.md rule #1): every feature at hour ``t`` uses only data
recorded STRICTLY BEFORE ``t``. This is done vectorised -- forward-fill within
patient, then shift down one row -- so no current-or-future value can leak, and
it is fast enough to run over the whole 1.5M-row cohort. Nothing here is tuned;
these are smoke-test baselines with library defaults.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import binomtest
from sklearn.isotonic import IsotonicRegression

from config import SEED
from cohort import (
    HOUR,
    LABEL_COLUMN,
    PATIENT_ID,
    load_cohort,
    split_patient_ids,
)
from metrics import auprc, ece, lead_time
from news2 import degraded_news2_score

# The five NEWS2 vitals present in PhysioNet-2019, plus FiO2 (kept only so NEWS2
# can infer supplemental oxygen -- FiO2 is NOT a GBM feature).
FIVE_VITALS = ["Resp", "O2Sat", "Temp", "SBP", "HR"]
ALL_VITALS = FIVE_VITALS + ["FiO2"]

# GBM inputs: last-observed value of each vital + how stale that value is.
GBM_FEATURES = (
    [f"{v}_locf" for v in FIVE_VITALS] + [f"{v}_hrs_since" for v in FIVE_VITALS]
)

# NEWS2 clinical high-risk threshold (aggregate score >= 5 = medium/high risk).
NEWS2_HIGH_RISK = 5


# --- Feature engineering (leakage-safe, vectorised) ------------------------

def build_features(df):
    """Build leakage-safe per-hour features from the raw cohort.

    For every row (patient, hour ``t``) we compute, using only rows with
    ``ICULOS < t``:

    * ``<vital>_locf`` -- the last observed value of the vital (LOCF).
    * ``<vital>_hrs_since`` -- hours since that vital was last measured.

    The trick is ``ffill().shift(1)`` within each patient: forward-fill carries
    the last value up to and including the current row, and the shift drops the
    current row -- leaving strictly-before-``t`` information. Missing (never
    measured before ``t``) stays ``NaN``; XGBoost handles that natively.
    """
    df = df.sort_values([PATIENT_ID, HOUR]).reset_index(drop=True)
    pid = df[PATIENT_ID]
    hour = df[HOUR]

    feat = pd.DataFrame(
        {PATIENT_ID: pid.to_numpy(), HOUR: hour.to_numpy(),
         LABEL_COLUMN: df[LABEL_COLUMN].to_numpy()}
    )

    # LOCF value, strictly before t.
    locf = df.groupby(pid, sort=False)[ALL_VITALS].ffill()
    locf = locf.groupby(pid, sort=False).shift(1)
    for v in ALL_VITALS:
        feat[f"{v}_locf"] = locf[v].to_numpy()

    # Hours since the vital was last measured, strictly before t.
    for v in FIVE_VITALS:
        measured_hour = hour.where(df[v].notna())          # hour if measured, else NaN
        last = measured_hour.groupby(pid, sort=False).ffill()
        last = last.groupby(pid, sort=False).shift(1)      # last measurement < t
        feat[f"{v}_hrs_since"] = (hour - last).to_numpy()

    return feat


def news2_risk_scores(feat):
    """Degraded NEWS2 score per row from the LOCF (before-t) vitals."""
    cols = {v: feat[f"{v}_locf"].to_numpy() for v in ALL_VITALS}
    return np.array(
        [
            degraded_news2_score(r, s, t, b, h, fio2=f)
            for r, s, t, b, h, f in zip(
                cols["Resp"], cols["O2Sat"], cols["Temp"],
                cols["SBP"], cols["HR"], cols["FiO2"],
            )
        ]
    )


# --- GBM -------------------------------------------------------------------

def train_gbm(X, y, seed=SEED):
    """Fit an XGBoost classifier with library defaults (no tuning).

    XGBoost splits on NaN natively, so the missing LOCF / staleness values need
    no imputation.
    """
    model = xgb.XGBClassifier(
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model


# --- Lead-time assembly ----------------------------------------------------

def _patient_series(feat, scores):
    """Group per-row scores into per-patient arrays for :func:`lead_time`.

    Returns (risk_scores, event_times, times): one entry per patient, with the
    event time = first hour ``SepsisLabel == 1`` (``None`` if never septic).
    """
    tmp = feat[[PATIENT_ID, HOUR, LABEL_COLUMN]].copy()
    tmp["score"] = np.asarray(scores)
    risk_scores, event_times, times = [], [], []
    for _, g in tmp.groupby(PATIENT_ID, sort=False):
        g = g.sort_values(HOUR)
        risk_scores.append(g["score"].to_numpy())
        times.append(g[HOUR].to_numpy())
        positives = g.loc[g[LABEL_COLUMN] == 1, HOUR]
        event_times.append(int(positives.min()) if len(positives) else None)
    return risk_scores, event_times, times


def _lead_at_rate(series, scores, threshold):
    """lead_time dict (+ actual alert rate) for a pre-assembled patient series.

    The per-patient series is assembled once per model; only the threshold
    changes across alert budgets, so this stays cheap to call repeatedly.
    """
    risk_scores, event_times, times = series
    result = lead_time(risk_scores, event_times, threshold, times=times)
    result["actual_rate"] = float((np.asarray(scores) >= threshold).mean())
    return result


def _first_cross_times(series, threshold):
    """Per-patient onset and first-crossing hour at ``threshold``.

    Returns two arrays aligned by patient (index ``i`` is the same patient
    across models, since both series come from the same test frame): ``onset``
    (NaN for non-septic patients) and ``first_cross`` (NaN if never crossed).
    """
    risk_scores, event_times, times = series
    onset = np.full(len(risk_scores), np.nan)
    first_cross = np.full(len(risk_scores), np.nan)
    for i, sc in enumerate(risk_scores):
        et = event_times[i]
        if et is None or (isinstance(et, float) and np.isnan(et)):
            continue  # not septic -> not in the paired population
        onset[i] = et
        t = np.asarray(times[i], dtype=float)
        order = np.argsort(t)
        crossed = np.flatnonzero(np.asarray(sc, dtype=float)[order] >= threshold)
        if crossed.size:
            first_cross[i] = t[order][crossed[0]]
    return onset, first_cross


def _paired_diff(series_news2, thr_news2, series_gbm, thr_gbm):
    """Paired first-crossing comparison on patients caught in time by BOTH.

    ``diff = first_cross_NEWS2 - first_cross_GBM``; positive means GBM fired
    earlier. Because the onset term cancels in a paired lead difference
    (``lead_GBM - lead_NEWS2 = first_cross_NEWS2 - first_cross_GBM``), this IS
    the paired lead advantage of GBM, with the onset-timing variance removed.
    Includes a two-sided sign test on the discordant pairs.
    """
    onset, fc_news2 = _first_cross_times(series_news2, thr_news2)
    _, fc_gbm = _first_cross_times(series_gbm, thr_gbm)
    caught_news2 = (~np.isnan(fc_news2)) & (fc_news2 <= onset)
    caught_gbm = (~np.isnan(fc_gbm)) & (fc_gbm <= onset)
    both = caught_news2 & caught_gbm

    diff = fc_news2[both] - fc_gbm[both]
    n = int(both.sum())
    n_gbm_earlier = int((diff > 0).sum())
    n_news2_earlier = int((diff < 0).sum())
    n_tie = int((diff == 0).sum())
    n_discordant = n_gbm_earlier + n_news2_earlier
    # Sign test: under H0 each discordant pair is 50/50; ties are dropped.
    sign_p = binomtest(n_gbm_earlier, n_discordant, 0.5).pvalue if n_discordant else 1.0

    if n:
        p25, p50, p75 = (float(np.percentile(diff, q)) for q in (25, 50, 75))
    else:
        p25 = p50 = p75 = float("nan")
    return {
        "n_paired": n,
        "median_diff": p50,
        "diff_p25": p25,
        "diff_p75": p75,
        "frac_gbm_earlier": (n_gbm_earlier / n if n else float("nan")),
        "n_gbm_earlier": n_gbm_earlier,
        "n_news2_earlier": n_news2_earlier,
        "n_tie": n_tie,
        "sign_p": float(sign_p),
    }


# --- Orchestration ---------------------------------------------------------

# Alert budgets to sweep: fraction of patient-hours flagged. Lower = scarcer,
# more selective alerts -- where a better-ranking model should pull ahead.
ALERT_RATES = [0.01, 0.02, 0.05, 0.10]

def run(df=None, data_dir="data/physionet2019", test_size=0.2, seed=SEED,
        news2_threshold=NEWS2_HIGH_RISK, alert_rates=ALERT_RATES):
    """Train on the train split, evaluate both baselines on the test split.

    Reports threshold-independent discrimination (AUPRC, ECE) and a lead-time
    sweep across matched alert budgets -- so we can see whether the models
    separate as alerts get scarcer. Prints a report and returns the results
    dict. Seeded by ``seed``.
    """
    if df is None:
        df = load_cohort(data_dir)  # cached parquet after first build

    feat = build_features(df)
    train_ids, test_ids = split_patient_ids(
        feat[PATIENT_ID].unique(), test_size=test_size, seed=seed
    )
    train = feat[feat[PATIENT_ID].isin(train_ids)]
    test = feat[feat[PATIENT_ID].isin(test_ids)]

    y_train = train[LABEL_COLUMN].to_numpy()
    y_test = test[LABEL_COLUMN].to_numpy()

    # --- Scores ---
    news2_train = news2_risk_scores(train)
    news2_test = news2_risk_scores(test)
    # Raw score drives AUPRC/lead-time (rank-based); isotonic-calibrated on TRAIN
    # for a fair ECE (the raw 0-20 score is not a probability).
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(news2_train, y_train)
    news2_test_probs = calibrator.transform(news2_test)

    model = train_gbm(train[GBM_FEATURES], y_train, seed=seed)
    gbm_test = model.predict_proba(test[GBM_FEATURES])[:, 1]

    scores = {"NEWS2": news2_test, "GBM": gbm_test}

    # --- Discrimination (threshold-independent) ---
    discrimination = {
        "NEWS2": {"auprc": auprc(y_test, news2_test),
                  "ece": ece(y_test, news2_test_probs)},
        "GBM": {"auprc": auprc(y_test, gbm_test),
                "ece": ece(y_test, gbm_test)},
    }

    # --- Lead-time across matched alert budgets ---
    # Assemble each model's per-patient series once, then re-threshold cheaply.
    # Each budget targets the same flagged fraction per model (matched via the
    # score quantile); the clinical row uses NEWS2's score>=5 cutoff.
    series = {name: _patient_series(test, s) for name, s in scores.items()}
    clinical_rate = float((news2_test >= news2_threshold).mean())

    # NEWS2's integer score can't hit a target rate exactly, so it overshoots.
    # The UNPAIRED sweep matches GBM to the *target* rate (surfacing that
    # overshoot as contrast); the PAIRED comparison matches GBM to NEWS2's
    # *actual* rate, so both flag the same fraction and "who fires earlier" is
    # not confounded by alert volume.
    specs = []
    for rate in alert_rates:
        thr_news2 = float(np.quantile(news2_test, 1.0 - rate))
        actual_news2 = float((news2_test >= thr_news2).mean())
        specs.append({
            "label": f"{rate * 100:.0f}%",
            "thr_news2": thr_news2,
            "actual_news2": actual_news2,
            "thr_gbm_target": float(np.quantile(gbm_test, 1.0 - rate)),
            "thr_gbm_matched": float(np.quantile(gbm_test, 1.0 - actual_news2)),
        })
    specs.append({
        "label": "clinical",
        "thr_news2": float(news2_threshold),
        "actual_news2": clinical_rate,
        "thr_gbm_target": float(np.quantile(gbm_test, 1.0 - clinical_rate)),
        "thr_gbm_matched": float(np.quantile(gbm_test, 1.0 - clinical_rate)),
    })

    sweep, paired = [], []
    for spec in specs:
        sweep.append({"budget": spec["label"], "model": "NEWS2",
                      **_lead_at_rate(series["NEWS2"], news2_test, spec["thr_news2"])})
        sweep.append({"budget": spec["label"], "model": "GBM",
                      **_lead_at_rate(series["GBM"], gbm_test, spec["thr_gbm_target"])})
        row = _paired_diff(series["NEWS2"], spec["thr_news2"],
                           series["GBM"], spec["thr_gbm_matched"])
        row["budget"] = spec["label"]
        row["matched_rate"] = spec["actual_news2"]
        row["gbm_actual_rate"] = float((gbm_test >= spec["thr_gbm_matched"]).mean())
        paired.append(row)

    results = {
        "summary": {
            "n_hours": int(len(y_test)),
            "n_patients": int(test[PATIENT_ID].nunique()),
            "n_events": int(sweep[0]["n_events"]),
            "prevalence": float(y_test.mean()),
            "clinical_rate": clinical_rate,
            "news2_threshold": news2_threshold,
        },
        "discrimination": discrimination,
        "lead_time_sweep": sweep,
        "paired_lead_time": paired,
    }
    _print_report(results)
    return results


def _fmt_band(row):
    """p25 / p50 / p75 lead band, NaN-safe."""
    if row["n_caught"] == 0:
        return "n/a (0 caught)"
    return (f"{row['lead_p25']:>3.0f} /{row['median_lead_time']:>4.0f} /"
            f"{row['lead_p75']:>4.0f}")


def _print_report(results):
    s = results["summary"]
    print(
        f"\nTest set: {s['n_hours']:,} patient-hours, {s['n_patients']:,} patients, "
        f"{s['n_events']:,} septic; positive-hour prevalence {s['prevalence'] * 100:.2f}%"
    )
    print("  NEWS2 = degraded NEWS2 score;  GBM = XGBoost on 5 vitals + staleness\n")

    print("Discrimination (threshold-independent):")
    print(f"  {'model':<6}{'AUPRC':>8}{'ECE':>8}")
    for name, d in results["discrimination"].items():
        print(f"  {name:<6}{d['auprc']:>8.3f}{d['ece']:>8.3f}")

    print("\nLead-time by alert budget (scarcer alerts first; matched rate per budget):")
    hdr = (f"  {'budget':>8}  {'model':<6}{'actual':>8}{'caught':>8}{'n':>6}"
           f"    {'lead p25/ p50/ p75 (h)'}")
    print(hdr)
    print("  " + "-" * (len(hdr) + 2))
    clinical_rate = s["clinical_rate"]
    for row in results["lead_time_sweep"]:
        if row["budget"] == "clinical":
            budget = f"{clinical_rate * 100:.1f}%*"
        else:
            budget = row["budget"]
        print(
            f"  {budget:>8}  {row['model']:<6}{row['actual_rate'] * 100:>7.1f}%"
            f"{row['caught_in_time_rate']:>8.2f}{row['n_caught']:>6}    {_fmt_band(row)}"
        )
    print(
        f"\n* clinical = NEWS2 cutoff score>={s['news2_threshold']} "
        f"(GBM matched to that {clinical_rate * 100:.1f}% rate)."
        "\n  caught = fraction of septic patients alerted before onset; "
        "n = patients entering the band;"
        "\n  lead = hours before sepsis onset (p25/p50/p75)."
    )

    print("\nPAIRED: patients caught by BOTH, GBM matched to NEWS2's ACTUAL rate; "
          "diff = first_cross_NEWS2 - first_cross_GBM  (+ = GBM earlier):")
    hdr = (f"  {'budget':>8}{'match%':>8}{'n_pair':>7}{'median':>8}{'p25':>6}{'p75':>6}"
           f"{'GBMwin%':>9}{'sign_p':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for row in results["paired_lead_time"]:
        budget = f"{clinical_rate * 100:.1f}%*" if row["budget"] == "clinical" else row["budget"]
        if row["n_paired"] == 0:
            print(f"  {budget:>8}{row['matched_rate'] * 100:>7.1f}%{0:>7}"
                  "     (no patients caught by both)")
            continue
        print(
            f"  {budget:>8}{row['matched_rate'] * 100:>7.1f}%{row['n_paired']:>7}"
            f"{row['median_diff']:>8.0f}{row['diff_p25']:>6.0f}{row['diff_p75']:>6.0f}"
            f"{row['frac_gbm_earlier'] * 100:>8.0f}%{row['sign_p']:>9.2g}"
        )
    print(
        "  match% = actual alert rate BOTH models are held to (NEWS2's, since it "
        "overshoots the target).\n  median/p25/p75 in hours (+ = GBM earlier); "
        "GBMwin% = fraction strictly earlier; sign_p =\n  two-sided sign test on "
        "discordant pairs (small => beyond chance)."
    )


if __name__ == "__main__":
    run()
