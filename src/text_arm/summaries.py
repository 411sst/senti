"""Templated agent summaries -- the text arm of the ablation.

This module is the *only* thing that differs between the two arms of the
decisive experiment. The latent orchestrator receives each agent's embedding
vector; the text orchestrator receives these sentences. Everything upstream
(cohort, features, leakage discipline, the agents' own calibrated risks) and
everything downstream (calibration, evaluation) is shared, so any difference in
results is attributable to the **communication medium**.

Two deliberate design choices:

* **Templated, not LLM-generated.** Each agent's summary is produced by a
  deterministic function of its own numbers. Putting an LLM at the agent level
  would inject sampling variability into three places at once and confound the
  thing being measured. Templates keep the agents identical across arms.
* **Same information, different encoding.** Every word here is derived from the
  same strictly-before-``t`` features the latent agents consume, and the risk
  word is a binning of the agent's own isotonic-calibrated risk. The text arm is
  therefore not handicapped by seeing *less*; it sees the same evidence written
  out in words. What it loses is precision and dimensionality -- which is
  exactly the thesis.
"""

import numpy as np
import pandas as pd

from mimic_cohort import NEWS2_COMPONENTS, VITALS

# How far back to look when describing a trend. Both endpoints are
# before-t features, so the comparison is leakage-safe.
TREND_LAG_HOURS = 4

# Vitals use ABSOLUTE change thresholds, one per channel. A single relative
# threshold does not work across these ranges: SpO2 lives in 90-100, so a 10%
# relative move (97 -> 87) is nearly never reached and every desaturation would
# be described as "stable" -- silently deleting the strongest respiratory signal
# from the text arm and handicapping the ablation. These are set at roughly the
# smallest clinically meaningful move for each channel.
VITAL_ABS_THRESHOLD = {
    "heart_rate": 10.0,     # bpm
    "resp_rate": 4.0,       # breaths/min
    "spo2": 2.0,            # percentage points
    "sbp": 10.0,            # mmHg
    "temperature": 0.5,     # degrees C
}
# Labs keep a RELATIVE threshold: analyte scales differ by orders of magnitude
# (lactate ~1-10, platelets ~150k), so proportional change is the common unit.
LAB_TREND_THRESHOLD = 0.15

# Risk wording, applied to each agent's own calibrated probability. Fixed cut
# points (not quantiles) so the mapping is stable across splits and arms.
RISK_BANDS = [(0.02, "low"), (0.05, "moderate"), (0.15, "high")]
RISK_TOP = "very high"

VITAL_WORDS = {
    "heart_rate": "Heart rate", "resp_rate": "Resp rate", "spo2": "SpO2",
    "sbp": "Systolic BP", "temperature": "Temperature",
}
LAB_WORDS = {
    "lab_lactate": "Lactate", "lab_creatinine": "Creatinine",
    "lab_wbc": "WBC", "lab_platelets": "Platelets", "lab_bun": "BUN",
}


def risk_word(risk):
    """Bin a calibrated probability into a word."""
    if risk is None or (isinstance(risk, float) and np.isnan(risk)):
        return "unknown"
    for edge, word in RISK_BANDS:
        if risk < edge:
            return word
    return RISK_TOP


def trend_word(now, before, threshold, relative=False):
    """Describe a change between two before-t observations.

    ``threshold`` is an absolute delta unless ``relative`` is set, in which case
    it is a fraction of the earlier value (see the constants above for why the
    two channel families differ).
    """
    if now is None or before is None:
        return "unknown"
    if (isinstance(now, float) and np.isnan(now)) or (
            isinstance(before, float) and np.isnan(before)):
        return "unknown"
    if relative:
        if before == 0:
            return "stable"
        change = (now - before) / abs(before)
    else:
        change = now - before
    if change > threshold:
        return "rising"
    if change < -threshold:
        return "falling"
    return "stable"


def _fmt(value, digits=0):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unmeasured"
    return f"{value:.{digits}f}"


def vitals_summary(current, past, risk, include_ranges=True):
    """One sentence about the vitals trajectory.

    ``current`` and ``past`` are the before-t feature rows at hour ``t`` and
    hour ``t - TREND_LAG_HOURS``; both are LOCF'd, strictly-before-t values.

    ``include_ranges`` controls the "(78->92 over 4h)" detail on a moving heart
    rate. The LLM arm keeps it (default); the encoded-text channel experiment
    drops it, because the before->now cross-product explodes the number of
    distinct sentences without adding information beyond the trend word.
    """
    parts = []
    for col in ("heart_rate", "spo2", "resp_rate", "sbp"):
        now, before = current.get(col), (past.get(col) if past is not None else None)
        trend = trend_word(now, before, VITAL_ABS_THRESHOLD[col])
        if include_ranges and col == "heart_rate" and trend in ("rising", "falling"):
            parts.append(f"{VITAL_WORDS[col]} {trend} "
                         f"({_fmt(before)}->{_fmt(now)} over {TREND_LAG_HOURS}h)")
        else:
            parts.append(f"{VITAL_WORDS[col]} {trend} ({_fmt(now)})")
    return f"Vitals: {', '.join(parts)}. Risk: {risk_word(risk)}."


def labs_summary(current, past, risk):
    """One sentence about the lab panel."""
    parts = []
    for col in ("lab_lactate", "lab_creatinine", "lab_wbc", "lab_bun"):
        now, before = current.get(col), (past.get(col) if past is not None else None)
        trend = trend_word(now, before, LAB_TREND_THRESHOLD, relative=True)
        digits = 1 if col == "lab_lactate" else 0
        if trend == "unknown" and (now is None or (isinstance(now, float) and np.isnan(now))):
            parts.append(f"{LAB_WORDS[col]} unmeasured")
        else:
            parts.append(f"{LAB_WORDS[col]} {_fmt(now, digits)} ({trend})")
    return f"Labs: {', '.join(parts)}. Risk: {risk_word(risk)}."


def news2_summary(score, components, risk):
    """One sentence about the NEWS2 score and what is driving it.

    ``components`` is the NEWS2 agent's 7-dim latent -- the per-parameter
    sub-scores -- so the text arm names the same drivers the latent arm passes
    as numbers.
    """
    named = [(NEWS2_COMPONENTS[i], float(v)) for i, v in enumerate(components)]
    drivers = [name for name, v in sorted(named, key=lambda kv: -kv[1]) if v > 0][:2]
    driver_text = (" and ".join(w.replace("_", " ") for w in drivers)
                   if drivers else "no single parameter")
    return (f"NEWS2: score {int(round(score))}, driven by {driver_text}. "
            f"Risk: {risk_word(risk)}.")


def build_summaries(cohort, lab_features, news2_latent, risks,
                    trend_lag=TREND_LAG_HOURS, include_ranges=True):
    """Build all three agents' summaries for every cohort row.

    Parameters
    ----------
    cohort : DataFrame
        Cohort rows (before-t vitals, ``news2``), sorted by ``(stay_id, hour)``.
    lab_features : DataFrame
        Before-t lab values aligned row-for-row with ``cohort``.
    news2_latent : ndarray (N, 7)
        The NEWS2 agent's per-parameter sub-scores.
    risks : dict
        ``{"vitals_tcn": array, "labs": array, "news2": array}`` -- each agent's
        calibrated risk, aligned row-for-row.

    Returns
    -------
    DataFrame with ``stay_id``, ``hour`` and one text column per agent.
    """
    df = cohort.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    labs = lab_features.reset_index(drop=True)

    # The row `trend_lag` hours earlier WITHIN the same stay (NaN at the start
    # of a stay, which the templates render as "unknown").
    shifted_idx = _lagged_row_index(df, trend_lag)

    out = {"stay_id": df["stay_id"].to_numpy(), "hour": df["hour"].to_numpy()}
    vit_txt, lab_txt, n2_txt = [], [], []
    vit_cols = list(VITALS)
    lab_cols = [c for c in labs.columns if c.startswith("lab_")
                and not c.endswith("_hrs_since")]

    vit_now = df[vit_cols].to_dict("records")
    lab_now = labs[lab_cols].to_dict("records") if lab_cols else [{}] * len(df)
    news2_scores = df["news2"].to_numpy(dtype=float)

    for i in range(len(df)):
        j = shifted_idx[i]
        vit_past = vit_now[j] if j >= 0 else None
        lab_past = lab_now[j] if j >= 0 else None
        vit_txt.append(vitals_summary(vit_now[i], vit_past, risks["vitals_tcn"][i],
                                      include_ranges=include_ranges))
        lab_txt.append(labs_summary(lab_now[i], lab_past, risks["labs"][i]))
        n2_txt.append(news2_summary(news2_scores[i], news2_latent[i], risks["news2"][i]))

    out["vitals_text"] = vit_txt
    out["labs_text"] = lab_txt
    out["news2_text"] = n2_txt
    return pd.DataFrame(out)


def _lagged_row_index(df, lag):
    """Row index of ``hour - lag`` within the same stay, or -1 if absent.

    Only ever looks BACKWARD, so the trend endpoints are both at hours <= t.
    """
    key = pd.MultiIndex.from_arrays([df["stay_id"].to_numpy(),
                                     df["hour"].to_numpy()])
    want = pd.MultiIndex.from_arrays([df["stay_id"].to_numpy(),
                                      df["hour"].to_numpy() - lag])
    return key.get_indexer(want)
