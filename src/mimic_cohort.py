"""MIMIC-IV in-ICU hourly deterioration cohort: features, NEWS2, splits.

Consumes the raw hourly bins fetched by ``scripts/fetch_mimic_cohort.py`` and
produces the compact per-hour table the rest of the pipeline trains on.

What happens here (and why):

* **Leakage safety.** The raw file holds values *observed in* hour ``t``. Every
  feature is built by forward-filling within a stay and then shifting one row,
  so the row for hour ``t`` only ever sees hours ``< t`` -- the same
  ``before_time_t`` discipline used on PhysioNet-2019, applied vectorised.
* **NEWS2.** Scored from those strictly-before-``t`` values using the bands in
  :mod:`news2`. The scoring here is vectorised for 2.5M rows;
  ``tests/test_mimic_cohort.py`` asserts it agrees exactly with the scalar
  reference implementation, so the fast path can't silently drift.
* **Splits.** By ``subject_id`` (patient), never by stay or row -- one patient's
  ICU stays never straddle train and test. Deterministic from ``SEED``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cohort import split_patient_ids_three_way
from config import SEED
from news2 import (
    FIO2_ROOM_AIR_PCT,
    O2_FLOW_MAX_LPM,
    score_heart_rate,
    score_respiratory_rate,
    score_spo2,
    score_systolic_bp,
    score_temperature,
)

RAW_PATH = "data/cache/mimic_raw_hourly.parquet"
OUT_PATH = "data/cache/mimic_cohort_hourly.parquet"

# Stabilization window (hours). Must match --stab-hours in
# scripts/fetch_mimic_cohort.py. Predictions start at this hour; the raw
# extract additionally carries hours [0, STAB_HOURS) as HISTORY rows
# (is_history = 1) so that after the strictly-before-t shift the first
# prediction row already holds real carried-forward context. History rows are
# dropped in :func:`build` after feature construction -- they are never
# prediction opportunities and never reach the exported cohort.
STAB_HOURS = 6

PATIENT_ID = "patient_id"
STAY_ID = "stay_id"
HOUR = "hour"
LABEL = "label_24h"

# Vitals carried forward. These are the five NEWS2 vitals available in MIMIC.
VITALS = ["heart_rate", "resp_rate", "spo2", "sbp", "temperature"]
# Other channels that also need the strictly-before-t treatment.
CARRIED = VITALS + ["gcs", "gcs_verbal", "gcs_motor", "gcs_eyes", "gcs_unable",
                    "o2_flow", "o2_device", "fio2"]

# How long a carried-forward value may still be considered current, per channel.
# Past this, the value is dropped (treated as never measured) rather than
# propagated for the rest of the stay.
#
# Why this matters: NEWS2 is meant to describe the patient NOW. An unbounded
# carry-forward let a single charted oxygen device stretch to a p90 of 33h and a
# max of 328h -- claiming a patient was on oxygen from a note days old. Each
# bound below is ~2x that channel's observed p90 charting gap, so routine
# charting survives untouched and only genuinely stale values are dropped:
#
#   channel      p50 gap  p90 gap   bound
#   vitals            1h       1h     12h   (hourly; 12h means monitoring stopped)
#   gcs               3h       5h     12h   (q3-4h neuro checks)
#   o2_device/flow    4h       6h      8h   (q4h respiratory assessment)
#   fio2              6h      26h     24h   (blood gas, genuinely infrequent)
MAX_STALENESS_HOURS = {
    **{v: 12 for v in VITALS},
    "gcs": 12, "gcs_verbal": 12, "gcs_motor": 12, "gcs_eyes": 12, "gcs_unable": 12,
    "o2_device": 8, "o2_flow": 8,
    "fio2": 24,
}

# NEWS2 bands as (right-edge boundaries, score per band). Each scalar helper in
# news2.py is an `if v <= b0 ... elif v <= b1 ...` chain, which is exactly
# `searchsorted(boundaries, v, side="left")` indexed into the scores -- so these
# tables mirror the helpers rather than reinterpreting them.
BANDS = {
    "resp_rate":   ([8, 11, 20, 24],          [3, 1, 0, 2, 3]),
    "spo2":        ([91, 93, 95],             [3, 2, 1, 0]),
    "temperature": ([35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2]),
    "sbp":         ([90, 100, 110, 219],      [3, 2, 1, 0, 3]),
    "heart_rate":  ([40, 50, 90, 110, 130],   [3, 1, 0, 1, 2, 3]),
}
# The scalar function each band table must reproduce (checked by tests).
SCALAR_SCORERS = {
    "resp_rate": score_respiratory_rate,
    "spo2": score_spo2,
    "temperature": score_temperature,
    "sbp": score_systolic_bp,
    "heart_rate": score_heart_rate,
}


# --- Leakage-safe features -------------------------------------------------

def build_before_t_features(df, max_staleness=None):
    """Carry each channel forward within a stay, then shift it out of the present.

    ``groupby(stay).ffill()`` propagates the last observed value up to and
    including the current row; ``.shift(1)`` then drops the current row, leaving
    only information from hours strictly before ``t``. Grouping by ``stay_id``
    (not patient) stops one ICU stay's values bleeding into the next.

    Each channel also gets ``<channel>_hrs_since`` -- how old that carried value
    is -- and any value older than its :data:`MAX_STALENESS_HOURS` bound is
    dropped, so a long-past observation cannot masquerade as the current state.
    """
    if max_staleness is None:
        max_staleness = MAX_STALENESS_HOURS

    out = df.sort_values([STAY_ID, HOUR]).reset_index(drop=True)
    stay = out[STAY_ID]
    hour = out[HOUR]

    for col in CARRIED:
        observed = out[col]
        # Last value seen strictly before t.
        value = observed.groupby(stay, sort=False).ffill()
        value = value.groupby(stay, sort=False).shift(1)
        # Hour that value was recorded, also strictly before t.
        last_hour = hour.where(observed.notna()).groupby(stay, sort=False).ffill()
        last_hour = last_hour.groupby(stay, sort=False).shift(1)
        staleness = hour - last_hour

        bound = max_staleness.get(col)
        if bound is not None:
            value = value.where(staleness <= bound)      # too old -> unknown
            staleness = staleness.where(staleness <= bound)

        out[col] = value.to_numpy()
        out[f"{col}_hrs_since"] = staleness.to_numpy()

    return out


# --- Per-event-type labels -------------------------------------------------

# Each component of the composite deterioration label, and the raw column
# holding its exact hour-offset from ICU admission.
EVENT_TYPES = {"vent": "vent_h", "vaso": "vaso_h", "death": "death_h"}


def add_event_labels(df, horizon=24):
    """Add a per-component 24h label plus the first-event type.

    ``label_24h`` (already in the raw extract) is the composite: ANY component
    within the horizon. These break it out so we can see which kind of
    deterioration a model is actually catching -- ventilation dominates the
    cohort, and a model that only predicts intubation should not be able to hide
    behind the composite.

    Note a component can fire inside the horizon even though the stay was
    censored at a different, earlier component; each label is computed
    independently, so they are not mutually exclusive.
    """
    out = df.copy()
    hour = out[HOUR]
    for name, col in EVENT_TYPES.items():
        t = out[col]
        out[f"label_24h_{name}"] = (
            t.notna() & (t > hour) & (t <= hour + horizon)
        ).astype(np.int8)

    # Which component was the FIRST event for this stay (censoring point).
    event_type = pd.Series(pd.NA, index=out.index, dtype="object")
    for name, col in EVENT_TYPES.items():
        is_first = out["event_h"].notna() & (out[col] == out["event_h"])
        event_type = event_type.mask(is_first & event_type.isna(), name)
    out["event_type"] = event_type
    return out


# --- Vectorised NEWS2 (verified against the scalar helpers in tests) -------

def _band_score(values, boundaries, scores, missing_score=0):
    """Score a vital by its NEWS2 bands; missing values get ``missing_score``."""
    v = np.asarray(values, dtype=float)
    idx = np.searchsorted(np.asarray(boundaries, dtype=float), v, side="left")
    out = np.asarray(scores, dtype=float)[idx]
    return np.where(np.isnan(v), float(missing_score), out)


def alert_flags(gcs, gcs_verbal, gcs_motor, gcs_eyes, gcs_unable):
    """Vectorised :func:`news2.is_alert_from_gcs`. Returns float 1/0/NaN.

    Intubated (``gcs_unable == 1``): verbal is not assessable, so alertness is
    judged on eyes == 4 AND motor == 6. Otherwise the standard total GCS >= 15.
    See the scalar version for the full rationale.
    """
    gcs, gv, gm, ge, gu = (np.asarray(x, dtype=float)
                           for x in (gcs, gcs_verbal, gcs_motor, gcs_eyes, gcs_unable))
    intubated = (gu == 1)

    with np.errstate(invalid="ignore"):
        by_components = (ge + gv + gm) >= 15
        by_total = gcs >= 15
        by_eyes_motor = (ge == 4) & (gm == 6)

    have_components = ~np.isnan(ge) & ~np.isnan(gv) & ~np.isnan(gm)
    have_eyes_motor = ~np.isnan(ge) & ~np.isnan(gm)
    have_total = ~np.isnan(gcs)

    out = np.full(gcs.shape, np.nan)
    # Not intubated: components if available, else the total.
    out = np.where(~intubated & have_total, by_total.astype(float), out)
    out = np.where(~intubated & have_components, by_components.astype(float), out)
    # Intubated: eyes+motor if available, else fall back to the total.
    out = np.where(intubated & have_total, by_total.astype(float), out)
    out = np.where(intubated & have_eyes_motor, by_eyes_motor.astype(float), out)
    return out


def oxygen_flags(o2_device, o2_flow, fio2):
    """Vectorised :func:`news2.supplemental_oxygen_flag` (with cleaning).

    FiO2 is normalised to a percentage and O2 flow clipped to a physiological
    range first, so charting junk (FiO2 up to 40100, flow down to -1) cannot
    manufacture an oxygen requirement. Nothing charted => room air.
    """
    dev, flow, f = (np.asarray(x, dtype=float) for x in (o2_device, o2_flow, fio2))

    # FiO2: 0.21-1.0 is a fraction, 21-100 already a percentage, else junk.
    pct = np.where((f >= 0.21) & (f <= 1.0), f * 100.0,
                   np.where((f >= FIO2_ROOM_AIR_PCT) & (f <= 100.0), f, np.nan))
    clean_flow = np.where((flow >= 0.0) & (flow <= O2_FLOW_MAX_LPM), flow, np.nan)

    with np.errstate(invalid="ignore"):
        return ((np.nan_to_num(dev, nan=0.0) > 0)
                | (np.nan_to_num(clean_flow, nan=0.0) > 0)
                | (np.nan_to_num(pct, nan=0.0) > FIO2_ROOM_AIR_PCT))


# The seven NEWS2 parameters, in the order news2_component_scores returns them.
NEWS2_COMPONENTS = list(BANDS) + ["oxygen", "consciousness"]


def news2_component_scores(df, missing_vital_score=0, assume_alert_if_unknown=True):
    """Per-parameter NEWS2 sub-scores: ``(N, 7)`` array, plus the two flags.

    Returning the breakdown rather than only the total lets the NEWS2 agent
    expose *which* parameters are driving a score -- that vector is its latent.

    Consciousness and oxygen come from the derived ``is_alert``/``on_oxygen``
    columns when the frame carries them (the exported cohort does, but not the
    raw GCS/FiO2 channels they were built from); otherwise they are recomputed
    from the raw channels. Either path uses the same underlying logic.
    """
    columns = [_band_score(df[col], bounds, scores, missing_vital_score)
               for col, (bounds, scores) in BANDS.items()]

    if "is_alert" in df.columns:
        alert = pd.to_numeric(df["is_alert"], errors="coerce").to_numpy(dtype=float)
    else:
        alert = alert_flags(df["gcs"], df["gcs_verbal"], df["gcs_motor"],
                            df["gcs_eyes"], df["gcs_unable"])
    if "on_oxygen" in df.columns:
        on_oxygen = pd.to_numeric(df["on_oxygen"], errors="coerce").to_numpy() > 0
    else:
        on_oxygen = oxygen_flags(df["o2_device"], df["o2_flow"], df["fio2"])

    unknown_points = 0 if assume_alert_if_unknown else missing_vital_score
    consciousness = np.where(np.isnan(alert), unknown_points,
                             np.where(alert == 1, 0, 3))

    columns.append(np.asarray(on_oxygen, dtype=float) * 2)   # oxygen: 0 or 2
    columns.append(np.asarray(consciousness, dtype=float))   # 0 or 3
    return np.stack(columns, axis=1), alert, on_oxygen


def compute_news2(df, missing_vital_score=0, assume_alert_if_unknown=True):
    """Full 7-parameter NEWS2 for every row, from before-t values."""
    components, alert, on_oxygen = news2_component_scores(
        df, missing_vital_score, assume_alert_if_unknown
    )
    return components.sum(axis=1), alert, on_oxygen


# --- Orchestration ---------------------------------------------------------

def build(raw_path=RAW_PATH, out_path=OUT_PATH, val_size=0.15, test_size=0.15,
          seed=SEED):
    """Build the compact per-hour table and write it to Parquet."""
    raw = pd.read_parquet(raw_path)
    raw = raw.rename(columns={"subject_id": PATIENT_ID})

    # BigQuery hands back NUMERIC as decimal.Decimal and INT64 as pandas Int64;
    # coerce every measurement channel to plain float so arithmetic, NaN
    # handling and downstream model libraries all behave.
    for col in CARRIED + ["event_h", "death_h", "vent_h", "vaso_h"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").astype(float)

    feat = build_before_t_features(raw)
    feat = add_event_labels(feat)
    news2, alert, on_oxygen = compute_news2(feat)
    feat["news2"] = news2.astype(np.int16)
    feat["is_alert"] = alert
    feat["on_oxygen"] = on_oxygen

    # History rows [0, STAB_HOURS) exist so the ffill+shift above could carry
    # their observations INTO the first prediction hour; they are not
    # prediction rows themselves and leave the table here. Raw extracts from
    # before the h6 root-cause fix have no is_history column and no history
    # rows, so there is nothing to drop.
    if "is_history" in feat.columns:
        feat = feat[pd.to_numeric(feat["is_history"]) == 0].reset_index(drop=True)

    # Patient-level three-way split: train / val (calibration) / test.
    # A patient's rows -- and therefore all of their ICU stays -- land in
    # exactly one of the three.
    train_ids, val_ids, test_ids = split_patient_ids_three_way(
        feat[PATIENT_ID].unique(), val_size=val_size, test_size=test_size, seed=seed
    )
    split = np.full(len(feat), "train", dtype=object)
    split[feat[PATIENT_ID].isin(set(val_ids)).to_numpy()] = "val"
    split[feat[PATIENT_ID].isin(set(test_ids)).to_numpy()] = "test"
    feat["split"] = split

    keep = ([PATIENT_ID, STAY_ID, HOUR] + VITALS
            + [f"{v}_hrs_since" for v in VITALS]
            + ["gcs", "gcs_unable", "is_alert", "on_oxygen", "news2", "event_h",
               "event_type", LABEL]
            + [f"{LABEL}_{n}" for n in EVENT_TYPES] + ["split"])
    out = feat[keep]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out
