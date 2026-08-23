"""Torch-free feature builders shared by the agents and the fair baselines.

These two functions were born inside ``agents/labs_agent.py`` and
``agents/notes_agent.py``. They are pure pandas/numpy, but those modules import
torch at the top -- and torch and XGBoost cannot share a process on macOS (each
bundles its own OpenMP runtime; see ``tests/conftest.py``). The equal-information
GBM baselines in ``baselines_fair.py`` need exactly these features next to
XGBoost, so the builders live here and the agents re-import them. Behaviour is
unchanged; the agents' own tests still exercise both functions.
"""

import numpy as np
import pandas as pd

# --- Labs -------------------------------------------------------------------

LABS_PATH = "data/cache/mimic_labs_hourly.parquet"

# The 24-analyte panel, in the order the extract produces them.
LAB_COLUMNS = [
    "lab_creatinine", "lab_bun", "lab_sodium", "lab_potassium", "lab_chloride",
    "lab_bicarbonate", "lab_anion_gap", "lab_glucose", "lab_calcium",
    "lab_magnesium", "lab_phosphate", "lab_hematocrit", "lab_hemoglobin",
    "lab_wbc", "lab_platelets", "lab_rdw", "lab_alt", "lab_ast",
    "lab_bilirubin", "lab_albumin", "lab_inr", "lab_ptt", "lab_lactate",
    "lab_ph",
]

# How long a lab result stays informative. Chemistry and haematology are drawn
# roughly daily in the ICU, so a 24h bound keeps the usual cadence while
# stopping a week-old creatinine from being read as the current one -- the same
# staleness discipline applied to vitals in mimic_cohort.py.
LAB_MAX_STALENESS_HOURS = 24

# Labs are drawn from 24h before ICU admission (ED bloods are real history).
LAB_LOOKBACK_HOURS = 24


def load_labs(path=LABS_PATH):
    return pd.read_parquet(path)


def build_lab_features(cohort, labs, max_staleness=LAB_MAX_STALENESS_HOURS):
    """Leakage-safe lab features aligned to the cohort's (stay_id, hour) rows.

    Builds a dense hourly grid per stay running from ``-LAB_LOOKBACK_HOURS`` to
    the stay's last prediction hour, forward-fills each analyte along it, shifts
    one hour so the present is excluded, drops values older than
    ``max_staleness``, and finally selects the cohort's rows.

    The grid has to extend before hour 0 because the cohort starts at hour 6 and
    a lab drawn in the ED must still be visible there.

    Returns a frame with one row per cohort row: the carried lab values plus a
    ``<lab>_hrs_since`` staleness channel for each.
    """
    cohort_keys = cohort[["stay_id", "hour"]].drop_duplicates()
    span = cohort_keys.groupby("stay_id")["hour"].max().reset_index(name="max_hour")

    # Dense grid: one row per (stay, hour) from -lookback to the stay's end.
    grid = span.assign(
        hour=[np.arange(-LAB_LOOKBACK_HOURS, int(m) + 1) for m in span["max_hour"]]
    ).explode("hour")[["stay_id", "hour"]]
    grid["hour"] = grid["hour"].astype(int)

    labs = labs[labs["hour"].between(-LAB_LOOKBACK_HOURS, int(span["max_hour"].max()))]
    merged = grid.merge(labs, on=["stay_id", "hour"], how="left")
    merged = merged.sort_values(["stay_id", "hour"]).reset_index(drop=True)

    stay, hour = merged["stay_id"], merged["hour"]
    out = merged[["stay_id", "hour"]].copy()
    for col in LAB_COLUMNS:
        observed = merged[col] if col in merged else pd.Series(np.nan, index=merged.index)
        value = observed.groupby(stay, sort=False).ffill()
        value = value.groupby(stay, sort=False).shift(1)          # strictly before t
        last = hour.where(observed.notna()).groupby(stay, sort=False).ffill()
        last = last.groupby(stay, sort=False).shift(1)
        staleness = hour - last
        out[col] = value.where(staleness <= max_staleness).to_numpy()
        out[f"{col}_hrs_since"] = staleness.where(staleness <= max_staleness).to_numpy()

    # Keep only the cohort's prediction hours, in cohort key order.
    return cohort_keys.merge(out, on=["stay_id", "hour"], how="left") \
                      .sort_values(["stay_id", "hour"]).reset_index(drop=True)


# --- Notes (radiology reports) -----------------------------------------------

NOTE_EMBED_CACHE = "data/cache/note_embeddings.npz"
NOTE_INDEX_PATH = "data/cache/note_index.parquet"
NOTE_STALENESS_HOURS = 72


def assemble_note_features(cohort, note_index, embeddings, note_ids,
                           staleness=NOTE_STALENESS_HOURS):
    """Per patient-hour: the most recent note AVAILABLE strictly before t.

    ``allow_exact_matches=False`` excludes a note that became available exactly
    at hour ``t``; ``direction="backward"`` looks only into the past. Anything
    older than ``staleness`` is dropped, so a week-old report is treated as no
    information rather than as current.

    The embedding width is taken from ``embeddings`` itself, so callers may pass
    the raw 768-dim ClinicalBERT matrix or any reduced projection of it.

    Returns ``(embeddings (N, D), has_note (N,), age (N,), keys)`` in
    ``(stay_id, hour)`` order.
    """
    embeddings = np.asarray(embeddings)
    lookup = {str(n): i for i, n in enumerate(note_ids)}

    left = cohort[["stay_id", "hour"]].copy()
    left["stay_id"] = left.stay_id.astype("int64")
    left["hour"] = left.hour.astype("float64")
    right = note_index[["stay_id", "note_hour", "note_id"]].copy()
    right["stay_id"] = right.stay_id.astype("int64")
    right["note_hour"] = right.note_hour.astype("float64")

    merged = pd.merge_asof(
        left.sort_values("hour").reset_index(drop=True),
        right.sort_values("note_hour").reset_index(drop=True),
        left_on="hour", right_on="note_hour", by="stay_id",
        direction="backward", allow_exact_matches=False,
    ).sort_values(["stay_id", "hour"]).reset_index(drop=True)

    age = (merged["hour"] - merged["note_hour"]).to_numpy(dtype=float)
    usable = merged["note_id"].notna().to_numpy() & (age <= staleness)

    out = np.zeros((len(merged), embeddings.shape[1]), dtype=np.float32)
    has = np.zeros(len(merged), dtype=np.float32)
    for i in np.flatnonzero(usable):
        idx = lookup.get(str(merged["note_id"].iloc[i]))
        if idx is None:
            continue                      # note not encoded -> treat as absent
        out[i] = embeddings[idx]
        has[i] = 1.0
    age = np.where(has > 0, age, 0.0)
    return out, has, age.astype(np.float32), merged[["stay_id", "hour"]].astype("int64")


# --- Measurement cadence ------------------------------------------------------

CADENCE_PATH = "data/cache/mimic_cadence_hourly.parquet"
CADENCE_CHANNELS = [
    "n_lab_draws", "n_lab_tests", "n_chart_rows", "n_chart_items",
    "n_orders_med", "n_orders_lab", "n_orders_imaging", "n_orders_other",
    "n_micro",
]
CADENCE_LOOKBACK_HOURS = 24
# The three channels whose short-term acceleration is the "clinician worry"
# signal (CONCERN's construct): lab draws, charting intensity, med orders.
CADENCE_ACCEL_CHANNELS = ["n_lab_draws", "n_chart_rows", "n_orders_med"]

CADENCE_FEATURES = (
    [f"{c}_1h" for c in CADENCE_CHANNELS]
    + [f"{c}_6h" for c in CADENCE_CHANNELS]
    + [f"{c}_24h" for c in CADENCE_CHANNELS]
    + [f"{c}_accel" for c in CADENCE_ACCEL_CHANNELS]
    + ["hrs_since_lab_draw", "hrs_since_med_order"]
)


def load_cadence(path=CADENCE_PATH):
    return pd.read_parquet(path)


def _dense_shifted_grid(cohort, events, columns, lookback):
    """Dense per-stay hourly grid of event columns, zero-filled and shifted.

    Counts differ from lab VALUES: an hour with no rows genuinely means zero
    events (observed silence), so missing buckets fill with 0 rather than
    carrying forward. The one-hour shift makes the row at hour ``t`` see
    buckets ``<= t-1`` only -- strictly before t, as everywhere else.
    """
    cohort_keys = cohort[["stay_id", "hour"]].drop_duplicates()
    span = cohort_keys.groupby("stay_id")["hour"].max().reset_index(name="max_hour")
    grid = span.assign(
        hour=[np.arange(-lookback, int(m) + 1) for m in span["max_hour"]]
    ).explode("hour")[["stay_id", "hour"]]
    grid["hour"] = grid["hour"].astype(int)

    events = events[events["hour"].between(-lookback, int(span["max_hour"].max()))]
    merged = grid.merge(events, on=["stay_id", "hour"], how="left")
    merged = merged.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    stay = merged["stay_id"]
    for col in columns:
        v = (merged[col] if col in merged
             else pd.Series(0.0, index=merged.index)).astype(float).fillna(0.0)
        merged[col] = v.groupby(stay, sort=False).shift(1).fillna(0.0)
    return merged, cohort_keys


def _rolling_sum(merged, col, window):
    return merged[col].groupby(merged["stay_id"], sort=False) \
        .rolling(window, min_periods=1).sum().droplevel(0).sort_index()


def _hours_since_positive(merged, col, cap=24.0):
    """Hours since the (already-shifted) channel was last non-zero; cap when
    never seen -- 'no lab drawn in >=24h' is itself the signal."""
    hour = merged["hour"]
    last = hour.where(merged[col] > 0).groupby(
        merged["stay_id"], sort=False).ffill()
    return (hour - last).clip(upper=cap).fillna(cap)


def build_cadence_features(cohort, cadence):
    """Leakage-safe clinician-cadence features per cohort (stay_id, hour) row.

    Counts are log1p'd (heavy right tails); the acceleration channels compare
    the last 6h to the trailing-24h per-6h average -- "is attention to this
    patient RISING", which is the precursor construct, not the level.
    """
    merged, cohort_keys = _dense_shifted_grid(
        cohort, cadence, CADENCE_CHANNELS, CADENCE_LOOKBACK_HOURS)
    out = merged[["stay_id", "hour"]].copy()
    for c in CADENCE_CHANNELS:
        s6 = _rolling_sum(merged, c, 6)
        s24 = _rolling_sum(merged, c, 24)
        out[f"{c}_1h"] = np.log1p(merged[c].to_numpy(dtype=float))
        out[f"{c}_6h"] = np.log1p(s6.to_numpy(dtype=float))
        out[f"{c}_24h"] = np.log1p(s24.to_numpy(dtype=float))
        if c in CADENCE_ACCEL_CHANNELS:
            out[f"{c}_accel"] = (s6 / (s24 / 4.0 + 1.0)).to_numpy(dtype=float)
    out["hrs_since_lab_draw"] = _hours_since_positive(merged, "n_lab_draws")
    out["hrs_since_med_order"] = _hours_since_positive(merged, "n_orders_med")
    return cohort_keys.merge(out, on=["stay_id", "hour"], how="left") \
        .sort_values(["stay_id", "hour"]).reset_index(drop=True)


# --- Urine output / fluid balance ---------------------------------------------

FLUIDS_PATH = "data/cache/mimic_fluids_hourly.parquet"
FLUID_CHANNELS = ["urine_ml", "n_urine_records", "other_out_ml", "in_ml",
                  "n_in_records"]

FLUID_FEATURES = [
    "urine_1h", "urine_6h", "urine_12h", "urine_24h",
    "oliguria_6h", "oliguria_12h",
    "n_urine_records_24h", "hrs_since_urine_record", "has_urine_history",
    "in_6h", "in_24h", "net_24h", "cum_net",
]


def load_fluids(path=FLUIDS_PATH):
    return pd.read_parquet(path)


def build_fluid_features(cohort, fluids):
    """Leakage-safe urine/balance features per cohort (stay_id, hour) row.

    An hour without a urine record is NOT zero production (charting is q1-4h
    with a catheter, absent without one), so oliguria flags require at least
    one record inside their window, and the record-cadence features
    (n_records, hrs_since, has_history) let the model separate "dry patient"
    from "no catheter".
    """
    merged, cohort_keys = _dense_shifted_grid(
        cohort, fluids, FLUID_CHANNELS, CADENCE_LOOKBACK_HOURS)
    # The GU-irrigant correction (227488 enters negatively) can leave an
    # HOURLY bucket net-negative -- an accounting artifact, not physiology
    # (log1p would NaN on it). Volumes are floored at zero per bucket.
    for col in ("urine_ml", "other_out_ml", "in_ml"):
        merged[col] = merged[col].clip(lower=0.0)
    out = merged[["stay_id", "hour"]].copy()

    u6 = _rolling_sum(merged, "urine_ml", 6)
    u12 = _rolling_sum(merged, "urine_ml", 12)
    u24 = _rolling_sum(merged, "urine_ml", 24)
    r6 = _rolling_sum(merged, "n_urine_records", 6)
    r12 = _rolling_sum(merged, "n_urine_records", 12)
    r24 = _rolling_sum(merged, "n_urine_records", 24)
    o24 = _rolling_sum(merged, "other_out_ml", 24)
    i6 = _rolling_sum(merged, "in_ml", 6)
    i24 = _rolling_sum(merged, "in_ml", 24)

    out["urine_1h"] = np.log1p(merged["urine_ml"].to_numpy(dtype=float))
    out["urine_6h"] = np.log1p(u6.to_numpy(dtype=float))
    out["urine_12h"] = np.log1p(u12.to_numpy(dtype=float))
    out["urine_24h"] = np.log1p(u24.to_numpy(dtype=float))
    out["oliguria_6h"] = ((u6 < 180) & (r6 >= 1)).to_numpy(dtype=float)
    out["oliguria_12h"] = ((u12 < 360) & (r12 >= 1)).to_numpy(dtype=float)
    out["n_urine_records_24h"] = np.log1p(r24.to_numpy(dtype=float))
    out["hrs_since_urine_record"] = _hours_since_positive(
        merged, "n_urine_records")
    seen = merged["n_urine_records"].groupby(
        merged["stay_id"], sort=False).cummax()
    out["has_urine_history"] = (seen > 0).to_numpy(dtype=float)
    out["in_6h"] = np.log1p(i6.to_numpy(dtype=float))
    out["in_24h"] = np.log1p(i24.to_numpy(dtype=float))
    out["net_24h"] = np.clip(
        (i24 - u24 - o24).to_numpy(dtype=float) / 1000.0, -10, 10)
    cum_in = merged["in_ml"].groupby(merged["stay_id"], sort=False).cumsum()
    cum_out = (merged["urine_ml"] + merged["other_out_ml"]).groupby(
        merged["stay_id"], sort=False).cumsum()
    out["cum_net"] = np.clip(
        (cum_in - cum_out).to_numpy(dtype=float) / 1000.0, -20, 20)
    return cohort_keys.merge(out, on=["stay_id", "hour"], how="left") \
        .sort_values(["stay_id", "hour"]).reset_index(drop=True)
