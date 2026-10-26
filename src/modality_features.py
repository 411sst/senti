"""Torch-free feature builders shared by the agents and the fair baselines.

These two functions were born inside ``agents/labs_agent.py`` and the notes
agent (benched; its code is on ``experiment-demo``). They are pure pandas/numpy,
but those modules import
torch at the top -- and torch and XGBoost cannot share a process on macOS (each
bundles its own OpenMP runtime; see ``tests/conftest.py``). The equal-information
GBM baselines in ``baselines_fair.py`` need exactly these features next to
XGBoost, so the builders live here and the labs agent re-imports them. Behaviour
is unchanged; ``tests/test_baselines_fair.py`` exercises both functions.
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
    return carry_forward_to_cohort(cohort, labs, LAB_COLUMNS,
                                   lookback=LAB_LOOKBACK_HOURS,
                                   max_staleness=max_staleness)


def carry_forward_to_cohort(cohort, source, columns, lookback, max_staleness):
    """LOCF + strictly-before-t shift + staleness bound, on the cohort's rows.

    The shared core of every irregularly-sampled modality builder: labs, and the
    supplementary vitals/blood-gas channels the published GBM specs need. Kept
    as one function because the leakage argument only has to be checked once.

    ``max_staleness`` is either a scalar bound applied to every column or a
    ``{column: bound}`` mapping, matching ``MAX_STALENESS_HOURS`` in
    ``mimic_cohort``.
    """
    cohort_keys = cohort[["stay_id", "hour"]].drop_duplicates()
    span = cohort_keys.groupby("stay_id")["hour"].max().reset_index(name="max_hour")

    # Dense grid: one row per (stay, hour) from -lookback to the stay's end.
    grid = span.assign(
        hour=[np.arange(-lookback, int(m) + 1) for m in span["max_hour"]]
    ).explode("hour")[["stay_id", "hour"]]
    grid["hour"] = grid["hour"].astype(int)

    source = source[source["hour"].between(-lookback, int(span["max_hour"].max()))]
    merged = grid.merge(source, on=["stay_id", "hour"], how="left")
    merged = merged.sort_values(["stay_id", "hour"]).reset_index(drop=True)

    stay, hour = merged["stay_id"], merged["hour"]
    out = merged[["stay_id", "hour"]].copy()
    for col in columns:
        bound = (max_staleness.get(col) if isinstance(max_staleness, dict)
                 else max_staleness)
        observed = merged[col] if col in merged else pd.Series(np.nan, index=merged.index)
        value = observed.groupby(stay, sort=False).ffill()
        value = value.groupby(stay, sort=False).shift(1)          # strictly before t
        last = hour.where(observed.notna()).groupby(stay, sort=False).ffill()
        last = last.groupby(stay, sort=False).shift(1)
        staleness = hour - last
        out[col] = value.where(staleness <= bound).to_numpy()
        out[f"{col}_hrs_since"] = staleness.where(staleness <= bound).to_numpy()

    # Keep only the cohort's prediction hours, in cohort key order.
    return cohort_keys.merge(out, on=["stay_id", "hour"], how="left") \
                      .sort_values(["stay_id", "hour"]).reset_index(drop=True)


# --- Supplementary vitals and blood gases (published GBM specs) --------------

EXTRA_VITALS_PATH = "data/cache/mimic_extra_vitals_hourly.parquet"

# Channels mimic_cohort.sql never pulled, required by the specs reproduced in
# src/lit_gbm.py: Sadanandan 2026 needs DBP, MAP, pCO2 and pO2; eCART needs MAP.
EXTRA_VITALS_COLUMNS = ["dbp", "mbp", "pco2", "po2"]

# Bounds follow mimic_cohort.MAX_STALENESS_HOURS: pressures are vitals (12h,
# beyond which monitoring has stopped); blood gases get 24h, the same bound as
# FiO2 and the expiry eCARTv5 applies to gases and lactate.
EXTRA_MAX_STALENESS = {"dbp": 12, "mbp": 12, "pco2": 24, "po2": 24}

# Physiologic plausibility bounds, applied on load. The raw extract contains a
# pO2 of -32 and one of 4242 mmHg; eCARTv5's stated rule is that "nonphysiologic
# flowsheet data were considered input errors and treated as missing", and a
# carried-forward impossible value is worse than a gap because the model reads
# it as a measurement.
EXTRA_PLAUSIBLE = {"dbp": (5, 200), "mbp": (5, 250),
                   "pco2": (5, 150), "po2": (10, 700)}


def load_extra_vitals(path=EXTRA_VITALS_PATH):
    """Supplementary hourly channels, with ids typed for a safe join.

    BigQuery hands these back as floats; joining a float ``stay_id`` against the
    cohort's integer grid silently matches nothing, which is how an earlier
    coverage audit ended up with bands summing to 200%.
    """
    df = pd.read_parquet(path)
    df["stay_id"] = df["stay_id"].astype("int64")
    df["hour"] = np.floor(df["hour"].astype(float)).astype("int64")
    for col, (lo, hi) in EXTRA_PLAUSIBLE.items():
        if col in df:
            df[col] = df[col].where(df[col].between(lo, hi))
    return df.groupby(["stay_id", "hour"], as_index=False).mean()


def build_extra_vitals_features(cohort, extra,
                                max_staleness=EXTRA_MAX_STALENESS):
    """DBP / MAP / pCO2 / pO2 carried to the cohort's rows, strictly before t.

    Same leakage guarantee as :func:`build_lab_features`, which it shares an
    implementation with. Lookback is 0: these channels are ICU-only, so unlike
    labs there is no pre-admission history to reach back for.
    """
    return carry_forward_to_cohort(cohort, extra, EXTRA_VITALS_COLUMNS,
                                   lookback=0, max_staleness=max_staleness)


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


def assert_one_row_per_key(events, name):
    """Every hourly extract must hold at most ONE row per (stay_id, hour).

    The builders left-join the extract onto a dense grid, so a duplicated key
    silently EXPANDS the result past the cohort's row count and misaligns
    every downstream feature against the labels. The SQL aggregates by
    (stay_id, hour), so this is a guard against a future extract edit (a
    dropped GROUP BY, a FULL OUTER JOIN gaining a fan-out) rather than a
    condition seen today.
    """
    dup = int(events.duplicated(["stay_id", "hour"]).sum())
    if dup:
        raise ValueError(f"{name} extract has {dup} duplicate (stay_id, hour) "
                         f"rows -- the hourly aggregation is broken")


def _dense_shifted_grid(cohort, events, columns, lookback, name="extract"):
    """Dense per-stay hourly grid of event columns, zero-filled and shifted.

    Counts differ from lab VALUES: an hour with no rows genuinely means zero
    events (observed silence), so missing buckets fill with 0 rather than
    carrying forward. The one-hour shift makes the row at hour ``t`` see
    buckets ``<= t-1`` only -- strictly before t, as everywhere else.
    """
    assert_one_row_per_key(events, name)
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
        cohort, cadence, CADENCE_CHANNELS, CADENCE_LOOKBACK_HOURS,
        name="cadence")
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
        cohort, fluids, FLUID_CHANNELS, CADENCE_LOOKBACK_HOURS,
        name="fluids")
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


# --- Respiratory support ------------------------------------------------------

RESP_PATH = "data/cache/mimic_resp_hourly.parquet"

# Escalation ladder (bedside-observable spontaneous support only). Artificial
# airways (ETT/trach and their circuit pieces) are NOT a rung: those patients
# are already ventilated -- a new vent-onset event cannot occur -- so they get
# a separate state flag instead. Unmapped strings -> NaN (LOCF carries the
# previous state; census printed by the builder).
RESP_LEVEL_MAP = {
    "None": 0,
    "Nasal cannula": 1, "Oxymizer": 1,
    "Venti mask": 2, "Medium conc mask": 2, "Face tent": 2,
    "Aerosol-cool": 2, "High flow neb": 2, "Ultrasonic neb": 2,
    "Vapomist": 2, "Other": 2,
    "Non-rebreather": 3,
    "High flow nasal cannula": 4,
    "CPAP mask": 5, "Bipap mask": 5,
}
RESP_AIRWAY = {"Endotracheal tube", "Tracheostomy tube", "Trach mask",
               "T-piece"}
RESP_STALENESS_HOURS = 24

RESP_FEATURES = [
    "resp_level", "resp_airway", "resp_level_max_6h", "resp_level_max_24h",
    "resp_escalated_6h", "resp_escalated_24h",
    "resp_o2_flow", "resp_flow_delta_6h",
    "resp_fio2", "resp_fio2_delta_6h",
    "resp_seen", "resp_hrs_since_record",
]


def load_resp(path=RESP_PATH):
    return pd.read_parquet(path)


def build_resp_features(cohort, resp, max_level=5, verbose=False):
    """Leakage-safe respiratory-support features per cohort row.

    State channels (level/airway/flow/FiO2) follow the LAB pattern -- LOCF
    within stay, shift(1) so hour t sees records strictly before t, staleness
    bound -- because support is a STATE, not a count. ``max_level`` caps the
    ladder (4 = the cap-at-HFNC sensitivity variant that drops the near-label
    NIV rung).
    """
    assert_one_row_per_key(resp, "resp")
    resp = resp.copy()
    # MIMIC device strings carry trailing whitespace ('Bipap mask ',
    # 'CPAP mask ') -- without the strip, the NIV rung silently vanishes
    # (caught when resp and resp_cap4 emitted identical scores).
    device = resp["o2_device"].astype("string").str.strip()
    level = device.map(RESP_LEVEL_MAP)
    airway = device.isin(RESP_AIRWAY)
    if verbose:
        unmapped = resp.loc[level.isna() & ~airway
                            & resp["o2_device"].notna(), "o2_device"]
        print(f"[resp] unmapped device strings: {len(unmapped):,} rows "
              f"({unmapped.value_counts().head(5).to_dict()})")
    resp["level"] = np.where(airway, np.nan, level.clip(upper=max_level))
    resp["airway"] = np.where(resp["o2_device"].notna(),
                              airway.astype(float), np.nan)

    cohort_keys = cohort[["stay_id", "hour"]].drop_duplicates()
    span = cohort_keys.groupby("stay_id")["hour"].max().reset_index(name="max_hour")
    grid = span.assign(
        hour=[np.arange(-CADENCE_LOOKBACK_HOURS, int(m) + 1)
              for m in span["max_hour"]]
    ).explode("hour")[["stay_id", "hour"]]
    grid["hour"] = grid["hour"].astype(int)
    resp = resp[resp["hour"].between(-CADENCE_LOOKBACK_HOURS,
                                     int(span["max_hour"].max()))]
    merged = grid.merge(
        resp[["stay_id", "hour", "level", "airway", "o2_flow", "fio2"]],
        on=["stay_id", "hour"], how="left")
    merged = merged.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    stay, hour = merged["stay_id"], merged["hour"]

    def locf(col, fill, staleness_cap=RESP_STALENESS_HOURS):
        observed = merged[col]
        value = observed.groupby(stay, sort=False).ffill()
        value = value.groupby(stay, sort=False).shift(1)
        last = hour.where(observed.notna()).groupby(stay, sort=False).ffill()
        last = last.groupby(stay, sort=False).shift(1)
        stale = hour - last
        return (value.where(stale <= staleness_cap).fillna(fill),
                stale.clip(upper=staleness_cap).fillna(staleness_cap))

    lev, lev_stale = locf("level", 0.0)
    awy, _ = locf("airway", 0.0)
    flow, _ = locf("o2_flow", 0.0)
    fio2, _ = locf("fio2", 21.0)

    out = merged[["stay_id", "hour"]].copy()
    out["resp_level"] = lev.to_numpy(dtype=float)
    out["resp_airway"] = awy.to_numpy(dtype=float)
    by = lev.groupby(stay, sort=False)
    out["resp_level_max_6h"] = by.rolling(6, min_periods=1).max() \
        .droplevel(0).sort_index().to_numpy(dtype=float)
    out["resp_level_max_24h"] = by.rolling(24, min_periods=1).max() \
        .droplevel(0).sort_index().to_numpy(dtype=float)
    out["resp_escalated_6h"] = ((lev - by.shift(6)) > 0).to_numpy(dtype=float)
    out["resp_escalated_24h"] = ((lev - lev.groupby(stay, sort=False)
                                  .shift(24)) > 0).to_numpy(dtype=float)
    out["resp_o2_flow"] = flow.to_numpy(dtype=float)
    out["resp_flow_delta_6h"] = (flow - flow.groupby(stay, sort=False)
                                 .shift(6)).fillna(0.0).to_numpy(dtype=float)
    out["resp_fio2"] = fio2.to_numpy(dtype=float)
    out["resp_fio2_delta_6h"] = (fio2 - fio2.groupby(stay, sort=False)
                                 .shift(6)).fillna(0.0).to_numpy(dtype=float)
    seen = (merged["level"].notna() | merged["airway"].notna()).astype(float)
    out["resp_seen"] = seen.groupby(stay, sort=False).cummax() \
        .groupby(stay, sort=False).shift(1).fillna(0.0) \
        .to_numpy(dtype=float)
    out["resp_hrs_since_record"] = lev_stale.to_numpy(dtype=float)
    return cohort_keys.merge(out, on=["stay_id", "hour"], how="left") \
        .sort_values(["stay_id", "hour"]).reset_index(drop=True)


# --- Consciousness / sedation / pain (neuro-status agent) ---------------------

NEURO_STATUS_PATH = "data/cache/mimic_neuro_status_hourly.parquet"

# Two semantics, two paths (see mimic_neuro_status.sql): levels are carried
# forward under a staleness bound like vitals; counts zero-fill like cadence.
NEURO_VALUE_COLS = ["rass_min", "rass_max", "pain_max"]
NEURO_COUNT_COLS = ["n_rass", "cam_assessed", "cam_positive", "n_pain"]

# RASS is protocolled ~q4h and pain q4-8h; 12h staleness matches the vitals
# discipline -- past it, reassessment has stopped, which is itself a state the
# count channels carry.
NEURO_MAX_STALENESS = {"rass_min": 12, "rass_max": 12, "pain_max": 12}

NEURO_STATUS_FEATURES = [
    "rass_now", "rass_low_now", "rass_hrs_since", "rass_delta_24h",
    "agitated_now", "deeply_sedated_now",
    "n_rass_24h", "rass_swing_24h",
    "cam_assessed_24h", "cam_positive_24h", "delirium_recent",
    "hrs_since_cam",
    "pain_now", "pain_hrs_since",
]


def load_neuro_status(path=NEURO_STATUS_PATH):
    df = pd.read_parquet(path)
    df["stay_id"] = df["stay_id"].astype("int64")
    df["hour"] = df["hour"].astype("int64")
    return df


def build_neuro_status_features(cohort, neuro):
    """Sedation / delirium / pain trajectory, strictly before t.

    The construct is the arc, not the level: a patient LIGHTENING toward
    agitation (rass_delta_24h > 0, agitated_now) or newly delirium-positive is
    the pre-deterioration signature; a deeply sedated patient whose RASS has
    not been reassessed in 12h is a different state from an alert one, which
    is why the staleness channel is a feature rather than an imputation
    artifact.

    Levels ride :func:`carry_forward_to_cohort` (cohort-row aligned); counts
    ride :func:`_dense_shifted_grid` and come back to cohort rows through the
    same final key-merge cadence uses.
    """
    keys = cohort[["stay_id", "hour"]]

    # -- level path: already one row per cohort key, dense hourly within stay
    vals = carry_forward_to_cohort(keys, neuro, NEURO_VALUE_COLS,
                                   lookback=0, max_staleness=NEURO_MAX_STALENESS)
    out = vals[["stay_id", "hour"]].copy()
    rass, rass_lo = vals["rass_max"], vals["rass_min"]
    stay = vals["stay_id"]
    out["rass_now"] = rass.to_numpy(np.float32)
    out["rass_low_now"] = rass_lo.to_numpy(np.float32)
    out["rass_hrs_since"] = vals["rass_max_hrs_since"].to_numpy(np.float32)
    out["rass_delta_24h"] = (rass - rass.groupby(stay, sort=False).shift(24)) \
        .to_numpy(np.float32)
    out["agitated_now"] = (rass >= 2).astype(np.float32).to_numpy()
    out["deeply_sedated_now"] = (rass_lo <= -3).astype(np.float32).to_numpy()
    hi = rass.groupby(stay, sort=False).rolling(24, min_periods=1).max() \
        .droplevel(0).sort_index()
    lo = rass_lo.groupby(stay, sort=False).rolling(24, min_periods=1).min() \
        .droplevel(0).sort_index()
    out["rass_swing_24h"] = (hi - lo).to_numpy(np.float32)
    out["pain_now"] = vals["pain_max"].to_numpy(np.float32)
    out["pain_hrs_since"] = vals["pain_max_hrs_since"].to_numpy(np.float32)

    # -- count path: dense grid, zero-filled, shifted; features then key-merged
    merged, cohort_keys = _dense_shifted_grid(
        keys, neuro[["stay_id", "hour"] + NEURO_COUNT_COLS],
        NEURO_COUNT_COLS, 0, name="neuro_status")
    cnt = merged[["stay_id", "hour"]].copy()
    cnt["n_rass_24h"] = np.log1p(
        _rolling_sum(merged, "n_rass", 24).to_numpy(dtype=float))
    cnt["cam_assessed_24h"] = np.log1p(
        _rolling_sum(merged, "cam_assessed", 24).to_numpy(dtype=float))
    cam_pos24 = _rolling_sum(merged, "cam_positive", 24)
    cnt["cam_positive_24h"] = np.log1p(cam_pos24.to_numpy(dtype=float))
    cnt["delirium_recent"] = (cam_pos24 > 0).to_numpy(dtype=np.float32)
    cnt["hrs_since_cam"] = _hours_since_positive(merged, "cam_assessed") \
        .to_numpy(np.float32)

    out = out.merge(cnt, on=["stay_id", "hour"], how="left")
    if len(out) != len(cohort_keys):
        raise ValueError("neuro-status feature join changed row count")
    return out[["stay_id", "hour"] + NEURO_STATUS_FEATURES]


# --- Infection workup (cultures + antibiotics) --------------------------------

INFECTION_PATH = "data/cache/mimic_infection_hourly.parquet"
INFECTION_CHANNELS = ["n_cultures", "n_blood_cultures", "n_abx_starts",
                      "n_abx_stops"]
INFECTION_LOOKBACK_HOURS = 24

INFECTION_FEATURES = [
    "cultures_6h", "cultures_24h", "cultures_72h",
    "blood_culture_24h", "hrs_since_culture",
    "abx_starts_24h", "abx_active_now", "abx_delta_24h",
    "on_abx", "new_abx_6h",
]


def load_infection(path=INFECTION_PATH):
    df = pd.read_parquet(path)
    df["stay_id"] = df["stay_id"].astype("int64")
    df["hour"] = df["hour"].astype("int64")
    return df


def build_infection_features(cohort, infection):
    """Suspicion-of-infection workup, strictly before t.

    A culture draw is Sepsis-3's t_suspicion made bedside-visible; a new
    antibiotic or a widening active count is the escalation that follows.
    Culture RESULTS are never read -- they lag days and would be a different,
    later signal. The active-antibiotic count is cumsum(starts) -
    cumsum(stops), clipped at zero because a stop whose start predates the
    24h lookback would otherwise push the count negative.
    """
    merged, cohort_keys = _dense_shifted_grid(
        cohort[["stay_id", "hour"]], infection, INFECTION_CHANNELS,
        INFECTION_LOOKBACK_HOURS, name="infection")

    out = merged[["stay_id", "hour"]].copy()
    for w in (6, 24, 72):
        out[f"cultures_{w}h"] = np.log1p(
            _rolling_sum(merged, "n_cultures", w).to_numpy(dtype=float))
    out["blood_culture_24h"] = (
        _rolling_sum(merged, "n_blood_cultures", 24) > 0
    ).to_numpy(dtype=np.float32)
    out["hrs_since_culture"] = _hours_since_positive(merged, "n_cultures") \
        .to_numpy(np.float32)

    stay = merged["stay_id"]
    starts = merged["n_abx_starts"].groupby(stay, sort=False).cumsum()
    stops = merged["n_abx_stops"].groupby(stay, sort=False).cumsum()
    active = (starts - stops).clip(lower=0)
    out["abx_starts_24h"] = np.log1p(
        _rolling_sum(merged, "n_abx_starts", 24).to_numpy(dtype=float))
    out["abx_active_now"] = active.to_numpy(np.float32)
    out["abx_delta_24h"] = (active - active.groupby(stay, sort=False)
                            .shift(24).fillna(0)).to_numpy(np.float32)
    out["on_abx"] = (active > 0).to_numpy(dtype=np.float32)
    out["new_abx_6h"] = (
        _rolling_sum(merged, "n_abx_starts", 6) > 0).to_numpy(dtype=np.float32)

    return cohort_keys.merge(out, on=["stay_id", "hour"], how="left") \
        .sort_values(["stay_id", "hour"]).reset_index(drop=True) \
        [["stay_id", "hour"] + INFECTION_FEATURES]
