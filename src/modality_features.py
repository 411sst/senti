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
