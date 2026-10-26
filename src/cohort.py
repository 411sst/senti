"""Cohort loading, leakage-safe filtering, labelling, and splits.

For the Week-1 skeleton this reads the open PhysioNet 2019 sepsis data
(pipe-separated, one file per patient, one row per hour). Structural facts
this module relies on:

* ``patient_id`` is derived from the filename (``p000001.psv`` -> ``p000001``);
  it is not a column in the raw files.
* ``ICULOS`` (ICU length-of-stay, in hours) is the per-patient hour index.
* Vitals are recorded intermittently, with ``NaN`` in between.

Non-negotiables enforced here (see CLAUDE.md):
* No leakage: :func:`before_time_t` returns only data strictly before hour ``t``,
  and :func:`forward_fill` can only ever propagate an earlier value forward.
* The event label lives in ONE place (:func:`event_mask`) so the sepsis
  stand-in can be swapped for the real composite deterioration label without
  touching anything else.
* Splits are deterministic from the global ``SEED`` and are made by patient, so
  a patient never appears in both train and test.
"""

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import SEED
from news2 import degraded_news2_score

# --- Canonical column names (swap points live here) ------------------------
PATIENT_ID = "patient_id"
HOUR = "ICULOS"
LABEL_COLUMN = "SepsisLabel"

# Default on-disk cache for the concatenated cohort (see load_cohort).
DEFAULT_CACHE_PATH = "data/cache/cohort.parquet"

# Sparse vitals we carry forward for NEWS2. HR/O2Sat/Temp/SBP/Resp map directly
# to five of the seven NEWS2 inputs; FiO2 is kept because supplemental-oxygen
# status can only be inferred from it on this dataset.
VITAL_COLUMNS = ["HR", "O2Sat", "Temp", "SBP", "Resp", "FiO2"]


# --- Loading ---------------------------------------------------------------

def load_patient_file(path):
    """Read one PhysioNet ``.psv`` file and tag it with its ``patient_id``.

    The patient id is the filename stem (``p000001.psv`` -> ``p000001``).
    """
    path = Path(path)
    df = pd.read_csv(path, sep="|")
    df.insert(0, PATIENT_ID, path.stem)
    return df


def _build_cohort(data_dir):
    """Read and concatenate every ``.psv`` under ``data_dir`` (uncached).

    Files are read in sorted-path order so the concatenation (and therefore
    everything downstream) is reproducible.
    """
    data_dir = Path(data_dir)
    paths = sorted(data_dir.rglob("*.psv"))
    if not paths:
        raise FileNotFoundError(f"No .psv files found under {data_dir}")
    frames = [load_patient_file(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def _loader_fingerprint():
    """SHA-256 of the loader source that determines the cached content.

    Hashing the source of the functions that actually build the dataframe means
    ANY edit to how files are read or concatenated changes the fingerprint, so a
    stale cache is rebuilt instead of silently served. (The caching wrapper
    itself is intentionally excluded -- changing it does not change the data.)
    """
    src = inspect.getsource(load_patient_file) + inspect.getsource(_build_cohort)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def load_cohort(data_dir, use_cache=True, force_rebuild=False, cache_path=None):
    """Load every ``.psv`` under ``data_dir`` into a single dataframe.

    Each row keeps its ``patient_id`` and ``ICULOS`` hour index.

    Caching
    -------
    The concatenated frame is expensive to build (tens of thousands of files),
    so it is cached to Parquet at ``cache_path`` (default
    ``data/cache/cohort.parquet``) with a sidecar ``*.meta.json``.

    * First load (or any cache miss): build from the ``.psv`` files, then write
      the Parquet + meta.
    * Later loads: read the Parquet directly -- unless the cache is stale.
    * ``force_rebuild=True`` rebuilds and overwrites regardless of the cache.
    * ``use_cache=False`` bypasses reading and writing entirely.

    The cache is invalidated (rebuilt) when either the loader source changes
    (detected via :func:`_loader_fingerprint`) or ``data_dir`` differs from the
    directory the cache was built from -- never serving data produced by old
    logic or from a different source.
    """
    cache_path = Path(cache_path or DEFAULT_CACHE_PATH)
    meta_path = cache_path.parent / (cache_path.stem + ".meta.json")
    fingerprint = _loader_fingerprint()
    source_dir = str(Path(data_dir).resolve())

    if use_cache and not force_rebuild and cache_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            meta = {}
        # Serve the cache only if the loader is unchanged AND it was built from
        # the same source directory -- otherwise loading a different data_dir
        # would silently return the previously-cached cohort.
        if (meta.get("loader_fingerprint") == fingerprint
                and meta.get("source_dir") == source_dir):
            return pd.read_parquet(cache_path)
        # stale (loader changed or different source) -> fall through and rebuild

    df = _build_cohort(data_dir)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Write the Parquet first, then the meta: the meta is the "commit"
        # marker, so a half-written cache is treated as a miss next time.
        df.to_parquet(cache_path, index=False)
        meta = {
            "loader_fingerprint": fingerprint,
            "n_rows": int(len(df)),
            "n_patients": int(df[PATIENT_ID].nunique()),
            "source_dir": source_dir,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    return df


# --- Leakage-safe access ---------------------------------------------------

def before_time_t(df, patient_id, t):
    """Rows for ``patient_id`` recorded strictly before hour ``t``.

    "Strictly before" means ``ICULOS < t`` -- the row at hour ``t`` itself is
    the prediction point and must not be visible. This is the single gate that
    every feature must pass through to guarantee no future data leaks.
    """
    mask = (df[PATIENT_ID] == patient_id) & (df[HOUR] < t)
    return df.loc[mask]


def forward_fill(df, columns=None):
    """Last-value-carried-forward of sparse vitals, past-only.

    Intended to be run on the output of :func:`before_time_t`, so the frame
    already contains only rows earlier than the prediction time. We still make
    the temporal ordering explicit: sort ascending by ``ICULOS`` *before*
    filling, because pandas ``ffill`` propagates values in row order. With the
    rows sorted oldest-to-newest, "forward" == "later in time", so a value can
    only ever be carried from an earlier hour to a later one -- a future value
    can never be pulled backward. Grouping by ``patient_id`` keeps one
    patient's readings from bleeding into another's.

    Returns a re-sorted copy; the original frame is not modified.
    """
    if columns is None:
        columns = VITAL_COLUMNS

    # Explicit oldest-to-newest ordering is the whole point (see docstring).
    out = df.sort_values([PATIENT_ID, HOUR]).reset_index(drop=True)
    out[columns] = out.groupby(PATIENT_ID)[columns].ffill()
    return out


# Which raw column feeds each available NEWS2 vital input.
NEWS2_VITAL_COLUMNS = {
    "respiratory_rate": "Resp",
    "spo2": "O2Sat",
    "temperature": "Temp",
    "systolic_bp": "SBP",
    "heart_rate": "HR",
    "fio2": "FiO2",  # used only to infer supplemental oxygen
}


def news2_at_hour(df, patient_id, t):
    """Degraded NEWS2 for ``patient_id`` at hour ``t``, using only data before t.

    Chains the leakage-safe primitives: :func:`before_time_t` restricts to rows
    earlier than ``t``, :func:`forward_fill` carries the last observed value of
    each vital forward (past-only), and the most recent row then feeds
    :func:`news2.degraded_news2_score`. See that function for the degradations
    (oxygen inferred from FiO2, consciousness pinned Alert, never-measured vital
    scored as normal). If the patient has no rows before ``t``, every vital is
    missing and the scorer's missing-vital rule applies.

    Returns the integer degraded NEWS2 score.
    """
    history = forward_fill(before_time_t(df, patient_id, t), columns=VITAL_COLUMNS)
    if history.empty:
        recent = {col: np.nan for col in VITAL_COLUMNS}
    else:
        recent = history.iloc[-1]  # most recent hour strictly before t (LOCF-filled)

    col = NEWS2_VITAL_COLUMNS
    return degraded_news2_score(
        respiratory_rate=recent[col["respiratory_rate"]],
        spo2=recent[col["spo2"]],
        temperature=recent[col["temperature"]],
        systolic_bp=recent[col["systolic_bp"]],
        heart_rate=recent[col["heart_rate"]],
        fio2=recent[col["fio2"]],
    )


# --- Event label (ONE place) ----------------------------------------------

def event_mask(df):
    """Boolean Series: True for hours that are event-positive.

    THIS IS THE ONE PLACE the event definition lives. For the Week-1 skeleton
    the event is the PhysioNet sepsis label. To move to the real composite
    deterioration label (ICU transfer OR in-hospital mortality OR
    mechanical-ventilation onset OR vasopressor onset), change only this
    function -- everything else (onset, labelling, splits) is defined in terms
    of it.
    """
    return df[LABEL_COLUMN] == 1


def event_onset_hour(df):
    """First event-positive hour (``ICULOS``) for a single patient's rows.

    Returns ``None`` if the patient never becomes event-positive.
    """
    positives = df.loc[event_mask(df)]
    if positives.empty:
        return None
    return int(positives[HOUR].min())


def onset_by_patient(df):
    """Series mapping ``patient_id`` -> onset hour (``NaN`` if never positive)."""
    onsets = df.loc[event_mask(df)].groupby(PATIENT_ID)[HOUR].min()
    all_ids = df[PATIENT_ID].drop_duplicates()
    return onsets.reindex(all_ids)


# --- Deterministic patient-level split ------------------------------------

def split_patient_ids(patient_ids, test_size=0.2, seed=SEED):
    """Split a collection of patient ids into (train_ids, test_ids).

    Deterministic from ``seed`` and independent of input order: ids are sorted
    to a canonical order first, then shuffled with a seeded generator and
    sliced. Same seed -> same split, every run.
    """
    ids = np.array(sorted(set(patient_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)  # in place; deterministic given the seed
    n_test = int(round(len(ids) * test_size))
    test_ids = ids[:n_test]
    train_ids = ids[n_test:]
    return list(train_ids), list(test_ids)


def split_patient_ids_three_way(patient_ids, val_size=0.15, test_size=0.15,
                                seed=SEED):
    """Split patient ids into (train_ids, val_ids, test_ids).

    The middle split exists for **calibration**: a model is fit on train, its
    probabilities are calibrated on val, and only then is it scored on test.
    Calibrating on train would reuse data the model has already fit, making ECE
    look better than it is -- and ECE is one of the metrics this project claims
    to win on, so it has to be measured honestly.

    Deterministic from ``seed`` and independent of input order. The shuffled ids
    are sliced test-first, then val, so changing ``val_size`` leaves the test
    set untouched.
    """
    ids = np.array(sorted(set(patient_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]
    return list(train_ids), list(val_ids), list(test_ids)


def train_test_split_by_patient(df, test_size=0.2, seed=SEED):
    """Split ``df`` into (train_df, test_df) by patient.

    The split is made on ``patient_id`` and applied with ``isin``, so every row
    of a patient lands in exactly one side -- a patient is never split across
    train and test. Reproducible from ``seed`` regardless of row order.
    """
    train_ids, test_ids = split_patient_ids(
        df[PATIENT_ID].unique(), test_size=test_size, seed=seed
    )
    train_df = df[df[PATIENT_ID].isin(train_ids)]
    test_df = df[df[PATIENT_ID].isin(test_ids)]
    return train_df, test_df
