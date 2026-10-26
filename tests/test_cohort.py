"""Unit tests for the cohort loader, leakage filter, LOCF, and splits."""

import json

import numpy as np
import pandas as pd
import pytest

from cohort import (
    HOUR,
    PATIENT_ID,
    before_time_t,
    event_onset_hour,
    forward_fill,
    load_cohort,
    news2_at_hour,
    train_test_split_by_patient,
)


def _toy_cohort(n_patients=20, n_hours=3):
    """A small multi-patient frame with sparse vitals."""
    rows = []
    for p in range(n_patients):
        for h in range(1, n_hours + 1):
            rows.append(
                {
                    PATIENT_ID: f"p{p:03d}",
                    HOUR: h,
                    "HR": np.nan if h % 2 == 0 else 60 + p,
                    "SBP": np.nan,
                    "SepsisLabel": 0,
                }
            )
    return pd.DataFrame(rows)


# --- Loader ----------------------------------------------------------------

def test_load_cohort_derives_patient_id_and_concatenates(tmp_path):
    (tmp_path / "p000001.psv").write_text(
        "HR|ICULOS|SepsisLabel\n80|1|0\n85|2|0\n"
    )
    (tmp_path / "p000002.psv").write_text(
        "HR|ICULOS|SepsisLabel\n70|1|0\n"
    )
    df = load_cohort(tmp_path, use_cache=False)  # isolate from the real cache

    assert set(df[PATIENT_ID]) == {"p000001", "p000002"}
    assert len(df) == 3  # 2 + 1 rows concatenated
    assert HOUR in df.columns
    assert len(df[df[PATIENT_ID] == "p000001"]) == 2


# --- before_time_t ---------------------------------------------------------

@pytest.mark.parametrize("t", [1, 2, 3, 4, 5, 6])
def test_before_time_t_never_returns_row_at_or_after_t(t):
    df = pd.DataFrame(
        {
            PATIENT_ID: ["p1"] * 5 + ["p2"] * 5,
            HOUR: [1, 2, 3, 4, 5] * 2,
            "HR": range(10),
        }
    )
    out = before_time_t(df, "p1", t)
    # nothing at or after t, and never another patient's rows
    assert (out[HOUR] < t).all()
    assert (out[PATIENT_ID] == "p1").all()


def test_before_time_t_selects_expected_hours():
    df = pd.DataFrame(
        {PATIENT_ID: ["p1"] * 5, HOUR: [1, 2, 3, 4, 5], "HR": range(5)}
    )
    assert sorted(before_time_t(df, "p1", 3)[HOUR]) == [1, 2]
    assert before_time_t(df, "p1", 1).empty  # no rows strictly before hour 1


# --- forward_fill ----------------------------------------------------------

def test_forward_fill_carries_past_value_and_never_future():
    # Rows deliberately out of chronological order to prove explicit sorting.
    df = pd.DataFrame(
        {
            PATIENT_ID: ["p1"] * 4,
            HOUR: [3, 1, 4, 2],
            "HR": [np.nan, 80.0, 95.0, np.nan],  # hour1=80, hour4=95, others NaN
            "SBP": [np.nan, np.nan, np.nan, 120.0],  # hour2=120 only
        }
    )
    filled = forward_fill(df, columns=["HR", "SBP"]).sort_values(HOUR)

    # HR: 80 carried forward until the hour-4 reading of 95.
    assert filled["HR"].tolist() == [80.0, 80.0, 80.0, 95.0]

    # SBP: hour 1 stays NaN -- the hour-2 reading of 120 must NOT be pulled
    # backward -- then 120 carries forward.
    sbp = filled["SBP"].tolist()
    assert np.isnan(sbp[0])
    assert sbp[1:] == [120.0, 120.0, 120.0]


def test_forward_fill_does_not_bleed_across_patients():
    df = pd.DataFrame(
        {
            PATIENT_ID: ["p1", "p1", "p2", "p2"],
            HOUR: [1, 2, 1, 2],
            "HR": [50.0, np.nan, np.nan, 90.0],
        }
    )
    filled = forward_fill(df, columns=["HR"]).sort_values([PATIENT_ID, HOUR])

    p2_first = filled[(filled[PATIENT_ID] == "p2") & (filled[HOUR] == 1)]["HR"]
    # p2's first hour has no prior reading of its own -> NaN, NOT p1's 50.
    assert np.isnan(p2_first.iloc[0])
    # p1 fills forward with its own value.
    p1_second = filled[(filled[PATIENT_ID] == "p1") & (filled[HOUR] == 2)]["HR"]
    assert p1_second.iloc[0] == 50.0


# --- Event label -----------------------------------------------------------

def test_event_onset_hour_is_first_positive():
    df = pd.DataFrame(
        {PATIENT_ID: ["p1"] * 4, HOUR: [1, 2, 3, 4], "SepsisLabel": [0, 0, 1, 1]}
    )
    assert event_onset_hour(df) == 3


def test_event_onset_hour_none_when_never_positive():
    df = pd.DataFrame(
        {PATIENT_ID: ["p1"] * 3, HOUR: [1, 2, 3], "SepsisLabel": [0, 0, 0]}
    )
    assert event_onset_hour(df) is None


# --- Deterministic patient-level split ------------------------------------

def test_split_is_reproducible_across_runs():
    df = _toy_cohort()
    train1, test1 = train_test_split_by_patient(df)
    train2, test2 = train_test_split_by_patient(df)
    assert set(train1[PATIENT_ID]) == set(train2[PATIENT_ID])
    assert set(test1[PATIENT_ID]) == set(test2[PATIENT_ID])


def test_split_is_independent_of_row_order():
    df = _toy_cohort()
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    _, test_a = train_test_split_by_patient(df)
    _, test_b = train_test_split_by_patient(shuffled)
    assert set(test_a[PATIENT_ID]) == set(test_b[PATIENT_ID])


def test_no_patient_in_both_train_and_test():
    df = _toy_cohort()
    train, test = train_test_split_by_patient(df)
    train_ids = set(train[PATIENT_ID])
    test_ids = set(test[PATIENT_ID])
    assert train_ids.isdisjoint(test_ids)               # no overlap
    assert train_ids | test_ids == set(df[PATIENT_ID])  # everyone is placed


# --- NEWS2 wired to the cohort --------------------------------------------

def _one_hour_patient(**vitals):
    """A single-patient, single-hour frame with the given vital values."""
    row = {PATIENT_ID: "p1", HOUR: 1, "SepsisLabel": 0}
    row.update(vitals)
    return pd.DataFrame([row])


def test_news2_at_hour_hand_computed():
    # Vitals chosen so the degraded NEWS2 is 2+2+2+2+2 + 2(O2) + 0 = 12.
    df = _one_hour_patient(
        Resp=22.0, O2Sat=93.0, Temp=39.5, SBP=95.0, HR=130.0, FiO2=0.5
    )
    # Evaluated at t=2, so only hour 1 (strictly before) is visible.
    assert news2_at_hour(df, "p1", 2) == 12


def test_news2_at_hour_uses_locf_and_respects_before_t():
    # Hour 1 has a full set; hour 2 re-measures only HR (now normal), the rest
    # stay NaN and must be carried forward from hour 1.
    df = pd.DataFrame(
        {
            PATIENT_ID: ["p1", "p1"],
            HOUR: [1, 2],
            "Resp": [22.0, np.nan],
            "O2Sat": [93.0, np.nan],
            "Temp": [39.5, np.nan],
            "SBP": [95.0, np.nan],
            "HR": [130.0, 60.0],   # hour 2: 60 -> band 0
            "FiO2": [0.5, np.nan],
            "SepsisLabel": [0, 0],
        }
    )
    # At t=2 only hour 1 is visible -> HR 130 -> score 12.
    assert news2_at_hour(df, "p1", 2) == 12
    # At t=3 the most recent hour is 2: HR carried as 60 (band 0), everything
    # else LOCF'd from hour 1 -> 2+2+2+2 + 0(HR) + 2(O2) = 10.
    assert news2_at_hour(df, "p1", 3) == 10


def test_news2_at_hour_all_nan_is_handled_not_crash():
    df = pd.DataFrame(
        {
            PATIENT_ID: ["p1", "p1"],
            HOUR: [1, 2],
            "Resp": [np.nan, np.nan],
            "O2Sat": [np.nan, np.nan],
            "Temp": [np.nan, np.nan],
            "SBP": [np.nan, np.nan],
            "HR": [np.nan, np.nan],
            "FiO2": [np.nan, np.nan],
            "SepsisLabel": [0, 0],
        }
    )
    # Never-measured vitals -> default missing rule (0), so a valid 0, no crash.
    assert news2_at_hour(df, "p1", 3) == 0


def test_news2_at_hour_no_prior_data_is_zero_not_crash():
    df = _one_hour_patient(
        Resp=22.0, O2Sat=93.0, Temp=39.5, SBP=95.0, HR=130.0, FiO2=0.5
    )
    # t=1: no rows strictly before hour 1 -> all vitals missing -> 0, no crash.
    assert news2_at_hour(df, "p1", 1) == 0


# --- Parquet cache --------------------------------------------------------

def _write_psv(path, rows):
    """Write a minimal .psv file. rows = list of (HR, ICULOS, SepsisLabel)."""
    lines = ["HR|ICULOS|SepsisLabel"]
    lines += [f"{hr}|{ic}|{lab}" for hr, ic, lab in rows]
    path.write_text("\n".join(lines) + "\n")


def test_cache_is_created_then_served(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_psv(data / "p000001.psv", [(80, 1, 0), (85, 2, 0)])
    cache = tmp_path / "cache" / "cohort.parquet"

    df1 = load_cohort(data, cache_path=cache)
    assert cache.exists()
    assert (cache.parent / "cohort.meta.json").exists()
    assert len(df1) == 2

    # Add a new source file AFTER caching. A cache hit must ignore it and serve
    # the original data -- proving the second load read the cache, not the disk.
    _write_psv(data / "p000002.psv", [(70, 1, 0)])
    df2 = load_cohort(data, cache_path=cache)
    assert set(df2[PATIENT_ID]) == {"p000001"}
    assert len(df2) == 2


def test_force_rebuild_ignores_cache(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_psv(data / "p000001.psv", [(80, 1, 0)])
    cache = tmp_path / "cache" / "cohort.parquet"

    load_cohort(data, cache_path=cache)          # build cache
    _write_psv(data / "p000002.psv", [(70, 1, 0)])  # new data after caching
    df = load_cohort(data, cache_path=cache, force_rebuild=True)
    assert set(df[PATIENT_ID]) == {"p000001", "p000002"}  # rebuilt from source


def test_loader_change_invalidates_cache(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_psv(data / "p000001.psv", [(80, 1, 0)])
    cache = tmp_path / "cache" / "cohort.parquet"

    load_cohort(data, cache_path=cache)          # build cache
    _write_psv(data / "p000002.psv", [(70, 1, 0)])  # new data after caching

    # Simulate the loader source changing by invalidating the stored fingerprint.
    meta_path = cache.parent / "cohort.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["loader_fingerprint"] = "stale-does-not-match"
    meta_path.write_text(json.dumps(meta))

    # Stale fingerprint -> rebuild rather than serve old data.
    df = load_cohort(data, cache_path=cache)
    assert set(df[PATIENT_ID]) == {"p000001", "p000002"}


def test_different_source_dir_invalidates_cache(tmp_path):
    cache = tmp_path / "cache" / "cohort.parquet"
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    _write_psv(dir_a / "p000001.psv", [(80, 1, 0)])
    load_cohort(dir_a, cache_path=cache)  # cache built from dir_a

    dir_b = tmp_path / "b"
    dir_b.mkdir()
    _write_psv(dir_b / "p000002.psv", [(70, 1, 0)])
    # Same cache path, different source dir -> must rebuild, not serve dir_a.
    df = load_cohort(dir_b, cache_path=cache)
    assert set(df[PATIENT_ID]) == {"p000002"}


def test_use_cache_false_neither_reads_nor_writes(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_psv(data / "p000001.psv", [(80, 1, 0)])
    cache = tmp_path / "cache" / "cohort.parquet"

    df = load_cohort(data, cache_path=cache, use_cache=False)
    assert len(df) == 1
    assert not cache.exists()  # nothing written
