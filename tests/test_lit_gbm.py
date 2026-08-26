"""Tests for the literature-anchored GBM specs.

The load-bearing guarantees: every derived feature is a function of the current
and earlier rows only (so a future observation cannot move a past prediction),
the eCART down-sample touches train only and hits the published prevalence, and
the score cache refuses to hand back rows that do not align with the cohort.
"""

import numpy as np
import pandas as pd
import pytest

from lit_gbm import (
    SADANANDAN_LABS,
    SADANANDAN_VITALS,
    SPECS,
    build_channels,
    downsample_to_prevalence,
    expanding_stats,
    load_lit_scores,
    trailing_deltas,
    trailing_stats,
)
from mimic_cohort import VITALS
from modality_features import EXTRA_VITALS_COLUMNS, LAB_COLUMNS

LABEL = "label_24h"


def _cohort(n_stays=4, hours=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stays):
        for h in range(6, 6 + hours):
            row = {"stay_id": i, "patient_id": 100 + i, "hour": h,
                   LABEL: int(h > 6 + hours - 8 and i % 2 == 0),
                   "gcs": 14.0, "gcs_unable": 0.0, "on_oxygen": float(i % 2)}
            for v in VITALS:
                row[v] = float(rng.normal(80, 10))
                row[f"{v}_hrs_since"] = float(h % 5)
            rows.append(row)
    return pd.DataFrame(rows)


def _sources(cohort):
    keys = cohort[["stay_id", "hour"]].drop_duplicates()
    labs = keys.copy()
    for c in LAB_COLUMNS:
        labs[c] = np.linspace(1, 2, len(labs))
    extra = keys.copy()
    for c in EXTRA_VITALS_COLUMNS:
        extra[c] = np.linspace(30, 60, len(extra))
    raw = keys.copy()
    raw["fio2"] = 40.0
    raw["o2_flow"] = 2.0
    meta = pd.DataFrame({"stay_id": sorted(cohort.stay_id.unique())})
    meta["gender"] = "F"
    meta["anchor_age"] = 70.0
    meta["admit_hour_of_day"] = 3
    return labs, extra, raw, meta


# --- channel assembly --------------------------------------------------------

def test_build_channels_aligns_and_covers_every_spec_input():
    cohort = _cohort()
    ordered, ch = build_channels(cohort, *_sources(cohort))
    assert len(ch) == len(cohort)
    assert (ordered["stay_id"].to_numpy() ==
            cohort.sort_values(["stay_id", "hour"])["stay_id"].to_numpy()).all()
    for col in SADANANDAN_VITALS + SADANANDAN_LABS + ["age", "sex_male", "time_of_day"]:
        assert col in ch.columns, col


def test_time_of_day_wraps_from_the_admission_hour():
    cohort = _cohort(n_stays=1, hours=30)
    ordered, ch = build_channels(cohort, *_sources(cohort))
    # admit_hour_of_day=3, first prediction row is hour 6 -> 09:00
    assert ch["time_of_day"].iloc[0] == pytest.approx(9.0)
    assert ch["time_of_day"].max() < 24


# --- leakage -----------------------------------------------------------------

def test_derived_features_ignore_the_future():
    """Perturbing the LAST hour must not change any earlier row's features."""
    cohort = _cohort(n_stays=2, hours=20)
    labs, extra, raw, meta = _sources(cohort)
    ordered, ch = build_channels(cohort, labs, extra, raw, meta)

    bumped = ch.copy()
    last = bumped.index[-1]
    bumped.loc[last, VITALS] = 999.0

    for fn in (lambda c: trailing_stats(ordered, c, VITALS, 6),
               lambda c: trailing_deltas(ordered, c, VITALS, 6),
               lambda c: expanding_stats(ordered, c, VITALS)):
        a, b = fn(ch), fn(bumped)
        for k in a:
            np.testing.assert_allclose(a[k][:-1], b[k][:-1], equal_nan=True,
                                       err_msg=f"{k} moved on a future value")


def test_expanding_stats_match_a_plain_python_expanding_window():
    ordered = pd.DataFrame({"stay_id": [0, 0, 0, 1, 1], "hour": [0, 1, 2, 0, 1]})
    ch = pd.DataFrame({"x": [3.0, 1.0, np.nan, 10.0, 20.0]})
    out = expanding_stats(ordered, ch, ["x"])
    np.testing.assert_allclose(out["x_emin"], [3, 1, np.nan, 10, 10], equal_nan=True)
    np.testing.assert_allclose(out["x_emax"], [3, 3, np.nan, 10, 20], equal_nan=True)
    np.testing.assert_allclose(out["x_ecount"], [1, 2, 2, 1, 2])
    np.testing.assert_allclose(out["x_emean"], [3, 2, 2, 10, 15], equal_nan=True)


def test_expanding_stats_do_not_cross_stays():
    ordered = pd.DataFrame({"stay_id": [0, 0, 1], "hour": [0, 1, 0]})
    ch = pd.DataFrame({"x": [100.0, 100.0, 1.0]})
    out = expanding_stats(ordered, ch, ["x"])
    assert out["x_emax"][2] == 1.0          # not 100 from the previous stay


# --- specs -------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SPECS))
def test_spec_builds_a_finite_width_design_with_provenance(name):
    cohort = _cohort()
    ordered, ch = build_channels(cohort, *_sources(cohort))
    X, prov = SPECS[name](ordered, ch)
    assert len(X) == len(cohort) and X.shape[1] > 0
    assert prov["library"] in ("xgboost", "lightgbm")
    assert prov["paper"] and prov["deviations"], "a spec must state its gaps"


def test_sadanandan_uses_exactly_its_published_channel_count():
    cohort = _cohort()
    ordered, ch = build_channels(cohort, *_sources(cohort))
    X, _ = SPECS["sadanandan"](ordered, ch)
    # 7 vital channels + 16 labs + age + sex
    assert list(X.columns) == SADANANDAN_VITALS + SADANANDAN_LABS + ["age", "sex_male"]
    assert len(SADANANDAN_LABS) == 16


def test_sadanandan_drops_vitals_older_than_four_hours():
    cohort = _cohort(n_stays=1, hours=10)
    cohort["heart_rate_hrs_since"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ordered, ch = build_channels(cohort, *_sources(cohort))
    X, _ = SPECS["sadanandan"](ordered, ch)
    assert X["heart_rate"].iloc[:5].notna().all()      # <= 4h old
    assert X["heart_rate"].iloc[5:].isna().all()       # stale, treated as unknown


# --- eCART down-sampling -----------------------------------------------------

def test_downsample_hits_the_published_prevalence_and_keeps_all_positives():
    y = np.concatenate([np.zeros(9000), np.ones(1000)])
    idx = downsample_to_prevalence(y, 0.5, seed=0)
    assert y[idx].mean() == pytest.approx(0.5, abs=1e-9)
    assert (y[idx] == 1).sum() == 1000
    assert np.all(np.diff(idx) > 0), "index must stay sorted for .iloc alignment"


def test_downsample_is_a_noop_when_negatives_are_already_scarce():
    y = np.concatenate([np.zeros(10), np.ones(1000)])
    idx = downsample_to_prevalence(y, 0.5, seed=0)
    assert len(idx) == len(y)


# --- cache contract ----------------------------------------------------------

def test_load_lit_scores_rejects_misaligned_rows(tmp_path):
    path = tmp_path / "scores.npz"
    np.savez(path, val_stay_id=np.array([1, 1]), val_hour=np.array([6, 7]),
             test_stay_id=np.array([2]), test_hour=np.array([6]),
             val_x=np.array([0.1, 0.2]), test_x=np.array([0.3]))
    val = pd.DataFrame({"stay_id": [1, 1], "hour": [6, 7]})
    test_ok = pd.DataFrame({"stay_id": [2], "hour": [6]})
    assert "x" in load_lit_scores(val, test_ok, path=path)

    test_bad = pd.DataFrame({"stay_id": [2], "hour": [99]})
    with pytest.raises(ValueError, match="do not align"):
        load_lit_scores(val, test_bad, path=path)
