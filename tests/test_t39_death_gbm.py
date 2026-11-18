"""Tests for the T39 helpers in scripts/t39_death_gbm.py.

Synthetic frames with hand-checked answers. Ids are small integers.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from t39_death_gbm import (  # noqa: E402
    death_calibrator,
    death_labels,
    resume_payload,
    seed_pairs,
)


def _split():
    # Deliberately unsorted; stay 102's death hour is its last.
    return pd.DataFrame({
        "stay_id": [102, 101, 102, 101, 102],
        "hour": [2, 0, 0, 1, 1],
        "label_24h_death": [1, 0, 0, 0, 0],
        "label_24h": [1, 1, 0, 1, 1],          # composite: must be ignored
    })


def _keys(split):
    return split.sort_values(["stay_id", "hour"])[["stay_id", "hour"]].reset_index(drop=True)


def test_death_labels_follow_the_design_row_order():
    split = _split()
    y = death_labels(split, _keys(split))
    # sorted rows: (101,0) (101,1) (102,0) (102,1) (102,2)
    assert y.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("change", ["drop_row", "swap_hour"])
def test_death_labels_refuse_rows_that_differ_from_the_design(change):
    split = _split()
    keys = _keys(split)
    if change == "drop_row":
        keys = keys.iloc[:-1]
    else:
        keys.loc[0, "hour"] = 5
    with pytest.raises(ValueError, match="do not match"):
        death_labels(split, keys)


@pytest.mark.parametrize("bad", [np.nan, 2.0])
def test_death_labels_refuse_a_non_binary_label(bad):
    split = _split()
    split["label_24h_death"] = split["label_24h_death"].astype(float)
    split.loc[0, "label_24h_death"] = bad
    with pytest.raises(ValueError, match="not binary"):
        death_labels(split, _keys(split))


def test_death_calibrator_fits_on_v_fit_rows_only():
    raw = np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])
    y = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=float)
    fit_mask = np.array([True] * 4 + [False] * 4)
    iso = death_calibrator(raw, y, fit_mask)
    # Only the first four rows count: low raw -> 0, high raw -> 1.
    assert iso.predict([0.1, 0.4]).tolist() == [0.0, 1.0]


def test_resume_payload_refuses_a_cache_from_other_inputs(tmp_path):
    path = tmp_path / "t39_death_gbm.npz"
    assert resume_payload(path, "abc") == {}
    np.savez(path, fingerprint=np.array("abc"), test_s42=np.arange(3.0))
    got = resume_payload(path, "abc")
    assert got["test_s42"].tolist() == [0.0, 1.0, 2.0]
    with pytest.raises(SystemExit, match="different fingerprint"):
        resume_payload(path, "xyz")


def test_seed_pairs_counts_each_seed_and_pools_the_mean():
    ramp = np.linspace(-1.0, 1.0, 1000)
    points_a = {1: 0.50, 2: 0.30, 3: 0.41}
    points_b = {1: 0.40, 2: 0.40, 3: 0.40}
    draws_b = {s: np.full(1000, 0.40) for s in points_b}
    draws_a = {1: 0.50 + 0.01 * ramp,      # delta about +0.10, CI above zero
               2: 0.30 + 0.01 * ramp,      # delta about -0.10, CI below zero
               3: 0.41 + 0.05 * ramp}      # delta +0.01, CI spans zero
    out = seed_pairs(points_a, points_b, draws_a, draws_b)
    assert out["ci_above_zero"] == 1
    assert out["ci_below_zero"] == 1
    assert out["point_above_zero"] == 2
    assert out["pooled"]["delta"] == pytest.approx((0.10 - 0.10 + 0.01) / 3)
    assert out["pooled"]["ci"][0] < 0 < out["pooled"]["ci"][1]
    assert out["pairs"][1]["ci_above_zero"] and not out["pairs"][3]["ci_above_zero"]
