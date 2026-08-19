"""Tests for the NEWS2 fairness band.

The band exists to defend the timing claim against NEWS2's strongest form, so
the tests pin the two things that make it strongest: unbounded carry-forward
really does keep old observations alive (where the bounded variant forgets
them), and the single-parameter-3 trigger fires on a component score of 3 even
when the aggregate stays below every aggregate threshold.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mimic_cohort import CARRIED, build_before_t_features, news2_component_scores


def _raw(rows):
    df = pd.DataFrame(rows)
    for c in CARRIED:
        if c not in df:
            df[c] = np.nan
    return df


def test_unbounded_locf_keeps_what_bounded_forgets():
    # HR 25 (scores 3) charted once at hour 0; nothing else ever charted.
    rows = [{"stay_id": 1, "hour": h, "heart_rate": 25.0 if h == 0 else np.nan}
            for h in range(0, 30)]
    raw = _raw(rows)

    bounded = build_before_t_features(raw)                    # 12h vitals bound
    unbounded = build_before_t_features(raw, max_staleness={})

    b25 = bounded.loc[bounded.hour == 25, "heart_rate"].iloc[0]
    u25 = unbounded.loc[unbounded.hour == 25, "heart_rate"].iloc[0]
    assert np.isnan(b25)          # bounded: stale after 12h, forgotten
    assert u25 == 25.0            # unbounded: still current

    comp_b, _, _ = news2_component_scores(bounded)
    comp_u, _, _ = news2_component_scores(unbounded)
    row25 = bounded.index[bounded.hour == 25][0]
    assert comp_b[row25].sum() == 0          # bounded variant scores nothing
    assert comp_u[row25].sum() == 3          # best-LOCF still sees HR 25 -> 3


def test_single_param_3_fires_below_aggregate_thresholds():
    # SpO2 91 scores 3 on its own; everything else normal -> total 3 (< 5).
    feat = pd.DataFrame({
        "stay_id": [1], "hour": [7],
        "heart_rate": [80.0], "resp_rate": [16.0], "spo2": [91.0],
        "sbp": [120.0], "temperature": [37.0],
        "is_alert": [1.0], "on_oxygen": [0.0],
    })
    comp, _, _ = news2_component_scores(feat)
    total = comp.sum(axis=1)
    assert total[0] == 3
    assert not (total >= 5).any()                 # aggregate triggers silent
    assert (comp == 3).any(axis=1)[0]             # single-parameter trigger fires
