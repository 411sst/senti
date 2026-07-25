"""Unit tests for the MIMIC paired first-alarm comparison."""

import numpy as np
import pandas as pd
import pytest

from baselines_mimic import paired_first_alarm


def _frame(stays):
    """stays: {stay_id: (event_h, [news2 per hour], [gbm per hour])}."""
    rows = []
    for sid, (event, n2, gb) in stays.items():
        for h, (a, b) in enumerate(zip(n2, gb)):
            rows.append({"stay_id": sid, "hour": h, "event_h": event,
                         "n2": a, "gb": b})
    df = pd.DataFrame(rows)
    return df, df["n2"].to_numpy(), df["gb"].to_numpy()


def test_paired_counts_direction_and_sign_test():
    # thresholds are 0.5 for both; scores are 0 until the alarm hour.
    stays = {
        1: (10, [0, 0, 0, 1, 1], [0, 1, 1, 1, 1]),   # N2@3 GBM@1 -> GBM earlier (+2)
        2: (10, [0, 0, 1, 1, 1], [0, 1, 1, 1, 1]),   # N2@2 GBM@1 -> GBM earlier (+1)
        3: (10, [0, 1, 1, 1, 1], [0, 0, 0, 1, 1]),   # N2@1 GBM@3 -> NEWS2 earlier (-2)
        4: (10, [0, 1, 1, 1, 1], [0, 1, 1, 1, 1]),   # same hour -> tie
    }
    df, n2, gb = _frame(stays)
    r = paired_first_alarm(df, n2, gb, 0.5, 0.5)

    assert r["n_paired"] == 4
    assert r["n_gbm_earlier"] == 2
    assert r["n_news2_earlier"] == 1
    assert r["n_tie"] == 1
    assert r["median_diff"] == pytest.approx(0.5)   # diffs [2, 1, -2, 0]
    assert r["frac_gbm_earlier"] == pytest.approx(0.5)
    assert 0.0 <= r["sign_p"] <= 1.0


def test_only_stays_caught_by_both_are_paired():
    stays = {
        1: (10, [0, 1, 1], [0, 1, 1]),   # both catch -> paired
        2: (10, [0, 1, 1], [0, 0, 0]),   # GBM never alarms -> excluded
        3: (10, [0, 0, 0], [0, 1, 1]),   # NEWS2 never alarms -> excluded
    }
    df, n2, gb = _frame(stays)
    assert paired_first_alarm(df, n2, gb, 0.5, 0.5)["n_paired"] == 1


def test_stays_without_an_event_are_excluded():
    stays = {
        1: (10, [0, 1, 1], [0, 1, 1]),        # has an event -> paired
        2: (np.nan, [0, 1, 1], [0, 1, 1]),    # no event -> no lead time
    }
    df, n2, gb = _frame(stays)
    assert paired_first_alarm(df, n2, gb, 0.5, 0.5)["n_paired"] == 1


def test_no_pairs_returns_nan_median_not_crash():
    stays = {1: (10, [0, 0, 0], [0, 0, 0])}
    df, n2, gb = _frame(stays)
    r = paired_first_alarm(df, n2, gb, 0.5, 0.5)
    assert r["n_paired"] == 0
    assert np.isnan(r["median_diff"])
    assert r["sign_p"] == 1.0
