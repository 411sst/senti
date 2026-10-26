"""Unit tests for the MIMIC paired first-alarm comparison."""

import numpy as np
import pandas as pd
import pytest

from baselines_mimic import _stay_series, paired_first_alarm
from mimic_cohort import STAB_HOURS


def _frame(stays, start_hour=STAB_HOURS + 1):
    """stays: {stay_id: (event_h, [news2 per hour], [gbm per hour])}."""
    rows = []
    for sid, (event, n2, gb) in stays.items():
        for i, (a, b) in enumerate(zip(n2, gb)):
            rows.append({"stay_id": sid, "hour": start_hour + i,
                         "event_h": event, "n2": a, "gb": b})
    df = pd.DataFrame(rows)
    return df, df["n2"].to_numpy(), df["gb"].to_numpy()


def test_paired_counts_direction_and_sign_test():
    # thresholds are 0.5 for both; scores are 0 until the alarm hour.
    # Hours run 7..11, event at 12.
    stays = {
        1: (12, [0, 0, 0, 1, 1], [0, 1, 1, 1, 1]),   # N2@10 GBM@8 -> GBM earlier (+2)
        2: (12, [0, 0, 1, 1, 1], [0, 1, 1, 1, 1]),   # N2@9  GBM@8 -> GBM earlier (+1)
        3: (12, [0, 1, 1, 1, 1], [0, 0, 0, 1, 1]),   # N2@8  GBM@10 -> NEWS2 earlier (-2)
        4: (12, [0, 1, 1, 1, 1], [0, 1, 1, 1, 1]),   # same hour -> tie
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
        1: (12, [0, 1, 1], [0, 1, 1]),   # both catch -> paired
        2: (12, [0, 1, 1], [0, 0, 0]),   # GBM never alarms -> excluded
        3: (12, [0, 0, 0], [0, 1, 1]),   # NEWS2 never alarms -> excluded
    }
    df, n2, gb = _frame(stays)
    assert paired_first_alarm(df, n2, gb, 0.5, 0.5)["n_paired"] == 1


def test_stays_without_an_event_are_excluded():
    stays = {
        1: (12, [0, 1, 1], [0, 1, 1]),        # has an event -> paired
        2: (np.nan, [0, 1, 1], [0, 1, 1]),    # no event -> no lead time
    }
    df, n2, gb = _frame(stays)
    assert paired_first_alarm(df, n2, gb, 0.5, 0.5)["n_paired"] == 1


def test_no_pairs_returns_nan_median_not_crash():
    stays = {1: (12, [0, 0, 0], [0, 0, 0])}
    df, n2, gb = _frame(stays)
    r = paired_first_alarm(df, n2, gb, 0.5, 0.5)
    assert r["n_paired"] == 0
    assert np.isnan(r["median_diff"])
    assert r["sign_p"] == 1.0


# --- h6 root-cause fix: the first prediction hour competes like any other ----
# (The old "warm-up rule" masked hour 6 because it was all-NaN by grid
# construction; the extract now carries hours 0-5 as LOCF history, so an h6
# alarm is a real prediction and must count.)

def test_boundary_hour_alarm_counts():
    # Hours run 6..10, event at 12. GBM first crosses at h6, NEWS2 at h8:
    # GBM is earlier by 2 -- the h6 row is a legitimate first alarm.
    stays = {1: (12, [0, 0, 1, 1, 1], [1, 0, 0, 1, 1])}
    df, n2, gb = _frame(stays, start_hour=STAB_HOURS)
    r = paired_first_alarm(df, n2, gb, 0.5, 0.5)
    assert r["n_paired"] == 1
    assert r["n_gbm_earlier"] == 1
    assert r["median_diff"] == pytest.approx(2.0)


def test_stay_alarming_only_at_boundary_is_caught():
    # GBM's only crossing is h6 -> that IS its first alarm now.
    stays = {1: (12, [0, 1, 1, 1, 1], [1, 0, 0, 0, 0])}
    df, n2, gb = _frame(stays, start_hour=STAB_HOURS)
    r = paired_first_alarm(df, n2, gb, 0.5, 0.5)
    assert r["n_paired"] == 1
    assert r["median_diff"] == pytest.approx(1.0)     # NEWS2 h7 - GBM h6


def test_stay_series_keeps_every_prediction_hour():
    stays = {1: (12, [0, 1, 1], [0, 0, 0])}
    df, n2, _ = _frame(stays, start_hour=STAB_HOURS)   # hours 6, 7, 8
    risk, events, times = _stay_series(df, n2)
    assert len(risk) == 1
    assert times[0].tolist() == [STAB_HOURS, STAB_HOURS + 1, STAB_HOURS + 2]
    assert risk[0].tolist() == [0, 1, 1]
    assert events[0] == 12.0
