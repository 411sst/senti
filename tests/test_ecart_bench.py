"""Pure-helper tests for the T19 eCART bench (src/ecart_bench.py).

All data is synthetic: small integer stay ids, deterministic patterns.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ecart_bench import (band_anchor_stats, cohort_bands, first_alarm_hours,
                         pre_event_news2, standing_alarm_share)


def test_cohort_bands_smallest_integer_thresholds():
    # 70 zeros, 10 sixes, 10 sevens, 7 nines, 3 elevens
    n2 = np.array([0] * 70 + [6] * 10 + [7] * 10 + [9] * 7 + [11] * 3, float)
    bands = cohort_bands(n2)
    assert [b["threshold"] for b in bands] == [7, 8, 10]
    assert [b["val_volume"] for b in bands] == pytest.approx([0.20, 0.10, 0.03])
    assert [b["percentile"] for b in bands] == [75, 90, 95]


def test_cohort_bands_respects_alertable_mask():
    n2 = np.array([0] * 70 + [6] * 10 + [7] * 10 + [9] * 7 + [11] * 3, float)
    mask = n2 < 11                     # 97 rows left, none reach 10
    bands = cohort_bands(n2, alertable=mask)
    assert [b["threshold"] for b in bands] == [7, 8, 10]
    assert bands[2]["val_volume"] == 0.0


def test_cohort_bands_terminates_past_the_max_score():
    bands = cohort_bands(np.full(100, 5.0), tail_shares=(0.001,))
    assert bands[0]["threshold"] == 6
    assert bands[0]["val_volume"] == 0.0


def _stay(stay_id, event_h, news2_by_hour, model_by_hour, h0=6, h1=25):
    rows = []
    for h in range(h0, h1 + 1):
        rows.append({"stay_id": stay_id, "hour": float(h),
                     "event_h": event_h,
                     "news2": news2_by_hour(h), "m": model_by_hour(h)})
    return rows


@pytest.fixture()
def anchor_frame():
    """Six stays against band >=7 at model threshold 0.5.

    1  event h20, model alarm h10, NEWS2 first >=7 at h14   -> gap +4
    2  event h20, model alarm h10, NEWS2 >=7 from h6        -> band first
    3  event h20, model alarm h10, NEWS2 never >=7          -> never in band
    4  no event, model alarm h8                              -> not a catch
    5  event h20, model never alarms                         -> not a catch
    6  event h10, model alarm h15 (after the event)          -> not a catch
    """
    rows = []
    rows += _stay(1, 20.0, lambda h: 7.0 if h >= 14 else 4.0,
                  lambda h: 0.9 if h >= 10 else 0.1)
    rows += _stay(2, 20.0, lambda h: 8.0,
                  lambda h: 0.9 if h >= 10 else 0.1)
    rows += _stay(3, 20.0, lambda h: 5.0,
                  lambda h: 0.9 if h >= 10 else 0.1)
    rows += _stay(4, np.nan, lambda h: 4.0,
                  lambda h: 0.9 if h >= 8 else 0.1)
    rows += _stay(5, 20.0, lambda h: 4.0, lambda h: 0.1)
    rows += _stay(6, 10.0, lambda h: 4.0,
                  lambda h: 0.9 if h >= 15 else 0.1)
    return pd.DataFrame(rows)


def test_band_anchor_stats(anchor_frame):
    r = band_anchor_stats(anchor_frame, anchor_frame["m"], 0.5, 7.0)
    assert r["n_caught"] == 3                         # stays 1, 2, 3
    assert r["share_before_stay_ever_in_band"] == pytest.approx(2 / 3)
    assert r["share_never_in_band_pre_event"] == pytest.approx(1 / 3)
    assert r["n_crossing_later"] == 1
    assert r["gap_median_h"] == pytest.approx(4.0)


def test_first_alarm_hours_drops_stays_that_never_cross(anchor_frame):
    fa = first_alarm_hours(anchor_frame, anchor_frame["m"], 0.5)
    assert 5 not in fa.index
    assert fa.loc[1] == 10.0 and fa.loc[4] == 8.0


def test_standing_alarm_share(anchor_frame):
    # stay 2 is >=7 on its first scored hour (h6); stay 1 crosses at h14;
    # stay 3 never crosses and must drop out of the denominator
    share = standing_alarm_share(anchor_frame, anchor_frame["news2"], 7.0,
                                 pd.Series([1, 2, 3]))
    assert share == pytest.approx(0.5)


def test_pre_event_news2_window(anchor_frame):
    # only pre-event hours inside the window count, one row per stay-hour
    one = anchor_frame[anchor_frame["stay_id"] == 1].copy()
    one["news2"] = one["hour"]                       # news2 == hour
    r = pre_event_news2(one, 4)
    assert r["n_hours"] == 4                          # hours 16..19
    assert r["mean"] == pytest.approx(17.5)
    assert r["median"] == pytest.approx(17.5)
