"""T49 bridge features: strictly before t, 48 h window, stays kept apart.
Synthetic stays only."""

import numpy as np
import pandas as pd
import pytest

from t49_bridge import CHANNELS, bridge_design, feature_names


def _hourly(stays_hours, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for stay, hours in stays_hours.items():
        for h in hours:
            row = {"stay_id": stay, "hour": h}
            row.update({c: float(rng.uniform(1, 100)) if rng.random() < 0.6 else np.nan
                        for c in CHANNELS})
            rows.append(row)
    return pd.DataFrame(rows)


def _stays(los):
    return pd.DataFrame({"stay_id": list(los), "los_h": list(los.values()),
                         "anchor_age": [70] * len(los), "gender": ["M", "F"][:len(los)]
                         if len(los) <= 2 else ["F"] * len(los)})


def _keys(pairs):
    return pd.DataFrame(pairs, columns=["stay_id", "hour"])


def test_features_at_t_ignore_bin_t_and_later():
    hourly = _hourly({1010: range(0, 100)})
    stays = _stays({1010: 99.5})
    keys = _keys([(1010, t) for t in (6, 30, 60)])
    base = bridge_design(hourly, stays, keys)
    for t in (6, 30, 60):
        moved = hourly.copy()
        moved.loc[moved.hour >= t, CHANNELS] += 1000.0
        row = keys.index[keys.hour == t][0]
        pd.testing.assert_series_equal(bridge_design(moved, stays, keys).iloc[row],
                                       base.iloc[row])


def test_bin_t_minus_one_is_the_last_value():
    hourly = _hourly({1010: range(0, 50)})
    hourly.loc[hourly.hour == 19, "heart_rate"] = 123.0
    out = bridge_design(hourly, _stays({1010: 49.0}), _keys([(1010, 20)]))
    assert out.loc[0, "heart_rate_last"] == 123.0
    assert out.loc[0, "heart_rate_hrs_since"] == 1


def test_window_is_the_48_bins_before_t():
    hourly = pd.DataFrame({"stay_id": [1010, 1010], "hour": [10, 11]})
    for c in CHANNELS:
        hourly[c] = np.nan
    hourly.loc[hourly.hour == 10, "sodium"] = 130.0     # t - 49 at t = 59: outside
    hourly.loc[hourly.hour == 11, "sodium"] = 140.0     # t - 48 at t = 59: inside
    out = bridge_design(hourly, _stays({1010: 80.0}), _keys([(1010, 59), (1010, 60)]))
    assert out.loc[0, "sodium_mean48"] == 140.0 and out.loc[0, "sodium_min48"] == 140.0
    assert out.loc[0, "sodium_last"] == 140.0 and out.loc[0, "sodium_hrs_since"] == 48
    assert np.isnan(out.loc[1, "sodium_mean48"]) and np.isnan(out.loc[1, "sodium_last"])


def test_stays_do_not_leak_into_each_other():
    hourly = _hourly({1010: range(0, 40), 1020: range(0, 40)})
    stays = _stays({1010: 39.0, 1020: 39.0})
    both = bridge_design(hourly, stays, _keys([(1020, 6), (1020, 30)]))
    alone = bridge_design(hourly[hourly.stay_id == 1020], stays[stays.stay_id == 1020],
                          _keys([(1020, 6), (1020, 30)]))
    pd.testing.assert_frame_equal(both, alone)
    first = bridge_design(hourly, stays, _keys([(1020, 0)]))
    assert first.drop(columns=["age", "male"]).isna().all(axis=None)


def test_columns_order_static_and_out_of_stay_keys():
    hourly = _hourly({1010: range(0, 20), 1020: range(0, 20)})
    stays = _stays({1010: 19.0, 1020: 19.0})
    out = bridge_design(hourly, stays, _keys([(1020, 10), (1010, 10)]))
    assert list(out.columns) == feature_names()
    assert out["male"].tolist() == [0.0, 1.0] and out["age"].tolist() == [70.0, 70.0]
    with pytest.raises(KeyError):
        bridge_design(hourly, stays, _keys([(1010, 25)]))


def test_long_stays_and_many_stays_on_one_grid():
    """A window at a stay's first hours stays inside that stay even when the
    previous stay is long and full of values; hundreds of stays build fine."""
    hourly = _hourly({1010: range(0, 300), 1020: range(0, 10)})
    hourly.loc[hourly.stay_id == 1020, CHANNELS] = np.nan
    stays = _stays({1010: 299.0, 1020: 9.0})
    out = bridge_design(hourly, stays, _keys([(1020, h) for h in range(10)]))
    assert out.drop(columns=["age", "male"]).isna().all(axis=None)
    many = {2000 + i: range(0, 60) for i in range(400)}
    big = _hourly(many, seed=1)
    st = pd.DataFrame({"stay_id": list(many), "los_h": 59.0, "anchor_age": 60, "gender": "F"})
    k = _keys([(s, 30) for s in many])
    one = bridge_design(big, st, k)
    solo = bridge_design(big[big.stay_id == 2399], st[st.stay_id == 2399], _keys([(2399, 30)]))
    pd.testing.assert_series_equal(one.iloc[-1].rename(None), solo.iloc[0].rename(None))
