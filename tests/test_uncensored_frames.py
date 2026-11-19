"""T44-T49: rows outside our cohort (no stabilisation filter, no censoring).
Synthetic stays only."""

import numpy as np
import pandas as pd
import pytest

from uncensored_frames import pick_rows, uncensored_frame

FEATURES = ["heart_rate", "resp_rate", "spo2", "sbp", "temperature",
            "heart_rate_hrs_since", "gcs", "news2", "on_oxygen"]


def _raw(events, last_hour=30, seed=0):
    """Uncensored raw rows, hours 0..last_hour, for ``{stay_id: vent_h}``."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, (stay, vent) in enumerate(events.items()):
        for h in range(last_hour + 1):
            rows.append({
                "stay_id": stay, "subject_id": 50 + k, "hour": h, "is_history": int(h < 6),
                "heart_rate": float(rng.uniform(50, 140)), "resp_rate": float(rng.uniform(8, 30)),
                "spo2": float(rng.uniform(85, 100)), "sbp": float(rng.uniform(80, 180)),
                "temperature": float(rng.uniform(35, 39)) if h % 4 == 0 else np.nan,
                "gcs": float(rng.integers(3, 16)) if h % 3 == 0 else np.nan,
                "gcs_verbal": np.nan, "gcs_motor": np.nan, "gcs_eyes": np.nan,
                "gcs_unable": np.nan, "o2_flow": np.nan,
                "o2_device": float(h % 2) if h % 5 == 0 else np.nan, "fio2": np.nan,
                "event_h": vent, "death_h": np.nan, "vent_h": vent, "vaso_h": np.nan,
                "label_24h": int(not np.isnan(vent) and h < vent <= h + 24),
            })
    return pd.DataFrame(rows)


def test_features_at_t_use_only_hours_before_t():
    raw = _raw({1010: np.nan})
    base = uncensored_frame(raw)
    for t in (6, 12, 20):
        later = raw.copy()
        later.loc[later.hour >= t, ["heart_rate", "spo2", "temperature", "gcs"]] += 7.0
        moved = uncensored_frame(later)
        upto = base.hour <= t
        pd.testing.assert_frame_equal(base.loc[upto, FEATURES], moved.loc[upto, FEATURES])
        earlier = raw.copy()
        earlier.loc[earlier.hour == t - 1, "heart_rate"] += 7.0
        changed = uncensored_frame(earlier)
        row = base.index[base.hour == t][0]
        assert changed.loc[row, "heart_rate"] == base.loc[row, "heart_rate"] + 7.0


def test_no_censoring_and_no_stabilisation_filter():
    raw = _raw({1010: 10.0, 1020: 3.0, 1030: np.nan})
    out = uncensored_frame(raw)
    for stay in (1010, 1020, 1030):
        assert out.loc[out.stay_id == stay, "hour"].tolist() == list(range(6, 31))
    s = out[out.stay_id == 1010].set_index("hour")
    assert s.loc[6:9, "label_24h_vent"].eq(1).all()      # event in (h, h + 24]
    assert s.loc[10:30, "label_24h_vent"].eq(0).all()    # event no longer ahead
    assert out.loc[out.stay_id == 1020, "label_24h_vent"].eq(0).all()


def test_stays_are_independent_and_selectable():
    raw = _raw({1010: np.nan, 1020: 15.0})
    both = uncensored_frame(raw)
    one = uncensored_frame(raw, stays=[1020])
    pd.testing.assert_frame_equal(
        both[both.stay_id == 1020].reset_index(drop=True)[FEATURES], one[FEATURES])


def test_pick_rows_follows_key_order_and_rejects_missing_rows():
    out = uncensored_frame(_raw({1010: np.nan, 1020: np.nan}))
    keys = pd.DataFrame({"stay_id": [1020, 1010, 1020], "hour": [6, 30, 12]})
    pos = pick_rows(out, keys)
    assert out.loc[pos, ["stay_id", "hour"]].to_numpy().tolist() == [[1020, 6], [1010, 30],
                                                                     [1020, 12]]
    with pytest.raises(KeyError):
        pick_rows(out, pd.DataFrame({"stay_id": [1010], "hour": [5]}))   # history hour


def test_reconcile_takes_old_values_on_shared_bins_only():
    from uncensored_frames import reconcile
    new = _raw({1010: np.nan, 1020: np.nan}, last_hour=10)
    old = new[(new.stay_id == 1010) & (new.hour <= 5)].copy()
    old["heart_rate"] = old["heart_rate"] + 1e-13          # last-bit differences
    old.loc[old.hour == 2, "spo2"] = np.nan                 # a value the old bin lacked
    out, n = reconcile(new, old, ["heart_rate", "spo2"])
    assert n == 6
    shared = (out.stay_id == 1010) & (out.hour <= 5)
    np.testing.assert_array_equal(out.loc[shared, "heart_rate"].to_numpy(),
                                  old["heart_rate"].to_numpy())
    assert np.isnan(out.loc[shared & (out.hour == 2), "spo2"]).all()
    pd.testing.assert_frame_equal(out[~shared].reset_index(drop=True),
                                  new[~shared].reset_index(drop=True), check_dtype=False)
