"""T44-T49: rows outside our cohort (no stabilisation filter, no censoring).
Synthetic stays only."""

import numpy as np
import pandas as pd
import pytest

from uncensored_frames import pick_rows, sample_frame, shared_positions, uncensored_frame

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


def _samples(spec):
    """``{stay: (first, last)}`` -> sample keys with a synthetic label."""
    rows = [(s, h) for s, (a, b) in spec.items() for h in range(a, b + 1)]
    df = pd.DataFrame(rows, columns=["stay_id", "hour"])
    df["label"] = (df["hour"] % 3 == 0).astype(np.int8)
    return df.sample(frac=1.0, random_state=0)          # any input order


def test_sample_frame_keeps_the_sample_rows_and_attaches_their_columns():
    frame = uncensored_frame(_raw({1010: np.nan, 1020: 18.0}))
    out = sample_frame(frame, _samples({1010: (6, 12), 1020: (6, 9)}))
    assert list(zip(out.stay_id, out.hour)) == [(1010, h) for h in range(6, 13)] + \
        [(1020, h) for h in range(6, 10)]
    assert (out["label"] == (out["hour"] % 3 == 0)).all()
    ref = frame.set_index(["stay_id", "hour"]).loc[list(zip(out.stay_id, out.hour))]
    assert np.allclose(out["heart_rate"].to_numpy(), ref["heart_rate"].to_numpy())


def test_sample_frame_rows_equal_the_full_frame_rows():
    # truncating a stay after its last sample hour changes no earlier row
    raw = _raw({1010: np.nan})
    full = uncensored_frame(raw)
    cut = uncensored_frame(raw[raw.hour <= 12])
    out = sample_frame(full, _samples({1010: (6, 12)}))
    pd.testing.assert_frame_equal(out[FEATURES], cut[FEATURES].reset_index(drop=True))


@pytest.mark.parametrize("spec", [{1010: (7, 12)}, {1010: (6, 8), 1020: (6, 6)}])
def test_sample_frame_refuses_runs_that_skip_hours(spec):
    frame = uncensored_frame(_raw({1010: np.nan, 1020: np.nan}))
    smp = _samples(spec)
    if 1020 in spec:
        smp = pd.concat([smp, pd.DataFrame({"stay_id": [1010], "hour": [10], "label": [0]})])
    with pytest.raises(ValueError):
        sample_frame(frame, smp)


def test_sample_frame_raises_for_keys_outside_the_frame():
    frame = uncensored_frame(_raw({1010: np.nan}, last_hour=10))
    with pytest.raises(KeyError):
        sample_frame(frame, _samples({1010: (6, 14)}))


def test_shared_positions():
    a = pd.DataFrame({"stay_id": [1010, 1010, 1020, 1030], "hour": [6, 7, 6, 9]})
    b = pd.DataFrame({"stay_id": [1030, 1010, 1040], "hour": [9, 7, 6]})
    pa, pb = shared_positions(a, b)
    assert pa.tolist() == [1, 3] and pb.tolist() == [1, 0]
    with pytest.raises(ValueError):
        shared_positions(pd.concat([a, a]), b)
