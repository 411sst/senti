"""Tests for the T35 eICU helpers in scripts/t35_eicu_operating_points.py.

Synthetic stays with hand-checked answers. Ids are small integers.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from t35_eicu_operating_points import (  # noqa: E402
    VITALS,
    _paired,
    _race,
    catch_at_quiet_share,
    positions,
    quiet_alarm_share,
    stay_max_scores,
    regular_mask,
)


def _frame():
    # (stay, patient, hours, event hour or None)
    spec = [(1, 11, range(0, 10), 9.0), (2, 12, range(0, 10), 8.0),
            (3, 13, range(0, 10), 7.0), (4, 14, range(0, 10), None)]
    rows = [{"stay_id": s, "patient_id": p, "hour": h, "event_h": ev}
            for s, p, hours, ev in spec for h in hours]
    return pd.DataFrame(rows)


def _scores(frame, alarm_hours):
    """1.0 at (stay, hour) pairs in ``alarm_hours``, else 0."""
    keys = set(alarm_hours)
    return np.array([1.0 if (s, h) in keys else 0.0
                     for s, h in zip(frame["stay_id"], frame["hour"])])


def test_paired_difference_is_the_seed_mean_minus_the_comparator():
    f = [np.array([1, 0, 1, 0.]), np.array([1, 1, 1, 0.])]
    c = np.array([0, 0, 1, 0.])
    out = _paired(f, c, [np.arange(4), np.array([0, 0, 1, 1])])
    assert out["delta"] == pytest.approx(0.625 - 0.25)
    # second resample: f means 0.5 and 1.0 -> 0.75; c 0.0 -> 0.75
    assert sorted(out["ci"]) == pytest.approx([0.375 + (0.75 - 0.375) * 0.025,
                                               0.375 + (0.75 - 0.375) * 0.975])


def test_race_counts_first_alarms_on_jointly_caught_stays():
    t = _frame()
    # comparator: stay 1 at h5, stay 2 at h2, stay 3 never, stay 4 (quiet) at h1
    comp = _scores(t, [(1, 5), (2, 2), (4, 1)])
    # seed a: stay 1 at h3 (first), stay 2 at h2 (tie), stay 3 at h4
    # seed b: stay 1 at h7 (comparator first), stay 2 at h9 (after the event)
    flag = {"a": _scores(t, [(1, 3), (2, 2), (3, 4)]), "b": _scores(t, [(1, 7), (2, 9)])}
    thr = {"a": 0.5, "b": 0.5}
    order = np.array([11, 12, 13])
    out = _race(t, flag, comp, thr, 0.5, [np.ones(3)], order)
    assert out["n_pair"] == 3                    # a: stays 1, 2; b: stay 1
    assert (out["flagship_first"], out["comparator_first"], out["ties"]) == (1, 1, 1)
    assert out["flagship_first_when_differ"] == pytest.approx(0.5)
    assert out["ci"] == pytest.approx([0.5, 0.5])
    assert out["median_lead_h"] == pytest.approx(0.0)
    # Dropping patient 11 from a resample leaves only the tie: no rate drawn.
    out2 = _race(t, flag, comp, thr, 0.5, [np.array([0, 2, 1.]), np.ones(3)], order)
    assert out2["ci"] == pytest.approx([0.5, 0.5])


def test_regular_mask_needs_all_five_vitals_within_two_hours():
    base = {f"{v}_hrs_since": 1.0 for v in VITALS}
    df = pd.DataFrame([base, {**base, "spo2_hrs_since": 2.0},
                       {**base, "sbp_hrs_since": 3.0}, {**base, "temperature_hrs_since": np.nan}])
    assert regular_mask(df).tolist() == [True, True, False, False]


def test_positions_maps_keys_and_refuses_missing_rows():
    stay, hour = np.array([5, 5, 6]), np.array([1, 2, 1])
    frame = pd.DataFrame({"stay_id": [6, 5], "hour": [1, 2]})
    assert positions(stay, hour, frame).tolist() == [2, 1]
    with pytest.raises(SystemExit):
        positions(stay, hour, pd.DataFrame({"stay_id": [7], "hour": [1]}))


def test_quiet_alarm_share_counts_quiet_stays_with_any_alarm():
    t = _frame()                     # stay 4 is the only quiet stay
    assert quiet_alarm_share(t, _scores(t, [(1, 2)]), 0.5) == (0.0, 1)
    assert quiet_alarm_share(t, _scores(t, [(4, 9)]), 0.5) == (1.0, 1)
    t2 = t.assign(event_h=np.where(t["stay_id"] == 3, np.nan, t["event_h"]))
    assert quiet_alarm_share(t2, _scores(t2, [(3, 0), (3, 5)]), 0.5) == (0.5, 2)


def test_stay_max_ignores_hours_after_the_event():
    t = _frame()                     # events at h9, h8, h7; stay 4 quiet
    sc = _scores(t, [(3, 8), (4, 2)]) + 0.1 * (t["hour"].to_numpy() == 0)
    e, q = stay_max_scores(t, sc)
    assert e.tolist() == pytest.approx([0.1, 0.1, 0.1])   # stay 3's h8 is after its event
    assert q.tolist() == [1.0]


def test_catch_at_quiet_share_takes_the_best_catch_within_the_quiet_budget():
    e = np.array([0.9, 0.8, 0.3, 0.1])
    q = np.array([0.85, 0.5, 0.2, 0.05])
    # no quiet alarms allowed: threshold above 0.85 -> catches only 0.9
    assert catch_at_quiet_share(e, q, 0.0) == pytest.approx(0.25)
    # one quiet stay of four (0.85): best is threshold 0.8 -> catch 0.5
    assert catch_at_quiet_share(e, q, 0.25) == pytest.approx(0.5)
    # threshold 0.3 alarms on 2 of 4 quiet stays and catches 3 of 4;
    # halfway between 1/4 and 2/4 quiet interpolates 0.5 -> 0.75
    assert catch_at_quiet_share(e, q, 0.375) == pytest.approx(0.625)
    # every score above 0.85 is an event stay, so the curve starts at quiet 0;
    # a curve that cannot reach quiet 0 is anchored at (0, 0)
    assert catch_at_quiet_share(np.array([0.5]), np.array([0.9]), 0.0) == pytest.approx(0.0)
    assert catch_at_quiet_share(e, q, 1.0) == pytest.approx(1.0)
