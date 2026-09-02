"""The low end of the NEWS2 band, encoded as executable tests.

The objection (mentor-review prep, Sep 2026): "racing NEWS2 at a volume-matched
threshold of ~11 is cheating -- the ward rule escalates at >=5, so by 11 the
patient is already being seen."  These tests fabricate the ICU situation that
objection lives in -- a cohort whose NEWS2 baseline sits at 5-6 from admission
(chronic ICU elevation), with a genuine NEWS2 spike and an earlier model ramp
only in the hours before the event -- and pin down what each choice of
operating point actually measures:

1. At the ward rule (>=5) with the model held to the SAME volume, the race
   degenerates: every stay "alarms" at its first scored hour and every pair is
   a tie.  Zero timing information at ward volume.
2. At the ward rule with the model at its deployable threshold, NEWS2 "wins"
   every pair by chart position: its first alarm is the stay's first scored
   hour (a standing alarm), not a detection.
3. Volume-matched at a deployable budget, the threshold freezing itself pushes
   NEWS2's threshold out of the baseline band and the race measures detection:
   the model's pre-event ramp beats NEWS2's pre-event spike by exactly the
   hours built into the data.
4. The alarm-burden face of the same fact: >=5 alarms on 100% of event-free
   synthetic stays; the volume-matched operating point alarms on none.

All data below is synthetic (small integer stay ids, deterministic patterns);
no MIMIC values are used.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from baselines_mimic import STAB_HOURS
from fair_capability import frozen_thresholds, paired_first_alarm_diffs

N_EVENT, N_QUIET = 25, 75
HOURS = 48                       # scored hours per stay: STAB_HOURS .. +47
EVENT_H = 40.0                   # event hour for every event stay
RAMP_START = 34                  # model ramp: hours 34..39 at 0.9
SPIKE = {36: 9.0, 37: 10.0, 38: 11.0, 39: 12.0}   # NEWS2's real deterioration
WARD_THR = 5.0


def synthetic_icu(first_stay_id=1):
    """Chronic-baseline ICU cohort: NEWS2 5-6 every hour from admission."""
    rows = []
    for k in range(N_EVENT + N_QUIET):
        stay = first_stay_id + k
        is_event = k < N_EVENT
        for h in range(STAB_HOURS, STAB_HOURS + HOURS):
            news2 = 5.0 + ((stay + h) % 2)             # 5 or 6, always >= ward rule
            model = 0.10 + 0.001 * h                   # quiet baseline < 0.2
            if is_event:
                if h in SPIKE:
                    news2 = SPIKE[h]
                if RAMP_START <= h < EVENT_H:
                    model = 0.9
            rows.append({"stay_id": stay, "hour": h,
                         "event_h": EVENT_H if is_event else np.nan,
                         "n2": news2, "m": model})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def val():
    return synthetic_icu(first_stay_id=1)


@pytest.fixture(scope="module")
def test_frame():
    return synthetic_icu(first_stay_id=1001)


def test_ward_volume_race_is_all_ties(val, test_frame):
    """Model matched to the ward rule's volume: the race carries no signal."""
    alertable = np.ones(len(val), dtype=bool)
    matched = float((val["n2"].to_numpy() >= WARD_THR).mean())
    assert matched == pytest.approx(1.0)               # >=5 covers every hour here
    thr_model = float(np.quantile(val["m"].to_numpy(), 1.0 - matched))
    diffs = paired_first_alarm_diffs(
        test_frame, test_frame["n2"], test_frame["m"], WARD_THR, thr_model)
    assert len(diffs) == N_EVENT
    assert (diffs["diff"] == 0.0).all()                # both alarm at the first hour


def test_ward_rule_wins_by_chart_position_not_detection(test_frame):
    """NEWS2 at >=5 vs the model at its deployable threshold: NEWS2 'wins'
    every pair, but its first alarm is the stay's first scored hour."""
    diffs = paired_first_alarm_diffs(
        test_frame, test_frame["n2"], test_frame["m"], WARD_THR, 0.9)
    assert len(diffs) == N_EVENT
    # every "win" is first-scored-hour minus the model's genuine ramp start
    assert (diffs["diff"] == STAB_HOURS - RAMP_START).all()
    assert (diffs["diff"] < 0).all()                   # NEWS2 "earlier" on 100%
    # the standing-alarm check: NEWS2 crosses >=5 on the very first scored hour
    first_alarm = (test_frame[test_frame["n2"] >= WARD_THR]
                   .groupby("stay_id")["hour"].min())
    assert (first_alarm == STAB_HOURS).all()


def test_volume_matching_leaves_the_baseline_band(val, test_frame):
    """Freezing at a deployable budget pushes the NEWS2 threshold out of the
    chronic 5-6 band, and the race then measures detection order."""
    alertable = np.ones(len(val), dtype=bool)
    budget = 0.02
    thr_n2, thr_m, matched = frozen_thresholds(
        val["n2"].to_numpy(), val["m"].to_numpy(), alertable, budget)
    assert thr_n2 == pytest.approx(9.0)                # out of the 5-6 baseline
    assert matched == pytest.approx(N_EVENT * len(SPIKE) / len(val))
    assert thr_m == pytest.approx(0.9)

    diffs = paired_first_alarm_diffs(
        test_frame, test_frame["n2"], test_frame["m"], thr_n2, thr_m)
    assert len(diffs) == N_EVENT
    # model ramp (h34) vs NEWS2's own deterioration spike (h36): +2h, all stays
    assert (diffs["diff"] == min(SPIKE) - RAMP_START).all()
    assert (diffs["diff"] > 0).all()


def test_alarm_burden_on_event_free_stays(val, test_frame):
    """The alarm-fatigue face: >=5 alarms every quiet stay; matched alarms none."""
    quiet = test_frame[test_frame["event_h"].isna()]
    ward = quiet[quiet["n2"] >= WARD_THR].groupby("stay_id").size()
    assert len(ward) == N_QUIET                        # 100% of quiet stays alarm

    alertable = np.ones(len(val), dtype=bool)
    thr_n2, thr_m, _ = frozen_thresholds(
        val["n2"].to_numpy(), val["m"].to_numpy(), alertable, 0.02)
    assert (quiet["n2"] >= thr_n2).sum() == 0          # NEWS2 @ matched: silent
    assert (quiet["m"] >= thr_m).sum() == 0            # model @ matched: silent
