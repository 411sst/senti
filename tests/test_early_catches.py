"""Tests for the T33 early-catch helpers in scripts/paper_operating_points.py.

Synthetic stays with hand-checked answers: a stay is caught when its first
alarm falls at or before the event, caught >= 6 h ahead when the warning is at
least six hours, and quiet stays never enter the event-stay table.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from paper_operating_points import early_flags, early_stats, stay_first_alarms  # noqa: E402


def _test_frame():
    # (stay, patient, first hour, last hour, event hour or None, hours that alarm)
    spec = [(1, 101, 6, 20, 20.0, {10, 11, 18}),
            (2, 102, 6, 15, 15.0, {12}),
            (3, 103, 6, 15, 15.0, {6}),
            (4, 104, 6, 15, None, {8}),
            (5, 105, 6, 12, 12.0, set())]
    rows = []
    for stay, pid, h0, h1, ev, alarms in spec:
        for h in range(h0, h1 + 1):
            rows.append({"stay_id": stay, "patient_id": pid, "hour": h,
                         "event_h": np.nan if ev is None else ev,
                         "score": 0.9 if h in alarms else 0.1})
    return pd.DataFrame(rows)


def test_first_alarm_table_and_stats():
    t = _test_frame()
    tab = stay_first_alarms(t, t["score"].to_numpy(), 0.5)
    assert tab["stay_id"].tolist() == [1, 2, 3, 5]          # quiet stay 4 dropped
    assert tab["first_alarm"].tolist()[:3] == [10.0, 12.0, 6.0]
    assert np.isnan(tab["first_alarm"].iloc[3])
    st = early_stats(tab)
    assert st["n_event_stays"] == 4
    assert st["catch"] == pytest.approx(3 / 4)
    assert st["catch_6h"] == pytest.approx(2 / 4)            # warnings 10, 3, 9
    assert st["warning_median"] == pytest.approx(9.0)
    assert early_flags(tab).tolist() == [1.0, 0.0, 1.0, 0.0]


def test_first_hour_alarms_can_be_ignored():
    t = _test_frame()
    tab = stay_first_alarms(t, t["score"].to_numpy(), 0.5, skip_first_hour=True)
    st = early_stats(tab)
    assert np.isnan(tab.loc[tab.stay_id == 3, "first_alarm"]).all()
    assert st["catch"] == pytest.approx(2 / 4)
    assert st["catch_6h"] == pytest.approx(1 / 4)


def test_alarm_after_the_event_is_not_a_catch():
    t = _test_frame()
    s = t["score"].to_numpy().copy()
    s[(t.stay_id == 5).to_numpy()] = 0.1
    t.loc[t.stay_id == 5, "event_h"] = 9.0                   # event before any alarm
    s[((t.stay_id == 5) & (t.hour == 11)).to_numpy()] = 0.9
    tab = stay_first_alarms(t, s, 0.5)
    assert np.isnan(tab.loc[tab.stay_id == 5, "first_alarm"]).all()
