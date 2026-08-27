"""Tests for charted code-status extraction.

The load-bearing guarantees: a palliative care plan is never counted as a
limitation of care (it is a conversation, not a decision), an unrecognised
status value fails loudly rather than silently landing in the unrestricted
group, and the before-event comparison uses strict inequality.
"""

import numpy as np
import pandas as pd
import pytest

from code_status import (
    attach_to_stays,
    classify,
    classify_census,
    coverage_report,
    load_code_status,
    stay_timeline,
)

# Every distinct value present in the real extract.
REAL_VALUES = [
    "Full code", "DNR / DNI", "DNR (do not resuscitate)", "Comfort measures only",
    "DNAR (Do Not Attempt Resuscitation) [DNR] / DNI",
    "DNAR (Do Not Attempt Resuscitation)  [DNR]", "DNI (do not intubate)",
]


def _records(rows):
    """rows: (stay_id, hour, value, kind)"""
    df = pd.DataFrame(rows, columns=["stay_id", "hour", "value", "kind"])
    df["itemid"] = 223758
    is_cs = df["kind"].eq("code_status")
    flags = pd.DataFrame(
        [classify(v) if k else {"is_full": False, "is_cmo": False,
                                "limits_cpr": False, "limits_intubation": False}
         for v, k in zip(df["value"], is_cs)], index=df.index)
    flags["is_palliative_plan"] = ~is_cs
    return pd.concat([df, flags], axis=1)


# --- classification ----------------------------------------------------------

@pytest.mark.parametrize("value", REAL_VALUES)
def test_every_real_value_is_recognised(value):
    assert any(classify(value).values()), f"{value!r} matched nothing"


def test_full_code_is_not_a_limitation():
    f = classify("Full code")
    assert f["is_full"] and not f["limits_cpr"] and not f["limits_intubation"]


def test_comfort_measures_implies_both_limits():
    """A CMO patient is neither resuscitated nor intubated -- coding it as CMO
    alone would leave the stay counted as unrestricted for the vent arm."""
    f = classify("Comfort measures only")
    assert f["is_cmo"] and f["limits_cpr"] and f["limits_intubation"]


def test_dnr_and_dni_are_tracked_separately():
    assert classify("DNR (do not resuscitate)")["limits_cpr"]
    assert not classify("DNR (do not resuscitate)")["limits_intubation"]
    assert classify("DNI (do not intubate)")["limits_intubation"]
    assert not classify("DNI (do not intubate)")["limits_cpr"]
    both = classify("DNR / DNI")
    assert both["limits_cpr"] and both["limits_intubation"]


def test_classify_raises_on_an_unknown_value():
    with pytest.raises(ValueError, match="unmapped"):
        classify("Limited interventions")


def test_missing_value_is_not_unknown():
    """None/"" are absence, not novelty -- they must not raise."""
    for v in (None, "", "   "):
        assert not any(classify(v).values())


def test_census_raises_on_an_unrecognised_value():
    with pytest.raises(ValueError, match="unmapped"):
        classify_census(pd.Series(["Full code", "Limited interventions"]))


def test_census_is_quiet_when_everything_maps():
    assert classify_census(pd.Series(REAL_VALUES)) == {}


# --- timeline ----------------------------------------------------------------

def test_timeline_picks_the_first_hour_of_each_limitation():
    rec = _records([
        (1, 2.0, "Full code", "code_status"),
        (1, 30.0, "DNR (do not resuscitate)", "code_status"),
        (1, 48.0, "Comfort measures only", "code_status"),
    ])
    t = stay_timeline(rec).iloc[0]
    assert t.first_status_h == 2.0 and not t.first_status_is_limit
    assert t.first_limit_h == 30.0
    assert t.first_cpr_limit_h == 30.0
    assert t.first_intub_limit_h == 48.0     # DNR alone does not limit intubation
    assert t.first_cmo_h == 48.0
    assert t.ever_full and t.ever_limited


def test_palliative_plan_is_never_a_limitation():
    rec = _records([
        (7, 5.0, "Encourage family discussion", "palliative_plan"),
        (7, 9.0, "Yes", "palliative_plan"),
    ])
    t = stay_timeline(rec).iloc[0]
    assert t.has_palliative_plan and not t.has_code_status
    assert not t.ever_limited
    assert np.isnan(t.first_limit_h)
    assert t.first_palliative_h == 5.0


def test_negative_hours_are_kept():
    """A code status set before the patient reached the unit is still a real,
    earlier decision -- clipping it would understate the lead."""
    rec = _records([(3, -12.0, "DNR / DNI", "code_status")])
    assert stay_timeline(rec).iloc[0].first_limit_h == -12.0


# --- attach ------------------------------------------------------------------

def _stays():
    return pd.DataFrame({"stay_id": [1, 2, 3], "event_h": [40.0, np.nan, 10.0]})


def test_attach_preserves_rows_and_flags_missing_stays():
    rec = _records([(1, 30.0, "DNR / DNI", "code_status")])
    out = attach_to_stays(_stays(), stay_timeline(rec))
    assert len(out) == 3
    assert list(out.has_code_status) == [True, False, False]
    assert list(out.limit_before_event) == [True, False, False]
    assert out.loc[0, "hours_limit_to_event"] == pytest.approx(10.0)


def test_a_limitation_after_the_event_does_not_count():
    """Strict inequality: a DNR written after the patient died is not evidence
    that the death was decided in advance."""
    rec = _records([(3, 25.0, "Comfort measures only", "code_status")])
    out = attach_to_stays(_stays(), stay_timeline(rec))
    row = out[out.stay_id == 3].iloc[0]
    assert row.has_code_status and not row.limit_before_event
    assert np.isnan(row.hours_limit_to_event)


def test_attach_rejects_a_duplicated_timeline():
    rec = _records([(1, 30.0, "DNR / DNI", "code_status")])
    dup = pd.concat([stay_timeline(rec)] * 2, ignore_index=True)
    with pytest.raises(Exception):
        attach_to_stays(_stays(), dup)


def test_coverage_report_counts_both_evidence_kinds():
    rec = _records([(1, 5.0, "Full code", "code_status"),
                    (2, 5.0, "Interventions", "palliative_plan")])
    att = attach_to_stays(_stays(), stay_timeline(rec))
    att["event_type"] = ["death", "event-free", "event-free"]
    cov = coverage_report(att)
    assert set(cov.columns) >= {"with_code_status", "with_palliative", "any_goc_pct"}
    assert cov.stays.sum() == 3


# --- the real extract, if present --------------------------------------------

def test_real_extract_classifies_cleanly_if_present():
    try:
        rec = load_code_status(strict=True)
    except FileNotFoundError:
        pytest.skip("code-status extract not fetched on this clone")
    assert rec["stay_id"].dtype == np.int64
    cs = rec[~rec["is_palliative_plan"]]
    assert cs[["is_full", "limits_cpr", "limits_intubation"]].any(axis=1).all()
