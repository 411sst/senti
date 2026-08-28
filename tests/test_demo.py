"""Tests for the demo-dashboard bundle helpers (no streamlit, no data)."""

import math

import numpy as np
import pandas as pd
import pytest

from demo_bundle import (
    DEATH_ROLE, ROLES, assert_aggregate_only, assert_no_identifiers,
    dumps_strict, first_alarm_hour, nan_to_none, select_demo_stays,
    stay_label, stay_summary, t_head_from_composite, validate_bundle,
)


# --- first alarms -----------------------------------------------------------------

def _stay(stay_id, hours, model, news2, event_h, event_type):
    return pd.DataFrame({"stay_id": stay_id, "hour": hours, "event_h": event_h,
                         "event_type": event_type, "m": model, "n2": news2})


def test_first_alarm_hour_is_the_first_crossing_or_nan():
    assert first_alarm_hour([6, 7, 8, 9], [0.1, 0.3, 0.2, 0.4], 0.25) == 7.0
    assert math.isnan(first_alarm_hour([6, 7], [0.1, 0.2], 0.5))


def test_stay_summary_lead_matches_the_paired_race():
    """The demo's lead per stay must be the paper's lead per stay."""
    from fair_capability import paired_first_alarm_diffs

    frame = pd.concat([
        _stay(1, [6, 7, 8, 9, 10], [0.1, 0.3, 0.3, 0.3, 0.3], [5, 5, 5, 12, 12], 10.0, "vent"),
        _stay(2, [6, 7, 8], [0.1, 0.1, 0.1], [12, 12, 12], 8.0, "vaso"),      # model never
        _stay(3, [6, 7, 8], [0.3, 0.3, 0.3], [12, 12, 12], np.nan, None),     # no event
    ], ignore_index=True).sort_values(["stay_id", "hour"]).reset_index(drop=True)
    summary = stay_summary(frame, frame["m"], frame["n2"], 0.25, 11.0)
    paper = paired_first_alarm_diffs(frame, frame["n2"], frame["m"], 11.0, 0.25)

    assert paper["stay_id"].tolist() == [1]
    row = summary.set_index("stay_id").loc[1]
    assert row["fa_news2"] - row["fa_model"] == paper["diff"].iloc[0] == 2.0
    assert math.isnan(summary.set_index("stay_id").loc[2, "fa_model"])
    assert summary.set_index("stay_id").loc[3, "event_type"] is None
    assert summary.set_index("stay_id").loc[3, "n_hours"] == 3


def test_t_head_round_trips_and_survives_a_certain_death_head():
    t, d = np.array([0.2, 0.5, 0.0]), np.array([0.1, 0.4, 1.0])
    comp = 1 - (1 - t) * (1 - d)
    back = t_head_from_composite(comp, d)
    assert np.allclose(back[:2], t[:2])
    assert np.isfinite(back).all() and 0.0 <= back[2] <= 1.0


# --- curated stay selection --------------------------------------------------

def _summary(rows):
    cols = ("stay_id", "h0", "h1", "n_hours", "fa_model", "fa_news2", "event_h",
            "event_type")
    return pd.DataFrame(rows, columns=cols)


GOOD_VENT = (30000001, 6, 40, 35, 14, 20, 30, "vent")
GOOD_VASO = (30000002, 6, 40, 35, 14, 20, 30, "vaso")
GOOD_DEATH = (30000003, 6, 40, 35, 14, 20, 30, "death")
GOOD_N2 = (30000004, 6, 40, 35, 20, 14, 30, "vent")
GOOD_QUIET = (30000005, 6, 40, 35, np.nan, np.nan, np.nan, None)
DECOYS = [
    (30000011, 6, 40, 35, 14, 15, 30, "vent"),      # lead only 1 h
    (30000012, 6, 40, 35, 9, 20, 30, "vent"),       # alarm 3 h into the stay
    (30000013, 6, 40, 35, 14, 20, 17, "vent"),      # event 11 h into the stay
    (30000014, 6, 40, 35, 14, 20, 45, "vent"),      # alarm > 24 h before event
    (30000015, 6, 200, 195, 14, 20, 30, "vent"),    # too long to plot
    (30000016, 6, 40, 35, 14, 32, 30, "vent"),      # NEWS2 alarm after the event
    (30000017, 6, 40, 35, np.nan, 14, np.nan, None),  # quiet but NEWS2 fires
    (30000018, 6, 100, 95, np.nan, np.nan, np.nan, None),  # quiet but too long
]


def test_select_demo_stays_picks_exactly_the_qualifying_member_per_role():
    summary = _summary([GOOD_VENT, GOOD_VASO, GOOD_DEATH, GOOD_N2, GOOD_QUIET]
                       + DECOYS)
    chosen = select_demo_stays(summary, seed=42)
    assert set(chosen) == set(ROLES)
    assert chosen == {"vent-model-first": 30000001, "vaso-model-first": 30000002,
                      "news2-first": 30000004, "quiet": 30000005}
    assert chosen == select_demo_stays(summary, seed=42)          # deterministic
    assert select_demo_stays(summary, with_death=True)[DEATH_ROLE] == 30000003


def test_select_demo_stays_ranks_wins_by_lead_and_losses_by_typicality():
    bigger_win = (30000021, 6, 40, 35, 12, 20, 30, "vent")        # lead 8 vs 6
    mild_loss = (30000031, 6, 40, 35, 17, 14, 30, "vent")         # lead -3
    bad_loss = (30000032, 6, 40, 35, 26, 14, 30, "vent")          # lead -12
    summary = _summary([GOOD_VENT, bigger_win, GOOD_VASO, GOOD_N2, mild_loss,
                        bad_loss, GOOD_QUIET])
    first = select_demo_stays(summary, seed=42, pick_index=0)
    second = select_demo_stays(summary, seed=42, pick_index=1)
    assert first["vent-model-first"] == 30000021           # clearest win first
    assert second["vent-model-first"] == 30000001
    # losses: -6 (GOOD_N2), -3, -12 -> median -6 -> GOOD_N2 is the typical one
    assert first["news2-first"] == 30000004
    with pytest.raises(ValueError, match="vaso-model-first"):
        select_demo_stays(_summary([GOOD_VENT, GOOD_N2, GOOD_QUIET]))


# --- labels and JSON hygiene -----------------------------------------------

def test_stay_label_never_mentions_the_id():
    stay = {"unit_group": "medical", "event_type": "vent",
            "first_alarm": {"model": 14, "news2": 20, "lead_h": 6.0}}
    label = stay_label(stay)
    assert label == "Ventilation — model alarm 6 h before NEWS2 · medical ICU"
    assert stay_label({**stay, "first_alarm": {"lead_h": -2.0}}).startswith(
        "Ventilation — NEWS2 alarm 2 h before the model")
    assert stay_label({"unit_group": "neuro", "event_type": None}) == \
        "No event — no alarm from either · neuro ICU"


def test_nan_to_none_and_strict_dumps():
    out = nan_to_none({"a": np.float32("nan"), "b": np.array([1, 2]),
                       "c": np.int64(3), "d": (np.bool_(True),)})
    assert out == {"a": None, "b": [1, 2], "c": 3, "d": [True]}
    assert '"a": null' in dumps_strict({"a": float("nan")})


def _bundle(n=3):
    hours = list(range(6, 6 + n))
    arr = [0.1] * n
    return {
        "agents": ["a", "b"], "news2_components": ["resp_rate"],
        "thresholds": {"thr_model": 0.28, "thr_news2": 11.0},
        "attention_source": "single-head refit",
        "stays": [{
            "hours": hours,
            "risk": {k: list(arr) for k in ("composite", "t_head", "d_head", "single_head")},
            "news2": {"total": [4] * n, "subscores": {"resp_rate": [1] * n}},
            "vitals_observed": {"heart_rate": [80.0, None, 82.0][:n]},
            "vitals_model_view": {"heart_rate": [80.0] * n},
            "vitals_hrs_since": {"heart_rate": [1.0, 2.0, None][:n]},
            "agent_risk": {"a": list(arr), "b": list(arr)},
            "attention": {"a": [0.5] * n, "b": [0.5] * n},
            "alarm": {"model": [False] * n, "news2": [False] * n},
            "contexts": {str(hours[0]): {}}, "cards": {str(hours[0]): "x"},
        }],
    }


def test_validate_bundle_accepts_a_minimal_bundle_and_rejects_defects():
    assert validate_bundle(_bundle())
    short = _bundle(); short["stays"][0]["attention"]["a"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="attention.a"):
        validate_bundle(short)
    nan = _bundle(); nan["stays"][0]["risk"]["composite"][1] = float("nan")
    with pytest.raises(ValueError, match="risk.composite"):
        validate_bundle(nan)
    missing = _bundle(); del missing["stays"][0]["alarm"]
    with pytest.raises(ValueError, match="alarm"):
        validate_bundle(missing)
    stray = _bundle(); stray["stays"][0]["cards"]["99"] = "x"
    with pytest.raises(ValueError, match="cards"):
        validate_bundle(stray)


def test_assert_aggregate_only_catches_ids_at_any_depth():
    clean = {"headline": {"auprc": 0.308, "n_test_hours": 361229},
             "burden": {"units": [{"unit": "medical", "hours": 51234}]},
             "git_commit": "30000000", "footprint": {"single_stay_hours": 40}}
    assert assert_aggregate_only(clean)
    with pytest.raises(ValueError, match="stay_id"):
        assert_aggregate_only({"x": [{"stay_id": 1}]})
    with pytest.raises(ValueError, match="patient_id"):
        assert_aggregate_only({"a": {"b": {"patient_id": 5}}})
    with pytest.raises(ValueError, match="hour"):
        assert_aggregate_only({"stays": [{"hour": 6}]})
    with pytest.raises(ValueError, match="stay id"):
        assert_aggregate_only({"ids": [30004306]})
    with pytest.raises(ValueError, match="stay-id-like"):
        assert_aggregate_only({"note": "see 30004306 for details"})


def test_assert_no_identifiers_reads_the_rendered_page():
    assert assert_no_identifiers("<p>AUPRC 0.308 over 361229 hours</p>")
    with pytest.raises(ValueError):
        assert_no_identifiers("<td>stay_id</td>")
    with pytest.raises(ValueError):
        assert_no_identifiers("<td>30004306</td>")
