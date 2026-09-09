"""Tests for the demo-dashboard bundle helpers (no streamlit, no data)."""

import math

import numpy as np
import pandas as pd
import pytest

from demo_bundle import (
    COMPARATOR_LABELS, DEATH_ROLE, ROLES, alarm_announcements,
    assert_aggregate_only, assert_no_identifiers, budget_sweep_row,
    dumps_strict, first_alarm_hour, first_alarm_table, h2h_summary,
    nan_to_none, select_demo_stays, stay_label, stay_summary,
    t_head_from_composite, truncate_stay, validate_bundle,
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


def test_first_alarm_table_generalizes_stay_summary_to_many_models():
    frame = pd.concat([
        _stay(1, [6, 7, 8, 9], [0.1, 0.3, 0.3, 0.3], [5, 12, 12, 12], 9.0, "vent"),
        _stay(2, [6, 7, 8], [0.1, 0.1, 0.4], [5, 5, 5], np.nan, None),
    ], ignore_index=True)
    frame["third"] = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]
    table = first_alarm_table(
        frame, {"m": frame["m"], "n2": frame["n2"], "third": frame["third"]},
        {"m": 0.25, "n2": 11.0, "third": 0.5})
    row = table.set_index("stay_id")
    assert row.loc[1, "fa_m"] == 7.0 and row.loc[1, "fa_n2"] == 7.0
    assert row.loc[1, "fa_third"] == 6.0 and row.loc[2, "fa_third"] == 8.0
    assert math.isnan(row.loc[2, "fa_n2"])
    # the two-scorer summary is the same table under its historical names
    summary = stay_summary(frame, frame["m"], frame["n2"], 0.25, 11.0)
    renamed = table.rename(columns={"fa_m": "fa_model", "fa_n2": "fa_news2"})
    pd.testing.assert_frame_equal(summary, renamed[summary.columns])


def _h2h_table(ids):
    """Nine event stays + two quiet ones with a designed catch geometry."""
    rows = [
        # (h0, event_h, fa_sent, fa_opp)
        (6, 30.0, 10, 14),        # joint, flagship 4 h first
        (6, 30.0, 12, 12),        # joint, same hour
        (6, 30.0, 20, 15),        # joint, opponent 5 h first
        (6, 30.0, 10, np.nan),    # only the flagship catches
        (6, 30.0, np.nan, 10),    # only the opponent catches
        (6, 30.0, np.nan, np.nan),  # missed by all
        (6, 30.0, 6, 6),          # caught at h0 -> excluded from midstay
        (6, 30.0, 27, 27),        # 3 h warning -> excluded from lead6
        (6, 30.0, 32, np.nan),    # alarm after the event: not a catch
        (6, np.nan, np.nan, 10),  # quiet stay, opponent false alarm
        (6, np.nan, np.nan, np.nan),  # clean quiet stay
    ]
    return pd.DataFrame(
        [(sid, h0, 40, 35, e, "vent" if not math.isnan(e) else None, fs, fo)
         for sid, (h0, e, fs, fo) in zip(ids, rows)],
        columns=("stay_id", "h0", "h1", "n_hours", "event_h", "event_type",
                 "fa_sent", "fa_opp"))


def test_h2h_summary_counts_catches_races_exclusives_and_quiet_cost():
    h2h = h2h_summary(_h2h_table(range(1, 12)), "sent", ["opp"])
    assert h2h["n_event_stays"] == 9 and h2h["n_quiet_stays"] == 2
    assert h2h["n_missed_by_all"] == 2               # never caught + too late
    assert h2h["funnel"]["any"]["sent"] == {"n": 6, "pct": 6 / 9}
    assert h2h["funnel"]["any"]["opp"] == {"n": 6, "pct": 6 / 9}
    assert h2h["funnel"]["lead6"]["sent"]["n"] == 5   # the 3 h catch drops out
    assert h2h["funnel"]["midstay"]["sent"]["n"] == 5  # the h0 catch drops out
    race, = h2h["race"]
    assert race["opponent"] == "opp" and race["n_joint"] == 5
    assert (race["flagship_first"], race["tie"], race["opponent_first"]) == (1, 3, 1)
    assert race["flagship_first_pct"] == race["opponent_first_pct"] == 1 / 5
    assert (race["only_flagship"], race["only_opponent"]) == (1, 1)
    assert race["median_gap_h"] == -0.5              # signed, over non-zero gaps
    assert h2h["cost"]["quiet_fp_rate"] == {"sent": 0.0, "opp": 0.5}
    assert h2h["cost"]["stay_precision"]["sent"] == 6 / 7  # the late alarm hurts
    assert h2h["cost"]["stay_precision"]["opp"] == 6 / 7   # the quiet FP hurts


def test_budget_sweep_row_reports_catch_and_quiet_fp_per_model():
    row = budget_sweep_row(_h2h_table(range(1, 12)), ["sent", "opp"])
    assert row["sent"] == {"catch": 6 / 9, "quiet_fp": 0.0}
    assert row["opp"] == {"catch": 6 / 9, "quiet_fp": 0.5}
    assert assert_aggregate_only(
        budget_sweep_row(_h2h_table(range(30000001, 30000012)), ["sent", "opp"]),
        "sweep")


def test_h2h_summary_output_passes_the_aggregate_gate():
    """Even fed real-range stay ids, no id can leak into the h2h block."""
    table = _h2h_table(range(30000001, 30000012))
    assert assert_aggregate_only(h2h_summary(table, "sent", ["opp"]), "h2h")


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

def test_stay_label_races_the_baselines_and_never_mentions_the_id_or_news2():
    stay = {"unit_group": "medical", "event_type": "vent", "event_h": 40.0,
            "first_alarm": {"model": 14, "news2": 20, "lead_h": 6.0},
            "comp_first_alarm": {"ecart": 28, "yaib": 30}}
    assert stay_label(stay) == \
        "Ventilation — Sentinel alarms 14 h before the baselines · medical ICU"
    behind = {**stay, "comp_first_alarm": {"ecart": 10, "yaib": None}}
    assert stay_label(behind).startswith(
        "Ventilation — the baselines alarm 4 h before Sentinel")
    missed = {**stay, "comp_first_alarm": {"ecart": None, "yaib": None}}
    assert stay_label(missed).startswith(
        "Ventilation — only Sentinel alarms — 26 h before the event")
    old_bundle = {"unit_group": "medical", "event_type": "vent", "event_h": 40.0,
                  "first_alarm": {"model": 14}}
    assert "Sentinel alarms 26 h before the event" in stay_label(old_bundle)
    assert stay_label({"unit_group": "neuro", "event_type": None}) == \
        "No event — no alarm from any model · neuro ICU"
    for s in (stay, behind, missed):
        assert "NEWS2" not in stay_label(s)


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
        "thresholds": {"thr_model": 0.28, "thr_news2": 11.0,
                       "thr_ecart": 0.84, "thr_yaib": 0.23},
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
            "comp_risk": {"ecart": list(arr), "yaib": list(arr)},
            "comp_alarm": {"ecart": [False] * n, "yaib": [False] * n},
            "comp_first_alarm": {"ecart": None, "yaib": None},
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


def test_validate_bundle_checks_comparator_alignment_and_consistency():
    short = _bundle(); short["stays"][0]["comp_risk"]["ecart"] = [0.1, 0.1]
    with pytest.raises(ValueError, match="comp_risk.ecart"):
        validate_bundle(short)
    keys = _bundle(); del keys["stays"][0]["comp_first_alarm"]["yaib"]
    with pytest.raises(ValueError, match="comp_first_alarm keys"):
        validate_bundle(keys)
    outside = _bundle()
    outside["stays"][0]["comp_alarm"]["ecart"] = [False, False, True]
    outside["stays"][0]["comp_first_alarm"]["ecart"] = 99
    with pytest.raises(ValueError, match="outside the stay"):
        validate_bundle(outside)
    disagree = _bundle()
    disagree["stays"][0]["comp_alarm"]["ecart"] = [False, True, False]
    with pytest.raises(ValueError, match="disagrees with comp_alarm"):
        validate_bundle(disagree)   # first_alarm still says None
    unthresholded = _bundle(); del unthresholded["thresholds"]["thr_ecart"]
    with pytest.raises(ValueError, match="thr_ecart"):
        validate_bundle(unthresholded)


# --- live replay -----------------------------------------------------------------

def test_truncate_stay_cuts_every_aligned_array_and_hides_the_future_event():
    stay = _bundle(3)["stays"][0]
    stay["event_h"], stay["event_type"] = 8.9, "vent"     # hours are 6, 7, 8
    cut = truncate_stay(stay, 2)
    assert cut["hours"] == [6, 7]
    for group in ("risk", "agent_risk", "attention", "alarm", "comp_risk",
                  "comp_alarm", "vitals_observed", "vitals_model_view",
                  "vitals_hrs_since"):
        assert all(len(arr) == 2 for arr in cut[group].values()), group
    assert len(cut["news2"]["total"]) == 2
    assert all(len(a) == 2 for a in cut["news2"]["subscores"].values())
    assert cut["event_h"] is None and cut["event_type"] is None  # 8.9 > 7 + 1
    assert cut["cards"] == stay["cards"] and cut["contexts"] == stay["contexts"]
    full = truncate_stay(stay, 3)
    assert full["event_h"] == 8.9                # within an hour of the window
    peek = truncate_stay(stay, 2, hide_event=False)
    assert peek["event_h"] == 8.9
    assert stay["hours"] == [6, 7, 8]            # the original is untouched


def test_alarm_announcements_are_sorted_and_skip_models_that_never_fire():
    stay = {"first_alarm": {"model": 23, "news2": 30, "lead_h": 7.0},
            "comp_first_alarm": {"ecart": 14, "yaib": None}}
    out = alarm_announcements(stay, COMPARATOR_LABELS)
    assert [(h, name) for h, name, _ in out] == [(14.0, "ecart"), (23.0, "sentinel")]
    assert "eCART" in out[0][2] and "hour 14" in out[0][2]
    assert not any(name == "yaib" for _, name, _ in out)
    assert alarm_announcements({"first_alarm": {}}, {}) == []


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
        assert_aggregate_only({"ids": [39999999]})
    with pytest.raises(ValueError, match="stay-id-like"):
        assert_aggregate_only({"note": "see 39999999 for details"})


def test_assert_no_identifiers_reads_the_rendered_page():
    assert assert_no_identifiers("<p>AUPRC 0.308 over 361229 hours</p>")
    with pytest.raises(ValueError):
        assert_no_identifiers("<td>stay_id</td>")
    with pytest.raises(ValueError):
        assert_no_identifiers("<td>39999999</td>")
