"""Tests for the alarm-budget helpers (src/alarm_budget.py; no data)."""

import math

import numpy as np
import pandas as pd
import pytest

from alarm_budget import (
    assert_aggregate_only, budget_sweep_row, catch_at_quiet_fp, first_alarm_hour,
    first_alarm_table, h2h_summary, lead_stats, stay_maxima,
)

# Synthetic ids stay outside the 8-digit 3xxxxxxx range real stay ids occupy.
GATE_BOUND = 30_000_000


# --- first alarms -----------------------------------------------------------------

def _stay(stay_id, hours, model, news2, event_h, event_type):
    return pd.DataFrame({"stay_id": stay_id, "hour": hours, "event_h": event_h,
                         "event_type": event_type, "m": model, "n2": news2})


def test_first_alarm_hour_is_the_first_crossing_or_nan():
    assert first_alarm_hour([6, 7, 8, 9], [0.1, 0.3, 0.2, 0.4], 0.25) == 7.0
    assert math.isnan(first_alarm_hour([6, 7], [0.1, 0.2], 0.5))


def test_first_alarm_table_lead_matches_the_paired_race():
    """A stay's lead here must be the lead the paired race counts for it."""
    from fair_capability import paired_first_alarm_diffs

    frame = pd.concat([
        _stay(1, [6, 7, 8, 9, 10], [0.1, 0.3, 0.3, 0.3, 0.3], [5, 5, 5, 12, 12], 10.0, "vent"),
        _stay(2, [6, 7, 8], [0.1, 0.1, 0.1], [12, 12, 12], 8.0, "vaso"),      # model never
        _stay(3, [6, 7, 8], [0.3, 0.3, 0.3], [12, 12, 12], np.nan, None),     # no event
    ], ignore_index=True).sort_values(["stay_id", "hour"]).reset_index(drop=True)
    table = first_alarm_table(frame, {"model": frame["m"], "news2": frame["n2"]},
                              {"model": 0.25, "news2": 11.0}).set_index("stay_id")
    paper = paired_first_alarm_diffs(frame, frame["n2"], frame["m"], 11.0, 0.25)

    assert paper["stay_id"].tolist() == [1]
    assert table.loc[1, "fa_news2"] - table.loc[1, "fa_model"] == paper["diff"].iloc[0] == 2.0
    assert math.isnan(table.loc[2, "fa_model"])
    assert table.loc[3, "event_type"] is None
    assert table.loc[3, "n_hours"] == 3


def test_first_alarm_table_handles_many_models():
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
    assert row.loc[1, "ah_m"] == 3 and row.loc[2, "ah_third"] == 1


# --- one operating point --------------------------------------------------------

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
    df = pd.DataFrame(
        [(sid, h0, 40, 35, e, "vent" if not math.isnan(e) else None, fs, fo)
         for sid, (h0, e, fs, fo) in zip(ids, rows)],
        columns=("stay_id", "h0", "h1", "n_hours", "event_h", "event_type",
                 "fa_sent", "fa_opp"))
    for n in ("sent", "opp"):
        df[f"ah_{n}"] = np.where(df[f"fa_{n}"].notna(), 3, 0)
    df.loc[df.index[9], "ah_opp"] = 4          # the quiet false alarm lasts 4 h
    return df


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
    assert race["n_differ"] == 2 and race["earlier_pct"] == 0.5
    assert race["median_abs_gap_h"] == 4.5 and race["mean_gap_h"] == -0.5
    # warning depth (event_h - fa) over each model's catches
    assert h2h["lead"]["sent"]["n"] == 6
    assert h2h["lead"]["sent"]["median_h"] == 19.0   # leads 3,10,18,20,20,24
    assert h2h["lead"]["sent"]["mean_h"] == pytest.approx(95 / 6)
    assert h2h["lead"]["opp"]["median_h"] == 17.0    # leads 3,15,16,18,20,24
    assert h2h["lead"]["opp"]["mean_h"] == pytest.approx(16.0)
    assert h2h["cost"]["quiet_fp_rate"] == {"sent": 0.0, "opp": 0.5}
    assert h2h["cost"]["stay_precision"]["sent"] == 6 / 7  # the late alarm hurts
    assert h2h["cost"]["stay_precision"]["opp"] == 6 / 7   # the quiet FP hurts
    assert h2h["cost"]["quiet_alarm_h_mean"] == {"sent": 0.0, "opp": 4.0}
    assert h2h["cost"]["quiet_alarm_h_median"]["opp"] == 4.0


def test_lead_stats_measures_warning_depth_over_catches_only():
    table = _h2h_table(range(1, 12))
    direct = lead_stats(table, "sent")
    assert direct == h2h_summary(table, "sent", ["opp"])["lead"]["sent"]
    assert direct["n"] == 6                    # the h32 late alarm is not a catch
    assert direct["median_h"] == 19.0 and direct["mean_h"] == pytest.approx(95 / 6)


# --- across budgets ---------------------------------------------------------------

def test_budget_sweep_row_reports_catch_and_quiet_fp_per_model():
    row = budget_sweep_row(_h2h_table(range(1, 12)), ["sent", "opp"])
    assert row["sent"] == {"catch": 6 / 9, "quiet_fp": 0.0}
    assert row["opp"] == {"catch": 6 / 9, "quiet_fp": 0.5}


def _random_cohort(n_stays=60, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for sid in range(1, n_stays + 1):
        n = int(rng.integers(3, 12))
        event_h = float(rng.integers(0, 14)) if rng.random() < 0.4 else np.nan
        parts.append(pd.DataFrame({
            "stay_id": sid, "hour": np.arange(n, dtype=float), "event_h": event_h,
            "event_type": "vent" if not math.isnan(event_h) else None,
            "s": rng.random(n).round(2)}))       # rounding forces score ties
    return pd.concat(parts, ignore_index=True)


def test_stay_maxima_reproduce_the_sweep_rates_at_any_threshold():
    frame = _random_cohort()
    ids, is_ev, mx = stay_maxima(frame, frame["s"])
    assert ids.tolist() == sorted(frame["stay_id"].unique())
    assert is_ev.sum() == frame.groupby("stay_id")["event_h"].first().notna().sum()
    for thr in (0.05, 0.3, 0.5, 0.77, 0.99):
        row = budget_sweep_row(first_alarm_table(frame, {"s": frame["s"]}, {"s": thr}),
                               ["s"])["s"]
        assert row["catch"] == (mx[is_ev] >= thr).mean()
        assert row["quiet_fp"] == (mx[~is_ev] >= thr).mean()


def test_stay_maxima_ignore_hours_after_the_event():
    frame = _stay(1, [0, 1, 2, 3], [0.1, 0.2, 0.9, 0.9], [0, 0, 0, 0], 1.0, "vent")
    _, is_ev, mx = stay_maxima(frame, frame["m"])
    assert is_ev.tolist() == [True] and mx.tolist() == [0.2]
    early = _stay(2, [3, 4], [0.5, 0.9], [0, 0], 1.0, "vent")    # event before h0
    _, _, mx = stay_maxima(early, early["m"])
    assert mx.tolist() == [-np.inf]


def test_catch_at_quiet_fp_interpolates_the_stay_level_curve():
    ev = [0.9, 0.7, 0.5, 0.3]
    qt = [0.8, 0.6, 0.4, 0.2]
    # plateau tops: fp 0 -> .25, .25 -> .5, .5 -> .75, .75 -> 1, 1 -> 1
    got = catch_at_quiet_fp(ev, qt, [0.0, 0.125, 0.25, 0.5, 0.9, 1.0])
    np.testing.assert_allclose(got, [0.25, 0.375, 0.5, 0.75, 1.0, 1.0])


def test_catch_at_quiet_fp_counts_uncatchable_stays_and_ties():
    ev = [0.6, 0.6, -np.inf, -np.inf]       # half the events can never be caught
    qt = [0.6, 0.1]                         # a tie with the events at 0.6
    got = catch_at_quiet_fp(ev, qt, [0.25, 0.5, 1.0])
    np.testing.assert_allclose(got, [0.25, 0.5, 0.5])


# --- the DUA gate -------------------------------------------------------------------

def _leaves(tree):
    if isinstance(tree, dict):
        for v in tree.values():
            yield from _leaves(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            yield from _leaves(v)
    else:
        yield tree


def test_h2h_and_sweep_outputs_carry_no_input_ids():
    ids = list(range(90_000_001, 90_000_012))
    table = _h2h_table(ids)
    for block in (h2h_summary(table, "sent", ["opp"]),
                  budget_sweep_row(table, ["sent", "opp"])):
        assert assert_aggregate_only(block)
        assert not set(ids) & {v for v in _leaves(block) if isinstance(v, (int, float))}


def test_assert_aggregate_only_rejects_ids_keys_and_id_like_text():
    with pytest.raises(ValueError, match="patient-level key"):
        assert_aggregate_only({"rows": [{"stay_id": 1}]})
    with pytest.raises(ValueError, match="looks like a stay id"):
        assert_aggregate_only({"n": GATE_BOUND})
    with pytest.raises(ValueError, match="stay-id-like"):
        assert_aggregate_only({"note": f"see {GATE_BOUND}"})
    assert assert_aggregate_only({"git_commit": str(GATE_BOUND), "n": 1540, "ok": True})
