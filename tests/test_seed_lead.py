"""Tests for the seed-consistency helpers (no data, no torch)."""

import math

import numpy as np
import pandas as pd
import pytest

from seed_lead import (ensemble_scores, race_anatomy, seed_level_ci,
                       seeds_needed, summarize_tail, tail_diagnosis)


def test_race_anatomy_counts_and_discordant_share():
    r = race_anatomy([3, 1, 0, 0, -2, 5, -1])
    assert (r["n"], r["wins"], r["ties"], r["losses"]) == (7, 3, 2, 2)
    assert r["median"] == 0.0 and r["win_given_discordant"] == 0.6
    assert 0 < r["sign_p"] <= 1
    empty = race_anatomy([])
    assert empty["n"] == 0 and math.isnan(empty["median"]) and empty["sign_p"] == 1.0


def test_seed_level_ci_is_a_t_interval_over_seeds():
    mean, (lo, hi), sd = seed_level_ci([1.0, 2.0, 1.5, 0.0, 0.0])
    assert mean == pytest.approx(0.9)
    assert lo < 0 < hi                      # five seeds this spread: not clear of zero
    assert sd == pytest.approx(np.std([1, 2, 1.5, 0, 0], ddof=1))
    m1, (l1, h1), s1 = seed_level_ci([2.0])
    assert m1 == 2.0 and math.isnan(l1) and math.isnan(s1)
    assert seeds_needed(sd=2.87, target_half_width=1.0) > 10
    assert seeds_needed(sd=0.1, target_half_width=1.0) == 2


def test_ensemble_scores_mean_and_median_and_shape_guard():
    stack = np.array([[0.1, 0.9], [0.3, 0.5], [0.2, 0.1]])
    assert np.allclose(ensemble_scores(stack), [0.2, 0.5])
    assert np.allclose(ensemble_scores(stack, "median"), [0.2, 0.5])
    with pytest.raises(ValueError):
        ensemble_scores(np.zeros((0, 3)))
    with pytest.raises(ValueError):
        ensemble_scores(stack, "max")


def _frame():
    # stay 1: NEWS2 >= 11 from the first hour, model fires at hour 20 -> loss -14, standing alarm
    # stay 2: NEWS2 rises at hour 15, model fires at hour 3 -> win +12, model NOT on from hour 0
    # stay 3: model >= thr from the first hour, NEWS2 at hour 14 -> win +14, standing model alarm
    # stay 4: small diff, excluded from the tail
    rows = []
    for h in range(0, 21):
        rows.append((1, h, 12.0, 0.9 if h >= 20 else 0.1))
        rows.append((2, h, 12.0 if h >= 15 else 3.0, 0.9 if h >= 3 else 0.1))
        rows.append((3, h, 12.0 if h >= 14 else 3.0, 0.9))
        rows.append((4, h, 12.0 if h >= 5 else 3.0, 0.9 if h >= 4 else 0.1))
    return pd.DataFrame(rows, columns=["stay_id", "hour", "n2", "m"])


def test_tail_diagnosis_flags_standing_alarms_on_both_sides():
    f = _frame()
    diffs = pd.DataFrame({"stay_id": [1, 2, 3, 4], "diff": [-14.0, 12.0, 14.0, 1.0]})
    tail = tail_diagnosis(f, f["n2"], f["m"], 11.0, 0.5, diffs, cutoff=12)
    assert set(tail["stay_id"]) == {1, 2, 3}
    by = tail.set_index("stay_id")
    assert by.loc[1, "side"] == "loss" and by.loc[1, "earlier_from_first_hour"]
    assert by.loc[2, "side"] == "win" and not by.loc[2, "earlier_from_first_hour"]
    assert by.loc[3, "side"] == "win" and by.loc[3, "earlier_from_first_hour"]
    s = summarize_tail(tail)
    assert s["loss"] == {"n": 1, "from_first_hour": 1, "share_from_first_hour": 1.0}
    assert s["win"]["n"] == 2 and s["win"]["share_from_first_hour"] == 0.5
    assert math.isnan(summarize_tail(tail.iloc[0:0])["loss"]["share_from_first_hour"])
