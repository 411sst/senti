"""Tests for the timing/calibration capability comparison.

The load-bearing guarantee: ``paired_first_alarm_diffs`` implements EXACTLY the
published pairing protocol (``baselines_mimic.paired_first_alarm``) -- same
warm-up masking, same jointly-caught restriction, same alarm-precedes-event
rule -- it just returns the per-stay differences so inference can be clustered
by patient. The equivalence test drives both implementations over the same
synthetic cohort and demands identical aggregates.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from baselines_mimic import STAB_HOURS, paired_first_alarm
from fair_capability import (
    frozen_thresholds,
    paired_first_alarm_diffs,
    summarize_paired,
)


def synthetic_cohort(seed=0, n_stays=40):
    """Stays with varied scores, events, and h6 boundary rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for stay in range(1, n_stays + 1):
        length = rng.integers(4, 30)
        event = float(rng.integers(STAB_HOURS + 2, STAB_HOURS + 40)) \
            if rng.random() < 0.6 else np.nan
        for h in range(STAB_HOURS, STAB_HOURS + length):
            rows.append({"stay_id": stay, "hour": h, "event_h": event,
                         "n2": float(rng.integers(0, 15)),
                         "m": float(rng.random())})
    return pd.DataFrame(rows)


def test_diffs_reproduce_the_published_aggregates():
    df = synthetic_cohort()
    thr_n2, thr_m = 8.0, 0.6
    published = paired_first_alarm(df, df["n2"], df["m"], thr_n2, thr_m)
    diffs = paired_first_alarm_diffs(df, df["n2"], df["m"], thr_n2, thr_m)

    assert len(diffs) == published["n_paired"]
    assert float(np.median(diffs["diff"])) == pytest.approx(published["median_diff"])
    assert int((diffs["diff"] > 0).sum()) == published["n_gbm_earlier"]
    assert int((diffs["diff"] < 0).sum()) == published["n_news2_earlier"]
    assert int((diffs["diff"] == 0).sum()) == published["n_tie"]

    summary = summarize_paired(diffs, {s: s for s in diffs["stay_id"]}, reps=50)
    assert summary["sign_p"] == pytest.approx(published["sign_p"])
    assert summary["win_rate"] == pytest.approx(published["frac_gbm_earlier"])


def test_boundary_hour_alarms_count_since_h6_fix():
    """h6 carries real LOCF history now, so a first alarm there is legitimate."""
    df = pd.DataFrame({
        "stay_id": [1] * 4, "hour": [6, 7, 8, 9], "event_h": [9.0] * 4,
        "n2": [0.0, 10.0, 0.0, 0.0],        # NEWS2 first alarms at h7
        "m": [1.0, 0.0, 0.0, 0.9],          # model first alarms at h6
    })
    diffs = paired_first_alarm_diffs(df, df["n2"], df["m"], 9.0, 0.8)
    assert len(diffs) == 1
    assert diffs["diff"].iloc[0] == pytest.approx(7.0 - 6.0)   # model 1h earlier


def test_frozen_thresholds_match_achieved_volume():
    news2 = np.array([0, 0, 0, 0, 0, 0, 3, 3, 9, 9], dtype=float)
    model = np.linspace(0, 1, 10)
    alertable = np.ones(10, dtype=bool)
    thr_n2, thr_m, matched = frozen_thresholds(news2, model, alertable, 0.10)
    # integer NEWS2 overshoots the 10% budget: threshold 9 catches exactly 20%
    assert (news2 >= thr_n2).mean() == pytest.approx(matched) == pytest.approx(0.2)
    # the model is then held to the ACHIEVED 20%, not the nominal 10%
    assert (model >= thr_m).mean() == pytest.approx(0.2)


def test_summarize_paired_clusters_by_patient():
    """CI must come from resampling patients, not stays: one dominant patient
    with many concordant stays should widen/shift the interval when clustered."""
    diffs = pd.DataFrame({
        "stay_id": list(range(10)),
        "diff": [5.0] * 5 + [-1.0] * 5,
    })
    # stays 0-4 all belong to patient A; 5-9 are separate patients
    mapping = {s: ("A" if s < 5 else f"p{s}") for s in range(10)}
    r = summarize_paired(diffs, mapping, reps=400, seed=1)
    assert r["n_pair"] == 10 and r["win_rate"] == pytest.approx(0.5)
    lo, hi = r["median_ci"]
    # when patient A is left out of a resample the median collapses to -1;
    # when A dominates it jumps to +5 -- the clustered CI must span both.
    assert lo <= -1.0 and hi >= 5.0

    r2 = summarize_paired(diffs, mapping, reps=400, seed=1)
    assert r2["median_ci"] == r["median_ci"]     # seeded determinism
