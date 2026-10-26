"""Tests for the NEWS2 tool-agent (agent #2).

The agent must be a faithful wrapper: its raw score has to equal the
deterministic scorer exactly, and its latent has to be the seven sub-scores
that sum to it. It also has to obey the same base-class calibration contract as
the learned agents.
"""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from agents.base import KEY_COLUMNS, Agent
from agents.news2_agent import NEWS2_MAX, News2Agent
from mimic_cohort import NEWS2_COMPONENTS, compute_news2
from news2 import full_news2_score

LABEL = "label_24h"


def _frame(rows):
    """A cohort-shaped frame; missing channels default to NaN."""
    df = pd.DataFrame(rows)
    for c in ["heart_rate", "resp_rate", "spo2", "sbp", "temperature",
              "gcs", "gcs_unable", "is_alert", "on_oxygen"]:
        if c not in df:
            df[c] = np.nan
    if LABEL not in df:
        df[LABEL] = 0
    return df


def _cohort(n_stays=6, n_hours=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stays):
        sick = s % 2 == 0
        for h in range(n_hours):
            rows.append({
                "stay_id": 100 + s, "patient_id": 50 + s, "hour": h,
                "heart_rate": (125 if sick else 70) + rng.normal(0, 2),
                "resp_rate": (23 if sick else 16) + rng.normal(0, 1),
                "spo2": (92 if sick else 98) + rng.normal(0, 1),
                "sbp": (95 if sick else 120) + rng.normal(0, 3),
                "temperature": 37.0 + rng.normal(0, 0.2),
                "gcs": 15.0, "gcs_unable": 0.0,
                "is_alert": 1.0, "on_oxygen": bool(sick),
                LABEL: int(sick and h >= n_hours - 3),
            })
    return _frame(rows)


# --- Faithfulness to the deterministic scorer ------------------------------

def test_raw_score_equals_the_scalar_news2_scorer():
    """The agent's raw score must equal news2.full_news2_score exactly."""
    df = _frame([{
        "stay_id": 1, "hour": 6,
        "resp_rate": 22.0, "spo2": 93.0, "temperature": 39.5, "sbp": 95.0,
        "heart_rate": 130.0, "is_alert": 0.0, "on_oxygen": True,
    }])
    agent = News2Agent()
    _, _, raw = agent._emit_raw(df)

    expected = full_news2_score(
        respiratory_rate=22.0, spo2=93.0, temperature=39.5, systolic_bp=95.0,
        heart_rate=130.0, gcs=9, gcs_verbal=2, gcs_motor=4, gcs_eyes=3,
        gcs_unable=0, o2_device=1,
    )
    assert raw[0] == expected == 15


def test_latent_is_the_seven_subscores_and_sums_to_the_total():
    df = _cohort()
    agent = News2Agent()
    keys, latent, raw = agent._emit_raw(df)

    assert latent.shape == (len(df), 7)
    assert agent.latent_dim == 7
    assert list(agent.component_names) == NEWS2_COMPONENTS
    np.testing.assert_allclose(latent.sum(axis=1), raw, atol=1e-6)


def test_subscores_are_in_their_documented_ranges():
    latent = News2Agent()._emit_raw(_cohort(seed=3))[1]
    for i, name in enumerate(NEWS2_COMPONENTS):
        col = latent[:, i]
        assert col.min() >= 0
        if name == "oxygen":
            assert set(np.unique(col)) <= {0.0, 2.0}
        elif name == "consciousness":
            assert set(np.unique(col)) <= {0.0, 3.0}
        else:
            assert col.max() <= 3


def test_agrees_with_compute_news2_on_the_same_frame():
    df = _cohort(seed=4)
    ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    total, _, _ = compute_news2(ordered)
    _, _, raw = News2Agent()._emit_raw(df)
    np.testing.assert_allclose(raw, total, atol=1e-6)


# --- Agent contract --------------------------------------------------------

def test_fit_model_is_a_no_op_but_calibration_still_happens():
    train, val = _cohort(seed=5), _cohort(seed=6)
    agent = News2Agent()
    assert agent.calibrator is None
    agent.fit(train, val)
    assert agent.calibrator is not None          # calibrated despite no training
    assert agent.calibration_summary["split_rows"] == len(val)


def test_emitted_rows_are_sorted_by_key():
    train, val = _cohort(seed=7), _cohort(seed=8)
    shuffled = val.sample(frac=1, random_state=0)
    agent = News2Agent().fit(train, val)
    out = agent.emit(shuffled)
    expected = val.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    assert (out.keys["stay_id"].to_numpy() == expected["stay_id"].to_numpy()).all()
    assert (out.keys["hour"].to_numpy() == expected["hour"].to_numpy()).all()


def test_emitted_risk_is_a_probability_unlike_the_raw_score():
    train, val = _cohort(seed=9), _cohort(seed=10)
    agent = News2Agent().fit(train, val)
    out = agent.emit(val)
    assert ((out.risk >= 0) & (out.risk <= 1)).all()
    assert agent.raw_risk(val).max() > 1.0      # raw is 0-20, not a probability
    assert (agent.raw_probability(val) <= 1.0).all()


def test_calibration_is_fit_only_on_val(monkeypatch):
    """Same guarantee as the learned agents: isotonic sees val labels only."""
    captured = {}
    original_fit = IsotonicRegression.fit

    def spy(self, X, y, **kwargs):
        captured["y"] = np.asarray(y).copy()
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(IsotonicRegression, "fit", spy)
    train = _cohort(n_stays=8, n_hours=10, seed=11)
    val = _cohort(n_stays=4, n_hours=6, seed=12)
    assert len(train) != len(val)

    News2Agent().fit(train, val)
    val_sorted = val.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    assert len(captured["y"]) == len(val)
    assert np.array_equal(captured["y"], val_sorted[LABEL].to_numpy(dtype=float))


def test_calibration_is_monotone_so_ranking_is_preserved():
    train, val = _cohort(seed=13), _cohort(seed=14)
    agent = News2Agent().fit(train, val)
    raw, cal = agent.raw_risk(val), agent.emit(val).risk
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(cal[order]) >= -1e-6)


def test_satisfies_the_agent_interface():
    assert issubclass(News2Agent, Agent)
    agent = News2Agent()
    assert agent.name == "news2"
    with pytest.raises(RuntimeError):
        agent.calibrate(np.array([5.0]))      # not calibrated yet
