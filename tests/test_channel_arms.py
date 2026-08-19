"""Tests for the controlled channel experiment's pure plumbing.

The arms must differ ONLY in the channel, so the plumbing that builds each
arm's inputs is what gets pinned here: sentence-code round-trips (every row
gets exactly its own sentence's embedding), the risk-only bank really carries
nothing but risks, hard votes use per-agent thresholds, and the achieved-rate
analysis reproduces the discrete-overshoot behaviour the analysis reports.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from channel_arms import (
    TEXT_COLUMNS,
    achieved_rates,
    build_risk_bank,
    hard_vote_scores,
    unique_sentences,
)
from orchestrator.fusion import AgentBank


def test_unique_sentences_round_trip():
    summaries = {
        "train": pd.DataFrame({
            "vitals_text": ["a", "b", "a"], "news2_text": ["x", "x", "y"],
            "labs_text": ["p", "q", "r"]}),
        "test": pd.DataFrame({
            "vitals_text": ["b", "c"], "news2_text": ["y", "x"],
            "labs_text": ["p", "p"]}),
    }
    uniques, codes = unique_sentences(summaries)
    for col in TEXT_COLUMNS:
        for split, frame in summaries.items():
            rebuilt = uniques[col][codes[col][split]]
            np.testing.assert_array_equal(rebuilt, frame[col].to_numpy())
    assert set(uniques["vitals_text"]) == {"a", "b", "c"}


def _tiny_bank():
    keys = pd.DataFrame({"stay_id": [1, 1, 2], "hour": [6, 7, 6]})
    # layout: [vit(2) | n2(2) | labs(2) | risks(3)]
    matrix = np.arange(3 * 9, dtype=np.float32).reshape(3, 9)
    matrix[:, 6:] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    return AgentBank(keys, matrix, {"vitals_tcn": 2, "news2": 2, "labs": 2},
                     ["vitals_tcn", "news2", "labs"])


def test_risk_bank_carries_only_risks():
    bank = _tiny_bank()
    splits = {"train": pd.DataFrame({"stay_id": [1, 1], "hour": [6, 7],
                                     "patient_id": [10, 10]}),
              "test": pd.DataFrame({"stay_id": [2], "hour": [6],
                                    "patient_id": [20]})}
    rb = build_risk_bank(bank, splits)
    assert rb.latent_dims == {"vitals_tcn": 1, "news2": 1, "labs": 1}
    assert rb.matrix.shape == (3, 6)                 # 3 x (1+1+1 latents + 3 risks)
    # each 'latent' is exactly the agent's own risk, duplicated in the block
    np.testing.assert_allclose(rb.matrix[:, :3], rb.matrix[:, 3:])
    np.testing.assert_allclose(rb.matrix[0, :3], [0.1, 0.2, 0.3])


def test_hard_vote_uses_per_agent_thresholds():
    rng = np.random.default_rng(0)
    risks_val = np.column_stack([rng.random(1000),
                                 rng.random(1000) * 0.1,     # low-scale agent
                                 rng.random(1000)])
    votes, thresholds = hard_vote_scores(risks_val, risks_val, budget=0.10)
    # every agent votes on ~10% of rows despite wildly different scales
    per_agent = (risks_val >= thresholds).mean(axis=0)
    assert np.allclose(per_agent, 0.10, atol=0.01)
    assert set(np.unique(votes)) <= {0.0, 1.0, 2.0, 3.0}


def test_achieved_rates_overshoot_on_discrete_scores():
    continuous = np.linspace(0, 1, 10_000)
    votes = np.repeat([0.0, 1.0, 2.0, 3.0], [9000, 700, 250, 50])
    fine = achieved_rates(continuous)
    coarse = achieved_rates(votes)
    for b in (0.01, 0.02, 0.05):
        assert fine[b] == pytest.approx(b, abs=0.001)   # continuous: on target
    # the pipeline's quantile rule lands on a vote step, so a 1% or 2% budget
    # both fire on 3% of hours (>=2 votes) and 5% fires on 10% (>=1 vote):
    assert coarse[0.01] == pytest.approx(0.03, abs=1e-6)
    assert coarse[0.02] == pytest.approx(0.03, abs=1e-6)
    assert coarse[0.05] == pytest.approx(0.10, abs=1e-6)
