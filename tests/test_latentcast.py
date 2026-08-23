"""Tests for LatentCast (forecast-then-alarm through the frozen fusion head).

The load-bearing test is the ORACLE-SUBSTITUTION one: with a perfect
forecaster (forecast bank := true bank shifted left by k), the frozen causal
head's output at ``t`` must EQUAL the parent orchestrator's risk at ``t+k``.
That single equality is simultaneously the semantics proof (whole-sequence
substitution computes "risk k hours ahead as estimable at t") and the
leakage/plumbing regression test.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from agents.base import KEY_COLUMNS
from explain.faithfulness import with_bank
from latentcast import (
    LatentForecaster,
    bank_sequences,
    build_forecast_bank,
    collate_forecast,
    exceedance_score,
    or_score,
    shifted_label,
    within_horizon_max,
)
from orchestrator.fusion import AgentBank, Orchestrator

LABEL = "label_24h"
AGENTS = ("a1", "a2")
DIMS = {"a1": 4, "a2": 3}


def _bank(stay_hours, seed=0):
    rng = np.random.default_rng(seed)
    rows = [(s, h) for s, hours in stay_hours.items() for h in hours]
    keys = pd.DataFrame(rows, columns=list(KEY_COLUMNS))
    matrix = rng.normal(
        size=(len(keys), sum(DIMS.values()) + len(AGENTS))).astype(np.float32)
    return AgentBank(keys, matrix, DIMS, AGENTS)


def _cohort(stay_hours, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"stay_id": s, "patient_id": 100 + s, "hour": h,
             LABEL: int(rng.random() < 0.3)}
            for s, hours in stay_hours.items() for h in hours]
    return pd.DataFrame(rows)


def _fitted_orchestrator(hours=20, seed=0):
    stay_hours = {i: list(range(6, 6 + hours)) for i in range(1, 7)}
    bank = _bank(stay_hours, seed=seed)
    orch = Orchestrator(bank=bank, d_model=8, dilations=(1, 2), max_epochs=1,
                        batch_size=4, verbose=False)
    orch.fit(_cohort(stay_hours, seed=seed), _cohort(stay_hours, seed=seed + 1))
    return orch, bank, stay_hours


# --- Forecaster causality ----------------------------------------------------

def test_forecaster_is_causal():
    """Rewriting future inputs must not move predictions at earlier hours."""
    torch.manual_seed(0)
    model = LatentForecaster(n_features=9, hidden=8, horizons=(2, 4))
    x = torch.randn(1, 30, 9)
    base = {k: v.detach().clone() for k, v in model(x).items()}
    x2 = x.clone()
    x2[:, 15:, :] = 99.0
    later = model(x2)
    for k in (2, 4):
        torch.testing.assert_close(later[k][:, :15], base[k][:, :15])
        assert not torch.allclose(later[k][:, 15:], base[k][:, 15:])


def test_collate_targets_never_cross_stays():
    """mask_k is zero exactly where t+k leaves the stay; targets match X[t+k]."""
    seqs = [{"stay_id": 1, "hours": np.arange(6, 16),
             "X": np.arange(10, dtype=np.float32).reshape(10, 1)},
            {"stay_id": 2, "hours": np.arange(6, 10),
             "X": (100 + np.arange(4, dtype=np.float32)).reshape(4, 1)}]
    x, targets, masks = collate_forecast(seqs, [0, 1], (2,), "cpu")
    m = masks[2].numpy()
    assert m[0].tolist() == [1] * 8 + [0] * 2
    assert m[1].tolist() == [1] * 2 + [0] * 8
    t = targets[2].numpy()
    np.testing.assert_allclose(t[0, :8, 0], np.arange(2, 10))
    np.testing.assert_allclose(t[1, :2, 0], [102, 103])
    assert (t[1, 2:] == 0).all()          # padding, masked out


# --- The oracle-substitution semantics proof ---------------------------------

def test_perfect_forecaster_reproduces_parent_risk_at_t_plus_k():
    """Forecast bank := true bank shifted left by k  =>  frozen-head
    risk_hat(t) == parent raw risk(t+k), for every t where BOTH sequences
    give the TCN a full receptive field.

    The shifted sequence is missing the parent's first k rows of context, so
    positions inside the TCN warm-up window (RF = 1 + 2*(kernel-1)*sum(dils)
    = 13 h for this test net; 29 h for the shipped one) legitimately differ --
    they see truncated history, which is still strictly-past information.
    """
    k = 3
    rf = 13
    orch, bank, stay_hours = _fitted_orchestrator()
    df = _cohort(stay_hours)

    # Perfect forecast: row at (s, t) holds the TRUE row at t+k; tail rows
    # (no future inside the stay) hold garbage -- causality means they can
    # only affect positions >= their own, which we exclude from comparison.
    matrix = bank.matrix.copy()
    stay = bank.keys["stay_id"].to_numpy()
    for s in np.unique(stay):
        rows = np.flatnonzero(stay == s)
        matrix[rows[:-k]] = bank.matrix[rows[k:]]
        matrix[rows[-k:]] = 7.7
    fbank = AgentBank(bank.keys, matrix, bank.latent_dims, bank.agent_names)

    parent = orch.raw_risk(df)                       # sorted (stay, hour)
    hat = with_bank(orch, fbank).raw_risk(df)
    ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    st = ordered["stay_id"].to_numpy()
    compared = 0
    for s in np.unique(st):
        rows = np.flatnonzero(st == s)
        full = np.arange(rf - 1, len(rows) - k)      # full RF in both
        np.testing.assert_allclose(hat[rows[full]], parent[rows[full + k]],
                                   atol=1e-6)
        compared += len(full)
    assert compared >= 20                            # the test actually bites


def test_forecast_bank_keys_match_the_real_bank():
    torch.manual_seed(0)
    orch, bank, stay_hours = _fitted_orchestrator()
    df = _cohort(stay_hours)
    model = LatentForecaster(bank.matrix.shape[1], hidden=8, horizons=(2,))
    stats = (np.zeros(bank.matrix.shape[1], dtype=np.float32),
             np.ones(bank.matrix.shape[1], dtype=np.float32))
    fbank = build_forecast_bank(bank, df, model, stats, 2, "cpu")
    assert fbank.keys.equals(bank.keys)
    assert fbank.matrix.shape == bank.matrix.shape
    # risk columns clamped to [0, 1]
    n_risk = len(bank.agent_names)
    assert fbank.matrix[:, -n_risk:].min() >= 0.0
    assert fbank.matrix[:, -n_risk:].max() <= 1.0
    # frozen head consumes it end to end
    hat = with_bank(orch, fbank).raw_risk(df)
    assert hat.shape == (len(df),)


# --- Exceedance-ensemble scoring ---------------------------------------------

def test_exceedance_votes_lattice_and_tiebreak_below_step():
    members = [np.array([0.9, 0.1, 0.5]), np.array([0.8, 0.2, 0.4]),
               np.array([0.1, 0.1, 0.6])]
    score, votes, mean = exceedance_score(members, tau=0.45)
    np.testing.assert_allclose(votes, [2 / 3, 0.0, 2 / 3])
    assert (np.abs(score - votes) <= 1e-6).all()      # tiebreak << 1/E step
    assert score[0] > score[2]                        # mean breaks the tie
    np.testing.assert_allclose(mean, [0.6, 0.4 / 3, 0.5])


def test_or_score_never_ranks_a_parent_alarm_below_threshold():
    """Every parent crossing gets the lattice maximum (1.0 + tie), so no
    exceedance threshold inside the vote range can drop a parent alarm --
    the OR rung provably never alarms later than the parent."""
    votes = np.array([0.0, 0.4, 1.0, 0.2])
    mean_within = np.array([0.1, 0.5, 0.9, 0.3])
    parent = np.array([0.9, 0.1, 0.1, 0.95])          # rows 0, 3 cross
    s = or_score(votes, mean_within, parent, tau=0.5)
    crossed = parent >= 0.5
    assert (s[crossed] >= 1.0).all()
    assert (s[~crossed] < 1.0 + 1e-5).all()
    assert s[1] == pytest.approx(0.4 + 1e-6 * 0.5)


def test_within_horizon_max_is_elementwise_over_selected_ks():
    hats = {2: np.array([0.1, 0.9]), 4: np.array([0.5, 0.2]),
            8: np.array([9.0, 9.0])}
    np.testing.assert_allclose(within_horizon_max(hats, [2, 4]), [0.5, 0.9])


# --- The shifted-label control guard ----------------------------------------

def test_shifted_label_at_zero_matches_the_shipped_convention():
    frame = pd.DataFrame({
        "hour": [6, 7, 8, 9, 6, 7],
        "event_h": [31.0, 31.0, 31.0, 31.0, np.nan, np.nan],
    })
    frame[LABEL] = ((frame["event_h"] > frame["hour"])
                    & (frame["event_h"] <= frame["hour"] + 24)).fillna(False
                    ).astype(float)
    np.testing.assert_array_equal(shifted_label(frame, 0),
                                  frame[LABEL].to_numpy())
    wider = shifted_label(frame, 6)
    assert wider.sum() >= frame[LABEL].sum()
    assert wider[0] == 1.0 and frame[LABEL][0] == 0.0   # hour 6, event 31
