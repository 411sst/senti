"""Tests for the ECG agent (agent #4).

Two families here. The first is the usual leakage discipline. The second is
specific to this modality: ECGs are *ordered*, not monitored, so their presence
correlates with acuity (5.17% event rate on covered hours vs 4.74% overall). A
model could score respectably by learning "an ECG was ordered -> this patient is
sicker" while knowing nothing about the waveform. These tests make that
shortcut structurally impossible to hide.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from agents.base import Agent
from agents.ecg_agent import (
    STALENESS_HOURS,
    EcgAgent,
    EcgHead,
    assemble_ecg_features,
    presence_only_auprc,
)
from agents.ecg_encoder import EMBED_DIM

LABEL = "label_24h"


def _cohort(stay_hours, label_fn=None):
    rows = []
    for stay, hours in stay_hours.items():
        for h in hours:
            rows.append({"stay_id": stay, "patient_id": 500 + stay, "hour": h,
                         LABEL: 0 if label_fn is None else label_fn(stay, h)})
    return pd.DataFrame(rows)


def _index(records):
    """records: [(stay_id, ecg_hour, study_id)]"""
    return pd.DataFrame(records, columns=["stay_id", "ecg_hour", "study_id"])


def _embeddings(study_ids, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(len(study_ids), EMBED_DIM)).astype(np.float32), \
        np.array(study_ids, dtype=np.int64)


# --- Leakage: strictly before t, bounded staleness -------------------------

def test_ecg_recorded_at_hour_t_is_not_used():
    """An ECG timestamped exactly at t must be invisible at t."""
    cohort = _cohort({1: [6, 7]})
    index = _index([(1, 7.0, 100)])
    emb, ids = _embeddings([100])
    X, has, age, keys = assemble_ecg_features(cohort, index, emb, ids)
    by_hour = dict(zip(keys["hour"], has))
    assert by_hour[6] == 0.0          # before the ECG exists
    assert by_hour[7] == 0.0          # its OWN hour -> excluded
    cohort2 = _cohort({1: [6, 7, 8]})
    _, has2, _, keys2 = assemble_ecg_features(cohort2, index, emb, ids)
    assert dict(zip(keys2["hour"], has2))[8] == 1.0    # visible the hour after


def test_stale_ecg_is_dropped_at_the_bound():
    ecg_hour = 9
    at_bound = ecg_hour + STALENESS_HOURS          # age exactly 72h
    past_bound = at_bound + 1                      # age 73h
    cohort = _cohort({1: [10, at_bound, past_bound]})
    index = _index([(1, float(ecg_hour), 100)])
    emb, ids = _embeddings([100])
    _, has, age, keys = assemble_ecg_features(cohort, index, emb, ids)
    by_hour = dict(zip(keys["hour"], has))
    age_by_hour = dict(zip(keys["hour"], age))
    assert by_hour[10] == 1.0                      # 1h old
    assert age_by_hour[at_bound] == pytest.approx(STALENESS_HOURS)
    assert by_hour[at_bound] == 1.0                # exactly at the bound: kept
    assert by_hour[past_bound] == 0.0              # one hour past: dropped


def test_most_recent_prior_ecg_wins():
    cohort = _cohort({1: [20]})
    index = _index([(1, 5.0, 100), (1, 18.0, 200), (1, 25.0, 300)])
    emb, ids = _embeddings([100, 200, 300])
    X, has, age, _ = assemble_ecg_features(cohort, index, emb, ids)
    assert has[0] == 1.0
    assert age[0] == pytest.approx(2.0)                      # 20 - 18
    np.testing.assert_allclose(X[0], emb[1])                 # study 200, not 100/300


def test_ecgs_do_not_cross_stays():
    cohort = _cohort({1: [10], 2: [10]})
    index = _index([(1, 5.0, 100)])
    emb, ids = _embeddings([100])
    _, has, _, keys = assemble_ecg_features(cohort, index, emb, ids)
    m = dict(zip(keys["stay_id"], has))
    assert m[1] == 1.0 and m[2] == 0.0


def test_missing_embedding_is_treated_as_no_ecg():
    """A study in the index but not downloaded must not fabricate a vector."""
    cohort = _cohort({1: [10]})
    index = _index([(1, 5.0, 999)])          # 999 was never encoded
    emb, ids = _embeddings([100])
    X, has, _, _ = assemble_ecg_features(cohort, index, emb, ids)
    assert has[0] == 0.0
    assert np.allclose(X[0], 0.0)


# --- Presence-leakage: the shortcut must be impossible to hide -------------

def test_all_no_ecg_rows_receive_an_identical_input_and_identical_risk():
    """No-ECG hours carry no varying input, so the head MUST emit a constant.

    This is what makes presence-separation auditable: the agent cannot rank
    within the no-ECG group, so any full-cohort AUPRC decomposes cleanly into
    "presence separation" plus "signal among covered hours".
    """
    torch.manual_seed(0)
    head = EcgHead(hidden=8, latent=4).eval()
    n = 6
    embedding = torch.randn(n, EMBED_DIM)      # deliberately DIFFERENT per row
    has = torch.zeros(n)                       # ...but none of them has an ECG
    age = torch.rand(n) * 50                   # ...and stale ages differ too
    with torch.no_grad():
        logits, latents = head(embedding, has, age)
    assert torch.allclose(logits, logits[0].expand(n), atol=1e-6), \
        "no-ECG rows produced different risks -- some input is leaking through"
    assert torch.allclose(latents, latents[0].expand_as(latents), atol=1e-6)


def test_no_ecg_uses_the_learned_vector_not_a_zero_vector():
    torch.manual_seed(1)
    head = EcgHead(hidden=8, latent=4).eval()
    with torch.no_grad():
        zeros = torch.zeros(1, EMBED_DIM)
        no_ecg, _ = head(zeros, torch.zeros(1), torch.zeros(1))
        as_real, _ = head(zeros, torch.ones(1), torch.zeros(1))
    # A zero embedding presented as a REAL ECG must be scored differently from
    # the no-ECG case, or the learned vector is doing nothing.
    assert not torch.allclose(no_ecg, as_real, atol=1e-5)


def test_covered_rows_still_vary_with_the_waveform():
    """Sanity in the other direction: real ECGs must produce differing risks."""
    torch.manual_seed(2)
    head = EcgHead(hidden=8, latent=4).eval()
    embedding = torch.randn(5, EMBED_DIM)
    with torch.no_grad():
        logits, _ = head(embedding, torch.ones(5), torch.zeros(5))
    assert logits.std() > 1e-4


def test_presence_only_auprc_recovers_the_base_rate_gap():
    """The floor the agent must beat is computable and behaves sensibly."""
    y = np.r_[np.ones(60), np.zeros(940), np.ones(40), np.zeros(960)]
    has = np.r_[np.ones(1000), np.zeros(1000)]        # covered hours are enriched
    floor = presence_only_auprc(y, has)
    assert floor > y.mean()            # presence alone is informative...
    assert floor < 0.20                # ...but only weakly
    # an uninformative indicator gives roughly prevalence
    assert presence_only_auprc(y, np.ones_like(y)) == pytest.approx(y.mean(), abs=1e-6)


# --- Agent contract --------------------------------------------------------

def _fitted(seed=0):
    hours = list(range(6, 26))
    stay_hours = {i: hours for i in range(1, 9)}
    # every other stay gets an ECG, so both branches are exercised
    index = _index([(i, 5.0, 100 + i) for i in range(1, 9) if i % 2 == 0])
    emb, ids = _embeddings([100 + i for i in range(1, 9) if i % 2 == 0], seed=seed)

    def lab(stay, h):
        return int(stay % 2 == 0 and h >= 20)

    train = _cohort(stay_hours, lab)
    val = _cohort({i + 10: hours for i in range(1, 9)}, lambda s, h: int(s % 2 == 0 and h >= 20))
    val_index = _index([(i + 10, 5.0, 200 + i) for i in range(1, 9) if i % 2 == 0])
    val_emb, val_ids = _embeddings([200 + i for i in range(1, 9) if i % 2 == 0], seed=seed + 1)

    agent = EcgAgent(hidden=8, latent=4, max_epochs=2, batch_size=64, verbose=False)
    agent._cache = (np.vstack([emb, val_emb]), np.r_[ids, val_ids],
                    pd.concat([index, val_index], ignore_index=True))
    return agent.fit(train, val), train, val


def test_agent_emits_latent_and_calibrated_risk():
    agent, _, val = _fitted()
    out = agent.emit(val)
    assert out.latent.shape == (len(val), 4)
    assert ((out.risk >= 0) & (out.risk <= 1)).all()
    expected = val.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    assert (out.keys["stay_id"].to_numpy() == expected["stay_id"].to_numpy()).all()


def test_emit_before_fit_raises():
    agent = EcgAgent(verbose=False)
    agent._cache = (np.zeros((1, EMBED_DIM), np.float32), np.array([1]),
                    _index([(1, 1.0, 1)]))
    with pytest.raises(RuntimeError):
        agent.emit(_cohort({1: [6]}))


def test_satisfies_the_agent_interface():
    assert issubclass(EcgAgent, Agent)
    assert EcgAgent(verbose=False).name == "ecg"
