"""Tests for the trainable ResNet ECG agents (V2 ``ecg_resnet``, V3 ``+FiLM``).

Ported from ``test_ecg_agent.py``: the same leakage discipline and the same
presence-leakage guards, now over a *trainable* encoder rather than a frozen
one. V3 adds one obligation -- age conditioning must NOT leak into the no-ECG
floor, so no-ECG hours stay a single constant even when patients differ in age.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from agents.base import Agent
from agents.ecg_encoder import EXPECTED_SAMPLES, N_LEADS
from agents.ecg_resnet_agent import EcgResnetAgent, EcgResnetFilmAgent, EcgResnetNet
from agents.ecg_resnet_encoder import (
    STALENESS_HOURS,
    WaveformStore,
    assemble_ecg_study_ids,
)

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


def _waveforms(study_ids, seed=0):
    rng = np.random.default_rng(seed)
    return {int(s): rng.normal(size=(N_LEADS, EXPECTED_SAMPLES)).astype(np.float32)
            for s in study_ids}


# --- Leakage: strictly before t, bounded staleness (the shared merge) -------

def test_ecg_recorded_at_hour_t_is_not_used():
    cohort = _cohort({1: [6, 7, 8]})
    study, age, keys = assemble_ecg_study_ids(cohort, _index([(1, 7.0, 100)]))
    by_hour = dict(zip(keys["hour"], study))
    assert np.isnan(by_hour[6])       # before the ECG exists
    assert np.isnan(by_hour[7])       # its OWN hour -> excluded
    assert by_hour[8] == 100          # visible the hour after


def test_stale_ecg_is_dropped_at_the_bound():
    ecg_hour = 9
    at_bound = ecg_hour + STALENESS_HOURS
    past_bound = at_bound + 1
    cohort = _cohort({1: [10, at_bound, past_bound]})
    study, age, keys = assemble_ecg_study_ids(cohort, _index([(1, float(ecg_hour), 100)]))
    by_hour = dict(zip(keys["hour"], study))
    age_by_hour = dict(zip(keys["hour"], age))
    assert by_hour[10] == 100
    assert age_by_hour[at_bound] == pytest.approx(STALENESS_HOURS)
    assert by_hour[at_bound] == 100                 # exactly at the bound: kept
    assert np.isnan(by_hour[past_bound])            # one hour past: dropped


def test_most_recent_prior_ecg_wins():
    cohort = _cohort({1: [20]})
    study, age, keys = assemble_ecg_study_ids(
        cohort, _index([(1, 5.0, 100), (1, 18.0, 200), (1, 25.0, 300)]))
    assert study[0] == 200                           # study 200, not 100/300
    assert age[0] == pytest.approx(2.0)              # 20 - 18


def test_ecgs_do_not_cross_stays():
    cohort = _cohort({1: [10], 2: [10]})
    study, _, keys = assemble_ecg_study_ids(cohort, _index([(1, 5.0, 100)]))
    m = dict(zip(keys["stay_id"], study))
    assert m[1] == 100 and np.isnan(m[2])


def test_study_absent_from_store_is_treated_as_no_ecg():
    """A study in the index but not downloaded must not fabricate coverage."""
    cohort = _cohort({1: [10]})
    store = WaveformStore.from_arrays(_waveforms([100]))     # only 100 on disk
    agent = EcgResnetAgent(verbose=False)
    agent.store = store
    agent._index = _index([(1, 5.0, 999)])                  # row points at 999
    aln = agent._align(cohort)
    assert aln["has"][0] == 0.0 and aln["sidx"][0] == -1


# --- Presence-leakage: the shortcut must be impossible to hide -------------

def test_all_no_ecg_rows_receive_identical_risk_v2():
    torch.manual_seed(0)
    net = EcgResnetNet(hidden=8, latent=4, use_film=False).eval()
    n = 6
    feat = torch.randn(n, net.embed_dim)     # different per row...
    has = torch.zeros(n)                     # ...but none has an ECG
    ecg_age = torch.rand(n) * 50             # ...and stale ages differ
    with torch.no_grad():
        logits, latents = net.head_from_feat(feat, has, ecg_age)
    assert torch.allclose(logits, logits[0].expand(n), atol=1e-6)
    assert torch.allclose(latents, latents[0].expand_as(latents), atol=1e-6)


def test_all_no_ecg_rows_receive_identical_risk_v3_even_with_differing_age():
    """V3: age modulates only the ECG-present pathway; the no-ECG floor is one constant."""
    torch.manual_seed(0)
    net = EcgResnetNet(hidden=8, latent=4, use_film=True).eval()
    n = 6
    feat = torch.randn(n, net.embed_dim)
    has = torch.zeros(n)
    ecg_age = torch.rand(n) * 50
    patient_age = torch.tensor([20., 40., 55., 70., 85., 90.])   # deliberately different
    with torch.no_grad():
        logits, latents = net.head_from_feat(feat, has, ecg_age, patient_age)
    assert torch.allclose(logits, logits[0].expand(n), atol=1e-6), \
        "age leaked into the no-ECG floor via FiLM"
    assert torch.allclose(latents, latents[0].expand_as(latents), atol=1e-6)


def test_no_ecg_uses_the_learned_vector_not_a_zero_vector():
    torch.manual_seed(1)
    net = EcgResnetNet(hidden=8, latent=4).eval()
    with torch.no_grad():
        zeros = torch.zeros(1, net.embed_dim)
        no_ecg, _ = net.head_from_feat(zeros, torch.zeros(1), torch.zeros(1))
        as_real, _ = net.head_from_feat(zeros, torch.ones(1), torch.zeros(1))
    assert not torch.allclose(no_ecg, as_real, atol=1e-5)


def test_covered_rows_still_vary_with_the_waveform():
    torch.manual_seed(2)
    net = EcgResnetNet(hidden=8, latent=4).eval()
    feat = torch.randn(5, net.embed_dim)
    with torch.no_grad():
        logits, _ = net.head_from_feat(feat, torch.ones(5), torch.zeros(5))
    assert logits.std() > 1e-4


def test_v3_covered_rows_vary_with_age():
    """FiLM should let age move the covered-row risk (else V3 collapses to V2)."""
    torch.manual_seed(3)
    net = EcgResnetNet(hidden=8, latent=4, use_film=True).eval()
    # Push FiLM away from its identity init so the age effect is visible.
    with torch.no_grad():
        net.film_generator.gamma_head.weight.normal_(std=0.5)
        net.film_generator.beta_head.weight.normal_(std=0.5)
    feat = torch.randn(1, net.embed_dim).expand(4, -1)         # same waveform
    ages = torch.tensor([20., 45., 70., 90.])
    with torch.no_grad():
        logits, _ = net.head_from_feat(feat, torch.ones(4), torch.zeros(4), ages)
    assert logits.std() > 1e-4


# --- Agent contract --------------------------------------------------------

def _fitted(cls=EcgResnetAgent, seed=0):
    hours = list(range(6, 26))
    stay_hours = {i: hours for i in range(1, 9)}
    ecg_stays = [i for i in range(1, 9) if i % 2 == 0]
    index = _index([(i, 5.0, 100 + i) for i in ecg_stays])
    val_stays = {i + 10: hours for i in range(1, 9)}
    val_ecg = [i + 10 for i in range(1, 9) if i % 2 == 0]
    val_index = _index([(i, 5.0, 200 + i) for i in val_ecg])

    def lab(stay, h):
        return int(stay % 2 == 0 and h >= 20)

    train = _cohort(stay_hours, lab)
    val = _cohort(val_stays, lambda s, h: int(s % 2 == 0 and h >= 20))
    waves = _waveforms([100 + i for i in ecg_stays] + [200 + i for i in val_ecg], seed)

    agent = cls(hidden=8, latent=4, n_studies=4, rows_per_study=2, max_epochs=2,
                encode_batch=8, verbose=False, device=torch.device("cpu"))
    agent.store = WaveformStore.from_arrays(waves)
    agent._index = pd.concat([index, val_index], ignore_index=True)
    if isinstance(agent, EcgResnetFilmAgent):
        agent._age_lookup = {500 + s: 40.0 + s for s in list(stay_hours) + list(val_stays)}
    return agent.fit(train, val), train, val


def test_v2_agent_emits_latent_and_calibrated_risk():
    agent, _, val = _fitted(EcgResnetAgent)
    out = agent.emit(val)
    assert out.latent.shape == (len(val), 4)
    assert ((out.risk >= 0) & (out.risk <= 1)).all()
    expected = val.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    assert (out.keys["stay_id"].to_numpy() == expected["stay_id"].to_numpy()).all()


def test_v3_agent_emits_and_no_ecg_is_constant_on_real_data():
    agent, _, val = _fitted(EcgResnetFilmAgent)
    out = agent.emit(val)
    raw = agent.raw_risk(val)
    has = agent.has_ecg(val)
    assert out.latent.shape == (len(val), 4)
    n_uniq = len(np.unique(np.round(raw[~has], 6)))
    assert n_uniq == 1, f"no-ECG hours not constant under FiLM ({n_uniq} distinct)"


def test_emit_before_fit_raises():
    agent = EcgResnetAgent(verbose=False)
    agent.store = WaveformStore.from_arrays(_waveforms([1]))
    agent._index = _index([(1, 1.0, 1)])
    with pytest.raises(RuntimeError):
        agent.emit(_cohort({1: [6]}))


def test_satisfies_the_agent_interface():
    assert issubclass(EcgResnetAgent, Agent) and issubclass(EcgResnetFilmAgent, Agent)
    assert EcgResnetAgent(verbose=False).name == "ecg_resnet"
    assert EcgResnetFilmAgent(verbose=False).name == "ecg_resnet_film"
