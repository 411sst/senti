"""Tests for the H1 gated-residual ECG omni-agent (``ecg_hybrid``)."""

import numpy as np
import pandas as pd
import pytest
import torch

from agents.base import Agent
from agents.ecg_hybrid_agent import (
    DELTA_NAMES,
    EcgHybridAgent,
    EcgHybridNet,
    build_delta_features,
)
from agents.ecg_meas_features import FEATURE_NAMES, REPORT_COLS, build_meas_features

LABEL = "label_24h"
MEAS_DIM = len(FEATURE_NAMES) + len(DELTA_NAMES)


def _meas_row(study_id, rr=800, afib=False):
    row = {"study_id": study_id, "rr_interval": rr, "p_onset": 100, "p_end": 200,
           "qrs_onset": 260, "qrs_end": 350, "t_end": 660, "p_axis": 45,
           "qrs_axis": 40, "t_axis": 45}
    reports = ("Atrial fibrillation",) if afib else ("Sinus rhythm",)
    for i, col in enumerate(REPORT_COLS):
        row[col] = reports[i] if i < len(reports) else None
    return row


def _meas(studies):
    ids, X, names = build_meas_features(pd.DataFrame(
        [_meas_row(s, rr=800 + 10 * k, afib=(k % 3 == 2))
         for k, s in enumerate(studies)]))
    lookup = {int(s): i for i, s in enumerate(ids)}
    return X, lookup, names


# --- Δmeas ------------------------------------------------------------------

def test_delta_features_previous_ecg_within_stay():
    X, lookup, names = _meas([100, 200])
    index = pd.DataFrame([(1, 5.0, 100), (1, 12.0, 200)],
                         columns=["stay_id", "ecg_hour", "study_id"])
    delta = build_delta_features(index, X, names, lookup)
    first, second = delta[(1, 100)], delta[(1, 200)]
    d = dict(zip(DELTA_NAMES, second))
    assert first[-1] == 0.0                      # no previous ECG -> has_prev 0
    assert (first == 0).all()
    assert d["has_prev"] == 1.0
    assert d["hours_since_prev"] == pytest.approx(7.0)      # 12 - 5
    assert d["d_hr"] == pytest.approx(60000/810 - 60000/800, abs=1e-3)


def test_delta_new_onset_afib_fires_once():
    X, lookup, names = _meas([100, 200, 300])    # study 300 (k=2) has AFib
    index = pd.DataFrame([(1, 2.0, 100), (1, 6.0, 200), (1, 9.0, 300)],
                         columns=["stay_id", "ecg_hour", "study_id"])
    delta = build_delta_features(index, X, names, lookup)
    assert dict(zip(DELTA_NAMES, delta[(1, 200)]))["new_afib"] == 0.0
    assert dict(zip(DELTA_NAMES, delta[(1, 300)]))["new_afib"] == 1.0


def test_delta_does_not_cross_stays():
    X, lookup, names = _meas([100, 200])
    index = pd.DataFrame([(1, 5.0, 100), (2, 12.0, 200)],
                         columns=["stay_id", "ecg_hour", "study_id"])
    delta = build_delta_features(index, X, names, lookup)
    assert delta[(2, 200)][-1] == 0.0            # stay 2's first ECG: no prev


# --- Presence guards --------------------------------------------------------

def test_no_ecg_rows_emit_one_constant_regardless_of_wave_and_age_inputs():
    torch.manual_seed(0)
    net = EcgHybridNet(meas_dim=MEAS_DIM, wave_dim=16, wave_proj=8,
                       hidden=8, latent=4).eval()
    n = 6
    meas = torch.randn(n, MEAS_DIM)
    wave = torch.randn(n, 16)
    has = torch.zeros(n)
    with torch.no_grad():
        # no-ECG implies no waveform; ages differ, stale ages differ, wave
        # inputs differ -- NOTHING may move the output on no-ECG rows.
        logits, latents = net(meas, has, wave, torch.zeros(n),
                              torch.rand(n) * 50,
                              torch.tensor([0.2, 0.4, 0.5, 0.7, 0.8, 0.9]))
    assert torch.allclose(logits, logits[0].expand(n), atol=1e-6)
    assert torch.allclose(latents, latents[0].expand_as(latents), atol=1e-6)


def test_film_preserves_no_ecg_constant_with_differing_ages():
    """H6: FiLM(age) modulates the ECG-present branch only -- no-ECG rows must
    stay ONE constant even when patients' ages differ."""
    torch.manual_seed(4)
    net = EcgHybridNet(meas_dim=MEAS_DIM, wave_dim=16, wave_proj=8,
                       hidden=8, latent=4, use_film=True).eval()
    n = 5
    meas = torch.randn(n, MEAS_DIM)
    ages = torch.tensor([0.21, 0.44, 0.58, 0.73, 0.90])
    with torch.no_grad():
        logits, latents = net(meas, torch.zeros(n), torch.randn(n, 16),
                              torch.zeros(n), torch.rand(n) * 50, ages)
    assert torch.allclose(logits, logits[0].expand(n), atol=1e-6), \
        "FiLM leaked age into the no-ECG floor"
    # and on covered rows, differing ages must move the output (FiLM is live)
    with torch.no_grad():
        net.film_generator.gamma_head.weight.data.normal_(std=0.3)
        same_meas = meas[:1].expand(n, -1)
        l2, _ = net(same_meas, torch.ones(n), torch.zeros(n, 16),
                    torch.zeros(n), torch.zeros(n), ages)
    assert l2.std() > 1e-5


def test_gate_starts_at_zero_so_waveform_is_inert_at_init():
    torch.manual_seed(1)
    net = EcgHybridNet(meas_dim=MEAS_DIM, wave_dim=16, wave_proj=8,
                       hidden=8, latent=4).eval()
    meas = torch.randn(1, MEAS_DIM).expand(3, -1)
    waves = torch.randn(3, 16)                   # three DIFFERENT waveforms
    with torch.no_grad():
        logits, _ = net(meas, torch.ones(3), waves, torch.ones(3),
                        torch.zeros(3), torch.full((3,), 0.5))
    assert torch.allclose(logits, logits[0].expand(3), atol=1e-6), \
        "waveform moved the output despite gate=0 init"


def test_open_gate_lets_waveform_vary_covered_rows_only():
    torch.manual_seed(2)
    net = EcgHybridNet(meas_dim=MEAS_DIM, wave_dim=16, wave_proj=8,
                       hidden=8, latent=4).eval()
    with torch.no_grad():
        net.gate.fill_(1.0)
    meas = torch.randn(1, MEAS_DIM).expand(4, -1)
    waves = torch.randn(4, 16)
    with torch.no_grad():
        with_wave, _ = net(meas, torch.ones(4), waves, torch.ones(4),
                           torch.zeros(4), torch.full((4,), 0.5))
        no_wave, _ = net(meas, torch.ones(4), waves, torch.zeros(4),
                         torch.zeros(4), torch.full((4,), 0.5))
    assert with_wave.std() > 1e-4                # waveform now matters...
    assert torch.allclose(no_wave, no_wave[0].expand(4), atol=1e-6)  # ...only when present


# --- Agent contract ---------------------------------------------------------

def _fitted():
    hours = list(range(6, 26))
    stays = {i: hours for i in range(1, 9)}
    ecg_stays = [i for i in range(1, 9) if i % 2 == 0]
    studies = [100 + i for i in ecg_stays] + [200 + i for i in ecg_stays]
    X, lookup, names = _meas(studies)
    index = pd.DataFrame(
        [(i, 5.0, 100 + i) for i in ecg_stays]
        + [(i + 10, 5.0, 200 + i) for i in ecg_stays],
        columns=["stay_id", "ecg_hour", "study_id"])
    delta = build_delta_features(index, X, names, lookup)
    rng = np.random.default_rng(0)
    wave = rng.normal(size=(len(studies), 16)).astype(np.float32)
    wlook = {int(s): i for i, s in enumerate(studies) if i % 2 == 0}  # half on disk
    age = {500 + s: 40.0 + s for s in list(stays) + [i + 10 for i in range(1, 9)]}

    def cohort(offset=0):
        rows = []
        for stay, hs in stays.items():
            for h in hs:
                rows.append({"stay_id": stay + offset, "patient_id": 500 + stay + offset,
                             "hour": h, LABEL: int(stay % 2 == 0 and h >= 20)})
        return pd.DataFrame(rows)

    agent = EcgHybridAgent(hidden=8, latent=4, wave_proj=8, max_epochs=2,
                           batch_size=64, verbose=False)
    agent._cache = (X, lookup, names, wave, wlook, index, delta, age)
    train, val = cohort(0), cohort(10)
    return agent.fit(train, val), train, val


def test_agent_emits_and_no_ecg_constant_on_real_flow():
    agent, _, val = _fitted()
    out = agent.emit(val)
    assert out.latent.shape == (len(val), 4)
    assert ((out.risk >= 0) & (out.risk <= 1)).all()
    raw = agent.raw_risk(val)
    has = agent.has_ecg(val)
    assert len(np.unique(np.round(raw[~has], 6))) == 1


def test_satisfies_the_agent_interface():
    assert issubclass(EcgHybridAgent, Agent)
    assert EcgHybridAgent(verbose=False).name == "ecg_hybrid"
