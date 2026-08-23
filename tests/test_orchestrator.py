"""Tests for the orchestrator (latent fusion + trajectory reasoning).

The load-bearing test is :func:`test_orchestrator_is_causal`. The orchestrator
is the deepest temporal model in the project and sits downstream of three
agents, so a causality break here would silently invalidate every headline
number. As with the vitals-TCN, it is checked empirically: rewrite the future,
assert the past does not move.
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.isotonic import IsotonicRegression

from agents.base import KEY_COLUMNS, Agent, AgentOutput
from orchestrator.fusion import AgentBank, LatentFusionNet, Orchestrator

LABEL = "label_24h"
AGENTS = ("a1", "a2")
DIMS = {"a1": 4, "a2": 3}


def _bank(stay_hours, seed=0):
    """A synthetic AgentBank over the given {stay_id: [hours]}."""
    rng = np.random.default_rng(seed)
    rows = [(s, h) for s, hours in stay_hours.items() for h in hours]
    keys = pd.DataFrame(rows, columns=list(KEY_COLUMNS))
    n = len(keys)
    matrix = rng.normal(size=(n, sum(DIMS.values()) + len(AGENTS))).astype(np.float32)
    return AgentBank(keys, matrix, DIMS, AGENTS)


def _cohort(stay_hours, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s, hours in stay_hours.items():
        for h in hours:
            rows.append({"stay_id": s, "patient_id": 100 + s, "hour": h,
                         LABEL: int(rng.random() < 0.3)})
    return pd.DataFrame(rows)


# --- Causality: the leakage guarantee for the fused temporal stack --------

def test_orchestrator_is_causal():
    """Rewriting agent signals at hour t must not change outputs before t."""
    torch.manual_seed(0)
    net = LatentFusionNet(DIMS, AGENTS, d_model=8, dilations=(1, 2)).eval()
    latents = {name: torch.randn(1, 20, dim) for name, dim in DIMS.items()}
    risks = torch.rand(1, 20, len(AGENTS))

    with torch.no_grad():
        base_logits, base_emb, base_attn = net(latents, risks)
        # tamper with EVERY input channel from hour 10 onwards
        tampered = {k: v.clone() for k, v in latents.items()}
        for v in tampered.values():
            v[:, 10:, :] += 5.0
        tampered_risks = risks.clone()
        tampered_risks[:, 10:, :] = 1.0
        new_logits, new_emb, new_attn = net(tampered, tampered_risks)

    assert torch.allclose(base_logits[:, :10], new_logits[:, :10], atol=1e-6)
    assert torch.allclose(base_emb[:, :10], new_emb[:, :10], atol=1e-6)
    assert torch.allclose(base_attn[:, :10], new_attn[:, :10], atol=1e-6)
    # the perturbation must actually register, or the test proves nothing
    assert not torch.allclose(base_logits[:, 10:], new_logits[:, 10:], atol=1e-6)


def test_attention_is_over_agents_not_over_time():
    """Attention weights sum to 1 across agents at every hour."""
    torch.manual_seed(1)
    net = LatentFusionNet(DIMS, AGENTS, d_model=8, dilations=(1,)).eval()
    latents = {name: torch.randn(2, 7, dim) for name, dim in DIMS.items()}
    with torch.no_grad():
        _, _, weights = net(latents, torch.rand(2, 7, len(AGENTS)))
    assert weights.shape == (2, 7, len(AGENTS))
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 7))


def test_output_length_matches_input_and_single_hour_works():
    net = LatentFusionNet(DIMS, AGENTS, d_model=8, dilations=(1, 2))
    for T in (1, 5):
        latents = {name: torch.randn(2, T, dim) for name, dim in DIMS.items()}
        logits, emb, attn = net(latents, torch.rand(2, T, len(AGENTS)))
        assert logits.shape == (2, T)
        assert emb.shape == (2, T, 8)
        assert attn.shape == (2, T, len(AGENTS))


# --- The agent bank: row-for-row joining ----------------------------------

def test_bank_lookup_returns_rows_in_sorted_key_order():
    bank = _bank({2: [6, 7], 1: [6, 7, 8]})
    cohort = _cohort({2: [6, 7], 1: [6, 7, 8]}).sample(frac=1, random_state=0)
    X, keys = bank.lookup(cohort)
    expected = cohort.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    assert len(X) == len(cohort)
    assert (keys["stay_id"].to_numpy() == expected["stay_id"].to_numpy()).all()
    assert (keys["hour"].to_numpy() == expected["hour"].to_numpy()).all()


def test_bank_raises_when_a_cohort_row_has_no_cached_emission():
    bank = _bank({1: [6, 7]})
    with pytest.raises(ValueError, match="no cached agent emission"):
        bank.lookup(_cohort({1: [6, 7, 99]}))


def test_bank_rejects_agents_with_mismatched_keys(tmp_path):
    """Agents must emit on matched keys; a mismatch is an error, not a silent join."""
    a = AgentOutput(keys=pd.DataFrame({"stay_id": [1, 1], "hour": [6, 7]}),
                    latent=np.zeros((2, 3), dtype=np.float32),
                    risk=np.zeros(2, dtype=np.float32))
    b = AgentOutput(keys=pd.DataFrame({"stay_id": [1, 1], "hour": [6, 8]}),  # differs
                    latent=np.zeros((2, 3), dtype=np.float32),
                    risk=np.zeros(2, dtype=np.float32))
    a.save(tmp_path / "x_train.npz")
    b.save(tmp_path / "y_train.npz")
    with pytest.raises(ValueError, match="different keys"):
        AgentBank.load(splits=("train",), agent_names=("x", "y"), emb_dir=tmp_path)


def test_split_features_recovers_each_agents_block():
    bank = _bank({1: [6, 7]})
    x = torch.arange(2 * 9, dtype=torch.float32).reshape(2, 9)
    latents, risks = bank.split_features(x)
    assert latents["a1"].shape == (2, 4)
    assert latents["a2"].shape == (2, 3)
    assert risks.shape == (2, 2)
    # blocks must be contiguous slices in agent order, then the risks
    torch.testing.assert_close(latents["a1"], x[:, 0:4])
    torch.testing.assert_close(latents["a2"], x[:, 4:7])
    torch.testing.assert_close(risks, x[:, 7:9])


# --- Orchestrator contract -------------------------------------------------

def _fitted(seed=0, hours=12):
    stay_hours = {i: list(range(6, 6 + hours)) for i in range(1, 7)}
    bank = _bank(stay_hours, seed=seed)
    train = _cohort(stay_hours, seed=seed)
    val = _cohort(stay_hours, seed=seed + 1)
    orch = Orchestrator(bank=bank, d_model=8, dilations=(1, 2), max_epochs=1,
                        batch_size=4, verbose=False)
    return orch.fit(train, val), train, val


def test_orchestrator_emits_fused_latent_and_calibrated_risk():
    orch, _, val = _fitted()
    out = orch.emit(val)
    assert out.latent.shape == (len(val), orch.latent_dim)
    assert out.risk.shape == (len(val),)
    assert ((out.risk >= 0) & (out.risk <= 1)).all()
    expected = val.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    assert (out.keys["stay_id"].to_numpy() == expected["stay_id"].to_numpy()).all()


def test_calibration_is_fit_only_on_val(monkeypatch):
    """Same guarantee as every agent: isotonic sees val labels, never train's."""
    captured = {}
    original = IsotonicRegression.fit

    def spy(self, X, y, **kw):
        captured["y"] = np.asarray(y).copy()
        return original(self, X, y, **kw)

    monkeypatch.setattr(IsotonicRegression, "fit", spy)

    stay_hours = {i: list(range(6, 18)) for i in range(1, 7)}
    bank = _bank(stay_hours)
    train = _cohort(stay_hours, seed=2)
    val = _cohort({i: list(range(6, 12)) for i in range(1, 4)}, seed=3)
    assert len(train) != len(val)

    Orchestrator(bank=bank, d_model=8, dilations=(1,), max_epochs=1,
                 batch_size=4, verbose=False).fit(train, val)
    val_sorted = val.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    assert len(captured["y"]) == len(val)
    assert np.array_equal(captured["y"], val_sorted[LABEL].to_numpy(dtype=float))


def test_attention_weights_are_reported_per_row():
    orch, _, val = _fitted(seed=4)
    keys, weights = orch.attention_weights(val)
    assert len(keys) == len(val)
    assert weights.shape == (len(val), len(AGENTS))
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)


def test_checkpoint_roundtrip_reproduces_scores(tmp_path):
    """save/load_checkpoint restore the exact net AND calibrator.

    Frozen-head experiments depend on this: a reloaded orchestrator must score
    a bank identically to the fitted original, raw and calibrated.
    """
    orch, _, val = _fitted()
    path = orch.save_checkpoint(tmp_path / "orch.pt")
    loaded = Orchestrator.load_checkpoint(path, orch.bank)
    np.testing.assert_allclose(loaded.raw_risk(val), orch.raw_risk(val),
                               atol=1e-6)
    np.testing.assert_allclose(loaded.emit(val).risk, orch.emit(val).risk,
                               atol=1e-6)


def test_checkpoint_rejects_mismatched_bank(tmp_path):
    orch, _, _ = _fitted()
    path = orch.save_checkpoint(tmp_path / "orch.pt")
    wrong_dims = AgentBank(orch.bank.keys, orch.bank.matrix[:, :-1],
                           {"a1": 4, "a2": 2}, AGENTS)
    with pytest.raises(ValueError, match="dims"):
        Orchestrator.load_checkpoint(path, wrong_dims)


def test_checkpoint_before_fit_raises(tmp_path):
    orch = Orchestrator(bank=_bank({1: [6, 7]}), verbose=False)
    with pytest.raises(RuntimeError, match="not fitted"):
        orch.save_checkpoint(tmp_path / "orch.pt")


def test_emit_before_fit_raises():
    orch = Orchestrator(bank=_bank({1: [6, 7]}), verbose=False)
    with pytest.raises(RuntimeError):
        orch.emit(_cohort({1: [6, 7]}))


def test_orchestrator_satisfies_the_agent_interface():
    assert issubclass(Orchestrator, Agent)
    assert Orchestrator(verbose=False).name == "orchestrator"


# --- Shipped configurations + graceful degradation -------------------------

def test_shipped_configs_are_what_the_ablation_selected():
    from orchestrator.fusion import DEFAULT_CONFIG, ORCHESTRATOR_CONFIGS
    assert DEFAULT_CONFIG == "primary"
    # Primary == core-3, selected on VAL (modality_ablation --split val):
    # notes and ECG are retained-but-benched until they clear the rejoin bar
    # (positive val fusion contribution, CI clear of zero).
    assert ORCHESTRATOR_CONFIGS["primary"] == ("vitals_tcn", "news2", "labs")
    assert ORCHESTRATOR_CONFIGS["core-3"] == ("vitals_tcn", "news2", "labs")
    assert ORCHESTRATOR_CONFIGS["core-3+notes"] == ("vitals_tcn", "news2", "labs", "notes")
    assert ORCHESTRATOR_CONFIGS["all-5"] == (
        "vitals_tcn", "news2", "labs", "ecg", "notes")
    # ECG significantly hurt the fusion, so it must not be in the primary path.
    assert "ecg" not in ORCHESTRATOR_CONFIGS["primary"]


def _touch_agents(tmp_path, agents, splits=("train", "val", "test")):
    for a in agents:
        for s in splits:
            (tmp_path / f"{a}_{s}.npz").write_bytes(b"")


def test_resolve_config_uses_primary_when_everything_is_cached(tmp_path):
    from orchestrator.fusion import ORCHESTRATOR_CONFIGS, resolve_config
    _touch_agents(tmp_path, ORCHESTRATOR_CONFIGS["all-5"])
    name, agents = resolve_config(emb_dir=tmp_path)
    assert name == "primary"
    assert agents == ORCHESTRATOR_CONFIGS["primary"]


def test_resolve_config_needs_no_benched_modalities(tmp_path):
    """A site with only the core three agents runs the PRIMARY config outright:
    since the val-based selection benched notes and ECG, primary == core-3 and
    there is nothing left to degrade to."""
    from orchestrator.fusion import resolve_config
    _touch_agents(tmp_path, ("vitals_tcn", "news2", "labs"))
    name, agents = resolve_config(emb_dir=tmp_path)
    assert name == "primary"
    assert "notes" not in agents and "ecg" not in agents


def test_degradation_never_adds_a_modality_the_caller_did_not_ask_for(tmp_path):
    from orchestrator.fusion import ORCHESTRATOR_CONFIGS, resolve_config
    _touch_agents(tmp_path, ORCHESTRATOR_CONFIGS["all-5"])
    for preferred in ("primary", "core-3"):
        _, agents = resolve_config(preferred, emb_dir=tmp_path)
        assert "ecg" not in agents      # only reachable by asking for all-5


def test_partial_cache_does_not_count_as_available(tmp_path):
    """An agent missing even one split must not be treated as usable."""
    from orchestrator.fusion import available_agents
    _touch_agents(tmp_path, ("vitals_tcn", "news2", "labs"))
    _touch_agents(tmp_path, ("notes",), splits=("train", "val"))   # no test split
    assert "notes" not in available_agents(emb_dir=tmp_path)


def test_resolve_config_raises_when_even_core3_is_incomplete(tmp_path):
    from orchestrator.fusion import resolve_config
    _touch_agents(tmp_path, ("vitals_tcn", "news2"))
    with pytest.raises(RuntimeError, match="no usable configuration"):
        resolve_config(emb_dir=tmp_path)


def test_flagship_chain_is_strictly_nested():
    """Degradation must only ever REMOVE modalities: every entry's agents are
    a subset of the entry before it."""
    from orchestrator.fusion import (FLAGSHIP_DEGRADATION_ORDER,
                                     ORCHESTRATOR_CONFIGS)
    chain = [set(ORCHESTRATOR_CONFIGS[c]) for c in FLAGSHIP_DEGRADATION_ORDER]
    for bigger, smaller in zip(chain, chain[1:]):
        assert smaller < bigger


def test_flagship_degrades_through_the_chain_not_straight_to_core3(tmp_path):
    """all-8 with resp uncached must land on all-7, keeping cadence/fluid."""
    from orchestrator.fusion import ORCHESTRATOR_CONFIGS, resolve_config
    _touch_agents(tmp_path, ORCHESTRATOR_CONFIGS["all-7"])
    name, agents = resolve_config("all-8", emb_dir=tmp_path)
    assert name == "all-7"
    assert "cadence" in agents and "fluid" in agents and "resp" not in agents


def test_unknown_config_name_is_rejected():
    from orchestrator.fusion import config_agents
    with pytest.raises(KeyError):
        config_agents("core-9")
