"""Alignment and leakage audit for the orchestrator's data flow.

The orchestrator's headline AUPRC depends on three agents' vectors being joined
to each other and to the label without a single row slipping. These tests pin
down the four ways that could silently go wrong:

1. **Cross-agent join** -- every fused row pairs the same ``(stay_id, hour)``
   across all agents, and the label belongs to that same key.
2. **Temporal stacking** -- the fused input at hour ``t`` uses agent outputs
   from hours ``<= t`` only.
3. **Agent-level leakage** -- an agent's vector for hour ``t`` was built from
   data before ``t``, and the orchestrator cannot bypass that by re-reading raw
   columns.
4. **Label timing** -- the label at hour ``t`` looks *forward* into
   ``(t, t+24]``, and is not shifted or backward-looking.

Each test is built so that the failure it guards against would actually make it
fail -- the "tampered" halves assert the perturbation registers, so a test that
passes vacuously is caught.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from agents.base import KEY_COLUMNS, AgentOutput
from mimic_cohort import add_event_labels
from orchestrator.fusion import AgentBank, LatentFusionNet, Orchestrator

LABEL = "label_24h"
AGENTS = ("a1", "a2", "a3")
DIMS = {"a1": 3, "a2": 2, "a3": 4}
TOTAL_LATENT = sum(DIMS.values())


def _bank_with_traceable_values(stay_hours):
    """A bank whose every number encodes the row and agent it came from.

    Agent ``k``'s latent for ``(stay, hour)`` is filled with
    ``stay*1000 + hour + k/10``, so any mis-join shows up as a value that
    decodes to the wrong key or the wrong agent.
    """
    rows = [(s, h) for s, hours in stay_hours.items() for h in hours]
    keys = pd.DataFrame(rows, columns=list(KEY_COLUMNS))
    blocks = []
    for k, name in enumerate(AGENTS):
        tag = np.array([s * 1000 + h + k / 10 for s, h in rows], dtype=np.float32)
        blocks.append(np.repeat(tag[:, None], DIMS[name], axis=1))
    risks = np.stack(
        [np.array([(s * 1000 + h) / 1e6 + k for s, h in rows], dtype=np.float32)
         for k in range(len(AGENTS))], axis=1)
    matrix = np.concatenate(blocks + [risks], axis=1).astype(np.float32)
    return AgentBank(keys, matrix, DIMS, AGENTS)


def _cohort(stay_hours, labels=None):
    rows = []
    for s, hours in stay_hours.items():
        for h in hours:
            rows.append({"stay_id": s, "patient_id": 100 + s, "hour": h,
                         LABEL: 0 if labels is None else labels[(s, h)]})
    return pd.DataFrame(rows)


# --- 1. Cross-agent join alignment ----------------------------------------

def test_every_fused_row_pairs_the_same_key_across_all_agents_and_label():
    """Trace each fused row back: all agents and the label must agree on the key."""
    stay_hours = {7: [6, 7, 8], 3: [6, 7]}
    labels = {(7, 6): 1, (7, 7): 0, (7, 8): 1, (3, 6): 0, (3, 7): 1}
    bank = _bank_with_traceable_values(stay_hours)
    cohort = _cohort(stay_hours, labels).sample(frac=1, random_state=0)  # shuffled

    orch = Orchestrator(bank=bank, verbose=False)
    sequences = orch._sequences(cohort)

    seen = set()
    for seq in sequences:
        stay = int(seq["stay_id"])
        for pos, hour in enumerate(seq["hours"]):
            hour = int(hour)
            row = seq["X"][pos]
            offset = 0
            for k, name in enumerate(AGENTS):
                block = row[offset:offset + DIMS[name]]
                expected = stay * 1000 + hour + k / 10
                assert np.allclose(block, expected), (
                    f"agent {name} at ({stay},{hour}) carries {block[0]}, "
                    f"which decodes to a different key/agent")
                offset += DIMS[name]
            # the label attached to this position must be this key's label
            assert seq["y"][pos] == labels[(stay, hour)]
            seen.add((stay, hour))
    assert seen == set(labels)          # every cohort row appeared exactly once


def test_shuffling_the_cohort_does_not_change_the_fused_rows():
    stay_hours = {1: [6, 7, 8], 2: [6, 7]}
    bank = _bank_with_traceable_values(stay_hours)
    cohort = _cohort(stay_hours)
    orch = Orchestrator(bank=bank, verbose=False)

    a = orch._sequences(cohort)
    b = orch._sequences(cohort.sample(frac=1, random_state=3))
    assert [int(s["stay_id"]) for s in a] == [int(s["stay_id"]) for s in b]
    for x, y in zip(a, b):
        assert np.array_equal(x["hours"], y["hours"])
        assert np.array_equal(x["X"], y["X"])
        assert np.array_equal(x["y"], y["y"])


def test_bank_refuses_to_join_agents_whose_keys_differ(tmp_path):
    good = AgentOutput(keys=pd.DataFrame({"stay_id": [1, 1], "hour": [6, 7]}),
                       latent=np.zeros((2, 2), dtype=np.float32),
                       risk=np.zeros(2, dtype=np.float32))
    offset_by_one = AgentOutput(
        keys=pd.DataFrame({"stay_id": [1, 1], "hour": [7, 8]}),   # shifted!
        latent=np.zeros((2, 2), dtype=np.float32),
        risk=np.zeros(2, dtype=np.float32))
    good.save(tmp_path / "p_train.npz")
    offset_by_one.save(tmp_path / "q_train.npz")
    with pytest.raises(ValueError, match="different keys"):
        AgentBank.load(splits=("train",), agent_names=("p", "q"), emb_dir=tmp_path)


def test_missing_agent_emission_is_an_error_not_a_silent_gap():
    bank = _bank_with_traceable_values({1: [6, 7]})
    with pytest.raises(ValueError, match="no cached agent emission"):
        bank.lookup(_cohort({1: [6, 7, 8]}))


# --- 2. Temporal stacking: only hours <= t --------------------------------

def test_fused_output_at_hour_t_ignores_all_later_hours():
    """Hand-built: rewrite agent signals after hour 5, assert hours 0-5 unmoved."""
    torch.manual_seed(0)
    net = LatentFusionNet(DIMS, AGENTS, d_model=8, dilations=(1, 2)).eval()
    T = 12
    latents = {n: torch.randn(1, T, d) for n, d in DIMS.items()}
    risks = torch.rand(1, T, len(AGENTS))

    with torch.no_grad():
        base, base_emb, base_attn = net(latents, risks)
        for cut in (3, 5, 8):
            tampered = {k: v.clone() for k, v in latents.items()}
            for v in tampered.values():
                v[:, cut:, :] = 99.0
            tampered_risks = risks.clone()
            tampered_risks[:, cut:, :] = 0.0
            out, emb, attn = net(tampered, tampered_risks)
            assert torch.allclose(base[:, :cut], out[:, :cut], atol=1e-6), \
                f"future beyond hour {cut} changed an earlier prediction"
            assert torch.allclose(base_emb[:, :cut], emb[:, :cut], atol=1e-6)
            assert torch.allclose(base_attn[:, :cut], attn[:, :cut], atol=1e-6)
            # the tampering must register from the cut onwards
            assert not torch.allclose(base[:, cut:], out[:, cut:], atol=1e-6)


def test_truncating_a_stay_does_not_change_the_surviving_predictions():
    """A prediction at hour t is identical whether or not later hours exist."""
    torch.manual_seed(1)
    net = LatentFusionNet(DIMS, AGENTS, d_model=8, dilations=(1, 2)).eval()
    T = 10
    latents = {n: torch.randn(1, T, d) for n, d in DIMS.items()}
    risks = torch.rand(1, T, len(AGENTS))
    with torch.no_grad():
        full, _, _ = net(latents, risks)
        short, _, _ = net({k: v[:, :6] for k, v in latents.items()}, risks[:, :6])
    torch.testing.assert_close(full[:, :6], short, atol=1e-6, rtol=1e-6)


def test_sequence_hours_are_ascending_and_positions_track_real_hours():
    stay_hours = {1: [6, 7, 8, 9]}
    orch = Orchestrator(bank=_bank_with_traceable_values(stay_hours), verbose=False)
    seq = orch._sequences(_cohort(stay_hours))[0]
    assert list(seq["hours"]) == [6, 7, 8, 9]
    assert (np.diff(seq["hours"]) == 1).all()   # consecutive: position i == hour_0 + i


# --- 3. Agent-level leakage / raw-data isolation --------------------------

def test_orchestrator_reads_only_keys_and_label_never_raw_features():
    """Scrambling raw cohort columns must not change the fused input at all."""
    stay_hours = {1: [6, 7, 8], 2: [6, 7]}
    bank = _bank_with_traceable_values(stay_hours)
    cohort = _cohort(stay_hours)
    # add raw-looking columns of the kind the real cohort carries
    rng = np.random.default_rng(0)
    for col in ("heart_rate", "spo2", "news2", "gcs", "on_oxygen"):
        cohort[col] = rng.normal(size=len(cohort))

    orch = Orchestrator(bank=bank, verbose=False)
    before = orch._sequences(cohort)
    scrambled = cohort.copy()
    for col in ("heart_rate", "spo2", "news2", "gcs", "on_oxygen"):
        scrambled[col] = rng.normal(size=len(scrambled)) * 1e6
    after = orch._sequences(scrambled)

    for a, b in zip(before, after):
        assert np.array_equal(a["X"], b["X"]), \
            "fused features changed when raw columns changed -- the orchestrator " \
            "is reading raw data instead of the agents' cached latents"
        assert np.array_equal(a["y"], b["y"])


# --- 4. Label timing -------------------------------------------------------

def test_label_looks_forward_into_the_next_24_hours():
    """Event at hour 30: label is 1 exactly for hours in [6, 29], not before."""
    hours = list(range(0, 30))
    df = pd.DataFrame({"stay_id": 1, "hour": hours, "event_h": 30.0,
                       "vent_h": 30.0, "vaso_h": np.nan, "death_h": np.nan})
    out = add_event_labels(df).set_index("hour")
    for h in hours:
        expected = int(30 > h and 30 <= h + 24)      # (h, h+24]
        assert int(out.loc[h, "label_24h_vent"]) == expected, f"hour {h}"
    assert int(out.loc[5, "label_24h_vent"]) == 0    # 30 > 5+24 -> outside window
    assert int(out.loc[6, "label_24h_vent"]) == 1    # 30 == 6+24 -> inclusive edge


def test_label_is_not_backward_looking():
    """An event already in the past must never light up the label."""
    df = pd.DataFrame({"stay_id": 1, "hour": [10, 11, 12], "event_h": 5.0,
                       "vent_h": 5.0, "vaso_h": np.nan, "death_h": np.nan})
    out = add_event_labels(df)
    assert (out["label_24h_vent"] == 0).all()


def test_label_is_not_shifted_by_one_hour():
    """Distinguish (h, h+24] from the off-by-one alternatives."""
    df = pd.DataFrame({"stay_id": 1, "hour": [5, 6, 7], "event_h": 30.0,
                       "vent_h": 30.0, "vaso_h": np.nan, "death_h": np.nan})
    got = add_event_labels(df)["label_24h_vent"].tolist()
    assert got == [0, 1, 1]                    # correct: window opens at hour 6
    assert got != [1, 1, 1]                    # would mean (h-1, h+23] or wider
    assert got != [0, 0, 1]                    # would mean (h+1, h+25]
