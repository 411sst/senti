"""Tests for the notes agent (agent #5, radiology reports).

Three families:

* **Leakage** -- a note is usable only from when it became AVAILABLE, strictly
  before the prediction hour, and only while fresh.
* **Presence-leakage** -- imaging is ordered on deteriorating patients (5.09% vs
  4.18% event rate), so "a report exists" is prognostic by itself. The head must
  be structurally incapable of ranking within the no-note group.
* **Discharge exclusion** -- discharge summaries narrate the outcome and must
  never enter the pipeline.
"""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import agents.notes_agent as notes_module
from agents.base import Agent
from agents.note_encoder import EMBED_DIM
from agents.notes_agent import (
    STALENESS_HOURS,
    NoteHead,
    NotesAgent,
    assemble_note_features,
    presence_only_auprc,
)

LABEL = "label_24h"


def _cohort(stay_hours, label_fn=None):
    rows = []
    for stay, hours in stay_hours.items():
        for h in hours:
            rows.append({"stay_id": stay, "patient_id": 700 + stay, "hour": h,
                         LABEL: 0 if label_fn is None else label_fn(stay, h)})
    return pd.DataFrame(rows)


def _index(records):
    """records: [(stay_id, note_hour, note_id)] -- note_id is a STRING."""
    return pd.DataFrame(records, columns=["stay_id", "note_hour", "note_id"])


def _embeddings(note_ids, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(len(note_ids), EMBED_DIM)).astype(np.float32),
            np.array([str(n) for n in note_ids], dtype=object))


# --- Leakage ---------------------------------------------------------------

def test_note_available_at_hour_t_is_not_used():
    """A report that lands exactly at t must be invisible at t."""
    index = _index([(1, 7.0, "n1")])
    emb, ids = _embeddings(["n1"])
    _, has, _, keys = assemble_note_features(_cohort({1: [6, 7, 8]}), index, emb, ids)
    by_hour = dict(zip(keys["hour"], has))
    assert by_hour[6] == 0.0        # before it exists
    assert by_hour[7] == 0.0        # its OWN availability hour -> excluded
    assert by_hour[8] == 1.0        # usable the hour after


def test_stale_note_is_dropped_at_the_bound():
    note_hour = 9
    at_bound = note_hour + STALENESS_HOURS
    index = _index([(1, float(note_hour), "n1")])
    emb, ids = _embeddings(["n1"])
    _, has, age, keys = assemble_note_features(
        _cohort({1: [10, at_bound, at_bound + 1]}), index, emb, ids)
    by_hour = dict(zip(keys["hour"], has))
    assert dict(zip(keys["hour"], age))[at_bound] == pytest.approx(STALENESS_HOURS)
    assert by_hour[at_bound] == 1.0          # exactly at the bound: kept
    assert by_hour[at_bound + 1] == 0.0      # one hour past: dropped


def test_most_recent_prior_note_wins():
    index = _index([(1, 5.0, "old"), (1, 18.0, "recent"), (1, 25.0, "future")])
    emb, ids = _embeddings(["old", "recent", "future"])
    X, has, age, _ = assemble_note_features(_cohort({1: [20]}), index, emb, ids)
    assert has[0] == 1.0
    assert age[0] == pytest.approx(2.0)
    order = list(ids)
    np.testing.assert_allclose(X[0], emb[order.index("recent")])


def test_notes_do_not_cross_stays():
    index = _index([(1, 5.0, "n1")])
    emb, ids = _embeddings(["n1"])
    _, has, _, keys = assemble_note_features(_cohort({1: [10], 2: [10]}), index, emb, ids)
    m = dict(zip(keys["stay_id"], has))
    assert m[1] == 1.0 and m[2] == 0.0


def test_unencoded_note_is_treated_as_absent():
    """A note in the index but missing from the embedding cache is not faked."""
    index = _index([(1, 5.0, "missing")])
    emb, ids = _embeddings(["other"])
    X, has, _, _ = assemble_note_features(_cohort({1: [10]}), index, emb, ids)
    assert has[0] == 0.0 and np.allclose(X[0], 0.0)


# --- Discharge summaries must never be read --------------------------------

def test_pipeline_never_references_the_discharge_table():
    """Structural check: no module in the notes path touches discharge notes.

    Discharge summaries are written a median of 57h AFTER ICU discharge and
    narrate the outcome, so reading them would be direct label leakage.
    """
    repo = Path(notes_module.__file__).resolve().parents[2]
    sources = [
        Path(notes_module.__file__).read_text(),
        (repo / "src" / "agents" / "note_encoder.py").read_text(),
        (repo / "scripts" / "fetch_note_texts.py").read_text(),
        (repo / "scripts" / "encode_notes.py").read_text(),
    ]
    for src in sources:
        code = "\n".join(line for line in src.splitlines()
                         if not line.strip().startswith(("#", '"', "*")))
        assert "mimiciv_note.discharge" not in code
        assert "discharge.csv" not in code


def test_features_come_only_from_the_note_index_and_cache():
    """assemble_note_features reads its three arguments and nothing else."""
    params = list(inspect.signature(assemble_note_features).parameters)
    assert params[:4] == ["cohort", "note_index", "embeddings", "note_ids"]


# --- Presence-leakage guard ------------------------------------------------

def test_all_no_note_rows_receive_identical_input_and_identical_risk():
    """No-note hours carry no varying signal, so the head must emit a constant.

    Without this, the agent could score well by learning "imaging was ordered ->
    sicker patient", which is workflow, not radiology.
    """
    torch.manual_seed(0)
    head = NoteHead(hidden=8, latent=4).eval()
    n = 6
    with torch.no_grad():
        logits, latents = head(torch.randn(n, EMBED_DIM),   # different embeddings
                               torch.zeros(n),               # but none available
                               torch.rand(n) * 50)           # and differing ages
    assert torch.allclose(logits, logits[0].expand(n), atol=1e-6), \
        "no-note rows produced different risks -- input is leaking through"
    assert torch.allclose(latents, latents[0].expand_as(latents), atol=1e-6)


def test_no_note_vector_is_learned_not_just_zeros():
    torch.manual_seed(1)
    head = NoteHead(hidden=8, latent=4).eval()
    with torch.no_grad():
        zeros = torch.zeros(1, EMBED_DIM)
        absent, _ = head(zeros, torch.zeros(1), torch.zeros(1))
        present, _ = head(zeros, torch.ones(1), torch.zeros(1))
    assert not torch.allclose(absent, present, atol=1e-5)


def test_covered_rows_vary_with_the_note_text():
    torch.manual_seed(2)
    head = NoteHead(hidden=8, latent=4).eval()
    with torch.no_grad():
        logits, _ = head(torch.randn(5, EMBED_DIM), torch.ones(5), torch.zeros(5))
    assert logits.std() > 1e-4


def test_presence_only_auprc_is_a_meaningful_floor():
    y = np.r_[np.ones(51), np.zeros(949), np.ones(42), np.zeros(958)]
    has = np.r_[np.ones(1000), np.zeros(1000)]
    assert presence_only_auprc(y, has) > y.mean()
    assert presence_only_auprc(y, np.ones_like(y)) == pytest.approx(y.mean(), abs=1e-6)


# --- Agent contract --------------------------------------------------------

def _fitted(seed=0):
    hours = list(range(6, 26))
    train_stays = {i: hours for i in range(1, 9)}
    val_stays = {i + 10: hours for i in range(1, 9)}
    recs = [(i, 5.0, f"t{i}") for i in range(1, 9) if i % 2 == 0]
    recs += [(i + 10, 5.0, f"v{i}") for i in range(1, 9) if i % 2 == 0]
    index = _index(recs)
    emb, ids = _embeddings([r[2] for r in recs], seed=seed)

    def lab(stay, h):
        return int(stay % 2 == 0 and h >= 20)

    agent = NotesAgent(hidden=8, latent=4, max_epochs=2, batch_size=64, verbose=False)
    agent._cache = (emb, ids, index)
    return agent.fit(_cohort(train_stays, lab), _cohort(val_stays, lab)), \
        _cohort(val_stays, lab)


def test_agent_emits_latent_and_calibrated_risk():
    agent, val = _fitted()
    out = agent.emit(val)
    assert out.latent.shape == (len(val), 4)
    assert ((out.risk >= 0) & (out.risk <= 1)).all()
    expected = val.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    assert (out.keys["stay_id"].to_numpy() == expected["stay_id"].to_numpy()).all()


def test_emit_before_fit_raises():
    agent = NotesAgent(verbose=False)
    agent._cache = (np.zeros((1, EMBED_DIM), np.float32), np.array(["a"], dtype=object),
                    _index([(1, 1.0, "a")]))
    with pytest.raises(RuntimeError):
        agent.emit(_cohort({1: [6]}))


def test_satisfies_the_agent_interface():
    assert issubclass(NotesAgent, Agent)
    assert NotesAgent(verbose=False).name == "notes"
