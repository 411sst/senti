"""Tests for the windowed (rich) notes agent.

The single-note agent's leakage discipline is one ``merge_asof`` call with
``allow_exact_matches=False``; the windowed agent re-implements the same rule by
hand over many notes, so it has to be re-proved rather than inherited:

* a report is visible at hour ``t`` only when ``0 < t - note_hour <= window``,
* the "most recent" slot picks the same report the single-note agent would,
* ``delta`` is exactly zero with one report, so any gain it buys really does
  come from having more than one,
* uncovered rows stay indistinguishable, preserving the presence-leakage guard.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from agents.note_encoder import EMBED_DIM
from agents.notes_rich_agent import (
    RichNoteHead, RichNotesAgent, aggregate_blocks, build_window_index,
    count_only_auprc,
)

WINDOW = 72


def make_notes(rows):
    """rows: list of (stay_id, note_hour, note_id)."""
    return pd.DataFrame(rows, columns=["stay_id", "note_hour", "note_id"])


def make_cohort(rows):
    return pd.DataFrame(rows, columns=["stay_id", "hour"])


def embeddings_for(note_ids, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(len(note_ids), EMBED_DIM)).astype(np.float32)


# --- window resolution ------------------------------------------------------

def test_note_filed_exactly_at_hour_t_is_not_visible():
    """The rule allow_exact_matches=False buys the single-note agent."""
    notes = make_notes([(1, 10.0, "n1")])
    cohort = make_cohort([(1, 10), (1, 11)])
    ids = np.array(["n1"])
    index = build_window_index(cohort, notes, ids, WINDOW)
    assert index.counts.tolist() == [0, 1]      # hour 10 blind, hour 11 sees it


def test_notes_outside_the_window_are_dropped():
    notes = make_notes([(1, 5.0, "old"), (1, 70.0, "new")])
    ids = np.array(["old", "new"])
    cohort = make_cohort([(1, 80)])
    assert build_window_index(cohort, notes, ids, 24).counts.tolist() == [1]
    assert build_window_index(cohort, notes, ids, 76).counts.tolist() == [2]


def test_future_notes_never_appear():
    notes = make_notes([(1, 50.0, "future")])
    index = build_window_index(make_cohort([(1, 20)]), notes,
                               np.array(["future"]), WINDOW)
    assert index.counts.tolist() == [0]


def test_positions_are_ordered_most_recent_first():
    notes = make_notes([(1, 10.0, "a"), (1, 30.0, "b"), (1, 20.0, "c")])
    ids = np.array(["a", "b", "c"])
    index = build_window_index(make_cohort([(1, 40)]), notes, ids, WINDOW)
    assert index.ages.tolist() == [10.0, 20.0, 30.0]        # b, c, a
    assert index.positions.tolist() == [1, 2, 0]


def test_notes_do_not_leak_across_stays():
    notes = make_notes([(1, 10.0, "a"), (2, 10.0, "b")])
    ids = np.array(["a", "b"])
    index = build_window_index(make_cohort([(1, 20), (2, 20)]), notes, ids, WINDOW)
    assert index.counts.tolist() == [1, 1]
    assert index.positions.tolist() == [0, 1]               # each sees its own


def test_unencoded_notes_count_as_absent():
    """A note with no embedding must not silently shift the 'most recent' slot."""
    notes = make_notes([(1, 30.0, "encoded"), (1, 35.0, "missing")])
    index = build_window_index(make_cohort([(1, 40)]), notes,
                               np.array(["encoded"]), WINDOW)
    assert index.counts.tolist() == [1]
    assert index.ages.tolist() == [10.0]


def test_rows_are_returned_in_key_sorted_order():
    notes = make_notes([(1, 1.0, "a"), (2, 1.0, "b")])
    cohort = make_cohort([(2, 9), (1, 5), (2, 4), (1, 8)])
    index = build_window_index(cohort, notes, np.array(["a", "b"]), WINDOW)
    assert index.keys.stay_id.tolist() == [1, 1, 2, 2]
    assert index.keys.hour.tolist() == [5, 8, 4, 9]


# --- aggregation ------------------------------------------------------------

@pytest.fixture
def three_notes():
    notes = make_notes([(1, 10.0, "a"), (1, 20.0, "b"), (1, 30.0, "c")])
    ids = np.array(["a", "b", "c"])
    emb = embeddings_for(ids)
    index = build_window_index(make_cohort([(1, 40)]), notes, ids, WINDOW)
    return index, emb


def test_recent_block_is_the_latest_note(three_notes):
    index, emb = three_notes
    blocks, has, age, n, span = aggregate_blocks(index, emb, [0], "recent")
    assert np.allclose(blocks[0], emb[2])       # note "c" at hour 30
    assert has[0] == 1.0 and age[0] == 10.0 and n[0] == 3 and span[0] == 30.0


def test_mean_block_averages_the_window(three_notes):
    index, emb = three_notes
    blocks, _, _, _, _ = aggregate_blocks(index, emb, [0], "mean")
    assert np.allclose(blocks[0], emb.mean(axis=0), atol=1e-5)


def test_rich_delta_is_recent_minus_mean_of_older(three_notes):
    index, emb = three_notes
    blocks, _, _, _, _ = aggregate_blocks(index, emb, [0], "rich")
    recent, mean, delta = np.split(blocks[0], 3)
    assert np.allclose(recent, emb[2], atol=1e-5)
    assert np.allclose(mean, emb.mean(axis=0), atol=1e-5)
    assert np.allclose(delta, emb[2] - emb[:2].mean(axis=0), atol=1e-5)


def test_delta_is_exactly_zero_with_a_single_note():
    """So any gain attributed to delta really comes from having >1 report."""
    notes = make_notes([(1, 10.0, "only")])
    ids = np.array(["only"])
    emb = embeddings_for(ids)
    index = build_window_index(make_cohort([(1, 20)]), notes, ids, WINDOW)
    blocks, _, _, _, _ = aggregate_blocks(index, emb, [0], "rich")
    _, _, delta = np.split(blocks[0], 3)
    assert np.allclose(delta, 0.0)


def test_uncovered_rows_get_zero_blocks():
    notes = make_notes([(1, 100.0, "a")])
    ids = np.array(["a"])
    index = build_window_index(make_cohort([(1, 5)]), notes, ids, WINDOW)
    blocks, has, age, n, span = aggregate_blocks(index, embeddings_for(ids),
                                                 [0], "rich")
    assert has[0] == 0.0 and n[0] == 0
    assert np.allclose(blocks[0], 0.0) and age[0] == 0.0 and span[0] == 0.0


def test_aggregation_batches_independently():
    """Aggregating a batch must equal aggregating each row alone."""
    notes = make_notes([(1, 5.0, "a"), (1, 9.0, "b"), (2, 3.0, "c")])
    ids = np.array(["a", "b", "c"])
    emb = embeddings_for(ids)
    cohort = make_cohort([(1, 10), (1, 20), (2, 10), (2, 90)])
    index = build_window_index(cohort, notes, ids, WINDOW)
    together, _, _, _, _ = aggregate_blocks(index, emb, [0, 1, 2, 3], "rich")
    for row in range(4):
        alone, _, _, _, _ = aggregate_blocks(index, emb, [row], "rich")
        assert np.allclose(together[row], alone[0], atol=1e-6)


# --- the presence-leakage guard survives windowing --------------------------

def test_head_gives_every_uncovered_row_the_same_risk():
    torch.manual_seed(0)
    head = RichNoteHead(n_blocks=3, n_scalars=1).eval()
    blocks = torch.randn(4, 3 * EMBED_DIM)      # different junk per row
    has = torch.zeros(4)
    scalars = torch.randn(4, 1)
    with torch.no_grad():
        logits, _ = head(blocks, has, scalars)
    assert torch.allclose(logits, logits[0].expand(4), atol=1e-6)


def test_head_does_vary_when_a_note_is_present():
    torch.manual_seed(0)
    head = RichNoteHead(n_blocks=3, n_scalars=1).eval()
    blocks = torch.randn(4, 3 * EMBED_DIM)
    with torch.no_grad():
        logits, _ = head(blocks, torch.ones(4), torch.zeros(4, 1))
    assert not torch.allclose(logits, logits[0].expand(4), atol=1e-6)


def test_count_only_auprc_is_the_suspicion_floor():
    y = np.array([0, 0, 1, 1, 0, 1])
    counts = np.array([0.0, 1.0, 5.0, 6.0, 0.0, 4.0])
    assert count_only_auprc(y, counts) > y.mean()


# --- the agent end to end ---------------------------------------------------

def synthetic_split(n_stays=14, n_hours=30, seed=0):
    rng = np.random.default_rng(seed)
    frames, notes, ids = [], [], []
    for stay in range(1, n_stays + 1):
        frame = pd.DataFrame({"stay_id": stay, "hour": np.arange(n_hours)})
        frame["label_24h"] = (rng.random(n_hours) < 0.2).astype(int)
        frames.append(frame)
        for k in range(rng.integers(0, 4)):
            note_id = f"s{stay}n{k}"
            notes.append((stay, float(rng.integers(1, n_hours)), note_id))
            ids.append(note_id)
    return (pd.concat(frames, ignore_index=True), make_notes(notes),
            np.array(ids))


def fitted_agent(tmp_path, aggregation="rich", **kwargs):
    cohort, notes, ids = synthetic_split()
    emb = embeddings_for(ids, seed=1)
    tmp_path.mkdir(parents=True, exist_ok=True)
    npz = tmp_path / "emb.npz"
    np.savez_compressed(npz, note_ids=ids.astype(str), embeddings=emb)
    index_path = tmp_path / "index.parquet"
    notes.assign(note_type="RR").to_parquet(index_path)

    agent = RichNotesAgent(aggregation=aggregation, embed_cache=str(npz),
                           note_index=str(index_path), max_epochs=2,
                           verbose=False, **kwargs)
    train = cohort[cohort.stay_id <= 10]
    val = cohort[cohort.stay_id > 10]
    return agent.fit(train, val), cohort, train, val


def test_agent_emits_key_sorted_rows_with_a_latent(tmp_path):
    agent, cohort, _, val = fitted_agent(tmp_path)
    out = agent.emit(val)
    expected = val.sort_values(["stay_id", "hour"])[["stay_id", "hour"]]
    assert out.keys.stay_id.tolist() == expected.stay_id.tolist()
    assert out.keys.hour.tolist() == expected.hour.tolist()
    assert out.latent.shape == (len(val), agent.latent_dim)
    assert np.all((out.risk >= 0) & (out.risk <= 1))


def test_recent_mode_uses_one_block_and_rich_uses_three(tmp_path):
    recent, _, _, _ = fitted_agent(tmp_path / "a", aggregation="recent")
    rich, _, _, _ = fitted_agent(tmp_path / "b", aggregation="rich")
    assert recent.n_blocks == 1 and rich.n_blocks == 3


def test_count_features_are_off_by_default(tmp_path):
    """Note count is a clinician-suspicion proxy -- CLAUDE.md rule #4."""
    agent, _, _, _ = fitted_agent(tmp_path)
    assert agent.use_count_features is False
    assert agent.n_scalars == 1


def test_enabling_count_features_widens_the_scalar_block(tmp_path):
    agent, _, _, _ = fitted_agent(tmp_path, use_count_features=True)
    assert agent.n_scalars == 3


def test_agent_is_deterministic(tmp_path):
    a, _, _, val = fitted_agent(tmp_path / "a")
    b, _, _, val_b = fitted_agent(tmp_path / "b")
    assert np.allclose(a.emit(val).risk, b.emit(val_b).risk)


def test_shrinking_the_window_can_only_remove_notes(tmp_path):
    cohort, notes, ids = synthetic_split()
    wide = build_window_index(cohort, notes, ids, 72)
    narrow = build_window_index(cohort, notes, ids, 12)
    assert (narrow.counts <= wide.counts).all()
    assert narrow.counts.sum() < wide.counts.sum()


# --- the contamination audit's helpers -------------------------------------

def load_audit():
    """Import scripts/audit_note_contamination.py for its helper functions."""
    import importlib.util
    from pathlib import Path
    path = (Path(__file__).resolve().parent.parent / "scripts"
            / "audit_note_contamination.py")
    spec = importlib.util.spec_from_file_location("audit_note_contamination", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_cohort():
    """Two stays: one deteriorates at hour 20, one never does."""
    return pd.DataFrame({
        "stay_id": [1] * 3 + [2] * 3,
        "hour": [0, 10, 19] * 2,
        "event_h": [20.0] * 3 + [np.nan] * 3,
        "event_type": ["vent"] * 3 + [None] * 3,
    })


def test_blackout_censors_only_notes_just_before_an_event():
    audit = load_audit()
    notes = make_notes([
        (1, 5.0, "far"),        # 15h before the event -> kept
        (1, 16.0, "near"),      # 4h before  -> censored at blackout=6
        (1, 25.0, "after"),     # after the event -> kept (lead < 0)
        (2, 16.0, "control"),   # event-free stay -> always kept
    ])
    kept, dropped = audit.censored_index(notes, audit_cohort(), blackout=6)
    assert dropped == 1
    assert set(kept.note_id) == {"far", "after", "control"}


def test_blackout_of_zero_removes_almost_nothing():
    audit = load_audit()
    notes = make_notes([(1, 16.0, "near"), (1, 20.0, "exact")])
    kept, dropped = audit.censored_index(notes, audit_cohort(), blackout=0)
    assert dropped == 1                    # only the note filed AT the event
    assert set(kept.note_id) == {"near"}


def test_widening_the_blackout_only_removes_more():
    audit = load_audit()
    notes = make_notes([(1, h, f"n{h}") for h in (2.0, 8.0, 14.0, 18.0)])
    counts = [audit.censored_index(notes, audit_cohort(), blackout=b)[1]
              for b in (2, 6, 12, 24)]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_lead_is_measured_from_the_event_not_the_prediction_hour():
    audit = load_audit()
    notes = make_notes([(1, 5.0, "a"), (2, 5.0, "b")])
    frame = audit.note_lead_frame(audit_cohort(), notes)
    lead = frame.set_index("note_id")["lead"]
    assert lead["a"] == pytest.approx(15.0)      # event 20 - note 5
    assert np.isnan(lead["b"])                   # event-free stay


def test_window_index_subset_keeps_rows_consistent():
    notes = make_notes([(1, 5.0, "a"), (2, 5.0, "b"), (3, 5.0, "c")])
    ids = np.array(["a", "b", "c"])
    cohort = make_cohort([(1, 10), (2, 10), (3, 10)])
    index = build_window_index(cohort, notes, ids, WINDOW)
    keep = np.array([True, False, True])
    sub = index.subset(keep)
    assert len(sub) == 2
    assert sub.keys.stay_id.tolist() == [1, 3]
    assert sub.positions.tolist() == [0, 2]
