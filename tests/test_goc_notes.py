"""Tests for the goals-of-care note extraction (T16) -- audit-only posture.

The load-bearing guarantees: the model pipeline never touches the note data
(discharge summaries narrate the predicted outcome), a verdict is never
guessed from a reply that did not parse, and a cache written under different
instructions or a different model serves nothing.
"""

import json
from pathlib import Path

import pytest

from goc_notes import (
    GocCache,
    UNSURE,
    adjudicate,
    build_goc_prompt,
    extract_snippets,
    goc_fingerprint,
    parse_verdict,
)

REPO = Path(__file__).resolve().parents[1]


# --- snippet gate ------------------------------------------------------------

def test_no_goc_vocabulary_means_no_snippets():
    text = "Patient admitted with pneumonia. Treated with antibiotics. " * 50
    assert extract_snippets(text) == []


def test_nearby_mentions_merge_into_one_snippet():
    text = ("x" * 300 + " code status was discussed and the patient was made "
            "DNR/DNI after a family meeting. " + "y" * 300)
    snips = extract_snippets(text, window=240)
    assert len(snips) == 1
    assert "DNR" in snips[0] and "family meeting" in snips[0]


def test_distant_mentions_stay_separate_and_cap_applies():
    passage = "the patient remained full code status here. "
    filler = "z" * 800
    text = filler.join(passage for _ in range(12))
    snips = extract_snippets(text, window=100, cap=8)
    assert len(snips) == 8
    assert all("code status" in s for s in snips)


def test_empty_text_is_safe():
    assert extract_snippets(None) == []
    assert extract_snippets("") == []


# --- verdict parsing ---------------------------------------------------------

def test_parse_valid_verdict():
    v = parse_verdict('{"any_limitation": "yes", "cmo": "no", '
                      '"evidence_quote": "made DNR/DNI"}')
    assert v == {"any_limitation": "yes", "cmo": "no",
                 "evidence_quote": "made DNR/DNI"}


def test_parse_verdict_wrapped_in_prose():
    reply = ('Here is my verdict: {"any_limitation": "no", "cmo": "no", '
             '"evidence_quote": ""} I hope this helps.')
    assert parse_verdict(reply)["any_limitation"] == "no"


@pytest.mark.parametrize("bad", [
    "", None, "no json here", '{"any_limitation": "maybe", "cmo": "no"}',
    '{"cmo": "no"}', '{"any_limitation": "yes"', '{broken json}',
])
def test_parse_rejects_malformed_or_out_of_enum(bad):
    assert parse_verdict(bad) is None


def test_adjudicate_falls_back_to_unsure_after_retries():
    class NeverJSON:
        calls = 0
        def generate(self, prompt):
            NeverJSON.calls += 1
            return "I cannot answer that."
    v = adjudicate(NeverJSON(), ["some snippet"], retries=1)
    assert v["any_limitation"] == "unsure" and v.get("parse_failed")
    assert NeverJSON.calls == 2                      # one retry, no more


def test_prompt_contains_every_snippet_and_only_snippets():
    snips = ["made DNR after discussion", "transitioned to comfort measures"]
    prompt = build_goc_prompt(snips)
    for s in snips:
        assert s in prompt
    assert "Excerpt 2" in prompt


# --- cache provenance --------------------------------------------------------

def test_cache_roundtrip_and_stale_fingerprint_skipped(tmp_path):
    path = tmp_path / "verdicts.jsonl"
    fp = goc_fingerprint("model-a")
    cache = GocCache(path, fp)
    cache.add(101, {"any_limitation": "yes", "cmo": "no",
                    "evidence_quote": "q"}, n_snippets=3)
    assert 101 in GocCache(path, fp)
    # same file opened under another model's fingerprint serves nothing
    other = GocCache(path, goc_fingerprint("model-b"))
    assert 101 not in other and len(other) == 0


def test_fingerprint_moves_with_model_and_prompt():
    a = goc_fingerprint("gemma4:12b")
    assert a != goc_fingerprint("llama3.1:8b")
    assert a != goc_fingerprint("gemma4:12b", system="different rules")


# --- the audit-only guard ----------------------------------------------------

def test_model_pipeline_never_references_goc_notes():
    """Discharge-note verdicts partition audit tables; they must never leak
    into anything that produces model inputs."""
    sources = (list((REPO / "src" / "agents").glob("*.py"))
               + list((REPO / "src" / "orchestrator").glob("*.py"))
               + [REPO / "src" / "modality_features.py"])
    assert len(sources) > 5, "guard is scanning the wrong tree"
    for path in sources:
        code = path.read_text()
        assert "goc_notes" not in code, f"{path} touches audit-only note data"
        assert "mimiciv_note.discharge" not in code, path
