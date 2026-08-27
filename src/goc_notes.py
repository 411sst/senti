"""Goals-of-care extraction from discharge summaries -- AUDIT ONLY (T16).

This module exists to fill the gap T11 measured: 1,496 of the cohort's
death-first stays have NO charted code status, so the charted-field audit
cannot say whether their outcomes were already decided. MIMIC-IV-Note
discharge summaries narrate those decisions; a local LLM reads targeted
excerpts and reports whether a limitation of care was documented.

**Never a model feature.** Discharge summaries are written after the outcome
and narrate the event being predicted; like charted code status, everything
here is used only to PARTITION evaluation tables (decided vs undecided), never
as input to any agent or fusion model. A guard test pins the model pipeline
(``src/agents/``, ``src/orchestrator/``, ``src/modality_features.py``) to zero
references to this data. Notes never leave the machine -- the LLM is a local
ollama model (PhysioNet DUA).

**Structured-first.** The LLM only ever sees short regex-gated excerpts around
goals-of-care vocabulary; a note with no such vocabulary is verdicted
"no documentation" without any model call. The 1,029 charted death stays give
a free validation set: extractor accuracy is measurable against the charted
field before the extractor is trusted on the uncharted ones.
"""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

GOC_NOTES_PATH = "data/cache/goc_notes.parquet"

# Vocabulary that marks a goals-of-care passage. Kept deliberately wide -- the
# LLM adjudicates; this gate only bounds what it must read.
GOC_PATTERNS = (
    r"\bDNR\b|\bDNAR\b|\bDNI\b|do not resuscitate|do not intubate|"
    r"comfort measures|\bCMO\b|goals of care|code status|palliative|hospice|"
    r"family meeting|withdraw(?:al)? of (?:care|life support)|"
    r"terminal(?:ly)? extubat"
)
_GOC_RE = re.compile(GOC_PATTERNS, re.I)
PATTERNS_VERSION = "goc-v1"

SYSTEM_PROMPT = (
    "You audit ICU discharge summaries for documented code-status decisions.\n"
    "You are given short excerpts from ONE patient's discharge summary.\n"
    "Decide from the excerpts alone:\n"
    "- any_limitation: was a limitation of care (DNR, DNAR, DNI, comfort "
    "measures only, withdrawal of life support, transition to hospice) "
    "documented as DECIDED or IN EFFECT for this patient? Answer \"yes\" only "
    "if the text states the decision was made or in effect. A discussion "
    "alone, or a patient who remained full code, is \"no\".\n"
    "- cmo: was the patient transitioned to comfort measures only, or was "
    "life support withdrawn? Same standard.\n"
    "- evidence_quote: the shortest exact quote (at most 25 words) that "
    "supports your answers, or \"\" if none.\n"
    "If the excerpts are ambiguous, answer \"unsure\".\n"
    "Reply with ONLY a JSON object, no other text:\n"
    '{"any_limitation": "yes|no|unsure", "cmo": "yes|no|unsure", '
    '"evidence_quote": "..."}'
)

_JSON_RE = re.compile(r"\{[^{}]*\}", re.S)
_VERDICTS = ("yes", "no", "unsure")


def extract_snippets(text, window=240, cap=8):
    """Regex-gated excerpts of ``text`` around goals-of-care vocabulary.

    Overlapping windows are merged so one passage is one excerpt, and at most
    ``cap`` excerpts survive (discharge summaries can mention code status
    dozens of times; the first ``cap`` merged passages carry the decision).
    Returns ``[]`` when the vocabulary never appears -- the caller then
    records "no documentation" without an LLM call.
    """
    if not text:
        return []
    t = str(text)
    spans = []
    for m in _GOC_RE.finditer(t):
        lo, hi = max(0, m.start() - window), min(len(t), m.end() + window)
        if spans and lo <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
        else:
            spans.append((lo, hi))
    return [t[lo:hi].strip() for lo, hi in spans[:cap]]


def build_goc_prompt(snippets):
    """Numbered excerpts + the answer cue. Pure function (testable)."""
    lines = [f"Excerpt {i}:\n{s}" for i, s in enumerate(snippets, start=1)]
    return "\n\n".join(lines) + "\n\nJSON verdict:"


def parse_verdict(reply):
    """First flat JSON object in ``reply`` with valid verdict enums, else None.

    The schema is flat, so ``{[^{}]*}`` finds it even when the model wraps it
    in prose. Anything malformed -- bad JSON, missing keys, out-of-enum values
    -- is None; the caller retries once and then records "unsure". A verdict
    is never guessed from a reply that did not parse.
    """
    if not reply:
        return None
    m = _JSON_RE.search(str(reply))
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    al = str(obj.get("any_limitation", "")).strip().lower()
    cmo = str(obj.get("cmo", "")).strip().lower()
    if al not in _VERDICTS or cmo not in _VERDICTS:
        return None
    return {"any_limitation": al, "cmo": cmo,
            "evidence_quote": str(obj.get("evidence_quote", ""))[:400]}


UNSURE = {"any_limitation": "unsure", "cmo": "unsure", "evidence_quote": "",
          "parse_failed": True}


def adjudicate(runner, snippets, retries=1):
    """One stay's excerpts -> one verdict dict, retrying a failed parse once."""
    prompt = build_goc_prompt(snippets)
    for _ in range(1 + retries):
        verdict = parse_verdict(runner.generate(prompt))
        if verdict is not None:
            return verdict
    return dict(UNSURE)


def goc_fingerprint(model, system=SYSTEM_PROMPT, patterns=PATTERNS_VERSION):
    """Cache provenance: instructions + gate vocabulary + model."""
    blob = "\n".join([system, patterns, str(model)])
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


class GocCache:
    """Append-only JSONL of per-stay verdicts, keyed ``stay_id``.

    One file per model (the fingerprint also embeds the model, so a cache file
    reused under the wrong model ignores every entry rather than serving stale
    verdicts -- the RationaleCache lesson). Written after every generation, so
    an interrupted run over 1,496 stays resumes instead of restarting.
    """

    def __init__(self, path, fingerprint):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.entries = {}
        stale = 0
        if self.path.exists():
            with self.path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue          # tolerate a partial final line
                    if rec.get("fingerprint") != self.fingerprint:
                        stale += 1
                        continue
                    self.entries[int(rec["stay_id"])] = rec
        if stale:
            print(f"[goc] ignoring {stale:,} cached verdicts written under a "
                  f"different (prompt, patterns, model)")

    def __contains__(self, stay_id):
        return int(stay_id) in self.entries

    def __len__(self):
        return len(self.entries)

    def get(self, stay_id):
        return self.entries.get(int(stay_id))

    def add(self, stay_id, verdict, n_snippets):
        rec = {"stay_id": int(stay_id), "n_snippets": int(n_snippets),
               "fingerprint": self.fingerprint, **verdict}
        self.entries[int(stay_id)] = rec
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")


def death_stays(cohort_path="data/cache/mimic_cohort_hourly.parquet",
                strict=True):
    """Death-first stays with their charted-code-status audit columns.

    One row per death-first stay: ``split``, ``has_code_status`` (the
    charted/uncharted partition -- charted stays are the free validation set),
    ``limit_before_event`` / ``cmo_before_event`` (the charted truth the
    extractor is validated against).
    """
    from code_status import attach_to_stays, load_code_status, stay_timeline

    df = pd.read_parquet(cohort_path,
                         columns=["stay_id", "patient_id", "split", "hour",
                                  "event_h", "event_type"])
    stays = df.groupby("stay_id", sort=False).agg(
        patient_id=("patient_id", "first"), split=("split", "first"),
        event_h=("event_h", "first"), last_hour=("hour", "max"),
        event_type=("event_type", "first")).reset_index()
    att = attach_to_stays(stays, stay_timeline(load_code_status(strict=strict)))
    deaths = att[att["event_type"] == "death"].reset_index(drop=True)
    # Death well past the stay's last grid hour = the patient left the ICU and
    # died elsewhere in the hospital. The charted code-status extract (ICU
    # chartevents) is structurally BLIND to decisions made on the ward after
    # transfer, so chart-vs-note comparisons must stratify on this: the chart
    # is authoritative only for in-ICU deaths. Empirically 71% of noted,
    # charted death stays are post-ICU -- and 87% of note-says-limited /
    # chart-says-no disagreements sit there, 85% of them CMO transitions.
    deaths["post_icu_death"] = (deaths["event_h"] - deaths["last_hour"]) > 2.0
    return deaths[["stay_id", "patient_id", "split", "event_h",
                   "post_icu_death", "has_code_status", "ever_limited",
                   "limit_before_event", "cmo_before_event"]]
