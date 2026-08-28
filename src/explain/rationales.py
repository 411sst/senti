"""Turning the orchestrator's structured drivers into one readable sentence.

The orchestrator is not a black box: at every patient-hour it already exposes
the NEWS2 agent's seven sub-scores, each agent's calibrated risk, and the
attention weight it put on each agent. This module renders those numbers into a
prompt and asks a **local** ollama model (no external API -- PhysioNet DUA) for
a single sentence naming the drivers. A second prompt variant
(:data:`MENU_SYSTEM_PROMPT`, :func:`build_menu_prompt`) adds an enumerated
candidate menu and worked examples; results_fusion.md T17 records why it is a
deployment guardrail rather than the reported bench.

Note what the prompt does **not** contain: the orchestrator's own fused risk,
and whether the alert fired. That omission is the leakage control for the
simulatability metric. If the model were told "risk 0.31, alert fired" it could
write "high risk" and a surrogate would recover the decision from that phrase
alone -- scoring high on simulatability while explaining nothing. Withholding it
means the only route from rationale text back to the decision is through the
*drivers the sentence names*, which is the thing we want to measure.

Two anti-gaming baselines live here too, and they matter as much as the real
arm. A metric that cannot tell a faithful explanation from a fluent-but-empty
one is not measuring faithfulness:

* :func:`shuffle_rationales` -- real sentences, re-assigned to the wrong rows.
  Identical fluency, identical vocabulary, identical driver density; only the
  correspondence to the patient-hour is broken.
* :data:`GENERIC_RATIONALE` -- one fixed, clinically plausible sentence for
  every row. It deliberately *does* name drivers, so it is not trivially zeroed
  out by the extractor; it just names the same ones regardless of the patient.
"""

import json
from pathlib import Path

from explain.drivers import (AGENT_LABELS, COMPONENT_LABELS, named_agents,
                             named_components, top_attention_agent)
from text_arm.llm_orchestrator import DEFAULT_MODEL, OllamaRunner

SYSTEM_PROMPT = (
    "You write one-sentence justifications for an ICU early-warning model.\n"
    "For a single patient-hour you are given: the NEWS2 score and its seven "
    "sub-scores, each monitoring agent's calibrated risk, how much the fusion "
    "model attended to each agent, and the 4-hour trend in the vital signs.\n"
    "Write ONE sentence of at most 35 words naming the 2-3 specific drivers "
    "that most account for the model's assessment at this hour.\n"
    "Rules:\n"
    "- Use ONLY the numbers given below. Never invent a laboratory value, a "
    "diagnosis, a symptom or a finding that is not written there. The labs and "
    "notes agents report a risk number only -- refer to them by name and never "
    "state what they contain.\n"
    "- Name concrete parameters and their direction, e.g. 'rising heart rate', "
    "'falling SpO2', 'respiratory rate sub-score 3'.\n"
    "- Say which agent the fusion leaned on.\n"
    "- Quote the NEWS2 score when it is informative, e.g. 'NEWS2 still low at 4'.\n"
    "- Do NOT state an overall risk level, do NOT say whether to raise an "
    "alert, do NOT recommend any action.\n"
    "- No preamble and no bullet points. Output only the sentence."
)

# The "menu-fewshot" variant: the same rules, plus an enumerated candidate menu
# the sentence must draw from, and worked examples ahead of the patient-hour.
# Built for readers that fail the hallucination bar by narrating normal-range
# trends (a zero sub-score is a parameter that did not drive anything).
MENU_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "\n- Every patient-hour comes with a CANDIDATE MENU. Name drivers ONLY from "
    "that menu: the agents listed and the NEWS2 parameters listed. Never "
    "mention a vital sign that is not on the menu, even if its 4-hour trend is "
    "shown -- a parameter with a zero sub-score did not drive the assessment.\n"
    "- Worked examples precede the patient-hour. Write the final sentence in "
    "the same style, for the LAST patient-hour only."
)

# Plausible, fluent, names real drivers -- and completely independent of the
# patient. This is the adversary the metrics have to defeat.
GENERIC_RATIONALE = (
    "Vital signs are trending abnormally with a rising heart rate and falling "
    "blood pressure, and the NEWS2 score reflects growing physiological "
    "instability across the monitored parameters."
)

# Words that would let a surrogate recover the decision without reading any
# driver. Stripped in the scrubbed variant of the simulatability metric.
LEAK_WORDS = (
    "risk", "risky", "high", "low", "moderate", "severe", "critical", "alert",
    "alarm", "warning", "deteriorat", "concern", "worrying", "unstable",
    "stable", "urgent", "escalat", "danger",
)


def _fmt(value, digits=0):
    return "unmeasured" if value is None else f"{value:.{digits}f}"


def _trend_phrase(name, trend):
    """e.g. ``heart rate rising (88->104)`` or ``SpO2 stable at 97``.

    "unknown" is spelled out as *no 4h history* rather than printed as the
    trend word: written bare it reads as though the vital itself were
    unmeasured, and the model then describes a value it can plainly see as
    missing.
    """
    if trend["now"] is None:
        return f"{name} unmeasured"
    if trend["word"] == "unknown":
        return f"{name} {_fmt(trend['now'])} (no 4h history)"
    if trend["word"] in ("rising", "falling"):
        return (f"{name} {trend['word']} "
                f"({_fmt(trend['before'])}->{_fmt(trend['now'])})")
    return f"{name} stable at {_fmt(trend['now'])}"


def _packet(context):
    """The evidence packet: the five lines of numbers every variant shows."""
    subs = context["subscores"]
    active = [f"{COMPONENT_LABELS[c]} {int(v)}"
              for c, v in sorted(subs.items(), key=lambda kv: -kv[1]) if v > 0]
    breakdown = ", ".join(active) if active else "all seven sub-scores zero"

    risks = "  ".join(f"{AGENT_LABELS.get(a, a)} {v:.3f}"
                      for a, v in context["agent_risks"].items())
    attention = ", ".join(f"{AGENT_LABELS.get(a, a)} {v:.2f}"
                          for a, v in sorted(context["attention"].items(),
                                             key=lambda kv: -kv[1]))
    trends = ", ".join(
        _trend_phrase(COMPONENT_LABELS.get(v, v), context["trends"][v])
        for v in ("heart_rate", "spo2", "resp_rate", "sbp", "temperature")
    )

    return "\n".join([
        f"NEWS2 score: {context['news2_total']:.0f} out of 20.",
        f"NEWS2 sub-scores above zero: {breakdown}.",
        f"Agent risks (calibrated P of deterioration in 24h): {risks}.",
        f"Fusion attention over agents: {attention}.",
        f"Vital-sign trend over the last 4 hours: {trends}.",
    ])


def build_prompt(context):
    """Render one structured-driver record as a prompt. Pure function.

    Being pure and side-effect free is what lets ``tests/test_explain.py``
    assert directly that the fused risk and the alert flag never appear in the
    text the model sees.
    """
    return _packet(context) + "\n\nOne sentence naming the drivers:"


def build_menu(context, n_agents=3):
    """The enumerated candidate menu: top-attention agents, non-zero sub-scores."""
    ranked = sorted(context["attention"].items(), key=lambda kv: -kv[1])[:n_agents]
    agents = ", ".join(f"{AGENT_LABELS.get(a, a)} (attention {w:.2f})"
                       for a, w in ranked)
    params = [f"{COMPONENT_LABELS[c]} (sub-score {int(v)})"
              for c, v in sorted(context["subscores"].items(),
                                 key=lambda kv: -kv[1]) if v > 0]
    params = (", ".join(params) if params
              else "none -- every sub-score is zero, so name no vital sign")
    return "\n".join([
        f"CANDIDATE MENU -- agents the fusion leaned on: {agents}.",
        f"CANDIDATE MENU -- NEWS2 parameters above zero: {params}.",
    ])


def build_menu_prompt(context, examples=()):
    """The menu-fewshot variant of :func:`build_prompt`. Pure function.

    ``examples`` are ``(context, sentence)`` pairs from OTHER patient-hours,
    rendered as worked examples ahead of the target. Same leakage guarantee as
    the plain prompt: no fused risk and no alert flag, for the target or for
    any example.
    """
    blocks = []
    for k, (ex_context, ex_text) in enumerate(examples, start=1):
        blocks.append("\n".join([f"EXAMPLE {k}", _packet(ex_context),
                                 build_menu(ex_context), f"Sentence: {ex_text}"]))
    blocks.append("\n".join(["NOW THIS PATIENT-HOUR", _packet(context),
                             build_menu(context), "",
                             "One sentence naming only drivers from the menu:"]))
    return "\n\n".join(blocks)


def few_shot_pool(contexts, texts, agent_names, fired, k_fired=2, k_quiet=2):
    """Exemplar rows for the few-shot variant, drawn from another reader's cards.

    An exemplar must be a non-empty sentence that names the top-attention agent
    and names no zero-scored NEWS2 parameter -- the behaviour the variant is
    trying to teach. Fired and quiet rows are interleaved so the examples do
    not teach "always sound alarmed". Order is deterministic (input order).
    """
    good_fired, good_quiet = [], []
    for context, text, is_fired in zip(contexts, texts, fired):
        if not text:
            continue
        if top_attention_agent(context) not in named_agents(text, agent_names):
            continue
        zero = {c for c, v in context["subscores"].items() if v <= 0}
        if named_components(text) & zero:
            continue
        (good_fired if is_fired else good_quiet).append((context, text))
    pool = []
    for a, b in zip(good_fired[:k_fired], good_quiet[:k_quiet]):
        pool.extend([a, b])
    return pool


def examples_for(pool, context, k=3):
    """The first ``k`` pool exemplars that are not this patient-hour itself."""
    key = (context["stay_id"], context["hour"])
    return [(c, t) for c, t in pool
            if (c["stay_id"], c["hour"]) != key][:k]


def clean_reply(text):
    """Collapse the model's reply to a single sentence of prose.

    Llama sometimes prefixes "Here is the sentence:" or wraps in quotes despite
    being told not to; stripping that here keeps the *scored* artifact free of
    boilerplate that would otherwise be identical across every row and so leak
    nothing but noise into the surrogate.
    """
    if not text:
        return ""
    line = " ".join(str(text).split())
    for prefix in ("Here is the sentence:", "Here is a sentence:",
                   "Sentence:", "Rationale:", "Here's the sentence:"):
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix):].strip()
    return line.strip().strip('"').strip()


def prompt_fingerprint(system=SYSTEM_PROMPT):
    """Short hash of the instructions a cached sentence was generated under."""
    import hashlib
    return hashlib.sha256(system.encode()).hexdigest()[:12]


class RationaleCache:
    """Append-only JSONL cache of generated sentences, keyed ``(stay_id, hour)``.

    Written after every generation, so an interrupted run resumes rather than
    paying for the whole subset again.

    Each record also stores a fingerprint of the system prompt it was generated
    under, and entries with a different fingerprint are ignored on load. Editing
    the instructions therefore regenerates automatically instead of silently
    scoring sentences written to older rules -- a stale cache has already caused
    one wrong result in this project.
    """

    def __init__(self, path, fingerprint=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint or prompt_fingerprint()
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
                        continue              # tolerate a partial final line
                    if rec.get("prompt_fingerprint") != self.fingerprint:
                        stale += 1
                        continue
                    self.entries[(int(rec["stay_id"]), int(rec["hour"]))] = rec
        if stale:
            print(f"[rationale] ignoring {stale:,} cached sentences written "
                  f"under different instructions")

    def __contains__(self, key):
        return key in self.entries

    def __len__(self):
        return len(self.entries)

    def add(self, stay_id, hour, text, prompt):
        rec = {"stay_id": int(stay_id), "hour": int(hour), "text": text,
               "prompt_chars": len(prompt),
               "prompt_fingerprint": self.fingerprint}
        self.entries[(int(stay_id), int(hour))] = rec
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def texts_for(self, contexts):
        """Sentences in the order of ``contexts``; ``""`` where absent."""
        return [self.entries.get((c["stay_id"], c["hour"]), {}).get("text", "")
                for c in contexts]


def generate_rationales(contexts, cache_path, model=DEFAULT_MODEL, runner=None,
                        progress_every=10, prompt_fn=build_prompt,
                        system=SYSTEM_PROMPT, fingerprint=None):
    """Generate (or resume) one rationale per context. Returns the cache.

    ``prompt_fn``/``system``/``fingerprint`` select a prompt variant; the
    defaults are the plain prompt and its fingerprint, unchanged.
    """
    cache = RationaleCache(cache_path, fingerprint or prompt_fingerprint(system))
    runner = runner or OllamaRunner(model=model, system=system, num_predict=90)

    todo = [c for c in contexts if (c["stay_id"], c["hour"]) not in cache]
    print(f"[rationale] {len(cache):,} cached, {len(todo):,} to generate "
          f"(model={model})")

    empty = 0
    for n, context in enumerate(todo, start=1):
        prompt = prompt_fn(context)
        text = clean_reply(runner.generate(prompt))
        if not text:
            empty += 1
        cache.add(context["stay_id"], context["hour"], text, prompt)
        if progress_every and n % progress_every == 0:
            print(f"  {n:,}/{len(todo):,} ({empty} empty)", flush=True)
    if todo:
        print(f"[rationale] done: {len(todo):,} generated, {empty} empty")
    return cache


def shuffle_rationales(texts, seed=42):
    """Real sentences re-assigned to the wrong rows (a derangement where possible).

    This is the RANDOM control. Using genuine sentences rather than synthetic
    noise is the point: fluency, vocabulary and driver density are held fixed,
    so anything the metrics still detect is the row-level correspondence and
    nothing else.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(texts)
    if n < 2:
        return list(texts)
    for _ in range(100):
        order = rng.permutation(n)
        if not (order == np.arange(n)).any():
            return [texts[i] for i in order]
    # Fall back to a rotation, which is a derangement by construction.
    return [texts[(i + 1) % n] for i in range(n)]


def generic_rationales(n):
    """The GENERIC control: one fixed plausible sentence, repeated."""
    return [GENERIC_RATIONALE] * n


def scrub_leak_words(text):
    """Remove numerals and risk-level vocabulary from a rationale.

    Used for the leakage-controlled variant of simulatability: if a surrogate
    can still recover the decision after this, it is reading the *named
    drivers*, not a tell-word like "high" or a quoted score.
    """
    import re
    out = re.sub(r"\d+(?:\.\d+)?", " ", str(text))
    for word in LEAK_WORDS:
        out = re.sub(r"\b" + word + r"\w*", " ", out, flags=re.IGNORECASE)
    return " ".join(out.split())
