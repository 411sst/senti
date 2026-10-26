"""The text-passing orchestrator: one local LLM fusing three agents' sentences.

This is the ablation arm. Structurally it mirrors the latent orchestrator --
same agents, same cohort, same splits, same trajectory window, same
isotonic-on-val calibration -- but the agents communicate in **natural language**
instead of latent vectors, and the fusion is done by a single local
Llama-3.1-8B rather than a learned attention + TCN.

Leakage: for a prediction at hour ``t`` the prompt contains agent summaries for
hours ``t - W + 1 .. t`` only. Each of those summaries was itself built from
strictly-before-``t`` features, so nothing after the prediction point can reach
the model. :func:`build_prompt` is a pure function of the window, which is what
``tests/test_text_arm.py`` exercises for causality: perturb the summaries after
hour ``k`` and the prompts for hours before ``k`` must be byte-identical.

Compute: every fused risk is generated once and appended to a JSONL cache as it
completes, so an interrupted run resumes instead of restarting. Nothing is
called twice.
"""

import json
import re
import subprocess
from pathlib import Path

import numpy as np

# Local only -- no external API, per the PhysioNet data use agreement.
DEFAULT_MODEL = "llama3.1:8b"
# How many hours of history the LLM sees. Mirrors "the fusion sees the sequence
# of agent signals over time" while keeping the prompt short enough to be fast.
WINDOW_HOURS = 4

SYSTEM_PROMPT = (
    "You are a clinical deterioration risk assessor in an ICU. Three monitoring "
    "agents report on a patient each hour: a vitals agent, a labs agent, and a "
    "NEWS2 score agent. Given their reports over the last few hours, estimate "
    "the probability that this patient will deteriorate within the next 24 hours "
    "(deterioration = invasive ventilation, vasopressors, or death).\n"
    "Base rate: about 5 in 100 patient-hours lead to deterioration.\n"
    "Reply with ONLY an integer from 0 to 100 -- the probability in percent. "
    "No words, no explanation, no percent sign."
)

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def build_prompt(window):
    """Build the user prompt from a list of per-hour summary dicts.

    ``window`` is ordered oldest-to-newest and contains only hours ``<= t``;
    the last entry is the prediction hour. Pure function -- no I/O, no state --
    so it can be tested for causality directly.
    """
    lines = []
    for i, row in enumerate(window):
        age = len(window) - 1 - i
        when = "now" if age == 0 else f"{age}h ago"
        lines.append(f"[{when}]")
        lines.append(f"  {row['vitals_text']}")
        lines.append(f"  {row['labs_text']}")
        lines.append(f"  {row['news2_text']}")
    lines.append("")
    lines.append("Probability (0-100) of deterioration within 24 hours:")
    return "\n".join(lines)


def parse_risk(text):
    """Extract a 0-1 probability from the model's reply.

    The prompt asks for an integer percent, but models sometimes answer with a
    decimal probability ("0.9"). A reply carrying a decimal point and lying in
    [0, 1] is therefore read as a probability, not a percent -- the old code
    read "0.9" as 0.9% (0.009), silently deflating every such risk 100-fold.
    Returns ``nan`` when the model does not produce a usable number, so those
    rows can be counted and reported rather than silently imputed.
    """
    if not text:
        return float("nan")
    match = _NUMBER.search(text.strip())
    if not match:
        return float("nan")
    raw = match.group()
    value = float(raw)
    if "." in raw and 0.0 <= value <= 1.0:
        return value                     # decimal probability, e.g. "0.9"
    if value < 0 or value > 100:
        return float("nan")
    return value / 100.0                 # integer/large value: percent


class OllamaRunner:
    """Thin wrapper over the local Ollama HTTP API (sequential, one model).

    ``system`` and ``num_predict`` default to the risk-fusion task above. The
    explainability layer reuses this runner with its own system prompt and a
    longer generation budget, so there is exactly one place that knows how to
    talk to the local model.
    """

    def __init__(self, model=DEFAULT_MODEL, timeout=120, temperature=0.0,
                 system=SYSTEM_PROMPT, num_predict=8, num_ctx=None,
                 think=None):
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.system = system
        self.num_predict = num_predict
        self.num_ctx = num_ctx               # None -> ollama's model default
        # think=False disables a reasoning model's hidden deliberation. A
        # capped num_predict otherwise gets consumed ENTIRELY by the think
        # block and the visible response comes back empty (observed with
        # gemma4:12b: eval_count == num_predict, response == ""). Harmless
        # for non-reasoning models (verified llama3.1/qwen2.5, ollama 0.30).
        self.think = think                   # None -> field omitted

    def generate(self, prompt):
        """Return the model's raw reply for one prompt."""
        options = {"temperature": self.temperature,
                   "num_predict": self.num_predict,
                   "top_p": 1.0, "seed": 42}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        body = {
            "model": self.model,
            "prompt": prompt,
            "system": self.system,
            "stream": False,
            # temperature 0 for reproducibility: the ablation should not turn on
            # sampling luck.
            "options": options,
        }
        if self.think is not None:
            body["think"] = self.think
        payload = json.dumps(body)
        result = subprocess.run(
            ["curl", "-s", "--max-time", str(self.timeout),
             "http://localhost:11434/api/generate",
             "-H", "Content-Type: application/json", "-d", payload],
            capture_output=True, text=True, timeout=self.timeout + 15,
        )
        if result.returncode != 0 or not result.stdout:
            return ""
        try:
            return json.loads(result.stdout).get("response", "")
        except json.JSONDecodeError:
            return ""


class RiskCache:
    """Append-only JSONL cache keyed by ``(stay_id, hour)``.

    Written after every single generation so a timeout or crash loses at most
    one call.

    ``fingerprint`` guards against silent staleness: a cache produced under one
    (model, window, prompt) combination must not be reused under another. The
    fingerprint is stored as a meta line; reopening with a DIFFERENT stored one
    raises. A legacy cache with no meta line is adopted with a loud warning --
    it predates the guard, so its provenance is genuinely unknown.
    """

    def __init__(self, path, fingerprint=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.entries = {}
        stored = None
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
                    if "__fingerprint__" in rec:
                        stored = rec["__fingerprint__"]
                        continue
                    self.entries[(int(rec["stay_id"]), int(rec["hour"]))] = rec
            if fingerprint is not None:
                if stored is not None and stored != fingerprint:
                    raise ValueError(
                        f"{self.path} was generated under a different "
                        f"(model, window, prompt) fingerprint: cached={stored!r}, "
                        f"requested={fingerprint!r}. Use a new cache path."
                    )
                if stored is None:
                    print(f"[cache] WARNING: {self.path} predates cache "
                          f"fingerprinting ({len(self.entries):,} rows of "
                          f"unknown provenance); adopting the current one")
                    with self.path.open("a") as fh:
                        fh.write(json.dumps({"__fingerprint__": fingerprint}) + "\n")
        elif fingerprint is not None:
            with self.path.open("a") as fh:
                fh.write(json.dumps({"__fingerprint__": fingerprint}) + "\n")

    def __contains__(self, key):
        return key in self.entries

    def __len__(self):
        return len(self.entries)

    def get(self, key):
        return self.entries.get(key)

    def add(self, stay_id, hour, risk, reply):
        rec = {"stay_id": int(stay_id), "hour": int(hour),
               "risk": None if np.isnan(risk) else float(risk), "reply": reply}
        self.entries[(int(stay_id), int(hour))] = rec
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def risks_for(self, keys):
        """Risks in the order of ``keys``; ``nan`` where absent or unparsed."""
        out = []
        for stay_id, hour in keys:
            rec = self.entries.get((int(stay_id), int(hour)))
            out.append(float("nan") if rec is None or rec["risk"] is None
                       else float(rec["risk"]))
        return np.array(out, dtype=float)


def windows_for(summaries, window_hours=WINDOW_HOURS):
    """Yield ``(stay_id, hour, window)`` for every row, oldest-to-newest.

    The window for hour ``t`` holds rows ``t - window_hours + 1 .. t`` from the
    same stay -- never anything later.
    """
    cols = ["vitals_text", "labs_text", "news2_text"]
    for stay_id, group in summaries.groupby("stay_id", sort=False):
        group = group.sort_values("hour").reset_index(drop=True)
        records = group[cols].to_dict("records")
        hours = group["hour"].to_numpy()
        for i, hour in enumerate(hours):
            lo = max(0, i - window_hours + 1)
            yield int(stay_id), int(hour), records[lo:i + 1]


def cache_fingerprint(model, window_hours, system=SYSTEM_PROMPT):
    """What a risk cache's contents depend on, hashed for the meta line."""
    import hashlib
    return {"model": model, "window_hours": int(window_hours),
            "system_sha": hashlib.sha256(system.encode()).hexdigest()[:12]}


def generate_risks(summaries, cache_path, model=DEFAULT_MODEL,
                   window_hours=WINDOW_HOURS, limit=None, progress_every=25,
                   runner=None):
    """Generate (or resume) fused risks for every row of ``summaries``.

    Returns the :class:`RiskCache`. Already-cached rows are skipped, so calling
    this twice costs nothing the second time. The cache is fingerprinted by
    (model, window, system prompt): resuming under different settings raises
    instead of silently mixing generations.
    """
    cache = RiskCache(cache_path,
                      fingerprint=cache_fingerprint(model, window_hours))
    runner = runner or OllamaRunner(model=model)

    todo = [(s, h, w) for s, h, w in windows_for(summaries, window_hours)
            if (s, h) not in cache]
    if limit is not None:
        todo = todo[:limit]
    print(f"[llm] {len(cache):,} cached, {len(todo):,} to generate "
          f"(model={model}, window={window_hours}h)")

    failures = 0
    for n, (stay_id, hour, window) in enumerate(todo, start=1):
        reply = runner.generate(build_prompt(window))
        risk = parse_risk(reply)
        if np.isnan(risk):
            failures += 1
        cache.add(stay_id, hour, risk, reply.strip()[:80])
        if progress_every and n % progress_every == 0:
            print(f"  {n:,}/{len(todo):,} ({failures} unparsed)", flush=True)
    if todo:
        print(f"[llm] done: {len(todo):,} generated, {failures} unparsed")
    return cache
