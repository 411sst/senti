# Sentinel demo dashboard

Minimal Streamlit app for a live walkthrough: **Patient** (why an alarm fired, hour by
hour, with the local-LLM explanation card), **Cohort** (ranking vs earliness, per-unit
alarm burden, the caveats), **Footprint** (parameters, CPU latency, what runs where).
`COMPONENTS.md` explains every component on the page — what it is and what it indicates.

```bash
.venv/bin/pip install -r requirements-demo.txt          # streamlit + altair, once
.venv/bin/python scripts/build_demo_bundle.py           # ~10 min, once (CPU + ollama)
OLLAMA_KEEP_ALIVE=1h .venv/bin/streamlit run demo/app.py
.venv/bin/python scripts/export_demo_static.py          # aggregate-only leave-behind
```

## What the build does

`scripts/build_demo_bundle.py` refits the single-head ALL-8 fusion once at seed 42 and
saves `models/orchestrator/all8_s42.pt` (attention weights come from it; the plotted
risk is the two-head composite from `data/cache/two_head.npz`), freezes the 2% global
thresholds exactly as the results doc does, picks four curated test stays
deterministically (ventilation and vasopressor stays where the model alarmed ≥ 2 h
before NEWS2, one stay where NEWS2 alarmed first, one quiet stay), precomputes an
explanation card for every hour of those stays with `gemma4:12b`, measures the CPU
footprint, and writes:

- `data/cache/demo/bundle.json` — **patient-level**. Stays under the gitignored
  `data/` tree and is only ever rendered on this laptop (PhysioNet DUA). Never
  screenshot the Patient tab into anything that leaves the machine.
- `data/cache/demo/cohort.json` — aggregate-only; the only input the leave-behind
  exporter can open (`demo_bundle.assert_aggregate_only` guards it).

Flags: `--skip-llm` (no ollama), `--refit` (rebuild the checkpoint), `--pick-index k`
(swap the curated stays), `--with-death` (add a death stay).

## Presentation flow (about six minutes, then questions)

1. **Patient**, the ventilation stay: point at the two alarm markers ("Sentinel here,
   NEWS2 sixteen hours later, intubation here"). The slider opens on the model's alarm
   hour: attention shows the agents it leaned on, the chips show what NEWS2 could and
   could not see, the card says it in one sentence. Press **Regenerate live** — about a
   second on the laptop GPU, same sentence (temperature 0, seed 42: reproducible by design).
2. The **quiet** stay: flat risk, no alarm from either — it also knows when to stay
   silent. Then the **NEWS2-first** stay — "we show the ones we lose too".
3. **Cohort**: the scatter ("five ranking models on the zero line, one above it"), the
   burden bars ("same model, seven thresholds"), then read the three caveats aloud.
4. **Footprint**: the numbers, one sentence on what runs where, then hand over to the
   Defence Brief's questions for review.
5. Leave-behind: the aggregate-only page from `scripts/export_demo_static.py`, plus the
   Defence Brief link.

## Running it on a teammate's machine (credentialed, no GPU)

The app needs neither the data pipeline nor a GPU: it reads two JSON files and nothing
else. Only PhysioNet-credentialed teammates may hold `bundle.json` (it is patient-level
MIMIC-IV data); send it over a private channel, never through git or a shared drive.

```bash
git clone <repo> sentinel && cd sentinel && git checkout experiment-demo
python3 -m venv .venv && .venv/bin/pip install -r requirements-demo.txt   # ~1 min, no torch
mkdir -p data/cache/demo
# copy bundle.json and cohort.json (from the build machine) into data/cache/demo/
.venv/bin/streamlit run demo/app.py
```

What works without a GPU: everything — all three tabs, every stay, every hour, and the
cached explanation card for each hour. The only difference is the **Regenerate live**
button: with no ollama running it shows a warning and keeps the cached card. (Optional:
`ollama pull gemma4:12b` and `ollama serve` on CPU makes the button work, at roughly
20–30 s per card instead of ~1 s.)

## Before the demo

- `OLLAMA_KEEP_ALIVE=1h` — ollama unloads the model after 5 min idle; the first live
  card would otherwise take ~10 s. The app falls back to the cached card if ollama is
  not reachable.
- The attention shown is the single-head ALL-8 refit's (the composite's shared
  substrate) — the same caveat the T17 explanation benchmark carries.
