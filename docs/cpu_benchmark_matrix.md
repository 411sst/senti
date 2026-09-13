# CPU-only benchmark matrix

**Purpose.** Intel's review comment #2 asks whether Sentinel can run on a single
compute node of "i7 category with integrated GPU", without data-centre compute.
Two machines have been measured so far and **neither is an i7**. This file is
both the *specification* — every measurement a comparable run must produce — and
the *results matrix* so far.

The two existing runs measured **different halves** of the problem and used
different batch shapes, which makes some of their numbers look contradictory
until the protocol is pinned. Section 6 reconciles them; section 5 is the
protocol that keeps a third run comparable.

---

## 1. Machines

| | **A — Core Ultra 9 275HX** | **B — Core i5-13500HX** | **C — i7 target** |
|---|---|---|---|
| Role | primary dev laptop | teammate's laptop | **not yet measured** |
| Codename | Arrow Lake-HX | Raptor Lake-HX | — |
| Cores | 24 logical | 14 physical / 20 logical | ≥ 4 physical |
| RAM | 32 GB (+50 GB swap) | 31.7 GB | ≥ 8 GB |
| GPU present | RTX 5070 Ti Laptop, 12 GB | RTX 4050 | integrated |
| GPU used in benchmark | **no** (`CUDA_VISIBLE_DEVICES=`) | **no** (`+cpu` torch build) | **no** |
| Torch | 2.11.0+cu128 | 2.14.0+cpu | any |
| Python | 3.12.3 | 3.12.10 | 3.12 |
| Thread setting | 1 (serving) / 1–14 (sweep) | `OMP_NUM_THREADS=8` | both |
| Source | `results_fusion.md` T21 | HW compatibility report, 13 Sep 2026 | — |

---

## 2. Part A — Serving path (answers "can it run at the bedside")

The deployment question: one patient, one hour, scored on demand.

| Measurement | A · Ultra 9 275HX | B · i5-13500HX | C · i7 |
|---|---|---|---|
| Learned parameters, whole path | 96,605 | not measured | — |
| Fusion checkpoint on disk | 121 KB | not measured | — |
| Latency, 1 patient-hour, 29 h context, **1 thread** | **0.832 ms** | not measured | — |
| …2 / 4 / 8 threads | 0.850 / 0.888 / 0.907 ms | not measured | — |
| Batched throughput (256 × 29 h), 1 thread | 253,194 patient-hours/s | not measured | — |
| Model contribution to RSS | **3.5 MB** | not measured | — |
| Python + torch RSS floor | 638 MB | not measured | — |
| 20-bed unit, hourly, prediction only | 17 ms CPU/hour | not measured | — |
| 20-bed unit, incl. explanation cards | **≈ 0.16% of one core** | not measured | — |

**Finding (A):** additional cores do not help at batch 1 — latency *rises* from
0.832 to 0.907 ms going 1 → 8 threads, because dispatch overhead exceeds the
arithmetic. The serving unit is one core.

---

## 3. Part B — Batch and training path (answers "can it be rebuilt here")

### 3.1 Vitals-TCN, batch 64 × 200 hours (19,681 params)

Same protocol on both machines.

| Threads | A forward | B forward | A throughput | B throughput | A train/step | B train/step |
|---|---|---|---|---|---|---|
| 1 | **10.4 ms** | 14.2 ms | 1,233,341 hr/s | 901,405 hr/s | **40.7 ms** | 87.0 ms |
| 4 | **2.2 ms** | 6.8 ms | 5,822,430 hr/s | 1,876,347 hr/s | **20.4 ms** | 56.4 ms |
| 8 | **1.6 ms** | 5.9 ms | 8,080,877 hr/s | 2,160,312 hr/s | **16.0 ms** | 49.1 ms |
| 14 | **1.4 ms** | 4.1 ms | 9,154,496 hr/s | 3,147,608 hr/s | 16.0 ms | 52.4 ms |
| Thread speedup 1 → 14 | **7.4×** | 3.46× | | | 2.5× | 1.66× |

### 3.2 XGBoost (300 trees, depth 6, 1,720,991 rows × 19 features)

| Measurement | A | B | C |
|---|---|---|---|
| Fit, 1 thread | not measured | 22.2 s | — |
| Fit, 8 threads | not measured | 8.8 s | — |
| Fit, 14 threads | not measured | 7.3 s | — |
| Predict 361k rows, 14 threads | not measured | 0.16 s (2,266,198 rows/s) | — |

### 3.3 Whole-pipeline fit

| Measurement | A | B | C |
|---|---|---|---|
| Full fit, 8 threads | **not measured — see §7** | **403 s (6 min 43 s)** | — |
| ⤷ NEWS2 / GBM | — | 0 s / 7 s | — |
| ⤷ vitals-TCN / labs | — | 113 s / 117 s | — |
| ⤷ orchestrator (from cached latents) | — | 166 s | — |
| Average CPU load | — | 37.7% (7.53 of 20 cores) | — |
| Peak resident memory | — | **2.0 GB** | — |
| Disk: environment | 7.6 GB (CUDA torch build) | **1.2 GB** (CPU-only build) | — |
| Disk: data | 6.0 GB (all caches + embeddings) | 467 MB | — |

**Note on disk:** machine A's figures are a full research tree — every agent
cache, ECG embeddings, and a CUDA torch build. Machine B's 1.7 GB total is the
representative number for a fresh CPU-only clone, and the right one to quote.

---

## 4. Part C — Explanation cards on CPU (`num_gpu: 0`)

Cards are generated **only when an alarm fires** (~0.52/hour for a 20-bed unit
at the frozen 2.62% alarm volume) and no prediction depends on them.

| Model | Weights | A tok/s | A s/card | B tok/s | B s/card | B GPU s/card |
|---|---|---|---|---|---|---|
| medgemma:4b | 3.3 GB | 12.1 | **3.1 s** | 15.1 | 9.9 s | 8.4 s |
| gemma3:4b | 3.3 GB | not measured | — | 13.0 | 12.0 s | 7.3 s |
| llama3.2:3b | 2.0 GB | not measured | — | 13.0 | 19.3 s | 13.2 s |
| qwen2.5:7b | 4.7 GB | not measured | — | 8.4 | 17.2 s | 9.3 s |
| gemma4:12b | 7.6 GB | 4.1 | **11.0 s** | not measured | — | — |

**The s/card figures are not comparable between A and B** — different prompts
and different output lengths (A: the explanation-card prompt at
`num_predict=90`; B: an ICU-trajectory prompt at ~70–100 output tokens). The
**tok/s column is the comparable one**. Section 5 fixes this for run C.

**Both runs agree on the conclusion:** a 4B model generates a card in single-digit
seconds on CPU alone, and even a 12B model at 11 s is far inside a budget of one
card every two hours.

---

## 5. Protocol — what a comparable run must fix

A third run (the i7) is only comparable if these are held constant:

| Knob | Required value | Why |
|---|---|---|
| GPU | hidden (`CUDA_VISIBLE_DEVICES=`) or `+cpu` torch build | the question is CPU-only |
| Serving batch | **1 patient × 29 h context** | the bedside operating point |
| Batch-path shape | **64 × 200 hours** | matches §3.1 on both machines |
| Threads | sweep **1, 2, 4, 8, 14** | thread scaling is the whole story (§6) |
| LLM prompt | the explanation-card prompt, `num_predict=90`, `temperature=0`, `seed=42` | s/card is prompt-dependent |
| LLM models | `medgemma:4b` **and** `gemma4:12b` | the small and large ends |
| Timing | median of ≥ 20 reps after ≥ 5 warmup | first call includes model load |
| Report | tok/s **and** s/card | only tok/s is portable |

Machine A's serving numbers come from `scripts/bench_single_node.py`, which
already implements this:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/bench_single_node.py \
  --threads 1,2,4,8 --reps 100 --llm-cards 4 --llm-models medgemma:4b,gemma4:12b
```

---

## 6. Reconciling the two runs

The two reports appear to contradict each other on threading. They do not.

- **Machine A, serving:** 1 → 8 threads makes latency *worse* (0.832 → 0.907 ms).
- **Machine B, batch:** 1 → 14 threads gives **3.46×** on TCN inference.
- **Machine A, same batch shape as B:** **7.4×**.

**Batch size is the entire explanation.** At batch 1 × 29 hours (29 patient-hours
of work) the per-op dispatch cost dominates and threads add coordination
overhead. At 64 × 200 (12,800 patient-hours) there is enough arithmetic to
amortise them. Both findings are correct and they apply to different paths:

| Path | Shape | Threads help? | Deploy as |
|---|---|---|---|
| Bedside scoring | 1 patient-hour | **no** | 1 core |
| Backfill / catch-up | many patients at once | **yes, 3.5–7.4×** | as many cores as available |
| Training / refit | full cohort | **yes, ~3×** | 8 threads is the sweet spot |

Machine A is 1.4× faster than B at one thread but 3.7× faster at eight — a
newer core plus more of them, and it scales further before flattening.

---

## 7. Hazards and open items

**`scripts/results_table.py` overwrites the cached agent bank.** At line 94 it
calls `emit_and_cache(...)` into `data/cache/embeddings/`, replacing the
`vitals_tcn`, `labs`, `news2` and orchestrator emissions that every published
result is paired against. The i5 report was correctly produced on an isolated
clone. **Do not run it on the working repository** — this is why machine A has
no full-fit timing.

**The i5 report's accuracy table is the older ladder,** not the current flagship:
it shows the orchestrator at 0.232 AUPRC / 0.786 AUROC (the core-3 configuration
from `results.md`), whereas the shipped two-head ALL-8 flagship is 0.309 / 0.823
over ten seeds. Both are correct for what they ran; only don't quote 0.232 as
the system's accuracy.

**Still required for the Intel answer:**

1. A run on an actual **i7** — any generation clears the floor per the i5
   report's analysis, but nothing has been measured on one.
2. Machine A's batch-path numbers (XGBoost fit, full fit) — needs an isolated
   clone.
3. Machine B's serving-path numbers — `bench_single_node.py` has not been run
   there; it is a three-minute job and would complete the matrix.
4. An actual **integrated-GPU** inference path (OpenVINO / IPEX-LLM). Not
   required — the CPU card numbers make it optional — but it is the one part of
   the "i7 with integrated graphics" phrasing that remains untested.

---

## 8. Bottom line for comment #2

| Claim | Evidence |
|---|---|
| No GPU is required anywhere, including explanation cards | A and B, independently |
| Prediction is ~0.8 ms per patient-hour on one core | A |
| A 20-bed ICU uses ~0.16% of one core, cards included | A |
| The whole system rebuilds from scratch in ~7 min on a laptop CPU | B |
| Peak memory for a full refit is 2.0 GB | B |
| A fresh CPU-only install is 1.7 GB on disk | B |
| Pre-AVX2 CPUs incur a 16% penalty, not a failure | B |
| RAM (8 GB), not the instruction set, is the real floor | B |
