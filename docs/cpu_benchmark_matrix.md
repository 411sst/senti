# CPU-only benchmark matrix

**Purpose.** Intel's review comment #2 asks whether Sentinel can run on a single
compute node of "i7 category with integrated GPU", without data-centre compute.
Two machines have been measured and **neither is an i7**. This file is both the
*specification* — every measurement a comparable run must produce — and the
*results matrix* so far.

**Status: Part A and Part B are complete on both machines and agree.** Machine
B's revision D closed the last gap in its column. Two cross-corrections have
been applied along the way: B's 300-tree XGBoost figure was wrong and had been
copied into A's protocol (§3.2), and **A's "hybrid P/E cores cause the ollama
collapse" finding did not survive B's counter-measurement and is retracted**
(§4). One mechanism remains undetermined, with a named test to settle it (§7).

---

## 1. Machines

| | **A — Core Ultra 9 275HX** | **B — Core i5-13500HX** | **C — i7 target** |
|---|---|---|---|
| Role | primary dev laptop | teammate's laptop | **not yet measured** |
| Codename | Arrow Lake-HX | Raptor Lake-HX | — |
| Cores | 24 logical | 14 physical / 20 logical | ≥ 4 physical |
| RAM | 32 GB (+50 GB swap) | 31.7 GB | ≥ 8 GB |
| GPU present | RTX 5070 Ti Laptop, 12 GB | RTX 4050 | integrated |
| GPU used | **no** (`CUDA_VISIBLE_DEVICES=`) | **no** (`+cpu` wheel, no CUDA) | **no** |
| Torch | 2.11.0+cu128 | **2.14.0+cpu** | prefer `+cpu` |
| Python | 3.12.3 | 3.12.10 | 3.12 |
| Source | `results_fusion.md` T21 + this file | HW compatibility report **rev B** | — |

---

## 2. Part A — Serving path (answers "can it run at the bedside")

Both machines have now run `scripts/bench_single_node.py`.

| Measurement | A · Ultra 9 275HX | B · i5-13500HX | C · i7 |
|---|---|---|---|
| Learned parameters, whole path | 96,605 | **96,605** *(exact match)* | — |
| Fusion checkpoint on disk | 121 KB | architecture-only in that clone | — |
| **End-to-end** 1 patient-hour, 29 h ctx, 1 thread | **0.832 ms** | **1.660 ms** (p95 2.11) | — |
| …2 / 4 / 8 threads | 0.850 / 0.888 / 0.907 ms | 1.725 / 1.930 / 2.203 ms | — |
| …14 threads | — | 2.284 ms | — |
| …degradation at max threads | 0.92× | 0.73× | — |
| **Vitals-TCN alone**, 1 × 29 h, 1 thread | 0.207 ms | 0.418 ms (0.45 in rev D) | — |
| Batched throughput (256 × 29 h), 1 thread | 253,194 patient-hours/s | 184,000 patient-hours/s | — |
| **Serving peak RSS** (no agent bank) | 691 MB `+cu128` / **537 MB `+cpu`** | **335 MB** `+cpu` | — |
| Python + torch floor | 638 MB `+cu128` / 455 MB `+cpu` | — | — |
| Model contribution to RSS | 3.5–11 MB (sampling noise) | 0.7 MB | — |
| 20-bed unit, prediction only | 17 ms/hour `+cu128` / 15 ms `+cpu` | 33 ms/hour (0.0009% of a core) | — |

### The CPU-only wheel is faster, not just smaller

Machine A was re-measured on a purpose-built `torch 2.14.0+cpu` environment —
the same version and build machine B uses — so the two columns compare like
with like:

| | A · `+cu128` | A · **`+cpu`** | B · `+cpu` |
|---|---|---|---|
| Bedside latency, 1 thread | 0.832 ms | **0.727 ms** | 1.660 ms |
| Serving peak RSS | 691 MB | **537 MB** | 335 MB |
| Full fit, 8 threads | 133 s | **126 s** | 403 s |
| Full-fit peak RSS | 8.3 GB | **5.75 GB** | 2.0 GB |
| Environment on disk | 7.6 GB | **1.8 GB** | 1.2 GB |

**Dropping CUDA costs nothing and pays 13% at inference** (0.832 → 0.727 ms,
medians of 100 reps, p95s 0.856 and 0.742 — the gap is outside run-to-run
noise). Peak serving memory falls 22%, full-fit memory 31%, and the environment
shrinks 4×. There is no argument for shipping the CUDA wheel on a CPU node.

**Where A's remaining 288 MB of NVIDIA libraries comes from — and it is not
torch.** B's report notes its `site-packages` has no `nvidia/` directory. On
Linux that is not achievable with xgboost installed: **`xgboost` hard-requires
`nvidia-nccl-cu13`**, 288 MB, regardless of the torch wheel. `torch 2.14.0+cpu`
itself pulls none. B's clean tree is a Windows property — xgboost's Windows
wheel has no such dependency — not a consequence of choosing the CPU wheel.
`pip install xgboost --no-deps` removes it; it is unused at inference.

**Two residual gaps against B on identical torch, both Linux-vs-Windows:**
serving RSS 537 vs 335 MB and full-fit RSS 5.75 vs 2.0 GB. glibc's per-thread
malloc arenas on a 24-core host are the obvious suspect and are untested;
neither figure threatens the 8 GB floor, so this was not chased further.

**A is ~2× faster on the whole path** (0.832 vs 1.660 ms) — the same ratio every
other benchmark shows. Parameter counts match exactly, which confirms both ran
the same architecture.

**Where the 1.66 ms goes** (B's breakdown): the two fusion heads are **1.04 ms**,
the vitals-TCN 0.45 ms, and every MLP agent under 0.05 ms. The fusion, not the
agents, is the serving cost.

**Serving memory is not fit memory.** B measures the serving process at
**335 MB** peak with the models themselves contributing 0.7 MB. The 2.0 GB
quoted for the full fit holds the 2.46 M-row agent bank in RAM; a bedside
deployment never does. Quoting 2.0 GB as a *serving* requirement overstates it
roughly 6×. Both numbers are real and answer different questions.

> **Do not put A's 0.832 ms next to B's 0.418 ms.** They measure different
> things: A's is the **whole path** (six agent forwards + two fusion heads);
> B's is the **vitals-TCN alone**. The like-for-like row is TCN-only, where A is
> 0.207 ms against B's 0.418 ms — about 2× faster, as expected.

**Both machines independently confirm:** at batch 1, adding threads makes
latency *worse*. The serving unit is one core, and the repository's pinned
`OMP_NUM_THREADS=1` is the correct setting for serving rather than a limitation.

---

## 3. Part B — Batch and training path

### 3.1 Vitals-TCN, batch 64 × 200 hours (19,681 params)

| Threads | A forward | B forward | A throughput | B throughput | A train/step | B train/step |
|---|---|---|---|---|---|---|
| 1 | **10.4 ms** | 11.51 ms | 1,233,341 hr/s | 1,111,623 hr/s | **40.7 ms** | 72.98 ms |
| 2 | — | 6.21 ms | — | 2,061,158 hr/s | — | 51.81 ms |
| 4 | **2.2 ms** | 5.40 ms | 5,822,430 hr/s | 2,370,326 hr/s | **20.4 ms** | 44.38 ms |
| 8 | **1.6 ms** | 3.89 ms | 8,080,877 hr/s | 3,293,282 hr/s | **16.0 ms** | 42.13 ms |
| 14 | **1.4 ms** | 4.05 ms *(regresses)* | 9,154,496 hr/s | 3,157,999 hr/s | 16.0 ms | 41.77 ms |
| Speedup 1 → best | **7.4×** | 2.96× (at 8) | | | 2.5× | 1.75× |

Both machines peak and then flatten; B regresses slightly at 14.

### 3.2 XGBoost — **corrected, both machines**

`train_gbm()` in `baselines_mimic.py` passes **no hyperparameters** —
`XGBClassifier(random_state, n_jobs, tree_method="hist", eval_metric)` and
nothing else, so XGBoost's library defaults apply: **100 trees, depth 6**.

> **Both machines originally timed 300 trees**, a configuration the repository
> never fits. B caught this in revision B; A's first run had copied B's
> protocol and inherited the same error. Both columns below are re-measured
> with `train_gbm`'s exact settings on 1,720,991 train rows × **13** features.

| Threads | A fit | B fit | A predict 361k | B predict 361k |
|---|---|---|---|---|
| 1 | **6.64 s** | 9.17 s | 0.216 s (1.67 M rows/s) | 0.428 s (0.84 M rows/s) |
| 2 | **4.30 s** | 5.56 s | 0.111 s | 0.236 s |
| 4 | **2.79 s** | 4.02 s | 0.056 s | 0.124 s |
| 8 | **2.08 s** | 3.09 s | 0.029 s | 0.086 s |
| 14 | **1.87 s** (3.55×) | 2.78 s (3.30×) | **0.022 s (16.1 M rows/s)** | 0.058 s (6.2 M rows/s) |

### 3.3 Whole-pipeline fit

Machine A ran in an isolated clone (source + symlinked parquets + its own empty
`embeddings/`); machine B likewise. See §7 for why this is mandatory.

| Measurement | A | B | C |
|---|---|---|---|
| Full fit, 8 threads | **133 s (2 min 13 s)** | 403 s (6 min 43 s) | — |
| ⤷ NEWS2 / GBM | 0 s / 2 s | 0 s / 7 s | — |
| ⤷ vitals-TCN / labs | 45 s / 42 s | 113 s / 117 s | — |
| ⤷ orchestrator (cached latents) | 38 s | 166 s | — |
| Average cores busy | 4.76 of 24 (19.8%) | 7.53 of 20 (37.7%) | — |
| Peak resident memory | 8.3 GB — **build artifact, see §7** | **2.0 GB** | — |
| Disk: environment | 7.6 GB (1.6 torch + **4.2 CUDA**) | **1.2 GB** (`+cpu` wheel) | — |
| Disk: data | 6.0 GB (full research tree) | 467 MB | — |
| Test suite | 669 passed, 0 skipped | 667 passed, 1 skipped (668 collected) | — |

**A is 3.0× faster end to end.** Quote **B's 2.0 GB and 1.7 GB disk** as the
requirement — those are the CPU-wheel figures a deployment would actually see.

---

## 4. Part C — Explanation cards on CPU (`num_gpu: 0`)

Cards are generated **only when an alarm fires** (~0.52/hour for a 20-bed unit
at the frozen 2.62% alarm volume) and no prediction depends on them. Both
machines now use the repository's real card prompt at `num_predict=90`,
`temperature=0`, `seed=42`, `think=false`.

| Model | Weights | A tok/s (default) | A tok/s (**pinned**) | B tok/s | B s/card | B out | B resident |
|---|---|---|---|---|---|---|---|
| llama3.2:3b | 2.0 GB | — | — | **23.3** | 2.2 s | 49 tok | 2.6 GB |
| **medgemma:4b** | 3.3 GB | 12.1 | **15.2** | 19.0 | 2.0 s | 34 tok | 2.9 GB |
| gemma3:4b | 3.3 GB | — | — | 18.9 | 1.7 s | 28 tok | 2.9 GB |
| qwen2.5:7b | 4.7 GB | — | — | 11.4 | 3.0 s | 33 tok | 5.1 GB |
| **gemma4:12b** *(ships)* | 7.6 GB | 4.1 | **5.4** | 5.9 | 10.1 s | 57 tok | **8.9 GB** |

### Ollama thread count — version A/B, run on one machine

The mechanism is now measured rather than inferred. Both ollama versions were
run **side by side on machine A** — same CPU, same OS, same kernel, same model
store, same prompt, same session — with 0.34.0 extracted to a scratch directory
and served on port 11435 while the system's 0.30.10 kept serving on 11434.
Nothing was upgraded or replaced.

| `num_thread` | A · **0.30.10** | A · **0.34.0** | B · 0.34.0 (Windows) |
|---|---|---|---|
| 4 | 12.6 | 12.8 | 14.9 |
| 8 | 15.2 | 15.1 | 16.9 |
| 14 | **15.5** | **15.7** | **17.8** |
| all cores (A 24 / B 20) | **4.9** | **13.6** | 17.0 |
| default (unset) | 12.6 | 12.8 | **17.4** |

Median of 3 reps for the 14 / all-cores / default rows on A; single rep at 4 and
8. The all-cores figure on 0.30.10 is the noisiest cell measured (4.7–6.6 across
reps), which is itself consistent with threads contending.

**Three conclusions, and they split the earlier claim in half:**

1. **The version fixes the catastrophic case.** At all-cores, 0.30.10 collapses
   to 4.9 tok/s — 68% below its own peak — while 0.34.0 holds 13.6, down only
   13%. That is a **2.8× difference attributable to the runtime alone**, with
   every hardware variable held constant.
2. **The version does not fix the default.** Both versions default to ~12.7
   tok/s on machine A, **19% below the 15.5–15.7 available at `num_thread=14`**.
   Upgrading changes the default's behaviour not at all here.
3. **The default problem is machine-specific, not version-specific.** On B, the
   same 0.34.0 defaults to 17.4 against a 17.8 peak — within 2%. On A it is 19%
   off. Same runtime, same setting, different outcome: what remains is the OS
   (Linux vs Windows) or the E-core share (16 of 24 vs 8 of 20), and this
   experiment does not separate those two.

**The earlier "hybrid P/E cores cause the collapse" claim was wrong** and stays
retracted — B is hybrid and does not collapse. But the corrected picture still
supports pinning, for a different reason: ollama's *default* thread choice is
poor on machine A on **either** version.

> **Two recommendations, both measured:**
> **(a) Pin `num_thread` to ~14.** Worth **+23%** on machine A (12.7 → 15.5) and
> +2% on machine B, on either version, at zero cost.
> **(b) Require ollama ≥ 0.34.** It does not improve the tuned case, but it
> converts the untuned all-cores failure from −68% to −13%.
>
> **What not to say to Intel:** that their hybrid architecture requires manual
> thread pinning. The catastrophic case is a runtime bug fixed in 0.34; the
> residual default gap is unattributed between OS and core ratio. Pinning is a
> tuning note that happens to pay well, not a hardware defect.

**`think: false` is mandatory for gemma4:12b.** It is a reasoning model in this
ollama build: without the flag the whole `num_predict` budget is consumed by
hidden deliberation and the card returns **empty** while `eval_count` still
reads 90 — a plausible tok/s attached to no output. `OllamaRunner` carries the
field (`llm_orchestrator.py:110-115`) and `scripts/bench_single_node.py:225`
sets it, so machine A's numbers are real cards.

**Deployment pick: `medgemma:4b`.** Near the top on generation rate,
domain-matched, and 2.9 GB resident against gemma4:12b's 8.9 GB — the 12B alone
pushes a deployment past the 16 GB tier.

---

## 4b. Kernel dispatch — what a pre-AVX2 CPU gives up

`ATEN_CPU_CAPABILITY=default`, 8 threads, TCN at 64 × 200.

| Tier | A forward | A train/step | B forward | B train/step |
|---|---|---|---|---|
| AVX2 (Haswell+, 2013 on) | 1.5 ms | 16.4 ms | 4.3 ms | 53.3 ms |
| DEFAULT (pre-AVX2) | 2.1 ms | 18.9 ms | 5.0 ms | 53.9 ms |
| **Penalty** | **40% / 15%** | | **16% / 1%** | |

**AVX2 is not a requirement** — it is a speed difference, agreed on both
machines. The penalty is larger on A because its AVX2 path is faster to begin
with; A's *fallback* (2.1 ms) still beats B's AVX2 path (4.3 ms). Caveat from
B's report: this forces torch's kernel tier only, MKL underneath still uses the
host instruction set, so a genuine pre-AVX2 chip gives up somewhat more.

---

## 4c. Accuracy reproduction

Both machines ran `scripts/results_table.py`, which fits the **core-3 bank**,
not the shipped flagship. Calibrated:

| Model | A | B | published | A AUROC | B AUROC |
|---|---|---|---|---|---|
| **Two-head ALL-8 (flagship, 10 seeds)** | — | — | **0.309** | — | **0.823** |
| Orchestrator (core-3) | 0.237 | 0.232 | 0.232 | 0.790 | 0.786 |
| vitals-TCN | 0.198 | 0.187 | 0.187 | 0.765 | 0.755 |
| GBM | 0.172 | 0.172 | 0.172 | 0.747 | 0.747 |
| labs | 0.167 | 0.165 | 0.165 | 0.726 | 0.726 |
| NEWS2 | 0.114 | 0.114 | 0.114 | 0.684 | 0.684 |
| random floor | 0.048 | 0.048 | 0.048 | 0.501 | 0.501 |

Test split identical on both: 361,229 patient-hours, 8,375 stays, 6,335
patients, 4.74% prevalence.

**Never quote 0.232 as the system's accuracy** — it belongs to the three-agent
bank this timing script happens to fit. The flagship is 0.309 / 0.823, CIs
[0.291, 0.337] and [0.813, 0.840] (`results_fusion.md`).

**A drifts upward on the two trained sequence models** (+0.011 vitals-TCN,
+0.005 orchestrator) where B reproduces published exactly.

**The torch-version explanation for this is now disproven.** Re-running A's full
fit on `torch 2.14.0+cpu` — byte-identical build to B's — produced *exactly the
same* drifted numbers (orchestrator 0.237, vitals-TCN 0.198). Neither the CUDA
wheel nor the torch version causes it. Both machines ran 8 threads, so what
remains is the host: thread scheduling and floating-point reduction order differ
between a 24-core Linux box and a 20-thread Windows one, and agent refits at
eight threads are not order-deterministic. This is why the repository pins
`OMP_NUM_THREADS=1` and claims bit-identity only for the **fusion** at one
thread. Published agent numbers should be regenerated at one thread, not eight.

---

## 5. Protocol — what run C must fix

| Knob | Required value |
|---|---|
| GPU | hidden (`CUDA_VISIBLE_DEVICES=`) or a `+cpu` wheel |
| Serving shape | 1 patient × 29 h context — report end-to-end **and** TCN-only |
| Batch shape | 64 × 200 hours |
| Threads | sweep 1, 2, 4, 8, 14 |
| GBM | `train_gbm`'s own settings — **library defaults, 100 trees**, 13 features |
| LLM | real card prompt, `num_predict=90`, `temperature=0`, `seed=42`, **`think=false`** |
| LLM threads | sweep `num_thread` 4/8/14/all — the default is 19% off peak on machine A |
| LLM models | `medgemma:4b` and `gemma4:12b` minimum |
| Timing | median of ≥ 20 reps after ≥ 5 warmup |
| Report | tok/s **and** s/card; tok/s is the portable one |

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/bench_single_node.py \
  --threads 1,2,4,8 --reps 100 --llm-cards 4 --llm-models medgemma:4b,gemma4:12b
```

---

## 6. The threading result, settled

| Path | Shape | Threads help? | Deploy as |
|---|---|---|---|
| Bedside scoring | 1 patient-hour | **no** — 0.90× (A), 0.70× (B) | 1 core |
| Backfill / catch-up | many patients | **yes** — up to 7.4× (A), 2.96× (B) | all cores |
| Training / refit | full cohort | **yes** — ~3× both machines | 8 threads |

Revision A's apparent 3.46×-versus-degradation contradiction was batch size and
nothing else; both machines now measure both regimes and agree.

---

## 7. Resolved, and still open

**Resolved since revision A**

- *GBM feature count* — 13, not 19. The 19 was the vitals-TCN's `N_FEATURES`.
- *GBM tree count* — 100 (library default), not 300. **Both machines re-measured.**
- *Test count* — B's 687 counted a dirty tree with 20 uncommitted tests; a clean
  `f7bbb62` collects 668. A collects 669 at a later commit with 0 skips; B sees
  1 skip. The remaining ±1 is commit drift plus one environment-dependent skip.
- *Memory* — B confirmed its venv is the `+cpu` wheel with **no `nvidia/`
  directory**. A's 8.3 GB peak is the `+cu128` wheel mapping CUDA libraries even
  with the GPU hidden. **2.0 GB is the requirement; ship the `+cpu` wheel** — 5×
  smaller on disk, ~4× lighter in RAM.

**Hazard — do not run `scripts/results_table.py` on a working repository.** It
calls `emit_and_cache()` at lines 94, 99 and 112, overwriting every agent's
cached latents in `data/cache/embeddings/` — the bank every published result is
paired against — silently. Both machines' timings come from isolated clones.

**Open**

1. **A real i7.** Neither machine is one. B's per-generation table is a
   projection anchored on one measurement and is the weakest evidence in the
   pair; it should be labelled as such wherever quoted. ~20 minutes on any i7
   laptop closes this.
2. **LLM thread behaviour** (§4) — **half resolved by the version A/B.** Running
   0.30.10 and 0.34.0 side by side on machine A proved the runtime causes the
   all-cores collapse (4.9 → 13.6 tok/s). It did **not** explain why ollama's
   default is 19% off peak on A but within 2% on B, on the same 0.34.0. That
   residue is OS or E-core share, unseparated. Isolating it needs either a
   Linux run on B's hardware or a Windows run on A's — neither is worth doing
   before the review, since pinning `num_thread` sidesteps it entirely.
3. ~~B's serving-path numbers~~ — **done in revision D.** Both machines now have
   a complete Part A.
4. **Integrated-GPU inference path** (OpenVINO / IPEX-LLM). Optional — the CPU
   card numbers make it unnecessary — but it is the one part of "i7 with
   integrated graphics" still untested.

---

## 8. Bottom line for comment #2

| Claim | Evidence |
|---|---|
| No GPU required anywhere, including explanation cards | A and B |
| Prediction is sub-millisecond per patient-hour on one core | A |
| Adding cores makes bedside latency worse — one core is correct | A and B |
| A 20-bed ICU uses ~0.16% of one core, cards included | A |
| The whole system rebuilds from scratch in 2–7 min on a laptop CPU | A and B |
| Peak memory for a full refit is 2.0 GB | B |
| A fresh CPU-only install is 1.7 GB on disk | B |
| XGBoost trains on 1.72 M rows in 2–3 s at 8 threads | A and B |
| Pre-AVX2 CPUs incur a 16–40% penalty, not a failure | A and B |
| RAM (8 GB), not the instruction set, is the real floor | B |
| The CPU-only torch wheel is 4× smaller **and 13% faster** at inference | A |
| On Linux, xgboost — not torch — pulls 288 MB of NCCL; `--no-deps` drops it | A |
| `medgemma:4b` is the deployment reader: 2.9 GB vs gemma4:12b's 8.9 GB | B |
| Require ollama ≥ 0.34 — turns an untuned −68% all-core case into −13% | A (version A/B) |
| Pin ollama `num_thread` ≈ 14: +23% on A, +2% on B, on either version | A and B |
