# Comments

1. Data is read hourly from its respective sources. Can you ensure it is on event basis and/or hourly. This ensures that any critical data is not waiting for its hourly slot. This combination ensures data is not later than 1hr old.
2. Call out some of the deployment requirements. Check if you can bring all of this on a single compute node of, say, i7 category with integrated GPU. This is mainly to ensure that the solution is not blocked due to high compute requirements needing data center compute. This is not an immediate ask but something to look at in the final phase.
3. Can you provide some details on each of the agents looking at the different modality of input data? What qualifies them as Agents?
4. The documentation points out what analytics was used for each sensor modality and why. I have not double clicked on the analytics done for each sensor modality. Was this vetted at your end beyond the literature survey you did? I am referring to subject matter experts in the campus - Prof Vinodha or some Prof's in medical college for example?
5. Orchestrator: One suggestion - Dont call it "Death Head". You can name the two networks as "Treatment" and "Risk" heads.
6. Different patients will not have their sensor data read at the same priority. Depending on what they are suffering from, the weightage varies. Isn't it? Please discuss if same weightage across makes sense or not.
7. I did not get the "LLM and Explainability" clearly. One request: Can you position the writeup as a technical document. I find the language a little flowery 🙂.
8. Provide a technical section which describes the architecture, components used, HW used, etc. in the solution. As I noted in #2, the HW story has a separate dimension that you can look into.

# Answers:

## 1.  Event-basis ingestion

Each channel is represented as the last value recorded strictly before the scoring instant, together with its age in hours. (Age is an explicit model input — the network is never told that a stale value is fresh.) verify
The hourly grid is the model's time axis, and not a polling interval. The TCN's ~29-hour receptive field is 29 steps. Because every agent is a pure function of "last value before now plus its age", the current step can be re-scored the moment new data arrives rather than at a fixed slot boundary. Worst-case data age is then the arrival latency of the event, not one hour.

This is the intended serving design; the evaluation in this document is retrospective on MIMIC-IV, where data arrives as a completed record, so the event-driven loop is specified but not yet benchmarked.

Note: Rephrase this, tldr we CAN do event based ingestion but dont due to limitations of the mimic dataset.

## 2. Deployment requirements

It runs on a single commodity laptop with no discrete GPU, on roughly one CPU core. **Four machines** have been measured end to end, spanning three operating systems and two instruction-set architectures, with the GPU disabled where one existed and absent where one did not.

| Metric | Ultra 9 275HX | i5-13500HX | **i9-13900H** | Apple M1 |
| --- | --- | --- | --- | --- |
| OS · cores · RAM | Linux · 24 · 32 GB | Windows · 20 · 32 GB | **Windows · 20 · 16 GB** | macOS · 8 · 8 GB |
| Graphics | discrete, hidden | discrete, unused | **integrated only** | on-die, unused |
| torch | 2.14.0+cpu | 2.14.0+cpu | 2.14.0+cpu | 2.14.0 arm64 |
| Learned parameters, whole path | 96,605 | 96,605 | **96,605** | 96,605 |
| Fusion checkpoint on disk | 121 KB | 121 KB | 121 KB | 121 KB |
| Score one patient-hour (29 h context), 1 thread | 0.727 ms | 1.66 ms | **1.99 ms** | 1.12 ms |
| Batched throughput, 1 thread | ~232,000 ph/s | ~184,000 ph/s | ~159,000 ph/s | see note |
| Serving process peak memory | 537 MB | 335 MB | **319 MB** | 373 MB |
| 20-bed ICU, hourly scoring | 15 ms CPU/h | 33 ms CPU/h | **40 ms CPU/h** | 22 ms CPU/h |
| Full rebuild of every model | 126 s (8 thr) | 403 s (8 thr) | — | 322 s (1 thr) |
| Peak memory during that rebuild | 5.75 GB | 2.0 GB | — | 2.4 GB |
| Environment on disk | 1.8 GB | 1.2 GB | 1.2 GB | 1.1 GB |

Parameter counts match to the unit on all four, confirming the same architecture ran everywhere. **The worst case across every machine is 2 ms per patient-hour, and 40 ms of CPU per hour for a 20-bed unit — about 0.001% of one core.**

Two of these machines matter beyond adding data points. The **i9-13900H has Iris Xe integrated graphics and no discrete GPU at all**, which is the configuration the review comment describes; it is a 13th-generation Intel laptop part with the same core layout as the i5 above it. The **Apple M1** is ARM rather than x86, and shows the system is not tied to an instruction set. Serving memory converges across the three non-Linux machines at 319–373 MB; our own 537 MB is the outlier, and **~320 MB is the figure to quote**.

Serving wants one core; training wants all of them. Latency rises as threads are added on every machine measured — 0.73 → 0.80 ms on the fastest, 1.99 → 3.06 ms on the i9, 1.12 → 2.79 ms on the M1 — because these networks are small enough that thread dispatch costs more than the arithmetic. Training is the opposite, roughly 3× from one thread to eight. The repository's single-thread default is therefore correct for serving rather than a limitation. Most of the serving cost is the fusion, not the agents: 1.04 ms of the i5's 1.66 ms, 1.23 ms of the i9's 1.99 ms.

For a 20-bed ICU the prediction path is at most 0.0011% of one core. Explanation cards, generated only when an alarm fires (~0.52/hour at the frozen alarm volume), dominate the remainder: about 0.16% of a core including them.

Minimum requirements:

|  | Floor | Recommended | If shipping the 12B card model |
| --- | --- | --- | --- |
| Cores | 2 | 8 | 8 |
| Memory | 8 GB | 16 GB | 24 GB |
| Disk | 2 GB | 6 GB | 14 GB |
| Full rebuild |  ~9 min | ~7 min | ~7 min |
| GPU | none | none | none |

**Both memory tiers are now measured rather than extrapolated.** The 8 GB floor was verified on an actual 8 GB machine: the M1's full rebuild peaked at 2.4 GB with zero swaps. The 16 GB row's warning about the 12B card model was demonstrated on the i9 — with one ordinary application running alongside, gemma4:12b fell from 4.9 to 1.5 tokens per second and 32 seconds per card, while medgemma:4b stayed usable at 7.9. On the 8 GB machine the 12B needs six and a half minutes per card. **medgemma:4b is the deployment reader at both tiers**, and on the 16 GB tier that is the difference between robust and fragile rather than a preference.

Memory is the only hard floor; cores cost time, never correctness — a dual-core machine produces bit-identical output, just slower. AVX2 is not required either, and the penalty shrinks as cores get newer: forcing the pre-2013 kernel fallback costs 40% on the oldest machine tested, 16% on the next, and **nothing measurable on the i9**. The M1 settles the point outright, since it has no AVX2 at all and runs the same code unmodified.

Two deployment notes:

Use the CPU-only torch wheel — smaller and faster. On one machine, switching from the CUDA build cut latency 0.832 → 0.727 ms (13%), serving memory 691 → 537 MB, and the environment 7.6 → 1.8 GB. The CUDA build maps its libraries even with the GPU hidden and returns nothing for them. On Linux a further 288 MB arrives via xgboost, which hard-requires nvidia-nccl-cu13 whatever torch wheel is used; it is unused at inference and --no-deps drops it.

For CPU explanation cards, use ollama ≥ 0.34 and **leave num_thread unset**. Across the four machines the default lands at or near peak on three; only the Linux box's default gives up 19%. What must never be done is pin the thread count to the logical-core count: that costs 4% on one machine and 45% on another, and the magnitude is not predictable from the hardware. Requiring ≥ 0.34 removes the catastrophic case on x86 — an older runtime lost two-thirds of its throughput — but not the underlying mechanism, since a chip with performance and efficiency cores can still lose half its rate when work is split evenly across both.

What is and is not measured:

The i9-13900H is the integrated-graphics configuration the comment describes, on a current-generation Intel laptop part. It is an i9 rather than an i7, and an i7 of that generation sits between it and the i5 already measured — both bracket it, and no i7 was available to us. One part of the phrasing does remain untested: the integrated GPU as an *inference device*, through OpenVINO or IPEX-LLM. Every figure above treats it as absent, which makes it a possible optimisation rather than a gap in the answer.

A note on the M1's batched throughput: the macOS arm64 torch build routes one convolution shape through a single-threaded fallback path, which costs that column roughly 4×. One configuration flag recovers it, and the bedside figure quoted above is unaffected.

## 3. What makes these agents?

Each is an independently trained, independently deployable model with a fixed output contract: given one modality at one hour, emit a latent vector, a calibrated risk, and a presence flag. They can be added, removed, or benched without retraining the others.

The clinical-notes agent is built and currently benched on exactly this basis. Each holds its own opinion that the orchestrator may weight or ignore, and the weighting is re-derived every hour from evidence.

They are not conversational agents: they do not exchange messages, call tools, or plan. Architecturally this is a gated mixture of experts with per-hour gating, and "modality expert" is an equally accurate name for the same component.

## 4.

Yes, on modality selection and deployment framing, with one clinician, and it changed the design. No, on the analytic choice per modality — that was literature plus internal ablation.

## 5. Orchestrator naming

Renamed as suggested. The two networks are now the Treatment head and the Mortality head:

1. Treatment head — trained on treatment onset: ventilation or vasopressor start within 24 hours.
2. Mortality head — trained on death within 24 hours.

We went with "Mortality" rather than "Risk" for one reason: both heads emit a risk, so naming one of them the Risk head implies the other is something else. "Mortality" names the label the head is trained on, which is the only thing that separates it from the treatment head. Both are isotonic-calibrated and combined by noisy-OR into a single probability of either event within 24 hours.

## 6. Do all patients get the same agent weighting?

No. The attention layer recomputes the eight agent weights every hour for every patient, from that patient's own evidence. Weighting is a per-hour, per-patient quantity, not a global constant.

We measured whether the roster is genuinely condition-dependent. Standalone lift (AUPRC ÷ within-condition prevalence) per agent, by discharge-coded condition:

| Condition | Stays | Top-3 agents | Notable |
| --- | --- | --- | --- |
| Sepsis | 1,222 | gbm · vitals · resp | fluid not elevated |
| Respiratory failure | 1,644 | gbm · vitals · labs |  |
| AKI | 2,448 | gbm · vitals · cadence |  |
| GI bleed | 554 | gbm · labs (3.9) · vitals | labs far above its usual 2.2–2.7 |

The rosters differ exactly as clinical reasoning predicts — falling haemoglobin is the early signal in GI bleeding, and labs is the only modality that sees it. A shared core (GBM, vitals) holds across four of six shown; the remaining seats are condition-specific.

We then tested the stronger form of this idea: train a separate specialist model per ICU scenario, each with its own agent roster selected on its own validation data. Three specialties × two specialist arms × three seeds = 18 comparisons against the generalist. The generalist won all 18, with 9 of 18 confidence intervals clear of zero. A specialist sees ~20% of the stays and loses the cross-scenario physiology the generalist learns from everyone else's deteriorations; the deficit scales with data starvation (respiratory, 2,500 stays: −0.06 AUPRC).

What survives from the specialist idea is the roster, not the retraining: within a specialty a 4-agent subset matches the full 8-agent bank, which matters for deployment cost. So the delivered decomposition is one model trained on everyone, per-unit operating points (alarm-burden spread across ICU types falls 5.3× → 1.7×), and attention doing per-patient weighting continuously — reading the roster from data hour by hour, rather than from a diagnosis label that only exists at discharge.

## 7. LLM and explainability

Role. The language model is a rendering layer. It converts a structured explanation into one sentence of English. It has no role in prediction.

Input. When an alarm fires, the system already holds the explanation as structured data: the attention weight on each of the eight agents at that hour, each agent's calibrated risk, the seven NEWS2 sub-scores, and 4-hour trends for five vitals. This structure is passed to a local model. The fused risk and the alarm flag are withheld, so the generated sentence cannot restate the decision it is explaining.

Output. One sentence naming the drivers. It is displayed alongside the alarm and is not stored as a model input.

Evaluation. Four local models were benchmarked on 200 held-out cases against two adversarial baselines. Selected model: **gemma4:12b**.

| Faithfulness Metric  | Result |
| --- | --- |
| Names the top-attended agent correctly | 97% of cards |
| Names the correct NEWS2 driver | 78% of cards  |
| Unsupported drivers invented | 0.13 per card |
| Alarm recoverable from card text by a surrogate model | 0.84 AUROC |
|  → same, with numbers and medical vocabulary stripped | 0.82 AUROC |

| Generation Latency | Per card |
| --- | --- |
| Local GPU  | ~0.7 s  |
| CPU only, ollama's default threads | ~3 s (4B) / ~11 s (12B) |
| CPU only, num_thread pinned | ~23% faster again |

The two surrogate rows test whether the card carries the decision's actual content rather than generic clinical language: a separate model, given only the card text, recovers the alarm at 0.84 AUROC — and still at 0.82 once every number and medical term is stripped out.

A medically fine-tuned model (medgemma:4b) was also tested and ranked last on six of seven benchmark metrics, inventing 0.36 unsupported drivers per card; domain tuning did not help on a task where the inputs are numbers supplied verbatim.

Guarantees. The model runs locally; no data leaves the node. It cannot modify a score, a threshold, or an alarm. Removing it leaves every prediction bit-identical.

## 8. Technical overview

### Pipeline:
Per patient-hour, four stages:

1. Agents. Eight independent models each read one modality and emit a latent vector, a calibrated risk, and a presence flag. Every input is the last value recorded strictly before the scoring instant, carried forward with its age in hours as an explicit feature.
2. Fusion. Latents are projected to a shared 32-dimensional space, weighted by softmax attention across agents, and passed through three causal TCN blocks spanning ~29 hours.
3. Heads. A treatment head and a mortality head each emit an isotonic-calibrated probability; the two are combined by noisy-OR.
4. Alarm. A threshold crossing on the fused risk. Thresholds are frozen on validation data and never recomputed at inference.

### Components:

| Agent | Modality | Analytic | Features | Params |
| --- | --- | --- | --- | --- |
| Vitals | 5 vital signs | causal TCN, dilations 1/2/4 | 19 | 19681 |
| ECG | cart machine-read | MLP + learned "no data" embedding | 51 | 10996 |
| Cadence | order/charting timestamps | MLP | 32 | 4225 |
| Labs | 24 analytes | MLP (2 hidden layers) | 24 | 3713 |
| Fluid | urine output, balance | MLP | 13 | 3009 |
| Resp | O₂ escalation ladder, FiO₂ | MLP | 12 | 2945 |
| NEWS2 | charted vitals | deterministic lookup | 7 sub-scores | 0 |
| GBM | tabular, all modalities | XGBoost | 145 | trees |
| Orchestrator | agent bank | attention + causal TCN, ×2 heads | 208 inputs/hour | 52036 |

Total learned neural parameters: 96,605. Fusion checkpoint on disk: 121 KB.

### Data and protocol:

MIMIC-IV v3.1 under the PhysioNet credentialed-access DUA, extracted via BigQuery. 

- 56,157 ICU stays
- 2,460,983 patient-hours

Splits are by patient, never by stay or row, so no patient appears in two splits. Held-out test: 361,229 patient-hours across 8,375 stays at 4.74% prevalence. Scoring begins at hour 6, after a stabilisation window. Labels are 24-hour-horizon composites, censored at the first event.

Validation is split in two halves with disjoint patients: one fits the isotonic calibrators, the other freezes the alarm thresholds. Test data is touched only for final scoring.

### Hardware:

Development. The entire system — cohort extraction, all eight agents, the fusion, the benchmarks and the demonstration dashboard — was built and trained on a **Single laptop**: 

- Intel Core Ultra 9 275HX (24 logical cores)
- NVIDIA RTX 5070 Ti Laptop GPU with 12 GB VRAM, 32 GB RAM plus 50 GB swap.

No cluster, no cloud training, no data-centre compute at any point. Agent training and the gradient-boosted baselines use the GPU; the fusion is trained and scored on CPU at a single thread, deliberately, so results are bit-identical across runs.

The 12 GB VRAM ceiling is also the reason the explanation-card benchmark tops out at a 12-billion-parameter model: that is the largest 4-bit local model that fits alongside everything else. The card generator was never assumed to have more.

Deployment figures are in section 2: at worst 2 ms per patient-hour on one CPU thread and a ~320 MB serving process, with no GPU required for any prediction. Additional cores do not help, so the deployment unit is one core.

The gap between the two is smaller than it looks: development needs the GPU and the RAM headroom because it holds 2.46 million patient-hours of cached agent output in memory for batch evaluation. A bedside deployment holds one patient's recent history — kilobytes — and needs neither.

Verified on three further machines. Teammates independently ran the same benchmarks on a Core i5-13500HX and a Core i9-13900H under Windows, and an Apple M1 under macOS — the last three on CPU-only builds with no CUDA present at all. Parameter counts matched exactly on all four (96,605) and every conclusion held. The process corrected several of our own figures: an XGBoost timing that used 300 trees where the repository uses the library default of 100, a claim that hybrid CPU cores caused an inference slowdown which a second hybrid machine failed to reproduce, and a memory figure inflated by the CUDA build. It also surfaced two portability bugs in our own benchmark script — a Unix-only module that stopped it running on Windows, and a units mismatch that would have reported macOS memory a thousandfold wrong.

Verified on second hardware. A teammate independently ran the same benchmarks on an Intel Core i5-13500HX under Windows with a CPU-only torch build — no CUDA present at all, cuda.is_available() false. Parameter counts matched exactly (96,605), the serving and training numbers came in at roughly two to three times slower as expected for that part, and the conclusions were unchanged. Two of our figures were corrected in the process: an XGBoost timing that had used 300 trees where the repository uses the library default of 100, and an early explanation of an ollama throughput anomaly that we attributed to hybrid CPU cores and then withdrew when the second machine — also hybrid — failed to reproduce it.