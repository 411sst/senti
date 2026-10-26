# ECG Results — the complete variant bake-off (V1–V6) and the hybrid thesis

Companion to `results.md` (which holds the core-3 / orchestrator / earliness
results). This documents the 2026-08-20/21 ECG rescue experiment: every ECG
approach the project has considered, trained and evaluated on **identical data**,
plus the fusion verdict and the design it motivates. All numbers reproduced on
the branch; commands at the bottom.

> **On the `paper` branch only V4 (`ecg_meas`, the shipped ECG agent) has its
> code.** The benched variants (V1 head training, V2, V3, V5, V6, H1) and their
> waveform download, embedding extraction and coverage-matched comparison were
> removed as unused by the flagship. Their numbers below stand as recorded; to
> reproduce them, use branch `experiment-demo` (or `feat/ecg-integration`).
> `src/agents/ecg_agent.py` and `ecg_encoder.py` stay because `ecg_meas`
> builds on them.

## Setup

- Cohort/test split as `results.md` (361,229 test hours, 8,375 stays, 4.74%).
- **ECG index** (`build_ecg_index.py`): 68,632 studies over 30,202 cohort stays;
  the leakage-safe pick (most recent ECG strictly before hour *t*, ≤72 h stale)
  covers **42.4% of cohort hours**.
- **Waveforms**: staged 5,000-stay subset = 7,547 studies (914 MB, 12-lead 10 s
  500 Hz) → **7.6% of test hours** ECG-on-disk. Full covered set (44k studies,
  ~5.3 GB) not yet downloaded.
- **Machine measurements** (`fetch_ecg_measurements.py`): cart-generated
  conduction intervals + up to 18 diagnostic statements for 67,270 of the
  indexed studies (100% coverage of hours with an indexed ECG).
- Every agent obeys the same contract: presence-guarded head (all no-ECG hours
  emit ONE constant risk — asserted in tests), isotonic-on-val calibration,
  strictly-before-*t* assembly.

## The variants

| id | agent | what it is |
|---|---|---|
| V1 | `ecg` | frozen **ECG-FM** (wav2vec2_cmsc, 90.9M, 768-d) + MLP head |
| V2 | `ecg_resnet` | 12-lead port of the stonecap 1D-ResNet (4.0M), trained end-to-end on `label_24h` |
| V3 | `ecg_resnet_film` | V2 + FiLM conditioning on patient age (ECG-present pathway only) |
| V4 | `ecg_meas` | **51 structured features** from the cart's machine read (12 numerics + validity + 26 diagnosis regex flags), no waveforms |
| V4d | `ecg_meas_disk` | V4 restricted to the 7.5k on-disk studies (training-data-matched control) |
| V5a | `ecg_resnet_pre` | V2's ResNet **pretrained** on machine-measurement targets (multi-task), frozen → head |
| V5b | `ecg_mamba` | CNN-stem → 8 Mamba blocks (pure PyTorch, 3.7M), same pretraining, frozen → head |
| V6 | `ecgfounder` | frozen **ECGFounder** (Net1D 30.7M, 1024-d; NEJM AI 2025, supervised on 10.77M ECGs) |
| H1 | `ecg_hybrid` | meas ⊕ Δmeas ⊕ gated ECGFounder residual ⊕ age (84.7k params) |
| H2 | `ecg_meas_hrv` | meas ⊕ gated **ultra-short HRV** slot (RMSSD/SD1/pNN50/CVNN from 7,540 waveforms via NeuroKit2) |
| H3a | `ecg_hyb_nodelta` | H1 minus Δmeas (waveform arm only) — ablation |
| H3b | `ecg_hyb_delta` | H1 minus waveform (**meas ⊕ Δmeas**) — ablation |
| H5 | `ecg_meas_wide` | V4-structured risk + auxiliary-trained founder latent concatenated into the emitted latent |
| H6 | `ecg_hyb_delta_film` | H3b + **FiLM(age)** modulating the measurement features (age-conditioned interpretation) |

## Grand table — every architecture, after all trainings

Standalone = covered-hours test AUROC on the agent's measured rows (42% of
hours; V1/V2/V5/V6 waveform-only agents are limited to the 7.6% disk rows —
marked †; the *identical-rows* comparison for V1–V6 is the intersection table
below). Fusion = ΔAUPRC vs core-3, mean over orchestrator seeds {42, 1, 2},
within-seed pairing; "✓" = sign-consistent at all 3 seeds.

| model | architecture | standalone AUROC | fusion Δ (3-seed) | sign |
|---|---|---|---|---|
| **H6 `hyb_delta_film`** | meas ⊕ Δmeas ⊕ FiLM(age) | **0.589** | +0.0033 | 2/3 |
| H2 `meas_hrv` | meas ⊕ gated HRV | 0.579 | +0.0026 | 2/3 |
| H1 `ecg_hybrid` | meas ⊕ Δ ⊕ gated founder | 0.578 | +0.0004 | 1/3 |
| H5 `meas_wide` | V4 risk + aux founder latent | 0.578 | −0.0012 (vs meas **−0.0062 ✗3/3**) | |
| H3a `hyb_nodelta` | meas ⊕ gated founder | 0.575 | +0.0005 | 1/3 |
| **H3b `hyb_delta`** | meas ⊕ Δmeas | 0.574 | **+0.0044** | **✓ 3/3** |
| **V4 `ecg_meas`** | 51 structured features | 0.568 | **+0.0049** | **✓ 3/3** |
| V4d `meas_disk` | V4, 7.6% coverage | 0.567* | −0.0098 | ✗ 3/3 |
| V6 `ecgfounder` | frozen Net1D 30.7M | 0.556 † | −0.0208 | ✗ 3/3 |
| V3 `film` | ResNet+FiLM e2e | 0.548* † | +0.0012 | 1/3 |
| V1 `ecg` | frozen ECG-FM 90.9M | 0.516 † | **−0.0385** | ✗ 3/3 (worst) |
| V5a `resnet_pre` | pretrained ResNet, frozen | 0.501 † | +0.0026 | 2/3 |
| V5b `mamba` | pretrained CNN→Mamba, frozen | 0.507 † | +0.0052 | ✓ 3/3 (latent effect) |
| V2 `resnet` | ResNet e2e on label | 0.483 † | +0.0023 | 2/3 |

All fusion cells above are now measured under the uniform 3-seed within-seed
protocol (no historical or single-seed values remain). V1's re-measured fusion
(−0.0385, our 7.6%-coverage reproduction) is far worse than its historical
−0.0101 (which came from its old full-embedding-coverage run) — a sparse,
near-chance 768-d-headed channel is the most damaging input tested.

\* intersection-rows figure (no own-coverage run). † waveform agents score only
disk-covered rows — see the intersection table for the fair V1–V6 comparison.

**Ship recommendation: `core-3 + meas` (V4)** — best fusion, sign-consistent,
simplest, earliness intact. **Equal-fusion alternate: H3b `hyb_delta`**
(+0.0044 ✓, better standalone 0.574). **Best standalone ECG risk model: H6**
(0.589) — for single-modality deployment, not for the fusion.

**Hybrid exploration conclusions (H1–H6):**
1. The Δmeas block (serial-ECG change) is fusion-neutral and standalone-positive
   — H1's fusion loss was caused by the **waveform arm's noise** (H3 ablation:
   wave-only +0.0005 ≈ full H1; Δ-only +0.0044 ≈ plain meas).
2. FiLM(age) is the strongest *standalone* addition (+0.015 over hyb_delta) but
   adds nothing in fusion (−0.0011 vs hyb_delta) — age×measurement interactions
   improve within-agent ranking, not orchestrator-level information.
3. Ultra-short HRV is real signal (0.579 standalone) but arrives through a
   7.6%-coverage slot and slightly dilutes fusion (−0.0023 vs meas). Worth
   re-testing at full coverage after the 44k download.
4. Every route for waveform embeddings degrades fusion: raw channel (−0.0208),
   gated residual (−0.0045 vs meas), aux latent (−0.0062 vs meas ✗3/3). The
   mamba-latent effect did not replicate by construction (H5) — it appears to
   require the latent to be an agent's own trained pipeline output.

## Standalone discrimination — identical test rows

Intersection rows (study on disk ∧ measured): **26,780 hours, 1,374 positives,
prevalence 5.13%**. Same rows for every model; stay-bootstrap 95% CIs.

| model | AUPRC | AUROC | 95% CI |
|---|---|---|---|
| **V4d meas_disk** | 0.062 | **0.567** | [0.523, 0.608] |
| **V4 meas** | 0.064 | 0.550 | [0.492, 0.607] |
| V3 film | 0.062 | 0.548 | [0.491, 0.605] |
| V1 ECG-FM | 0.057 | 0.519 | [0.453, 0.581] |
| V6 ECGFounder | 0.053 | 0.511 | [0.454, 0.563] |
| V5a resnet_pre | 0.053 | 0.510 | [0.450, 0.564] |
| V2 resnet | 0.053 | 0.502 | [0.446, 0.559] |
| V5b mamba | 0.048 | 0.471 | [0.437, 0.506] |
| *labs (same rows)* | 0.226 | 0.769 | [0.720, 0.813] |
| *orchestrator (same rows)* | 0.224 | 0.766 | [0.717, 0.812] |
| *vitals-TCN / NEWS2* | 0.152 / 0.093 | 0.701 / 0.632 | |

Labs/vitals hold their full-cohort level on these rows → the subset is not
"harder"; the ECG modality is genuinely weak for this label.

Key paired differences (stay bootstrap, same rows): meas_disk − mamba **+0.098
[+0.043, +0.159]**; meas − mamba +0.080 [+0.026, +0.133]; film − mamba +0.077
[+0.022, +0.134]; meas_disk − resnet_pre +0.055 (P=96%); film − resnet +0.046
(P=97%); meas − meas_disk −0.016 (P=29% — V4 does NOT need its 9× larger
training set to win).

Full-coverage / own-subset numbers (different row sets — do not compare across
rows): V4 at its full 42.0% coverage: covered test AUROC **0.571** (no-rate
variant **0.582**), Gate-1 val 0.563 [0.535, 0.589]. V6 on the 27,344
disk-covered hours: **0.556** (best per-event of the waveform pack: vent 0.044,
vaso 0.043, death 0.012); val 0.522 [0.465, 0.582]. V1 on the same disk rows:
0.516, gain over presence floor +0.0002 (does not beat presence). V2's original
run: 0.483.

**V1's historical "0.607" is an artifact.** Reproduced end-to-end on today's
subset (fairseq_signals + checkpoint + fresh extraction), frozen ECG-FM is
barely above chance — consistent with the independent ICLR-2026 FM benchmark
(arXiv:2509.25095), which found ECG-FM competitive only when fine-tuned.

## Pretraining (V5): encoders learn ECGs, not this label

Multi-task pretraining on machine-measurement targets (statement flags BCE +
masked interval regression), 5,259 train-split studies, patient-grouped 90/10:

| backbone | best pretrain-val loss | macro-AUROC | epochs |
|---|---|---|---|
| ResNet (4.0M) | 0.842 @ ep 10 | **0.899** | 17 (1 min GPU) |
| CNN→Mamba (3.7M) | 0.862 @ ep 6 | 0.882 | 13 (26 min GPU) |

Both encoders read electrophysiology (AFib, blocks, QTc…) at ~0.9 AUROC from
raw waveforms — then transfer to `label_24h` at ≈ chance. **Supervision was the
bottleneck for training, but information content is the bottleneck for the
label: the waveform adds almost nothing beyond the machine's own read at this
data scale.** Architecture (Mamba vs ResNet, controlled A/B: 0.471 vs 0.510) is
not the missing piece.

## Fusion — the only gate that matters

Single-run ablation (val, seed 42, patient-clustered bootstrap):

| config | ΔAUPRC vs core-3 | 95% CI | attention |
|---|---|---|---|
| **core-3 + meas** | **+0.0068** | [+0.0040, +0.0097] | 0.051 |
| core-3 + mamba | +0.0071 | [+0.0043, +0.0098] | |
| core-3 + meas_disk | −0.0114 | [−0.0159, −0.0074] | |
| core-3 + resnet / resnet_pre / film | ≈ 0 | | |
| core-3 + ECG (V1, historical) | −0.0101 | | ~0.04 |

**Seed-robustness (methodology catch).** core-3 alone spans val AUPRC
0.2217–0.2399 across orchestrator seeds (spread **0.0182**) — larger than any
fusion delta. Within-seed-paired deltas over seeds {42, 1, 2}:

| config | Δ per seed | mean | sign-consistent? |
|---|---|---|---|
| + meas | +0.0068 / +0.0017 / +0.0063 | **+0.0049** | ✓ 3/3 positive |
| + mamba | +0.0071 / +0.0003 / +0.0082 | +0.0052 | ✓ 3/3 non-negative |
| + meas_disk | −0.0115 / −0.0133 / −0.0045 | −0.0098 | ✓ 3/3 negative |

⇒ (a) **V4 is the first ECG agent to genuinely improve the fusion** — positive
at every seed, mean +0.005. (b) Single-run fusion deltas with
conditional-bootstrap CIs overstate certainty; **all fusion comparisons
(including those feeding `results.md`) should be reported as mean-over-seeds
with within-seed pairing.** (c) The mamba row is a finding, not noise: its
*risk* ranks covered hours below chance yet its *latent* consistently helps —
direct support for the latents-over-risk thesis (meas_disk shows "any extra
channel" does NOT help). (d) Identical features at 7% coverage *hurt* (−0.010)
while at 42% they help: sparse channels are net-harmful to this fusion.

Additional multi-seed fusion rows (same protocol):

| config | Δ per seed vs core-3 | mean | verdict |
|---|---|---|---|
| + founder (V6, raw channel) | −0.0283 / −0.0223 / −0.0119 | −0.0208 | sparse FM latent damages fusion |
| + ecg (V1, re-measured) | −0.0398 / −0.0428 / −0.0330 | **−0.0385** | worst of all tested channels |
| + hybrid (H1) | −0.0011 / −0.0029 / +0.0051 | +0.0004 | ≈ 0; and **hybrid − meas = −0.0079/−0.0046/−0.0011 (negative 3/3)** |
| + resnet (V2) / + film (V3) / + resnet_pre (V5a) | mixed signs | +0.0023 / +0.0012 / +0.0026 | ≈ noise, none sign-consistent |

**Earliness under core-3+meas: headline SURVIVES** (re-checked at 3 seeds after
a single-seed run suggested a dip). Val, volume-matched vs NEWS2, mean over
seeds: 1% budget — core-3 median +1.67 h / win 59% vs core-3+meas +1.33 h /
58% (per-seed win%: 62/55/60 vs 56/55/64 — the earlier "56 vs 62" was a seed-42
artifact); 5% budget — identical (+1.00 h, 55% both). All runs p ≤ 4e-13,
false-alarm rates matched. No earliness objection to shipping core-3+meas.

## Findings (summary)

1. The **machine's own reading of the ECG beats every learned waveform
   encoder** — including two foundation models — for 24 h deterioration, and is
   the only fusion-positive ECG agent. ECG enters Sentinel as 51 structured
   features, not through a 90M-parameter encoder.
2. The signal is **rhythm/conduction/ischemia content, not stale heart rate**:
   removing all rate features *improves* the covered-hours AUROC (0.582).
3. Frozen foundation models underdeliver here: ECGFounder > ECG-FM head-to-head
   (0.556 vs 0.516 on disk rows), but both lose to the machine read.
4. Task-adapted pretraining works (0.9 macro-AUROC) but does not transfer to
   the weak label; **Mamba is not the fix** (controlled A/B).
5. Latents ≠ risk: a below-chance agent's latent can still help the fusion.
6. Orchestrator seed noise dwarfs fusion deltas — multi-seed pairing is
   mandatory methodology going forward.

## H1: the gated-residual ECG omni-agent (`ecg_hybrid`) — built & evaluated

One agent, one orchestrator slot, machine-read backbone + waveform residual:

```
input  = [meas (51) ⊕ Δmeas vs previous ECG (13) ⊕ staleness ⊕ age·has_ecg]
latent = meas_body(input) + gate · wave_body(Linear(founder 1024→64))   # gate init 0
```

Three presence states extend the tested guard (no-ECG → learned constant —
verified, ONE risk; measured-only → learned `no_wave` slot; both → full).
Δmeas = QTc drift, HR trend, axis shift, new-onset AFib/ischemia/flutter/VT
flags, time-since-previous-ECG, prior-ECG count — serial-ECG change, unused by
any other variant or baseline. Age is ECG-presence-gated (V3 rationale: it must
not leak into the no-ECG floor). 84.7k params; gate opened to −0.105.

**Standalone: H1 > V4 exactly where designed** (identical rows):

| rows | H1 | V4 | Δ |
|---|---|---|---|
| all measured (n=151,799, 42%) | **0.578** | 0.568 | +0.010 |
| measured + waveform (n=26,780) | **0.582** | 0.550 | **+0.032** |
| Gate 1 (val covered) | 0.565 [0.538, 0.592] ✓ | 0.563 | |

**Fusion: V4 still wins** (multi-seed, within-seed pairing): hybrid − core-3 ≈
0 (mean +0.0004); **hybrid − meas negative at all 3 seeds (mean −0.0045)**.
Interpretation: Δmeas's trajectory content (HR trend, drift) improves
within-agent ranking but duplicates what the orchestrator already receives from
the vitals-TCN, and the waveform arm adds noise on 93% of hours — so the
snapshot-only V4 latent remains the more complementary fusion input. **Ship
verdict: core-3 + meas.** H1 stands as the better *standalone* ECG risk model
(e.g. for a deployment where only ECG is available).

## Literature context (must-cite; surveyed 2026-08)

- **MDS-ICU** (arXiv:2601.06645, AI4HealthUOL) is the closest prior work —
  MIMIC-IV + MIMIC-IV-ECG, ICU outcomes, supervised S4 waveform encoder + tabular
  late fusion; ECG adds significant gains on 14/33 tasks. **MDS-ED**
  (arXiv:2407.17856) shows the same stack ordering we found: clinical 0.8365 <
  +ECG *machine features* 0.8656 < +**supervised-on-target** S4 waveform 0.9115
  — waveform beats the machine read only when a small encoder is trained
  end-to-end on the target task, never frozen. Our frozen-FM results replicate
  the ICLR-2026 benchmark (arXiv:2509.25095): ECG-FM weak frozen; diagnostic
  FMs transfer poorly to outcome tasks.

## Ranked next steps

1. **ECG-CPC + learnable query-attention pooling** (Figshare ckpt; 3.8M SSM —
   the only benchmarked FM that *leads* deterioration/mortality/ICU categories;
   frozen use requires their `LearnableQueryAttentionPool1d`, not mean-pool).
   Free side-experiment: retrofit the same attention pool onto our ECGFounder /
   ECG-FM token sequences — tests whether mean-pooling caused part of the
   frozen-FM failure.
2. **Small supervised S4 waveform arm** replicating MDS-ICU's recipe (their
   MIT-licensed code targets our exact dataset) — the literature's one proven
   route to waveform > machine-read; requires the full 44k download for
   coverage.
3. **Risk-feature augmentation of V4** (deadline-friendly): EchoNext-mini
   structural-heart-disease probabilities (public 4.3 MB weights, 250 Hz),
   ECG-age gap (Zenodo weights + our age table), NeuroKit2 ultra-short-valid
   HRV (RMSSD/SD1, pNN50) + QT dispersion / Tpeak–Tend from per-lead
   delineation — all append to the winning structured agent.

## Reproduce

On `paper`, only the V4 lines below run (the others are marked; see the note at
the top).

```bash
# data (BigQuery: GOOGLE_CLOUD_PROJECT=<project>, SENTINEL_BQ_STORAGE=0)
.venv/bin/python scripts/build_ecg_index.py
.venv/bin/python scripts/fetch_ecg_subset.py --stays 5000     # 914 MB waveforms   # experiment-demo only
.venv/bin/python scripts/fetch_ecg_measurements.py
.venv/bin/python scripts/fetch_patient_age.py   # experiment-demo only
.venv/bin/python scripts/build_ecg_meas_features.py

# agents
.venv/bin/python scripts/train_ecg_meas_agent.py              # V4 (+ --no-rate)
.venv/bin/python scripts/train_ecg_hybrid_agent.py            # H1   # experiment-demo only
.venv/bin/python scripts/train_ecg_resnet_agent.py --variant resnet   # V2   # experiment-demo only
.venv/bin/python scripts/train_ecg_resnet_agent.py --variant film     # V3   # experiment-demo only
.venv/bin/python scripts/pretrain_ecg_encoder.py --backbone resnet    # V5a   # experiment-demo only
.venv/bin/python scripts/pretrain_ecg_encoder.py --backbone mamba     # V5b   # experiment-demo only
.venv/bin/python scripts/extract_pretrained_ecg_embeddings.py --backbone {resnet,mamba}   # experiment-demo only
.venv/bin/python scripts/train_ecg_pretrained_agent.py --backbone {resnet,mamba,ecgfounder}   # experiment-demo only
.venv-ecgfm/bin/python scripts/extract_ecg_embeddings.py      # V1 (isolated env)   # experiment-demo only
.venv/bin/python scripts/train_ecg_agent.py                   # V1 head   # experiment-demo only
.venv/bin/python scripts/extract_ecgfounder_embeddings.py     # V6   # experiment-demo only

# comparisons
.venv/bin/python scripts/ecg_coverage_matched.py --coverage both   # experiment-demo only
.venv/bin/python scripts/modality_ablation.py --split val
# multi-seed fusion: scratchpad/seed_robustness.py pattern (logs/seed_robustness.log)
```

Checkpoints: `models/ecg_pretrain/{resnet,mamba}.pt`, `models/ecg_fm/*`,
`models/ecgfounder/*`. Caches: `data/cache/embeddings/{agent}_{split}.npz`.
Tests: `pytest tests/test_ecg_*.py tests/test_ecg_pretrain.py` (all green;
full suite 491 passed).
