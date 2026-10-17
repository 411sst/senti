# Evidence table for the paper

Every number in `main.tex` cites a row here as `\ev{ID}`. A number with no row does
not go in the paper. Check the source before quoting; these rows were copied from
the files named, not from memory.

**Status**
- `VERIFIED`: found in a committed file on `experiment-demo` (line given where useful)
- `BRANCH`: committed, but on another branch
- `CACHE`: computed by a committed script, stored only in `data/cache/` (gitignored)
- `UNCOMMITTED`: exists only in an artifact or an uncommitted working tree. Commit before citing
- `RERUN`: must be re-run before the paper can state it
- `CHECK`: plausible, but the exact source still needs confirming

Main sources: `results_fusion.md` (T1–T23), `results_ecg.md`, `results.md`,
`docs/blog.md`, `docs/cpu_benchmark_matrix.md`, `origin/eicu-finetune:docs/eicu_finetune.md`,
`origin/live-version:results_live.md` (T24).

---

## A. Cohort, label and protocol

| ID | Claim | Numbers | Source | Seeds | Status |
|---|---|---|---|---|---|
| A1 | Development cohort | MIMIC-IV v3.1; 56,157 ICU stays; 2,460,983 scored patient-hours. Test: 361,229 hours, 8,375 stays, 6,335 patients, 4.74% positive | `results_fusion.md` l.115; `cpu_benchmark_matrix.md` §4c; row/stay counts recomputed from `data/cache/mimic_cohort_hourly.parquet` | — | VERIFIED |
| A2 | Label definition | First of: invasive ventilation onset (`InvasiveVent`/`Tracheostomy`), vasopressor or inotrope onset (norepinephrine, epinephrine, dopamine, phenylephrine, vasopressin, dobutamine, milrinone), in-hospital death; within 24 h; stay censored at first event | `scripts/mimic_cohort.sql` l.12–21, 29–42 | — | VERIFIED |
| A3 | Stabilisation window | Stay must be event-free for 6 h; without it 73% of events fall in hours 0–5 (eICU 71.7%). Hours 0–5 feed carry-forward only, not TCN steps | `mimic_cohort.sql` l.12–16; `src/mimic_cohort.py` l.40–46; T22 | — | VERIFIED |
| A4 | Splits | Patient-level 70/15/15, seed 42; validation split 50/50 by patient into V_fit (fusion calibration) and V_freeze (thresholds, variant selection). Agents early-stop and calibrate on the FULL validation set | `src/cohort.py` `split_patient_ids_three_way`; `src/fusion_common.py` l.8–18, 56–67 | — | VERIFIED |
| A5 | Leakage controls | Strictly-before-t features; labs binned by storetime (median 70 min after draw); causality tests rewrite the future | `src/agents/labs_agent.py` l.6–16; `src/agents/vitals_tcn.py` l.13–20; `tests/` | — | VERIFIED; test **count** → CHECK (`pytest --collect-only` before quoting) |
| A6 | Metrics | AUPRC = scikit-learn `average_precision_score`; CIs = patient-clustered bootstrap, 1,000 reps; thresholds frozen on V_freeze, volume-matched to NEWS2 | `src/metrics.py` l.14–20; `results_fusion.md` l.119–120; `src/fair_capability.py` `frozen_thresholds` | — | VERIFIED |

## B. Discrimination (MIMIC test)

| ID | Claim | Numbers | Source | Seeds | Status |
|---|---|---|---|---|---|
| B1 | Flagship two-head ALL-8 | AUPRC 0.309 (0.3087, SD 0.005), AUROC 0.823; CI [0.291, 0.337] / [0.813, 0.840] | T9, T18; `results_fusion.md` l.39, 157, 1574 | 10 | VERIFIED |
| B2 | Beats the equal-information GBM | FAIR+TRAJ (145 features) 0.287 [0.264, 0.311], AUROC 0.807; paired CI clear of zero | `results_fusion.md` l.39–41, 134 | 10/10 | VERIFIED |
| B3 | Published-spec rebuilds on our cohort | eCART-style 0.280 [0.257, 0.304] / 0.814; YAIB LightGBM 0.281 [0.258, 0.306] / 0.808; Sadanandan XGBoost 0.239 [0.220, 0.261] / 0.766; NEWS2 0.114 [0.102, 0.128] / 0.683. Feature counts 95 / 218 / ~26+age, sex | T10 `results_fusion.md` l.736–766; deviations in `src/lit_gbm.py` | comparators seed-fixed | VERIFIED |
| B4 | vs eCART-style | ΔAUPRC +0.0287 [+0.0252, +0.0323] | T19 `results_fusion.md` l.1676 | 10/10 | VERIFIED |
| B5 | Mortality head, scored on death | AUPRC 0.340 [0.288, 0.415], AUROC 0.946 [0.928, 0.960]; GBM 0.189 on death | T9, T18 `results_fusion.md` l.43, 158 | 10 | VERIFIED |
| B6 | Standalone agents (AUPRC / AUROC) | gbm 0.279/0.806; vitals 0.189/0.754; labs 0.166/0.727; cadence 0.156/0.703; resp 0.134/0.712; fluid 0.093/0.654; ECG 0.055/0.539 | `results_fusion.md` l.160–163 | — | VERIFIED |

## C. Patients caught at a fixed alarm budget

| ID | Claim | Numbers | Source | Seeds | Status |
|---|---|---|---|---|---|
| C1 | Catch rate at the NEWS2-matched budget | Of 1,540 event stays: Sentinel 65.5% (1,008), eCART-style 53.6% (825), YAIB 51.8% (798). Caught ≥6 h early: 34.9 / 28.2 / 27.1% | `data/cache/demo/cohort.json` `h2h.funnel`, built by `scripts/build_demo_bundle.py` | s42 | CACHE |
| C2 | **Realized test alarm rates are not equal** | Budget set on V_freeze (matched 2.617%); realized on test: Sentinel 2.79%, eCART-style 2.56%, YAIB 2.43%. Paper must report these. The sweep (C4) shows eCART-style at a 3% budget still catches only 55.8% | `cohort.json` `h2h.alarm_rate_test` | s42 | CACHE; **must be stated in the paper** |
| C3 | Cost | Quiet stays with ≥1 alarm: 11.8 / 6.8 / 6.2%. Stay-level precision: 55.5 / 63.9 / 65.5% | `cohort.json` `h2h.cost` | s42 | CACHE |
| C4 | Budget sweep, 0.5–10% | Sentinel 34.7→86.8%, eCART-style 26.8→79.7%, YAIB 26.8→77.2%; gap +7 to +13 points at every budget. Each model thresholded at its own (1−b) V_freeze quantile | `scripts/build_demo_bundle.py` l.470–488 (`h2h.sweep`); the local `cohort.json` predates this code | s42 | CACHE: re-run the bundle builder to regenerate before citing |
| C5 | First-alarm race (jointly caught stays) | vs eCART-style: Sentinel first 214, same hour 470, eCART first 78 (of 762). vs YAIB: 191 / 458 / 94 (of 743) | `cohort.json` `h2h.race` | s42 | CACHE |
| C7 | **Mid-stay catches** (first alarm after the stay's first scored hour) | Sentinel 23.2% (357), eCART-style 24.7% (380), YAIB 24.5% (377). Sentinel's catch-rate lead comes from stays it flags at the first scored hour. Must be discussed in the paper | `cohort.json` `h2h.funnel.midstay`; definition in `src/demo_bundle.py` l.95–115 | s42 | CACHE; **must be stated** |
| C6 | Race over 10 seeds; median warning | Significant 10/10 vs eCART-style, 9/10 vs YAIB; median warning 7.3 / 6.9 / 6.7 h (10 seeds: 7.1 h, 6.3–7.8) | "Sentinel in Numbers" artifact | 10 | CHECK: locate source |

## D. How agents communicate (the thesis claim)

| ID | Claim | Numbers | Source | Seeds | Status |
|---|---|---|---|---|---|
| D1 | LLM-as-fuser text arm loses | 150 stays, 6,185 hours, 4.41% positive; llama3.1:8b, 4 h window, temp 0. AUPRC text 0.100 raw / 0.090 calibrated vs latent 0.244; AUROC 0.732 vs 0.787; death component 0.025 vs 0.193 | T20 `results_fusion.md` l.1798–1827 | 1 (core-3, subset) | VERIFIED |
| D2 | Same-fuser channel arms | Core-3, full test: TEXT (templated sentences → frozen Bio_ClinicalBERT → PCA-128) 0.254 [0.236, 0.275] **>** LATENT 0.232 [0.215, 0.251]; RISK-ONLY 0.231; MEAN-RISK 0.244; HARD-VOTE 0.133. Timing: LATENT +2 h vs TEXT 0–1 h | `src/channel_arms.py`; `data/cache/channel_arms_scores.npz`; Aug-19 session notes. **Not written up in any results file** | 1 (core-3) | **RERUN on the flagship and write up. The paper's central claim depends on it** |

## E. Architecture choices and ablations

| ID | Claim | Numbers | Source | Seeds | Status |
|---|---|---|---|---|---|
| E1 | Adding agents | ALL-7 0.282; ALL-8 0.306; two-head 0.309 | T6, T8, T9; `results_fusion.md` l.152–157 | 3–10 | VERIFIED |
| E2 | ECG: machine read beats waveform models | 6 waveform variants incl. ECG-FM (90.9M) and ECGFounder (30.7M); machine-read fusion Δ +0.0049 (3/3 seeds); ECG-FM −0.0385 | `results_ecg.md` grand table | 3 | VERIFIED (core-3 era) |
| E3 | One model beats specialists | 18/18 specialist cells below the generalist, 9/18 CI clear; cardiac 0.353 vs 0.340/0.337; neuro 0.187 vs 0.168/0.171; respiratory 0.310 vs 0.244/0.250 | T14 `results_fusion.md` l.1190–1272 | 3 | VERIFIED |
| E4 | Per-unit thresholds | Alarm-burden spread across ICU types 5.3× → 1.7× | T12 `results_fusion.md` l.1071, 1138 | 5 | VERIFIED |
| E5 | Condition → top agents | Sepsis/pneumonia lean on resp; AKI on cadence; GI bleed on labs (lift 3.9) | T13 `results_fusion.md` l.1142–1188 | — | VERIFIED |
| E6 | YAIB TCN swap (vitals agent) | Inside flagship ΔAUPRC −0.0045 [−0.0076, −0.0013], lower in 9/10 seeds; agent alone −0.0036 (CI spans 0); 914k vs 19,681 params; receptive field 55 h vs 29 h; 16 random draws of YAIB's search space | T26 artifact `6tFTHBgxSoXnx5N3yAfevD`; `results_live.md` T26 not committed | 10 | UNCOMMITTED |
| E7 | Notes and other candidate agents did not ship | core-3+notes val Δ −0.0038 [−0.0081, +0.0007]; neuro-status and infection agents benched | `src/orchestrator/fusion.py` l.51; T15 | 3–5 | CHECK: cite the results rows, not code comments |
| E8 | Untested design choices | TCN depth/dilations/kernel; per-agent sequence models; MLP widths; hyperparameter search | none in any branch | — | OPEN: ablate, or list as a limitation |

## F. Timing

| ID | Claim | Numbers | Source | Seeds | Status |
|---|---|---|---|---|---|
| F1 | Lead over NEWS2 | Median lead @2% +0.95 h [+0.52, +1.38], positive 8/10; earlier in 68.5% [64.7, 72.4] of stays where alarms differ; ~70% of ≥12 h leads are first-hour flags | T18 `results_fusion.md` l.49–58, 1572 | 10 | VERIFIED |
| F2 | Death stays | On 68 death stays with no documented care-limitation decision: +0.5 h, 50% win | T16 `results_fusion.md` l.1413–1414 | — | VERIFIED |
| F3 | Treatment-only supervision | Death-first stays lead +3 / +3 / +4 h | T7 Part ii | 3 | VERIFIED |

## G. External validation and adaptation (eICU-CRD v2.0)

| ID | Claim | Numbers | Source | Seeds | Status |
|---|---|---|---|---|---|
| G1 | Cohort and protocol | 208 hospitals; 133,229 stays; 5,965,716 hours; 3.38% positive; nothing fitted on eICU. Model = ALL-8 minus cadence (no order-entry data); ECG sends its no-data value | T22 `results_fusion.md` l.1913–1990, 2175–2195, 2374–2380 | — | VERIFIED |
| G2 | Matched-density result | AUPRC 0.1526 vs YAIB 0.1391, eCART-style 0.1394, Sadanandan 0.1330; 30/30 intervals clear; full re-seed 30/30, larger margin | T22 l.2184, 2223 | 10 + 10 | VERIFIED |
| G3 | All hours, pooled | Ties YAIB and eCART-style; beats Sadanandan | T22 | 10 | VERIFIED |
| G4 | NEWS2 as a control | Lift 2.41 (MIMIC) vs 2.46 (eICU) | T22 l.2053–2054 | — | VERIFIED |
| G5 | Per-hospital AUROC (91 sites) | p10 / median / p90: two-head 0.689 / 0.759 / 0.842; one head 0.677 / 0.754 / 0.835; core-3 0.658 / 0.728 / 0.815 | T22 | — | CHECK: confirm line |
| G6 | Cost of dropping cadence (on MIMIC) | Single-head ALL-8 0.3009 vs 7-agent 0.2863 (~0.015); two-head cost roughly half | T22 l.2175–2195 | 1–10 | VERIFIED |
| G7 | Fine-tuning on eICU | Hospital split 129 / 30 / 49 (test 23,668 stays, 990,683 h, 3.76%): zero-shot 0.1293 → recalibrate 0.1435 → heads 0.1742 → full 0.1888 (+46%); AUROC 0.7516 → 0.7950; heads ≈ ¾ of gain; MIMIC after fine-tune 0.305 → 0.271 | `origin/eicu-finetune:docs/eicu_finetune.md` | 5 | BRANCH |
| G8 | Calibration transfer | ECE 0.0020 → 0.0381; label-free prior correction 0.0202; 500 local patients 0.0059; 5,000 0.0017; per-hospital median 0.0365 | T23 `results_fusion.md` l.2300–2360 | s42 | VERIFIED |

## H. Real-time scoring

| ID | Claim | Numbers | Source | Seeds | Status |
|---|---|---|---|---|---|
| H1 | Reading-level replay (T24) | Rebuild lossless (0 mismatches / 2,460,983 rows); on-arrival scoring +30 min mean [12, 55], median 0, +28% alarm onsets; ~80% of later first alarms wait on hourly agents; AUPRC 0.314 (monitor clock) → 0.292 (EHR clock); 52.5% of vitals reach the EHR after their hour is scored; 4.5% of vital hour-cells hold >1 reading | `origin/live-version:results_live.md` T24 | s42 | BRANCH; CHECK the two "median minutes late" figures (29 vs 34 min) |
| H2 | Vitals-only real time (T25) | TCN alone AUPRC 0.188 vs 0.314; real-time TCN alarms on 25.7% of quiet stays vs 10.8%; TCN-direct at matched false-alarm cost catches 55.5% vs 64.6%, first alarm −239 min [−336, −142]; 81% of TCN catches at first scored hour | T25 artifact `QrzrYZs61gUurvqo1BSakB` | s42 | UNCOMMITTED |

## I. Deployment footprint

| ID | Claim | Numbers | Source | Seeds | Status |
|---|---|---|---|---|---|
| I1 | Size | 96,605 trained parameters; fusion checkpoint 121 KB; per-agent counts (vitals 19,681; ECG 10,996; cadence 4,225; labs 3,713; fluid 3,009; resp 2,945; fusion 52,036) | T21 `results_fusion.md` l.1840–1860 | — | VERIFIED |
| I2 | Latency, 1 thread, 29 h context | Core Ultra 9 0.727 ms (CPU-only build); i5-13500HX 1.66 ms; i9-13900H (integrated graphics only) 1.99 ms; Apple M1 1.12 ms. Serving RSS 537 / 335 / 319 / 373 MB | `docs/blog.md` l.27–40; `docs/cpu_benchmark_matrix.md` §2 | — | VERIFIED (i9 and M1 run by teammates) |
| I3 | Load | Threads do not help at batch 1; 20-bed unit ≤ 40 ms CPU per hour; with cards ≈ 0.16% of one core | `docs/blog.md` l.42–48; matrix §6 | — | VERIFIED |
| I4 | Requirements | Minimum 2 cores / 8 GB / 2 GB disk (M1 rebuild peak 2.4 GB); recommended 8 cores / 16 GB; 24 GB with the 12B card model | `docs/blog.md` l.50–60 | — | VERIFIED |

## J. Explanations

| ID | Claim | Numbers | Source | Seeds | Status |
|---|---|---|---|---|---|
| J1 | Card faithfulness (200 cases) | gemma4:12b: top agent named 97%, top NEWS2 item 78%, invented drivers 0.13/card, alarm recoverable from text AUROC 0.836 (0.823 scrubbed). llama3.1:8b 0.71 / 0.40 / 0.27 / 0.776; qwen2.5-coder:14b 0.72 / 0.47 / 0.23 / 0.723; medgemma:4b 0.71 / 0.36 / 0.36 / 0.703 | T17 `results_fusion.md` l.1437–1536 | — | VERIFIED |
| J2 | Card cost on CPU | medgemma:4b 3.1 s, gemma4:12b 11.0 s (default threads, machine A); 2.0 / 10.1 s on machine B | T21; matrix §4 | — | VERIFIED |

## K. Limitations the paper must state

| ID | Limitation | Source |
|---|---|---|
| K1 | One development site; retrospective only; no prospective or silent deployment | — |
| K2 | Cadence counts orders and culture draws, i.e. clinician behaviour; discuss as a deliberate choice (T6/T7) and point to G1 (eICU result without cadence) | `src/modality_features.py` l.209–229; T6, T7 |
| K3 | Per-modality design not reviewed by a clinician; one clinician shaped deployment design (T12–T14) | `docs/references.md` §5 |
| K4 | Single-seed experiments: C1–C5, G8, H1, H2 | rows above |
| K5 | Hour-level AUPRC; patient-level results only at fixed budgets | — |
| K6 | Code default is still core-3 (`DEFAULT_CONFIG`); paper results use two-head ALL-8. Fix before releasing code | `src/orchestrator/fusion.py` |
| K7 | Untested design choices (E8) | — |

## Citations to add to `docs/references.bib`

Verify each entry before adding (title, venue, year). Add a matching register entry to `docs/references.md`.

- Royal College of Physicians, NEWS2 (2017)
- Pollard et al., eICU-CRD, *Scientific Data* 5, 180178 (2018)
- Bai, Kolter, Koltun, TCN vs RNN, arXiv:1803.01271 (2018)
- van den Oord et al., WaveNet, arXiv:1609.03499 (2016)
- Moor et al., sepsis TCN, MLHC 2019 / arXiv:1902.01659
- Lauritsen et al., *Nature Communications* 11, 3852 (2020)
- Chen and Guestrin, XGBoost, KDD 2016
- Bahdanau, Cho, Bengio, additive attention, ICLR 2015
- Zadrozny and Elkan, isotonic calibration, KDD 2002
- Rossetti et al., CONCERN trial, *Nature Medicine* 2025, doi:10.1038/s41591-025-03609-7
- TRIPOD+AI reporting guideline (Collins et al., BMJ 2024): verify
- Multi-agent LLM systems in medicine (e.g. MedAgents, MDAgents): find and verify
