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
| B2 | Beats the equal-information GBM | FAIR+TRAJ (145 features) 0.287 [0.264, 0.311], AUROC 0.807; paired CI clear of zero | `results_fusion.md` l.39–41, 134 | 10/10 | VERIFIED; "equal-information" holds only for its 145 columns, see B7 |
| B3 | Published-spec rebuilds on our cohort | eCART-style 0.280 [0.257, 0.304] / 0.814; YAIB LightGBM 0.281 [0.258, 0.306] / 0.808; Sadanandan XGBoost 0.239 [0.220, 0.261] / 0.766; NEWS2 0.114 [0.102, 0.128] / 0.683. Feature counts 95 / 218 / ~26+age, sex | T10 `results_fusion.md` l.736–766; deviations in `src/lit_gbm.py` | comparators seed-fixed; ten-seed refits in B8 | VERIFIED |
| B4 | vs eCART-style | ΔAUPRC +0.0287 [+0.0252, +0.0323] | T19 `results_fusion.md` l.1676 | 10/10 | VERIFIED |
| B5 | Mortality head, scored on death | AUPRC 0.340 [0.288, 0.415], AUROC 0.946 [0.928, 0.960]; GBM 0.189 on death | T9, T18 `results_fusion.md` l.43, 158 | 10 | VERIFIED |
| B6 | Standalone agents (AUPRC / AUROC) | gbm 0.279/0.806; vitals 0.189/0.754; labs 0.166/0.727; cadence 0.156/0.703; resp 0.134/0.712; fluid 0.093/0.654; ECG 0.055/0.539 | `results_fusion.md` l.160–163 | — | VERIFIED |
| B7 | **Does not beat an all-features GBM on ranking (pre-registered, failed)** | One XGBoost on all eight agents' inputs (262 columns): AUPRC 0.3268, AUROC 0.8363 vs flagship 0.3087 / 0.8225. Pooled ΔAUPRC (flagship − GBM) −0.0181 [−0.0282, −0.0082], CI below zero in 10/10 seeds; ΔAUROC −0.0138 [−0.0182, −0.0090]. Without cadence (230): GBM 0.3218, Δ −0.0131 [−0.0233, −0.0031]. GBM lead is on vent/vaso; on death the flagship is ahead (+0.0259 [+0.0039, +0.0473]; D head 0.340 vs GBM 0.154) | T30 `results_fusion.md`; `scripts/t30_allfeat_gbm.py`; `data/cache/t30_allfeat_gbm.json` | flagship 10; GBM s42 | BRANCH (`paper-tests`) |
| B8 | **Seed-paired against every baseline (pre-registered A2: 4 of 5 pass)** | Each baseline refit at seeds 42, 1–9 (seed-42 refits exactly equal the cached fits) and paired with flagship seed s. Pooled ΔAUPRC (flagship − baseline): FAIR+TRAJ +0.0224 [+0.0121, +0.0333], Sadanandan +0.0725 [+0.0570, +0.0885], eCART-style +0.0302 [+0.0192, +0.0422], YAIB +0.0330 [+0.0194, +0.0466], each CI clear in 10/10 pairs; all-features GBM −0.0195 [−0.0298, −0.0102], CI below zero in 10/10 pairs. Baseline ten-seed AUPRC 0.2864 / 0.2363 / 0.2785 / 0.2757 / 0.3282 (SD 0.0014–0.0029; flagship SD 0.0049) | T31 `results_fusion.md`; `scripts/t31_baseline_seeds.py`; `data/cache/t31_baseline_seeds.json` | 10 × 10 paired | BRANCH (`paper-tests`) |

## C. Patients caught at a fixed alarm budget

| ID | Claim | Numbers | Source | Seeds | Status |
|---|---|---|---|---|---|
| C1 | Catch rate at the NEWS2-matched budget | Of 1,540 event stays: Sentinel 65.5% (1,008), eCART-style 53.6% (825), YAIB 51.8% (798). Caught ≥6 h early: 34.9 / 28.2 / 27.1% | `data/cache/demo/cohort.json` `h2h.funnel`, built by `scripts/build_demo_bundle.py` | s42 | CACHE |
| C2 | **Realized test alarm rates are not equal** | Budget set on V_freeze (matched 2.617%); realized on test: Sentinel 2.79%, eCART-style 2.56%, YAIB 2.43%. Paper must report these. The sweep (C4) shows eCART-style at a 3% budget still catches only 55.8% | `cohort.json` `h2h.alarm_rate_test` | s42 | CACHE; **must be stated in the paper** |
| C3 | Cost | Quiet stays with ≥1 alarm: 11.8 / 6.8 / 6.2%. Stay-level precision: 55.5 / 63.9 / 65.5% | `cohort.json` `h2h.cost` | s42 | CACHE |
| C4 | Budget sweep, 0.5–10% (10 seeds) | Sentinel mean catch 34.4 / 46.0 / 58.6 / 66.3 / 74.6 / 82.0 / 86.5% at 0.5 / 1 / 2 / 3 / 5 / 7.5 / 10%. Best comparator per budget: Sadanandan 28.8 / 38.1 / 50.3 / 56.9 / 66.5% (≤5%), eCART-style 73.6 / 79.7% (7.5, 10%). Gap over the best: +5.6 / +8.0 / +8.4 / +9.4 / +8.1 / +8.4 / +6.8 points. Realized test alarm rate slightly higher for Sentinel (2.14% vs 1.84–2.06% at 2%); Sentinel at 2% (2.14%) still beats Sadanandan at 3% (3.02%): 58.6 vs 56.9% | T28 `results_fusion.md` l.2570+; `scripts/paper_operating_points.py`; seed-42 values reproduce the demo bundle | 10 | VERIFIED |
| C5 | First-alarm race (jointly caught stays) | vs eCART-style: Sentinel first 214, same hour 470, eCART first 78 (of 762). vs YAIB: 191 / 458 / 94 (of 743) | `cohort.json` `h2h.race` | s42 | CACHE |
| C7 | Catches are not just first-hour flags | Caught by an alarm AFTER the first scored hour (first-hour alarms ignored), 2% budget: Sentinel 53.3% vs Sadanandan 48.1%, eCART-style 43.8%, YAIB 42.7%; Sentinel ahead at every budget. (The earlier 'mid-stay' count, first alarm > first hour, penalised models that flag at hour 6 and is not a fair comparison) | T28 `results_fusion.md` l.2570+; `scripts/paper_operating_points.py` | 10 | VERIFIED |
| C8 | Lead over NEWS2 at equal realized volume | NEWS2 alarms 1.03 / 2.28 / 4.71% of test hours at 1 / 2 / 5% budgets; Sentinel at protocol thresholds 1.36 / 2.76 / 5.21%. Re-thresholded to NEWS2's exact volume: 2% first-when-differ 69 → 62%, median lead +0.95 → +0.5 h, sign test 10/10 → 9/10 seeds; 5% 73 → 71%, 10/10; 1% 55 → 51%, 0/10. The lead shrinks but survives at 2% and 5% | T28 `results_fusion.md` l.2570+; `scripts/paper_operating_points.py`; diagnostic (test-score quantiles), not protocol | 10 | VERIFIED; **report alongside F1** |
| C6 | Race vs comparators at the 2% NEWS2-matched volume | First alarm when the two differ (mean over 10 seeds), protocol → equal realized volume: eCART-style 70 → 67%, YAIB 66 → 59%, Sadanandan 69 → 68%. Sign test significant at equal volume in 9 / 8 / 10 of 10 seeds. Median lead 0 h (most jointly caught stays tie) | T28 `results_fusion.md` l.2570+; `scripts/paper_operating_points.py` | 10 | VERIFIED |
| C9 | Catches vs the all-features GBM, and the cost | Flagship catches more event stays at every budget: 10/10 seeds at protocol thresholds; at the GBM's realized volume the ten-seed mean is ahead at every budget in 8–10 of 10 seeds (corrected 2026-11-18): 2% 58.6% (57.6% at equal volume) vs 53.9%; ≥6 h 29.4 vs 25.6%. But more quiet stays alarmed: 7.5 vs 5.3% at 2% equal volume; stay-level precision 63.3 vs 69.5% (exploratory). Races: median lead 0 h; first-when-differ at equal volume 53 / 56 / 70% at 1 / 2 / 5% | T30 `results_fusion.md`; `data/cache/t30_allfeat_gbm.json` | flagship 10; GBM s42 | BRANCH (`paper-tests`) |

## D. How agents communicate (the thesis claim)

| ID | Claim | Numbers | Source | Seeds | Status |
|---|---|---|---|---|---|
| D1 | LLM-as-fuser text arm loses | 150 stays, 6,185 hours, 4.41% positive; llama3.1:8b, 4 h window, temp 0. AUPRC text 0.100 raw / 0.090 calibrated vs latent 0.244; AUROC 0.732 vs 0.787; death component 0.025 vs 0.193 | T20 `results_fusion.md` l.1798–1827 | 1 (core-3, subset) | VERIFIED |
| D2 | T27: latents add nothing to ranking | Two-head ALL-8, 3 seeds. LATENT 0.3075 (mean) vs RISK-ONLY 0.3043; pooled Δ −0.0032 [−0.0099, +0.0033]. MEAN-RISK (untrained mean of the 8 calibrated risks) 0.3076. HARD-VOTE 0.1897 (9 distinct scores; overshoots budget, 3.89% at 2%) | T27 `results_fusion.md` l.2462+; `scripts/channel_flagship.py` | 3 | VERIFIED |
| D3 | T27: latents buy earliness | Lead vs NEWS2 @2%: LATENT +1.0 to +2.0 h, win 57–59% (CIs clear of 50%); RISK-ONLY and MEAN-RISK +0.0 h, 42–48%. Head-to-head at equal test volume: LATENT first more often in 9/9 cells (7/9 significant), catches more stays the other misses in 8/9; strong at seeds 1, 2, marginal at 42 | T27 | 3 | VERIFIED (head-to-head uses test-quantile thresholds: diagnostic, not protocol) |
| D4 | Text arm on the flagship | ClinicalBERT sentence channel through the same fusion; the remaining test of latent vs text | not run | — | OPEN (optional; ~half day code + 2–4 h compute) |

## E. Architecture choices and ablations

| ID | Claim | Numbers | Source | Seeds | Status |
|---|---|---|---|---|---|
| E1 | Adding agents | ALL-7 0.282; ALL-8 0.306; two-head 0.309 | T6, T8, T9; `results_fusion.md` l.152–157 | 3–10 | VERIFIED |
| E2 | ECG: machine read beats waveform models | 6 waveform variants incl. ECG-FM (90.9M) and ECGFounder (30.7M); machine-read fusion Δ +0.0049 (3/3 seeds); ECG-FM −0.0385 | `results_ecg.md` grand table | 3 | VERIFIED (core-3 era) |
| E3 | One model beats specialists | 18/18 specialist cells below the generalist, 9/18 CI clear; cardiac 0.353 vs 0.340/0.337; neuro 0.187 vs 0.168/0.171; respiratory 0.310 vs 0.244/0.250 | T14 `results_fusion.md` l.1190–1272 | 3 | VERIFIED |
| E4 | Per-unit thresholds | Alarm-burden spread across ICU types 5.3× → 1.7× | T12 `results_fusion.md` l.1071, 1138 | 5 | VERIFIED |
| E5 | Condition → top agents | Sepsis/pneumonia lean on resp; AKI on cadence; GI bleed on labs (lift 3.9) | T13 `results_fusion.md` l.1142–1188 | — | VERIFIED |
| E6 | YAIB TCN swap (vitals agent) | Inside flagship ΔAUPRC −0.0045 [−0.0076, −0.0013], lower in 9/10 seeds; agent alone −0.0036 (CI spans 0); 914k vs 19,681 params; receptive field 55 h vs 29 h; 16 random draws of YAIB's search space | T26 `results_fusion.md` l.2392+; `scripts/yaib_tcn_swap.py`, `src/agents/yaib_tcn.py`, `tests/test_yaib_tcn.py` | 10 | VERIFIED |
| E7 | Notes and other candidate agents did not ship | core-3+notes val Δ −0.0038 [−0.0081, +0.0007]; neuro-status and infection agents benched | `src/orchestrator/fusion.py` l.51; T15 | 3–5 | CHECK: cite the results rows, not code comments |
| E8 | Untested design choices | TCN depth/dilations/kernel; per-agent sequence models; MLP widths; hyperparameter search | none in any branch | — | OPEN: ablate, or list as a limitation |

## F. Timing

| ID | Claim | Numbers | Source | Seeds | Status |
|---|---|---|---|---|---|
| F1 | Lead over NEWS2 | Median lead @2% +0.95 h [+0.52, +1.38], positive 8/10; earlier in 68.5% [64.7, 72.4] of stays where alarms differ; ~70% of ≥12 h leads are first-hour flags | T18 `results_fusion.md` l.49–58, 1572 | 10 | VERIFIED |
| F2 | Death stays | On 68 death stays with no documented care-limitation decision: +0.5 h, 50% win | T16 `results_fusion.md` l.1413–1414 | — | VERIFIED |
| F3 | Treatment-only supervision | Death-first stays lead +3 / +3 / +4 h | T7 Part ii | 3 | VERIFIED |
| F4 | ≥6 h catches at equal realized volume (pre-registered L1, FAILED at 0.5% and at 1% vs Sadanandan) | Flagship re-thresholded to each comparator's test volume, ≥6 h catch of 1,540 event stays: at 2% 27.8 / 26.6 / 28.8% vs eCART-style 24.0, YAIB 23.5, Sadanandan 24.7% (Δ +3.9 / +3.1 / +4.0 pts, CIs clear); at 5% Δ +4.1 / +4.8 / +4.3; at 1% vs Sadanandan +0.8 [−0.7, +2.4]; at 0.5% behind eCART-style (−0.7) and Sadanandan (−0.9). Ignoring first-hour alarms the lead holds only from ~2–3%. Warning times among caught stays match the comparators' | T33 `results_fusion.md`; `scripts/paper_operating_points.py --early`; `data/cache/paper_early_catches.json` | flagship 10; comparators single fits | BRANCH (`paper-tests`) |

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
| H2 | Vitals-only real time (T25) | TCN alone AUPRC 0.188 vs 0.314; real-time TCN alarms on 25.7% of quiet stays vs 10.8%; TCN-direct at matched false-alarm cost catches 55.5% vs 64.6%, first alarm −239 min [−336, −142]; 81% of TCN catches at first scored hour | T25 `origin/live-version:results_live.md` l.212+; `scripts/live_tcn_direct.py` | s42 | BRANCH |

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
| K8 | A single GBM given all the agents' inputs ranks hours better than the fusion (B7). The fusion's measured gains are death ranking and stay-level catch, the latter at lower stay-level precision (C9) | T30 |

## Citations

Added to `docs/references.bib` (with register entries in `docs/references.md` §6) on 2026-10-18:
RCP NEWS2 (`rcp2017news2`), Pollard eICU (`pollard2018eicu`), Bai TCN (`bai2018tcn`),
WaveNet (`oord2016wavenet`), Moor (`moor2019sepsis`), Lauritsen (`lauritsen2020xaiews`),
XGBoost (`chen2016xgboost`), Bahdanau (`bahdanau2015attention`), Zadrozny and Elkan
(`zadrozny2002calibration`), CONCERN (`rossetti2025concern`).

The register's own notes on the older entries were moved from `note` to `annote`, so they
no longer print in the reference list.

Still to find and verify before Related work:
- TRIPOD+AI reporting guideline (Collins et al., BMJ 2024)
- Multi-agent LLM systems in medicine (e.g. MedAgents, MDAgents)
- Full author lists for `lauritsen2020xaiews` and `rossetti2025concern` (currently "and others")
