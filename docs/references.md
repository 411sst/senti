# Sentinel — source register

Every external number that appears in `results_*.md`, in a figure, or in the
clinician-facing artifact must trace to an entry here. Each entry carries the
full citation, the URL, the date it was read, what we use it for, and **the
exact figures quoted**, so a claim can be checked without re-reading the paper.

Convention: numbers are reproduced as the source reports them (AUROC/AUPRC on
whatever scale that source uses — several benchmark papers report ×100). Where
a source's cohort or label differs from ours, the difference is stated in the
entry rather than left for the reader to infer.

---

## 1. Primary anchor — same label, same database

**Sadanandan, B.** *Multimodal Deep Learning for Early Prediction of Patient
Deterioration in the ICU: Integrating Time-Series EHR Data with Clinical Notes.*
arXiv:2603.14719v1 [cs.LG], 16 March 2026.
<https://arxiv.org/abs/2603.14719> — read 2026-08-26.

*Used for:* the literature-anchored XGBoost comparator (`spec_sadanandan` in
`src/lit_gbm.py`) and the "our GBM is not a strawman" argument. This is the only
paper found whose label is our label.

- Label: positive if **ICU mortality ∨ vasopressor initiation ∨ mechanical
  ventilation initiation** occurs in the next 24 h, evaluated hourly. Patients
  already on vasopressors or ventilation are not labelled positive for
  continuation.
- Cohort: MIMIC-IV (2008–2019); 94,459 → **74,822 ICU stays / 61,437 patients**;
  inclusion stays ≥ 24 h, age ≥ 18, vitals in first 6 h; excludes death within
  6 h. Samples drawn hours 6–48 with a 48 h look-back.
- Samples: **5,720,904** total — 4,019,343 train / 877,920 val / **823,641 test**,
  split at patient level. Positive rate **2.8 %**.
- Features: 26 structured (10 vitals: HR, arterial + non-invasive SBP and DBP,
  MAP, RR, SpO₂, temperature; 16 labs: lactate, creatinine, BUN, potassium,
  sodium, glucose, WBC, haemoglobin, haematocrit, platelets, bilirubin, albumin,
  pH, pCO₂, pO₂, bicarbonate) + ClinicalBERT radiology-note embedding + age, sex.
- Results (test, n = 823,641), AUROC / AUPRC:
  | model | AUROC | AUPRC | F1 | Brier |
  |---|---|---|---|---|
  | Multimodal (theirs) | 0.7857 | 0.1908 | 0.2584 | 0.0760 |
  | Structured-only | 0.7603 | 0.1371 | 0.1905 | 0.0768 |
  | **XGBoost** | **0.7486** | **0.1123** | 0.1684 | 0.1161 |
  | Logistic regression | 0.7171 | 0.0865 | 0.1508 | 0.2103 |
  | Text-only | 0.5271 | 0.0297 | 0.0626 | 0.1325 |
- Reports **no NEWS/NEWS2 baseline and no lead-time analysis**. Single operating
  point at threshold 0.47: recall 26.6 % (6,193 of 23,273 events), precision
  25.1 %, specificity 97.7 %.
- **Differences from our cohort:** they require stays ≥ 24 h and sample hours
  6–48 only; we require ≥ 12 h and score the whole stay, censoring at the first
  event. Their prevalence is 2.8 %, ours 4.74 %.
- *Re-checked 2026-11-19 against the v1 PDF for T49, with table numbers.*
  Cohort counts: Section 3.1 and Table 3 (94,459 ICU stays identified → 74,822
  stays, 61,437 patients). Samples and positives: Table 4 (5,720,904 samples,
  160,395 positive, 2.8 %; test 823,641 / 23,273). Model results: Table 5 (test
  AUROC column; Table 5 also lists a validation AUROC, e.g. 0.7914 for the
  multimodal model). Operating point: Table 6 (threshold 0.47; precision 25.1 %,
  recall 26.6 %, specificity 97.7 %; ECE 0.2129). Trained on six NVIDIA A100
  80 GB GPUs (Section 2.6). Event definitions (Section 2.3): vasopressors by
  `inputevents` itemids 221906, 221289, 221662, 221749, 222315; ventilation by
  `procedureevents` itemids 224385 (intubation) and 225792 (invasive
  ventilation); ICU death = hospital expiry flag with the death time inside the
  ICU stay. **Version note:** Section 2.1 says 2008–2019, but 94,459 stays is
  the MIMIC-IV v3.x ICU-stay count (v3.1: 94,458; v2.2: 73,181), so the cohort
  was probably drawn from v3.x. The paper does not state a version.
  Local copy: `~/code/paper-sources/sadanandan2026_arxiv_v1.pdf`.

## 2. Clinical-deployment anchor

**Churpek, M.M., et al.** *Multicenter Development and Prospective Validation of
eCARTv5: A Gradient-Boosted Machine-Learning Early Warning Score.*
Critical Care Explorations, 2025. PMID 38562803; PMC11949291.
<https://pmc.ncbi.nlm.nih.gov/articles/PMC11949291/> — read 2026-08-26;
re-validated 2026-09-02 (full text via Europe PMC) for the T19 redesign.

*Used for:* the `spec_ecart` feature philosophy, and — more importantly — the
**ranking-vs-earliness dissociation**, which this paper demonstrates for an
FDA-cleared model.

- Outcome: ward death or ICU transfer within 24 h of a score.
- Model: GBM via **discrete-time survival analysis**, time discretised into
  **8-hour blocks**; training data **down-sampled to 50 % outcome prevalence**;
  **no variable selection** ("research has shown that this may degrade the
  accuracy of tree-based algorithms"); **97 predictors** — patient
  characteristics (age, BMI), vitals, labs, time of day, nursing/RT
  documentation (delivered FiO₂, Braden scale), plus trends. Non-physiologic
  values treated as missing; last value pulled forward, but **blood gas and
  lactate expire after 24 h**. Library and hyperparameters not disclosed; the
  full variable list is in supplemental Table E2 (not web-accessible).
- Most important variables: max respiratory rate in prior 24 h, delivered FiO₂,
  min systolic BP in prior 24 h, heart rate.
- Cohorts: development 901,491 admissions (4.8 % event rate); retrospective
  validation **1,769,461 admissions / 132,873,833 observations** (4.8 %);
  prospective validation 205,946 admissions (4.2 %).
- AUROC, retrospective: **eCARTv5 0.834 (0.834–0.835)**, eCARTv2 0.775
  (0.775–0.776), NEWS 0.766 (0.766–0.767), MEWS 0.704 (0.703–0.704).
  Prospective: 0.828 / 0.768 / 0.767 / 0.709.
- Thresholds were **not volume-matched**: each score's "commonly used
  moderate (higher sensitivity) and high-risk (higher PPV) thresholds" were
  chosen post hoc — eCARTv5 ≥ 93 / ≥ 97, eCARTv2 ≥ 21 / ≥ 35, **NEWS ≥ 5 /
  ≥ 7**, MEWS ≥ 3 / ≥ 4.
- Operating points, eCARTv5: moderate (≥ 93) sensitivity 51.8 %, PPV 9.0 %;
  high (≥ 97) sensitivity 38.6 %, PPV 14.2 %. NEWS: moderate (≥ 5) 49.7 % /
  5.4 %; high (≥ 7) 28.0 % / 9.9 %. Specificity/alert rates not reported.
- Lead time = interval from the score's **first threshold crossing anywhere
  in the admission** to the event — unpaired, volume-unmatched, standing
  alarms counted (IQRs run 1–73 h).
- **Lead time at the moderate-risk threshold: NEWS 17 h (IQR 1–73), eCARTv5
  16 h (IQR 1–68), MEWS 13 h (IQR 1–66), eCARTv2 9 h (IQR 0–57).**
  eCARTv5 beats NEWS by 6.8 AUROC points and warns *no earlier*. This is
  Sentinel's central claim, stated by the comparator class's own validation.
- At the high-risk threshold the paper does claim earliness: eCARTv5 5 h
  (IQR 0–43) vs NEWS 3 h (0–40), MEWS 2 h, eCARTv2 2 h, p < 0.01 — but on
  different caught populations (sensitivity 38.6 % vs 28.0 %), unpaired,
  with standing alarms credited. Our harness (frozen thresholds, matched
  volume, jointly-caught pairs) would not admit that comparison as stated.
- Comparator is the **original NEWS, not NEWS2**: "choosing the SpO₂ scale
  in NEWS2 requires real-time clinician determination."

## 3. Benchmark anchors

**van de Water, R., Schmidt, H., Elbers, P., Thoral, P., Arnrich, B.,
Rockenschaub, P.** *Yet Another ICU Benchmark: A Flexible Multi-Center Framework
for Clinical ML.* arXiv:2306.05109; ICLR 2024 (confirm venue string before
submission). <https://arxiv.org/abs/2306.05109> — read 2026-08-26.
Config read from `rvandewater/YAIB`,
`configs/prediction_models/LGBMClassifier.gin`.

*Used for:* `spec_yaib` — the community-standard MIMIC-IV GBM recipe.

- Datasets: MIMIC-III, **MIMIC-IV**, eICU, HiRID, AUMCdb. Tasks: mortality (0–24 h,
  once per stay), AKI (hourly), sepsis (hourly), kidney function, length of stay.
- Features: **52 — 4 static (age, sex, height, weight) + 48 time series.**
  Forward-fill within stay, then sample-mean fill for values with no prior
  measurement, then zero-mean/unit-variance scaling. For classical ML only:
  **min, max, mean and count** generated per dynamic feature.
- LGBM search space (from the gin config): `max_depth` 3–7, `num_leaves`
  log-uniform 8–128 (base 2), `colsample_bytree` 0.33–1.0, `subsample`
  0.33–1.0; fixed `n_estimators = 100000`, `min_child_samples = 1000`,
  `subsample_freq = 1`. Tuning: 50 iterations of Bayesian optimisation for ML
  models (30 for DL).
- MIMIC-IV results, AUROC / AUPRC (×100):
  | task | LR | **LGBM** | GRU | LSTM | TCN | Transformer |
  |---|---|---|---|---|---|---|
  | Mortality | 86.1 / 39.7 | **87.7 / 44.2** | 87.6 / 42.8 | 86.7 / 41.0 | 87.1 / 41.4 | 86.9 / 42.2 |
  | AKI | 77.1 / 37.7 | **83.8 / 53.3** | 90.7 / 69.6 | 89.7 / 66.5 | 89.8 / 66.8 | 89.6 / 65.6 |
  | Sepsis | 77.1 / 4.6 | **77.5 / 5.9** | 83.6 / 9.1 | 82.0 / 8.0 | 82.7 / 8.8 | 80.0 / 6.6 |
- **Difference from ours:** none of these labels is our composite. This entry
  anchors the *method*, not the task.
- *Re-checked 2026-11-19 for T46 against arXiv:2306.05109v4 (19 March 2024),
  whose first page reads "Published as a conference paper at ICLR 2024" (the
  OpenReview PDF is behind a browser check).* All numbers ×100, mean ± SD over
  runs, not confidence intervals. **Table 3**, ICU mortality:
  | model | AUMCdb | HiRID | **eICU** | **MIMIC-IV** |
  |---|---|---|---|---|
  | LR | 83.7±0.6 / 52.9±1.2 | 84.0±0.3 / 36.9±1.1 | 84.8±0.2 / 33.0±0.7 | 86.1±0.1 / 39.7±0.6 |
  | LGBM | 84.5±0.5 / 53.7±1.2 | 84.4±0.3 / 40.6±0.8 | 85.7±0.2 / **36.0±0.6** | **87.7±0.2 / 44.2±0.7** |
  | GRU | 83.9±0.3 / 53.8±0.7 | 84.8±0.2 / 39.4±0.4 | **86.0±0.1** / 35.6±0.1 | 87.6±0.1 / 42.8±0.3 |
  | LSTM | 83.7±0.7 / 53.6±1.4 | 84.0±0.7 / 37.8±1.0 | 85.5±0.2 / 35.7±0.8 | 86.7±0.4 / 41.0±0.7 |
  | TCN | 84.0±0.6 / 54.2±1.4 | 84.6±0.7 / 39.2±1.3 | 85.4±0.2 / 34.3±0.6 | 87.1±0.3 / 41.4±0.8 |
  | TF | 84.1±0.2 / 54.4±1.1 | 84.9±0.7 / 39.3±1.5 | 85.9±0.2 / 34.7±0.8 | 86.9±0.3 / 42.2±0.3 |

  Best published eICU mortality AUROC: GRU 86.0. **Table 14**, cohorts: MIMIC-IV
  (listed as version **2.0**) 52,045 included stays, 3,779 died (7.3 %); eICU
  v2.0 113,382 stays, 6,253 died (5.5 %). Section 4: mortality uses data from
  hours 0–24; exclusions are age < 18, missing discharge time, < 6 h in the ICU,
  measurements in < 4 time bins, a gap of > 12 consecutive hours without a
  measurement, plus the task's own criteria. **Version note:** the paper lists
  MIMIC-IV 2.0, but the public cohort code (`rvandewater/YAIB-cohorts`, commit
  74ac699) pins `ricu` 0.6.0 (`prockenschaub/ricu-package@0969a52`), whose
  MIMIC-IV source is `physionet.org/files/mimiciv/2.2`. T46 rebuilds with the
  code as published, so on v2.2; its count check against 52,045 uses a 10 %
  tolerance for that reason (agreed 2026-11-19). The README also warns that
  later `ricu` fixes can make cohorts differ slightly from the paper's.
  Local copy: `~/code/paper-sources/yaib_arxiv_2306.05109v4.pdf`.

**Sheikhalishahi, S., Balaraman, V., Osmani, V.** *Benchmarking machine
learning models on multi-centre eICU critical care dataset.* PLOS ONE 15(7):
e0235424, 2 July 2020. DOI 10.1371/journal.pone.0235424 (peer-reviewed).
Also arXiv:1910.00964: v1 2 Oct 2019, v2 6 May 2020, **v3 5 Aug 2021**.
Read 2026-11-19. Code: `github.com/mostafaalishahi/eICU_Benchmark_updated`.
No correction notice was found for the PLOS ONE article.

*Used for:* T44 (decompensation) and T45 (in-hospital mortality) on eICU.

- Cohort (both versions, PLOS ONE Fig 1 and Table 3): eICU v2.0, 200,859 ICU
  stays → exclude age ≤ 18 or ≥ 89 and more than one ICU stay → 96,212 →
  exclude ≤ 15 or ≥ 200 records, missing gender or discharge status → 73,718
  stays, 4,564,844 records. Decompensation (LoS ≥ 24 h): **55,933 stays,
  2,800,711 records**, about 6.5 % positive (3,664 patients). Mortality (LoS ≥
  48 h): **30,680 stays, 1,164,966 records**. Evaluation: 5-fold
  cross-validation within eICU.
- **The two versions report different decompensation tasks.** PLOS ONE: a
  bidirectional LSTM labels every record of the stay (many-to-many), so each
  prediction can draw on later records of the same stay. arXiv v3: the past
  12 h ("derivation window") predicts on a 6-hour rolling window, with 95 % CIs.
- **PLOS ONE, Table 8** (decompensation, AUROC ×100 ± SD over folds / AUPRC):
  LR 67.63±5.89 / 16.53; ANN 80.59±0.60 / 22.86; **BiLSTM (num + cat, EMB)
  95.35±0.60 / 68.69**; BiLSTM cat only 86.82±0.70 / 36.34; BiLSTM num only
  95.15±0.16 / 68.28.
- **arXiv v3, Table 8** (decompensation, rolling window, 95 % CI): LR 78.99
  (77.06–80.95) / 22.78; **ANN 83.57 (82.31–84.87) / 27.86 (24.95–30.78)**;
  **BiLSTM 83.78 (81.86–85.73) / 30.79 (28.09–33.49)**.
- **PLOS ONE, Table 5** (in-hospital mortality, AUROC ± SD / AUPRC): first
  24 h — APACHE 77.30 / 41.23; LR 79.88±0.67 / 40.50; ANN 82.60±0.58 / 46.17;
  **BiLSTM (EMB) 83.80±0.42 / 48.72**; BiLSTM (OHE) 82.78±0.32 / 46.34. First
  48 h — LR 82.34±0.65 / 45.39; ANN 85.36±0.66 / 52.59; **BiLSTM (EMB)
  86.63±0.66 / 55.20**; BiLSTM (OHE) 84.96±0.63 / 51.63.
- **arXiv v3, Table 5** (mortality, 95 % CI): 24 h BiLSTM 83.70 (83.07–84.40)
  / 48.47; 48 h BiLSTM 86.55 (85.65–87.52) / 54.98; APACHE 77.30 / 41.23.
- **Which numbers the paper tests use (agreed 2026-11-19).** T45 uses PLOS ONE
  Table 5 (the task is the same in both versions), with v3 alongside. T44 uses
  arXiv v3's task and Table 8, labelled "arXiv v3, not peer-reviewed", with the
  PLOS ONE Table 8 result reported beside it: a model that sees later records
  of the stay is not comparable with a forward-only one.
  Local copies: `~/code/paper-sources/sheikhalishahi2020_plosone.pdf`,
  `sheikhalishahi_arxiv_1910.00964v{1,2,3}.pdf`.

**Yèche, H., Kuznetsova, R., Zimmermann, M., Hüser, M., Lyu, X., Faltys, M.,
Rätsch, G.** *HiRID-ICU-Benchmark — A Comprehensive Machine Learning Benchmark on
High-resolution ICU Data.* NeurIPS 2021 Datasets & Benchmarks Track;
arXiv:2111.08536. <https://arxiv.org/abs/2111.08536> — read 2026-08-26.

*Used for:* evidence that a tuned GBM is the bar to beat on ICU deterioration,
not a weak baseline — and the min/max/mean + measurement-density feature recipe.

- Tasks: circulatory failure (lactate > 2 mmol/L and either MAP < 65 mmHg or any
  vasoactive drug) and respiratory failure (P/F < 300, Berlin), both "onset in
  the next 12 h given not in failure now".
- Data: HiRID — 712 variables, 5-minute resolution, 33,905 admissions, 70/15/15.
- LGBM grid: depth (3, 4, 5, 6, 7); `colsample_bytree` (0.33, 0.66, 1.00);
  `subsample` (0.33, 0.66, 1.00). Hand-engineered features: min, max, mean of
  past values plus **measurement density** (proportion of non-empty timepoints).
- AUPRC: circulatory failure **LGBM-with-features 38.8 ± 0.2** vs GRU 36.8 ± 0.5;
  respiratory failure **LGBM 60.4 ± 0.2** vs GRU 59.2 ± 0.3; ICU mortality
  **LGBM 62.6 ± 0.0** vs Transformer 61.0 ± 0.8. "For both Circulatory and
  Respiratory Failure tasks, LGBM trained only on the current variables
  outperforms all deep learning methods."
- **Difference from ours:** HiRID, not MIMIC; failure-onset labels, not ours.

**Wang, S., McDermott, M.B.A., Chauhan, G., Ghassemi, M., Hughes, M.C.,
Naumann, T.** *MIMIC-Extract: A Data Extraction, Preprocessing, and
Representation Pipeline for MIMIC-III.* ACM CHIL 2020; arXiv:1907.08322.
<https://arxiv.org/abs/1907.08322> — read 2026-08-26.

*Used for:* the canonical framing of vent/vaso **intervention-onset** prediction,
the closest published task to our treatment arm.

- Task: 6 h observation window, 6 h gap, 4 h prediction window; four classes —
  Onset, Stay On, Wean, Stay Off — for mechanical ventilation and vasopressors.
- Cohort: adults ≥ 15, first ICU stay, total stay 12 h–10 days;
  **104 clinically aggregated time-series variables**.
- Baselines are LR, RF, GRU-D, CNN, LSTM — **no gradient-boosting baseline**
  (checked; this is why it anchors the task and not the comparator).
  Ventilator-onset AUROC spans 70.1–87.1 across models. In-ICU mortality
  AUROC / AUPRC: LR 88.7 / 46.4, RF 89.7 / 49.8, GRU-D 89.1 / 50.9.

## 4. Supporting

**Hyland, S.L., Faltys, M., Hüser, M., Lyu, X., Gumbsch, T., Esteban, C., Bock,
C., Horn, M., Moor, M., Rieck, B., Zimmermann, M., Bodenham, D., Borgwardt, K.,
Rätsch, G., Merz, T.M.** *Early prediction of circulatory failure in the
intensive care unit using machine learning.* Nature Medicine 26, 364–373 (2020);
preprint arXiv:1904.07990. <https://arxiv.org/abs/1904.07990> — read 2026-08-26.

- 240 patient-years of high-resolution ICU data; circulatory-failure prevalence
  **3.1 %**; predicts **90.0 %** of events, **81.8 % more than two hours in
  advance**; **AUROC 94.0 %, AUPRC 63.0 %**; externally validated.
- ⚠ The gradient-boosting library is not named in the abstract. **Verify in the
  full text before describing this as a GBM result.**

**Solaiman, K.M.A., Sebastian, J., Tobden, K.** *Benchmarking Early Deterioration
Prediction Across Hospital-Rich and MCI-Like Emergency Triage Under Constrained
Sensing.* arXiv:2602.20168v1 [cs.CY], 9 February 2026.
<https://arxiv.org/abs/2602.20168> — read 2026-08-26.

- MIMIC-IV v3.1 + MIMIC-IV-ED v2.2; 10,000 adults, first ED encounter,
  features restricted to the first hour after arrival; **default hyperparameters,
  no tuning**.
- In-hospital mortality (prevalence 12.10 %), AUROC / AUPRC: XGBoost
  0.809 ± 0.011 / 0.382 ± 0.026; LightGBM 0.777 ± 0.013 / 0.358 ± 0.016;
  TabNet 0.786 ± 0.026 / 0.334 ± 0.035; RF 0.771 ± 0.020 / 0.332 ± 0.037;
  LR 0.668 ± 0.020 / 0.197 ± 0.012.
- ICU transfer within 24 h (prevalence 45.10 %), XGBoost AUPRC:
  all features 0.815 ± 0.002 → vitals only 0.719 ± 0.001.
- **Difference from ours:** emergency-department triage, single first-hour
  snapshot, 45 % prevalence for the ICU-transfer label. Not comparable to an
  hourly in-ICU AUPRC; cited only for the tree-model-beats-linear pattern.

**Gao, S., Zhang, Y., Deng, M., Liu, H., Xu, W., Luo, M., Tian, Y., Zhang, B.**
*Machine learning-based early warning system for hemodynamic deterioration in
cardiovascular ICU patients: a bidirectional cross-validation study.*
Frontiers in Cardiovascular Medicine, 21 January 2026.
DOI 10.3389/fcvm.2025.1694001.
<https://pmc.ncbi.nlm.nih.gov/articles/PMC12868282/> — read 2026-08-26.

*Used for:* the counterweight in the ICU-generalization question — a
specialty-specific ICU model that transfers across databases.

- Cardiovascular ICU patients, ≥ 18, stay ≥ 24 h. MIMIC-IV 46,007 admissions /
  35,182 patients; eICU-CRD 50,949 / 41,512. 14 features.
- Random forest selected: MIMIC internal 0.870, eICU internal 0.883,
  **MIMIC→eICU 0.841, eICU→MIMIC 0.852 — degradation held under 4 %.**
- Algorithm comparison: SVM AUROC 0.934 / AUPRC 0.903; RF 0.914 / 0.878;
  XGBoost 0.907. Clinical scores: SOFA 0.681, APACHE II 0.747.
- Subgroups: female 0.841 vs male 0.848 (p = 0.045); age < 65 0.849 vs ≥ 65
  0.840 (p = 0.016); SOFA < 6 0.834 vs ≥ 6 0.866 (p < 0.001).

**Bukhari, S.A.C., Singh, A., Hossain, S., Wajahat, I.** *Early Prediction of
Multi-Label Care Escalation Triggers in the Intensive Care Unit Using Electronic
Health Records.* arXiv:2509.18145v1, 15 September 2025.
<https://arxiv.org/abs/2509.18145> — read 2026-08-26.

- MIMIC-IV, 85,242 ICU stays, features from the first 24 h, outcomes from hours
  24–72; four triggers (respiratory failure, haemodynamic instability, renal
  compromise, neurologic deterioration) via label-powerset into 16 classes.
- XGBoost average AUC 0.80 (RF 0.76); per-trigger F1 0.66 / 0.72 / 0.76 / 0.62.
  **No AUPRC reported**, and the triggers are threshold-crossings on vitals
  rather than clinical events — cited for framing only.

## 5. Primary (non-literature) sources

**Clinician consultation, 2026-08-26.** Notes provided by the reviewing medical
professional (`~/Downloads/direc.jpeg`). Verbatim points:

1. "CXR not always useful because anything non respiratory is not reflected in
   it. Say kidney and stuff you cannot see it early on."
2. "Camera will spoil doctor working env."
3. "Keep models on standby because ICU cannot be generalized."
4. "ICU: Cardio, neuro, respiratory, urology, burns."
5. "Camera audio can be kept far away, it can be useful but near the patient
   from a doctor POV it might actually harm the process of caring for the
   patient."

Points 3 and 4 drive the ICU-stratification work (`src/icu_strata.py`). Points
1, 2 and 5 are recorded as scoping constraints on future imaging/ambient
modalities and are not acted on here.

**MIMIC-IV v3.1**, Johnson, A.E.W., Bulgarelli, L., Shen, L., et al.
*MIMIC-IV, a freely accessible electronic health record dataset.*
Scientific Data 10, 1 (2023). Accessed via BigQuery `physionet-data.mimiciv_3_1_*`
under the PhysioNet credentialed-access DUA. No patient-level data appears in any
committed file or published artifact.


## 6. Methods and data cited in the paper draft

Added 2026-10-18 for `paper/main.tex`. These are cited for a method, a dataset
or a scoring standard, not for a number, so no figures are quoted. BibTeX keys
in brackets.

- **Royal College of Physicians**, *NEWS2* (2017) [`rcp2017news2`]. The scoring
  bands implemented in `src/news2.py` (SpO2 Scale 1).
- **Pollard et al.**, *The eICU Collaborative Research Database*, Scientific
  Data 5, 180178 (2018) [`pollard2018eicu`]. External validation cohort (T22).
- **Bai, Kolter, Koltun**, arXiv:1803.01271 (2018) [`bai2018tcn`] and **van den
  Oord et al.**, *WaveNet*, arXiv:1609.03499 (2016) [`oord2016wavenet`]. Causal,
  dilated temporal convolutions (vitals agent, fusion heads).
- **Moor et al.**, MLHC 2019 [`moor2019sepsis`] and **Lauritsen et al.**,
  Nature Communications 11, 3852 (2020) [`lauritsen2020xaiews`]. Prior TCNs on
  ICU and hospital EHR data.
- **Chen and Guestrin**, *XGBoost*, KDD 2016 [`chen2016xgboost`]. GBM agent and
  baselines.
- **Bahdanau, Cho, Bengio**, ICLR 2015 [`bahdanau2015attention`]. Additive
  attention over agents.
- **Zadrozny and Elkan**, KDD 2002 [`zadrozny2002calibration`]. Isotonic
  calibration of agent and head scores.
- **Rossetti et al.**, *Nature Medicine* (2025),
  doi:10.1038/s41591-025-03609-7 [`rossetti2025concern`]. The CONCERN trial:
  nursing documentation patterns as an early-warning signal; context for the
  cadence agent. Only the trial's design is cited, not its effect sizes.

---

## Where these are used

`results_fusion.md` T10 Part 1 (literature-anchored GBMs) draws on entries 1-5;
Part 2 (ICU-category strata) draws on entry 8 and the clinician notes in §5.
`src/lit_gbm.py` encodes entries 1, 2 and 3 as executable specifications, each
carrying its own list of deviations from the published description.

---

## Deliberately not cited

Web-search result snippets reporting a MIMIC-IV `first_careunit` distribution.
They describe a ~53,000-stay subset of unclear provenance and disagree with one
another. Our composition figures are computed from `icustays` directly
(`scripts/mimic_stay_meta.sql`) — see `results_fusion.md` T10.
