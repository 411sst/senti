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
