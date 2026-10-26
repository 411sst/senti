-- Stay-level metadata for ICU-category stratification (MIMIC-IV 3.1).
--
-- scripts/mimic_cohort.sql selects no stay-level metadata, so the local cohort
-- parquet cannot answer "which kind of ICU is this?". This extract adds one row
-- per stay_id with the three views needed, because the five categories a
-- clinician names (cardio / neuro / respiratory / urology / burns) are NOT all
-- ICU units in MIMIC-IV:
--
--   * care unit    -- cardiac (CCU, CVICU) and neuro units exist as places
--   * hospital service at ICU admission -- GU is urology, PSURG is plastics/burns
--   * diagnosis    -- respiratory and burns are diagnoses, not wards
--
-- No hourly data and no timestamps leave this query: it is stay-level only, and
-- every field is fixed at or before ICU admission (services is filtered to
-- transfertime <= intime), so nothing here can leak a future event.
WITH stays AS (
  SELECT i.stay_id, i.subject_id, i.hadm_id, i.intime,
         i.first_careunit, i.last_careunit,
         -- Wall-clock hour of ICU admission. The cohort's `hour` column counts
         -- hours since intime, so time-of-day (an eCARTv5 predictor) is only
         -- recoverable with this offset: (admit_hour + hour) % 24.
         EXTRACT(HOUR FROM i.intime) AS admit_hour_of_day
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  WHERE i.los >= 0.5                         -- identical inclusion to the cohort
),
-- Hospital service in effect when the patient entered the ICU: the last service
-- row transferred to at or before intime. Falls back to the earliest service row
-- for stays whose ICU admission precedes any services entry.
svc_prior AS (
  SELECT s.stay_id, sv.curr_service,
         ROW_NUMBER() OVER (PARTITION BY s.stay_id
                            ORDER BY sv.transfertime DESC) AS rn
  FROM stays s
  JOIN `physionet-data.mimiciv_3_1_hosp.services` sv USING (hadm_id)
  WHERE sv.transfertime <= s.intime
),
svc_any AS (
  SELECT s.stay_id, sv.curr_service,
         ROW_NUMBER() OVER (PARTITION BY s.stay_id
                            ORDER BY sv.transfertime ASC) AS rn
  FROM stays s
  JOIN `physionet-data.mimiciv_3_1_hosp.services` sv USING (hadm_id)
),
svc AS (
  SELECT s.stay_id,
         COALESCE(p.curr_service, a.curr_service) AS curr_service,
         p.curr_service IS NULL AS service_is_fallback
  FROM stays s
  LEFT JOIN (SELECT stay_id, curr_service FROM svc_prior WHERE rn = 1) p USING (stay_id)
  LEFT JOIN (SELECT stay_id, curr_service FROM svc_any   WHERE rn = 1) a USING (stay_id)
),
-- Primary diagnosis (seq_num = 1) for the admission.
dx1 AS (
  SELECT s.stay_id, d.icd_code, d.icd_version, dd.long_title
  FROM stays s
  JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d USING (hadm_id)
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses` dd
         ON dd.icd_code = d.icd_code AND dd.icd_version = d.icd_version
  WHERE d.seq_num = 1
),
-- Any-position diagnosis flags. Burns and respiratory failure are rarely the
-- primary code, so the primary-diagnosis view alone would undercount them.
dxflags AS (
  SELECT s.stay_id,
    MAX(CASE WHEN (d.icd_version = 10 AND REGEXP_CONTAINS(d.icd_code, r'^T2[0-9]|^T3[0-2]'))
              OR (d.icd_version = 9  AND REGEXP_CONTAINS(d.icd_code, r'^94[0-9]'))
             THEN 1 ELSE 0 END) AS dx_burn,
    MAX(CASE WHEN (d.icd_version = 10 AND REGEXP_CONTAINS(d.icd_code, r'^J'))
              OR (d.icd_version = 9  AND REGEXP_CONTAINS(d.icd_code, r'^4[6-9][0-9]|^5[0-1][0-9]'))
             THEN 1 ELSE 0 END) AS dx_resp,
    MAX(CASE WHEN (d.icd_version = 10 AND REGEXP_CONTAINS(d.icd_code, r'^N[0-3]|^N4[0-9]|^N5[0-3]'))
              OR (d.icd_version = 9  AND REGEXP_CONTAINS(d.icd_code, r'^5[89][0-9]|^6[0-2][0-9]'))
             THEN 1 ELSE 0 END) AS dx_gu,
    MAX(CASE WHEN (d.icd_version = 10 AND REGEXP_CONTAINS(d.icd_code, r'^I'))
              OR (d.icd_version = 9  AND REGEXP_CONTAINS(d.icd_code, r'^39[0-9]|^4[0-5][0-9]'))
             THEN 1 ELSE 0 END) AS dx_cardio,
    MAX(CASE WHEN (d.icd_version = 10 AND REGEXP_CONTAINS(d.icd_code, r'^G|^I6'))
              OR (d.icd_version = 9  AND REGEXP_CONTAINS(d.icd_code, r'^3[2-5][0-9]|^43[0-8]'))
             THEN 1 ELSE 0 END) AS dx_neuro,
    -- Condition flags for the condition->agent roster (T13). Any-position ICD:
    -- these label WHAT THE STAY WAS, for retrospective stratification -- they
    -- are discharge-coded and must never become a real-time selector input.
    MAX(CASE WHEN (d.icd_version = 10 AND REGEXP_CONTAINS(d.icd_code, r'^A4[01]|^R652'))
              OR (d.icd_version = 9  AND REGEXP_CONTAINS(d.icd_code, r'^038|^99591|^99592|^78552'))
             THEN 1 ELSE 0 END) AS dx_sepsis,
    MAX(CASE WHEN (d.icd_version = 10 AND REGEXP_CONTAINS(d.icd_code, r'^N17'))
              OR (d.icd_version = 9  AND REGEXP_CONTAINS(d.icd_code, r'^584'))
             THEN 1 ELSE 0 END) AS dx_aki,
    MAX(CASE WHEN (d.icd_version = 10 AND REGEXP_CONTAINS(d.icd_code, r'^I50'))
              OR (d.icd_version = 9  AND REGEXP_CONTAINS(d.icd_code, r'^428'))
             THEN 1 ELSE 0 END) AS dx_hf,
    MAX(CASE WHEN (d.icd_version = 10 AND REGEXP_CONTAINS(d.icd_code, r'^J1[2-8]'))
              OR (d.icd_version = 9  AND REGEXP_CONTAINS(d.icd_code, r'^48[0-6]'))
             THEN 1 ELSE 0 END) AS dx_pneumonia,
    MAX(CASE WHEN (d.icd_version = 10 AND REGEXP_CONTAINS(d.icd_code, r'^J96'))
              OR (d.icd_version = 9  AND REGEXP_CONTAINS(d.icd_code, r'^5188[124]'))
             THEN 1 ELSE 0 END) AS dx_respfail,
    MAX(CASE WHEN (d.icd_version = 10 AND REGEXP_CONTAINS(d.icd_code, r'^K92[012]|^K2[5-8]'))
              OR (d.icd_version = 9  AND REGEXP_CONTAINS(d.icd_code, r'^578'))
             THEN 1 ELSE 0 END) AS dx_gibleed,
    COUNT(*) AS n_diagnoses
  FROM stays s
  JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d USING (hadm_id)
  GROUP BY 1
)
SELECT
  s.stay_id, s.subject_id,
  s.admit_hour_of_day,
  p.gender, p.anchor_age,
  s.first_careunit, s.last_careunit,
  a.admission_type, a.admission_location,
  v.curr_service, v.service_is_fallback,
  x.icd_code AS dx1_code, x.icd_version AS dx1_version, x.long_title AS dx1_title,
  f.dx_burn, f.dx_resp, f.dx_gu, f.dx_cardio, f.dx_neuro,
  f.dx_sepsis, f.dx_aki, f.dx_hf, f.dx_pneumonia, f.dx_respfail, f.dx_gibleed,
  f.n_diagnoses
FROM stays s
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.patients` p USING (subject_id)
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a USING (hadm_id)
LEFT JOIN svc     v USING (stay_id)
LEFT JOIN dx1     x USING (stay_id)
LEFT JOIN dxflags f USING (stay_id)
