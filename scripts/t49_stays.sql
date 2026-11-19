-- T49: stay-level inputs for Sadanandan's cohort (arXiv:2603.14719v1,
-- Sections 2.2-2.3), MIMIC-IV 3.1 on BigQuery. One row per ICU stay, ALL stays:
-- every inclusion rule is applied afterwards in Python (src/sadanandan_cohort.py)
-- so each exclusion step can be counted. Times are hours from ICU intime.
--
-- Their vitals (Table 1): HR 220045; SBP 220050, 220179; DBP 220051, 220180;
-- MAP 220052, 220181; RR 220210; SpO2 220277; temperature 223761, 223762.
-- Their labs (Table 2): 50813, 50912, 51006, 50971, 50983, 50931, 51301, 51222,
-- 51221, 51265, 50885, 50862, 50820, 50818, 50821, 50882.
WITH st AS (
  SELECT i.stay_id, i.subject_id, i.hadm_id, i.intime, i.outtime, i.first_careunit,
         TIMESTAMP_DIFF(TIMESTAMP(i.outtime), TIMESTAMP(i.intime), MINUTE) / 60.0 AS los_h,
         EXTRACT(YEAR FROM i.intime) AS intime_year,
         p.anchor_age, p.anchor_year, p.anchor_year_group, p.gender,
         a.admittime, a.dischtime, a.deathtime, a.hospital_expire_flag
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  JOIN `physionet-data.mimiciv_3_1_hosp.patients` p USING (subject_id)
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a USING (hadm_id)
),
vit AS (
  SELECT s.stay_id,
         MIN(TIMESTAMP_DIFF(TIMESTAMP(c.charttime), TIMESTAMP(s.intime), MINUTE)) / 60.0
           AS first_vital_h,
         COUNT(*) AS n_vitals
  FROM st s
  JOIN `physionet-data.mimiciv_3_1_icu.chartevents` c USING (stay_id)
  WHERE c.itemid IN (220045, 220050, 220179, 220051, 220180, 220052, 220181,
                     220210, 220277, 223761, 223762)
    AND c.valuenum IS NOT NULL
    AND c.charttime >= s.intime AND c.charttime <= s.outtime
  GROUP BY s.stay_id
),
lab AS (
  SELECT s.stay_id,
         COUNTIF(l.charttime >= s.intime AND l.charttime <= s.outtime) AS n_labs_icu,
         COUNT(*) AS n_labs_hadm
  FROM st s
  JOIN `physionet-data.mimiciv_3_1_hosp.labevents` l ON l.hadm_id = s.hadm_id
  WHERE l.itemid IN (50813, 50912, 51006, 50971, 50983, 50931, 51301, 51222,
                     51221, 51265, 50885, 50862, 50820, 50818, 50821, 50882)
    AND l.valuenum IS NOT NULL
  GROUP BY s.stay_id
)
SELECT s.stay_id, s.subject_id, s.hadm_id, s.first_careunit, s.los_h,
       s.intime_year, s.anchor_age, s.anchor_year, s.anchor_year_group, s.gender,
       s.hospital_expire_flag,
       s.intime IS NULL OR s.outtime IS NULL AS icu_time_missing,
       s.admittime IS NULL OR s.dischtime IS NULL AS hosp_time_missing,
       TIMESTAMP_DIFF(TIMESTAMP(s.deathtime), TIMESTAMP(s.intime), MINUTE) / 60.0 AS death_h,
       v.first_vital_h, IFNULL(v.n_vitals, 0) AS n_vitals,
       IFNULL(l.n_labs_icu, 0) AS n_labs_icu, IFNULL(l.n_labs_hadm, 0) AS n_labs_hadm
FROM st s
LEFT JOIN vit v USING (stay_id)
LEFT JOIN lab l USING (stay_id)
