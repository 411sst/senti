-- T49 bridge: hourly bins of Sadanandan's 26 structured inputs, MIMIC-IV 3.1.
--
-- Items exactly as their code (github.com/thedatasense/Patient_Early_Deterioration_Risk_Prediction
-- at 785d8d7, src/data/vitals.py and src/data/labs.py): 10 chartevents vitals and
-- 16 labevents labs, values recorded within the ICU stay. One row per (stay,
-- hour) that has any value; hour = floor(hours from ICU intime).
--
-- Vitals: hourly mean, binned by charttime, as their code (and as our cohort
-- extract bins vitals). Labs: their code bins by charttime, which hands a model
-- a result before it exists; our leakage rule needs the time the result was
-- recorded, so labs are binned by GREATEST(storetime, charttime) exactly as
-- scripts/mimic_labs.sql does (NULL storetime dropped), keeping their window
-- of charttime within [intime, outtime]; within a bin the last value by
-- charttime is kept (their code keeps the last value in the hour).
-- The strictly-before-t shift is applied later in Python.
-- All stays of at least 24 h: a superset of their cohort.
WITH st AS (
  SELECT stay_id, hadm_id, intime, outtime
  FROM `physionet-data.mimiciv_3_1_icu.icustays`
  WHERE TIMESTAMP_DIFF(TIMESTAMP(outtime), TIMESTAMP(intime), MINUTE) >= 24 * 60
),
vit AS (
  SELECT s.stay_id,
         DIV(TIMESTAMP_DIFF(TIMESTAMP(c.charttime), TIMESTAMP(s.intime), SECOND), 3600) AS hour,
         AVG(IF(c.itemid = 220045, c.valuenum, NULL)) AS heart_rate,
         AVG(IF(c.itemid = 220050, c.valuenum, NULL)) AS abp_systolic,
         AVG(IF(c.itemid = 220051, c.valuenum, NULL)) AS abp_diastolic,
         AVG(IF(c.itemid = 220052, c.valuenum, NULL)) AS abp_mean,
         AVG(IF(c.itemid = 220179, c.valuenum, NULL)) AS nibp_systolic,
         AVG(IF(c.itemid = 220180, c.valuenum, NULL)) AS nibp_diastolic,
         AVG(IF(c.itemid = 220210, c.valuenum, NULL)) AS respiratory_rate,
         AVG(IF(c.itemid = 223761, c.valuenum, NULL)) AS temperature_f,
         AVG(IF(c.itemid = 223762, c.valuenum, NULL)) AS temperature_c,
         AVG(IF(c.itemid = 220277, c.valuenum, NULL)) AS spo2
  FROM st s
  JOIN `physionet-data.mimiciv_3_1_icu.chartevents` c USING (stay_id)
  WHERE c.itemid IN (220045, 220050, 220051, 220052, 220179, 220180, 220210,
                     223761, 223762, 220277)
    AND c.valuenum IS NOT NULL
    AND c.charttime >= s.intime AND c.charttime <= s.outtime
  GROUP BY 1, 2
),
lab_rows AS (
  SELECT s.stay_id, l.itemid, l.valuenum, l.charttime,
         DIV(TIMESTAMP_DIFF(TIMESTAMP(GREATEST(l.storetime, l.charttime)),
                            TIMESTAMP(s.intime), SECOND), 3600) AS hour
  FROM st s
  JOIN `physionet-data.mimiciv_3_1_hosp.labevents` l ON l.hadm_id = s.hadm_id
  WHERE l.itemid IN (50813, 50820, 50821, 50818, 50912, 50971, 50983, 50882,
                     50893, 51301, 51222, 51265, 51237, 51275, 50885, 50931)
    AND l.valuenum IS NOT NULL
    AND l.storetime IS NOT NULL
    AND l.charttime >= s.intime AND l.charttime <= s.outtime
),
lab AS (
  SELECT stay_id, hour,
    ARRAY_AGG(IF(itemid = 50813, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS lactate,
    ARRAY_AGG(IF(itemid = 50820, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS ph,
    ARRAY_AGG(IF(itemid = 50821, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS po2,
    ARRAY_AGG(IF(itemid = 50818, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS pco2,
    ARRAY_AGG(IF(itemid = 50912, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS creatinine,
    ARRAY_AGG(IF(itemid = 50971, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS potassium,
    ARRAY_AGG(IF(itemid = 50983, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS sodium,
    ARRAY_AGG(IF(itemid = 50882, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS bicarbonate,
    ARRAY_AGG(IF(itemid = 50893, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS calcium,
    ARRAY_AGG(IF(itemid = 51301, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS wbc,
    ARRAY_AGG(IF(itemid = 51222, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS hemoglobin,
    ARRAY_AGG(IF(itemid = 51265, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS platelets,
    ARRAY_AGG(IF(itemid = 51237, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS inr,
    ARRAY_AGG(IF(itemid = 51275, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS ptt,
    ARRAY_AGG(IF(itemid = 50885, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS bilirubin_total,
    ARRAY_AGG(IF(itemid = 50931, valuenum, NULL) IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS glucose
  FROM lab_rows
  GROUP BY 1, 2
)
SELECT COALESCE(v.stay_id, l.stay_id) AS stay_id, COALESCE(v.hour, l.hour) AS hour,
       v.* EXCEPT (stay_id, hour), l.* EXCEPT (stay_id, hour)
FROM vit v
FULL OUTER JOIN lab l ON l.stay_id = v.stay_id AND l.hour = v.hour
