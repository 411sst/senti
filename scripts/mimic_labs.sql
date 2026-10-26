-- Hourly lab panel for the in-ICU cohort (MIMIC-IV 3.1, BigQuery).
--
-- LEAKAGE-CRITICAL DESIGN DECISION: labs are binned by **storetime** (when the
-- result became available in the system), NOT charttime (when the specimen was
-- drawn). The median gap between the two is 70 minutes (p75 = 131 min), so
-- binning by charttime would hand the model a result an hour or more before any
-- clinician could have seen it -- a leak of exactly the kind CLAUDE.md rule #1
-- forbids. Rows with a NULL storetime (1.6%) are dropped rather than guessed at,
-- and the handful where storetime < charttime are pushed to charttime.
--
-- Hours are offsets from ICU intime and may be NEGATIVE (down to -24): labs
-- drawn in the ED before ICU admission are legitimate history at hour 6, so the
-- window opens a day early. Feature construction (LOCF, shift, staleness
-- bounds) happens in Python, as for vitals.
WITH stays AS (
  SELECT stay_id, hadm_id, intime, outtime
  FROM `physionet-data.mimiciv_3_1_icu.icustays`
  WHERE los >= 0.5
),
panel AS (
  SELECT * FROM UNNEST([
    STRUCT(50912 AS itemid, 'creatinine' AS name), (51006,'bun'),
    (50983,'sodium'), (50971,'potassium'), (50902,'chloride'),
    (50882,'bicarbonate'), (50868,'anion_gap'), (50931,'glucose'),
    (50893,'calcium'), (50960,'magnesium'), (50970,'phosphate'),
    (51221,'hematocrit'), (51222,'hemoglobin'), (51301,'wbc'),
    (51265,'platelets'), (51277,'rdw'),
    (50861,'alt'), (50878,'ast'), (50885,'bilirubin'), (50862,'albumin'),
    (51237,'inr'), (51275,'ptt'),
    (50813,'lactate'), (50820,'ph')
  ])
),
observed AS (
  SELECT s.stay_id, p.name,
         -- availability time: the result exists only once it is stored
         TIMESTAMP_DIFF(
           TIMESTAMP(GREATEST(l.storetime, l.charttime)),
           TIMESTAMP(s.intime), HOUR) AS hour,
         l.valuenum AS value
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
  JOIN panel p USING (itemid)
  JOIN stays s ON s.hadm_id = l.hadm_id
  WHERE l.valuenum IS NOT NULL
    AND l.storetime IS NOT NULL
    AND GREATEST(l.storetime, l.charttime) <= s.outtime
    AND GREATEST(l.storetime, l.charttime) >= DATETIME_SUB(s.intime, INTERVAL 24 HOUR)
)
SELECT stay_id, hour,
  AVG(IF(name='creatinine',  value, NULL)) AS lab_creatinine,
  AVG(IF(name='bun',         value, NULL)) AS lab_bun,
  AVG(IF(name='sodium',      value, NULL)) AS lab_sodium,
  AVG(IF(name='potassium',   value, NULL)) AS lab_potassium,
  AVG(IF(name='chloride',    value, NULL)) AS lab_chloride,
  AVG(IF(name='bicarbonate', value, NULL)) AS lab_bicarbonate,
  AVG(IF(name='anion_gap',   value, NULL)) AS lab_anion_gap,
  AVG(IF(name='glucose',     value, NULL)) AS lab_glucose,
  AVG(IF(name='calcium',     value, NULL)) AS lab_calcium,
  AVG(IF(name='magnesium',   value, NULL)) AS lab_magnesium,
  AVG(IF(name='phosphate',   value, NULL)) AS lab_phosphate,
  AVG(IF(name='hematocrit',  value, NULL)) AS lab_hematocrit,
  AVG(IF(name='hemoglobin',  value, NULL)) AS lab_hemoglobin,
  AVG(IF(name='wbc',         value, NULL)) AS lab_wbc,
  AVG(IF(name='platelets',   value, NULL)) AS lab_platelets,
  AVG(IF(name='rdw',         value, NULL)) AS lab_rdw,
  AVG(IF(name='alt',         value, NULL)) AS lab_alt,
  AVG(IF(name='ast',         value, NULL)) AS lab_ast,
  AVG(IF(name='bilirubin',   value, NULL)) AS lab_bilirubin,
  AVG(IF(name='albumin',     value, NULL)) AS lab_albumin,
  AVG(IF(name='inr',         value, NULL)) AS lab_inr,
  AVG(IF(name='ptt',         value, NULL)) AS lab_ptt,
  AVG(IF(name='lactate',     value, NULL)) AS lab_lactate,
  AVG(IF(name='ph',          value, NULL)) AS lab_ph
FROM observed
GROUP BY stay_id, hour
