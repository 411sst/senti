-- Hourly lab panel for the in-ICU cohort (eICU-CRD v2.0, BigQuery).
--
-- External-validation twin of scripts/mimic_labs.sql: same 24 analytes, same
-- column names, same hourly-bin contract, so src/modality_features.py builds
-- lab features from this file unchanged.
--
-- TIMING CAVEAT (the one real asymmetry against the MIMIC extract). The MIMIC
-- version bins labs by **storetime** -- when the result became visible in the
-- system -- because the median storetime-charttime gap there is 70 minutes and
-- binning by draw time would hand the model a result an hour before any
-- clinician could have seen it. eICU has no storetime: `labresultoffset` is
-- the only timing column (`labresultrevisedoffset` marks later corrections).
-- The eICU documentation describes labresultoffset as when the result was
-- recorded, which is nearer storetime than draw time, but it is not the same
-- guarantee. Direction of the bias: if eICU lab timing is optimistic, eICU
-- scores are flattered, which makes measured transfer degradation SMALLER than
-- the truth. Must be reported alongside any transfer number.
--
-- All 24 analytes were confirmed present by name (180k-190k of 200,859 stays
-- for the core chemistry; lactate 71,794 and pH 84,164 are the thinnest).
--
-- Hours are offsets from unit admission and may be NEGATIVE (down to -24):
-- labs drawn before ICU admission are legitimate history at hour 6, so the
-- window opens a day early, exactly as in MIMIC.
WITH stays AS (
  SELECT patientunitstayid AS stay_id, unitdischargeoffset
  FROM `physionet-data.eicu_crd_derived.icustay_detail`
  WHERE icu_los_hours >= 12
),
panel AS (
  SELECT * FROM UNNEST([
    STRUCT('creatinine' AS src, 'creatinine' AS name), ('bun','bun'),
    ('sodium','sodium'), ('potassium','potassium'), ('chloride','chloride'),
    ('bicarbonate','bicarbonate'), ('anion gap','anion_gap'), ('glucose','glucose'),
    ('calcium','calcium'), ('magnesium','magnesium'), ('phosphate','phosphate'),
    ('hct','hematocrit'), ('hgb','hemoglobin'), ('wbc x 1000','wbc'),
    ('platelets x 1000','platelets'), ('rdw','rdw'),
    ('alt (sgpt)','alt'), ('ast (sgot)','ast'), ('total bilirubin','bilirubin'),
    ('albumin','albumin'), ('pt - inr','inr'), ('ptt','ptt'),
    ('lactate','lactate'), ('ph','ph')
  ])
),
observed AS (
  SELECT s.stay_id, p.name,
         DIV(l.labresultoffset, 60) AS hour,
         l.labresult AS value
  FROM `physionet-data.eicu_crd.lab` l
  JOIN panel p ON p.src = LOWER(l.labname)
  JOIN stays s ON s.stay_id = l.patientunitstayid
  WHERE l.labresult IS NOT NULL
    AND l.labresultoffset <= s.unitdischargeoffset
    AND l.labresultoffset >= -24 * 60
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
