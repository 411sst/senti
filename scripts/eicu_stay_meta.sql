-- Stay-level demographics for the in-ICU cohort (eICU-CRD v2.0).
--
-- External-validation twin of the columns src/lit_gbm.py reads from
-- scripts/mimic_stay_meta.sql: stay_id, gender, anchor_age, admit_hour_of_day.
-- The published GBM specs use all three (eCARTv5 scores age, sex and time of
-- day explicitly), so a fair external comparison needs them.
--
-- THREE eICU-SPECIFIC CONVERSIONS, each one a place a silent mismatch could hide:
--
--   * gender. eICU spells it 'Male'/'Female'; MIMIC uses 'M'/'F' and lit_gbm
--     tests `gender == "M"`. Mapped here so the test means the same thing.
--   * age. eICU stores it as a STRING and writes '> 89' for every patient over
--     89, exactly as MIMIC caps its own. '> 89' becomes 90; anything
--     non-numeric becomes NULL rather than 0, so an unparsed value cannot read
--     as a newborn.
--   * time of day. MIMIC's stay_meta carries admit_hour_of_day because its
--     timestamps are date-shifted; eICU keeps a real wall-clock unitadmittime24
--     (a TIME value), so the hour is EXTRACTed directly from it. lit_gbm then adds
--     the hour offset back to recover time of day, unchanged.
SELECT
  p.patientunitstayid AS stay_id,
  CASE WHEN p.gender = 'Male' THEN 'M'
       WHEN p.gender = 'Female' THEN 'F' END AS gender,
  CASE WHEN TRIM(p.age) = '> 89' THEN 90
       ELSE SAFE_CAST(TRIM(p.age) AS INT64) END AS anchor_age,
  EXTRACT(HOUR FROM p.unitadmittime24) AS admit_hour_of_day
FROM `physionet-data.eicu_crd.patient` p
JOIN `physionet-data.eicu_crd_derived.icustay_detail` d
  USING (patientunitstayid)
WHERE d.icu_los_hours >= 12
