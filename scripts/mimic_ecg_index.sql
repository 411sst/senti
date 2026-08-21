-- Candidate ECG studies for each cohort ICU stay.
--
-- Joins MIMIC-IV-ECG's record_list to icustays on subject_id and keeps every
-- study whose acquisition time falls from @staleness hours BEFORE ICU admission
-- through discharge. This only *bounds* the fetch set; the precise leakage-safe
-- pick ("most recent ECG strictly before hour t, within 72h") happens later in
-- assemble_ecg_features()/covered_hours() via merge_asof. `ecg_hour` is measured
-- from ICU intime, matching the cohort's `hour` axis.
--
-- Both ecg_time and intime are DATETIME (MIMIC wall-clock, deidentified) so the
-- difference needs no timezone handling.
SELECT
  icu.stay_id                                             AS stay_id,
  r.study_id                                              AS study_id,
  r.path                                                  AS path,
  DATETIME_DIFF(r.ecg_time, icu.intime, SECOND) / 3600.0  AS ecg_hour
FROM `physionet-data.mimiciv_ecg.record_list`  AS r
JOIN `physionet-data.mimiciv_3_1_icu.icustays` AS icu
  USING (subject_id)
WHERE DATETIME_DIFF(r.ecg_time, icu.intime, SECOND) / 3600.0
        BETWEEN -@staleness AND icu.los * 24.0
