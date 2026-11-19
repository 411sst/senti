-- T49: vasopressor and ventilation records for Sadanandan's label
-- (arXiv:2603.14719v1, Section 2.3), MIMIC-IV 3.1 on BigQuery. Times are hours
-- from ICU intime; one row per source record.
--
-- Vasopressors: inputevents itemids 221906 (norepinephrine), 221289
-- (epinephrine), 221662 (dopamine), 221749 (phenylephrine), 222315
-- (vasopressin). Ventilation: procedureevents itemids 224385 (intubation) and
-- 225792 (invasive ventilation).
SELECT ie.stay_id, 'vaso' AS kind, ie.itemid, ie.statusdescription AS status,
       TIMESTAMP_DIFF(TIMESTAMP(ie.starttime), TIMESTAMP(i.intime), MINUTE) / 60.0 AS start_h,
       TIMESTAMP_DIFF(TIMESTAMP(ie.endtime), TIMESTAMP(i.intime), MINUTE) / 60.0 AS end_h
FROM `physionet-data.mimiciv_3_1_icu.inputevents` ie
JOIN `physionet-data.mimiciv_3_1_icu.icustays` i USING (stay_id)
WHERE ie.itemid IN (221906, 221289, 221662, 221749, 222315)
UNION ALL
SELECT pe.stay_id, 'vent' AS kind, pe.itemid, pe.statusdescription AS status,
       TIMESTAMP_DIFF(TIMESTAMP(pe.starttime), TIMESTAMP(i.intime), MINUTE) / 60.0 AS start_h,
       TIMESTAMP_DIFF(TIMESTAMP(pe.endtime), TIMESTAMP(i.intime), MINUTE) / 60.0 AS end_h
FROM `physionet-data.mimiciv_3_1_icu.procedureevents` pe
JOIN `physionet-data.mimiciv_3_1_icu.icustays` i USING (stay_id)
WHERE pe.itemid IN (224385, 225792)
