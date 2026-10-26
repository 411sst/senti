-- DBP / MAP / pCO2 / pO2 for the in-ICU cohort (eICU-CRD v2.0).
--
-- External-validation twin of scripts/mimic_extra_vitals.sql. These four
-- channels exist only because the published GBM specs need them
-- (src/lit_gbm.py): eCARTv5 and YAIB both read diastolic and mean pressure,
-- and the blood-gas pair. Same column names and hourly-bin contract, so
-- build_extra_vitals_features runs over this file unchanged.
--
-- Invasive arterial pressure is preferred over the cuff where both exist, the
-- same COALESCE order the cohort extract uses for systolic.
WITH stays AS (
  SELECT patientunitstayid AS stay_id, unitdischargeoffset
  FROM `physionet-data.eicu_crd_derived.icustay_detail`
  WHERE icu_los_hours >= 12
),
vit AS (
  SELECT s.stay_id, DIV(v.chartoffset, 60) AS hour,
    AVG(COALESCE(v.ibp_diastolic, v.nibp_diastolic)) AS dbp,
    AVG(COALESCE(v.ibp_mean,      v.nibp_mean))      AS mbp
  FROM `physionet-data.eicu_crd_derived.pivoted_vital` v
  JOIN stays s ON s.stay_id = v.patientunitstayid
  WHERE v.chartoffset >= 0 AND v.chartoffset <= s.unitdischargeoffset
  GROUP BY 1, 2
),
gas AS (
  SELECT s.stay_id, DIV(b.chartoffset, 60) AS hour,
    AVG(b.paco2) AS pco2, AVG(b.pao2) AS po2
  FROM `physionet-data.eicu_crd_derived.pivoted_bg` b
  JOIN stays s ON s.stay_id = b.patientunitstayid
  WHERE b.chartoffset >= 0 AND b.chartoffset <= s.unitdischargeoffset
  GROUP BY 1, 2
)
SELECT
  COALESCE(v.stay_id, g.stay_id) AS stay_id,
  COALESCE(v.hour,    g.hour)    AS hour,
  v.dbp, v.mbp, g.pco2, g.po2
FROM vit v
FULL OUTER JOIN gas g ON g.stay_id = v.stay_id AND g.hour = v.hour
