-- Supplementary hourly channels the published GBM specs need (MIMIC-IV 3.1).
--
-- scripts/mimic_cohort.sql pulls 5 vitals and no blood gases. Two published
-- specs we reproduce in src/lit_gbm.py require more:
--   * Sadanandan 2026 (arXiv:2603.14719): 10 vitals incl. DBP and MAP, and
--     16 labs incl. pCO2 and pO2.
--   * eCARTv5 (Churpek 2025): FiO2 is already in the cohort; MAP is not.
--
-- Rather than re-run the whole cohort extract, this mirrors the `vit` and `fio`
-- CTEs of mimic_cohort.sql exactly (same stay set, same hourly binning, same
-- [intime, outtime] window) and returns ONLY the missing channels, to be joined
-- onto the existing (stay_id, hour) grid.
--
-- Same leakage posture as the cohort extract: this file is a faithful record of
-- what was OBSERVED in each hour. The strictly-before-t shift happens in Python
-- (src/modality_features.py::build_extra_vitals_features).
WITH stays AS (
  SELECT i.stay_id, i.hadm_id, i.intime, i.outtime
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  WHERE i.los >= 0.5                         -- identical inclusion to the cohort
),
vit AS (
  SELECT s.stay_id,
    TIMESTAMP_DIFF(TIMESTAMP(v.charttime), TIMESTAMP(s.intime), HOUR) AS hour,
    AVG(COALESCE(v.dbp, v.dbp_ni)) AS dbp,
    AVG(COALESCE(v.mbp, v.mbp_ni)) AS mbp
  FROM stays s
  JOIN `physionet-data.mimiciv_3_1_derived.vitalsign` v USING (stay_id)
  WHERE v.charttime >= s.intime AND v.charttime <= s.outtime
  GROUP BY 1, 2
),
gas AS (
  SELECT s.stay_id,
    TIMESTAMP_DIFF(TIMESTAMP(b.charttime), TIMESTAMP(s.intime), HOUR) AS hour,
    AVG(b.pco2) AS pco2,
    AVG(b.po2)  AS po2
  FROM stays s
  JOIN `physionet-data.mimiciv_3_1_derived.bg` b ON b.hadm_id = s.hadm_id
  WHERE b.charttime >= s.intime AND b.charttime <= s.outtime
  GROUP BY 1, 2
)
SELECT
  COALESCE(v.stay_id, g.stay_id) AS stay_id,
  COALESCE(v.hour,    g.hour)    AS hour,
  v.dbp, v.mbp, g.pco2, g.po2
FROM vit v
FULL OUTER JOIN gas g ON g.stay_id = v.stay_id AND g.hour = v.hour
