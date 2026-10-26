-- Hourly respiratory-support state for the in-ICU cohort (MIMIC-IV 3.1).
--
-- The escalation ladder (room air -> nasal cannula -> masks -> HFNC -> NIV)
-- precedes mechanical-ventilation onset by hours; device changes, O2 flow and
-- FiO2 are bedside observables charted in real time (charttime binning is
-- availability-correct, as for vitals). NEAR-LABEL WARNING, handled
-- downstream: the agent ships with a cap-at-HFNC sensitivity variant and
-- per-component reporting, since NIV is one step from the vent label.
--
-- Itemids (MetaVision): 226732 O2 Delivery Device(s) [string], 223834 O2 Flow
-- [L/min], 223835 Inspired O2 Fraction [%]. Device takes the LAST string per
-- hour; numerics take the hourly MAX (the clinically conservative read).
WITH stays AS (
  SELECT stay_id, hadm_id, intime, outtime
  FROM `physionet-data.mimiciv_3_1_icu.icustays`
  WHERE los >= 0.5
),
dev AS (
  SELECT c.stay_id,
         TIMESTAMP_DIFF(TIMESTAMP(c.charttime), TIMESTAMP(s.intime), HOUR) AS hour,
         ARRAY_AGG(c.value ORDER BY c.charttime DESC LIMIT 1)[OFFSET(0)]
           AS o2_device
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` c
  JOIN stays s USING (stay_id)
  WHERE c.itemid = 226732
    AND c.value IS NOT NULL AND c.value != ''
    AND c.charttime >= DATETIME_SUB(s.intime, INTERVAL 24 HOUR)
    AND c.charttime <= s.outtime
  GROUP BY 1, 2
),
num AS (
  SELECT c.stay_id,
         TIMESTAMP_DIFF(TIMESTAMP(c.charttime), TIMESTAMP(s.intime), HOUR) AS hour,
         MAX(IF(c.itemid = 223834, c.valuenum, NULL)) AS o2_flow,
         MAX(IF(c.itemid = 223835, c.valuenum, NULL)) AS fio2
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` c
  JOIN stays s USING (stay_id)
  WHERE c.itemid IN (223834, 223835)
    AND c.valuenum IS NOT NULL AND c.valuenum >= 0 AND c.valuenum <= 100
    AND c.charttime >= DATETIME_SUB(s.intime, INTERVAL 24 HOUR)
    AND c.charttime <= s.outtime
  GROUP BY 1, 2
)
SELECT stay_id, hour, o2_device, o2_flow, fio2
FROM dev
FULL OUTER JOIN num USING (stay_id, hour)
ORDER BY stay_id, hour
