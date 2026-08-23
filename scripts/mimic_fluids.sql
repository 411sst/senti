-- Hourly urine output + fluid balance for the in-ICU cohort (MIMIC-IV 3.1).
--
-- Physiology the current agents do not see: oliguria classically precedes
-- hypotension/vasopressors by hours, and the orchestrator's weakest
-- per-event component is death (AUPRC 0.073 vs the fair GBM's 0.19) -- the
-- component renal trajectory speaks to most directly.
--
-- LEAKAGE: bedside-charted volumes are available at charttime (the nurse
-- writes the number at the bedside; there is no storetime lag as with lab
-- results). Bucket h holds volumes charted in [intime+h, intime+h+1); the
-- Python builder shifts one hour so the row at t sees buckets <= t-1.
--
-- Urine itemids follow the mimic-code urine_output concept; 227488 (GU
-- irrigant IN) enters negatively, 227489 (irrigant/urine out) positively.
-- inputevents volumes are binned at starttime -- an approximation for long
-- infusions (the full amount lands in the starting hour); acceptable for a
-- balance TREND feature and stated here rather than hidden.
WITH stays AS (
  SELECT stay_id, hadm_id, intime, outtime
  FROM `physionet-data.mimiciv_3_1_icu.icustays`
  WHERE los >= 0.5
),
urine AS (
  SELECT o.stay_id,
         TIMESTAMP_DIFF(TIMESTAMP(o.charttime), TIMESTAMP(s.intime), HOUR) AS hour,
         SUM(CASE WHEN o.itemid = 227488 THEN -o.value ELSE o.value END)
           AS urine_ml,
         COUNT(*) AS n_urine_records
  FROM `physionet-data.mimiciv_3_1_icu.outputevents` o
  JOIN stays s USING (stay_id)
  WHERE o.itemid IN (226559, 226560, 226561, 226584, 226563, 226564, 226565,
                     226567, 226557, 226558, 227488, 227489)
    AND o.value IS NOT NULL AND o.value >= 0 AND o.value < 5000
    AND o.charttime >= DATETIME_SUB(s.intime, INTERVAL 24 HOUR)
    AND o.charttime <= s.outtime
  GROUP BY 1, 2
),
out_other AS (
  SELECT o.stay_id,
         TIMESTAMP_DIFF(TIMESTAMP(o.charttime), TIMESTAMP(s.intime), HOUR) AS hour,
         SUM(o.value) AS other_out_ml
  FROM `physionet-data.mimiciv_3_1_icu.outputevents` o
  JOIN stays s USING (stay_id)
  WHERE o.itemid NOT IN (226559, 226560, 226561, 226584, 226563, 226564,
                         226565, 226567, 226557, 226558, 227488, 227489)
    AND o.value IS NOT NULL AND o.value >= 0 AND o.value < 5000
    AND o.charttime >= DATETIME_SUB(s.intime, INTERVAL 24 HOUR)
    AND o.charttime <= s.outtime
  GROUP BY 1, 2
),
fluid_in AS (
  SELECT i.stay_id,
         TIMESTAMP_DIFF(TIMESTAMP(i.starttime), TIMESTAMP(s.intime), HOUR) AS hour,
         SUM(i.amount) AS in_ml,
         COUNT(*) AS n_in_records
  FROM `physionet-data.mimiciv_3_1_icu.inputevents` i
  JOIN stays s USING (stay_id)
  WHERE LOWER(i.amountuom) = 'ml'
    AND i.amount > 0 AND i.amount < 10000
    AND i.starttime >= DATETIME_SUB(s.intime, INTERVAL 24 HOUR)
    AND i.starttime <= s.outtime
  GROUP BY 1, 2
)
SELECT stay_id, hour,
       IFNULL(urine_ml, 0)        AS urine_ml,
       IFNULL(n_urine_records, 0) AS n_urine_records,
       IFNULL(other_out_ml, 0)    AS other_out_ml,
       IFNULL(in_ml, 0)           AS in_ml,
       IFNULL(n_in_records, 0)    AS n_in_records
FROM urine
FULL OUTER JOIN out_other USING (stay_id, hour)
FULL OUTER JOIN fluid_in  USING (stay_id, hour)
ORDER BY stay_id, hour
