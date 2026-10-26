-- Hourly measurement-cadence counts for the in-ICU cohort (MIMIC-IV 3.1).
--
-- The modality is CLINICIAN BEHAVIOUR, not physiology: how often labs are
-- drawn, vitals are charted, and orders are placed. The construct is
-- RCT-validated (CONCERN, Nature Medicine 2025: documentation-pattern-based
-- deterioration alerts cut mortality 35.6%); what Sentinel adds is the
-- separable-agent framing and paired earliness attribution.
--
-- LEAKAGE: counts are binned by ACTION time (charttime for lab draws and
-- bedside charting, ordertime for orders) -- unlike lab VALUES, which must be
-- binned by storetime (see mimic_labs.sql), the *act* of drawing/ordering is
-- observable the moment it happens, so action time is the correct
-- availability time for a cadence signal. Bucket h holds events in
-- [intime+h, intime+h+1); the Python feature builder shifts one hour so the
-- row at hour t sees buckets <= t-1 only (strictly before t).
--
-- Hours may be negative (down to -24): ED ordering intensity before ICU
-- admission is legitimate history at hour 6, mirroring the labs window.
WITH stays AS (
  SELECT stay_id, hadm_id, intime, outtime
  FROM `physionet-data.mimiciv_3_1_icu.icustays`
  WHERE los >= 0.5
),
lab_draws AS (
  SELECT s.stay_id,
         TIMESTAMP_DIFF(TIMESTAMP(l.charttime), TIMESTAMP(s.intime), HOUR) AS hour,
         COUNT(DISTINCT l.specimen_id) AS n_lab_draws,
         COUNT(*) AS n_lab_tests
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
  JOIN stays s ON s.hadm_id = l.hadm_id
  WHERE l.charttime >= DATETIME_SUB(s.intime, INTERVAL 24 HOUR)
    AND l.charttime <= s.outtime
  GROUP BY 1, 2
),
charting AS (
  SELECT c.stay_id,
         TIMESTAMP_DIFF(TIMESTAMP(c.charttime), TIMESTAMP(s.intime), HOUR) AS hour,
         COUNT(*) AS n_chart_rows,
         COUNT(DISTINCT c.itemid) AS n_chart_items
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` c
  JOIN stays s USING (stay_id)
  WHERE c.charttime >= DATETIME_SUB(s.intime, INTERVAL 24 HOUR)
    AND c.charttime <= s.outtime
  GROUP BY 1, 2
),
orders AS (
  SELECT s.stay_id,
         TIMESTAMP_DIFF(TIMESTAMP(p.ordertime), TIMESTAMP(s.intime), HOUR) AS hour,
         COUNTIF(p.order_type = 'Medications') AS n_orders_med,
         COUNTIF(p.order_type = 'Lab') AS n_orders_lab,
         COUNTIF(p.order_type = 'Radiology') AS n_orders_imaging,
         COUNTIF(p.order_type NOT IN ('Medications', 'Lab', 'Radiology'))
           AS n_orders_other
  FROM `physionet-data.mimiciv_3_1_hosp.poe` p
  JOIN stays s ON s.hadm_id = p.hadm_id
  WHERE p.ordertime >= DATETIME_SUB(s.intime, INTERVAL 24 HOUR)
    AND p.ordertime <= s.outtime
  GROUP BY 1, 2
),
micro AS (
  -- Culture collection: "a culture was sent" is the infection-concern signal.
  SELECT s.stay_id,
         TIMESTAMP_DIFF(TIMESTAMP(m.charttime), TIMESTAMP(s.intime), HOUR) AS hour,
         COUNT(*) AS n_micro
  FROM `physionet-data.mimiciv_3_1_hosp.microbiologyevents` m
  JOIN stays s ON s.hadm_id = m.hadm_id
  WHERE m.charttime IS NOT NULL
    AND m.charttime >= DATETIME_SUB(s.intime, INTERVAL 24 HOUR)
    AND m.charttime <= s.outtime
  GROUP BY 1, 2
)
SELECT stay_id, hour,
       IFNULL(n_lab_draws, 0)      AS n_lab_draws,
       IFNULL(n_lab_tests, 0)      AS n_lab_tests,
       IFNULL(n_chart_rows, 0)     AS n_chart_rows,
       IFNULL(n_chart_items, 0)    AS n_chart_items,
       IFNULL(n_orders_med, 0)     AS n_orders_med,
       IFNULL(n_orders_lab, 0)     AS n_orders_lab,
       IFNULL(n_orders_imaging, 0) AS n_orders_imaging,
       IFNULL(n_orders_other, 0)   AS n_orders_other,
       IFNULL(n_micro, 0)          AS n_micro
FROM lab_draws
FULL OUTER JOIN charting USING (stay_id, hour)
FULL OUTER JOIN orders   USING (stay_id, hour)
FULL OUTER JOIN micro    USING (stay_id, hour)
ORDER BY stay_id, hour
