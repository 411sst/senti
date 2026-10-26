-- In-ICU hourly decompensation cohort (MIMIC-IV 3.1, BigQuery).
--
-- One row per (stay_id, hour) during an ICU stay. Feature columns hold the
-- values OBSERVED IN THAT HOUR; the leakage-safe "only data strictly before t"
-- shift is applied later in Python (src/mimic_cohort.py), so this file stays a
-- faithful, auditable extract of the source.
--
-- Design decisions (see the plan discussion):
--   * Ward->ICU is NOT buildable in MIMIC-IV: chartevents/derived.vitalsign are
--     ICU-only, so ward periods have no vitals and NEWS2 is uncomputable there.
--     This is the in-ICU decompensation cohort instead.
--   * Stabilization window @stab_hours (default 6): a patient must be
--     event-free through that hour, and predictions start there. Without it,
--     73% of "events" land in the first 6 hours -- patients who arrived
--     critically ill or post-op, i.e. already deteriorated rather than
--     predictably deteriorating.
--   * Deterioration = invasive ventilation onset OR vasopressor/inotrope onset
--     OR in-hospital death, whichever comes first. (ICU transfer is definitionally
--     unavailable once the patient is already in the ICU.)
--   * Each stay is censored at its first event: hours after it are not
--     prediction opportunities.
WITH stays AS (
  SELECT i.stay_id, i.subject_id, i.hadm_id, i.intime, i.outtime, i.los
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  WHERE i.los >= 0.5                       -- >= 12h so an hourly trajectory exists
),
-- ---- deterioration components -------------------------------------------
death AS (
  SELECT hadm_id, deathtime FROM `physionet-data.mimiciv_3_1_hosp.admissions`
  WHERE deathtime IS NOT NULL              -- minute-resolved in-hospital death
),
vent AS (                                  -- invasive ventilation ONSET
  SELECT stay_id, MIN(starttime) AS t
  FROM `physionet-data.mimiciv_3_1_derived.ventilation`
  WHERE ventilation_status IN ('InvasiveVent','Tracheostomy')
  GROUP BY stay_id
),
vaso AS (                                  -- vasopressor / inotrope ONSET
  SELECT stay_id, MIN(starttime) AS t
  FROM `physionet-data.mimiciv_3_1_derived.vasoactive_agent`
  WHERE COALESCE(norepinephrine, epinephrine, dopamine, phenylephrine,
                 vasopressin, dobutamine, milrinone) IS NOT NULL
  GROUP BY stay_id
),
evt AS (
  SELECT s.stay_id, s.subject_id, s.intime, s.outtime, s.los,
    TIMESTAMP_DIFF(TIMESTAMP(d.deathtime), TIMESTAMP(s.intime), MINUTE)/60.0 AS death_h,
    TIMESTAMP_DIFF(TIMESTAMP(v.t),         TIMESTAMP(s.intime), MINUTE)/60.0 AS vent_h,
    TIMESTAMP_DIFF(TIMESTAMP(p.t),         TIMESTAMP(s.intime), MINUTE)/60.0 AS vaso_h
  FROM stays s
  LEFT JOIN death d USING (hadm_id)
  LEFT JOIN vent  v USING (stay_id)
  LEFT JOIN vaso  p USING (stay_id)
),
cohort AS (
  SELECT * EXCEPT(event_h_raw), NULLIF(event_h_raw, 1e9) AS event_h
  FROM (
    SELECT *, LEAST(IFNULL(death_h,1e9), IFNULL(vent_h,1e9), IFNULL(vaso_h,1e9)) AS event_h_raw
    FROM evt
  )
  -- must survive event-free through the stabilization window
  WHERE event_h_raw = 1e9 OR event_h_raw > @stab_hours
),
-- ---- hourly grid (capped at 14 days to bound size) -----------------------
-- Starts at hour 0, not @stab_hours: hours [0, @stab_hours) are emitted as
-- HISTORY rows (is_history = 1) so that after the strictly-before-t shift in
-- Python the first prediction hour carries real carried-forward context --
-- the same reason the labs extract opens its window 24h before admission.
-- History rows are never prediction rows: src/mimic_cohort.py drops them
-- after feature construction. Censoring needs no history clause because the
-- cohort is event-free through @stab_hours by definition.
grid AS (
  SELECT c.stay_id, c.subject_id, c.event_h, c.death_h, c.vent_h, c.vaso_h, h AS hour
  FROM cohort c,
  UNNEST(GENERATE_ARRAY(0, LEAST(CAST(FLOOR(c.los*24) AS INT64), 336))) AS h
  WHERE c.event_h IS NULL OR h < c.event_h   -- censor at the first event
),
-- ---- hourly observation bins --------------------------------------------
vit AS (
  SELECT c.stay_id,
    TIMESTAMP_DIFF(TIMESTAMP(v.charttime), TIMESTAMP(c.intime), HOUR) AS hour,
    AVG(v.heart_rate) AS heart_rate, AVG(v.resp_rate) AS resp_rate,
    AVG(v.spo2) AS spo2, AVG(COALESCE(v.sbp, v.sbp_ni)) AS sbp,
    AVG(v.temperature) AS temperature
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.vitalsign` v USING (stay_id)
  WHERE v.charttime >= c.intime AND v.charttime <= c.outtime
  GROUP BY 1, 2
),
gcs AS (
  SELECT c.stay_id,
    TIMESTAMP_DIFF(TIMESTAMP(g.charttime), TIMESTAMP(c.intime), HOUR) AS hour,
    MIN(g.gcs) AS gcs, MIN(g.gcs_verbal) AS gcs_verbal,
    MIN(g.gcs_motor) AS gcs_motor, MIN(g.gcs_eyes) AS gcs_eyes,
    MAX(g.gcs_unable) AS gcs_unable          -- 1 = intubated, verbal not assessable
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.gcs` g USING (stay_id)
  WHERE g.charttime >= c.intime AND g.charttime <= c.outtime
  GROUP BY 1, 2
),
oxy AS (
  SELECT c.stay_id,
    TIMESTAMP_DIFF(TIMESTAMP(o.charttime), TIMESTAMP(c.intime), HOUR) AS hour,
    MAX(o.o2_flow) AS o2_flow,
    MAX(CASE WHEN o.o2_delivery_device_1 IS NOT NULL
              AND o.o2_delivery_device_1 NOT IN ('None') THEN 1 ELSE 0 END) AS o2_device
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.oxygen_delivery` o USING (stay_id)
  WHERE o.charttime >= c.intime AND o.charttime <= c.outtime
  GROUP BY 1, 2
),
fio AS (
  SELECT c.stay_id,
    TIMESTAMP_DIFF(TIMESTAMP(b.charttime), TIMESTAMP(c.intime), HOUR) AS hour,
    MAX(COALESCE(b.fio2, b.fio2_chartevents)) AS fio2
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_icu.icustays` i USING (stay_id)
  JOIN `physionet-data.mimiciv_3_1_derived.bg` b ON b.hadm_id = i.hadm_id
  WHERE b.charttime >= c.intime AND b.charttime <= c.outtime
  GROUP BY 1, 2
)
SELECT g.stay_id, g.subject_id, g.hour,
       CAST(g.hour < @stab_hours AS INT64) AS is_history,
       v.heart_rate, v.resp_rate, v.spo2, v.sbp, v.temperature,
       gc.gcs, gc.gcs_verbal, gc.gcs_motor, gc.gcs_eyes, gc.gcs_unable,
       o.o2_flow, o.o2_device, f.fio2,
       g.event_h, g.death_h, g.vent_h, g.vaso_h,
       -- label: first deterioration event within 24h AFTER this prediction hour
       CAST(g.event_h IS NOT NULL AND g.event_h > g.hour
            AND g.event_h <= g.hour + 24 AS INT64) AS label_24h
FROM grid g
LEFT JOIN vit v  ON v.stay_id = g.stay_id AND v.hour = g.hour
LEFT JOIN gcs gc ON gc.stay_id = g.stay_id AND gc.hour = g.hour
LEFT JOIN oxy o  ON o.stay_id = g.stay_id AND o.hour = g.hour
LEFT JOIN fio f  ON f.stay_id = g.stay_id AND f.hour = g.hour
