-- Hourly infection-workup events: culture draws + antibiotic exposure
-- (MIMIC-IV 3.1).
--
-- T13 found no agent specifically elevated in sepsis beyond the shared core.
-- The earliest bedside-visible sepsis signal is behavioral: a clinician
-- SUSPECTS infection, draws cultures, and starts or escalates antibiotics --
-- Sepsis-3's t_suspicion, hours before any vasoactive order. This is the
-- cadence family of evidence (T7: clinician actions are recognition
-- timestamps, an asset), one construct deeper: not "how much attention" but
-- "attention of the infectious kind".
--
-- Timing discipline: cultures are binned by CHARTTIME (specimen collection --
-- bedside-observable; results lag days and are never read). Antibiotics come
-- from mimic-code's derived.antibiotic (prescription-classified, 245 agents;
-- 97% carry a real time-of-day). Every channel here is a count of events in
-- the hour; the builder zero-fills silence and reconstructs the running
-- active-antibiotic count from starts and stops.
--
-- The 24h lookback mirrors the labs extract: ED blood cultures and
-- pre-ICU antibiotics are real history that must be visible at hour 6.
WITH stays AS (
  SELECT i.stay_id, i.hadm_id, i.intime, i.outtime
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  WHERE i.los >= 0.5                         -- identical inclusion to the cohort
),
micro AS (
  SELECT s.stay_id,
    TIMESTAMP_DIFF(TIMESTAMP(m.charttime), TIMESTAMP(s.intime), HOUR) AS hour,
    COUNT(DISTINCT m.micro_specimen_id) AS n_cultures,
    COUNT(DISTINCT CASE WHEN UPPER(m.spec_type_desc) LIKE '%BLOOD%'
                        THEN m.micro_specimen_id END) AS n_blood_cultures
  FROM stays s
  JOIN `physionet-data.mimiciv_3_1_hosp.microbiologyevents` m
       ON m.hadm_id = s.hadm_id
  WHERE TIMESTAMP(m.charttime) >= TIMESTAMP_SUB(TIMESTAMP(s.intime), INTERVAL 24 HOUR)
    AND TIMESTAMP(m.charttime) <= TIMESTAMP(s.outtime)
  GROUP BY 1, 2
),
abx AS (
  SELECT s.stay_id, h.hour,
    COUNTIF(h.kind = 'start') AS n_abx_starts,
    COUNTIF(h.kind = 'stop')  AS n_abx_stops
  FROM stays s
  JOIN (
    SELECT hadm_id,
      TIMESTAMP_DIFF(TIMESTAMP(starttime), TIMESTAMP(x.intime), HOUR) AS hour,
      'start' AS kind
    FROM `physionet-data.mimiciv_3_1_derived.antibiotic` a
    JOIN (SELECT hadm_id, intime FROM stays) x USING (hadm_id)
    WHERE a.starttime IS NOT NULL
    UNION ALL
    SELECT hadm_id,
      TIMESTAMP_DIFF(TIMESTAMP(stoptime), TIMESTAMP(x.intime), HOUR) AS hour,
      'stop' AS kind
    FROM `physionet-data.mimiciv_3_1_derived.antibiotic` a
    JOIN (SELECT hadm_id, intime FROM stays) x USING (hadm_id)
    WHERE a.stoptime IS NOT NULL
  ) h ON h.hadm_id = s.hadm_id
  WHERE h.hour >= -24
    AND h.hour <= TIMESTAMP_DIFF(TIMESTAMP(s.outtime), TIMESTAMP(s.intime), HOUR)
  GROUP BY 1, 2
)
SELECT
  COALESCE(m.stay_id, a.stay_id) AS stay_id,
  COALESCE(m.hour, a.hour) AS hour,
  m.n_cultures, m.n_blood_cultures, a.n_abx_starts, a.n_abx_stops
FROM micro m
FULL OUTER JOIN abx a ON a.stay_id = m.stay_id AND a.hour = m.hour
