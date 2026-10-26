-- Hourly consciousness / sedation / pain observations (MIMIC-IV 3.1).
--
-- The bank reads GCS through the vitals agent, but GCS moves late and is
-- charted sparsely once a patient is "stable". The bedside instruments that
-- move EARLY are RASS (sedation-agitation, charted ~q4h per protocol,
-- 89% of stays), CAM-ICU (delirium screen, 78%), and the pain score.
-- Agitation -> delirium -> decline is a classic pre-deterioration arc that no
-- current agent can see.
--
-- Value channels (rass_*, pain_max) are levels: an hour without a row means
-- "not reassessed", so the builder carries them forward under a staleness
-- bound, exactly like vitals. Count channels (n_*, cam_*) are events: an hour
-- without a row genuinely means zero assessments, so the builder zero-fills
-- them, exactly like cadence. The two semantics are kept in separate columns
-- here so the builder cannot confuse them.
--
-- CAM-ICU positivity is counted as value = 'Yes' on any component item; the
-- builder reads "any Yes in the last 24h" as the delirium flag rather than
-- reconstructing the formal 4-feature algorithm from rows that nurses chart
-- component-by-component.
WITH stays AS (
  SELECT i.stay_id, i.intime, i.outtime
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  WHERE i.los >= 0.5                         -- identical inclusion to the cohort
)
SELECT
  ce.stay_id,
  TIMESTAMP_DIFF(TIMESTAMP(ce.charttime), TIMESTAMP(s.intime), HOUR) AS hour,
  -- RASS: -5 (unarousable) .. +4 (combative); valuenum carries the score
  MIN(CASE WHEN ce.itemid = 228096 THEN ce.valuenum END) AS rass_min,
  MAX(CASE WHEN ce.itemid = 228096 THEN ce.valuenum END) AS rass_max,
  COUNTIF(ce.itemid = 228096) AS n_rass,
  -- CAM-ICU component items (both generations of itemids)
  COUNTIF(ce.itemid IN (228300,228301,228302,228303,228334,228335,228336,
                        228337,229324,229325,229326)) AS cam_assessed,
  COUNTIF(ce.itemid IN (228300,228301,228302,228303,228334,228335,228336,
                        228337,229324,229325,229326)
          AND ce.value = 'Yes') AS cam_positive,
  -- Pain level: charted as words ("None".."Worst") but valuenum carries the
  -- ordinal for every scoreable row; "Unable to Score" has no valuenum and
  -- correctly becomes NULL rather than a fake zero.
  MAX(CASE WHEN ce.itemid = 223791 THEN ce.valuenum END) AS pain_max,
  COUNTIF(ce.itemid = 223791) AS n_pain
FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
JOIN stays s USING (stay_id)
WHERE ce.itemid IN (228096, 223791,
                    228300,228301,228302,228303,228334,228335,228336,228337,
                    229324,229325,229326)
  AND ce.charttime >= s.intime AND ce.charttime <= s.outtime
GROUP BY 1, 2
