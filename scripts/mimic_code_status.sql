-- Charted code status / limitation of care (MIMIC-IV 3.1).
--
-- T7 recorded a caveat it could not act on: death is itself treatment-mediated
-- (DNR, withdrawal of care), so a death-only label is a LOWER bound on
-- physiological signal, and comfort-care exclusion was "listed as future work".
-- Acting on it needs to know WHEN a limitation of care was documented, and
-- MIMIC-IV charts exactly that: itemid 223758 ("Code Status") is a timestamped
-- ICU chart item, not free text.
--
-- Nothing here is a model feature. A code-status entry is a clinician DECISION
-- about the outcome we predict, so feeding it to a model would be predicting
-- the order set -- the exact failure T7 exists to rule out. It is used only to
-- PARTITION stays for a sensitivity analysis, the same audit-only posture as
-- the note-blackout counterfactual in audit_note_contamination.py.
--
-- Hours are offsets from ICU intime, matching every other extract, so the
-- timeline can be compared against event_h without re-deriving anything.
WITH stays AS (
  SELECT i.stay_id, i.intime, i.outtime
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  WHERE i.los >= 0.5                         -- identical inclusion to the cohort
),
-- Two kinds of evidence, kept apart because they mean different things.
-- A code-status entry states what will not be done. A palliative care plan
-- says a goals-of-care conversation is underway, which is weaker and must not
-- be read as a limitation -- so the `kind` column carries the distinction
-- forward instead of collapsing it here.
--
-- Deliberately EXCLUDED after inspection: 227954 and 224859 ("Status and
-- Comfort") are sedation/agitation scales -- their top values are Asleep /
-- Calm / Awake -- despite labels that read like goals of care.
cs AS (
  SELECT ce.stay_id, ce.charttime, ce.itemid, ce.value,
         CASE WHEN ce.itemid IN (223758, 228687, 229784) THEN 'code_status'
              ELSE 'palliative_plan' END AS kind
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
  WHERE ce.itemid IN (223758,   -- Code Status (nursing flowsheet, the main one)
                      228687,   -- Code Status. (MD progress note)
                      229784,   -- Code Status (Intubation)
                      229150,   -- Palliative Care NCP - Expected outcomes
                      229152,   -- Palliative Care NCP - Interventions
                      229154,   -- Palliative Care NCP - Plan revised
                      229155)   -- Palliative Care NCP - Problem resolved
    AND ce.value IS NOT NULL
)
SELECT
  s.stay_id,
  TIMESTAMP_DIFF(TIMESTAMP(cs.charttime), TIMESTAMP(s.intime), MINUTE) / 60.0 AS hour,
  cs.itemid,
  cs.kind,
  cs.value
FROM stays s
JOIN cs USING (stay_id)
