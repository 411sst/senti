-- Hourly urine output + fluid balance for the in-ICU cohort (eICU-CRD v2.0).
--
-- External-validation twin of scripts/mimic_fluids.sql: same five channels,
-- same column names, so src/modality_features.py builds fluid features from
-- this file unchanged.
--
-- SOURCE. eICU records intake and output in one table (intakeoutput) whose
-- cellpath carries the category: "Intake (ml)" (6.1M rows / 121,502 stays),
-- "Output (ml)" (5.0M / 168,987), plus Weight and Dialysis which we ignore.
-- That maps onto MIMIC's inputevents/outputevents split cleanly. Urine is
-- separated from other output by celllabel, the way MIMIC separates it by
-- itemid -- one source, one rule, as in the MIMIC extract.
--
-- We deliberately take urine from intakeoutput rather than from the derived
-- pivoted_uo table: mixing a derived urine source with a raw other-output
-- source risks double counting the same charted volume.
--
-- LEAKAGE: bedside-charted volumes are available when the nurse writes them,
-- so the charting offset is the correct availability time (no storetime lag,
-- exactly as argued in mimic_fluids.sql). Bucket h holds volumes charted in
-- [h, h+1); the Python builder shifts one hour so the row at t sees <= t-1.
--
-- Hours may be negative (down to -24), mirroring the MIMIC window.
WITH stays AS (
  SELECT patientunitstayid AS stay_id, unitdischargeoffset
  FROM `physionet-data.eicu_crd_derived.icustay_detail`
  WHERE icu_los_hours >= 12
),
io AS (
  SELECT s.stay_id,
         DIV(o.intakeoutputoffset, 60) AS hour,
         REGEXP_EXTRACT(o.cellpath, r'\|I&O\|([^|]+)\|') AS cat,
         LOWER(o.celllabel) AS label,
         o.cellvaluenumeric AS value
  FROM `physionet-data.eicu_crd.intakeoutput` o
  JOIN stays s ON s.stay_id = o.patientunitstayid
  WHERE o.cellvaluenumeric IS NOT NULL
    AND o.intakeoutputoffset <= s.unitdischargeoffset
    AND o.intakeoutputoffset >= -24 * 60
),
tagged AS (
  SELECT stay_id, hour, cat, value,
         -- urine vs other output, by label (mirrors MIMIC's itemid split)
         (REGEXP_CONTAINS(label, r'urine|foley|urethral|void') AND
          NOT REGEXP_CONTAINS(label, r'stool')) AS is_urine
  FROM io
),
urine AS (
  SELECT stay_id, hour, SUM(value) AS urine_ml, COUNT(*) AS n_urine_records
  FROM tagged
  WHERE cat = 'Output (ml)' AND is_urine AND value >= 0 AND value < 5000
  GROUP BY 1, 2
),
out_other AS (
  SELECT stay_id, hour, SUM(value) AS other_out_ml
  FROM tagged
  WHERE cat = 'Output (ml)' AND NOT is_urine AND value >= 0 AND value < 5000
  GROUP BY 1, 2
),
fluid_in AS (
  SELECT stay_id, hour, SUM(value) AS in_ml, COUNT(*) AS n_in_records
  FROM tagged
  WHERE cat = 'Intake (ml)' AND value > 0 AND value < 10000
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
