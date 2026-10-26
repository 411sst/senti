-- Hourly respiratory-support state for the in-ICU cohort (eICU-CRD v2.0).
--
-- External-validation twin of scripts/mimic_resp.sql: same three columns
-- (o2_device, o2_flow, fio2), so src/modality_features.py's build_resp_features
-- and its RESP_LEVEL_MAP run over this file unchanged.
--
-- DEVICE VOCABULARY. MIMIC charts a controlled itemid-226732 vocabulary; eICU
-- charts free text, and the same device appears many ways ("RA" / "ra" /
-- "Room Air" / "ROOM AIR" / "R/A"; "NC" / "nc" / "nasal cannula" / "nasal
-- canula"; "vent" / "VENT" / "vented" / "ventilator"). Rather than widen
-- RESP_LEVEL_MAP with 40 eICU spellings -- which would put the two datasets on
-- two different ladders -- this query normalises eICU strings ONTO MIMIC's
-- vocabulary. The mapping is explicit and auditable here, and the escalation
-- ladder downstream stays one shared definition.
--
-- Note "ventilator" and "trach collar" map to MIMIC's ARTIFICIAL AIRWAY
-- strings, not to ladder rungs: those patients are already ventilated, so a
-- vent-onset event cannot occur and RESP_AIRWAY handles them separately. This
-- is the same treatment ETT/trach get in MIMIC.
--
-- FIO2 UNITS -- the trap this extract exists to avoid. MIMIC charts FiO2 as a
-- PERCENTAGE (itemid 223835; 99.66% of values in 21-100, median 40) while
-- every eICU FiO2 source is a FRACTION (100.00% in 0.21-1.0, median 0.48).
-- build_resp_features does NOT normalise: it LOCFs with a room-air default of
-- 21.0 on the percentage scale and feeds resp_fio2 / resp_fio2_delta_6h raw.
-- Handing it fractions would make every real eICU FiO2 look sub-room-air and
-- turn the 6h delta into noise -- a silent corruption, not a visible failure.
-- So this query converts to percentage here, matching MIMIC's scale.
--
-- FIO2 SOURCE. respiratorycharting carries 3.08M FiO2 rows over 75,234 stays;
-- pivoted_bg (blood gas) only 300,884. The former is the analogue of MIMIC's
-- charted FiO2, so it leads and the blood gas fills gaps. Values are free text
-- ("40", "0.4", "40%"), hence the strip-and-cast.
--
-- Unmapped strings become NULL, which LOCF carries as "no new information" --
-- identical to an unmapped MIMIC string.
WITH stays AS (
  SELECT patientunitstayid AS stay_id, unitdischargeoffset
  FROM `physionet-data.eicu_crd_derived.icustay_detail`
  WHERE icu_los_hours >= 12
),
norm AS (
  SELECT s.stay_id,
         DIV(o.chartoffset, 60) AS hour,
         o.chartoffset,
         CASE
           WHEN REGEXP_CONTAINS(LOWER(TRIM(o.o2_device)),
                r'^(ra|r/a|room ?air)$')                    THEN 'None'
           WHEN REGEXP_CONTAINS(LOWER(TRIM(o.o2_device)),
                r'nasal ?can|^nc$|^nch$|oxymizer|oximizer|oxymask')
                                                            THEN 'Nasal cannula'
           WHEN REGEXP_CONTAINS(LOWER(TRIM(o.o2_device)),
                r'venturi|^vm$')                            THEN 'Venti mask'
           WHEN REGEXP_CONTAINS(LOWER(TRIM(o.o2_device)),
                r'aerosol')                                 THEN 'Aerosol-cool'
           WHEN REGEXP_CONTAINS(LOWER(TRIM(o.o2_device)),
                r'face tent')                               THEN 'Face tent'
           WHEN REGEXP_CONTAINS(LOWER(TRIM(o.o2_device)),
                r'non.?rebreather|nrb')                     THEN 'Non-rebreather'
           WHEN REGEXP_CONTAINS(LOWER(TRIM(o.o2_device)),
                r'hfnc|hhfnc|high ?flow')       THEN 'High flow nasal cannula'
           WHEN REGEXP_CONTAINS(LOWER(TRIM(o.o2_device)),
                r'bipap|cpap|niv|non.?invasive')            THEN 'Bipap mask'
           -- artificial airways: a state flag downstream, never a ladder rung
           WHEN REGEXP_CONTAINS(LOWER(TRIM(o.o2_device)),
                r'^vent|ventilator|vented|ett|endotracheal') THEN 'Endotracheal tube'
           WHEN REGEXP_CONTAINS(LOWER(TRIM(o.o2_device)),
                r'trach|^tc$')                              THEN 'Trach mask'
           WHEN REGEXP_CONTAINS(LOWER(TRIM(o.o2_device)),
                r'^other$')                                 THEN 'Other'
           ELSE NULL
         END AS o2_device,
         o.o2_flow
  FROM `physionet-data.eicu_crd_derived.pivoted_o2` o
  JOIN stays s ON s.stay_id = o.patientunitstayid
  WHERE o.chartoffset <= s.unitdischargeoffset
    AND o.chartoffset >= -24 * 60
),
dev AS (                                   -- LAST device string in the hour
  SELECT stay_id, hour,
         ARRAY_AGG(o2_device IGNORE NULLS ORDER BY chartoffset DESC
                   LIMIT 1)[SAFE_OFFSET(0)] AS o2_device
  FROM norm
  GROUP BY 1, 2
),
flow AS (                                  -- hourly MAX (conservative read)
  SELECT stay_id, hour, MAX(o2_flow) AS o2_flow
  FROM norm
  WHERE o2_flow IS NOT NULL AND o2_flow >= 0 AND o2_flow <= 100
  GROUP BY 1, 2
),
fio_raw AS (
  -- charted ventilator/device FiO2 (primary)
  SELECT s.stay_id, DIV(r.respchartoffset, 60) AS hour,
         SAFE_CAST(REGEXP_REPLACE(r.respchartvalue, r'[^0-9.]', '')
                   AS FLOAT64) AS v
  FROM `physionet-data.eicu_crd.respiratorycharting` r
  JOIN stays s ON s.stay_id = r.patientunitstayid
  WHERE LOWER(r.respchartvaluelabel) LIKE '%fio2%'
    AND r.respchartoffset <= s.unitdischargeoffset
    AND r.respchartoffset >= -24 * 60
  UNION ALL
  -- blood-gas FiO2 (fills the gaps)
  SELECT s.stay_id, DIV(b.chartoffset, 60) AS hour, b.fio2
  FROM `physionet-data.eicu_crd_derived.pivoted_bg` b
  JOIN stays s ON s.stay_id = b.patientunitstayid
  WHERE b.fio2 IS NOT NULL
    AND b.chartoffset <= s.unitdischargeoffset
    AND b.chartoffset >= -24 * 60
),
fio AS (
  SELECT stay_id, hour, MAX(pct) AS fio2
  FROM (
    -- normalise to PERCENTAGE: a fraction (0.21-1.0) scales by 100, a value
    -- already in 21-100 passes through, anything else is charting junk.
    SELECT stay_id, hour,
           CASE WHEN v >= 0.21 AND v <= 1.0  THEN v * 100.0
                WHEN v >= 21.0 AND v <= 100.0 THEN v
                ELSE NULL END AS pct
    FROM fio_raw
  )
  WHERE pct IS NOT NULL
  GROUP BY 1, 2
)
SELECT stay_id, hour, o2_device, o2_flow, fio2
FROM dev
FULL OUTER JOIN flow USING (stay_id, hour)
FULL OUTER JOIN fio  USING (stay_id, hour)
ORDER BY stay_id, hour
