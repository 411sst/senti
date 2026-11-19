-- T44/T45 preparation: the cohort extract of scripts/eicu_cohort.sql for ALL
-- eICU stays of at least 24 h (a superset of Sheikhalishahi's decompensation
-- and mortality cohorts), WITHOUT the stabilisation filter, WITHOUT censoring
-- at the first event and WITHOUT the 14-day cap: every hour of every stay.
--
-- Everything else is eicu_cohort.sql verbatim: the same observation bins and
-- columns, history rows for hours < @stab_hours, and our composite label
-- columns (not the paper's; T44 and T45 build their own). On rows T22's
-- cohort also has, the built features must equal T22's (checked by
-- scripts/t4x_eicu_extracts.py).
WITH stays AS (
  SELECT d.patientunitstayid AS stay_id,
         d.uniquepid         AS subject_key,     -- patient id, for patient-level splits
         d.hospitalid, d.unittype, d.icu_los_hours AS los_h,
         d.unitdischargeoffset, d.hospitaldischargeoffset,
         p.unitdischargestatus, p.hospitaldischargestatus
  FROM `physionet-data.eicu_crd_derived.icustay_detail` d
  JOIN `physionet-data.eicu_crd.patient` p USING (patientunitstayid)
  WHERE d.icu_los_hours >= 24                    -- superset of both published cohorts
),
-- ---- deterioration components -------------------------------------------
vent AS (                                        -- invasive ventilation ONSET
  SELECT stay_id, MIN(h) AS t FROM (
    SELECT patientunitstayid AS stay_id, ventstartoffset / 60.0 AS h
    FROM `physionet-data.eicu_crd.respiratorycare`
    WHERE ventstartoffset IS NOT NULL AND ventstartoffset > 0
    UNION ALL
    SELECT patientunitstayid, treatmentoffset / 60.0
    FROM `physionet-data.eicu_crd.treatment`
    WHERE treatmentstring LIKE '%mechanical ventilation%' AND treatmentoffset > 0
  ) GROUP BY stay_id
),
vaso AS (                                        -- vasopressor / inotrope ONSET
  SELECT patientunitstayid AS stay_id, MIN(chartoffset) / 60.0 AS t
  FROM `physionet-data.eicu_crd_derived.pivoted_treatment_vasopressor`
  WHERE vasopressor = 1 AND chartoffset > 0
  GROUP BY stay_id
),
evt AS (
  SELECT s.*,
    -- ICU deaths get the unit discharge offset; ward deaths after ICU
    -- discharge get the hospital one. Both are in minutes from unit admit.
    CASE WHEN s.hospitaldischargestatus = 'Expired'
         THEN IF(s.unitdischargestatus = 'Expired',
                 s.unitdischargeoffset / 60.0,
                 s.hospitaldischargeoffset / 60.0) END AS death_h,
    v.t AS vent_h, p.t AS vaso_h
  FROM stays s
  LEFT JOIN vent v USING (stay_id)
  LEFT JOIN vaso p USING (stay_id)
),
cohort AS (
  SELECT * EXCEPT(event_h_raw), NULLIF(event_h_raw, 1e9) AS event_h
  FROM (
    SELECT *, LEAST(IFNULL(death_h,1e9), IFNULL(vent_h,1e9), IFNULL(vaso_h,1e9)) AS event_h_raw
    FROM evt
  )
  -- no stabilisation filter
),
-- ---- hourly grid (uncapped) -----------------------------------------------
-- Hours [0, @stab_hours) are HISTORY rows (is_history = 1) so the first
-- prediction hour carries real carried-forward context, exactly as in MIMIC.
grid AS (
  SELECT c.stay_id, c.subject_key, c.hospitalid, c.unittype,
         c.event_h, c.death_h, c.vent_h, c.vaso_h, h AS hour
  FROM cohort c,
  UNNEST(GENERATE_ARRAY(0, CAST(FLOOR(c.los_h) AS INT64))) AS h
  -- no censoring, no cap: every hour of the stay
),
-- ---- hourly observation bins --------------------------------------------
vit AS (
  SELECT c.stay_id, DIV(v.chartoffset, 60) AS hour,
    AVG(v.heartrate) AS heart_rate, AVG(v.respiratoryrate) AS resp_rate,
    AVG(v.spo2) AS spo2,
    AVG(COALESCE(v.ibp_systolic, v.nibp_systolic)) AS sbp,
    AVG(v.temperature) AS temperature
  FROM cohort c
  JOIN `physionet-data.eicu_crd_derived.pivoted_vital` v
    ON v.patientunitstayid = c.stay_id
  WHERE v.chartoffset >= 0 AND v.chartoffset <= c.unitdischargeoffset
  GROUP BY 1, 2
),
gcs AS (
  SELECT c.stay_id, DIV(g.chartoffset, 60) AS hour,
    MIN(g.gcs) AS gcs, MIN(g.gcsverbal) AS gcs_verbal,
    MIN(g.gcsmotor) AS gcs_motor, MIN(g.gcseyes) AS gcs_eyes
  FROM cohort c
  JOIN `physionet-data.eicu_crd_derived.pivoted_gcs` g
    ON g.patientunitstayid = c.stay_id
  WHERE g.chartoffset >= 0 AND g.chartoffset <= c.unitdischargeoffset
  GROUP BY 1, 2
),
oxy AS (
  SELECT c.stay_id, DIV(o.chartoffset, 60) AS hour,
    MAX(o.o2_flow) AS o2_flow,
    MAX(CASE WHEN o.o2_device IS NOT NULL AND o.o2_device NOT IN ('None')
             THEN 1 ELSE 0 END) AS o2_device
  FROM cohort c
  JOIN `physionet-data.eicu_crd_derived.pivoted_o2` o
    ON o.patientunitstayid = c.stay_id
  WHERE o.chartoffset >= 0 AND o.chartoffset <= c.unitdischargeoffset
  GROUP BY 1, 2
),
fio AS (
  -- FiO2 AS A PERCENTAGE, matching MIMIC. eICU stores it as a FRACTION
  -- (0.21-1.0) while MIMIC's extract yields a percentage (21-100). The
  -- cohort's own use of this column is oxygen_flags, which normalises either
  -- scale identically -- but scripts/mimic_extra_vitals.sql's consumers read
  -- the RAW column (src/lit_gbm.py masks it by staleness and feeds it to the
  -- published GBM specs), so a fraction here would silently hand those specs
  -- values 100x smaller than the ones they were fitted on.
  SELECT c.stay_id, DIV(b.chartoffset, 60) AS hour, MAX(pct) AS fio2
  FROM cohort c
  JOIN (
    SELECT patientunitstayid, chartoffset,
           CASE WHEN fio2 >= 0.21 AND fio2 <= 1.0   THEN fio2 * 100.0
                WHEN fio2 >= 21.0 AND fio2 <= 100.0 THEN fio2
                ELSE NULL END AS pct
    FROM `physionet-data.eicu_crd_derived.pivoted_bg`
    WHERE fio2 IS NOT NULL
  ) b ON b.patientunitstayid = c.stay_id
  WHERE b.chartoffset >= 0 AND b.chartoffset <= c.unitdischargeoffset
    AND b.pct IS NOT NULL
  GROUP BY 1, 2
)
SELECT g.stay_id, g.subject_key, g.hospitalid, g.unittype, g.hour,
       CAST(g.hour < @stab_hours AS INT64) AS is_history,
       v.heart_rate, v.resp_rate, v.spo2, v.sbp, v.temperature,
       gc.gcs, gc.gcs_verbal, gc.gcs_motor, gc.gcs_eyes,
       CAST(NULL AS INT64) AS gcs_unable,        -- no eICU equivalent (see header)
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
