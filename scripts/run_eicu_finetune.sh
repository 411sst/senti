#!/usr/bin/env bash
# End-to-end: fetch MIMIC-IV, MIMIC-IV-ECG and eICU-CRD caches from BigQuery,
# then fine-tune the flagship on eICU (scripts/finetune_eicu.py). Every fetch
# step skips if its cache already exists, so the script is safe to re-run.
#
#   GOOGLE_CLOUD_PROJECT=<project> bash scripts/run_eicu_finetune.sh [finetune args...]
#
# Optional per-dataset projects (default: GOOGLE_CLOUD_PROJECT):
#   MIMIC_PROJECT, ECG_PROJECT, EICU_PROJECT
# SKIP_ECG=1 skips the MIMIC-IV-ECG steps; then also pass an --agents list
# without ecg_meas.
#
# Needs PhysioNet credentialed access, linked to the Google account gcloud uses,
# for MIMIC-IV v3.1, MIMIC-IV-ECG v1.0 and eICU-CRD v2.0 on BigQuery. On a
# project without billing (the BigQuery sandbox), also export
# SENTINEL_BQ_STORAGE=0.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

: "${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT to the project that runs the queries}"
MIMIC_PROJECT="${MIMIC_PROJECT:-$GOOGLE_CLOUD_PROJECT}"
ECG_PROJECT="${ECG_PROJECT:-$GOOGLE_CLOUD_PROJECT}"
EICU_PROJECT="${EICU_PROJECT:-$GOOGLE_CLOUD_PROJECT}"

echo "== MIMIC-IV caches (source cohort)"
export GOOGLE_CLOUD_PROJECT="$MIMIC_PROJECT"
"$PY" scripts/fetch_mimic_cohort.py
[ -f data/cache/mimic_cohort_hourly.parquet ] || \
    "$PY" -c "import sys; sys.path.insert(0, 'src'); import mimic_cohort; mimic_cohort.build()"
"$PY" scripts/fetch_mimic_labs.py
"$PY" scripts/fetch_cadence_fluids.py --which fluids,resp

if [ "${SKIP_ECG:-0}" != "1" ]; then
    echo "== MIMIC-IV-ECG caches (ecg_meas agent)"
    export GOOGLE_CLOUD_PROJECT="$ECG_PROJECT"
    "$PY" scripts/build_ecg_index.py
    "$PY" scripts/fetch_ecg_measurements.py
    "$PY" scripts/build_ecg_meas_features.py
fi

echo "== eICU-CRD caches (target cohort)"
export GOOGLE_CLOUD_PROJECT="$EICU_PROJECT"
"$PY" scripts/fetch_eicu_cohort.py
[ -f data/cache/eicu_cohort_hourly.parquet ] || "$PY" src/eicu_cohort.py
"$PY" scripts/fetch_eicu_labs.py
"$PY" scripts/fetch_eicu_modalities.py --which fluids,resp,extra_vitals,stay_meta

echo "== fine-tune"
"$PY" scripts/finetune_eicu.py --config configs/eicu_finetune.json "$@"
