"""Fetch machine-read ECG measurements for the cohort's indexed studies.

MIMIC-IV-ECG ships ``machine_measurements``: per-study conduction intervals
(RR, P/QRS onsets+ends, T end, P/QRS/T axes) plus up to 18 machine diagnostic
statements ("Sinus rhythm", "Atrial fibrillation with rapid ventricular
response", "*** CONSIDER ACUTE ST ELEVATION MI ***", ...). These are produced
by the ECG cart at acquisition time, so they are available at ``ecg_time`` and
the cohort's strictly-before-t merge applies unchanged.

Mirrors scripts/mimic_ecg_index.sql's join (record-level table x icustays on
subject_id, windowed to [-@staleness h, discharge]) because ecg_index.parquet
does not carry subject_id; the result is then restricted locally to the studies
actually in the index and deduped on study_id.

Writes ``data/cache/ecg_machine_measurements.parquet`` (~68k rows, a few MB).

Usage:
    python scripts/fetch_ecg_measurements.py [--staleness 72] [--force]
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or "sentinel-503510"
USE_BQSTORAGE = os.environ.get("SENTINEL_BQ_STORAGE", "1") != "0"
REPO = Path(__file__).resolve().parent.parent
INDEX = REPO / "data" / "cache" / "ecg_index.parquet"
OUT_PATH = REPO / "data" / "cache" / "ecg_machine_measurements.parquet"

REPORT_COLS = [f"report_{i}" for i in range(18)]
SQL = f"""
SELECT DISTINCT
  m.study_id,
  m.rr_interval, m.p_onset, m.p_end, m.qrs_onset, m.qrs_end, m.t_end,
  m.p_axis, m.qrs_axis, m.t_axis,
  {', '.join(f'm.{c}' for c in REPORT_COLS)}
FROM `physionet-data.mimiciv_ecg.machine_measurements` AS m
JOIN `physionet-data.mimiciv_3_1_icu.icustays` AS icu
  USING (subject_id)
WHERE DATETIME_DIFF(m.ecg_time, icu.intime, SECOND) / 3600.0
        BETWEEN -@staleness AND icu.los * 24.0
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staleness", type=int, default=72)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if OUT_PATH.exists() and not args.force:
        print(f"[skip] {OUT_PATH} exists (use --force to refetch)")
        return
    if not INDEX.exists():
        raise SystemExit(f"[error] {INDEX} missing -- run build_ecg_index.py first")

    params = [bigquery.ScalarQueryParameter("staleness", "INT64", args.staleness)]
    client = bigquery.Client(project=PROJECT)
    dry = client.query(SQL, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False, query_parameters=params))
    print(f"[dry-run] would scan {dry.total_bytes_processed/1e9:.2f} GB")

    df = client.query(
        SQL, job_config=bigquery.QueryJobConfig(query_parameters=params)
    ).result().to_dataframe(create_bqstorage_client=USE_BQSTORAGE)
    print(f"[bq] {len(df):,} rows over {df.study_id.nunique():,} studies")

    # Restrict to the index's studies; one row per study.
    wanted = set(pd.read_parquet(INDEX, columns=["study_id"]).study_id.astype("int64"))
    df["study_id"] = df.study_id.astype("int64")
    df = (df[df.study_id.isin(wanted)]
          .drop_duplicates("study_id")
          .sort_values("study_id")
          .reset_index(drop=True))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    n_stmt = df[REPORT_COLS].notna().any(axis=1).sum()
    usable_rr = ((df.rr_interval > 0) & (df.rr_interval < 3000)).sum()
    print(f"[done] {len(df):,}/{len(wanted):,} indexed studies -> {OUT_PATH} "
          f"({OUT_PATH.stat().st_size/1e6:.1f} MB)")
    print(f"       usable rr_interval: {usable_rr:,} ({100*usable_rr/len(df):.1f}%) | "
          f">=1 statement: {n_stmt:,} ({100*n_stmt/len(df):.1f}%)")


if __name__ == "__main__":
    main()
