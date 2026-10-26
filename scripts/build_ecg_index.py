"""Build the ECG index mapping cohort ICU-stay hours to fetchable waveforms.

Runs scripts/mimic_ecg_index.sql (record_list join icustays, windowed to
[-staleness h, discharge]), restricts to the cohort's stays, and writes
``data/cache/ecg_index.parquet`` with columns

    stay_id:int64, ecg_hour:float64, study_id:int64, path:str

which is exactly the schema consumed by scripts/fetch_ecg_measurements.py and
src/agents/ecg_agent.py. This is metadata only (a few MB). The waveform
download for the benched waveform agents (fetch_ecg_subset.py) is on
experiment-demo; the shipped ecg_meas agent needs no waveforms.

Usage:
    python scripts/build_ecg_index.py [--staleness 72] [--force]
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

# Billing/quota project (reads shared physionet-data tables). Defaults to the
# team project; set GOOGLE_CLOUD_PROJECT to bill your own.
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or "sentinel-503510"
# Sandbox projects lack the BigQuery Storage API; set SENTINEL_BQ_STORAGE=0 to
# fall back to the slower REST download.
USE_BQSTORAGE = os.environ.get("SENTINEL_BQ_STORAGE", "1") != "0"

REPO = Path(__file__).resolve().parent.parent
SQL_PATH = REPO / "scripts" / "mimic_ecg_index.sql"
COHORT = REPO / "data" / "cache" / "mimic_cohort_hourly.parquet"
OUT_PATH = REPO / "data" / "cache" / "ecg_index.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staleness", type=int, default=72,
                    help="keep ECGs up to this many hours before ICU admission")
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()

    if OUT_PATH.exists() and not args.force:
        print(f"[skip] {OUT_PATH} exists (use --force to rebuild)")
        return
    if not COHORT.exists():
        raise SystemExit(f"[error] cohort missing: {COHORT}\n"
                         "        run fetch_mimic_cohort.py + mimic_cohort.build() first")

    sql = SQL_PATH.read_text()
    params = [bigquery.ScalarQueryParameter("staleness", "INT64", args.staleness)]
    client = bigquery.Client(project=PROJECT)

    dry = client.query(sql, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False, query_parameters=params))
    print(f"[dry-run] would scan {dry.total_bytes_processed/1e9:.2f} GB")

    df = client.query(
        sql, job_config=bigquery.QueryJobConfig(query_parameters=params)
    ).result().to_dataframe(create_bqstorage_client=USE_BQSTORAGE)
    print(f"[bq] {len(df):,} (stay, ECG) candidate rows over "
          f"{df.stay_id.nunique():,} stays, {df.study_id.nunique():,} studies")

    # Restrict to the modelling cohort's stays (the join above spans all ICU stays).
    cohort_stays = set(
        pd.read_parquet(COHORT, columns=["stay_id"]).stay_id.astype("int64"))
    df["stay_id"] = df.stay_id.astype("int64")
    df["study_id"] = df.study_id.astype("int64")
    df["ecg_hour"] = df.ecg_hour.astype("float64")
    idx = (df[df.stay_id.isin(cohort_stays)]
           [["stay_id", "ecg_hour", "study_id", "path"]]
           .sort_values(["stay_id", "ecg_hour"])
           .reset_index(drop=True))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(OUT_PATH, index=False)
    print(f"[done] {len(idx):,} rows -> {OUT_PATH} "
          f"({OUT_PATH.stat().st_size/1e6:.1f} MB)")
    print(f"       {idx.stay_id.nunique():,}/{len(cohort_stays):,} cohort stays "
          f"have >=1 candidate ECG | {idx.study_id.nunique():,} unique studies to fetch")


if __name__ == "__main__":
    main()
