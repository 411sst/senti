"""Fetch the hourly MIMIC-IV lab panel to a local Parquet.

Runs scripts/mimic_labs.sql (labs binned by result-availability time, see that
file) and writes ``data/cache/mimic_labs_hourly.parquet``. Leakage-safe feature
construction happens afterwards in ``src/agents/labs_agent.py``.

Usage:
    python scripts/fetch_mimic_labs.py [--force]
"""

import argparse
import os
from pathlib import Path

from google.cloud import bigquery

# Billing/quota project for the query (it reads the shared physionet-data
# tables but bytes scanned are billed here). Defaults to the team project;
# set GOOGLE_CLOUD_PROJECT to bill your own project instead.
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or "sentinel-503510"

# The BigQuery Storage API gives a much faster to_dataframe() but requires a
# billing-enabled project. On a no-billing / sandbox project it is unavailable,
# so set SENTINEL_BQ_STORAGE=0 to fall back to the (slower) REST download.
USE_BQSTORAGE = os.environ.get("SENTINEL_BQ_STORAGE", "1") != "0"
REPO = Path(__file__).resolve().parent.parent
SQL_PATH = REPO / "scripts" / "mimic_labs.sql"
OUT_PATH = REPO / "data" / "cache" / "mimic_labs_hourly.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if OUT_PATH.exists() and not args.force:
        print(f"[skip] {OUT_PATH} exists (use --force to refetch)")
        return

    sql = SQL_PATH.read_text()
    client = bigquery.Client(project=PROJECT)
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False))
    print(f"[dry-run] would scan {dry.total_bytes_processed / 1e9:.2f} GB")

    df = client.query(sql).result().to_dataframe(create_bqstorage_client=USE_BQSTORAGE)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[done] {len(df):,} rows x {df.shape[1]} cols -> {OUT_PATH} "
          f"({OUT_PATH.stat().st_size / 1e6:.0f} MB), "
          f"{df.stay_id.nunique():,} stays")


if __name__ == "__main__":
    main()
