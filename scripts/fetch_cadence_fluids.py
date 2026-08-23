"""Fetch the hourly cadence and fluid-balance extracts to local Parquet.

Runs scripts/mimic_cadence.sql and/or scripts/mimic_fluids.sql (see those
files for the leakage rationale per modality) and writes
``data/cache/mimic_{cadence,fluids}_hourly.parquet``. Feature construction
happens afterwards in ``src/modality_features.py``.

Usage:
    GOOGLE_CLOUD_PROJECT=<project> SENTINEL_BQ_STORAGE=0 \
        python scripts/fetch_cadence_fluids.py [--which cadence,fluids] [--force]
"""

import argparse
import os
from pathlib import Path

from google.cloud import bigquery

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or "sentinel-503510"
USE_BQSTORAGE = os.environ.get("SENTINEL_BQ_STORAGE", "1") != "0"
REPO = Path(__file__).resolve().parent.parent

EXTRACTS = {
    "cadence": (REPO / "scripts" / "mimic_cadence.sql",
                REPO / "data" / "cache" / "mimic_cadence_hourly.parquet"),
    "fluids": (REPO / "scripts" / "mimic_fluids.sql",
               REPO / "data" / "cache" / "mimic_fluids_hourly.parquet"),
}


def fetch(which, force=False):
    sql_path, out_path = EXTRACTS[which]
    if out_path.exists() and not force:
        print(f"[skip] {out_path} exists (use --force to refetch)")
        return
    sql = sql_path.read_text()
    client = bigquery.Client(project=PROJECT)
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False))
    print(f"[{which}] dry-run: would scan "
          f"{dry.total_bytes_processed / 1e9:.2f} GB")
    df = client.query(sql).result().to_dataframe(
        create_bqstorage_client=USE_BQSTORAGE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[{which}] {len(df):,} rows x {df.shape[1]} cols -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.0f} MB), "
          f"{df.stay_id.nunique():,} stays")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="cadence,fluids")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for which in args.which.split(","):
        fetch(which.strip(), force=args.force)


if __name__ == "__main__":
    main()
