"""Fetch per-patient age for the FiLM (V3) ECG agent.

Writes ``data/cache/patient_age.parquet`` with columns ``patient_id`` (=
subject_id) and ``anchor_age`` from ``physionet-data.mimiciv_3_1_hosp.patients``.
Patient-level and tiny; only the FiLM variant (``ecg_resnet_film``) needs it, and
it requires no cohort re-fetch.

Usage:
    python scripts/fetch_patient_age.py [--force]
"""

import argparse
import os
from pathlib import Path

from google.cloud import bigquery

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or "sentinel-503510"
USE_BQSTORAGE = os.environ.get("SENTINEL_BQ_STORAGE", "1") != "0"
REPO = Path(__file__).resolve().parent.parent
OUT_PATH = REPO / "data" / "cache" / "patient_age.parquet"

SQL = """
SELECT subject_id AS patient_id, anchor_age
FROM `physionet-data.mimiciv_3_1_hosp.patients`
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if OUT_PATH.exists() and not args.force:
        print(f"[skip] {OUT_PATH} exists (use --force to refetch)")
        return

    client = bigquery.Client(project=PROJECT)
    dry = client.query(SQL, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False))
    print(f"[dry-run] would scan {dry.total_bytes_processed/1e6:.1f} MB")

    df = client.query(SQL).result().to_dataframe(create_bqstorage_client=USE_BQSTORAGE)
    df["patient_id"] = df.patient_id.astype("int64")
    df["anchor_age"] = df.anchor_age.astype("float32")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[done] {len(df):,} patients -> {OUT_PATH} "
          f"(age {df.anchor_age.min():.0f}-{df.anchor_age.max():.0f}, "
          f"median {df.anchor_age.median():.0f})")


if __name__ == "__main__":
    main()
