"""Fetch the supplementary MIMIC-IV extracts to local Parquet.

Runs one of scripts/mimic_*.sql (see each file for its leakage rationale) and
writes the matching ``data/cache/*.parquet``. Feature construction happens
afterwards in ``src/modality_features.py``.

Despite the name this is the general extract runner: cadence, fluids and resp
are hourly modality extracts; ``extra_vitals`` adds the DBP/MAP/pCO2/pO2
channels the published GBM specs need (src/lit_gbm.py) and ``stay_meta`` adds
the stay-level care-unit / service / diagnosis metadata used for ICU-category
stratification (src/icu_strata.py).

Usage:
    GOOGLE_CLOUD_PROJECT=<project> SENTINEL_BQ_STORAGE=0 \
        python scripts/fetch_cadence_fluids.py [--which cadence,fluids]
                                               [--force] [--dry-run]
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
    "resp": (REPO / "scripts" / "mimic_resp.sql",
             REPO / "data" / "cache" / "mimic_resp_hourly.parquet"),
    "extra_vitals": (REPO / "scripts" / "mimic_extra_vitals.sql",
                     REPO / "data" / "cache" / "mimic_extra_vitals_hourly.parquet"),
    "stay_meta": (REPO / "scripts" / "mimic_stay_meta.sql",
                  REPO / "data" / "cache" / "mimic_stay_meta.parquet"),
    "code_status": (REPO / "scripts" / "mimic_code_status.sql",
                    REPO / "data" / "cache" / "mimic_code_status.parquet"),
    "neuro_status": (REPO / "scripts" / "mimic_neuro_status.sql",
                     REPO / "data" / "cache" / "mimic_neuro_status_hourly.parquet"),
    "infection": (REPO / "scripts" / "mimic_infection.sql",
                  REPO / "data" / "cache" / "mimic_infection_hourly.parquet"),
}


def fetch(which, force=False, dry_run=False):
    sql_path, out_path = EXTRACTS[which]
    if out_path.exists() and not force and not dry_run:
        print(f"[skip] {out_path} exists (use --force to refetch)")
        return
    sql = sql_path.read_text()
    client = bigquery.Client(project=PROJECT)
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False))
    print(f"[{which}] dry-run: would scan "
          f"{dry.total_bytes_processed / 1e9:.2f} GB")
    if dry_run:
        return
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
    ap.add_argument("--dry-run", action="store_true",
                    help="print the byte estimate and stop (sandbox has no billing)")
    args = ap.parse_args()
    for which in args.which.split(","):
        fetch(which.strip(), force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
