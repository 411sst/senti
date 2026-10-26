"""Fetch the supplementary eICU-CRD extracts to local Parquet.

External-validation twin of ``scripts/fetch_cadence_fluids.py``. Runs one of
scripts/eicu_*.sql (see each file for its leakage rationale and its mapping
back onto the MIMIC vocabulary) and writes the matching
``data/cache/eicu_*.parquet``. Feature construction happens afterwards in
``src/modality_features.py``, unchanged -- these extracts are shaped so the
MIMIC builders run over them as-is.

eICU is credentialed PhysioNet data under the same DUA as MIMIC-IV: every
output lives under the gitignored ``data/`` tree and never leaves this machine.

Usage:
    SENTINEL_BQ_STORAGE=0 python scripts/fetch_eicu_modalities.py \
        [--which fluids,resp] [--force] [--dry-run]
"""

import argparse
import os
from pathlib import Path

from google.cloud import bigquery

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or "gen-lang-client-0883047128"
USE_BQSTORAGE = os.environ.get("SENTINEL_BQ_STORAGE", "1") != "0"
REPO = Path(__file__).resolve().parent.parent

EXTRACTS = {
    "fluids": (REPO / "scripts" / "eicu_fluids.sql",
               REPO / "data" / "cache" / "eicu_fluids_hourly.parquet"),
    "resp": (REPO / "scripts" / "eicu_resp.sql",
             REPO / "data" / "cache" / "eicu_resp_hourly.parquet"),
    # Needed only by the published GBM specs (src/lit_gbm.py), which read
    # diastolic/mean pressure, the blood-gas pair, and age/sex/time-of-day.
    "extra_vitals": (REPO / "scripts" / "eicu_extra_vitals.sql",
                     REPO / "data" / "cache" / "eicu_extra_vitals_hourly.parquet"),
    "stay_meta": (REPO / "scripts" / "eicu_stay_meta.sql",
                  REPO / "data" / "cache" / "eicu_stay_meta.parquet"),
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
    result = client.query(sql).result()
    try:
        df = result.to_dataframe(create_bqstorage_client=USE_BQSTORAGE)
    except Exception as exc:              # sandbox projects have no Storage API
        print(f"[{which}] BQ Storage unavailable ({type(exc).__name__}); REST")
        df = result.to_dataframe(create_bqstorage_client=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[{which}] {len(df):,} rows x {df.shape[1]} cols -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.0f} MB), "
          f"{df.stay_id.nunique():,} stays")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="fluids,resp")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the byte estimate and stop")
    args = ap.parse_args()
    for which in args.which.split(","):
        fetch(which.strip(), force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
