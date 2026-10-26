"""Fetch discharge summaries for the cohort's DEATH-FIRST stays -- AUDIT ONLY.

    python scripts/fetch_goc_notes.py --dry-run     # byte check, no fetch
    python scripts/fetch_goc_notes.py               # fetch (skips if cached)
    python scripts/fetch_goc_notes.py --force       # refetch

T16 companion to ``src/goc_notes.py``. This is the one place the pipeline
reads ``mimiciv_note.discharge``, and it is deliberately NOT the model
pipeline: discharge summaries narrate the predicted outcome, so they are used
only to audit whether a death was preceded by a documented limitation of care
(the T11 partition), never as a feature. No model-facing source ever reads a
discharge summary; tests/test_goc_notes.py pins it.

BigQuery scans the full ``text`` column regardless of the stay filter
(columnar storage), so expect ~3.5 GB scanned -- run ``--dry-run`` first, per
house rules. No local cache carries ``hadm_id``, so the join to
``icustays`` happens server-side and the stay filter travels as a query
parameter. Output stays under gitignored ``data/`` (PhysioNet DUA).
"""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402
from google.cloud import bigquery  # noqa: E402

from goc_notes import death_stays  # noqa: E402

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or "sentinel-503510"
USE_BQSTORAGE = os.environ.get("SENTINEL_BQ_STORAGE", "1") != "0"
OUT = REPO / "data" / "cache" / "goc_notes.parquet"

SQL = """
SELECT n.note_id, n.subject_id, n.hadm_id, i.stay_id,
       n.charttime, n.storetime, n.text
FROM `physionet-data.mimiciv_note.discharge` n
JOIN `physionet-data.mimiciv_3_1_icu.icustays` i ON n.hadm_id = i.hadm_id
WHERE i.stay_id IN UNNEST(@stay_ids)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if OUT.exists() and not args.force and not args.dry_run:
        print(f"[skip] {OUT} exists -- use --force to refetch")
        return

    deaths = death_stays(cohort_path=str(REPO / "data/cache/mimic_cohort_hourly.parquet"))
    ids = sorted(int(s) for s in deaths["stay_id"].unique())
    charted = int(deaths["has_code_status"].sum())
    print(f"[resolve] {len(ids):,} death-first stays "
          f"({charted:,} charted / {len(ids) - charted:,} uncharted)")

    client = bigquery.Client(project=PROJECT)
    params = [bigquery.ArrayQueryParameter("stay_ids", "INT64", ids)]
    dry = client.query(SQL, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False, query_parameters=params))
    print(f"[dry-run] {dry.total_bytes_processed / 1e9:.2f} GB")
    if args.dry_run:
        return

    job = bigquery.QueryJobConfig(query_parameters=params)
    df = client.query(SQL, job_config=job).result().to_dataframe(
        create_bqstorage_client=USE_BQSTORAGE)
    df["stay_id"] = df["stay_id"].astype("int64")
    df = df.drop_duplicates(["note_id", "stay_id"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    covered = df["stay_id"].nunique()
    chars = df["text"].str.len()
    print(f"[done] {len(df):,} discharge notes covering {covered:,} of "
          f"{len(ids):,} death stays ({len(ids) - covered:,} without a note) "
          f"-> {OUT} ({OUT.stat().st_size / 1e6:.0f} MB); "
          f"median {chars.median():.0f} chars, p90 {chars.quantile(.9):.0f}")


if __name__ == "__main__":
    main()
