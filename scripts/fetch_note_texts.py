"""Fetch the radiology note texts our cohort can actually see.

    python scripts/fetch_note_texts.py                    # most-recent note only
    python scripts/fetch_note_texts.py --mode window      # every note in the window

Only ~90k of the 2.3M radiology reports are ever selected as "the most recent
note available before hour t, within 72h" for some cohort patient-hour. Encoding
all 452k cohort-linked notes would take hours for no benefit, so we resolve the
needed set locally first and fetch just those texts.

**Discharge summaries are never read.** They are written a median of 57h AFTER
ICU discharge and narrate the outcome being predicted; the discharge table is not
touched anywhere in this pipeline.

Notes are keyed by ``GREATEST(storetime, charttime)`` -- when the report became
available, not when the study was performed. The median gap is 136 minutes, so
using charttime would leak an impression 2+ hours before any clinician saw it.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from google.cloud import bigquery  # noqa: E402

PROJECT = "sentinel-503510"
INDEX = REPO / "data" / "cache" / "note_index.parquet"
COHORT = REPO / "data" / "cache" / "mimic_cohort_hourly.parquet"
OUT = REPO / "data" / "cache" / "note_texts.parquet"
STALENESS_HOURS = 72


def _load():
    notes = pd.read_parquet(INDEX)
    cohort = pd.read_parquet(COHORT)[["stay_id", "hour"]]
    cohort["stay_id"] = cohort.stay_id.astype("int64")
    cohort["hour"] = cohort.hour.astype("float64")
    notes["stay_id"] = notes.stay_id.astype("int64")
    notes["note_hour"] = notes.note_hour.astype("float64")

    return cohort, notes


def needed_note_ids():
    """note_ids selected as the most recent available note for some cohort hour."""
    cohort, notes = _load()
    merged = pd.merge_asof(
        cohort.sort_values("hour").reset_index(drop=True),
        notes[["stay_id", "note_hour", "note_id"]]
            .sort_values("note_hour").reset_index(drop=True),
        left_on="hour", right_on="note_hour", by="stay_id",
        direction="backward", allow_exact_matches=False,   # STRICTLY before t
    )
    age = merged["hour"] - merged["note_hour"]
    usable = merged[(age <= STALENESS_HOURS) & merged.note_id.notna()]
    return sorted(usable.note_id.unique())


def window_note_ids(window=STALENESS_HOURS):
    """Every note reachable by a cohort hour within ``window`` hours BEFORE it.

    The most-recent-only set above is a strict subset of this one. The rich
    notes agent aggregates over the whole window, so it needs all of them --
    but only ~37k more than the most-recent set, because the same report is the
    most recent one for many consecutive hours.

    ``age > 0`` is the same strictly-before-t rule ``merge_asof`` enforces with
    ``allow_exact_matches=False``: a report filed exactly at hour t is not yet
    available at hour t.
    """
    cohort, notes = _load()
    pairs = cohort.merge(notes[["stay_id", "note_hour", "note_id"]], on="stay_id")
    age = pairs["hour"] - pairs["note_hour"]
    reachable = pairs[(age > 0) & (age <= window)]
    return sorted(reachable.note_id.unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["recent", "window"], default="recent")
    ap.add_argument("--window", type=float, default=STALENESS_HOURS)
    args = ap.parse_args()

    ids = (needed_note_ids() if args.mode == "recent"
           else window_note_ids(args.window))
    label = ("most recent before t" if args.mode == "recent"
             else f"anywhere in the {args.window:g}h window before t")
    print(f"[resolve] {len(ids):,} distinct notes are reachable by some "
          f"patient-hour ({label})")

    existing = pd.read_parquet(OUT) if OUT.exists() else None
    if existing is not None:
        have = set(existing.note_id.astype(str))
        ids = [i for i in ids if str(i) not in have]
        print(f"[resolve] {len(have):,} already cached, {len(ids):,} to fetch")
        if not ids:
            print(f"[skip] {OUT} already covers them")
            return

    client = bigquery.Client(project=PROJECT)
    sql = """
    SELECT note_id, text
    FROM `physionet-data.mimiciv_note.radiology`
    WHERE note_id IN UNNEST(@ids)
    """
    job = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", ids)])
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False,
        query_parameters=job.query_parameters))
    print(f"[dry-run] {dry.total_bytes_processed/1e9:.2f} GB")

    df = client.query(sql, job_config=job).result().to_dataframe(
        create_bqstorage_client=True)
    if existing is not None:
        # append rather than overwrite: the window set is a superset of the
        # most-recent set, and re-downloading what we already have is waste
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates("note_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    chars = df.text.str.len()
    print(f"[done] {len(df):,} note texts -> {OUT} "
          f"({OUT.stat().st_size/1e6:.0f} MB); "
          f"median {chars.median():.0f} chars, p90 {chars.quantile(.9):.0f}")


if __name__ == "__main__":
    main()
