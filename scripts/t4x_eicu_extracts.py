"""T44/T45 preparation: eICU hourly rows for every hour of every stay of 24 h or more.

    python scripts/t4x_eicu_extracts.py fetch    # BigQuery, read-only; new file only
    python scripts/t4x_eicu_extracts.py reconcile  # local: T22's values on shared bins
    python scripts/t4x_eicu_extracts.py build    # local: the unchanged eICU builder
    python scripts/t4x_eicu_extracts.py gate     # local checks against T22's caches

T22's eICU cohort drops stays with an event in the first 6 h, censors at the
first event and stops at 14 days. Sheikhalishahi's cohorts keep all of these
(T44: "drop no records"), so the vitals extract is rerun without them
(``scripts/t4x_eicu_raw.sql``) and built with the unchanged
``eicu_cohort.build``. The labs, fluids and resp extracts already cover every
hour of every stay of 12 h or more and are reused as they are (checked).

As on MIMIC, BigQuery's AVG can differ in the last bit between runs, so
``reconcile`` copies T22's raw observation values onto every bin both extracts
have before building (``uncensored_frames.reconcile``).

Files: ``data/cache/t4x/eicu_raw_uncensored.parquet`` (raw bins),
``data/cache/t4x/eicu_raw_uncensored_reconciled.parquet``,
``data/cache/t4x/eicu_frame_uncensored.parquet`` (built rows),
``data/cache/t4x/eicu_extracts_gate.json`` (aggregates).
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CACHE = REPO / "data" / "cache"
OUT = CACHE / "t4x"
RAW = OUT / "eicu_raw_uncensored.parquet"
RAW_RECON = OUT / "eicu_raw_uncensored_reconciled.parquet"
FRAME = OUT / "eicu_frame_uncensored.parquet"
GATE = OUT / "eicu_extracts_gate.json"
STAB_HOURS = 6


def fetch():
    from google.cloud import bigquery
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise SystemExit("set GOOGLE_CLOUD_PROJECT")
    if RAW.exists():
        raise SystemExit(f"refusing to overwrite {RAW}")
    use_storage = os.environ.get("SENTINEL_BQ_STORAGE", "1") != "0"
    client = bigquery.Client(project=project)
    sql = (REPO / "scripts" / "t4x_eicu_raw.sql").read_text()
    params = [bigquery.ScalarQueryParameter("stab_hours", "INT64", STAB_HOURS)]
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False, query_parameters=params))
    print(f"[dry-run] {dry.total_bytes_processed / 1e9:.2f} GB", flush=True)
    df = client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=params)).result().to_dataframe(create_bqstorage_client=use_storage)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW, index=False)
    print(f"[done] {len(df):,} rows, {df.stay_id.nunique():,} stays -> {RAW.name}", flush=True)


def reconcile():
    from eicu_cohort import RAW_PATH as T22_RAW
    from mimic_cohort import CARRIED
    from uncensored_frames import reconcile as rec
    if RAW_RECON.exists():
        raise SystemExit(f"refusing to overwrite {RAW_RECON}")
    new, old = pd.read_parquet(RAW), pd.read_parquet(T22_RAW)
    m = old.merge(new, on=["stay_id", "hour"], suffixes=("_o", "_n"))
    stats = {}
    for c in CARRIED:
        o = pd.to_numeric(m[c + "_o"], errors="coerce").to_numpy(float)
        n = pd.to_numeric(m[c + "_n"], errors="coerce").to_numpy(float)
        both = ~np.isnan(o) & ~np.isnan(n)
        stats[c] = {"bins_differing": int((o[both] != n[both]).sum()),
                    "max_abs_diff": float(np.abs(o[both] - n[both]).max()) if both.any() else 0.0,
                    "missing_pattern_mismatches": int((np.isnan(o) != np.isnan(n)).sum())}
    del m
    out, n_shared = rec(new, old, CARRIED)
    out.to_parquet(RAW_RECON, index=False)
    res = {"shared_bins": n_shared, "per_channel_before": stats}
    (OUT / "eicu_reconcile.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def build():
    from eicu_cohort import build as build_eicu
    if FRAME.exists():
        raise SystemExit(f"refusing to overwrite {FRAME}")
    out = build_eicu(raw_path=RAW_RECON, out_path=FRAME)
    print(f"[done] {len(out):,} rows, {out.stay_id.nunique():,} stays -> {FRAME.name}")


def gate():
    from eicu_cohort import OUT_PATH as T22_FRAME
    from feed_outage import frame_differences
    from modality_features import load_fluids, load_labs, load_resp
    from score_eicu import EICU_FLUIDS, EICU_LABS, EICU_RESP

    new = pd.read_parquet(FRAME)
    old = pd.read_parquet(T22_FRAME)
    res = {"new": {"rows": int(len(new)), "stays": int(new.stay_id.nunique()),
                   "patients": int(new.patient_id.nunique()),
                   "max_hour": int(new.hour.max())},
           "t22": {"rows": int(len(old)), "stays": int(old.stay_id.nunique())}}
    k_new = pd.MultiIndex.from_frame(new[["stay_id", "hour"]].astype("int64"))
    k_old = pd.MultiIndex.from_frame(old[["stay_id", "hour"]].astype("int64"))
    # T22 stays shorter than 24 h are outside this superset by design
    los24 = set(new.stay_id.unique())
    in_scope = old.stay_id.isin(los24).to_numpy()
    missing = int((~k_old[in_scope].isin(k_new)).sum())
    a = new[k_new.isin(k_old)].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    b = old[in_scope].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    cols = [c for c in old.columns if c in new.columns and c != "split"]
    diff = frame_differences(a, b, cols) if missing == 0 else ["<rows missing>"]
    res["t22_rows_in_scope"] = int(in_scope.sum())
    res["t22_rows_missing_from_new"] = missing
    res["columns_compared"] = len(cols)
    res["columns_differing"] = diff
    cover = {}
    ids = np.sort(new.stay_id.unique())
    for name, df in (("labs", load_labs(EICU_LABS)), ("fluids", load_fluids(EICU_FLUIDS)),
                     ("resp", load_resp(EICU_RESP))):
        sub = df[np.isin(df.stay_id.to_numpy("int64"), ids)]
        cover[name] = {"stays_with_rows": int(sub.stay_id.nunique()),
                       "max_hour": float(sub.hour.max())}
    res["modality_coverage_of_new_stays"] = cover
    res["gate_passed"] = bool(missing == 0 and not diff)
    GATE.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    if not res["gate_passed"]:
        raise SystemExit("GATE FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "reconcile", "build", "gate"])
    {"fetch": fetch, "reconcile": reconcile, "build": build, "gate": gate}[ap.parse_args().stage]()


if __name__ == "__main__":
    main()
