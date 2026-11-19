"""T49 preparation: Sadanandan's cohort on MIMIC-IV 3.1, and its count check.

    python scripts/t49_cohort.py fetch    # BigQuery, read-only; new files only
    python scripts/t49_cohort.py build    # local: paper-text rules, samples, counts
    python scripts/t49_cohort.py build-code     # local: THEIR CODE's rules (T49 uses these)
    python scripts/t49_cohort.py fetch-bridge   # BigQuery: the bridge's 26 hourly inputs

``fetch`` runs ``scripts/t49_stays.sql`` and ``scripts/t49_therapy.sql``
(dry-running each first to report the scan) and writes
``data/cache/t49_stays.parquet`` and ``data/cache/t49_therapy.parquet``.
Nothing is uploaded: the queries read PhysioNet's tables and the results are
downloaded.

``build`` applies ``src/sadanandan_cohort.py`` and writes aggregates to
``data/cache/t49_cohort.json`` and the sample tables (local cache, never
committed) to ``data/cache/t49_samples_{restricted,unrestricted}.parquet``.

``build-code`` applies their public code's rules (``code_cohort``,
``code_samples``, the therapy items of ``scripts/t49_therapy_code.sql``) and
writes ``data/cache/t49_cohort_code.json`` and ``data/cache/t49_samples_code.parquet``.
``fetch-bridge`` runs ``scripts/t49_bridge.sql`` -> ``data/cache/t49_bridge_hourly.parquet``.

Count check (spec): 74,822 stays, 61,437 patients, 5,720,904 samples, 2.8 %
positive; any difference above 10 % stops T49 for a decision.
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
STAYS = CACHE / "t49_stays.parquet"
THERAPY = CACHE / "t49_therapy.parquet"
THERAPY_CODE = CACHE / "t49_therapy_code.parquet"
SAMPLES_CODE = CACHE / "t49_samples_code.parquet"
OUT_JSON_CODE = CACHE / "t49_cohort_code.json"
BRIDGE = CACHE / "t49_bridge_hourly.parquet"
BRIDGE_DESIGN = CACHE / "t49"          # bridge_{train,val,test}.parquet
OUT_JSON = CACHE / "t49_cohort.json"
PUBLISHED = {"stays": 74822, "patients": 61437, "samples": 5720904,
             "positive_rate": 0.028, "positives": 160395,
             "table3": {"icu_death": 5612, "vasopressor": 28445,
                        "ventilation": 19234, "any_event": 35891}}
TOLERANCE = 0.10


def fetch():
    from google.cloud import bigquery

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise SystemExit("set GOOGLE_CLOUD_PROJECT")
    use_storage = os.environ.get("SENTINEL_BQ_STORAGE", "1") != "0"
    client = bigquery.Client(project=project)
    for sql_name, out in (("t49_stays.sql", STAYS), ("t49_therapy.sql", THERAPY)):
        if out.exists():
            print(f"[skip] {out} exists")
            continue
        sql = (REPO / "scripts" / sql_name).read_text()
        dry = client.query(sql, job_config=bigquery.QueryJobConfig(
            dry_run=True, use_query_cache=False))
        print(f"[dry-run] {sql_name}: {dry.total_bytes_processed / 1e9:.2f} GB", flush=True)
        df = client.query(sql).result().to_dataframe(create_bqstorage_client=use_storage)
        for c in df.columns:
            if str(df[c].dtype) in ("Int64", "Float64", "boolean"):
                df[c] = df[c].astype(float if str(df[c].dtype) != "boolean" else bool)
        df.to_parquet(out, index=False)
        print(f"[done] {sql_name}: {len(df):,} rows -> {out.name}", flush=True)


def _rel(ours, theirs):
    return (ours - theirs) / theirs


def build():
    from baselines_mimic import load_splits
    from sadanandan_cohort import (first_starts, include, samples,
                                   stay_outcomes)

    stays = pd.read_parquet(STAYS)
    therapy = pd.read_parquet(THERAPY)
    starts = first_starts(therapy)
    splits = load_splits()
    split_of = {}
    for name, df in splits.items():
        for p in df["patient_id"].unique():
            split_of[int(p)] = name

    result = {"published": PUBLISHED, "tolerance": TOLERANCE, "variants": {}}
    for restrict in (True, False):
        for labs_in in ("icu", "hadm"):
            kept, steps = include(stays, restrict_years=restrict, labs_in=labs_in)
            key = f"{'restricted' if restrict else 'unrestricted'}_labs_{labs_in}"
            v = {"steps": [{"step": s, "stays": n, "patients": p} for s, n, p in steps],
                 "stays": int(len(kept)), "patients": int(kept["subject_id"].nunique()),
                 "outcomes": stay_outcomes(kept, starts)}
            for last, tag in ((48, "hours_6_48"), (None, "hours_6_discharge")):
                smp = samples(kept, starts, last_hour=last)
                sp = smp["subject_id"].map(split_of).fillna("not_in_our_cohort")
                v[tag] = {
                    "samples": int(len(smp)),
                    "positives": int(smp["label"].sum()),
                    "positive_rate": float(smp["label"].mean()),
                    "started_share": float(smp["started"].mean()),
                    "by_our_split": {k: {"samples": int((sp == k).sum()),
                                         "stays": int(smp.loc[sp == k, "stay_id"].nunique()),
                                         "patients": int(smp.loc[sp == k, "subject_id"].nunique()),
                                         "positives": int(smp.loc[sp == k, "label"].sum())}
                                     for k in ("train", "val", "test", "not_in_our_cohort")},
                }
                if labs_in == "icu" and last == 48:
                    out = CACHE / f"t49_samples_{'restricted' if restrict else 'unrestricted'}.parquet"
                    if not out.exists():
                        smp.assign(our_split=sp.to_numpy()).to_parquet(out, index=False)
            checks = {
                "stays": _rel(v["stays"], PUBLISHED["stays"]),
                "patients": _rel(v["patients"], PUBLISHED["patients"]),
                "samples_6_48": _rel(v["hours_6_48"]["samples"], PUBLISHED["samples"]),
                "samples_6_discharge": _rel(v["hours_6_discharge"]["samples"], PUBLISHED["samples"]),
                "positive_rate_6_48": _rel(v["hours_6_48"]["positive_rate"], PUBLISHED["positive_rate"]),
                "positive_rate_6_discharge": _rel(v["hours_6_discharge"]["positive_rate"],
                                                  PUBLISHED["positive_rate"]),
            }
            v["relative_difference"] = {k: round(float(x), 4) for k, x in checks.items()}
            result["variants"][key] = v
    OUT_JSON.write_text(json.dumps(result, indent=2))

    for key, v in result["variants"].items():
        print(f"\n== {key}: {v['stays']:,} stays, {v['patients']:,} patients "
              f"(published 74,822 / 61,437)")
        for st in v["steps"]:
            print(f"   {st['step']:<40s} {st['stays']:>7,} stays {st['patients']:>7,} patients")
        for tag in ("hours_6_48", "hours_6_discharge"):
            s = v[tag]
            print(f"   {tag:<18s} samples {s['samples']:>10,}  positive {100 * s['positive_rate']:.2f}%"
                  f"  ({s['positives']:,})  test split: {s['by_our_split']['test']['samples']:,} samples, "
                  f"{s['by_our_split']['test']['patients']:,} patients")
        print(f"   outcomes {v['outcomes']} (published {PUBLISHED['table3']})")
        print(f"   relative difference {v['relative_difference']}")


def _split_of():
    from baselines_mimic import load_splits
    split_of = {}
    for name, df in load_splits().items():
        for p in df["patient_id"].unique():
            split_of[int(p)] = name
    return split_of


def build_code():
    from sadanandan_cohort import code_cohort, code_samples, first_starts, stay_outcomes
    if SAMPLES_CODE.exists() or OUT_JSON_CODE.exists():
        raise SystemExit("refusing to overwrite the code-rule outputs")
    stays = pd.read_parquet(STAYS)
    starts = first_starts(pd.read_parquet(THERAPY_CODE))
    kept, steps = code_cohort(stays)
    smp = code_samples(kept, starts)
    sp = smp["subject_id"].map(_split_of()).fillna("not_in_our_cohort")
    v = {"rules": "their code at 785d8d7", "published": PUBLISHED,
         "steps": [{"step": a, "stays": n, "patients": q} for a, n, q in steps],
         "stays": int(len(kept)), "patients": int(kept["subject_id"].nunique()),
         "samples": int(len(smp)), "positives": int(smp["label"].sum()),
         "positive_rate": float(smp["label"].mean()),
         "started_share": float(smp["started"].mean()),
         "outcomes": stay_outcomes(kept, starts),
         "by_our_split": {k: {"samples": int((sp == k).sum()),
                              "stays": int(smp.loc[sp == k, "stay_id"].nunique()),
                              "patients": int(smp.loc[sp == k, "subject_id"].nunique()),
                              "positives": int(smp.loc[sp == k, "label"].sum())}
                          for k in ("train", "val", "test", "not_in_our_cohort")}}
    v["relative_difference"] = {
        "stays": round(_rel(v["stays"], PUBLISHED["stays"]), 4),
        "patients": round(_rel(v["patients"], PUBLISHED["patients"]), 4),
        "samples": round(_rel(v["samples"], PUBLISHED["samples"]), 4),
        "positives": round(_rel(v["positives"], PUBLISHED["positives"]), 4),
        "positive_rate": round(_rel(v["positive_rate"], PUBLISHED["positive_rate"]), 4)}
    smp.assign(our_split=sp.to_numpy()).to_parquet(SAMPLES_CODE, index=False)
    OUT_JSON_CODE.write_text(json.dumps(v, indent=2))
    print(json.dumps({k: v[k] for k in ("stays", "patients", "samples", "positives",
                                        "positive_rate", "relative_difference",
                                        "by_our_split")}, indent=2))


def fetch_bridge():
    from google.cloud import bigquery
    if BRIDGE.exists():
        raise SystemExit(f"refusing to overwrite {BRIDGE}")
    client = bigquery.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"])
    sql = (REPO / "scripts" / "t49_bridge.sql").read_text()
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    print(f"[dry-run] t49_bridge.sql: {dry.total_bytes_processed / 1e9:.2f} GB", flush=True)
    df = client.query(sql).result().to_dataframe(
        create_bqstorage_client=os.environ.get("SENTINEL_BQ_STORAGE", "1") != "0")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(BRIDGE, index=False)
    print(f"[done] {len(df):,} rows, {df.stay_id.nunique():,} stays -> {BRIDGE.name}")


def build_bridge():
    """Bridge features (src/t49_bridge.py) for every code-rule sample in our
    train, validation and test splits; one file per split, rows in sample order."""
    from t49_bridge import bridge_design
    BRIDGE_DESIGN.mkdir(exist_ok=True)
    hourly = pd.read_parquet(BRIDGE)
    stays = pd.read_parquet(STAYS, columns=["stay_id", "los_h", "anchor_age", "gender"])
    smp = pd.read_parquet(SAMPLES_CODE, columns=["stay_id", "hour", "our_split"])
    for split in ("train", "val", "test"):
        path = BRIDGE_DESIGN / f"bridge_{split}.parquet"
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        keys = smp[smp.our_split == split].reset_index(drop=True)
        X = bridge_design(hourly, stays, keys).astype("float32")
        X.to_parquet(path, index=False)
        print(f"[done] {split}: {len(X):,} rows x {X.shape[1]} -> {path.relative_to(REPO)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "build", "build-code", "fetch-bridge",
                                      "build-bridge"])
    args = ap.parse_args()
    {"fetch": fetch, "build": build, "build-code": build_code,
     "fetch-bridge": fetch_bridge, "build-bridge": build_bridge}[args.stage]()


if __name__ == "__main__":
    main()
