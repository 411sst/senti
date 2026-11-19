"""T44-T49 preparation: MIMIC-IV extracts for stays and hours our cohort leaves out.

    python scripts/t4x_mimic_extracts.py fetch    # BigQuery, read-only; new files only
    # add --variant all for every hour of every stay (T49 samples run to discharge - 24 h)
    python scripts/t4x_mimic_extracts.py reconcile  # local: original values on shared bins
    python scripts/t4x_mimic_extracts.py gate     # local checks against our caches

Our cohort drops stays with an event in the first 6 h and censors every stay
at its first event, but the published cohorts (T45, T46, T49) keep both. The
labs, cadence, fluid and resp extracts already cover every ICU stay of 12 h or
more at every hour, uncensored, so only two sources need new extracts:

* ``scripts/t4x_mimic_raw.sql``: the cohort's vitals/GCS/oxygen bins for all
  stays of 12 h or more, hours 0 to 48, no stabilisation filter, no censoring
  -> ``data/cache/t4x/mimic_raw_uncensored_48h.parquet``.
* ECG: ``scripts/mimic_ecg_index.sql`` without the cohort restriction, and the
  machine measurements of every study it lists, featurised with the unchanged
  ``build_meas_features`` -> ``data/cache/t4x/ecg_index_all.parquet``,
  ``ecg_machine_measurements_all.parquet``, ``ecg_meas_features_all.npz``.

``reconcile`` copies the original extract's observation values onto every
(stay, hour) bin both extracts have: BigQuery's AVG sums in a nondeterministic
order, so the re-run differs in the last bit on ~0.1 % of bins, which flips a
NEWS2 band on a handful of rows. The new extract then supplies only the bins our
cohort never had -> ``mimic_raw_uncensored_48h_reconciled.parquet``.

``gate`` rebuilds features from the reconciled raw extract with the unchanged
``cohort_from_raw`` and requires that every row our cohort also has comes out
identical, and that the new ECG files agree with ours on every shared stay and
study. Aggregates go to ``data/cache/t4x/mimic_extracts_gate.json``.
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
RAW = OUT / "mimic_raw_uncensored_48h.parquet"
RAW_RECON = OUT / "mimic_raw_uncensored_48h_reconciled.parquet"
ECG_INDEX = OUT / "ecg_index_all.parquet"
ECG_MEAS = OUT / "ecg_machine_measurements_all.parquet"
ECG_FEAT = OUT / "ecg_meas_features_all.npz"
GATE = OUT / "mimic_extracts_gate.json"
STAB_HOURS = 6
MAX_HOUR = 48
STALENESS = 72


def _set_variant(variant):
    """``48h`` (hours 0-48, T45/T46) or ``all`` (no cap, T49)."""
    global RAW, RAW_RECON, GATE, MAX_HOUR
    if variant == "all":
        RAW = OUT / "mimic_raw_uncensored_all.parquet"
        RAW_RECON = OUT / "mimic_raw_uncensored_all_reconciled.parquet"
        GATE = OUT / "mimic_extracts_gate_all.json"
        MAX_HOUR = 1_000_000


def _client():
    from google.cloud import bigquery
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise SystemExit("set GOOGLE_CLOUD_PROJECT")
    return bigquery, bigquery.Client(project=project)


def _run(sql, params, label):
    bigquery, client = _client()
    use_storage = os.environ.get("SENTINEL_BQ_STORAGE", "1") != "0"
    cfg = dict(query_parameters=params)
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False, **cfg))
    print(f"[dry-run] {label}: {dry.total_bytes_processed / 1e9:.2f} GB", flush=True)
    return client.query(sql, job_config=bigquery.QueryJobConfig(**cfg)).result().to_dataframe(
        create_bqstorage_client=use_storage)


def _write(df, path):
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    df.to_parquet(path, index=False)
    print(f"[done] {len(df):,} rows -> {path.relative_to(REPO)}", flush=True)


def fetch():
    from google.cloud.bigquery import ScalarQueryParameter as P

    OUT.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        sql = (REPO / "scripts" / "t4x_mimic_raw.sql").read_text()
        df = _run(sql, [P("stab_hours", "INT64", STAB_HOURS), P("max_hour", "INT64", MAX_HOUR)],
                  "uncensored raw")
        _write(df, RAW)
    if not ECG_INDEX.exists():
        sql = (REPO / "scripts" / "mimic_ecg_index.sql").read_text()
        df = _run(sql, [P("staleness", "INT64", STALENESS)], "ECG index, all stays")
        df["stay_id"] = df.stay_id.astype("int64")
        df["study_id"] = df.study_id.astype("int64")
        df["ecg_hour"] = df.ecg_hour.astype("float64")
        idx = (df[["stay_id", "ecg_hour", "study_id", "path"]]
               .sort_values(["stay_id", "ecg_hour"]).reset_index(drop=True))
        _write(idx, ECG_INDEX)
    if not ECG_MEAS.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fetch_ecg_measurements", REPO / "scripts" / "fetch_ecg_measurements.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        df = _run(mod.SQL, [P("staleness", "INT64", STALENESS)], "ECG measurements, all stays")
        df["study_id"] = df.study_id.astype("int64")
        keep = set(pd.read_parquet(ECG_INDEX, columns=["study_id"]).study_id)
        df = df[df.study_id.isin(keep)].drop_duplicates("study_id").reset_index(drop=True)
        _write(df, ECG_MEAS)
    if not ECG_FEAT.exists():
        from agents.ecg_meas_features import build_meas_features
        study_ids, X, names = build_meas_features(pd.read_parquet(ECG_MEAS))
        assert np.isfinite(X).all()
        np.savez_compressed(ECG_FEAT, embeddings=X, study_ids=study_ids,
                            feature_names=np.array(names))
        print(f"[done] {X.shape[0]:,} studies x {X.shape[1]} -> {ECG_FEAT.relative_to(REPO)}")


def reconcile():
    from mimic_cohort import CARRIED, RAW_PATH
    from uncensored_frames import reconcile as rec
    if RAW_RECON.exists():
        raise SystemExit(f"refusing to overwrite {RAW_RECON}")
    new, old = pd.read_parquet(RAW), pd.read_parquet(RAW_PATH)
    stats = {}
    keys = ["stay_id", "hour"]
    m = old.merge(new, on=keys, suffixes=("_o", "_n"))
    for c in CARRIED:
        o = pd.to_numeric(m[c + "_o"], errors="coerce").to_numpy(float)
        n = pd.to_numeric(m[c + "_n"], errors="coerce").to_numpy(float)
        both = ~np.isnan(o) & ~np.isnan(n)
        stats[c] = {"bins_differing": int((o[both] != n[both]).sum()),
                    "max_abs_diff": float(np.abs(o[both] - n[both]).max()) if both.any() else 0.0,
                    "missing_pattern_mismatches": int((np.isnan(o) != np.isnan(n)).sum())}
    out, n_shared = rec(new, old, CARRIED)
    out.to_parquet(RAW_RECON, index=False)
    res = {"shared_bins": n_shared, "per_channel_before": stats}
    (OUT / RAW_RECON.name.replace(".parquet", ".json")).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def gate():
    from agents.ecg_agent import ECG_INDEX as OUR_INDEX
    from agents.ecg_meas_agent import MEAS_CACHE
    from feed_outage import COHORT_KEEP, cohort_from_raw, frame_differences
    from mimic_cohort import OUT_PATH as COHORT

    result = {}
    raw = pd.read_parquet(RAW_RECON)
    result["raw"] = {"rows": int(len(raw)), "stays": int(raw.stay_id.nunique()),
                     "patients": int(raw.subject_id.nunique()),
                     "max_hour": int(pd.to_numeric(raw.hour).max())}
    cohort = pd.read_parquet(COHORT)
    ours = np.sort(cohort.stay_id.unique())
    sub = raw[np.isin(raw.stay_id.to_numpy("int64"), ours)].copy()
    feat = cohort_from_raw(sub).sort_values(["stay_id", "hour"]).reset_index(drop=True)
    ref = cohort[cohort.hour <= MAX_HOUR].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    key_f = pd.MultiIndex.from_frame(feat[["stay_id", "hour"]].astype("int64"))
    key_r = pd.MultiIndex.from_frame(ref[["stay_id", "hour"]].astype("int64"))
    shared = key_f.isin(key_r)
    a = feat[shared].reset_index(drop=True)
    missing = int((~key_r.isin(key_f)).sum())
    cols = [c for c in COHORT_KEEP if c in ref.columns and c != "patient_id"]
    diff = frame_differences(a, ref, cols) if missing == 0 else ["<rows missing>"]
    result["cohort_rows"] = {"cohort_rows_to_48h": int(len(ref)), "missing_from_new": missing,
                             "columns_compared": len(cols), "columns_differing": diff,
                             "new_rows_outside_cohort": int((~shared).sum())}

    old_idx = pd.read_parquet(OUR_INDEX)
    new_idx = pd.read_parquet(ECG_INDEX)
    mine = new_idx[new_idx.stay_id.isin(set(old_idx.stay_id))]
    same = (mine.sort_values(["stay_id", "study_id", "ecg_hour"])[["stay_id", "study_id", "ecg_hour"]]
            .reset_index(drop=True)
            .equals(old_idx.sort_values(["stay_id", "study_id", "ecg_hour"])
                    [["stay_id", "study_id", "ecg_hour"]].reset_index(drop=True)))
    result["ecg_index"] = {"rows": int(len(new_idx)), "stays": int(new_idx.stay_id.nunique()),
                           "our_stays_identical": bool(same)}
    z0, z1 = np.load(MEAS_CACHE), np.load(ECG_FEAT)
    pos = {int(s): i for i, s in enumerate(z1["study_ids"])}
    common = [i for i, s in enumerate(z0["study_ids"]) if int(s) in pos]
    j = [pos[int(z0["study_ids"][i])] for i in common]
    result["ecg_features"] = {
        "studies": int(len(z1["study_ids"])), "ours": int(len(z0["study_ids"])),
        "ours_present": int(len(common)),
        "names_identical": bool(list(z0["feature_names"]) == list(z1["feature_names"])),
        "values_identical": bool(np.array_equal(z0["embeddings"][common], z1["embeddings"][j]))}
    ok = (missing == 0 and not diff and same and len(common) == len(z0["study_ids"])
          and result["ecg_features"]["names_identical"]
          and result["ecg_features"]["values_identical"])
    result["gate_passed"] = bool(ok)
    GATE.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not ok:
        raise SystemExit("GATE FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "reconcile", "gate"])
    ap.add_argument("--variant", choices=["48h", "all"], default="48h")
    a = ap.parse_args()
    _set_variant(a.variant)
    {"fetch": fetch, "reconcile": reconcile, "gate": gate}[a.stage]()


if __name__ == "__main__":
    main()
