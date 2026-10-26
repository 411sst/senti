"""Build the ecg_meas feature cache from the fetched machine measurements.

Reads ``data/cache/ecg_machine_measurements.parquet`` (fetch_ecg_measurements.py),
featurizes every study (agents/ecg_meas_features.py) and writes
``data/cache/ecg_meas_features.npz`` in the exact npz format the frozen-embedding
EcgAgent machinery consumes: keys ``embeddings`` (N, F), ``study_ids`` (N,),
plus ``feature_names`` for the record.

Prints the per-flag match census and the most frequent UNMATCHED statements --
regex fragility is the likely silent killer of this agent, so make it visible.

Usage:
    python scripts/build_ecg_meas_features.py [--force]
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agents.ecg_meas_features import build_meas_features, match_census  # noqa: E402

IN_PATH = REPO / "data" / "cache" / "ecg_machine_measurements.parquet"
OUT_PATH = REPO / "data" / "cache" / "ecg_meas_features.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if OUT_PATH.exists() and not args.force:
        print(f"[skip] {OUT_PATH} exists (use --force to rebuild)")
        return
    if not IN_PATH.exists():
        raise SystemExit(f"[error] {IN_PATH} missing -- run fetch_ecg_measurements.py")

    df = pd.read_parquet(IN_PATH)
    print(f"[in] {len(df):,} studies")

    counts, unmatched = match_census(df)
    print("\n[census] per-flag positives:")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<16} {n:>7,}  ({100*n/len(df):.1f}%)")
    print("\n[census] top unmatched report_0 statements:")
    for stmt, n in unmatched.items():
        print(f"    {n:>6,}  {stmt[:90]}")

    study_ids, X, names = build_meas_features(df)
    dropped = len(df) - len(study_ids)
    print(f"\n[features] {X.shape[0]:,} studies x {X.shape[1]} features "
          f"({dropped:,} dropped as fully uninformative, "
          f"{100*dropped/len(df):.2f}%)")
    assert np.isfinite(X).all(), "non-finite feature values"

    np.savez_compressed(OUT_PATH, embeddings=X, study_ids=study_ids,
                        feature_names=np.array(names))
    print(f"[done] -> {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
