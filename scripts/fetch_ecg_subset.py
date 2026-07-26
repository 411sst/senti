"""Download the ECG waveforms needed for a stay-level subset of the cohort.

    python scripts/fetch_ecg_subset.py [--stays 1500]

Pulls ONLY the studies that a cohort patient-hour can actually see: the most
recent ECG strictly before hour ``t``, within the 72h staleness bound. For 1,500
stays that is ~2,300 of the 800,035 ECGs in MIMIC-IV-ECG.

PhysioNet rate-limits us to ~70 KB/s regardless of concurrency (measured: 72
KB/s on one stream, 63 KB/s aggregate across eight), so this is bandwidth-bound
at roughly an hour for the subset. It is resumable -- already-downloaded records
are skipped -- so an interruption costs nothing.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = "https://physionet.org/files/mimic-iv-ecg/1.0"
OUT_DIR = REPO / "data" / "ecg_waveforms"
INDEX = REPO / "data" / "cache" / "ecg_index.parquet"
COHORT = REPO / "data" / "cache" / "mimic_cohort_hourly.parquet"
STALENESS_HOURS = 72


def covered_hours(cohort, ecg, bound=STALENESS_HOURS):
    """Cohort rows joined to the most recent ECG strictly before each hour."""
    cohort = cohort.copy()
    cohort["stay_id"] = cohort.stay_id.astype("int64")
    cohort["hour"] = cohort.hour.astype("float64")
    ecg = ecg.copy()
    ecg["stay_id"] = ecg.stay_id.astype("int64")
    ecg["ecg_hour"] = ecg.ecg_hour.astype("float64")

    merged = pd.merge_asof(
        cohort.sort_values("hour").reset_index(drop=True),
        ecg[["stay_id", "ecg_hour", "study_id", "path"]]
            .sort_values("ecg_hour").reset_index(drop=True),
        left_on="hour", right_on="ecg_hour", by="stay_id",
        direction="backward", allow_exact_matches=False,   # STRICTLY before t
    )
    merged["ecg_age"] = merged["hour"] - merged["ecg_hour"]
    merged.loc[merged["ecg_age"] > bound, ["study_id", "path", "ecg_age"]] = np.nan
    return merged


def select_studies(n_stays, seed=42):
    """The exact study set the sampled stays need (same sampling as the sizing run)."""
    cohort = pd.read_parquet(COHORT)[["stay_id", "hour", "split", "label_24h"]]
    ecg = pd.read_parquet(INDEX)
    merged = covered_hours(cohort, ecg)
    cov = merged[merged.study_id.notna()]

    rng = np.random.default_rng(seed)
    stays = cov.stay_id.unique()
    pick = rng.choice(stays, min(n_stays, len(stays)), replace=False)
    subset = cov[cov.stay_id.isin(set(pick))]
    studies = subset[["study_id", "path"]].drop_duplicates("study_id")
    return set(pick), subset, studies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stays", type=int, default=1500)
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()

    pick, subset, studies = select_studies(args.stays)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[subset] {len(pick):,} stays | {len(subset):,} ECG-covered patient-hours "
          f"| {int(subset.label_24h.sum()):,} positive")
    print(f"[subset] {len(studies):,} unique ECG studies to fetch")

    todo = []
    for _, row in studies.iterrows():
        sid = str(int(row.study_id))
        if (OUT_DIR / f"{sid}.dat").exists() and (OUT_DIR / f"{sid}.hea").exists():
            continue
        todo.append((sid, row.path))
    print(f"[fetch] {len(studies) - len(todo):,} already on disk, {len(todo):,} to go")
    if not todo:
        print("[fetch] nothing to do")
        return

    # Batch into curl config files so progress is checkpointed by the filesystem:
    # anything already written is skipped on the next run.
    t0 = time.time()
    chunk = 40
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        cfg = OUT_DIR / "_dl.cfg"
        with cfg.open("w") as fh:
            for sid, path in batch:
                for ext in (".hea", ".dat"):
                    fh.write(f'url = "{BASE}/{path}{ext}"\n')
                    fh.write(f'output = "{OUT_DIR}/{sid}{ext}"\n')
        subprocess.run(["curl", "-sS", "-Z", "--parallel-max", str(args.parallel),
                        "--retry", "3", "--retry-delay", "2", "--max-time", "120",
                        "-K", str(cfg)], check=False)
        done = i + len(batch)
        rate = done / max(time.time() - t0, 1e-9)
        eta = (len(todo) - done) / max(rate, 1e-9) / 60
        print(f"  {done:,}/{len(todo):,} studies  ({rate:.2f}/s, ETA {eta:.0f} min)",
              flush=True)

    have = len(list(OUT_DIR.glob("*.dat")))
    print(f"[done] {have:,} records on disk in {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
