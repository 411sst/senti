"""Do the published GBM specs still lose on eICU? The comparative external test.

    python scripts/score_eicu_baselines.py [--specs ecart yaib sadanandan]

``scripts/score_eicu.py`` answers "does Sentinel transfer" -- degradation
against itself. This answers the question the shipping claim actually rests
on: **does Sentinel still beat the published specs on 208 hospitals it has
never seen?** Beating them on one hospital could be tuning to that hospital;
beating them across 208 unseen ones cannot.

Protocol, identical to src/lit_gbm.run() and to the flagship's: every spec is
fitted on MIMIC train, early-stopped on MIMIC val (eCART additionally
down-samples train to its published 50% prevalence), then scored unchanged on
eICU. No spec sees an eICU row during fitting, exactly as Sentinel does not.

The honest risk this run has to control for: the specs read channels eICU
charts more thinly than MIMIC does, so a spec could degrade for want of data
rather than for want of generality. Every table below therefore reports the
matched-charting-density cut alongside the pooled one, the same way the
flagship's does -- if Sentinel's margin only survives pooled, that is a data
artifact and not a result.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from baselines_mimic import load_splits  # noqa: E402
from lit_gbm import (  # noqa: E402
    FITTERS, OXY_COLUMNS, SPECS, build_channels, build_spec_design,
    downsample_to_prevalence, load_inputs,
)
from metrics import auprc, auroc  # noqa: E402
from modality_features import load_extra_vitals, load_labs  # noqa: E402

LABEL = "label_24h"
VITALS = ["heart_rate", "resp_rate", "spo2", "sbp", "temperature"]
CACHE = REPO / "data" / "cache"
SENTINEL_SCORES = CACHE / "eicu_twohead_all7.npz"


def as_fio2_percent(values):
    """Force FiO2 onto MIMIC's PERCENTAGE scale, idempotently.

    This has now bitten twice in two different code paths -- the resp extract
    and the cohort's raw parquet -- because eICU stores FiO2 as a fraction
    (0.21-1.0) and MIMIC as a percentage (21-100), and nothing downstream
    validates the scale. Neither failure raised: the values were simply 100x
    too small, which reads as "this patient is on sub-room-air oxygen" and
    quietly degrades whatever consumes it.

    So the scale is enforced here rather than trusted. A fraction scales by
    100, a value already in 21-100 passes through untouched (so running this
    on an already-correct column is a no-op), and anything else becomes NaN.
    """
    v = np.asarray(values, dtype=float)
    out = np.where((v >= 0.21) & (v <= 1.0), v * 100.0,
                   np.where((v >= 21.0) & (v <= 100.0), v, np.nan))
    seen = out[~np.isnan(out)]
    if seen.size and (seen.min() < 21.0 or seen.max() > 100.0):
        raise ValueError(f"FiO2 outside 21-100 after normalising: "
                         f"[{seen.min():.3f}, {seen.max():.3f}]")
    return out


def eicu_inputs():
    """The eICU twins of lit_gbm.load_inputs()'s five tables."""
    cohort = pd.read_parquet(CACHE / "eicu_cohort_hourly.parquet")
    labs = load_labs(CACHE / "eicu_labs_hourly.parquet")
    extra = load_extra_vitals(CACHE / "eicu_extra_vitals_hourly.parquet")
    raw = pd.read_parquet(CACHE / "eicu_raw_hourly.parquet",
                          columns=["stay_id", "hour"] + OXY_COLUMNS)
    raw["stay_id"] = raw["stay_id"].astype("int64")
    raw["hour"] = raw["hour"].astype("int64")
    before = float(pd.to_numeric(raw["fio2"], errors="coerce").median())
    raw["fio2"] = as_fio2_percent(raw["fio2"])
    after = float(pd.to_numeric(raw["fio2"], errors="coerce").median())
    print(f"[guard] eICU raw FiO2 median {before:.2f} -> {after:.2f} (percent scale)")
    meta = pd.read_parquet(
        CACHE / "eicu_stay_meta.parquet",
        columns=["stay_id", "gender", "anchor_age", "admit_hour_of_day"])
    meta["stay_id"] = meta["stay_id"].astype("int64")
    return cohort, labs, extra, raw, meta


def fresh_mask(df, bound):
    m = np.ones(len(df), dtype=bool)
    for c in VITALS:
        m &= df[f"{c}_hrs_since"].to_numpy() <= bound
    return m


def row(tag, y, r):
    p = float(y.mean())
    a = auprc(y, r)
    print(f"  {tag:<34} n={len(y):>9,}  prev {100*p:5.2f}%  "
          f"AUPRC {a:.4f}  lift {a/p:5.2f}  AUROC {auroc(y, r):.4f}")
    return a / p, auroc(y, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", default=["ecart", "yaib", "sadanandan"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    splits, labs, extra, raw, meta = load_inputs()
    e_cohort, e_labs, e_extra, e_raw, e_meta = eicu_inputs()
    assert not (set(e_cohort.stay_id.unique())
                & set(pd.concat([splits[s] for s in splits]).stay_id.unique())), \
        "eICU/MIMIC stay_id collision"

    built = {}
    for split in ("train", "val", "test"):
        built[split] = build_channels(splits[split], labs, extra, raw, meta)
        print(f"[channels] MIMIC {split}: {len(built[split][0]):,} rows")
    e_ordered, e_channels = build_channels(e_cohort, e_labs, e_extra, e_raw, e_meta)
    print(f"[channels] eICU: {len(e_ordered):,} rows x {e_channels.shape[1]} channels")

    y = {s: built[s][0][LABEL].to_numpy(float) for s in built}
    y_e = e_ordered[LABEL].to_numpy(float)
    results = {}

    for name in args.specs:
        print(f"\n=== {name} ===")
        t0 = time.time()
        designs = {s: build_spec_design(name, *built[s])[0] for s in built}
        X_e, prov = build_spec_design(name, e_ordered, e_channels)
        X_tr, y_tr = designs["train"], y["train"]
        if "downsample_prevalence" in prov:
            keep = downsample_to_prevalence(y_tr, prov["downsample_prevalence"])
            print(f"  down-sampled train -> {len(keep):,} rows "
                  f"({y_tr[keep].mean():.1%}, as published)")
            X_tr, y_tr = X_tr.iloc[keep], y_tr[keep]
        model, _ = FITTERS[prov["library"]](X_tr, y_tr, designs["val"],
                                            y["val"], device=args.device)
        s_test = model.predict_proba(designs["test"])[:, 1].astype(np.float64)
        s_e = model.predict_proba(X_e)[:, 1].astype(np.float64)
        results[name] = {"test": s_test, "eicu": s_e}
        print(f"  fitted + scored in {time.time()-t0:.0f}s")
        del designs, X_e

    np.savez_compressed(CACHE / "eicu_lit_scores.npz",
                        stay_id=e_ordered.stay_id.to_numpy(),
                        hour=e_ordered.hour.to_numpy(),
                        **{f"eicu_{k}": v["eicu"] for k, v in results.items()},
                        **{f"test_{k}": v["test"] for k, v in results.items()})

    # --- Sentinel, aligned on the same rows -----------------------------
    sent = None
    if SENTINEL_SCORES.exists():
        z = np.load(SENTINEL_SCORES)
        if (z["stay_id"] == e_ordered.stay_id.to_numpy()).all() and \
           (z["hour"] == e_ordered.hour.to_numpy()).all():
            sent = z["risk"].astype(float)
            print(f"\n[sentinel] two-head flagship scores aligned on eICU rows")
        else:
            print("\n[sentinel] score cache misaligned with these rows; skipped")

    t_ordered = built["test"][0]
    for tag, bound in (("pooled", None), ("fresh <= 2h", 2)):
        print(f"\n=== eICU, {tag} — Sentinel vs the published specs ===")
        me = np.ones(len(e_ordered), bool) if bound is None else fresh_mask(e_ordered, bound)
        mt = np.ones(len(t_ordered), bool) if bound is None else fresh_mask(t_ordered, bound)
        for name in args.specs:
            row(f"{name:<10} MIMIC test", y["test"][mt], results[name]["test"][mt])
            row(f"{name:<10} eICU", y_e[me], results[name]["eicu"][me])
        if sent is not None:
            print("  " + "-" * 92)
            row("SENTINEL two-head  eICU", y_e[me], sent[me])


if __name__ == "__main__":
    main()
