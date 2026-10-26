"""Train the machine-measurements ECG agent (V4, ``ecg_meas``) and report it.

    python scripts/train_ecg_meas_agent.py [--no-rate]

Same reporting contract as the other ECG training scripts: full-cohort AND
ECG-available-subset tables (the subset is the verdict on the modality), the
presence-only floor, the no-ECG single-constant check, and per-event AUPRC with
**death highlighted** -- the orchestrator's known deficit (0.073) makes a death
component gain a secondary win condition.

``--no-rate`` drops the heart-rate-carrying features (hr, rr, sinus_tachy,
sinus_brady, rvr): vitals already stream HR hourly, so if the no-rate variant
collapses to ~chance the honest conclusion is "V4 ~= stale HR".

GATE 1 (proceed to V5): covered-hours VAL AUROC >= 0.55, stay-bootstrap CI
clear of 0.5 -- printed at the end.
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from agents.ecg_agent import presence_only_auprc  # noqa: E402
from agents.ecg_meas_agent import RATE_FEATURES, EcgMeasAgent  # noqa: E402
from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, auroc, ece  # noqa: E402
from mimic_cohort import EVENT_TYPES  # noqa: E402

LABEL = "label_24h"
EMB_DIR = REPO / "data" / "cache" / "embeddings"

OTHERS = {                       # full-cohort test numbers, same split (results.md)
    "labs":       (0.171, 0.162, 0.728, 0.0035, 0.0012, 0.128, 0.100, 0.035),
    "vitals-TCN": (0.159, 0.155, 0.750, 0.0524, 0.0021, 0.084, 0.054, 0.139),
    "NEWS2":      (0.108, 0.107, 0.657, 0.0845, 0.0014, 0.059, 0.041, 0.062),
    "GBM":        (0.170, 0.166, 0.745, 0.0036, 0.0016, 0.081, 0.049, 0.187),
}


def report(title, y, raw, cal, has, evaluated, types, label="ecg_meas"):
    print(f"\n  {title}")
    hdr = (f"    {'model':<14}{'AUPRC raw':>11}{'AUPRC cal':>11}{'AUROC':>8}"
           f"{'ECE raw':>9}{'ECE cal':>9}   " + "".join(f"{t:>9}" for t in types))
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    per_type = {}
    for t in types:
        yt = evaluated[f"{LABEL}_{t}"].to_numpy(dtype=float)
        per_type[t] = auprc(yt, raw) if yt.sum() > 0 else float("nan")
    print(f"    {label:<14}{auprc(y, raw):>11.3f}{auprc(y, cal):>11.3f}"
          f"{auroc(y, raw):>8.3f}{ece(y, raw):>9.4f}{ece(y, cal):>9.4f}   "
          + "".join(f"{per_type[t]:>9.3f}" for t in types))
    if has is not None:
        print(f"    {'presence-only':<14}{presence_only_auprc(y, has):>11.3f}"
              f"{'':>11}{auroc(y, has):>8.3f}")
    print(f"    {'(random)':<14}{y.mean():>11.3f}{y.mean():>11.3f}{0.5:>8.3f}")


def stay_bootstrap_auroc_ci(y, scores, stays, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(stays)
    idx = {s: np.flatnonzero(stays == s) for s in uniq}
    out = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([idx[s] for s in pick])
        if 0 < y[ii].sum() < len(ii):
            out.append(auroc(y[ii], scores[ii]))
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rate", action="store_true",
                    help="drop HR-carrying features (redundancy check vs vitals)")
    args = ap.parse_args()
    drop = RATE_FEATURES if args.no_rate else ()
    tag = "ecg_meas (no-rate)" if args.no_rate else "ecg_meas"

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    agent = EcgMeasAgent(drop_features=drop)
    t0 = time.time()
    agent.fit(train, val)
    print(f"[train] {time.time() - t0:.0f}s")

    if not args.no_rate:                     # only the full variant is cached
        for name, df in (("train", train), ("val", val), ("test", test)):
            out = agent.emit_and_cache(df, EMB_DIR / f"{agent.name}_{name}.npz")
            print(f"[cache] {name}: {out.latent.shape} latents")

    test_sorted = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    out = agent.emit(test)
    assert (out.keys["stay_id"].to_numpy() == test_sorted["stay_id"].to_numpy()).all()
    y = test_sorted[LABEL].to_numpy(dtype=float)
    raw, cal = agent.raw_risk(test), out.risk
    has = agent.has_ecg(test).astype(float)
    types = list(EVENT_TYPES)

    print(f"\n{'='*100}")
    print(f"ECG AGENT ({tag}) -- machine-read measurements + diagnostic statements, no waveforms")
    print(f"  test: {len(y):,} patient-hours, {test_sorted.stay_id.nunique():,} stays, "
          f"prevalence {100*y.mean():.2f}%")
    print(f"  ECG measured (<=72h): {int(has.sum()):,} hours ({100*has.mean():.1f}%), "
          f"prevalence there {100*y[has>0].mean():.2f}% vs {100*y[has==0].mean():.2f}% without")
    print("="*100)

    report("FULL COHORT (inflated by ECG-presence being prognostic):",
           y, raw, cal, has, test_sorted, types, label=tag[:14])
    for name, v in OTHERS.items():
        print(f"    {'(' + name + ')':<14}{v[0]:>11.3f}{v[1]:>11.3f}{v[2]:>8.3f}"
              f"{v[3]:>9.4f}{v[4]:>9.4f}   " + "".join(f"{x:>9.3f}" for x in v[5:]))

    cov = has > 0
    report("ECG-MEASURED SUBSET  <-- the verdict on the modality:",
           y[cov], raw[cov], cal[cov], None,
           test_sorted[cov].reset_index(drop=True), types, label=tag[:14])
    death = test_sorted.loc[cov, f"{LABEL}_death"].to_numpy(dtype=float)
    print(f"    NOTE death column above vs orchestrator's full-cohort death AUPRC 0.073 "
          f"(known deficit; subset death prevalence {100*death.mean():.2f}%)")

    # ---- presence-leakage diagnostics ----------------------------------
    print(f"\n  Presence-leakage checks:")
    n_uniq = len(np.unique(np.round(raw[~cov], 6)))
    print(f"    distinct risks on no-ECG hours : {n_uniq} "
          f"({'constant, as designed' if n_uniq == 1 else 'NOT constant -- investigate'})")
    floor = presence_only_auprc(y, has)
    print(f"    presence-only AUPRC (floor)    : {floor:.4f}")
    print(f"    agent full-cohort AUPRC        : {auprc(y, raw):.4f} "
          f"(gain {auprc(y, raw) - floor:+.4f})")

    # ---- GATE 1: covered-hours VAL AUROC -------------------------------
    val_sorted = val.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    yv = val_sorted[LABEL].to_numpy(dtype=float)
    rawv = agent.raw_risk(val)
    hasv = agent.has_ecg(val)
    stays = val_sorted["stay_id"].to_numpy()
    a = auroc(yv[hasv], rawv[hasv])
    lo, hi = stay_bootstrap_auroc_ci(yv[hasv], rawv[hasv], stays[hasv])
    passed = a >= 0.55 and lo > 0.5
    print(f"\n  GATE 1 -- covered-hours VAL AUROC: {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}] "
          f"-> {'PASS (>=0.55, CI clear of 0.5)' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
