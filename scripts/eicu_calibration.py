"""Calibration transfer: what breaks, and what a new site must pay to repair it.

    python scripts/eicu_calibration.py

T22 found that the flagship's *ranking* survives the move to 208 unseen
hospitals (AUROC 0.801 -> 0.764) while its *calibration* does not
(ECE 0.0020 -> 0.0378). That split matters operationally: alarm thresholds are
quantiles of a calibrated probability, so a site inheriting MIMIC's calibrators
inherits the wrong alarm volume even though the model still ranks its patients
correctly.

This asks three questions the T22 result leaves open:

1. **What shape is the failure?** A uniform shift (every probability too high
   or too low by a constant factor) is repairable with no labels at all. A
   distortion of the curve's shape is not.
2. **Does a label-free prior correction fix it?** eICU's prevalence is 3.38%
   against the 4.74% the calibrators were fitted at. Rescaling the odds by the
   prevalence ratio needs only the local base rate -- a number a site knows
   without labelling anything.
3. **If labels are needed, how many?** Re-fitting isotonic on N local patients
   and testing on held-out ones turns "re-freeze thresholds locally" from an
   instruction into a number a deployment can plan around.

Every map applied here is monotone, so AUPRC and AUROC are unchanged by
construction -- asserted, not assumed. This is a calibration experiment only;
it cannot flatter the ranking results.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, auroc, ece, ece_adaptive  # noqa: E402

LABEL = "label_24h"
CACHE = REPO / "data" / "cache"
SEED = 42


def reliability(y, p, n_bins=10):
    """Equal-MASS reliability table: predicted vs observed, per decile."""
    order = np.argsort(p, kind="stable")
    rows = []
    for k, idx in enumerate(np.array_split(order, n_bins)):
        if not len(idx):
            continue
        rows.append((k + 1, len(idx), float(p[idx].mean()), float(y[idx].mean())))
    return pd.DataFrame(rows, columns=["decile", "n", "predicted", "observed"])


def prior_shift(p, src_prev, dst_prev, eps=1e-9):
    """Rescale calibrated probabilities from a source to a target base rate.

    The standard prior correction: multiply the odds by the ratio of target to
    source prior odds. Needs no labels -- only the local prevalence, which a
    site can estimate from its own event counts without touching the model.
    Monotone in p, so ranking is untouched.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    odds = p / (1 - p)
    ratio = (dst_prev / (1 - dst_prev)) / (src_prev / (1 - src_prev))
    shifted = odds * ratio
    return shifted / (1 + shifted)


def main():
    cohort = pd.read_parquet(CACHE / "eicu_cohort_hourly.parquet")
    o = cohort.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    z = np.load(CACHE / "eicu_twohead_all7.npz")
    assert (z["stay_id"] == o.stay_id.to_numpy()).all(), "score cache misaligned"
    y = o[LABEL].to_numpy(float)
    p = z["risk"].astype(float)
    pid = o["patient_id"].to_numpy()

    splits = load_splits()
    src_prev = float(splits["val"][LABEL].mean())   # what the calibrators saw
    dst_prev = float(y.mean())
    print(f"[data] eICU {len(y):,} hours, {len(np.unique(pid)):,} patients")
    print(f"[prev] MIMIC val (calibrator source) {100*src_prev:.2f}%  ->  "
          f"eICU {100*dst_prev:.2f}%")

    base_rank = (auprc(y, p), auroc(y, p))
    print(f"\n=== 1. The frozen flagship on eICU ===")
    print(f"  ECE (equal-width)  {ece(y, p):.4f}")
    print(f"  ECE (equal-mass)   {ece_adaptive(y, p):.4f}   <- the honest one at 3.4% prevalence")
    print(f"  AUPRC {base_rank[0]:.4f}  AUROC {base_rank[1]:.4f}")

    print(f"\n=== 2. What shape is the failure? (equal-mass deciles) ===")
    rel = reliability(y, p)
    rel["ratio"] = rel.observed / rel.predicted.clip(lower=1e-9)
    print(rel.to_string(index=False,
                        float_format=lambda v: f"{v:.4f}"))
    top = rel.iloc[-1]
    print(f"  top decile: predicted {top.predicted:.3f} vs observed {top.observed:.3f}")

    print(f"\n=== 3. Label-free prior correction ===")
    p_shift = prior_shift(p, src_prev, dst_prev)
    print(f"  ECE (equal-mass)   {ece_adaptive(y, p):.4f}  ->  {ece_adaptive(y, p_shift):.4f}")
    print(f"  AUPRC {auprc(y, p_shift):.4f}  AUROC {auroc(y, p_shift):.4f}  "
          f"(monotone: ranking must be unchanged)")
    assert abs(auroc(y, p_shift) - base_rank[1]) < 1e-9, "prior shift moved the ranking"

    print(f"\n=== 4. How many local patients buy back calibration? ===")
    rng = np.random.default_rng(SEED)
    patients = np.unique(pid)
    rng.shuffle(patients)
    n_eval = len(patients) // 4
    eval_set = set(patients[:n_eval].tolist())
    pool = patients[n_eval:]
    m_eval = np.fromiter((q in eval_set for q in pid), bool, len(pid))
    y_ev, p_ev = y[m_eval], p[m_eval]
    print(f"  held-out: {int(m_eval.sum()):,} hours from {n_eval:,} patients "
          f"(never used for fitting)")
    print(f"  {'local patients':>15} {'local hours':>12} {'ECE':>9} {'vs frozen':>11}")
    frozen = ece_adaptive(y_ev, p_ev)
    print(f"  {'0 (frozen)':>15} {0:>12,} {frozen:>9.4f} {'-':>11}")
    for n_pat in (25, 50, 100, 250, 500, 1000, 2500, 5000, 10000):
        if n_pat > len(pool):
            break
        eces, hours = [], []
        for rep in range(5):
            r = np.random.default_rng(SEED + rep)
            pick = set(r.choice(pool, n_pat, replace=False).tolist())
            m_fit = np.fromiter((q in pick for q in pid), bool, len(pid))
            if y[m_fit].sum() < 2:
                continue
            iso = IsotonicRegression(out_of_bounds="clip").fit(p[m_fit], y[m_fit])
            eces.append(ece_adaptive(y_ev, iso.transform(p_ev)))
            hours.append(int(m_fit.sum()))
        if not eces:
            continue
        med = float(np.median(eces))
        print(f"  {n_pat:>15,} {int(np.median(hours)):>12,} {med:>9.4f} "
              f"{(frozen-med)/frozen*100:>10.0f}%")

    print(f"\n=== 5. Does the failure vary by site? ===")
    tmp = pd.DataFrame({"h": o.hospitalid.to_numpy(), "y": y, "p": p})
    rows = []
    for h, g in tmp.groupby("h"):
        if len(g) >= 20000 and g.y.sum() >= 50:
            rows.append(ece_adaptive(g.y.to_numpy(), g.p.to_numpy()))
    d = np.array(rows)
    print(f"  {len(d)} hospitals | ECE p10 {np.percentile(d,10):.4f} "
          f"median {np.median(d):.4f} p90 {np.percentile(d,90):.4f}")


if __name__ == "__main__":
    main()
