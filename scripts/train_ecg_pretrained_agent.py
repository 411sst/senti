"""Train the label head over a frozen pretrained ECG encoder (V5) and report.

    python scripts/train_ecg_pretrained_agent.py --backbone resnet   # ecg_resnet_pre
    python scripts/train_ecg_pretrained_agent.py --backbone mamba    # ecg_mamba

Same reporting contract as the other ECG training scripts. The coverage here is
the DOWNLOADED-studies subset (~7.6% of test hours at the current 5k-stay
fetch), not the 42% the measurements agent sees -- variant-vs-variant fairness
lives in scripts/ecg_coverage_matched.py, which pairs rows on the intersection.

GATE 4 context printed at the end: covered-hours val AUROC vs V2's 0.483 (the
from-scratch failure) and V1's 0.607 (frozen ECG-FM reference).
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
from agents.ecg_pretrained_agents import (  # noqa: E402
    EcgFounderAgent,
    EcgMambaAgent,
    EcgResnetPreAgent,
)
from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, auroc, ece  # noqa: E402
from mimic_cohort import EVENT_TYPES  # noqa: E402

LABEL = "label_24h"
EMB_DIR = REPO / "data" / "cache" / "embeddings"
AGENTS = {"resnet": EcgResnetPreAgent, "mamba": EcgMambaAgent,
          "ecgfounder": EcgFounderAgent}


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
    ap.add_argument("--backbone", choices=list(AGENTS), required=True)
    args = ap.parse_args()

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    agent = AGENTS[args.backbone]()
    t0 = time.time()
    agent.fit(train, val)
    print(f"[train] {time.time() - t0:.0f}s")

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
    print(f"ECG AGENT ({agent.name}) -- frozen {args.backbone} encoder pretrained "
          f"on machine-measurement targets")
    print(f"  test: {len(y):,} patient-hours | ECG on disk (<=72h): "
          f"{int(has.sum()):,} hours ({100*has.mean():.1f}%)")
    print("="*100)

    cov = has > 0
    per_type = {t: auprc(test_sorted.loc[cov, f'{LABEL}_{t}'].to_numpy(dtype=float),
                         raw[cov]) for t in types}
    print(f"\n  ECG-COVERED SUBSET (the verdict):")
    print(f"    AUPRC raw {auprc(y[cov], raw[cov]):.3f} | cal "
          f"{auprc(y[cov], cal[cov]):.3f} | AUROC {auroc(y[cov], raw[cov]):.3f} | "
          f"ECE cal {ece(y[cov], cal[cov]):.4f} | prevalence {y[cov].mean():.4f}")
    print("    per-event AUPRC: "
          + " ".join(f"{t}={per_type[t]:.3f}" for t in types))
    n_uniq = len(np.unique(np.round(raw[~cov], 6)))
    print(f"    distinct risks on no-ECG hours: {n_uniq} "
          f"({'constant, as designed' if n_uniq == 1 else 'NOT constant'})")
    print(f"    presence-only floor {presence_only_auprc(y, has):.4f} | "
          f"full-cohort AUPRC {auprc(y, raw):.4f}")

    # ---- GATE 4 context: covered-hours VAL AUROC ------------------------
    val_sorted = val.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    yv = val_sorted[LABEL].to_numpy(dtype=float)
    rawv = agent.raw_risk(val)
    hasv = agent.has_ecg(val)
    stays = val_sorted["stay_id"].to_numpy()
    a = auroc(yv[hasv], rawv[hasv])
    lo, hi = stay_bootstrap_auroc_ci(yv[hasv], rawv[hasv], stays[hasv])
    print(f"\n  GATE 4 -- covered-hours VAL AUROC: {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"    references: V2 from-scratch ResNet 0.483 (fail) | "
          f"V1 frozen ECG-FM 0.607 | chance 0.5")


if __name__ == "__main__":
    main()
