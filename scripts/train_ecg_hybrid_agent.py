"""Train the H1 gated-residual ECG omni-agent and report it.

    python scripts/train_ecg_hybrid_agent.py

Reports the standard tables plus H1's specific questions:
- the learned **gate** (waveform residual weight; started at 0 -- how far did
  the waveform earn its way in?),
- head-to-head vs the cached V4 (``ecg_meas``) risks on identical covered rows
  ("H1 - V4" = Δmeas + waveform's incremental value),
- Gate 1 (covered-hours val AUROC, bar = V4's 0.563).
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

from agents.base import AgentOutput  # noqa: E402
from agents.ecg_agent import presence_only_auprc  # noqa: E402
from agents.ecg_hybrid_agent import EcgHybridAgent  # noqa: E402
from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, auroc, ece  # noqa: E402
from mimic_cohort import EVENT_TYPES  # noqa: E402

LABEL = "label_24h"
EMB_DIR = REPO / "data" / "cache" / "embeddings"


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
    argparse.ArgumentParser().parse_args()
    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    agent = EcgHybridAgent()
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
    hasw = agent.has_wave(test)
    types = list(EVENT_TYPES)

    print(f"\n{'='*100}")
    print("ECG AGENT (ecg_hybrid, H1) -- machine read + Δmeas + gated ECGFounder residual + age")
    print(f"  test: {len(y):,} hours | measured {int(has.sum()):,} ({100*has.mean():.1f}%) | "
          f"waveform also on disk {int(hasw.sum()):,} ({100*hasw.mean():.1f}%)")
    print(f"  learned gate: {float(agent.net.gate):+.4f}  (0 = waveform ignored)")
    print("="*100)

    cov = has > 0
    per_type = {t: auprc(test_sorted.loc[cov, f"{LABEL}_{t}"].to_numpy(dtype=float),
                         raw[cov]) for t in types}
    print(f"\n  ECG-MEASURED SUBSET (42% of hours -- the verdict):")
    print(f"    AUPRC raw {auprc(y[cov], raw[cov]):.3f} | cal {auprc(y[cov], cal[cov]):.3f} "
          f"| AUROC {auroc(y[cov], raw[cov]):.3f} | ECE cal {ece(y[cov], cal[cov]):.4f}")
    print("    per-event AUPRC: " + " ".join(f"{t}={per_type[t]:.3f}" for t in types))
    n_uniq = len(np.unique(np.round(raw[~cov], 6)))
    print(f"    distinct risks on no-ECG hours: {n_uniq} "
          f"({'constant, as designed' if n_uniq == 1 else 'NOT constant -- investigate'})")
    print(f"    presence-only floor {presence_only_auprc(y, has):.4f} | "
          f"full-cohort AUPRC {auprc(y, raw):.4f}")

    # ---- H1 vs V4 on identical rows -------------------------------------
    v4_path = EMB_DIR / "ecg_meas_test.npz"
    if v4_path.exists():
        v4 = AgentOutput.load(v4_path)
        assert (v4.keys["stay_id"].to_numpy() == test_sorted["stay_id"].to_numpy()).all()
        v4r = v4.risk.astype(float)
        stays = test_sorted["stay_id"].to_numpy()
        print(f"\n  H1 vs V4 (ecg_meas) on identical rows:")
        for label, mask in (("measured rows", cov),
                            ("measured+waveform rows", hasw)):
            a1, a4 = auroc(y[mask], raw[mask]), auroc(y[mask], v4r[mask])
            print(f"    {label:<24} n={int(mask.sum()):>7,}  "
                  f"H1 AUROC {a1:.3f} vs V4 {a4:.3f}  (Δ {a1-a4:+.3f})")

    # ---- GATE 1 ---------------------------------------------------------
    val_sorted = val.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    yv = val_sorted[LABEL].to_numpy(dtype=float)
    rawv = agent.raw_risk(val)
    hv = agent.has_ecg(val)
    sv = val_sorted["stay_id"].to_numpy()
    a = auroc(yv[hv], rawv[hv])
    lo, hi = stay_bootstrap_auroc_ci(yv[hv], rawv[hv], sv[hv])
    print(f"\n  GATE 1 -- covered-hours VAL AUROC: {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"    bar: V4's 0.563; CI must clear 0.5")


if __name__ == "__main__":
    main()
