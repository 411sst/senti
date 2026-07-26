"""Coverage-matched comparison: all agents on the ECG-available hours only.

    python scripts/ecg_coverage_matched.py

The ECG agent scores AUROC 0.608 on the 8,334 test hours where a recent ECG
exists, against 0.728/0.750 for labs/vitals on the *full* cohort. That is not a
fair comparison -- different patient-hours. This script scores every agent on
the identical covered rows, which separates two very different conclusions:

* if labs/vitals also drop sharply on covered hours, the subset is simply harder
  and ECG's number is less damning than it looks;
* if labs/vitals hold near their full-cohort level while ECG sits at 0.61, the
  modality is genuinely weak for 24h deterioration.

Uses each agent's cached calibrated risk. Calibration is monotone, so AUROC is
unaffected by it and AUPRC only via ties -- the ranking comparison is clean.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agents.base import AgentOutput  # noqa: E402
from agents.ecg_agent import assemble_ecg_features  # noqa: E402
from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, auroc  # noqa: E402

LABEL = "label_24h"
EMB_DIR = REPO / "data" / "cache" / "embeddings"
AGENTS = ("ecg", "labs", "vitals_tcn", "news2", "orchestrator")


def bootstrap_ci(y, scores, stays, reps=1000, seed=0):
    """Stay-level bootstrap CI for AUROC (hours within a stay are correlated)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(stays)
    idx = {s: np.flatnonzero(stays == s) for s in uniq}
    out = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([idx[s] for s in pick])
        if y[ii].sum() < 3 or y[ii].sum() == len(ii):
            continue
        out.append(auroc(y[ii], scores[ii]))
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def main():
    sp = load_splits()
    test = sp["test"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    y = test[LABEL].to_numpy(dtype=float)
    stays = test["stay_id"].to_numpy()

    emb = np.load(REPO / "data" / "cache" / "ecg_embeddings.npz")
    index = pd.read_parquet(REPO / "data" / "cache" / "ecg_index.parquet")
    _, has, _, keys = assemble_ecg_features(test, index, emb["embeddings"],
                                            emb["study_ids"])
    assert (keys["stay_id"].to_numpy() == stays).all(), "ECG features misaligned"
    covered = has > 0

    risks = {}
    for name in AGENTS:
        path = EMB_DIR / f"{name}_test.npz"
        if not path.exists():
            continue
        out = AgentOutput.load(path)
        assert (out.keys["stay_id"].to_numpy() == stays).all(), f"{name} misaligned"
        risks[name] = out.risk.astype(float)
    risks["NEWS2 score"] = test["news2"].to_numpy(dtype=float)

    print(f"test split: {len(y):,} hours, prevalence {100*y.mean():.2f}%")
    print(f"ECG-covered: {int(covered.sum()):,} hours ({100*covered.mean():.1f}%), "
          f"{int(y[covered].sum()):,} positives, prevalence {100*y[covered].mean():.2f}%")
    print(f"uncovered  : {int((~covered).sum()):,} hours, "
          f"prevalence {100*y[~covered].mean():.2f}%")

    for title, mask in (("ECG-COVERED HOURS  (identical rows for every agent)", covered),
                        ("UNCOVERED HOURS    (for contrast)", ~covered),
                        ("FULL TEST SPLIT    (reference)", np.ones(len(y), bool))):
        ym, sm = y[mask], stays[mask]
        print(f"\n{'=' * 82}\n{title}\n"
              f"  n={int(mask.sum()):,}  positives={int(ym.sum()):,}  "
              f"prevalence={ym.mean():.4f}  (AUPRC floor)\n{'=' * 82}")
        hdr = f"  {'model':<16}{'AUPRC':>9}{'AUPRC/floor':>13}{'AUROC':>9}{'AUROC 95% CI':>20}"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for name, s in risks.items():
            sub = s[mask]
            pr, roc = auprc(ym, sub), auroc(ym, sub)
            lo, hi = bootstrap_ci(ym, sub, sm)
            print(f"  {name:<16}{pr:>9.3f}{pr/ym.mean():>12.2f}x{roc:>9.3f}"
                  f"      [{lo:.3f}, {hi:.3f}]")

    # Head-to-head: does ECG add anything the others do not already have?
    print(f"\n{'=' * 82}\nPAIRED on covered hours: ECG vs each agent (stay bootstrap)\n{'=' * 82}")
    rng = np.random.default_rng(1)
    yc, sc = y[covered], stays[covered]
    uniq = np.unique(sc)
    idx = {s: np.flatnonzero(sc == s) for s in uniq}
    for name in ("labs", "vitals_tcn", "orchestrator"):
        if name not in risks:
            continue
        a, b = risks["ecg"][covered], risks[name][covered]
        d = []
        for _ in range(1000):
            pick = rng.choice(uniq, len(uniq), replace=True)
            ii = np.concatenate([idx[s] for s in pick])
            if yc[ii].sum() < 3:
                continue
            d.append(auroc(yc[ii], a[ii]) - auroc(yc[ii], b[ii]))
        d = np.array(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  ECG - {name:<14} AUROC {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"P(ECG better)={100*(d > 0).mean():.0f}%")


if __name__ == "__main__":
    main()
