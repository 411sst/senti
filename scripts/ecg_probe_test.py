"""Is the ECG head data-starved, or is the modality weak? A cheap discriminator.

    python scripts/ecg_probe_test.py

The MLP head peaked at val epoch 3 then declined, which is the signature of
overfitting on ~2,060 positive covered hours. Three diagnostics, all on the
ECG-available subset and all cheap:

1. **Linear probe** on the frozen 768-dim embeddings, across a regularisation
   sweep. If a heavily-regularised linear model matches or beats the MLP, the
   MLP was burning capacity on noise.
2. **Train-vs-test gap.** A large gap means overfitting (fixable with data); a
   small gap with low test scores means the embeddings simply do not separate
   these labels (not fixable with data).
3. **Learning curve** over 25/50/100% of the available training stays. This is
   the decisive one for the download decision: if test performance is still
   climbing at 100%, the full 44k-study set would likely help; if it has
   plateaued, another 20x of data buys nothing.

Subsampling is by STAY so trajectories stay intact and no patient's hours are
split across the curve.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from agents.ecg_agent import assemble_ecg_features  # noqa: E402
from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, auroc  # noqa: E402

LABEL = "label_24h"
EMB = REPO / "data" / "cache" / "ecg_embeddings.npz"
INDEX = REPO / "data" / "cache" / "ecg_index.parquet"

# Reference: the MLP head reported on the covered test subset.
MLP_TEST = {"auprc": 0.057, "auroc": 0.590}


def covered(split_df, index, emb, ids):
    """Covered rows only: (X, y, stay_id) for hours with an ECG within 72h."""
    X, has, age, keys = assemble_ecg_features(split_df, index, emb, ids)
    ordered = split_df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    y = ordered[LABEL].to_numpy(dtype=float)
    m = has > 0
    return X[m], y[m], keys["stay_id"].to_numpy()[m]


def fit_probe(Xtr, ytr, C, mean, std):
    model = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    model.fit((Xtr - mean) / std, ytr)
    return model


def score(model, X, y, mean, std):
    p = model.predict_proba((X - mean) / std)[:, 1]
    return auprc(y, p), auroc(y, p)


def main():
    z = np.load(EMB)
    emb, ids = z["embeddings"], z["study_ids"]
    index = pd.read_parquet(INDEX)
    sp = load_splits()

    Xtr, ytr, str_ = covered(sp["train"], index, emb, ids)
    Xva, yva, _ = covered(sp["val"], index, emb, ids)
    Xte, yte, _ = covered(sp["test"], index, emb, ids)
    mean, std = Xtr.mean(axis=0), np.where(Xtr.std(axis=0) > 1e-6, Xtr.std(axis=0), 1.0)

    print(f"covered hours -- train {len(ytr):,} ({int(ytr.sum()):,} pos, "
          f"{str_.__len__() and len(np.unique(str_)):,} stays) | "
          f"val {len(yva):,} ({int(yva.sum()):,} pos) | "
          f"test {len(yte):,} ({int(yte.sum()):,} pos)")
    print(f"test prevalence (the AUPRC floor): {yte.mean():.4f}\n")

    # ---- 1 + 2: regularisation sweep with the train/test gap ------------
    print("=" * 86)
    print("LINEAR PROBE on frozen embeddings -- regularisation sweep")
    print("=" * 86)
    hdr = (f"  {'C (1/L2)':>10}{'train AUPRC':>13}{'train AUROC':>13}"
           f"{'val AUROC':>11}{'test AUPRC':>12}{'test AUROC':>12}{'gap':>8}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    best = None
    for C in (0.0001, 0.001, 0.01, 0.1, 1.0):
        m = fit_probe(Xtr, ytr, C, mean, std)
        tr_pr, tr_roc = score(m, Xtr, ytr, mean, std)
        _, va_roc = score(m, Xva, yva, mean, std)
        te_pr, te_roc = score(m, Xte, yte, mean, std)
        print(f"  {C:>10g}{tr_pr:>13.3f}{tr_roc:>13.3f}{va_roc:>11.3f}"
              f"{te_pr:>12.3f}{te_roc:>12.3f}{tr_roc - te_roc:>8.3f}")
        if best is None or va_roc > best[0]:
            best = (va_roc, C, te_pr, te_roc)
    print(f"\n  {'MLP head (current)':<24}{'':>22}{'':>11}"
          f"{MLP_TEST['auprc']:>12.3f}{MLP_TEST['auroc']:>12.3f}")
    print(f"  best probe by VAL AUROC: C={best[1]:g} -> "
          f"test AUPRC {best[2]:.3f}, AUROC {best[3]:.3f}")

    # ---- 3: learning curve over training stays --------------------------
    print("\n" + "=" * 86)
    print("LEARNING CURVE -- would 20x more ECGs help? (subsample training STAYS)")
    print("=" * 86)
    C = best[1]
    stays = np.unique(str_)
    rng = np.random.default_rng(42)
    order = rng.permutation(stays)
    print(f"  {'frac':>6}{'stays':>8}{'hours':>9}{'pos':>7}"
          f"{'test AUPRC':>12}{'test AUROC':>12}")
    print("  " + "-" * 54)
    curve = []
    for frac in (0.25, 0.50, 0.75, 1.0):
        keep = set(order[:max(1, int(len(order) * frac))])
        m_ = np.isin(str_, list(keep))
        if ytr[m_].sum() < 5:
            continue
        mdl = fit_probe(Xtr[m_], ytr[m_], C, mean, std)
        te_pr, te_roc = score(mdl, Xte, yte, mean, std)
        curve.append((frac, te_pr, te_roc))
        print(f"  {frac:>6.0%}{len(keep):>8,}{int(m_.sum()):>9,}"
              f"{int(ytr[m_].sum()):>7,}{te_pr:>12.3f}{te_roc:>12.3f}")

    if len(curve) >= 2:
        d_roc = curve[-1][2] - curve[-2][2]
        d_pr = curve[-1][1] - curve[-2][1]
        print(f"\n  slope over the last step: AUROC {d_roc:+.3f}, AUPRC {d_pr:+.3f}")
        print("  -> " + ("still climbing: more ECGs would likely help"
                         if d_roc > 0.01 else
                         "plateaued: another 20x of data is unlikely to move it"))


if __name__ == "__main__":
    main()
