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
from agents.ecg_resnet_encoder import (  # noqa: E402
    WaveformStore,
    assemble_ecg_study_ids,
)
from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, auroc  # noqa: E402

LABEL = "label_24h"
EMB_DIR = REPO / "data" / "cache" / "embeddings"
AGENTS = ("ecg", "ecg_resnet", "ecg_resnet_film", "ecg_meas", "ecg_meas_disk",
          "ecg_resnet_pre", "ecg_mamba", "ecgfounder", "labs", "vitals_tcn",
          "news2", "orchestrator")
ECG_VARIANTS = ("ecg", "ecg_resnet", "ecg_resnet_film", "ecg_meas",
                "ecg_meas_disk", "ecg_resnet_pre", "ecg_mamba", "ecgfounder")
MEAS_CACHE = REPO / "data" / "cache" / "ecg_meas_features.npz"


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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", choices=("disk", "measured", "both"),
                    default="both",
                    help="which rows count as ECG-covered: waveform on disk, "
                         "machine measurements available, or their intersection "
                         "(the fair variant-vs-variant set)")
    args = ap.parse_args()

    sp = load_splits()
    test = sp["test"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    y = test[LABEL].to_numpy(dtype=float)
    stays = test["stay_id"].to_numpy()

    # Covered = rows whose most-recent prior ECG (within staleness) satisfies
    # the chosen coverage criterion. Waveform agents (V2/V5) only see on-disk
    # studies; the measurements agent (V4) sees every measured study (~42% of
    # hours) -- comparing across those different row sets in one table would be
    # meaningless, so the default pairs everyone on the intersection.
    index = pd.read_parquet(REPO / "data" / "cache" / "ecg_index.parquet")
    study, _, keys = assemble_ecg_study_ids(test, index)
    assert (keys["stay_id"].to_numpy() == stays).all(), "ECG index misaligned"
    ok = set()
    if args.coverage in ("disk", "both"):
        ok_disk = WaveformStore(verbose=False).available_studies()
    if args.coverage in ("measured", "both"):
        ok_meas = {int(s) for s in np.load(MEAS_CACHE)["study_ids"]}
    ok = (ok_disk & ok_meas if args.coverage == "both"
          else ok_disk if args.coverage == "disk" else ok_meas)
    print(f"[coverage={args.coverage}] {len(ok):,} qualifying studies")
    covered = np.array([not np.isnan(s) and int(s) in ok for s in study])

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

    # Head-to-head on covered hours: each ECG variant vs the other agents, and
    # variant vs variant (V1 ECG-FM vs V2 ResNet vs V3 +FiLM) -- the core of the
    # experiment. Only pairs whose caches both exist are run.
    print(f"\n{'=' * 82}\nPAIRED on covered hours: AUROC difference (stay bootstrap)\n{'=' * 82}")
    rng = np.random.default_rng(1)
    yc, sc = y[covered], stays[covered]
    uniq = np.unique(sc)
    idx = {s: np.flatnonzero(sc == s) for s in uniq}

    def paired(a_name, b_name):
        a, b = risks[a_name][covered], risks[b_name][covered]
        d = []
        for _ in range(1000):
            pick = rng.choice(uniq, len(uniq), replace=True)
            ii = np.concatenate([idx[s] for s in pick])
            if yc[ii].sum() < 3:
                continue
            d.append(auroc(yc[ii], a[ii]) - auroc(yc[ii], b[ii]))
        d = np.array(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {a_name} - {b_name:<16} AUROC {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"P({a_name} better)={100*(d > 0).mean():.0f}%")

    present_variants = [v for v in ECG_VARIANTS if v in risks]
    for ev in present_variants:                       # variant vs the rest
        for name in ("labs", "vitals_tcn", "orchestrator"):
            if name in risks:
                paired(ev, name)
    for i, a_name in enumerate(present_variants):     # variant vs variant
        for b_name in present_variants[i + 1:]:
            paired(a_name, b_name)


if __name__ == "__main__":
    main()
