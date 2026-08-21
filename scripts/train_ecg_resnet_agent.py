"""Train the trainable-ResNet ECG agent (V2 / V3) and report it.

    python scripts/train_ecg_resnet_agent.py --variant resnet   # V2
    python scripts/train_ecg_resnet_agent.py --variant film     # V3 (+FiLM on age)

Same reporting contract as scripts/train_ecg_agent.py (the frozen ECG-FM V1):
the table is printed on the full cohort AND on the ECG-available subset -- the
subset figure is the honest verdict on the modality, the full-cohort one being
inflated by ECG-presence itself being mildly prognostic. Also prints the
presence-only floor the agent must beat, and asserts the no-ECG hours emit a
single constant risk (the presence-leakage guard).

Requires waveforms on disk (scripts/fetch_ecg_subset.py) and, for --variant film,
data/cache/patient_age.parquet (scripts/fetch_patient_age.py).
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

from agents.ecg_agent import presence_only_auprc  # noqa: E402  (shared floor)
from agents.ecg_resnet_agent import EcgResnetAgent, EcgResnetFilmAgent  # noqa: E402
from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, auroc, ece  # noqa: E402
from mimic_cohort import EVENT_TYPES  # noqa: E402

LABEL = "label_24h"
EMB_DIR = REPO / "data" / "cache" / "embeddings"

VARIANTS = {
    "resnet": (EcgResnetAgent, "1D-ResNet trained end-to-end on label_24h (V2)"),
    "film":   (EcgResnetFilmAgent, "1D-ResNet + FiLM(age), ECG-present pathway (V3)"),
}

OTHERS = {                       # full-cohort test numbers, same split (results.md)
    "labs":       (0.171, 0.162, 0.728, 0.0035, 0.0012, 0.128, 0.100, 0.035),
    "vitals-TCN": (0.159, 0.155, 0.750, 0.0524, 0.0021, 0.084, 0.054, 0.139),
    "NEWS2":      (0.108, 0.107, 0.657, 0.0845, 0.0014, 0.059, 0.041, 0.062),
    "GBM":        (0.170, 0.166, 0.745, 0.0036, 0.0016, 0.081, 0.049, 0.187),
    "ECG-FM(V1)": (None,),       # frozen baseline; benched at AUROC 0.607 (covered)
}


def report(title, y, raw, cal, has, evaluated, types, label="ECG-ResNet"):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="resnet")
    ap.add_argument("--max-epochs", type=int, default=12)
    args = ap.parse_args()
    cls, desc = VARIANTS[args.variant]

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    agent = cls(max_epochs=args.max_epochs)
    t0 = time.time()
    agent.fit(train, val)
    print(f"[train] {time.time() - t0:.0f}s")

    for name, df in (("train", train), ("val", val), ("test", test)):
        out = agent.emit_and_cache(df, EMB_DIR / f"{agent.name}_{name}.npz")
        print(f"[cache] {name}: {out.latent.shape} latents -> {agent.name}_{name}.npz")

    test_sorted = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    out = agent.emit(test)
    assert (out.keys["stay_id"].to_numpy() == test_sorted["stay_id"].to_numpy()).all()
    y = test_sorted[LABEL].to_numpy(dtype=float)
    raw, cal = agent.raw_risk(test), out.risk
    has = agent.has_ecg(test).astype(float)
    types = list(EVENT_TYPES)

    print(f"\n{'='*100}")
    print(f"ECG AGENT ({agent.name}) -- {desc}")
    print(f"  test: {len(y):,} patient-hours, {test_sorted.stay_id.nunique():,} stays, "
          f"prevalence {100*y.mean():.2f}%")
    print(f"  ECG available (<=72h, on disk): {int(has.sum()):,} hours ({100*has.mean():.1f}%), "
          f"prevalence there {100*y[has>0].mean():.2f}% vs {100*y[has==0].mean():.2f}% without")
    print("="*100)

    report("FULL COHORT (inflated by ECG-presence being prognostic):",
           y, raw, cal, has, test_sorted, types, label=agent.name)
    for name, v in OTHERS.items():
        if v[0] is None:
            print(f"    {'(' + name + ')':<14}{'benched (AUROC 0.607 on covered)':>33}")
            continue
        print(f"    {'(' + name + ')':<14}{v[0]:>11.3f}{v[1]:>11.3f}{v[2]:>8.3f}"
              f"{v[3]:>9.4f}{v[4]:>9.4f}   " + "".join(f"{x:>9.3f}" for x in v[5:]))

    cov = has > 0
    report("ECG-AVAILABLE SUBSET  <-- the verdict on the modality:",
           y[cov], raw[cov], cal[cov], None,
           test_sorted[cov].reset_index(drop=True), types, label=agent.name)

    # ---- presence-leakage diagnostics ----------------------------------
    print(f"\n  Presence-leakage checks:")
    n_uniq = len(np.unique(np.round(raw[~cov], 6)))
    print(f"    distinct risks on no-ECG hours : {n_uniq} "
          f"({'constant, as designed' if n_uniq == 1 else 'NOT constant -- investigate'})")
    print(f"    mean risk  ECG / no-ECG        : {raw[cov].mean():.4f} / {raw[~cov].mean():.4f}")
    print(f"    true rate  ECG / no-ECG        : {y[cov].mean():.4f} / {y[~cov].mean():.4f}")
    floor = presence_only_auprc(y, has)
    print(f"    presence-only AUPRC (floor)    : {floor:.4f}")
    print(f"    agent full-cohort AUPRC        : {auprc(y, raw):.4f}")
    subset_auprc = auprc(y[cov], raw[cov])
    print(f"    ECG-available-subset AUPRC     : {subset_auprc:.4f} "
          f"(prevalence there {y[cov].mean():.4f})")
    gain = subset_auprc - y[cov].mean()
    print(f"    -> subset gain over prevalence : {gain:+.4f} "
          f"({'real waveform signal' if gain > 0.005 else 'NOT beating prevalence'})")


if __name__ == "__main__":
    main()
