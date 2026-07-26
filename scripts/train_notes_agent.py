"""Train the notes agent and report it against the other agents.

    python scripts/train_notes_agent.py

Reports the standard table twice: on the full cohort and, separately, on the
**note-available subset**. The subset number is the verdict on the modality --
the full-cohort figure is inflated by the fact that having imaging ordered is
itself prognostic (5.09% vs 4.18% event rate), which is a property of clinical
workflow rather than of the radiology text.

Also reports the presence-only baseline: what an "is there a report?" indicator
scores by itself. The agent has to beat that to have learned anything.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from agents.notes_agent import NotesAgent, presence_only_auprc  # noqa: E402
from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, auroc, ece  # noqa: E402
from mimic_cohort import EVENT_TYPES  # noqa: E402

LABEL = "label_24h"
EMB_DIR = REPO / "data" / "cache" / "embeddings"

OTHERS = {                       # from scripts/results_table.py, same test split
    "labs":       (0.171, 0.162, 0.728, 0.0035, 0.0012, 0.128, 0.100, 0.035),
    "vitals-TCN": (0.159, 0.155, 0.750, 0.0524, 0.0021, 0.084, 0.054, 0.139),
    "NEWS2":      (0.108, 0.107, 0.657, 0.0845, 0.0014, 0.059, 0.041, 0.062),
    "GBM":        (0.170, 0.166, 0.745, 0.0036, 0.0016, 0.081, 0.049, 0.187),
    "ECG":        (0.048, 0.048, 0.503, 0.0043, 0.0002, 0.033, 0.021, 0.007),
}


def report(title, y, raw, cal, has, evaluated, types):
    print(f"\n  {title}")
    hdr = (f"    {'model':<14}{'AUPRC raw':>11}{'AUPRC cal':>11}{'AUROC':>8}"
           f"{'ECE raw':>9}{'ECE cal':>9}   " + "".join(f"{t:>9}" for t in types))
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    per_type = {}
    for t in types:
        yt = evaluated[f"{LABEL}_{t}"].to_numpy(dtype=float)
        per_type[t] = auprc(yt, raw) if yt.sum() > 0 else float("nan")
    print(f"    {'notes':<14}{auprc(y, raw):>11.3f}{auprc(y, cal):>11.3f}"
          f"{auroc(y, raw):>8.3f}{ece(y, raw):>9.4f}{ece(y, cal):>9.4f}   "
          + "".join(f"{per_type[t]:>9.3f}" for t in types))
    if has is not None:
        print(f"    {'presence-only':<14}{presence_only_auprc(y, has):>11.3f}"
              f"{'':>11}{auroc(y, has):>8.3f}")
    print(f"    {'(random)':<14}{y.mean():>11.3f}{y.mean():>11.3f}{0.5:>8.3f}")


def main():
    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    agent = NotesAgent()
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
    has = agent.has_note(test).astype(float)
    types = list(EVENT_TYPES)

    print(f"\n{'='*100}")
    print(f"NOTES AGENT -- frozen Bio_ClinicalBERT (768d) + small head")
    print(f"  test: {len(y):,} patient-hours, {test_sorted.stay_id.nunique():,} stays, "
          f"prevalence {100*y.mean():.2f}%")
    print(f"  note available (<=72h): {int(has.sum()):,} hours ({100*has.mean():.1f}%), "
          f"prevalence there {100*y[has>0].mean():.2f}% "
          f"vs {100*y[has==0].mean():.2f}% without")
    print("="*100)

    report("FULL COHORT (inflated by note-presence being prognostic):",
           y, raw, cal, has, test_sorted, types)
    for name, v in OTHERS.items():
        print(f"    {'(' + name + ')':<14}{v[0]:>11.3f}{v[1]:>11.3f}{v[2]:>8.3f}"
              f"{v[3]:>9.4f}{v[4]:>9.4f}   " + "".join(f"{x:>9.3f}" for x in v[5:]))

    cov = has > 0
    report("NOTE-AVAILABLE SUBSET  <-- the verdict on the modality:",
           y[cov], raw[cov], cal[cov], None, test_sorted[cov].reset_index(drop=True), types)

    # ---- presence-leakage diagnostics ----------------------------------
    print(f"\n  Presence-leakage checks:")
    n_uniq = len(np.unique(np.round(raw[~cov], 6)))
    print(f"    distinct risks on no-note hours : {n_uniq} "
          f"({'constant, as designed' if n_uniq == 1 else 'NOT constant -- investigate'})")
    print(f"    mean risk  note / no-note        : {raw[cov].mean():.4f} / {raw[~cov].mean():.4f}")
    print(f"    true rate  note / no-note        : {y[cov].mean():.4f} / {y[~cov].mean():.4f}")
    print(f"    presence-only AUPRC (floor)    : {presence_only_auprc(y, has):.4f}")
    print(f"    agent full-cohort AUPRC        : {auprc(y, raw):.4f}")
    gain = auprc(y, raw) - presence_only_auprc(y, has)
    print(f"    -> gain over presence alone    : {gain:+.4f} "
          f"({'real waveform signal' if gain > 0.005 else 'NOT beating presence'})")


if __name__ == "__main__":
    main()
