"""Train the labs agent and report it in the standard agent table.

    python scripts/train_labs_agent.py [--epochs 12]

Trains on train, isotonic-calibrates on val (inherited from the base class),
caches latents, and reports on test alongside the other agents. The per-event-
type row is the interesting one: labs should carry mortality signal, which is
where the vitals-TCN was weakest.
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from agents.labs_agent import LabsAgent  # noqa: E402
from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, ece  # noqa: E402
from mimic_cohort import EVENT_TYPES  # noqa: E402

LABEL = "label_24h"
EMB_DIR = REPO / "data" / "cache" / "embeddings"

# Previously reported rows, for comparison (see the other agent scripts).
OTHERS = {
    "vitals-TCN": dict(raw=0.161, cal=0.156, ece_raw=0.0519, ece_cal=0.0023,
                       vent=0.083, vaso=0.054, death=0.132),
    "NEWS2 agent": dict(raw=0.108, cal=0.107, ece_raw=0.1623, ece_cal=0.0014,
                        vent=0.059, vaso=0.041, death=0.062),
    "GBM baseline": dict(raw=0.170, cal=0.170, ece_raw=0.0036, ece_cal=0.0016,
                         vent=0.081, vaso=0.049, death=0.187),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    agent = LabsAgent(max_epochs=args.epochs)
    t0 = time.time()
    agent.fit(train, val)
    print(f"[train] {time.time() - t0:.0f}s")

    for name, df in (("train", train), ("val", val), ("test", test)):
        out = agent.emit_and_cache(df, EMB_DIR / f"{agent.name}_{name}.npz")
        print(f"[cache] {name}: {out.latent.shape} latents")

    test_sorted = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    out = agent.emit(test)
    assert (out.keys["stay_id"].to_numpy() == test_sorted["stay_id"].to_numpy()).all()

    y = test_sorted[LABEL].to_numpy()
    risk_cal, risk_raw = out.risk, agent.raw_risk(test)
    cal = agent.calibration_summary

    print(f"\nLabs agent on test ({len(y):,} patient-hours, "
          f"{test_sorted['stay_id'].nunique():,} stays, "
          f"prevalence {100 * y.mean():.2f}%)")
    print(f"  latent dim {out.latent_dim}; isotonic calibration fit on val "
          f"({cal['split_rows']:,} rows, {cal['positives']:,} positive)\n")

    print(f"  {'model':<14}{'AUPRC raw':>11}{'AUPRC cal':>11}{'ECE raw':>10}{'ECE cal':>10}")
    print(f"  {'labs':<14}{auprc(y, risk_raw):>11.3f}{auprc(y, risk_cal):>11.3f}"
          f"{ece(y, risk_raw):>10.4f}{ece(y, risk_cal):>10.4f}")
    for name, v in OTHERS.items():
        print(f"  {'(' + name + ')':<14}{v['raw']:>11.3f}{v['cal']:>11.3f}"
              f"{v['ece_raw']:>10.4f}{v['ece_cal']:>10.4f}")
    print(f"  {'(random)':<14}{y.mean():>11.3f}{y.mean():>11.3f}")

    print("\n  AUPRC by event type  <- the complementarity check:")
    types = list(EVENT_TYPES)
    print("    " + f"{'model':<14}" + "".join(f"{t:>10}" for t in types))
    per_type = {t: auprc(test_sorted[f"{LABEL}_{t}"].to_numpy(), risk_raw) for t in types}
    print("    " + f"{'labs':<14}" + "".join(f"{per_type[t]:>10.3f}" for t in types))
    for name, v in OTHERS.items():
        print("    " + f"{'(' + name + ')':<14}"
              + "".join(f"{v[t]:>10.3f}" for t in types))


if __name__ == "__main__":
    main()
