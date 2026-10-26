"""Train the vitals-TCN agent, calibrate it, cache its embeddings, report.

    python scripts/train_vitals_tcn.py [--epochs 8] [--channels 32]

Trains on the train split only, fits isotonic calibration on val (inherited from
``agents.base.Agent``), and reports on test in the same shape as the GBM
baseline table so the two are directly comparable. Latents for all three splits are cached to
``data/cache/embeddings/`` so downstream fusion never re-runs the encoder.
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from agents.vitals_tcn import VitalsTCNAgent  # noqa: E402
from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, ece  # noqa: E402
from mimic_cohort import EVENT_TYPES  # noqa: E402

LABEL = "label_24h"
EMB_DIR = REPO / "data" / "cache" / "embeddings"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    agent = VitalsTCNAgent(channels=args.channels, max_epochs=args.epochs,
                           batch_size=args.batch_size)
    t0 = time.time()
    agent.fit(train, val)
    print(f"[train] {time.time() - t0:.0f}s")

    # --- emit + cache for every split -----------------------------------
    outputs = {}
    for name, df in (("train", train), ("val", val), ("test", test)):
        t = time.time()
        out = agent.emit_and_cache(df, EMB_DIR / f"{agent.name}_{name}.npz")
        outputs[name] = out
        print(f"[cache] {name}: {out.latent.shape} latents -> "
              f"{EMB_DIR / f'{agent.name}_{name}.npz'} ({time.time() - t:.0f}s)")

    # --- report on test --------------------------------------------------
    # emit() returns rows ordered by (stay_id, hour); align the labels the same
    # way rather than assuming the frame order matches.
    test_sorted = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    keys = outputs["test"].keys
    assert (keys["stay_id"].to_numpy() == test_sorted["stay_id"].to_numpy()).all()
    assert (keys["hour"].to_numpy() == test_sorted["hour"].to_numpy()).all()

    y = test_sorted[LABEL].to_numpy()
    risk_cal = outputs["test"].risk
    risk_raw = agent.raw_risk(test)

    print(f"\nVitals-TCN agent on test ({len(y):,} patient-hours, "
          f"{test_sorted['stay_id'].nunique():,} stays, "
          f"prevalence {100 * y.mean():.2f}%)")
    cal = agent.calibration_summary
    print(f"  latent dim {outputs['test'].latent_dim}; isotonic calibration fit on "
          f"val ({cal['split_rows']:,} rows, {cal['positives']:,} positive)\n")
    # Report AUPRC on BOTH raw and calibrated scores. Isotonic is monotone but
    # it is a STEP function, so it merges nearby scores into ties -- that costs
    # a little ranking resolution even though nothing is reordered. Showing both
    # keeps the calibration/discrimination trade-off visible instead of hiding
    # it behind one number.
    print(f"  {'model':<12}{'AUPRC raw':>11}{'AUPRC cal':>11}{'ECE raw':>10}{'ECE cal':>10}")
    print(f"  {'vitals-TCN':<12}{auprc(y, risk_raw):>11.3f}{auprc(y, risk_cal):>11.3f}"
          f"{ece(y, risk_raw):>10.4f}{ece(y, risk_cal):>10.4f}")
    print(f"  {'(GBM)':<12}{0.170:>11.3f}{0.170:>11.3f}{0.0036:>10.4f}{0.0016:>10.4f}")
    print(f"  {'(NEWS2)':<12}{0.108:>11.3f}{0.108:>11.3f}{0.1623:>10.4f}{0.0014:>10.4f}")
    print(f"  {'(random)':<12}{y.mean():>11.3f}{y.mean():>11.3f}")

    print("\n  AUPRC by event type:")
    types = list(EVENT_TYPES)
    print("    " + f"{'model':<12}" + "".join(f"{t:>10}" for t in types))
    per_type = {t: auprc(test_sorted[f"{LABEL}_{t}"].to_numpy(), risk_cal) for t in types}
    print("    " + f"{'vitals-TCN':<12}" + "".join(f"{per_type[t]:>10.3f}" for t in types))
    print("    " + f"{'(GBM)':<12}" + "".join(
        f"{v:>10.3f}" for v in (0.081, 0.049, 0.187)))
    print("    " + f"{'(NEWS2)':<12}" + "".join(
        f"{v:>10.3f}" for v in (0.059, 0.041, 0.062)))


if __name__ == "__main__":
    main()
