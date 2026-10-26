"""The single source of truth for the results table.

    python scripts/results_table.py [--epochs 8]

Fits every model -- NEWS2, GBM, the three agents, and the orchestrator -- on the
same splits, scores them all on the same test rows, and prints one table. The
per-model numbers used to be copied by hand between scripts, which is a
maintenance hazard: this file recomputes them instead so they cannot drift.

Reported per model: AUPRC and AUROC (raw and calibrated), ECE (raw and
calibrated), and AUPRC per event type. **AUPRC is the primary discrimination
metric** -- with ~4.7% positive hours, AUROC flatters everything (see the note
in metrics.auroc) and is included as context only.

Every model is calibrated the same way: isotonic regression fit on the val split
only, never train, never test.
"""

# PyTorch and XGBoost each bundle an OpenMP runtime and deadlock when both are
# active in one process on macOS (see tests/conftest.py). This script is the one
# place that genuinely needs both, so it pins them to a single thread. Must run
# before either library is imported.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from agents.labs_agent import LabsAgent  # noqa: E402
from agents.news2_agent import News2Agent  # noqa: E402
from agents.vitals_tcn import VitalsTCNAgent  # noqa: E402
from baselines_mimic import GBM_FEATURES, load_splits, train_gbm  # noqa: E402
from metrics import auprc, auroc, ece  # noqa: E402
from mimic_cohort import EVENT_TYPES  # noqa: E402
from orchestrator.fusion import AgentBank, Orchestrator  # noqa: E402

LABEL = "label_24h"
EMB_DIR = REPO / "data" / "cache" / "embeddings"


def calibrate(raw_val, y_val, raw_test):
    """Isotonic fit on val, applied to test -- the same protocol agents use."""
    return IsotonicRegression(out_of_bounds="clip").fit(raw_val, y_val).transform(raw_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--skip-agents", action="store_true",
                    help="reuse cached agent latents instead of refitting")
    args = ap.parse_args()

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    test_sorted = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    y_val = val.sort_values(["stay_id", "hour"])[LABEL].to_numpy(dtype=float)
    y_test = test_sorted[LABEL].to_numpy(dtype=float)

    results = {}   # name -> (raw_scores, calibrated_scores) on test_sorted rows

    # --- NEWS2 (deterministic, no training) ------------------------------
    # raw_probability (score/20) rather than the 0-20 raw_risk: the "raw" slot
    # feeds ece(), which rejects values outside [0, 1]. The rescaling is
    # monotone, so the AUPRC/AUROC columns are identical either way.
    t0 = time.time()
    news2 = News2Agent().fit(train, val)
    results["NEWS2"] = (news2.raw_probability(test), news2.emit(test).risk)
    print(f"[fit] NEWS2 {time.time() - t0:.0f}s")

    # --- GBM baseline ----------------------------------------------------
    t0 = time.time()
    gbm = train_gbm(train)
    gbm_val = gbm.predict_proba(val.sort_values(["stay_id", "hour"])[GBM_FEATURES])[:, 1]
    gbm_test = gbm.predict_proba(test_sorted[GBM_FEATURES])[:, 1]
    results["GBM"] = (gbm_test, calibrate(gbm_val, y_val, gbm_test))
    print(f"[fit] GBM {time.time() - t0:.0f}s")

    # --- learned agents --------------------------------------------------
    for label, agent in (("vitals-TCN", VitalsTCNAgent(max_epochs=args.epochs, verbose=False)),
                         ("labs", LabsAgent(max_epochs=12, verbose=False))):
        t0 = time.time()
        agent.fit(train, val)
        results[label] = (agent.raw_risk(test), agent.emit(test).risk)
        for split_name, df in (("train", train), ("val", val), ("test", test)):
            agent.emit_and_cache(df, EMB_DIR / f"{agent.name}_{split_name}.npz")
        print(f"[fit] {label} {time.time() - t0:.0f}s")

    # NEWS2 latents are needed by the bank too.
    for split_name, df in (("train", train), ("val", val), ("test", test)):
        news2.emit_and_cache(df, EMB_DIR / f"news2_{split_name}.npz")

    # --- orchestrator ----------------------------------------------------
    t0 = time.time()
    orch = Orchestrator(bank=AgentBank.load(), max_epochs=args.epochs, verbose=False)
    orch.fit(train, val)
    results["ORCHESTRATOR"] = (orch.raw_risk(test), orch.emit(test).risk)
    # Re-cache the orchestrator too. Its emissions are a function of the agent
    # latents cached moments ago, so leaving an older file on disk would strand a
    # model trained on superseded inputs -- which is exactly what happened once:
    # the ablation's LATENT column and this table's ORCHESTRATOR row came from
    # two different training runs. audit_alignment.py now fails on that skew.
    for split_name, df in (("train", train), ("val", val), ("test", test)):
        orch.emit_and_cache(df, EMB_DIR / f"{orch.name}_{split_name}.npz")
    print(f"[fit] orchestrator {time.time() - t0:.0f}s (cached)")

    # --- random reference -------------------------------------------------
    rng = np.random.default_rng(42)
    coin = rng.random(len(y_test))
    results["(random)"] = (coin, coin)

    # --- table ------------------------------------------------------------
    order = ["ORCHESTRATOR", "GBM", "labs", "vitals-TCN", "NEWS2", "(random)"]
    types = list(EVENT_TYPES)
    print(f"\n{'=' * 100}")
    print(f"RESULTS -- test split: {len(y_test):,} patient-hours, "
          f"{test_sorted['stay_id'].nunique():,} ICU stays, "
          f"{test_sorted['patient_id'].nunique():,} patients, "
          f"prevalence {100 * y_test.mean():.2f}%")
    print(f"{'=' * 100}")
    header = (f"  {'model':<14}{'AUPRC raw':>10}{'AUPRC cal':>10}"
              f"{'AUROC raw':>10}{'AUROC cal':>10}{'ECE raw':>9}{'ECE cal':>9}"
              f"   " + "".join(f"{t:>8}" for t in types))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in order:
        raw, cal = results[name]
        per_type = {t: auprc(test_sorted[f"{LABEL}_{t}"].to_numpy(), raw) for t in types}
        print(f"  {name:<14}{auprc(y_test, raw):>10.3f}{auprc(y_test, cal):>10.3f}"
              f"{auroc(y_test, raw):>10.3f}{auroc(y_test, cal):>10.3f}"
              f"{ece(y_test, raw):>9.4f}{ece(y_test, cal):>9.4f}   "
              + "".join(f"{per_type[t]:>8.3f}" for t in types))

    prev = {t: test_sorted[f"{LABEL}_{t}"].mean() for t in types}
    print(f"\n  Per-event-type columns are AUPRC on the raw score. Each type has its "
          f"own floor:\n    " + "  ".join(f"{t} {100*p:.2f}% (AUPRC floor {p:.4f})"
                                          for t, p in prev.items()))
    print(f"  AUPRC floor overall = prevalence = {y_test.mean():.4f}; AUROC floor = 0.500.")
    print("  AUPRC is the primary metric: AUROC's specificity axis barely moves under")
    print("  4.7% prevalence, so it reads high even when most alerts are wrong.")
    print("  All calibration is isotonic, fit on val only.")


if __name__ == "__main__":
    main()
