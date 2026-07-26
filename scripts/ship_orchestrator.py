"""Train and cache the shipped orchestrator configurations.

    python scripts/ship_orchestrator.py

Trains the PRIMARY configuration (core-3 + notes) plus both fallbacks (core-3,
all-5), caches each one's emissions under a config-suffixed name, and mirrors
the primary to the unsuffixed ``orchestrator_*.npz`` that downstream code reads
by default.

Caching all three means graceful degradation is a *load*, not a retrain: if a
site has no notes pipeline, ``core-3`` is already on disk.

Everything else is held to the audited settings -- death_weight=1,
isotonic-on-val calibration, seed 42.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from baselines_mimic import load_splits  # noqa: E402
from metrics import auprc, auroc, ece  # noqa: E402
from mimic_cohort import EVENT_TYPES  # noqa: E402
from orchestrator.fusion import (  # noqa: E402
    DEFAULT_CONFIG, ORCHESTRATOR_CONFIGS, AgentBank, Orchestrator,
)

LABEL = "label_24h"
EMB_DIR = REPO / "data" / "cache" / "embeddings"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    test_sorted = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    y = test_sorted[LABEL].to_numpy(dtype=float)
    types = list(EVENT_TYPES)

    results = {}
    for config, agents in ORCHESTRATOR_CONFIGS.items():
        t0 = time.time()
        bank = AgentBank.load(agent_names=agents)
        orch = Orchestrator(bank=bank, max_epochs=args.epochs, verbose=False)
        orch.fit(train, val)

        for split_name, df in (("train", train), ("val", val), ("test", test)):
            orch.emit_and_cache(df, EMB_DIR / f"orchestrator-{config}_{split_name}.npz")
        raw, cal = orch.raw_risk(test), orch.emit(test).risk
        _, attn = orch.attention_weights(test)
        results[config] = {"raw": raw, "cal": cal, "agents": agents,
                           "attn": dict(zip(agents, attn.mean(axis=0)))}
        tag = "  <- PRIMARY" if config == DEFAULT_CONFIG else ""
        print(f"[ship] {config:<10} {len(agents)} agents  {time.time()-t0:.0f}s  cached{tag}")

    # The unsuffixed name is what audit_alignment.py and downstream consumers
    # read; point it at the primary so "the orchestrator" means one thing.
    for split_name in ("train", "val", "test"):
        shutil.copyfile(EMB_DIR / f"orchestrator-{DEFAULT_CONFIG}_{split_name}.npz",
                        EMB_DIR / f"orchestrator_{split_name}.npz")
    print(f"[ship] orchestrator_*.npz  <- {DEFAULT_CONFIG}")

    print(f"\n{'=' * 96}")
    print(f"SHIPPED CONFIGURATIONS -- test {len(y):,} patient-hours, "
          f"prevalence {100*y.mean():.2f}%")
    print("=" * 96)
    hdr = (f"  {'config':<10}{'role':<10}{'agents':>7}{'AUPRC raw':>11}"
           f"{'AUPRC cal':>11}{'AUROC':>8}{'ECE cal':>9}   "
           + "".join(f"{t:>9}" for t in types))
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for config, r in results.items():
        role = "PRIMARY" if config == DEFAULT_CONFIG else "fallback"
        per_type = {t: auprc(test_sorted[f"{LABEL}_{t}"].to_numpy(), r["raw"])
                    for t in types}
        print(f"  {config:<10}{role:<10}{len(r['agents']):>7}"
              f"{auprc(y, r['raw']):>11.3f}{auprc(y, r['cal']):>11.3f}"
              f"{auroc(y, r['raw']):>8.3f}{ece(y, r['cal']):>9.4f}   "
              + "".join(f"{per_type[t]:>9.3f}" for t in types))

    print("\n  Attention (mean over test hours):")
    for config, r in results.items():
        parts = "  ".join(f"{a}={w:.3f}" for a, w in r["attn"].items())
        print(f"    {config:<10} {parts}")

    print("\n  Graceful degradation: if a modality's cache is absent, "
          "AgentBank.load()\n  falls back primary -> core-3 automatically "
          "(orchestrator.fusion.resolve_config).")


if __name__ == "__main__":
    main()
