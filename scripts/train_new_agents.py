"""Train the cadence and/or fluid agents; report standalone; cache emissions.

Standalone reporting mirrors the other agent trainers: full-cohort AUROC/AUPRC
on val and test plus per-event-type AUPRC with DEATH highlighted -- the
orchestrator's known death deficit (0.073 vs the GBM's 0.19) is the hole the
fluid agent aims at, and a death-component gain is a secondary win condition.
Fusion verdicts come later from scripts/eval_new_agents.py, never from here.

    python scripts/train_new_agents.py [--agent cadence,fluid] [--epochs 12]
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

EMB_DIR = REPO / "data" / "cache" / "embeddings"
LABEL = "label_24h"
EVENT_TYPES = ("vent", "vaso", "death")


def report(name, df, split_name, risk):
    from metrics import auprc, auroc

    ordered = df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    y = ordered[LABEL].to_numpy(dtype=float)
    line = (f"  [{name}] {split_name}: AUPRC {auprc(y, risk):.4f}  "
            f"AUROC {auroc(y, risk):.4f}")
    for t in EVENT_TYPES:
        yt = ordered[f"{LABEL}_{t}"].to_numpy(dtype=float)
        mark = " <-- death" if t == "death" else ""
        line += f"  {t} {auprc(yt, risk):.4f}{mark}"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="cadence,fluid")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    from agents.tabular_agent import (CadenceAgent, FluidAgent,
                                      InfectionAgent, NeuroStatusAgent,
                                      RespAgent)

    from baselines_mimic import load_splits

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    classes = {
        "cadence": CadenceAgent,
        "fluid": FluidAgent,
        "resp": RespAgent,
        "resp_cap4": lambda **kw: RespAgent(max_level=4, **kw),
        "neuro_status": NeuroStatusAgent,
        "infection": InfectionAgent,
    }

    for which in args.agent.split(","):
        cls = classes[which.strip()]
        t0 = time.time()
        agent = cls(max_epochs=args.epochs)
        agent.fit(train, val)
        for split_name, df in (("val", val), ("test", test)):
            report(which, df, split_name, agent.raw_risk(df))
        for split_name, df in (("train", train), ("val", val),
                               ("test", test)):
            agent.emit_and_cache(df, EMB_DIR / f"{which}_{split_name}.npz")
        print(f"  [{which}] emissions cached ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
