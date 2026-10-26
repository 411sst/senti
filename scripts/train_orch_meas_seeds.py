"""Train the core-3+meas orchestrator across seeds: the fusion-work substrate.

The orchestrator's run-to-run seed noise (core-3 val AUPRC 0.2217-0.2399 across
seeds, results_ecg.md) exceeds typical fusion deltas, so every experiment that
consumes its scores must be run per-seed with within-seed pairing. This script
produces everything those experiments need, once:

  * emission caches   data/cache/embeddings/orchestrator-meas-s{seed}_{split}.npz
  * checkpoints       models/orchestrator/meas_s{seed}.pt   (frozen-head reuse)
  * one consolidated  data/cache/orch_meas_seeds.npz with RAW val/test scores
    per seed -- raw on purpose: score fusion (max/stack/gate) is non-monotone
    in its inputs, so the fusion layer fits its own calibrators on its own
    fit-half rather than trusting full-val isotonics baked into the caches.
  * with --oof: out-of-fold calibrated risks on TRAIN (patient-level 2-fold,
    fold split fixed across seeds) -> orchestrator-meas-s{seed}_trainoof.npz,
    the honest teacher for lead-time distillation.

    python scripts/train_orch_meas_seeds.py --seeds 42,1,2 [--oof] [--epochs 8]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

CONFIG = "core-3+meas"
SEEDS_NPZ = REPO / "data/cache/orch_meas_seeds.npz"
EMB_DIR = REPO / "data/cache/embeddings"
CKPT_DIR = REPO / "models/orchestrator"


def sorted_split(df):
    return df.sort_values(["stay_id", "hour"]).reset_index(drop=True)


def oof_teacher(bank, train, val, seed, epochs):
    """Out-of-fold calibrated train risks: patient-level 2-fold, fixed folds.

    The fold split uses a FIXED rng (42) so every seed's teacher is scored on
    identical folds -- only the orchestrator seed varies. Each fold model early
    stops and calibrates on the real val split, exactly like the full model.
    """
    from agents.base import AgentOutput
    from orchestrator.fusion import Orchestrator

    patients = np.sort(train["patient_id"].unique())
    perm = np.random.default_rng(42).permutation(patients)
    half = len(patients) // 2
    folds = (set(perm[:half]), set(perm[half:]))

    outs = []
    for i, hold in enumerate(folds):
        fit_df = train[~train["patient_id"].isin(hold)]
        hold_df = train[train["patient_id"].isin(hold)]
        t0 = time.time()
        orch = Orchestrator(bank=bank, max_epochs=epochs, seed=seed,
                            verbose=False)
        orch.fit(fit_df, val)
        outs.append(orch.emit(hold_df))
        print(f"    [oof fold {i}] fit on {fit_df['patient_id'].nunique():,} "
              f"patients, emitted {len(hold_df):,} held-out rows "
              f"({time.time() - t0:.0f}s)")

    keys = pd.concat([o.keys for o in outs], ignore_index=True)
    latent = np.concatenate([o.latent for o in outs])
    risk = np.concatenate([o.risk for o in outs])
    order = np.lexsort((keys["hour"].to_numpy(), keys["stay_id"].to_numpy()))
    return AgentOutput(keys=keys.iloc[order].reset_index(drop=True),
                       latent=latent[order], risk=risk[order])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,2,3,4", help="comma list")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--oof", action="store_true",
                    help="also produce out-of-fold train risks (distillation "
                         "teacher)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    from baselines_mimic import load_splits
    from metrics import auprc
    from orchestrator.fusion import AgentBank, Orchestrator, config_agents

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    agents = config_agents(CONFIG)
    bank = AgentBank.load(agent_names=agents)
    print(f"[bank] {CONFIG}: {bank.n_features} features from {list(agents)}")

    payload, val_auprcs = {}, {}
    for split_name, df in (("val", val), ("test", test)):
        ordered = sorted_split(df)
        payload[f"stay_id_{split_name}"] = ordered["stay_id"].to_numpy()
        payload[f"hour_{split_name}"] = ordered["hour"].to_numpy()
        payload[f"patient_id_{split_name}"] = ordered["patient_id"].to_numpy()
        payload[f"y_{split_name}"] = ordered["label_24h"].to_numpy(dtype=float)

    for seed in seeds:
        t0 = time.time()
        orch = Orchestrator(bank=bank, max_epochs=args.epochs, seed=seed,
                            verbose=False)
        orch.fit(train, val)
        orch.save_checkpoint(CKPT_DIR / f"meas_s{seed}.pt")
        for split_name, df in (("train", train), ("val", val), ("test", test)):
            orch.emit_and_cache(
                df, EMB_DIR / f"orchestrator-meas-s{seed}_{split_name}.npz")
        raw = {s: orch.raw_risk(df)
               for s, df in (("val", val), ("test", test))}
        for split_name, r in raw.items():
            payload[f"raw_s{seed}_{split_name}"] = r.astype(float)
        val_auprcs[str(seed)] = float(auprc(payload["y_val"], raw["val"]))
        print(f"[seed {seed}] val AUPRC {val_auprcs[str(seed)]:.4f} "
              f"({time.time() - t0:.0f}s incl. emissions)")

        if args.oof:
            out = oof_teacher(bank, train, val, seed, args.epochs)
            out.save(EMB_DIR / f"orchestrator-meas-s{seed}_trainoof.npz")

    payload["meta_json"] = np.array(json.dumps({
        "config": CONFIG, "agents": list(agents), "epochs": args.epochs,
        "seeds": seeds, "val_auprc": val_auprcs, "oof": bool(args.oof),
    }))
    SEEDS_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SEEDS_NPZ, **payload)
    print(f"[cache] raw per-seed scores -> {SEEDS_NPZ}")


if __name__ == "__main__":
    main()
