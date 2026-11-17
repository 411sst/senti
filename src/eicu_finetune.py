"""Site adaptation on eICU-CRD: splits, settings and checkpoints.

T22 scored the MIMIC-trained flagship on eICU with nothing fitted there. This
module supports the follow-up question: if a site is allowed to train on its
own data, how much of the transfer gap does fine-tuning close, and what does
it cost the source cohort?

The eICU cohort (:mod:`eicu_cohort`) is built with every row marked ``test``.
:func:`assign_splits` replaces that with a train/val/test assignment. The
default is **hospital-level**: test hospitals are never seen during
fine-tuning, so the fine-tuned number is still an unseen-hospital number and
stays comparable to T22's zero-shot result. A patient-level split is available
for comparison, but it measures within-hospital generalisation, which is an
easier question.

Everything here is pure pandas/numpy so the split logic is unit-testable
without the credentialed data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import SEED

SPLITS = ("train", "val", "test")

# Defaults for scripts/finetune_eicu.py. configs/eicu_finetune.json overrides
# them; the CLI overrides both.
DEFAULTS = {
    "agents": ["vitals_tcn", "news2", "labs", "ecg_meas", "gbm", "fluid", "resp"],
    "split_by": "hospital",
    "split_fracs": [0.6, 0.2, 0.2],
    "split_seed": SEED,
    "train_patients": 0,
    "seed": SEED,
    "seeds": [SEED, 1, 2, 3, 4],
    "source_epochs": 8,
    "source_fusion_epochs": 8,
    "agent_ft_lr": 1e-4,
    "agent_ft_epochs": 6,
    "head_ft_lr": 1e-4,
    "head_ft_epochs": 8,
    "gbm_ft_eta": 0.02,
    "gbm_ft_rounds": 500,
    "gbm_device": "cpu",
    "freeze_agents": ["ecg_meas"],
    "scratch": False,
    "bootstrap_reps": 300,
}


def load_settings(path=None):
    """``DEFAULTS`` updated by a JSON file, rejecting unknown keys.

    Unknown keys fail loudly: a misspelt setting would otherwise fall back to
    its default without anyone noticing.
    """
    settings = dict(DEFAULTS)
    if path is None:
        return settings
    with open(path, encoding="utf-8") as fh:
        overrides = json.load(fh)
    overrides = {k: v for k, v in overrides.items() if not k.startswith("_")}
    unknown = sorted(set(overrides) - set(DEFAULTS))
    if unknown:
        raise KeyError(f"unknown settings in {path}: {unknown}")
    settings.update(overrides)
    return settings


def _cut(units, weights, fracs, seed):
    """Shuffle ``units`` and cut them so each split gets ~``fracs`` of weight."""
    fracs = np.asarray(fracs, dtype=float)
    if len(fracs) != 3 or (fracs <= 0).any() or not np.isclose(fracs.sum(), 1.0):
        raise ValueError(f"split fractions must be three positives summing to 1, got {fracs}")
    perm = np.random.default_rng(seed).permutation(len(units))
    share = np.cumsum(np.asarray(weights, dtype=float)[perm])
    share /= share[-1]
    # A unit goes to the split its cumulative share STARTS in, so a large
    # hospital straddling a boundary is never split across two splits.
    start = np.r_[0.0, share[:-1]]
    bounds = np.cumsum(fracs)[:-1]
    which = np.searchsorted(bounds, start, side="right")
    return dict(zip(np.asarray(units)[perm], np.asarray(SPLITS)[which]))


def assign_splits(cohort, by="hospital", fracs=(0.6, 0.2, 0.2), seed=SEED,
                  train_patients=0):
    """Return ``(cohort with a fresh split column, report dict)``.

    ``by="hospital"`` cuts hospitals, weighted by stay count, so the three
    splits hold disjoint hospitals. eICU's ``uniquepid`` can recur across
    hospitals in the same health system, so patients are then made disjoint
    too: a patient's rows in val or test are dropped if that patient appears
    in an earlier split (train before val before test). The drop is counted
    in the report rather than hidden.

    ``by="patient"`` cuts patients directly; hospitals are then shared.

    ``train_patients > 0`` subsamples the train split to that many patients
    (seeded), for learning curves: how much local data does fine-tuning need?
    """
    if by not in ("hospital", "patient"):
        raise ValueError(f"split_by must be 'hospital' or 'patient', got {by!r}")
    if by == "hospital":
        stays = cohort.groupby("hospitalid")["stay_id"].nunique()
        mapping = _cut(stays.index.to_numpy(), stays.to_numpy(), fracs, seed)
        split = cohort["hospitalid"].map(mapping).to_numpy()
    else:
        patients = np.sort(cohort["patient_id"].unique())
        mapping = _cut(patients, np.ones(len(patients)), fracs, seed)
        split = cohort["patient_id"].map(mapping).to_numpy()

    # Work on arrays and copy the (multi-GB) cohort only once, at the end.
    pid = cohort["patient_id"].to_numpy()
    dropped = {}
    seen = set()
    keep = np.ones(len(cohort), dtype=bool)
    for name in SPLITS:
        in_split = split == name
        clash = in_split & pd.Series(pid).isin(seen).to_numpy()
        dropped[name] = int(pd.unique(pid[clash]).size)
        keep &= ~clash
        seen |= set(pd.unique(pid[in_split & ~clash]))

    if train_patients:
        train_ids = np.sort(pd.unique(pid[keep & (split == "train")]))
        if train_patients < len(train_ids):
            chosen = np.random.default_rng(seed).choice(
                train_ids, train_patients, replace=False)
            keep &= (split != "train") | pd.Series(pid).isin(chosen).to_numpy()

    df = cohort[keep]
    df = df.assign(split=split[keep])
    report = {"by": by, "fracs": list(fracs), "seed": seed,
              "dropped_overlap_patients": dropped}
    for split in SPLITS:
        part = df[df["split"] == split]
        report[split] = {
            "hours": int(len(part)),
            "stays": int(part["stay_id"].nunique()),
            "patients": int(part["patient_id"].nunique()),
            "hospitals": int(part["hospitalid"].nunique()),
            "positive_hours": int(part["label_24h"].sum()),
        }
    return df.reset_index(drop=True), report


def agent_checkpoint(agent):
    """Weights + everything needed to reuse them, for a torch agent."""
    return {
        "name": agent.name,
        "state_dict": {k: v.detach().cpu() for k, v in agent.net.state_dict().items()},
        "hparams": dict(getattr(agent, "hparams", {})),
        "stats": getattr(agent, "stats", None),
        "calibrator": agent.calibrator,
    }


def save_agent(agent, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent_checkpoint(agent), path)
    return path
