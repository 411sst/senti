"""Fine-tune the MIMIC-trained flagship on eICU-CRD and measure what it buys.

    python scripts/finetune_eicu.py [--config configs/eicu_finetune.json]
                                    [--seeds 42 1 2 3 4] [--resume]
                                    [--split-by hospital] [--train-patients N]
                                    [--scratch] [--label run]

T22 scored the frozen MIMIC flagship on 208 unseen eICU hospitals. This script
asks the next question: if a site may train on its own labelled data, what
does fine-tuning add over the frozen model, and what does it cost on MIMIC?

PROTOCOL. eICU is split into train / val / test by HOSPITAL by default
(src/eicu_finetune.py), so every number on eICU test is still an
unseen-hospital number and stays comparable to T22. Then, on the same eICU
test rows:

* ``zero_shot``  -- the T22 setup: agents, heads and calibrators all fitted on
  MIMIC, nothing fitted on eICU.
* ``recal``      -- zero-shot weights, head isotonics refit on eICU val
  (the T23 "local labels" repair, at full eICU-val size).
* ``heads_ft``   -- agents frozen at their MIMIC weights; the treatment and
  mortality heads warm-started from MIMIC and fine-tuned on eICU train.
* ``full_ft``    -- every learned agent fine-tuned on eICU train from its
  MIMIC weights (XGBoost continues boosting), agent calibrators refit on eICU
  val, then heads warm-started and fine-tuned as above.
* ``scratch``    -- optional (``--scratch``): the same stack trained on eICU
  alone, no MIMIC at all. It tells whether MIMIC pretraining is helping.

SEEDS. The MIMIC source stack and the eICU split are fitted once, at the
audited ``seed`` (42), and shared -- the starting point a site would receive.
Each of ``seeds`` then re-rolls everything trained on eICU: the agents'
fine-tune, the booster's continued boosting and its out-of-fold folds, and
both heads. The spread across seeds is therefore fine-tuning variance alone,
and all arms of all seeds go through one paired, patient-clustered bootstrap
on identical resamples.

Early stopping and isotonic calibration on eICU use eICU val only; eICU test
is touched once, for the report. Fine-tuning is seeded with the pretrained
weights' own val score, so a component whose fine-tune never beats the
pretrained net on eICU val comes back unchanged.

WHAT IS NOT FINE-TUNED. NEWS2 is a deterministic lookup; only its isotonic
calibrator is refit. ecg_meas is frozen by default: eICU has no ECGs, so every
eICU row takes the agent's learned no-ECG path and there is nothing to adapt.
Cadence is excluded for the reason given in scripts/score_eicu.py.

FORGETTING. The fine-tuned stacks are also scored on MIMIC test, so the report
shows what adapting to eICU costs on the source cohort.

RESUMING. ``--resume`` skips any agent whose emission files for a stage are
already on disk, so a long multi-seed run can pick up where it stopped.

OUTPUTS (all gitignored, derived from credentialed data):
    data/cache/embeddings_eicu_ft/<label>/source/*.npz
    data/cache/embeddings_eicu_ft/<label>/s<seed>/{finetuned,scratch}/*.npz
    data/cache/eicu_ft_split_<label>.parquet     eICU split assignment
    data/cache/eicu_finetune_<label>.npz         per-arm eICU test scores
    data/cache/eicu_finetune_<label>.json        settings + metrics
    models/eicu_finetune/<label>/s<seed>/        fine-tuned weights
"""

import argparse
import copy
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from baselines_mimic import load_splits  # noqa: E402
from eicu_finetune import (DEFAULTS, assign_splits, load_settings,  # noqa: E402
                           save_agent)
from metrics import auprc, auroc, ece, ece_adaptive  # noqa: E402
from score_eicu import (EICU_COHORT, EICU_LABS, build_agent,  # noqa: E402
                        fresh_mask, point_at_eicu, restore_mimic)

LABEL = "label_24h"
# Emission split names. eICU splits get their own names so MIMIC and eICU
# emissions can share one directory (and one AgentBank) without colliding.
MIMIC_SPLITS = ("train", "val", "test")
EICU_SPLITS = {"train": "etrain", "val": "eval", "test": "etest"}
EICU_NAMES = tuple(EICU_SPLITS.values())
# Splits each stage emits, per agent.
SOURCE_SPLITS = MIMIC_SPLITS + EICU_NAMES
FINETUNED_SPLITS = EICU_NAMES + ("test",)
SCRATCH_SPLITS = EICU_NAMES
# The val-tuned FAIR+TRAJ booster settings (results_fusion.md, Protocol).
PUBLISHED_FAIR_TRAJ = {"max_depth": 4, "scale_pos_weight": 1.0}


# --- helpers ------------------------------------------------------------------

def ordered(df):
    return df.sort_values(["stay_id", "hour"]).reset_index(drop=True)


def cached(out_dir, name, splits):
    return all((out_dir / f"{name}_{sp}.npz").exists() for sp in splits)


def metrics_row(tag, y, r, quiet=False):
    p = float(y.mean())
    out = dict(tag=tag, n=int(len(y)), prev=p, auprc=auprc(y, r),
               auroc=auroc(y, r), ece=ece(y, r), ece_mass=ece_adaptive(y, r))
    out["lift"] = out["auprc"] / p
    if not quiet:
        print(f"  {tag:<34} n={out['n']:>9,}  prev {100*p:5.2f}%  "
              f"AUPRC {out['auprc']:.4f}  lift {out['lift']:5.2f}  "
              f"AUROC {out['auroc']:.4f}  ECE {out['ece_mass']:.4f}")
    return out


# --- agents -------------------------------------------------------------------

def emit_all(agent, name, frames, out_dir):
    """Emit ``frames`` ({split_name: df}) with the right auxiliary source."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, df in frames.items():
        if split in EICU_NAMES:
            point_at_eicu(name, agent)
        else:
            restore_mimic(name, agent)
        agent.emit_and_cache(df, out_dir / f"{name}_{split}.npz")
    restore_mimic(name, agent)


def fit_source_agent(name, frames, out_dir, s):
    """Fit one agent on MIMIC at the audited seed and emit every split."""
    t0 = time.time()
    agent = build_agent(name, s["source_epochs"], seed=s["seed"])
    agent.fit(frames["train"], frames["val"])
    emit_all(agent, name, {k: frames[k] for k in SOURCE_SPLITS}, out_dir)
    print(f"[{name}] source fit + emit in {time.time()-t0:.0f}s")
    return agent


def finetune_agent(source, name, frames, out_dir, s, seed, save_dir):
    """A seeded copy of ``source`` adapted to eICU, emitted to ``out_dir``."""
    t0 = time.time()
    agent = copy.deepcopy(source)
    if name == "news2":
        # Deterministic: nothing to train, only the calibrator moves.
        agent._fit_calibrator(frames["eval"])
    elif name in s["freeze_agents"]:
        print(f"[{name}] frozen (settings.freeze_agents)")
    else:
        agent.seed = seed
        point_at_eicu(name, agent)
        agent.finetune(frames["etrain"], frames["eval"], lr=s["agent_ft_lr"],
                       max_epochs=s["agent_ft_epochs"])
        restore_mimic(name, agent)
        save_agent(agent, save_dir / "agents" / f"{name}.pt")
    emit_all(agent, name, {k: frames[k] for k in FINETUNED_SPLITS}, out_dir)
    print(f"[{name}] s{seed} fine-tune + emit in {time.time()-t0:.0f}s")


def scratch_agent(name, frames, out_dir, s, seed):
    """The same agent trained on eICU alone, for the no-pretraining arm."""
    t0 = time.time()
    agent = build_agent(name, s["source_epochs"], seed=seed)
    point_at_eicu(name, agent)
    agent.fit(frames["etrain"], frames["eval"])
    emit_all(agent, name, {k: frames[k] for k in SCRATCH_SPLITS}, out_dir)
    print(f"[{name}] s{seed} scratch fit + emit in {time.time()-t0:.0f}s")


class GbmStage:
    """The XGBoost pseudo-agent: source fit, continued boosting, scratch.

    Mirrors score_eicu.build_gbm_emissions. Train-split emissions are always
    OUT-OF-FOLD (patient-level two-fold), so the heads never see in-sample GBM
    scores; that holds for the eICU train split after fine-tuning too.

    Design matrices are 145 float32 columns; holding all six splits at once
    (~8.7M rows) is ~5 GB on its own. Each is built on demand. eICU train and
    val stay cached across seeds (every seed fits on them); the others are
    dropped as soon as they are scored.
    """

    def __init__(self, frames, s):
        import distill_lead
        from baselines_fair import variant_columns
        from modality_features import load_labs

        if not Path(distill_lead.SCORE_CACHE).exists():
            # fit_gbm_student reads the tuned depth / class weight from the
            # fair GBM cache, which only exists after the full MIMIC benchmark
            # (src/baselines_fair.py, which itself needs the shipped
            # orchestrator scores). Fall back to the published FAIR+TRAJ
            # winner instead.
            print(f"[gbm] {distill_lead.SCORE_CACHE} absent; using the "
                  f"published FAIR+TRAJ winner {PUBLISHED_FAIR_TRAJ}")
            distill_lead._fair_record = lambda: dict(PUBLISHED_FAIR_TRAJ)
        self.s = s
        self.frames = frames
        self.cols = variant_columns()["FAIR+TRAJ"]
        self.labs = {"mimic": load_labs(), "eicu": load_labs(EICU_LABS)}
        self.X, self.y, self.keys = {}, {}, {}
        self.source = None

    # -- design matrices ---------------------------------------------------
    def design(self, k):
        from baselines_fair import build_design
        if k not in self.X:
            lab = self.labs["eicu" if k in EICU_NAMES else "mimic"]
            full, self.y[k], self.keys[k] = build_design(self.frames[k], lab, None)
            self.X[k] = full[self.cols]
            del full
        return self.X[k]

    def drop(self, *ks):
        for k in ks:
            self.X.pop(k, None)
        gc.collect()

    # -- fitting -------------------------------------------------------------
    def fit(self, train_key, val_key, seed, rows=None, **kw):
        from distill_lead import fit_gbm_student
        Xt, yt = self.design(train_key), self.y[train_key]
        if rows is not None:
            Xt, yt = Xt[rows], yt[rows]
        return fit_gbm_student(Xt, yt, np.ones(len(yt)), self.design(val_key),
                               self.y[val_key], device=self.s["gbm_device"],
                               seed=seed, **kw)

    def oof(self, train_key, val_key, seed, **kw):
        from distill_lead import gbm_predict
        self.design(train_key)
        pid = self.keys[train_key]["patient_id"].to_numpy()
        patients = np.sort(np.unique(pid))
        perm = np.random.default_rng(seed).permutation(patients)
        half = len(patients) // 2
        out = np.full(len(pid), np.nan)
        for hold in (perm[:half], perm[half:]):
            m = np.isin(pid, hold)
            out[m] = gbm_predict(self.fit(train_key, val_key, seed, rows=~m, **kw),
                                 self.X[train_key][m])
        assert not np.isnan(out).any()
        return out

    def predict(self, booster, k, keep=False):
        from distill_lead import gbm_predict
        r = gbm_predict(booster, self.design(k))
        if not keep:
            self.drop(k)
        return r

    def save(self, out_dir, raw, iso):
        from agents.base import AgentOutput
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, r in raw.items():
            r = np.clip(r, 1e-6, 1 - 1e-6)
            AgentOutput(keys=self.keys[split][["stay_id", "hour"]].copy(),
                        latent=np.log(r / (1 - r)).astype(np.float32)[:, None],
                        risk=iso.transform(r).astype(np.float32)
                        ).save(out_dir / f"gbm_{split}.npz")

    # -- stages --------------------------------------------------------------
    def fit_source(self, out_dir):
        """MIMIC booster at the audited seed; every split scored."""
        from sklearn.isotonic import IsotonicRegression
        t0 = time.time()
        seed = self.s["seed"]
        self.source = self.fit("train", "val", seed)
        raw = {"train": self.oof("train", "val", seed)}
        self.drop("train")
        raw["val"] = self.predict(self.source, "val")
        for k in ("test", "etest"):
            raw[k] = self.predict(self.source, k)
        for k in ("etrain", "eval"):                 # kept for every seed
            raw[k] = self.predict(self.source, k, keep=True)
        self.src_raw = raw
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw["val"], self.y["val"])
        self.save(out_dir, raw, iso)
        print(f"[gbm] source fit + emit in {time.time()-t0:.0f}s")

    def finetune(self, seed, out_dir, save_dir):
        """Continue boosting on eICU train; keep it only if it wins on val."""
        from distill_lead import gbm_predict
        from sklearn.isotonic import IsotonicRegression
        t0 = time.time()
        kw = dict(eta=self.s["gbm_ft_eta"], num_boost_round=self.s["gbm_ft_rounds"])
        tuned = self.fit("etrain", "eval", seed, init_model=self.source.copy(), **kw)
        raw = {"eval": gbm_predict(tuned, self.design("eval"))}
        a0 = auprc(self.y["eval"], self.src_raw["eval"])
        a1 = auprc(self.y["eval"], raw["eval"])
        print(f"[gbm] s{seed} eICU val AUPRC {a0:.4f} -> {a1:.4f} after continued boosting")
        if a1 > a0:
            # OOF on eICU train: each half is scored by the MIMIC booster
            # continued on the OTHER half.
            raw["etrain"] = self.oof("etrain", "eval", seed,
                                     init_model=self.source.copy(), **kw)
        else:
            tuned = self.source
            raw["eval"], raw["etrain"] = self.src_raw["eval"], self.src_raw["etrain"]
        for k in ("etest", "test"):
            raw[k] = self.predict(tuned, k)
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw["eval"], self.y["eval"])
        self.save(out_dir, raw, iso)
        save_dir.mkdir(parents=True, exist_ok=True)
        tuned.save_model(save_dir / "gbm.json")
        with open(save_dir / "gbm_iso.pkl", "wb") as fh:
            pickle.dump(iso, fh)
        print(f"[gbm] s{seed} fine-tune + emit in {time.time()-t0:.0f}s")

    def scratch(self, seed, out_dir):
        from distill_lead import gbm_predict
        from sklearn.isotonic import IsotonicRegression
        t0 = time.time()
        fresh = self.fit("etrain", "eval", seed)
        raw = {"eval": gbm_predict(fresh, self.design("eval")),
               "etrain": self.oof("etrain", "eval", seed),
               "etest": self.predict(fresh, "etest")}
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw["eval"], self.y["eval"])
        self.save(out_dir, raw, iso)
        print(f"[gbm] s{seed} scratch fit + emit in {time.time()-t0:.0f}s")


# --- heads --------------------------------------------------------------------

def fit_heads(bank, train, val, epochs, seed, lr=1e-3, init=None):
    """Treatment + mortality heads, as in scripts/two_head.py.

    ``init`` ({"T": head, "D": head}) warm-starts each head from a fitted one;
    the bank may differ (eICU instead of MIMIC) as long as agents and latent
    dims match. Each head's isotonic is fit on the V_fit half of ``val``.
    """
    from decompose import treatment_only_label
    from distill_lead import DistilledOrchestrator, make_soft_frame
    from fusion_common import split_val_rows
    from sklearn.isotonic import IsotonicRegression

    train_o = ordered(train)
    val_o, fit_mask, _ = split_val_rows(val)
    targets = {
        "T": (treatment_only_label(train_o), treatment_only_label(val_o)),
        "D": (train_o["label_24h_death"].to_numpy(float),
              val_o["label_24h_death"].to_numpy(float)),
    }
    heads = {}
    for name, (y_tr, y_va) in targets.items():
        t0 = time.time()
        soft = pd.concat([make_soft_frame(train_o, y_tr),
                          make_soft_frame(val_o, y_va)])
        head = DistilledOrchestrator(bank=bank, max_epochs=epochs, seed=seed,
                                     lr=lr, verbose=False, soft_frame=soft)
        if init is not None:
            head.net = copy.deepcopy(init[name].net)
            head.warm_start = True
        head._fit_model(train, val)
        head.warm_start = False
        raw = head.raw_risk(val)
        head.calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            raw[fit_mask], y_va[fit_mask])
        heads[name] = head
        print(f"[heads] s{seed} {name}-head {'fine-tuned' if init else 'fit'} in "
              f"{time.time()-t0:.0f}s ({int(y_tr.sum()):,} train positives)")
    return heads


def recalibrated(heads, val):
    """Copies of ``heads`` with isotonics refit on ``val``'s V_fit half."""
    from decompose import treatment_only_label
    from fusion_common import split_val_rows
    from sklearn.isotonic import IsotonicRegression

    val_o, fit_mask, _ = split_val_rows(val)
    y = {"T": treatment_only_label(val_o),
         "D": val_o["label_24h_death"].to_numpy(float)}
    out = {}
    for name, head in heads.items():
        h = copy.copy(head)
        raw = h.raw_risk(val)
        h.calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            raw[fit_mask], y[name][fit_mask])
        out[name] = h
    return out


def score_heads(heads, df):
    """Noisy-OR of the two calibrated, strictified heads (T9)."""
    from fusion_ladder import _strictify
    cal = {}
    for name, head in heads.items():
        raw = head.raw_risk(df)
        cal[name] = np.clip(_strictify(head.calibrator.transform(raw), raw), 0, 1)
    return 1.0 - (1.0 - cal["T"]) * (1.0 - cal["D"])


def release(*head_dicts):
    """Drop the heads' bank references so the bank itself can be freed."""
    for heads in head_dicts:
        for head in heads.values():
            head.bank = None
    gc.collect()


# --- report -------------------------------------------------------------------

def report(scores, mimic_scores, et, mt, seeds, s, split_report):
    from baselines_fair import bootstrap_indices
    y = et[LABEL].to_numpy(float)
    dense = fresh_mask(et, 2)
    out = {"settings": s, "split": split_report, "eicu_test": {},
           "eicu_test_matched": {}, "mimic_test": {}, "delta_vs_zero_shot": {},
           "seed_summary": {}}
    print(f"\n=== eICU test ({split_report['test']['hospitals']} held-out hospitals) ===")
    for arm, r in scores.items():
        out["eicu_test"][arm] = metrics_row(arm, y, r)
    print("\n=== eICU test, matched density (all 5 vitals fresh <= 2h) ===")
    for arm, r in scores.items():
        out["eicu_test_matched"][arm] = metrics_row(arm, y[dense], r[dense])
    print("\n=== MIMIC test (forgetting check) ===")
    ym = mt[LABEL].to_numpy(float)
    for arm, r in mimic_scores.items():
        out["mimic_test"][arm] = metrics_row(arm, ym, r)

    # One bootstrap pass for every arm of every seed, so every delta below is
    # paired against every other on identical patient-level resamples.
    reps = s["bootstrap_reps"]
    print(f"\n=== paired dAUPRC, eICU test, patient-clustered bootstrap ({reps} reps) ===")
    draws = {arm: [] for arm in scores}
    for idx in bootstrap_indices(et["patient_id"].to_numpy(), reps, s["seed"]):
        if y[idx].sum() in (0, len(idx)):
            continue
        for arm, r in scores.items():
            draws[arm].append(auprc(y[idx], r[idx]))
    draws = {k: np.asarray(v) for k, v in draws.items()}

    def delta(a, b):
        d = draws[a] - draws[b]
        point = out["eicu_test"][a]["auprc"] - out["eicu_test"][b]["auprc"]
        return dict(point=point, lo=float(np.percentile(d, 2.5)),
                    hi=float(np.percentile(d, 97.5)))

    for arm in scores:
        if arm != "zero_shot":
            out["delta_vs_zero_shot"][arm] = dv = delta(arm, "zero_shot")
            print(f"  {arm:<18} {dv['point']:+.4f} [{dv['lo']:+.4f}, {dv['hi']:+.4f}]")

    families = ["heads_ft", "full_ft"] + (["scratch"] if s["scratch"] else [])
    print(f"\n=== across {len(seeds)} fine-tuning seeds ===")
    for fam in families:
        arms = [f"{fam}_s{sd}" for sd in seeds]
        a = np.array([out["eicu_test"][k]["auprc"] for k in arms])
        r = np.array([out["eicu_test"][k]["auroc"] for k in arms])
        clear = sum(out["delta_vs_zero_shot"][k]["lo"] > 0 for k in arms)
        dmin = min(out["delta_vs_zero_shot"][k]["point"] for k in arms)
        best, worst = arms[int(a.argmax())], arms[int(a.argmin())]
        summ = dict(auprc_mean=float(a.mean()), auprc_sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                    auprc_min=float(a.min()), auprc_max=float(a.max()),
                    auroc_mean=float(r.mean()), clear_of_zero=f"{clear}/{len(arms)}",
                    smallest_gain=float(dmin))
        if len(arms) > 1:
            summ["seed_gap"] = delta(best, worst)
            summ["seed_gap"]["between"] = [best, worst]
        out["seed_summary"][fam] = summ
        print(f"  {fam:<9} AUPRC {summ['auprc_mean']:.4f} +/- {summ['auprc_sd']:.4f} "
              f"[{summ['auprc_min']:.4f}, {summ['auprc_max']:.4f}]  "
              f"AUROC {summ['auroc_mean']:.4f}  clear of zero {summ['clear_of_zero']}  "
              f"smallest gain {summ['smallest_gain']:+.4f}")
        if "seed_gap" in summ:
            g = summ["seed_gap"]
            print(f"  {'':<9} widest seed gap {best} - {worst}: "
                  f"{g['point']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]")
    return out


# --- main ---------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="JSON settings file (see configs/)")
    ap.add_argument("--label", default="run", help="suffix for every output")
    ap.add_argument("--resume", action="store_true",
                    help="skip agents whose emissions are already cached")
    ap.add_argument("--agents", nargs="+")
    ap.add_argument("--split-by", choices=["hospital", "patient"])
    ap.add_argument("--split-fracs", nargs=3, type=float)
    ap.add_argument("--split-seed", type=int)
    ap.add_argument("--train-patients", type=int,
                    help="subsample eICU train to N patients (0 = all)")
    ap.add_argument("--seed", type=int, help="source stack seed")
    ap.add_argument("--seeds", nargs="+", type=int, help="fine-tuning seeds")
    ap.add_argument("--source-epochs", type=int)
    ap.add_argument("--source-fusion-epochs", type=int)
    ap.add_argument("--agent-ft-lr", type=float)
    ap.add_argument("--agent-ft-epochs", type=int)
    ap.add_argument("--head-ft-lr", type=float)
    ap.add_argument("--head-ft-epochs", type=int)
    ap.add_argument("--gbm-ft-eta", type=float)
    ap.add_argument("--gbm-ft-rounds", type=int)
    ap.add_argument("--gbm-device", choices=["cpu", "cuda"])
    ap.add_argument("--freeze-agents", nargs="*")
    ap.add_argument("--scratch", action="store_true", default=None,
                    help="also train the stack on eICU alone (no MIMIC)")
    ap.add_argument("--bootstrap-reps", type=int)
    args = ap.parse_args()
    s = load_settings(args.config)
    for key in DEFAULTS:
        v = getattr(args, key)
        if v is not None:
            s[key] = v
    return args.label, args.resume, s


def main():
    label, resume, s = parse_args()
    seeds = list(s["seeds"])
    print("[settings]", json.dumps(s))
    cache = REPO / "data" / "cache"
    root = cache / "embeddings_eicu_ft" / label
    src_dir = root / "source"
    ft_dir = {sd: root / f"s{sd}" / "finetuned" for sd in seeds}
    sc_dir = {sd: root / f"s{sd}" / "scratch" for sd in seeds}
    model_dir = {sd: REPO / "models" / "eicu_finetune" / label / f"s{sd}" for sd in seeds}

    mimic = load_splits()
    cohort = pd.read_parquet(EICU_COHORT)
    clash = set(pd.concat(mimic.values())["stay_id"]) & set(cohort["stay_id"])
    assert not clash, f"{len(clash)} eICU/MIMIC stay_id collisions"
    cohort, split_report = assign_splits(cohort, by=s["split_by"],
                                         fracs=s["split_fracs"],
                                         seed=s["split_seed"],
                                         train_patients=s["train_patients"])
    eicu = {k: cohort[cohort["split"] == k] for k in ("train", "val", "test")}
    cohort[["stay_id", "patient_id", "hospitalid", "split"]].drop_duplicates(
        "stay_id").to_parquet(cache / f"eicu_ft_split_{label}.parquet")
    print("[split]", json.dumps(split_report))
    del cohort                                   # the split frames are copies
    gc.collect()
    frames = {**mimic, **{EICU_SPLITS[k]: v for k, v in eicu.items()}}

    # ---- agents: one source fit, then every seed's fine-tune ---------------
    for name in s["agents"]:
        todo = [sd for sd in seeds
                if not (resume and cached(ft_dir[sd], name, FINETUNED_SPLITS))]
        todo_scratch = [sd for sd in seeds if s["scratch"] and not (
            resume and cached(sc_dir[sd], name, SCRATCH_SPLITS))]
        need_source = not (resume and cached(src_dir, name, SOURCE_SPLITS)) or todo
        if name == "gbm":
            if not (need_source or todo_scratch):
                continue
            gbm = GbmStage(frames, s)
            gbm.fit_source(src_dir)              # also needed by every fine-tune
            for sd in todo:
                gbm.finetune(sd, ft_dir[sd], model_dir[sd])
            for sd in todo_scratch:
                gbm.scratch(sd, sc_dir[sd])
            del gbm
        else:
            if need_source:
                source = fit_source_agent(name, frames, src_dir, s)
                for sd in todo:
                    finetune_agent(source, name, frames, ft_dir[sd], s, sd, model_dir[sd])
                del source
            for sd in todo_scratch:
                scratch_agent(name, frames, sc_dir[sd], s, sd)
        gc.collect()

    # ---- heads ---------------------------------------------------------------
    from orchestrator.fusion import AgentBank
    names = tuple(s["agents"])
    scores, mimic_scores = {}, {}

    bank = AgentBank.load(splits=SOURCE_SPLITS, agent_names=names, emb_dir=src_dir)
    source_heads = fit_heads(bank, mimic["train"], mimic["val"],
                             s["source_fusion_epochs"], s["seed"])
    scores["zero_shot"] = score_heads(source_heads, eicu["test"])
    mimic_scores["zero_shot"] = score_heads(source_heads, mimic["test"])
    scores["recal"] = score_heads(recalibrated(source_heads, eicu["val"]), eicu["test"])
    for sd in seeds:
        heads = fit_heads(bank, eicu["train"], eicu["val"], s["head_ft_epochs"],
                          sd, lr=s["head_ft_lr"], init=source_heads)
        scores[f"heads_ft_s{sd}"] = score_heads(heads, eicu["test"])
        mimic_scores[f"heads_ft_s{sd}"] = score_heads(heads, mimic["test"])
        release(heads)
    # source_heads are still needed, but only for their nets (warm-start init).
    release(source_heads)
    del bank
    gc.collect()

    for sd in seeds:
        bank = AgentBank.load(splits=FINETUNED_SPLITS, agent_names=names,
                              emb_dir=ft_dir[sd])
        heads = fit_heads(bank, eicu["train"], eicu["val"], s["head_ft_epochs"],
                          sd, lr=s["head_ft_lr"], init=source_heads)
        scores[f"full_ft_s{sd}"] = score_heads(heads, eicu["test"])
        mimic_scores[f"full_ft_s{sd}"] = score_heads(heads, mimic["test"])
        model_dir[sd].mkdir(parents=True, exist_ok=True)
        for hname, head in heads.items():
            head.save_checkpoint(model_dir[sd] / f"head_{hname}.pt")
        release(heads)
        del bank
        gc.collect()

    for sd in (seeds if s["scratch"] else []):
        bank = AgentBank.load(splits=SCRATCH_SPLITS, agent_names=names,
                              emb_dir=sc_dir[sd])
        heads = fit_heads(bank, eicu["train"], eicu["val"],
                          s["source_fusion_epochs"], sd)
        scores[f"scratch_s{sd}"] = score_heads(heads, eicu["test"])
        release(heads)
        del bank
        gc.collect()

    # ---- report --------------------------------------------------------------
    results = report(scores, mimic_scores, ordered(eicu["test"]),
                     ordered(mimic["test"]), seeds, s, split_report)
    et = ordered(eicu["test"])
    np.savez_compressed(cache / f"eicu_finetune_{label}.npz",
                        stay_id=et["stay_id"].to_numpy(), hour=et["hour"].to_numpy(),
                        **{f"risk_{arm}": r for arm, r in scores.items()})
    with open(cache / f"eicu_finetune_{label}.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\n[done] scores -> data/cache/eicu_finetune_{label}.npz, "
          f"weights -> models/eicu_finetune/{label}/")


if __name__ == "__main__":
    main()
