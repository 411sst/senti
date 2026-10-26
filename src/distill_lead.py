"""Bundle 4c: is earliness a property of the objective, or the representation?

The GBM out-ranks; the orchestrator alarms earlier. This module tries to move
the earliness by changing only the TRAINING SIGNAL, inside a 2-family x
4-objective grid (identical objective definitions for both families):

  bce       the shipped hard-label parents (from caches; no new fits)
  tls       temporal label smoothing: clip((24 + H - tte) / H, 0, 1) on the
            ramp tte in (24, 24+H]; 1 inside the true 24h window; 0 elsewhere
  utility   hard labels, re-weighted: early positives up-weighted
            (1 + beta*tte/24), just-too-early negatives (tte in (24, 36])
            down-weighted toward 0.5 -- "stop punishing slightly-early alarms"
  distill   soft targets from the OTHER family's out-of-fold teacher:
            y_soft = (1-lambda)*y_hard + lambda*q~, where q~ is the teacher's
            risk trajectory made monotone-nondecreasing inside the pre-event
            window (running max over tte <= 24 + 12) -- the student is taught
            a clean early rise, not the teacher's jitter

If loss shaping cannot give the GBM earliness but nothing loses it for the
orchestrator either, earliness is representational -- exactly what the
channel decomposition predicts. Either answer is a claim.

Protocol (fusion_common): students early-stop on the V_fit half of val;
V_freeze supplies the selection surface (AUPRC + median lead@2% vs NEWS2) and
the Pareto frontier per family; ONLY frontier points are evaluated on test.
Teachers are strictly out-of-fold on train (patient-level 2-fold, the fold
split shared with scripts/train_orch_meas_seeds.py). Everything re-runs per
orchestrator seed {42, 1, 2}.

    python src/distill_lead.py [--families gbm,orch]
                               [--objectives tls,utility,distill]
                               [--seeds 42,1,2] [--reps 2000] [--quick]
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from baselines_fair import SCORE_CACHE
from baselines_mimic import load_splits
from config import SEED
from fusion_common import FusionContext, load_fair_scores, load_orch_seeds
from metrics import auprc

LABEL = "label_24h"
OUT_CACHE = "data/cache/distill_grid.npz"
TLS_H = (6, 12, 24)
UTILITY_BETA = (0.5, 1.0)
DISTILL_LAMBDA = (0.25, 0.5, 0.75, 1.0)
ENVELOPE_H = 12


# --- Objective construction (shared by both families) ------------------------

def _tte(ordered):
    return (ordered["event_h"].to_numpy(dtype=float)
            - ordered["hour"].to_numpy(dtype=float))


def tls_targets(ordered, H):
    """Temporal label smoothing on the pre-window ramp."""
    tte = _tte(ordered)
    y = ordered[LABEL].to_numpy(dtype=float).copy()
    ramp = ~np.isnan(tte) & (tte > 24) & (tte <= 24 + H)
    y[ramp] = np.clip((24 + H - tte[ramp]) / H, 0.0, 1.0)
    return y


def utility_weights(ordered, beta):
    """Sample weights: reward early catches, forgive slightly-early alarms."""
    tte = _tte(ordered)
    y = ordered[LABEL].to_numpy(dtype=float)
    w = np.ones(len(y))
    pos = y == 1
    w[pos] = 1.0 + beta * (tte[pos] / 24.0)
    near = (~pos) & ~np.isnan(tte) & (tte > 24) & (tte <= 36)
    w[near] = np.maximum(0.5, 1.0 - beta * ((36 - tte[near]) / 12.0) * 0.5)
    return w


def soft_targets(ordered, teacher_q, lam, envelope=ENVELOPE_H):
    """(1-lam)*y_hard + lam*q~ with the monotone pre-event envelope.

    Inside the window (tte <= 24 + envelope; a contiguous SUFFIX of each event
    stay, since tte falls as hour rises) the teacher trajectory is replaced by
    its running max, teaching a clean early rise. Outside, q~ = q untouched.
    Uses event_h on TRAIN rows only -- train labels inside training targets.
    """
    tte = _tte(ordered)
    q = np.asarray(teacher_q, dtype=float)
    in_win = ~np.isnan(tte) & (tte <= 24 + envelope)
    stay = ordered["stay_id"].to_numpy()
    masked = pd.Series(np.where(in_win, q, -np.inf))
    run_max = masked.groupby(stay).cummax().to_numpy()
    q_tilde = np.where(in_win, np.maximum(q, run_max), q)
    y_hard = ordered[LABEL].to_numpy(dtype=float)
    return (1.0 - lam) * y_hard + lam * q_tilde


# --- Students ----------------------------------------------------------------

def _fair_record():
    meta = json.loads(str(np.load(SCORE_CACHE, allow_pickle=True)["meta_json"]))
    return meta["records"]["FAIR+TRAJ"]


def fit_gbm_student(X_tr, y_tr, w_tr, X_es, y_es, device="cuda", seed=SEED,
                    init_model=None, eta=0.05, num_boost_round=3000):
    """FAIR+TRAJ-configured booster on (possibly soft) targets.

    Low-level ``xgb.train`` on purpose: ``binary:logistic`` accepts labels in
    [0, 1] (exact soft-target cross-entropy), which ``XGBClassifier`` refuses.
    Early stopping watches HARD-label aucpr on the V_fit half of val.

    ``init_model`` continues boosting from an existing booster (site
    adaptation: new trees are added on the target cohort, the source trees
    are kept); pair it with a smaller ``eta`` and ``num_boost_round``.
    ``best_iteration`` stays absolute, so :func:`gbm_predict` works unchanged.
    """
    import xgboost as xgb

    rec = _fair_record()
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    des = xgb.DMatrix(X_es, label=y_es)
    params = {"objective": "binary:logistic", "eval_metric": "aucpr",
              "eta": eta, "max_depth": rec["max_depth"],
              "min_child_weight": 10, "subsample": 0.8,
              "colsample_bytree": 0.8,
              "scale_pos_weight": rec["scale_pos_weight"],
              "seed": seed, "tree_method": "hist", "device": device}
    booster = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                        evals=[(des, "vfit")], early_stopping_rounds=50,
                        verbose_eval=False, xgb_model=init_model)
    return booster


def gbm_predict(booster, X):
    import xgboost as xgb
    return booster.predict(
        xgb.DMatrix(X),
        iteration_range=(0, booster.best_iteration + 1)).astype(float)


class DistilledOrchestrator:
    """Orchestrator student with per-row target/weight overrides on train.

    Composition, not fusion.py edits: overrides live in a (stay_id, hour) ->
    (y_soft, w) frame consulted inside ``_sequences``; val rows are never in
    the frame, so early stopping keeps scoring against hard labels.
    """

    def __new__(cls, *args, **kwargs):
        from orchestrator.fusion import Orchestrator

        soft_frame = kwargs.pop("soft_frame", None)

        class _Student(Orchestrator):
            def _sequences(self, df, with_labels=True):
                seqs = super()._sequences(df, with_labels)
                if with_labels and soft_frame is not None:
                    idx = soft_frame.index
                    y_arr = soft_frame["y_soft"].to_numpy(dtype=float)
                    w_arr = soft_frame["w"].to_numpy(dtype=float)
                    for s in seqs:
                        want = pd.MultiIndex.from_arrays(
                            [np.full(len(s["hours"]), s["stay_id"]),
                             s["hours"]])
                        pos = idx.get_indexer(want)
                        hit = pos >= 0
                        if hit.any():
                            s["y"][hit] = y_arr[pos[hit]]
                            s["w"][hit] = w_arr[pos[hit]]
                    return seqs
                return seqs

        return _Student(*args, **kwargs)


def make_soft_frame(ordered, y_soft, w=None):
    return pd.DataFrame({
        "y_soft": y_soft,
        "w": np.ones(len(y_soft)) if w is None else w,
    }, index=pd.MultiIndex.from_arrays(
        [ordered["stay_id"].to_numpy(), ordered["hour"].to_numpy()]))


# --- Teachers ----------------------------------------------------------------

def orch_teacher_oof(seed):
    """Calibrated OOF orchestrator risks on train (from train_orch_meas_seeds)."""
    z = np.load("data/cache/embeddings/"
                f"orchestrator-meas-s{seed}_trainoof.npz")
    return z["stay_id"], z["hour"], z["risk"].astype(float)


def gbm_teacher_oof(X_tr, y_tr, train_ordered, device="cuda"):
    """FAIR+TRAJ OOF probabilities on train: the reverse-distillation teacher.

    Patient-level 2-fold with the SAME fixed fold rng (42) as the orchestrator
    OOF, so both directions of distillation see identical fold structure.
    """
    patients = np.sort(train_ordered["patient_id"].unique())
    perm = np.random.default_rng(42).permutation(patients)
    half = len(patients) // 2
    q = np.full(len(train_ordered), np.nan)
    pid = train_ordered["patient_id"].to_numpy()
    for hold in (set(perm[:half]), set(perm[half:])):
        mask = np.isin(pid, list(hold))
        booster = fit_gbm_student(X_tr[~mask], y_tr[~mask],
                                  np.ones(int((~mask).sum())),
                                  X_tr[mask], y_tr[mask], device=device)
        q[mask] = gbm_predict(booster, X_tr[mask])
    assert not np.isnan(q).any()
    return q


# --- The grid ----------------------------------------------------------------

def _cells(objectives, seeds, quick):
    """(family, objective, hyper, seed) cells. GBM cells without a teacher are
    seed-independent (the GBM is seed-fixed) and run once."""
    out = []
    tls = (12,) if quick else TLS_H
    betas = (1.0,) if quick else UTILITY_BETA
    lams = (0.5,) if quick else DISTILL_LAMBDA
    for family in ("gbm", "orch"):
        if "tls" in objectives:
            for H in tls:
                for s in ([None] if family == "gbm" else seeds):
                    out.append((family, "tls", H, s))
        if "utility" in objectives:
            for b in betas:
                for s in ([None] if family == "gbm" else seeds):
                    out.append((family, "utility", b, s))
        if "distill" in objectives:
            for lam in lams:
                for s in seeds:        # teacher (gbm fam) / student (orch fam)
                    out.append((family, "distill", lam, s))
    return out


def pareto_front(points):
    """Indices of the maximal set under (freeze_auprc, freeze_lead)."""
    front = []
    for i, p in enumerate(points):
        dominated = any(
            (o["freeze_auprc"] >= p["freeze_auprc"]
             and o["freeze_lead"] >= p["freeze_lead"]
             and (o["freeze_auprc"] > p["freeze_auprc"]
                  or o["freeze_lead"] > p["freeze_lead"]))
            for j, o in enumerate(points) if j != i)
        if not dominated:
            front.append(i)
    return front


def run(families=("gbm", "orch"), objectives=("tls", "utility", "distill"),
        seeds=(42, 1, 2), reps=2000, device="cuda", quick=False,
        out_cache=OUT_CACHE):
    from orchestrator.fusion import AgentBank, config_agents

    splits = load_splits()
    ctx = FusionContext(splits)
    train_ordered = splits["train"].sort_values(
        ["stay_id", "hour"]).reset_index(drop=True)
    val_fit_frame = ctx.val_ordered[ctx.fit_mask]
    seeds = list(seeds)[:1] if quick else list(seeds)

    # Parents as BCE reference points (no fits).
    fair_val, fair_test = load_fair_scores(ctx.val_ordered, ctx.test_ordered)
    _, orch_scores, _ = load_orch_seeds(ctx.val_ordered, ctx.test_ordered)
    points = [{"family": "gbm", "objective": "bce", "hyper": None,
               "seed": None,
               "freeze_auprc": ctx.freeze_auprc(fair_val),
               "freeze_lead": ctx.freeze_lead(fair_val),
               "scores": (fair_val, fair_test)}]
    for s in seeds:
        points.append({"family": "orch", "objective": "bce", "hyper": None,
                       "seed": s,
                       "freeze_auprc": ctx.freeze_auprc(orch_scores[s]["val"]),
                       "freeze_lead": ctx.freeze_lead(orch_scores[s]["val"]),
                       "scores": (orch_scores[s]["val"],
                                  orch_scores[s]["test"])})

    # GBM design matrices, once (heavy).
    need_gbm = "gbm" in families or ("orch" in families
                                     and "distill" in objectives)
    if need_gbm:
        from baselines_fair import build_design, variant_columns
        from modality_features import load_labs
        print("[distill] building design matrices once...")
        labs = load_labs()
        cols = variant_columns()["FAIR+TRAJ"]
        X_tr = build_design(splits["train"], labs, None)[0][cols]
        X_va = build_design(splits["val"], labs, None)[0][cols]
        X_te = build_design(splits["test"], labs, None)[0][cols]
        y_tr_hard = train_ordered[LABEL].to_numpy(dtype=float)
        X_es = X_va[ctx.fit_mask]
        y_es = ctx.y_val[ctx.fit_mask]

    bank = None
    if "orch" in families:
        bank = AgentBank.load(agent_names=config_agents("core-3+meas"))

    gbm_q = None   # reverse teacher, built lazily once
    for family, objective, hyper, seed in _cells(objectives, seeds, quick):
        if family not in families:
            continue
        tag = f"{family}/{objective}/{hyper}" + (f"/s{seed}" if seed else "")
        t0 = time.time()

        if family == "gbm":
            y, w = y_tr_hard, np.ones(len(y_tr_hard))
            if objective == "tls":
                y = tls_targets(train_ordered, hyper)
            elif objective == "utility":
                w = utility_weights(train_ordered, hyper)
            else:
                sid, hr, q = orch_teacher_oof(seed)
                if not ((sid == train_ordered["stay_id"].to_numpy()).all()
                        and (hr == train_ordered["hour"].to_numpy()).all()):
                    raise ValueError("OOF teacher misaligned with train")
                y = soft_targets(train_ordered, q, hyper)
            booster = fit_gbm_student(X_tr, y, w, X_es, y_es, device=device)
            s_val = gbm_predict(booster, X_va)
            s_test = gbm_predict(booster, X_te)
            extra = {"n_trees": int(booster.best_iteration + 1)}
        else:
            from orchestrator.fusion import Orchestrator  # noqa: F401
            y, w = None, None
            if objective == "tls":
                y = tls_targets(train_ordered, hyper)
            elif objective == "utility":
                y = train_ordered[LABEL].to_numpy(dtype=float)
                w = utility_weights(train_ordered, hyper)
            else:
                if gbm_q is None:
                    print("[distill] fitting the reverse (GBM OOF) teacher...")
                    gbm_q = gbm_teacher_oof(X_tr, y_tr_hard, train_ordered,
                                            device=device)
                y = soft_targets(train_ordered, gbm_q, hyper)
            soft = make_soft_frame(train_ordered, y, w)
            student = DistilledOrchestrator(bank=bank, max_epochs=8,
                                            seed=seed or SEED, verbose=False,
                                            soft_frame=soft)
            student.fit(splits["train"], val_fit_frame)
            s_val = student.raw_risk(splits["val"])
            s_test = student.raw_risk(splits["test"])
            extra = {}

        point = {"family": family, "objective": objective, "hyper": hyper,
                 "seed": seed,
                 "freeze_auprc": ctx.freeze_auprc(s_val),
                 "freeze_lead": ctx.freeze_lead(s_val),
                 "scores": (s_val, s_test), **extra}
        points.append(point)
        print(f"[cell {tag}] freeze AUPRC {point['freeze_auprc']:.4f}, "
              f"freeze lead@2% {point['freeze_lead']:+.1f} "
              f"({time.time() - t0:.0f}s)")

    # Pareto per family on the V_freeze surface; ONLY frontier points touch
    # test (plus the BCE parents, whose test rows exist in prior tables).
    payload = {"stay_id_test": ctx.test_ordered["stay_id"].to_numpy(),
               "hour_test": ctx.test_ordered["hour"].to_numpy()}
    report = []
    for family in ("gbm", "orch"):
        fam_idx = [i for i, p in enumerate(points) if p["family"] == family]
        if not fam_idx:
            continue
        front = {fam_idx[j] for j in
                 pareto_front([points[i] for i in fam_idx])}
        print(f"\n=== {family} grid (V_freeze surface; * = Pareto frontier, "
              f"evaluated on test) ===")
        for i in fam_idx:
            p = points[i]
            row = {k: p[k] for k in ("family", "objective", "hyper", "seed",
                                     "freeze_auprc", "freeze_lead")}
            star = " "
            if i in front:
                star = "*"
                r = ctx.evaluate_rung(p["scores"][0], p["scores"][1],
                                      reps=reps)
                row["test"] = r
                tag = (f"{family}_{p['objective']}_{p['hyper']}"
                       f"_s{p['seed']}").replace(".", "p")
                payload[f"val_{tag}"] = p["scores"][0].astype(np.float32)
                payload[f"test_{tag}"] = p["scores"][1].astype(np.float32)
            report.append(row)
            test_str = ""
            if "test" in row:
                b = row["test"]["budgets"][0.02]
                test_str = (f"  test AUPRC {row['test']['auprc']:.4f}, "
                            f"lead@2% {b['median']:+.1f} (win "
                            f"{100*b['win_rate']:.0f}%)")
            print(f" {star} {p['objective']:<8}{str(p['hyper']):<6}"
                  f"s{str(p['seed']):<5}freeze AUPRC "
                  f"{p['freeze_auprc']:.4f}  lead@2% "
                  f"{p['freeze_lead']:+.1f}{test_str}")

    payload["meta_json"] = np.array(json.dumps(_jsonable(
        {"seeds": seeds, "objectives": list(objectives),
         "families": list(families), "reps": reps, "report": report})))
    Path(out_cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_cache, **payload)
    print(f"\n[cache] -> {out_cache}")
    return report


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="gbm,orch")
    ap.add_argument("--objectives", default="tls,utility,distill")
    ap.add_argument("--seeds", default="42,1,2")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(families=tuple(args.families.split(",")),
        objectives=tuple(args.objectives.split(",")),
        seeds=[int(s) for s in args.seeds.split(",")],
        reps=args.reps, device=args.device, quick=args.quick)
