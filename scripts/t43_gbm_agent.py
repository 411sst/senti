"""T43 preparation: the all-features GBM as Sentinel's GBM agent.

Builds the agent's emissions with the shipped GBM agent's construction
(``scripts/orch_gbm_agent.py``), swapping FAIR+TRAJ (145 columns) for the T30
all-features GBM (262 columns):

* train rows: out of fold, two patient-level halves drawn exactly as the
  shipped agent draws them (``gbm_agent_oof.patient_halves``). Each fold model
  uses T30's winning settings (depth 6, no class weight) with the tuner's
  fixed ones (lr 0.05, min_child_weight 10, subsample and colsample 0.8, up to
  3,000 trees, early stopping after 50 rounds on validation AUPRC), seed 42,
  GPU ``hist``.
* validation and test rows: T30's seed-42 fit on all of train, read from
  ``allfeat_gbm_scores.npz`` (the fit T31 reproduced exactly).
* latent = log-odds of the raw probability; risk = isotonic calibration fit on
  V_fit. The shipped agent fits its calibrator on the full validation split;
  T43 follows the paper-test rule (calibrate on V_fit), a stated difference.

Outputs, all new paths:
* ``data/cache/embeddings_t43/``: ``gbm_{train,val,test}.npz`` plus links to
  the seven unchanged agents' shipped emissions, so ``AgentBank.load`` reads the
  T43 bank from one directory and the shipped ``embeddings/`` is never written.
* ``models/t43/gbm_agent_fold{0,1}.ubj``: the two fold boosters.
* ``data/cache/t43_gbm_agent.json``: aggregates only.

No test-split metric is computed by the ``agent`` stage.

Stages ``heads`` and ``eval`` run the pre-registered test (results_fusion.md,
T43):

* ``heads`` (CPU only, one thread per worker): both heads refit with
  ``two_head.fit_head`` at seeds 42 and 1-9 on the T43 bank (the new GBM agent
  plus the seven shipped agents), and on the GBM agent alone (secondary arm);
  plus one refit on the published bank at seed 42 for gate 1. Raw head scores
  and each head's mean test attention per agent go to ``data/cache/t43/heads/``;
  checkpoints to ``models/t43/heads/``.
* ``eval``: gates, then the measures with T47's shared-resample code
  (``scripts/t47_stack.py``, NEWS2-matched race protocol). Aggregates go to
  ``data/cache/t43_eval.json``.

    python scripts/t43_gbm_agent.py agent [--device cuda]
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python scripts/t43_gbm_agent.py heads --workers 3
    OMP_NUM_THREADS=1 python scripts/t43_gbm_agent.py eval --workers 16
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

SHIPPED_EMB = REPO / "data" / "cache" / "embeddings"
T43_EMB = REPO / "data" / "cache" / "embeddings_t43"
MODELS = REPO / "models" / "t43"
ALLFEAT_CACHE = REPO / "data" / "cache" / "allfeat_gbm_scores.npz"
OUT_JSON = REPO / "data" / "cache" / "t43_gbm_agent.json"
OTHER_AGENTS = ("vitals_tcn", "news2", "labs", "ecg_meas", "cadence", "fluid",
                "resp")
SPLITS = ("train", "val", "test")
HEADS = REPO / "data" / "cache" / "t43" / "heads"
EVAL_JSON = REPO / "data" / "cache" / "t43_eval.json"
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
EPOCHS = 8
ARMS = {"new": "new Sentinel (all-features GBM agent)",
        "gbm_only": "heads on the GBM agent alone (secondary)"}


def _keys_equal(a_stay, a_hour, frame):
    return (np.array_equal(np.asarray(a_stay), frame["stay_id"].to_numpy())
            and np.array_equal(np.asarray(a_hour), frame["hour"].to_numpy()))


def stage_agent(device):
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression

    from agents.base import AgentOutput
    from allfeat_gbm import build_allfeat_design, column_sets, load_sources
    from baselines_mimic import load_splits
    from fusion_common import FusionContext
    from gbm_agent_oof import logit, oof_scores, patient_halves
    from metrics import auprc, auroc

    t0 = time.time()
    z = np.load(ALLFEAT_CACHE, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    rec = meta["records"]["allfeat"]
    if (rec["max_depth"], rec["scale_pos_weight"]) != (6, 1.0):
        raise SystemExit(f"unexpected T30 winner {rec}")

    splits = load_splits()
    ctx = FusionContext(splits)
    train_o = splits["train"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    val_o, test_o = ctx.val_ordered, ctx.test_ordered
    if not (_keys_equal(z["val_stay_id"], z["val_hour"], val_o)
            and _keys_equal(z["test_stay_id"], z["test_hour"], test_o)):
        raise SystemExit("T30 score cache rows do not match the cohort order")
    y_val = val_o["label_24h"].to_numpy(dtype=float)
    raw = {"val": z["rawval_allfeat_s42"].astype(float),
           "test": z["rawtest_allfeat_s42"].astype(float)}
    check = auprc(y_val, raw["val"])
    if abs(check - rec["val_auprc"]) > 1e-9:
        raise SystemExit(f"cached val AUPRC {check} != recorded {rec['val_auprc']}")

    print("[t43] building the 262-column design for train and val...", flush=True)
    sources = load_sources()
    cols = column_sets(sources["ecg"]["names"])["allfeat"]
    X_tr, y_tr, keys_tr = build_allfeat_design(splits["train"], sources)
    X_va, y_va, keys_va = build_allfeat_design(splits["val"], sources)
    if not (_keys_equal(keys_tr["stay_id"], keys_tr["hour"], train_o)
            and _keys_equal(keys_va["stay_id"], keys_va["hour"], val_o)):
        raise SystemExit("design rows do not match the cohort order")
    y_tr = np.asarray(y_tr, dtype=float)
    if not np.array_equal(y_tr, train_o["label_24h"].to_numpy(dtype=float)):
        raise SystemExit("design label is not the cohort's label_24h")
    X_tr, X_va = X_tr[cols], X_va[cols]
    d_val = xgb.DMatrix(X_va, label=y_val)
    params = {"objective": "binary:logistic", "eval_metric": "aucpr",
              "eta": 0.05, "max_depth": rec["max_depth"],
              "min_child_weight": 10, "subsample": 0.8,
              "colsample_bytree": 0.8,
              "scale_pos_weight": rec["scale_pos_weight"],
              "seed": 42, "tree_method": "hist", "device": device}
    fold_info = []

    def fit(Xr, yr):
        t = time.time()
        b = xgb.train(params, xgb.DMatrix(Xr, label=yr), num_boost_round=3000,
                      evals=[(d_val, "val")], early_stopping_rounds=50,
                      verbose_eval=False)
        fold_info.append({"rows": int(len(yr)), "trees": int(b.best_iteration + 1),
                          "val_auprc_es": float(b.best_score),
                          "seconds": round(time.time() - t, 1)})
        print(f"[t43] fold fit: {len(yr):,} rows, {b.best_iteration + 1} trees, "
              f"val AUPRC {b.best_score:.4f} ({time.time() - t:.0f}s)", flush=True)
        return b

    def predict(b, Xr):
        return b.predict(xgb.DMatrix(Xr),
                         iteration_range=(0, b.best_iteration + 1)).astype(float)

    pid = train_o["patient_id"].to_numpy()
    raw["train"], boosters = oof_scores(X_tr, y_tr, pid, fit, predict)
    halves = patient_halves(pid)

    iso = IsotonicRegression(out_of_bounds="clip").fit(
        raw["val"][ctx.fit_mask], y_val[ctx.fit_mask])
    T43_EMB.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    frames = {"train": train_o, "val": val_o, "test": test_o}
    for split in SPLITS:
        out_path = T43_EMB / f"gbm_{split}.npz"
        if out_path.exists():
            raise SystemExit(f"{out_path} exists; new outputs go to new files")
        AgentOutput(keys=frames[split][["stay_id", "hour"]].copy(),
                    latent=logit(raw[split]).astype(np.float32)[:, None],
                    risk=iso.transform(raw[split]).astype(np.float32)
                    ).save(out_path)
    for name in OTHER_AGENTS:
        for split in SPLITS:
            link = T43_EMB / f"{name}_{split}.npz"
            if not link.exists():
                link.symlink_to(SHIPPED_EMB / f"{name}_{split}.npz")
    for k, b in enumerate(boosters):
        b.save_model(MODELS / f"gbm_agent_fold{k}.ubj")

    shipped_tr = np.load(SHIPPED_EMB / "gbm_train.npz")
    y_shipped = train_o["label_24h"].to_numpy(dtype=float)
    summary = {
        "stage": "agent",
        "run_seconds": round(time.time() - t0, 1),
        "columns": len(cols),
        "columns_sha256": hashlib.sha256("|".join(cols).encode()).hexdigest(),
        "t30_winner": rec,
        "folds": fold_info,
        "fold_patients": [len(h) for h in halves],
        "train_rows": int(len(train_o)),
        "train_oof_auprc": float(auprc(y_tr, raw["train"])),
        "train_oof_auroc": float(auroc(y_tr, raw["train"])),
        "shipped_agent_train_oof_auprc": float(
            auprc(y_shipped, shipped_tr["risk"].astype(float))),
        "val_auprc_cached_fit": float(check),
        "calibrator": "isotonic on V_fit",
        "xgboost": xgb.__version__,
        "device": device,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# --- heads ------------------------------------------------------------------------

def _agents(arm):
    from feed_outage import ALL8
    return ("gbm",) if arm == "gbm_only" else ALL8


def _fit_job(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    arm, seed = job
    path = HEADS / f"{arm}_s{seed}.npz"
    if path.exists():
        return job, "cached", 0.0
    t0 = time.time()
    import pandas as pd  # noqa: F401
    from baselines_mimic import load_splits
    from decompose import treatment_only_label
    from fusion_common import FusionContext
    from orchestrator.fusion import AgentBank
    from two_head import fit_head
    splits = load_splits()
    ctx = FusionContext(splits)
    train_o = splits["train"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    val_o = ctx.val_ordered
    labels = {"T": (treatment_only_label(train_o), treatment_only_label(val_o)),
              "D": (train_o["label_24h_death"].to_numpy(dtype=float),
                    val_o["label_24h_death"].to_numpy(dtype=float))}
    bank = AgentBank.load(agent_names=_agents(arm),
                          emb_dir=SHIPPED_EMB if arm == "published" else T43_EMB)
    payload = {"agents": np.array(list(bank.agent_names))}
    for h, (y_tr, y_va) in labels.items():
        head = fit_head(bank, splits, y_tr, y_va, train_o, val_o, seed, EPOCHS)
        payload[f"rawval_{h}"] = head.raw_risk(splits["val"])
        payload[f"rawtest_{h}"] = head.raw_risk(splits["test"])
        if arm != "published":
            _, w = head.attention_weights(splits["test"])
            payload[f"attn_{h}"] = np.asarray(w, dtype=float).mean(axis=0)
            (MODELS / "heads").mkdir(parents=True, exist_ok=True)
            head.save_checkpoint(MODELS / "heads" / f"{arm}_{h}_s{seed}.pt")
    HEADS.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)
    return job, "fitted", round(time.time() - t0, 1)


def stage_heads(workers):
    import multiprocessing as mp
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("run heads with CUDA_VISIBLE_DEVICES= (CPU only)")
    jobs = [("published", 42)] + [(a, s) for a in ARMS for s in SEEDS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for job, status, secs in pool.imap_unordered(_fit_job, jobs):
            print(f"[{job[0]} s{job[1]}] {status} {secs} s ({time.time() - t0:.0f} s elapsed)",
                  flush=True)


# --- eval -------------------------------------------------------------------------

def _composite(ctx, z):
    """(val, test) composite: per-head isotonic on V_fit against its own label,
    rank tiebreak, noisy-OR (as shipped and as T42)."""
    from decompose import treatment_only_label
    from feed_outage import calibrated, noisy_or
    from sklearn.isotonic import IsotonicRegression
    y = {"T": treatment_only_label(ctx.val_ordered),
         "D": ctx.val_ordered["label_24h_death"].to_numpy(dtype=float)}
    iso = {h: IsotonicRegression(out_of_bounds="clip").fit(
        z[f"rawval_{h}"][ctx.fit_mask], y[h][ctx.fit_mask]) for h in ("T", "D")}
    return tuple(noisy_or(calibrated(iso["T"], z[f"raw{sp}_T"]), calibrated(iso["D"], z[f"raw{sp}_D"]))
                 for sp in ("val", "test"))


def _gates(ctx):
    from agents.base import AgentOutput
    from gbm_agent_oof import logit
    out = {}
    pub = np.load(REPO / "data" / "cache" / "two_head.npz")
    val, test = _composite(ctx, np.load(HEADS / "published_s42.npz"))
    out["head_code_reproduces_two_head_s42"] = bool(
        np.array_equal(val, pub["val_s42"]) and np.array_equal(test, pub["test_s42"]))
    same = []
    for name in OTHER_AGENTS:
        for split in SPLITS:
            a, b = T43_EMB / f"{name}_{split}.npz", SHIPPED_EMB / f"{name}_{split}.npz"
            same.append(hashlib.sha256(a.read_bytes()).digest()
                        == hashlib.sha256(b.read_bytes()).digest())
    out["seven_agents_byte_identical"] = bool(all(same))
    z = np.load(ALLFEAT_CACHE, allow_pickle=True)
    ok = True
    for split, key in (("val", "rawval_allfeat_s42"), ("test", "rawtest_allfeat_s42")):
        o = AgentOutput.load(T43_EMB / f"gbm_{split}.npz")
        ok &= bool(np.array_equal(np.asarray(o.latent, np.float32)[:, 0],
                                  logit(z[key].astype(float)).astype(np.float32)))
    out["new_agent_logodds_equal_t30_s42"] = ok
    print(f"[gates] {out}", flush=True)
    if not all(out.values()):
        raise SystemExit("GATE FAILED")
    return out


def _arm_measures(ctx, arm, G, mode, test):
    """T47's measures for one ten-seed arm ``arm`` {seed: (val, test)}."""
    from demo_bundle import first_alarm_table
    from paper_operating_points import SWEEP_BUDGETS, catch_stats, freeze_quantile, test_rate
    from t35_eicu_operating_points import (catch_at_quiet_share, quiet_alarm_share,
                                           stay_max_scores)
    from t47_stack import RACE_BUDGETS, pooled_race, test_race
    out = {"news2_race": {}, "gbm_race": {}}
    for b in RACE_BUDGETS:
        per = [test_race(ctx, arm[s][0], arm[s][1], b, mode)[0] for s in SEEDS]
        out["news2_race"][str(b)] = pooled_race(ctx, per)
        if G is not None:
            per = [test_race(ctx, arm[s][0], arm[s][1], b, mode, opp=G[s])[0] for s in SEEDS]
            out["gbm_race"][str(b)] = pooled_race(ctx, per)
    sweep = {}
    for b in SWEEP_BUDGETS:
        cs = []
        for s in SEEDS:
            thr = freeze_quantile(ctx, arm[s][0], b)
            st = catch_stats(first_alarm_table(test, {"m": arm[s][1]}, {"m": thr}), "m")
            cs.append({"catch": st["catch"], "catch_6h": st["catch_6h"],
                       "quiet_alarmed": st["quiet_fp"],
                       "test_alarm_rate": test_rate(ctx, arm[s][1], thr)})
        sweep[str(b)] = {m: float(np.mean([c[m] for c in cs])) for m in cs[0]}
    out["catch_sweep"] = sweep
    if G is not None:
        quiet = {}
        for b in SWEEP_BUDGETS:
            own, gb = [], []
            for s in SEEDS:
                q, _ = quiet_alarm_share(test, G[s][1], freeze_quantile(ctx, G[s][0], b))
                own.append(catch_at_quiet_share(*stay_max_scores(test, arm[s][1]), q))
                gb.append(catch_at_quiet_share(*stay_max_scores(test, G[s][1]), q))
            quiet[str(b)] = {"model_catch_mean": float(np.mean(own)),
                             "gbm_catch_mean": float(np.mean(gb)),
                             "seeds_ahead": int(sum(o > g for o, g in zip(own, gb)))}
        out["catch_at_gbm_quiet_share"] = quiet
    return out


def stage_eval(workers):
    from demo_bundle import assert_aggregate_only
    from feed_outage import ALL8
    from metrics import auprc, auroc
    from paper_stats import metric_draws, pooled_delta
    from t47_stack import LEAD_FLOOR, REPS, _context, load_parents
    mode = "matched"
    t0 = time.time()
    ctx = _context()
    gates = _gates(ctx)
    shipped, G = load_parents(ctx)
    test, y = ctx.test_ordered, ctx.y_test
    arms, attn = {"shipped": shipped}, {}
    for a in ARMS:
        arms[a] = {}
        for s in SEEDS:
            z = np.load(HEADS / f"{a}_s{s}.npz")
            arms[a][s] = _composite(ctx, z)
            if a == "new":
                names = [str(n) for n in z["agents"]]
                if names != list(ALL8):
                    raise SystemExit(f"unexpected bank order {names}")
                attn.setdefault("T", []).append(z["attn_T"][names.index("gbm")])
                attn.setdefault("D", []).append(z["attn_D"][names.index("gbm")])
    arms["gbm"] = G
    scores = {f"{a}_s{s}": v[s][1] for a, v in arms.items() for s in SEEDS}
    pts = {m: {n: float(fn(y, v)) for n, v in scores.items()}
           for m, fn in (("auprc", auprc), ("auroc", auroc))}
    t1 = time.time()
    draws = metric_draws({"auprc": ("auprc", y), "auroc": ("auroc", y)}, scores,
                         ctx.patient_test, reps=REPS, workers=workers)
    if any(len(d) != REPS for t in draws.values() for d in t.values()):
        raise SystemExit("a resample was skipped")
    print(f"[bootstrap] {len(scores)} scores x {REPS} in {time.time() - t1:.0f} s", flush=True)

    def ranking(a):
        out = {}
        for m in ("auprc", "auroc"):
            vals = [pts[m][f"{a}_s{s}"] for s in SEEDS]
            dr = np.mean([draws[m][f"{a}_s{s}"] for s in SEEDS], axis=0)
            out[m] = {"mean": float(np.mean(vals)), "min": float(min(vals)), "max": float(max(vals)),
                      "ci": [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))]}
        return out

    def delta(a, b):
        out = {}
        for m in ("auprc", "auroc"):
            d = pooled_delta([pts[m][f"{a}_s{s}"] for s in SEEDS], [pts[m][f"{b}_s{s}"] for s in SEEDS],
                             [draws[m][f"{a}_s{s}"] for s in SEEDS], [draws[m][f"{b}_s{s}"] for s in SEEDS])
            d["seeds_positive"] = int(sum(pts[m][f"{a}_s{s}"] > pts[m][f"{b}_s{s}"] for s in SEEDS))
            out[m] = d
        return out

    names = {"new": ARMS["new"], "gbm_only": ARMS["gbm_only"],
             "shipped": "shipped Sentinel (reference)", "gbm": "all-features GBM (T31 refits)"}
    res = {"test": "T43 all-features GBM as Sentinel's GBM agent", "race_thresholds": mode,
           "seeds": list(SEEDS), "reps": REPS, "gates": gates, "rows": int(len(y)),
           "positives": int(y.sum()), "patients": int(len(np.unique(ctx.patient_test))),
           "gbm_agent_attention_share": {h: {"mean": float(np.mean(v)), "min": float(min(v)),
                                              "max": float(max(v))} for h, v in attn.items()},
           "arms": {}}
    for a in ("new", "gbm_only", "shipped", "gbm"):
        arm = {"name": names[a], "ranking": ranking(a)}
        if a != "gbm":
            arm["vs_gbm"] = delta(a, "gbm")
        if a == "new":
            arm["vs_shipped"] = delta("new", "shipped")
        arm.update(_arm_measures(ctx, arms[a], None if a == "gbm" else G, mode, test))
        res["arms"][a] = arm
        r = arm["ranking"]["auprc"]
        r2 = arm["news2_race"]["0.02"]
        print(f"[{a}] AUPRC {r['mean']:.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]"
              + (f" | dAUPRC vs GBM {arm['vs_gbm']['auprc']['delta']:+.4f} {arm['vs_gbm']['auprc']['ci']}"
                 if a != "gbm" else "")
              + f" | lead@2% {r2['lead_h']:+.2f} h, share-first {r2['share_first']:.3f} {r2['share_first_ci']}",
              flush=True)
    new = res["arms"]["new"]
    r2 = new["news2_race"]["0.02"]
    res["verdicts"] = {
        "ranking_pass": bool(new["vs_gbm"]["auprc"]["ci_above_zero"]),
        "timing_pass": bool(r2["lead_h"] >= LEAD_FLOOR and r2["share_first_ci"][0] > 0.5)}
    res["note"] = ("aggregate-only; the new GBM agent reads every feed; catches at the GBM's "
                   "quiet-stay share are a test-score diagnostic")
    res["seconds"] = round(time.time() - t0, 1)
    res = json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    assert_aggregate_only(res)
    EVAL_JSON.write_text(json.dumps(res, indent=1))
    print(f"[verdict] ranking {'PASS' if res['verdicts']['ranking_pass'] else 'FAIL'}; timing "
          f"{'PASS' if res['verdicts']['timing_pass'] else 'FAIL'}")
    print(f"[saved] {EVAL_JSON.relative_to(REPO)} ({res['seconds']} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["agent", "heads", "eval"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if args.stage == "agent":
        stage_agent(args.device)
    elif args.stage == "heads":
        stage_heads(args.workers)
    else:
        stage_eval(args.workers)


if __name__ == "__main__":
    main()
