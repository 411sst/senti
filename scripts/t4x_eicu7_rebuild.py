"""T44/T45 preparation: rebuild T22's zero-shot seven-agent model WITH weights.

T22 (``scripts/score_eicu.py --two-head --label all7``) fitted seven agents on
MIMIC (vitals_tcn, news2, labs, ecg_meas, gbm, fluid, resp; 8 epochs, seed 42)
and its heads at seeds 42 and 1-9, but saved only emissions and scores. T44
scores that model's death head on eICU rows T22 never built, so the model is
refitted here with T22's own code (``score_eicu.build_agent``, the
``build_gbm_emissions`` recipe, ``two_head.fit_head``) and saved.

Nothing touches eICU or the MIMIC test split. Gates use MIMIC train and val:

* news2, labs, ecg_meas (CPU fits in T22) and the GBM agent (deterministic
  GPU XGBoost) must reproduce T22's cached emissions exactly;
* vitals_tcn, fluid and resp were GPU fits that do not reproduce (T36); they
  are refitted on the CPU (deterministic from now on) and must land within
  ``TOL`` validation AUPRC of T22's cached emissions;
* the death head at seed 42 fitted on the rebuilt bank must land within
  ``TOL`` validation death-AUPRC of the same head fitted on T22's cached bank.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python scripts/t4x_eicu7_rebuild.py fit-agents
    OMP_NUM_THREADS=1 python scripts/t4x_eicu7_rebuild.py fit-gbm
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python scripts/t4x_eicu7_rebuild.py fit-heads --workers 2

Weights go to ``models/t4x/eicu7/``; emissions and head raw scores to
``data/cache/t4x/eicu7/``; aggregates to ``data/cache/t4x/eicu7_gate.json``.
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

CACHE = REPO / "data" / "cache"
T22_EMB = CACHE / "embeddings_eicu"
WORK = CACHE / "t4x" / "eicu7"
EMB = WORK / "embeddings"
HEADS = WORK / "heads"
MODELS = REPO / "models" / "t4x" / "eicu7"
GATE = CACHE / "t4x" / "eicu7_gate.json"
AGENTS = ("vitals_tcn", "news2", "labs", "ecg_meas", "gbm", "fluid", "resp")   # T22 order
EXACT = ("news2", "labs", "ecg_meas")
EPOCHS = 8
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
TOL = 0.010
SPLITS = ("train", "val")


def _gate_update(section, payload):
    g = json.loads(GATE.read_text()) if GATE.exists() else {}
    g[section] = payload
    GATE.parent.mkdir(parents=True, exist_ok=True)
    GATE.write_text(json.dumps(g, indent=2))


def _one_thread():
    import torch
    torch.set_num_threads(1)


def _same(a, b):
    return bool(a.keys.reset_index(drop=True).astype("int64").equals(
        b.keys.reset_index(drop=True).astype("int64"))
        and np.array_equal(a.risk, b.risk) and np.array_equal(a.latent, b.latent))


def _strip(agent):
    for attr in ("labs", "source", "_cache"):
        if hasattr(agent, attr):
            setattr(agent, attr, None)
    return agent


def stage_fit_agents():
    import torch

    from agents.base import AgentOutput
    from baselines_mimic import load_splits
    from metrics import auprc
    from score_eicu import build_agent
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("run fit-agents with CUDA_VISIBLE_DEVICES= (CPU only)")
    _one_thread()
    splits = load_splits()
    EMB.mkdir(parents=True, exist_ok=True)
    (MODELS / "agents").mkdir(parents=True, exist_ok=True)
    y_val = splits["val"].sort_values(["stay_id", "hour"])["label_24h"].to_numpy(float)
    report = {}
    for name in [a for a in AGENTS if a != "gbm"]:
        wpath = MODELS / "agents" / f"{name}.pt"
        if wpath.exists():
            raise SystemExit(f"refusing to overwrite {wpath}")
        t0 = time.time()
        agent = build_agent(name, EPOCHS, seed=42)
        agent.fit(splits["train"], splits["val"])
        rec = {"fit_seconds": round(time.time() - t0, 1)}
        for sp in SPLITS:
            out = agent.emit(splits[sp])
            ref = AgentOutput.load(T22_EMB / f"{name}_{sp}.npz")
            rec[f"{sp}_identical_to_t22"] = _same(out, ref)
            if sp == "val":
                rec["val_auprc_rebuilt"] = float(auprc(y_val, out.risk))
                rec["val_auprc_t22"] = float(auprc(y_val, ref.risk))
                rec["val_risk_corr"] = float(np.corrcoef(out.risk, ref.risk)[0, 1])
            out.save(EMB / f"{name}_{sp}.npz")
        rec["passed"] = (all(rec[f"{s}_identical_to_t22"] for s in SPLITS) if name in EXACT
                         else abs(rec["val_auprc_rebuilt"] - rec["val_auprc_t22"]) <= TOL)
        torch.save(_strip(agent), wpath)
        report[name] = rec
        print(f"[{name}] {rec}", flush=True)
    _gate_update("agents", report)
    bad = [n for n, r in report.items() if not r["passed"]]
    if bad:
        raise SystemExit(f"GATE: {bad} do not match T22")


def stage_fit_gbm():
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression

    from agents.base import AgentOutput
    from baselines_fair import build_design, variant_columns
    from baselines_mimic import load_splits
    from distill_lead import fit_gbm_student, gbm_predict
    from gbm_agent_oof import patient_halves
    from modality_features import load_labs

    (MODELS / "gbm").mkdir(parents=True, exist_ok=True)
    paths = {k: MODELS / "gbm" / f"{k}.ubj" for k in ("full", "fold0", "fold1")}
    if any(p.exists() for p in paths.values()):
        raise SystemExit("refusing to overwrite the eICU-7 GBM boosters")
    splits = load_splits()
    cols = variant_columns()["FAIR+TRAJ"]
    labs = load_labs()
    ordered = {k: splits[k].sort_values(["stay_id", "hour"]).reset_index(drop=True)
               for k in SPLITS}
    y = {k: ordered[k]["label_24h"].to_numpy(float) for k in SPLITS}
    X = {k: build_design(splits[k], labs, None)[0][cols] for k in SPLITS}
    full = fit_gbm_student(X["train"], y["train"], np.ones(len(y["train"])),
                           X["val"], y["val"], device="cuda", seed=42)
    pid = ordered["train"]["patient_id"].to_numpy()
    oof = np.full(len(pid), np.nan)
    folds = []
    for held in patient_halves(pid):
        m = np.isin(pid, list(held))
        b = fit_gbm_student(X["train"][~m], y["train"][~m], np.ones(int((~m).sum())),
                            X["val"], y["val"], device="cuda", seed=42)
        oof[m] = gbm_predict(b, X["train"][m])
        folds.append(b)
    raw = {"train": oof, "val": gbm_predict(full, X["val"])}
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw["val"], y["val"])
    rec = {"full_best_iteration": int(full.best_iteration),
           "fold_best_iterations": [int(b.best_iteration) for b in folds]}
    for sp in SPLITS:
        r = np.clip(raw[sp], 1e-6, 1 - 1e-6)
        out = AgentOutput(keys=ordered[sp][["stay_id", "hour"]].copy(),
                          latent=np.log(r / (1 - r)).astype(np.float32)[:, None],
                          risk=iso.transform(raw[sp]).astype(np.float32))
        ref = AgentOutput.load(T22_EMB / f"gbm_{sp}.npz")
        rec[f"{sp}_identical_to_t22"] = _same(out, ref)
        rec[f"{sp}_latent_max_abs_diff"] = float(np.abs(out.latent - ref.latent).max())
        rec[f"{sp}_risk_max_abs_diff"] = float(np.abs(out.risk - ref.risk).max())
        out.save(EMB / f"gbm_{sp}.npz")
    rec["passed"] = all(rec[f"{s}_identical_to_t22"] for s in SPLITS)
    print(f"[gbm] {rec}", flush=True)
    if rec["passed"]:
        full.save_model(paths["full"])
        for b, k in zip(folds, ("fold0", "fold1")):
            b.save_model(paths[k])
        with open(MODELS / "gbm" / "iso.pkl", "wb") as f:
            pickle.dump(iso, f)
        # reloaded (GPU) boosters give the same val scores
        b = xgb.Booster()
        b.load_model(paths["full"])
        b.set_param({"device": "cuda"})
        rec["reloaded_identical"] = bool(np.array_equal(gbm_predict(b, X["val"]), raw["val"]))
        rec["passed"] = rec["reloaded_identical"]
    _gate_update("gbm", rec)
    if not rec["passed"]:
        raise SystemExit("GATE: the GBM agent does not reproduce T22")


def _fit_head_job(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    _one_thread()
    bank_name, seed = job
    path = HEADS / f"{bank_name}_D_s{seed}.npz"
    if path.exists():
        return job, "cached", 0.0
    t0 = time.time()
    from baselines_mimic import load_splits
    from fusion_common import FusionContext
    from orchestrator.fusion import AgentBank
    from two_head import fit_head
    splits = load_splits()
    ctx = FusionContext(splits)
    train_o = splits["train"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    val_o = ctx.val_ordered
    y_tr = train_o["label_24h_death"].to_numpy(dtype=float)
    y_va = val_o["label_24h_death"].to_numpy(dtype=float)
    bank = AgentBank.load(splits=SPLITS, agent_names=AGENTS,
                          emb_dir=EMB if bank_name == "rebuilt" else T22_EMB)
    head = fit_head(bank, splits, y_tr, y_va, train_o, val_o, seed, EPOCHS)
    if bank_name == "rebuilt":
        head.save_checkpoint(MODELS / "heads" / f"D_s{seed}.pt")
    np.savez_compressed(path, rawval_D=head.raw_risk(splits["val"]))
    return job, "fitted", round(time.time() - t0, 1)


def stage_fit_heads(workers, despite_agent_gate=False):
    from baselines_mimic import load_splits
    from fusion_common import FusionContext
    from metrics import auprc
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("run fit-heads with CUDA_VISIBLE_DEVICES= (CPU only)")
    g = json.loads(GATE.read_text())
    failed = [n for n, r in g["agents"].items() if not r["passed"]]
    if not g["gbm"]["passed"]:
        raise SystemExit("GBM gate not passed")
    if failed and not despite_agent_gate:
        raise SystemExit(f"agent gate not passed for {failed}")
    if failed:   # recorded, never silent: the user decides whether the rebuild stands
        _gate_update("heads_fitted_despite_agent_gate", failed)
    HEADS.mkdir(parents=True, exist_ok=True)
    (MODELS / "heads").mkdir(parents=True, exist_ok=True)
    jobs = [("t22bank", 42)] + [("rebuilt", s) for s in SEEDS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for job, status, secs in pool.imap_unordered(_fit_head_job, jobs):
            print(f"[{job[0]} s{job[1]}] {status} {secs} s ({time.time() - t0:.0f} s)", flush=True)
    ctx = FusionContext(load_splits())
    y = ctx.val_ordered["label_24h_death"].to_numpy(dtype=float)
    val = {f"{b}_s{s}": float(auprc(y, np.load(HEADS / f"{b}_D_s{s}.npz")["rawval_D"]))
           for b, s in jobs}
    rec = {"val_death_auprc": val,
           "rebuilt_ten_seed_mean": float(np.mean([val[f"rebuilt_s{s}"] for s in SEEDS])),
           "seed42_diff_rebuilt_minus_t22bank": val["rebuilt_s42"] - val["t22bank_s42"]}
    rec["passed"] = abs(rec["seed42_diff_rebuilt_minus_t22bank"]) <= TOL
    _gate_update("heads", rec)
    print(json.dumps(rec, indent=2))
    if not rec["passed"]:
        raise SystemExit("GATE: rebuilt death head differs from the T22-bank head")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit-agents", "fit-gbm", "fit-heads"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--despite-agent-gate", action="store_true",
                    help="fit heads although an agent missed its gate (recorded in the gate file)")
    a = ap.parse_args()
    if a.stage == "fit-agents":
        stage_fit_agents()
    elif a.stage == "fit-gbm":
        stage_fit_gbm()
    else:
        stage_fit_heads(a.workers, a.despite_agent_gate)


if __name__ == "__main__":
    main()
