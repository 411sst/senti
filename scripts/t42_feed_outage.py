"""T42: does Sentinel hold up better than the all-features GBM when a data
feed fails? (results_fusion.md, T42, pre-registered.)

An outage deletes one feed's raw records for test stays before any feature is
built (``src/feed_outage.py``); each system then rebuilds its inputs with its
own unchanged code and is scored with its models and thresholds unchanged.

Stages, in order. Aggregates go to ``data/cache/t42_feed_outage.json``;
working files go to ``data/cache/t42/`` and ``models/t42/`` (both new):

    # agents, CPU only and one thread (deterministic; T36's path)
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python scripts/t42_feed_outage.py --stage fit-agents
    # GBMs on the GPU, one OpenMP thread (as T30 and T31)
    OMP_NUM_THREADS=1 python scripts/t42_feed_outage.py --stage fit-gbm
    # fusion heads, CPU, one thread per worker
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python scripts/t42_feed_outage.py --stage fit-heads --workers 3
    # outages: agents and heads on the CPU, GBMs on the GPU
    OMP_NUM_THREADS=1 python scripts/t42_feed_outage.py --stage score --workers 3
    # metrics, CIs and verdicts
    OMP_NUM_THREADS=1 python scripts/t42_feed_outage.py --stage eval --workers 8

Why the agents are refit: training kept only their outputs, never weights.
NEWS2, labs and ecg_meas (CPU fits) reproduce the published outputs exactly,
and so does the GBM agent's booster (GPU XGBoost); the run stops otherwise.
vitals_tcn, cadence, fluid and resp were GPU fits that do not reproduce
(T36) and are refit on the CPU. Both heads are refit on that bank at every
seed. Everything is trained once, before any outage, and then frozen.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
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
PUB_EMB = CACHE / "embeddings"
WORK = CACHE / "t42"
EMB = WORK / "embeddings"
HEADS = WORK / "heads"
SCORES = WORK / "scores"
GBM_NPZ = WORK / "gbm.npz"
AGENT_GATE = WORK / "agents_gate.json"
MODELS = REPO / "models" / "t42"
OUT = CACHE / "t42_feed_outage.json"

SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
FB_SEEDS = (42, 1, 2)
EPOCHS = 8
BUDGET = 0.02
REPS = 1000
PUBLISHED_AUPRC = 0.3087            # flagship ten-seed mean (T30)
QUALITY_TOL = 0.010                 # rebuilt ten-seed mean must sit within this
SCENARIOS = ("A", "B")
CODE = ["src/allfeat_gbm.py", "src/baselines_fair.py",
        "src/modality_features.py", "src/mimic_cohort.py", "src/agents/base.py",
        "src/agents/vitals_tcn.py", "src/agents/labs_agent.py",
        "src/agents/news2_agent.py", "src/agents/ecg_agent.py",
        "src/agents/ecg_meas_agent.py", "src/agents/tabular_agent.py",
        "src/orchestrator/fusion.py", "src/distill_lead.py",
        "src/fusion_ladder.py", "scripts/two_head.py"]


# --- bookkeeping -------------------------------------------------------------

def fingerprint(stage="fit"):
    """Hash of the inputs, the model code, and library versions. ``"score"``
    adds the outage code, so outage scores are rebuilt when it changes while
    fitted models are not."""
    import torch
    import xgboost

    from agents.ecg_agent import ECG_INDEX
    from agents.ecg_meas_agent import MEAS_CACHE
    from baselines_mimic import OUT_PATH
    from mimic_cohort import RAW_PATH
    from modality_features import CADENCE_PATH, FLUIDS_PATH, LABS_PATH, RESP_PATH
    files = []
    for p in (OUT_PATH, RAW_PATH, LABS_PATH, CADENCE_PATH, FLUIDS_PATH,
              RESP_PATH, MEAS_CACHE, ECG_INDEX):
        st = Path(p).stat()
        files.append([str(p), st.st_size, st.st_mtime_ns])
    spec = {"files": files, "torch": torch.__version__,
            "xgboost": xgboost.__version__,
            "code": {c: hashlib.sha256(Path(c).read_bytes()).hexdigest()
                     for c in CODE + (["src/feed_outage.py"] if stage == "score" else [])}}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def _save_npz(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def _check_fp(path, fp):
    z = np.load(path, allow_pickle=False)
    if str(z["fingerprint"]) != fp:
        raise SystemExit(f"{path.name} was made from different inputs or code; "
                         f"move it aside to rebuild it")
    return z


def _one_thread():
    import torch
    torch.set_num_threads(1)


def _same_output(a, b):
    """Two AgentOutputs: identical keys, latents and risks?"""
    keys = (np.array_equal(a.keys["stay_id"].to_numpy(dtype="int64"),
                           b.keys["stay_id"].to_numpy(dtype="int64"))
            and np.array_equal(a.keys["hour"].to_numpy(dtype="int64"),
                               b.keys["hour"].to_numpy(dtype="int64")))
    lat = np.array_equal(np.asarray(a.latent, np.float32), np.asarray(b.latent, np.float32))
    risk = np.array_equal(np.asarray(a.risk, np.float32), np.asarray(b.risk, np.float32))
    return bool(keys and lat and risk)


def _max_diff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _sorted(df):
    return df.sort_values(["stay_id", "hour"]).reset_index(drop=True)


# --- stage: fit-agents (CPU) ----------------------------------------------------

def _new_agents():
    import torch

    from agents.ecg_meas_agent import EcgMeasAgent
    from agents.labs_agent import LabsAgent
    from agents.news2_agent import News2Agent
    from agents.tabular_agent import CadenceAgent, FluidAgent, RespAgent
    from agents.vitals_tcn import VitalsTCNAgent
    cpu = torch.device("cpu")
    # The training scripts' settings: results_table.py (vitals_tcn, labs,
    # NEWS2), train_ecg_meas_agent.py, train_new_agents.py (12 epochs).
    return {
        "news2": News2Agent(),
        "labs": LabsAgent(max_epochs=12, verbose=False),
        "ecg_meas": EcgMeasAgent(verbose=False),
        "vitals_tcn": VitalsTCNAgent(max_epochs=EPOCHS, verbose=False, device=cpu),
        "cadence": CadenceAgent(max_epochs=12, verbose=False, device="cpu"),
        "fluid": FluidAgent(max_epochs=12, verbose=False, device="cpu"),
        "resp": RespAgent(max_epochs=12, verbose=False, device="cpu"),
    }


def _strip(agent):
    """Drop the data an agent keeps around, so its pickle is weights only."""
    for attr in ("labs", "source", "_cache"):
        if hasattr(agent, attr):
            setattr(agent, attr, None)
    return agent


def stage_fit_agents():
    import torch

    from agents.base import AgentOutput
    from baselines_mimic import load_splits
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("run fit-agents with CUDA_VISIBLE_DEVICES= (CPU only)")
    _one_thread()
    fp = fingerprint()
    if AGENT_GATE.exists():
        meta = json.loads(AGENT_GATE.read_text())
        if meta.get("fingerprint") == fp and meta.get("done"):
            print("[agents] already fitted for this fingerprint")
            return
    splits = load_splits()
    (MODELS / "agents").mkdir(parents=True, exist_ok=True)
    EMB.mkdir(parents=True, exist_ok=True)
    gate = {"fingerprint": fp, "agents": {}}
    for name, agent in _new_agents().items():
        t0 = time.time()
        agent.fit(splits["train"], splits["val"])
        rec = {"fit_seconds": round(time.time() - t0, 1)}
        for split in ("train", "val", "test"):
            out = agent.emit(splits[split])
            pub = AgentOutput.load(PUB_EMB / f"{name}_{split}.npz")
            rec[f"{split}_identical_to_published"] = _same_output(out, pub)
            rec[f"{split}_max_abs_risk_diff"] = _max_diff(out.risk, pub.risk)
            out.save(EMB / f"{name}_{split}.npz")
        torch.save(_strip(agent), MODELS / "agents" / f"{name}.pt")
        gate["agents"][name] = rec
        print(f"[{name}] {rec}", flush=True)
        must = name in ("news2", "labs", "ecg_meas")
        if must and not all(rec[f"{s}_identical_to_published"]
                            for s in ("train", "val", "test")):
            AGENT_GATE.write_text(json.dumps(gate, indent=2))
            raise SystemExit(f"GATE: the {name} refit does not reproduce the "
                             f"published outputs; stopping")
    # The GBM agent's published outputs are kept as they are; its booster is
    # refit and checked against them in fit-gbm.
    for split in ("train", "val", "test"):
        AgentOutput.load(PUB_EMB / f"gbm_{split}.npz").save(EMB / f"gbm_{split}.npz")
    gate["done"] = True
    AGENT_GATE.write_text(json.dumps(gate, indent=2))
    print(f"[agents] -> {EMB}")


# --- stage: fit-gbm (GPU) -----------------------------------------------------

def _gbm_agent_output(raw, keys, iso):
    """The GBM agent's emission from FAIR+TRAJ raw scores
    (``scripts/orch_gbm_agent.build_gbm_agent_cache``)."""
    from agents.base import AgentOutput
    r = np.asarray(raw, dtype=float)
    c = np.clip(r, 1e-6, 1 - 1e-6)
    logit = np.log(c / (1 - c))
    return AgentOutput(keys=keys[["stay_id", "hour"]].reset_index(drop=True).astype("int64"),
                       latent=logit.astype(np.float32)[:, None],
                       risk=iso.transform(r).astype(np.float32))


def _gbm_agent_iso(ctx):
    from baselines_fair import SCORE_CACHE
    from sklearn.isotonic import IsotonicRegression
    z = np.load(SCORE_CACHE, allow_pickle=True)
    return IsotonicRegression(out_of_bounds="clip").fit(
        z["rawval_FAIR+TRAJ"].astype(float), ctx.y_val)


def _predict(model, X):
    return model.predict_proba(X)[:, 1].astype(np.float64)


def _load_booster(path, device="cuda"):
    import xgboost as xgb
    m = xgb.XGBClassifier()
    m.load_model(path)
    m.set_params(device=device)
    return m


def stage_fit_gbm(device="cuda"):
    from agents.base import AgentOutput
    from allfeat_gbm import build_allfeat_design, column_sets, load_sources
    from baselines_fair import SCORE_CACHE, tune_gbm, variant_columns
    from baselines_mimic import load_splits
    from distill_lead import _fair_record
    from feed_outage import FEEDS, fallback_columns
    from fusion_common import FusionContext
    from t40_fair_reuse import fixed_cell

    fp = fingerprint()
    t0 = time.time()
    splits = load_splits()
    ctx = FusionContext(splits)
    sources = load_sources()
    names = sources["ecg"]["names"]
    design = {s: build_allfeat_design(splits[s], sources) for s in ("train", "val", "test")}
    print(f"[design] {time.time() - t0:.0f} s", flush=True)
    for s, ref in (("val", ctx.val_ordered), ("test", ctx.test_ordered)):
        k = design[s][2]
        assert (k["stay_id"].to_numpy() == ref["stay_id"].to_numpy()).all()
        assert (k["hour"].to_numpy() == ref["hour"].to_numpy()).all()
    X_tr, y_tr = design["train"][0], design["train"][1]
    X_va, y_va = design["val"][0], design["val"][1]
    X_te = design["test"][0]
    cols = column_sets(names)["allfeat"]
    fair_cols = list(variant_columns()["FAIR+TRAJ"])

    payload, records = {}, {}
    if GBM_NPZ.exists():
        old = _check_fp(GBM_NPZ, fp)
        payload = {k: old[k] for k in old.files if k not in ("fingerprint", "records_json")}
        records = json.loads(str(old["records_json"]))
    payload.update({"fingerprint": np.array(fp),
                    "val_stay_id": ctx.val_ordered["stay_id"].to_numpy(),
                    "val_hour": ctx.val_ordered["hour"].to_numpy(),
                    "test_stay_id": ctx.test_ordered["stay_id"].to_numpy(),
                    "test_hour": ctx.test_ordered["hour"].to_numpy()})
    (MODELS / "gbm").mkdir(parents=True, exist_ok=True)

    def save():
        p = dict(payload)
        p["records_json"] = np.array(json.dumps(records))
        _save_npz(GBM_NPZ, p)

    def fit_cell(tag, cell, cols_, seed, reference=None):
        path = MODELS / "gbm" / f"{tag}.ubj"
        if f"test_{tag}" in payload and path.exists():
            print(f"[{tag}] cached")
            return
        t1 = time.time()
        if cell is None:
            model, rec = tune_gbm(X_tr[cols_], y_tr, X_va[cols_], y_va, seed=seed,
                                  device=device, verbose=False)
        else:
            model, rec = fixed_cell(**cell)(X_tr[cols_], y_tr, X_va[cols_], y_va,
                                            seed=seed, device=device, verbose=False)
        s_va, s_te = _predict(model, X_va[cols_]), _predict(model, X_te[cols_])
        if (cell is not None and reference is not None
                and not (np.array_equal(s_va, reference[0])
                         and np.array_equal(s_te, reference[1]))):
            # The published fit ran the whole grid; rerun it the same way.
            print(f"[{tag}] the recorded cell alone differs; rerunning the grid", flush=True)
            model, rec = tune_gbm(X_tr[cols_], y_tr, X_va[cols_], y_va, seed=seed,
                                  device=device, verbose=False)
            rec["refit_with_full_grid"] = True
            s_va, s_te = _predict(model, X_va[cols_]), _predict(model, X_te[cols_])
        model.save_model(path)
        back = _load_booster(path, device)
        rec["reload_identical"] = bool(np.array_equal(_predict(back, X_va[cols_]), s_va)
                                       and np.array_equal(_predict(back, X_te[cols_]), s_te))
        if reference is not None:
            ok = (np.array_equal(s_va, reference[0]) and np.array_equal(s_te, reference[1]))
            rec["identical_to_published"] = bool(ok)
            rec["max_abs_diff_to_published"] = max(_max_diff(s_va, reference[0]),
                                                   _max_diff(s_te, reference[1]))
        rec["fit_seconds"] = round(time.time() - t1, 1)
        rec.pop("grid", None)
        records[tag] = rec
        payload[f"val_{tag}"], payload[f"test_{tag}"] = s_va, s_te
        save()
        print(f"[{tag}] {rec}", flush=True)
        if not rec["reload_identical"]:
            raise SystemExit(f"GATE: {tag} predicts differently after save/load")
        if reference is not None and not rec["identical_to_published"]:
            raise SystemExit(f"GATE: {tag} does not reproduce the published scores")

    # (a) the GBM agent's FAIR+TRAJ booster: the published winner cell, seed 42
    fr = _fair_record()
    z = np.load(SCORE_CACHE, allow_pickle=True)
    fit_cell("fair_traj", {"max_depth": fr["max_depth"],
                           "scale_pos_weight": float(fr["scale_pos_weight"])},
             fair_cols, 42, (z["rawval_FAIR+TRAJ"].astype(float),
                             z["raw_FAIR+TRAJ"].astype(float)))
    iso = _gbm_agent_iso(ctx)
    for split, ordered in (("val", ctx.val_ordered), ("test", ctx.test_ordered)):
        mine = _gbm_agent_output(payload[f"{split}_fair_traj"], ordered, iso)
        same = _same_output(mine, AgentOutput.load(PUB_EMB / f"gbm_{split}.npz"))
        records[f"gbm_agent_{split}_identical_to_published"] = same
        if not same:
            save()
            raise SystemExit(f"GATE: GBM agent {split} emission differs from published")
    save()

    # (b) the all-features GBM at the T31 seeds, each seed's recorded cell
    t31 = np.load(CACHE / "t31_seeds_tab.npz")
    t31_rec = json.loads(str(t31["records_json"]))
    for seed in SEEDS:
        r = t31_rec[f"allfeat_s{seed}"]
        if r["scale_pos_weight"] != 1.0:
            raise SystemExit("a T31 cell used the rebalanced weight; refit it exactly")
        fit_cell(f"allfeat_s{seed}", {"max_depth": r["max_depth"], "scale_pos_weight": 1.0},
                 cols, seed, (t31[f"val_allfeat_s{seed}"], t31[f"test_allfeat_s{seed}"]))

    # (c) planned-ahead fallbacks: the T30 recipe without the feed's columns
    for feed in FEEDS:
        fb = fallback_columns(feed, cols, names)
        for seed in FB_SEEDS:
            fit_cell(f"fb_{feed}_s{seed}", None, fb, seed)
            records[f"fb_{feed}_s{seed}"]["n_columns"] = len(fb)
    save()
    print(f"[done] {GBM_NPZ.name} in {time.time() - t0:.0f} s")


# --- stage: fit-heads (CPU workers) --------------------------------------------

def _head_jobs():
    from feed_outage import FEEDS
    jobs = [("published", 42)] + [("all8", s) for s in SEEDS]
    jobs += [(f"fb_{f}", s) for f in FEEDS for s in FB_SEEDS]
    return jobs


def _arm_agents(arm):
    from feed_outage import ALL8, fallback_agents
    if arm in ("published", "all8"):
        return ALL8
    return fallback_agents(arm[3:])


def _fit_head_job(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    _one_thread()
    arm, seed = job
    path = HEADS / f"{arm}_s{seed}.npz"
    fp = fingerprint()
    if path.exists():
        _check_fp(path, fp)
        return job, "cached", 0.0
    t0 = time.time()
    from baselines_mimic import load_splits
    from decompose import treatment_only_label
    from fusion_common import FusionContext
    from orchestrator.fusion import AgentBank
    from two_head import fit_head
    splits = load_splits()
    ctx = FusionContext(splits)
    train_o = _sorted(splits["train"])
    val_o = ctx.val_ordered
    labels = {"T": (treatment_only_label(train_o), treatment_only_label(val_o)),
              "D": (train_o["label_24h_death"].to_numpy(dtype=float),
                    val_o["label_24h_death"].to_numpy(dtype=float))}
    bank = AgentBank.load(agent_names=_arm_agents(arm),
                          emb_dir=PUB_EMB if arm == "published" else EMB)
    payload = {"fingerprint": np.array(fp)}
    for h, (y_tr, y_va) in labels.items():
        head = fit_head(bank, splits, y_tr, y_va, train_o, val_o, seed, EPOCHS)
        payload[f"rawval_{h}"] = head.raw_risk(splits["val"])
        payload[f"rawtest_{h}"] = head.raw_risk(splits["test"])
        if arm != "published":
            head.save_checkpoint(MODELS / "heads" / f"{arm}_{h}_s{seed}.pt")
    _save_npz(path, payload)
    return job, "fitted", round(time.time() - t0, 1)


def stage_fit_heads(workers, reverse=False):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("run fit-heads with CUDA_VISIBLE_DEVICES= (CPU only)")
    meta = json.loads(AGENT_GATE.read_text())
    if meta.get("fingerprint") != fingerprint() or not meta.get("done"):
        raise SystemExit("run fit-agents first (or its outputs are stale)")
    HEADS.mkdir(parents=True, exist_ok=True)
    (MODELS / "heads").mkdir(parents=True, exist_ok=True)
    jobs = _head_jobs()
    if reverse:           # a second process can work the list from the other end
        jobs = jobs[::-1]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for job, status, secs in pool.imap_unordered(_fit_head_job, jobs):
            print(f"[{job[0]} s{job[1]}] {status} {secs} s "
                  f"({time.time() - t0:.0f} s elapsed)", flush=True)
    _gate_heads()


def _composite(ctx, raw, isos=None):
    """Two-head composite from raw head scores: per-head isotonic on V_fit
    (against that head's own label), rank tiebreak, noisy-OR. ``isos`` reuses
    fitted calibrators for scores outside val (test, outage)."""
    from decompose import treatment_only_label
    from sklearn.isotonic import IsotonicRegression
    if isos is None:
        y = {"T": treatment_only_label(ctx.val_ordered),
             "D": ctx.val_ordered["label_24h_death"].to_numpy(dtype=float)}
        isos = {h: IsotonicRegression(out_of_bounds="clip").fit(
            raw[f"rawval_{h}"][ctx.fit_mask], y[h][ctx.fit_mask]) for h in ("T", "D")}
    return isos


def _sentinel_scores(ctx, z):
    """``(val, test, isos)`` composite scores from a heads npz."""
    from feed_outage import calibrated, noisy_or
    isos = _composite(ctx, z)
    val = noisy_or(calibrated(isos["T"], z["rawval_T"]), calibrated(isos["D"], z["rawval_D"]))
    test = noisy_or(calibrated(isos["T"], z["rawtest_T"]), calibrated(isos["D"], z["rawtest_D"]))
    return val, test, isos


def _gate_heads():
    """The head code path reproduces the published two-head at seed 42."""
    from baselines_mimic import load_splits
    from fusion_common import FusionContext
    ctx = FusionContext(load_splits())
    z = np.load(HEADS / "published_s42.npz")
    val, test, _ = _sentinel_scores(ctx, z)
    pub = np.load(CACHE / "two_head.npz")
    ok = (np.array_equal(val, pub["val_s42"]) and np.array_equal(test, pub["test_s42"]))
    print(f"[gate] T42 head code on the published bank equals two_head.npz s42: {ok} "
          f"(max |d| {max(_max_diff(val, pub['val_s42']), _max_diff(test, pub['test_s42'])):.3g})")
    if not ok:
        raise SystemExit("GATE: the head code path does not reproduce two_head.npz")


# --- stage: score (outages) -----------------------------------------------------

_W = {}


def _score_init():
    """Load once per worker: frozen models, raw sources and intact references."""
    os.environ["OMP_NUM_THREADS"] = "1"
    _one_thread()
    import pandas as pd
    import torch

    from agents.base import AgentOutput
    from agents.ecg_agent import ECG_INDEX
    from allfeat_gbm import column_sets
    from baselines_fair import variant_columns
    from baselines_mimic import load_splits
    from feed_outage import ALL8, FEEDS, draw_windows, fallback_agents, fallback_columns
    from fusion_common import FusionContext
    from mimic_cohort import RAW_PATH
    from modality_features import load_cadence, load_fluids, load_labs, load_resp
    from orchestrator.fusion import Orchestrator
    fp = fingerprint()
    fp_score = fingerprint("score")
    splits = load_splits()
    ctx = FusionContext(splits)
    test = ctx.test_ordered
    stays = np.sort(test["stay_id"].unique())

    def mine(df):
        return df[np.isin(df["stay_id"].to_numpy(dtype="int64"), stays)].reset_index(drop=True)
    raw = pd.read_parquet(RAW_PATH)
    # The hourly tables are cut to test stays (checked: no feature changes).
    # The ECG index stays whole: its builder breaks same-hour ties with an
    # unstable sort over the whole index, so cutting it re-rolls ties.
    src = {"cohort_raw": mine(raw), "labs": mine(load_labs()),
           "ecg_index": pd.read_parquet(ECG_INDEX), "cadence": mine(load_cadence()),
           "fluid": mine(load_fluids()), "resp": mine(load_resp())}
    del raw
    from agents.ecg_meas_agent import MEAS_CACHE
    zm = np.load(MEAS_CACHE)
    ecg = {"values": zm["embeddings"].astype(np.float32), "study_ids": zm["study_ids"],
           "names": [str(n) for n in zm["feature_names"]]}

    agents = {n: torch.load(MODELS / "agents" / f"{n}.pt", weights_only=False)
              for n in ALL8 if n != "gbm"}
    gz = _check_fp(GBM_NPZ, fp)
    gbm_ref = {k: gz[k] for k in gz.files if k.startswith("test_")}
    boosters = {f"allfeat_s{s}": _load_booster(MODELS / "gbm" / f"allfeat_s{s}.ubj")
                for s in SEEDS}
    boosters["fair_traj"] = _load_booster(MODELS / "gbm" / "fair_traj.ubj")
    for f in FEEDS:
        for s in FB_SEEDS:
            boosters[f"fb_{f}_s{s}"] = _load_booster(MODELS / "gbm" / f"fb_{f}_s{s}.ubj")
    intact = {n: AgentOutput.load(EMB / f"{n}_test.npz") for n in ALL8}
    from orchestrator.fusion import AgentBank
    bank0 = AgentBank.load(splits=("test",), agent_names=ALL8, emb_dir=EMB)
    heads = {}
    for s in SEEDS:
        for h in ("T", "D"):
            heads[("all8", h, s)] = Orchestrator.load_checkpoint(
                MODELS / "heads" / f"all8_{h}_s{s}.pt", bank0)
    for f in FEEDS:
        sub = AgentBank.load(splits=("test",), agent_names=fallback_agents(f), emb_dir=EMB)
        for s in FB_SEEDS:
            for h in ("T", "D"):
                heads[(f"fb_{f}", h, s)] = Orchestrator.load_checkpoint(
                    MODELS / "heads" / f"fb_{f}_{h}_s{s}.pt", sub)
    head_ref = {}
    for arm, s in [("all8", s) for s in SEEDS] + [(f"fb_{f}", s) for f in FEEDS for s in FB_SEEDS]:
        z = _check_fp(HEADS / f"{arm}_s{s}.npz", fp)
        for h in ("T", "D"):
            head_ref[(arm, h, s)] = z[f"rawtest_{h}"]
    names = ecg["names"]
    cols = column_sets(names)["allfeat"]
    _W.update(fp=fp_score, ctx=ctx, test=test, stays=stays, src=src, ecg=ecg, agents=agents,
              boosters=boosters, gbm_ref=gbm_ref, intact=intact, heads=heads, head_ref=head_ref,
              cols=cols, fair_cols=list(variant_columns()["FAIR+TRAJ"]),
              fb_cols={f: fallback_columns(f, cols, names) for f in FEEDS},
              iso=_gbm_agent_iso(ctx), starts=draw_windows(test[["stay_id", "hour"]], seed=42),
              split_col=splits["test"]["split"].iloc[0])


def _run_systems(src, feed):
    """Every system on test under ``src``: agent outputs, head raws, GBM raws.

    All-8 heads and all-features GBMs always; the planned-ahead fallbacks only
    for ``feed`` (every fallback when ``feed`` is None, the intact path)."""
    import pandas as pd

    from allfeat_gbm import build_allfeat_design
    from feed_outage import ALL8, COHORT_CHANNELS, bank_from_outputs, cohort_from_raw
    W = _W
    if feed is None or feed in COHORT_CHANNELS:
        frame = _sorted(cohort_from_raw(src["cohort_raw"].copy()))
        frame["split"] = W["split_col"]
    else:
        frame = W["test"]
    af = {"labs": src["labs"], "cadence": src["cadence"], "fluid": src["fluid"],
          "resp": src["resp"], "ecg": {"index": src["ecg_index"], **W["ecg"]}}
    X, y, keys = build_allfeat_design(frame, af)
    a = W["agents"]
    outs = {"news2": a["news2"].emit(frame), "vitals_tcn": a["vitals_tcn"].emit(frame)}
    a["labs"].labs = src["labs"]
    outs["labs"] = a["labs"].emit(frame)
    a["ecg_meas"]._cache = (W["ecg"]["values"], W["ecg"]["study_ids"], src["ecg_index"])
    outs["ecg_meas"] = a["ecg_meas"].emit(frame)
    for n in ("cadence", "fluid", "resp"):
        a[n].source = src[n]
        outs[n] = a[n].emit(frame)
    raw_fair = _predict(W["boosters"]["fair_traj"], X[W["fair_cols"]])
    outs["gbm"] = _gbm_agent_output(raw_fair, keys, W["iso"])
    banks = {tuple(ALL8): bank_from_outputs(outs, list(ALL8))}
    wanted = ("all8",) + ((f"fb_{feed}",) if feed else tuple(
        arm_ for arm_, _, _ in W["heads"] if arm_ != "all8"))
    head_raw = {}
    for (arm, h, s), head in W["heads"].items():
        if arm not in wanted:
            continue
        names = tuple(head.bank.agent_names)
        if names not in banks:
            banks[names] = bank_from_outputs(outs, list(names))
        head.bank = banks[names]
        head_raw[(arm, h, s)] = head.raw_risk(frame)
    gbm_raw = {}
    for k, b in W["boosters"].items():
        if k.startswith("allfeat_"):
            gbm_raw[k] = _predict(b, X[W["cols"]])
        elif k.startswith("fb_") and (feed is None or k.startswith(f"fb_{feed}_")):
            gbm_raw[k] = _predict(b, X[W["fb_cols"][k[3:].rsplit("_s", 1)[0]]])
    return frame, X, outs, head_raw, gbm_raw


def _score_job(job):
    import pandas as pd

    from feed_outage import FEED_AGENTS, apply_outage, before_window_rows, feed_columns
    scen, feed = job
    W = _W
    path = SCORES / f"{scen}_{feed or 'intact'}.npz"
    if path.exists():
        _check_fp(path, W["fp"])
        return job, "cached", None
    t0 = time.time()
    if feed is None:
        src = W["src"]
    else:
        starts = W["starts"] if scen == "B" else None
        src = apply_outage(W["src"], feed, W["stays"], starts)
    frame, X, outs, head_raw, gbm_raw = _run_systems(src, feed)
    test = W["test"]
    checks = {"keys_and_labels_identical": bool(
        (frame["stay_id"].to_numpy() == test["stay_id"].to_numpy()).all()
        and (frame["hour"].to_numpy() == test["hour"].to_numpy()).all()
        and np.array_equal(frame["label_24h"].to_numpy(dtype=float),
                           test["label_24h"].to_numpy(dtype=float)))}
    affected = set(FEED_AGENTS[feed]) if feed else set()
    checks["unaffected_agents_identical"] = all(
        _same_output(outs[n], W["intact"][n]) for n in outs if n not in affected)
    checks["affected_agents_changed"] = {n: not _same_output(outs[n], W["intact"][n])
                                         for n in sorted(affected)}
    checks["fallback_heads_identical"] = all(
        np.array_equal(v, W["head_ref"][k]) for k, v in head_raw.items() if k[0] != "all8")
    checks["fallback_gbms_identical"] = all(
        np.array_equal(v, W["gbm_ref"][f"test_{k}"]) for k, v in gbm_raw.items()
        if k.startswith("fb_"))
    if feed is None:
        checks["intact_path_reproduces_references"] = bool(
            checks["unaffected_agents_identical"]
            and all(np.array_equal(v, W["head_ref"][k]) for k, v in head_raw.items())
            and all(np.array_equal(v, W["gbm_ref"][f"test_{k}"]) for k, v in gbm_raw.items()))
    else:
        X0 = W.get("X0")
        mine = set(feed_columns(feed, W["ecg"]["names"]))
        checks["other_feeds_gbm_columns_identical"] = all(
            np.array_equal(X0[c].to_numpy(dtype=float), X[c].to_numpy(dtype=float),
                           equal_nan=True) for c in X0.columns if c not in mine)
        if scen == "B":
            before = before_window_rows(test[["stay_id", "hour"]], W["starts"])
            checks["before_window_agents_identical"] = all(
                np.array_equal(outs[n].risk[before], W["intact"][n].risk[before])
                and np.array_equal(outs[n].latent[before], W["intact"][n].latent[before])
                for n in outs)
            d_head = max(_max_diff(v[before], W["head_ref"][k][before])
                         for k, v in head_raw.items())
            checks["before_window_head_max_abs_diff"] = d_head
            checks["before_window_gbm_identical"] = all(
                np.array_equal(v[before], W["gbm_ref"][f"test_{k}"][before])
                for k, v in gbm_raw.items())
    payload = {"fingerprint": np.array(W["fp"]), "checks_json": np.array(json.dumps(checks))}
    for (arm, h, s), v in head_raw.items():
        if arm == "all8":
            payload[f"S_{h}_s{s}"] = v
    for k, v in gbm_raw.items():
        if k.startswith("allfeat_"):
            payload[f"G_s{k.rsplit('_s', 1)[1]}"] = v
    _save_npz(path, payload)
    return job, checks, round(time.time() - t0, 1)


def _score_worker_init():
    _score_init()
    # the intact design, for the other-feeds column check
    from allfeat_gbm import build_allfeat_design
    W = _W
    af = {"labs": W["src"]["labs"], "cadence": W["src"]["cadence"], "fluid": W["src"]["fluid"],
          "resp": W["src"]["resp"], "ecg": {"index": W["src"]["ecg_index"], **W["ecg"]}}
    W["X0"] = build_allfeat_design(W["test"], af)[0]


REQUIRED = ("keys_and_labels_identical", "unaffected_agents_identical",
            "fallback_heads_identical", "fallback_gbms_identical")


def _gate_checks(job, checks):
    scen, feed = job
    bad = [k for k in REQUIRED if not checks.get(k)]
    if feed is None and not checks.get("intact_path_reproduces_references"):
        bad.append("intact_path_reproduces_references")
    if feed is not None:
        if not checks.get("other_feeds_gbm_columns_identical"):
            bad.append("other_feeds_gbm_columns_identical")
        if scen == "B":
            for k in ("before_window_agents_identical", "before_window_gbm_identical"):
                if not checks.get(k):
                    bad.append(k)
            if checks.get("before_window_head_max_abs_diff", 0) != 0:
                bad.append("before_window_head_max_abs_diff")
    return bad


def stage_score(workers):
    from feed_outage import FEEDS
    SCORES.mkdir(parents=True, exist_ok=True)
    jobs = [("A", None)] + [(s, f) for s in SCENARIOS for f in FEEDS]
    t0 = time.time()
    # the intact path first, alone: everything downstream depends on it
    with mp.get_context("spawn").Pool(1, initializer=_score_worker_init) as pool:
        for job, checks, secs in pool.imap_unordered(_score_job, jobs[:1]):
            _report(job, checks, secs, t0)
    with mp.get_context("spawn").Pool(workers, initializer=_score_worker_init) as pool:
        for job, checks, secs in pool.imap_unordered(_score_job, jobs[1:]):
            _report(job, checks, secs, t0)


def _report(job, checks, secs, t0):
    if checks == "cached":
        z = np.load(SCORES / f"{job[0]}_{job[1] or 'intact'}.npz")
        checks = json.loads(str(z["checks_json"]))
    bad = _gate_checks(job, checks)
    print(f"[{job[0]} {job[1] or 'intact'}] {secs} s ({time.time() - t0:.0f} s elapsed) "
          f"checks: {'all pass' if not bad else 'FAILED ' + ', '.join(bad)}", flush=True)
    if bad:
        raise SystemExit(f"GATE: {job} failed {bad}")


# --- stage: eval ------------------------------------------------------------------

def _gbm_scores(ctx, val, test):
    from feed_outage import calibrated
    iso = ctx.fit_calibrator(np.asarray(val, dtype=float))
    return calibrated(iso, val), calibrated(iso, test), iso


def _summ(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


def stage_eval(workers):
    import pandas as pd

    from demo_bundle import assert_aggregate_only, first_alarm_table
    from feed_outage import (FEEDS, absolute_verdict, alarm_label, calibrated,
                             draw_windows, headline_pass, noisy_or,
                             robustness_verdict, window_catch, window_rows)
    from baselines_mimic import load_splits
    from fusion_common import FusionContext
    from metrics import auprc, auroc
    from paper_operating_points import catch_stats, freeze_quantile
    from paper_stats import metric_draws, pooled_delta

    t0 = time.time()
    fp, fp_score = fingerprint(), fingerprint("score")
    ctx = FusionContext(load_splits())
    test = ctx.test_ordered
    gz = _check_fp(GBM_NPZ, fp)
    gbm_records = json.loads(str(gz["records_json"]))

    # --- every score, calibrated, with its intact threshold ----------------
    S, G, thr = {}, {}, {}
    out_S, out_G = {}, {}
    for s in SEEDS:
        z = _check_fp(HEADS / f"all8_s{s}.npz", fp)
        val, tst, isos = _sentinel_scores(ctx, z)
        S[s], thr[("S", s)] = tst, freeze_quantile(ctx, val, BUDGET)
        gval, gtst, giso = _gbm_scores(ctx, gz[f"val_allfeat_s{s}"], gz[f"test_allfeat_s{s}"])
        G[s], thr[("G", s)] = gtst, freeze_quantile(ctx, gval, BUDGET)
        for scen in SCENARIOS:
            for f in FEEDS:
                o = _check_fp(SCORES / f"{scen}_{f}.npz", fp_score)
                out_S[(scen, f, s)] = noisy_or(calibrated(isos["T"], o[f"S_T_s{s}"]),
                                               calibrated(isos["D"], o[f"S_D_s{s}"]))
                out_G[(scen, f, s)] = calibrated(giso, o[f"G_s{s}"])
    fb_S, fb_G = {}, {}
    for f in FEEDS:
        for s in FB_SEEDS:
            z = _check_fp(HEADS / f"fb_{f}_s{s}.npz", fp)
            val, tst, _ = _sentinel_scores(ctx, z)
            fb_S[(f, s)], thr[("Sfb", f, s)] = tst, freeze_quantile(ctx, val, BUDGET)
            gval, gtst, _ = _gbm_scores(ctx, gz[f"val_fb_{f}_s{s}"], gz[f"test_fb_{f}_s{s}"])
            fb_G[(f, s)], thr[("Gfb", f, s)] = gtst, freeze_quantile(ctx, gval, BUDGET)

    y_all = ctx.y_test
    quality = {"rebuilt_sentinel_auprc": _summ([auprc(y_all, S[s]) for s in SEEDS]),
               "gbm_auprc": _summ([auprc(y_all, G[s]) for s in SEEDS]),
               "published_sentinel_auprc_mean": PUBLISHED_AUPRC, "tolerance": QUALITY_TOL}
    quality["pass"] = bool(abs(quality["rebuilt_sentinel_auprc"]["mean"] - PUBLISHED_AUPRC)
                           <= QUALITY_TOL)
    print(f"[quality] rebuilt Sentinel ten-seed AUPRC "
          f"{quality['rebuilt_sentinel_auprc']['mean']:.4f} (published {PUBLISHED_AUPRC}); "
          f"GBM {quality['gbm_auprc']['mean']:.4f}", flush=True)
    if not quality["pass"]:
        raise SystemExit("GATE: the rebuilt Sentinel is not the published system in quality")

    starts = draw_windows(test[["stay_id", "hour"]], seed=42)
    rows_by = {"A": np.ones(len(test), dtype=bool),
               "B": window_rows(test[["stay_id", "hour"]], starts)}
    result = {"fingerprint": fp_score, "quality_gate": quality, "budget": BUDGET, "reps": REPS,
              "seeds": list(SEEDS), "fallback_seeds": list(FB_SEEDS),
              "gbm_fit_records": gbm_records, "scenarios": {}}
    checks = {"intact": json.loads(str(np.load(SCORES / "A_intact.npz")["checks_json"]))}
    for scen in SCENARIOS:
        for f in FEEDS:
            z = np.load(SCORES / f"{scen}_{f}.npz")
            checks[f"{scen}_{f}"] = json.loads(str(z["checks_json"]))
    result["score_checks"] = checks

    for scen in SCENARIOS:
        rows = rows_by[scen]
        y = y_all[rows]
        pats = ctx.patient_test[rows]
        frame = test[rows].reset_index(drop=True)
        scores = {}
        for s in SEEDS:
            scores[f"S_int_s{s}"], scores[f"G_int_s{s}"] = S[s][rows], G[s][rows]
            for f in FEEDS:
                scores[f"S_out_{f}_s{s}"] = out_S[(scen, f, s)][rows]
                scores[f"G_out_{f}_s{s}"] = out_G[(scen, f, s)][rows]
        for f in FEEDS:
            for s in FB_SEEDS:
                scores[f"S_fb_{f}_s{s}"], scores[f"G_fb_{f}_s{s}"] = fb_S[(f, s)][rows], fb_G[(f, s)][rows]
        t1 = time.time()
        pts = {m: {n: float(fn(y, v)) for n, v in scores.items()}
               for m, fn in (("auprc", auprc), ("auroc", auroc))}
        draws = metric_draws({"auprc": ("auprc", y), "auroc": ("auroc", y)}, scores, pats,
                             reps=REPS, workers=workers)
        print(f"[{scen}] {len(scores)} scores bootstrapped in {time.time() - t1:.0f} s", flush=True)

        def catch(name_scores, thr_):
            if scen == "A":
                tab = first_alarm_table(test, {"m": name_scores}, {"m": thr_})
                return catch_stats(tab, "m")["catch"]
            return window_catch(frame, y, name_scores[rows], thr_)[0]

        def rate(v, thr_):
            return float((np.asarray(v)[rows] >= thr_).mean())

        sres = {"rows": int(rows.sum()), "stays": int(frame["stay_id"].nunique()),
                "positives": int(y.sum()), "patients": int(len(np.unique(pats)))}
        if scen == "B":
            sres["event_stays_with_positive_window_hour"] = window_catch(
                frame, y, np.zeros(len(y)), 1.0)[1]
        sres["feeds"] = {}
        for f in FEEDS:
            fres = {}
            for arm, seeds_, s_int, s_out, g_int, g_out in (
                    ("no_retraining", SEEDS,
                     lambda s: f"S_int_s{s}", lambda s: f"S_out_{f}_s{s}",
                     lambda s: f"G_int_s{s}", lambda s: f"G_out_{f}_s{s}"),
                    ("planned_ahead", FB_SEEDS,
                     lambda s: f"S_int_s{s}", lambda s: f"S_fb_{f}_s{s}",
                     lambda s: f"G_int_s{s}", lambda s: f"G_fb_{f}_s{s}")):
                a = {}
                for m in ("auprc", "auroc"):
                    P, D = pts[m], draws[m]
                    sd_p = [P[s_int(s)] - P[s_out(s)] for s in seeds_]
                    gd_p = [P[g_int(s)] - P[g_out(s)] for s in seeds_]
                    sd_d = [D[s_int(s)] - D[s_out(s)] for s in seeds_]
                    gd_d = [D[g_int(s)] - D[g_out(s)] for s in seeds_]
                    rob = pooled_delta(gd_p, sd_p, gd_d, sd_d)
                    per_seed = [np.percentile(g - sd, [2.5, 97.5]) for g, sd in zip(gd_d, sd_d)]
                    absd = pooled_delta([P[s_out(s)] for s in seeds_], [P[g_out(s)] for s in seeds_],
                                        [D[s_out(s)] for s in seeds_], [D[g_out(s)] for s in seeds_])
                    zero = np.zeros_like(D[s_int(seeds_[0])])
                    a[m] = {
                        "sentinel_intact": _summ([P[s_int(s)] for s in seeds_]),
                        "sentinel_outage": _summ([P[s_out(s)] for s in seeds_]),
                        "gbm_intact": _summ([P[g_int(s)] for s in seeds_]),
                        "gbm_outage": _summ([P[g_out(s)] for s in seeds_]),
                        "sentinel_drop": pooled_delta(sd_p, [0.0] * len(seeds_), sd_d,
                                                      [zero] * len(seeds_)),
                        "gbm_drop": pooled_delta(gd_p, [0.0] * len(seeds_), gd_d,
                                                 [zero] * len(seeds_)),
                        "robustness_gbm_drop_minus_sentinel_drop": rob,
                        "robustness_seed_pairs_ci_above_below": [
                            int(sum(lo > 0 for lo, _ in per_seed)),
                            int(sum(hi < 0 for _, hi in per_seed))],
                        "outage_sentinel_minus_gbm": absd,
                    }
                a["verdict_robustness"] = robustness_verdict(
                    a["auprc"]["robustness_gbm_drop_minus_sentinel_drop"]["ci"])
                a["verdict_absolute"] = absolute_verdict(a["auprc"]["outage_sentinel_minus_gbm"]["ci"])
                # alarms and catches at the 2% budget
                al = {}
                for sysname, int_v, out_v, t_int, t_out in (
                        ("sentinel", lambda s: S[s],
                         (lambda s: out_S[(scen, f, s)]) if arm == "no_retraining" else (lambda s: fb_S[(f, s)]),
                         lambda s: thr[("S", s)],
                         (lambda s: thr[("S", s)]) if arm == "no_retraining" else (lambda s: thr[("Sfb", f, s)])),
                        ("gbm", lambda s: G[s],
                         (lambda s: out_G[(scen, f, s)]) if arm == "no_retraining" else (lambda s: fb_G[(f, s)]),
                         lambda s: thr[("G", s)],
                         (lambda s: thr[("G", s)]) if arm == "no_retraining" else (lambda s: thr[("Gfb", f, s)]))):
                    r_int = [rate(int_v(s), t_int(s)) for s in seeds_]
                    r_out = [rate(out_v(s), t_out(s)) for s in seeds_]
                    c_int = [catch(int_v(s), t_int(s)) for s in seeds_]
                    c_out = [catch(out_v(s), t_out(s)) for s in seeds_]
                    al[sysname] = {"alarm_rate_intact": _summ(r_int), "alarm_rate_outage": _summ(r_out),
                                   "alarm_label": alarm_label(np.mean(r_out), np.mean(r_int)),
                                   "catch_intact": _summ(c_int), "catch_outage": _summ(c_out)}
                a["at_2pct"] = al
                fres[arm] = a
            sres["feeds"][f] = fres
            print(f"  [{scen} {f}] robustness "
                  f"{fres['no_retraining']['auprc']['robustness_gbm_drop_minus_sentinel_drop']['delta']:+.4f} "
                  f"{fres['no_retraining']['verdict_robustness']}; planned "
                  f"{fres['planned_ahead']['verdict_robustness']}", flush=True)
        verdicts = {f: sres["feeds"][f]["no_retraining"]["verdict_robustness"] for f in FEEDS}
        ok, n_s, n_g = headline_pass(verdicts)
        sres["robustness_verdicts_no_retraining"] = verdicts
        sres["counts"] = {"sentinel_more_robust": n_s, "gbm_more_robust": n_g}
        if scen == "A":
            sres["headline_pass"] = ok
        result["scenarios"][scen] = sres
    assert_aggregate_only(result)
    OUT.write_text(json.dumps(result, indent=1))
    print(f"[saved] {OUT} ({time.time() - t0:.0f} s); headline "
          f"{'PASS' if result['scenarios']['A']['headline_pass'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("fit-agents", "fit-gbm", "fit-heads", "score", "eval"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reverse", action="store_true",
                    help="fit-heads: take the jobs in reverse order")
    args = ap.parse_args()
    if args.stage == "fit-agents":
        stage_fit_agents()
    elif args.stage == "fit-gbm":
        stage_fit_gbm(args.device)
    elif args.stage == "fit-heads":
        stage_fit_heads(args.workers, args.reverse)
    elif args.stage == "score":
        stage_score(args.workers)
    else:
        stage_eval(args.workers)


if __name__ == "__main__":
    main()
