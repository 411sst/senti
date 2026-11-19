"""T49: Sentinel, the bridge and the reference GBM on Sadanandan's cohort.

Pre-registered in results_fusion.md (T49). Rows are their code-rule samples
(``src/sadanandan_cohort.py``, hours 6 to int(LoS) - 24) of patients in our
splits: training rows train, validation rows early-stop and calibrate, and
only our test split's patients are scored.

Stages, in order (no test metric is computed before ``eval``):

* ``envelope`` (CPU, one thread): the rounding envelope of gate 1 (the dated
  clarification under T49). T42's own frames are re-emitted through the five
  agents that emit in fixed-size batches at four other batch sizes, inputs
  unchanged; the T42 heads score the re-emitted test banks.

* ``agents --split S`` (CPU, one thread; the GBM agent predicts on the GPU):
  the split's rows from the all-hours reconciled extract
  (``uncensored_frames.sample_frame``), the T42 rebuild's seven agents
  (``models/t42/agents``), the GBM agent (FAIR+TRAJ; training rows out of fold
  through the shipped folds, ``models/t4x``), and the 262-column all-features
  design for the reference arm. On the (stay, hour) rows our cohort also has,
  the batched agents must match the T42 bank within the envelope and the
  others exactly (gate 1, agent part; checked on every split, required on test
  by the pre-registration).
* ``shipped`` (CPU, one thread): the T42 heads (both, seeds 42 and 1-9) on
  validation and test rows. Head scores must match T42's on shared test rows
  within the head envelope (gate 1, head part).
* ``heads --workers N`` (CPU, one thread per worker): two heads refit with
  ``two_head.fit_head`` on their label's parts (T: first vasopressor or
  ventilation start in 24 h; D: ICU death in 24 h) at ten seeds. Gate 2: the
  parts rebuild their label on every training and validation row.
* ``gbms`` (GPU): the bridge (132 features, ``src/t49_bridge.py``) and the
  reference all-features GBM, each with ``baselines_fair.tune_gbm`` unchanged
  at ten seeds, on the same rows.
* ``eval --workers N``: the pre-registered measures on test, the bridge
  comparability gate and the pass rule. Aggregates go to
  ``data/cache/t49_eval.json``.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python scripts/t49_score.py envelope
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python scripts/t49_score.py envelope-tcn
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python scripts/t49_score.py envelope-heads
    OMP_NUM_THREADS=1 python scripts/t49_score.py agents --split test
    OMP_NUM_THREADS=1 python scripts/t49_score.py agents --split val
    OMP_NUM_THREADS=1 python scripts/t49_score.py agents --split train
    OMP_NUM_THREADS=1 python scripts/t49_score.py shipped
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python scripts/t49_score.py heads --workers 2
    OMP_NUM_THREADS=1 python scripts/t49_score.py gbms

Outputs (never committed, never overwritten): ``data/cache/t49/`` (frames
with keys and labels, agent outputs, designs, raw scores, aggregate JSON
reports) and ``models/t49/`` (head checkpoints).
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
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CACHE = REPO / "data" / "cache"
WORK = CACHE / "t49"
EMB = WORK / "embeddings"
SCORES = WORK / "scores"
STAYS = CACHE / "t49_stays.parquet"
THERAPY_CODE = CACHE / "t49_therapy_code.parquet"
SAMPLES_CODE = CACHE / "t49_samples_code.parquet"
RAW_ALL = CACHE / "t4x" / "mimic_raw_uncensored_all_reconciled.parquet"
ECG_INDEX_ALL = CACHE / "t4x" / "ecg_index_all.parquet"
ECG_MEAS_ALL = CACHE / "t4x" / "ecg_meas_features_all.npz"
T42_EMB = CACHE / "t42" / "embeddings"
T42_HEADS = CACHE / "t42" / "heads"
T42_MODELS = REPO / "models" / "t42"
FOLD_MODELS = REPO / "models" / "t4x"
MODELS = REPO / "models" / "t49"
SPLITS = ("train", "val", "test")
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
EPOCHS = 8
BATCHED = ("labs", "ecg_meas", "cadence", "fluid", "resp")
ENV_BATCHES = (1024, 2048, 4000, 8192)
# envelope.json (first measurement, absolute risk) is kept; the gate reads the
# second amendment's envelope, with risk differences in float32 ulps.
ENVELOPE = WORK / "envelope_ulps.json"
GATE_KEYS = ("share", "latent", "risk_ulps")
# vitals_tcn batches whole stays (64 per batch), so a different stay mix changes
# padded batch shapes; its envelope for the extra val/train check is measured
# the same way at other batch sizes. The extra check stops on magnitude only.
ENVELOPE_TCN = WORK / "envelope_vitals_tcn.json"
TCN_BATCHES = (16, 32, 128, 256)
EXTRA_KEYS = ("latent", "risk_ulps")
# Third amendment: the head envelope with both rounding sources, re-emitted
# agent inputs and the heads' own stay batching.
ENVELOPE_HEADS = WORK / "envelope_heads.json"
HEAD_BATCHES = (16, 32, 64, 128, 256)
FRAME_COLS = ["stay_id", "hour", "patient_id", "label", "label_t", "label_d", "started",
              "label_24h", "label_24h_death", "split"]


def _one_thread():
    import torch
    torch.set_num_threads(1)


def _refuse(path):
    if Path(path).exists():
        raise SystemExit(f"{path} exists; new outputs go to new files")


def _write_json(path, obj):
    from demo_bundle import assert_aggregate_only
    obj = json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    assert_aggregate_only(obj)
    Path(path).write_text(json.dumps(obj, indent=1))


def _samples():
    """Their samples with the label's two parts; must equal the cached
    Phase 1 samples (keys, label, started) exactly."""
    from sadanandan_cohort import code_cohort, code_samples, first_starts
    kept, _ = code_cohort(pd.read_parquet(STAYS))
    smp = code_samples(kept, first_starts(pd.read_parquet(THERAPY_CODE)))
    cached = pd.read_parquet(SAMPLES_CODE)
    for c in ("stay_id", "hour", "label", "started"):
        if not np.array_equal(smp[c].to_numpy(), cached[c].to_numpy()):
            raise SystemExit(f"rebuilt samples differ from the cached ones in {c!r}")
    smp["our_split"] = cached["our_split"].to_numpy()
    return smp


def _same(a, b):
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


# --- stage: envelope --------------------------------------------------------------

def _load_agents():
    import torch
    from feed_outage import ALL8
    return {n: torch.load(T42_MODELS / "agents" / f"{n}.pt", weights_only=False)
            for n in ALL8 if n != "gbm"}


def _emit_batched(agents, frame, src, ecg, ecg_index):
    agents["labs"].labs = src["labs"]
    outs = {"labs": agents["labs"].emit(frame)}
    agents["ecg_meas"]._cache = (ecg["values"], ecg["study_ids"], ecg_index)
    outs["ecg_meas"] = agents["ecg_meas"].emit(frame)
    for n in ("cadence", "fluid", "resp"):
        agents[n].source = src[n]
        outs[n] = agents[n].emit(frame)
    return outs


def _diff(a_lat, a_risk, b_lat, b_risk):
    dl = np.abs(np.asarray(a_lat, float) - np.asarray(b_lat, float))
    dl = dl.max(axis=1) if dl.ndim == 2 else dl
    dr = np.abs(np.asarray(a_risk, float) - np.asarray(b_risk, float))
    ulp = np.spacing(np.abs(np.asarray(b_risk, np.float32))).astype(float)
    return {"share": float(((dl > 0) | (dr > 0)).mean()) if len(dl) else 0.0,
            "latent": float(dl.max()) if len(dl) else 0.0,
            "risk": float(dr.max()) if len(dr) else 0.0,
            "risk_ulps": float((dr / ulp).max()) if len(dr) else 0.0}


def _raw_diff(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return {"share": float((d > 0).mean()), "raw": float(d.max())}


def stage_envelope():
    from agents.base import AgentOutput
    from agents.ecg_agent import ECG_INDEX
    from agents.ecg_meas_agent import MEAS_CACHE
    from baselines_mimic import load_splits
    from feed_outage import ALL8, bank_from_outputs
    from modality_features import load_cadence, load_fluids, load_labs, load_resp
    from orchestrator.fusion import Orchestrator
    _one_thread()
    _refuse(ENVELOPE)
    t0 = time.time()
    splits = load_splits()
    zm = np.load(MEAS_CACHE)
    ecg = {"values": zm["embeddings"].astype(np.float32), "study_ids": zm["study_ids"]}
    ecg_index = pd.read_parquet(ECG_INDEX)
    full = {"labs": load_labs(), "cadence": load_cadence(), "fluid": load_fluids(),
            "resp": load_resp()}
    agents = _load_agents()
    env = {"batch_sizes": list(ENV_BATCHES), "agents": {}, "heads": {}}
    test_outs = {}
    for split in SPLITS:
        frame = splits[split].sort_values(["stay_id", "hour"]).reset_index(drop=True)
        stays = np.unique(frame["stay_id"].to_numpy(dtype="int64"))
        src = {k: v[np.isin(v["stay_id"].to_numpy(dtype="int64"), stays)].reset_index(drop=True)
               for k, v in full.items()}
        refs = {n: AgentOutput.load(T42_EMB / f"{n}_{split}.npz") for n in BATCHED}
        per = {n: [] for n in BATCHED}
        for bs in (4096,) + ENV_BATCHES:
            for n in BATCHED:
                agents[n].batch_size = bs
            outs = _emit_batched(agents, frame, src, ecg, ecg_index)
            for n in BATCHED:
                if not (_same(outs[n].keys["stay_id"], refs[n].keys["stay_id"])
                        and _same(outs[n].keys["hour"], refs[n].keys["hour"])):
                    raise SystemExit(f"{n} {split}: keys differ from T42")
                d = _diff(outs[n].latent, outs[n].risk, refs[n].latent, refs[n].risk)
                if bs == 4096:
                    if d["share"] != 0.0:
                        raise SystemExit(f"{n} {split}: batch 4096 does not reproduce T42 exactly")
                else:
                    per[n].append(d)
            if split == "test" and bs != 4096:
                test_outs[bs] = outs
            print(f"[envelope {split} batch {bs}] "
                  f"{ {n: per[n][-1] for n in BATCHED} if bs != 4096 else 'exact'} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        for n in BATCHED:
            agents[n].batch_size = 4096
        env["agents"][split] = {n: {k: max(d[k] for d in per[n])
                                    for k in ("share", "latent", "risk", "risk_ulps")}
                                for n in BATCHED}
    # heads: the T42 heads on the T42 test bank with the batched agents re-emitted
    test = splits["test"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    base = {n: AgentOutput.load(T42_EMB / f"{n}_test.npz") for n in ALL8}
    worst = {"share": 0.0, "raw": 0.0}
    for bs, outs in test_outs.items():
        bank = bank_from_outputs({**base, **outs}, list(ALL8))
        for s in SEEDS:
            ref = np.load(T42_HEADS / f"all8_s{s}.npz")
            for h in ("T", "D"):
                head = Orchestrator.load_checkpoint(T42_MODELS / "heads" / f"all8_{h}_s{s}.pt", bank)
                d = _raw_diff(head.raw_risk(test), ref[f"rawtest_{h}"])
                env["heads"][f"b{bs}_s{s}_{h}"] = d
                worst = {k: max(worst[k], d[k]) for k in worst}
        print(f"[envelope heads batch {bs}] worst so far {worst} ({time.time() - t0:.0f} s)", flush=True)
    env["heads_max"] = worst
    env["seconds"] = round(time.time() - t0, 1)
    _write_json(ENVELOPE, env)
    print(json.dumps({"agents_test": env["agents"]["test"], "heads_max": worst}, indent=1))


def stage_envelope_tcn():
    from agents.base import AgentOutput
    from baselines_mimic import load_splits
    _one_thread()
    _refuse(ENVELOPE_TCN)
    t0 = time.time()
    splits = load_splits()
    agent = _load_agents()["vitals_tcn"]
    env = {"batch_sizes": list(TCN_BATCHES), "agents": {}}
    for split in SPLITS:
        frame = splits[split].sort_values(["stay_id", "hour"]).reset_index(drop=True)
        ref = AgentOutput.load(T42_EMB / f"vitals_tcn_{split}.npz")
        per = []
        for bs in (64,) + TCN_BATCHES:
            agent.batch_size = bs
            o = agent.emit(frame)
            d = _diff(o.latent, o.risk, ref.latent, ref.risk)
            if bs == 64 and d["share"] != 0.0:
                raise SystemExit(f"vitals_tcn {split}: batch 64 does not reproduce T42 exactly")
            if bs != 64:
                per.append(d)
            print(f"[envelope vitals_tcn {split} batch {bs}] {d} ({time.time() - t0:.0f} s)", flush=True)
        env["agents"][split] = {"vitals_tcn": {k: max(d[k] for d in per) for k in per[0]}}
    env["seconds"] = round(time.time() - t0, 1)
    _write_json(ENVELOPE_TCN, env)


def stage_envelope_heads():
    from agents.base import AgentOutput
    from agents.ecg_agent import ECG_INDEX
    from agents.ecg_meas_agent import MEAS_CACHE
    from baselines_mimic import load_splits
    from feed_outage import ALL8, bank_from_outputs
    from modality_features import load_cadence, load_fluids, load_labs, load_resp
    from orchestrator.fusion import Orchestrator
    _one_thread()
    _refuse(ENVELOPE_HEADS)
    t0 = time.time()
    test = load_splits()["test"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    stays = np.unique(test["stay_id"].to_numpy(dtype="int64"))
    src = {k: v[np.isin(v["stay_id"].to_numpy(dtype="int64"), stays)].reset_index(drop=True)
           for k, v in {"labs": load_labs(), "cadence": load_cadence(), "fluid": load_fluids(),
                        "resp": load_resp()}.items()}
    zm = np.load(MEAS_CACHE)
    ecg = {"values": zm["embeddings"].astype(np.float32), "study_ids": zm["study_ids"]}
    ecg_index = pd.read_parquet(ECG_INDEX)
    agents = _load_agents()
    base = {n: AgentOutput.load(T42_EMB / f"{n}_test.npz") for n in ALL8}
    runs, worst = {}, {"share": 0.0, "raw": 0.0}
    for bs_a in ENV_BATCHES:
        for n in BATCHED:
            agents[n].batch_size = bs_a
        bank = bank_from_outputs({**base, **_emit_batched(agents, test, src, ecg, ecg_index)},
                                 list(ALL8))
        for s in SEEDS:
            ref = np.load(T42_HEADS / f"all8_s{s}.npz")
            for h in ("T", "D"):
                head = Orchestrator.load_checkpoint(T42_MODELS / "heads" / f"all8_{h}_s{s}.pt", bank)
                for bs_h in HEAD_BATCHES:
                    head.batch_size = bs_h
                    d = _raw_diff(head.raw_risk(test), ref[f"rawtest_{h}"])
                    runs[f"a{bs_a}_h{bs_h}_s{s}_{h}"] = d
                    worst = {k: max(worst[k], d[k]) for k in worst}
        print(f"[envelope heads, agent batch {bs_a}] worst so far {worst} ({time.time() - t0:.0f} s)",
              flush=True)
    _write_json(ENVELOPE_HEADS, {"agent_batch_sizes": list(ENV_BATCHES),
                                 "head_batch_sizes": list(HEAD_BATCHES), "heads_max": worst,
                                 "runs": runs, "seconds": round(time.time() - t0, 1)})


# --- stage: agents ----------------------------------------------------------------

def stage_agents(split):
    import torch
    import xgboost as xgb

    from agents.base import AgentOutput
    from agents.ecg_agent import ECG_INDEX
    from agents.ecg_meas_agent import MEAS_CACHE
    from allfeat_gbm import build_allfeat_design, column_sets
    from baselines_fair import variant_columns
    from baselines_mimic import load_splits
    from distill_lead import gbm_predict
    from feed_outage import ALL8
    from fusion_common import FusionContext
    from gbm_agent_oof import held_out_fold
    from modality_features import load_cadence, load_fluids, load_labs, load_resp
    from t42_feed_outage import _gbm_agent_iso, _gbm_agent_output, _load_booster, _predict
    from uncensored_frames import sample_frame, shared_positions, uncensored_frame

    _one_thread()
    report_path = WORK / f"agents_{split}.json"
    _refuse(report_path)
    if not ENVELOPE.exists():
        raise SystemExit("run the envelope stage first")
    t0 = time.time()
    smp = _samples()
    smp = smp[smp["our_split"] == split].reset_index(drop=True)
    stays = np.unique(smp["stay_id"].to_numpy(dtype="int64"))

    raw = pd.read_parquet(RAW_ALL)
    frame = uncensored_frame(raw, stays)
    del raw
    frame = sample_frame(frame, smp[["stay_id", "hour", "subject_id", "label", "label_t",
                                     "label_d", "started"]])
    if not _same(frame["patient_id"].to_numpy(dtype="int64"),
                 frame["subject_id"].to_numpy(dtype="int64")):
        raise SystemExit("frame patient ids differ from the samples' subject ids")
    frame = frame.drop(columns=["subject_id"])
    frame["label_24h"] = frame["label"].astype(frame["label_24h"].dtype)
    frame["label_24h_death"] = frame["label_d"].astype(frame["label_24h_death"].dtype)
    frame["split"] = split
    print(f"[{split}] frame: {len(frame):,} rows, {len(stays):,} stays ({time.time() - t0:.0f} s)",
          flush=True)

    def mine(df):
        return df[np.isin(df["stay_id"].to_numpy(dtype="int64"), stays)].reset_index(drop=True)
    src = {"labs": mine(load_labs()), "cadence": mine(load_cadence()),
           "fluid": mine(load_fluids()), "resp": mine(load_resp())}

    # ECG inputs: stays our cohort has use our cohort's ECG index and features
    # (the builder breaks same-hour ties with an unstable sort over the whole
    # index, so a larger index re-rolls them); other stays use the all-stays
    # files. Each ECG feature reads only its own stay's rows, so the two groups
    # are built separately and recombined in (stay_id, hour) order.
    splits = load_splits()
    ours = np.isin(frame["stay_id"].to_numpy(dtype="int64"),
                   splits[split]["stay_id"].unique().astype("int64"))
    ecg_files = {True: (MEAS_CACHE, ECG_INDEX), False: (ECG_MEAS_ALL, ECG_INDEX_ALL)}
    agents = _load_agents()
    X_parts, ecg_parts, names = [], [], None
    for flag in (True, False):
        fg = frame[ours == flag].reset_index(drop=True)
        if not len(fg):
            continue
        zm = np.load(ecg_files[flag][0])
        ecg_index = pd.read_parquet(ecg_files[flag][1])
        ecg = {"values": zm["embeddings"].astype(np.float32), "study_ids": zm["study_ids"],
               "names": [str(n) for n in zm["feature_names"]]}
        if names is not None and ecg["names"] != names:
            raise SystemExit("the two ECG feature files name different features")
        names = ecg["names"]
        Xg, _, kg = build_allfeat_design(fg, {**src, "ecg": {"index": ecg_index, **ecg}})
        if not (_same(kg["stay_id"].to_numpy(dtype="int64"), fg["stay_id"].to_numpy(dtype="int64"))
                and _same(kg["hour"].to_numpy(dtype="int64"), fg["hour"].to_numpy(dtype="int64"))):
            raise SystemExit("design rows are not in frame order")
        X_parts.append(Xg.assign(stay_id=fg["stay_id"].to_numpy(), hour=fg["hour"].to_numpy()))
        agents["ecg_meas"]._cache = (ecg["values"], ecg["study_ids"], ecg_index)
        ecg_parts.append(agents["ecg_meas"].emit(fg))
    X = pd.concat(X_parts, ignore_index=True)
    order = np.lexsort((X["hour"].to_numpy(), X["stay_id"].to_numpy()))
    X = X.iloc[order].reset_index(drop=True)
    keys = X[["stay_id", "hour"]].astype("int64")
    if not (_same(keys["stay_id"], frame["stay_id"].to_numpy(dtype="int64"))
            and _same(keys["hour"], frame["hour"].to_numpy(dtype="int64"))):
        raise SystemExit("recombined design rows are not in frame order")
    keys = keys.assign(patient_id=frame["patient_id"].to_numpy())
    X = X.drop(columns=["stay_id", "hour"])
    cols = column_sets(names)["allfeat"]
    print(f"[{split}] design: {X.shape}; {int(ours.sum()):,} rows on our cohort's ECG files "
          f"({time.time() - t0:.0f} s)", flush=True)

    outs = {"news2": agents["news2"].emit(frame), "vitals_tcn": agents["vitals_tcn"].emit(frame)}
    agents["labs"].labs = src["labs"]
    outs["labs"] = agents["labs"].emit(frame)
    for n in ("cadence", "fluid", "resp"):
        agents[n].source = src[n]
        outs[n] = agents[n].emit(frame)
    ek = pd.concat([e.keys for e in ecg_parts], ignore_index=True)
    eo = np.lexsort((ek["hour"].to_numpy(), ek["stay_id"].to_numpy()))
    outs["ecg_meas"] = AgentOutput(keys=ek.iloc[eo].reset_index(drop=True),
                                   latent=np.concatenate([e.latent for e in ecg_parts])[eo],
                                   risk=np.concatenate([e.risk for e in ecg_parts])[eo])
    if not _same(outs["ecg_meas"].keys["stay_id"].to_numpy(dtype="int64"),
                 outs["labs"].keys["stay_id"].to_numpy(dtype="int64")):
        raise SystemExit("ECG output rows are not in the other agents' order")
    ctx = FusionContext(splits)
    Xf = X[list(variant_columns()["FAIR+TRAJ"])]
    if split == "train":
        fold = held_out_fold(frame["patient_id"].to_numpy(), splits["train"]["patient_id"].to_numpy())
        raw_gbm = np.full(len(Xf), np.nan)
        for k in (0, 1):
            b = xgb.Booster()
            b.load_model(FOLD_MODELS / f"gbm_agent_fold{k}.ubj")
            b.set_param({"device": "cuda"})      # fitted on the GPU
            m = fold == k
            raw_gbm[m] = gbm_predict(b, Xf[m])
    else:
        raw_gbm = _predict(_load_booster(T42_MODELS / "gbm" / "fair_traj.ubj"), Xf)
    outs["gbm"] = _gbm_agent_output(raw_gbm, keys, _gbm_agent_iso(ctx))
    print(f"[{split}] agents emitted ({time.time() - t0:.0f} s)", flush=True)

    # gate 1, agent part: on rows our cohort also has, the batched agents within
    # the rounding envelope, the others exactly equal to the T42 bank. Test
    # (pre-registered) uses the test envelope; the extra val/train check stops
    # only beyond rounding, i.e. beyond the agent's largest envelope over splits.
    E = json.loads(ENVELOPE.read_text())["agents"]
    if split == "test":
        env, keys_checked = E["test"], GATE_KEYS
    else:
        env = {n: {k: max(E[sp][n][k] for sp in SPLITS) for k in E["test"][n]} for n in BATCHED}
        if ENVELOPE_TCN.exists():
            T = json.loads(ENVELOPE_TCN.read_text())["agents"]
            env["vitals_tcn"] = {k: max(T[sp]["vitals_tcn"][k] for sp in SPLITS)
                                 for k in T["test"]["vitals_tcn"]}
        keys_checked = EXTRA_KEYS
    gate = {}
    for n in ALL8:
        ref = AgentOutput.load(T42_EMB / f"{n}_{split}.npz")
        pa, pb = shared_positions(outs[n].keys, ref.keys)
        d = _diff(outs[n].latent[pa], outs[n].risk[pa], ref.latent[pb], ref.risk[pb])
        bound = env.get(n, {k: 0.0 for k in ("share", "latent", "risk", "risk_ulps")})
        gate[n] = {"shared_rows": int(len(pa)), **d, "envelope": bound,
                   "within": all(d[k] <= bound[k] for k in keys_checked),
                   "within_absolute_risk": d["risk"] <= bound["risk"]}
    ok = all(g["within"] for g in gate.values())
    print(f"[{split}] gate 1 (agents, shared rows): {'PASS' if ok else 'FAIL'} "
          f"{ {n: (round(100 * g['share'], 2), g['latent'], g['risk_ulps'], g['within']) for n, g in gate.items()} }",
          flush=True)
    report = {"split": split, "checked": list(keys_checked),
              "rows": int(len(frame)), "stays": int(len(stays)),
              "patients": int(frame["patient_id"].nunique()),
              "design_columns": len(cols), "gate1_agents": gate, "gate1_agents_pass": ok,
              "seconds": round(time.time() - t0, 1)}
    if not ok:
        _write_json(WORK / f"agents_{split}_FAILED.json", report)
        raise SystemExit("GATE 1 FAILED: agent outputs fall outside the envelope on shared rows")

    EMB.mkdir(parents=True, exist_ok=True)
    for n in ALL8:
        _refuse(EMB / f"{n}_{split}.npz")
        outs[n].save(EMB / f"{n}_{split}.npz")
    frame[FRAME_COLS].to_parquet(WORK / f"frame_{split}.parquet", index=False)
    X[cols].astype(np.float32).to_parquet(WORK / f"allfeat_{split}.parquet", index=False)
    _write_json(report_path, report)
    print(f"[{split}] saved ({report['seconds']} s)")


# --- stage: shipped heads -----------------------------------------------------------

def _frames(splits=SPLITS):
    return {sp: pd.read_parquet(WORK / f"frame_{sp}.parquet") for sp in splits}


def _bank(splits):
    """One AgentBank over the named splits' T49 agent outputs."""
    from agents.base import AgentOutput
    from feed_outage import ALL8, bank_from_outputs
    outs = {}
    for n in ALL8:
        parts = [AgentOutput.load(EMB / f"{n}_{sp}.npz") for sp in splits]
        outs[n] = AgentOutput(keys=pd.concat([p.keys for p in parts], ignore_index=True),
                              latent=np.concatenate([p.latent for p in parts]),
                              risk=np.concatenate([p.risk for p in parts]))
    return bank_from_outputs(outs, list(ALL8))


def stage_shipped():
    from baselines_mimic import load_splits
    from fusion_common import FusionContext
    from orchestrator.fusion import Orchestrator
    from uncensored_frames import shared_positions
    _one_thread()
    report_path = WORK / "shipped.json"
    _refuse(report_path)
    if not ENVELOPE_HEADS.exists():
        raise SystemExit("run the envelope-heads stage first")
    t0 = time.time()
    frames = _frames(("val", "test"))
    banks = {sp: _bank((sp,)) for sp in ("val", "test")}
    ctx = FusionContext(load_splits())
    shared = {"val": shared_positions(frames["val"], ctx.val_ordered),
              "test": shared_positions(frames["test"], ctx.test_ordered)}
    SCORES.mkdir(parents=True, exist_ok=True)
    gate = {}
    for s in SEEDS:
        path = SCORES / f"shipped_s{s}.npz"
        _refuse(path)
        payload, ref = {}, np.load(T42_HEADS / f"all8_s{s}.npz")
        for h in ("T", "D"):
            head = Orchestrator.load_checkpoint(T42_MODELS / "heads" / f"all8_{h}_s{s}.pt",
                                                banks["val"])
            for sp in ("val", "test"):
                head.bank = banks[sp]
                payload[f"raw{sp}_{h}"] = head.raw_risk(frames[sp])
                pa, pb = shared[sp]
                gate[f"s{s}_{h}_{sp}"] = _raw_diff(payload[f"raw{sp}_{h}"][pa], ref[f"raw{sp}_{h}"][pb])
        np.savez_compressed(path, **payload)
        print(f"[shipped s{s}] {time.time() - t0:.0f} s; shared test rows "
              f"{ {k: v for k, v in gate.items() if k.startswith(f's{s}_') and k.endswith('_test')} }",
              flush=True)
    bound = json.loads(ENVELOPE_HEADS.read_text())["heads_max"]
    test_d = {k: v for k, v in gate.items() if k.endswith("_test")}
    ok_test = all(v["share"] <= bound["share"] and v["raw"] <= bound["raw"] for v in test_d.values())
    report = {"shared_rows": {sp: int(len(shared[sp][0])) for sp in shared},
              "head_envelope": bound, "gate1_heads_test_pass": ok_test,
              "test_max": {k: max(v[k] for v in test_d.values()) for k in ("share", "raw")},
              "val_max_diagnostic": {k: max(v[k] for kk, v in gate.items() if kk.endswith("_val"))
                                     for k in ("share", "raw")},
              "per_seed_head": gate, "seconds": round(time.time() - t0, 1)}
    _write_json(report_path if ok_test else WORK / "shipped_FAILED.json", report)
    print(f"[gate 1, heads] {'PASS' if ok_test else 'FAIL'} {report}")
    if not ok_test:
        raise SystemExit("GATE 1 FAILED: head scores fall outside the envelope on shared test rows")


# --- stage: retrained heads ---------------------------------------------------------

def _fit_job(seed):
    os.environ["OMP_NUM_THREADS"] = "1"
    _one_thread()
    from two_head import fit_head
    path = SCORES / f"retrained_s{seed}.npz"
    if path.exists():
        return seed, "cached", 0.0
    t0 = time.time()
    frames = _frames()
    bank = _bank(SPLITS)
    payload = {}
    for h, col in (("T", "label_t"), ("D", "label_d")):
        f_tr, f_va = frames["train"], frames["val"]
        head = fit_head(bank, {"train": f_tr, "val": f_va}, f_tr[col].to_numpy(dtype=float),
                        f_va[col].to_numpy(dtype=float), f_tr, f_va, seed, EPOCHS)
        payload[f"rawval_{h}"] = head.raw_risk(f_va)
        payload[f"rawtest_{h}"] = head.raw_risk(frames["test"])
        (MODELS / "heads").mkdir(parents=True, exist_ok=True)
        head.save_checkpoint(MODELS / "heads" / f"{h}_s{seed}.pt")
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)
    return seed, "fitted", round(time.time() - t0, 1)


def stage_heads(workers):
    import multiprocessing as mp
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("run heads with CUDA_VISIBLE_DEVICES= (CPU only)")
    frames = _frames(("train", "val"))
    gate2 = {sp: _same((f["label_t"] | f["label_d"]).to_numpy(), f["label"].to_numpy())
             for sp, f in frames.items()}
    print(f"[gate 2] the parts rebuild their label: {gate2}", flush=True)
    if not all(gate2.values()):
        raise SystemExit("GATE 2 FAILED")
    del frames
    SCORES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for seed, status, secs in pool.imap_unordered(_fit_job, SEEDS):
            print(f"[retrained s{seed}] {status} {secs} s ({time.time() - t0:.0f} s elapsed)",
                  flush=True)
    _write_json(WORK / "heads.json", {"gate2": gate2, "seeds": list(SEEDS), "epochs": EPOCHS,
                                      "seconds": round(time.time() - t0, 1)})


# --- stage: GBMs ----------------------------------------------------------------------

def _bridge(split, frame):
    """Bridge rows (stored in sample order) in frame order."""
    smp = pd.read_parquet(SAMPLES_CODE, columns=["stay_id", "hour", "our_split"])
    keys = smp[smp["our_split"] == split].reset_index(drop=True)
    order = np.lexsort((keys["hour"].to_numpy(), keys["stay_id"].to_numpy()))
    X = pd.read_parquet(WORK / f"bridge_{split}.parquet").iloc[order].reset_index(drop=True)
    if not (_same(keys["stay_id"].to_numpy()[order], frame["stay_id"].to_numpy())
            and _same(keys["hour"].to_numpy()[order], frame["hour"].to_numpy())):
        raise SystemExit("bridge rows do not match the frame")
    return X


def stage_gbms(device):
    from baselines_fair import tune_gbm
    out_dir = SCORES / "gbms"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _frames()
    y = {sp: frames[sp]["label"].to_numpy(dtype=int) for sp in SPLITS}
    t0 = time.time()
    for model in ("bridge", "reference"):
        if model == "bridge":
            X = {sp: _bridge(sp, frames[sp]) for sp in SPLITS}
        else:
            X = {sp: pd.read_parquet(WORK / f"allfeat_{sp}.parquet") for sp in SPLITS}
        for s in SEEDS:
            path = out_dir / f"{model}_s{s}.npz"
            if path.exists():
                print(f"[{model} s{s}] cached", flush=True)
                continue
            t1 = time.time()
            m, rec = tune_gbm(X["train"], y["train"], X["val"], y["val"], seed=s,
                              device=device, verbose=False)
            rec["fit_seconds"] = round(time.time() - t1, 1)
            np.savez_compressed(path, val=m.predict_proba(X["val"])[:, 1].astype(np.float64),
                                test=m.predict_proba(X["test"])[:, 1].astype(np.float64),
                                record_json=np.array(json.dumps(rec)))
            print(f"[{model} s{s}] depth {rec['max_depth']} spw {rec['scale_pos_weight']} "
                  f"{rec['n_trees']} trees, val AUPRC {rec['val_auprc']:.4f} "
                  f"({rec['fit_seconds']} s; {time.time() - t0:.0f} s elapsed)", flush=True)
        del X


# --- stage: eval ---------------------------------------------------------------------

PUBLISHED = {"multimodal": {"auroc": 0.7857, "auprc": 0.1908, "lift": 6.81},
             "structured_only": {"auroc": 0.7603, "auprc": 0.1371},
             "xgboost": {"auroc": 0.7486, "auprc": 0.1123},
             "operating_point": {"recall": 0.266, "precision": 0.251, "specificity": 0.977},
             "positive_rate": 0.028, "source": "arXiv 2603.14719v1 Tables 5-6 (preprint)"}
BRIDGE_GATE = (0.7286, 0.7686)
SPECIFICITY = 0.977
REPS = 1000


def _arm_scores(ctx, frames, fit):
    """{arm: {seed: (val, test)}} on the T49 rows."""
    from decompose import treatment_only_label
    from feed_outage import calibrated, noisy_or
    from sklearn.isotonic import IsotonicRegression
    fv, ft = frames["val"], frames["test"]
    y_ours = {"T": treatment_only_label(ctx.val_ordered),
              "D": ctx.val_ordered["label_24h_death"].to_numpy(dtype=float)}
    y_t49 = {"T": fv["label_t"].to_numpy(dtype=float), "D": fv["label_d"].to_numpy(dtype=float)}
    arms = {"shipped": {}, "retrained": {}, "bridge": {}, "reference": {}}
    for s in SEEDS:
        # shipped: the T42 composite exactly as shipped (calibrators from our V_fit)
        ref = np.load(T42_HEADS / f"all8_s{s}.npz")
        iso = {h: IsotonicRegression(out_of_bounds="clip").fit(
            ref[f"rawval_{h}"][ctx.fit_mask], y_ours[h][ctx.fit_mask]) for h in ("T", "D")}
        z = np.load(SCORES / f"shipped_s{s}.npz")
        arms["shipped"][s] = tuple(noisy_or(calibrated(iso["T"], z[f"raw{sp}_T"]),
                                            calibrated(iso["D"], z[f"raw{sp}_D"]))
                                   for sp in ("val", "test"))
        # retrained: calibrators on T49 V_fit rows, each head against its part
        z = np.load(SCORES / f"retrained_s{s}.npz")
        iso = {h: IsotonicRegression(out_of_bounds="clip").fit(
            z[f"rawval_{h}"][fit], y_t49[h][fit]) for h in ("T", "D")}
        arms["retrained"][s] = tuple(noisy_or(calibrated(iso["T"], z[f"raw{sp}_T"]),
                                              calibrated(iso["D"], z[f"raw{sp}_D"]))
                                     for sp in ("val", "test"))
        for m in ("bridge", "reference"):
            g = np.load(SCORES / "gbms" / f"{m}_s{s}.npz")
            arms[m][s] = (g["val"].astype(float), g["test"].astype(float))
    for a in arms.values():
        for v, t in a.values():
            if len(v) != len(fv) or len(t) != len(ft):
                raise SystemExit("score arrays do not match the frames")
    return arms


def _measures(arms, y_val, freeze, y, pid, workers, label):
    """Ranking, lift and the 97.7%-specificity operating point for every arm on
    one set of test rows, with ten-seed pooled CIs on shared resamples."""
    from metrics import auprc, auroc
    from paper_stats import ci, metric_draws
    from stacking import resample_counts
    from t49_metrics import patient_counts, rates, spec_threshold
    from config import SEED
    names = {f"{a}_s{s}": v[s][1] for a, v in arms.items() for s in SEEDS}
    pts = {m: {n: float(fn(y, v)) for n, v in names.items()}
           for m, fn in (("auprc", auprc), ("auroc", auroc))}
    t0 = time.time()
    draws = metric_draws({"auprc": ("auprc", y), "auroc": ("auroc", y)}, names, pid,
                         reps=REPS, workers=workers)
    if any(len(d) != REPS for t in draws.values() for d in t.values()):
        raise SystemExit("a resample was skipped")
    print(f"[{label}] bootstrap {len(names)} scores x {REPS} ({time.time() - t0:.0f} s)",
          flush=True)
    patients, base = patient_counts(y, np.zeros(len(y)), 1.0, pid)
    weights = [w for _, w in resample_counts(pid, REPS, SEED)]
    prev_point = float(np.mean(y))
    prev = np.array([rates(base, w)["positive_rate"] for w in weights])
    out = {"rows": int(len(y)), "positives": int(y.sum()), "positive_rate": prev_point,
           "patients": int(len(patients)), "arms": {}}
    for a, v in arms.items():
        r = {}
        for m in ("auroc", "auprc"):
            vals = [pts[m][f"{a}_s{s}"] for s in SEEDS]
            dr = np.mean([draws[m][f"{a}_s{s}"] for s in SEEDS], axis=0)
            r[m] = {"mean": float(np.mean(vals)), "min": float(min(vals)),
                    "max": float(max(vals)), "ci": list(ci(dr))}
            if m == "auprc":
                lift = dr / prev
                r["lift"] = {"mean": float(np.mean(vals)) / prev_point, "ci": list(ci(lift))}
        ops = {}
        for where in ("v_freeze", "test"):
            per = []
            for s in SEEDS:
                sv, st = v[s]
                neg = (sv[freeze][y_val[freeze] == 0] if where == "v_freeze"
                       else st[y == 0])
                thr = spec_threshold(neg, SPECIFICITY)
                _, c = patient_counts(y, st, thr, pid)
                per.append((rates(c), [rates(c, w) for w in weights]))
            op = {}
            for k in ("recall", "precision", "specificity"):
                dr = np.mean([[d[k] for d in p[1]] for p in per], axis=0)
                op[k] = {"mean": float(np.mean([p[0][k] for p in per])), "ci": list(ci(dr))}
            ops[f"threshold_on_{where}"] = op
        r["at_97.7_specificity"] = ops
        out["arms"][a] = r
        print(f"[{label}] {a}: AUROC {r['auroc']['mean']:.4f} {r['auroc']['ci']} | AUPRC "
              f"{r['auprc']['mean']:.4f} | lift {r['lift']['mean']:.2f} {r['lift']['ci']} | "
              f"recall@97.7 (V_freeze thr) {ops['threshold_on_v_freeze']['recall']['mean']:.3f}",
              flush=True)
    return out


def stage_eval(workers):
    from baselines_mimic import load_splits
    from fusion_common import FusionContext
    t0 = time.time()
    out_path = CACHE / "t49_eval.json"
    _refuse(out_path)
    ctx = FusionContext(load_splits())
    frames = _frames(("val", "test"))
    fv, ft = frames["val"], frames["test"]
    fit_patients = set(ctx.val_ordered["patient_id"].to_numpy()[ctx.fit_mask].tolist())
    fit = fv["patient_id"].isin(fit_patients).to_numpy()
    if not fv["patient_id"].isin(set(ctx.val_ordered["patient_id"])).all():
        raise SystemExit("a T49 validation patient is outside our validation split")
    test_patients = set(ctx.test_ordered["patient_id"])
    if not ft["patient_id"].isin(test_patients).all():
        raise SystemExit("a scored patient is outside our test split")
    arms = _arm_scores(ctx, frames, fit)
    y_val = fv["label"].to_numpy(dtype=int)
    res = {"test": "T49 Sadanandan 2026 (preprint) cohort, label and samples",
           "seeds": list(SEEDS), "reps": REPS, "published": PUBLISHED,
           "bridge_gate": list(BRIDGE_GATE)}
    y, pid = ft["label"].to_numpy(dtype=int), ft["patient_id"].to_numpy()
    res["all_rows"] = _measures(arms, y_val, ~fit, y, pid, workers, "all rows")
    keep_t, keep_v = ft["started"].to_numpy() == 0, fv["started"].to_numpy() == 0
    sub = {a: {s: (v[0][keep_v], v[1][keep_t]) for s, v in d.items()} for a, d in arms.items()}
    res["not_started_rows"] = _measures(sub, y_val[keep_v], (~fit)[keep_v], y[keep_t],
                                        pid[keep_t], workers, "not started")
    b = res["all_rows"]["arms"]["bridge"]["auroc"]["mean"]
    comparable = bool(BRIDGE_GATE[0] <= b <= BRIDGE_GATE[1])
    sh = res["all_rows"]["arms"]["shipped"]
    res["verdicts"] = {
        "bridge_auroc_ten_seed_mean": b, "comparable": comparable,
        "primary_auroc_ci_lower_above_0.7857": bool(sh["auroc"]["ci"][0] > 0.7857),
        "primary_lift_ci_lower_above_6.81": bool(sh["lift"]["ci"][0] > 6.81),
        "pass": bool(comparable and sh["auroc"]["ci"][0] > 0.7857 and sh["lift"]["ci"][0] > 6.81)}
    res["note"] = ("aggregate-only; scored rows are the cohort's patients in our test split; "
                   "the threshold row 'test' mirrors their Table 6 and is description only")
    res["seconds"] = round(time.time() - t0, 1)
    _write_json(out_path, res)
    print(f"[verdict] {res['verdicts']}")
    print(f"[saved] {out_path.relative_to(REPO)} ({res['seconds']} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["envelope", "envelope-tcn", "envelope-heads", "agents",
                                      "shipped", "heads", "gbms", "eval"])
    ap.add_argument("--split", choices=SPLITS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.stage == "envelope":
        stage_envelope()
    elif args.stage == "envelope-tcn":
        stage_envelope_tcn()
    elif args.stage == "envelope-heads":
        stage_envelope_heads()
    elif args.stage == "agents":
        if not args.split:
            ap.error("agents needs --split")
        stage_agents(args.split)
    elif args.stage == "shipped":
        stage_shipped()
    elif args.stage == "heads":
        stage_heads(args.workers)
    elif args.stage == "gbms":
        stage_gbms(args.device)
    else:
        stage_eval(args.workers)


if __name__ == "__main__":
    main()
