"""T32 (S1): the flagship without its clinician-behaviour agent.

The two-head flagship rebuilt on seven agents (ALL-8 less cadence, the shipped
MIMIC agent caches), at seeds 42 and 1-9, against the three published recipes
and the no-cadence all-features GBM (T30).

    OMP_NUM_THREADS=1 .venv/bin/python scripts/t32_no_cadence.py --stage heads --parallel 2
    .venv/bin/python scripts/t32_no_cadence.py --stage gbm --device cuda
    OMP_NUM_THREADS=1 .venv/bin/python scripts/t32_no_cadence.py --stage eval

``heads`` runs two_head.py's per-seed construction (``fit_head`` per head,
isotonic on V_fit, ``_strictify``, noisy-OR) in worker processes on the CPU
with one thread each. It also fits ALL-8 at seed 42, which must equal
``two_head.npz`` element for element (the gate). Each job writes
``data/cache/t32_heads/<arm>_s<seed>.npz`` with a fingerprint.

``gbm`` fits one descriptive arm: the no-cadence all-features GBM without its
36 measurement-timing columns (``src/measurement_timing.py``), seed 42, to
``data/cache/t32_gbm_notiming.npz``.

``eval`` writes aggregates only to ``data/cache/t32_no_cadence.json``.
"""

import argparse
import hashlib
import json
import os
import subprocess
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
JOB_DIR = CACHE / "t32_heads"
GBM_CACHE = CACHE / "t32_gbm_notiming.npz"
OUT = CACHE / "t32_no_cadence.json"
SHIPPED = CACHE / "two_head.npz"
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
COMPARATORS = ("ecart", "yaib", "sadanandan", "allfeat_nocad")
EPOCHS = 8
REPS = 1000
HEAD_CODE = ["scripts/two_head.py", "src/distill_lead.py", "src/orchestrator/fusion.py",
             "src/fusion_common.py", "src/fusion_ladder.py", "src/decompose.py",
             "src/agents/base.py"]


def log(msg):
    print(msg, flush=True)


def agents_for(arm):
    from two_head import ALL8
    if arm == "all8":
        return tuple(ALL8)
    if arm == "all7":
        return tuple(a for a in ALL8 if a != "cadence")
    raise ValueError(arm)


def _hash_files(paths):
    out = {}
    for p in paths:
        p = Path(p)
        if p.exists():
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def head_fingerprint(arm, seed):
    import torch
    from orchestrator.fusion import EMB_DIR
    banks = []
    for a in agents_for(arm):
        for split in ("train", "val", "test"):
            st = (Path(EMB_DIR) / f"{a}_{split}.npz").stat()
            banks.append([f"{a}_{split}", st.st_size, st.st_mtime_ns])
    spec = {"arm": arm, "agents": list(agents_for(arm)), "seed": seed,
            "epochs": EPOCHS, "torch": torch.__version__,
            "code": _hash_files(HEAD_CODE), "banks": banks}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


# --- heads ----------------------------------------------------------------------

def worker(arm, seed):
    """two_head.py's per-seed loop for one agent set, one seed."""
    import torch
    torch.set_num_threads(1)
    from baselines_mimic import load_splits
    from decompose import treatment_only_label
    from fusion_common import FusionContext
    from fusion_ladder import _strictify
    from orchestrator.fusion import AgentBank
    from sklearn.isotonic import IsotonicRegression
    from two_head import fit_head

    out_path = JOB_DIR / f"{arm}_s{seed}.npz"
    if out_path.exists():
        return
    t0 = time.time()
    splits = load_splits()
    ctx = FusionContext(splits)
    train_o = splits["train"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    val_o = ctx.val_ordered
    y_death = {"train": train_o["label_24h_death"].to_numpy(dtype=float),
               "val": val_o["label_24h_death"].to_numpy(dtype=float)}
    y_treat = {"train": treatment_only_label(train_o),
               "val": treatment_only_label(val_o)}
    bank = AgentBank.load(agent_names=agents_for(arm))
    heads = {}
    for name, y in (("T", y_treat), ("D", y_death)):
        head = fit_head(bank, splits, y["train"], y["val"], train_o, val_o, seed, EPOCHS)
        raw_val = head.raw_risk(splits["val"])
        raw_test = head.raw_risk(splits["test"])
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            raw_val[ctx.fit_mask], y["val"][ctx.fit_mask])
        heads[name] = {"val": _strictify(iso.transform(raw_val), raw_val),
                       "test": _strictify(iso.transform(raw_test), raw_test)}
    s_val = 1.0 - (1.0 - heads["T"]["val"]) * (1.0 - heads["D"]["val"])
    s_test = 1.0 - (1.0 - heads["T"]["test"]) * (1.0 - heads["D"]["test"])
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, val=s_val, test=s_test, dhead_test=heads["D"]["test"],
                        stay_id_test=ctx.test_ordered["stay_id"].to_numpy(),
                        hour_test=ctx.test_ordered["hour"].to_numpy(),
                        fingerprint=np.array(head_fingerprint(arm, seed)))
    tmp.replace(out_path)
    log(f"[worker] {arm} seed {seed} done in {time.time() - t0:.0f} s")


def run_jobs(jobs, parallel):
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs if not (JOB_DIR / f"{j[0]}_s{j[1]}.npz").exists()]
    running, t0 = [], time.time()
    while pending or running:
        while pending and len(running) < parallel:
            arm, seed = pending.pop(0)
            lf = open(JOB_DIR / f"{arm}_s{seed}.log", "w")
            p = subprocess.Popen([sys.executable, __file__, "--worker", "--arm", arm,
                                  "--seed", str(seed)], env=env, stdout=lf,
                                 stderr=subprocess.STDOUT)
            running.append((p, arm, seed, lf))
        time.sleep(5)
        for item in list(running):
            p, arm, seed, lf = item
            if p.poll() is None:
                continue
            lf.close()
            running.remove(item)
            if p.returncode != 0:
                for q, *_ in running:
                    q.terminate()
                raise SystemExit(f"{arm} seed {seed} failed; see its log in {JOB_DIR}")
            log(f"[heads] {arm} seed {seed} done ({time.time() - t0:.0f} s elapsed)")


def gate_all8():
    """ALL-8 at seed 42 must equal the shipped flagship element for element."""
    z, c = np.load(JOB_DIR / "all8_s42.npz"), np.load(SHIPPED)
    out = {}
    for split in ("val", "test"):
        a, b = z[split], c[f"{split}_s42"]
        out[split] = {"identical": bool(np.array_equal(a, b)),
                      "max_abs_diff": float(np.abs(a - b).max())}
    return out


def stage_heads(parallel):
    run_jobs([("all8", 42)], 1)
    g = gate_all8()
    log(f"[gate] ALL-8 seed 42 vs two_head.npz: {g}")
    if not (g["val"]["identical"] and g["test"]["identical"]):
        raise SystemExit("ALL-8 seed 42 does not reproduce two_head.npz; stopping")
    run_jobs([("all7", s) for s in SEEDS], parallel)


# --- descriptive GBM arm ---------------------------------------------------------

def gbm_fingerprint(cols, ctx, device):
    import xgboost
    from agents.ecg_agent import ECG_INDEX
    from agents.ecg_meas_agent import MEAS_CACHE
    from baselines_mimic import OUT_PATH
    from modality_features import CADENCE_PATH, FLUIDS_PATH, LABS_PATH, RESP_PATH
    files = [[str(p), Path(p).stat().st_size, Path(p).stat().st_mtime_ns]
             for p in (OUT_PATH, LABS_PATH, CADENCE_PATH, FLUIDS_PATH, RESP_PATH,
                       MEAS_CACHE, ECG_INDEX)]
    h = hashlib.sha256()
    for a in (ctx.val_ordered["stay_id"].to_numpy(), ctx.val_ordered["hour"].to_numpy(),
              ctx.fit_mask, ctx.test_ordered["stay_id"].to_numpy(),
              ctx.test_ordered["hour"].to_numpy()):
        h.update(np.ascontiguousarray(a).tobytes())
    spec = {"cols": cols, "seed": 42, "device": device, "xgboost": xgboost.__version__,
            "code": _hash_files(["src/allfeat_gbm.py", "src/baselines_fair.py",
                                 "src/modality_features.py", "src/measurement_timing.py",
                                 "src/agents/ecg_agent.py", "src/mimic_cohort.py"]),
            "sources": files, "keys": h.hexdigest()}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def stage_gbm(device):
    from allfeat_gbm import build_allfeat_design, column_sets, load_sources
    from baselines_fair import tune_gbm
    from baselines_mimic import load_splits
    from fusion_common import FusionContext
    from fusion_ladder import _strictify
    from measurement_timing import measurement_timing_columns

    splits = load_splits()
    ctx = FusionContext(splits)
    sources = load_sources()
    nocad = column_sets(sources["ecg"]["names"])["allfeat_nocad"]
    timing = set(measurement_timing_columns(nocad))
    cols = [c for c in nocad if c not in timing]
    fp = gbm_fingerprint(cols, ctx, device)
    if GBM_CACHE.exists():
        if str(np.load(GBM_CACHE)["fingerprint"]) == fp:
            log("[gbm] cache is current")
            return
        raise SystemExit(f"{GBM_CACHE.name} exists with a different fingerprint")
    log(f"[gbm] {len(nocad)} no-cadence columns less {len(timing)} timing = {len(cols)}")
    d = {s: build_allfeat_design(splits[s], sources) for s in ("train", "val", "test")}
    model, rec = tune_gbm(d["train"][0][cols], d["train"][1], d["val"][0][cols],
                          d["val"][1], seed=42, device=device)
    raw_va = model.predict_proba(d["val"][0][cols])[:, 1].astype(float)
    raw_te = model.predict_proba(d["test"][0][cols])[:, 1].astype(float)
    iso = ctx.fit_calibrator(raw_va)
    np.savez_compressed(
        GBM_CACHE, val=_strictify(iso.transform(raw_va), raw_va),
        test=_strictify(iso.transform(raw_te), raw_te), rawtest=raw_te,
        val_stay_id=d["val"][2]["stay_id"].to_numpy(), val_hour=d["val"][2]["hour"].to_numpy(),
        test_stay_id=d["test"][2]["stay_id"].to_numpy(), test_hour=d["test"][2]["hour"].to_numpy(),
        fingerprint=np.array(fp),
        meta_json=np.array(json.dumps({"record": rec, "n_columns": len(cols), "device": device,
                                       "dropped": sorted(timing)})))
    log(f"[gbm] {rec} -> {GBM_CACHE.name}")


# --- evaluation --------------------------------------------------------------------

def _sweep_row(ctx, val, test, thr=None, budget=None):
    from demo_bundle import first_alarm_table
    from paper_operating_points import (after_first_hour_catch, catch_stats,
                                        freeze_quantile, test_rate)
    if thr is None:
        thr = freeze_quantile(ctx, val, budget)
    table = first_alarm_table(ctx.test_ordered, {"m": test}, {"m": thr})
    return {**catch_stats(table, "m"),
            "catch_after_h0": after_first_hour_catch(ctx, test, thr),
            "test_alarm_rate": test_rate(ctx, test, thr)}


def stage_eval(workers):
    from baselines_mimic import load_splits
    from config import SEED
    from demo_bundle import assert_aggregate_only
    from fusion_common import FusionContext
    from lit_gbm import load_lit_scores
    from metrics import auprc, auroc
    from paper_operating_points import SWEEP_BUDGETS
    from paper_stats import metric_draws, paired_delta, pooled_delta

    t0 = time.time()
    ctx = FusionContext(load_splits())
    gate = gate_all8()
    log(f"[gate] ALL-8 seed 42 vs two_head.npz: {gate}")
    if not (gate["val"]["identical"] and gate["test"]["identical"]):
        raise SystemExit("gate failed")

    all7, dh7 = {}, {}
    for s in SEEDS:
        z = np.load(JOB_DIR / f"all7_s{s}.npz")
        if str(z["fingerprint"]) != head_fingerprint("all7", s):
            raise SystemExit(f"all7 seed {s}: fingerprint does not match the inputs")
        if not (z["stay_id_test"] == ctx.test_ordered["stay_id"].to_numpy()).all():
            raise SystemExit(f"all7 seed {s} is not aligned with test")
        all7[s] = {"val": z["val"].astype(float), "test": z["test"].astype(float)}
        dh7[s] = z["dhead_test"].astype(float)
    zt = np.load(SHIPPED)
    all8 = {s: {"val": zt[f"val_s{s}"].astype(float), "test": zt[f"test_s{s}"].astype(float)}
            for s in SEEDS}
    lit = load_lit_scores(ctx.val_ordered, ctx.test_ordered)
    za = np.load(CACHE / "allfeat_gbm_scores.npz")
    for side, ref in (("val", ctx.val_ordered), ("test", ctx.test_ordered)):
        assert (za[f"{side}_stay_id"] == ref["stay_id"].to_numpy()).all()
    zg = np.load(GBM_CACHE)
    gmeta = json.loads(str(zg["meta_json"]))
    if str(zg["fingerprint"]) != gbm_fingerprint(_notiming_cols(), ctx, gmeta["device"]):
        raise SystemExit(f"{GBM_CACHE.name}: fingerprint does not match the inputs")
    comps = {n: {"val": lit[n][0], "test": lit[n][1]} for n in ("ecart", "yaib", "sadanandan")}
    comps["allfeat_nocad"] = {"val": za["val_allfeat_nocad_s42"].astype(float),
                              "test": za["test_allfeat_nocad_s42"].astype(float)}
    notiming = {"val": zg["val"].astype(float), "test": zg["test"].astype(float)}

    y = ctx.y_test
    y_death = ctx.test_ordered["label_24h_death"].to_numpy(dtype=float)
    named = {f"all7_s{s}": all7[s]["test"] for s in SEEDS}
    named.update({f"all8_s{s}": all8[s]["test"] for s in SEEDS})
    named.update({n: c["test"] for n, c in comps.items()})
    named["allfeat_nocad_notiming"] = notiming["test"]
    targets = {"auprc": ("auprc", y), "auroc": ("auroc", y), "death_auprc": ("auprc", y_death)}
    fns = {"auprc": (auprc, y), "auroc": (auroc, y), "death_auprc": (auprc, y_death)}
    points = {t: {n: float(f(yy, s)) for n, s in named.items()} for t, (f, yy) in fns.items()}
    points["death_auprc"].update({f"all7_dhead_s{s}": float(auprc(y_death, dh7[s])) for s in SEEDS})
    t1 = time.time()
    draws = metric_draws(targets, named, ctx.patient_test, reps=REPS, seed=SEED, workers=workers)
    n_used = len(draws["auprc"]["all7_s42"])
    log(f"[bootstrap] {n_used} shared resamples x {len(named)} scores ({time.time() - t1:.0f} s)")

    def pooled_vs(t, opp):
        per = {s: paired_delta(points[t][f"all7_s{s}"], points[t][opp],
                               draws[t][f"all7_s{s}"], draws[t][opp]) for s in SEEDS}
        return {"pooled": pooled_delta([points[t][f"all7_s{s}"] for s in SEEDS], points[t][opp],
                                       [draws[t][f"all7_s{s}"] for s in SEEDS], draws[t][opp]),
                "seeds_ci_above_zero": int(sum(p["ci_above_zero"] for p in per.values())),
                "seeds_ci_below_zero": int(sum(p["ci"][1] < 0 for p in per.values())),
                "per_seed": per}

    ranking = {}
    for t in targets:
        ranking[t] = {"points": points[t], "vs": {o: pooled_vs(t, o) for o in
                                                  (*COMPARATORS, "allfeat_nocad_notiming")}}
        # Cadence's cost: ALL-8 seed s minus ALL-7 seed s.
        ranking[t]["all8_minus_all7"] = pooled_delta(
            [points[t][f"all8_s{s}"] for s in SEEDS], [points[t][f"all7_s{s}"] for s in SEEDS],
            [draws[t][f"all8_s{s}"] for s in SEEDS], [draws[t][f"all7_s{s}"] for s in SEEDS])
        for o in COMPARATORS:
            p = ranking[t]["vs"][o]["pooled"]
            log(f"[{t}] all7 - {o:<14} {p['delta']:+.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]")

    sweep = []
    t = ctx.test_ordered
    n_stays = int(t["stay_id"].nunique())
    n_event = int(t.groupby("stay_id")["event_h"].first().notna().sum())

    def precision(x):
        c, q = x["catch"] * n_event, x["quiet_fp"] * (n_stays - n_event)
        return float(c / (c + q)) if c + q else float("nan")

    for b in SWEEP_BUDGETS:
        row = {"budget": b, "comparators": {}, "all7_by_seed": {}, "all8_by_seed": {},
               "all7_at_comparator_test_volume": {}}
        for n, c in {**comps, "allfeat_nocad_notiming": notiming}.items():
            row["comparators"][n] = _sweep_row(ctx, c["val"], c["test"], budget=b)
            row["comparators"][n]["stay_precision"] = precision(row["comparators"][n])
        for s in SEEDS:
            row["all7_by_seed"][s] = _sweep_row(ctx, all7[s]["val"], all7[s]["test"], budget=b)
            row["all8_by_seed"][s] = _sweep_row(ctx, all8[s]["val"], all8[s]["test"], budget=b)
        for n in COMPARATORS:
            rate = row["comparators"][n]["test_alarm_rate"]
            per = {}
            for s in SEEDS:
                # Diagnostic, not protocol: threshold from TEST score quantiles.
                thr = float(np.quantile(all7[s]["test"][ctx.alertable_test], 1.0 - rate))
                per[s] = _sweep_row(ctx, None, all7[s]["test"], thr=thr)
            row["all7_at_comparator_test_volume"][n] = {
                k: float(np.mean([per[s][k] for s in SEEDS]))
                for k in ("catch", "catch_6h", "catch_after_h0", "quiet_fp", "test_alarm_rate")}
            row["all7_at_comparator_test_volume"][n]["stay_precision"] = float(
                np.mean([precision(per[s]) for s in SEEDS]))
        c7 = [row["all7_by_seed"][s]["catch"] for s in SEEDS]
        c8 = [row["all8_by_seed"][s]["catch"] for s in SEEDS]
        row["summary"] = {
            "all7_mean": float(np.mean(c7)), "all7_min": float(min(c7)), "all7_max": float(max(c7)),
            "all8_mean": float(np.mean(c8)),
            "all7_rate_mean": float(np.mean([row["all7_by_seed"][s]["test_alarm_rate"] for s in SEEDS])),
            "all7_stay_precision_mean": float(np.mean([precision(row["all7_by_seed"][s]) for s in SEEDS])),
            "all7_mean_beats": {n: bool(np.mean(c7) > row["comparators"][n]["catch"]) for n in COMPARATORS},
            "all7_seeds_ahead": {n: int(sum(x > row["comparators"][n]["catch"] for x in c7))
                                 for n in COMPARATORS},
        }
        sweep.append(row)
        log(f"[sweep] {b:>5.1%}: all7 {np.mean(c7):.1%} [{min(c7):.1%}, {max(c7):.1%}] | all8 "
            f"{np.mean(c8):.1%} | " + " ".join(
                f"{n} {row['comparators'][n]['catch']:.1%}" for n in COMPARATORS))

    auprc_ok = {o: ranking["auprc"]["vs"][o]["pooled"]["ci_above_zero"] for o in COMPARATORS}
    catch_ok = {o: all(r["summary"]["all7_mean_beats"][o] for r in sweep) for o in COMPARATORS}
    passed = all(auprc_ok.values()) and all(catch_ok.values())
    payload = {
        "test": "T32", "seeds": list(SEEDS), "reps": REPS, "resamples_used": n_used,
        "gate": gate, "notiming_meta": gmeta,
        "n_test_stays": n_stays, "n_event_stays": n_event,
        "ranking": ranking, "sweep": sweep,
        "pass_parts": {"auprc_ci_above_zero": auprc_ok, "catch_more_every_budget": catch_ok},
        "pass": passed,
        "pass_rule": "vs each of eCART-style, YAIB, Sadanandan and the no-cadence GBM: "
                     "pooled dAUPRC (all7 - comparator) CI above zero AND the ten-seed mean "
                     "catch above the comparator's at all seven budgets (protocol thresholds)",
        "note": "aggregate-only; equal-volume rows use test score quantiles (diagnostic)",
    }
    payload = json.loads(json.dumps(payload, default=float))
    assert_aggregate_only(payload)
    OUT.write_text(json.dumps(payload, indent=1))
    log(f"[verdict] AUPRC {auprc_ok}; catch {catch_ok} -> {'PASS' if passed else 'FAIL'}")
    log(f"[saved] {OUT} ({time.time() - t0:.0f} s)")


def _notiming_cols():
    from agents.ecg_meas_agent import MEAS_CACHE
    from allfeat_gbm import column_sets
    from measurement_timing import measurement_timing_columns
    names = [str(n) for n in np.load(MEAS_CACHE)["feature_names"]]
    nocad = column_sets(names)["allfeat_nocad"]
    timing = set(measurement_timing_columns(nocad))
    return [c for c in nocad if c not in timing]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("heads", "gbm", "eval"))
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm", choices=("all7", "all8"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.worker:
        worker(args.arm, args.seed)
    elif args.stage == "heads":
        stage_heads(args.parallel)
    elif args.stage == "gbm":
        stage_gbm(args.device)
    elif args.stage == "eval":
        stage_eval(args.workers)
    else:
        ap.error("give --stage or --worker")


if __name__ == "__main__":
    main()
