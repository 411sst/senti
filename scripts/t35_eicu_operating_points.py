"""T35 (A3 + L2): operating points on eICU, every model trained on MIMIC only.

Models, all zero-shot on eICU:
    flagship      two-head ALL-7 (no cadence; eICU has no cadence mapping),
                  ten seeds, cached as data/cache/eicu_twohead_all7{,_s1..s9}.npz
    ecart, yaib, sadanandan
                  the three published recipes, data/cache/eicu_lit_scores.npz
    news2         the cohort NEWS2 score
    allfeat_nocad the no-cadence all-features GBM (T30, seed 42), refit here on
                  MIMIC with T30's exact recipe and scored on eICU features
                  built by the same builders (src/eicu_allfeat.py)

Split: ``eicu_finetune.assign_splits(by="hospital", fracs=(0.6, 0.2, 0.2))``,
giving 30 val and 49 test hospitals. Train hospitals are not used.
Thresholds: each model's own (1 - b) quantile on the val hospitals' hours;
evaluation on the test hospitals. Everything is done twice: on regularly
charted hours (all five vitals measured in the previous 2 h) and on all hours.
In each hour set, thresholds and alarms both use only that set's hours.

    .venv/bin/python scripts/t35_eicu_operating_points.py --stage gbm --device cuda
    OMP_NUM_THREADS=1 .venv/bin/python scripts/t35_eicu_operating_points.py --stage eval

Outputs (new files): data/cache/t35_eicu_gbm.npz (eICU val/test raw GBM
scores, fingerprinted) and data/cache/t35_eicu_operating_points.json
(aggregates only).
"""

import argparse
import hashlib
import inspect
import json
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
import pandas as pd  # noqa: E402

CACHE = REPO / "data" / "cache"
EICU_COHORT = CACHE / "eicu_cohort_hourly.parquet"
FLAGSHIP = {42: CACHE / "eicu_twohead_all7.npz",
            **{s: CACHE / f"eicu_twohead_all7_s{s}.npz" for s in range(1, 10)}}
LIT = CACHE / "eicu_lit_scores.npz"
T30_SCORES = CACHE / "allfeat_gbm_scores.npz"
GBM_OUT = CACHE / "t35_eicu_gbm.npz"
OUT = CACHE / "t35_eicu_operating_points.json"
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
SEED_GBM = 42
LIT_NAMES = ("ecart", "yaib", "sadanandan")
COMPARATORS = LIT_NAMES + ("news2", "allfeat_nocad")
VITALS = ("heart_rate", "resp_rate", "spo2", "sbp", "temperature")
REGULAR_BOUND_H = 2.0
SWEEP_BUDGETS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
CI_BUDGETS = (0.01, 0.02, 0.05)
EXPECTED_HOSPITALS = {"val": 30, "test": 49}
REPS = 1000
GATE_AUPRC_TOL = 5e-4
CODE = ["src/allfeat_gbm.py", "src/eicu_allfeat.py", "src/baselines_fair.py",
        "src/modality_features.py", "src/agents/ecg_agent.py", "src/mimic_cohort.py"]


def log(msg):
    print(msg, flush=True)


def eicu_frames():
    """``{val, test}`` eICU frames sorted by (stay_id, hour), plus the split report."""
    from eicu_finetune import assign_splits
    cols = (["patient_id", "stay_id", "hour", "hospitalid", "news2", "event_h",
             "event_type", "label_24h"] + [f"{v}_hrs_since" for v in VITALS])
    cohort = pd.read_parquet(EICU_COHORT, columns=cols)
    cohort, report = assign_splits(cohort, by="hospital", fracs=(0.6, 0.2, 0.2))
    out = {}
    for name in ("val", "test"):
        f = cohort[cohort["split"] == name].sort_values(["stay_id", "hour"])
        out[name] = f.reset_index(drop=True)
        n_h = out[name]["hospitalid"].nunique()
        if n_h != EXPECTED_HOSPITALS[name]:
            raise SystemExit(f"eICU {name} has {n_h} hospitals, expected "
                             f"{EXPECTED_HOSPITALS[name]}")
    return out, report


def regular_mask(frame):
    m = np.ones(len(frame), dtype=bool)
    for v in VITALS:
        m &= frame[f"{v}_hrs_since"].to_numpy(dtype=float) <= REGULAR_BOUND_H
    return m


def positions(stay, hour, frame):
    """Row positions of ``frame``'s (stay_id, hour) keys in a cached score array."""
    idx = pd.MultiIndex.from_arrays([np.asarray(stay), np.asarray(hour)])
    pos = idx.get_indexer(pd.MultiIndex.from_arrays(
        [frame["stay_id"].to_numpy(), frame["hour"].to_numpy()]))
    if (pos < 0).any():
        raise SystemExit(f"{int((pos < 0).sum())} eICU rows have no cached score")
    return pos


# --- stage: gbm ----------------------------------------------------------------------

def gbm_fingerprint(device):
    import xgboost
    from baselines_fair import tune_gbm
    files = [[str(p), Path(p).stat().st_size, Path(p).stat().st_mtime_ns]
             for p in (EICU_COHORT, "data/cache/eicu_labs_hourly.parquet",
                       "data/cache/eicu_fluids_hourly.parquet",
                       "data/cache/eicu_resp_hourly.parquet", T30_SCORES)]
    spec = {"arm": "allfeat_nocad", "seed": SEED_GBM, "device": device,
            "xgboost": xgboost.__version__,
            "tune_gbm": hashlib.sha256(inspect.getsource(tune_gbm).encode()).hexdigest(),
            "code": {c: hashlib.sha256(Path(c).read_bytes()).hexdigest() for c in CODE},
            "t30_fingerprint": str(np.load(T30_SCORES)["fingerprint"]), "files": files}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def stage_gbm(device):
    from allfeat_gbm import build_allfeat_design, column_sets, load_sources
    from baselines_fair import tune_gbm
    from baselines_mimic import load_splits
    from eicu_allfeat import EICU_FLUIDS, EICU_LABS, EICU_RESP, eicu_sources
    from metrics import auprc
    from modality_features import load_fluids, load_labs, load_resp

    fp = gbm_fingerprint(device)
    if GBM_OUT.exists():
        if str(np.load(GBM_OUT)["fingerprint"]) == fp:
            log(f"[cache] {GBM_OUT.name} is current")
            return
        raise SystemExit(f"{GBM_OUT.name} exists with a different fingerprint")
    t0 = time.time()
    splits = load_splits()
    sources = load_sources()
    cols = column_sets(sources["ecg"]["names"])["allfeat_nocad"]
    design = {}
    for s in ("train", "val", "test"):
        X, y, keys = build_allfeat_design(splits[s], sources)
        design[s] = (X[cols].copy(), y, keys)
        del X
        log(f"[mimic] {s}: {len(keys):,} rows x {len(cols)} ({time.time() - t0:.0f} s)")
    X_tr, y_tr, _ = design.pop("train")
    X_va, y_va, _ = design["val"]
    model, record = tune_gbm(X_tr, y_tr, X_va, y_va, seed=SEED_GBM, device=device)
    del X_tr
    raw_te = model.predict_proba(design["test"][0])[:, 1].astype(float)
    t30 = np.load(T30_SCORES)
    ref = t30["rawtest_allfeat_nocad_s42"].astype(float)
    y_te = design["test"][1]
    gate = {"max_abs_diff": float(np.abs(raw_te - ref).max()),
            "identical": bool(np.array_equal(raw_te, ref)),
            "auprc_refit": float(auprc(y_te, raw_te)), "auprc_t30": float(auprc(y_te, ref))}
    log(f"[gate] refit vs T30 on MIMIC test: {gate}")
    if abs(gate["auprc_refit"] - gate["auprc_t30"]) > GATE_AUPRC_TOL:
        raise SystemExit("the refit does not reproduce T30's no-cadence GBM; stopping")
    del design

    frames, _ = eicu_frames()
    mimic_ecg_sources = {"ecg": sources["ecg"], "cadence": sources["cadence"]}
    labs, fluid, resp = load_labs(EICU_LABS), load_fluids(EICU_FLUIDS), load_resp(EICU_RESP)
    del sources
    payload = {"fingerprint": np.array(fp)}
    for s in ("val", "test"):
        f = frames[s]
        src = eicu_sources(mimic_ecg_sources, labs, fluid, resp, f["stay_id"].unique())
        cohort = pd.read_parquet(EICU_COHORT)
        cohort = cohort[cohort["stay_id"].isin(f["stay_id"].unique())]
        cohort = cohort.merge(f[["stay_id", "hour"]], on=["stay_id", "hour"])
        X, _, keys = build_allfeat_design(cohort, src)
        del cohort
        if not ((keys["stay_id"].to_numpy() == f["stay_id"].to_numpy()).all()
                and (keys["hour"].to_numpy() == f["hour"].to_numpy()).all()):
            raise SystemExit(f"eICU {s} design rows do not align with the split frame")
        if X["ecg_has"].any():
            raise SystemExit("an eICU row picked up a MIMIC ECG")
        payload[f"{s}_stay_id"] = keys["stay_id"].to_numpy()
        payload[f"{s}_hour"] = keys["hour"].to_numpy()
        payload[f"{s}_raw"] = model.predict_proba(X[cols])[:, 1].astype(float)
        log(f"[eicu] {s}: {len(keys):,} rows scored ({time.time() - t0:.0f} s)")
        del X
    meta = {"device": device, "record": record, "gate": gate,
            "seconds": round(time.time() - t0, 1)}
    payload["meta_json"] = np.array(json.dumps(meta, default=float))
    tmp = GBM_OUT.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(GBM_OUT)
    log(f"[saved] {GBM_OUT.name}")


# --- stage: eval ---------------------------------------------------------------------

def load_scores(frames):
    """``(flagship {seed: {val, test}}, comps {name: {val, test}})`` on eICU rows."""
    flag = {}
    for s, path in FLAGSHIP.items():
        z = np.load(path)
        if "cadence" in str(z["agents"]):
            raise SystemExit(f"{path.name} includes cadence")
        flag[s] = {k: z["risk"][positions(z["stay_id"], z["hour"], f)].astype(float)
                   for k, f in frames.items()}
    z = np.load(LIT)
    comps = {n: {k: z[f"eicu_{n}"][positions(z["stay_id"], z["hour"], f)].astype(float)
                 for k, f in frames.items()} for n in LIT_NAMES}
    comps["news2"] = {k: f["news2"].to_numpy(dtype=float) for k, f in frames.items()}
    g = np.load(GBM_OUT)
    if str(g["fingerprint"]) != gbm_fingerprint(json.loads(str(g["meta_json"]))["device"]):
        raise SystemExit(f"{GBM_OUT.name} does not match its inputs")
    comps["allfeat_nocad"] = {k: g[f"{k}_raw"][positions(g[f"{k}_stay_id"], g[f"{k}_hour"], f)]
                              .astype(float) for k, f in frames.items()}
    return flag, comps


def _flags(table):
    caught = table["first_alarm"].notna().to_numpy()
    warn = (table["event_h"] - table["first_alarm"]).to_numpy(dtype=float)
    return caught.astype(float), (caught & (warn >= 6.0)).astype(float)


def _paired(flag_f, flag_c, idx_list):
    """Pooled ten-seed paired difference (flagship mean - comparator) with a CI."""
    point = float(np.mean([f.mean() for f in flag_f]) - flag_c.mean())
    draws = [np.mean([f[idx].mean() for f in flag_f]) - flag_c[idx].mean() for idx in idx_list]
    return {"delta": point, "ci": [float(np.percentile(draws, 2.5)),
                                   float(np.percentile(draws, 97.5))]}


def _race(t, f_by_seed, c_scores, thr_f_by_seed, thr_c, idx_rows, patients_order):
    """Pooled first-alarm race on jointly caught stays; diff > 0 = flagship first."""
    from paper_operating_points import stay_first_alarms
    tc = stay_first_alarms(t, c_scores, thr_c)
    diffs, pats = [], []
    for s, f in f_by_seed.items():
        tf = stay_first_alarms(t, f, thr_f_by_seed[s])
        both = tf["first_alarm"].notna().to_numpy() & tc["first_alarm"].notna().to_numpy()
        diffs.append((tc["first_alarm"] - tf["first_alarm"]).to_numpy(dtype=float)[both])
        pats.append(tf["patient_id"].to_numpy()[both])
    d, p = np.concatenate(diffs), np.concatenate(pats)
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    # Patient-clustered CI: a resample's multiplicity per patient weights every
    # pooled row of that patient.
    pos = {pid: i for i, pid in enumerate(patients_order)}
    pi = np.array([pos[x] for x in p], dtype=np.int64)
    rates = []
    for counts in idx_rows:
        w = counts[pi]
        a, b = float(w[d > 0].sum()), float(w[d < 0].sum())
        if a + b:
            rates.append(a / (a + b))
    return {"n_pair": int(len(d)), "flagship_first": wins, "comparator_first": losses,
            "ties": int(len(d) - wins - losses),
            "flagship_first_when_differ": float(wins / (wins + losses)) if wins + losses else float("nan"),
            "ci": [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))],
            "median_lead_h": float(np.median(d)) if len(d) else float("nan")}


def evaluate_set(name, val, test, flag, comps, reps):
    from baselines_fair import bootstrap_indices
    from config import SEED
    from paper_operating_points import early_stats, stay_first_alarms

    log(f"\n[{name}] val {len(val):,} hours, test {len(test):,} hours, "
        f"{test['stay_id'].nunique():,} test stays")
    rows = []
    patients_order = None
    for b in SWEEP_BUDGETS:
        row = {"budget": b, "comparators": {}, "flagship_by_seed": {},
               "flagship_equal_volume": {}, "paired_catch": {}, "paired_catch_6h": {}}
        thr_c = {n: float(np.quantile(c["val"], 1.0 - b)) for n, c in comps.items()}
        flags_c, tab0 = {}, None
        for n, c in comps.items():
            tab = stay_first_alarms(test, c["test"], thr_c[n])
            tab0 = tab if tab0 is None else tab0
            if not (tab["stay_id"].to_numpy() == tab0["stay_id"].to_numpy()).all():
                raise SystemExit("event-stay tables are misaligned")
            rate = float((c["test"] >= thr_c[n]).mean())
            row["comparators"][n] = {**early_stats(tab), "test_alarm_rate": rate,
                                     "val_alarm_rate": float((c["val"] >= thr_c[n]).mean())}
            flags_c[n] = _flags(tab)
        patients = tab0["patient_id"].to_numpy()
        if patients_order is None:
            patients_order = np.unique(patients)
        idx_list = list(bootstrap_indices(patients, reps, SEED))
        flags_f, thr_f = {}, {}
        for s, f in flag.items():
            thr_f[s] = float(np.quantile(f["val"], 1.0 - b))
            tab = stay_first_alarms(test, f["test"], thr_f[s])
            row["flagship_by_seed"][s] = {**early_stats(tab),
                                          "test_alarm_rate": float((f["test"] >= thr_f[s]).mean())}
            flags_f[s] = _flags(tab)
        for n in comps:
            rate = row["comparators"][n]["test_alarm_rate"]
            eq = []
            for s, f in flag.items():
                thr = float(np.quantile(f["test"], 1.0 - rate))
                eq.append(early_stats(stay_first_alarms(test, f["test"], thr)))
            row["flagship_equal_volume"][n] = {
                k: float(np.mean([e[k] for e in eq])) for k in ("catch", "catch_6h", "warning_median")}
            row["paired_catch"][n] = _paired([flags_f[s][0] for s in flag], flags_c[n][0], idx_list)
            row["paired_catch_6h"][n] = _paired([flags_f[s][1] for s in flag], flags_c[n][1], idx_list)
            seeds_ahead = sum(row["flagship_by_seed"][s]["catch"] > row["comparators"][n]["catch"]
                              for s in flag)
            row["paired_catch"][n]["seeds_ahead"] = int(seeds_ahead)
        keys = ("catch", "catch_6h", "warning_median", "warning_q25", "warning_q75",
                "test_alarm_rate")
        row["flagship_mean"] = {k: float(np.mean([v[k] for v in row["flagship_by_seed"].values()]))
                                for k in keys}
        if b in CI_BUDGETS:
            # Times each patient is drawn in each resample, for the clustered
            # race CI. ``idx`` lists event-stay rows, so a patient with k event
            # stays appears k times per draw; divide back to draws.
            pat_pos = np.searchsorted(patients_order, patients)
            per = np.bincount(pat_pos, minlength=len(patients_order)).astype(float)
            counts = [np.bincount(pat_pos[idx], minlength=len(patients_order))
                      / np.maximum(per, 1.0) for idx in idx_list]
            row["races"] = {n: _race(test, {s: f["test"] for s, f in flag.items()},
                                     comps[n]["test"], thr_f, thr_c[n], counts, patients_order)
                            for n in comps}
        rows.append(row)
        fm = row["flagship_mean"]
        log(f"[{name}] {b:>5.1%}: flagship catch {fm['catch']:.1%} (alarm {fm['test_alarm_rate']:.2%}) | "
            + " ".join(f"{n} {row['comparators'][n]['catch']:.1%} ({row['comparators'][n]['test_alarm_rate']:.2%}) "
                       f"d {row['paired_catch'][n]['delta']:+.1%} [{row['paired_catch'][n]['ci'][0]:+.1%}, "
                       f"{row['paired_catch'][n]['ci'][1]:+.1%}]" for n in comps))
    beats = {str(r["budget"]): {n: r["flagship_mean"]["catch"] > r["comparators"][n]["catch"]
                                for n in comps} for r in rows}
    ci_ok = {str(r["budget"]): {n: r["paired_catch"][n]["ci"][0] > 0 for n in comps}
             for r in rows if r["budget"] in CI_BUDGETS}
    passed = (all(all(v.values()) for v in beats.values())
              and all(all(v.values()) for v in ci_ok.values()))
    return {"n_val_hours": int(len(val)), "n_test_hours": int(len(test)),
            "n_test_stays": int(test["stay_id"].nunique()),
            "n_test_event_stays": int(rows[0]["comparators"]["news2"]["n_event_stays"]),
            "rows": rows, "verdict": {"beats_every_budget": beats,
                                      "ci_above_zero_at_1_2_5": ci_ok, "pass": passed}}


def stage_eval(reps):
    from demo_bundle import assert_aggregate_only
    from metrics import auprc, auroc

    t0 = time.time()
    frames, report = eicu_frames()
    flag, comps = load_scores(frames)
    log(f"[load] {len(flag)} flagship seeds, comparators {list(comps)} ({time.time() - t0:.0f} s)")
    y = frames["test"]["label_24h"].to_numpy(dtype=float)
    ranking = {"flagship_by_seed": {s: {"auprc": float(auprc(y, f["test"])),
                                        "auroc": float(auroc(y, f["test"]))} for s, f in flag.items()},
               **{n: {"auprc": float(auprc(y, c["test"])), "auroc": float(auroc(y, c["test"]))}
                  for n, c in comps.items()},
               "prevalence": float(y.mean())}
    sets = {}
    for name in ("regular", "all"):
        if name == "regular":
            mv, mt = regular_mask(frames["val"]), regular_mask(frames["test"])
        else:
            mv, mt = np.ones(len(frames["val"]), bool), np.ones(len(frames["test"]), bool)
        val = frames["val"][mv].reset_index(drop=True)
        test = frames["test"][mt].reset_index(drop=True)
        f_sub = {s: {"val": f["val"][mv], "test": f["test"][mt]} for s, f in flag.items()}
        c_sub = {n: {"val": c["val"][mv], "test": c["test"][mt]} for n, c in comps.items()}
        sets[name] = evaluate_set(name, val, test, f_sub, c_sub, reps)
        sets[name]["share_of_test_hours"] = float(mt.mean())
        log(f"[{name}] verdict: {'PASS' if sets[name]['verdict']['pass'] else 'FAIL'}")
    payload = {"test": "T35", "seeds": list(SEEDS), "reps": REPS,
               "split": report,
               "hospitals": {k: int(f["hospitalid"].nunique()) for k, f in frames.items()},
               "ranking_test_all_hours": ranking, "hour_sets": sets,
               "rules": "per hour set: the flagship's ten-seed mean catch beats every comparator at "
                        "all 7 budgets, and the pooled paired catch CI is above zero at 1/2/5% vs every "
                        "comparator; thresholds are each model's own (1-b) quantile on val-hospital "
                        "hours of that set",
               "note": "aggregate-only; equal-volume rows are a diagnostic (test-score quantiles)"}
    payload = json.loads(json.dumps(payload, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    assert_aggregate_only(payload)
    OUT.write_text(json.dumps(payload, indent=1))
    log(f"[saved] {OUT} ({time.time() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("gbm", "eval"), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()
    if args.stage == "gbm":
        stage_gbm(args.device)
    else:
        stage_eval(args.reps)


if __name__ == "__main__":
    main()
