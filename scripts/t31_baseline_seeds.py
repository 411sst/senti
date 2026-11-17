"""T31: the baselines refit at ten seeds and paired seed by seed with the flagship.

Every baseline has a source of training randomness, so each is refit under
seeds 42 and 1-9:

* ``fair_traj``  FAIR+TRAJ, ``baselines_fair.tune_gbm`` (row and column
                 subsampling 0.8, ``random_state``).
* ``allfeat``    the T30 all-features GBM, same tuner on 262 columns.
* ``sadanandan`` ``lit_gbm.fit_xgb`` (subsampling 0.8, ``random_state``).
* ``ecart``      ``lit_gbm.fit_xgb`` after ``downsample_to_prevalence``,
                 whose negative sample is drawn with the seed.
* ``yaib``       ``lit_gbm.fit_lgbm`` (bagging 0.66, ``random_state``).

The same seed drives every random step of a fit. FAIR+TRAJ is the first 145
columns of the T30 design (both come from ``build_design``), so the two share
one design matrix. Raw scores are cached; AUPRC and AUROC are rank metrics.

Stage ``fit`` writes one cache per process (new filenames, fingerprinted;
the cache is rewritten after every fit so an interrupted run keeps what it
has). The original runs used the GPU for XGBoost and 14 CPU threads for
LightGBM; the stages repeat that:

    OMP_NUM_THREADS=1 .venv/bin/python scripts/t31_baseline_seeds.py --stage fit \
        --models fair_traj,allfeat --device cuda --out data/cache/t31_seeds_tab.npz
    OMP_NUM_THREADS=8 .venv/bin/python scripts/t31_baseline_seeds.py --stage fit \
        --models sadanandan,ecart --device cuda --out data/cache/t31_seeds_lit_gpu.npz
    OMP_NUM_THREADS=14 .venv/bin/python scripts/t31_baseline_seeds.py --stage fit \
        --models yaib --out data/cache/t31_seeds_lit_cpu.npz
    OMP_NUM_THREADS=1 .venv/bin/python scripts/t31_baseline_seeds.py --stage eval

The tabular process runs with OMP_NUM_THREADS=1, as the T30 fit did.

Stage ``eval`` gates the seed-42 refits against the cached single fits, then
pairs flagship seed s with baseline seed s on 1,000 shared patient-clustered
resamples. Aggregates only, to ``data/cache/t31_baseline_seeds.json``.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

CACHE = REPO / "data" / "cache"
OUT = CACHE / "t31_baseline_seeds.json"
PARTS = ("t31_seeds_tab.npz", "t31_seeds_lit_gpu.npz", "t31_seeds_lit_cpu.npz")
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
MODELS = ("fair_traj", "allfeat", "sadanandan", "ecart", "yaib")
TAB, LIT = ("fair_traj", "allfeat"), ("sadanandan", "ecart", "yaib")
REPS = 1000
GATE_AUPRC_TOL = 5e-4
CODE = ["src/allfeat_gbm.py", "src/baselines_fair.py", "src/lit_gbm.py",
        "src/modality_features.py", "src/agents/ecg_agent.py",
        "src/mimic_cohort.py"]


def _fingerprint(models, seeds, device, omp, keys):
    import lightgbm
    import xgboost

    from agents.ecg_agent import ECG_INDEX
    from agents.ecg_meas_agent import MEAS_CACHE
    from baselines_mimic import OUT_PATH
    from lit_gbm import RAW_PATH, STAY_META_PATH
    from modality_features import (CADENCE_PATH, EXTRA_VITALS_PATH, FLUIDS_PATH,
                                   LABS_PATH, RESP_PATH)
    files = []
    for p in (OUT_PATH, LABS_PATH, CADENCE_PATH, FLUIDS_PATH, RESP_PATH,
              MEAS_CACHE, ECG_INDEX, EXTRA_VITALS_PATH, RAW_PATH, STAY_META_PATH):
        st = Path(p).stat()
        files.append([str(p), st.st_size, st.st_mtime_ns])
    h = hashlib.sha256()
    for a in keys:
        h.update(np.ascontiguousarray(a).tobytes())
    spec = {"models": list(models), "seeds": list(seeds), "device": device,
            "omp": omp,
            "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
            "code": {c: hashlib.sha256(Path(c).read_bytes()).hexdigest() for c in CODE},
            "sources": files, "keys": h.hexdigest()}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def _save(out, payload, records):
    payload = dict(payload)
    payload["records_json"] = np.array(json.dumps(records))
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(out)


# --- stage: fit -------------------------------------------------------------

def _designs_tab():
    from allfeat_gbm import build_allfeat_design, column_sets, load_sources
    from baselines_fair import variant_columns
    from baselines_mimic import load_splits
    splits = load_splits()
    sources = load_sources()
    out = {}
    for split in ("train", "val", "test"):
        X, y, keys = build_allfeat_design(splits[split], sources)
        out[split] = (X, y, keys)
    sets = column_sets(sources["ecg"]["names"])
    cols = {"fair_traj": list(variant_columns()["FAIR+TRAJ"]),
            "allfeat": sets["allfeat"]}
    return out, cols


def _designs_lit(models):
    from lit_gbm import LABEL, build_channels, build_spec_design, load_inputs
    splits, labs, extra, raw, meta = load_inputs()
    built = {s: build_channels(splits[s], labs, extra, raw, meta)
             for s in ("train", "val", "test")}
    designs = {}
    for name in models:
        d = {}
        for s in ("train", "val", "test"):
            ordered, channels = built[s]
            X, prov = build_spec_design(name, ordered, channels)
            d[s] = (X, ordered[LABEL].to_numpy(dtype=float), ordered[["stay_id", "hour"]])
        designs[name] = (d, prov)
    return designs


def stage_fit(models, seeds, device, out):
    out = Path(out)
    if out.exists() and not out.name.startswith("t31_"):
        raise SystemExit("refusing to write over a non-T31 cache")
    t0 = time.time()
    payload, records = {}, {}
    if set(models) <= set(TAB):
        designs, cols = _designs_tab()
        keys = {s: designs[s][2] for s in ("val", "test")}
    elif set(models) <= set(LIT):
        lit = _designs_lit(models)
        keys = {s: lit[models[0]][0][s][2] for s in ("val", "test")}
    else:
        raise SystemExit("one process fits either tabular or published-spec models")
    for s in ("val", "test"):
        payload[f"{s}_stay_id"] = keys[s]["stay_id"].to_numpy()
        payload[f"{s}_hour"] = keys[s]["hour"].to_numpy()
    omp = os.environ.get("OMP_NUM_THREADS")
    fp = _fingerprint(models, seeds, device, omp,
                      [payload[k] for k in ("val_stay_id", "val_hour",
                                            "test_stay_id", "test_hour")])
    if out.exists():
        old = np.load(out)
        if str(old["fingerprint"]) != fp:
            raise SystemExit(f"{out.name} exists with a different fingerprint")
        payload.update({k: old[k] for k in old.files
                        if k.startswith(("val_", "test_")) and k not in payload})
        records = json.loads(str(old["records_json"]))
    payload["fingerprint"] = np.array(fp)
    payload["meta"] = np.array(json.dumps({"models": list(models), "seeds": list(seeds),
                                           "device": device,
                                           "omp": omp}))
    print(f"[design] built in {time.time() - t0:.0f} s; fingerprint {fp[:12]}")

    for name in models:
        if name in TAB:
            from baselines_fair import tune_gbm
            X_tr = designs["train"][0][cols[name]]
            X_va = designs["val"][0][cols[name]]
            X_te = designs["test"][0][cols[name]]
            y_tr, y_va = designs["train"][1], designs["val"][1]
        else:
            from lit_gbm import FITTERS, downsample_to_prevalence
            d, prov = lit[name]
            fitter = FITTERS[prov["library"]]
        for seed in seeds:
            tag = f"{name}_s{seed}"
            if f"test_{tag}" in payload:
                print(f"[{tag}] cached")
                continue
            t1 = time.time()
            if name in TAB:
                model, rec = tune_gbm(X_tr, y_tr, X_va, y_va, seed=seed,
                                      device=device, verbose=False)
                s_va = model.predict_proba(X_va)[:, 1].astype(np.float64)
                s_te = model.predict_proba(X_te)[:, 1].astype(np.float64)
            else:
                X_tr, y_tr = d["train"][0], d["train"][1]
                if "downsample_prevalence" in prov:
                    keep = downsample_to_prevalence(y_tr, prov["downsample_prevalence"], seed)
                    X_tr, y_tr = X_tr.iloc[keep], y_tr[keep]
                model, rec = fitter(X_tr, y_tr, d["val"][0], d["val"][1],
                                    seed=seed, device=device)
                s_va = model.predict_proba(d["val"][0])[:, 1].astype(np.float64)
                s_te = model.predict_proba(d["test"][0])[:, 1].astype(np.float64)
            rec["fit_seconds"] = round(time.time() - t1, 1)
            records[tag] = rec
            payload[f"val_{tag}"], payload[f"test_{tag}"] = s_va, s_te
            _save(out, payload, records)
            print(f"[{tag}] {rec}")
            del model
    print(f"[done] {out.name} in {time.time() - t0:.0f} s")


# --- stage: eval ------------------------------------------------------------

def _load_parts(ctx):
    scores, records = {}, {}
    for part in PARTS:
        z = np.load(CACHE / part, allow_pickle=False)
        for side, ref in (("val", ctx.val_ordered), ("test", ctx.test_ordered)):
            if not ((z[f"{side}_stay_id"] == ref["stay_id"].to_numpy()).all()
                    and (z[f"{side}_hour"] == ref["hour"].to_numpy()).all()):
                raise SystemExit(f"{part} is not aligned with {side}")
        meta = json.loads(str(z["meta"]))
        fp = _fingerprint(meta["models"], meta["seeds"], meta["device"], meta["omp"],
                          [z[k] for k in ("val_stay_id", "val_hour",
                                          "test_stay_id", "test_hour")])
        if fp != str(z["fingerprint"]):
            raise SystemExit(f"{part}: fingerprint does not match the current inputs")
        for k in z.files:
            if k.startswith("test_") and k not in ("test_stay_id", "test_hour"):
                scores[k[len("test_"):]] = z[k].astype(float)
        records.update(json.loads(str(z["records_json"])))
    return scores, records


def stage_eval(workers):
    from baselines_mimic import load_splits
    from config import SEED
    from demo_bundle import assert_aggregate_only
    from fusion_common import FusionContext, load_fair_scores
    from lit_gbm import load_lit_scores
    from metrics import auprc, auroc
    from paper_operating_points import load_scores
    from paper_stats import metric_draws, paired_delta, pooled_delta

    t0 = time.time()
    ctx = FusionContext(load_splits())
    seeds, flagship, _ = load_scores(ctx)
    scores, records = _load_parts(ctx)
    missing = [f"{m}_s{s}" for m in MODELS for s in SEEDS if f"{m}_s{s}" not in scores]
    if missing:
        raise SystemExit(f"missing fits: {missing}")
    y = ctx.y_test

    # Gate: each seed-42 refit against the cached single fit.
    _, fair_test = load_fair_scores(ctx.val_ordered, ctx.test_ordered)
    lit = load_lit_scores(ctx.val_ordered, ctx.test_ordered)
    za = np.load(CACHE / "allfeat_gbm_scores.npz")
    cached = {"fair_traj": fair_test, "allfeat": za["rawtest_allfeat_s42"].astype(float),
              **{n: lit[n][1] for n in LIT}}
    gate = {}
    for m in MODELS:
        a, b = scores[f"{m}_s42"], cached[m]
        gate[m] = {"max_abs_diff": float(np.max(np.abs(a - b))),
                   "identical": bool(np.array_equal(a, b)),
                   "auprc_refit": float(auprc(y, a)), "auprc_cached": float(auprc(y, b))}
        g = gate[m]
        print(f"[gate] {m:<10} seed-42 refit AUPRC {g['auprc_refit']:.4f} vs cached "
              f"{g['auprc_cached']:.4f}; identical {g['identical']} "
              f"(max |diff| {g['max_abs_diff']:.2e})")
    bad = [m for m, g in gate.items()
           if abs(g["auprc_refit"] - g["auprc_cached"]) > GATE_AUPRC_TOL]
    if bad:
        raise SystemExit(f"seed-42 refits do not reproduce the cached fits: {bad}")

    named = {f"flagship_s{s}": flagship[s]["test"] for s in seeds}
    named.update({f"{m}_s{s}": scores[f"{m}_s{s}"] for m in MODELS for s in SEEDS})
    targets = {"auprc": ("auprc", y), "auroc": ("auroc", y)}
    fns = {"auprc": auprc, "auroc": auroc}
    points = {t: {n: float(fns[t](y, s)) for n, s in named.items()} for t in targets}
    t1 = time.time()
    draws = metric_draws(targets, named, ctx.patient_test, reps=REPS, seed=SEED,
                         workers=workers)
    n_used = len(draws["auprc"]["flagship_s42"])
    print(f"[bootstrap] {n_used} shared resamples x {len(named)} scores x 2 targets "
          f"({time.time() - t1:.0f} s)")

    results = {}
    for m in MODELS:
        r = {}
        for t in targets:
            vals = [points[t][f"{m}_s{s}"] for s in SEEDS]
            pairs = {s: paired_delta(points[t][f"flagship_s{s}"], points[t][f"{m}_s{s}"],
                                     draws[t][f"flagship_s{s}"], draws[t][f"{m}_s{s}"])
                     for s in SEEDS}
            r[t] = {
                "baseline_mean": float(np.mean(vals)), "baseline_sd": float(np.std(vals, ddof=1)),
                "baseline_min": float(min(vals)), "baseline_max": float(max(vals)),
                "pairs": pairs,
                "pairs_ci_above_zero": int(sum(p["ci_above_zero"] for p in pairs.values())),
                "pairs_ci_below_zero": int(sum(p["ci"][1] < 0 for p in pairs.values())),
                "pooled": pooled_delta([points[t][f"flagship_s{s}"] for s in SEEDS],
                                       [points[t][f"{m}_s{s}"] for s in SEEDS],
                                       [draws[t][f"flagship_s{s}"] for s in SEEDS],
                                       [draws[t][f"{m}_s{s}"] for s in SEEDS]),
            }
        r["pass"] = r["auprc"]["pairs_ci_above_zero"] == len(SEEDS)
        results[m] = r
        a, p = r["auprc"], r["auprc"]["pooled"]
        print(f"[{m:<10}] AUPRC {a['baseline_mean']:.4f} (SD {a['baseline_sd']:.4f}, "
              f"{a['baseline_min']:.4f}-{a['baseline_max']:.4f}) | pooled d "
              f"{p['delta']:+.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}] | pairs CI>0 "
              f"{a['pairs_ci_above_zero']}/10, CI<0 {a['pairs_ci_below_zero']}/10 -> "
              f"{'PASS' if r['pass'] else 'FAIL'}")
    fl = [points["auprc"][f"flagship_s{s}"] for s in SEEDS]
    payload = {
        "test": "T31", "seeds": list(SEEDS), "reps": REPS, "resamples_used": n_used,
        "flagship_auprc": {"mean": float(np.mean(fl)), "sd": float(np.std(fl, ddof=1))},
        "gate": gate, "fit_records": records, "results": results,
        "pass_rule": "per baseline: paired dAUPRC (flagship seed s - baseline seed s) "
                     "95% CI above zero in all 10 pairs",
        "note": "aggregate-only",
    }
    payload = json.loads(json.dumps(payload, default=float))
    assert_aggregate_only(payload)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"[saved] {OUT} ({time.time() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("fit", "eval"), required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    if args.stage == "fit":
        if not args.out:
            raise SystemExit("--out is required for --stage fit")
        stage_fit(tuple(args.models.split(",")),
                  tuple(int(s) for s in args.seeds.split(",")), args.device, args.out)
    else:
        stage_eval(args.workers)


if __name__ == "__main__":
    main()
