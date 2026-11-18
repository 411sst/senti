"""T39: a death-only all-features GBM against the D head (pre-registered).

The flagship's D head is trained and early-stopped on ``label_24h_death``.
Until now its death AUPRC (0.340) was compared with GBMs trained on the
composite label, which is not a like-for-like comparison. T39 trains the T30
all-features GBM on the D head's own label and rows:

* design: the same 262 columns (``allfeat_gbm.build_allfeat_design``);
* recipe: ``baselines_fair.tune_gbm`` unchanged (depth {4, 6, 8} x class
  weight {1, rebalanced}, learning rate 0.05, up to 3,000 trees, early
  stopping after 50 rounds on validation AUPRC). The rebalanced weight is
  recomputed from the death label;
* rows: every train row (fit), every val row (early stopping and grid
  choice), V_fit (isotonic), every test row (scoring), as in
  ``scripts/two_head.py``;
* label: the cohort's ``label_24h_death``; the design's composite label is
  discarded;
* seeds 42 and 1-9, on the GPU with one OpenMP thread, as T30 and T31.

Stage ``fit`` writes ``data/cache/t39_death_gbm.npz`` (new file,
fingerprinted, rewritten after every seed). Stage ``eval`` scores death
within 24 h on the test split for the D head (``two_head.npz``), the
death-only GBM, the composite all-features GBM (``t31_seeds_tab.npz``) and
the flagship's composite score, on 1,000 shared patient-clustered
resamples, and writes aggregates only to ``data/cache/t39_death_gbm.json``.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/t39_death_gbm.py --stage fit --device cuda
    OMP_NUM_THREADS=1 .venv/bin/python scripts/t39_death_gbm.py --stage eval
"""

import argparse
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
SCORES = CACHE / "t39_death_gbm.npz"
OUT = CACHE / "t39_death_gbm.json"
DEATH = "label_24h_death"
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
MODEL_TAG = "allfeat_death"
REPS = 1000
GATE_TOL = 1e-9
N_COLUMNS = 262


# --- helpers (unit-tested) ---------------------------------------------------

def death_labels(split_df, keys):
    """``label_24h_death`` in the design's row order.

    ``split_df`` is a cohort split in any order; ``keys`` holds the design's
    ``stay_id`` and ``hour`` columns. Raises unless the sorted split has
    exactly the design's rows, and unless the label is binary.
    """
    ordered = split_df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    if len(ordered) != len(keys) or not (
            (ordered["stay_id"].to_numpy() == keys["stay_id"].to_numpy()).all()
            and (ordered["hour"].to_numpy() == keys["hour"].to_numpy()).all()):
        raise ValueError("design rows do not match the cohort split")
    y = ordered[DEATH].to_numpy(dtype=float)
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError(f"{DEATH} is not binary")
    return y


def death_calibrator(raw_val, y_val_death, fit_mask):
    """Isotonic raw -> death probability, fit on V_fit only (as the D head)."""
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(out_of_bounds="clip").fit(
        raw_val[fit_mask], y_val_death[fit_mask])


def resume_payload(path, fingerprint):
    """Arrays already saved under the same fingerprint, or ``{}``.

    Refuses to continue over a cache written from different inputs.
    """
    path = Path(path)
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=False)
    if str(z["fingerprint"]) != fingerprint:
        raise SystemExit(f"{path.name} exists with a different fingerprint; "
                         "refusing to overwrite it")
    return {k: z[k] for k in z.files}


def seed_pairs(points_a, points_b, draws_a, draws_b):
    """Per-seed paired deltas (a - b) and their counts, plus the pooled delta.

    Each argument is a dict keyed by seed. The pooled delta is the mean over
    seeds of the per-resample deltas (``paper_stats.pooled_delta``).
    """
    from paper_stats import paired_delta, pooled_delta
    seeds = list(points_a)
    pairs = {s: paired_delta(points_a[s], points_b[s], draws_a[s], draws_b[s])
             for s in seeds}
    return {
        "pairs": pairs,
        "ci_above_zero": int(sum(p["ci"][0] > 0 for p in pairs.values())),
        "ci_below_zero": int(sum(p["ci"][1] < 0 for p in pairs.values())),
        "point_above_zero": int(sum(p["delta"] > 0 for p in pairs.values())),
        "pooled": pooled_delta([points_a[s] for s in seeds],
                               [points_b[s] for s in seeds],
                               [draws_a[s] for s in seeds],
                               [draws_b[s] for s in seeds]),
    }


def _fingerprint(seeds, device, omp, keys):
    """T31's fingerprint (input files, code, versions, row keys), tagged T39."""
    from t31_baseline_seeds import _fingerprint as t31_fingerprint
    return t31_fingerprint((MODEL_TAG, DEATH), seeds, device, omp, keys)


def _key_arrays(payload):
    return [payload[k] for k in ("val_stay_id", "val_hour", "test_stay_id", "test_hour")]


def _check_aligned(stay_id, hour, ref, what):
    if not ((stay_id == ref["stay_id"].to_numpy()).all()
            and (hour == ref["hour"].to_numpy()).all()):
        raise SystemExit(f"{what} is not aligned with the cohort rows")


# --- stage: fit ----------------------------------------------------------------

def stage_fit(seeds, device):
    from allfeat_gbm import build_allfeat_design, column_sets, load_sources
    from baselines_fair import tune_gbm
    from baselines_mimic import load_splits
    from fusion_common import FusionContext
    from fusion_ladder import _strictify
    from metrics import auprc
    from t31_baseline_seeds import _save

    t0 = time.time()
    splits = load_splits()
    ctx = FusionContext(splits)
    sources = load_sources()
    cols = column_sets(sources["ecg"]["names"])["allfeat"]
    if len(cols) != N_COLUMNS:
        raise SystemExit(f"expected {N_COLUMNS} columns, got {len(cols)}")

    design = {}
    for split in ("train", "val", "test"):
        X, _composite, keys = build_allfeat_design(splits[split], sources)
        if list(X.columns) != cols:
            raise SystemExit(f"{split} design columns differ from the T30 set")
        design[split] = (X, death_labels(splits[split], keys), keys)
        print(f"[features] {split}: {X.shape[0]:,} rows x {len(cols)}, death "
              f"prevalence {design[split][1].mean():.4%} ({time.time() - t0:.0f} s)")
    for split, ref in (("val", ctx.val_ordered), ("test", ctx.test_ordered)):
        keys = design[split][2]
        _check_aligned(keys["stay_id"].to_numpy(), keys["hour"].to_numpy(), ref,
                       f"{split} design")
        if not np.array_equal(design[split][1], ref[DEATH].to_numpy(dtype=float)):
            raise SystemExit(f"{split} death label differs from FusionContext")

    payload = {f"{s}_{c}": design[s][2][c].to_numpy()
               for s in ("val", "test") for c in ("stay_id", "hour")}
    omp = os.environ.get("OMP_NUM_THREADS")
    fp = _fingerprint(seeds, device, omp, _key_arrays(payload))
    old = resume_payload(SCORES, fp)
    records = json.loads(str(old.pop("records_json"))) if old else {}
    payload.update({k: v for k, v in old.items() if k not in payload})
    payload["fingerprint"] = np.array(fp)
    payload["meta"] = np.array(json.dumps({
        "seeds": list(seeds), "device": device, "label": DEATH,
        "omp": omp, "n_columns": len(cols)}))
    print(f"[design] built in {time.time() - t0:.0f} s; fingerprint {fp[:12]}")

    X_tr, y_tr, _ = design["train"]
    X_va, y_va, _ = design["val"]
    X_te = design["test"][0]
    for seed in seeds:
        tag = f"s{seed}"
        if f"test_{tag}" in payload:
            print(f"[{tag}] cached")
            continue
        t1 = time.time()
        model, rec = tune_gbm(X_tr, y_tr, X_va, y_va, seed=seed, device=device,
                              verbose=False)
        raw_va = model.predict_proba(X_va)[:, 1].astype(np.float64)
        raw_te = model.predict_proba(X_te)[:, 1].astype(np.float64)
        iso = death_calibrator(raw_va, y_va, ctx.fit_mask)
        cal_va = _strictify(iso.transform(raw_va), raw_va)
        cal_te = _strictify(iso.transform(raw_te), raw_te)
        fm = ctx.freeze_mask
        rec["fit_seconds"] = round(time.time() - t1, 1)
        rec["v_freeze_death_auprc"] = float(auprc(y_va[fm], cal_va[fm]))
        records[tag] = rec
        payload.update({f"rawval_{tag}": raw_va, f"rawtest_{tag}": raw_te,
                        f"val_{tag}": cal_va, f"test_{tag}": cal_te})
        _save(SCORES, payload, records)
        print(f"[{tag}] {rec}")
        del model
    print(f"[done] {SCORES.name} in {time.time() - t0:.0f} s")


# --- stage: eval ---------------------------------------------------------------

def _load_death_gbm(ctx):
    z = np.load(SCORES, allow_pickle=False)
    _check_aligned(z["val_stay_id"], z["val_hour"], ctx.val_ordered, SCORES.name)
    _check_aligned(z["test_stay_id"], z["test_hour"], ctx.test_ordered, SCORES.name)
    meta = json.loads(str(z["meta"]))
    fp = _fingerprint(tuple(meta["seeds"]), meta["device"], meta["omp"], _key_arrays(z))
    if fp != str(z["fingerprint"]):
        raise SystemExit(f"{SCORES.name}: fingerprint does not match the current inputs")
    missing = [s for s in SEEDS if f"test_s{s}" not in z.files]
    if missing:
        raise SystemExit(f"{SCORES.name} is missing seeds {missing}")
    return ({s: z[f"test_s{s}"].astype(float) for s in SEEDS},
            json.loads(str(z["records_json"])), meta)


def _load_composite_gbm(ctx):
    from t31_baseline_seeds import _fingerprint as t31_fingerprint
    z = np.load(CACHE / "t31_seeds_tab.npz", allow_pickle=False)
    _check_aligned(z["test_stay_id"], z["test_hour"], ctx.test_ordered, "t31_seeds_tab.npz")
    meta = json.loads(str(z["meta"]))
    fp = t31_fingerprint(meta["models"], meta["seeds"], meta["device"], meta["omp"],
                         _key_arrays(z))
    if fp != str(z["fingerprint"]):
        raise SystemExit("t31_seeds_tab.npz: fingerprint does not match the current inputs")
    return {s: z[f"test_allfeat_s{s}"].astype(float) for s in SEEDS}


def stage_eval(workers):
    from baselines_mimic import load_splits
    from config import SEED
    from demo_bundle import assert_aggregate_only
    from fusion_common import FusionContext
    from metrics import auprc, auroc
    from paper_stats import metric_draws

    t0 = time.time()
    ctx = FusionContext(load_splits())
    test = ctx.test_ordered
    y_death = test[DEATH].to_numpy(dtype=float)

    zt = np.load(CACHE / "two_head.npz", allow_pickle=False)
    _check_aligned(zt["stay_id_test"], zt["hour_test"], test, "two_head.npz")
    models = {
        "dhead": {s: zt[f"dhead_test_s{s}"].astype(float) for s in SEEDS},
        "death_gbm": None,
        "composite_gbm": _load_composite_gbm(ctx),
        "flagship": {s: zt[f"test_s{s}"].astype(float) for s in SEEDS},
    }
    models["death_gbm"], fit_records, fit_meta = _load_death_gbm(ctx)

    # Gate: the cached D-head scores give the death AUPRC T30 recorded.
    t30 = json.loads((CACHE / "t30_allfeat_gbm.json").read_text())
    t30_points = t30["ranking"]["death_auprc"]["points"]
    gate = {s: {"auprc": float(auprc(y_death, models["dhead"][s])),
                "t30": float(t30_points[f"dhead_s{s}"])} for s in SEEDS}
    bad = [s for s, g in gate.items() if abs(g["auprc"] - g["t30"]) > GATE_TOL]
    print(f"[gate] D-head death AUPRC mean "
          f"{np.mean([g['auprc'] for g in gate.values()]):.4f}; matches T30 in "
          f"{len(SEEDS) - len(bad)}/{len(SEEDS)} seeds")
    if bad:
        raise SystemExit(f"D-head gate failed for seeds {bad}")

    named = {f"{m}_s{s}": v[s] for m, v in models.items() for s in SEEDS}
    targets = {"death_auprc": ("auprc", y_death), "death_auroc": ("auroc", y_death)}
    fns = {"death_auprc": auprc, "death_auroc": auroc}
    points = {t: {n: float(fns[t](y_death, s)) for n, s in named.items()} for t in targets}
    t1 = time.time()
    draws = metric_draws(targets, named, ctx.patient_test, reps=REPS, seed=SEED,
                         workers=workers)
    n_used = len(draws["death_auprc"]["dhead_s42"])
    print(f"[bootstrap] {n_used} shared resamples x {len(named)} scores x 2 targets "
          f"({time.time() - t1:.0f} s)")

    summary = {}
    for m in models:
        summary[m] = {}
        for t in targets:
            vals = [points[t][f"{m}_s{s}"] for s in SEEDS]
            summary[m][t] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                             "min": float(min(vals)), "max": float(max(vals)),
                             "per_seed": {str(s): v for s, v in zip(SEEDS, vals)}}
        print(f"[{m:<13}] death AUPRC {summary[m]['death_auprc']['mean']:.4f} "
              f"(SD {summary[m]['death_auprc']['sd']:.4f}) | AUROC "
              f"{summary[m]['death_auroc']['mean']:.4f}")

    comparisons = {"dhead_minus_death_gbm": ("dhead", "death_gbm"),
                   "dhead_minus_composite_gbm": ("dhead", "composite_gbm"),
                   "death_gbm_minus_composite_gbm": ("death_gbm", "composite_gbm"),
                   "flagship_minus_death_gbm": ("flagship", "death_gbm")}
    deltas = {}
    for name, (a, b) in comparisons.items():
        deltas[name] = {}
        for t in targets:
            pick = lambda m, src: {s: src[f"{m}_s{s}"] for s in SEEDS}  # noqa: E731
            deltas[name][t] = seed_pairs(pick(a, points[t]), pick(b, points[t]),
                                         pick(a, draws[t]), pick(b, draws[t]))
            p = deltas[name][t]["pooled"]
            print(f"[{name:<30}] {t:<11} pooled {p['delta']:+.4f} "
                  f"[{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}] | seeds CI>0 "
                  f"{deltas[name][t]['ci_above_zero']}/10, CI<0 "
                  f"{deltas[name][t]['ci_below_zero']}/10, point>0 "
                  f"{deltas[name][t]['point_above_zero']}/10")

    primary = deltas["dhead_minus_death_gbm"]["death_auprc"]["pooled"]
    verdict = "PASS" if primary["ci"][0] > 0 else "FAIL"
    print(f"[verdict] pooled dAUPRC (D head - death-only GBM) {primary['delta']:+.4f} "
          f"[{primary['ci'][0]:+.4f}, {primary['ci'][1]:+.4f}] -> {verdict}")

    payload = {
        "test": "T39", "seeds": list(SEEDS), "reps": REPS, "resamples_used": n_used,
        "test_hours": int(len(y_death)), "death_prevalence": float(y_death.mean()),
        "gate": {str(s): g for s, g in gate.items()},
        "fit_records": fit_records, "fit_meta": fit_meta,
        "summary": summary, "deltas": deltas, "verdict": verdict,
        "pass_rule": "pooled paired dAUPRC on death (D head seed s - death-only GBM "
                     "seed s) has its 95% CI above zero",
        "note": "aggregate-only",
    }
    payload = json.loads(json.dumps(payload, default=float))
    assert_aggregate_only(payload)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"[saved] {OUT} ({time.time() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("fit", "eval"), required=True)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    if args.stage == "fit":
        stage_fit(tuple(int(s) for s in args.seeds.split(",")), args.device)
    else:
        stage_eval(args.workers)


if __name__ == "__main__":
    main()
