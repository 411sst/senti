"""T30: the all-features GBM against the two-head flagship.

One XGBoost on the union of all eight agents' inputs (``src/allfeat_gbm.py``),
plus a no-cadence version, against the flagship's ten seeds.

Stage ``fit`` (GPU for the trees): builds the design for train/val/test with
the agents' own feature code, fits both arms with the unchanged FAIR+TRAJ
recipe (``baselines_fair.tune_gbm``: depth {4,6,8} x class weight {none,
rebalanced}, lr 0.05, <= 3,000 trees, early stop 50 rounds on val AUPRC, seed
42), calibrates on V_fit with the flagship's rank tiebreak, and caches the
scores to ``data/cache/allfeat_gbm_scores.npz`` with a fingerprint. An existing
cache with a different fingerprint is never overwritten.

Stage ``eval`` (CPU, one thread per process): gates, then AUPRC, AUROC and
death-only AUPRC on 1,000 shared patient-clustered resamples, the catch sweep,
and first-alarm races at 1/2/5%. Aggregates only, to
``data/cache/t30_allfeat_gbm.json``.

    .venv/bin/python scripts/t30_allfeat_gbm.py --stage fit
    OMP_NUM_THREADS=1 .venv/bin/python scripts/t30_allfeat_gbm.py --stage eval
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

CACHE = REPO / "data" / "cache"
SCORES = CACHE / "allfeat_gbm_scores.npz"
OUT = CACHE / "t30_allfeat_gbm.json"
SEED_GBM = 42
REPS = 1000
EXPECTED_S42_AUPRC = 0.3141
CODE = ["src/allfeat_gbm.py", "src/baselines_fair.py", "src/modality_features.py",
        "src/agents/ecg_agent.py", "src/agents/ecg_meas_agent.py",
        "src/mimic_cohort.py", "src/fusion_common.py", "src/fusion_ladder.py"]


# --- fingerprint -------------------------------------------------------------

def _sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def fingerprint(ctx, ecg_names, device):
    """Hash of everything the cached scores depend on."""
    import xgboost

    from agents.ecg_agent import ECG_INDEX
    from agents.ecg_meas_agent import MEAS_CACHE
    from allfeat_gbm import column_sets
    from baselines_fair import tune_gbm
    from baselines_mimic import OUT_PATH
    from modality_features import CADENCE_PATH, FLUIDS_PATH, LABS_PATH, RESP_PATH

    files = []
    for p in (OUT_PATH, LABS_PATH, CADENCE_PATH, FLUIDS_PATH, RESP_PATH,
              MEAS_CACHE, ECG_INDEX):
        st = Path(p).stat()
        files.append([str(p), st.st_size, st.st_mtime_ns])
    spec = {
        "columns": column_sets(ecg_names),
        "seed": SEED_GBM, "device": device, "xgboost": xgboost.__version__,
        "tune_gbm": hashlib.sha256(inspect.getsource(tune_gbm).encode()).hexdigest(),
        "code": {c: hashlib.sha256(Path(c).read_bytes()).hexdigest() for c in CODE},
        "calibration": "isotonic on V_fit + _strictify",
        "sources": files,
        "val_keys": _sha(ctx.val_ordered["stay_id"].to_numpy(),
                         ctx.val_ordered["hour"].to_numpy(), ctx.fit_mask),
        "test_keys": _sha(ctx.test_ordered["stay_id"].to_numpy(),
                          ctx.test_ordered["hour"].to_numpy()),
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


# --- stage: fit ----------------------------------------------------------------

def _gain_by_block(model, cols, ecg_names):
    from allfeat_gbm import NEWS2_SUBSCORES, ecg_columns
    from baselines_fair import variant_columns
    from modality_features import (CADENCE_FEATURES, FLUID_FEATURES,
                                   RESP_FEATURES)
    blocks = {"FAIR+TRAJ": set(variant_columns()["FAIR+TRAJ"]),
              "cadence": set(CADENCE_FEATURES), "fluid": set(FLUID_FEATURES),
              "resp": set(RESP_FEATURES), "ecg": set(ecg_columns(ecg_names)),
              "news2": set(NEWS2_SUBSCORES)}
    gain = model.get_booster().get_score(importance_type="total_gain")
    total = sum(gain.values()) or 1.0
    share = {b: float(sum(v for k, v in gain.items() if k in members) / total)
             for b, members in blocks.items()}
    top = sorted(gain.items(), key=lambda kv: -kv[1])[:15]
    return share, [[k, float(v / total)] for k, v in top]


def stage_fit(device):
    from allfeat_gbm import ARMS, build_allfeat_design, column_sets, load_sources
    from baselines_fair import tune_gbm
    from baselines_mimic import load_splits
    from fusion_common import FusionContext
    from fusion_ladder import _strictify
    from metrics import auprc

    splits = load_splits()
    ctx = FusionContext(splits)
    sources = load_sources()
    ecg_names = sources["ecg"]["names"]
    fp = fingerprint(ctx, ecg_names, device)
    if SCORES.exists():
        old = str(np.load(SCORES)["fingerprint"])
        if old == fp:
            print(f"[cache] {SCORES.name} is current (fingerprint {fp[:12]}); nothing to fit")
            return
        raise SystemExit(f"{SCORES} exists with a different fingerprint "
                         f"({old[:12]} vs {fp[:12]}); refusing to overwrite it")

    sets = column_sets(ecg_names)
    print(f"[columns] allfeat {len(sets['allfeat'])}, allfeat_nocad "
          f"{len(sets['allfeat_nocad'])}; ECG machine-read {len(ecg_names)}")
    if (len(sets["allfeat"]), len(sets["allfeat_nocad"])) != (262, 230):
        raise SystemExit("column counts are not 262 / 230")

    design, t0 = {}, time.time()
    for split, frame in (("train", splits["train"]), ("val", splits["val"]),
                         ("test", splits["test"])):
        X, y, keys = build_allfeat_design(frame, sources)
        design[split] = (X, y, keys)
        print(f"[features] {split}: {X.shape[0]:,} rows x {X.shape[1]} "
              f"({time.time() - t0:.0f} s)")
    for split, ref in (("val", ctx.val_ordered), ("test", ctx.test_ordered)):
        keys = design[split][2]
        if not ((keys["stay_id"].to_numpy() == ref["stay_id"].to_numpy()).all()
                and (keys["hour"].to_numpy() == ref["hour"].to_numpy()).all()):
            raise SystemExit(f"{split} design rows do not align with FusionContext")
    news2 = np.concatenate([
        design[s][0][[c for c in sets["allfeat"] if c.startswith("news2_")]]
        .sum(axis=1).to_numpy() for s in ("train", "val", "test")])
    cohort_news2 = np.concatenate([
        splits[s].sort_values(["stay_id", "hour"])["news2"].to_numpy(dtype=float)
        for s in ("train", "val", "test")])
    agree = float(np.mean(np.isclose(news2, cohort_news2)))
    print(f"[check] NEWS2 sub-scores sum to the cohort NEWS2 on {agree:.4%} of rows")

    X_tr, y_tr, _ = design["train"]
    X_va, y_va, keys_va = design["val"]
    X_te, _, keys_te = design["test"]
    payload = {"val_stay_id": keys_va["stay_id"].to_numpy(),
               "val_hour": keys_va["hour"].to_numpy(),
               "test_stay_id": keys_te["stay_id"].to_numpy(),
               "test_hour": keys_te["hour"].to_numpy(),
               "fingerprint": np.array(fp)}
    meta = {"seed": SEED_GBM, "device": device, "news2_sum_agreement": agree,
            "n_columns": {a: len(sets[a]) for a in ARMS}, "records": {},
            "gain_share_by_block": {}, "top_gain": {}}
    for arm in ARMS:
        cols = sets[arm]
        t1 = time.time()
        print(f"[{arm}] tuning on val ({len(cols)} features, {device})...")
        model, record = tune_gbm(X_tr[cols], y_tr, X_va[cols], y_va,
                                 seed=SEED_GBM, device=device)
        raw_va = model.predict_proba(X_va[cols])[:, 1].astype(float)
        raw_te = model.predict_proba(X_te[cols])[:, 1].astype(float)
        iso = ctx.fit_calibrator(raw_va)
        cal_va = _strictify(iso.transform(raw_va), raw_va)
        cal_te = _strictify(iso.transform(raw_te), raw_te)
        record["fit_seconds"] = round(time.time() - t1, 1)
        record["v_freeze_auprc"] = ctx.freeze_auprc(cal_va)
        record["val_auprc_check"] = float(auprc(y_va, raw_va))
        meta["records"][arm] = record
        meta["gain_share_by_block"][arm], meta["top_gain"][arm] = \
            _gain_by_block(model, cols, ecg_names)
        tag = f"{arm}_s{SEED_GBM}"
        payload.update({f"rawval_{tag}": raw_va, f"rawtest_{tag}": raw_te,
                        f"val_{tag}": cal_va, f"test_{tag}": cal_te})
        print(f"[{arm}] {record}")
        del model
    payload["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(SCORES, **payload)
    print(f"[saved] {SCORES} (fingerprint {fp[:12]})")


# --- stage: eval -------------------------------------------------------------

def _load_allfeat(ctx):
    from allfeat_gbm import ARMS
    z = np.load(SCORES, allow_pickle=False)
    for split, ref in (("val", ctx.val_ordered), ("test", ctx.test_ordered)):
        if not ((z[f"{split}_stay_id"] == ref["stay_id"].to_numpy()).all()
                and (z[f"{split}_hour"] == ref["hour"].to_numpy()).all()):
            raise SystemExit(f"{SCORES.name} is not aligned with {split}")
    tag = f"_s{SEED_GBM}"
    arms = {a: {"val": z[f"val_{a}{tag}"].astype(float),
                "test": z[f"test_{a}{tag}"].astype(float),
                "rawtest": z[f"rawtest_{a}{tag}"].astype(float)} for a in ARMS}
    return arms, json.loads(str(z["meta_json"])), str(z["fingerprint"])


def _sweep_row(ctx, name, val, test, thr=None, budget=None):
    from demo_bundle import first_alarm_table
    from paper_operating_points import (after_first_hour_catch, catch_stats,
                                        freeze_quantile, test_rate)
    if thr is None:
        thr = freeze_quantile(ctx, val, budget)
    table = first_alarm_table(ctx.test_ordered, {name: test}, {name: thr})
    return {**catch_stats(table, name),
            "catch_after_h0": after_first_hour_catch(ctx, test, thr),
            "test_alarm_rate": test_rate(ctx, test, thr), "threshold": float(thr)}


def stage_eval(workers):
    from allfeat_gbm import ARMS
    from baselines_mimic import load_splits
    from config import SEED
    from demo_bundle import assert_aggregate_only
    from fusion_common import FusionContext, load_fair_scores
    from metrics import auprc, auroc
    from paper_operating_points import (RACE_BUDGETS, SWEEP_BUDGETS,
                                        freeze_quantile, load_scores, race,
                                        test_rate)
    from paper_stats import metric_draws, paired_delta, pooled_delta

    t0 = time.time()
    ctx = FusionContext(load_splits())
    seeds, flagship, _ = load_scores(ctx)
    arms, meta, fp_cached = _load_allfeat(ctx)
    from agents.ecg_meas_agent import MEAS_CACHE
    ecg_names = [str(n) for n in np.load(MEAS_CACHE)["feature_names"]]
    fp_now = fingerprint(ctx, ecg_names, meta["device"])
    if fp_now != fp_cached:
        raise SystemExit("allfeat_gbm_scores.npz fingerprint does not match the "
                         "current inputs; refit to a new filename")
    zt = np.load(CACHE / "two_head.npz", allow_pickle=False)
    dhead = {s: zt[f"dhead_test_s{s}"].astype(float) for s in seeds}
    _, fair_test = load_fair_scores(ctx.val_ordered, ctx.test_ordered)
    y = ctx.y_test
    y_death = ctx.test_ordered["label_24h_death"].to_numpy(dtype=float)
    print(f"[load] seeds {seeds}; arms {list(ARMS)}; fingerprint ok "
          f"({time.time() - t0:.0f} s)")

    # Gates: the published flagship numbers reproduce before anything new.
    gates = {"s42_auprc": auprc(y, flagship[42]["test"])}
    sw = {b: _sweep_row(ctx, "m", flagship[42]["val"], flagship[42]["test"],
                        budget=b)["catch"] for b in (0.005, 0.10)}
    gates.update({"sweep_s42_0.5%": sw[0.005], "sweep_s42_10%": sw[0.10]})
    print(f"[gate] s42 AUPRC {gates['s42_auprc']:.4f} (expect {EXPECTED_S42_AUPRC}); "
          f"sweep s42 {sw[0.005]:.4f} / {sw[0.10]:.4f} (expect 0.3468 / 0.8682)")
    if (abs(gates["s42_auprc"] - EXPECTED_S42_AUPRC) > 5e-5
            or abs(sw[0.005] - 0.3468) > 5e-4 or abs(sw[0.10] - 0.8682) > 5e-4):
        raise SystemExit("flagship gate failed")

    # Ranking: one set of 1,000 patient-clustered resamples for every score.
    scores = {f"flagship_s{s}": flagship[s]["test"] for s in seeds}
    scores.update({a: arms[a]["test"] for a in ARMS})
    scores["fair_traj"] = fair_test
    targets = {"auprc": ("auprc", y), "auroc": ("auroc", y),
               "death_auprc": ("auprc", y_death)}
    fns = {"auprc": (auprc, y), "auroc": (auroc, y), "death_auprc": (auprc, y_death)}
    points = {t: {n: float(f(yy, s)) for n, s in scores.items()}
              for t, (f, yy) in fns.items()}
    points["death_auprc"].update(
        {f"dhead_s{s}": float(auprc(y_death, dhead[s])) for s in seeds})
    t1 = time.time()
    draws = metric_draws(targets, scores, ctx.patient_test, reps=REPS, seed=SEED,
                         workers=workers)
    n_used = len(draws["auprc"]["allfeat"])
    print(f"[bootstrap] {n_used} shared resamples x {len(scores)} scores x 3 "
          f"targets ({time.time() - t1:.0f} s)")

    ranking = {}
    for t in targets:
        ranking[t] = {"points": points[t], "vs": {}}
        for opp in (*ARMS, "fair_traj"):
            per_seed = {s: paired_delta(points[t][f"flagship_s{s}"], points[t][opp],
                                        draws[t][f"flagship_s{s}"], draws[t][opp])
                        for s in seeds}
            ranking[t]["vs"][opp] = {
                "pooled": pooled_delta([points[t][f"flagship_s{s}"] for s in seeds],
                                       points[t][opp],
                                       [draws[t][f"flagship_s{s}"] for s in seeds],
                                       draws[t][opp]),
                "per_seed": per_seed,
                "seeds_ci_above_zero": sum(d["ci_above_zero"] for d in per_seed.values()),
                "seeds_ci_below_zero": sum(d["ci"][1] < 0 for d in per_seed.values()),
            }
    raw_auprc = {a: float(auprc(y, arms[a]["rawtest"])) for a in ARMS}

    # Catch sweep: each model at its own (1 - b) V_freeze quantile.
    sweep = []
    for b in SWEEP_BUDGETS:
        row = {"budget": b, "arms": {}, "flagship_by_seed": {},
               "flagship_at_allfeat_test_volume": {}}
        for a in ARMS:
            row["arms"][a] = _sweep_row(ctx, a, arms[a]["val"], arms[a]["test"],
                                        budget=b)
        target_rate = row["arms"]["allfeat"]["test_alarm_rate"]
        for s in seeds:
            f = flagship[s]
            row["flagship_by_seed"][s] = _sweep_row(ctx, "m", f["val"], f["test"],
                                                    budget=b)
            # Diagnostic, not protocol: threshold from TEST score quantiles so
            # the flagship fires on the GBM's realized test volume.
            thr_vm = float(np.quantile(f["test"][ctx.alertable_test], 1.0 - target_rate))
            row["flagship_at_allfeat_test_volume"][s] = _sweep_row(
                ctx, "m", f["val"], f["test"], thr=thr_vm)
        c = [row["flagship_by_seed"][s]["catch"] for s in seeds]
        cv = [row["flagship_at_allfeat_test_volume"][s]["catch"] for s in seeds]
        g = row["arms"]["allfeat"]["catch"]
        row["summary"] = {
            "flagship_mean": float(np.mean(c)), "flagship_min": float(min(c)),
            "flagship_max": float(max(c)),
            "seeds_ahead_of_allfeat": int(sum(x > g for x in c)),
            "equal_volume_mean": float(np.mean(cv)),
            "equal_volume_seeds_ahead": int(sum(x > g for x in cv)),
        }
        sweep.append(row)
        print(f"[sweep] {b:>5.1%}: flagship {np.mean(c):.1%} [{min(c):.1%}, {max(c):.1%}] "
              f"| allfeat {g:.1%} ({row['arms']['allfeat']['test_alarm_rate']:.2%}) "
              f"| nocad {row['arms']['allfeat_nocad']['catch']:.1%} "
              f"| equal-volume flagship {np.mean(cv):.1%} "
              f"| seeds ahead {row['summary']['seeds_ahead_of_allfeat']}/10")

    # Races vs the all-features GBM: protocol, then the equal-volume diagnostic.
    races = []
    for b in RACE_BUDGETS:
        thr_g = freeze_quantile(ctx, arms["allfeat"]["val"], b)
        g_rate = test_rate(ctx, arms["allfeat"]["test"], thr_g)
        for s in seeds:
            f = flagship[s]
            thr_m = freeze_quantile(ctx, f["val"], b)
            thr_vm = float(np.quantile(f["test"][ctx.alertable_test], 1.0 - g_rate))
            races.append({
                "budget": b, "seed": s, "allfeat_test_rate": g_rate,
                "protocol": {"model_test_rate": test_rate(ctx, f["test"], thr_m),
                             **race(ctx, arms["allfeat"]["test"], f["test"],
                                    thr_g, thr_m, REPS)},
                "equal_volume_diagnostic": {
                    "model_test_rate": test_rate(ctx, f["test"], thr_vm),
                    **race(ctx, arms["allfeat"]["test"], f["test"], thr_g, thr_vm, REPS)},
            })
        for tag in ("protocol", "equal_volume_diagnostic"):
            r = [x[tag] for x in races if x["budget"] == b]
            first = sum(1 for x in r if x["sign_p"] < 0.05 and x["earlier_when_differ"] > 0.5)
            print(f"[race] {b:.0%} {tag:<24} first-when-differ "
                  f"{np.mean([x['earlier_when_differ'] for x in r]):.0%} | median lead "
                  f"{np.mean([x['median'] for x in r]):+.2f} h | flagship-first p<0.05 "
                  f"{first}/10")

    # Exploratory, added after the pre-registered run and NOT part of the pass
    # rule: where the ranking gap comes from (AUPRC per event part), and what
    # the stay-level catch lead costs (quiet stays alarmed, stay precision).
    from decompose import treatment_only_label
    t = ctx.test_ordered
    parts = {"vent": t["label_24h_vent"].to_numpy(dtype=float),
             "vaso": t["label_24h_vaso"].to_numpy(dtype=float),
             "death": y_death,
             "treatment_only": np.asarray(treatment_only_label(t), dtype=float)}
    part_auprc = {p: {"prevalence": float(yy.mean()),
                      **{n: float(auprc(yy, s)) for n, s in scores.items()}}
                  for p, yy in parts.items()}
    n_stays = int(t["stay_id"].nunique())
    n_event = int(t.groupby("stay_id")["event_h"].first().notna().sum())
    n_alertable = int(ctx.alertable_test.sum())

    def stay_cost(x):
        caught, quiet = x["catch"] * n_event, x["quiet_fp"] * (n_stays - n_event)
        alarmed = caught + quiet
        return {"stay_precision": float(caught / alarmed) if alarmed else float("nan"),
                "stays_alarmed": float(alarmed),
                "alarm_hours_per_alarmed_stay":
                    float(x["test_alarm_rate"] * n_alertable / alarmed) if alarmed else float("nan")}

    stay_costs = []
    for r in sweep:
        row = {"budget": r["budget"],
               **{a: stay_cost(r["arms"][a]) for a in ARMS}}
        for key, src in (("flagship", "flagship_by_seed"),
                         ("flagship_at_allfeat_test_volume", "flagship_at_allfeat_test_volume")):
            per = [stay_cost(x) for x in r[src].values()]
            row[key] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        stay_costs.append(row)
        print(f"[cost] {r['budget']:>5.1%}: stay precision flagship "
              f"{row['flagship']['stay_precision']:.1%} | at GBM volume "
              f"{row['flagship_at_allfeat_test_volume']['stay_precision']:.1%} | allfeat "
              f"{row['allfeat']['stay_precision']:.1%}")
    exploratory = {"note": "added after the pre-registered run; not part of the pass rule",
                   "n_test_stays": n_stays, "n_event_stays": n_event,
                   "auprc_by_event_part": part_auprc, "stay_costs": stay_costs}

    verdict = ranking["auprc"]["vs"]["allfeat"]
    payload = {
        "test": "T30", "seeds": seeds, "reps": REPS, "resamples_used": n_used,
        "gbm_seed": SEED_GBM, "fingerprint": fp_cached[:16], "gates": gates,
        "gbm_meta": meta, "raw_auprc_check": raw_auprc, "ranking": ranking,
        "sweep": sweep, "races": races, "exploratory": exploratory,
        "pass_rule": "pooled paired dAUPRC (flagship - allfeat) 95% CI above zero",
        "pass": verdict["pooled"]["ci_above_zero"],
        "note": "aggregate-only; equal-volume rows set the flagship threshold "
                "from test score quantiles (diagnostic, not protocol)",
    }
    payload = json.loads(json.dumps(payload, default=float))
    assert_aggregate_only(payload)
    OUT.write_text(json.dumps(payload, indent=1))
    p = verdict["pooled"]
    print(f"\n[verdict] pooled dAUPRC flagship - allfeat {p['delta']:+.4f} "
          f"[{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]; seeds clear "
          f"{verdict['seeds_ci_above_zero']}/10 -> {'PASS' if payload['pass'] else 'FAIL'}")
    print(f"[saved] {OUT} ({time.time() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("fit", "eval"), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    if args.stage == "fit":
        stage_fit(args.device)
    else:
        stage_eval(args.workers)


if __name__ == "__main__":
    main()
