"""T47: stack Sentinel with the all-features GBM (results_fusion.md, T47,
pre-registered). MIMIC test split, ten seed pairs.

Parents per seed pair i: the cached flagship composite (``two_head.npz``
``val_s{i}``/``test_s{i}``) and T31's all-features GBM refit
(``t31_seeds_tab.npz`` ``val_allfeat_s{i}``/``test_allfeat_s{i}``), both
calibrated, entered as clipped logits (``src/stacking.py``).

* Arm A (T1 LOGIT3): unpenalised logistic regression on the two logits plus
  an intercept, fit on V_fit.
* Arm B (T1 CONSTR-1H): s = w * logit(Sentinel) + (1 - w) * logit(GBM), w on
  the 0.1 grid, chosen on V_freeze as the highest AUPRC among weights with a
  V_freeze median lead over NEWS2 at 2% of at least +1 h; no qualifying
  weight -> the arm fails.

Stages:

    # V_fit / V_freeze only: arm A's fit, arm B's grid and choice. No test row.
    OMP_NUM_THREADS=1 python scripts/t47_stack.py select --race-thresholds matched
    # test scoring, measures, verdicts
    OMP_NUM_THREADS=1 python scripts/t47_stack.py eval --race-thresholds matched --workers 16

``--race-thresholds`` sets where a model sits in a race at budget b:
``matched`` = NEWS2's achieved V_freeze volume at b (``frozen_thresholds``,
the protocol of T1, T30 and T33); ``own`` = the model's own (1 - b) V_freeze
quantile. It is required and recorded in every output.

Outputs (aggregates only): ``data/cache/t47/selection.json`` and
``data/cache/t47_stack.json``.
"""

import argparse
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
WORK = CACHE / "t47"
SELECTION = WORK / "selection.json"
OUT = CACHE / "t47_stack.json"
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
BUDGET = 0.02
RACE_BUDGETS = (0.01, 0.02, 0.05)
LEAD_FLOOR = 1.0
REPS = 1000
PUBLISHED_SENTINEL_AUPRC = 0.3087       # flagship ten-seed mean (T30)
GATE_TOL = 5e-4


# --- loading -----------------------------------------------------------------

def load_parents(ctx):
    z = np.load(CACHE / "two_head.npz")
    t = np.load(CACHE / "t31_seeds_tab.npz")
    test, val = ctx.test_ordered, ctx.val_ordered
    ok = (np.array_equal(z["stay_id_test"], test["stay_id"].to_numpy())
          and np.array_equal(z["hour_test"], test["hour"].to_numpy())
          and np.array_equal(t["test_stay_id"], test["stay_id"].to_numpy())
          and np.array_equal(t["test_hour"], test["hour"].to_numpy())
          and np.array_equal(t["val_stay_id"], val["stay_id"].to_numpy())
          and np.array_equal(t["val_hour"], val["hour"].to_numpy())
          and all(len(z[f"val_s{s}"]) == len(val) for s in SEEDS))
    if not ok:
        raise SystemExit("GATE: parent scores are not aligned with the FusionContext rows")
    S = {s: (z[f"val_s{s}"].astype(float), z[f"test_s{s}"].astype(float)) for s in SEEDS}
    G = {s: (t[f"val_allfeat_s{s}"].astype(float), t[f"test_allfeat_s{s}"].astype(float))
         for s in SEEDS}
    return S, G


# --- thresholds and races ----------------------------------------------------------

def race_thresholds(ctx, s_val, b, mode, opp_val=None):
    """(opponent threshold, model threshold, volume) at budget ``b``. The
    opponent is NEWS2 unless ``opp_val`` is given."""
    from fair_capability import frozen_thresholds
    from paper_operating_points import freeze_quantile
    fm = ctx.freeze_mask
    nf, af = ctx.news2_val[fm], ctx.alertable_val[fm]
    thr_n2, thr_m, matched = frozen_thresholds(nf, s_val[fm], af, b)
    vol = matched if mode == "matched" else b
    if mode == "own":
        thr_n2 = float(np.quantile(nf[af], 1.0 - b))
        thr_m = freeze_quantile(ctx, s_val, b)
    if opp_val is None:
        return thr_n2, thr_m, vol
    return freeze_quantile(ctx, opp_val, vol), freeze_quantile(ctx, s_val, vol), vol


def freeze_lead(ctx, s_val, mode, b=BUDGET):
    """Median lead over NEWS2 on V_freeze (arm B's selection surface)."""
    from fair_capability import paired_first_alarm_diffs
    fm = ctx.freeze_mask
    thr_n2, thr_m, _ = race_thresholds(ctx, s_val, b, mode)
    d = paired_first_alarm_diffs(ctx.freeze_frame, ctx.news2_val[fm], s_val[fm], thr_n2, thr_m)
    return float(np.median(d["diff"])) if len(d) else float("nan")


def test_race(ctx, s_val, s_test, b, mode, opp=None):
    """Per-stay diffs on test (opponent's first alarm - model's)."""
    from fair_capability import paired_first_alarm_diffs
    if opp is None:
        thr_o, thr_m, vol = race_thresholds(ctx, s_val, b, mode)
        o_test = ctx.news2_test
    else:
        thr_o, thr_m, vol = race_thresholds(ctx, s_val, b, mode, opp_val=opp[0])
        o_test = opp[1]
    d = paired_first_alarm_diffs(ctx.test_ordered, o_test, s_test, thr_o, thr_m)
    return d, vol


def pooled_race(ctx, per_seed):
    """``per_seed``: list of diff frames. Ten-seed mean of per-seed medians,
    and share-first with its CI on the shared resamples."""
    from stacking import pooled_share_first_ci
    from config import SEED
    diffs = [d["diff"].to_numpy(dtype=float) for d in per_seed]
    pats = [d["stay_id"].map(ctx.stay_to_patient).to_numpy() for d in per_seed]
    point, ci, n = pooled_share_first_ci(diffs, pats, ctx.patient_test, REPS, SEED)
    medians = [float(np.median(x)) if len(x) else float("nan") for x in diffs]
    return {"lead_h": float(np.nanmean(medians)), "per_seed_median_h": medians,
            "share_first": point, "share_first_ci": ci, "resamples": n,
            "n_pairs_per_seed": [int(len(x)) for x in diffs]}


# --- stages --------------------------------------------------------------------

def _context():
    from baselines_mimic import load_splits
    from fusion_common import FusionContext
    return FusionContext(load_splits())


def stage_select(mode):
    from metrics import auprc
    from stacking import W_GRID, blend, choose_weight, fit_logit3, logit
    ctx = _context()
    S, G = load_parents(ctx)
    fm, fit = ctx.freeze_mask, ctx.fit_mask
    gate = freeze_lead(ctx, S[42][0], "matched")
    if gate != ctx.freeze_lead(S[42][0]):
        raise SystemExit("GATE: freeze_lead does not reproduce FusionContext.freeze_lead")
    sel = {"race_thresholds": mode, "lead_floor_h": LEAD_FLOOR, "budget": BUDGET, "seeds": {}}
    for s in SEEDS:
        lS, lG = logit(S[s][0]), logit(G[s][0])
        m = fit_logit3(lS[fit], lG[fit], ctx.y_val[fit])
        grid = []
        for w in W_GRID:
            v = blend(w, lS, lG)
            grid.append({"w": w, "lead": freeze_lead(ctx, v, mode),
                         "auprc": float(auprc(ctx.y_val[fm], v[fm]))})
        chosen = choose_weight(grid, LEAD_FLOOR)
        sel["seeds"][str(s)] = {
            "logit3": {"coef_sentinel": float(m.coef_[0, 0]), "coef_gbm": float(m.coef_[0, 1]),
                       "intercept": float(m.intercept_[0])},
            "grid": grid, "w": None if chosen is None else chosen["w"]}
        print(f"[select] s{s}: LOGIT3 coef S {m.coef_[0, 0]:+.3f} G {m.coef_[0, 1]:+.3f} "
              f"b {m.intercept_[0]:+.3f} | CONSTR-1H w = "
              f"{'none' if chosen is None else chosen['w']}", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    SELECTION.write_text(json.dumps(sel, indent=1))
    print(f"[saved] {SELECTION.relative_to(REPO)}")


def _scores(ctx, S, G, sel):
    from stacking import blend, fit_logit3, logit, logit3_score
    fit = ctx.fit_mask
    A, B = {}, {}
    for s in SEEDS:
        lSv, lGv, lSt, lGt = logit(S[s][0]), logit(G[s][0]), logit(S[s][1]), logit(G[s][1])
        m = fit_logit3(lSv[fit], lGv[fit], ctx.y_val[fit])
        rec = sel["seeds"][str(s)]["logit3"]
        if not np.allclose([m.coef_[0, 0], m.coef_[0, 1], m.intercept_[0]],
                           [rec["coef_sentinel"], rec["coef_gbm"], rec["intercept"]], rtol=0, atol=1e-9):
            raise SystemExit("GATE: arm A's refit differs from the recorded selection")
        A[s] = (logit3_score(m, lSv, lGv), logit3_score(m, lSt, lGt))
        w = sel["seeds"][str(s)]["w"]
        if w is not None:
            B[s] = (blend(w, lSv, lGv), blend(w, lSt, lGt))
    return A, B


def stage_eval(mode, workers):
    from demo_bundle import assert_aggregate_only, first_alarm_table
    from metrics import auprc, auroc
    from paper_operating_points import SWEEP_BUDGETS, catch_stats, freeze_quantile, test_rate
    from paper_stats import metric_draws, pooled_delta
    from t35_eicu_operating_points import (catch_at_quiet_share, quiet_alarm_share,
                                           stay_max_scores)
    t0 = time.time()
    sel = json.loads(SELECTION.read_text())
    if sel["race_thresholds"] != mode:
        raise SystemExit("the selection was made under the other race-threshold rule")
    ctx = _context()
    S, G = load_parents(ctx)
    A, B = _scores(ctx, S, G, sel)
    test, y = ctx.test_ordered, ctx.y_test
    b_seeds = [s for s in SEEDS if s in B]

    s_auprc = [float(auprc(y, S[s][1])) for s in SEEDS]
    gate = {"sentinel_ten_seed_auprc": float(np.mean(s_auprc)),
            "published": PUBLISHED_SENTINEL_AUPRC}
    gate["pass"] = bool(abs(gate["sentinel_ten_seed_auprc"] - PUBLISHED_SENTINEL_AUPRC) <= GATE_TOL)
    print(f"[gate] Sentinel ten-seed AUPRC {gate['sentinel_ten_seed_auprc']:.4f} "
          f"(published {PUBLISHED_SENTINEL_AUPRC})", flush=True)
    if not gate["pass"]:
        raise SystemExit("GATE: cached Sentinel scores do not give the published AUPRC")

    arms = {"S": S, "G": G, "A": A, "B": B}
    scores = {f"{k}_s{s}": v[s][1] for k, v in arms.items() for s in v}
    pts = {m: {n: float(fn(y, v)) for n, v in scores.items()}
           for m, fn in (("auprc", auprc), ("auroc", auroc))}
    t1 = time.time()
    draws = metric_draws({"auprc": ("auprc", y), "auroc": ("auroc", y)}, scores,
                         ctx.patient_test, reps=REPS, workers=workers)
    if any(len(d) != REPS for t in draws.values() for d in t.values()):
        raise SystemExit("a resample was skipped; the shared-resample race CIs assume none is")
    print(f"[bootstrap] {len(scores)} scores x {REPS} in {time.time() - t1:.0f} s", flush=True)

    def summary(k, seeds_):
        out = {}
        for m in ("auprc", "auroc"):
            vals = [pts[m][f"{k}_s{s}"] for s in seeds_]
            dr = np.mean([draws[m][f"{k}_s{s}"] for s in seeds_], axis=0)
            out[m] = {"mean": float(np.mean(vals)), "min": float(min(vals)),
                      "max": float(max(vals)), "ci": [float(np.percentile(dr, 2.5)),
                                                      float(np.percentile(dr, 97.5))]}
        return out

    def delta_vs_gbm(k, seeds_):
        out = {}
        for m in ("auprc", "auroc"):
            d = pooled_delta([pts[m][f"{k}_s{s}"] for s in seeds_],
                             [pts[m][f"G_s{s}"] for s in seeds_],
                             [draws[m][f"{k}_s{s}"] for s in seeds_],
                             [draws[m][f"G_s{s}"] for s in seeds_])
            d["seeds_positive"] = int(sum(pts[m][f"{k}_s{s}"] > pts[m][f"G_s{s}"] for s in seeds_))
            d["seeds"] = len(seeds_)
            out[m] = d
        return out

    res = {"test": "T47 stacking Sentinel with the all-features GBM", "race_thresholds": mode,
           "seeds": list(SEEDS), "reps": REPS, "gate": gate, "selection": sel,
           "rows": int(len(y)), "positives": int(y.sum()),
           "patients": int(len(np.unique(ctx.patient_test))), "arms": {}}
    names = {"A": "LOGIT3 stack", "B": "CONSTR-1H stack", "S": "Sentinel (parent)",
             "G": "all-features GBM (parent)"}
    for k in ("A", "B", "S", "G"):
        seeds_ = b_seeds if k == "B" else list(SEEDS)
        if not seeds_:
            res["arms"][k] = {"name": names[k], "seeds": []}
            continue
        arm = {"name": names[k], "seeds": seeds_, "ranking": summary(k, seeds_)}
        if k != "G":
            arm["vs_gbm"] = delta_vs_gbm(k, seeds_)
        arm["news2_race"], arm["gbm_race"] = {}, {}
        for b in RACE_BUDGETS:
            per, vols = [], []
            for s in seeds_:
                d, vol = test_race(ctx, arms[k][s][0], arms[k][s][1], b, mode)
                per.append(d)
                vols.append(vol)
            arm["news2_race"][str(b)] = {**pooled_race(ctx, per), "volume_mean": float(np.mean(vols))}
            if k != "G":
                per = [test_race(ctx, arms[k][s][0], arms[k][s][1], b, mode, opp=G[s])[0]
                       for s in seeds_]
                arm["gbm_race"][str(b)] = pooled_race(ctx, per)
        sweep = {}
        for b in SWEEP_BUDGETS:
            cs = []
            for s in seeds_:
                thr = freeze_quantile(ctx, arms[k][s][0], b)
                st = catch_stats(first_alarm_table(test, {"m": arms[k][s][1]}, {"m": thr}), "m")
                cs.append({"catch": st["catch"], "catch_6h": st["catch_6h"],
                           "quiet_alarmed": st["quiet_fp"],
                           "test_alarm_rate": test_rate(ctx, arms[k][s][1], thr)})
            sweep[str(b)] = {m: float(np.mean([c[m] for c in cs])) for m in cs[0]}
        arm["catch_sweep"] = sweep
        if k != "G":
            quiet = {}
            for b in SWEEP_BUDGETS:
                own, gb = [], []
                for s in seeds_:
                    q, _ = quiet_alarm_share(test, G[s][1], freeze_quantile(ctx, G[s][0], b))
                    own.append(catch_at_quiet_share(*stay_max_scores(test, arms[k][s][1]), q))
                    gb.append(catch_at_quiet_share(*stay_max_scores(test, G[s][1]), q))
                quiet[str(b)] = {"model_catch_mean": float(np.mean(own)),
                                 "gbm_catch_mean": float(np.mean(gb)),
                                 "seeds_ahead": int(sum(o > g for o, g in zip(own, gb)))}
            arm["catch_at_gbm_quiet_share"] = quiet
        res["arms"][k] = arm
        r = arm["ranking"]["auprc"]
        print(f"[{k}] AUPRC {r['mean']:.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]"
              + (f" | dAUPRC vs GBM {arm['vs_gbm']['auprc']['delta']:+.4f} "
                 f"{arm['vs_gbm']['auprc']['ci']}" if k != "G" else "")
              + f" | lead@2% {arm['news2_race']['0.02']['lead_h']:+.2f} h, share-first "
              f"{arm['news2_race']['0.02']['share_first']:.3f} {arm['news2_race']['0.02']['share_first_ci']}",
              flush=True)

    a, bb = res["arms"]["A"], res["arms"]["B"]
    verdict_a = bool(a["vs_gbm"]["auprc"]["ci_above_zero"])
    if len(b_seeds) < len(SEEDS):
        verdict_b = False
        reason_b = f"no qualifying weight at seeds {[s for s in SEEDS if s not in B]}"
    else:
        r2 = bb["news2_race"]["0.02"]
        ranking_ok = bool(bb["vs_gbm"]["auprc"]["ci"][1] >= 0)
        timing_ok = bool(r2["lead_h"] >= LEAD_FLOOR and r2["share_first_ci"][0] > 0.5)
        verdict_b = ranking_ok and timing_ok
        reason_b = {"ranking_ci_includes_or_above_zero": ranking_ok, "timing_pass": timing_ok}
    res["verdicts"] = {"arm_A_pass": verdict_a, "arm_B_pass": verdict_b, "arm_B_detail": reason_b}
    res["note"] = ("aggregate-only; both parents early-stop on validation data; arm A fit on "
                   "V_fit, arm B chosen on V_freeze; catches at the GBM's quiet-stay share are "
                   "a test-score diagnostic")
    res["seconds"] = round(time.time() - t0, 1)
    res = json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    assert_aggregate_only(res)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"[verdict] arm A {'PASS' if verdict_a else 'FAIL'}; arm B "
          f"{'PASS' if verdict_b else 'FAIL'} ({reason_b})")
    print(f"[saved] {OUT.relative_to(REPO)} ({res['seconds']} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "eval"])
    ap.add_argument("--race-thresholds", choices=["matched", "own"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.stage == "select":
        stage_select(a.race_thresholds)
    else:
        stage_eval(a.race_thresholds, a.workers)


if __name__ == "__main__":
    main()
