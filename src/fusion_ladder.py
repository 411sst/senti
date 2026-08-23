"""Bundle 4a/4b: can GBM ranking and orchestrator earliness be combined?

The corrected results read as a tradeoff -- FAIR+TRAJ wins test AUPRC
(0.287 vs 0.232) with zero earliness, the core-3+meas orchestrator alarms
median +2h earlier than NEWS2 at matched 1/2/5% budgets. Nobody has tested
whether the tradeoff is real: clinical stacking papers report only ranking
metrics. Every rung here is scored on BOTH axes with the fair_capability
harness under the fusion_common protocol (calibrators/weights on V_fit,
thresholds on V_freeze, test once, all rules re-fit per orchestrator seed).

The ladder (4a), cheapest to strongest:

  MAX          max of the two calibrated scores -- alarms at whichever parent
               crosses its threshold-scale first; zero fitted parameters
               beyond the calibrators.
  LOGIT3       sigmoid(a*logit(cO) + b*logit(cG) + c), 3 params on V_fit.
  CONSTRAINED  w*cO + (1-w)*cG with w chosen on V_freeze to maximise AUPRC
               subject to giving up NO median lead vs the orchestrator alone.

The gate (4b):

  GATED        g*cO + (1-g)*cG with g = sigmoid(a + b1*cO + b2*rise6h(cO)) --
               hands control to the orchestrator exactly when its trajectory
               is moving, which is its documented capability.
  ORACLE-GATE  cO on hours within 24h of the true event, cG elsewhere.
               Label-using HEADROOM BOUND, never a model row: it caps what any
               proximity gate could achieve. If it fails to beat the ladder,
               gating is dead.

    python src/fusion_ladder.py [--stage ladder|gate|all] [--seeds 42,1,2]
                                [--reps 2000] [--auprc-reps 1000] [--quick]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from baselines_mimic import load_splits
from fair_capability import BUDGETS
from fusion_common import (
    FusionContext,
    delta_ci,
    load_fair_scores,
    load_orch_seeds,
    paired_auprc_draws,
)
from metrics import auprc

OUT_CACHE = "data/cache/fusion_ladder_scores.npz"
W_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _strictify(cal, raw):
    """Break isotonic plateaus in raw-score order (magnitude <= 1e-9).

    Isotonic calibration is only WEAKLY monotone: a plateau sitting on the
    val-quantile threshold makes every tied test row fire, blowing far past
    the matched alert volume (observed: 1.97% achieved at a 1.03% match --
    the lead time that "bought" was an artifact). The rank tiebreak keeps the
    probability scale intact for blending while restoring the strict
    monotonicity that quantile volume matching assumes.
    """
    ranks = np.argsort(np.argsort(raw, kind="stable"), kind="stable")
    return cal + 1e-9 * (ranks / max(1, len(raw) - 1))


def _rise6(ordered, s, hours=6):
    """Per-stay ``s_t - s_{t-hours}`` (0.0 in the first hours of a stay)."""
    ser = pd.Series(np.asarray(s, dtype=float))
    prev = ser.groupby(ordered["stay_id"].to_numpy()).shift(hours)
    return (ser - prev).fillna(0.0).to_numpy()


# --- 4a: the ladder ----------------------------------------------------------

def build_ladder(ctx, orch_val_raw, orch_test_raw, fair_val_raw, fair_test_raw):
    """Calibrate both parents on V_fit, then the three rungs.

    Returns ``(rungs, info, cal)`` where each rung maps to ``(s_val, s_test)``
    and ``cal`` carries the calibrated parent scores for downstream stages.
    """
    cO = ctx.fit_calibrator(orch_val_raw)
    cG = ctx.fit_calibrator(fair_val_raw)
    oV = _strictify(cO.transform(orch_val_raw), orch_val_raw)
    oT = _strictify(cO.transform(orch_test_raw), orch_test_raw)
    gV = _strictify(cG.transform(fair_val_raw), fair_val_raw)
    gT = _strictify(cG.transform(fair_test_raw), fair_test_raw)
    cal = {"oV": oV, "oT": oT, "gV": gV, "gT": gT}

    rungs = {"MAX": (np.maximum(oV, gV), np.maximum(oT, gT))}
    info = {}

    X_val = np.column_stack([_logit(oV), _logit(gV)])
    X_test = np.column_stack([_logit(oT), _logit(gT)])
    lr = LogisticRegression().fit(X_val[ctx.fit_mask],
                                  ctx.y_val[ctx.fit_mask])
    rungs["LOGIT3"] = (lr.predict_proba(X_val)[:, 1],
                       lr.predict_proba(X_test)[:, 1])
    info["logit3"] = {"coef": lr.coef_.ravel().tolist(),
                      "intercept": float(lr.intercept_[0])}

    # Earliness-constrained stack: never give up lead the orchestrator has.
    # w=1 recovers the orchestrator, so the constraint set is never empty.
    lead_floor = ctx.freeze_lead(oV)
    grid = []
    for w in W_GRID:
        s_val = w * oV + (1 - w) * gV
        grid.append({"w": float(w), "lead": ctx.freeze_lead(s_val),
                     "auprc": ctx.freeze_auprc(s_val)})
    feasible = [g for g in grid if g["lead"] >= lead_floor]
    chosen = max(feasible, key=lambda g: g["auprc"])
    w = chosen["w"]
    rungs["CONSTRAINED"] = (w * oV + (1 - w) * gV, w * oT + (1 - w) * gT)
    info["constrained"] = {"w": w, "lead_floor": lead_floor, "grid": grid}

    # Team-target rung: "GBM-level AUPRC at lead >= 1h" -- same stack, the
    # floor relaxed from "the parent's own lead" to an absolute +1h, freeing
    # the selection to trade the second hour of lead for ranking.
    feasible_1h = [g for g in grid if g["lead"] >= 1.0]
    if feasible_1h:
        c1 = max(feasible_1h, key=lambda g: g["auprc"])
        w1 = c1["w"]
        rungs["CONSTR-1H"] = (w1 * oV + (1 - w1) * gV,
                              w1 * oT + (1 - w1) * gT)
        info["constr_1h"] = {"w": w1}
    return rungs, info, cal


# --- 4b: proximity gate + oracle bound ---------------------------------------

def _fit_gate(y_fit, o_fit, g_fit, rise_fit):
    """BCE-fit the 3-parameter gate on the V_fit rows (Nelder-Mead: the blend
    is not logistic in its features, so sklearn does not apply)."""
    from scipy.optimize import minimize

    def nll(params):
        a, b1, b2 = params
        gate = 1.0 / (1.0 + np.exp(-(a + b1 * o_fit + b2 * rise_fit)))
        prob = np.clip(gate * o_fit + (1 - gate) * g_fit, 1e-7, 1 - 1e-7)
        return -np.mean(y_fit * np.log(prob) + (1 - y_fit) * np.log(1 - prob))

    res = minimize(nll, x0=np.zeros(3), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-9, "maxiter": 2000})
    return res.x, float(res.fun)


def gate_blend(params, o, g, rise):
    a, b1, b2 = params
    gate = 1.0 / (1.0 + np.exp(-(a + b1 * o + b2 * rise)))
    return gate * o + (1 - gate) * g, gate


def build_gate(ctx, cal):
    """The learned proximity gate and the oracle headroom bound."""
    oV, oT, gV, gT = cal["oV"], cal["oT"], cal["gV"], cal["gT"]
    rise_val = _rise6(ctx.val_ordered, oV)
    rise_test = _rise6(ctx.test_ordered, oT)

    params, nll = _fit_gate(ctx.y_val[ctx.fit_mask], oV[ctx.fit_mask],
                            gV[ctx.fit_mask], rise_val[ctx.fit_mask])
    f_val, gate_val = gate_blend(params, oV, gV, rise_val)
    f_test, _ = gate_blend(params, oT, gT, rise_test)

    # Oracle: true time-to-event decides the parent. The ONLY rung allowed to
    # read event_h -- it bounds every conceivable proximity gate.
    rungs, info = {"GATED": (f_val, f_test)}, {}
    near = []
    for frame in (ctx.val_ordered, ctx.test_ordered):
        tte = frame["event_h"].to_numpy(dtype=float) - frame["hour"].to_numpy()
        near.append(frame["event_h"].notna().to_numpy() & (tte <= 24))
    rungs["ORACLE-GATE"] = (np.where(near[0], oV, gV),
                            np.where(near[1], oT, gT))
    info["gate"] = {"params": [float(p) for p in params], "fit_nll": nll,
                    "mean_gate_val": float(np.mean(gate_val))}
    return rungs, info


# --- Report ------------------------------------------------------------------

def _print_seed_table(seed, evals, fair_name):
    print(f"\n=== seed {seed} ===")
    print(f"  {'model':<13}{'AUPRC':>7}  {'Δ vs ' + fair_name:<22}"
          f"{'Δ vs ORCH-meas':<22}")
    for name, r in evals.items():
        def fmt(ci):
            return ("--" if ci is None
                    else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]")
        print(f"  {name:<13}{r['auprc']:>7.3f}  "
              f"{fmt(r.get('delta_vs_fair_ci')):<22}"
              f"{fmt(r.get('delta_vs_orch_ci')):<22}")
    hdr = (f"  {'budget':>7} {'model':<13}{'n_pair':>7}{'median':>7}"
           f"  {'95% CI':<14}{'win':>5}  {'95% CI':<14}{'sign_p':>9}")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for budget in BUDGETS:
        for name, r in evals.items():
            b = r["budgets"][budget]
            print(f"  {f'{budget*100:.0f}%':>7} {name:<13}{b['n_pair']:>7}"
                  f"{b['median']:>7.1f}  "
                  f"[{b['median_ci'][0]:>4.1f}, {b['median_ci'][1]:>4.1f}]"
                  f"{100*b['win_rate']:>4.0f}%  "
                  f"[{100*b['win_rate_ci'][0]:>3.0f}%, "
                  f"{100*b['win_rate_ci'][1]:>3.0f}%]"
                  f"{b['sign_p']:>9.2g}")


def _print_aggregate(results, fair_name):
    seeds = list(results)
    names = [n for n in results[seeds[0]]["evals"]
             if all(n in results[s]["evals"] for s in seeds)]
    print(f"\n=== aggregate over seeds {seeds} "
          f"(mean; sign = seeds with Δ>0 / lead>0) ===")
    print(f"  {'model':<13}{'AUPRC':>7}{'ΔvsFAIR':>9}{'sign':>6}"
          + "".join(f"{f'lead@{int(b*100)}%':>9}{'sign':>6}" for b in BUDGETS))
    fair_auprcs = {s: results[s]["evals"][fair_name]["auprc"] for s in seeds}
    for name in names:
        rows = [results[s]["evals"][name] for s in seeds]
        mean_auprc = np.mean([r["auprc"] for r in rows])
        deltas = [r["auprc"] - fair_auprcs[s] for s, r in zip(seeds, rows)]
        line = (f"  {name:<13}{mean_auprc:>7.3f}{np.mean(deltas):>+9.4f}"
                f"{sum(d > 0 for d in deltas):>5}/{len(seeds)}")
        for b in BUDGETS:
            leads = [r["budgets"][b]["median"] for r in rows]
            line += (f"{np.mean(leads):>9.1f}"
                     f"{sum(l > 0 for l in leads):>5}/{len(seeds)}")
        print(line)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run(stage="all", reps=2000, auprc_reps=1000, quick=False,
        fair_variant="FAIR+TRAJ", out_cache=OUT_CACHE):
    splits = load_splits()
    ctx = FusionContext(splits)
    fair_val, fair_test = load_fair_scores(ctx.val_ordered, ctx.test_ordered,
                                           fair_variant)
    seeds, orch, meta = load_orch_seeds(ctx.val_ordered, ctx.test_ordered)
    if quick:
        seeds, reps, auprc_reps = seeds[:1], 200, 200
    print(f"[protocol] V_fit {int(ctx.fit_mask.sum()):,} rows / V_freeze "
          f"{int(ctx.freeze_mask.sum()):,} rows (patient-disjoint); "
          f"seeds {seeds}; fair parent {fair_variant}")

    results = {}
    payload = {
        "stay_id_val": ctx.val_ordered["stay_id"].to_numpy(),
        "hour_val": ctx.val_ordered["hour"].to_numpy(),
        "stay_id_test": ctx.test_ordered["stay_id"].to_numpy(),
        "hour_test": ctx.test_ordered["hour"].to_numpy(),
    }
    for seed in seeds:
        rungs, info, cal = build_ladder(ctx, orch[seed]["val"],
                                        orch[seed]["test"],
                                        fair_val, fair_test)
        if stage == "gate":
            rungs = {}
        if stage in ("gate", "all"):
            g_rungs, g_info = build_gate(ctx, cal)
            rungs.update(g_rungs)
            info.update(g_info)

        entries = {"ORCH-meas": (orch[seed]["val"], orch[seed]["test"]),
                   fair_variant: (fair_val, fair_test), **rungs}
        evals = {name: ctx.evaluate_rung(s_val, s_test, reps=reps)
                 for name, (s_val, s_test) in entries.items()}
        draws = paired_auprc_draws(
            ctx.y_test, {n: st for n, (_, st) in entries.items()},
            ctx.patient_test, reps=auprc_reps)
        for name in evals:
            evals[name]["delta_vs_fair_ci"] = (
                None if name == fair_variant
                else delta_ci(draws[name], draws[fair_variant]))
            evals[name]["delta_vs_orch_ci"] = (
                None if name == "ORCH-meas"
                else delta_ci(draws[name], draws["ORCH-meas"]))
        results[seed] = {"evals": evals, "info": info}
        _print_seed_table(seed, evals, fair_variant)

        for name, (s_val, s_test) in rungs.items():
            tag = name.replace("-", "_")
            # float64: every rung is built from _strictify'd parents, whose
            # 1e-9 rank tiebreak is below float32's resolution (7.5e-9 near
            # 0.1) -- storing float32 would re-introduce the isotonic
            # plateaus the tiebreak exists to remove.
            payload[f"val_{tag}_s{seed}"] = s_val
            payload[f"test_{tag}_s{seed}"] = s_test

    if len(seeds) > 1:
        _print_aggregate(results, fair_variant)

    payload["meta_json"] = np.array(json.dumps(_jsonable(
        {"stage": stage, "seeds": seeds, "fair_variant": fair_variant,
         "reps": reps, "auprc_reps": auprc_reps,
         "orch_meta": meta, "results": results})))
    Path(out_cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_cache, **payload)
    print(f"\n[cache] rung scores + results -> {out_cache}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("ladder", "gate", "all"))
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--auprc-reps", type=int, default=1000)
    ap.add_argument("--quick", action="store_true",
                    help="first seed only, 200 bootstrap reps")
    ap.add_argument("--fair", default="FAIR+TRAJ",
                    choices=("FAIR", "FAIR+TRAJ", "FAIR+NOTES"))
    args = ap.parse_args()
    run(stage=args.stage, reps=args.reps, auprc_reps=args.auprc_reps,
        quick=args.quick, fair_variant=args.fair)
