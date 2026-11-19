"""T48: alarm-persistence rules (pre-registered in results_fusion.md, T48).

Models (seeds 42 and 1-9): shipped Sentinel, the T43 variant, the T47 stacks
(LOGIT3 and CONSTR-1H) and the all-features GBM (T31 refits), each with the
same rule options: alarm after k = 1, 2 or 3 consecutive hours above
threshold, or hysteresis.

* Budget b is NEWS2's achieved V_freeze volume at nominal b (the dated
  clarification under the T43-T49 shared rules): about 1.23%, 2.62% and 5.05%
  at 1%, 2% and 5%.
* The GBM's quiet-stay share is fixed by its plain k = 1 rule at its own
  V_freeze quantile for that volume. On V_freeze, each model and seed picks
  the rule with the most catches at the GBM's V_freeze quiet-stay share; ties
  go to the simpler rule (k = 1, 2, 3, then hysteresis).
* On test, the GBM's k = 1 rule at its V_freeze threshold fixes the quiet-stay
  share, and each model's catch with its chosen rule is read at that share
  (T38's method, ``persistence.catch_at_quiet_share``).
* Pass: a Sentinel variant catches more test event stays than the GBM at 2%
  and at 5%, with the CI of the ten-seed mean difference above zero. The CI
  uses 1,000 patient-clustered resamples of stays (the shared resamples), and
  the whole test step is recomputed in each.
* Hysteresis (on at the model's (1 - b) V_freeze quantile, off below its
  (1 - 2b) quantile) gives the same first alarms as k = 1, so on catches and
  quiet-stay share it equals k = 1 (unit-tested). Its share of test hours in
  alarm is reported beside k = 1's.

    OMP_NUM_THREADS=1 python scripts/t48_persistence.py --workers 16

Aggregates only, to ``data/cache/t48_persistence.json``.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

OUT = REPO / "data" / "cache" / "t48_persistence.json"
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
BUDGETS = (0.01, 0.02, 0.05)
PASS_BUDGETS = (0.02, 0.05)
RULES = ("k1", "k2", "k3", "hysteresis")
K = {"k1": 1, "k2": 2, "k3": 3, "hysteresis": 1}   # stay-level equivalence (see docstring)
SENTINELS = ("sentinel", "t43", "logit3", "constr1h")
MODELS = SENTINELS + ("gbm",)
REPS = 1000


def _load():
    from t43_gbm_agent import HEADS as T43_HEADS
    from t43_gbm_agent import _composite
    from t47_stack import SELECTION, _context, _scores, load_parents
    ctx = _context()
    S, G = load_parents(ctx)
    sel = json.loads(SELECTION.read_text())
    if sel["race_thresholds"] != "matched":
        raise SystemExit("T47's selection was made under the other race-threshold rule")
    A, B = _scores(ctx, S, G, sel)
    if set(B) != set(SEEDS):
        raise SystemExit("CONSTR-1H lacks a seed")
    N = {s: _composite(ctx, np.load(T43_HEADS / f"new_s{s}.npz")) for s in SEEDS}
    return ctx, {"sentinel": S, "t43": N, "logit3": A, "constr1h": B, "gbm": G}


_W = {}


def _boot_init(payload):
    _W.update(payload)


def _boot_chunk(rows):
    """Ten-seed mean catch differences (model - GBM) per resample in ``rows``."""
    from persistence import catch_at_quiet_share, share_at_or_above
    P = _W
    out = {b: {m: [] for m in SENTINELS + ("gbm",)} for b in P["budgets"]}
    for r in rows:
        w = P["counts"][r][P["stay_patient"]]
        we, wq = w[P["ep"]], w[P["qp"]]
        for b in P["budgets"]:
            per = {m: [] for m in MODELS}
            for i, s in enumerate(SEEDS):
                q = share_at_or_above(P["gbm_k1_quiet"][b][i], P["thr_g"][b][i], wq)
                for m in MODELS:
                    em, qm = P["chosen"][b][m][i]
                    per[m].append(catch_at_quiet_share(em, qm, q, we, wq))
            g = np.array(per["gbm"])
            for m in SENTINELS:
                out[b][m].append(float(np.mean(np.array(per[m]) - g)))
            out[b]["gbm"].append(float(np.mean(g)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    from config import SEED
    from demo_bundle import assert_aggregate_only
    from paper_operating_points import freeze_quantile
    from paper_stats import ci
    from persistence import (catch_at_quiet_share, hysteresis_state, run_min,
                             share_at_or_above, stay_maxima)
    from stacking import resample_counts
    from t47_stack import race_thresholds
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; new outputs go to new files")
    t0 = time.time()
    ctx, models = _load()
    val, test, fm = ctx.val_ordered, ctx.test_ordered, ctx.freeze_mask
    ff = val[fm]
    print(f"[load] {time.time() - t0:.0f} s", flush=True)

    def rule_scores(frame, s, k):
        return run_min(s, frame["stay_id"].to_numpy(), frame["hour"].to_numpy(), k)

    def maxima(frame, s):
        em, qm, ep, qp = stay_maxima(frame["stay_id"].to_numpy(), frame["hour"].to_numpy(),
                                     frame["event_h"].to_numpy(dtype=float), s)
        return em, qm, ep, qp

    # per model, seed and k: stay maxima on V_freeze and on test
    mx_f, mx_t = {}, {}
    for m in MODELS:
        for s in SEEDS:
            v, t = models[m][s]
            for k in (1, 2, 3):
                mx_f[(m, s, k)] = maxima(ff, rule_scores(val, v, k)[fm])
                mx_t[(m, s, k)] = maxima(test, rule_scores(test, t, k))
    _, _, ep, qp = mx_t[("gbm", 42, 1)]
    print(f"[maxima] {time.time() - t0:.0f} s; test stays: {len(ep):,} event, {len(qp):,} quiet",
          flush=True)

    res = {"test": "T48 alarm-persistence rules", "seeds": list(SEEDS), "reps": REPS,
           "rules": list(RULES), "budget_protocol": "NEWS2-matched V_freeze volume",
           "budgets": {}}
    boot = {"budgets": BUDGETS, "thr_g": {}, "gbm_k1_quiet": {}, "chosen": {}}
    for b in BUDGETS:
        vol = race_thresholds(ctx, models["gbm"][42][0], b, "matched")[2]
        rb = {"volume": float(vol), "models": {}}
        thr_g = [freeze_quantile(ctx, models["gbm"][s][0], vol) for s in SEEDS]
        q_f = [share_at_or_above(mx_f[("gbm", s, 1)][1], thr_g[i]) for i, s in enumerate(SEEDS)]
        q_t = [share_at_or_above(mx_t[("gbm", s, 1)][1], thr_g[i]) for i, s in enumerate(SEEDS)]
        rb["gbm_quiet_share_v_freeze"] = float(np.mean(q_f))
        rb["gbm_quiet_share_test"] = float(np.mean(q_t))
        boot["thr_g"][b] = thr_g
        boot["gbm_k1_quiet"][b] = [mx_t[("gbm", s, 1)][1] for s in SEEDS]
        boot["chosen"][b] = {}
        point = {}
        for m in MODELS:
            chosen, f_by_rule, t_by_rule, t_chosen, alarm_hours = [], {}, {}, [], {"k1": [], "hysteresis": []}
            boot["chosen"][b][m] = []
            for i, s in enumerate(SEEDS):
                cf = {r: catch_at_quiet_share(*mx_f[(m, s, K[r])][:2], q_f[i]) for r in RULES}
                best = max(cf.values())
                r_star = next(r for r in RULES if cf[r] == best)
                chosen.append(r_star)
                for r in RULES:
                    f_by_rule.setdefault(r, []).append(cf[r])
                    t_by_rule.setdefault(r, []).append(
                        catch_at_quiet_share(*mx_t[(m, s, K[r])][:2], q_t[i]))
                t_chosen.append(t_by_rule[r_star][-1])
                boot["chosen"][b][m].append(mx_t[(m, s, K[r_star])][:2])
                v, t = models[m][s]
                on, off = freeze_quantile(ctx, v, vol), freeze_quantile(ctx, v, 2 * vol)
                alarm_hours["k1"].append(float((t >= on).mean()))
                alarm_hours["hysteresis"].append(float(hysteresis_state(
                    t, test["stay_id"].to_numpy(), on, off).mean()))
            point[m] = t_chosen
            rb["models"][m] = {
                "chosen_rules": {r: chosen.count(r) for r in RULES},
                "v_freeze_catch_by_rule": {r: float(np.mean(x)) for r, x in f_by_rule.items()},
                "test_catch_by_rule": {r: float(np.mean(x)) for r, x in t_by_rule.items()},
                "test_catch_chosen": float(np.mean(t_chosen)),
                "test_alarm_hour_share": {r: float(np.mean(x)) for r, x in alarm_hours.items()}}
        for m in SENTINELS:
            d = np.array(point[m]) - np.array(point["gbm"])
            rb["models"][m]["vs_gbm"] = {"delta": float(d.mean()),
                                         "seeds_ahead": int((d > 0).sum())}
        res["budgets"][str(b)] = rb
        print(f"[b={b}] vol {vol:.4f}; GBM quiet share test {np.mean(q_t):.3f}; " + "; ".join(
            f"{m} {rb['models'][m]['test_catch_chosen']:.3f} {rb['models'][m]['chosen_rules']}"
            for m in MODELS), flush=True)

    # patient-clustered resamples of stays, shared (resample_counts == bootstrap_indices)
    stays = test.drop_duplicates("stay_id")["stay_id"].to_numpy()
    patients, _ = next(resample_counts(ctx.patient_test, 1, SEED))
    stay_patient = np.searchsorted(patients, np.array([ctx.stay_to_patient[s] for s in stays]))
    counts = np.stack([c for _, c in resample_counts(ctx.patient_test, REPS, SEED)])
    boot.update(counts=counts, stay_patient=stay_patient, ep=ep, qp=qp)
    t1 = time.time()
    chunks = np.array_split(np.arange(REPS), args.workers * 4)
    with mp.get_context("spawn").Pool(args.workers, initializer=_boot_init, initargs=(boot,)) as pool:
        parts = pool.map(_boot_chunk, chunks)
    print(f"[bootstrap] {REPS} resamples in {time.time() - t1:.0f} s", flush=True)
    for b in BUDGETS:
        rb = res["budgets"][str(b)]
        for m in SENTINELS:
            draws = np.concatenate([p[b][m] for p in parts])
            rb["models"][m]["vs_gbm"]["ci"] = ci(draws)
            rb["models"][m]["vs_gbm"]["ci_above_zero"] = bool(ci(draws)[0] > 0)
        rb["models"]["gbm"]["test_catch_chosen_ci"] = ci(np.concatenate([p[b]["gbm"] for p in parts]))
    passing = [m for m in SENTINELS if all(
        res["budgets"][str(b)]["models"][m]["vs_gbm"]["ci_above_zero"] for b in PASS_BUDGETS)]
    res["verdict"] = {"pass": bool(passing), "passing_variants": passing,
                      "note": "any one of four Sentinel variants may pass (stated multiplicity)"}
    res["seconds"] = round(time.time() - t0, 1)
    res = json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    assert_aggregate_only(res)
    OUT.write_text(json.dumps(res, indent=1))
    for b in BUDGETS:
        rb = res["budgets"][str(b)]["models"]
        print(f"[b={b}] " + "; ".join(f"{m} {rb[m]['vs_gbm']['delta']:+.4f} "
                                      f"[{rb[m]['vs_gbm']['ci'][0]:+.4f}, {rb[m]['vs_gbm']['ci'][1]:+.4f}] "
                                      f"{rb[m]['vs_gbm']['seeds_ahead']}/10" for m in SENTINELS))
    print(f"[verdict] {'PASS' if res['verdict']['pass'] else 'FAIL'} {passing}")
    print(f"[saved] {OUT.relative_to(REPO)} ({res['seconds']} s)")


if __name__ == "__main__":
    main()
