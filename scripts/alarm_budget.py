"""T28 -- the alarm budget: catch, cost and timing against the published GBMs.

    OMP_NUM_THREADS=1 python scripts/alarm_budget.py [--reps 2000] [--reference PATH]

Nothing is trained. Reads three caches:

    data/cache/two_head.npz          the two-head flagship, ten seeds (T9/T18)
    data/cache/lit_gbm_scores.npz    the eCART- and YAIB-style GBMs (T10)
    data/cache/standby_report.json   the published 2% thresholds (T12), as a pin

and writes the aggregate-only data/cache/alarm_budget.json:

* ``h2h``    -- seed 42, every model frozen on V_freeze at the matched volume of
                the 2% NEWS2 budget: catch funnel, warning depth, the
                discordant race, false-alarm cost;
* ``seeds``  -- the same race with the flagship refrozen per seed (the GBMs are
                deterministic single fits);
* ``sweep``  -- catch and quiet-stay false-alarm rate at seven nominal budgets
                plus the operating point;
* ``equal_cost`` -- catch at EQUAL quiet-stay false-alarm rates (stay-level
                curve), with patient-clustered CIs on the flagship's margin.

``--reference`` compares the h2h / seeds / sweep blocks against an earlier
build of the same numbers (the experiment-demo dashboard's cohort.json) and
fails on any difference.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)                       # the loaders default to repo-relative paths
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from alarm_budget import (  # noqa: E402
    COMPARATOR_LABELS, COMPARATORS, assert_aggregate_only, budget_sweep_row,
    catch_at_quiet_fp, first_alarm_table, h2h_summary, lead_stats, stay_maxima,
)
from baselines_fair import bootstrap_indices  # noqa: E402
from baselines_mimic import load_splits  # noqa: E402
from fair_capability import frozen_thresholds  # noqa: E402
from fusion_common import FusionContext  # noqa: E402

CACHE = REPO / "data" / "cache"
OUT_JSON = CACHE / "alarm_budget.json"
BUDGET = 0.02
SWEEP = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
FP_TARGETS = (0.05, 0.10, 0.15, 0.20)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True,
                              cwd=REPO).stdout.strip()
    except Exception:
        return "unknown"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_scores(ctx):
    z = np.load(CACHE / "two_head.npz")
    if not ((z["stay_id_test"] == ctx.test_ordered["stay_id"].to_numpy()).all()
            and (z["hour_test"] == ctx.test_ordered["hour"].to_numpy()).all()):
        raise RuntimeError("two_head.npz does not align with the cohort test rows")
    zl = np.load(CACHE / "lit_gbm_scores.npz")
    if not ((zl["test_stay_id"] == ctx.test_ordered["stay_id"].to_numpy()).all()
            and (zl["test_hour"] == ctx.test_ordered["hour"].to_numpy()).all()):
        raise RuntimeError("lit_gbm_scores.npz does not align with the cohort test rows")
    if not ((zl["val_stay_id"] == ctx.val_ordered["stay_id"].to_numpy()).all()
            and (zl["val_hour"] == ctx.val_ordered["hour"].to_numpy()).all()):
        raise RuntimeError("lit_gbm_scores.npz does not align with the cohort val rows")
    return z, zl


def head_to_head(ctx, z, zl):
    """Seed 42 at the matched volume of the 2% NEWS2 budget."""
    fm, av = ctx.freeze_mask, ctx.alertable_val[ctx.freeze_mask]
    thr_n2, thr_m, matched = frozen_thresholds(
        ctx.news2_val[fm], z["val_s42"].astype(float)[fm], ctx.alertable_val[fm], BUDGET)
    standby = json.load(open(CACHE / "standby_report.json"))
    g = standby["thresholds"]["__global__"]
    if abs(g["thr_model"] - thr_m) > 1e-9 or abs(g["thr_news2"] - thr_n2) > 1e-9:
        raise RuntimeError("thresholds differ from the published standby report")
    flag_test = z["test_s42"].astype(float)

    # Comparators frozen on the same V_freeze half at the same matched volume,
    # so every model in the race spends the same alarm budget.
    thr_comp = {n: float(np.quantile(zl[f"val_{n}"].astype(float)[fm][av], 1.0 - matched))
                for n in COMPARATORS}
    lit_test = {n: zl[f"test_{n}"].astype(float) for n in COMPARATORS}
    table = first_alarm_table(ctx.test_ordered, {"sentinel": flag_test, **lit_test},
                              {"sentinel": thr_m, **thr_comp})
    h2h = h2h_summary(table, "sentinel", list(COMPARATORS))
    h2h.update({
        "budget": BUDGET, "matched": matched, "labels": dict(COMPARATOR_LABELS),
        "thresholds": {"sentinel": thr_m, **thr_comp},
        "alarm_rate_test": {
            "sentinel": float((flag_test >= thr_m).mean()),
            **{n: float((lit_test[n] >= thr_comp[n]).mean()) for n in COMPARATORS}},
        "caveat": ("published feature specs retrained on our cohort, "
                   "not the commercial products"),
    })
    return h2h, matched, thr_comp, lit_test


def seed_race(ctx, z, matched, thr_comp, lit_test):
    """The flagship refrozen per seed at the matched volume.

    The GBM baselines are deterministic single fits, so only the flagship's
    warning depth and the discordant race vary by seed.
    """
    fm, av = ctx.freeze_mask, ctx.alertable_val[ctx.freeze_mask]
    seed_list = sorted({int(k.split("_s")[1]) for k in z.files if k.startswith("test_s")})
    per_seed, earlier = [], {n: [] for n in COMPARATORS}
    for s_ in seed_list:
        thr_s = float(np.quantile(z[f"val_s{s_}"].astype(float)[fm][av], 1.0 - matched))
        tab_s = first_alarm_table(
            ctx.test_ordered,
            {"sentinel": z[f"test_s{s_}"].astype(float), **lit_test},
            {"sentinel": thr_s, **thr_comp})
        ls = lead_stats(tab_s, "sentinel")
        per_seed.append({"seed": s_, "n": ls["n"], "median_h": ls["median_h"],
                         "mean_h": ls["mean_h"]})
        ev_s = tab_s[tab_s["event_h"].notna()]
        c_sent = ev_s["fa_sentinel"].notna() & (ev_s["fa_sentinel"] <= ev_s["event_h"])
        for n in COMPARATORS:
            c_opp = ev_s[f"fa_{n}"].notna() & (ev_s[f"fa_{n}"] <= ev_s["event_h"])
            d = (ev_s[f"fa_{n}"] - ev_s["fa_sentinel"])[c_sent & c_opp]
            nz = d[d != 0]
            p = binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else 1.0
            earlier[n].append({
                "seed": s_, "n_differ": int(len(nz)),
                "earlier_pct": float((nz > 0).mean()) if len(nz) else 0.0,
                "sig": bool(p < 0.05)})
    meds = [r["median_h"] for r in per_seed]
    return {
        "n_seeds": len(seed_list),
        "sentinel_lead": per_seed,
        "lead_median_h": {"mean": float(np.mean(meds)), "min": float(min(meds)),
                          "max": float(max(meds))},
        "lead_mean_h": {"mean": float(np.mean([r["mean_h"] for r in per_seed]))},
        "earlier": {n: {
            "per_seed": earlier[n],
            "mean_pct": float(np.mean([r["earlier_pct"] for r in earlier[n]])),
            "min_pct": float(min(r["earlier_pct"] for r in earlier[n])),
            "max_pct": float(max(r["earlier_pct"] for r in earlier[n])),
            "sig_seeds": int(sum(r["sig"] for r in earlier[n]))}
            for n in COMPARATORS},
    }


def sweep(ctx, z, zl, matched, lit_test, maxima, is_ev):
    """The frozen-threshold rates at seven nominal budgets plus the operating
    point, cross-checked against the stay-maxima curve the equal-cost block
    interpolates on."""
    fm, av = ctx.freeze_mask, ctx.alertable_val[ctx.freeze_mask]
    budgets = sorted({*SWEEP, round(matched, 6)})
    names = ["sentinel", *COMPARATORS]
    val = {"sentinel": z["val_s42"].astype(float),
           **{n: zl[f"val_{n}"].astype(float) for n in COMPARATORS}}
    test = {"sentinel": z["test_s42"].astype(float), **lit_test}
    rows = []
    for b in budgets:
        thr_b = {n: float(np.quantile(val[n][fm][av], 1.0 - b)) for n in names}
        row = budget_sweep_row(first_alarm_table(ctx.test_ordered, test, thr_b), names)
        for n in names:
            mx = maxima[n]
            if (abs(row[n]["catch"] - (mx[is_ev] >= thr_b[n]).mean()) > 1e-12
                    or abs(row[n]["quiet_fp"] - (mx[~is_ev] >= thr_b[n]).mean()) > 1e-12):
                raise RuntimeError(f"stay maxima disagree with the sweep at {b} for {n}")
        rows.append({"budget": float(b), **row})
    return {"models": names, "operating": matched, "rows": rows}


def equal_cost(ctx, z, lit_test, maxima, ids, is_ev, reps):
    """Catch at equal quiet-stay false-alarm rate, seed 42 with paired
    patient-clustered CIs on the margin, plus the flagship's ten-seed spread."""
    targets = np.asarray(FP_TARGETS)
    point = {n: catch_at_quiet_fp(m[is_ev], m[~is_ev], targets) for n, m in maxima.items()}
    patients = np.array([ctx.stay_to_patient[i] for i in ids])
    draws = {n: [] for n in COMPARATORS}
    for idx in bootstrap_indices(patients, reps):
        e = is_ev[idx]
        c = {n: catch_at_quiet_fp(m[idx][e], m[idx][~e], targets) for n, m in maxima.items()}
        for n in COMPARATORS:
            draws[n].append(c["sentinel"] - c[n])
    margin = {}
    for n in COMPARATORS:
        d = np.asarray(draws[n])
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        margin[n] = [{"quiet_fp": float(t), "delta": float(point["sentinel"][i] - point[n][i]),
                      "ci": [float(lo[i]), float(hi[i])]}
                     for i, t in enumerate(targets)]

    seeds = sorted({int(k.split("_s")[1]) for k in z.files if k.startswith("test_s")})
    by_seed = []
    for s_ in seeds:
        _, _, m = stay_maxima(ctx.test_ordered, z[f"test_s{s_}"].astype(float))
        by_seed.append(catch_at_quiet_fp(m[is_ev], m[~is_ev], targets))
    by_seed = np.asarray(by_seed)
    return {
        "targets": [float(t) for t in targets], "reps": int(reps),
        "catch": {n: [float(v) for v in c] for n, c in point.items()},
        "margin": margin,
        "sentinel_seeds": {
            "n_seeds": len(seeds),
            "mean": [float(v) for v in by_seed.mean(0)],
            "min": [float(v) for v in by_seed.min(0)],
            "max": [float(v) for v in by_seed.max(0)],
            "above": {n: [int(v) for v in (by_seed > point[n]).sum(0)]
                      for n in COMPARATORS}},
    }


def diff_tree(a, b, path=""):
    """Every leaf where two JSON trees differ (exact, floats included)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}.{k}: missing" for k in a.keys() ^ b.keys()]
        for k in a.keys() & b.keys():
            out += diff_tree(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff_tree(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--reference", default=None,
                    help="an earlier build's JSON carrying an 'h2h' block to match")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    ctx = FusionContext(load_splits())
    z, zl = load_scores(ctx)

    h2h, matched, thr_comp, lit_test = head_to_head(ctx, z, zl)
    seeds = seed_race(ctx, z, matched, thr_comp, lit_test)

    test42 = {"sentinel": z["test_s42"].astype(float), **lit_test}
    maxima = {}
    for n, s in test42.items():
        ids, is_ev, maxima[n] = stay_maxima(ctx.test_ordered, s)
    sw = sweep(ctx, z, zl, matched, lit_test, maxima, is_ev)
    eq = equal_cost(ctx, z, lit_test, maxima, ids, is_ev, args.reps)

    out = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "inputs": {p: sha256(CACHE / p) for p in
                   ("two_head.npz", "lit_gbm_scores.npz", "standby_report.json")},
        "h2h": h2h, "seeds": seeds, "sweep": sw, "equal_cost": eq,
    }
    assert_aggregate_only(out)

    f_any = h2h["funnel"]["any"]
    print("[h2h] catch " + " / ".join(f"{m} {f_any[m]['pct']:.1%}" for m in h2h["models"])
          + "; race " + "; ".join(
              f"vs {r['opponent']}: {r['flagship_first']}-{r['tie']}-{r['opponent_first']}"
              f" (only {r['only_flagship']}:{r['only_opponent']})" for r in h2h["race"])
          + "; quiet FP " + " / ".join(
              f"{h2h['cost']['quiet_fp_rate'][m]:.1%}" for m in h2h["models"]))
    print(f"[seeds] flagship lead median {seeds['lead_median_h']['mean']:.2f} h; earlier "
          + "; ".join(f"{n} {seeds['earlier'][n]['mean_pct']:.1%} "
                      f"(sig {seeds['earlier'][n]['sig_seeds']}/{seeds['n_seeds']})"
                      for n in COMPARATORS))
    print("[sweep] catch / quiet FP at " + "; ".join(
        f"{r['budget']:.1%}: " + " ".join(
            f"{r[n]['catch']:.1%}/{r[n]['quiet_fp']:.1%}" for n in sw["models"])
        for r in sw["rows"]))
    for n in COMPARATORS:
        print(f"[equal cost] vs {n}: " + "; ".join(
            f"FP {m['quiet_fp']:.0%} {m['delta']:+.3f} [{m['ci'][0]:+.3f}, {m['ci'][1]:+.3f}]"
            for m in eq["margin"][n]))

    if args.reference:
        ref = json.load(open(args.reference))["h2h"]
        mine = json.loads(json.dumps({**h2h, "seeds": seeds, "sweep": sw}))
        diffs = diff_tree(mine, ref)
        if diffs:
            raise SystemExit("reference mismatch:\n" + "\n".join(diffs[:20]))
        print(f"[reference] h2h, seeds and sweep match {args.reference} exactly")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
