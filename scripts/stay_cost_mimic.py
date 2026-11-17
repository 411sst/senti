"""Exploratory stay-level cost of the MIMIC catch lead (for claim 4).

T35 found that on all eICU hours the flagship's extra catches came from
alarming on many more quiet stays (stays with no event). This runs the same
check on the MIMIC test set. It is added after T28/T30/T32/T33 and is not part
of any pre-registered pass rule.

For each model and budget, at protocol thresholds (own (1 - b) quantile on
V_freeze):
* catch: share of event stays alarmed at or before the event;
* quiet alarmed: share of quiet stays with at least one alarm;
* stay precision.

Then, as a diagnostic, the best catch the flagship reaches while alarming on
no more quiet stays than each comparator does at its protocol threshold. This
is read off the flagship's stay-level curve, built from per-stay maximum
scores, so it uses test scores (as T35's diagnostic does). It also reports the
stay-level AUROC.

Models:
* flagship two-head ALL-8, 10 seeds (`two_head.npz`);
* eCART-style, YAIB and Sadanandan (`lit_gbm_scores.npz`);
* the T30 all-features GBM and its no-cadence arm, seed 42
  (`allfeat_gbm_scores.npz`).

Output is aggregate only: data/cache/stay_cost_mimic.json.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/stay_cost_mimic.py
"""

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
from sklearn.metrics import roc_auc_score  # noqa: E402

from paper_operating_points import (  # noqa: E402
    SWEEP_BUDGETS,
    early_stats,
    freeze_quantile,
    load_scores,
    stay_first_alarms,
    test_rate,
)
from t35_eicu_operating_points import (  # noqa: E402
    catch_at_quiet_share,
    quiet_alarm_share,
    stay_max_scores,
)

CACHE = REPO / "data" / "cache"
OUT = CACHE / "stay_cost_mimic.json"


def load_allfeat(ctx):
    z = np.load(CACHE / "allfeat_gbm_scores.npz", allow_pickle=False)
    for split, frame in (("val", ctx.val_ordered), ("test", ctx.test_ordered)):
        if not (z[f"{split}_stay_id"] == frame["stay_id"].to_numpy()).all():
            raise RuntimeError(f"allfeat_gbm_scores.npz is not aligned with {split}")
    return {arm: {"val": z[f"val_{arm}_s42"].astype(float), "test": z[f"test_{arm}_s42"].astype(float)}
            for arm in ("allfeat", "allfeat_nocad")}


def stay_auroc(event_max, quiet_max):
    lo = min(np.min(event_max[np.isfinite(event_max)]), np.min(quiet_max)) - 1.0
    x = np.concatenate([np.where(np.isfinite(event_max), event_max, lo), quiet_max])
    return float(roc_auc_score(np.r_[np.ones(len(event_max)), np.zeros(len(quiet_max))], x))


def main():
    from baselines_mimic import load_splits
    from demo_bundle import assert_aggregate_only
    from fusion_common import FusionContext

    t0 = time.time()
    ctx = FusionContext(load_splits())
    if not ctx.alertable_test.all():
        raise SystemExit("expected every MIMIC test hour to be alertable")
    seeds, flagship, comps = load_scores(ctx)
    comps.update(load_allfeat(ctx))
    test = ctx.test_ordered
    models = {**{f"flagship_s{s}": f for s, f in flagship.items()}, **comps}
    smax = {m: stay_max_scores(test, v["test"]) for m, v in models.items()}
    auc = {m: stay_auroc(*smax[m]) for m in models}
    fl_auc = [auc[f"flagship_s{s}"] for s in seeds]
    rows = []
    for b in SWEEP_BUDGETS:
        row = {"budget": b}
        for m, v in models.items():
            thr = freeze_quantile(ctx, v["val"], b)
            st = early_stats(stay_first_alarms(test, v["test"], thr))
            q, n_quiet = quiet_alarm_share(test, v["test"], thr)
            caught, quiet = st["catch"] * st["n_event_stays"], q * n_quiet
            row[m] = {"catch": st["catch"], "quiet_alarmed": q,
                      "stay_precision": float(caught / (caught + quiet)),
                      "test_alarm_rate": test_rate(ctx, v["test"], thr)}
        fl = [row.pop(f"flagship_s{s}") for s in seeds]
        row["flagship_mean"] = {k: float(np.mean([x[k] for x in fl])) for k in fl[0]}
        row["matched_quiet"] = {}
        for n in comps:
            qn = row[n]["quiet_alarmed"]
            per = [catch_at_quiet_share(*smax[f"flagship_s{s}"], qn) for s in seeds]
            own = catch_at_quiet_share(*smax[n], qn)
            row["matched_quiet"][n] = {"quiet_alarmed": qn, "comparator_envelope_catch": own,
                                       "flagship_catch_mean": float(np.mean(per)),
                                       "seeds_ahead": int(sum(x > own for x in per))}
        rows.append(row)
        f = row["flagship_mean"]
        print(f"[cost] {b:>5.1%} flagship catch {f['catch']:.1%} quiet {f['quiet_alarmed']:.1%} "
              f"prec {f['stay_precision']:.1%} | " + " | ".join(
                  f"{n} {row[n]['catch']:.1%} q{row[n]['quiet_alarmed']:.1%} p{row[n]['stay_precision']:.1%}"
                  for n in comps))
        print(f"[matched] {b:>5.1%} " + " | ".join(
            f"{n} q{x['quiet_alarmed']:.1%}: {x['comparator_envelope_catch']:.1%} vs F "
            f"{x['flagship_catch_mean']:.1%} ({x['seeds_ahead']}/10)" for n, x in row["matched_quiet"].items()))
    print(f"[stay AUROC] flagship {np.mean(fl_auc):.4f} ({min(fl_auc):.4f}-{max(fl_auc):.4f}) "
          + ", ".join(f"{n} {auc[n]:.4f}" for n in comps))
    n_e, n_q = (len(x) for x in smax[next(iter(comps))])
    payload = {"test": "stay-level cost on MIMIC (exploratory)", "seeds": seeds,
               "n_event_stays": n_e, "n_quiet_stays": n_q, "rows": rows,
               "stay_auroc": {"flagship_mean": float(np.mean(fl_auc)), "flagship_min": float(min(fl_auc)),
                              "flagship_max": float(max(fl_auc)), **{n: auc[n] for n in comps}},
               "note": "aggregate-only; protocol V_freeze thresholds; matched-quiet catch is a "
                       "test-score diagnostic; added after the pre-registered runs"}
    payload = json.loads(json.dumps(payload, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    assert_aggregate_only(payload)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"[saved] {OUT} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
