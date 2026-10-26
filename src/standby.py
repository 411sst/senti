"""Standby operating points: the per-unit layer T10 says the evidence supports.

A reviewing clinician's instinct — "keep models on standby because ICU cannot
be generalized" — decomposed, in T10, into three measured facts:

  * discrimination varies across units only modestly (AUROC spread 0.078,
    barely past noise; within noise on the category view),
  * but **alarm burden varies five-fold** at one global threshold
    (med-surg 4.03% of alertable hours vs neuro 0.75%),
  * and the flagship's lead is unit-dependent (+2h medical/surgical/trauma,
    0h med-surg/neuro).

So what deserves to be per-unit is the **operating point**, not the model.
This module is that layer: one flagship, one score, and a small table of
per-unit thresholds frozen on V_freeze *within* each unit — six numbers on
standby, selected by `first_careunit`, a database column known at ICU
admission. Nothing here retrains, gates, or reweights the agents; the
dynamic, per-patient form of "standby" already exists inside the
orchestrator, whose attention reweights agents hour by hour as modalities
appear and go stale.

**The anti-optimism rule carries over.** Per-unit thresholds are still frozen
on V_freeze only, still NEWS2-anchored within the unit (both sides of the race
re-anchor, so the race stays volume-matched *within* unit), and test is still
scored once. A unit whose freeze half is too thin to support a quantile falls
back to the global threshold and says so — a threshold fit on a handful of
rows is noise wearing a decimal point.
"""

import numpy as np
import pandas as pd

from config import SEED
from fair_capability import (
    frozen_thresholds,
    paired_first_alarm_diffs,
    summarize_paired,
)

# An alertable V_freeze row count below this cannot pin a 2% quantile with any
# stability; the unit keeps the global operating point instead of a noisy one.
MIN_FREEZE_ROWS = 5000

# The flagship's agent bank, in emission-cache naming.
FLAGSHIP_AGENTS = ("vitals_tcn", "news2", "labs", "ecg_meas",
                   "gbm", "cadence", "fluid", "resp")


def rows_stratum(ordered, strata, view="unit_group"):
    """Map a split's (stay_id, hour) rows to their stay's stratum label."""
    lookup = strata.set_index("stay_id")[view]
    out = ordered["stay_id"].map(lookup)
    if out.isna().any():
        n = int(out.isna().sum())
        raise ValueError(f"{n} rows belong to stays with no stratum label")
    return out.to_numpy()


def standby_thresholds(ctx, stratum_val, s_val, budget,
                       min_rows=MIN_FREEZE_ROWS):
    """Per-unit NEWS2-anchored thresholds, frozen on each unit's V_freeze rows.

    Returns ``{unit: {"thr_news2", "thr_model", "matched", "source"}}`` where
    ``source`` is "unit" or "global-fallback". The global pair is computed the
    same way on all freeze rows and is both the fallback and the comparator.
    """
    nf_all = ctx.news2_val[ctx.freeze_mask]
    af_all = ctx.alertable_val[ctx.freeze_mask]
    sf_all = s_val[ctx.freeze_mask]
    strat_f = stratum_val[ctx.freeze_mask]

    # Two anchors per unit, because two different questions are being asked.
    #
    #   thr_model_budget   the model's own (1 - budget) quantile within the
    #                      unit — the DEPLOYMENT threshold. This is what
    #                      equalizes burden: every unit runs at the budget.
    #   thr_news2/thr_model  the NEWS2-anchored pair within the unit — the
    #                      RACE thresholds. NEWS2's integer score overshoots
    #                      fine budgets by different amounts per unit (badly
    #                      in low-acuity neuro), so a fair volume-matched race
    #                      must re-anchor per unit; but that same overshoot is
    #                      why the anchored pair cannot BE the deployment
    #                      threshold.
    g_n2, g_m, g_matched = frozen_thresholds(nf_all, sf_all, af_all, budget)
    g_budget = float(np.quantile(sf_all[af_all], 1.0 - budget))
    out = {"__global__": {"thr_news2": g_n2, "thr_model": g_m,
                          "matched": g_matched, "thr_model_budget": g_budget,
                          "source": "global"}}
    for unit in sorted(pd.unique(strat_f)):
        sel = strat_f == unit
        n_alertable = int((af_all & sel).sum())
        if n_alertable < min_rows:
            out[unit] = {**out["__global__"], "source": "global-fallback",
                         "n_freeze_alertable": n_alertable}
            continue
        u_n2, u_m, u_matched = frozen_thresholds(
            nf_all[sel], sf_all[sel], af_all[sel], budget)
        u_budget = float(np.quantile(sf_all[af_all & sel], 1.0 - budget))
        out[unit] = {"thr_news2": u_n2, "thr_model": u_m,
                     "matched": u_matched, "thr_model_budget": u_budget,
                     "source": "unit", "n_freeze_alertable": n_alertable}
    return out


def evaluate_standby(ctx, stratum_test, s_test, thresholds, reps=1000):
    """Per-unit alarm rate and paired race, global vs standby operating points.

    Both threshold sets are applied to the SAME test scores — the model never
    changes, only where the line is drawn — so any difference between the two
    columns is purely the operating point.
    """
    g = thresholds["__global__"]
    rows = {}
    # Iterate over the units actually PRESENT IN TEST: a unit that never
    # appeared in the val freeze half (possible for tiny strata under a
    # patient-level split) has no fitted entry and runs on the global point.
    for unit in sorted(pd.unique(stratum_test)):
        thr = thresholds.get(unit, {**g, "source": "global-fallback"})
        sel = stratum_test == unit
        alert = sel & ctx.alertable_test
        n_alert = int(alert.sum())
        rec = {
            "hours": int(sel.sum()),
            "prevalence": float(ctx.y_test[sel].mean()),
            "source": thr["source"],
            "alarm_global": float((s_test[alert] >= g["thr_model"]).mean()),
            "alarm_standby": float(
                (s_test[alert] >= thr["thr_model_budget"]).mean()),
            "matched_volume": thr["matched"],
        }
        sub = ctx.test_ordered[sel]
        for name, tn2, tm in (("race_global", g["thr_news2"], g["thr_model"]),
                              ("race_standby", thr["thr_news2"], thr["thr_model"])):
            diffs = paired_first_alarm_diffs(sub, ctx.news2_test[sel],
                                             s_test[sel], tn2, tm)
            if len(diffs) >= 20:
                r = summarize_paired(diffs, ctx.stay_to_patient, reps=reps)
                rec[name] = {"n_pair": r["n_pair"], "median": r["median"],
                             "win_rate": r["win_rate"],
                             "median_ci": r.get("median_ci")}
            else:
                rec[name] = {"n_pair": int(len(diffs))}
        rows[unit] = rec
    return rows


def burden_spread(rows, key):
    """max/min realised alarm rate across units — the equalization headline."""
    vals = [r[key] for r in rows.values() if r[key] > 0]
    return float(max(vals) / min(vals)) if len(vals) >= 2 else float("nan")


def threshold_stability(per_seed_thresholds, key="thr_model_budget"):
    """Cross-seed spread of the per-unit deployment thresholds.

    ``per_seed_thresholds`` is ``{seed: standby_thresholds(...) result}``. The
    T12 claim rests on thresholds frozen from ONE seed's scores; this measures
    how much those numbers move when the fusion net is retrained. Per unit:
    min/max/mean of ``key`` across seeds and the coefficient of variation —
    plus how often the unit kept its own threshold vs falling back to global,
    since a unit that flips between "unit" and "global-fallback" across seeds
    is itself a form of instability.
    """
    units = sorted({u for thr in per_seed_thresholds.values() for u in thr})
    out = {}
    for unit in units:
        vals, sources = [], []
        for thr in per_seed_thresholds.values():
            if unit in thr:
                vals.append(float(thr[unit][key]))
                sources.append(thr[unit]["source"])
        mean = float(np.mean(vals))
        out[unit] = {
            "n_seeds": len(vals),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": mean,
            "cv": float(np.std(vals) / mean) if mean else float("nan"),
            "source_counts": {s: sources.count(s) for s in set(sources)},
        }
    return out


def agent_roster(ctx, stratum_test, risks, min_pos=25):
    """Per-unit standalone AUPRC (and lift) for each agent's calibrated risk.

    The static evidence behind a per-unit standby roster: which agents earn
    their seat where. ``risks`` is {agent: test-risk array on ctx's rows}.
    Lift (AUPRC / prevalence) is reported beside AUPRC because units differ
    four-fold in event rate and raw AUPRC would rank prevalence, not agents —
    the same trap Part C of T10 documents.
    """
    from metrics import auprc

    out = {}
    for unit in sorted(pd.unique(stratum_test)):
        sel = stratum_test == unit
        y = ctx.y_test[sel]
        if y.sum() < min_pos:
            out[unit] = {"underpowered": True, "n_pos": int(y.sum())}
            continue
        prev = float(y.mean())
        cell = {}
        for name, r in risks.items():
            a = float(auprc(y, r[sel]))
            cell[name] = {"auprc": a, "lift": a / prev}
        out[unit] = {"underpowered": False, "prevalence": prev,
                     "hours": int(sel.sum()), "agents": cell}
    return out
