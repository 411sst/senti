"""Does a documented limitation of care precede the events we predict?

Acts on the caveat T7 could not: death is treatment-mediated, so a death-only
label is a lower bound on physiological signal. MIMIC-IV charts code status
directly, so this reads the field rather than inferring it from notes.

Three questions, in order:

  A. Coverage -- who has a charted code status at all? This is what decides
     whether running a language model over notes is worth it: the field is only
     worth supplementing where it is silent.
  B. Timing -- among stays that reached an event, how often was a limitation
     documented BEFORE it, and how long before?
  C. Exposure -- how much of the flagship's measured lead sits on stays where
     the outcome had already been decided by a person?

    python scripts/run_code_status.py [--json out.json]
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

OUT_JSON = REPO / "data" / "cache" / "code_status_report.json"


def cohort_stays():
    df = pd.read_parquet(REPO / "data/cache/mimic_cohort_hourly.parquet",
                         columns=["stay_id", "patient_id", "split",
                                  "event_h", "event_type"])
    g = df.groupby("stay_id", sort=False).agg(
        patient_id=("patient_id", "first"), split=("split", "first"),
        event_h=("event_h", "first"), event_type=("event_type", "first"),
        n_hours=("event_h", "size")).reset_index()
    return g


BUDGET = 0.02


def lead_by_decision(att, budget=BUDGET, reps=2000):
    """Split the flagship's paired first-alarm race on death stays by whether a
    limitation of care was already documented.

    The question T7 could not answer: when the model alarms early on a stay that
    ends in death, is it reading physiology, or is it reading a ward where a
    decision has already been made? Thresholds are the standard globally frozen
    ones -- only the SUMMARY is split, exactly as decompose.component_race does,
    so the composite row still reproduces the published harness.

    n is small by construction (death-first stays in the test split, charted,
    jointly caught), so every cell is printed with its count and nothing is
    concluded from a cell that cannot support it.
    """
    from baselines_mimic import load_splits
    from fair_capability import frozen_thresholds, paired_first_alarm_diffs, summarize_paired
    from fusion_common import FusionContext

    ctx = FusionContext(load_splits())
    th = np.load(REPO / "data/cache/two_head.npz")
    if not (th["stay_id_test"] == ctx.test_ordered["stay_id"].to_numpy()).all():
        raise ValueError("two_head cache does not align with the cohort")
    s_val, s_test = th["val_s42"].astype(float), th["test_s42"].astype(float)

    thr_n2, thr_m, _ = frozen_thresholds(ctx.news2_val[ctx.freeze_mask],
                                         s_val[ctx.freeze_mask],
                                         ctx.alertable_val[ctx.freeze_mask], budget)
    diffs = paired_first_alarm_diffs(ctx.test_ordered, ctx.news2_test, s_test,
                                     thr_n2, thr_m)

    idx = att.set_index("stay_id")
    d = diffs.copy()
    d["event_type"] = d["stay_id"].map(idx["event_type"])
    d["charted"] = d["stay_id"].map(idx["has_code_status"]).fillna(False)
    d["decided"] = d["stay_id"].map(idx["limit_before_event"]).fillna(False)

    deaths = d[d["event_type"] == "death"]
    cells = {
        "all death stays": deaths,
        "decision documented before death": deaths[deaths["decided"]],
        "charted, no limitation": deaths[deaths["charted"] & ~deaths["decided"]],
        "no charted status": deaths[~deaths["charted"]],
        "EXCLUDING decided stays": deaths[~deaths["decided"]],
    }
    print(f"  budget {budget:.0%}, thresholds frozen globally · "
          f"{len(deaths)} jointly-caught death stays in the test split\n")
    print(f"  {'subgroup':<36}{'n':>5}{'median lead':>13}{'win':>7}")
    print("  " + "-" * 61)
    out = {}
    for name, cell in cells.items():
        if len(cell) == 0:
            print(f"  {name:<36}{0:>5}{'--':>13}{'--':>7}")
            out[name] = {"n_pair": 0}
            continue
        r = summarize_paired(cell, ctx.stay_to_patient, reps=min(reps, 1000))
        flag = "  *" if len(cell) < 20 else ""
        print(f"  {name:<36}{r['n_pair']:>5}{r['median']:>12.1f}h"
              f"{r['win_rate']:>7.0%}{flag}")
        out[name] = {"n_pair": r["n_pair"], "median": r["median"],
                     "win_rate": r["win_rate"], "median_ci": r.get("median_ci")}
    print("\n  * fewer than 20 pairs -- reported for completeness, not evidence")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(OUT_JSON))
    args = ap.parse_args()

    from code_status import attach_to_stays, coverage_report, load_code_status, stay_timeline

    stays = cohort_stays()
    records = load_code_status()
    timeline = stay_timeline(records)
    att = attach_to_stays(stays, timeline)
    att["event_type"] = att["event_type"].fillna("event-free")

    print("=" * 78)
    print("A -- CHARTED CODE STATUS: COVERAGE")
    print("=" * 78)
    print(f"  cohort {len(att):,} stays")
    print(f"    code status charted      {int(att.has_code_status.sum()):>7,}  "
          f"({att.has_code_status.mean():.1%})")
    print(f"    palliative care plan     {int(att.has_palliative_plan.sum()):>7,}  "
          f"({att.has_palliative_plan.mean():.1%})")
    print(f"    either                   {int(att.has_any_goc.sum()):>7,}  "
          f"({att.has_any_goc.mean():.1%})")
    only_pall = int((att.has_palliative_plan & ~att.has_code_status).sum())
    print(f"    palliative plan but NO charted code status: {only_pall:,} "
          f"(what the weaker signal adds)")
    neg = int((records.hour < 0).sum())
    print(f"  {len(records):,} charted entries | {neg:,} ({neg/len(records):.1%}) "
          f"timestamped before ICU intime (kept: an earlier decision is still a decision)")
    cov = coverage_report(att)
    print()
    print(cov.to_string(index=False))

    print("\n" + "=" * 78)
    print("B -- WAS THE OUTCOME ALREADY DECIDED?")
    print("=" * 78)
    rows = []
    for et in ("death", "vent", "vaso"):
        g = att[att.event_type == et]
        if not len(g):
            continue
        n = len(g)
        cov_n = int(g.has_code_status.sum())
        lim = int(g.limit_before_event.sum())
        cmo = int(g.cmo_before_event.sum())
        dni = int(g.intub_limit_before_event.sum())
        med = g.loc[g.limit_before_event, "hours_limit_to_event"].median()
        rows.append({"first event": et, "stays": n, "charted": cov_n,
                     "limit before event": lim,
                     "pct of charted": round(100 * lim / max(1, cov_n), 1),
                     "CMO before": cmo, "DNI before": dni,
                     "median h limit->event": round(float(med), 1) if lim else np.nan})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))

    d = att[att.event_type == "death"]
    dc = d[d.has_code_status]
    print(f"\n  Among death-first stays WITH a charted status ({len(dc):,}):")
    print(f"    {int(dc.limit_before_event.sum()):,} "
          f"({dc.limit_before_event.mean():.1%}) had a limitation documented before death")
    print(f"    {int(dc.cmo_before_event.sum()):,} "
          f"({dc.cmo_before_event.mean():.1%}) were on comfort measures only before death")
    if dc.limit_before_event.any():
        h = dc.loc[dc.limit_before_event, "hours_limit_to_event"]
        print(f"    lead from decision to death: median {h.median():.0f}h "
              f"(IQR {h.quantile(.25):.0f}-{h.quantile(.75):.0f}h)")

    v = att[(att.event_type == "vent") & att.has_code_status]
    print(f"\n  Ventilation arm -- DNI forecloses the event we score:")
    print(f"    {int(v.intub_limit_before_event.sum()):,} of {len(v):,} charted vent-first "
          f"stays ({v.intub_limit_before_event.mean():.1%}) had DNI documented beforehand")

    print("\n" + "=" * 78)
    print("C -- IS A LANGUAGE MODEL WORTH RUNNING OVER NOTES?")
    print("=" * 78)
    gap = att[~att.has_any_goc]
    gap_death = gap[gap.event_type == "death"]
    print(f"  silent for {len(gap):,} stays ({1-att.has_code_status.mean():.1%} of cohort), "
          f"of which {len(gap_death):,} are death-first")
    dr = att.groupby("event_type").has_any_goc.mean()
    print("  coverage by first event: "
          + " · ".join(f"{k} {v:.0%}" for k, v in dr.items()))
    spread = dr.max() - dr.min()
    print(f"  spread across event types: {spread:.1%} — "
          + ("coverage is NOT uniform; the silent stays are a biased sample"
             if spread > 0.05 else
             "coverage is close to uniform, so the silence is roughly missing-at-random"))

    # ---- D: does the measured lead ride on already-decided stays? ----------
    print("\n" + "=" * 78)
    print("D -- DOES THE FLAGSHIP'S DEATH-STAY LEAD SURVIVE THE EXCLUSION?")
    print("=" * 78)
    try:
        part_d = lead_by_decision(att)
    except Exception as e:
        part_d = {"error": f"{type(e).__name__}: {e}"}
        print(f"  skipped: {part_d['error']}")

    payload = {"coverage": cov.to_dict("records"), "timing": tab.to_dict("records"),
               "n_stays": len(att), "charted": int(att.has_code_status.sum()),
               "gap_stays": len(gap), "gap_death_stays": len(gap_death),
               "lead_by_decision": part_d}
    Path(args.json).write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
