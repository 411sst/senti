"""Which ICU is this cohort, and does one model hold across it? (T10, part 2)

Answers the two clinician points recorded in docs/references.md §5:

    "ICU: Cardio, neuro, respiratory, urology, burns."
        -> Part A: what our 56,157 stays actually contain, under three views,
           with an explicit not-represented verdict for the categories MIMIC-IV
           has no unit for.

    "Keep models on standby because ICU cannot be generalized."
        -> Part B: per-stratum AUPRC/AUROC with patient-clustered CIs, and the
           paired first-alarm race vs NEWS2 at a fixed alarm budget.
        -> Part C: is the spread across strata larger than the noise within one?

Two protocol rules make Part B mean something:

  * Thresholds are frozen ONCE, globally, on V_freeze -- never refit per
    stratum. Refitting would turn this into per-stratum tuning and guarantee a
    flattering answer. Each stratum's realised alarm rate is reported instead,
    as the diagnostic it is.
  * Under-powered strata are printed and flagged, never dropped. Silently
    omitting the small units is how "the model generalises" becomes unfalsifiable.

    python scripts/run_icu_strata.py [--parts a,b,c] [--reps 2000]
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

OUT_JSON = REPO / "data" / "cache" / "icu_strata.json"
VIEWS = ("unit_group", "service_group", "clinician_category", "dx1_chapter")
BUDGET = 0.02


def stay_frame(splits):
    """One row per stay: patient, split, hours, event type, hourly prevalence."""
    df = pd.concat(splits.values())
    g = df.groupby("stay_id", sort=False)
    out = g.agg(patient_id=("patient_id", "first"),
                split=("split", "first"),
                event_type=("event_type", "first"),
                n_hours=("hour", "size"),
                n_positive=("label_24h", "sum")).reset_index()
    out["deteriorated"] = out["event_type"].notna()
    return out


# --- Part A ------------------------------------------------------------------

def composition(strata, view, subset=None):
    frame = strata if subset is None else strata[subset]
    g = frame.groupby(view, dropna=False)
    tab = g.agg(stays=("stay_id", "size"),
                patients=("patient_id", "nunique"),
                hours=("n_hours", "sum"),
                positive_hours=("n_positive", "sum"),
                deteriorated=("deteriorated", "sum")).reset_index()
    tab["pct_stays"] = (100 * tab.stays / tab.stays.sum()).round(2)
    tab["pct_deteriorated"] = (100 * tab.deteriorated / tab.stays).round(1)
    tab["hourly_prevalence_pct"] = (100 * tab.positive_hours / tab.hours).round(2)
    return tab.sort_values("stays", ascending=False).reset_index(drop=True)


def run_part_a(strata):
    from icu_strata import category_coverage

    print("\n" + "=" * 78)
    print("PART A -- what kind of ICU is this cohort?")
    print("=" * 78)
    out = {}
    for view in VIEWS:
        tab = composition(strata, view)
        print(f"\n--- by {view} ({len(tab)} strata) ---")
        print(tab.head(20).to_string(index=False))
        out[view] = tab.to_dict("records")
        total = tab.pct_stays.sum()
        if not np.isclose(total, 100.0, atol=0.05):
            raise ValueError(f"{view}: composition sums to {total:.2f}%, not 100%")

    cov = category_coverage(strata)
    print("\n--- the five categories the clinician named ---")
    print(cov.to_string(index=False))
    out["category_coverage"] = cov.to_dict("records")

    missing = cov[~cov.has_dedicated_unit].category.tolist()
    print(f"\n  no dedicated ICU exists in MIMIC-IV for: {', '.join(missing)}")
    for _, r in cov.iterrows():
        if r.n_assigned == 0:
            print(f"  {r.category}: NOT REPRESENTED in this cohort")
        elif r.pct_assigned < 1.0:
            print(f"  {r.category}: {r.n_assigned:,} stays ({r.pct_assigned}%) "
                  f"-- too few to model or evaluate separately")
    return out


# --- Part B ------------------------------------------------------------------

def load_models(ctx):
    """name -> (val_score or None, test_score). None means test-only."""
    from fusion_common import load_fair_scores

    models = {"NEWS2": (ctx.news2_val, ctx.news2_test)}
    fv, ft = load_fair_scores(ctx.val_ordered, ctx.test_ordered)
    models["FAIR+TRAJ"] = (fv, ft)

    z = np.load(REPO / "data/cache/fair_gbm_scores.npz")
    if (z["stay_id"] == ctx.test_ordered["stay_id"].to_numpy()).all():
        models["GBM13"] = (None, z["raw_GBM13"].astype(float))   # no val cache

    th = np.load(REPO / "data/cache/two_head.npz")
    if (th["stay_id_test"] == ctx.test_ordered["stay_id"].to_numpy()).all():
        models["two-head_s42"] = (th["val_s42"].astype(float),
                                  th["test_s42"].astype(float))

    lit_path = REPO / "data/cache/lit_gbm_scores.npz"
    if lit_path.exists():
        from lit_gbm import load_lit_scores
        for name, (v, t) in load_lit_scores(ctx.val_ordered, ctx.test_ordered,
                                            path=lit_path).items():
            models[f"lit:{name}"] = (v, t)
    else:
        print("  [note] lit_gbm_scores.npz absent -- run src/lit_gbm.py first "
              "to include the literature specs")
    return models


def run_part_b(ctx, strata, models, reps, min_stays, min_pos):
    from fair_capability import frozen_thresholds, paired_first_alarm_diffs, summarize_paired
    from icu_strata import stratum_metrics

    print("\n" + "=" * 78)
    print("PART B -- does one model hold across ICU types?")
    print("=" * 78)

    test_stays = ctx.test_ordered["stay_id"].to_numpy()
    lookup = strata.set_index("stay_id")
    y = ctx.y_test
    patients = ctx.patient_test
    scores_test = {k: v[1] for k, v in models.items()}

    # Thresholds frozen ONCE on V_freeze, globally -- see the module docstring.
    nf = ctx.news2_val[ctx.freeze_mask]
    af = ctx.alertable_val[ctx.freeze_mask]
    thresholds = {}
    for name, (s_val, _) in models.items():
        if s_val is None:
            continue
        thr_n2, thr_m, _ = frozen_thresholds(nf, s_val[ctx.freeze_mask], af, BUDGET)
        thresholds[name] = (thr_n2, thr_m)

    out = {}
    for view in ("unit_group", "clinician_category"):
        row_stratum = lookup.loc[test_stays, view].to_numpy()
        print(f"\n--- by {view} (budget {BUDGET:.0%}, thresholds frozen globally) ---")
        rows = []
        for name in sorted(set(row_stratum)):
            mask = row_stratum == name
            n_stays = int(pd.unique(test_stays[mask]).size)
            n_pos_stays = int(lookup.loc[pd.unique(test_stays[mask]),
                                         "deteriorated"].sum())
            rec = {"stratum": name, "test_stays": n_stays,
                   "deteriorated_stays": n_pos_stays,
                   "hours": int(mask.sum()),
                   "hourly_prevalence": float(y[mask].mean())}
            underpowered = n_stays < min_stays or n_pos_stays < min_pos
            rec["underpowered"] = bool(underpowered)

            boots = stratum_metrics(y, scores_test, patients, mask,
                                    reference="NEWS2", reps=reps,
                                    min_pos=min_pos)
            rec["models"] = {}
            for m, s in scores_test.items():
                entry = {"auprc": float(_safe(y[mask], s[mask], "auprc")),
                         "auroc": float(_safe(y[mask], s[mask], "auroc"))}
                if boots is not None:
                    entry["auprc_ci"] = boots[m]["auprc_ci"]
                    entry["auroc_ci"] = boots[m]["auroc_ci"]
                if m in thresholds:
                    entry["alarm_rate"] = float(
                        (s[mask & ctx.alertable_test] >= thresholds[m][1]).mean())
                rec["models"][m] = entry

            # Paired first-alarm race, restricted to this stratum's stays.
            sub = ctx.test_ordered[mask]
            for m, (s_val, s_test) in models.items():
                if m == "NEWS2" or s_val is None:
                    continue
                thr_n2, thr_m = thresholds[m]
                diffs = paired_first_alarm_diffs(sub, ctx.news2_test[mask],
                                                 s_test[mask], thr_n2, thr_m)
                if len(diffs) >= 20:
                    r = summarize_paired(diffs, ctx.stay_to_patient, reps=min(reps, 500))
                    rec["models"][m]["lead"] = {
                        "median_h": r["median"], "n_pairs": int(len(diffs)),
                        "win_rate": r.get("win_rate")}
                else:
                    rec["models"][m]["lead"] = {"n_pairs": int(len(diffs)),
                                                "note": "too few jointly-caught stays"}
            rows.append(rec)
        _print_part_b(rows, scores_test)
        out[view] = rows
    return out


def _safe(y, s, metric):
    from metrics import auprc, auroc
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return auprc(y, s) if metric == "auprc" else auroc(y, s)


def _print_part_b(rows, scores_test):
    names = list(scores_test)
    head = f"{'stratum':<20}{'stays':>7}{'det':>6}{'prev':>7}  " + \
           "".join(f"{n[:14]:>16}" for n in names)
    print(head)
    print("-" * len(head))
    for r in sorted(rows, key=lambda x: -x["test_stays"]):
        flag = " *" if r["underpowered"] else "  "
        line = (f"{r['stratum'][:18]:<20}{r['test_stays']:>7}"
                f"{r['deteriorated_stays']:>6}{r['hourly_prevalence']:>6.1%}{flag}")
        for n in names:
            m = r["models"][n]
            line += f"{m['auprc']:>8.3f}/{m['auroc']:.3f}"
        print(line)
    print("  * under-powered: too few test stays or too few deteriorations for a "
          "bootstrap CI to mean anything (numbers shown, uncertainty not)")
    print("  cells are AUPRC/AUROC")


# --- Part C ------------------------------------------------------------------

def run_part_c(part_b, model="two-head_s42"):
    """Spread across strata vs the noise within one -- on three metrics.

    AUPRC alone cannot answer this. It is bounded below by prevalence, and our
    strata differ four-fold in event rate (neuro 1.4%, cardiac 6.1%), so an
    AUPRC ranking across units is partly just a prevalence ranking. Reporting
    the spread on AUPRC alone would manufacture heterogeneity that is really
    arithmetic. So three views are printed together:

        AUPRC        the headline metric, prevalence-bound
        AUROC        prevalence-free discrimination
        lift         AUPRC / prevalence -- what the ranking is worth relative
                     to the base rate a random alarm would achieve here

    If the three disagree about which unit is worst, that disagreement IS the
    finding, and it is reported rather than resolved by choosing a metric.
    """
    print("\n" + "=" * 78)
    print("PART C -- is the between-stratum spread bigger than the within-stratum noise?")
    print("=" * 78)
    out = {}
    for view, rows in part_b.items():
        usable = [r for r in rows if not r["underpowered"]
                  and model in r["models"] and "auprc_ci" in r["models"][model]]
        if len(usable) < 2:
            print(f"  {view}: fewer than two adequately powered strata")
            continue

        for r in usable:
            m = r["models"][model]
            r["_lift"] = m["auprc"] / max(1e-9, r["hourly_prevalence"])

        def spread_of(key):
            v = np.array([r["models"][model][key] if key in ("auprc", "auroc")
                          else r["_lift"] for r in usable])
            return float(v.max() - v.min())

        auprc_ci = np.array([r["models"][model]["auprc_ci"][1]
                             - r["models"][model]["auprc_ci"][0] for r in usable])
        auroc_ci = np.array([r["models"][model]["auroc_ci"][1]
                             - r["models"][model]["auroc_ci"][0] for r in usable])

        print(f"\n  {view} ({model}, {len(usable)} adequately powered strata)")
        print(f"    {'stratum':<20}{'prev':>7}{'AUPRC':>18}{'AUROC':>18}{'lift':>8}")
        for r in sorted(usable, key=lambda x: -x["models"][model]["auroc"]):
            m = r["models"][model]
            plo, phi = m["auprc_ci"]
            rlo, rhi = m["auroc_ci"]
            print(f"    {r['stratum'][:18]:<20}{r['hourly_prevalence']:>6.1%}"
                  f"{m['auprc']:>8.3f} [{plo:.2f},{phi:.2f}]"
                  f"{m['auroc']:>8.3f} [{rlo:.2f},{rhi:.2f}]"
                  f"{r['_lift']:>8.1f}")

        s_auprc, s_auroc, s_lift = (spread_of("auprc"), spread_of("auroc"),
                                    spread_of("lift"))
        w_auprc, w_auroc = float(np.median(auprc_ci)), float(np.median(auroc_ci))
        het_auprc, het_auroc = s_auprc > w_auprc, s_auroc > w_auroc
        print(f"    AUPRC spread {s_auprc:.3f} vs median CI width {w_auprc:.3f}"
              f"  -> {'exceeds noise' if het_auprc else 'within noise'}")
        print(f"    AUROC spread {s_auroc:.3f} vs median CI width {w_auroc:.3f}"
              f"  -> {'exceeds noise' if het_auroc else 'within noise'}")
        print(f"    lift spread  {s_lift:.1f}x")

        worst_auprc = min(usable, key=lambda r: r["models"][model]["auprc"])["stratum"]
        worst_auroc = min(usable, key=lambda r: r["models"][model]["auroc"])["stratum"]
        agree = worst_auprc == worst_auroc
        if het_auprc and het_auroc and agree:
            verdict = (f"heterogeneous, and both metrics agree the weakest unit "
                       f"is '{worst_auprc}' -- a per-unit model is justified")
        elif het_auprc or het_auroc:
            verdict = (f"heterogeneous, but the metrics DISAGREE on which unit is "
                       f"weakest (AUPRC says '{worst_auprc}', AUROC says "
                       f"'{worst_auroc}') -- the AUPRC ordering is largely the "
                       f"prevalence ordering, so 'which ICU is hardest' has no "
                       f"metric-free answer")
        else:
            verdict = "within noise on both metrics -- no per-unit model indicated"
        print(f"    => {verdict}")
        out[view] = {"model": model, "auprc_spread": s_auprc,
                     "auroc_spread": s_auroc, "lift_spread": s_lift,
                     "median_auprc_ci_width": w_auprc,
                     "median_auroc_ci_width": w_auroc,
                     "worst_by_auprc": worst_auprc, "worst_by_auroc": worst_auroc,
                     "verdict": verdict}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="a,b,c")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--min-stays", type=int, default=200)
    ap.add_argument("--min-pos", type=int, default=25)
    ap.add_argument("--from-json", action="store_true",
                    help="recompute Part C from the saved Part B (no refit)")
    args = ap.parse_args()

    if args.from_json:
        saved = json.loads(OUT_JSON.read_text())
        saved["part_c"] = run_part_c(saved["part_b"])
        OUT_JSON.write_text(json.dumps(saved, indent=2, default=float))
        print(f"\n[saved] {OUT_JSON}")
        return
    parts = {p.strip() for p in args.parts.split(",")}

    from baselines_mimic import load_splits
    from fusion_common import FusionContext
    from icu_strata import assign_strata

    splits = load_splits()
    stays = stay_frame(splits)
    strata = assign_strata(stays)
    print(f"[cohort] {len(strata):,} stays | "
          f"{strata.patient_id.nunique():,} patients | "
          f"{strata.n_hours.sum():,} hours")

    result = {}
    if "a" in parts:
        result["part_a"] = run_part_a(strata)
    if "b" in parts or "c" in parts:
        ctx = FusionContext(splits)
        models = load_models(ctx)
        result["part_b"] = run_part_b(ctx, strata, models, args.reps,
                                      args.min_stays, args.min_pos)
        if "c" in parts:
            result["part_c"] = run_part_c(result["part_b"])

    OUT_JSON.write_text(json.dumps(result, indent=2, default=float))
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
