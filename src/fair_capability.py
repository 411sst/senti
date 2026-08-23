"""Timing and calibration: what survives the fair-baseline result?

The equal-information GBM (FAIR+NOTES, ``baselines_fair.py``) beats the
orchestrator on discrimination. Two capabilities could still separate them, and
each gets its own analysis here:

1. **Timing.** The paired, alert-volume-matched first-alarm race against NEWS2
   that produced the orchestrator's "+2h earlier on 66% of stays" headline --
   identical protocol (h6 warm-up mask, matching to NEWS2's achieved alert
   volume, jointly-caught pairing, sign test) with one audit-mandated upgrade:
   thresholds are selected on VAL, frozen, and applied to test. If the
   discrimination-winning GBM also beats NEWS2 on timing, the orchestrator has
   no timing story; if it cannot, timing is a real orchestrator capability.

2. **Calibration.** Clinical-standard measures -- equal-mass-binned ECE,
   Cox calibration slope/intercept, and ICI -- instead of the fragile
   equal-width ECE, each with patient-clustered bootstrap CIs.

Volume matching is threshold-based, so any monotone rescaling of a model's
scores (isotonic calibration included) selects the same alarms; using the
orchestrator's calibrated cache and the GBM's raw scores is therefore fair.

    python src/fair_capability.py [--reps 2000] [--cal-reps 500]
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from baselines_fair import (
    SCORE_CACHE,
    bootstrap_indices,
    build_design,
    fit_note_pca,
    tune_gbm,
    variant_columns,
)
from baselines_mimic import alertable_mask, load_splits
from config import SEED
from metrics import calibration_slope_intercept, ece_adaptive, ici
from modality_features import load_labs

LABEL = "label_24h"
FAIR = "FAIR+NOTES"
ORCH_NPZ = {"val": "data/cache/embeddings/orchestrator_val.npz",
            "test": "data/cache/embeddings/orchestrator_test.npz"}
BUDGETS = (0.01, 0.02, 0.05)


# --- Frozen operating points -------------------------------------------------

def frozen_thresholds(news2_val, model_val, alertable_val, budget):
    """Operating points chosen on VAL only, to be applied unchanged to test.

    NEWS2's integer score overshoots fine budgets, so (as in the published
    protocol) the model is matched to NEWS2's ACHIEVED alert volume rather than
    the nominal budget -- otherwise NEWS2 would spend ~2x the alerts and win
    earliness for free. Both quantiles come from val's alertable hours.
    """
    n2 = news2_val[alertable_val]
    thr_news2 = float(np.quantile(n2, 1.0 - budget))
    matched = float((n2 >= thr_news2).mean())
    thr_model = float(np.quantile(model_val[alertable_val], 1.0 - matched))
    return thr_news2, thr_model, matched


# --- Paired first-alarm, per-stay ---------------------------------------------

def paired_first_alarm_diffs(test, s_news2, s_model, thr_news2, thr_model):
    """Per-stay paired first-alarm differences (baselines_mimic protocol).

    Same rules as :func:`baselines_mimic.paired_first_alarm` -- warm-up hour
    masked, only stays BOTH models catch at or before the event -- but returns
    the per-stay ``diff = first_alarm_NEWS2 - first_alarm_model`` (positive =
    model earlier) so the caller can run clustered inference over stays.
    """
    tmp = test[["stay_id", "hour", "event_h"]].copy()
    tmp["n2"] = np.asarray(s_news2)
    tmp["m"] = np.asarray(s_model)
    tmp = tmp[alertable_mask(tmp)]

    stays, diffs = [], []
    for stay, g in tmp.groupby("stay_id", sort=False):
        event = g["event_h"].iloc[0]
        if pd.isna(event):
            continue
        g = g.sort_values("hour")
        hours = g["hour"].to_numpy(dtype=float)
        cross_n2 = hours[g["n2"].to_numpy() >= thr_news2]
        cross_m = hours[g["m"].to_numpy() >= thr_model]
        if cross_n2.size == 0 or cross_m.size == 0:
            continue
        if cross_n2[0] > event or cross_m[0] > event:
            continue
        stays.append(stay)
        diffs.append(cross_n2[0] - cross_m[0])
    return pd.DataFrame({"stay_id": stays, "diff": diffs})


def summarize_paired(diffs, stay_to_patient, reps=2000, seed=SEED):
    """Aggregates + patient-clustered bootstrap CIs for one paired comparison.

    The sign test is kept for continuity with the published tables; the
    clustered CIs are the statistically honest companion (multiple stays per
    patient are not independent pairs).
    """
    d = diffs["diff"].to_numpy(dtype=float)
    n = len(d)
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    discordant = wins + losses
    out = {
        "n_pair": n,
        "median": float(np.median(d)) if n else float("nan"),
        "win_rate": wins / n if n else float("nan"),
        "ties": n - discordant,
        "sign_p": float(binomtest(wins, discordant, 0.5).pvalue) if discordant else 1.0,
    }
    if n == 0:
        out["median_ci"] = out["win_rate_ci"] = (float("nan"), float("nan"))
        return out

    patients = diffs["stay_id"].map(stay_to_patient).to_numpy()
    medians, rates = [], []
    for idx in bootstrap_indices(patients, reps, seed):
        db = d[idx]
        medians.append(np.median(db))
        rates.append((db > 0).mean())
    out["median_ci"] = (float(np.percentile(medians, 2.5)),
                        float(np.percentile(medians, 97.5)))
    out["win_rate_ci"] = (float(np.percentile(rates, 2.5)),
                          float(np.percentile(rates, 97.5)))
    return out


# --- Calibration with clustered CIs ---------------------------------------------

def calibration_with_cis(y, scores_by_model, patient_ids, reps=500, seed=SEED):
    """Point estimates + patient-clustered percentile CIs for the four
    clinical-standard calibration numbers, per model."""
    def all_four(yb, pb):
        slope, intercept = calibration_slope_intercept(yb, pb)
        return {"ece_em": ece_adaptive(yb, pb), "slope": slope,
                "intercept": intercept, "ici": ici(yb, pb)}

    out = {name: {"point": all_four(y, p)} for name, p in scores_by_model.items()}
    draws = {name: {k: [] for k in out[name]["point"]} for name in scores_by_model}
    for idx in bootstrap_indices(patient_ids, reps, seed):
        yb = y[idx]
        if yb.sum() in (0, len(yb)):
            continue
        for name, p in scores_by_model.items():
            for k, v in all_four(yb, p[idx]).items():
                draws[name][k].append(v)
    for name in out:
        out[name]["ci"] = {
            k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in draws[name].items()
        }
    return out


# --- Score plumbing ---------------------------------------------------------------

def load_orch(split_frame, split_name, paths=ORCH_NPZ):
    """Orchestrator scores for a split, alignment-checked.

    ``paths`` swaps in a non-canonical emission cache (e.g. a per-seed
    core-3+meas run) without touching the canonical constant.
    """
    z = np.load(paths[split_name])
    ordered = split_frame.sort_values(["stay_id", "hour"])
    if not ((z["stay_id"] == ordered["stay_id"].to_numpy()).all()
            and (z["hour"] == ordered["hour"].to_numpy()).all()):
        raise ValueError(f"{paths[split_name]} does not align with the "
                         f"{split_name} split")
    return z["risk"].astype(float)


def fair_scores(splits, cache_path=SCORE_CACHE, variant=FAIR):
    """Fair-GBM scores for one variant: test from the cache, val cached or refit.

    Newer ``baselines_fair.py`` runs cache val scores directly (``rawval_*``);
    older caches only held test scores, in which case the val scores are
    produced by refitting the tuned configuration through the same code path
    and seed, verified against the cached test scores, and written back into
    the cache so the refit never runs twice.
    """
    z = dict(np.load(cache_path, allow_pickle=True))
    meta = json.loads(str(z["meta_json"]))
    if f"rawval_{variant}" in z:
        return z[f"rawval_{variant}"].astype(float), \
            z[f"raw_{variant}"].astype(float), \
            z[f"cal_{variant}"].astype(float)

    train, val = splits["train"], splits["val"]
    print(f"[fair] rebuilding design matrices to emit {variant} val scores...")
    note_pack = (fit_note_pca(train["stay_id"].unique(), seed=SEED)
                 if variant == "FAIR+NOTES" else None)
    labs = load_labs()
    X_tr, y_tr, _ = build_design(train, labs, note_pack)
    X_va, y_va, _ = build_design(val, labs, note_pack)
    X_te, _, _ = build_design(splits["test"], labs, note_pack)

    cols = variant_columns()[variant]
    rec = meta["records"][variant]
    print(f"[fair] refitting recorded winner {rec} ...")
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=3000, learning_rate=0.05, max_depth=rec["max_depth"],
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=rec["scale_pos_weight"], random_state=SEED, n_jobs=-1,
        tree_method="hist", eval_metric="aucpr", early_stopping_rounds=50,
    )
    model.fit(X_tr[cols], y_tr, eval_set=[(X_va[cols], y_va)], verbose=False)

    raw_te = model.predict_proba(X_te[cols])[:, 1].astype(float)
    drift = float(np.abs(raw_te - z[f"raw_{variant}"]).max())
    if drift > 1e-6:
        print(f"[fair] WARNING: refit test scores differ from cache "
              f"(max |diff| {drift:.2e}); keeping the cached canonical scores")
    raw_va = model.predict_proba(X_va[cols])[:, 1].astype(float)

    z[f"rawval_{variant}"] = raw_va
    np.savez_compressed(cache_path, **z)
    print(f"[fair] val scores cached back into {cache_path}")
    return raw_va, z[f"raw_{variant}"].astype(float), \
        z[f"cal_{variant}"].astype(float)


# --- Report ---------------------------------------------------------------------

def run(reps=2000, cal_reps=500, fair=FAIR):
    splits = load_splits()
    val, test = splits["val"], splits["test"]
    val_ordered = val.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    test_ordered = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)

    news2_val = val_ordered["news2"].to_numpy(dtype=float)
    news2_test = test_ordered["news2"].to_numpy(dtype=float)
    alertable_val = alertable_mask(val_ordered)
    alertable_test = alertable_mask(test_ordered)

    orch_val, orch_test = load_orch(val, "val"), load_orch(test, "test")
    fair_val, fair_test_raw, fair_test_cal = fair_scores(splits, variant=fair)

    stay_to_patient = dict(zip(test_ordered["stay_id"], test_ordered["patient_id"]))
    y_test = test_ordered[LABEL].to_numpy(dtype=float)

    # ---- 1. paired lead time vs NEWS2, val-frozen thresholds -----------------
    print(f"\nPAIRED FIRST-ALARM vs NEWS2 -- thresholds selected on val, frozen, "
          f"applied to test\n(every hour alertable since the h6 root-cause fix; "
          f"jointly-caught stays; diff > 0 = model fires earlier; "
          f"{reps}-rep patient-clustered CIs)\n")
    hdr = (f"  {'budget':>7} {'model':<11}{'n2%':>6}{'mdl%':>6}{'n_pair':>7}"
           f"{'median':>7}  {'95% CI':<14}{'win':>5}  {'95% CI':<14}{'ties':>5}"
           f"{'sign_p':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for budget in BUDGETS:
        for name, s_val, s_test in (("ORCH", orch_val, orch_test),
                                    (fair, fair_val, fair_test_raw)):
            thr_n2, thr_m, _ = frozen_thresholds(news2_val, s_val,
                                                 alertable_val, budget)
            diffs = paired_first_alarm_diffs(test_ordered, news2_test, s_test,
                                             thr_n2, thr_m)
            r = summarize_paired(diffs, stay_to_patient, reps=reps)
            n2_rate = float((news2_test[alertable_test] >= thr_n2).mean())
            m_rate = float((s_test[alertable_test] >= thr_m).mean())
            print(f"  {f'{budget*100:.0f}%':>7} {name:<11}"
                  f"{100*n2_rate:>5.1f}%{100*m_rate:>5.1f}%{r['n_pair']:>7}"
                  f"{r['median']:>7.1f}  "
                  f"[{r['median_ci'][0]:>4.1f}, {r['median_ci'][1]:>4.1f}]"
                  f"{100*r['win_rate']:>4.0f}%  "
                  f"[{100*r['win_rate_ci'][0]:>3.0f}%, {100*r['win_rate_ci'][1]:>3.0f}%]"
                  f"{r['ties']:>5}{r['sign_p']:>10.2g}")
    print("  n2% / mdl% = achieved TEST alert volumes at the val-frozen thresholds.")

    # ---- 2. calibration, clinical-standard metrics ---------------------------
    print(f"\nCALIBRATION on test (isotonic-on-val scores; {cal_reps}-rep "
          f"patient-clustered CIs)\ntargets: ECE-em 0, slope 1, intercept 0, ICI 0\n")
    cal = calibration_with_cis(
        y_test, {"ORCH": orch_test, fair: fair_test_cal},
        test_ordered["patient_id"].to_numpy(), reps=cal_reps)
    hdr = (f"  {'model':<11}" + "".join(f"{m:>10}  {'95% CI':<18}"
           for m in ("ECE-em", "slope", "intercept", "ICI")))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, r in cal.items():
        cells = []
        for key in ("ece_em", "slope", "intercept", "ici"):
            lo, hi = r["ci"][key]
            cells.append(f"{r['point'][key]:>10.4f}  [{lo:>7.4f}, {hi:>7.4f}]")
        print(f"  {name:<11}" + "".join(cells))
    return cal


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000,
                    help="bootstrap resamples for the lead-time CIs")
    ap.add_argument("--cal-reps", type=int, default=500,
                    help="bootstrap resamples for the calibration CIs")
    ap.add_argument("--fair", default=FAIR,
                    choices=("FAIR", "FAIR+TRAJ", "FAIR+NOTES"),
                    help="which fair-GBM variant to race (FAIR+TRAJ on clones "
                         "without the note caches)")
    args = ap.parse_args()
    run(reps=args.reps, cal_reps=args.cal_reps, fair=args.fair)
