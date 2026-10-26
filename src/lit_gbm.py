"""Literature-anchored GBM comparators: the published bar, on our cohort.

The flagship's claim is that a strong ranker wins on ranking and still warns no
earlier. That argument only lands if the ranker is credible, and until now the
comparator has been OUR XGBoost (``baselines_fair.py``, FAIR+TRAJ) -- which a
reviewer can dismiss as a self-built strawman. This module reproduces three
gradient-boosted models whose SPECIFICATION comes from a paper, on our cohort,
under our protocol, so the comparison is to published practice rather than to a
model we chose the shape of.

    sadanandan   arXiv:2603.14719 -- the only published model found whose label
                 is OUR label: ICU death or vasopressor onset or ventilation
                 onset within 24h, scored hourly on MIMIC-IV. Reports
                 XGBoost AUROC 0.7486 / AUPRC 0.1123 on 823,641 test hours.
    yaib         arXiv:2306.05109 -- the community MIMIC-IV benchmark recipe:
                 forward-fill, missing indicators, mean-fill, then min/max/mean/
                 count per channel, into LightGBM.
    ecart        Churpek 2025 (Crit Care Explor) -- the FDA-cleared GBM early
                 warning score: 24h min/max and trends, delivered FiO2, time of
                 day, gases expiring at 24h, training down-sampled to 50%
                 prevalence.

See docs/references.md for the full citations and the numbers each reports.

**What is reproduced and what is not.** Every spec carries a ``deviations``
list naming each place our data forced a departure, and those lists are printed
with the results and saved into the score cache. A reproduction whose gaps are
invisible is worse than no reproduction, because it invites the reader to treat
a handicapped model as the published one.

**Protocol is ours, not theirs.** Fit on train; tune and isotonic-calibrate on
V_fit; freeze alarm thresholds on V_freeze; touch test once. Mixing our protocol
with their reported numbers inside one table is exactly the error this module
exists to avoid, so the two live in separate tables in results_fusion.md.

    python src/lit_gbm.py [--specs sadanandan,yaib,ecart] [--device cuda]
                          [--reps 2000] [--quick]
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from baselines_mimic import load_splits
from config import SEED
from metrics import auprc, auroc
from mimic_cohort import VITALS
from modality_features import (
    EXTRA_VITALS_COLUMNS,
    LAB_COLUMNS,
    build_extra_vitals_features,
    build_lab_features,
    carry_forward_to_cohort,
    load_extra_vitals,
    load_labs,
)

LABEL = "label_24h"
SCORE_CACHE = "data/cache/lit_gbm_scores.npz"
RAW_PATH = "data/cache/mimic_raw_hourly.parquet"
STAY_META_PATH = "data/cache/mimic_stay_meta.parquet"

# Oxygen channels the cohort parquet drops (mimic_cohort.py's `keep` list keeps
# only the derived `on_oxygen` flag). eCARTv5 names delivered FiO2 as its second
# most important predictor, so it is rebuilt here from the same raw extract the
# cohort itself was built from.
OXY_COLUMNS = ["fio2", "o2_flow"]
OXY_MAX_STALENESS = {"fio2": 24, "o2_flow": 8}      # mimic_cohort.MAX_STALENESS_HOURS

# Sadanandan's 16 labs, in their order. All 16 are available once the
# supplementary blood-gas fetch has run.
SADANANDAN_LABS = [
    "lab_lactate", "lab_creatinine", "lab_bun", "lab_potassium", "lab_sodium",
    "lab_glucose", "lab_wbc", "lab_hemoglobin", "lab_hematocrit",
    "lab_platelets", "lab_bilirubin", "lab_albumin", "lab_ph", "pco2", "po2",
    "lab_bicarbonate",
]
# Their 10 vital columns are arterial and non-invasive SBP/DBP counted
# separately; MIMIC's derived.vitalsign coalesces those pairs, so the same
# physiology arrives as 7 channels.
SADANANDAN_VITALS = VITALS + ["dbp", "mbp"]
SADANANDAN_VITAL_STALENESS = 4        # "forward-fill, max 4h gap for vitals"

ECART_TREND_WINDOW = 24               # "maximum RR in the prior 24 hours"
ECART_TARGET_PREVALENCE = 0.5         # "down-sampling ... to obtain 50% outcome prevalence"


# --- Input assembly ----------------------------------------------------------

def load_inputs():
    """Everything the specs draw on, loaded once."""
    splits = load_splits()
    labs = load_labs()
    extra = load_extra_vitals()
    raw = pd.read_parquet(RAW_PATH, columns=["stay_id", "hour"] + OXY_COLUMNS)
    raw["stay_id"] = raw["stay_id"].astype("int64")
    raw["hour"] = raw["hour"].astype("int64")
    meta = pd.read_parquet(
        STAY_META_PATH,
        columns=["stay_id", "gender", "anchor_age", "admit_hour_of_day"])
    meta["stay_id"] = meta["stay_id"].astype("int64")
    return splits, labs, extra, raw, meta


def build_channels(split_df, labs, extra, raw, meta):
    """All candidate channels for one split, on its sorted (stay_id, hour) rows.

    Returns ``(ordered, channels)`` where ``channels`` is a DataFrame of raw
    per-hour values -- every one of them strictly before t, either because the
    cohort builder already shifted it or because
    ``carry_forward_to_cohort`` did. Specs select columns from here and derive
    their own summaries; nothing spec-specific happens at this stage.
    """
    ordered = split_df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    keys = ordered[["stay_id", "hour"]]
    ch = pd.DataFrame(index=ordered.index)

    for col in VITALS:
        ch[col] = ordered[col].to_numpy(np.float32)
        ch[f"{col}_hrs_since"] = ordered[f"{col}_hrs_since"].to_numpy(np.float32)
    for col in ("gcs", "gcs_unable", "on_oxygen"):
        ch[col] = ordered[col].to_numpy(np.float32)

    def attach(frame, columns, what):
        if not (frame["stay_id"].to_numpy() == ordered["stay_id"].to_numpy()).all():
            raise ValueError(f"{what} features misaligned with cohort rows")
        for c in columns:
            ch[c] = frame[c].to_numpy(np.float32)
            ch[f"{c}_hrs_since"] = frame[f"{c}_hrs_since"].to_numpy(np.float32)

    attach(build_lab_features(keys, labs), LAB_COLUMNS, "lab")
    attach(build_extra_vitals_features(keys, extra), EXTRA_VITALS_COLUMNS, "extra-vitals")
    attach(carry_forward_to_cohort(keys, raw, OXY_COLUMNS, lookback=0,
                                   max_staleness=OXY_MAX_STALENESS),
           OXY_COLUMNS, "oxygen")

    stat = keys.merge(meta, on="stay_id", how="left", validate="many_to_one")
    if len(stat) != len(ordered):
        raise ValueError("stay metadata join changed row count")
    ch["age"] = stat["anchor_age"].to_numpy(np.float32)
    ch["sex_male"] = (stat["gender"] == "M").to_numpy(np.float32)
    # eCARTv5 uses time of day. `hour` counts from ICU admission, so wall-clock
    # time only exists once the admission hour is added back.
    ch["time_of_day"] = ((stat["admit_hour_of_day"].to_numpy(np.float32)
                          + ordered["hour"].to_numpy(np.float32)) % 24)
    return ordered, ch


# --- Shared feature derivations ----------------------------------------------

def _mask_stale(values, hrs_since, bound):
    """Drop a carried value once it is older than ``bound`` hours."""
    return np.where(np.asarray(hrs_since) <= bound, values, np.nan).astype(np.float32)


def trailing_stats(ordered, channels, columns, window, stats=("min", "max")):
    """Trailing-window aggregates within a stay, ending at the current row.

    The inputs are already strictly-before-t, so a trailing window over them is
    too. Grouping by stay stops one stay's history reaching the next.
    """
    out = {}
    stay = ordered["stay_id"]
    for col in columns:
        v = pd.Series(channels[col].to_numpy(), index=ordered.index)
        roll = v.groupby(stay, sort=False).rolling(window, min_periods=1)
        for s in stats:
            agg = getattr(roll, s)().droplevel(0).sort_index()
            out[f"{col}_{s}_{window}h"] = agg.to_numpy(np.float32)
    return out


def trailing_deltas(ordered, channels, columns, window):
    """Change over ``window`` hours -- eCART's 'trend' block."""
    out = {}
    stay = ordered["stay_id"]
    for col in columns:
        v = pd.Series(channels[col].to_numpy(), index=ordered.index)
        out[f"{col}_delta_{window}h"] = (
            v - v.groupby(stay, sort=False).shift(window)).to_numpy(np.float32)
    return out


def expanding_stats(ordered, channels, columns):
    """min / max / mean / count of everything seen so far in the stay.

    YAIB's recipe for classical models ("the min, max, mean, and count of
    measurements for each feature"), computed over the whole history up to t.
    Written with cummin/cummax/cumsum rather than ``rolling`` because an
    expanding ``groupby.rolling`` over 2.4M rows x 37 channels does not finish.
    """
    out = {}
    stay = ordered["stay_id"]
    for col in columns:
        v = pd.Series(channels[col].to_numpy(), index=ordered.index)
        g = v.groupby(stay, sort=False)
        out[f"{col}_emin"] = g.cummin().to_numpy(np.float32)
        out[f"{col}_emax"] = g.cummax().to_numpy(np.float32)
        seen = v.notna()
        count = seen.groupby(stay, sort=False).cumsum()
        total = v.fillna(0).groupby(stay, sort=False).cumsum()
        out[f"{col}_ecount"] = count.to_numpy(np.float32)
        out[f"{col}_emean"] = np.where(count > 0, total / count.replace(0, np.nan),
                                       np.nan).astype(np.float32)
    return out


# --- The three specifications ------------------------------------------------

def spec_sadanandan(ordered, channels):
    """arXiv:2603.14719 -- 26 structured channels + demographics, current values.

    The paper states a 48h look-back and an input dimension of 26 for the
    sequence model, but never says how that window was reduced for its XGBoost
    baseline. The literal reading is taken here (the current carried value of
    each channel), and the ambiguity is recorded as a deviation rather than
    resolved in whichever direction flatters us.
    """
    X = {}
    for col in SADANANDAN_VITALS:
        X[col] = _mask_stale(channels[col], channels[f"{col}_hrs_since"],
                             SADANANDAN_VITAL_STALENESS)
    for col in SADANANDAN_LABS:
        X[col] = channels[col].to_numpy(np.float32)
    X["age"] = channels["age"].to_numpy(np.float32)
    X["sex_male"] = channels["sex_male"].to_numpy(np.float32)
    return pd.DataFrame(X), {
        "paper": "Sadanandan 2026 (arXiv:2603.14719)",
        "reported": {"auroc": 0.7486, "auprc": 0.1123,
                     "cohort": "74,822 MIMIC-IV stays, 2.8% prevalence, "
                               "stays >=24h, hours 6-48"},
        "library": "xgboost",
        "deviations": [
            "their 10 vital COLUMNS are arterial and non-invasive SBP/DBP counted "
            "separately; MIMIC's derived.vitalsign coalesces each pair, so the "
            "same physiology arrives as 7 channels",
            "the 48h look-back is not specified for their tree baseline; current "
            "carried values are used",
            "lab staleness uses our 24h bound; theirs is 'variable per lab type', "
            "unspecified",
            "no radiology-note channel (that is their multimodal arm, not the "
            "XGBoost baseline)",
        ],
    }


def spec_yaib(ordered, channels):
    """arXiv:2306.05109 -- forward-fill, missing indicators, min/max/mean/count.

    Applied to the dynamic channels we hold rather than to YAIB's own 48, which
    are defined against its harmonised concept set and are not reconstructible
    here. What is reproduced is the recipe and the model, not the variable list.
    """
    dynamic = (VITALS + EXTRA_VITALS_COLUMNS + OXY_COLUMNS + LAB_COLUMNS
               + ["gcs"])
    X = {}
    for col in dynamic:
        v = channels[col].to_numpy(np.float32)
        X[col] = v
        X[f"{col}_missing"] = np.isnan(v).astype(np.float32)   # missing indicator
    X.update(expanding_stats(ordered, channels, dynamic))
    X["age"] = channels["age"].to_numpy(np.float32)
    X["sex_male"] = channels["sex_male"].to_numpy(np.float32)
    return pd.DataFrame(X), {
        "paper": "van de Water et al. (arXiv:2306.05109), YAIB",
        "reported": {"note": "MIMIC-IV LGBM: mortality 87.7/44.2, AKI 83.8/53.3, "
                             "sepsis 77.5/5.9 (AUROC/AUPRC x100) -- none of these "
                             "labels is ours; anchors the method, not the task"},
        "library": "lightgbm",
        "deviations": [
            f"{len(dynamic)} dynamic channels from our cohort, not YAIB's 48 "
            "harmonised concepts",
            "static block is age and sex; height and weight are not in our extract",
            "mean-fill of never-measured values is left to the model instead: "
            "LightGBM handles NaN natively and mean-filling would erase the "
            "missing-indicator's meaning",
        ],
    }


def spec_ecart(ordered, channels):
    """Churpek 2025 -- 24h min/max and trends, FiO2, time of day.

    The published model is a discrete-time survival GBM over 8h blocks trained
    on a 50%-prevalence down-sample; the down-sampling is reproduced (see
    :func:`downsample_to_prevalence`), the 8h discretisation is not -- our grid
    is hourly and re-blocking it would change the prediction task rather than
    the model.
    """
    summarised = VITALS + ["dbp", "mbp", "fio2", "o2_flow", "gcs"]
    X = {}
    for col in summarised + LAB_COLUMNS + ["pco2", "po2"]:
        X[col] = channels[col].to_numpy(np.float32)
    X.update(trailing_stats(ordered, channels, summarised, ECART_TREND_WINDOW,
                            stats=("min", "max")))
    X.update(trailing_deltas(ordered, channels, summarised + LAB_COLUMNS,
                             ECART_TREND_WINDOW))
    for col in ("on_oxygen", "gcs_unable", "age", "sex_male", "time_of_day"):
        X[col] = channels[col].to_numpy(np.float32)
    return pd.DataFrame(X), {
        "paper": "Churpek et al. 2025, Crit Care Explor (eCARTv5)",
        "reported": {"auroc": 0.834, "vs_news": 0.766,
                     "lead_h": 16, "news_lead_h": 17,
                     "cohort": "1,769,461 ward admissions, 4.8% event rate, "
                               "outcome = ward death or ICU transfer <=24h"},
        "library": "xgboost",
        "downsample_prevalence": ECART_TARGET_PREVALENCE,
        "deviations": [
            "97-variable list is in supplemental Table E2 and is not publicly "
            "retrievable; the main text's named categories are reproduced",
            "no Braden scale, BMI, height or weight in our extract",
            "hourly grid retained; the published model discretises to 8h blocks "
            "with discrete-time survival",
            "ward outcome (death or ICU transfer) vs our in-ICU composite -- the "
            "model form is reproduced, the task is ours",
        ],
    }


SPECS = {
    "sadanandan": spec_sadanandan,
    "yaib": spec_yaib,
    "ecart": spec_ecart,
}


# --- Fitting -----------------------------------------------------------------

def downsample_to_prevalence(y, target, seed=SEED):
    """Row index keeping every positive and enough negatives to hit ``target``.

    eCARTv5 down-samples the training data to 50% outcome prevalence before
    fitting. Applied to TRAIN ONLY -- val and test keep their real prevalence,
    or every calibration and AUPRC number downstream would be meaningless.
    """
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_neg = int(round(len(pos) * (1 - target) / target))
    if n_neg >= len(neg):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]))


def fit_xgb(X_tr, y_tr, X_va, y_va, seed=SEED, device=None, quick=False,
            spec_params=None):
    """Val-tuned XGBoost, same modest grid discipline as ``tune_gbm``."""
    import xgboost as xgb

    pos_weight = float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum()))
    depths = (6,) if quick else (4, 6, 8)
    extra = {"device": device} if device else {}
    params = dict(n_estimators=3000, learning_rate=0.05, min_child_weight=10,
                  subsample=0.8, colsample_bytree=0.8, random_state=seed,
                  n_jobs=-1, tree_method="hist", eval_metric="aucpr",
                  early_stopping_rounds=50)
    params.update(spec_params or {})
    best_model, best = None, None
    # dict.fromkeys dedupes: after eCART's 50% down-sample pos_weight IS 1.0,
    # and fitting the identical model twice only burns GPU time.
    for depth in depths:
        for spw in dict.fromkeys((1.0, round(pos_weight, 6))):
            model = xgb.XGBClassifier(max_depth=depth, scale_pos_weight=spw,
                                      **params, **extra)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            score = auprc(y_va, model.predict_proba(X_va)[:, 1])
            print(f"      depth={depth} spw={spw:>6.1f}: val AUPRC {score:.4f} "
                  f"({model.best_iteration + 1} trees)")
            if best is None or score > best["val_auprc"]:
                best_model = model
                best = {"max_depth": depth, "scale_pos_weight": round(spw, 2),
                        "n_trees": int(model.best_iteration + 1),
                        "val_auprc": float(score)}
    return best_model, best


def fit_lgbm(X_tr, y_tr, X_va, y_va, seed=SEED, quick=False, **_):
    """LightGBM over YAIB's published search space.

    YAIB tunes with 50 Bayesian iterations; a grid over the same space is used
    here so the run is reproducible and finishes -- the space is theirs, the
    search is ours, and that is recorded as a deviation.
    """
    import lightgbm as lgb

    depths = (6,) if quick else (3, 5, 7)
    leaves = (32,) if quick else (32, 128)
    best_model, best = None, None
    for depth in depths:
        for num_leaves in leaves:
            model = lgb.LGBMClassifier(
                n_estimators=3000, learning_rate=0.05, max_depth=depth,
                num_leaves=num_leaves, min_child_samples=1000,
                colsample_bytree=0.66, subsample=0.66, subsample_freq=1,
                random_state=seed, n_jobs=-1, verbose=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            score = auprc(y_va, model.predict_proba(X_va)[:, 1])
            print(f"      depth={depth} leaves={num_leaves}: val AUPRC {score:.4f} "
                  f"({model.best_iteration_} trees)")
            if best is None or score > best["val_auprc"]:
                best_model = model
                best = {"max_depth": depth, "num_leaves": num_leaves,
                        "n_trees": int(model.best_iteration_),
                        "val_auprc": float(score)}
    return best_model, best


FITTERS = {"xgboost": fit_xgb, "lightgbm": fit_lgbm}


# --- Driver ------------------------------------------------------------------

def build_spec_design(name, ordered, channels):
    """Design frame + provenance for one spec on one split's rows."""
    X, prov = SPECS[name](ordered, channels)
    return X.astype(np.float32), prov


def run(specs=("sadanandan", "yaib", "ecart"), device=None, quick=False,
        out_cache=SCORE_CACHE, seed=SEED):
    """Fit each spec, score val and test, cache RAW scores.

    Raw, not calibrated: ``FusionContext`` fits the isotonic map on V_fit, so a
    score calibrated here would be calibrated on the wrong half. Cached as
    float64 -- a float32 score cache silently collapsed rank ties in an earlier
    experiment and cost a day of debugging.
    """
    t0 = time.time()
    splits, labs, extra, raw, meta = load_inputs()
    built = {}
    for split in ("train", "val", "test"):
        ordered, channels = build_channels(splits[split], labs, extra, raw, meta)
        built[split] = (ordered, channels)
        print(f"[channels] {split}: {len(ordered):,} rows x {channels.shape[1]} channels")

    payload, records = {}, {}
    for name in specs:
        print(f"\n=== {name} ===")
        designs, ys = {}, {}
        for split in ("train", "val", "test"):
            ordered, channels = built[split]
            X, prov = build_spec_design(name, ordered, channels)
            designs[split] = X
            ys[split] = ordered[LABEL].to_numpy(dtype=float)
        print(f"  {designs['train'].shape[1]} features | "
              f"paper: {prov['paper']}")
        for d in prov["deviations"]:
            print(f"    deviation: {d}")

        X_tr, y_tr = designs["train"], ys["train"]
        if "downsample_prevalence" in prov:
            keep = downsample_to_prevalence(y_tr, prov["downsample_prevalence"], seed)
            print(f"  down-sampled train {len(y_tr):,} -> {len(keep):,} rows "
                  f"({y_tr[keep].mean():.1%} prevalence, as published)")
            X_tr, y_tr = X_tr.iloc[keep], y_tr[keep]

        fitter = FITTERS[prov["library"]]
        model, best = fitter(X_tr, y_tr, designs["val"], ys["val"],
                             seed=seed, device=device, quick=quick)
        s_val = model.predict_proba(designs["val"])[:, 1].astype(np.float64)
        s_test = model.predict_proba(designs["test"])[:, 1].astype(np.float64)

        payload[f"val_{name}"] = s_val
        payload[f"test_{name}"] = s_test
        records[name] = {
            "tuning": best,
            "n_features": int(designs["train"].shape[1]),
            "provenance": prov,
            "test_auprc": float(auprc(ys["test"], s_test)),
            "test_auroc": float(auroc(ys["test"], s_test)),
        }
        print(f"  -> test AUPRC {records[name]['test_auprc']:.4f} | "
              f"AUROC {records[name]['test_auroc']:.4f}")

    ordered_val, _ = built["val"]
    ordered_test, _ = built["test"]
    payload["val_stay_id"] = ordered_val["stay_id"].to_numpy()
    payload["val_hour"] = ordered_val["hour"].to_numpy()
    payload["test_stay_id"] = ordered_test["stay_id"].to_numpy()
    payload["test_hour"] = ordered_test["hour"].to_numpy()
    payload["records_json"] = json.dumps(records)
    Path(out_cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_cache, **payload)
    print(f"\n[done] {len(specs)} specs in {time.time() - t0:.0f}s -> {out_cache}")
    return records


def load_lit_scores(val_ordered, test_ordered, specs=None, path=SCORE_CACHE):
    """Cached raw scores, asserted against the caller's row order.

    Same alignment contract as ``load_fair_scores``: the cache stores the keys
    it was built on and refuses to hand back a misaligned array.
    """
    z = np.load(path, allow_pickle=False)
    for side, ordered in (("val", val_ordered), ("test", test_ordered)):
        if not ((z[f"{side}_stay_id"] == ordered["stay_id"].to_numpy()).all()
                and (z[f"{side}_hour"] == ordered["hour"].to_numpy()).all()):
            raise ValueError(f"{path}: {side} rows do not align with the cohort")
    names = specs or [k[4:] for k in z.files if k.startswith("val_")
                      and k not in ("val_stay_id", "val_hour")]
    return {n: (z[f"val_{n}"].astype(float), z[f"test_{n}"].astype(float))
            for n in names}


def merge_caches(parts, out=SCORE_CACHE):
    """Combine per-process score caches into one, checking they agree on rows.

    The specs are trained in parallel across devices (XGBoost on the GPU,
    LightGBM on the CPU), so each process writes its own cache; this stitches
    them back together for the evaluation script.
    """
    payload, records = {}, {}
    keys = None
    for path in parts:
        z = np.load(path, allow_pickle=False)
        this = {k: z[k] for k in ("val_stay_id", "val_hour",
                                  "test_stay_id", "test_hour")}
        if keys is None:
            keys = this
            payload.update(this)
        else:
            for k, v in this.items():
                if not (v == keys[k]).all():
                    raise ValueError(f"{path}: {k} disagrees with the first cache")
        for k in z.files:
            if k.startswith(("val_", "test_")) and k not in keys:
                payload[k] = z[k]
        records.update(json.loads(str(z["records_json"])))
    payload["records_json"] = json.dumps(records)
    np.savez_compressed(out, **payload)
    print(f"[merged] {len(records)} specs -> {out}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="sadanandan,yaib,ecart")
    ap.add_argument("--device", default=None,
                    help="xgboost tree-construction device, e.g. cuda")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=SCORE_CACHE)
    ap.add_argument("--merge", nargs="*", default=None,
                    help="merge these caches into --out instead of training")
    args = ap.parse_args()
    if args.merge:
        merge_caches(args.merge, out=args.out)
        return
    run(specs=tuple(s.strip() for s in args.specs.split(",")),
        device=args.device, quick=args.quick, out_cache=args.out)


if __name__ == "__main__":
    main()
