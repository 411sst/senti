"""Equal-information GBM baselines: the fair bar the orchestrator must beat.

The audited GBM in ``baselines_mimic.py`` sees 13 vitals-derived columns, while
the orchestrator fuses labs + vitals + notes + NEWS2. Beating that GBM therefore
demonstrates modality access, not architecture. Each variant here hands ONE
XGBoost model the same underlying information the orchestrator's agents consume,
with honest val-based tuning, so whatever advantage the orchestrator retains is
attributable to its fusion/trajectory mechanism rather than to seeing more data.

Three nested feature sets::

    FAIR        vitals13 + 24 lab values + lab staleness   (labs agent's inputs)
    FAIR+TRAJ   + 6/12/24h deltas and trailing min/max/mean (the TCN's receptive
                  field, summarised the way a tree model can use it)
    FAIR+NOTES  + PCA of the ClinicalBERT note embedding + presence/age
                  (the notes agent's inputs, linearly compressed)

Slope features were considered and skipped: over a fixed dense window an
endpoint delta carries the same split information as delta/w, and a
least-squares slope has no NaN-safe closed form under LOCF gaps.

Protocol (mirrors the project's rules): a small grid is tuned on VAL AUPRC with
early stopping -- test is never touched until the final report; isotonic
calibration is fit on val; every headline number gets a patient-level clustered
bootstrap CI, because hours within a stay and stays within a patient are not
independent. NEWS2 is never a feature -- it is the competing baseline.

    python src/baselines_fair.py [--reps 1000] [--quick]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

from baselines_mimic import GBM_FEATURES, load_splits, train_gbm
from config import SEED
from metrics import auprc, auroc, ece
from mimic_cohort import EVENT_TYPES, VITALS
from modality_features import (
    LAB_COLUMNS,
    NOTE_EMBED_CACHE,
    NOTE_INDEX_PATH,
    assemble_note_features,
    build_lab_features,
    load_labs,
)

LABEL = "label_24h"
ORCH_TEST_NPZ = "data/cache/embeddings/orchestrator_test.npz"
SCORE_CACHE = "data/cache/fair_gbm_scores.npz"

TRAJ_WINDOWS = (6, 12, 24)
TRAJ_STATS = ("delta", "min", "max", "mean")
NOTE_COMPONENTS = 64

LAB_FEATURES = LAB_COLUMNS + [f"{c}_hrs_since" for c in LAB_COLUMNS]


# --- Feature construction ---------------------------------------------------

def _assert_dense_hours(ordered):
    """The trajectory features assume one row per hour within a stay."""
    step = ordered["hour"].diff()
    new_stay = ordered["stay_id"].diff() != 0
    if not ((step == 1) | new_stay).all():
        raise ValueError("hourly grid is not dense within stays -- "
                         "shift(w) would no longer mean 'w hours earlier'")


def vitals_trajectory_features(ordered, windows=TRAJ_WINDOWS):
    """Deltas and trailing-window aggregates of the before-t vitals.

    The vitals columns are already strictly-before-t (built in mimic_cohort),
    so any function of the CURRENT and EARLIER rows within a stay is also
    strictly-before-t: ``shift(w)`` looks back w hours and a trailing
    ``rolling(w)`` window ends at the current row. Grouping by stay stops one
    stay's history bleeding into the next; rows near the start of a stay get
    NaN, which XGBoost treats as first-class.

    Returns a dict of float32 arrays aligned to ``ordered``'s rows.
    """
    _assert_dense_hours(ordered)
    out = {}
    stay = ordered["stay_id"]
    for col in VITALS:
        v = ordered[col]
        by_stay = v.groupby(stay, sort=False)
        for w in windows:
            out[f"{col}_delta_{w}h"] = (v - by_stay.shift(w)).to_numpy(np.float32)
            roll = v.groupby(stay, sort=False).rolling(w, min_periods=1)
            for stat in ("min", "max", "mean"):
                agg = getattr(roll, stat)().droplevel(0).sort_index()
                out[f"{col}_{stat}_{w}h"] = agg.to_numpy(np.float32)
    return out


def lab_delta_features(lab_feats, window=24):
    """24h change in each carried lab value -- the 'is it trending?' signal.

    Computed over the cohort's prediction rows (dense hourly within a stay), so
    early hours where the look-back leaves the stay are NaN rather than guessed.
    """
    out = {}
    stay = lab_feats["stay_id"]
    for col in LAB_COLUMNS:
        v = lab_feats[col]
        out[f"{col}_delta_{window}h"] = (
            v - v.groupby(stay, sort=False).shift(window)
        ).to_numpy(np.float32)
    return out


def fit_note_pca(train_stay_ids, n_components=NOTE_COMPONENTS, seed=SEED,
                 embed_cache=NOTE_EMBED_CACHE, index_path=NOTE_INDEX_PATH):
    """PCA of the cached ClinicalBERT note embeddings, fit on TRAIN notes only.

    768 raw dimensions x 1.7M patient-hours does not fit in memory, so the
    embeddings are linearly compressed first. The projection is fit exclusively
    on notes belonging to train-split stays -- val/test note statistics never
    shape the axes -- then applied to every note.

    Returns ``(projected (n_notes, k), note_ids, note_index, explained)``.
    """
    z = np.load(embed_cache, allow_pickle=True)
    emb, ids = z["embeddings"], z["note_ids"]
    index = pd.read_parquet(index_path)

    train_note_ids = train_only_note_ids(index, train_stay_ids)
    ids_str = np.array([str(i) for i in ids])
    fit_mask = np.isin(ids_str, np.array(sorted(train_note_ids)))
    if fit_mask.sum() < n_components:
        raise ValueError(f"only {int(fit_mask.sum())} train notes for "
                         f"{n_components} PCA components")

    pca = PCA(n_components=n_components, random_state=seed,
              svd_solver="randomized")
    pca.fit(emb[fit_mask])
    projected = pca.transform(emb).astype(np.float32)
    return projected, ids, index, float(pca.explained_variance_ratio_.sum())


def train_only_note_ids(note_index, train_stay_ids):
    """Note ids belonging to train-split stays (the PCA's fit population)."""
    mask = note_index["stay_id"].isin(set(train_stay_ids))
    return set(note_index.loc[mask, "note_id"].astype(str))


def build_design(split_df, labs, note_pack):
    """One wide float32 design frame for a split; variants pick column subsets.

    Returns ``(X, y, keys)`` where ``keys`` carries stay_id/hour/patient_id in
    the same sorted (stay_id, hour) row order as ``X`` -- the order every score
    array in this module lives in.

    ``note_pack=None`` skips the note block (the note_* columns are simply
    absent), for clones without the note-embedding caches; only FAIR+NOTES
    needs them.
    """
    ordered = split_df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    blocks = {c: ordered[c].to_numpy(np.float32) for c in GBM_FEATURES}

    lab_feats = build_lab_features(ordered[["stay_id", "hour"]], labs)
    if not (lab_feats["stay_id"].to_numpy() == ordered["stay_id"].to_numpy()).all():
        raise ValueError("lab features misaligned with cohort rows")
    for c in LAB_FEATURES:
        blocks[c] = lab_feats[c].to_numpy(np.float32)

    blocks.update(vitals_trajectory_features(ordered))
    blocks.update(lab_delta_features(lab_feats))

    if note_pack is not None:
        projected, ids, index, _ = note_pack
        emb, has, age, nkeys = assemble_note_features(
            ordered[["stay_id", "hour"]], index, projected, ids)
        if not (nkeys["stay_id"].to_numpy() == ordered["stay_id"].to_numpy()).all():
            raise ValueError("note features misaligned with cohort rows")
        for k in range(emb.shape[1]):
            blocks[f"note_pc{k}"] = emb[:, k]
        blocks["note_has"] = has
        blocks["note_age"] = age

    X = pd.DataFrame(blocks)
    y = ordered[LABEL].to_numpy(dtype=float)
    keys = ordered[["stay_id", "hour", "patient_id"]].copy()
    return X, y, keys


def variant_columns(n_note_components=NOTE_COMPONENTS):
    """Nested feature lists: each variant adds one information source."""
    fair = list(GBM_FEATURES) + LAB_FEATURES
    traj = fair + [f"{c}_{s}_{w}h" for c in VITALS for w in TRAJ_WINDOWS
                   for s in TRAJ_STATS] \
                + [f"{c}_delta_24h" for c in LAB_COLUMNS]
    notes = traj + [f"note_pc{k}" for k in range(n_note_components)] \
                 + ["note_has", "note_age"]
    return {"FAIR": fair, "FAIR+TRAJ": traj, "FAIR+NOTES": notes}


# --- Tuning (val only -- test is never touched here) -------------------------

def tune_gbm(X_tr, y_tr, X_va, y_va, seed=SEED, quick=False, verbose=True,
             device=None):
    """Small val-tuned grid with early stopping; returns (model, record).

    Deliberately modest -- depth x class-weight, fixed conservative shrinkage --
    per the project's small-models rule. ``scale_pos_weight`` gets the two
    settings that matter at 4.7% prevalence: none, and fully rebalanced.

    ``device="cuda"`` moves tree construction to the GPU (hist method is
    identical in expectation; scores drift at float tolerance vs CPU, which is
    why regenerated caches are re-measured rather than compared bit-for-bit).
    """
    import xgboost as xgb

    pos_weight = float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum()))
    depths = (6,) if quick else (4, 6, 8)
    extra = {"device": device} if device else {}
    best_model, best = None, None
    for depth in depths:
        for spw in (1.0, pos_weight):
            model = xgb.XGBClassifier(
                n_estimators=3000, learning_rate=0.05, max_depth=depth,
                min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=spw, random_state=seed, n_jobs=-1,
                tree_method="hist", eval_metric="aucpr",
                early_stopping_rounds=50, **extra,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            score = auprc(y_va, model.predict_proba(X_va)[:, 1])
            if verbose:
                print(f"    depth={depth} spw={spw:>6.1f}: "
                      f"val AUPRC {score:.4f} ({model.best_iteration + 1} trees)")
            if best is None or score > best["val_auprc"]:
                best_model = model
                best = {"max_depth": depth, "scale_pos_weight": round(spw, 2),
                        "n_trees": int(model.best_iteration + 1),
                        "val_auprc": float(score)}
    return best_model, best


# --- Clustered inference ------------------------------------------------------

def bootstrap_indices(patient_ids, reps, seed=SEED):
    """Row-index arrays for patient-level bootstrap resamples.

    Patients are drawn with replacement; a drawn patient contributes ALL of
    their rows, preserving the within-patient correlation that a row-level
    bootstrap would ignore.
    """
    patients, inverse = np.unique(np.asarray(patient_ids), return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(patients)))
    rows = np.split(order, bounds[1:])
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        draw = rng.integers(0, len(patients), len(patients))
        yield np.concatenate([rows[d] for d in draw])


def clustered_bootstrap(y, scores_by_model, patient_ids, reference,
                        reps=1000, seed=SEED):
    """Percentile 95% CIs for AUPRC/AUROC per model and paired deltas vs one.

    All models are evaluated on the SAME resample each rep, so the delta CIs
    are paired -- between-patient sampling noise cancels out of the difference.
    """
    names = list(scores_by_model)
    stats = {n: {"auprc": [], "auroc": []} for n in names}
    for idx in bootstrap_indices(patient_ids, reps, seed):
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue                       # degenerate resample; effectively never
        for n in names:
            sb = scores_by_model[n][idx]
            stats[n]["auprc"].append(auprc(yb, sb))
            stats[n]["auroc"].append(auroc(yb, sb))

    def ci(v):
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    out = {}
    ref_auprc = np.asarray(stats[reference]["auprc"])
    for n in names:
        a = np.asarray(stats[n]["auprc"])
        out[n] = {
            "auprc_ci": ci(a),
            "auroc_ci": ci(np.asarray(stats[n]["auroc"])),
            "delta_auprc_vs_ref": None if n == reference else ci(ref_auprc - a),
        }
    return out


# --- Orchestrator comparator ---------------------------------------------------

def load_orchestrator_scores(keys, path=ORCH_TEST_NPZ):
    """Canonical calibrated orchestrator test scores, aligned to ``keys``."""
    z = np.load(path)
    same = ((z["stay_id"] == keys["stay_id"].to_numpy())
            & (z["hour"] == keys["hour"].to_numpy()))
    if not same.all():
        raise ValueError(f"{path} rows do not align with the test split -- "
                         "re-run scripts/train_orchestrator.py")
    return z["risk"].astype(float)


# --- Report --------------------------------------------------------------------

def run(reps=1000, quick=False, out_cache=SCORE_CACHE,
        variants=("FAIR", "FAIR+TRAJ", "FAIR+NOTES"), device=None):
    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    labs = load_labs()

    note_pack = None
    if "FAIR+NOTES" in variants:
        print("[notes] fitting PCA on train-split notes only...")
        note_pack = fit_note_pca(train["stay_id"].unique(),
                                 seed=SEED)
        print(f"[notes] {NOTE_COMPONENTS} components explain "
              f"{100 * note_pack[3]:.1f}% of embedding variance")

    print("[features] building design matrices (train/val/test)...")
    X_tr, y_tr, _ = build_design(train, labs, note_pack)
    X_va, y_va, _ = build_design(val, labs, note_pack)
    X_te, y_te, keys_te = build_design(test, labs, note_pack)
    test_ordered = test.sort_values(["stay_id", "hour"]).reset_index(drop=True)

    scores = {"ORCH": load_orchestrator_scores(keys_te),
              "NEWS2": test_ordered["news2"].to_numpy(dtype=float)}

    # The audited vitals-only, library-default GBM: the row every fair variant
    # is measured against to decompose "modality access" from "architecture".
    print("[GBM13] training the audited vitals-only baseline...")
    legacy = train_gbm(train, seed=SEED)
    scores["GBM13"] = legacy.predict_proba(
        test_ordered[list(GBM_FEATURES)])[:, 1].astype(float)

    records, calibrated, raw_val = {}, {}, {}
    for name, cols in variant_columns().items():
        if name not in variants:
            continue
        print(f"[{name}] tuning on val ({len(cols)} features)...")
        model, record = tune_gbm(X_tr[cols], y_tr, X_va[cols], y_va,
                                 quick=quick, device=device)
        raw_va = model.predict_proba(X_va[cols])[:, 1].astype(float)
        raw_te = model.predict_proba(X_te[cols])[:, 1].astype(float)
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_va, y_va)
        scores[name] = raw_te
        calibrated[name] = iso.transform(raw_te).astype(float)
        raw_val[name] = raw_va
        record["n_features"] = len(cols)
        records[name] = record

    print(f"[bootstrap] {reps} patient-level resamples...")
    boots = clustered_bootstrap(y_te, scores, keys_te["patient_id"].to_numpy(),
                                reference="ORCH", reps=reps)

    _print_report(y_te, test_ordered, scores, calibrated, records, boots)
    _save_cache(out_cache, keys_te, y_te, scores, calibrated, records,
                boots, reps, note_pack[3] if note_pack else float("nan"),
                raw_val=raw_val)
    return scores, boots


def _print_report(y, test_ordered, scores, calibrated, records, boots):
    prevalence = float(y.mean())
    print(f"\nEQUAL-INFORMATION GBM vs ORCHESTRATOR "
          f"({len(y):,} test hours, prevalence {100 * prevalence:.2f}%)")
    print("threshold-free discrimination on ALL test hours "
          "(same protocol as results.md); CIs are patient-clustered bootstrap\n")

    hdr = (f"  {'model':<12}{'AUPRC':>8}  {'95% CI':<17}{'AUROC':>7}  "
           f"{'95% CI':<17}{'ECEcal':>7}" + "".join(f"{t:>8}" for t in EVENT_TYPES))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in ("ORCH", "FAIR+NOTES", "FAIR+TRAJ", "FAIR", "GBM13", "NEWS2"):
        if name not in scores:
            continue
        s = scores[name]
        b = boots[name]
        e = (ece(y, calibrated[name]) if name in calibrated else float("nan"))
        per_type = [auprc(test_ordered[f"{LABEL}_{t}"].to_numpy(), s)
                    for t in EVENT_TYPES]
        print(f"  {name:<12}{auprc(y, s):>8.3f}  "
              f"[{b['auprc_ci'][0]:.3f}, {b['auprc_ci'][1]:.3f}]"
              f"{auroc(y, s):>7.3f}  "
              f"[{b['auroc_ci'][0]:.3f}, {b['auroc_ci'][1]:.3f}]"
              f"{e:>7.4f}" + "".join(f"{p:>8.3f}" for p in per_type))

    print("\n  Paired AUPRC deltas (ORCH minus model; positive = orchestrator ahead):")
    for name in ("FAIR+NOTES", "FAIR+TRAJ", "FAIR", "GBM13"):
        if name not in scores:
            continue
        lo, hi = boots[name]["delta_auprc_vs_ref"]
        point = auprc(y, scores["ORCH"]) - auprc(y, scores[name])
        verdict = ("orchestrator ahead" if lo > 0
                   else "model ahead" if hi < 0 else "CI crosses zero")
        print(f"    vs {name:<12} {point:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict}")

    gap_old = auprc(y, scores["ORCH"]) - auprc(y, scores["GBM13"])
    best_fair = max((n for n in ("FAIR", "FAIR+TRAJ", "FAIR+NOTES")
                     if n in scores),
                    key=lambda n: auprc(y, scores[n]))
    gap_new = auprc(y, scores["ORCH"]) - auprc(y, scores[best_fair])
    if gap_new > 0:
        print(f"\n  Of the published +{gap_old:.3f} AUPRC gap over the vitals-only "
              f"GBM, +{gap_new:.3f} survives against {best_fair} "
              f"({100 * gap_new / gap_old:.0f}% of the advantage).")
    else:
        print(f"\n  None of the published +{gap_old:.3f} AUPRC gap over the "
              f"vitals-only GBM survives: {best_fair} is AHEAD of the "
              f"orchestrator by {-gap_new:.3f}.")
    print("\n  Chosen hyperparameters (selected on val, never test):")
    for name, r in records.items():
        print(f"    {name:<12} {r}")


def run_from_cache(reps=1000, cache_path=SCORE_CACHE):
    """Refresh the comparison against the CURRENT canonical orchestrator scores.

    Exists for when the shipped orchestrator configuration changes: the fair
    variants' features and fitted models are untouched by that, so retraining
    18 GBMs to update one comparator column would be waste (and, under the
    fixed-seed protocol, would reproduce the cached scores bit-for-bit anyway).
    Variant scores, labels, keys, and any cached val scores are reused from the
    cache; only the ORCH column and the bootstrap are recomputed.
    """
    splits = load_splits()
    test_ordered = splits["test"].sort_values(
        ["stay_id", "hour"]).reset_index(drop=True)

    z = dict(np.load(cache_path, allow_pickle=True))
    meta = json.loads(str(z["meta_json"]))
    keys = pd.DataFrame({"stay_id": z["stay_id"], "hour": z["hour"],
                         "patient_id": z["patient_id"]})
    if not ((keys["stay_id"].to_numpy() == test_ordered["stay_id"].to_numpy()).all()
            and (keys["hour"].to_numpy() == test_ordered["hour"].to_numpy()).all()):
        raise ValueError(f"{cache_path} rows no longer match the cohort test "
                         "split -- re-run the full baselines_fair pipeline")

    y = z["y"].astype(float)
    scores = {k[len("raw_"):]: z[k].astype(float)
              for k in z if k.startswith("raw_")}
    calibrated = {k[len("cal_"):]: z[k].astype(float)
                  for k in z if k.startswith("cal_")}
    scores["ORCH"] = load_orchestrator_scores(keys)

    print(f"[bootstrap] {reps} patient-level resamples "
          f"(fair scores from cache, ORCH from canonical)...")
    boots = clustered_bootstrap(y, scores, keys["patient_id"].to_numpy(),
                                reference="ORCH", reps=reps)
    _print_report(y, test_ordered, scores, calibrated, meta["records"], boots)

    z["raw_ORCH"] = scores["ORCH"]
    meta.update(reps=reps, boots=boots)
    z["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(cache_path, **z)
    print(f"\n[cache] refreshed ORCH scores + bootstrap -> {cache_path}")
    return scores, boots


def _save_cache(path, keys, y, scores, calibrated, records, boots, reps,
                pca_explained, raw_val=None):
    """Scores + provenance, so the bootstrap/report can re-run without retraining.

    ``raw_val`` (variant -> val raw scores) is cached as ``rawval_<name>`` while
    the tuned models are still in hand -- threshold freezing and score fusion
    need val scores, and caching them here retires the refit path in
    ``fair_capability.fair_scores``.
    """
    meta = {"seed": SEED, "reps": reps, "pca_explained": pca_explained,
            "records": records,
            "boots": boots}
    payload = {
        "stay_id": keys["stay_id"].to_numpy(),
        "hour": keys["hour"].to_numpy(),
        "patient_id": keys["patient_id"].to_numpy(),
        "y": y,
        "meta_json": np.array(json.dumps(meta)),
    }
    for name, s in scores.items():
        payload[f"raw_{name}"] = s
    for name, s in calibrated.items():
        payload[f"cal_{name}"] = s
    for name, s in (raw_val or {}).items():
        payload[f"rawval_{name}"] = s
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"\n[cache] scores + bootstrap results -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000,
                    help="patient-level bootstrap resamples")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: single-depth grid")
    ap.add_argument("--from-cache", action="store_true",
                    help="reuse cached fair scores; refresh only the ORCH "
                         "column and the bootstrap")
    ap.add_argument("--variants", default="FAIR,FAIR+TRAJ,FAIR+NOTES",
                    help="comma list of variants to train (skip FAIR+NOTES on "
                         "clones without the note-embedding caches)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="XGBoost tree-construction device")
    args = ap.parse_args()
    if args.from_cache:
        run_from_cache(reps=args.reps)
    else:
        run(reps=args.reps, quick=args.quick,
            variants=tuple(v.strip() for v in args.variants.split(",") if v.strip()),
            device=None if args.device == "cpu" else args.device)
