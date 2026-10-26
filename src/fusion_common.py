"""Shared plumbing for the score-fusion experiments (results_fusion.md).

The corrected results read as a tradeoff -- the fair GBM wins ranking, the
latent orchestrator wins earliness (results.md). The fusion experiments ask
whether that tradeoff is real, and every one of them grades on the same
substrate defined here.

**Anti-optimism protocol.** Fusion rules have free parameters fit against
labels, and the paired-timing harness freezes thresholds on val; letting both
happen on the same rows would let a combiner overfit its own operating point.
So val is split ONCE, at the patient level, 50/50 (seeded):

    V_fit     everything with free label-reading parameters: per-parent
              isotonic calibrators, stack/gate weights, per-horizon LatentCast
              isotonics, distillation early stopping.
    V_freeze  NEWS2-anchored threshold freezing (fair_capability protocol) and
              val-side model SELECTION (constrained-stack w, Pareto points).

Test is touched exactly once per reported rung. Selection-on-V_freeze retains
the same val-selection optimism the repo already accepts in ``tune_gbm``.
Parents are re-evaluated under this same protocol, so their rows here differ
slightly from results.md (which froze on full val) -- protocols are never
mixed within a table.

**Raw scores in, fusion calibrates.** Volume-matched thresholding is invariant
to monotone rescaling, but max/stack/gate arithmetic is not -- both parents
must sit on a common probability scale fit on the same data half. So loaders
return RAW scores and :meth:`FusionContext.fit_calibrator` maps them.

**Seeds.** The orchestrator's seed noise exceeds typical fusion deltas
(results_ecg.md), so callers loop over the seeds in ``orch_meas_seeds.npz``
(built by scripts/train_orch_meas_seeds.py), re-fit every rule within seed,
and report mean + per-seed sign consistency. The fair GBM is seed-fixed.
"""

import json

import numpy as np
from sklearn.isotonic import IsotonicRegression

from baselines_fair import SCORE_CACHE, bootstrap_indices
from baselines_mimic import alertable_mask
from config import SEED
from fair_capability import (
    BUDGETS,
    frozen_thresholds,
    paired_first_alarm_diffs,
    summarize_paired,
)
from metrics import auprc

LABEL = "label_24h"
SEEDS_NPZ = "data/cache/orch_meas_seeds.npz"


def split_val_rows(val, seed=SEED):
    """Sorted val frame + patient-disjoint boolean row masks (fit, freeze).

    Deterministic in ``seed``; every patient lands in exactly one half, so no
    patient's hours inform both a calibrator/stacker and the frozen thresholds.
    """
    ordered = val.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    patients = np.sort(val["patient_id"].unique())
    perm = np.random.default_rng(seed).permutation(patients)
    fit_set = set(perm[: len(patients) // 2].tolist())
    fit_mask = ordered["patient_id"].isin(fit_set).to_numpy()
    return ordered, fit_mask, ~fit_mask


def load_orch_seeds(val_ordered, test_ordered, path=SEEDS_NPZ):
    """Per-seed RAW core-3+meas orchestrator scores, alignment-checked.

    Returns ``(seeds, {seed: {"val": ..., "test": ...}}, meta)``.
    """
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    for name, ordered in (("val", val_ordered), ("test", test_ordered)):
        if not ((z[f"stay_id_{name}"] == ordered["stay_id"].to_numpy()).all()
                and (z[f"hour_{name}"] == ordered["hour"].to_numpy()).all()):
            raise ValueError(f"{path} {name} rows do not align with the "
                             f"cohort -- re-run scripts/train_orch_meas_seeds.py")
    scores = {s: {"val": z[f"raw_s{s}_val"].astype(float),
                  "test": z[f"raw_s{s}_test"].astype(float)}
              for s in meta["seeds"]}
    return list(meta["seeds"]), scores, meta


def load_fair_scores(val_ordered, test_ordered, variant="FAIR+TRAJ",
                     path=SCORE_CACHE):
    """RAW fair-GBM scores ``(val, test)`` for one variant, alignment-checked.

    The cache stores test keys (checked exactly); val scores were emitted in
    the same sorted-(stay_id, hour) convention, so only their length can be
    checked.
    """
    z = np.load(path, allow_pickle=True)
    if not ((z["stay_id"] == test_ordered["stay_id"].to_numpy()).all()
            and (z["hour"] == test_ordered["hour"].to_numpy()).all()):
        raise ValueError(f"{path} rows do not align with the cohort test "
                         "split -- re-run src/baselines_fair.py")
    key = f"rawval_{variant}"
    if key not in z:
        raise KeyError(f"{path} lacks {key} -- re-run src/baselines_fair.py "
                       f"--variants ...,{variant} (newer runs cache val scores)")
    raw_val = z[key].astype(float)
    if len(raw_val) != len(val_ordered):
        raise ValueError(f"{key} has {len(raw_val)} rows, val has "
                         f"{len(val_ordered)}")
    return raw_val, z[f"raw_{variant}"].astype(float)


def paired_auprc_draws(y, scores_by_model, patient_ids, reps=1000, seed=SEED):
    """Patient-clustered bootstrap AUPRC draws, same resample for every model.

    Returns ``{name: (reps,) array}``; any pairwise delta CI falls out as the
    percentile CI of ``draws[a] - draws[b]``. One resampling loop serves every
    comparison, unlike ``clustered_bootstrap``'s single fixed reference.
    """
    names = list(scores_by_model)
    draws = {n: [] for n in names}
    for idx in bootstrap_indices(patient_ids, reps, seed):
        yb = y[idx]
        if yb.sum() in (0, len(yb)):
            continue
        for n in names:
            draws[n].append(auprc(yb, scores_by_model[n][idx]))
    return {n: np.asarray(v) for n, v in draws.items()}


def delta_ci(draws_a, draws_b):
    """Percentile 95% CI for AUPRC(a) - AUPRC(b) over paired draws."""
    d = draws_a - draws_b
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


class FusionContext:
    """Everything the harness needs, built once from the cohort splits.

    Not a stats object -- just the sorted frames, masks, label arrays and the
    NEWS2 comparator, plus the two operations every experiment repeats:
    calibrating a score on V_fit and pushing a rung through the frozen-
    threshold paired-timing harness.
    """

    def __init__(self, splits, seed=SEED):
        self.val_ordered, self.fit_mask, self.freeze_mask = \
            split_val_rows(splits["val"], seed)
        self.test_ordered = splits["test"].sort_values(
            ["stay_id", "hour"]).reset_index(drop=True)
        self.freeze_frame = self.val_ordered[self.freeze_mask]

        self.news2_val = self.val_ordered["news2"].to_numpy(dtype=float)
        self.news2_test = self.test_ordered["news2"].to_numpy(dtype=float)
        self.alertable_val = alertable_mask(self.val_ordered)
        self.alertable_test = alertable_mask(self.test_ordered)

        self.y_val = self.val_ordered[LABEL].to_numpy(dtype=float)
        self.y_test = self.test_ordered[LABEL].to_numpy(dtype=float)
        self.patient_test = self.test_ordered["patient_id"].to_numpy()
        self.stay_to_patient = dict(zip(self.test_ordered["stay_id"],
                                        self.test_ordered["patient_id"]))

    # -- calibration -------------------------------------------------------
    def fit_calibrator(self, raw_val):
        """Isotonic raw -> probability, fit on the V_fit half only."""
        return IsotonicRegression(out_of_bounds="clip").fit(
            raw_val[self.fit_mask], self.y_val[self.fit_mask])

    # -- selection surface (V_freeze, never test) --------------------------
    def freeze_auprc(self, s_val):
        return float(auprc(self.y_val[self.freeze_mask],
                           s_val[self.freeze_mask]))

    def freeze_lead(self, s_val, budget=0.02):
        """Median paired lead vs NEWS2 on the FREEZE half.

        Selection surface only: thresholds and race both live on V_freeze,
        which is in-sample for the threshold but identical across the
        candidates being compared -- the final test row never reuses it.
        """
        nf = self.news2_val[self.freeze_mask]
        af = self.alertable_val[self.freeze_mask]
        sf = s_val[self.freeze_mask]
        thr_n2, thr_m, _ = frozen_thresholds(nf, sf, af, budget)
        diffs = paired_first_alarm_diffs(self.freeze_frame, nf, sf,
                                         thr_n2, thr_m)
        return (float(np.median(diffs["diff"])) if len(diffs)
                else float("nan"))

    # -- the harness row ---------------------------------------------------
    def evaluate_rung(self, s_val, s_test, reps=2000):
        """Test AUPRC + per-budget paired first-alarm vs NEWS2.

        Thresholds come from the V_freeze rows of ``s_val`` (the
        fair_capability protocol with val restricted to the freeze half);
        ``s_test`` is scored once.
        """
        out = {"auprc": float(auprc(self.y_test, s_test)), "budgets": {}}
        nf = self.news2_val[self.freeze_mask]
        af = self.alertable_val[self.freeze_mask]
        for budget in BUDGETS:
            thr_n2, thr_m, _ = frozen_thresholds(nf, s_val[self.freeze_mask],
                                                 af, budget)
            diffs = paired_first_alarm_diffs(self.test_ordered,
                                             self.news2_test, s_test,
                                             thr_n2, thr_m)
            r = summarize_paired(diffs, self.stay_to_patient, reps=reps)
            r["news2_rate"] = float(
                (self.news2_test[self.alertable_test] >= thr_n2).mean())
            r["model_rate"] = float(
                (s_test[self.alertable_test] >= thr_m).mean())
            out["budgets"][budget] = r
        return out
