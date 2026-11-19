"""T47: stacking Sentinel with the all-features GBM (pure helpers, no I/O).

* ``logit``: log-odds of a calibrated score, clipped to [eps, 1 - eps]
  (isotonic calibration outputs exact 0s and 1s).
* ``fit_logit3``: arm A, an unpenalised logistic regression on the two logits
  plus an intercept, fit on the rows it is given (V_fit).
* ``choose_weight``: arm B's rule. Highest V_freeze AUPRC among weights whose
  V_freeze median lead is at least the floor; ``None`` when no weight
  qualifies (the arm then fails; there is no fallback weight).
* ``resample_counts``: per-patient draw counts of the shared patient-clustered
  resamples. They are the same resamples ``baselines_fair.bootstrap_indices``
  yields for the same ids, reps and seed, so race statistics share resamples
  with the AUPRC and AUROC draws of ``paper_stats.metric_draws``.
* ``share_first``: among stays where the two first alarms differ, the share
  where the model alarms first, optionally weighted by those counts.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

W_GRID = tuple(round(0.1 * i, 1) for i in range(11))


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def fit_logit3(l_sentinel, l_gbm, y):
    """Arm A on the given rows; returns the fitted model (3 parameters)."""
    X = np.column_stack([l_sentinel, l_gbm])
    return LogisticRegression(C=np.inf, max_iter=1000).fit(X, np.asarray(y, dtype=int))


def logit3_score(model, l_sentinel, l_gbm):
    """Arm A's score on the log-odds scale (monotone in its probability, so it
    ranks and thresholds identically, without saturating to exact 1)."""
    return model.decision_function(np.column_stack([l_sentinel, l_gbm]))


def blend(w, l_sentinel, l_gbm):
    """Arm B: s = w * logit(Sentinel) + (1 - w) * logit(GBM)."""
    return w * np.asarray(l_sentinel) + (1.0 - w) * np.asarray(l_gbm)


def choose_weight(grid, lead_floor=1.0):
    """``grid``: list of {"w", "lead", "auprc"} measured on V_freeze. The
    first weight in grid order wins an exact AUPRC tie. NaN leads never
    qualify."""
    ok = [g for g in grid if np.isfinite(g["lead"]) and g["lead"] >= lead_floor]
    if not ok:
        return None
    best = max(g["auprc"] for g in ok)
    return next(g for g in ok if g["auprc"] == best)


def resample_counts(patient_ids, reps, seed):
    """Yield ``(patients, counts)`` per resample: ``patients`` is the sorted
    unique id array and ``counts[i]`` how often patient i was drawn."""
    patients = np.unique(np.asarray(patient_ids))
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        draw = rng.integers(0, len(patients), len(patients))
        yield patients, np.bincount(draw, minlength=len(patients)).astype(float)


def share_first(diff, weights=None):
    """``diff`` = opponent's first alarm - model's (> 0: model first)."""
    d = np.asarray(diff, dtype=float)
    w = np.ones(len(d)) if weights is None else np.asarray(weights, dtype=float)
    a, b = w[d > 0].sum(), w[d < 0].sum()
    return float(a / (a + b)) if a + b > 0 else float("nan")


def pooled_share_first_ci(diffs_by_seed, patients_by_seed, all_patient_ids, reps, seed):
    """Ten-seed mean share-first and its CI over the shared resamples: in each
    resample every seed's share is computed with the drawn patients' counts as
    weights, and the seeds are averaged. Returns (point, [lo, hi], n_used)."""
    point = float(np.nanmean([share_first(d) for d in diffs_by_seed]))
    draws = []
    pos = None
    for patients, counts in resample_counts(all_patient_ids, reps, seed):
        if pos is None:
            pos = [np.searchsorted(patients, np.asarray(p)) for p in patients_by_seed]
        vals = [share_first(d, counts[i]) for d, i in zip(diffs_by_seed, pos)]
        if np.all(np.isnan(vals)):
            continue
        draws.append(np.nanmean(vals))
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan))
    return point, [float(lo), float(hi)], len(draws)
