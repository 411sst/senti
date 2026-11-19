"""T49 measures that the shared bootstrap helpers do not cover (pure, no I/O).

* ``spec_threshold``: the score threshold that gives a target specificity on
  the negatives it is given (alarm when score > threshold).
* ``patient_counts``: per-patient row counts (rows, positives, true and false
  positives at a threshold), in ``np.unique`` patient order. Weighted by a
  resample's per-patient draw counts (``stacking.resample_counts``) they give
  that resample's positive rate, recall, precision and specificity without
  materialising the resampled rows.
* ``rates``: those four rates from (optionally weighted) patient counts.
"""

import numpy as np


def spec_threshold(negative_scores, specificity):
    """Smallest observed negative score ``thr`` such that the share of
    negatives with score <= thr is at least ``specificity``."""
    s = np.sort(np.asarray(negative_scores, dtype=float))
    if len(s) == 0:
        raise ValueError("no negatives")
    k = int(np.ceil(specificity * len(s))) - 1
    return float(s[min(max(k, 0), len(s) - 1)])


def patient_counts(y, scores, thr, patient_ids):
    """``(patients, counts)`` with ``counts`` of shape (n_patients, 4): rows,
    positives, true positives and false positives (alarm = score > thr)."""
    y = np.asarray(y).astype(bool)
    alarm = np.asarray(scores, dtype=float) > thr
    patients, inv = np.unique(np.asarray(patient_ids), return_inverse=True)
    n = len(patients)
    counts = np.stack([np.bincount(inv, minlength=n),
                       np.bincount(inv, weights=y, minlength=n),
                       np.bincount(inv, weights=y & alarm, minlength=n),
                       np.bincount(inv, weights=~y & alarm, minlength=n)], axis=1)
    return patients, counts.astype(float)


def rates(counts, weights=None):
    """Positive rate, recall, precision and specificity from patient counts."""
    w = np.ones(len(counts)) if weights is None else np.asarray(weights, dtype=float)
    rows, pos, tp, fp = (w @ counts).tolist()
    neg = rows - pos
    return {"positive_rate": pos / rows if rows else np.nan,
            "recall": tp / pos if pos else np.nan,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "specificity": 1.0 - fp / neg if neg else np.nan}
