"""Evaluation metrics for deterioration predictions.

Three metrics, each answering a different question about the model:

* :func:`auprc`    -- how well does it rank rare events? (primary metric)
* :func:`ece`      -- are its probabilities honest? (calibration)
* :func:`lead_time`-- how much earlier does it warn? (clinical value)
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auprc(y_true, y_scores):
    """Area under the precision-recall curve.

    This is our primary metric: deterioration events are rare, so the
    precision-recall curve reflects performance on the positive class far
    better than ROC. Implemented via scikit-learn's ``average_precision_score``
    (average precision), the standard, interpolation-free AUPRC estimator.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Ground-truth binary labels (1 = deterioration event).
    y_scores : array-like of float
        Predicted scores or probabilities; higher means higher risk.

    Returns
    -------
    float
        AUPRC in [0, 1]. A perfect ranking scores 1.0; a random model scores
        approximately the event prevalence.
    """
    return float(average_precision_score(y_true, y_scores))


def auroc(y_true, y_scores):
    """Area under the ROC curve -- reported alongside AUPRC, never instead of it.

    AUROC is included because it is the number most readers expect, but AUPRC
    remains this project's primary discrimination metric. With ~4.7% positive
    hours the negative class dominates, and AUROC's specificity axis moves very
    little as false positives accumulate -- a model can look strong on AUROC
    while its alerts are mostly wrong. AUPRC tracks precision directly, which is
    what an alerting system lives or dies by. Read AUROC as context, and note
    that its "random" baseline is a fixed 0.5 regardless of prevalence, whereas
    AUPRC's floor is the prevalence itself.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Ground-truth binary labels (1 = deterioration event).
    y_scores : array-like of float
        Predicted scores or probabilities; higher means higher risk.

    Returns
    -------
    float
        AUROC in [0, 1]; 0.5 is random, 1.0 is a perfect ranking.
    """
    return float(roc_auc_score(y_true, y_scores))


def ece(y_true, y_probs, n_bins=10):
    """Expected Calibration Error (binary).

    Predictions are grouped into ``n_bins`` equal-width confidence bins over
    [0, 1]. Within each bin we compare the average predicted probability
    (confidence) to the observed event frequency (accuracy); the ECE is the
    count-weighted average of those absolute gaps. A value of 0 means the
    predicted probabilities match reality exactly.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Ground-truth binary labels.
    y_probs : array-like of float
        Predicted event probabilities in [0, 1].
    n_bins : int, optional
        Number of equal-width bins (default 10).

    Returns
    -------
    float
        Expected Calibration Error in [0, 1]; lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_probs = np.asarray(y_probs, dtype=float)
    n = y_true.shape[0]
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Bins are half-open [lo, hi); the final bin includes the right edge
        # so that a probability of exactly 1.0 is counted.
        if i == n_bins - 1:
            mask = (y_probs >= lo) & (y_probs <= hi)
        else:
            mask = (y_probs >= lo) & (y_probs < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = y_probs[mask].mean()
        frequency = y_true[mask].mean()
        total += (count / n) * abs(confidence - frequency)
    return float(total)


def lead_time(risk_scores, event_times, threshold, times=None):
    """Median warning lead time at a chosen high-risk threshold.

    For each patient we find the first hour the risk score crosses
    ``threshold`` and measure how far ahead of the true event that warning
    was. The median of those lead times is our "we warn earlier" metric.

    Only patients who are flagged **at or before** their event contribute:
    a patient who never crosses the threshold got no warning, and one who
    crosses only after the event was not warned early -- both are excluded.

    Parameters
    ----------
    risk_scores : sequence of array-like
        One entry per patient: their sequence of per-hour risk scores.
    event_times : sequence
        One entry per patient: the true event time (same units as ``times``).
        ``None`` or ``NaN`` marks a patient with no event, who is skipped.
    threshold : float
        Risk score at or above which a patient is flagged high-risk.
    times : sequence of array-like, optional
        One entry per patient: timestamps aligned to each patient's
        ``risk_scores``. If omitted, hour indices ``0, 1, 2, ...`` are used.

    Returns
    -------
    dict
        Summary of the lead-time distribution over patients flagged in time:

        * ``median_lead_time`` / ``lead_p25`` / ``lead_p75`` -- the 50th/25th/
          75th percentiles of lead time (event time minus first-crossing time),
          or ``nan`` if nobody was caught. The band is reported alongside the
          median because the distribution is wide and right-skewed.
        * ``caught_in_time_rate`` is the fraction of deteriorating patients
          (those with a real event time) flagged at or before their event.
          It is ``0.0`` -- not ``nan`` -- when nobody is caught.
        * ``n_caught`` / ``n_events`` -- the numerator and denominator behind
          the rate (how many leads actually enter the percentiles).
    """
    leads = []
    n_deteriorated = 0
    for i, scores in enumerate(risk_scores):
        scores = np.asarray(scores, dtype=float)

        event_time = event_times[i]
        if event_time is None:
            continue
        event_time = float(event_time)
        if np.isnan(event_time):
            continue
        n_deteriorated += 1  # patient has a real event -> in the denominator

        if times is None:
            t = np.arange(scores.shape[0], dtype=float)
        else:
            t = np.asarray(times[i], dtype=float)

        # Evaluate in chronological order regardless of input ordering.
        order = np.argsort(t)
        t = t[order]
        scores = scores[order]

        crossed = np.flatnonzero(scores >= threshold)
        if crossed.size == 0:
            continue  # never flagged: no warning was issued
        first_cross_time = t[crossed[0]]
        if first_cross_time > event_time:
            continue  # flagged only after the event: not an early warning

        leads.append(event_time - first_cross_time)

    if leads:
        p25, p50, p75 = (float(np.percentile(leads, q)) for q in (25, 50, 75))
    else:
        p25 = p50 = p75 = float("nan")
    rate = len(leads) / n_deteriorated if n_deteriorated else 0.0
    return {
        "median_lead_time": p50,
        "lead_p25": p25,
        "lead_p75": p75,
        "caught_in_time_rate": rate,
        "n_caught": len(leads),
        "n_events": n_deteriorated,
    }
