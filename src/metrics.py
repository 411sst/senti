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


def _validated_probs(y_probs):
    """Probabilities must live in [0, 1] -- anything else is a caller bug.

    The old behaviour silently dropped out-of-range values from every bin while
    still counting them in the denominator, which once made a NEWS2 score of
    0-20 produce a small-looking "ECE" with 79% of rows contributing nothing.
    Failing loudly is the fix.
    """
    p = np.asarray(y_probs, dtype=float)
    if p.size and (np.isnan(p).any() or p.min() < 0.0 or p.max() > 1.0):
        raise ValueError(
            "calibration metrics need probabilities in [0, 1]; got values in "
            f"[{np.nanmin(p) if p.size else float('nan'):.4g}, "
            f"{np.nanmax(p) if p.size else float('nan'):.4g}]"
            + (" with NaNs" if np.isnan(p).any() else "")
        )
    return p


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
    y_probs = _validated_probs(y_probs)
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


def ece_adaptive(y_true, y_probs, n_bins=15):
    """Equal-mass-binned ECE -- the honest version at low prevalence.

    Equal-WIDTH bins are nearly unfalsifiable when 4.7% prevalence pushes ~90%
    of predictions into the lowest bin: one bin dominates the weighted average
    and within-bin miscalibration averages away. Here the bins are quantiles of
    the score distribution instead, so every bin carries comparable weight and
    the low-risk region -- where alerting thresholds actually live -- is
    examined at full resolution. Duplicate quantile edges (isotonic scores tie
    heavily) are merged rather than left as empty bins.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_probs = _validated_probs(y_probs)
    if y_true.shape[0] == 0:
        return 0.0

    edges = np.unique(np.quantile(y_probs, np.linspace(0.0, 1.0, n_bins + 1)))
    assign = np.searchsorted(edges[1:-1], y_probs, side="right")
    total = 0.0
    for b in np.unique(assign):
        mask = assign == b
        total += (mask.mean()) * abs(y_probs[mask].mean() - y_true[mask].mean())
    return float(total)


def calibration_slope_intercept(y_true, y_probs, eps=1e-6):
    """Cox recalibration: logistic fit of the outcome on logit(predicted risk).

    The clinical-statistics standard for "weak calibration" (Van Calster et
    al.): slope 1 means risk spread is right (slope < 1 = overfitted/too
    extreme, > 1 = too timid); intercept 0 (given slope) means the overall
    level is right. Probabilities are clipped away from {0, 1} so isotonic's
    exact zeros stay finite in logit space.

    Returns ``(slope, intercept)``.
    """
    from sklearn.linear_model import LogisticRegression

    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(_validated_probs(y_probs), eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    fit = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    fit.fit(logit, y_true)
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def ici(y_true, y_probs, frac=0.1):
    """Integrated Calibration Index: mean |smoothed observed rate - predicted|.

    Austin & Steyerberg define ICI against a loess-smoothed calibration curve;
    statsmodels is not a dependency here, so the curve is estimated with the
    nearest equivalent: a Gaussian kernel smoother over the RANK of the
    prediction (equal-mass, like loess's span), with ``frac`` of the sample as
    the kernel's ~2-sigma width. Averaging |curve - prediction| over the
    observations weights the index by the score density, exactly as ICI
    prescribes.
    """
    from scipy.signal import fftconvolve

    y_true = np.asarray(y_true, dtype=float)
    p = _validated_probs(y_probs)
    n = y_true.shape[0]
    if n == 0:
        return 0.0

    order = np.argsort(p, kind="stable")
    y_sorted, p_sorted = y_true[order], p[order]

    sigma = max(1.0, frac * n / 2.0)
    half = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= kernel.sum()

    padded = np.concatenate([y_sorted[half:0:-1], y_sorted, y_sorted[-2:-half - 2:-1]])
    smoothed = fftconvolve(padded, kernel, mode="valid")
    return float(np.mean(np.abs(smoothed - p_sorted)))


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
