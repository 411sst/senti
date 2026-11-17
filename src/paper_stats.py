"""Shared paired bootstrap for the paper tests (T30 onwards).

Every score in a comparison is evaluated on the SAME patient-clustered
resamples (``baselines_fair.bootstrap_indices`` with one seed), so a paired
difference is a per-resample difference. The resample generator is
deterministic for a seed, so each worker process regenerates the identical
index arrays rather than sharing gigabytes of them; the result does not depend
on the number of workers.

For AUPRC the draws are identical to ``fusion_common.paired_auprc_draws``
(same metric, same resamples, same skip rule); this module adds other targets
(AUROC, death-only AUPRC) and runs the scores in parallel.
"""

import multiprocessing as mp

import numpy as np

from baselines_fair import bootstrap_indices
from config import SEED
from metrics import auprc, auroc

METRICS = {"auprc": auprc, "auroc": auroc}

_STATE = {}


def _init(targets, patient_ids, reps, seed):
    _STATE.update(targets=targets, patient_ids=patient_ids, reps=reps,
                  seed=seed)


def _draws_one(item):
    name, scores = item
    targets, reps, seed = _STATE["targets"], _STATE["reps"], _STATE["seed"]
    out = {t: [] for t in targets}
    for idx in bootstrap_indices(_STATE["patient_ids"], reps, seed):
        ys = {t: y[idx] for t, (_, y) in targets.items()}
        if any(yb.sum() in (0, len(yb)) for yb in ys.values()):
            continue
        sb = scores[idx]
        for t, (metric, _) in targets.items():
            out[t].append(METRICS[metric](ys[t], sb))
    return name, {t: np.asarray(v) for t, v in out.items()}


def metric_draws(targets, scores_by_name, patient_ids, reps=1000, seed=SEED,
                 workers=1):
    """Bootstrap draws of several metrics for several scores, shared resamples.

    ``targets`` maps a target name to ``(metric, y)`` with ``metric`` one of
    ``METRICS``, e.g. ``{"auprc": ("auprc", y), "death_auprc": ("auprc",
    y_death)}``. Returns ``{target: {score name: (reps,) array}}``. A resample
    is skipped for every score and target if any target is single-class in it.
    """
    targets = {t: (m, np.asarray(y, dtype=float)) for t, (m, y) in targets.items()}
    for m, _ in targets.values():
        if m not in METRICS:
            raise ValueError(f"unknown metric {m!r}")
    items = [(n, np.asarray(s, dtype=float)) for n, s in scores_by_name.items()]
    args = (targets, np.asarray(patient_ids), reps, seed)
    if workers <= 1:
        _init(*args)
        results = [_draws_one(it) for it in items]
    else:
        ctx = mp.get_context("spawn")    # fork is unsafe once torch has threads
        with ctx.Pool(min(workers, len(items)), initializer=_init,
                      initargs=args) as pool:
            results = pool.map(_draws_one, items, chunksize=1)
    draws = {t: {} for t in targets}
    for name, per_target in results:
        for t, d in per_target.items():
            draws[t][name] = d
    return draws


def ci(values):
    v = np.asarray(values, dtype=float)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired_delta(point_a, point_b, draws_a, draws_b):
    """Point difference a - b with the percentile CI of per-resample deltas."""
    lo, hi = ci(np.asarray(draws_a) - np.asarray(draws_b))
    return {"delta": float(point_a - point_b), "ci": [lo, hi],
            "ci_above_zero": bool(lo > 0)}


def pooled_delta(points_a, point_b, draws_a, draws_b):
    """Mean over seeds of (a_seed - b), with the CI of that mean per resample.

    ``points_a`` / ``draws_a`` are lists over seeds; ``b`` is one fixed score
    (or, if ``draws_b`` is a list, paired seed by seed).
    """
    if isinstance(draws_b, list):
        deltas = [np.asarray(a) - np.asarray(b) for a, b in zip(draws_a, draws_b)]
        point = float(np.mean([pa - pb for pa, pb in zip(points_a, point_b)]))
    else:
        deltas = [np.asarray(a) - np.asarray(draws_b) for a in draws_a]
        point = float(np.mean(points_a) - point_b)
    pooled = np.mean(deltas, axis=0)
    lo, hi = ci(pooled)
    return {"delta": point, "ci": [lo, hi], "ci_above_zero": bool(lo > 0),
            "draws_mean": float(pooled.mean())}
