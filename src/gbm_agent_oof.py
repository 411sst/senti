"""Out-of-fold training scores for a GBM agent (T43).

The shipped GBM agent (``scripts/orch_gbm_agent.py``) scores its TRAINING rows
out of fold: training patients are split once into two halves with a fixed
generator (``numpy.random.default_rng(42)`` over the sorted patient ids), a
model is fit on each half and scores the other. The fusion heads are trained on
those rows, so they never learn to trust in-sample scores they will not get at
inference. Validation and test rows are scored by the model fit on all of train.

This module holds the fold logic so T43's all-features agent uses exactly the
same split as the shipped agent, and so the split can be unit-tested.
"""

import numpy as np

FOLD_SEED = 42


def patient_halves(patient_ids, seed=FOLD_SEED):
    """The two held-out patient sets, as the shipped agent draws them.

    ``patient_ids`` is the per-row patient id of the sorted training frame.
    Returns ``(first, second)``: sets of patient ids, disjoint, covering every
    patient once.
    """
    patients = np.sort(np.unique(np.asarray(patient_ids)))
    perm = np.random.default_rng(seed).permutation(patients)
    half = len(patients) // 2
    return set(perm[:half].tolist()), set(perm[half:].tolist())


def oof_scores(X, y, patient_ids, fit, predict, seed=FOLD_SEED):
    """Out-of-fold scores: each row is scored by the model that never saw its
    patient.

    ``fit(X_rows, y_rows) -> model`` and ``predict(model, X_rows) -> scores``
    are supplied by the caller. Returns ``(scores, models)`` with the two fold
    models in held-out-half order.
    """
    pid = np.asarray(patient_ids)
    scores = np.full(len(pid), np.nan)
    models = []
    for held in patient_halves(pid, seed):
        mask = np.isin(pid, list(held))
        model = fit(X[~mask], y[~mask])
        scores[mask] = predict(model, X[mask])
        models.append(model)
    if np.isnan(scores).any():
        raise ValueError("some training rows were not scored out of fold")
    return scores, models


def logit(p, eps=1e-6):
    """The agent's latent: log-odds of the raw probability, clipped as the
    shipped agent clips it."""
    q = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(q / (1 - q))


def held_out_fold(patient_ids, train_patient_ids, seed=FOLD_SEED):
    """For new training rows of patients in our training split: the index of
    the half that holds each row's patient, so row i is scored by the fold
    model that never saw that patient (fold k scores ``patient_halves(...)[k]``).
    Raises for a patient outside the training split."""
    halves = patient_halves(train_patient_ids, seed)
    pid = np.asarray(patient_ids)
    fold = np.full(len(pid), -1)
    for k, held in enumerate(halves):
        fold[np.isin(pid, list(held))] = k
    if (fold < 0).any():
        raise ValueError(f"{int((fold < 0).sum())} rows belong to patients outside the training split")
    return fold
