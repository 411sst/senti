"""T43: the GBM agent's out-of-fold training scores. Synthetic ids only."""

import numpy as np
import pytest

from gbm_agent_oof import held_out_fold, logit, oof_scores, patient_halves


def _rows(n_patients=37, seed=0):
    rng = np.random.default_rng(seed)
    patients = 1000 + np.arange(n_patients) * 7          # far from 3xxxxxxx
    pid = np.repeat(patients, rng.integers(1, 6, n_patients))
    return pid


def test_halves_are_disjoint_and_cover_every_patient():
    pid = _rows()
    a, b = patient_halves(pid)
    assert a.isdisjoint(b)
    assert a | b == set(np.unique(pid).tolist())
    assert abs(len(a) - len(b)) <= 1


def test_halves_match_the_shipped_agent_construction():
    # scripts/orch_gbm_agent.py: sort unique ids, rng(42).permutation, split at len // 2
    pid = _rows(seed=3)
    patients = np.sort(np.unique(pid))
    perm = np.random.default_rng(42).permutation(patients)
    half = len(patients) // 2
    a, b = patient_halves(pid)
    assert a == set(perm[:half].tolist())
    assert b == set(perm[half:].tolist())


def test_halves_do_not_depend_on_row_order():
    pid = _rows(seed=5)
    shuffled = np.random.default_rng(9).permutation(pid)
    assert patient_halves(pid) == patient_halves(shuffled)


def test_every_row_is_scored_by_the_model_that_never_saw_its_patient():
    pid = _rows(seed=1)
    X = pid.reshape(-1, 1).astype(float)
    y = np.zeros(len(pid))

    def fit(Xr, yr):
        return set(Xr[:, 0].astype(int).tolist())          # patients seen

    def predict(seen, Xr):
        return np.array([1.0 if int(p) in seen else 0.0 for p in Xr[:, 0]])

    scores, models = oof_scores(X, y, pid, fit, predict)
    assert len(models) == 2
    assert (scores == 0.0).all()                            # never in-sample
    assert models[0].isdisjoint(models[1])                  # each fold sees one half


def test_unscored_rows_raise():
    pid = _rows(seed=2)
    X = np.zeros((len(pid), 1))

    def predict(model, Xr):
        return np.full(len(Xr), np.nan)

    with pytest.raises(ValueError):
        oof_scores(X, np.zeros(len(pid)), pid, lambda a, b: None, predict)


def test_logit_matches_the_shipped_agent_formula():
    r = np.array([0.0, 1e-9, 0.2, 0.5, 0.97, 1.0])
    c = np.clip(r, 1e-6, 1 - 1e-6)
    np.testing.assert_allclose(logit(r), np.log(c / (1 - c)))


def test_held_out_fold_names_the_half_that_holds_each_patient():
    pid = _rows(seed=5)
    a, b = patient_halves(pid)
    new = np.array([min(a), min(b), min(a), max(b)])
    assert held_out_fold(new, pid).tolist() == [0, 1, 0, 1]
    with pytest.raises(ValueError):
        held_out_fold(np.array([min(a), 999]), pid)        # 999 not in training
