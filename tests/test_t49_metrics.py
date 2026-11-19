"""T49 operating-point and rate helpers. Synthetic data only."""

import numpy as np
import pytest

from stacking import resample_counts
from t49_metrics import patient_counts, rates, spec_threshold


def test_spec_threshold_gives_at_least_the_target_specificity():
    rng = np.random.default_rng(0)
    neg = rng.normal(size=1000)
    for spec in (0.5, 0.9, 0.977):
        thr = spec_threshold(neg, spec)
        assert (neg <= thr).mean() >= spec
        assert (neg < thr).mean() < spec            # the smallest such threshold
    assert spec_threshold([3.0, 1.0, 2.0], 1.0) == 3.0
    with pytest.raises(ValueError):
        spec_threshold([], 0.9)


def test_counts_and_rates_by_hand():
    y = np.array([1, 0, 0, 1, 0, 1])
    s = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.95])
    pid = np.array([1010, 1010, 1020, 1020, 1030, 1030])
    patients, c = patient_counts(y, s, 0.5, pid)
    assert patients.tolist() == [1010, 1020, 1030]
    assert c.tolist() == [[2, 1, 1, 1], [2, 1, 0, 0], [2, 1, 1, 1]]
    r = rates(c)
    assert r == pytest.approx({"positive_rate": 0.5, "recall": 2 / 3, "precision": 0.5,
                               "specificity": 1 / 3})


def test_weighted_rates_equal_rates_on_the_resampled_rows():
    rng = np.random.default_rng(1)
    pid = rng.integers(1000, 1060, 800)
    y = (rng.random(800) < 0.2).astype(int)
    s = rng.random(800) + 0.3 * y
    patients, c = patient_counts(y, s, 0.7, pid)
    for (pats, w), _ in zip(resample_counts(pid, 5, 42), range(5)):
        assert np.array_equal(pats, patients)
        rows = np.concatenate([np.repeat(np.flatnonzero(pid == p), int(k))
                               for p, k in zip(pats, w)])
        _, c_rows = patient_counts(y[rows], s[rows], 0.7, pid[rows])
        assert rates(c, w) == pytest.approx(rates(c_rows))
