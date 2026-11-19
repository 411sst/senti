"""T47 stacking helpers. Synthetic data only."""

import numpy as np

from baselines_fair import bootstrap_indices
from stacking import (W_GRID, blend, choose_weight, fit_logit3, logit, logit3_score,
                      pooled_share_first_ci, resample_counts, share_first)


def test_logit_clips_exact_zeros_and_ones():
    out = logit([0.0, 0.5, 1.0])
    assert np.isfinite(out).all()
    assert out[1] == 0.0 and np.isclose(out[0], -out[2])
    assert np.isclose(out[2], np.log((1 - 1e-6) / 1e-6))


def test_grid_is_eleven_weights():
    assert W_GRID == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def test_logit3_is_fit_only_on_given_rows_and_unpenalised():
    rng = np.random.default_rng(0)
    n = 4000
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2 * a - b - 1)))).astype(int)
    fit_rows = np.arange(n) < n // 2
    m = fit_logit3(a[fit_rows], b[fit_rows], y[fit_rows])
    # changing the other rows' labels leaves the fit unchanged
    y2 = y.copy()
    y2[~fit_rows] = 1 - y2[~fit_rows]
    m2 = fit_logit3(a[fit_rows], b[fit_rows], y2[fit_rows])
    assert np.allclose(m.coef_, m2.coef_) and np.allclose(m.intercept_, m2.intercept_)
    assert m.C == np.inf             # no penalty
    assert np.allclose(m.coef_.ravel(), [2, -1], atol=0.3)
    s = logit3_score(m, a, b)
    assert np.allclose(1 / (1 + np.exp(-s)), m.predict_proba(np.column_stack([a, b]))[:, 1])


def test_blend_end_points():
    ls, lg = np.array([1.0, -2.0]), np.array([3.0, 0.5])
    assert np.array_equal(blend(1.0, ls, lg), ls) and np.array_equal(blend(0.0, ls, lg), lg)
    assert np.allclose(blend(0.3, ls, lg), 0.3 * ls + 0.7 * lg)


def test_choose_weight_rule():
    grid = [{"w": 0.0, "lead": 0.0, "auprc": 0.40},
            {"w": 0.5, "lead": 1.0, "auprc": 0.35},
            {"w": 0.6, "lead": 2.0, "auprc": 0.33},
            {"w": 0.9, "lead": float("nan"), "auprc": 0.50}]
    assert choose_weight(grid)["w"] == 0.5          # best AUPRC among lead >= 1 h
    tie = grid + [{"w": 0.7, "lead": 1.5, "auprc": 0.35}]
    assert choose_weight(tie)["w"] == 0.5           # first in grid order wins a tie


def test_choose_weight_none_when_nothing_qualifies():
    grid = [{"w": w, "lead": 0.5, "auprc": 0.3} for w in W_GRID]
    assert choose_weight(grid) is None


def test_resample_counts_match_bootstrap_indices():
    rng = np.random.default_rng(1)
    pats = rng.integers(1000, 1050, 400)
    rows_per = {p: int((pats == p).sum()) for p in np.unique(pats)}
    for (patients, counts), idx in zip(resample_counts(pats, 5, 42),
                                       bootstrap_indices(pats, 5, 42)):
        drawn = pats[idx]
        for i, p in enumerate(patients):
            assert (drawn == p).sum() == counts[i] * rows_per[p]


def test_share_first_ignores_ties_and_uses_weights():
    d = np.array([2.0, -1.0, 0.0, 3.0])
    assert share_first(d) == 2 / 3
    assert share_first(d, [0, 1, 5, 1]) == 0.5
    assert np.isnan(share_first(np.array([0.0, 0.0])))


def test_pooled_share_first_ci():
    rng = np.random.default_rng(2)
    pats_all = np.arange(1000, 1200)
    diffs, pats = [], []
    for _ in range(3):
        p = rng.choice(pats_all, 80, replace=False)
        diffs.append(rng.choice([-2.0, 1.0, 3.0, 0.0], 80, p=[0.3, 0.4, 0.2, 0.1]))
        pats.append(p)
    point, (lo, hi), n = pooled_share_first_ci(diffs, pats, pats_all, 200, 42)
    assert n == 200 and lo <= point <= hi
    assert np.isclose(point, np.mean([share_first(d) for d in diffs]))
