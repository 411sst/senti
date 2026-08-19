"""Unit tests for the evaluation metrics."""

import numpy as np
import pytest

from metrics import (auprc, auroc, calibration_slope_intercept, ece,
                     ece_adaptive, ici, lead_time)


# --- AUPRC -----------------------------------------------------------------

class TestAUPRC:
    def test_perfect_classifier_is_one(self):
        """Positives ranked strictly above negatives -> AUPRC = 1.0."""
        y_true = [0, 0, 1, 1]
        y_scores = [0.05, 0.2, 0.8, 0.95]
        assert auprc(y_true, y_scores) == pytest.approx(1.0)

    def test_random_model_scores_near_prevalence(self):
        """A constant (uninformative) score gives roughly the event rate."""
        y_true = [0, 0, 0, 1]  # prevalence 0.25
        y_scores = [0.5, 0.5, 0.5, 0.5]
        assert auprc(y_true, y_scores) == pytest.approx(0.25)

    def test_inverted_ranking_is_worse_than_prevalence(self):
        """Positives ranked below negatives -> AUPRC below prevalence."""
        y_true = [1, 1, 0, 0]  # prevalence 0.5
        y_scores = [0.1, 0.2, 0.8, 0.9]
        assert auprc(y_true, y_scores) < 0.5


# --- AUROC -----------------------------------------------------------------

class TestAUROC:
    def test_perfect_classifier_is_one(self):
        assert auroc([0, 0, 1, 1], [0.05, 0.2, 0.8, 0.95]) == pytest.approx(1.0)

    def test_random_scores_are_half(self):
        """A constant score gives exactly 0.5 -- AUROC's floor is prevalence-free."""
        assert auroc([0, 0, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)

    def test_inverted_ranking_is_zero(self):
        assert auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)

    def test_is_insensitive_to_prevalence_unlike_auprc(self):
        """The point of keeping AUPRC primary: AUROC hides class imbalance.

        Both conditions draw from the SAME score distributions and share one
        negative pool -- only how many negatives are included changes. So the
        ranking quality is identical by construction and any movement is the
        prevalence effect, not sampling noise.
        """
        rng = np.random.default_rng(0)
        pos = rng.normal(1.0, 1.0, 500)
        neg_pool = rng.normal(0.0, 1.0, 50_000)

        rocs, prcs = [], []
        for n_neg in (1_000, 50_000):              # 33% vs 1% prevalence
            neg = neg_pool[:n_neg]
            y = np.r_[np.ones(len(pos)), np.zeros(n_neg)]
            s = np.r_[pos, neg]
            rocs.append(auroc(y, s))
            prcs.append(auprc(y, s))

        assert abs(rocs[0] - rocs[1]) < 0.05        # AUROC barely moves
        assert prcs[0] - prcs[1] > 0.4              # AUPRC collapses

    def test_monotone_rescaling_does_not_change_auroc(self):
        rng = np.random.default_rng(1)
        y = rng.binomial(1, 0.3, 200)
        s = rng.normal(size=200)
        assert auroc(y, s) == pytest.approx(auroc(y, 3.0 * s + 7.0))


# --- ECE -------------------------------------------------------------------

class TestECE:
    def test_perfectly_calibrated_is_zero(self):
        """In each confidence group the predicted prob equals the event rate."""
        y_probs = [0.1] * 10 + [0.5] * 10 + [0.9] * 10
        y_true = (
            [1] + [0] * 9          # 0.1 group: 1/10 positive
            + [1] * 5 + [0] * 5    # 0.5 group: 5/10 positive
            + [1] * 9 + [0]        # 0.9 group: 9/10 positive
        )
        assert ece(y_true, y_probs, n_bins=10) == pytest.approx(0.0, abs=1e-9)

    def test_maximally_miscalibrated_is_one(self):
        """100%-confident predictions that are always wrong -> ECE = 1.0."""
        y_probs = [1.0] * 10
        y_true = [0] * 10
        assert ece(y_true, y_probs, n_bins=10) == pytest.approx(1.0)

    def test_known_gap(self):
        """One bin, confidence 0.8, observed frequency 0.5 -> gap 0.3."""
        y_probs = [0.8, 0.8, 0.8, 0.8]
        y_true = [1, 1, 0, 0]  # frequency 0.5
        assert ece(y_true, y_probs, n_bins=10) == pytest.approx(0.3)


# --- Lead time -------------------------------------------------------------

class TestLeadTime:
    def test_hand_built_case(self):
        """Three patients crossing 0.5 at hours 2, 3, 1; event at hour 6.

        Lead times are 6-2=4, 6-3=3, 6-1=5, so the median is 4; all three
        deteriorating patients are caught -> rate 1.0.
        """
        threshold = 0.5
        risk_scores = [
            [0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.95],  # first crosses at hour 2
            [0.1, 0.2, 0.3, 0.55, 0.6, 0.7, 0.8],  # first crosses at hour 3
            [0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],   # first crosses at hour 1
        ]
        event_times = [6, 6, 6]
        result = lead_time(risk_scores, event_times, threshold)
        assert result["median_lead_time"] == pytest.approx(4.0)
        assert result["caught_in_time_rate"] == pytest.approx(1.0)
        # leads are [4, 3, 5] -> sorted [3, 4, 5]
        assert result["lead_p25"] == pytest.approx(3.5)
        assert result["lead_p75"] == pytest.approx(4.5)
        assert result["n_caught"] == 3
        assert result["n_events"] == 3

    def test_partial_catch_rate_and_median(self):
        """2 of 4 deteriorating patients are warned in time.

        Caught leads are 6-2=4 and 6-5=1, so the median over the caught pair
        is 2.5, and the caught-in-time rate is 2/4 = 0.5.
        """
        threshold = 0.5
        risk_scores = [
            [0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.95],  # crosses hour 2, event 6 -> lead 4
            [0.1, 0.1, 0.1, 0.2, 0.3, 0.7, 0.8],   # crosses hour 5, event 6 -> lead 1
            [0.1, 0.2, 0.3, 0.4],                  # never crosses          -> missed
            [0.1, 0.1, 0.1, 0.9],                  # crosses hour 3 > event 2 -> late
        ]
        event_times = [6, 6, 6, 2]
        result = lead_time(risk_scores, event_times, threshold)
        assert result["median_lead_time"] == pytest.approx(2.5)
        assert result["caught_in_time_rate"] == pytest.approx(0.5)

    def test_excludes_non_crossers_and_late_crossers(self):
        """Only patients flagged at or before their event count toward median."""
        threshold = 0.5
        risk_scores = [
            [0.6, 0.7],        # crosses at hour 0, event hour 1 -> lead 1
            [0.1, 0.2, 0.3],   # never crosses            -> excluded
            [0.1, 0.1, 0.9],   # crosses at hour 2 > event hour 1 -> excluded
        ]
        event_times = [1, 5, 1]
        result = lead_time(risk_scores, event_times, threshold)
        assert result["median_lead_time"] == pytest.approx(1.0)
        assert result["caught_in_time_rate"] == pytest.approx(1 / 3)

    def test_respects_explicit_timestamps(self):
        """Crossing at t=2.0 with event at t=10.0 gives a lead of 8.0."""
        threshold = 0.5
        risk_scores = [[0.1, 0.8, 0.9]]
        times = [[0.0, 2.0, 4.0]]
        event_times = [10.0]
        result = lead_time(risk_scores, event_times, threshold, times=times)
        assert result["median_lead_time"] == pytest.approx(8.0)
        assert result["caught_in_time_rate"] == pytest.approx(1.0)

    def test_nan_median_but_zero_rate_when_no_patient_flagged(self):
        """Nobody caught -> median is nan, but the rate is 0.0 (not nan)."""
        risk_scores = [[0.1, 0.2], [0.0, 0.3]]
        event_times = [5, 5]
        result = lead_time(risk_scores, event_times, 0.5)
        assert np.isnan(result["median_lead_time"])
        assert np.isnan(result["lead_p25"])
        assert np.isnan(result["lead_p75"])
        assert result["caught_in_time_rate"] == 0.0
        assert result["n_caught"] == 0
        assert result["n_events"] == 2


# --- Input validation (regression: NEWS2 0-20 scores once slipped through) ---

def test_ece_rejects_out_of_range_scores():
    y = np.array([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ece(y, np.array([0.0, 7.0, 20.0]))          # a raw NEWS2 score
    with pytest.raises(ValueError):
        ece(y, np.array([-0.1, 0.5, 0.5]))
    with pytest.raises(ValueError, match="NaN"):
        ece(y, np.array([0.1, np.nan, 0.5]))


def test_ece_adaptive_rejects_out_of_range_scores():
    with pytest.raises(ValueError):
        ece_adaptive(np.array([0.0, 1.0]), np.array([0.5, 1.5]))


# --- Adaptive (equal-mass) ECE ------------------------------------------------

def test_ece_adaptive_perfectly_calibrated_two_groups():
    # 100 rows at p=0.1 with 10% positives, 100 at p=0.8 with 80% positives.
    y = np.array([1] * 10 + [0] * 90 + [1] * 80 + [0] * 20, dtype=float)
    p = np.array([0.1] * 100 + [0.8] * 100)
    assert ece_adaptive(y, p, n_bins=4) == pytest.approx(0.0, abs=1e-12)


def test_ece_adaptive_sees_miscalibration_equal_width_hides():
    # Everything predicted in [0, 0.1): equal-width ECE lumps it into one bin
    # where over- and under-confidence cancel; equal-mass bins keep them apart.
    y = np.array([0] * 500 + [1] * 500, dtype=float)
    p = np.array([0.05] * 500 + [0.055] * 500)       # both wrong, opposite ways
    assert ece(y, p, n_bins=10) == pytest.approx(abs(0.0525 - 0.5), abs=1e-3)
    assert ece_adaptive(y, p, n_bins=2) == pytest.approx(
        0.5 * 0.05 + 0.5 * (1 - 0.055), abs=1e-6)


# --- Calibration slope / intercept ---------------------------------------------

def test_calibration_slope_recovers_the_generating_model():
    rng = np.random.default_rng(42)
    logit = rng.normal(-3.0, 1.5, 60_000)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(p.shape) < p).astype(float)
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.05)
    assert intercept == pytest.approx(0.0, abs=0.15)


def test_calibration_slope_flags_overconfident_scores():
    rng = np.random.default_rng(7)
    logit = rng.normal(-3.0, 1.5, 60_000)
    p_true = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(p_true.shape) < p_true).astype(float)
    p_overconfident = 1.0 / (1.0 + np.exp(-2.0 * logit))   # doubled spread
    slope, _ = calibration_slope_intercept(y, p_overconfident)
    assert slope == pytest.approx(0.5, abs=0.05)


# --- ICI --------------------------------------------------------------------------

def test_ici_near_zero_when_calibrated_and_catches_a_shift():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 40_000)
    y = (rng.random(p.shape) < p).astype(float)
    assert ici(y, p) < 0.02
    shifted = np.clip(p + 0.10, 0.0, 1.0)
    assert ici(y, shifted) == pytest.approx(0.10, abs=0.03)
