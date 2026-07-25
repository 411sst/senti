"""Tests for the agent interface and the vitals-TCN agent.

The load-bearing test here is :func:`test_network_is_causal`: it rewrites the
future and asserts the past does not move. Everything else about this project's
leakage discipline is enforced at the dataframe level; for a sequence model the
architecture itself has to be checked.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from agents.base import KEY_COLUMNS, Agent, AgentOutput
from sklearn.isotonic import IsotonicRegression

from agents.vitals_tcn import (
    CONTINUOUS,
    N_FEATURES,
    VitalsTCNAgent,
    VitalsTCNNet,
    build_sequences,
    encode_features,
    fit_normalizer,
)
from mimic_cohort import VITALS

LABEL = "label_24h"


def _cohort(n_stays=6, n_hours=12, seed=0):
    """A small synthetic cohort shaped like the real one."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stays):
        deteriorates = s % 2 == 0
        for h in range(n_hours):
            # deteriorating stays drift upward in HR late in the stay
            drift = 40 * (h / n_hours) if deteriorates else 0
            row = {
                "stay_id": 1000 + s, "patient_id": 500 + s, "hour": h,
                "heart_rate": 70 + drift + rng.normal(0, 2),
                "resp_rate": 16 + rng.normal(0, 1),
                "spo2": 97 + rng.normal(0, 1),
                "sbp": 120 + rng.normal(0, 3),
                "temperature": 37 + rng.normal(0, 0.2),
                "gcs": 15.0, "gcs_unable": 0.0, "on_oxygen": float(deteriorates),
                LABEL: int(deteriorates and h >= n_hours - 4),
            }
            for v in VITALS:
                row[f"{v}_hrs_since"] = 1.0
            rows.append(row)
    return pd.DataFrame(rows)


def _variable_length_cohort():
    """Stays of differing length, so length-bucketed batching reorders rows."""
    frames = []
    for i, n_hours in enumerate((3, 11, 5, 8)):
        f = _cohort(n_stays=1, n_hours=n_hours, seed=10 + i)
        f["stay_id"] = 2000 + i
        f["patient_id"] = 700 + i
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


# --- Causality (the leakage test for a sequence model) ---------------------

def test_network_is_causal():
    """Changing inputs at hour t must not change outputs at any hour < t."""
    torch.manual_seed(0)
    net = VitalsTCNNet(n_features=4, channels=8, dilations=(1, 2)).eval()
    x = torch.randn(1, 4, 20)

    with torch.no_grad():
        base_logits, base_emb = net(x)
        tampered = x.clone()
        tampered[:, :, 10:] += 5.0          # rewrite the future
        new_logits, new_emb = net(tampered)

    # Everything strictly before the tampered point is untouched...
    assert torch.allclose(base_logits[:, :10], new_logits[:, :10], atol=1e-6)
    assert torch.allclose(base_emb[:, :10], new_emb[:, :10], atol=1e-6)
    # ...and the change does register from that point on (the test can fail).
    assert not torch.allclose(base_logits[:, 10:], new_logits[:, 10:], atol=1e-6)


def test_causal_conv_output_length_matches_input():
    net = VitalsTCNNet(n_features=N_FEATURES, channels=8, dilations=(1, 2))
    x = torch.randn(2, N_FEATURES, 7)
    logits, emb = net(x)
    assert logits.shape == (2, 7)
    assert emb.shape == (2, 7, 8)


def test_single_hour_sequence_works():
    """Stays can be one hour long; padding must not break."""
    net = VitalsTCNNet(n_features=N_FEATURES, channels=8, dilations=(1, 2))
    logits, emb = net(torch.randn(1, N_FEATURES, 1))
    assert logits.shape == (1, 1) and emb.shape == (1, 1, 8)


# --- Feature encoding ------------------------------------------------------

def test_normalizer_uses_train_statistics_only():
    train = _cohort(seed=1)
    stats = fit_normalizer(train)
    assert set(stats) == set(CONTINUOUS)
    hr_mean, hr_std = stats["heart_rate"]
    assert hr_mean == pytest.approx(train["heart_rate"].mean(), rel=1e-6)
    assert hr_std > 0


def test_encoding_marks_missing_values_and_imputes_to_mean():
    df = _cohort(n_stays=2, n_hours=4)
    df.loc[0, "heart_rate"] = np.nan
    stats = fit_normalizer(df)
    X = encode_features(df, stats)

    assert X.shape == (len(df), N_FEATURES)
    assert not np.isnan(X).any()          # network never sees NaN
    hr_idx = CONTINUOUS.index("heart_rate")
    mask_idx = len(CONTINUOUS) + 2 + VITALS.index("heart_rate")  # 2 binary channels
    assert X[0, hr_idx] == 0.0            # missing -> channel mean (z = 0)
    assert X[0, mask_idx] == 1.0          # and flagged as missing
    assert X[1, mask_idx] == 0.0


def test_sequences_split_by_stay_and_sort_by_hour():
    df = _cohort(n_stays=3, n_hours=5).sample(frac=1, random_state=0)  # shuffled
    stats = fit_normalizer(df)
    seqs = build_sequences(df, stats)
    assert len(seqs) == 3
    for s in seqs:
        assert list(s["hours"]) == sorted(s["hours"])   # chronological
        assert len(s["hours"]) == 5


# --- Isotonic calibration (inherited from Agent) ---------------------------

def test_calibration_is_fit_only_on_val(monkeypatch):
    """THE calibration guarantee: isotonic sees val labels, never train's.

    Spies on the isotonic fit and checks the exact label vector it received.
    Train and val are given different sizes and different positive rates, so
    fitting on the wrong split cannot coincidentally pass.
    """
    captured = {}
    original_fit = IsotonicRegression.fit

    def spy(self, X, y, **kwargs):
        captured["X"] = np.asarray(X).copy()
        captured["y"] = np.asarray(y).copy()
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(IsotonicRegression, "fit", spy)

    train = _cohort(n_stays=8, n_hours=10, seed=20)     # 80 rows
    val = _cohort(n_stays=4, n_hours=6, seed=21)        # 24 rows
    assert len(train) != len(val)

    agent = VitalsTCNAgent(channels=8, max_epochs=1, batch_size=4, verbose=False)
    agent.fit(train, val)

    val_sorted = val.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    assert len(captured["y"]) == len(val)
    assert np.array_equal(captured["y"], val_sorted[LABEL].to_numpy(dtype=float))
    # and the summary the agent reports agrees
    assert agent.calibration_summary["split_rows"] == len(val)


def test_calibration_is_not_refit_by_emit(monkeypatch):
    """Calling emit() on test data must never update the calibrator."""
    train, val = _cohort(seed=22), _cohort(seed=23)
    test = _cohort(n_stays=4, n_hours=9, seed=24)

    agent = VitalsTCNAgent(channels=8, max_epochs=1, batch_size=4, verbose=False)
    agent.fit(train, val)
    before = agent.calibrator

    fits = []
    monkeypatch.setattr(IsotonicRegression, "fit",
                        lambda self, X, y, **kw: fits.append(1))
    agent.emit(test)
    agent.raw_risk(test)
    assert fits == []                    # no refit happened
    assert agent.calibrator is before    # same fitted object


def test_calibration_is_monotone_so_ranking_is_preserved():
    """Isotonic changes probabilities but must not reorder anything."""
    train, val = _cohort(seed=25), _cohort(seed=26)
    agent = VitalsTCNAgent(channels=8, max_epochs=1, batch_size=4, verbose=False)
    agent.fit(train, val)

    raw = agent.raw_risk(val)
    cal = agent.emit(val).risk
    # A monotone map cannot invert any pair: wherever raw increases, calibrated
    # must not decrease.
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(cal[order]) >= -1e-6)


def test_calibrate_before_fit_raises():
    agent = VitalsTCNAgent(channels=8, verbose=False)
    with pytest.raises(RuntimeError):
        agent.calibrate(np.array([0.5]))


def test_temperature_scaling_is_monotone_so_ranking_is_unchanged():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=500)
    t = 2.5
    assert np.array_equal(np.argsort(logits), np.argsort(logits / t))


# --- Agent contract --------------------------------------------------------

def test_agent_emits_latent_and_calibrated_risk():
    train, val = _cohort(seed=2), _cohort(seed=3)
    agent = VitalsTCNAgent(channels=8, max_epochs=1, batch_size=4, verbose=False)
    agent.fit(train, val)

    out = agent.emit(val)
    assert isinstance(out, AgentOutput)
    assert list(out.keys.columns) == KEY_COLUMNS
    assert len(out.keys) == len(val)
    assert out.latent.shape == (len(val), agent.latent_dim)
    assert out.risk.shape == (len(val),)
    assert ((out.risk >= 0) & (out.risk <= 1)).all()   # calibrated probability


def test_emitted_keys_align_with_sorted_frame():
    """Rows come back in (stay_id, hour) order so agents can be joined by key.

    Uses VARIABLE-length stays on purpose: inference batches are bucketed by
    length, so an equal-length cohort would pass this test even if emit() leaked
    that internal ordering to the caller.
    """
    train = _cohort(seed=4)
    val = _variable_length_cohort()
    assert val.groupby("stay_id").size().nunique() > 1, "stays must differ in length"

    agent = VitalsTCNAgent(channels=8, max_epochs=1, batch_size=2, verbose=False)
    agent.fit(train, val)
    out = agent.emit(val)

    expected = val.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    assert (out.keys["stay_id"].to_numpy() == expected["stay_id"].to_numpy()).all()
    assert (out.keys["hour"].to_numpy() == expected["hour"].to_numpy()).all()


def test_raw_risk_uses_the_same_row_order_as_emit():
    """raw_risk() and emit() must return the same rows in the same order.

    Checked with an identity calibrator, so any remaining difference is a
    row-ordering difference rather than a calibration effect.
    """
    train, val = _cohort(seed=6), _variable_length_cohort()
    agent = VitalsTCNAgent(channels=8, max_epochs=1, batch_size=2, verbose=False)
    agent.fit(train, val)

    identity = IsotonicRegression(out_of_bounds="clip").fit([0.0, 1.0], [0.0, 1.0])
    agent.calibrator = identity
    np.testing.assert_allclose(agent.raw_risk(val), agent.emit(val).risk, rtol=1e-5)


def test_emit_before_fit_raises():
    agent = VitalsTCNAgent(channels=8, verbose=False)
    with pytest.raises(RuntimeError):
        agent.emit(_cohort())


def test_agent_output_roundtrips_through_cache(tmp_path):
    keys = pd.DataFrame({"stay_id": [1, 1, 2], "hour": [6, 7, 6]})
    out = AgentOutput(keys=keys, latent=np.random.rand(3, 4).astype(np.float32),
                      risk=np.array([0.1, 0.2, 0.3], dtype=np.float32))
    path = out.save(tmp_path / "emb.npz")
    back = AgentOutput.load(path)
    assert back.latent.shape == out.latent.shape
    np.testing.assert_allclose(back.latent, out.latent, rtol=1e-6)
    np.testing.assert_allclose(back.risk, out.risk, rtol=1e-6)
    assert back.keys.equals(out.keys)


def test_agent_output_rejects_length_mismatch():
    keys = pd.DataFrame({"stay_id": [1, 2], "hour": [6, 6]})
    with pytest.raises(ValueError):
        AgentOutput(keys=keys, latent=np.zeros((3, 4)), risk=np.zeros(2))


def test_vitals_agent_satisfies_the_agent_interface():
    assert issubclass(VitalsTCNAgent, Agent)
    agent = VitalsTCNAgent(channels=8, verbose=False)
    assert agent.name == "vitals_tcn"
    assert agent.latent_dim == 8
