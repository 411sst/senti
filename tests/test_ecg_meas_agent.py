"""Tests for the machine-measurements ECG agent (V4, ``ecg_meas``).

Feature-engineering pitfalls first (sentinel rules, the NORMAL/ABNORMAL regex
collision, QTc, axis wrap), then the inherited agent contract over a
49-wide "embedding" -- including the presence guard the whole design rests on.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from agents.base import Agent
from agents.ecg_agent import EcgHead
from agents.ecg_meas_agent import EcgMeasAgent
from agents.ecg_meas_features import (
    FEATURE_NAMES,
    NUMERIC_SPEC,
    RATE_FEATURES,
    REPORT_COLS,
    build_meas_features,
)

LABEL = "label_24h"


def _row(study_id=1, rr=800, p_onset=100, p_end=200, qrs_onset=260, qrs_end=350,
         t_end=660, p_axis=45, qrs_axis=40, t_axis=45, reports=("Sinus rhythm",)):
    row = {"study_id": study_id, "rr_interval": rr, "p_onset": p_onset,
           "p_end": p_end, "qrs_onset": qrs_onset, "qrs_end": qrs_end,
           "t_end": t_end, "p_axis": p_axis, "qrs_axis": qrs_axis, "t_axis": t_axis}
    for i, col in enumerate(REPORT_COLS):
        row[col] = reports[i] if i < len(reports) else None
    return row


def _features(rows):
    ids, X, names = build_meas_features(pd.DataFrame(rows))
    return ids, X, names


def _get(X, names, feat, i=0):
    return X[i, names.index(feat)]


# --- Sentinels & derived features ------------------------------------------

def test_sentinels_invalidate_but_zero_axis_is_valid():
    ids, X, names = _features([
        _row(study_id=1, p_axis=0),                    # 0 deg is a REAL axis
        _row(study_id=2, p_axis=29999),                # sentinel
        _row(study_id=3, p_onset=29999),               # kills p_dur and pr only
    ])
    assert _get(X, names, "p_axis_valid", 0) == 1.0 and _get(X, names, "p_axis", 0) == 0.0
    assert _get(X, names, "p_axis_valid", 1) == 0.0
    assert _get(X, names, "p_axis", 1) == NUMERIC_SPEC["p_axis"][0]   # imputed normal
    # partial sentinel: row 3 loses P-derived features, keeps QRS/QT
    assert _get(X, names, "p_dur_valid", 2) == 0.0
    assert _get(X, names, "pr_valid", 2) == 0.0
    assert _get(X, names, "qrs_dur_valid", 2) == 1.0
    assert _get(X, names, "qt_valid", 2) == 1.0


def test_derived_intervals_and_qtc():
    ids, X, names = _features([_row()])
    assert _get(X, names, "hr") == pytest.approx(75.0)          # 60000/800
    assert _get(X, names, "qrs_dur") == pytest.approx(90.0)     # 350-260
    assert _get(X, names, "qt") == pytest.approx(400.0)         # 660-260
    assert _get(X, names, "qtc_bazett") == pytest.approx(400.0 / np.sqrt(0.8))
    assert _get(X, names, "qtc_fridericia") == pytest.approx(400.0 / np.cbrt(0.8))


def test_qrst_angle_wraps():
    ids, X, names = _features([_row(qrs_axis=-175, t_axis=175)])  # 350 apart -> 10
    assert _get(X, names, "qrst_angle") == pytest.approx(10.0)


def test_out_of_bounds_is_invalid():
    ids, X, names = _features([_row(rr=100)])       # rr below the 200ms bound
    assert _get(X, names, "rr_valid") == 0.0
    assert _get(X, names, "hr_valid") == 0.0
    assert _get(X, names, "rr") == NUMERIC_SPEC["rr"][0]


# --- Statement regexes ------------------------------------------------------

def test_normal_ecg_does_not_match_inside_abnormal_ecg():
    ids, X, names = _features([
        _row(study_id=1, reports=("Normal ECG",)),
        _row(study_id=2, reports=("Abnormal ECG",)),
    ])
    assert _get(X, names, "normal_ecg", 0) == 1.0
    assert _get(X, names, "abnormal_ecg", 0) == 0.0
    assert _get(X, names, "normal_ecg", 1) == 0.0        # the collision guard
    assert _get(X, names, "abnormal_ecg", 1) == 1.0


def test_case_and_punctuation_variants_match():
    for stmt in ("Sinus rhythm", "Sinus rhythm.", "SINUS RHYTHM"):
        ids, X, names = _features([_row(reports=(stmt,))])
        assert _get(X, names, "sinus_rhythm") == 1.0, stmt


def test_clinical_patterns():
    ids, X, names = _features([
        _row(study_id=1, reports=("Atrial fibrillation with rapid ventricular response",)),
        _row(study_id=2, reports=("*** CONSIDER ACUTE ST ELEVATION MI ***",)),
        _row(study_id=3, reports=("Sinus rhythm with PVC(s)", "Left bundle branch block")),
        _row(study_id=4, reports=("Old inferior infarct",)),
        _row(study_id=5, reports=("Acute anterior infarct",)),
    ])
    assert _get(X, names, "afib", 0) == 1.0 and _get(X, names, "rvr", 0) == 1.0
    assert _get(X, names, "acute_ischemia", 1) == 1.0 and _get(X, names, "has_urgent", 1) == 1.0
    assert _get(X, names, "pvc", 2) == 1.0 and _get(X, names, "lbbb", 2) == 1.0
    assert _get(X, names, "old_infarct", 3) == 1.0 and _get(X, names, "acute_ischemia", 3) == 0.0
    assert _get(X, names, "old_infarct", 4) == 0.0 and _get(X, names, "acute_ischemia", 4) == 1.0


def test_fully_corrupt_study_is_dropped():
    ids, X, names = _features([
        _row(study_id=1),
        _row(study_id=2, rr=0, p_onset=29999, p_end=29999, qrs_onset=0,
             qrs_end=29999, t_end=29999, p_axis=29999, qrs_axis=29999,
             t_axis=32767, reports=("All 12 leads are missing",)),
    ])
    assert list(ids) == [1]                       # corrupt study 2 dropped


# --- Presence guard over a 49-d "embedding" ---------------------------------

def test_no_ecg_rows_emit_identical_risk_with_meas_width():
    torch.manual_seed(0)
    head = EcgHead(embed_dim=len(FEATURE_NAMES), hidden=8, latent=4).eval()
    n = 6
    feats = torch.randn(n, len(FEATURE_NAMES))
    with torch.no_grad():
        logits, latents = head(feats, torch.zeros(n), torch.rand(n) * 50)
    assert torch.allclose(logits, logits[0].expand(n), atol=1e-6)
    assert torch.allclose(latents, latents[0].expand_as(latents), atol=1e-6)


def test_fully_imputed_covered_row_differs_from_no_ecg_constant():
    """A study with every numeric invalid still has has_ecg=1 -> real pathway."""
    torch.manual_seed(1)
    head = EcgHead(embed_dim=len(FEATURE_NAMES), hidden=8, latent=4).eval()
    ids, X, names = _features([_row(rr=0, p_onset=29999, p_end=29999,
                                    qrs_onset=29999, qrs_end=29999, t_end=29999,
                                    p_axis=29999, qrs_axis=29999, t_axis=32767,
                                    reports=("Atrial fibrillation",))])
    x = torch.from_numpy(X)
    with torch.no_grad():
        covered, _ = head(x, torch.ones(1), torch.zeros(1))
        no_ecg, _ = head(x, torch.zeros(1), torch.zeros(1))
    assert not torch.allclose(covered, no_ecg, atol=1e-5)


# --- Agent contract ---------------------------------------------------------

def _cohort(stay_hours, label_fn=None):
    rows = []
    for stay, hours in stay_hours.items():
        for h in hours:
            rows.append({"stay_id": stay, "patient_id": 500 + stay, "hour": h,
                         LABEL: 0 if label_fn is None else label_fn(stay, h)})
    return pd.DataFrame(rows)


def _fitted(drop_features=()):
    hours = list(range(6, 26))
    stay_hours = {i: hours for i in range(1, 9)}
    ecg_stays = [i for i in range(1, 9) if i % 2 == 0]
    index = pd.DataFrame([(i, 5.0, 100 + i) for i in ecg_stays]
                         + [(i + 10, 5.0, 200 + i) for i in ecg_stays],
                         columns=["stay_id", "ecg_hour", "study_id"])
    rng = np.random.default_rng(0)
    meas_rows = [_row(study_id=s, rr=int(rng.integers(400, 1200)),
                      qrs_axis=int(rng.integers(-30, 90)))
                 for s in [100 + i for i in ecg_stays] + [200 + i for i in ecg_stays]]
    ids, X, names = build_meas_features(pd.DataFrame(meas_rows))

    agent = EcgMeasAgent(hidden=8, latent=4, max_epochs=2, batch_size=64,
                         verbose=False, drop_features=drop_features)
    # inject the cache the way tests do for V1 (bypasses npz on disk)
    if drop_features:
        keep = [i for i, n in enumerate(names) if n not in drop_features]
        X = X[:, keep]
    agent._cache = (X.astype(np.float32), ids, index)

    train = _cohort(stay_hours, lambda s, h: int(s % 2 == 0 and h >= 20))
    val = _cohort({i + 10: hours for i in range(1, 9)},
                  lambda s, h: int(s % 2 == 0 and h >= 20))
    return agent.fit(train, val), train, val


def test_agent_emits_latent_and_calibrated_risk():
    agent, _, val = _fitted()
    out = agent.emit(val)
    assert out.latent.shape == (len(val), 4)
    assert ((out.risk >= 0) & (out.risk <= 1)).all()
    raw = agent.raw_risk(val)
    has = agent.has_ecg(val)
    assert len(np.unique(np.round(raw[~has], 6))) == 1     # no-ECG constant


def test_drop_features_narrows_input_and_still_fits():
    agent, _, val = _fitted(drop_features=RATE_FEATURES)
    out = agent.emit(val)
    assert out.latent.shape == (len(val), 4)
    # the head saw fewer inputs than the full feature set
    assert agent.net.body[0].in_features == len(FEATURE_NAMES) - len(RATE_FEATURES) + 1


def test_satisfies_the_agent_interface():
    assert issubclass(EcgMeasAgent, Agent)
    assert EcgMeasAgent(verbose=False).name == "ecg_meas"
