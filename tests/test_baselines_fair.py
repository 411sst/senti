"""Tests for the equal-information GBM baselines.

The rules these tests enforce:

* the modality_features refactor changed NOTHING -- the agents re-export the
  exact same objects, so all existing agent tests keep guarding the behaviour;
* every new trajectory feature is a function of data strictly before (or at)
  hours <= t whose inputs were themselves strictly-before-t -- rewriting the
  future must not move the past, and one stay must never see another's history;
* the note PCA is fit on train-split notes only;
* the bootstrap resamples PATIENTS, keeping each patient's rows together.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import modality_features
from agents import labs_agent
from baselines_fair import (
    bootstrap_indices,
    clustered_bootstrap,
    lab_delta_features,
    train_only_note_ids,
    variant_columns,
    vitals_trajectory_features,
)
from mimic_cohort import VITALS


# --- The refactor is a re-export, not a copy ---------------------------------

def test_agents_reexport_the_shared_builders():
    assert labs_agent.build_lab_features is modality_features.build_lab_features
    assert labs_agent.load_labs is modality_features.load_labs
    assert labs_agent.LAB_COLUMNS is modality_features.LAB_COLUMNS


def test_note_assembler_takes_width_from_embeddings():
    """The assembler must accept reduced (e.g. PCA) embeddings, not just 768-dim."""
    cohort = pd.DataFrame({"stay_id": [1, 1], "hour": [7, 8]})
    index = pd.DataFrame({"stay_id": [1], "note_id": ["n1"], "note_hour": [6.0]})
    emb = np.arange(5, dtype=np.float32).reshape(1, 5)
    out, has, age, keys = modality_features.assemble_note_features(
        cohort, index, emb, np.array(["n1"]))
    assert out.shape == (2, 5)
    assert has.tolist() == [1.0, 1.0]
    np.testing.assert_array_equal(out[0], emb[0])


# --- Trajectory features: leakage and correctness ------------------------------

def _stay(stay_id, hours, values):
    frame = {"stay_id": stay_id, "hour": hours}
    for v in VITALS:
        frame[v] = values
    return pd.DataFrame(frame)


def test_trajectory_values_are_exact():
    hours = list(range(6, 16))
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    df = _stay(1, hours, values).astype({"hour": int})
    feats = vitals_trajectory_features(df, windows=(6,))
    v = VITALS[0]
    # at hour 15 (last row): delta_6 = 100 - 40; trailing 6-window = rows 10..15
    assert feats[f"{v}_delta_6h"][-1] == pytest.approx(60.0)
    assert feats[f"{v}_min_6h"][-1] == pytest.approx(50.0)
    assert feats[f"{v}_max_6h"][-1] == pytest.approx(100.0)
    assert feats[f"{v}_mean_6h"][-1] == pytest.approx(75.0)
    # too little history -> NaN delta, aggregates over what exists
    assert np.isnan(feats[f"{v}_delta_6h"][0])
    assert feats[f"{v}_min_6h"][0] == pytest.approx(10.0)


def test_trajectory_uses_no_future_data():
    """Rewrite every hour after t: features at and before t must not move."""
    hours = list(range(6, 26))
    rng = np.random.default_rng(0)
    values = rng.normal(80, 10, len(hours))
    df = _stay(1, hours, values)
    feats = vitals_trajectory_features(df)

    corrupted = df.copy()
    t_row = 12                                        # hour 18
    for v in VITALS:
        corrupted.loc[corrupted.index > t_row, v] = 9999.0
    feats_c = vitals_trajectory_features(corrupted)

    for name, arr in feats.items():
        np.testing.assert_array_equal(
            arr[:t_row + 1], feats_c[name][:t_row + 1],
            err_msg=f"{name} changed when the future was rewritten")


def test_trajectory_does_not_cross_stays():
    a = _stay(1, range(6, 12), [100] * 6)
    b = _stay(2, range(6, 12), [40] * 6)
    feats = vitals_trajectory_features(pd.concat([a, b], ignore_index=True))
    v = VITALS[0]
    # stay 2's first row: no within-stay history -> NaN delta, own-value min --
    # stay 1's 100s must not leak into stay 2's window.
    assert np.isnan(feats[f"{v}_delta_6h"][6])
    assert feats[f"{v}_min_6h"][6] == pytest.approx(40.0)
    assert feats[f"{v}_max_6h"][6] == pytest.approx(40.0)


def test_trajectory_rejects_gappy_grid():
    df = _stay(1, [6, 7, 9, 10], [1, 2, 3, 4])       # hour 8 missing
    with pytest.raises(ValueError, match="dense"):
        vitals_trajectory_features(df)


def test_lab_deltas_do_not_cross_stays():
    lab_feats = pd.DataFrame({
        "stay_id": [1] * 3 + [2] * 3,
        "hour": [6, 7, 8] * 2,
        **{c: np.nan for c in modality_features.LAB_COLUMNS},
    })
    lab_feats["lab_lactate"] = [1.0, 2.0, 3.0, 9.0, 9.0, 9.0]
    out = lab_delta_features(lab_feats, window=2)
    d = out["lab_lactate_delta_2h"]
    assert d[2] == pytest.approx(2.0)                # within stay 1: 3 - 1
    assert np.isnan(d[3]) and np.isnan(d[4])         # stay 2 has no 2h history


# --- Note PCA population --------------------------------------------------------

def test_note_pca_population_is_train_only():
    index = pd.DataFrame({
        "stay_id": [1, 1, 2, 3],
        "note_id": ["a", "b", "c", "d"],
        "note_hour": [1.0, 2.0, 1.0, 1.0],
    })
    assert train_only_note_ids(index, train_stay_ids=[1, 3]) == {"a", "b", "d"}


# --- Variant nesting --------------------------------------------------------------

def test_variants_are_nested_and_unique():
    cols = variant_columns()
    assert set(cols["FAIR"]) < set(cols["FAIR+TRAJ"]) < set(cols["FAIR+NOTES"])
    for names in cols.values():
        assert len(names) == len(set(names))


# --- Clustered bootstrap ------------------------------------------------------------

def test_bootstrap_keeps_patient_rows_together():
    patients = np.array([10, 10, 10, 20, 20, 30])
    rows_per_patient = {10: 3, 20: 2, 30: 1}
    for idx in bootstrap_indices(patients, reps=20, seed=0):
        drawn = patients[idx]
        counts = pd.Series(drawn).value_counts()
        for pid, n in counts.items():
            assert n % rows_per_patient[pid] == 0, (
                "a patient's rows were split across the resample boundary")


def test_bootstrap_is_deterministic_and_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    n = 400
    y = (rng.random(n) < 0.2).astype(float)
    good = y + rng.normal(0, 0.6, n)                  # informative scores
    patients = np.repeat(np.arange(80), 5)
    scores = {"REF": good, "OTHER": rng.random(n)}

    b1 = clustered_bootstrap(y, scores, patients, "REF", reps=200, seed=7)
    b2 = clustered_bootstrap(y, scores, patients, "REF", reps=200, seed=7)
    assert b1["OTHER"]["auprc_ci"] == b2["OTHER"]["auprc_ci"]

    from metrics import auprc
    lo, hi = b1["REF"]["auprc_ci"]
    assert lo <= auprc(y, good) <= hi
    # REF is genuinely better, and the paired delta CI should say so
    d_lo, d_hi = b1["OTHER"]["delta_auprc_vs_ref"]
    assert d_lo > 0 and d_hi > d_lo
