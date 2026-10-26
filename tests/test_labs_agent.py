"""Tests for the labs agent (agent #3).

Labs are the easiest modality to leak, so most of these are leakage tests:
values must come from results AVAILABLE strictly before the prediction hour,
must go stale, and must not cross between stays. (The other half of the
guarantee -- binning by storetime rather than charttime -- lives in
scripts/mimic_labs.sql and is enforced at extract time.)
"""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from agents.base import Agent
from agents.labs_agent import (
    LAB_COLUMNS,
    LAB_MAX_STALENESS_HOURS,
    LabsAgent,
    build_lab_features,
)

LABEL = "label_24h"


def _labs(rows):
    """Sparse lab observations; unlisted analytes are absent.

    Always carries the key columns, so an empty panel (no labs drawn at all) is
    still a well-formed frame rather than a column-less one.
    """
    df = pd.DataFrame(rows)
    for c in ["stay_id", "hour"]:
        if c not in df:
            df[c] = pd.Series(dtype="int64")
    for c in LAB_COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df


def _cohort(stay_hours, seed=0):
    """{stay_id: [hours]} -> a cohort-shaped frame."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid, hours in stay_hours.items():
        for h in hours:
            rows.append({"stay_id": sid, "patient_id": 900 + sid, "hour": h,
                         LABEL: int(rng.random() < 0.3)})
    return pd.DataFrame(rows)


# --- Leakage: strictly before t --------------------------------------------

def test_lab_result_is_not_visible_in_its_own_hour():
    """A result stored in hour t must not be a feature at hour t."""
    cohort = _cohort({1: [6, 7, 8]})
    labs = _labs([{"stay_id": 1, "hour": 7, "lab_lactate": 4.0}])
    f = build_lab_features(cohort, labs).set_index("hour")

    assert math.isnan(f.loc[6, "lab_lactate"])   # before it existed
    assert math.isnan(f.loc[7, "lab_lactate"])   # its OWN hour -> still hidden
    assert f.loc[8, "lab_lactate"] == 4.0        # visible the hour after


def test_lab_value_is_carried_forward_until_it_goes_stale():
    cohort = _cohort({1: list(range(6, 6 + 40))})
    labs = _labs([{"stay_id": 1, "hour": 6, "lab_creatinine": 2.5}])
    f = build_lab_features(cohort, labs).set_index("hour")

    assert f.loc[7, "lab_creatinine"] == 2.5
    assert f.loc[6 + LAB_MAX_STALENESS_HOURS, "lab_creatinine"] == 2.5   # at bound
    assert math.isnan(f.loc[6 + LAB_MAX_STALENESS_HOURS + 1, "lab_creatinine"])
    assert f.loc[7, "lab_creatinine_hrs_since"] == 1


def test_pre_icu_labs_are_visible_as_history():
    """ED bloods drawn before ICU admission are legitimate history at hour 6."""
    cohort = _cohort({1: [6, 7]})
    labs = _labs([{"stay_id": 1, "hour": -3, "lab_wbc": 18.0}])
    f = build_lab_features(cohort, labs).set_index("hour")
    assert f.loc[6, "lab_wbc"] == 18.0
    assert f.loc[6, "lab_wbc_hrs_since"] == 9      # drawn 9 hours earlier


def test_labs_do_not_bleed_across_stays():
    cohort = _cohort({1: [6, 7], 2: [6, 7]})
    labs = _labs([{"stay_id": 1, "hour": 6, "lab_sodium": 130.0}])
    f = build_lab_features(cohort, labs).set_index(["stay_id", "hour"])
    assert f.loc[(1, 7), "lab_sodium"] == 130.0
    assert math.isnan(f.loc[(2, 7), "lab_sodium"])   # stay 2 must not see it


def test_features_are_aligned_to_cohort_rows():
    cohort = _cohort({3: [6, 7, 8], 1: [6, 7]})
    f = build_lab_features(cohort, _labs([]))
    expected = cohort.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    assert len(f) == len(cohort)
    assert (f["stay_id"].to_numpy() == expected["stay_id"].to_numpy()).all()
    assert (f["hour"].to_numpy() == expected["hour"].to_numpy()).all()


def test_missing_labs_stay_missing():
    cohort = _cohort({1: [6, 7]})
    f = build_lab_features(cohort, _labs([]))
    assert f[LAB_COLUMNS].isna().all().all()


# --- Agent contract --------------------------------------------------------

def _fittable(seed=0):
    """A cohort + labs pair big enough to fit a tiny model on."""
    hours = list(range(6, 22))
    cohort = _cohort({i: hours for i in range(1, 9)}, seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(1, 9):
        for h in range(-2, 20, 4):
            rows.append({"stay_id": sid, "hour": h,
                         "lab_lactate": float(rng.uniform(0.5, 6)),
                         "lab_creatinine": float(rng.uniform(0.5, 4)),
                         "lab_wbc": float(rng.uniform(2, 25))})
    return cohort, _labs(rows)


def test_agent_emits_latent_and_calibrated_risk():
    cohort, labs = _fittable(seed=1)
    val, _ = _fittable(seed=2)
    agent = LabsAgent(hidden=8, latent=4, max_epochs=1, batch_size=64,
                      labs=labs, verbose=False)
    agent.fit(cohort, val)

    out = agent.emit(val)
    assert out.latent.shape == (len(val), 4)
    assert out.risk.shape == (len(val),)
    assert ((out.risk >= 0) & (out.risk <= 1)).all()
    expected = val.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    assert (out.keys["stay_id"].to_numpy() == expected["stay_id"].to_numpy()).all()


def test_calibration_is_fit_only_on_val(monkeypatch):
    captured = {}
    original = IsotonicRegression.fit

    def spy(self, X, y, **kw):
        captured["y"] = np.asarray(y).copy()
        return original(self, X, y, **kw)

    monkeypatch.setattr(IsotonicRegression, "fit", spy)
    train, labs = _fittable(seed=3)
    val = _cohort({i: list(range(6, 14)) for i in range(1, 5)}, seed=4)
    assert len(train) != len(val)

    LabsAgent(hidden=8, latent=4, max_epochs=1, batch_size=64, labs=labs,
              verbose=False).fit(train, val)
    val_sorted = val.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    assert len(captured["y"]) == len(val)
    assert np.array_equal(captured["y"], val_sorted[LABEL].to_numpy(dtype=float))


def test_normalizer_uses_train_statistics_only():
    train, labs = _fittable(seed=5)
    val, _ = _fittable(seed=6)
    agent = LabsAgent(hidden=8, latent=4, max_epochs=1, batch_size=64,
                      labs=labs, verbose=False)
    agent.fit(train, val)
    # stats were fit before any val data was seen; spot-check one channel
    feats = build_lab_features(train, labs)["lab_lactate"].dropna()
    mean, _ = agent.stats["lab_lactate"]
    assert mean == pytest.approx(feats.mean(), rel=1e-6)


def test_emit_before_fit_raises():
    _, labs = _fittable()
    with pytest.raises(RuntimeError):
        LabsAgent(labs=labs, verbose=False).emit(_cohort({1: [6]}))


def test_satisfies_the_agent_interface():
    assert issubclass(LabsAgent, Agent)
    assert LabsAgent(verbose=False).name == "labs"
