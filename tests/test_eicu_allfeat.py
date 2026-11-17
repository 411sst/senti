"""Tests for the eICU source swap of the all-features design (src/eicu_allfeat.py).

The features themselves are ``build_allfeat_design``'s, tested for leakage in
tests/test_allfeat_gbm.py. Here: the swap blanks exactly the ECG and cadence
blocks (to their no-record values), leaves every other column as the same
builders compute it, refuses an id overlap with the MIMIC ECG index, and still
never reads the future.
Every row is synthetic; ids are small integers.
"""

import numpy as np
import pytest

from allfeat_gbm import build_allfeat_design, column_sets
from eicu_allfeat import eicu_sources
from modality_features import CADENCE_FEATURES
from test_allfeat_gbm import ECG_NAMES, T0, _cohort, _rewrite_from, _sources


def _shift_ecg_ids(src, offset=1000):
    src["ecg"]["index"] = src["ecg"]["index"].assign(
        stay_id=src["ecg"]["index"]["stay_id"] + offset)
    return src


def test_swap_blanks_ecg_and_cadence_and_keeps_the_rest():
    cohort, src = _cohort(), _sources()
    X_full, y_full, keys = build_allfeat_design(cohort, src)
    mimic = _shift_ecg_ids(_sources())        # MIMIC ECG ids never match eICU stays
    swapped = eicu_sources(mimic, src["labs"], src["fluid"], src["resp"],
                           cohort["stay_id"].unique())
    X, y, keys2 = build_allfeat_design(cohort, swapped)
    assert list(X.columns) == list(X_full.columns)
    assert (keys2.to_numpy() == keys.to_numpy()).all() and (y == y_full).all()
    ecg = [f"ecg_{n}" for n in ECG_NAMES] + ["ecg_has", "ecg_age"]
    assert (X[ecg].to_numpy() == 0).all()
    # No cadence events: counts are 0 and hours-since sit at their cap, the
    # same for every row (the no-cadence arm drops the block anyway).
    cad = X[list(CADENCE_FEATURES)].to_numpy()
    assert (cad == cad[0]).all() and set(np.unique(cad)) <= {0.0, 24.0}
    assert X_full["ecg_has"].sum() > 0
    rest = [c for c in X.columns if c not in set(ecg) | set(CADENCE_FEATURES)]
    assert np.array_equal(X[rest].to_numpy(), X_full[rest].to_numpy(), equal_nan=True)
    assert len(column_sets(ECG_NAMES)["allfeat_nocad"]) == len(X.columns) - len(CADENCE_FEATURES)


def test_swap_refuses_an_id_overlap_with_the_mimic_ecg_index():
    cohort, src = _cohort(), _sources()
    with pytest.raises(ValueError, match="collide"):
        eicu_sources(src, src["labs"], src["fluid"], src["resp"], cohort["stay_id"].unique())


def test_swapped_design_never_reads_the_future():
    cohort, src = _cohort(), _sources()
    mimic = _shift_ecg_ids(_sources())
    a, _, keys = build_allfeat_design(
        cohort, eicu_sources(mimic, src["labs"], src["fluid"], src["resp"], [1, 2, 3]))
    src2, coh2 = _rewrite_from(src, cohort, T0)
    b, _, _ = build_allfeat_design(
        coh2, eicu_sources(mimic, src2["labs"], src2["fluid"], src2["resp"], [1, 2, 3]))
    past = (keys["hour"] <= T0).to_numpy()
    assert np.array_equal(a[past].to_numpy(), b[past].to_numpy(), equal_nan=True)
    assert not np.array_equal(a[~past].to_numpy(), b[~past].to_numpy(), equal_nan=True)
