"""Tests for ICU-category stratification.

The load-bearing guarantees: the strata join never changes the stay count, an
unmapped care unit or service fails loudly instead of being bucketed away, and
the clinician-category assignment prefers the evidence it says it prefers.
"""

import numpy as np
import pandas as pd
import pytest

from icu_strata import (
    CAREUNIT_GROUP,
    CLINICIAN_CATEGORIES,
    SERVICE_GROUP,
    assign_strata,
    category_coverage,
    icd_chapter,
    stratum_metrics,
    unmapped_census,
)


def _meta(rows):
    base = {"first_careunit": "Medical Intensive Care Unit (MICU)",
            "last_careunit": "Medical Intensive Care Unit (MICU)",
            "admission_type": "EW EMER.", "admission_location": "ED",
            "curr_service": "MED", "service_is_fallback": False,
            "dx1_code": "I509", "dx1_version": 10, "dx1_title": "heart failure",
            "dx_burn": 0, "dx_resp": 0, "dx_gu": 0, "dx_cardio": 1,
            "dx_neuro": 0, "n_diagnoses": 5, "subject_id": 1,
            "gender": "M", "anchor_age": 60, "admit_hour_of_day": 12}
    return pd.DataFrame([{**base, **r} for r in rows])


# --- chapter mapping ---------------------------------------------------------

@pytest.mark.parametrize("code,ver,expected", [
    ("J960", 10, "respiratory"),
    ("I2101", 10, "circulatory"),
    ("N179", 10, "genitourinary"),
    ("T2131XA", 10, "injury/poisoning"),
    ("486", 9, "respiratory"),
    ("41401", 9, "circulatory"),
    ("5849", 9, "genitourinary"),
    ("94100", 9, "injury/poisoning"),
    ("E8790", 9, "unknown"),          # ICD-9 supplementary codes are not chapters
    (None, 10, "unknown"),
])
def test_icd_chapter(code, ver, expected):
    assert icd_chapter(code, ver) == expected


# --- mapping completeness ----------------------------------------------------

def test_unmapped_census_raises_above_threshold():
    values = pd.Series(["MED"] * 90 + ["NOT_A_SERVICE"] * 10)
    with pytest.raises(ValueError, match="unmapped"):
        unmapped_census(values, SERVICE_GROUP, "curr_service", max_frac=0.01)


def test_unmapped_census_tolerates_rare_values():
    values = pd.Series(["MED"] * 999 + ["NOT_A_SERVICE"])
    frac = unmapped_census(values, SERVICE_GROUP, "curr_service", max_frac=0.01)
    assert frac == pytest.approx(0.001)


def test_every_mapped_unit_has_a_group():
    assert set(CAREUNIT_GROUP.values()) <= {
        "cardiac", "neuro", "medical", "med-surg", "surgical", "trauma",
        "unspecified"}
    assert all(isinstance(v, str) and v for v in SERVICE_GROUP.values())


# --- the join ----------------------------------------------------------------

def test_assign_strata_preserves_row_count_and_partitions():
    meta = _meta([
        {"first_careunit": "Coronary Care Unit (CCU)", "curr_service": "CMED"},
        {"first_careunit": "Neuro Stepdown", "curr_service": "NSURG"},
        {"first_careunit": "Medical Intensive Care Unit (MICU)",
         "curr_service": "GU", "dx1_code": "N390"},
        {"first_careunit": "Surgical Intensive Care Unit (SICU)",
         "curr_service": "PSURG", "dx_burn": 1, "dx1_code": "T3110"},
        {"first_careunit": "Trauma SICU (TSICU)", "curr_service": "TRAUM",
         "dx1_code": "J189"},
    ])
    meta["stay_id"] = np.arange(len(meta))
    stays = pd.DataFrame({"stay_id": np.arange(len(meta)),
                          "patient_id": np.arange(len(meta))})

    out = assign_strata(stays, meta)
    assert len(out) == len(stays)
    assert list(out["unit_group"]) == ["cardiac", "neuro", "medical",
                                       "surgical", "trauma"]
    # service is the primary evidence, so GU wins over the MICU it sat in
    assert list(out["clinician_category"]) == [
        "cardio", "neuro", "urology", "burns", "respiratory"]


def test_assign_strata_rejects_missing_metadata():
    meta = _meta([{}])
    meta["stay_id"] = [0]
    stays = pd.DataFrame({"stay_id": [0, 99]})
    with pytest.raises(ValueError, match="no metadata row"):
        assign_strata(stays, meta)


def test_respiratory_only_reachable_by_diagnosis():
    """No MIMIC-IV unit or service is respiratory -- so nothing else may assign it."""
    from icu_strata import SERVICE_TO_CLINICIAN, UNIT_TO_CLINICIAN
    assert "respiratory" not in SERVICE_TO_CLINICIAN.values()
    assert "respiratory" not in UNIT_TO_CLINICIAN.values()


def test_category_coverage_reports_every_category():
    meta = _meta([{"first_careunit": "Coronary Care Unit (CCU)",
                   "curr_service": "CMED"}])
    meta["stay_id"] = [0]
    out = assign_strata(pd.DataFrame({"stay_id": [0]}), meta)
    cov = category_coverage(out)
    assert list(cov["category"]) == list(CLINICIAN_CATEGORIES)
    assert cov.loc[cov.category == "cardio", "n_assigned"].item() == 1
    assert cov.loc[cov.category == "burns", "n_assigned"].item() == 0


# --- subgroup metrics --------------------------------------------------------

def test_stratum_metrics_returns_none_when_underpowered():
    y = np.array([0] * 100 + [1] * 3)
    scores = {"m": np.linspace(0, 1, len(y))}
    out = stratum_metrics(y, scores, np.arange(len(y)),
                          np.ones(len(y), bool), reference="m", reps=20)
    assert out is None


def test_stratum_metrics_scores_only_the_masked_rows():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(200), np.ones(60)])
    good = np.concatenate([rng.uniform(0, .5, 200), rng.uniform(.5, 1, 60)])
    mask = np.ones(len(y), bool)
    out = stratum_metrics(y, {"good": good}, np.arange(len(y)), mask,
                          reference="good", reps=50)
    lo, hi = out["good"]["auroc_ci"]
    assert lo > 0.9 and hi <= 1.0

    # A mask that keeps only negatives has no positives left -> under-powered.
    neg_only = np.zeros(len(y), bool)
    neg_only[:200] = True
    assert stratum_metrics(y, {"good": good}, np.arange(len(y)), neg_only,
                           reference="good", reps=20) is None
