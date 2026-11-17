"""Tests for the eICU hospital-level split used by the operating-point tests.

``src/eicu_finetune.py`` is taken unchanged from ``origin/eicu-finetune``
(9e19973). Only its split is needed here, so only the split tests are carried
over; the warm-start tests in that branch's ``tests/test_eicu_finetune.py``
need agent code that is not on this branch.

The split guards the claim the eICU results rest on: test hospitals, and test
patients, are never seen in the train or validation hospitals. Every row is
synthetic.
"""

import numpy as np
import pandas as pd
import pytest

from eicu_finetune import assign_splits

LABEL = "label_24h"


def _eicu(n_hospitals=20, stays_per=5, hours=4, seed=0):
    """Hourly rows over hospitals; one patient per stay unless overridden."""
    rng = np.random.default_rng(seed)
    rows = []
    stay = 0
    for h in range(n_hospitals):
        for _ in range(stays_per + h % 3):
            stay += 1
            for hour in range(hours):
                rows.append({"stay_id": stay, "patient_id": f"p{stay}",
                             "hospitalid": 100 + h, "hour": hour,
                             LABEL: int(rng.random() < 0.1), "split": "test"})
    return pd.DataFrame(rows)


def test_hospital_split_keeps_hospitals_disjoint():
    df, report = assign_splits(_eicu(), by="hospital")
    hosp = {s: set(df.loc[df.split == s, "hospitalid"]) for s in ("train", "val", "test")}
    assert not hosp["train"] & hosp["val"]
    assert not hosp["train"] & hosp["test"]
    assert not hosp["val"] & hosp["test"]
    assert all(report[s]["hours"] > 0 for s in ("train", "val", "test"))


def test_hospital_split_roughly_follows_fractions():
    df, _ = assign_splits(_eicu(n_hospitals=200), by="hospital", fracs=(0.6, 0.2, 0.2))
    share = df.drop_duplicates("stay_id").split.value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.6, abs=0.05)
    assert share["test"] == pytest.approx(0.2, abs=0.05)


def test_patient_seen_in_train_is_dropped_from_later_splits():
    df = _eicu()
    df0, _ = assign_splits(df, by="hospital")
    train_h = df0.loc[df0.split == "train", "hospitalid"].iloc[0]
    test_h = df0.loc[df0.split == "test", "hospitalid"].iloc[0]
    # One patient with a stay in a train hospital and another in a test one.
    a = df.loc[df.hospitalid == train_h, "stay_id"].iloc[0]
    b = df.loc[df.hospitalid == test_h, "stay_id"].iloc[0]
    df.loc[df.stay_id.isin([a, b]), "patient_id"] = "shared"

    out, report = assign_splits(df, by="hospital")
    assert set(out.loc[out.patient_id == "shared", "split"]) == {"train"}
    assert report["dropped_overlap_patients"]["test"] == 1
    for s, t in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (set(out.loc[out.split == s, "patient_id"])
                    & set(out.loc[out.split == t, "patient_id"]))


def test_patient_split_is_patient_disjoint_and_deterministic():
    a, _ = assign_splits(_eicu(), by="patient", seed=7)
    b, _ = assign_splits(_eicu(), by="patient", seed=7)
    assert a.split.tolist() == b.split.tolist()
    per_patient = a.groupby("patient_id").split.nunique()
    assert (per_patient == 1).all()


def test_train_patients_subsamples_train_only():
    full, _ = assign_splits(_eicu(), by="hospital")
    small, report = assign_splits(_eicu(), by="hospital", train_patients=5)
    assert report["train"]["patients"] == 5
    for s in ("val", "test"):
        assert len(small[small.split == s]) == len(full[full.split == s])


def test_bad_split_arguments_raise():
    with pytest.raises(ValueError):
        assign_splits(_eicu(), by="unit")
    with pytest.raises(ValueError):
        assign_splits(_eicu(), fracs=(0.5, 0.5, 0.5))
