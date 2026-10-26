"""Tests for eICU site adaptation: the split and the warm-start path.

The split tests guard the claim the fine-tuning results rest on: eICU test
hospitals and patients are never seen while fine-tuning. The warm-start tests
guard the other half: fine-tuning really starts from the source weights and
keeps the source normaliser, rather than silently retraining from scratch.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from agents.vitals_tcn import VitalsTCNAgent
from eicu_finetune import DEFAULTS, assign_splits, load_settings
from mimic_cohort import VITALS

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


# --- splits -------------------------------------------------------------

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


# --- settings -----------------------------------------------------------

def test_settings_file_overrides_defaults_and_ignores_comments(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"_note": "x", "agent_ft_lr": 0.5}))
    s = load_settings(p)
    assert s["agent_ft_lr"] == 0.5
    assert s["split_by"] == DEFAULTS["split_by"]


def test_settings_file_rejects_unknown_keys(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"agent_ft_rl": 0.5}))
    with pytest.raises(KeyError):
        load_settings(p)


def test_shipped_settings_file_loads():
    from pathlib import Path
    s = load_settings(Path(__file__).resolve().parent.parent
                      / "configs" / "eicu_finetune.json")
    assert set(s) == set(DEFAULTS)


# --- warm start ---------------------------------------------------------

def _cohort(n_stays=6, n_hours=12, seed=0, shift=0.0):
    """Synthetic vitals cohort; ``shift`` moves heart rate (a 'new site')."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stays):
        bad = s % 2 == 0
        for h in range(n_hours):
            row = {"stay_id": 1000 + s, "patient_id": 500 + s, "hour": h,
                   "heart_rate": 70 + shift + (40 * h / n_hours if bad else 0)
                   + rng.normal(0, 2),
                   "resp_rate": 16 + rng.normal(0, 1),
                   "spo2": 97 + rng.normal(0, 1),
                   "sbp": 120 + rng.normal(0, 3),
                   "temperature": 37 + rng.normal(0, 0.2),
                   "gcs": 15.0, "gcs_unable": 0.0, "on_oxygen": float(bad),
                   LABEL: int(bad and h >= n_hours - 4)}
            for v in VITALS:
                row[f"{v}_hrs_since"] = 1.0
            rows.append(row)
    return pd.DataFrame(rows)


def _state(agent):
    return {k: v.clone() for k, v in agent.net.state_dict().items()}


def test_finetune_keeps_source_normaliser_and_restores_hparams():
    agent = VitalsTCNAgent(channels=8, max_epochs=1, batch_size=4, verbose=False)
    agent.fit(_cohort(seed=0), _cohort(seed=1))
    stats = dict(agent.stats)
    agent.finetune(_cohort(seed=2, shift=30), _cohort(seed=3, shift=30),
                   lr=1e-2, max_epochs=2)
    assert agent.stats == stats
    assert agent.lr == 1e-3 and agent.max_epochs == 1
    assert agent.warm_start is False


def test_finetune_starts_from_the_fitted_weights():
    agent = VitalsTCNAgent(channels=8, max_epochs=1, batch_size=4, verbose=False)
    agent.fit(_cohort(seed=0), _cohort(seed=1))
    before = _state(agent)
    # lr=0: training cannot move the weights, so if fine-tuning rebuilt the
    # net from scratch the weights would differ from the fitted ones.
    agent.finetune(_cohort(seed=2, shift=30), _cohort(seed=3, shift=30), lr=0.0)
    for k, v in agent.net.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_finetune_refits_calibrator_on_target_val():
    agent = VitalsTCNAgent(channels=8, max_epochs=1, batch_size=4, verbose=False)
    agent.fit(_cohort(seed=0), _cohort(seed=1))
    source_cal = agent.calibrator
    target_val = _cohort(seed=3, shift=30)
    agent.finetune(_cohort(seed=2, shift=30), target_val, lr=0.0)
    assert agent.calibrator is not source_cal
    assert agent.calibration_summary["split_rows"] == len(target_val)


def test_finetune_before_fit_raises():
    agent = VitalsTCNAgent(channels=8, verbose=False)
    with pytest.raises(RuntimeError):
        agent.finetune(_cohort(), _cohort(seed=1))


def test_orchestrator_warm_start_keeps_weights_at_zero_lr():
    from orchestrator.fusion import AgentBank, Orchestrator
    rng = np.random.default_rng(0)
    stay_hours = {i: list(range(6, 18)) for i in range(1, 7)}
    keys = pd.DataFrame([(s, h) for s, hs in stay_hours.items() for h in hs],
                        columns=["stay_id", "hour"])
    dims = {"a1": 4, "a2": 3}
    bank = AgentBank(keys, rng.normal(size=(len(keys), 9)).astype(np.float32),
                     dims, ("a1", "a2"))

    def cohort(seed):
        r = np.random.default_rng(seed)
        out = keys.copy()
        out["patient_id"] = 100 + out["stay_id"]
        out[LABEL] = (r.random(len(out)) < 0.3).astype(int)
        return out

    train, val = cohort(1), cohort(2)
    orch = Orchestrator(bank=bank, d_model=8, dilations=(1, 2), max_epochs=1,
                        batch_size=4, verbose=False).fit(train, val)
    before = _state(orch)
    orch.finetune(train, val, lr=0.0)
    for k, v in orch.net.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_gbm_continued_boosting_adds_trees(monkeypatch):
    import distill_lead
    monkeypatch.setattr(distill_lead, "_fair_record",
                        lambda: {"max_depth": 3, "scale_pos_weight": 1.0})
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 5)), columns=list("abcde"))
    y = (X["a"] + rng.normal(0, 0.5, 400) > 1).astype(float).to_numpy()
    base = distill_lead.fit_gbm_student(X[:200], y[:200], np.ones(200),
                                        X[200:300], y[200:300], device="cpu",
                                        num_boost_round=20)
    n0 = base.num_boosted_rounds()
    more = distill_lead.fit_gbm_student(X[300:], y[300:], np.ones(100),
                                        X[200:300], y[200:300], device="cpu",
                                        init_model=base.copy(), eta=0.02,
                                        num_boost_round=10)
    assert base.num_boosted_rounds() == n0          # source booster untouched
    assert more.num_boosted_rounds() > n0
    assert more.best_iteration >= n0 - 1            # absolute, not relative
    preds = distill_lead.gbm_predict(more, X[200:300])
    assert preds.shape == (100,)
