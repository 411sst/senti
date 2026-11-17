"""Tests for the all-features GBM design (T30) and the shared paired bootstrap.

The load-bearing test is the leakage one: rewriting every source event at hour
>= t0, and every cohort row after t0, must leave every feature row at hour
<= t0 unchanged. A boundary check proves each block DOES see hour t0 - 1, so
the leakage test is not passing because a block ignores its source.

Every row is synthetic; ids are small integers.
"""

import numpy as np
import pandas as pd
import pytest

from allfeat_gbm import (
    NEWS2_SUBSCORES,
    build_allfeat_design,
    column_sets,
    ecg_columns,
)
from baselines_fair import variant_columns
from fusion_common import paired_auprc_draws
from mimic_cohort import VITALS, compute_news2
from modality_features import (
    CADENCE_CHANNELS,
    CADENCE_FEATURES,
    FLUID_FEATURES,
    LAB_COLUMNS,
    RESP_FEATURES,
)
from paper_stats import metric_draws, paired_delta, pooled_delta

LABEL = "label_24h"
T0 = 15
ECG_NAMES = ["hr", "qrs_dur", "qtc_bazett"]


def _cohort(stays=(1, 2, 3), hours=range(6, 31), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in stays:
        for h in hours:
            row = {"stay_id": s, "patient_id": 100 + s, "hour": h,
                   LABEL: int(h > 26 and s == 1),
                   "gcs": float(rng.integers(9, 16)), "gcs_unable": 0.0,
                   "is_alert": float(rng.random() < 0.9),
                   "on_oxygen": float(rng.random() < 0.3)}
            for v, (lo, hi) in zip(VITALS, ((50, 130), (8, 30), (88, 100),
                                            (85, 180), (35.5, 39.5))):
                row[v] = float(rng.uniform(lo, hi))
                row[f"{v}_hrs_since"] = float(rng.integers(0, 3))
            rows.append(row)
    return pd.DataFrame(rows)


def _sources(stays=(1, 2, 3), seed=1):
    rng = np.random.default_rng(seed)
    hours = range(-4, 31)
    labs = pd.DataFrame([{"stay_id": s, "hour": h,
                          **{c: (float(rng.uniform(1, 10)) if rng.random() < 0.3
                                 else np.nan) for c in LAB_COLUMNS}}
                         for s in stays for h in hours])
    cadence = pd.DataFrame([{"stay_id": s, "hour": h,
                             **{c: int(rng.integers(0, 4)) for c in CADENCE_CHANNELS}}
                            for s in stays for h in hours])
    fluid = pd.DataFrame([{"stay_id": s, "hour": h,
                           "urine_ml": float(rng.integers(0, 120)),
                           "n_urine_records": 1, "other_out_ml": 0.0,
                           "in_ml": float(rng.integers(0, 300)),
                           "n_in_records": 1}
                          for s in stays for h in hours])
    devices = ["None", "Nasal cannula", "Venti mask", "High flow nasal cannula"]
    resp = pd.DataFrame([{"stay_id": s, "hour": h,
                          "o2_device": devices[int(rng.integers(0, 4))],
                          "o2_flow": float(rng.integers(0, 40)),
                          "fio2": float(rng.integers(21, 80))}
                         for s in stays for h in range(-2, 31, 3)])
    ecg_rows, study = [], 0
    for s in stays:
        for h in (4.5, 9.0, 13.25, 14.0, 15.0, 18.5, 24.0):
            study += 1
            ecg_rows.append({"stay_id": s, "ecg_hour": h, "study_id": 900 + study})
    index = pd.DataFrame(ecg_rows)
    values = rng.normal(size=(len(index), len(ECG_NAMES))).astype(np.float32)
    ecg = {"index": index, "values": values,
           "study_ids": index["study_id"].to_numpy(), "names": list(ECG_NAMES)}
    return {"labs": labs, "cadence": cadence, "fluid": fluid, "resp": resp,
            "ecg": ecg}


def _copy(sources):
    out = {k: v.copy() for k, v in sources.items() if k != "ecg"}
    e = sources["ecg"]
    out["ecg"] = {"index": e["index"].copy(), "values": e["values"].copy(),
                  "study_ids": e["study_ids"].copy(), "names": list(e["names"])}
    return out


def _rewrite_from(sources, cohort, t):
    """Change every source event at hour >= t and every cohort row after t."""
    src = _copy(sources)
    rng = np.random.default_rng(99)
    for name in ("labs", "cadence", "fluid"):
        df = src[name]
        m = df["hour"] >= t
        for c in df.columns.drop(["stay_id", "hour"]):
            df.loc[m, c] = rng.integers(50, 500, m.sum())
    r = src["resp"]
    m = r["hour"] >= t
    r.loc[m, "o2_device"] = "Non-invasive ventilation"
    r.loc[m, "fio2"] = 95.0
    extra = pd.DataFrame([{"stay_id": 1, "hour": t, "o2_device": "Bipap mask",
                           "o2_flow": 60.0, "fio2": 100.0}])
    src["resp"] = pd.concat([r[~((r.stay_id == 1) & (r.hour == t))], extra],
                            ignore_index=True)
    e = src["ecg"]
    m = (e["index"]["ecg_hour"] >= t).to_numpy()
    e["values"][m] = 77.0
    new = pd.DataFrame([{"stay_id": 2, "ecg_hour": float(t), "study_id": 999}])
    e["index"] = pd.concat([e["index"], new], ignore_index=True)
    e["values"] = np.vstack([e["values"], np.full((1, e["values"].shape[1]), 55.0,
                                                  np.float32)])
    e["study_ids"] = e["index"]["study_id"].to_numpy()

    coh = cohort.copy()
    later = coh["hour"] > t
    for v in VITALS + ["gcs", "is_alert", "on_oxygen"]:
        coh.loc[later, v] = coh.loc[later, v] * 0.5 + 1.0
    return src, coh


def _past_rows(X, keys, t):
    return X[(keys["hour"] <= t).to_numpy()].reset_index(drop=True)


# --- leakage -----------------------------------------------------------------

def test_rewriting_the_future_never_moves_the_past():
    cohort, sources = _cohort(), _sources()
    X, _, keys = build_allfeat_design(cohort, sources)
    src2, coh2 = _rewrite_from(sources, cohort, T0)
    X2, _, keys2 = build_allfeat_design(coh2, src2)
    assert keys.equals(keys2)
    pd.testing.assert_frame_equal(_past_rows(X, keys, T0), _past_rows(X2, keys2, T0))
    # Not vacuous: the rewrite does reach the rows after t0.
    later = (keys["hour"] > T0).to_numpy()
    assert not X[later].equals(X2[later])


@pytest.mark.parametrize("block,cols", [
    ("labs", ["lab_creatinine", "lab_creatinine_hrs_since"]),
    ("cadence", list(CADENCE_FEATURES)),
    ("fluid", list(FLUID_FEATURES)),
    ("resp", list(RESP_FEATURES)),
    ("ecg", ecg_columns(ECG_NAMES)),
])
def test_each_block_first_sees_an_event_one_hour_later(block, cols):
    """An event in hour bucket t0 - 1 (an ECG at t0 - 0.5) moves row t0; an
    event AT t0 does not, which the leakage test covers. So the boundary is
    exactly strictly-before, and no block passes by ignoring its source."""
    cohort, sources = _cohort(), _sources()
    X, _, keys = build_allfeat_design(cohort, sources)
    src = _copy(sources)
    h = T0 - 1
    if block == "ecg":
        new = pd.DataFrame([{"stay_id": 1, "ecg_hour": T0 - 0.5, "study_id": 4242}])
        src["ecg"]["index"] = pd.concat([src["ecg"]["index"], new], ignore_index=True)
        src["ecg"]["values"] = np.vstack([src["ecg"]["values"],
                                          np.full((1, len(ECG_NAMES)), 33.0, np.float32)])
        src["ecg"]["study_ids"] = src["ecg"]["index"]["study_id"].to_numpy()
    elif block == "resp":
        r = src["resp"]
        src["resp"] = pd.concat(
            [r[~((r.stay_id == 1) & (r.hour == h))],
             pd.DataFrame([{"stay_id": 1, "hour": h, "o2_device": "Bipap mask",
                            "o2_flow": 60.0, "fio2": 100.0}])], ignore_index=True)
    else:
        df = src[block]
        m = (df["stay_id"] == 1) & (df["hour"] == h)
        for c in df.columns.drop(["stay_id", "hour"]):
            df.loc[m, c] = 250
    X2, _, _ = build_allfeat_design(cohort, src)
    row = ((keys["stay_id"] == 1) & (keys["hour"] == T0)).to_numpy()
    before = ((keys["stay_id"] == 1) & (keys["hour"] <= h)).to_numpy()
    assert not X.loc[row, cols].equals(X2.loc[row, cols])
    pd.testing.assert_frame_equal(X[before], X2[before])


# --- shape, order, content -----------------------------------------------------

def test_column_sets_full_and_no_cadence():
    sets = column_sets([f"m{i}" for i in range(51)])
    full, nocad = sets["allfeat"], sets["allfeat_nocad"]
    assert len(variant_columns()["FAIR+TRAJ"]) == 145
    assert len(full) == 145 + 32 + 13 + 12 + 53 + 7 == 262
    assert len(nocad) == 230
    assert set(full) - set(nocad) == set(CADENCE_FEATURES)
    assert full[:145] == list(variant_columns()["FAIR+TRAJ"])


def test_rows_are_sorted_and_independent_of_input_order():
    cohort, sources = _cohort(), _sources()
    X, y, keys = build_allfeat_design(cohort, sources)
    shuffled = cohort.sample(frac=1.0, random_state=3)
    Xs, ys, keys_s = build_allfeat_design(shuffled, sources)
    pd.testing.assert_frame_equal(X, Xs)
    assert np.array_equal(y, ys) and keys.equals(keys_s)
    assert keys.equals(keys.sort_values(["stay_id", "hour"]).reset_index(drop=True))
    expect = cohort.sort_values(["stay_id", "hour"])[LABEL].to_numpy(dtype=float)
    assert np.array_equal(y, expect)
    assert list(X.columns) == column_sets(ECG_NAMES)["allfeat"]
    assert (X.dtypes == np.float32).all()


def test_news2_subscores_sum_to_the_news2_total():
    cohort, sources = _cohort(), _sources()
    X, _, _ = build_allfeat_design(cohort, sources)
    total, _, _ = compute_news2(cohort.sort_values(["stay_id", "hour"]))
    assert np.allclose(X[NEWS2_SUBSCORES].sum(axis=1).to_numpy(), total)


def test_hours_without_a_recent_ecg_get_the_agents_zero_fill():
    cohort, sources = _cohort(), _sources()
    # Every stay has an ECG at 4.5 h, within 72 h of all its hours, so drop
    # stay 3's ECGs to get a stay with none.
    e = sources["ecg"]
    keep = (e["index"]["stay_id"] != 3).to_numpy()
    e["index"] = e["index"][keep].reset_index(drop=True)
    e["values"] = e["values"][keep]
    e["study_ids"] = e["index"]["study_id"].to_numpy()
    X, _, keys = build_allfeat_design(cohort, sources)
    none = (keys["stay_id"] == 3).to_numpy()
    cols = ecg_columns(ECG_NAMES)
    assert (X.loc[none, cols].to_numpy() == 0).all()
    assert (X.loc[~none, "ecg_has"] == 1).all()


# --- shared paired bootstrap ---------------------------------------------------

def _scores(n=600, seed=5):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n // 6), 6) + 100
    y = (rng.random(n) < 0.15).astype(float)
    a = y * 0.8 + rng.random(n)
    b = y * 0.4 + rng.random(n)
    return y, pid, {"a": a, "b": b}


def test_metric_draws_match_the_existing_auprc_draws_exactly():
    y, pid, scores = _scores()
    ours = metric_draws({"auprc": ("auprc", y)}, scores, pid, reps=40, seed=7)
    ref = paired_auprc_draws(y, scores, pid, reps=40, seed=7)
    for n in scores:
        assert np.array_equal(ours["auprc"][n], ref[n])


def test_metric_draws_do_not_depend_on_worker_count():
    y, pid, scores = _scores()
    y2 = np.roll(y, 3)
    targets = {"auprc": ("auprc", y), "auroc": ("auroc", y), "other": ("auprc", y2)}
    one = metric_draws(targets, scores, pid, reps=25, seed=3, workers=1)
    two = metric_draws(targets, scores, pid, reps=25, seed=3, workers=2)
    for t in targets:
        for n in scores:
            assert np.array_equal(one[t][n], two[t][n])


def test_paired_and_pooled_deltas():
    d = paired_delta(0.5, 0.3, [0.5, 0.6, 0.4], [0.3, 0.3, 0.3])
    assert d["delta"] == pytest.approx(0.2)
    assert d["ci_above_zero"]
    # Per resample: ((0.5-0.3) + (0.7-0.3))/2 = 0.3 and ((0.4-0.3) + (0.6-0.3))/2 = 0.2.
    p = pooled_delta([0.5, 0.7], 0.3, [np.array([0.5, 0.4]), np.array([0.7, 0.6])],
                     np.array([0.3, 0.3]))
    assert p["delta"] == pytest.approx(0.3)
    assert p["draws_mean"] == pytest.approx(0.25)
    assert p["ci"] == pytest.approx([0.2025, 0.2975])
    assert p["ci_above_zero"]


def test_measurement_timing_columns():
    from measurement_timing import MEASUREMENT_TIMING_EXTRA, measurement_timing_columns
    from mimic_cohort import VITALS
    cols = column_sets([f"m{i}" for i in range(51)])["allfeat_nocad"]
    timing = measurement_timing_columns(cols)
    assert len(timing) == len(VITALS) + len(LAB_COLUMNS) + len(MEASUREMENT_TIMING_EXTRA) == 36
    assert set(MEASUREMENT_TIMING_EXTRA) <= set(cols)
    assert timing == [c for c in cols if c in set(timing)]
    assert "lab_ph_hrs_since" in timing and "lab_ph" not in timing
    assert not set(timing) & set(CADENCE_FEATURES)


# --- paper_stats.pooled_race ------------------------------------------------------

def _race_frame():
    rows = [{"stay_id": s, "patient_id": 10 + s, "hour": h, "event_h": ev}
            for s, ev in ((1, 9.0), (2, 8.0), (3, 7.0), (4, None)) for h in range(10)]
    return pd.DataFrame(rows)


def _alarms(frame, pairs):
    keys = set(pairs)
    return np.array([1.0 if (s, h) in keys else 0.0
                     for s, h in zip(frame["stay_id"], frame["hour"])])


def test_pooled_race_counts_jointly_caught_stays_over_seeds():
    from paper_stats import event_first_alarms, pooled_race
    t = _race_frame()
    opp = _alarms(t, [(1, 5), (2, 2), (4, 1)])          # stay 3 never; 4 is quiet
    seed_a = _alarms(t, [(1, 3), (2, 2), (3, 4)])        # first, tie, (opp misses)
    seed_b = _alarms(t, [(1, 7), (2, 9)])                # second; stay 2 after event
    tab = event_first_alarms(t, opp, 0.5)
    assert tab["stay_id"].tolist() == [1, 2, 3]
    assert tab["first_alarm"].tolist()[:2] == [5.0, 2.0] and np.isnan(tab["first_alarm"][2])
    out = pooled_race(t, [seed_a, seed_b], [0.5, 0.5], opp, 0.5, reps=200)
    assert (out["n_pair"], out["model_first"], out["opponent_first"], out["ties"]) == (3, 1, 1, 1)
    assert out["share_when_differ"] == pytest.approx(0.5)
    assert out["median_lead_h"] == pytest.approx(0.0)
    assert [p["n_pair"] for p in out["per_seed"]] == [2, 1]
    assert 0.0 <= out["ci"][0] <= 0.5 <= out["ci"][1] <= 1.0
    # A per-seed opponent gives the same answer when the arrays are the same.
    same = pooled_race(t, [seed_a, seed_b], [0.5, 0.5], [opp, opp], [0.5, 0.5], reps=200)
    assert same["share_when_differ"] == out["share_when_differ"] and same["ci"] == out["ci"]
