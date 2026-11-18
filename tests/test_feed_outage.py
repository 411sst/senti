"""Tests for the T42 feed-outage machinery (src/feed_outage.py).

A small synthetic world: two stays with raw cohort channels and every hourly
source table, run through the unchanged builders (``mimic_cohort`` steps and
``allfeat_gbm.build_allfeat_design``). Ids are small integers.
"""

import numpy as np
import pandas as pd
import pytest

import feed_outage as fo
from allfeat_gbm import build_allfeat_design, column_sets
from measurement_timing import measurement_timing_columns
from modality_features import (CADENCE_CHANNELS, FLUID_CHANNELS, LAB_COLUMNS,
                               RESP_LEVEL_MAP)

STAYS = (101, 102)
LAST_HOUR = 40
ECG_NAMES = ["qrs", "pr", "qt"]


def _world(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k, stay in enumerate(STAYS):
        event = 38.5 if stay == 102 else np.nan
        for h in range(LAST_HOUR + 1):
            if not np.isnan(event) and h >= event:
                break
            def obs(p, lo, hi):
                return float(rng.uniform(lo, hi)) if rng.random() < p else np.nan
            rows.append({
                "stay_id": stay, "subject_id": 11 + k, "hour": h,
                "is_history": int(h < 6),
                "heart_rate": obs(.8, 50, 140), "resp_rate": obs(.8, 8, 30),
                "spo2": obs(.8, 85, 100), "sbp": obs(.8, 80, 180),
                "temperature": obs(.5, 35, 39.5),
                "gcs": obs(.3, 3, 15), "gcs_verbal": obs(.3, 1, 5),
                "gcs_motor": obs(.3, 1, 6), "gcs_eyes": obs(.3, 1, 4),
                "gcs_unable": float(rng.random() < .1) if rng.random() < .3 else np.nan,
                "o2_flow": obs(.3, 0, 10),
                "o2_device": float(rng.random() < .5) if rng.random() < .3 else np.nan,
                "fio2": obs(.2, 21, 80),
                "event_h": event, "death_h": np.nan,
                "vent_h": event, "vaso_h": np.nan,
                "label_24h": int(not np.isnan(event) and h < event <= h + 24),
            })
    raw = pd.DataFrame(rows)

    def table(columns, p, lo=-24, gen=None):
        out = []
        for stay in STAYS:
            for h in range(lo, LAST_HOUR + 1):
                if rng.random() < p:
                    row = {"stay_id": stay, "hour": h}
                    row.update(gen() if gen else
                               {c: float(rng.uniform(1, 9)) for c in columns})
                    out.append(row)
        return pd.DataFrame(out)

    labs = table(LAB_COLUMNS, .3)
    for c in LAB_COLUMNS:                                    # sparse analytes
        labs.loc[rng.random(len(labs)) < .5, c] = np.nan
    cadence = table(CADENCE_CHANNELS, .7,
                    gen=lambda: {c: float(rng.integers(0, 4)) for c in CADENCE_CHANNELS})
    fluid = table(FLUID_CHANNELS, .6,
                  gen=lambda: {"urine_ml": float(rng.uniform(0, 200)),
                               "n_urine_records": float(rng.integers(0, 3)),
                               "other_out_ml": float(rng.uniform(0, 50)),
                               "in_ml": float(rng.uniform(0, 300)),
                               "n_in_records": float(rng.integers(0, 3))})
    devices = list(RESP_LEVEL_MAP) + ["Endotracheal tube"]
    resp = table(None, .4, gen=lambda: {
        "o2_device": devices[rng.integers(len(devices))],
        "o2_flow": float(rng.uniform(0, 15)), "fio2": float(rng.uniform(21, 90))})
    ecg_index = pd.DataFrame({
        "stay_id": [101, 101, 101, 102, 102],
        "ecg_hour": [-3.5, 9.25, 20.0, 4.0, 25.5],
        "study_id": [1, 2, 3, 4, 5], "path": ["x"] * 5})
    ecg = {"values": rng.normal(size=(5, len(ECG_NAMES))).astype(np.float32),
           "study_ids": np.arange(1, 6), "names": ECG_NAMES}
    sources = {"cohort_raw": raw, "labs": labs, "ecg_index": ecg_index,
               "cadence": cadence, "fluid": fluid, "resp": resp}
    return sources, ecg


def _design(sources, ecg):
    cohort = fo.cohort_from_raw(sources["cohort_raw"])
    af = {"labs": sources["labs"], "cadence": sources["cadence"],
          "fluid": sources["fluid"], "resp": sources["resp"],
          "ecg": {"index": sources["ecg_index"], **ecg}}
    X, y, keys = build_allfeat_design(cohort, af)
    return X, keys


def _same(a, b):
    return np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                          equal_nan=True)


# --- the mapping ---------------------------------------------------------------

def test_feed_columns_partition_the_262_columns():
    names = [f"m{i}" for i in range(51)]
    full = column_sets(names)["allfeat"]
    per = {f: fo.feed_columns(f, names) for f in fo.FEEDS}
    assert {f: len(c) for f, c in per.items()} == {
        "vitals": 78, "labs": 72, "ecg": 53, "cadence": 32, "fluid": 13, "resp": 14}
    union = [c for f in fo.FEEDS for c in per[f]]
    assert len(union) == len(set(union)) == 262 and set(union) == set(full)
    timing = {f: len(measurement_timing_columns(c)) for f, c in per.items()}
    assert timing == {"vitals": 5, "labs": 24, "ecg": 2, "cadence": 0,
                      "fluid": 3, "resp": 2}


def test_fallbacks_drop_every_agent_that_reads_the_feed():
    assert fo.fallback_agents("vitals") == ("labs", "ecg_meas", "cadence",
                                            "fluid", "resp")
    assert fo.fallback_agents("resp") == ("labs", "ecg_meas", "cadence", "fluid")
    assert fo.fallback_agents("labs") == ("vitals_tcn", "news2", "ecg_meas",
                                          "cadence", "fluid", "resp")
    for f in ("ecg", "cadence", "fluid"):
        assert len(fo.fallback_agents(f)) == 7
    names = [f"m{i}" for i in range(51)]
    full = column_sets(names)["allfeat"]
    assert len(fo.fallback_columns("vitals", full, names)) == 262 - 78


# --- deleting records ----------------------------------------------------------

def test_window_outage_deletes_exactly_the_window_records():
    sources, _ = _world()
    starts = pd.Series({101: 10})
    out = fo.apply_outage(sources, "labs", [101], starts)
    labs = sources["labs"]
    gone = (labs["stay_id"] == 101) & labs["hour"].between(10, 33)
    assert len(out["labs"]) == len(labs) - int(gone.sum())
    kept = out["labs"]
    assert not ((kept["stay_id"] == 101) & kept["hour"].between(10, 33)).any()
    pd.testing.assert_frame_equal(kept, labs.loc[~gone])
    ecg = fo.apply_outage(sources, "ecg", [101], starts)["ecg_index"]
    assert ecg["study_id"].tolist() == [1, 2, 4, 5]          # 20.0 h is in [10, 34)
    ecg = fo.apply_outage(sources, "ecg", [101], pd.Series({101: 9}))["ecg_index"]
    assert ecg["study_id"].tolist() == [1, 4, 5]             # 9.25 and 20.0 deleted


def test_cohort_channels_are_blanked_not_deleted():
    sources, _ = _world()
    raw = sources["cohort_raw"]
    out = fo.apply_outage(sources, "resp", [101])["cohort_raw"]
    assert len(out) == len(raw)
    s = (out["stay_id"] == 101).to_numpy()
    assert out.loc[s, ["o2_flow", "o2_device", "fio2"]].isna().all().all()
    assert _same(out.loc[s, "heart_rate"], raw.loc[s, "heart_rate"])
    assert _same(out.loc[~s, "o2_flow"], raw.loc[~s, "o2_flow"])


@pytest.mark.parametrize("feed", fo.FEEDS)
@pytest.mark.parametrize("starts", [None, pd.Series({101: 12, 102: 8})])
def test_an_outage_leaves_every_other_feed_byte_identical(feed, starts):
    sources, ecg = _world()
    out = fo.apply_outage(sources, feed, list(STAYS), starts)
    owned = set()
    if feed in fo.FEED_TABLES:
        owned.add(fo.FEED_TABLES[feed])
    if feed in fo.COHORT_CHANNELS:
        owned.add("cohort_raw")
    for name in fo.SOURCE_NAMES:
        if name not in owned:
            assert out[name] is sources[name]               # untouched object
    X0, k0 = _design(sources, ecg)
    X1, k1 = _design(out, ecg)
    assert k0.equals(k1)
    mine = set(fo.feed_columns(feed, ECG_NAMES))
    for c in X0.columns:
        if c not in mine:
            assert _same(X0[c], X1[c]), f"{feed} outage changed {c}"
    assert any(not _same(X0[c], X1[c]) for c in mine), f"{feed} outage changed nothing"


def test_a_whole_stay_vitals_outage_reads_as_missing_vitals():
    sources, ecg = _world()
    X, keys = _design(fo.apply_outage(sources, "vitals", [101]), ecg)
    s = (keys["stay_id"] == 101).to_numpy()
    for c in ("heart_rate", "spo2", "heart_rate_hrs_since", "gcs"):
        assert X.loc[s, c].isna().all()
    for c in ("news2_heart_rate", "news2_spo2", "news2_consciousness"):
        assert (X.loc[s, c] == 0).all()


def test_a_whole_stay_resp_outage_reads_as_room_air():
    sources, ecg = _world()
    X, keys = _design(fo.apply_outage(sources, "resp", [101]), ecg)
    s = (keys["stay_id"] == 101).to_numpy()
    assert (X.loc[s, "resp_level"] == 0).all() and (X.loc[s, "resp_fio2"] == 21).all()
    assert (X.loc[s, "resp_seen"] == 0).all() and (X.loc[s, "on_oxygen"] == 0).all()
    assert (X.loc[s, "news2_oxygen"] == 0).all()


# --- time ------------------------------------------------------------------------

def _rewrite_from(sources, t0, rng):
    """Every record at or after t0 rewritten: values scrambled, some deleted."""
    out = {}
    for name, tab in sources.items():
        tcol = fo.TIME_COLUMN.get(name, "hour")
        late = tab[tcol].to_numpy(dtype=float) >= t0
        tab = tab.copy()
        if name == "cohort_raw":
            for c in fo.COHORT_CHANNELS["vitals"] + fo.COHORT_CHANNELS["resp"]:
                v = tab[c].to_numpy(dtype=float)
                tab[c] = np.where(late, rng.uniform(0, 200, len(tab)), v)
        elif name == "ecg_index":
            tab = tab[~late | (rng.random(len(tab)) < .5)]
        else:
            num = [c for c in tab.columns if c not in ("stay_id", "hour", "o2_device")]
            for c in num:
                tab.loc[late, c] = rng.uniform(0, 50, int(late.sum()))
            tab = tab[~late | (rng.random(len(tab)) < .7)]
        out[name] = tab.reset_index(drop=True)
    return out


@pytest.mark.parametrize("t0", [6, 15, 30])
def test_no_feature_at_hour_t_uses_records_at_or_after_t(t0):
    sources, ecg = _world()
    X0, k0 = _design(sources, ecg)
    X1, k1 = _design(_rewrite_from(sources, t0, np.random.default_rng(t0)), ecg)
    early = (k0["hour"] <= t0).to_numpy()
    for c in X0.columns:
        assert _same(X0.loc[early, c], X1.loc[early, c]), c


def test_rows_up_to_the_window_start_are_untouched_and_the_next_hour_is_not():
    sources, ecg = _world()
    starts = pd.Series({101: 15, 102: 12})
    X0, k0 = _design(sources, ecg)
    for feed in fo.FEEDS:
        X1, _ = _design(fo.apply_outage(sources, feed, list(STAYS), starts), ecg)
        before = fo.before_window_rows(k0, starts)
        for c in X0.columns:
            assert _same(X0.loc[before, c], X1.loc[before, c]), (feed, c)
    # the first vital charted at the window start disappears from the next hour
    raw = sources["cohort_raw"]
    s101 = raw[(raw["stay_id"] == 101) & (raw["hour"] >= 10)]
    h0 = int(s101.loc[s101["heart_rate"].notna(), "hour"].iloc[0])
    starts = pd.Series({101: h0, 102: 12})
    X1, _ = _design(fo.apply_outage(sources, "vitals", list(STAYS), starts), ecg)
    row = ((k0["stay_id"] == 101) & (k0["hour"] == h0 + 1)).to_numpy()
    assert X0.loc[row, "heart_rate_hrs_since"].item() == 1
    assert not X1.loc[row, "heart_rate_hrs_since"].item() == 1


# --- windows -----------------------------------------------------------------------

def _keys(n_stays=300, hours=(6, 7, 8)):
    return pd.DataFrame([(s, h) for s in range(1, n_stays + 1) for h in hours],
                        columns=["stay_id", "hour"]).sample(frac=1, random_state=0)


def test_draw_windows_is_seeded_and_uniform_over_scored_hours():
    keys = _keys()
    a, b = fo.draw_windows(keys, 42), fo.draw_windows(keys, 42)
    assert a.equals(b) and not a.equals(fo.draw_windows(keys, 7))
    assert set(a.unique()) <= {6, 7, 8} and len(a) == 300
    share = a.value_counts(normalize=True)
    assert (share - 1 / 3).abs().max() < 0.08


def test_window_rows_are_the_24_hours_after_the_start():
    keys = pd.DataFrame({"stay_id": [1] * 40, "hour": np.arange(6, 46)})
    starts = pd.Series({1: 10})
    rows = fo.window_rows(keys, starts)
    assert keys["hour"][rows].tolist() == list(range(11, 35))
    assert keys["hour"][fo.before_window_rows(keys, starts)].tolist() == list(range(6, 11))


# --- pipeline pieces -------------------------------------------------------------

def test_cohort_from_raw_matches_the_cohort_builder(tmp_path):
    from mimic_cohort import build
    sources, _ = _world()
    raw_path, out_path = tmp_path / "raw.parquet", tmp_path / "out.parquet"
    sources["cohort_raw"].to_parquet(raw_path)
    built = build(raw_path=raw_path, out_path=out_path)
    mine = fo.cohort_from_raw(sources["cohort_raw"].copy())
    order = ["stay_id", "hour"]
    built = built.sort_values(order).reset_index(drop=True)
    mine = mine.sort_values(order).reset_index(drop=True)
    assert fo.frame_differences(mine, built, fo.COHORT_KEEP) == []


def test_frame_differences_names_the_changed_column():
    a = pd.DataFrame({"x": [1.0, np.nan], "t": ["a", None]})
    b = pd.DataFrame({"x": [1.0, np.nan], "t": ["a", pd.NA]})
    assert fo.frame_differences(a, b, ["x", "t"]) == []
    b.loc[1, "x"] = 2.0
    assert fo.frame_differences(a, b, ["x", "t"]) == ["x"]


def test_bank_from_outputs_matches_agentbank_load(tmp_path):
    from agents.base import AgentOutput
    from orchestrator.fusion import AgentBank
    keys = pd.DataFrame({"stay_id": [7, 7, 8], "hour": [6, 7, 6]})
    rng = np.random.default_rng(0)
    outs = {n: AgentOutput(keys=keys.copy(),
                           latent=rng.normal(size=(3, d)).astype(np.float32),
                           risk=rng.random(3).astype(np.float32))
            for n, d in (("a", 2), ("b", 3))}
    for n, o in outs.items():
        o.save(tmp_path / f"{n}_test.npz")
    ref = AgentBank.load(splits=("test",), agent_names=["a", "b"], emb_dir=tmp_path)
    mine = fo.bank_from_outputs(outs, ["a", "b"])
    assert np.array_equal(ref.matrix, mine.matrix)
    assert ref.latent_dims == mine.latent_dims
    assert (ref.keys.to_numpy() == mine.keys.to_numpy()).all()


# --- verdicts and summaries ------------------------------------------------------

def test_verdict_rules():
    assert fo.robustness_verdict([0.001, 0.02]) == "Sentinel more robust"
    assert fo.robustness_verdict([-0.02, -0.001]) == "GBM more robust"
    assert fo.robustness_verdict([-0.01, 0.01]) == "tie"
    assert fo.absolute_verdict([0.001, 0.02]) == "Sentinel better during this outage"
    assert fo.absolute_verdict([-0.01, 0.01]) == "tie"
    v = dict(zip(fo.FEEDS, ["Sentinel more robust"] * 3 + ["tie"] * 3))
    assert fo.headline_pass(v) == (True, 3, 0)
    v["fluid"] = "GBM more robust"
    assert fo.headline_pass(v) == (False, 3, 1)
    v = dict(zip(fo.FEEDS, ["Sentinel more robust"] * 2 + ["tie"] * 4))
    assert fo.headline_pass(v)[0] is False


def test_alarm_labels_flag_both_directions():
    assert fo.alarm_label(0.05, 0.02) == "flood"
    assert fo.alarm_label(0.009, 0.02) == "silence"
    assert fo.alarm_label(0.021, 0.02) == "within 2x"


def test_window_catch_counts_stays_with_a_positive_window_hour():
    frame = pd.DataFrame({"stay_id": [1, 1, 2, 2, 3, 3]})
    y = [0, 1, 1, 1, 0, 0]               # stay 3 has no positive window hour
    s = [0.9, 0.1, 0.1, 0.2, 0.9, 0.9]
    catch, n = fo.window_catch(frame, y, s, thr=0.5)
    assert n == 2 and catch == 0.5       # stay 1 alarmed, stay 2 not


def test_noisy_or_and_calibration_keep_order():
    from sklearn.isotonic import IsotonicRegression
    assert np.allclose(fo.noisy_or([0.5, 0.0], [0.5, 0.2]), [0.75, 0.2])
    raw = np.array([0.1, 0.2, 0.2, 0.3, 0.9])
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw, [0, 0, 1, 0, 1])
    c = fo.calibrated(iso, raw)
    assert (np.diff(c[np.argsort(raw, kind="stable")]) >= 0).all()
