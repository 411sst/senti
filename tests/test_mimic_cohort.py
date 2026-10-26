"""Unit tests for the MIMIC-IV in-ICU hourly cohort builder.

The most important tests here are the equivalence ones: ``mimic_cohort``
scores 2.5M rows with vectorised numpy, but the authoritative NEWS2 logic lives
in scalar form in :mod:`news2`. These tests assert the fast path reproduces the
scalar reference exactly, so it cannot silently drift.
"""

import math

import numpy as np
import pandas as pd
import pytest

from mimic_cohort import (
    BANDS,
    CARRIED,
    SCALAR_SCORERS,
    _band_score,
    add_event_labels,
    alert_flags,
    build_before_t_features,
    compute_news2,
    oxygen_flags,
)
from news2 import is_alert_from_gcs, supplemental_oxygen_flag


# --- Vectorised == scalar --------------------------------------------------

@pytest.mark.parametrize("col", list(BANDS))
def test_vectorised_bands_match_scalar_helpers(col):
    """Sweep every band boundary; the numpy path must equal news2.py exactly."""
    bounds, scores = BANDS[col]
    # dense sweep across and beyond the bands, including .5 steps for temperature
    lo, hi = min(bounds) - 10, max(bounds) + 10
    values = np.arange(lo, hi, 0.5)
    vec = _band_score(values, bounds, scores)
    scalar = np.array([SCALAR_SCORERS[col](v) for v in values], dtype=float)
    assert np.array_equal(vec, scalar), f"{col}: vectorised != scalar"


def test_band_score_missing_uses_missing_score():
    bounds, scores = BANDS["heart_rate"]
    out = _band_score([np.nan, 70.0], bounds, scores, missing_score=0)
    assert out[0] == 0        # NaN -> missing score, NOT the worst band
    assert out[1] == 0        # 70 bpm is normal


@pytest.mark.parametrize(
    "gcs, gv, gm, ge, gu",
    [
        (15, 5, 6, 4, 0),      # alert, not intubated
        (12, 4, 5, 3, 0),      # not alert
        (10, 0, 6, 4, 1),      # intubated but awake  <- the edge case
        (4, 0, 3, 1, 1),       # intubated, obtunded
        (8, 0, 4, 4, 1),       # intubated, eyes open, not obeying
        (15, None, None, None, 0),   # only a total available
        (None, None, None, None, None),  # nothing recorded
    ],
)
def test_alert_flags_match_scalar(gcs, gv, gm, ge, gu):
    vec = alert_flags([gcs], [gv], [gm], [ge], [gu])[0]
    scalar = is_alert_from_gcs(gcs, gv, gm, ge, gu)
    if isinstance(scalar, float) and math.isnan(scalar):
        assert math.isnan(vec)
    else:
        assert bool(vec) == bool(scalar)


@pytest.mark.parametrize(
    "dev, flow, fio2",
    [
        (None, None, None),    # nothing charted -> room air
        (1, None, None),       # device
        (None, 2, None),       # flow
        (None, 0, None),       # zero flow is not oxygen
        (None, None, 0.4),     # FiO2 as fraction
        (None, None, 40),      # FiO2 as percent
        (None, None, 0.21),    # room air
        (None, -1, None),      # junk flow must not imply oxygen
        (None, None, 40100),   # junk FiO2 must not imply oxygen
    ],
)
def test_oxygen_flags_match_scalar(dev, flow, fio2):
    vec = bool(oxygen_flags([dev], [flow], [fio2])[0])
    assert vec is supplemental_oxygen_flag(dev, flow, fio2)


# --- Leakage safety --------------------------------------------------------

def _raw_frame(rows):
    """Build a raw-shaped frame; unspecified channels/event columns default to NaN."""
    df = pd.DataFrame(rows)
    for c in CARRIED + ["event_h", "vent_h", "vaso_h", "death_h"]:
        if c not in df:
            df[c] = np.nan
    return df


def test_features_use_only_data_strictly_before_t():
    raw = _raw_frame([
        {"stay_id": 1, "hour": 6, "heart_rate": 80.0},
        {"stay_id": 1, "hour": 7, "heart_rate": np.nan},
        {"stay_id": 1, "hour": 8, "heart_rate": np.nan},
        {"stay_id": 1, "hour": 9, "heart_rate": 130.0},
    ])
    f = build_before_t_features(raw).sort_values("hour").reset_index(drop=True)

    hr = f["heart_rate"].tolist()
    assert math.isnan(hr[0])       # hour 6: nothing before it
    assert hr[1] == 80.0           # hour 7 sees hour 6
    assert hr[2] == 80.0           # carried forward
    assert hr[3] == 80.0           # hour 9 must NOT see its own 130
    assert f["heart_rate_hrs_since"].tolist()[3] == 3   # last measured at hour 6


def test_features_do_not_bleed_across_stays():
    raw = _raw_frame([
        {"stay_id": 1, "hour": 6, "heart_rate": 50.0},
        {"stay_id": 1, "hour": 7, "heart_rate": np.nan},
        {"stay_id": 2, "hour": 6, "heart_rate": np.nan},
        {"stay_id": 2, "hour": 7, "heart_rate": 90.0},
    ])
    f = build_before_t_features(raw)
    s2h6 = f[(f.stay_id == 2) & (f.hour == 6)]["heart_rate"].iloc[0]
    assert math.isnan(s2h6)        # stay 1's 50 must not leak into stay 2
    s1h7 = f[(f.stay_id == 1) & (f.hour == 7)]["heart_rate"].iloc[0]
    assert s1h7 == 50.0


# --- NEWS2 over a frame ----------------------------------------------------

def test_compute_news2_hand_computed():
    df = _raw_frame([{
        "stay_id": 1, "hour": 6,
        "resp_rate": 22.0, "spo2": 93.0, "temperature": 39.5,
        "sbp": 95.0, "heart_rate": 130.0,
        "gcs": 9.0, "gcs_verbal": 2.0, "gcs_motor": 4.0, "gcs_eyes": 3.0,
        "gcs_unable": 0.0, "o2_device": 1.0,
    }])
    score, alert, oxy = compute_news2(df)
    # 2+2+2+2+2 vitals + 2 oxygen + 3 not-alert = 15
    assert score[0] == 15
    assert alert[0] == 0
    assert bool(oxy[0]) is True


def test_compute_news2_intubated_awake_scores_no_consciousness_points():
    df = _raw_frame([{
        "stay_id": 1, "hour": 6,
        "resp_rate": 16.0, "spo2": 98.0, "temperature": 37.0,
        "sbp": 120.0, "heart_rate": 70.0,
        "gcs": 10.0, "gcs_verbal": 0.0, "gcs_motor": 6.0, "gcs_eyes": 4.0,
        "gcs_unable": 1.0, "o2_device": 1.0,
    }])
    score, alert, _ = compute_news2(df)
    assert alert[0] == 1     # awake despite the tube
    assert score[0] == 2     # oxygen only -- no +3 for being intubated


def test_compute_news2_all_missing_is_zero_not_worst_case():
    df = _raw_frame([{"stay_id": 1, "hour": 6}])
    score, _, _ = compute_news2(df)
    assert score[0] == 0     # NaN must not fall through to the worst bands


# --- Bounded carry-forward -------------------------------------------------

def test_stale_value_is_dropped_not_carried_forever():
    """A value older than its staleness bound must stop being propagated.

    This is the oxygen-carry-forward bug: one charted device used to imply
    "on oxygen" for the rest of the stay (observed max 328h).
    """
    raw = _raw_frame([{"stay_id": 1, "hour": h, "o2_device": 1.0 if h == 6 else np.nan}
                      for h in range(6, 22)])
    f = build_before_t_features(raw).sort_values("hour").reset_index(drop=True)
    by_hour = dict(zip(f["hour"], f["o2_device"]))

    assert by_hour[7] == 1.0            # 1h old -> still current
    assert by_hour[14] == 1.0           # 8h old -> at the bound, still current
    assert math.isnan(by_hour[15])      # 9h old -> past the 8h bound, dropped
    assert math.isnan(by_hour[21])


def test_staleness_bound_clears_the_oxygen_flag():
    """Once the device reading expires, the patient is no longer 'on oxygen'."""
    raw = _raw_frame([{"stay_id": 1, "hour": h, "o2_device": 1.0 if h == 6 else np.nan}
                      for h in range(6, 22)])
    f = build_before_t_features(raw)
    _, _, oxy = compute_news2(f)
    f = f.assign(oxy=oxy).sort_values("hour")
    assert bool(f[f.hour == 7]["oxy"].iloc[0]) is True
    assert bool(f[f.hour == 20]["oxy"].iloc[0]) is False   # expired -> room air


def test_vitals_use_their_own_longer_bound():
    """Bounds are per-channel: vitals get 12h, not the oxygen 8h."""
    raw = _raw_frame([{"stay_id": 1, "hour": h, "heart_rate": 80.0 if h == 6 else np.nan}
                      for h in range(6, 22)])
    f = build_before_t_features(raw).sort_values("hour").reset_index(drop=True)
    by_hour = dict(zip(f["hour"], f["heart_rate"]))
    assert by_hour[18] == 80.0          # 12h old -> at the vitals bound
    assert math.isnan(by_hour[19])      # 13h old -> dropped


# --- Per-event-type labels -------------------------------------------------

def test_per_event_labels_are_independent_and_horizon_bounded():
    raw = _raw_frame([
        {"stay_id": 1, "hour": h, "event_h": 20.0, "vent_h": 20.0,
         "vaso_h": 25.0, "death_h": 50.0}
        for h in range(6, 30)
    ])
    f = add_event_labels(raw).sort_values("hour").reset_index(drop=True)
    at = lambda h, c: f.loc[f.hour == h, c].iloc[0]

    # Both vent (h=20) and vaso (h=25) fall inside the 24h horizon from hour 6,
    # so both labels fire at once -- they are NOT mutually exclusive.
    assert at(6, "label_24h_vent") == 1
    assert at(6, "label_24h_vaso") == 1
    # death at h=50 is beyond that horizon until hour 26 (26+24=50, inclusive edge)
    assert at(6, "label_24h_death") == 0
    assert at(25, "label_24h_death") == 0
    assert at(26, "label_24h_death") == 1
    # a component already in the past is not a future event
    assert at(21, "label_24h_vent") == 0
    # first-event type is the censoring component
    assert at(6, "event_type") == "vent"


def test_event_type_is_none_when_no_event():
    raw = _raw_frame([{"stay_id": 1, "hour": 6, "event_h": np.nan}])
    f = add_event_labels(raw)
    assert pd.isna(f["event_type"].iloc[0])
    assert f["label_24h_vent"].iloc[0] == 0


# --- h6 root-cause fix: history rows feed the first prediction hour ---------

def test_build_uses_history_rows_then_drops_them(tmp_path):
    """Hours [0, STAB_HOURS) must supply LOCF context to hour 6 via the
    strictly-before-t shift, and must not appear in the exported cohort."""
    from mimic_cohort import STAB_HOURS, build

    rows = []
    for patient in range(1, 11):                    # enough patients to split
        stay = 100 + patient
        for hour in range(0, STAB_HOURS + 3):       # history 0-5 + predictions 6-8
            rows.append({
                "subject_id": patient, "stay_id": stay, "hour": hour,
                "is_history": int(hour < STAB_HOURS),
                # heart rate charted ONLY at hour 5 (the last history hour)
                "heart_rate": 77.0 if hour == 5 else np.nan,
                "label_24h": 0,
            })
    raw = _raw_frame(rows)
    raw_path = tmp_path / "raw.parquet"
    raw.to_parquet(raw_path, index=False)

    out = build(raw_path=str(raw_path), out_path=str(tmp_path / "cohort.parquet"))

    assert (out["hour"] >= STAB_HOURS).all()        # history rows are gone
    h6 = out[out["hour"] == STAB_HOURS]
    assert (h6["heart_rate"] == 77.0).all()         # ...but their data arrived
    assert (h6["heart_rate_hrs_since"] == 1).all()  # charted one hour before t=6
    # NEWS2 at h6 is scored from that carried value, not pinned by missingness:
    # HR 77 scores 0, so the score is a real 0 from data -- assert the carried
    # channel is populated rather than absent.
    assert not h6["heart_rate"].isna().any()


def test_build_without_history_column_is_backward_compatible(tmp_path):
    """Old raw extracts (grid starting at STAB_HOURS) must still build."""
    from mimic_cohort import STAB_HOURS, build

    rows = [{"subject_id": p, "stay_id": 100 + p, "hour": h, "label_24h": 0,
             "heart_rate": 80.0}
            for p in range(1, 11) for h in range(STAB_HOURS, STAB_HOURS + 3)]
    raw = _raw_frame(rows)
    raw_path = tmp_path / "raw.parquet"
    raw.to_parquet(raw_path, index=False)

    out = build(raw_path=str(raw_path), out_path=str(tmp_path / "cohort.parquet"))
    assert len(out) == len(raw)
