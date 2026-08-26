"""Tests for the neuro-status and infection-workup builders (Tier-1 modalities).

The load-bearing guarantees: level channels (RASS, pain) are never zero-filled
(RASS 0 means "alert and calm", not "unobserved"), count channels never carry
forward (silence is a real zero), every feature is strictly before t, and the
active-antibiotic reconstruction cannot go negative on a stop whose start
predates the lookback window.
"""

import numpy as np
import pandas as pd
import pytest

from modality_features import (
    INFECTION_FEATURES,
    NEURO_STATUS_FEATURES,
    build_infection_features,
    build_neuro_status_features,
)


def _cohort(n_stays=2, hours=30):
    rows = [{"stay_id": s, "hour": h}
            for s in range(n_stays) for h in range(6, 6 + hours)]
    return pd.DataFrame(rows)


def _neuro(rows):
    base = {"rass_min": np.nan, "rass_max": np.nan, "pain_max": np.nan,
            "n_rass": 0, "cam_assessed": 0, "cam_positive": 0, "n_pain": 0}
    return pd.DataFrame([{**base, **r} for r in rows])


def _infection(rows):
    base = {"n_cultures": 0, "n_blood_cultures": 0,
            "n_abx_starts": 0, "n_abx_stops": 0}
    return pd.DataFrame([{**base, **r} for r in rows])


# --- neuro-status ------------------------------------------------------------

def test_neuro_features_align_and_cover_the_declared_list():
    co = _cohort()
    ne = _neuro([{"stay_id": 0, "hour": 8, "rass_min": -1.0, "rass_max": 0.0,
                  "n_rass": 1}])
    out = build_neuro_status_features(co, ne)
    assert len(out) == len(co)
    assert list(out.columns) == ["stay_id", "hour"] + NEURO_STATUS_FEATURES


def test_rass_is_carried_not_zero_filled():
    """RASS 0 is a real score; an unobserved hour must be NaN, not 0."""
    co = _cohort(n_stays=1)
    ne = _neuro([{"stay_id": 0, "hour": 10, "rass_min": 2.0, "rass_max": 2.0,
                  "n_rass": 1}])
    out = build_neuro_status_features(co, ne)
    assert np.isnan(out.loc[out.hour == 8, "rass_now"]).all()   # before charting
    assert out.loc[out.hour == 12, "rass_now"].item() == 2.0    # carried
    assert out.loc[out.hour == 12, "agitated_now"].item() == 1.0


def test_rass_is_strictly_before_t_and_expires():
    co = _cohort(n_stays=1)
    ne = _neuro([{"stay_id": 0, "hour": 10, "rass_max": 1.0, "rass_min": 1.0,
                  "n_rass": 1}])
    out = build_neuro_status_features(co, ne)
    # at hour 10 the observation is in the PRESENT hour -> not yet visible
    assert np.isnan(out.loc[out.hour == 10, "rass_now"]).all()
    assert out.loc[out.hour == 11, "rass_now"].item() == 1.0
    # 12h staleness: hour 10 obs is last visible at hour 22 (staleness 12)
    assert out.loc[out.hour == 22, "rass_now"].item() == 1.0
    assert np.isnan(out.loc[out.hour == 24, "rass_now"]).all()


def test_delirium_flag_is_a_24h_window_not_forever():
    co = _cohort(n_stays=1)
    ne = _neuro([{"stay_id": 0, "hour": 7, "cam_assessed": 4, "cam_positive": 2}])
    out = build_neuro_status_features(co, ne)
    assert out.loc[out.hour == 8, "delirium_recent"].item() == 1.0
    assert out.loc[out.hour == 30, "delirium_recent"].item() == 1.0   # 7+24=31 exclusive
    assert out.loc[out.hour == 32, "delirium_recent"].item() == 0.0


def test_neuro_counts_do_not_cross_stays():
    co = _cohort(n_stays=2)
    ne = _neuro([{"stay_id": 0, "hour": 8, "cam_assessed": 3, "cam_positive": 3}])
    out = build_neuro_status_features(co, ne)
    s1 = out[out.stay_id == 1]
    assert (s1["cam_positive_24h"] == 0).all()
    assert (s1["delirium_recent"] == 0).all()


# --- infection ---------------------------------------------------------------

def test_infection_features_align_and_zero_fill():
    co = _cohort()
    inf = _infection([{"stay_id": 0, "hour": 7, "n_cultures": 1}])
    out = build_infection_features(co, inf)
    assert len(out) == len(co)
    assert list(out.columns) == ["stay_id", "hour"] + INFECTION_FEATURES
    # a stay with no events reads all-zero counts, not NaN
    s1 = out[out.stay_id == 1]
    assert (s1["cultures_24h"] == 0).all()
    assert (s1["on_abx"] == 0).all()


def test_culture_draw_is_visible_one_hour_later_only():
    co = _cohort(n_stays=1)
    inf = _infection([{"stay_id": 0, "hour": 9, "n_cultures": 2,
                       "n_blood_cultures": 1}])
    out = build_infection_features(co, inf)
    assert out.loc[out.hour == 9, "cultures_6h"].item() == 0.0     # present hour
    assert out.loc[out.hour == 10, "cultures_6h"].item() == pytest.approx(np.log1p(2))
    assert out.loc[out.hour == 10, "blood_culture_24h"].item() == 1.0


def test_pre_icu_lookback_reaches_hour_six():
    """An ED blood culture at hour -3 must be visible at the first prediction row."""
    co = _cohort(n_stays=1)
    inf = _infection([{"stay_id": 0, "hour": -3, "n_cultures": 1,
                       "n_blood_cultures": 1}])
    out = build_infection_features(co, inf)
    assert out.loc[out.hour == 6, "cultures_24h"].item() > 0
    assert out.loc[out.hour == 6, "blood_culture_24h"].item() == 1.0


def test_abx_active_count_rises_and_falls():
    co = _cohort(n_stays=1)
    inf = _infection([
        {"stay_id": 0, "hour": 8, "n_abx_starts": 2},
        {"stay_id": 0, "hour": 20, "n_abx_stops": 1},
    ])
    out = build_infection_features(co, inf)
    assert out.loc[out.hour == 9, "abx_active_now"].item() == 2.0
    assert out.loc[out.hour == 9, "new_abx_6h"].item() == 1.0
    assert out.loc[out.hour == 21, "abx_active_now"].item() == 1.0
    assert out.loc[out.hour == 21, "on_abx"].item() == 1.0


def test_orphan_stop_cannot_go_negative():
    """A stop whose start predates the lookback clips at zero active."""
    co = _cohort(n_stays=1)
    inf = _infection([{"stay_id": 0, "hour": 7, "n_abx_stops": 3}])
    out = build_infection_features(co, inf)
    assert (out["abx_active_now"] >= 0).all()
    assert (out["on_abx"] == 0).all()


def test_infection_future_events_do_not_leak():
    co = _cohort(n_stays=1)
    inf1 = _infection([{"stay_id": 0, "hour": 12, "n_cultures": 1}])
    inf2 = _infection([{"stay_id": 0, "hour": 12, "n_cultures": 1},
                       {"stay_id": 0, "hour": 30, "n_cultures": 5,
                        "n_abx_starts": 4}])
    a = build_infection_features(co, inf1)
    b = build_infection_features(co, inf2)
    early = a.hour <= 30
    for c in INFECTION_FEATURES:
        np.testing.assert_allclose(a.loc[early, c], b.loc[early, c],
                                   err_msg=f"{c} moved on a future event")
