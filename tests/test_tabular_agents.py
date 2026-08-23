"""Tests for the cadence and fluid agents (feature builders + end-to-end).

The load-bearing test is the SHIFT one: an event charted in hour-bucket h must
be invisible to the feature row at hour h and first visible at h+1 --
strictly-before-t, the same contract as every other modality.
"""

import numpy as np
import pandas as pd
import pytest

from modality_features import (
    CADENCE_CHANNELS,
    CADENCE_FEATURES,
    FLUID_FEATURES,
    build_cadence_features,
    build_fluid_features,
)

LABEL = "label_24h"


def _cohort(stays={1: 20, 2: 12}):
    rows = [{"stay_id": s, "patient_id": 100 + s, "hour": h,
             LABEL: int(h > n - 6)}
            for s, n in stays.items() for h in range(6, 6 + n)]
    return pd.DataFrame(rows)


def _events(rows):
    base = {c: 0 for c in CADENCE_CHANNELS}
    return pd.DataFrame([{**base, **r} for r in rows])


# --- The strictly-before-t shift ---------------------------------------------

def test_event_in_bucket_h_is_first_visible_at_h_plus_1():
    cohort = _cohort()
    ev = _events([{"stay_id": 1, "hour": 10, "n_lab_draws": 3}])
    feats = build_cadence_features(cohort, ev)
    s1 = feats[feats["stay_id"] == 1].set_index("hour")
    assert s1.loc[10, "n_lab_draws_1h"] == 0.0          # own bucket: invisible
    assert s1.loc[11, "n_lab_draws_1h"] == pytest.approx(np.log1p(3))
    assert s1.loc[12, "n_lab_draws_1h"] == 0.0          # 1h window moved on
    # rolling windows carry it forward
    assert s1.loc[16, "n_lab_draws_6h"] == pytest.approx(np.log1p(3))
    assert s1.loc[17, "n_lab_draws_6h"] == 0.0          # 6h window closed


def test_no_cross_stay_bleed_and_zero_fill():
    cohort = _cohort()
    ev = _events([{"stay_id": 1, "hour": 8, "n_chart_rows": 50}])
    feats = build_cadence_features(cohort, ev)
    s2 = feats[feats["stay_id"] == 2]
    assert (s2[[f"{c}_24h" for c in CADENCE_CHANNELS]].to_numpy() == 0).all()
    assert np.isfinite(feats[CADENCE_FEATURES].to_numpy(dtype=float)).all()


def test_hours_since_caps_and_resets():
    cohort = _cohort({1: 20})
    ev = _events([{"stay_id": 1, "hour": 7, "n_lab_draws": 1}])
    feats = build_cadence_features(cohort, ev).set_index("hour")
    assert feats.loc[6, "hrs_since_lab_draw"] == 24.0    # nothing seen yet
    assert feats.loc[8, "hrs_since_lab_draw"] == 0.0     # visible bucket = 7
    assert feats.loc[12, "hrs_since_lab_draw"] == 4.0
    assert feats.loc[6 + 19, "hrs_since_lab_draw"] == 17.0


def test_acceleration_rises_when_recent_activity_exceeds_baseline():
    cohort = _cohort({1: 30})
    rows = [{"stay_id": 1, "hour": h, "n_orders_med": 1} for h in range(6, 26)]
    rows += [{"stay_id": 1, "hour": h, "n_orders_med": 6} for h in (26, 27)]
    feats = build_cadence_features(cohort, _events(rows)).set_index("hour")
    assert feats.loc[28, "n_orders_med_accel"] > feats.loc[20, "n_orders_med_accel"]


# --- Fluids ------------------------------------------------------------------

def _fluid_events(rows):
    base = {"urine_ml": 0.0, "n_urine_records": 0, "other_out_ml": 0.0,
            "in_ml": 0.0, "n_in_records": 0}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_duplicate_extract_keys_are_rejected_not_silently_fanned_out():
    """A duplicated (stay_id, hour) would left-join into EXTRA rows and
    misalign every feature against the labels -- the builders must refuse."""
    from modality_features import build_cadence_features, build_fluid_features
    cohort = _cohort({1: 12})
    dup_c = _events([{"stay_id": 1, "hour": 8, "n_lab_draws": 1},
                     {"stay_id": 1, "hour": 8, "n_lab_draws": 2}])
    with pytest.raises(ValueError, match="duplicate"):
        build_cadence_features(cohort, dup_c)
    dup_f = _fluid_events([{"stay_id": 1, "hour": 8, "urine_ml": 10.0},
                           {"stay_id": 1, "hour": 8, "urine_ml": 20.0}])
    with pytest.raises(ValueError, match="duplicate"):
        build_fluid_features(cohort, dup_f)


def test_oliguria_requires_records_not_just_low_volume():
    cohort = _cohort({1: 24, 2: 24})
    ev = _fluid_events(
        # stay 1: catheterised, genuinely oliguric (20 ml/h charted hourly)
        [{"stay_id": 1, "hour": h, "urine_ml": 20.0, "n_urine_records": 1}
         for h in range(6, 26)]
        # stay 2: no urine records at all -- must NOT read as oliguric
    )
    feats = build_fluid_features(cohort, ev)
    s1 = feats[feats["stay_id"] == 1].set_index("hour")
    s2 = feats[feats["stay_id"] == 2].set_index("hour")
    assert s1.loc[14, "oliguria_6h"] == 1.0              # 6x20 = 120 < 180
    assert (s2["oliguria_6h"] == 0).all()
    assert (s2["has_urine_history"] == 0).all()
    assert (s1.loc[8:, "has_urine_history"] == 1).all()
    assert np.isfinite(feats[FLUID_FEATURES].to_numpy(dtype=float)).all()


def test_net_balance_sign():
    cohort = _cohort({1: 24})
    # ONE row per (stay, hour): the extracts aggregate hourly, and the
    # builders now reject duplicate keys (they would fan out the join).
    ev = _fluid_events(
        [{"stay_id": 1, "hour": h, "in_ml": 500.0, "n_in_records": 1,
          "urine_ml": 30.0, "n_urine_records": 1} for h in range(6, 16)]
    )
    feats = build_fluid_features(cohort, ev).set_index("hour")
    assert feats.loc[20, "net_24h"] > 0                  # 5000 in vs 300 out
    assert feats.loc[20, "cum_net"] > 0


# --- Respiratory support -----------------------------------------------------

def _resp_events(rows):
    base = {"o2_device": None, "o2_flow": np.nan, "fio2": np.nan}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_resp_ladder_locf_shift_and_escalation():
    from modality_features import RESP_FEATURES, build_resp_features
    cohort = _cohort({1: 20})
    ev = _resp_events([
        {"stay_id": 1, "hour": 8, "o2_device": "Nasal cannula",
         "o2_flow": 2.0},
        {"stay_id": 1, "hour": 14, "o2_device": "High flow nasal cannula",
         "o2_flow": 40.0, "fio2": 60.0},
    ])
    feats = build_resp_features(cohort, ev).set_index("hour")
    assert feats.loc[8, "resp_level"] == 0.0          # own bucket invisible
    assert feats.loc[9, "resp_level"] == 1.0          # NC visible at h+1
    assert feats.loc[14, "resp_level"] == 1.0
    assert feats.loc[15, "resp_level"] == 4.0         # HFNC
    assert feats.loc[15, "resp_escalated_6h"] == 1.0
    assert feats.loc[15, "resp_fio2"] == 60.0
    assert feats.loc[9, "resp_fio2"] == 21.0          # default room air
    assert np.isfinite(feats[RESP_FEATURES].to_numpy(dtype=float)).all()


def test_resp_airway_is_a_state_not_a_rung():
    from modality_features import build_resp_features
    cohort = _cohort({1: 16})
    ev = _resp_events([
        {"stay_id": 1, "hour": 7, "o2_device": "Endotracheal tube"},
    ])
    feats = build_resp_features(cohort, ev).set_index("hour")
    assert feats.loc[8, "resp_airway"] == 1.0
    assert feats.loc[8, "resp_level"] == 0.0          # not on the ladder
    assert feats.loc[8, "resp_seen"] == 1.0


def test_resp_device_strings_with_trailing_whitespace_still_map():
    """MIMIC ships 'Bipap mask ' / 'CPAP mask ' WITH trailing spaces; the
    unstripped map silently dropped the whole NIV rung (caught because resp
    and resp_cap4 emitted byte-identical scores)."""
    from modality_features import build_resp_features
    cohort = _cohort({1: 12})
    ev = _resp_events([{"stay_id": 1, "hour": 7, "o2_device": "Bipap mask "}])
    feats = build_resp_features(cohort, ev).set_index("hour")
    assert feats.loc[8, "resp_level"] == 5.0


def test_resp_cap_variant_drops_the_niv_rung():
    from modality_features import build_resp_features
    cohort = _cohort({1: 16})
    ev = _resp_events([
        {"stay_id": 1, "hour": 7, "o2_device": "Bipap mask"},
    ])
    full = build_resp_features(cohort, ev).set_index("hour")
    capped = build_resp_features(cohort, ev, max_level=4).set_index("hour")
    assert full.loc[8, "resp_level"] == 5.0
    assert capped.loc[8, "resp_level"] == 4.0


# --- Agents end-to-end -------------------------------------------------------

def test_agents_fit_and_emit_on_synthetic_sources():
    from agents.tabular_agent import CadenceAgent, FluidAgent

    cohort = _cohort({i: 16 for i in range(1, 9)})
    rng = np.random.default_rng(0)
    cad = _events([{"stay_id": s, "hour": h,
                    "n_lab_draws": int(rng.integers(0, 3)),
                    "n_chart_rows": int(rng.integers(0, 30))}
                   for s in range(1, 9) for h in range(4, 22)])
    flu = _fluid_events([{"stay_id": s, "hour": h,
                          "urine_ml": float(rng.integers(0, 200)),
                          "n_urine_records": 1}
                         for s in range(1, 9) for h in range(4, 22)])
    train = cohort[cohort["stay_id"] <= 6]
    val = cohort[cohort["stay_id"] > 6]
    for cls, src in ((CadenceAgent, cad), (FluidAgent, flu)):
        agent = cls(source=src, max_epochs=1, batch_size=32, verbose=False)
        agent.fit(train, val)
        out = agent.emit(val)
        assert out.latent.shape == (len(val), 32)
        assert ((out.risk >= 0) & (out.risk <= 1)).all()
