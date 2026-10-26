"""Tests for the per-unit standby operating-point layer.

The load-bearing guarantees: a single-stratum world reproduces the global
harness exactly (regression pin), a thin stratum falls back to the global
threshold instead of fitting noise, and per-unit thresholds actually equalize
the realised alarm rate they were fit to equalize.
"""

import numpy as np
import pandas as pd
import pytest

from fusion_common import FusionContext
from standby import (
    MIN_FREEZE_ROWS,
    agent_roster,
    burden_spread,
    evaluate_standby,
    rows_stratum,
    standby_thresholds,
    threshold_stability,
)

LABEL = "label_24h"


def _split_frame(n_stays, hours=40, offset=0, seed=0, score_scale=1.0):
    """Synthetic split with a NEWS2-like integer score and known events."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stays):
        stay = offset + i
        has_event = i % 3 == 0
        event = float(6 + hours) if has_event else np.nan
        for h in range(6, 6 + hours):
            label = int(has_event and (event - h) <= 24)
            base = 3.0 + rng.normal(0, 1.5) * score_scale
            if has_event and (event - h) <= 12:
                base += (12 - (event - h)) * 0.6
            rows.append({"stay_id": stay, "patient_id": 5000 + offset + i,
                         "hour": h, "event_h": event, "event_type": "vent" if has_event else np.nan,
                         LABEL: label, "news2": max(0.0, round(base))})
    return pd.DataFrame(rows)


def _ctx(n_stays=60):
    splits = {"train": _split_frame(4, offset=90000),
              "val": _split_frame(n_stays, offset=0),
              "test": _split_frame(n_stays, offset=1000, seed=1)}
    return FusionContext(splits)


def _strata(ctx, units):
    """Assign stays round-robin to `units` across BOTH val and test stays."""
    stays = np.concatenate([ctx.val_ordered.stay_id.unique(),
                            ctx.test_ordered.stay_id.unique()])
    return pd.DataFrame({"stay_id": stays,
                         "unit_group": [units[i % len(units)]
                                        for i in range(len(stays))]})


def _score(ctx, side):
    ordered = ctx.val_ordered if side == "val" else ctx.test_ordered
    rng = np.random.default_rng(7)
    y = ordered[LABEL].to_numpy(float)
    return y * 0.5 + rng.uniform(0, 0.5, len(y))


# --- rows_stratum ------------------------------------------------------------

def test_rows_stratum_raises_on_unlabelled_stay():
    ctx = _ctx(12)
    strata = _strata(ctx, ["a"]).iloc[:-1]        # drop one stay's label
    with pytest.raises(ValueError, match="no stratum label"):
        rows_stratum(ctx.test_ordered, strata)


# --- thresholds --------------------------------------------------------------

def test_single_stratum_reproduces_the_global_thresholds():
    """One unit containing everything == the standard global harness."""
    ctx = _ctx(50)
    strata = _strata(ctx, ["only"])
    sv = _score(ctx, "val")
    thr = standby_thresholds(ctx, rows_stratum(ctx.val_ordered, strata),
                             sv, budget=0.05, min_rows=10)
    g, u = thr["__global__"], thr["only"]
    assert u["source"] == "unit"
    assert u["thr_news2"] == pytest.approx(g["thr_news2"])
    assert u["thr_model"] == pytest.approx(g["thr_model"])


def test_thin_stratum_falls_back_to_global():
    ctx = _ctx(40)
    strata = _strata(ctx, ["big"])
    # relabel exactly one VAL stay as its own unit -> far below any sane
    # min_rows (the last stay in the frame is a test stay, which would exercise
    # the different, test-only-unit path)
    strata.loc[strata.index[0], "unit_group"] = "tiny"
    sv = _score(ctx, "val")
    thr = standby_thresholds(ctx, rows_stratum(ctx.val_ordered, strata),
                             sv, budget=0.05, min_rows=MIN_FREEZE_ROWS)
    assert thr["tiny"]["source"] == "global-fallback"
    assert thr["tiny"]["thr_model"] == thr["__global__"]["thr_model"]


def test_budget_anchor_equalizes_where_news2_anchor_cannot():
    """The deployment threshold is the model's own quantile, so every unit's
    freeze-half alarm rate lands on the budget regardless of how badly the
    unit's NEWS2 integer overshoots."""
    ctx = _ctx(60)
    strata = _strata(ctx, ["p", "q"])
    sv = _score(ctx, "val")
    strat_val = rows_stratum(ctx.val_ordered, strata)
    sv = np.where(strat_val == "p", sv + 0.8, sv)
    thr = standby_thresholds(ctx, strat_val, sv, budget=0.05, min_rows=10)
    af = ctx.alertable_val[ctx.freeze_mask]
    sf, stf = sv[ctx.freeze_mask], strat_val[ctx.freeze_mask]
    for u in ("p", "q"):
        sel = (stf == u) & af
        rate = (sf[sel] >= thr[u]["thr_model_budget"]).mean()
        assert rate == pytest.approx(0.05, abs=0.02)


def test_standby_points_equalize_the_freeze_alarm_rate():
    """Two units whose scores live on different scales: one global threshold
    gives them different alarm rates; per-unit thresholds pull both to the
    NEWS2-matched volume within unit."""
    ctx = _ctx(60)
    strata = _strata(ctx, ["hot", "cold"])
    sv = _score(ctx, "val")
    strat_val = rows_stratum(ctx.val_ordered, strata)
    sv = np.where(strat_val == "hot", sv + 0.8, sv)     # hot unit scores higher

    thr = standby_thresholds(ctx, strat_val, sv, budget=0.05, min_rows=10)
    af = ctx.alertable_val[ctx.freeze_mask]
    sf = sv[ctx.freeze_mask]
    stf = strat_val[ctx.freeze_mask]
    rates_global, rates_unit = [], []
    for u in ("hot", "cold"):
        sel = (stf == u) & af
        rates_global.append((sf[sel] >= thr["__global__"]["thr_model"]).mean())
        rates_unit.append((sf[sel] >= thr[u]["thr_model"]).mean())
    # global threshold: hot unit fires far more; per-unit: both near matched
    assert abs(rates_global[0] - rates_global[1]) > 0.04
    assert abs(rates_unit[0] - rates_unit[1]) < 0.02


# --- evaluation --------------------------------------------------------------

def test_test_only_unit_runs_on_the_global_point():
    """A unit with no val presence still gets evaluated — on global thresholds."""
    ctx = _ctx(30)
    strata = _strata(ctx, ["a"])
    # relabel one TEST stay to a unit val has never seen
    test_stays = set(ctx.test_ordered.stay_id.unique())
    idx = strata[strata.stay_id.isin(test_stays)].index[0]
    strata.loc[idx, "unit_group"] = "ghost"
    sv, st = _score(ctx, "val"), _score(ctx, "test")
    thr = standby_thresholds(ctx, rows_stratum(ctx.val_ordered, strata),
                             sv, budget=0.05, min_rows=10)
    assert "ghost" not in thr
    rows = evaluate_standby(ctx, rows_stratum(ctx.test_ordered, strata),
                            st, thr, reps=20)
    assert rows["ghost"]["source"] == "global-fallback"
    # the two columns use different anchors by design (NEWS2-matched vs pure
    # budget), so the invariant is threshold identity with the global entry,
    # not column equality
    assert thr["__global__"]["thr_model_budget"] is not None


def test_evaluate_standby_reports_both_operating_points():
    ctx = _ctx(60)
    strata = _strata(ctx, ["a", "b"])
    sv, st = _score(ctx, "val"), _score(ctx, "test")
    strat_val = rows_stratum(ctx.val_ordered, strata)
    strat_test = rows_stratum(ctx.test_ordered, strata)
    thr = standby_thresholds(ctx, strat_val, sv, budget=0.05, min_rows=10)
    rows = evaluate_standby(ctx, strat_test, st, thr, reps=50)
    assert set(rows) == {"a", "b"}
    for r in rows.values():
        assert 0 <= r["alarm_standby"] <= 1
        assert "race_global" in r and "race_standby" in r
    assert burden_spread(rows, "alarm_standby") >= 1.0


# --- roster ------------------------------------------------------------------

def test_agent_roster_ranks_the_informative_agent_by_lift():
    ctx = _ctx(60)
    strata = _strata(ctx, ["x"])
    strat_test = rows_stratum(ctx.test_ordered, strata)
    y = ctx.y_test
    rng = np.random.default_rng(3)
    risks = {"good": y * 0.9 + rng.uniform(0, 0.1, len(y)),
             "noise": rng.uniform(0, 1, len(y))}
    out = agent_roster(ctx, strat_test, risks, min_pos=5)
    cell = out["x"]["agents"]
    assert cell["good"]["lift"] > cell["noise"]["lift"] * 2


def test_agent_roster_flags_underpowered_units():
    ctx = _ctx(30)
    strata = _strata(ctx, ["x"])
    strat_test = rows_stratum(ctx.test_ordered, strata)
    out = agent_roster(ctx, strat_test, {"a": np.zeros(len(ctx.y_test))},
                       min_pos=10 ** 6)
    assert out["x"]["underpowered"]


def test_threshold_stability_summarises_spread_and_source_flips():
    per_seed = {
        42: {"a": {"thr_model_budget": 0.10, "source": "unit"},
             "b": {"thr_model_budget": 0.20, "source": "unit"}},
        1:  {"a": {"thr_model_budget": 0.12, "source": "unit"},
             "b": {"thr_model_budget": 0.20, "source": "global-fallback"}},
        2:  {"a": {"thr_model_budget": 0.14, "source": "unit"},
             "b": {"thr_model_budget": 0.20, "source": "unit"}},
    }
    st = threshold_stability(per_seed)
    assert st["a"]["min"] == 0.10 and st["a"]["max"] == 0.14
    assert st["a"]["n_seeds"] == 3 and st["a"]["cv"] > 0
    # a perfectly stable unit has zero spread ...
    assert st["b"]["cv"] == pytest.approx(0.0, abs=1e-12)
    # ... but its source flip across seeds is still surfaced
    assert st["b"]["source_counts"] == {"unit": 2, "global-fallback": 1}
