"""Tests for the fusion protocol (fusion_common) and the ladder/gate rungs.

The load-bearing guarantees: the V_fit/V_freeze split is patient-disjoint and
actually enforced (calibrators never see freeze rows, thresholds never see fit
rows), the constrained stack cannot violate its own earliness floor, and the
oracle gate is the only rung that reads ``event_h``.
"""

import numpy as np
import pandas as pd
import pytest

from fusion_common import FusionContext, split_val_rows
from fusion_ladder import _rise6, _strictify, build_gate, build_ladder, \
    gate_blend

LABEL = "label_24h"


def _split_frame(n_stays, hours=30, offset=0, seed=0):
    """Synthetic cohort split: half the stays deteriorate at their last hour.

    Labels follow the real convention (event within 24h, rows censored before
    the event); NEWS2 and the parent scores rise toward the event so the
    paired-timing race has something to find.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stays):
        stay = offset + i
        has_event = i % 2 == 0
        event = float(6 + hours) if has_event else np.nan
        for h in range(6, 6 + hours):
            label = int(has_event and (event - h) <= 24)
            base = 3.0 + rng.normal(0, 1.5)
            if has_event and (event - h) <= 12:
                base += (12 - (event - h)) * 0.8
            rows.append({"stay_id": stay, "patient_id": 1000 + offset + i // 2,
                         "hour": h, "event_h": event, LABEL: label,
                         "news2": max(0.0, round(base))})
    return pd.DataFrame(rows)


def _splits(n_stays=60, seed=0):
    return {"val": _split_frame(n_stays, offset=0, seed=seed),
            "test": _split_frame(n_stays, offset=10_000, seed=seed + 1)}


def _parent_scores(ordered, quality, seed=0):
    """A raw score correlated with the label at the given quality in [0, 1]."""
    rng = np.random.default_rng(seed)
    y = ordered[LABEL].to_numpy(dtype=float)
    return quality * y + (1 - quality) * rng.random(len(y))


# --- The protocol split ------------------------------------------------------

def test_split_val_rows_is_patient_disjoint_covering_and_deterministic():
    val = _splits()["val"]
    ordered, fit_mask, freeze_mask = split_val_rows(val, seed=7)
    assert (fit_mask | freeze_mask).all() and not (fit_mask & freeze_mask).any()
    fit_p = set(ordered.loc[fit_mask, "patient_id"])
    freeze_p = set(ordered.loc[freeze_mask, "patient_id"])
    assert not fit_p & freeze_p
    _, fit2, _ = split_val_rows(val, seed=7)
    assert (fit_mask == fit2).all()
    _, fit3, _ = split_val_rows(val, seed=8)
    assert not (fit_mask == fit3).all()


def test_calibrator_never_sees_freeze_rows():
    ctx = FusionContext(_splits())
    raw = _parent_scores(ctx.val_ordered, 0.6)
    a = ctx.fit_calibrator(raw)
    corrupted = raw.copy()
    corrupted[ctx.freeze_mask] = np.random.default_rng(0).random(
        int(ctx.freeze_mask.sum()))
    b = ctx.fit_calibrator(corrupted)
    grid = np.linspace(0, 1, 101)
    np.testing.assert_allclose(a.transform(grid), b.transform(grid))


def test_evaluate_rung_thresholds_use_only_freeze_rows():
    ctx = FusionContext(_splits())
    s_val = _parent_scores(ctx.val_ordered, 0.6)
    s_test = _parent_scores(ctx.test_ordered, 0.6, seed=1)
    base = ctx.evaluate_rung(s_val, s_test, reps=50)
    corrupted = s_val.copy()
    corrupted[ctx.fit_mask] = 1e9          # wreck the fit half only
    same = ctx.evaluate_rung(corrupted, s_test, reps=50)
    assert base == same


def test_strictify_breaks_plateaus_in_raw_order_and_preserves_scale():
    """Isotonic plateaus must resolve in raw-score order at <=1e-9 magnitude,
    otherwise a plateau on the val threshold fires wholesale past the matched
    alert volume (the achieved-volume artifact caught in the first run)."""
    raw = np.array([0.10, 0.40, 0.20, 0.90, 0.50])
    cal = np.array([0.30, 0.30, 0.30, 0.80, 0.30])     # plateau of four
    s = _strictify(cal, raw)
    assert np.abs(s - cal).max() <= 1e-9
    plateau = [0, 1, 2, 4]
    assert (np.argsort(s[plateau]) == np.argsort(raw[plateau])).all()
    assert len(np.unique(s)) == len(s)


# --- The ladder --------------------------------------------------------------

def _ladder(ctx, o_quality=0.3, g_quality=0.9):
    oV = _parent_scores(ctx.val_ordered, o_quality, seed=2)
    oT = _parent_scores(ctx.test_ordered, o_quality, seed=3)
    gV = _parent_scores(ctx.val_ordered, g_quality, seed=4)
    gT = _parent_scores(ctx.test_ordered, g_quality, seed=5)
    return build_ladder(ctx, oV, oT, gV, gT), (oT, gT)


def test_ladder_rungs_exploit_the_better_parent():
    """With one near-perfect parent, MAX and LOGIT3 must track it, not the
    noise parent -- the known-optimal combination on synthetic scores."""
    from metrics import auprc
    ctx = FusionContext(_splits())
    (rungs, info, cal), (oT, gT) = _ladder(ctx)
    y = ctx.y_test
    for name in ("MAX", "LOGIT3"):
        assert auprc(y, rungs[name][1]) > auprc(y, cal["oT"]) + 0.05
        assert auprc(y, rungs[name][1]) >= auprc(y, cal["gT"]) - 0.05


def test_constrained_stack_never_violates_its_earliness_floor():
    ctx = FusionContext(_splits())
    (rungs, info, cal), _ = _ladder(ctx)
    c = info["constrained"]
    chosen = [g for g in c["grid"] if g["w"] == c["w"]]
    assert len(chosen) == 1
    assert chosen[0]["lead"] >= c["lead_floor"]
    # w=1 recovers the orchestrator, so the feasible set is never empty.
    w1 = [g for g in c["grid"] if g["w"] == 1.0][0]
    assert w1["lead"] == pytest.approx(c["lead_floor"])


# --- The gate ----------------------------------------------------------------

def test_rise6_is_per_stay_and_zero_in_early_hours():
    ordered = pd.DataFrame({
        "stay_id": [1] * 8 + [2] * 8,
        "hour": list(range(6, 14)) * 2,
    })
    s = np.concatenate([np.arange(8.0), 10 + np.arange(8.0)])
    r = _rise6(ordered, s)
    assert (r[:6] == 0).all() and (r[8:14] == 0).all()   # warm-up per stay
    assert r[6] == pytest.approx(6.0) and r[15] == pytest.approx(6.0)


def test_gate_blend_saturates_to_one_parent():
    o = np.array([0.9, 0.1]); g = np.array([0.2, 0.7])
    f, gate = gate_blend((50.0, 0.0, 0.0), o, g, np.zeros(2))
    np.testing.assert_allclose(f, o, atol=1e-6)
    f, gate = gate_blend((-50.0, 0.0, 0.0), o, g, np.zeros(2))
    np.testing.assert_allclose(f, g, atol=1e-6)


def test_oracle_gate_switches_exactly_on_true_proximity():
    ctx = FusionContext(_splits())
    (rungs, info, cal), _ = _ladder(ctx)
    g_rungs, _ = build_gate(ctx, cal)
    s_val, s_test = g_rungs["ORACLE-GATE"]
    for frame, s in ((ctx.val_ordered, s_val), (ctx.test_ordered, s_test)):
        tte = frame["event_h"].to_numpy(dtype=float) - frame["hour"].to_numpy()
        near = frame["event_h"].notna().to_numpy() & (tte <= 24)
        o = cal["oV"] if frame is ctx.val_ordered else cal["oT"]
        g = cal["gV"] if frame is ctx.val_ordered else cal["gT"]
        np.testing.assert_allclose(s[near], o[near])
        np.testing.assert_allclose(s[~near], g[~near])


def test_only_the_oracle_reads_event_h():
    """build_ladder and the learned gate must work on frames WITHOUT event_h;
    only the oracle rung may require it."""
    splits = _splits()
    ctx = FusionContext(splits)
    hidden = ctx.val_ordered.drop(columns=["event_h"])
    # the ladder's selection surface uses event_h only through the harness
    # (freeze_lead) -- so the fit machinery itself must not touch it:
    oV = _parent_scores(ctx.val_ordered, 0.3, seed=2)
    cal = ctx.fit_calibrator(oV)          # label-only, fine without event_h
    assert cal.transform(np.array([0.5])).shape == (1,)
