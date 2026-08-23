"""Tests for the lead-time distillation grid (objectives + override plumbing).

Key guarantees: lambda=0 reproduces hard labels exactly, the monotone envelope
touches only the pre-event window of event stays, utility weights never
undercut their floor, and the orchestrator student's target override reaches
train sequences while val sequences keep their hard labels (else early
stopping would score against soft targets and AUPRC would be meaningless).
"""

import numpy as np
import pandas as pd
import pytest

from distill_lead import (
    DistilledOrchestrator,
    make_soft_frame,
    soft_targets,
    tls_targets,
    utility_weights,
)

LABEL = "label_24h"


def _ordered(hours=40, event_at=40.0):
    """One event stay + one non-event stay, sorted, real label convention."""
    rows = []
    for stay, ev in ((1, event_at), (2, np.nan)):
        for h in range(6, 6 + hours):
            label = int(not np.isnan(ev) and h < ev <= h + 24)
            rows.append({"stay_id": stay, "hour": h, "event_h": ev,
                         LABEL: label})
    return pd.DataFrame(rows)


# --- Soft targets ------------------------------------------------------------

def test_lambda_zero_reproduces_hard_labels():
    ordered = _ordered()
    q = np.random.default_rng(0).random(len(ordered))
    np.testing.assert_array_equal(
        soft_targets(ordered, q, lam=0.0),
        ordered[LABEL].to_numpy(dtype=float))


def test_soft_targets_stay_in_unit_interval():
    ordered = _ordered()
    q = np.random.default_rng(0).random(len(ordered))
    for lam in (0.25, 0.5, 1.0):
        y = soft_targets(ordered, q, lam)
        assert y.min() >= 0.0 and y.max() <= 1.0


def test_envelope_is_monotone_inside_the_window_and_absent_outside():
    ordered = _ordered(hours=40, event_at=46.0)
    rng = np.random.default_rng(1)
    q = rng.random(len(ordered))
    y = soft_targets(ordered, q, lam=1.0, envelope=12)   # y == q~ exactly

    tte = ordered["event_h"].to_numpy() - ordered["hour"].to_numpy()
    stay1 = ordered["stay_id"].to_numpy() == 1
    in_win = stay1 & (tte <= 36)
    # hard labels are 0 on the ramp (tte>24), so lam=1 exposes q~ directly
    ramp = y[in_win]
    assert (np.diff(ramp) >= -1e-12).all()               # running max
    out = stay1 & ~in_win[..., ]
    np.testing.assert_array_equal(y[stay1 & (tte > 36)],
                                  q[stay1 & (tte > 36)])
    stay2 = ~stay1
    np.testing.assert_array_equal(y[stay2], q[stay2])    # non-event untouched


# --- TLS + utility -----------------------------------------------------------

def test_tls_ramp_values_and_bounds():
    ordered = _ordered(hours=40, event_at=46.0)
    y = tls_targets(ordered, H=12)
    tte = ordered["event_h"].to_numpy() - ordered["hour"].to_numpy()
    stay1 = ordered["stay_id"].to_numpy() == 1
    np.testing.assert_array_equal(y[stay1 & (tte <= 24)], 1.0)
    np.testing.assert_array_equal(y[stay1 & (tte > 36)], 0.0)
    ramp = stay1 & (tte > 24) & (tte <= 36)
    np.testing.assert_allclose(y[ramp], (36 - tte[ramp]) / 12.0)
    assert (y[~stay1] == 0).all()


def test_utility_weights_shape_and_floor():
    ordered = _ordered(hours=40, event_at=46.0)
    tte = ordered["event_h"].to_numpy() - ordered["hour"].to_numpy()
    y = ordered[LABEL].to_numpy(dtype=float)
    w = utility_weights(ordered, beta=1.0)
    pos = y == 1
    np.testing.assert_allclose(w[pos], 1.0 + tte[pos] / 24.0)
    near = (y == 0) & (tte > 24) & (tte <= 36)
    assert (w[near] >= 0.5).all() and (w[near] <= 1.0).all()
    assert (w[near] < 1.0).any()          # the down-weighting actually bites
    far = (ordered["stay_id"].to_numpy() == 1) & (tte > 36)
    np.testing.assert_array_equal(w[far], 1.0)
    np.testing.assert_array_equal(w[np.isnan(tte)], 1.0)   # non-event stays


# --- The orchestrator student override ---------------------------------------

def test_soft_frame_overrides_train_but_never_val():
    from agents.base import KEY_COLUMNS
    from orchestrator.fusion import AgentBank

    rng = np.random.default_rng(0)
    stay_hours = {i: list(range(6, 18)) for i in (1, 2, 3, 4)}
    rows = [(s, h) for s, hs in stay_hours.items() for h in hs]
    keys = pd.DataFrame(rows, columns=list(KEY_COLUMNS))
    dims = {"a1": 4, "a2": 3}
    matrix = rng.normal(size=(len(keys), 9)).astype(np.float32)
    bank = AgentBank(keys, matrix, dims, ("a1", "a2"))

    def cohort(stays):
        return pd.DataFrame(
            [{"stay_id": s, "patient_id": 100 + s, "hour": h,
              LABEL: int(rng.random() < 0.3)}
             for s in stays for h in stay_hours[s]])

    train, val = cohort((1, 2)), cohort((3, 4))
    train_ordered = train.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    soft = make_soft_frame(train_ordered,
                           np.full(len(train_ordered), 0.37),
                           np.full(len(train_ordered), 2.0))
    student = DistilledOrchestrator(bank=bank, d_model=8, dilations=(1,),
                                    max_epochs=1, batch_size=2,
                                    verbose=False, soft_frame=soft)
    train_seqs = student._sequences(train)
    assert all((s["y"] == 0.37).all() and (s["w"] == 2.0).all()
               for s in train_seqs)
    val_seqs = student._sequences(val)
    hard = val.sort_values(["stay_id", "hour"])[LABEL].to_numpy(dtype=float)
    got = np.concatenate([s["y"] for s in val_seqs])
    np.testing.assert_array_equal(got, hard)             # val stays hard
