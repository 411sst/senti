"""Tests for the clinician-vs-physiology decomposition.

The load-bearing pin: the composite row of the subgroup race must reproduce
the standard harness exactly (same thresholds, same diffs -- only the summary
splits), and the treatment/death subgroups must partition the composite pairs.
"""

import numpy as np
import pandas as pd
import pytest

from decompose import (
    anticipation_share,
    component_race,
    stay_event_types,
    treatment_only_label,
)
from fusion_common import FusionContext

LABEL = "label_24h"


def _split_frame(n_stays, hours=30, offset=0, seed=0):
    """Synthetic split: half the stays deteriorate; event types alternate
    vent/death among event stays (real convention: type of the FIRST event)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stays):
        stay = offset + i
        has_event = i % 2 == 0
        event = float(6 + hours) if has_event else np.nan
        etype = ("death" if i % 4 == 0 else "vent") if has_event else np.nan
        for h in range(6, 6 + hours):
            label = int(has_event and (event - h) <= 24)
            base = 3.0 + rng.normal(0, 1.5)
            if has_event and (event - h) <= 12:
                base += (12 - (event - h)) * 0.8
            rows.append({"stay_id": stay, "patient_id": 1000 + offset + i // 2,
                         "hour": h, "event_h": event, "event_type": etype,
                         LABEL: label,
                         f"{LABEL}_vent": int(label and etype == "vent"),
                         f"{LABEL}_vaso": 0,
                         f"{LABEL}_death": int(label and etype == "death"),
                         "news2": max(0.0, round(base))})
    return pd.DataFrame(rows)


def _ctx(seed=0):
    return FusionContext({"val": _split_frame(60, seed=seed),
                          "test": _split_frame(60, offset=10_000,
                                               seed=seed + 1)})


def _score(ordered, quality, seed):
    rng = np.random.default_rng(seed)
    y = ordered[LABEL].to_numpy(dtype=float)
    return quality * y + (1 - quality) * rng.random(len(y))


def test_composite_row_reproduces_the_standard_harness():
    ctx = _ctx()
    sv = _score(ctx.val_ordered, 0.6, 2)
    st = _score(ctx.test_ordered, 0.6, 3)
    race = component_race(ctx, sv, st, reps=50)
    standard = ctx.evaluate_rung(sv, st, reps=50)
    for budget, cell in race.items():
        assert cell["composite"] == standard["budgets"][budget] or all(
            cell["composite"][k] == standard["budgets"][budget][k]
            for k in ("n_pair", "median", "win_rate", "ties", "sign_p"))


def test_subgroups_partition_the_composite_pairs():
    ctx = _ctx()
    sv = _score(ctx.val_ordered, 0.6, 2)
    st = _score(ctx.test_ordered, 0.6, 3)
    race = component_race(ctx, sv, st, reps=50)
    for cell in race.values():
        assert (cell["treatment"]["n_pair"] + cell["death"]["n_pair"]
                == cell["composite"]["n_pair"])
        assert cell["death"]["n_pair"] > 0                 # subgroup nonempty


def test_stay_event_types_takes_the_first_tag():
    frame = _split_frame(8)
    types = stay_event_types(frame)
    assert types[0] == "death" and types[2] == "vent"
    assert pd.isna(types[1])


def test_anticipation_share_formula():
    fake = {0.02: {
        "treatment": {"n_pair": 100, "win_rate": 0.60},
        "death": {"n_pair": 50, "win_rate": 0.40},
    }}
    # wins: 60 treatment vs 20 death -> share 0.75
    assert anticipation_share(fake) == pytest.approx(0.75)
    empty = {0.02: {"treatment": {"n_pair": 0, "win_rate": 0.0},
                    "death": {"n_pair": 0, "win_rate": 0.0}}}
    assert np.isnan(anticipation_share(empty))


def test_treatment_only_label_is_the_union():
    frame = pd.DataFrame({f"{LABEL}_vent": [0, 1, 0, 1],
                          f"{LABEL}_vaso": [0, 0, 1, 1]})
    np.testing.assert_array_equal(treatment_only_label(frame), [0, 1, 1, 1])
