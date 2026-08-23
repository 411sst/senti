"""Clinician-vs-physiology decomposition of the earliness claim (Bundle 1).

The composite label mixes two different things: death (physiology) and
vent/vaso onset (a clinician's decision). Any model predicting the composite
may be predicting the order set, not the patient -- the strongest attack a
reviewer can make on the +1-2h headline, made acute by the flagship's cadence
agent (openly clinician-behavioural). Nobody has measured how much of an EWS
lead is clinician anticipation; either answer is a result.

Two parts:

**Part (i) -- decompose EXISTING models, no retraining.** The cohort censors
every stay at its FIRST event and tags it (``event_type``), so the paired
first-alarm race splits cleanly into death-first stays (physiology-led: no
proximate treatment label) and treatment-first stays (vent/vaso). Same cached
scores, same val-frozen thresholds, same harness -- the per-stay diffs are
just summarized per subgroup. The cadence agent is the built-in positive
control (expected treatment-loaded); its mirror is the death-component AUPRC
already reported everywhere.

**Part (ii) -- retrain on component labels** (``label_24h_death`` /
vent-or-vaso). Does the architecture's lead survive when the target is
physiology only? The orchestrator student reuses distill_lead's label
override; the GBM refits the recorded winner. A death-only model is a LOWER
bound on physiological signal: death is itself treatment-mediated
(DNR/withdrawal); comfort-care exclusion is documented future work.
"""

import numpy as np
import pandas as pd

from fair_capability import (
    BUDGETS,
    frozen_thresholds,
    paired_first_alarm_diffs,
    summarize_paired,
)

LABEL = "label_24h"
EVENT_TYPES = ("vent", "vaso", "death")
TREATMENT_TYPES = ("vent", "vaso")


def stay_event_types(test_ordered):
    """stay_id -> first-event type ('vent'/'vaso'/'death'/nan) for the split."""
    g = test_ordered.groupby("stay_id")["event_type"].first()
    return g.to_dict()


def component_race(ctx, s_val, s_test, reps=2000):
    """The paired first-alarm race, summarized per first-event subgroup.

    Thresholds are the same val-frozen, NEWS2-volume-matched operating points
    as every other table (fusion_common protocol); only the SUMMARY is split,
    so the composite row must reproduce the standard harness exactly (pinned
    by a test). Returns {budget: {"composite"|"treatment"|"death": summary}}.
    """
    types = stay_event_types(ctx.test_ordered)
    nf = ctx.news2_val[ctx.freeze_mask]
    af = ctx.alertable_val[ctx.freeze_mask]
    out = {}
    for budget in BUDGETS:
        thr_n2, thr_m, _ = frozen_thresholds(nf, s_val[ctx.freeze_mask],
                                             af, budget)
        diffs = paired_first_alarm_diffs(ctx.test_ordered, ctx.news2_test,
                                         s_test, thr_n2, thr_m)
        stay_type = diffs["stay_id"].map(types)
        cells = {"composite": diffs,
                 "treatment": diffs[stay_type.isin(TREATMENT_TYPES)],
                 "death": diffs[stay_type == "death"]}
        out[budget] = {name: summarize_paired(d, ctx.stay_to_patient,
                                              reps=reps)
                       for name, d in cells.items()}
    return out


def anticipation_share(race_summary, budget=0.02):
    """How much of the composite win mass sits on treatment-first stays.

    share = treatment wins / (treatment wins + death wins), computed from the
    subgroup n_pair x win_rate. 1.0 = the lead is pure clinician
    anticipation; the death-subgroup median lead is the companion number (the
    physiology lead in hours).
    """
    cell = race_summary[budget]
    wins = {k: cell[k]["n_pair"] * cell[k]["win_rate"]
            for k in ("treatment", "death")}
    total = wins["treatment"] + wins["death"]
    return (wins["treatment"] / total) if total > 0 else float("nan")


def component_auprc(ctx, s_test):
    """Test AUPRC against each component label (and the composite)."""
    from metrics import auprc
    out = {"composite": float(auprc(ctx.y_test, s_test))}
    for t in EVENT_TYPES:
        y = ctx.test_ordered[f"{LABEL}_{t}"].to_numpy(dtype=float)
        out[t] = float(auprc(y, s_test))
    return out


def treatment_only_label(frame):
    """Vent-or-vaso within 24h -- the clinician-decision target."""
    return np.maximum(frame[f"{LABEL}_vent"].to_numpy(dtype=float),
                      frame[f"{LABEL}_vaso"].to_numpy(dtype=float))
