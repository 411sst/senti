"""Tests for the D4 agent templates (src/text_arm/agent_templates.py).

* the vectorised risk binning matches ``summaries.risk_word`` exactly;
* sentences are deterministic and independent of row order;
* no future data: built end to end from synthetic sources through the agents'
  own feature builders, rewriting every event at hour >= t0 leaves every
  sentence at hour <= t0 unchanged;
* hand-checked renderings for the main branches.

Every row is synthetic; ids are small integers.
"""

import numpy as np
import pandas as pd
import pytest

from allfeat_gbm import build_allfeat_design
from test_allfeat_gbm import T0, _cohort, _rewrite_from, _sources
from text_arm.agent_templates import (
    ECG_FINDINGS,
    TEMPLATES,
    render_ecg,
    render_fluid,
    render_resp,
    risk_code,
    RISK_WORDS,
    sentences,
)
from text_arm.summaries import risk_word

ECG_NAMES = (["hr", "qrs_dur", "qtc_bazett", "hr_valid", "qrs_dur_valid",
              "qtc_bazett_valid", "normal_ecg"] + [c for c, _ in ECG_FINDINGS])


def _rich_sources(seed=1):
    src = _sources(seed=seed)
    rng = np.random.default_rng(seed)
    e = src["ecg"]
    n = len(e["index"])
    vals = np.zeros((n, len(ECG_NAMES)), dtype=np.float32)
    col = {c: i for i, c in enumerate(ECG_NAMES)}
    vals[:, col["hr"]] = rng.uniform(50, 140, n)
    vals[:, col["qrs_dur"]] = rng.uniform(80, 160, n)
    vals[:, col["qtc_bazett"]] = rng.uniform(380, 520, n)
    vals[:, col["hr_valid"]] = 1
    vals[:, col["qrs_dur_valid"]] = 1
    vals[:, col["qtc_bazett_valid"]] = rng.random(n) < 0.8
    for c, _ in ECG_FINDINGS:
        vals[:, col[c]] = rng.random(n) < 0.2
    e["values"], e["names"] = vals, list(ECG_NAMES)
    return src


def _all_sentences(cohort, sources):
    X, _, keys = build_allfeat_design(cohort, sources)
    # A risk that is itself a function of before-t inputs only (the row's
    # NEWS2 respiratory sub-score), like an agent's calibrated risk.
    risk = 0.01 + 0.04 * X["news2_resp_rate"].to_numpy(dtype=float)
    out = {}
    for agent in TEMPLATES:
        uniq, codes = sentences(agent, X, risk)
        out[agent] = np.asarray(uniq, dtype=object)[codes]
    return out, keys


def test_risk_code_matches_risk_word():
    r = np.array([0.0, 0.0199, 0.02, 0.049, 0.05, 0.149, 0.15, 0.9, np.nan])
    words = [RISK_WORDS[c] for c in risk_code(r)]
    assert words == [risk_word(float(x)) for x in r]


def test_sentences_are_deterministic_and_row_order_free():
    cohort, sources = _cohort(), _rich_sources()
    a, keys = _all_sentences(cohort, sources)
    b, _ = _all_sentences(cohort.sample(frac=1.0, random_state=4), sources)
    for agent in TEMPLATES:
        assert (a[agent] == b[agent]).all()
        assert len(a[agent]) == len(keys)


def test_rewriting_the_future_never_moves_a_past_sentence():
    cohort, sources = _cohort(), _rich_sources()
    a, keys = _all_sentences(cohort, sources)
    src2, coh2 = _rewrite_from(sources, cohort, T0)
    e = src2["ecg"]
    if e["values"].shape[1] != len(ECG_NAMES):          # the rewrite appends rows
        raise AssertionError("ECG value width changed")
    b, keys2 = _all_sentences(coh2, src2)
    past = (keys["hour"] <= T0).to_numpy()
    for agent in TEMPLATES:
        assert (a[agent][past] == b[agent][past]).all(), agent
    changed = [agent for agent in TEMPLATES if not (a[agent][~past] == b[agent][~past]).all()]
    assert {"cadence", "fluid", "resp", "ecg_meas"} <= set(changed)


def test_renderings():
    no_ecg = {"has": 0, "age": -1, "rate": -1, "wide_qrs": -1, "qtc": -1, "normal": 0,
              "risk": 0, "finding0": -1, "finding1": -1, "finding2": -1}
    assert render_ecg(no_ecg) == "ECG: none in the last 72 h. Risk: low."
    af = [i for i, (c, _) in enumerate(ECG_FINDINGS) if c == "afib"][0]
    ecg = {**no_ecg, "has": 1, "age": 1, "rate": 130, "wide_qrs": 0, "qtc": 480,
           "risk": 2, "finding0": af}
    assert render_ecg(ecg) == ("ECG 6-24 h ago: rate 130, narrow QRS, QTc 480; "
                               "atrial fibrillation. Risk: high.")
    fluid = {"has": 1, "recent": 1, "rate": 1, "oliguric": 1, "net_half_l": 3, "risk": 3}
    assert render_fluid(fluid) == ("Fluids: urine 10-20 ml/h over 6 h, oliguric; "
                                   "net balance +1.5 L over 24 h. Risk: very high.")
    resp = {"seen": 1, "level": 4, "airway": 0, "escalated": 1, "flow": 40, "fio2": 60,
            "risk": 1}
    assert render_resp(resp) == ("Breathing support: high-flow nasal cannula, 40 L/min, "
                                 "FiO2 60%; escalated in the last 6 h. Risk: moderate.")


def test_sentence_codes_index_unique_sentences():
    cohort, sources = _cohort(), _rich_sources()
    X, _, _ = build_allfeat_design(cohort, sources)
    for agent in TEMPLATES:
        uniq, codes = sentences(agent, X, np.full(len(X), 0.03))
        assert len(set(uniq)) == len(uniq)
        assert codes.min() >= 0 and codes.max() == len(uniq) - 1
        assert len(np.unique(codes)) == len(uniq)
    uniq, _ = sentences("gbm", X, np.full(len(X), 0.03))
    assert uniq == ["Tabular risk model: moderate risk."]
