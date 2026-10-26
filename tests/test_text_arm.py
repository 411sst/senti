"""Tests for the text-passing ablation arm.

The ablation is only meaningful if the text arm is a fair fight: it must see the
same evidence as the latent arm, be equally leakage-safe, and be deterministic
apart from the LLM call itself. These tests pin all three.
"""

import json

import numpy as np
import pandas as pd
import pytest

from text_arm.llm_orchestrator import (
    RiskCache,
    build_prompt,
    generate_risks,
    parse_risk,
    windows_for,
)
from text_arm.summaries import (
    TREND_LAG_HOURS,
    build_summaries,
    labs_summary,
    news2_summary,
    risk_word,
    trend_word,
    vitals_summary,
)


def _summaries(stay_hours):
    """Summary frame whose text encodes its own (stay, hour)."""
    rows = []
    for stay, hours in stay_hours.items():
        for h in hours:
            rows.append({"stay_id": stay, "hour": h,
                         "vitals_text": f"V s{stay} h{h}",
                         "labs_text": f"L s{stay} h{h}",
                         "news2_text": f"N s{stay} h{h}"})
    return pd.DataFrame(rows)


# --- Deterministic wording -------------------------------------------------

@pytest.mark.parametrize("risk, expected", [
    (0.0, "low"), (0.019, "low"), (0.02, "moderate"), (0.049, "moderate"),
    (0.05, "high"), (0.149, "high"), (0.15, "very high"), (0.9, "very high"),
    (float("nan"), "unknown"), (None, "unknown"),
])
def test_risk_word_bins(risk, expected):
    assert risk_word(risk) == expected


@pytest.mark.parametrize("now, before, expected", [
    (131, 94, "rising"), (94, 131, "falling"), (100, 100, "stable"),
    (105, 100, "stable"),                      # inside the 10 bpm band
    (float("nan"), 100, "unknown"), (100, None, "unknown"),
])
def test_trend_word_absolute(now, before, expected):
    assert trend_word(now, before, 10.0) == expected


@pytest.mark.parametrize("now, before, expected", [
    (2.3, 1.5, "rising"), (1.5, 2.3, "falling"), (1.6, 1.5, "stable"),
])
def test_trend_word_relative(now, before, expected):
    assert trend_word(now, before, 0.15, relative=True) == expected


def test_spo2_desaturation_is_not_described_as_stable():
    """A 97->90 desat is clinically dramatic; a relative threshold would hide it.

    This is why vitals use absolute per-channel thresholds -- otherwise the text
    arm would lose the respiratory signal and the ablation would be unfair.
    """
    from text_arm.summaries import VITAL_ABS_THRESHOLD
    assert trend_word(90.0, 97.0, VITAL_ABS_THRESHOLD["spo2"]) == "falling"


def test_vitals_summary_is_deterministic_and_readable():
    current = {"heart_rate": 131.0, "spo2": 90.0, "resp_rate": 30.0, "sbp": 104.0}
    past = {"heart_rate": 94.0, "spo2": 97.0, "resp_rate": 29.0, "sbp": 145.0}
    text = vitals_summary(current, past, risk=0.20)
    assert text == vitals_summary(current, past, risk=0.20)   # deterministic
    assert "Heart rate rising (94->131 over 4h)" in text
    assert "SpO2 falling" in text
    assert "Resp rate stable" in text
    assert text.endswith("Risk: very high.")


def test_labs_summary_names_unmeasured_analytes():
    text = labs_summary({"lab_lactate": 2.3, "lab_creatinine": float("nan")},
                        {"lab_lactate": 1.5}, risk=0.03)
    assert "Lactate 2.3 (rising)" in text
    assert "Creatinine unmeasured" in text
    assert text.endswith("Risk: moderate.")


def test_news2_summary_names_the_top_drivers():
    # sub-scores: resp_rate, spo2, temperature, sbp, heart_rate, oxygen, consciousness
    components = np.array([3, 2, 0, 0, 1, 0, 0], dtype=float)
    text = news2_summary(6, components, risk=0.06)
    assert "score 6" in text
    assert "resp rate" in text and "spo2" in text     # the two largest
    assert "temperature" not in text                  # zero contribution
    assert text.endswith("Risk: high.")


def test_news2_summary_handles_an_all_zero_score():
    text = news2_summary(0, np.zeros(7), risk=0.001)
    assert "no single parameter" in text


# --- The text arm sees the same information, from before t -----------------

def test_summaries_use_only_before_t_features_and_look_backward():
    """Trend endpoints come from hour t and t-4 of the SAME stay, never later."""
    hours = list(range(6, 16))
    cohort = pd.DataFrame({
        "stay_id": 1, "hour": hours,
        "heart_rate": np.linspace(80, 130, len(hours)),
        "resp_rate": 18.0, "spo2": 96.0, "sbp": 120.0, "temperature": 37.0,
        "news2": 4,
    })
    labs = pd.DataFrame({"lab_lactate": np.linspace(1.0, 3.0, len(hours))})
    latent = np.tile(np.array([1, 1, 0, 0, 1, 0, 1], dtype=float), (len(hours), 1))
    risks = {k: np.full(len(hours), 0.03) for k in ("vitals_tcn", "labs", "news2")}

    out = build_summaries(cohort, labs, latent, risks)
    assert len(out) == len(hours)
    # the first TREND_LAG_HOURS rows have no lagged partner -> "unknown"
    assert "unknown" in out.iloc[0]["vitals_text"]
    # later rows describe a real rise
    assert "rising" in out.iloc[-1]["vitals_text"]
    assert "rising" in out.iloc[-1]["labs_text"]


def test_summaries_do_not_cross_stays():
    cohort = pd.DataFrame({
        "stay_id": [1, 1, 2, 2], "hour": [6, 10, 6, 10],
        "heart_rate": [80.0, 90.0, 200.0, 40.0],
        "resp_rate": 18.0, "spo2": 96.0, "sbp": 120.0, "temperature": 37.0,
        "news2": 3,
    })
    labs = pd.DataFrame({"lab_lactate": [1.0, 1.0, 1.0, 1.0]})
    latent = np.zeros((4, 7))
    risks = {k: np.full(4, 0.01) for k in ("vitals_tcn", "labs", "news2")}
    out = build_summaries(cohort, labs, latent, risks, trend_lag=4)
    # stay 2 hour 6 has no earlier row of its own; it must NOT borrow stay 1's
    assert "unknown" in out[(out.stay_id == 2) & (out.hour == 6)]["vitals_text"].iloc[0]


# --- Leakage: the LLM only ever sees hours <= t ----------------------------

def test_window_contains_only_hours_up_to_t():
    frame = _summaries({1: list(range(6, 16))})
    for stay_id, hour, window in windows_for(frame, window_hours=4):
        hours_in_window = [int(r["vitals_text"].split("h")[-1]) for r in window]
        assert max(hours_in_window) == hour        # newest entry IS hour t
        assert all(h <= hour for h in hours_in_window)
        assert len(window) <= 4


def test_prompts_before_hour_k_are_unchanged_when_the_future_is_rewritten():
    """The causality test, text-arm edition: perturb from hour k, past unmoved."""
    frame = _summaries({1: list(range(6, 20))})
    base = {h: build_prompt(w) for _, h, w in windows_for(frame, 4)}

    for cut in (9, 13, 17):
        tampered = frame.copy()
        mask = tampered["hour"] >= cut
        for col in ("vitals_text", "labs_text", "news2_text"):
            tampered.loc[mask, col] = "CATASTROPHIC DETERIORATION IMMINENT"
        after = {h: build_prompt(w) for _, h, w in windows_for(tampered, 4)}

        for hour in base:
            if hour < cut:
                assert base[hour] == after[hour], \
                    f"prompt for hour {hour} changed when hour {cut}+ was rewritten"
            else:
                assert base[hour] != after[hour]     # perturbation must register


def test_prompt_orders_history_oldest_to_newest_and_labels_now():
    frame = _summaries({1: [6, 7, 8]})
    _, hour, window = list(windows_for(frame, window_hours=3))[-1]
    prompt = build_prompt(window)
    assert hour == 8
    assert prompt.index("2h ago") < prompt.index("1h ago") < prompt.index("[now]")
    assert "V s1 h8" in prompt.split("[now]")[1]     # newest block is hour 8


# --- Reply parsing ---------------------------------------------------------

@pytest.mark.parametrize("reply, expected", [
    ("42", 0.42), (" 7 ", 0.07), ("0", 0.0), ("100", 1.0),
    ("Probability: 35", 0.35), ("35%", 0.35),
])
def test_parse_risk_valid(reply, expected):
    assert parse_risk(reply) == pytest.approx(expected)


@pytest.mark.parametrize("reply", ["", None, "no idea", "-5", "150"])
def test_parse_risk_rejects_unusable(reply):
    assert np.isnan(parse_risk(reply))


# --- Cache + resume --------------------------------------------------------

def test_cache_appends_and_reloads(tmp_path):
    path = tmp_path / "risks.jsonl"
    cache = RiskCache(path)
    cache.add(1, 6, 0.3, "30")
    cache.add(1, 7, float("nan"), "junk")
    reloaded = RiskCache(path)
    assert len(reloaded) == 2
    assert (1, 6) in reloaded
    risks = reloaded.risks_for([(1, 6), (1, 7), (9, 9)])
    assert risks[0] == pytest.approx(0.3)
    assert np.isnan(risks[1])      # unparsed stays nan, never imputed
    assert np.isnan(risks[2])      # absent


def test_cache_tolerates_a_truncated_final_line(tmp_path):
    """A run killed mid-write must still resume."""
    path = tmp_path / "risks.jsonl"
    path.write_text(json.dumps({"stay_id": 1, "hour": 6, "risk": 0.2,
                                "reply": "20"}) + "\n" + '{"stay_id": 1, "ho')
    cache = RiskCache(path)
    assert len(cache) == 1
    assert (1, 6) in cache


def test_generate_skips_cached_rows(tmp_path):
    """Resuming must not re-call the model for rows already done."""
    calls = []

    class FakeRunner:
        def generate(self, prompt):
            calls.append(prompt)
            return "12"

    frame = _summaries({1: [6, 7, 8]})
    path = tmp_path / "risks.jsonl"
    generate_risks(frame, path, runner=FakeRunner(), progress_every=0)
    assert len(calls) == 3

    calls.clear()
    generate_risks(frame, path, runner=FakeRunner(), progress_every=0)
    assert calls == []               # everything already cached


def test_generate_records_unparsed_replies_without_crashing(tmp_path):
    class BadRunner:
        def generate(self, prompt):
            return "I cannot help with that."

    frame = _summaries({1: [6, 7]})
    cache = generate_risks(frame, tmp_path / "r.jsonl", runner=BadRunner(),
                           progress_every=0)
    assert len(cache) == 2
    assert np.isnan(cache.risks_for([(1, 6), (1, 7)])).all()


# --- Regression: decimal-probability replies (the "0.9" bug) -----------------

@pytest.mark.parametrize("reply, expected", [
    ("0.9", 0.9), ("0.05", 0.05), ("risk: 0.35", 0.35), ("1.0", 1.0),
    ("42", 0.42),                      # integers stay percent
])
def test_parse_risk_decimal_probabilities(reply, expected):
    assert parse_risk(reply) == pytest.approx(expected)


# --- Cache fingerprinting ------------------------------------------------------

def test_cache_rejects_a_different_fingerprint(tmp_path):
    path = tmp_path / "r.jsonl"
    cache = RiskCache(path, fingerprint={"model": "a", "window_hours": 4})
    cache.add(1, 6, 0.3, "30")
    RiskCache(path, fingerprint={"model": "a", "window_hours": 4})   # same: fine
    with pytest.raises(ValueError, match="different"):
        RiskCache(path, fingerprint={"model": "b", "window_hours": 4})


def test_cache_adopts_legacy_files_with_a_warning(tmp_path, capsys):
    path = tmp_path / "r.jsonl"
    RiskCache(path).add(1, 6, 0.3, "30")             # legacy: no meta line
    cache = RiskCache(path, fingerprint={"model": "a", "window_hours": 4})
    assert (1, 6) in cache
    assert "unknown provenance" in capsys.readouterr().out
    # and the adopted fingerprint is now enforced
    with pytest.raises(ValueError):
        RiskCache(path, fingerprint={"model": "b", "window_hours": 4})
