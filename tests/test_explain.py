"""Tests for the explainability layer and its faithfulness benchmark.

The properties that matter here are not "does it run" but:

* the prompt cannot leak the answer (no fused risk, no alert flag),
* the driver extractor identifies drivers from prose the way we claim,
* ablating a driver moves the fused risk and does so **causally** -- zeroing
  hours ``<= t`` must never change any hour before the earliest hour touched,
* the metrics can actually tell a faithful rationale from a shuffled or generic
  one, which is the whole reason the baselines exist.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from agents.base import KEY_COLUMNS
from explain.drivers import (
    build_contexts, driver_vocabulary, named_agents, named_components,
    named_drivers,
)
from explain.faithfulness import (
    ablated_bank, attention_baseline, attention_consistency, column_reference,
    column_spans, comprehensiveness, component_consistency, driver_columns,
    raw_risk_at, receptive_field_hours, restrict_bank, simulatability,
    sufficiency, summarise_consistency, with_bank,
)
from explain.rationales import (
    GENERIC_RATIONALE, RationaleCache, build_menu, build_menu_prompt,
    build_prompt, clean_reply, examples_for, few_shot_pool,
    generic_rationales, scrub_leak_words, shuffle_rationales,
)
from mimic_cohort import NEWS2_COMPONENTS, VITALS
from orchestrator.fusion import AgentBank, Orchestrator

AGENTS = ("vitals_tcn", "news2", "labs")
LATENT_DIMS = {"vitals_tcn": 6, "news2": len(NEWS2_COMPONENTS), "labs": 4}


# --- fixtures ---------------------------------------------------------------

def make_cohort(n_stays=8, n_hours=40, seed=0):
    """A small hour-contiguous cohort with the columns the explainer reads."""
    rng = np.random.default_rng(seed)
    frames = []
    for stay in range(1, n_stays + 1):
        hours = np.arange(n_hours)
        frame = pd.DataFrame({"stay_id": stay, "hour": hours})
        for vital in VITALS:
            frame[vital] = rng.normal(80, 10, n_hours)
        frame["news2"] = rng.integers(0, 12, n_hours)
        frame["label_24h"] = (rng.random(n_hours) < 0.1).astype(int)
        frame["label_24h_death"] = 0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def make_bank(cohort, seed=0):
    rng = np.random.default_rng(seed)
    keys = cohort[list(KEY_COLUMNS)].copy().reset_index(drop=True)
    width = sum(LATENT_DIMS.values()) + len(AGENTS)
    matrix = rng.normal(size=(len(keys), width)).astype(np.float32)
    return AgentBank(keys, matrix, dict(LATENT_DIMS), list(AGENTS))


@pytest.fixture
def fitted():
    """A tiny fitted orchestrator plus its cohort and bank."""
    cohort = make_cohort()
    bank = make_bank(cohort)
    train = cohort[cohort.stay_id <= 5]
    val = cohort[cohort.stay_id > 5]
    orch = Orchestrator(bank=bank, d_model=8, dilations=(1, 2), max_epochs=1,
                        verbose=False)
    orch.fit(train, val)
    return cohort, bank, orch


def make_context(attention, subscores, risks=None):
    return {
        "stay_id": 1, "hour": 10, "news2_total": float(sum(subscores.values())),
        "subscores": {c: float(subscores.get(c, 0)) for c in NEWS2_COMPONENTS},
        "agent_risks": risks or {a: 0.1 for a in AGENTS},
        "attention": dict(attention),
        "trends": {v: {"now": 90.0, "before": 80.0, "word": "rising"}
                   for v in VITALS},
    }


# --- driver extraction ------------------------------------------------------

def test_extractor_finds_named_parameters():
    text = ("Rising heart rate and falling SpO2 with an elevated lactate; "
            "NEWS2 still low at 4.")
    assert named_components(text) == {"heart_rate", "spo2"}
    assert named_agents(text, AGENTS) == {"vitals_tcn", "news2", "labs"}


def test_extractor_matches_word_endings_but_not_substrings():
    assert named_components("the patient is tachycardic") == {"heart_rate"}
    assert named_components("progressive hypoxaemia") == {"spo2"}
    # "noted" must not count as naming the notes agent
    assert "notes" not in named_agents("it is noted that", ("notes",))
    assert named_agents("the nursing notes describe", ("notes",)) == {"notes"}


def test_oxygen_saturation_is_spo2_not_the_oxygen_subscore():
    """A bare 'oxygen' inside 'oxygen saturation' must not fire the O2 driver."""
    assert named_components("falling oxygen saturation") == {"spo2"}
    assert named_components("on supplemental oxygen") == {"oxygen"}


def test_alert_is_not_a_consciousness_word():
    """'alert' names the thing being explained, so it cannot name a driver."""
    assert named_components("an alert was raised") == set()
    assert named_components("the patient is drowsy") == {"consciousness"}


def test_named_drivers_are_all_in_the_vocabulary():
    vocabulary = set(driver_vocabulary(AGENTS))
    text = "Rising heart rate, rising lactate, NEWS2 of 9, falling SpO2."
    assert named_drivers(text, AGENTS) <= vocabulary


def test_empty_rationale_names_nothing():
    assert named_drivers("", AGENTS) == set()


# --- the prompt cannot leak the answer -------------------------------------

def test_prompt_never_contains_the_fused_risk_or_alert_flag():
    """The leakage control for simulatability, asserted rather than assumed."""
    context = make_context({"vitals_tcn": 0.7, "news2": 0.2, "labs": 0.1},
                           {"heart_rate": 2, "spo2": 1})
    context["fused_risk"] = 0.9312               # even if a caller adds it
    context["fired"] = True
    prompt = build_prompt(context)
    assert "0.9312" not in prompt
    assert "fired" not in prompt.lower()
    assert "alert" not in prompt.lower()


def test_prompt_is_a_pure_function_of_the_context():
    context = make_context({"vitals_tcn": 0.5, "news2": 0.3, "labs": 0.2},
                           {"resp_rate": 3})
    assert build_prompt(context) == build_prompt(context)


def test_prompt_distinguishes_a_missing_trend_from_a_missing_value():
    """"unknown" written bare reads as "the vital was not measured".

    It is not -- it means the 4h-earlier endpoint is missing, which happens at
    the start of every stay. Printing it as the trend word had the model
    describing values it could plainly see.
    """
    context = make_context({"vitals_tcn": 0.5, "news2": 0.3, "labs": 0.2},
                           {"heart_rate": 2})
    context["trends"]["heart_rate"] = {"now": 94.0, "before": None,
                                       "word": "unknown"}
    context["trends"]["spo2"] = {"now": None, "before": None, "word": "unknown"}
    context["trends"]["sbp"] = {"now": 120.0, "before": 118.0, "word": "stable"}
    prompt = build_prompt(context)
    assert "heart rate 94 (no 4h history)" in prompt
    assert "SpO2 unmeasured" in prompt
    assert "systolic BP stable at 120" in prompt
    assert "unknown" not in prompt


def test_prompt_names_the_nonzero_subscores():
    context = make_context({"vitals_tcn": 0.5, "news2": 0.3, "labs": 0.2},
                           {"resp_rate": 3, "spo2": 2})
    prompt = build_prompt(context)
    assert "respiratory rate 3" in prompt
    assert "SpO2 2" in prompt
    assert "temperature" not in prompt.split("Vital-sign trend")[0]


def test_clean_reply_strips_boilerplate():
    assert clean_reply('Here is the sentence: "Rising HR."') == "Rising HR."
    assert clean_reply("  Rising   HR.  ") == "Rising HR."
    assert clean_reply(None) == ""


# --- the menu-fewshot variant ----------------------------------------------

def test_menu_lists_only_nonzero_parameters_and_top_agents():
    context = make_context({"vitals_tcn": 0.7, "news2": 0.2, "labs": 0.1},
                           {"heart_rate": 2, "spo2": 1})
    menu = build_menu(context, n_agents=2)
    assert "heart rate (sub-score 2)" in menu
    assert "SpO2 (sub-score 1)" in menu
    assert "temperature" not in menu               # zero-scored: off the menu
    assert menu.index("vitals-TCN") < menu.index("NEWS2")
    assert "labs" not in menu                      # third agent, n_agents=2
    empty = build_menu(make_context({"vitals_tcn": 0.7, "news2": 0.3}, {}))
    assert "name no vital sign" in empty


def test_menu_prompt_never_contains_the_fused_risk_or_alert_flag():
    """The variant keeps the plain prompt's leakage guarantee, examples included."""
    target = make_context({"vitals_tcn": 0.7, "news2": 0.2, "labs": 0.1},
                          {"heart_rate": 2})
    target["fused_risk"], target["fired"] = 0.9312, True
    example = make_context({"vitals_tcn": 0.5, "news2": 0.4, "labs": 0.1},
                           {"resp_rate": 3})
    example.update({"stay_id": 2, "fused_risk": 0.8765, "fired": True})
    prompt = build_menu_prompt(target, [(example, "Rising respiratory rate.")])
    for leak in ("0.9312", "0.8765", "fired", "alert"):
        assert leak not in prompt.lower()
    assert "EXAMPLE 1" in prompt and "Rising respiratory rate." in prompt
    assert prompt.index("EXAMPLE 1") < prompt.index("NOW THIS PATIENT-HOUR")
    assert prompt == build_menu_prompt(target, [(example, "Rising respiratory rate.")])


def test_plain_prompt_is_untouched_by_the_variant():
    """The three published T17 reports were generated under this exact prompt."""
    context = make_context({"vitals_tcn": 0.5, "news2": 0.3, "labs": 0.2},
                           {"resp_rate": 3})
    prompt = build_prompt(context)
    assert prompt.endswith("\n\nOne sentence naming the drivers:")
    assert "CANDIDATE MENU" not in prompt and "EXAMPLE" not in prompt


def test_few_shot_pool_keeps_only_faithful_exemplars():
    faithful = make_context({"vitals_tcn": 0.7, "news2": 0.2, "labs": 0.1},
                            {"heart_rate": 2})
    wrong_agent = dict(faithful, stay_id=2)
    hallucinated = dict(faithful, stay_id=3)
    silent = dict(faithful, stay_id=4)
    quiet = dict(faithful, stay_id=5)
    contexts = [faithful, wrong_agent, hallucinated, silent, quiet]
    texts = ["Rising heart rate drove the vitals-TCN agent.",   # keep
             "Labs carried the assessment.",                    # not top agent
             "Rising heart rate and falling SpO2 via vitals.",  # SpO2 scores 0
             "",                                                # empty
             "Vitals-TCN with a rising heart rate."]            # keep (quiet)
    fired = [True, True, True, True, False]
    pool = few_shot_pool(contexts, texts, AGENTS, fired, k_fired=2, k_quiet=2)
    assert [c["stay_id"] for c, _ in pool] == [1, 5]     # interleaved fired/quiet


def test_examples_never_include_the_row_itself():
    a = make_context({"vitals_tcn": 0.7, "news2": 0.2, "labs": 0.1}, {"heart_rate": 2})
    b, c = dict(a, stay_id=2), dict(a, stay_id=3)
    pool = [(a, "A"), (b, "B"), (c, "C")]
    assert [t for _, t in examples_for(pool, a, k=3)] == ["B", "C"]
    assert [t for _, t in examples_for(pool, dict(a, stay_id=9), k=2)] == ["A", "B"]
    prompt = build_menu_prompt(a, examples_for(pool, a))
    assert "Sentence: A" not in prompt


# --- anti-gaming baselines --------------------------------------------------

def test_shuffle_is_a_derangement():
    texts = [f"sentence {i}" for i in range(20)]
    shuffled = shuffle_rationales(texts, seed=1)
    assert sorted(shuffled) == sorted(texts)          # same sentences
    assert all(a != b for a, b in zip(texts, shuffled))   # none in place


def test_generic_names_drivers_so_it_is_a_real_adversary():
    """A generic sentence that named nothing would be trivially beaten."""
    named = named_drivers(GENERIC_RATIONALE, AGENTS)
    assert named, "the generic baseline must commit to some drivers"
    assert "news2:heart_rate" in named


def test_scrub_removes_numbers_and_risk_words():
    scrubbed = scrub_leak_words("High risk with NEWS2 of 12 and rising lactate")
    assert "12" not in scrubbed
    assert "risk" not in scrubbed.lower()
    assert "lactate" in scrubbed          # the driver survives


def test_rationale_cache_round_trips(tmp_path):
    cache = RationaleCache(tmp_path / "r.jsonl")
    cache.add(7, 3, "rising heart rate", "prompt")
    assert (7, 3) in cache
    reloaded = RationaleCache(tmp_path / "r.jsonl")
    assert reloaded.texts_for([{"stay_id": 7, "hour": 3}]) == ["rising heart rate"]
    assert reloaded.texts_for([{"stay_id": 7, "hour": 4}]) == [""]


def test_cache_ignores_sentences_written_under_other_instructions(tmp_path):
    """Editing the prompt must not silently score sentences from the old one."""
    old = RationaleCache(tmp_path / "r.jsonl", fingerprint="oldprompt")
    old.add(7, 3, "written under the old rules", "prompt")
    fresh = RationaleCache(tmp_path / "r.jsonl", fingerprint="newprompt")
    assert (7, 3) not in fresh
    assert fresh.texts_for([{"stay_id": 7, "hour": 3}]) == [""]


def test_prompt_fingerprint_changes_with_the_instructions():
    from explain.rationales import SYSTEM_PROMPT, prompt_fingerprint
    assert prompt_fingerprint(SYSTEM_PROMPT) == prompt_fingerprint(SYSTEM_PROMPT)
    assert prompt_fingerprint(SYSTEM_PROMPT) != prompt_fingerprint(
        SYSTEM_PROMPT + " and one more rule")


# --- driver -> matrix columns ----------------------------------------------

def test_driver_columns_cover_the_right_block():
    cohort = make_cohort(n_stays=2, n_hours=5)
    bank = make_bank(cohort)
    spans, risk_offset = column_spans(bank)
    columns = driver_columns(bank, "agent:news2")
    lo, hi = spans["news2"]
    assert set(columns) == set(range(lo, hi)) | {risk_offset + 1}

    only = driver_columns(bank, "news2:spo2")
    assert only == [lo + NEWS2_COMPONENTS.index("spo2")]


def test_driver_columns_returns_none_for_an_absent_agent():
    cohort = make_cohort(n_stays=2, n_hours=5)
    bank = make_bank(cohort)
    assert driver_columns(bank, "agent:ecg") is None


def test_ablation_zeroes_only_the_named_rows_and_columns():
    cohort = make_cohort(n_stays=3, n_hours=10)
    bank = make_bank(cohort)
    target = {"stay_id": 2, "hours": np.array([4, 5]), "drivers": {"news2:spo2"}}
    perturbed = ablated_bank(bank, [target])

    lo, _ = column_spans(bank)[0]["news2"]
    column = lo + NEWS2_COMPONENTS.index("spo2")
    keys = perturbed.keys
    touched = (keys.stay_id == 2) & keys.hour.isin([4, 5])
    assert np.all(perturbed.matrix[touched.to_numpy(), column] == 0)
    # everything else is untouched
    untouched = ~touched.to_numpy()
    assert np.array_equal(perturbed.matrix[untouched], bank.matrix[untouched])
    other = [c for c in range(bank.n_features) if c != column]
    assert np.array_equal(perturbed.matrix[touched.to_numpy()][:, other],
                          bank.matrix[touched.to_numpy()][:, other])


def test_ablation_does_not_mutate_the_original_bank():
    cohort = make_cohort(n_stays=2, n_hours=6)
    bank = make_bank(cohort)
    before = bank.matrix.copy()
    ablated_bank(bank, [{"stay_id": 1, "hours": np.array([2]),
                         "drivers": {"agent:labs"}}])
    assert np.array_equal(bank.matrix, before)


def test_restrict_bank_keeps_the_layout():
    cohort = make_cohort(n_stays=4, n_hours=6)
    bank = make_bank(cohort)
    small = restrict_bank(bank, [2, 3])
    assert set(small.keys.stay_id.unique()) == {2, 3}
    assert small.latent_dims == bank.latent_dims
    assert small.agent_names == bank.agent_names
    assert small.n_features == bank.n_features


# --- ablation is causal -----------------------------------------------------

def test_ablating_hours_up_to_t_never_moves_an_earlier_hour(fitted):
    """The trajectory guarantee, re-checked through the ablation path.

    Zeroing a driver over ``[t-w+1, t]`` may only change the fused risk from
    the earliest touched hour onward. If an hour before that moved, the
    perturbation would have travelled backwards in time.
    """
    cohort, bank, orch = fitted
    stay = 1
    sub = cohort[cohort.stay_id == stay]
    cut, window = 20, 6
    hours = np.arange(cut - window + 1, cut + 1)

    base = orch.raw_risk(sub)
    perturbed = ablated_bank(bank, [{"stay_id": stay, "hours": hours,
                                     "drivers": {"agent:vitals_tcn"}}])
    after = with_bank(orch, perturbed).raw_risk(sub)

    first = int(hours[0])
    assert np.allclose(base[:first], after[:first], atol=1e-6)
    assert not np.allclose(base[first:], after[first:], atol=1e-6)


def test_ablating_a_driver_changes_the_fused_risk(fitted):
    """If zeroing an agent moved nothing, comprehensiveness could not work."""
    cohort, bank, orch = fitted
    rows = cohort[(cohort.stay_id == 1) & (cohort.hour == 30)]
    sub = cohort[cohort.stay_id == 1]
    base = raw_risk_at(with_bank(orch, bank), sub, rows)
    perturbed = ablated_bank(bank, [{"stay_id": 1, "hours": np.arange(2, 31),
                                     "drivers": {"agent:vitals_tcn"}}])
    after = raw_risk_at(with_bank(orch, perturbed), sub, rows)
    assert not np.isclose(base[0], after[0])


def test_with_bank_does_not_disturb_the_original_orchestrator(fitted):
    cohort, bank, orch = fitted
    clone = with_bank(orch, restrict_bank(bank, [1]))
    assert orch.bank is bank
    assert clone.net is orch.net           # same trained weights, new input


def test_receptive_field_matches_the_shipped_settings():
    assert receptive_field_hours(3, (1, 2, 4), 2) == 29


# --- metric 1: comprehensiveness -------------------------------------------

def test_naming_the_real_drivers_beats_naming_nothing(fitted):
    """Comprehensiveness must be zero when the rationale commits to nothing."""
    cohort, bank, orch = fitted
    rows = cohort[cohort.hour == 30].reset_index(drop=True)
    sub = cohort
    named = comprehensiveness(orch, bank, sub, rows,
                              [{"agent:vitals_tcn"}] * len(rows))
    nothing = comprehensiveness(orch, bank, sub, rows, [set()] * len(rows))
    assert np.allclose(nothing["drop_window"], 0.0)
    assert np.abs(named["drop_window"]).mean() > 0


def test_window_ablation_is_at_least_as_strong_as_hour_t_alone(fitted):
    """Deleting 29 hours of a driver cannot inform the model MORE than one."""
    cohort, bank, orch = fitted
    rows = cohort[cohort.hour == 35].reset_index(drop=True)
    result = comprehensiveness(orch, bank, sub_of(cohort, rows), rows,
                               [{"agent:vitals_tcn", "agent:labs"}] * len(rows))
    assert (np.abs(result["drop_window"]).mean()
            >= np.abs(result["drop_at_t"]).mean())


def sub_of(cohort, rows):
    return cohort[cohort.stay_id.isin(set(rows.stay_id))]


def test_sufficiency_is_zero_when_everything_is_named(fitted):
    """Naming every driver wins comprehensiveness -- and loses sufficiency.

    This is what stops "list all the drivers" from gaming the benchmark.
    """
    cohort, bank, orch = fitted
    rows = cohort[cohort.hour == 30].reset_index(drop=True)
    vocabulary = driver_vocabulary(AGENTS)
    everything = sufficiency(orch, bank, cohort, rows,
                             [set(vocabulary)] * len(rows), vocabulary)
    assert np.allclose(everything["drop"], 0.0, atol=1e-6)


def test_ablation_reference_replaces_with_the_given_values():
    cohort = make_cohort(n_stays=2, n_hours=8)
    bank = make_bank(cohort)
    reference = np.arange(bank.n_features, dtype=np.float32)
    perturbed = ablated_bank(
        bank, [{"stay_id": 1, "hours": np.array([3]), "drivers": {"news2:spo2"}}],
        reference=reference)
    lo, _ = column_spans(bank)[0]["news2"]
    column = lo + NEWS2_COMPONENTS.index("spo2")
    row = ((perturbed.keys.stay_id == 1) & (perturbed.keys.hour == 3)).to_numpy()
    assert perturbed.matrix[row, column] == pytest.approx(reference[column])


def test_zero_is_not_a_neutral_reference_when_latents_are_offset():
    """The bug that made comprehensiveness come out negative on real data.

    Replacing every driver with a genuinely uninformative value should land the
    model near its *average* prediction. Zero only does that when the latents
    happen to be centred on zero -- and the real ones are not (median |train
    column mean| 0.47, max 4.10), so zeroing everything moved the mean fused
    risk from 0.499 UP to 0.763 and every "risk drop" came out negative.

    Reproduced here with an offset bank; the plain fixture cannot show it
    because its latents are standard-normal, so there zero *is* the mean.
    """
    cohort = make_cohort(n_stays=8, n_hours=40)
    bank = make_bank(cohort)
    bank.matrix = (bank.matrix + 5.0).astype(np.float32)      # off-centre, as real
    orch = Orchestrator(bank=bank, d_model=8, dilations=(1, 2), max_epochs=1,
                        verbose=False)
    orch.fit(cohort[cohort.stay_id <= 5], cohort[cohort.stay_id > 5])

    rows = cohort[cohort.hour == 35].reset_index(drop=True)
    everything = [set(driver_vocabulary(AGENTS))] * len(rows)
    average = float(orch.raw_risk(cohort).mean())

    with_mean = comprehensiveness(orch, bank, cohort, rows, everything,
                                  window=36,
                                  reference=column_reference(bank, cohort))
    with_zero = comprehensiveness(orch, bank, cohort, rows, everything,
                                  window=36, reference=None)
    assert (abs(with_mean["window"].mean() - average)
            < abs(with_zero["window"].mean() - average))


def test_column_reference_is_the_column_mean():
    cohort = make_cohort(n_stays=3, n_hours=10)
    bank = make_bank(cohort)
    reference = column_reference(bank, cohort)
    assert reference.shape == (bank.n_features,)
    assert np.allclose(reference, bank.matrix.mean(axis=0), atol=1e-5)


# --- metric 2: consistency with attention / sub-scores ---------------------

def test_attention_consistency_rewards_naming_the_attended_agent():
    contexts = [make_context({"vitals_tcn": 0.8, "news2": 0.15, "labs": 0.05},
                             {"heart_rate": 3})]
    right = attention_consistency(contexts, [{"vitals_tcn"}], AGENTS)[0]
    wrong = attention_consistency(contexts, [{"labs"}], AGENTS)[0]
    assert right["rho"] > wrong["rho"]
    assert right["top1_hit"] == 1.0 and wrong["top1_hit"] == 0.0
    assert right["mean_named"] > right["mean_unnamed"]


def test_consistency_is_undefined_when_everything_is_named():
    contexts = [make_context({"vitals_tcn": 0.8, "news2": 0.15, "labs": 0.05},
                             {"heart_rate": 3})]
    row = attention_consistency(contexts, [set(AGENTS)], AGENTS)[0]
    assert np.isnan(row["rho"])           # no contrast left to correlate
    assert row["top1_hit"] == 1.0


def test_centring_rescues_a_saturated_attention_head():
    """When one agent always dominates, raw attention cannot discriminate.

    This is the real situation: labs is the argmax on 99.7% of test hours. Both
    a real and a shuffled rationale then score a perfect top-1 hit by always
    naming it. Centring on each agent's mean asks which agent is unusually high
    *at this hour*, which does discriminate.
    """
    contexts = [
        make_context({"vitals_tcn": 0.20, "news2": 0.15, "labs": 0.65}, {}),
        make_context({"vitals_tcn": 0.30, "news2": 0.15, "labs": 0.55}, {}),
    ]
    baseline = attention_baseline(contexts, AGENTS)
    always_labs = [{"labs"}, {"labs"}]
    reads_the_row = [{"labs"}, {"vitals_tcn"}]

    raw_a = summarise_consistency(
        attention_consistency(contexts, always_labs, AGENTS))
    raw_b = summarise_consistency(
        attention_consistency(contexts, reads_the_row, AGENTS))
    assert raw_a["top1_hit"] == 1.0 and raw_a["top1_hit"] > raw_b["top1_hit"]

    cen_a = summarise_consistency(
        attention_consistency(contexts, always_labs, AGENTS, baseline=baseline))
    cen_b = summarise_consistency(
        attention_consistency(contexts, reads_the_row, AGENTS, baseline=baseline))
    assert cen_b["top1_hit"] > cen_a["top1_hit"]


def test_component_consistency_reports_a_hallucination_rate():
    """Naming a parameter whose sub-score is zero is inventing a finding."""
    contexts = [make_context({"vitals_tcn": 0.5, "news2": 0.3, "labs": 0.2},
                             {"resp_rate": 3})]
    honest = component_consistency(contexts, [{"resp_rate"}])[0]
    invented = component_consistency(contexts, [{"temperature"}])[0]
    both = component_consistency(contexts, [{"resp_rate", "temperature"}])[0]
    assert honest["unsupported"] == 0.0
    assert invented["unsupported"] == 1.0
    assert both["unsupported"] == pytest.approx(0.5)
    assert np.isnan(component_consistency(contexts, [set()])[0]["unsupported"])


def test_component_consistency_uses_the_subscores():
    contexts = [make_context({"vitals_tcn": 0.5, "news2": 0.3, "labs": 0.2},
                             {"resp_rate": 3, "spo2": 1})]
    right = component_consistency(contexts, [{"resp_rate"}])[0]
    wrong = component_consistency(contexts, [{"temperature"}])[0]
    assert right["top1_hit"] == 1.0 and wrong["top1_hit"] == 0.0
    assert right["rho"] > wrong["rho"]


def test_summarise_skips_undefined_correlations():
    rows = [{"rho": np.nan, "top1_hit": 1.0, "mean_named": 0.5,
             "mean_unnamed": np.nan, "n_named": 3},
            {"rho": 0.6, "top1_hit": 0.0, "mean_named": 0.2,
             "mean_unnamed": 0.4, "n_named": 1}]
    summary = summarise_consistency(rows)
    assert summary["rho"] == pytest.approx(0.6)
    assert summary["n_rho"] == 1 and summary["n_rows"] == 2


# --- metric 3: simulatability ----------------------------------------------

def test_simulatability_detects_a_signal_carrying_rationale():
    texts = (["rising heart rate with falling saturations"] * 20
             + ["all parameters within normal limits"] * 20)
    alerts = [1] * 20 + [0] * 20
    assert simulatability(texts, alerts)["auroc"] > 0.9


def test_simulatability_is_chance_on_a_generic_rationale():
    """The GENERIC baseline must score at chance, or the metric is broken."""
    alerts = [1] * 20 + [0] * 20
    result = simulatability(generic_rationales(40), alerts)
    assert result["auroc"] == pytest.approx(0.5, abs=1e-9)
    assert result["accuracy"] == pytest.approx(result["majority"])


def test_simulatability_falls_apart_when_rationales_are_shuffled():
    texts = (["rising heart rate with falling saturations"] * 20
             + ["all parameters within normal limits"] * 20)
    alerts = np.array([1] * 20 + [0] * 20)
    rng = np.random.default_rng(0)
    scrambled = [texts[i] for i in rng.permutation(len(texts))]
    real = simulatability(texts, alerts)["auroc"]
    broken = simulatability(scrambled, alerts)["auroc"]
    assert real > broken + 0.2


def test_simulatability_is_nan_when_only_one_class_is_present():
    result = simulatability(["a sentence"] * 10, [1] * 10)
    assert np.isnan(result["auroc"])


def test_simulatability_survives_fully_scrubbed_text():
    """Scrubbing can empty every sentence; that must report nan, not crash."""
    texts = ["high risk"] * 10 + ["low risk"] * 10
    scrubbed = [scrub_leak_words(t) for t in texts]
    result = simulatability(scrubbed, [1] * 10 + [0] * 10)
    assert np.isnan(result["auroc"])


# --- is a gap between two arms real? ---------------------------------------

def test_bootstrap_gap_brackets_a_real_difference():
    from explain.faithfulness import bootstrap_gap
    rng = np.random.default_rng(0)
    better = rng.normal(0.5, 0.1, 300)
    worse = rng.normal(0.2, 0.1, 300)
    gap = bootstrap_gap(better, worse, reps=500)
    assert gap["lo"] > 0 and gap["p_gt0"] > 0.99
    assert gap["lo"] < 0.3 < gap["hi"]


def test_bootstrap_gap_includes_zero_when_arms_are_the_same():
    from explain.faithfulness import bootstrap_gap
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.4, 0.1, 200), rng.normal(0.4, 0.1, 200)
    gap = bootstrap_gap(a, b, reps=500)
    assert gap["lo"] < 0 < gap["hi"]


def test_bootstrap_gap_ignores_undefined_rows():
    """Rows where a correlation was undefined must not poison the interval."""
    from explain.faithfulness import bootstrap_gap
    a = np.array([1.0, np.nan, 1.0, 1.0] * 25)
    b = np.zeros(100)
    gap = bootstrap_gap(a, b, reps=300)
    assert np.isfinite(gap["mean"]) and gap["lo"] > 0


def test_bootstrap_auroc_gap_is_paired():
    from explain.faithfulness import bootstrap_auroc_gap
    y = np.array([0] * 50 + [1] * 50)
    good = y + np.random.default_rng(0).normal(0, 0.2, 100)
    noise = np.random.default_rng(1).normal(0, 1, 100)
    gap = bootstrap_auroc_gap(y, good, noise, reps=300)
    assert gap["lo"] > 0 and gap["p_gt0"] > 0.95


def test_simulatability_returns_predictions_for_bootstrapping():
    texts = ["rising heart rate"] * 20 + ["normal limits"] * 20
    result = simulatability(texts, [1] * 20 + [0] * 20)
    assert len(result["proba"]) == 40 and len(result["y"]) == 40


# --- context assembly -------------------------------------------------------

def test_build_contexts_reads_trends_backward_only():
    """The 4h trend endpoint must come from hour t-4, never t+4."""
    cohort = make_cohort(n_stays=1, n_hours=20)
    cohort.loc[cohort.hour == 6, "heart_rate"] = 60.0
    cohort.loc[cohort.hour == 10, "heart_rate"] = 100.0
    cohort.loc[cohort.hour == 14, "heart_rate"] = 999.0
    rows = cohort[cohort.hour == 10].reset_index(drop=True)
    contexts = build_contexts(rows, cohort, np.zeros((1, 7)),
                              {a: np.array([0.1]) for a in AGENTS},
                              np.full((1, 3), 1 / 3), AGENTS)
    trend = contexts[0]["trends"]["heart_rate"]
    assert trend["before"] == pytest.approx(60.0)
    assert trend["now"] == pytest.approx(100.0)
    assert trend["word"] == "rising"


def test_build_contexts_marks_a_missing_history_as_unknown():
    cohort = make_cohort(n_stays=1, n_hours=20)
    rows = cohort[cohort.hour == 1].reset_index(drop=True)
    contexts = build_contexts(rows, cohort, np.zeros((1, 7)),
                              {a: np.array([0.1]) for a in AGENTS},
                              np.full((1, 3), 1 / 3), AGENTS)
    assert contexts[0]["trends"]["heart_rate"]["word"] == "unknown"


def test_build_contexts_exposes_subscores_and_attention_by_name():
    cohort = make_cohort(n_stays=1, n_hours=12)
    rows = cohort[cohort.hour == 8].reset_index(drop=True)
    subscores = np.arange(7, dtype=float).reshape(1, 7)
    contexts = build_contexts(rows, cohort, subscores,
                              {a: np.array([0.2]) for a in AGENTS},
                              np.array([[0.5, 0.3, 0.2]]), AGENTS)
    context = contexts[0]
    assert context["subscores"]["resp_rate"] == 0.0
    assert context["subscores"]["consciousness"] == 6.0
    assert context["attention"]["vitals_tcn"] == pytest.approx(0.5)


# --- row sampling -----------------------------------------------------------

def load_script():
    """Import scripts/run_explainability.py as a module for its sampling logic."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_explainability.py"
    spec = importlib.util.spec_from_file_location("run_explainability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sampling_filters_fired_and_quiet_rows_identically():
    """The eligibility rule must not differ between the two groups.

    If fired rows were allowed at the start of a stay while quiet rows were
    not, every fired rationale would say "no 4h history" and the surrogate
    would score well by reading that artifact rather than any driver.
    """
    script = load_script()
    cohort = make_cohort(n_stays=12, n_hours=40)
    ordered = cohort.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    # these stays start at hour 0, so the eligibility floor is hour 5:
    # stays 1-6 spike at 2 and 3 (too early to have a trend) and again at 25.
    risk = np.where((ordered.stay_id <= 6) & ordered.hour.isin([2, 3, 25]), 0.9, 0.1)

    rows = script.sample_rows(ordered, risk, 0.5, n_alerts=6, n_null=6, seed=1)
    start = ordered.groupby("stay_id")["hour"].min()
    floor = rows.stay_id.map(start) + 4 + 1
    assert (rows["hour"] >= floor).all(), "an ineligible hour was sampled"
    # the early spikes at hours 2/3 are ineligible, so hour 25 must be chosen
    assert set(rows[rows["_fired"]]["hour"]) == {25}


def test_sampling_takes_one_row_per_stay_from_disjoint_stays():
    """One row per stay is what makes each row its own CV group."""
    script = load_script()
    cohort = make_cohort(n_stays=10, n_hours=40)
    ordered = cohort.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    risk = np.where((ordered.stay_id <= 5) & (ordered.hour > 20), 0.9, 0.1)
    rows = script.sample_rows(ordered, risk, 0.5, n_alerts=5, n_null=5, seed=3)
    assert len(rows) == rows.stay_id.nunique() == 10
    fired_stays = set(rows[rows["_fired"]].stay_id)
    quiet_stays = set(rows[~rows["_fired"]].stay_id)
    assert not (fired_stays & quiet_stays)


def test_sampling_takes_the_first_eligible_alarm_not_the_last():
    script = load_script()
    cohort = make_cohort(n_stays=3, n_hours=40)
    ordered = cohort.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    risk = np.where(ordered.hour.isin([15, 30]), 0.9, 0.1)
    rows = script.sample_rows(ordered, risk, 0.5, n_alerts=3, n_null=0, seed=0)
    assert set(rows["hour"]) == {15}


def test_torch_stays_deterministic_for_the_ablation(fitted):
    """Two identical ablation passes must give identical numbers."""
    cohort, bank, orch = fitted
    rows = cohort[cohort.hour == 25].reset_index(drop=True)
    kwargs = dict(orch=orch, bank=bank, sub=cohort, rows=rows,
                  driver_sets=[{"agent:labs"}] * len(rows))
    first = comprehensiveness(**kwargs)
    second = comprehensiveness(**kwargs)
    assert np.array_equal(first["drop_window"], second["drop_window"])
    assert torch.is_grad_enabled()      # we never left grad mode disabled


def test_flagship_agents_have_label_and_phrase_coverage():
    """T17 explains the all-8 flagship: every agent in that config must be
    nameable in a rationale (label for the prompt, phrases for the extractor),
    and its own phrases must actually trigger the extractor."""
    from explain.drivers import AGENT_INHERITS_COMPONENTS, AGENT_LABELS, AGENT_PHRASES
    from orchestrator.fusion import ORCHESTRATOR_CONFIGS

    flagship = ORCHESTRATOR_CONFIGS["all-8"]
    for agent in flagship:
        assert agent in AGENT_LABELS, f"{agent} has no prompt label"
        assert AGENT_PHRASES.get(agent), f"{agent} has no extractor phrases"
        assert agent in AGENT_INHERITS_COMPONENTS, agent
        phrase = AGENT_PHRASES[agent][0].replace("*", "ing")
        assert agent in named_agents(f"driven by the {phrase}", flagship), \
            f"{agent}'s own first phrase does not trigger the extractor"
