"""What a rationale is allowed to name, and how we detect that it named it.

The faithfulness harness needs a **closed vocabulary of drivers**: things the
orchestrator's own structure already exposes, that a sentence can name, and that
we can switch off to see whether the fused risk actually depended on them. Two
families, both addressable as columns of the :class:`~orchestrator.fusion.AgentBank`
feature matrix:

``agent:<name>``
    One of the fused agents. Ablated by zeroing that agent's whole latent block
    plus its calibrated-risk entry.
``news2:<component>``
    One of the seven NEWS2 sub-scores -- which *are* the NEWS2 agent's latent,
    so a single column is exactly one clinical parameter. Ablated by zeroing
    that column.

**Detection is a deterministic phrase matcher over the generated prose**, not a
machine-readable list requested from the model. That is a deliberate choice: the
sentence a clinician reads is the artifact being scored, so if a driver is not
recoverable from that sentence it was not, for our purposes, named. It also
keeps the anti-gaming baselines honest -- the random and generic rationales go
through exactly the same extractor as the real ones, with no special-casing.

A phrase ending in ``*`` is a prefix match (``tachycard*`` catches
"tachycardia" and "tachycardic"); everything else is matched on word
boundaries, so "note" does not fire on "noted".
"""

import re

import numpy as np

from mimic_cohort import NEWS2_COMPONENTS, VITALS
from text_arm.summaries import (
    TREND_LAG_HOURS, VITAL_ABS_THRESHOLD, risk_word, trend_word,
)

# --- the vocabulary --------------------------------------------------------

AGENT_LABELS = {
    "vitals_tcn": "vitals-TCN", "news2": "NEWS2", "labs": "labs",
    "notes": "notes", "ecg": "ECG",
    # The flagship (all-8) additions -- T17 explains the shipped agent set.
    "ecg_meas": "ECG measurements", "gbm": "GBM risk model",
    "cadence": "charting cadence", "fluid": "fluids and urine output",
    "resp": "respiratory support",
}

COMPONENT_LABELS = {
    "resp_rate": "respiratory rate", "spo2": "SpO2", "temperature": "temperature",
    "sbp": "systolic BP", "heart_rate": "heart rate", "oxygen": "supplemental oxygen",
    "consciousness": "consciousness",
}


def driver_vocabulary(agent_names):
    """Every driver id a rationale for this configuration could name."""
    return ([f"agent:{a}" for a in agent_names]
            + [f"news2:{c}" for c in NEWS2_COMPONENTS])


# --- phrase tables ---------------------------------------------------------
# Wording a local Llama actually produces when handed these numbers. Kept
# explicit (rather than an embedding-similarity match) so that "did the
# rationale name driver X?" is reproducible and auditable.

COMPONENT_PHRASES = {
    "heart_rate": ("heart rate", "heart-rate", "hr", "pulse", "tachycard*",
                   "bradycard*"),
    "resp_rate": ("respiratory rate", "resp rate", "respiration rate", "rr",
                  "breathing rate", "tachypn*", "bradypn*"),
    "spo2": ("spo2", "sao2", "sp02", "oxygen saturation", "saturation",
             "saturations", "desatur*", "hypox*"),
    "sbp": ("systolic", "blood pressure", "bp", "hypotens*", "hypertens*"),
    "temperature": ("temperature", "temp", "fever", "febrile", "afebrile",
                    "pyrex*", "hypotherm*"),
    # NB: never a bare "oxygen" -- that is inside "oxygen saturation", which is
    # the SpO2 driver, not the supplemental-oxygen one.
    "oxygen": ("supplemental oxygen", "oxygen therapy", "oxygen requirement",
               "oxygen support", "on oxygen", "fio2", "o2 flow", "o2 device",
               "nasal cannula"),
    # "alert" is excluded on purpose: it is also the word for the thing being
    # explained, so it would fire on almost every rationale.
    "consciousness": ("consciousness", "conscious", "acvpu", "gcs",
                      "mental status", "responsive", "unresponsive", "drowsy",
                      "confus*", "obtund*", "lethargic", "sedat*"),
}

# Naming a vital sign names the vitals agent's domain; naming an analyte names
# the labs agent's. So the agent matcher inherits the component phrases rather
# than requiring the model to say the word "agent".
AGENT_PHRASES = {
    "vitals_tcn": ("vitals", "vital signs", "vital-sign", "vitals agent",
                   "vitals tcn", "vitals-tcn", "tcn"),
    "news2": ("news2", "news 2", "news-2", "news score", "early warning score"),
    "labs": ("lab", "labs", "laboratory", "lactate", "creatinine", "wbc",
             "white cell", "white blood cell", "platelet", "platelets", "bun",
             "bicarbonate", "anion gap", "urea", "renal function", "acidosis",
             "acidaemia", "acidemia", "leukocytosis", "leucocytosis"),
    "notes": ("notes", "clinical note", "nursing note", "radiology",
              "documentation", "narrative", "chart note"),
    "ecg": ("ecg", "ekg", "electrocardiogram", "electrocardiographic"),
    # NB: bare "ecg"/"ekg" belongs to ecg_meas in the flagship config -- the
    # raw-waveform agent is benched, so the words cannot collide in one bank.
    "ecg_meas": ("ecg", "ekg", "electrocardiogram", "electrocardiographic",
                 "ecg measurements", "qt", "qtc", "qrs", "pr interval",
                 "conduction"),
    "gbm": ("gbm", "gradient boost", "gradient-boost", "boosted", "xgboost",
            "risk model", "tabular model"),
    "cadence": ("cadence", "charting", "charting frequency",
                "measurement frequency", "observation frequency",
                "monitoring intensity", "monitoring frequency",
                "nursing attention"),
    "fluid": ("fluid", "fluids", "urine", "urine output", "oliguri*",
              "anuri*", "fluid balance", "diuresis", "intake and output"),
    "resp": ("respiratory support", "ventilat*", "airway support",
             "high-flow", "high flow", "bipap", "cpap", "niv",
             "non-invasive", "noninvasive"),
}

AGENT_INHERITS_COMPONENTS = {
    "vitals_tcn": ("heart_rate", "resp_rate", "spo2", "sbp", "temperature"),
    "news2": (),          # NEWS2 must be named as such, not via a parameter
    "labs": (),
    "notes": (),
    "ecg": (),
    "ecg_meas": (),
    "gbm": (),
    # naming supplemental oxygen names the resp-support agent's domain
    "resp": ("oxygen",),
    "cadence": (),
    "fluid": (),
}


def _normalise(text):
    """Lowercase, strip punctuation, pad -- so word-boundary matching is simple."""
    return " " + re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip() + " "


def _matches(normalised, phrases):
    for phrase in phrases:
        if phrase.endswith("*"):
            pattern = r"\b" + re.escape(phrase[:-1])
        else:
            pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, normalised):
            return True
    return False


def named_components(text):
    """Which of the seven NEWS2 parameters this sentence names."""
    normalised = _normalise(text)
    return {c for c in NEWS2_COMPONENTS
            if _matches(normalised, COMPONENT_PHRASES[c])}


def named_agents(text, agent_names):
    """Which fused agents this sentence names, directly or via their domain."""
    normalised = _normalise(text)
    components = named_components(text)
    out = set()
    for agent in agent_names:
        direct = _matches(normalised, AGENT_PHRASES.get(agent, ()))
        via_domain = bool(components & set(AGENT_INHERITS_COMPONENTS.get(agent, ())))
        if direct or via_domain:
            out.add(agent)
    return out


def named_drivers(text, agent_names):
    """Full set of driver ids this sentence names (agents + NEWS2 components)."""
    return ({f"agent:{a}" for a in named_agents(text, agent_names)}
            | {f"news2:{c}" for c in named_components(text)})


# --- assembling what the model actually saw --------------------------------

def _lagged_lookup(sub, rows, lag):
    """Row positions of ``hour - lag`` within the same stay, or -1 if absent.

    Looks BACKWARD only, so both trend endpoints are strictly-before-``t``
    features and the prompt cannot see the future.
    """
    import pandas as pd
    have = pd.MultiIndex.from_arrays([sub["stay_id"].to_numpy(),
                                      sub["hour"].to_numpy()])
    want = pd.MultiIndex.from_arrays([rows["stay_id"].to_numpy(),
                                      rows["hour"].to_numpy() - lag])
    return have.get_indexer(want)


def build_contexts(rows, sub, subscores, agent_risks, attention, agent_names,
                   trend_lag=TREND_LAG_HOURS):
    """One structured-driver record per sampled row.

    Parameters
    ----------
    rows : DataFrame
        The sampled patient-hours (``stay_id``, ``hour``, ``news2``, vitals).
    sub : DataFrame
        The full trajectories of those stays, used only to read the value
        ``trend_lag`` hours earlier for the trend wording.
    subscores : ndarray (n_rows, 7)
        The NEWS2 agent's latent -- its per-parameter sub-scores.
    agent_risks : dict
        ``{agent_name: array}`` of calibrated risks aligned to ``rows``.
    attention : ndarray (n_rows, n_agents)
        The orchestrator's attention over agents at each row.

    Returns
    -------
    list of dict -- everything the prompt renderer and the metrics need. The
    orchestrator's own fused risk is deliberately NOT included: see
    :mod:`explain.rationales` for why.
    """
    lagged = _lagged_lookup(sub, rows, trend_lag)
    sub_vitals = {v: sub[v].to_numpy(dtype=float) for v in VITALS}
    now_vitals = {v: rows[v].to_numpy(dtype=float) for v in VITALS}
    news2_total = rows["news2"].to_numpy(dtype=float)
    stay_ids = rows["stay_id"].to_numpy()
    hours = rows["hour"].to_numpy()

    contexts = []
    for i in range(len(rows)):
        j = lagged[i]
        trends = {}
        for v in VITALS:
            now = now_vitals[v][i]
            before = sub_vitals[v][j] if j >= 0 else np.nan
            trends[v] = {
                "now": None if np.isnan(now) else float(now),
                "before": None if (j < 0 or np.isnan(before)) else float(before),
                "word": trend_word(now, before, VITAL_ABS_THRESHOLD[v]),
            }
        contexts.append({
            "stay_id": int(stay_ids[i]),
            "hour": int(hours[i]),
            "news2_total": float(news2_total[i]),
            "subscores": {c: float(subscores[i, k])
                          for k, c in enumerate(NEWS2_COMPONENTS)},
            "agent_risks": {a: float(agent_risks[a][i]) for a in agent_names},
            "attention": {a: float(attention[i, k])
                          for k, a in enumerate(agent_names)},
            "trends": trends,
        })
    return contexts


def top_attention_agent(context):
    """The agent the fusion leaned on most at this hour."""
    return max(context["attention"], key=context["attention"].get)


def top_components(context, k=3):
    """The ``k`` highest NEWS2 sub-scores, strongest first, ignoring zeros."""
    ranked = sorted(context["subscores"].items(), key=lambda kv: -kv[1])
    return [name for name, value in ranked if value > 0][:k]


def describe_risk(context, agent):
    """Calibrated risk of one agent as a word, for prompt rendering."""
    return risk_word(context["agent_risks"][agent])
