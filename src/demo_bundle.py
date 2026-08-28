"""Pure helpers behind the demo dashboard bundle.

Everything the Streamlit app and the leave-behind exporter share lives here,
deliberately free of torch, the agent bank and the cohort parquet, so the app
starts in a couple of seconds and the tests need no data.

Two files come out of ``scripts/build_demo_bundle.py``:

* ``bundle.json`` -- patient-level (curated test stays, hour by hour). It stays
  under the gitignored ``data/`` tree and is only ever rendered on the analysis
  laptop (PhysioNet DUA).
* ``cohort.json`` -- aggregate-only, the material the hosted leave-behind may
  carry. :func:`assert_aggregate_only` is the gate between the two.
"""

import json
import math
import re
import urllib.request

import numpy as np
import pandas as pd

ROLES = ("vent-model-first", "vaso-model-first", "news2-first", "quiet")
DEATH_ROLE = "death-model-first"

# Keys that can only exist on patient-level material.
ID_KEYS = frozenset({"stay_id", "stay_ids", "patient_id", "subject_id", "hour"})
# MIMIC-IV stay ids are 8-digit integers starting with 3.
_STAY_ID = re.compile(r"\b3\d{7}\b")
_ID_WORDS = ("stay_id", "patient_id", "subject_id")

EVENT_WORD = {"vent": "Ventilation", "vaso": "Vasopressors", "death": "Death"}


# --- first alarms and the per-stay summary -----------------------------------

def first_alarm_hour(hours, scores, threshold):
    """First hour whose score crosses ``threshold``; NaN if it never does.

    Same rule as :func:`fair_capability.paired_first_alarm_diffs` uses, so the
    lead the demo shows for a stay is the lead the paper counted for it.
    """
    hours = np.asarray(hours, dtype=float)
    crossed = hours[np.asarray(scores, dtype=float) >= threshold]
    return float(crossed[0]) if crossed.size else float("nan")


def stay_summary(test_ordered, s_model, s_news2, thr_model, thr_news2):
    """One row per stay: first alarms, event, extent.

    ``test_ordered`` must be sorted by ``(stay_id, hour)`` with ``s_model`` and
    ``s_news2`` aligned to it.
    """
    tmp = test_ordered[["stay_id", "hour", "event_h", "event_type"]].copy()
    tmp["hour"] = tmp["hour"].astype(float)
    m = np.asarray(s_model, dtype=float) >= thr_model
    n2 = np.asarray(s_news2, dtype=float) >= thr_news2
    tmp["m_cross"] = np.where(m, tmp["hour"], np.nan)
    tmp["n2_cross"] = np.where(n2, tmp["hour"], np.nan)
    g = tmp.groupby("stay_id", sort=True)
    out = g.agg(h0=("hour", "min"), h1=("hour", "max"), n_hours=("hour", "size"),
                fa_model=("m_cross", "min"), fa_news2=("n2_cross", "min"),
                event_h=("event_h", "first"),
                event_type=("event_type", "first")).reset_index()
    out["event_h"] = out["event_h"].astype(float)
    out["event_type"] = out["event_type"].where(out["event_h"].notna(), None)
    return out


def select_demo_stays(summary, seed=42, pick_index=0, with_death=False,
                      max_hours=120):
    """Deterministic curated stays, one per role.

    model-first roles: both alarms at or before the event, the model at least
    2 h ahead, at least 12 h of stay before the event, at least 5 h of stay
    before the model's alarm (so the alarm hour carries a 4 h trend, the same
    eligibility rule the T17 sampler uses), the alarm inside the 24 h label
    horizon, and a plottable length. ``news2-first`` mirrors it; ``quiet`` is a
    stay with no event and no alarm from either scorer.

    Ranking inside a pool is deterministic and stated: model-first roles take
    the clearest win (largest lead, stay id as tie-break); ``news2-first`` takes
    the most typical loss (closest to the pool's median lead); ``quiet`` is a
    seeded draw. ``pick_index`` walks each ranking so a rehearsal can swap an
    unphotogenic stay. The label the app shows always states the lead, so a
    showcase win is presented as exactly that.
    """
    s = summary
    lead = s["fa_news2"] - s["fa_model"]
    both_before = (s["event_h"].notna() & s["fa_model"].notna()
                   & s["fa_news2"].notna()
                   & (s["fa_model"] <= s["event_h"])
                   & (s["fa_news2"] <= s["event_h"])
                   & (s["event_h"] - s["h0"] >= 12)
                   & (s["n_hours"] <= max_hours))
    model_first = (both_before & (lead >= 2) & (s["fa_model"] - s["h0"] >= 5)
                   & (s["event_h"] - s["fa_model"] <= 24))
    news2_first = (both_before & (lead <= -2) & (s["fa_news2"] - s["h0"] >= 5)
                   & (s["event_h"] - s["fa_news2"] <= 24))
    quiet = (s["event_h"].isna() & s["fa_model"].isna() & s["fa_news2"].isna()
             & s["n_hours"].between(24, 72))

    pools = {
        "vent-model-first": s[model_first & (s["event_type"] == "vent")],
        "vaso-model-first": s[model_first & (s["event_type"] == "vaso")],
        "news2-first": s[news2_first],
        "quiet": s[quiet],
    }
    if with_death:
        pools[DEATH_ROLE] = s[model_first & (s["event_type"] == "death")]

    rng = np.random.default_rng(seed)
    chosen = {}
    for role, pool in pools.items():
        if len(pool) == 0:
            raise ValueError(f"no candidate stay for role {role!r}")
        pool = pool.assign(_lead=pool["fa_news2"] - pool["fa_model"])
        if role == "quiet":
            ranked = rng.permutation(np.sort(pool["stay_id"].to_numpy()))
        elif role == "news2-first":
            centre = float(pool["_lead"].median())
            ranked = pool.assign(_d=(pool["_lead"] - centre).abs()).sort_values(
                ["_d", "stay_id"])["stay_id"].to_numpy()
        else:
            ranked = pool.sort_values(["_lead", "stay_id"],
                                      ascending=[False, True])["stay_id"].to_numpy()
        chosen[role] = int(ranked[pick_index % len(ranked)])
    return chosen


# --- arithmetic ----------------------------------------------------------------

def t_head_from_composite(composite, d_head):
    """Recover the T-head from ``composite = 1 - (1-T)(1-D)``; clipped to [0, 1]."""
    comp = np.asarray(composite, dtype=float)
    d = np.minimum(np.asarray(d_head, dtype=float), 1.0 - 1e-12)
    return np.clip(1.0 - (1.0 - comp) / (1.0 - d), 0.0, 1.0)


def stay_label(stay):
    """Selectbox label: what happened and who alarmed first, never the id."""
    unit = f"{stay['unit_group']} ICU"
    event = stay.get("event_type")
    if event is None:
        return f"No event — no alarm from either · {unit}"
    word = EVENT_WORD.get(event, "Event")
    lead = (stay.get("first_alarm") or {}).get("lead_h")
    if lead is None:
        timing = "alarms not both before the event"
    elif lead > 0:
        timing = f"model alarm {lead:g} h before NEWS2"
    elif lead < 0:
        timing = f"NEWS2 alarm {-lead:g} h before the model"
    else:
        timing = "model and NEWS2 alarm in the same hour"
    return f"{word} — {timing} · {unit}"


# --- JSON hygiene ----------------------------------------------------------------

def nan_to_none(obj):
    """Recursively make ``obj`` JSON-native: NaN -> None, numpy -> Python."""
    if isinstance(obj, dict):
        return {str(k): nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [nan_to_none(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [nan_to_none(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    return obj


def dumps_strict(obj):
    """JSON text with NaN rejected -- a stray NaN fails the build, not the page."""
    return json.dumps(nan_to_none(obj), allow_nan=False, indent=1)


_ALIGNED = {
    "risk": ("composite", "t_head", "d_head", "single_head"),
    "agent_risk": None, "attention": None,
    "alarm": ("model", "news2"),
}
_ALIGNED_NULLABLE = ("vitals_observed", "vitals_model_view", "vitals_hrs_since")


def validate_bundle(bundle):
    """Raise ``ValueError`` on a bundle the app could not render safely."""
    for key in ("agents", "news2_components", "thresholds", "attention_source",
                "stays"):
        if key not in bundle:
            raise ValueError(f"bundle is missing {key!r}")
    if not bundle["stays"]:
        raise ValueError("bundle has no stays")
    for stay in bundle["stays"]:
        hours = stay.get("hours")
        if not hours:
            raise ValueError("a stay has no hours")
        n = len(hours)

        def check(name, arr, nullable=False):
            if len(arr) != n:
                raise ValueError(f"{name}: {len(arr)} values for {n} hours")
            for v in arr:
                if v is None and nullable:
                    continue
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    raise ValueError(f"{name}: missing value")

        for group, names in _ALIGNED.items():
            block = stay.get(group)
            if block is None:
                raise ValueError(f"stay is missing {group!r}")
            for name in (names or block.keys()):
                check(f"{group}.{name}", block[name])
        check("news2.total", stay["news2"]["total"])
        for c, arr in stay["news2"]["subscores"].items():
            check(f"news2.subscores.{c}", arr)
        for group in _ALIGNED_NULLABLE:
            for name, arr in stay[group].items():
                check(f"{group}.{name}", arr, nullable=True)
        hour_keys = {str(h) for h in hours}
        for group in ("contexts", "cards"):
            extra = set(stay[group]) - hour_keys
            if extra:
                raise ValueError(f"{group} has hours outside the stay: {sorted(extra)[:3]}")
    return True


def assert_aggregate_only(tree, path="cohort"):
    """Raise if anything patient-level could have leaked into the cohort file.

    Walks dicts and lists; forbids the id keys and any value that looks like a
    MIMIC-IV stay id (8 digits starting with 3). ``git_commit`` values are
    skipped because a short hash can, in principle, be all digits.
    """
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k in ID_KEYS:
                raise ValueError(f"patient-level key {k!r} at {path}")
            if k == "git_commit":
                continue
            assert_aggregate_only(v, f"{path}.{k}")
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            assert_aggregate_only(v, f"{path}[{i}]")
    elif isinstance(tree, bool):
        return True
    elif isinstance(tree, (int, np.integer)):
        if 30_000_000 <= int(tree) <= 39_999_999:
            raise ValueError(f"value at {path} looks like a stay id")
    elif isinstance(tree, str):
        if _STAY_ID.search(tree):
            raise ValueError(f"string at {path} contains a stay-id-like number")
    return True


def assert_no_identifiers(html):
    """The leave-behind page must not carry ids or id-like numbers."""
    low = html.lower()
    for word in _ID_WORDS:
        if word in low:
            raise ValueError(f"leave-behind contains {word!r}")
    if _STAY_ID.search(html):
        raise ValueError("leave-behind contains a stay-id-like number")
    return True


# --- runtime -----------------------------------------------------------------------

def ollama_alive(url="http://localhost:11434/api/tags", timeout=1.0):
    """True when a local ollama server answers; the app falls back otherwise."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False
