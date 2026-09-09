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

# The two published-spec ML baselines the head-to-head story races against.
COMPARATORS = ("ecart", "yaib")
COMPARATOR_LABELS = {
    "sentinel": "Sentinel two-head",
    "ecart": "eCART-style GBM (published spec)",
    "yaib": "YAIB LightGBM (published spec)",
}

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


def first_alarm_table(test_ordered, scores_by_name, thr_by_name):
    """One row per stay: extent, event, and ``fa_<name>`` per scorer.

    Generalizes the two-scorer summary to any number of models; every value in
    ``scores_by_name`` must be aligned to ``test_ordered``'s rows, and the
    crossing rule is :func:`first_alarm_hour`'s.
    """
    tmp = test_ordered[["stay_id", "hour", "event_h", "event_type"]].copy()
    tmp["hour"] = tmp["hour"].astype(float)
    aggs = dict(h0=("hour", "min"), h1=("hour", "max"), n_hours=("hour", "size"),
                event_h=("event_h", "first"), event_type=("event_type", "first"))
    for name, scores in scores_by_name.items():
        crossed = np.asarray(scores, dtype=float) >= thr_by_name[name]
        tmp[f"_c_{name}"] = np.where(crossed, tmp["hour"], np.nan)
        aggs[f"fa_{name}"] = (f"_c_{name}", "min")
    out = tmp.groupby("stay_id", sort=True).agg(**aggs).reset_index()
    out["event_h"] = out["event_h"].astype(float)
    out["event_type"] = out["event_type"].where(out["event_h"].notna(), None)
    return out


def stay_summary(test_ordered, s_model, s_news2, thr_model, thr_news2):
    """One row per stay: first alarms, event, extent.

    ``test_ordered`` must be sorted by ``(stay_id, hour)`` with ``s_model`` and
    ``s_news2`` aligned to it.
    """
    out = first_alarm_table(test_ordered,
                            {"model": s_model, "news2": s_news2},
                            {"model": thr_model, "news2": thr_news2})
    return out[["stay_id", "h0", "h1", "n_hours", "fa_model", "fa_news2",
                "event_h", "event_type"]]


def h2h_summary(table, flagship, opponents, min_lead_h=6.0):
    """Aggregate head-to-head block for ``cohort.json``.

    ``table`` is :func:`first_alarm_table` output carrying ``fa_<flagship>``
    and ``fa_<opponent>`` columns. Catch definitions, per model over the event
    stays: ``any`` = first alarm at or before the event; ``lead6`` = at least
    ``min_lead_h`` hours before it; ``midstay`` = an any-catch whose first
    alarm falls strictly after the stay's first scored hour (i.e. a change
    detection, not an arrival-profile flag). Everything returned is a count or
    a fraction -- the output must pass :func:`assert_aggregate_only`.
    """
    is_event = table["event_h"].notna()
    ev, qt = table[is_event], table[~is_event]
    n_ev, n_qt = int(len(ev)), int(len(qt))
    models = [flagship, *opponents]

    def catches(name, mode):
        fa = ev[f"fa_{name}"]
        if mode == "lead6":
            return fa.notna() & (fa <= ev["event_h"] - min_lead_h)
        any_c = fa.notna() & (fa <= ev["event_h"])
        if mode == "midstay":
            return any_c & (fa > ev["h0"])
        return any_c

    funnel = {}
    for mode in ("any", "lead6", "midstay"):
        funnel[mode] = {
            m: {"n": int(catches(m, mode).sum()),
                "pct": float(catches(m, mode).sum() / n_ev) if n_ev else 0.0}
            for m in models}

    caught = {m: catches(m, "any") for m in models}
    race = []
    for opp in opponents:
        both = caught[flagship] & caught[opp]
        n_joint = int(both.sum())
        d = (ev[f"fa_{opp}"] - ev[f"fa_{flagship}"])[both]
        wins, losses = int((d > 0).sum()), int((d < 0).sum())
        nz = d[d != 0]
        race.append({
            "opponent": opp, "n_joint": n_joint,
            "flagship_first": wins, "tie": n_joint - wins - losses,
            "opponent_first": losses,
            "flagship_first_pct": float(wins / n_joint) if n_joint else 0.0,
            "tie_pct": float((n_joint - wins - losses) / n_joint) if n_joint else 0.0,
            "opponent_first_pct": float(losses / n_joint) if n_joint else 0.0,
            "only_flagship": int((caught[flagship] & ~caught[opp]).sum()),
            "only_opponent": int((caught[opp] & ~caught[flagship]).sum()),
            "median_gap_h": float(nz.median()) if len(nz) else 0.0,
        })

    cost = {"quiet_fp_rate": {}, "stay_precision": {}}
    for m in models:
        fa_all = table[f"fa_{m}"]
        alarmed = fa_all.notna()
        valid = alarmed & (fa_all <= table["event_h"])  # NaN event -> False
        cost["quiet_fp_rate"][m] = (
            float(qt[f"fa_{m}"].notna().mean()) if n_qt else 0.0)
        cost["stay_precision"][m] = (
            float(valid.sum() / alarmed.sum()) if alarmed.any() else 0.0)

    caught_by_any = np.zeros(n_ev, dtype=bool)
    for m in models:
        caught_by_any |= caught[m].to_numpy()
    return {
        "models": models,
        "n_event_stays": n_ev, "n_quiet_stays": n_qt,
        "n_missed_by_all": int((~caught_by_any).sum()),
        "min_lead_h": float(min_lead_h),
        "funnel": funnel, "race": race, "cost": cost,
    }


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
    "comp_risk": None, "comp_alarm": None,
}
_ALIGNED_NULLABLE = ("vitals_observed", "vitals_model_view", "vitals_hrs_since")


# --- live replay ----------------------------------------------------------------

def truncate_stay(stay, n, hide_event=True):
    """Copy of a stay dict cut to its first ``n`` hours, for the live replay.

    Slices ``hours``, every aligned per-hour array, and the NEWS2 block; the
    event is hidden while it still lies more than an hour past the truncated
    window (the replay must not foreshadow it), and ``contexts``/``cards``/
    ``first_alarm`` are left whole -- the caller indexes those by hour.
    """
    t = dict(stay)
    hours = list(stay["hours"][:n])
    t["hours"] = hours
    for group in (*_ALIGNED, *_ALIGNED_NULLABLE):
        block = stay.get(group)
        if block:
            t[group] = {name: arr[:n] for name, arr in block.items()}
    t["news2"] = {"total": stay["news2"]["total"][:n],
                  "subscores": {c: arr[:n]
                                for c, arr in stay["news2"]["subscores"].items()}}
    event_h = stay.get("event_h")
    if hide_event and event_h is not None and event_h > hours[-1] + 1.0:
        t["event_h"] = None
        t["event_type"] = None
    return t


def alarm_announcements(stay, labels):
    """Sorted ``[(alarm_hour, name, message)]`` for the replay's popups.

    ``sentinel`` comes from ``first_alarm["model"]``, the comparators from
    ``comp_first_alarm``; models that never fire are omitted.
    """
    out = []
    fa = (stay.get("first_alarm") or {}).get("model")
    if fa is not None:
        out.append((float(fa), "sentinel",
                    f"{labels.get('sentinel', 'Sentinel')} — alarm at hour {fa:g}"))
    for name, h in (stay.get("comp_first_alarm") or {}).items():
        if h is not None:
            out.append((float(h), name,
                        f"{labels.get(name, name)} — alarm at hour {h:g}"))
    out.sort(key=lambda item: (item[0], item[1]))
    return out


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

        comp_fa = stay.get("comp_first_alarm")
        if comp_fa is None or set(comp_fa) != set(stay["comp_risk"]):
            raise ValueError("comp_first_alarm keys do not match comp_risk")
        for name, h in comp_fa.items():
            if h is not None and h not in hours:
                raise ValueError(f"comp_first_alarm.{name} is outside the stay")
            first = next((hr for hr, hit in zip(hours, stay["comp_alarm"][name])
                          if hit), None)
            if h != first:
                raise ValueError(f"comp_first_alarm.{name} disagrees with comp_alarm")

    comp_names = {n for stay in bundle["stays"] for n in stay["comp_risk"]}
    for name in sorted(comp_names):
        if f"thr_{name}" not in bundle["thresholds"]:
            raise ValueError(f"thresholds is missing thr_{name}")
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
