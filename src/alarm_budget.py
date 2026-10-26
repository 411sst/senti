"""The alarm budget: catch, cost and timing against the published-spec GBMs.

Pure helpers behind ``scripts/alarm_budget.py`` (T28). Every scorer is frozen
on V_freeze at the same alarm volume and then raced stay by stay on test:

* :func:`first_alarm_table` -- one row per stay, the first hour each scorer
  crosses its threshold (the crossing rule of
  :func:`fair_capability.paired_first_alarm_diffs`, no warm-up mask);
* :func:`h2h_summary` -- catch funnel, warning depth, the discordant race and
  the false-alarm cost at one operating point;
* :func:`budget_sweep_row` -- catch and quiet-stay false-alarm rate at one
  budget;
* :func:`stay_maxima` / :func:`catch_at_quiet_fp` -- the same two rates as a
  stay-level curve, so models can be compared at EQUAL quiet-stay cost rather
  than equal alarm volume.

Everything returned is a count, a rate or a duration; the output file must
pass :func:`assert_aggregate_only` (PhysioNet DUA).
"""

import re

import numpy as np

# The two published-spec ML baselines the head-to-head races against.
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


# --- first alarms ---------------------------------------------------------------

def first_alarm_hour(hours, scores, threshold):
    """First hour whose score crosses ``threshold``; NaN if it never does.

    Same rule as :func:`fair_capability.paired_first_alarm_diffs` uses, so a
    stay's lead here is the lead the paired race counted for it.
    """
    hours = np.asarray(hours, dtype=float)
    crossed = hours[np.asarray(scores, dtype=float) >= threshold]
    return float(crossed[0]) if crossed.size else float("nan")


def first_alarm_table(test_ordered, scores_by_name, thr_by_name):
    """One row per stay: extent, event, and ``fa_<name>`` per scorer.

    Every value in ``scores_by_name`` must be aligned to ``test_ordered``'s
    rows; the crossing rule is :func:`first_alarm_hour`'s. ``ah_<name>`` counts
    the stay's alarm-hours.
    """
    tmp = test_ordered[["stay_id", "hour", "event_h", "event_type"]].copy()
    tmp["hour"] = tmp["hour"].astype(float)
    aggs = dict(h0=("hour", "min"), h1=("hour", "max"), n_hours=("hour", "size"),
                event_h=("event_h", "first"), event_type=("event_type", "first"))
    for name, scores in scores_by_name.items():
        crossed = np.asarray(scores, dtype=float) >= thr_by_name[name]
        tmp[f"_c_{name}"] = np.where(crossed, tmp["hour"], np.nan)
        aggs[f"fa_{name}"] = (f"_c_{name}", "min")
        aggs[f"ah_{name}"] = (f"_c_{name}", "count")   # alarm-hours per stay
    out = tmp.groupby("stay_id", sort=True).agg(**aggs).reset_index()
    out["event_h"] = out["event_h"].astype(float)
    out["event_type"] = out["event_type"].where(out["event_h"].notna(), None)
    return out


# --- one operating point --------------------------------------------------------

def h2h_summary(table, flagship, opponents, min_lead_h=6.0):
    """Aggregate head-to-head block at one operating point.

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
    lead = {m: lead_stats(table, m) for m in models}

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
            "n_differ": int(len(nz)),
            "earlier_pct": float((nz > 0).mean()) if len(nz) else 0.0,
            "median_abs_gap_h": float(nz.abs().median()) if len(nz) else 0.0,
            "mean_gap_h": float(nz.mean()) if len(nz) else 0.0,
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
        if f"ah_{m}" in table:                 # how heavy is a quiet false alarm
            qa = qt[f"ah_{m}"][qt[f"ah_{m}"] > 0]
            cost.setdefault("quiet_alarm_h_mean", {})[m] = (
                float(qa.mean()) if len(qa) else 0.0)
            cost.setdefault("quiet_alarm_h_median", {})[m] = (
                float(qa.median()) if len(qa) else 0.0)

    caught_by_any = np.zeros(n_ev, dtype=bool)
    for m in models:
        caught_by_any |= caught[m].to_numpy()
    return {
        "models": models,
        "n_event_stays": n_ev, "n_quiet_stays": n_qt,
        "n_missed_by_all": int((~caught_by_any).sum()),
        "min_lead_h": float(min_lead_h),
        "funnel": funnel, "lead": lead, "race": race, "cost": cost,
    }


def lead_stats(table, name):
    """Warning depth (event_h - first alarm) over one model's any-catches.

    ``table`` is :func:`first_alarm_table` output; aggregate-only by
    construction.
    """
    ev = table[table["event_h"].notna()]
    fa = ev[f"fa_{name}"]
    lh = (ev["event_h"] - fa)[fa.notna() & (fa <= ev["event_h"])]
    return {
        "n": int(len(lh)),
        "median_h": float(lh.median()) if len(lh) else 0.0,
        "mean_h": float(lh.mean()) if len(lh) else 0.0,
        "p25_h": float(lh.quantile(0.25)) if len(lh) else 0.0,
        "p75_h": float(lh.quantile(0.75)) if len(lh) else 0.0,
    }


# --- across budgets ---------------------------------------------------------------

def budget_sweep_row(table, names):
    """Catch rate over event stays and quiet-stay FP rate per model.

    One row of the budget sweep; ``table`` is :func:`first_alarm_table` output
    at one budget's thresholds. Aggregate-only by construction.
    """
    is_event = table["event_h"].notna()
    ev, qt = table[is_event], table[~is_event]
    row = {}
    for n in names:
        fa_ev = ev[f"fa_{n}"]
        row[n] = {
            "catch": float((fa_ev.notna() & (fa_ev <= ev["event_h"])).mean())
            if len(ev) else 0.0,
            "quiet_fp": float(qt[f"fa_{n}"].notna().mean()) if len(qt) else 0.0,
        }
    return row


def stay_maxima(test_ordered, scores):
    """Per stay, the highest score a counted first alarm could come from.

    Event stays take the max over hours at or before the event (a later alarm
    is not a catch), quiet stays the max over every hour; a stay with no such
    hour gets -inf. Then, at any threshold t, "caught" is ``max >= t`` and a
    quiet false alarm is ``max >= t`` -- exactly :func:`budget_sweep_row`'s
    rates. Returns ``(stay_ids, is_event, maxima)`` sorted by stay.
    """
    t = test_ordered[["stay_id", "hour", "event_h"]]
    s = np.asarray(scores, dtype=float)
    eligible = t["event_h"].isna().to_numpy() | (
        t["hour"].to_numpy(dtype=float) <= t["event_h"].to_numpy(dtype=float))
    stay = t["stay_id"].to_numpy()
    ids, inv = np.unique(stay, return_inverse=True)
    mx = np.full(len(ids), -np.inf)
    np.maximum.at(mx, inv[eligible], s[eligible])
    is_event = np.zeros(len(ids), dtype=bool)
    is_event[inv] = t["event_h"].notna().to_numpy()
    return ids, is_event, mx


def catch_at_quiet_fp(ev_max, qt_max, targets):
    """Catch rate at each target quiet-stay false-alarm rate.

    Traces the stay-level curve (quiet FP rate, catch rate) over every
    distinct threshold and interpolates linearly between the two achievable
    points that bracket each target (score ties make most exact rates
    unattainable). Where several thresholds share one FP rate the highest
    catch is kept.
    """
    ev_max = np.asarray(ev_max, dtype=float)
    qt_max = np.asarray(qt_max, dtype=float)
    pool = np.concatenate([ev_max, qt_max])
    thr = np.unique(pool[np.isfinite(pool)])[::-1]          # descending
    ev_s, qt_s = np.sort(ev_max), np.sort(qt_max)
    catch = (len(ev_s) - np.searchsorted(ev_s, thr, "left")) / max(len(ev_s), 1)
    fp = (len(qt_s) - np.searchsorted(qt_s, thr, "left")) / max(len(qt_s), 1)
    fp = np.concatenate([[0.0], fp])
    catch = np.concatenate([[0.0], catch])
    # fp and catch are both non-decreasing along descending thresholds, so the
    # last point of each fp plateau carries its highest catch.
    last = np.r_[fp[1:] != fp[:-1], True]
    return np.interp(np.asarray(targets, dtype=float), fp[last], catch[last])


# --- the DUA gate -------------------------------------------------------------------

def assert_aggregate_only(tree, path="alarm_budget"):
    """Raise if anything patient-level could have leaked into the output.

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
