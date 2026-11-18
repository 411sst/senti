"""Data-feed outages at the source (T42).

An outage deletes one feed's raw records for the affected stays BEFORE any
feature is built. Every system then rebuilds its inputs with its own,
unchanged feature code, so carry-forward limits, staleness bounds and
missingness flags behave as they would in a real outage, and nothing
downstream can see a deleted record.

The six feeds follow the hourly extracts (results_fusion.md, T42 mapping):

=======  ==============================================  =========================
feed     raw records deleted                              where they live
=======  ==============================================  =========================
vitals   HR, RR, SpO2, SBP, temperature; GCS total,      cohort extract
         components and "unable" flag                     (``mimic_raw_hourly``)
labs     every row (24 analytes, binned by storetime)     ``mimic_labs_hourly``
ecg      every ECG index row (study -> stay, hour)        ``ecg_index``
cadence  every row (orders, draws, chart rows, cultures)  ``mimic_cadence_hourly``
fluid    every row (urine, other output, intake)          ``mimic_fluids_hourly``
resp     every row (device, O2 flow, FiO2), and the       ``mimic_resp_hourly`` and
         cohort extract's O2 device, O2 flow and FiO2     the cohort extract
=======  ==============================================  =========================

Hourly tables lose rows. The cohort extract's rows ARE the hourly grid, so
there a missing record is a blank channel (NaN), which is exactly what an
hour without that chart entry looks like in the extract.

Scenario A deletes the feed for the whole stay, look-back hours included.
Scenario B deletes records stamped in ``[start, start + 24)``; the hours whose
inputs fall in that window are ``start + 1 .. start + 24`` (every builder reads
records strictly before ``t``).
"""

import numpy as np
import pandas as pd

from mimic_cohort import (
    CARRIED,
    EVENT_TYPES,
    LABEL,
    PATIENT_ID,
    VITALS,
    add_event_labels,
    build_before_t_features,
    compute_news2,
)

# Stay id given to detached ECG records; no cohort stay has it.
DETACHED_STAY = -1
FEEDS = ("vitals", "labs", "ecg", "cadence", "fluid", "resp")
WINDOW_HOURS = 24

# Cohort-extract channels owned by a feed (blanked, not row-deleted).
COHORT_CHANNELS = {
    "vitals": list(VITALS) + ["gcs", "gcs_verbal", "gcs_motor", "gcs_eyes",
                              "gcs_unable"],
    "resp": ["o2_flow", "o2_device", "fio2"],
}
# Hourly tables owned by a feed (rows deleted), with their time column.
FEED_TABLES = {"labs": "labs", "ecg": "ecg_index", "cadence": "cadence",
               "fluid": "fluid", "resp": "resp"}
TIME_COLUMN = {"ecg_index": "ecg_hour"}
SOURCE_NAMES = ("cohort_raw", "labs", "ecg_index", "cadence", "fluid", "resp")

ALL8 = ("vitals_tcn", "news2", "labs", "ecg_meas", "gbm", "cadence",
        "fluid", "resp")
# Agents that read anything from a feed (the T42 mapping). The resp feed
# reaches vitals_tcn, NEWS2 and the GBM agent through ``on_oxygen``.
FEED_AGENTS = {
    "vitals": ("vitals_tcn", "news2", "gbm"),
    "labs": ("labs", "gbm"),
    "ecg": ("ecg_meas",),
    "cadence": ("cadence",),
    "fluid": ("fluid",),
    "resp": ("resp", "vitals_tcn", "news2", "gbm"),
}

# The cohort columns the exported table keeps (mimic_cohort.build, minus split).
COHORT_KEEP = ([PATIENT_ID, "stay_id", "hour"] + list(VITALS)
               + [f"{v}_hrs_since" for v in VITALS]
               + ["gcs", "gcs_unable", "is_alert", "on_oxygen", "news2",
                  "event_h", "event_type", LABEL]
               + [f"{LABEL}_{n}" for n in EVENT_TYPES])


def fallback_agents(feed):
    """ALL-8 minus every agent that reads ``feed``, in ALL-8 order."""
    return tuple(a for a in ALL8 if a not in FEED_AGENTS[feed])


# --- which GBM columns each feed owns ------------------------------------------

def feed_columns(feed, ecg_names):
    """The all-features design columns built from ``feed``'s records."""
    from allfeat_gbm import NEWS2_SUBSCORES, ecg_columns
    from baselines_fair import LAB_FEATURES, TRAJ_STATS, TRAJ_WINDOWS
    from modality_features import (CADENCE_FEATURES, FLUID_FEATURES,
                                   LAB_COLUMNS, RESP_FEATURES)
    if feed == "vitals":
        traj = [f"{c}_{s}_{w}h" for c in VITALS for w in TRAJ_WINDOWS
                for s in TRAJ_STATS]
        return (list(VITALS) + [f"{v}_hrs_since" for v in VITALS]
                + ["gcs", "gcs_unable"] + traj
                + [n for n in NEWS2_SUBSCORES if n != "news2_oxygen"])
    if feed == "labs":
        return list(LAB_FEATURES) + [f"{c}_delta_24h" for c in LAB_COLUMNS]
    if feed == "ecg":
        return ecg_columns(ecg_names)
    if feed == "cadence":
        return list(CADENCE_FEATURES)
    if feed == "fluid":
        return list(FLUID_FEATURES)
    if feed == "resp":
        return list(RESP_FEATURES) + ["on_oxygen", "news2_oxygen"]
    raise ValueError(f"unknown feed {feed!r}")


def fallback_columns(feed, all_columns, ecg_names):
    """``all_columns`` without ``feed``'s columns, order kept."""
    drop = set(feed_columns(feed, ecg_names))
    return [c for c in all_columns if c not in drop]


# --- deleting records ----------------------------------------------------------

def outage_mask(stay, time, stays, starts=None, length=WINDOW_HOURS):
    """Records to delete: rows of ``stays`` (all of them when ``starts`` is
    None, else those stamped in ``[start, start + length)`` of their stay).

    ``starts`` is a Series indexed by stay id.
    """
    stay = np.asarray(stay, dtype="int64")
    hit = np.isin(stay, np.asarray(stays, dtype="int64"))
    if starts is None:
        return hit
    s = pd.Series(starts).reindex(stay).to_numpy(dtype=float)
    t = np.asarray(time, dtype=float)
    with np.errstate(invalid="ignore"):
        return hit & (t >= s) & (t < s + length)


def apply_outage(sources, feed, stays, starts=None, length=WINDOW_HOURS):
    """A copy of ``sources`` with ``feed``'s records deleted for ``stays``.

    Tables of every other feed are returned as the same objects, untouched.

    ECG index rows are detached (stay id set to ``DETACHED_STAY``, which no
    cohort stay has) rather than dropped. For every cohort row that is the
    same as deleting them: the builder joins by stay, so no row can reach a
    detached ECG. Dropping would also change the builder's sort: it orders
    the whole index by ``ecg_hour`` with an unstable sort, and that order
    breaks ties between two ECGs of one stay stamped at the same hour. So
    dropping a record can change which tied ECG another hour uses. Detaching
    leaves the ``ecg_hour`` array, and with it the order, exactly as it was.
    """
    if feed not in FEEDS:
        raise ValueError(f"unknown feed {feed!r}")
    out = dict(sources)
    if feed in FEED_TABLES:
        name = FEED_TABLES[feed]
        table = sources[name]
        tcol = TIME_COLUMN.get(name, "hour")
        drop = outage_mask(table["stay_id"], table[tcol], stays, starts, length)
        if name == "ecg_index":
            table = table.copy()
            table["stay_id"] = table["stay_id"].where(~drop, DETACHED_STAY)
            out[name] = table
        else:
            out[name] = table.loc[~drop]
    if feed in COHORT_CHANNELS:
        raw = sources["cohort_raw"].copy()
        blank = outage_mask(raw["stay_id"], raw["hour"], stays, starts, length)
        for col in COHORT_CHANNELS[feed]:
            raw[col] = raw[col].where(~blank)        # NaN / <NA> by dtype
        out["cohort_raw"] = raw
    return out


# --- scenario B windows ----------------------------------------------------------

def draw_windows(keys, seed=42):
    """One outage start per stay, uniform over that stay's scored hours.

    Stays are visited in ascending id order and each draws from one generator
    seeded with ``seed``, so the draw depends only on the keys and the seed.
    Returns a Series ``start`` indexed by stay id.
    """
    k = keys[["stay_id", "hour"]].sort_values(["stay_id", "hour"])
    rng = np.random.default_rng(seed)
    starts = {}
    for stay, hours in k.groupby("stay_id", sort=True)["hour"]:
        h = hours.to_numpy()
        starts[stay] = int(h[rng.integers(len(h))])
    return pd.Series(starts, name="start", dtype="int64")


def window_rows(keys, starts, length=WINDOW_HOURS):
    """Rows whose inputs lie in their stay's outage: ``start < hour <= start + length``."""
    s = starts.reindex(keys["stay_id"].to_numpy()).to_numpy(dtype=float)
    h = keys["hour"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return (h > s) & (h <= s + length)


def before_window_rows(keys, starts):
    """Rows at or before their stay's outage start (inputs untouched)."""
    s = starts.reindex(keys["stay_id"].to_numpy()).to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return keys["hour"].to_numpy(dtype=float) <= s


# --- the cohort pipeline, without its I/O --------------------------------------

def cohort_from_raw(raw):
    """``mimic_cohort.build``'s per-row steps on ``raw``, without the file I/O
    and the patient split: carry forward strictly before t, event labels,
    NEWS2 and its flags, history rows dropped, the exported columns kept.

    Every step is per stay, so a subset of stays gives those stays' rows.
    T42 gates this against the cached cohort before using it.
    """
    raw = raw.rename(columns={"subject_id": PATIENT_ID})
    for col in CARRIED + ["event_h", "death_h", "vent_h", "vaso_h"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").astype(float)
    feat = build_before_t_features(raw)
    feat = add_event_labels(feat)
    news2, alert, on_oxygen = compute_news2(feat)
    feat["news2"] = news2.astype(np.int16)
    feat["is_alert"] = alert
    feat["on_oxygen"] = on_oxygen
    if "is_history" in feat.columns:
        feat = feat[pd.to_numeric(feat["is_history"]) == 0].reset_index(drop=True)
    return feat[COHORT_KEEP]


def frame_differences(a, b, columns):
    """Columns whose values differ between two row-aligned frames (NaN == NaN).

    Numeric columns compare exactly; anything else compares as text with
    missing values normalised, so ``None``/``pd.NA``/``NaN`` all match.
    """
    if len(a) != len(b):
        return ["<row count>"]
    bad = []
    for c in columns:
        x, y = a[c], b[c]
        if (pd.api.types.is_numeric_dtype(x) or pd.api.types.is_bool_dtype(x)) and \
                (pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y)):
            if not np.array_equal(x.to_numpy(dtype=float), y.to_numpy(dtype=float),
                                  equal_nan=True):
                bad.append(c)
        else:
            xs = x.astype(object).where(x.notna(), None).astype(str).to_numpy()
            ys = y.astype(object).where(y.notna(), None).astype(str).to_numpy()
            if not (xs == ys).all():
                bad.append(c)
    return bad


# --- a bank from in-memory emissions -------------------------------------------

def bank_from_outputs(outputs, agent_names):
    """An ``AgentBank`` laid out exactly as ``AgentBank.load`` lays it out,
    from ``{name: AgentOutput}`` for one split."""
    from orchestrator.fusion import AgentBank

    def key_arrays(o):
        return (o.keys["stay_id"].to_numpy(dtype="int64"),
                o.keys["hour"].to_numpy(dtype="int64"))
    ref_stay, ref_hour = key_arrays(outputs[agent_names[0]])
    for name in agent_names:
        st, hr = key_arrays(outputs[name])
        if not (np.array_equal(st, ref_stay) and np.array_equal(hr, ref_hour)):
            raise ValueError(f"agent {name!r} emitted different keys")
    reference = pd.DataFrame({"stay_id": ref_stay, "hour": ref_hour})
    block = np.concatenate(
        [outputs[n].latent for n in agent_names]
        + [np.stack([outputs[n].risk for n in agent_names], axis=1)],
        axis=1).astype(np.float32)
    dims = {n: int(outputs[n].latent.shape[1]) for n in agent_names}
    return AgentBank(reference, block, dims, agent_names)


# --- scoring helpers -------------------------------------------------------------

def noisy_or(c_t, c_d):
    """The two-head composite: P(any event) under independence."""
    return 1.0 - (1.0 - np.asarray(c_t)) * (1.0 - np.asarray(c_d))


def calibrated(iso, raw):
    """Isotonic probabilities with the flagship's rank tiebreak."""
    from fusion_ladder import _strictify
    raw = np.asarray(raw, dtype=float)
    return _strictify(iso.transform(raw), raw)


def robustness_verdict(ci):
    """``ci`` of (GBM drop - Sentinel drop)."""
    lo, hi = ci
    if lo > 0:
        return "Sentinel more robust"
    if hi < 0:
        return "GBM more robust"
    return "tie"


def absolute_verdict(ci):
    """``ci`` of (Sentinel - GBM) AUPRC during the outage."""
    lo, hi = ci
    if lo > 0:
        return "Sentinel better during this outage"
    if hi < 0:
        return "GBM better during this outage"
    return "tie"


def alarm_label(rate_outage, rate_intact):
    """Descriptive: 'flood' at >= 2x the intact rate, 'silence' at <= 0.5x."""
    if rate_intact <= 0:
        return "n/a"
    r = rate_outage / rate_intact
    if r >= 2.0:
        return "flood"
    if r <= 0.5:
        return "silence"
    return "within 2x"


def headline_pass(verdicts):
    """Scenario A, no retraining: Sentinel more robust for >= 3 of the 6
    feeds and the GBM more robust for none."""
    n_s = sum(v == "Sentinel more robust" for v in verdicts.values())
    n_g = sum(v == "GBM more robust" for v in verdicts.values())
    return bool(n_s >= 3 and n_g == 0), n_s, n_g


def window_catch(frame, y, scores, thr):
    """Scenario B catch: among event stays with at least one positive hour in
    the evaluated rows, the share alarmed (score >= thr) on any evaluated row.

    ``frame`` (stay_id per row), ``y`` and ``scores`` are the evaluated rows.
    """
    df = pd.DataFrame({"stay_id": np.asarray(frame["stay_id"]),
                       "y": np.asarray(y, dtype=float),
                       "alarm": np.asarray(scores, dtype=float) >= thr})
    g = df.groupby("stay_id", sort=True).agg(pos=("y", "max"), hit=("alarm", "max"))
    g = g[g["pos"] > 0]
    return float(g["hit"].mean()) if len(g) else float("nan"), int(len(g))
