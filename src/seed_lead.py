"""Seed-consistency helpers for the paired first-alarm race (T18).

Pure numpy/pandas/scipy -- no torch, no data access. The per-seed race is
summarised by :func:`race_anatomy` (wins : ties : losses, median, mean, the
share of the model being earlier among *discordant* stays -- the statistic
that stays informative when the median lands on the tie atom), seeds are
treated as the sampling unit by :func:`seed_level_ci`, calibrated scores are
pooled by :func:`ensemble_scores`, and :func:`tail_diagnosis` asks whether a
long "loss" (NEWS2 far earlier) is a real late alarm or NEWS2 sitting at its
threshold from the first hour of the stay.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats


def race_anatomy(diffs):
    """Summarise per-stay paired differences (positive = model earlier)."""
    d = np.asarray(diffs, dtype=float)
    n = int(len(d))
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    ties, disc = n - wins - losses, wins + losses
    nan = float("nan")
    return {
        "n": n, "wins": wins, "ties": ties, "losses": losses,
        "median": float(np.median(d)) if n else nan,
        "mean": float(d.mean()) if n else nan,
        "p10": float(np.percentile(d, 10)) if n else nan,
        "p90": float(np.percentile(d, 90)) if n else nan,
        "win_given_discordant": wins / disc if disc else nan,
        "sign_p": (float(stats.binomtest(wins, disc, 0.5).pvalue)
                   if disc else 1.0),
    }


def seed_level_ci(values, level=0.95):
    """Mean over seeds with a t-interval that treats each seed as one draw.

    Returns ``(mean, (lo, hi), sd)``; the interval is NaN below two seeds.
    """
    v = np.asarray(values, dtype=float)
    k = len(v)
    if k == 0:
        return float("nan"), (float("nan"), float("nan")), float("nan")
    mean = float(v.mean())
    if k < 2:
        return mean, (float("nan"), float("nan")), float("nan")
    sd = float(v.std(ddof=1))
    half = float(stats.t.ppf(0.5 + level / 2.0, k - 1)) * sd / math.sqrt(k)
    return mean, (mean - half, mean + half), sd


def seeds_needed(sd, target_half_width, level=0.95, k_max=200):
    """Smallest k whose t-interval half-width at this SD is <= the target."""
    for k in range(2, k_max + 1):
        half = float(stats.t.ppf(0.5 + level / 2.0, k - 1)) * sd / math.sqrt(k)
        if half <= target_half_width:
            return k
    return None


def ensemble_scores(stack, how="mean"):
    """Seed ensemble of calibrated scores: ``stack`` is (seeds, rows)."""
    arr = np.asarray(stack, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("stack must be (n_seeds, n_rows) with n_seeds >= 1")
    if how == "mean":
        return arr.mean(axis=0)
    if how == "median":
        return np.median(arr, axis=0)
    raise ValueError(f"unknown ensemble rule {how!r}")


def tail_diagnosis(test, s_news2, s_model, thr_news2, thr_model, diffs,
                   cutoff=12.0):
    """Anatomy of the long tails of the paired race.

    For every paired stay whose |diff| >= ``cutoff`` hours, records whether
    the *earlier* system was already at or above its threshold on the stay's
    first scored hour -- a score that is "on" from admission is not an early
    warning, it is a standing alarm, and the race credits it with the whole
    gap. Returns one row per tail stay with ``side`` ("loss" = NEWS2 earlier,
    "win" = model earlier) and ``earlier_from_first_hour``.
    """
    tmp = test[["stay_id", "hour"]].copy()
    tmp["n2"] = np.asarray(s_news2, dtype=float)
    tmp["m"] = np.asarray(s_model, dtype=float)
    d = diffs.set_index("stay_id")["diff"]
    tail = d[d.abs() >= cutoff]
    rows = []
    for stay, diff in tail.items():
        g = tmp[tmp["stay_id"] == stay].sort_values("hour")
        first = g.iloc[0]
        if diff < 0:
            rows.append({"stay_id": stay, "diff": float(diff), "side": "loss",
                         "earlier_from_first_hour": bool(first["n2"] >= thr_news2)})
        else:
            rows.append({"stay_id": stay, "diff": float(diff), "side": "win",
                         "earlier_from_first_hour": bool(first["m"] >= thr_model)})
    return pd.DataFrame(rows, columns=["stay_id", "diff", "side",
                                       "earlier_from_first_hour"])


def summarize_tail(tail):
    """Per side: how many tail stays, and what share were standing alarms."""
    out = {}
    for side in ("loss", "win"):
        sub = tail[tail["side"] == side]
        out[side] = {
            "n": int(len(sub)),
            "from_first_hour": int(sub["earlier_from_first_hour"].sum()),
            "share_from_first_hour": (float(sub["earlier_from_first_hour"].mean())
                                      if len(sub) else float("nan")),
        }
    return out
