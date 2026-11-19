"""T48: alarm-persistence rules at the stay level (pure helpers, no I/O).

* ``run_min``: the score of the "k consecutive hours above threshold" rule.
  At hour t it is the minimum score over hours t-k+1..t of the same stay, when
  all k hours are present and consecutive, and -inf otherwise. The rule alarms
  at t exactly when ``run_min >= thr``. Only hours up to t are read.
* ``hysteresis_state``: the hourly alarm state under hysteresis. The alarm
  turns on when the score reaches ``thr_on`` and stays on until it falls below
  ``thr_off``. A stay's first alarm is its first hour at or above ``thr_on``,
  so stay-level catches and quiet alarms equal the k = 1 rule's at ``thr_on``.
  Hysteresis only changes how long alarms last.
* ``stay_maxima``: per-stay maxima over the hours that can raise a catch (an
  event stay's hours at or before its event) or a quiet alarm (every hour of a
  stay without an event), as in T38's ``stay_max_scores``, with each stay's
  position for resample weights.
* ``share_at_or_above`` and ``catch_at_quiet_share``: T38's stay-level quiet
  share and best catch at a quiet share, with optional per-stay weights (a
  patient-clustered resample's draw counts). With unit weights they equal
  T38's functions.
"""

import numpy as np
import pandas as pd


def _runs(stay_ids, hours):
    """Start index of the consecutive-hour run each row belongs to."""
    st = np.asarray(stay_ids)
    hr = np.asarray(hours, dtype=float)
    new = np.r_[True, (st[1:] != st[:-1]) | (hr[1:] - hr[:-1] != 1.0)]
    return np.maximum.accumulate(np.where(new, np.arange(len(st)), 0))


def run_min(scores, stay_ids, hours, k):
    """k-hour rule score; rows must be sorted by (stay_id, hour)."""
    s = np.asarray(scores, dtype=float)
    if k == 1:
        return s.copy()
    start = _runs(stay_ids, hours)
    pos = np.arange(len(s))
    out = pd.Series(s).rolling(k, min_periods=k).min().to_numpy()
    out[pos - start < k - 1] = -np.inf          # fewer than k consecutive hours so far
    return out


def hysteresis_state(scores, stay_ids, thr_on, thr_off):
    """Hourly alarm state (bool); rows sorted by (stay_id, hour); needs
    ``thr_on >= thr_off``. Within each run of rows of one stay at or above
    ``thr_off``, the alarm is on from the run's first hour at or above
    ``thr_on`` to the end of the run."""
    if thr_on < thr_off:
        raise ValueError("thr_on must be at least thr_off")
    s = np.asarray(scores, dtype=float)
    st = np.asarray(stay_ids)
    keep = s >= thr_off
    brk = np.r_[True, (st[1:] != st[:-1]) | ~keep[:-1]] | ~keep
    seg = np.cumsum(brk)
    trig = pd.Series((s >= thr_on).astype(np.int8)).groupby(seg).cummax().to_numpy()
    return keep & (trig > 0)


def stay_maxima(stay_ids, hours, event_h, scores):
    """``(event_max, quiet_max, event_pos, quiet_pos)``: per-stay maxima and
    the stays' positions in first-appearance order (for weights)."""
    codes, _ = pd.factorize(np.asarray(stay_ids), sort=False)
    n = int(codes.max()) + 1
    ev = np.asarray(event_h, dtype=float)
    counts = np.isnan(ev) | (np.asarray(hours, dtype=float) <= ev)
    m = np.full(n, -np.inf)
    np.maximum.at(m, codes, np.where(counts, np.asarray(scores, dtype=float), -np.inf))
    is_event = np.zeros(n, dtype=bool)
    is_event[codes[~np.isnan(ev)]] = True
    pos = np.arange(n)
    return m[is_event], m[~is_event], pos[is_event], pos[~is_event]


def share_at_or_above(values, thr, weights=None):
    v = np.asarray(values, dtype=float)
    w = np.ones(len(v)) if weights is None else np.asarray(weights, dtype=float)
    tot = w.sum()
    return float(w[v >= thr].sum() / tot) if tot > 0 else float("nan")


def catch_at_quiet_share(event_max, quiet_max, q, w_event=None, w_quiet=None):
    """Best catch reachable while alarming on at most a share ``q`` of quiet
    stays: the upper envelope of the stay-level curve, linearly interpolated
    (T38's method), with optional per-stay weights."""
    em, qm = np.asarray(event_max, dtype=float), np.asarray(quiet_max, dtype=float)
    we = np.ones(len(em)) if w_event is None else np.asarray(w_event, dtype=float)
    wq = np.ones(len(qm)) if w_quiet is None else np.asarray(w_quiet, dtype=float)
    thr = np.unique(np.concatenate([em, qm]))[::-1]
    thr = thr[np.isfinite(thr)]

    def above(vals, w):
        o = np.argsort(vals, kind="stable")
        cw = np.r_[0.0, np.cumsum(w[o])]
        return 1.0 - cw[np.searchsorted(vals[o], thr, side="left")] / cw[-1]
    quiet, catch = above(qm, wq), above(em, we)
    qu, inv = np.unique(quiet, return_inverse=True)
    best = np.zeros(len(qu))
    np.maximum.at(best, inv, catch)
    if qu[0] > 0:
        qu, best = np.r_[0.0, qu], np.r_[0.0, best]
    return float(np.interp(q, qu, best))
