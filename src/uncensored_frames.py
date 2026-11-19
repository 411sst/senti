"""Model inputs for stays and hours our cohort leaves out (T44-T49).

Our cohort drops stays with an event in the first 6 h and censors each stay at
its first event. The published cohorts keep both, so their rows are built from
the uncensored extract (``scripts/t4x_mimic_raw.sql``) with the cohort's own
per-row code, ``feed_outage.cohort_from_raw``: carry forward strictly before t,
NEWS2, history rows (hours before 6) dropped. Every step is per stay.
"""

import numpy as np
import pandas as pd

from feed_outage import cohort_from_raw


def uncensored_frame(raw, stays=None):
    """Prediction rows (hour 6 on) for ``stays`` (all when None), sorted by
    stay and hour, with no stabilisation filter and no censoring."""
    if stays is not None:
        keep = np.isin(raw["stay_id"].to_numpy(dtype="int64"),
                       np.asarray(stays, dtype="int64"))
        raw = raw[keep]
    out = cohort_from_raw(raw.copy())
    return out.sort_values(["stay_id", "hour"]).reset_index(drop=True)


def pick_rows(frame, keys):
    """Positions in ``frame`` of the ``(stay_id, hour)`` rows in ``keys``, in
    ``keys`` order. Raises if any key has no row."""
    index = pd.MultiIndex.from_arrays([frame["stay_id"].to_numpy(dtype="int64"),
                                       frame["hour"].to_numpy(dtype="int64")])
    if not index.is_unique:
        raise ValueError("frame has repeated (stay_id, hour) rows")
    want = pd.MultiIndex.from_arrays([np.asarray(keys["stay_id"], dtype="int64"),
                                      np.asarray(keys["hour"], dtype="int64")])
    pos = index.get_indexer(want)
    if (pos < 0).any():
        raise KeyError(f"{int((pos < 0).sum())} requested rows are not in the frame")
    return pos


def reconcile(new_raw, old_raw, columns):
    """``new_raw`` with ``columns`` taken from ``old_raw`` on every (stay_id,
    hour) bin both extracts have.

    BigQuery's AVG sums in a nondeterministic order, so a re-run of the same
    query can differ in the last bit (relative ~5e-16) on bins with several
    measurements, and a value sitting exactly on a NEWS2 band edge can flip a
    band. Taking the original extract's values on shared bins makes every row
    our cohort already has bit-identical to it; the new extract only supplies
    bins the original never had. Returns ``(frame, n_replaced_bins)``.
    """
    keys = ["stay_id", "hour"]
    out = new_raw.copy()
    left = pd.MultiIndex.from_arrays([out[k].to_numpy(dtype="int64") for k in keys])
    right = pd.MultiIndex.from_arrays([old_raw[k].to_numpy(dtype="int64") for k in keys])
    if not left.is_unique or not right.is_unique:
        raise ValueError("repeated (stay_id, hour) bins")
    pos = right.get_indexer(left)
    shared = pos >= 0
    for c in columns:
        col = pd.to_numeric(out[c], errors="coerce").astype(float).to_numpy()
        src = pd.to_numeric(old_raw[c], errors="coerce").astype(float).to_numpy()
        col[shared] = src[pos[shared]]
        out[c] = col
    return out, int(shared.sum())
