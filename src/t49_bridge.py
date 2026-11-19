"""T49 bridge inputs: Sadanandan's 26 structured channels as XGBoost features.

Their code (github.com/thedatasense/Patient_Early_Deterioration_Risk_Prediction
at 785d8d7) feeds a model the hourly bins of the 48 h up to AND INCLUDING the
prediction hour, so a sample at hour t sees data up to t + 1. Our leakage rule
allows only data recorded strictly before t, so every feature here reads bins
t - 48 .. t - 1 (``scripts/t49_bridge.sql`` builds the bins; labs are binned by
the time the result was recorded). Their XGBoost code is not public; the
features per channel are the last value in the window and its age in hours,
and the window's mean, minimum and maximum; plus age (``anchor_age``, as their
code) and sex.
"""

import numpy as np
import pandas as pd

VITALS = ["heart_rate", "abp_systolic", "abp_diastolic", "abp_mean", "nibp_systolic",
          "nibp_diastolic", "respiratory_rate", "temperature_f", "temperature_c", "spo2"]
LABS = ["lactate", "ph", "po2", "pco2", "creatinine", "potassium", "sodium", "bicarbonate",
        "calcium", "wbc", "hemoglobin", "platelets", "inr", "ptt", "bilirubin_total", "glucose"]
CHANNELS = VITALS + LABS
WINDOW = 48


def feature_names(window=WINDOW):
    out = []
    for c in CHANNELS:
        out += [f"{c}_last", f"{c}_hrs_since", f"{c}_mean{window}", f"{c}_min{window}",
                f"{c}_max{window}"]
    return out + ["age", "male"]


def bridge_design(hourly, stays, keys, window=WINDOW):
    """Feature matrix for the ``(stay_id, hour)`` rows of ``keys``, in order.

    ``hourly``: one row per (stay_id, hour) bin with the 26 channel columns.
    ``stays``: stay_id, los_h, anchor_age, gender for every stay in ``keys``.
    A key at hour t uses bins t - window .. t - 1 only.
    """
    st = stays.drop_duplicates("stay_id").set_index("stay_id")
    want = np.unique(np.asarray(keys["stay_id"], dtype="int64"))
    st = st.loc[want]
    # every hour 0 .. floor(los) of every stay, each stay preceded by `window`
    # empty padding rows (hours -window .. -1), so a window of `window` rows
    # (= hours, the grid being complete) never reaches into the previous stay
    n = np.floor(st["los_h"].to_numpy(float)).astype(int) + 1 + window
    sid = np.repeat(st.index.to_numpy(dtype="int64"), n)
    hour = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) - window
    grid = pd.DataFrame({"stay_id": sid, "hour": hour})
    h = hourly[np.isin(hourly["stay_id"].to_numpy(dtype="int64"), want)]
    h = h[["stay_id", "hour"] + CHANNELS].astype({"stay_id": "int64", "hour": "int64"})
    h = h[h["hour"] >= 0]
    grid = grid.merge(h, on=["stay_id", "hour"], how="left")
    if len(grid) != n.sum():
        raise ValueError("repeated (stay_id, hour) bins in the hourly table")

    by = grid["stay_id"]
    out = {}
    for c in CHANNELS:
        x = grid[c].astype(float)
        prev = x.shift(1)                       # the bin before t (a padding row at hour 0)
        seen = grid["hour"].where(x.notna())
        last_hour = seen.groupby(by, sort=False).ffill().groupby(by, sort=False).shift(1)
        last_val = x.groupby(by, sort=False).ffill().groupby(by, sort=False).shift(1)
        age = grid["hour"] - last_hour
        fresh = age <= window
        out[f"{c}_last"] = last_val.where(fresh).to_numpy()
        out[f"{c}_hrs_since"] = age.where(fresh).to_numpy()
        roll = prev.rolling(window, min_periods=1)          # bins t - window .. t - 1
        out[f"{c}_mean{window}"] = roll.mean().to_numpy()
        out[f"{c}_min{window}"] = roll.min().to_numpy()
        out[f"{c}_max{window}"] = roll.max().to_numpy()
    feat = pd.DataFrame(out)
    feat["age"] = st.loc[grid["stay_id"], "anchor_age"].to_numpy(float)
    feat["male"] = (st.loc[grid["stay_id"], "gender"].to_numpy() == "M").astype(float)
    index = pd.MultiIndex.from_arrays([grid["stay_id"].to_numpy(), grid["hour"].to_numpy()])
    pos = index.get_indexer(pd.MultiIndex.from_arrays(
        [np.asarray(keys["stay_id"], dtype="int64"), np.asarray(keys["hour"], dtype="int64")]))
    if (pos < 0).any():
        raise KeyError(f"{int((pos < 0).sum())} keys fall outside their stay")
    return feat.iloc[pos].reset_index(drop=True)[feature_names(window)]
