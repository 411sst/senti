"""Unit tests for the baseline models and their leakage-safe features."""

import numpy as np
import pandas as pd
import pytest

from baselines import _paired_diff, build_features, news2_risk_scores, run


def _vitals_row(**over):
    row = {"Resp": 16.0, "O2Sat": 98.0, "Temp": 37.0, "SBP": 120.0,
           "HR": 70.0, "FiO2": np.nan, "SepsisLabel": 0}
    row.update(over)
    return row


def _synthetic_cohort(n_normal=30, n_septic=30, hours=15, onset=12):
    """Normal patients vs septic patients whose HR rises 2h before onset."""
    rows = []
    for i in range(n_normal):
        for h in range(1, hours + 1):
            rows.append({"patient_id": f"n{i:03d}", "ICULOS": h, **_vitals_row()})
    for i in range(n_septic):
        for h in range(1, hours + 1):
            hr = 130.0 if h >= onset - 2 else 70.0
            lab = 1 if h >= onset else 0
            rows.append({"patient_id": f"s{i:03d}", "ICULOS": h,
                         **_vitals_row(HR=hr, SepsisLabel=lab)})
    return pd.DataFrame(rows)


# --- Leakage safety --------------------------------------------------------

def test_features_use_only_data_before_t():
    df = pd.DataFrame(
        {
            "patient_id": ["p1"] * 4,
            "ICULOS": [1, 2, 3, 4],
            "HR": [80.0, np.nan, np.nan, 95.0],
            "Resp": [np.nan, 18.0, np.nan, np.nan],
            "O2Sat": [98.0] * 4, "Temp": [37.0] * 4, "SBP": [120.0] * 4,
            "FiO2": [np.nan] * 4, "SepsisLabel": [0, 0, 0, 0],
        }
    )
    feat = build_features(df).sort_values("ICULOS").reset_index(drop=True)

    hr_locf = feat["HR_locf"].tolist()
    assert np.isnan(hr_locf[0])       # hour 1: nothing before it
    assert hr_locf[1] == 80.0         # hour 2 sees hour 1
    assert hr_locf[3] == 80.0         # hour 4: its own 95 must NOT leak in
    assert feat["HR_hrs_since"].tolist()[3] == 3   # last measured hour 1

    resp_locf = feat["Resp_locf"].tolist()
    assert np.isnan(resp_locf[0])
    assert resp_locf[2] == 18.0       # hour 3 sees hour 2's reading


def test_features_do_not_bleed_across_patients():
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p2"],
            "ICULOS": [1, 2, 1, 2],
            "HR": [50.0, np.nan, np.nan, 90.0],
            "Resp": [16.0] * 4, "O2Sat": [98.0] * 4, "Temp": [37.0] * 4,
            "SBP": [120.0] * 4, "FiO2": [np.nan] * 4, "SepsisLabel": [0, 0, 0, 0],
        }
    )
    feat = build_features(df)
    p2h1 = feat[(feat.patient_id == "p2") & (feat.ICULOS == 1)]["HR_locf"]
    assert np.isnan(p2h1.iloc[0])     # p1's 50 must not bleed into p2
    p1h2 = feat[(feat.patient_id == "p1") & (feat.ICULOS == 2)]["HR_locf"]
    assert p1h2.iloc[0] == 50.0


# --- Scoring ---------------------------------------------------------------

def test_news2_risk_scores_matches_degraded_score():
    feat = pd.DataFrame(
        {"Resp_locf": [22.0], "O2Sat_locf": [93.0], "Temp_locf": [39.5],
         "SBP_locf": [95.0], "HR_locf": [130.0], "FiO2_locf": [0.5]}
    )
    # 2+2+2+2+2 + 2(O2) + 0 = 12
    assert news2_risk_scores(feat).tolist() == [12]


# --- End to end ------------------------------------------------------------

def _crossing_series(cross_hours, onset=10, n_hours=12):
    """Build a (risk_scores, event_times, times) series where each patient's
    score is 0 until ``cross_hour`` then 1 (None = never crosses)."""
    risk, events, times = [], [], []
    for ch in cross_hours:
        hours = np.arange(1, n_hours + 1)
        risk.append(np.zeros(n_hours) if ch is None else (hours >= ch).astype(float))
        times.append(hours)
        events.append(onset)
    return risk, events, times


def test_paired_diff_counts_sign_and_both_restriction():
    thr = 0.5
    #                p1 p2 p3 p4 p5   p6(NEWS2-only)
    news2_cross = [   5, 5, 5, 5, 5,   3]
    gbm_cross = [     3, 2, 4, 1, 7,   None]   # p6: GBM never crosses -> excluded
    r = _paired_diff(_crossing_series(news2_cross), thr,
                     _crossing_series(gbm_cross), thr)
    # p6 caught by NEWS2 only -> not in the paired set
    assert r["n_paired"] == 5
    # diffs = [2, 3, 1, 4, -2]: GBM earlier in 4, NEWS2 earlier in 1
    assert r["n_gbm_earlier"] == 4
    assert r["n_news2_earlier"] == 1
    assert r["median_diff"] == pytest.approx(2.0)
    assert r["frac_gbm_earlier"] == pytest.approx(0.8)
    assert 0.0 <= r["sign_p"] <= 1.0


def test_run_end_to_end_learns_signal():
    df = _synthetic_cohort()
    results = run(df=df)

    assert set(results) == {
        "summary", "discrimination", "lead_time_sweep", "paired_lead_time"
    }

    disc = results["discrimination"]
    assert set(disc) == {"NEWS2", "GBM"}
    for d in disc.values():
        assert 0.0 <= d["auprc"] <= 1.0
        assert 0.0 <= d["ece"] <= 1.0

    sweep = results["lead_time_sweep"]
    assert {r["budget"] for r in sweep} >= {"1%", "2%", "5%", "10%", "clinical"}
    for r in sweep:
        assert r["model"] in {"NEWS2", "GBM"}
        assert 0.0 <= r["caught_in_time_rate"] <= 1.0
        assert 0.0 <= r["actual_rate"] <= 1.0
        for key in ("median_lead_time", "lead_p25", "lead_p75", "n_caught", "n_events"):
            assert key in r

    paired = results["paired_lead_time"]
    assert {r["budget"] for r in paired} >= {"1%", "2%", "5%", "10%", "clinical"}
    for r in paired:
        for key in ("n_paired", "median_diff", "diff_p25", "diff_p75",
                    "frac_gbm_earlier", "sign_p"):
            assert key in r
        assert 0.0 <= r["sign_p"] <= 1.0

    prevalence = df["SepsisLabel"].mean()
    assert disc["GBM"]["auprc"] > prevalence  # learned the signal
