"""Featurize machine-read ECG measurements into a fixed-width vector per study.

Turns one ``machine_measurements`` row (conduction intervals + up to 18 machine
diagnostic statements) into ~49 features consumed by the ``ecg_meas`` agent:

* 13 numeric features (rates, durations, QT corrections, axes, QRS-T angle,
  statement count) with **per-feature** validity flags -- the cart emits partial
  sentinels on real rows (an AFib read has no P wave, so ``p_axis`` is sentinel
  while QRS/QT are fine), so validity cannot be a row-level property.
* ~24 binary flags regex-matched against the joined statement text (rhythm,
  conduction, ectopy, ischemia, hypertrophy, quality).

Invalid numerics are imputed with **fixed clinical normals**, not train
statistics: constants are split-independent (no leakage question, the npz can be
built once before any split is touched) and the agent's inherited z-scoring on
covered train rows handles scale. The paired validity flag carries the
"missing" signal itself -- which is often informative (no P wave == AFib).

Sentinel rules (verified against the BigQuery table):
* onset/end columns: ``0``, ``29999``, ``32767`` are sentinels (0 is not a
  physical onset on the shared ms time base).
* axis columns: ``29999``, ``32767`` only -- **0 degrees is a legitimate axis**.
* ``rr_interval``: 0 plus plausibility bounds.
Every derived feature also has plausibility bounds; out-of-bounds => invalid.
"""

import re

import numpy as np
import pandas as pd

REPORT_COLS = [f"report_{i}" for i in range(18)]

ONSET_SENTINELS = {0, 29999, 32767}
AXIS_SENTINELS = {29999, 32767}          # 0 is a valid axis, keep it

# feature -> (imputed normal, low bound, high bound)
NUMERIC_SPEC = {
    "hr":            (75.0,   20.0,  300.0),
    "rr":            (800.0,  200.0, 3000.0),
    "p_dur":         (100.0,  30.0,  200.0),
    "pr":            (160.0,  50.0,  400.0),
    "qrs_dur":       (90.0,   40.0,  250.0),
    "qt":            (400.0,  200.0, 700.0),
    "qtc_bazett":    (420.0,  200.0, 800.0),
    "qtc_fridericia": (420.0, 200.0, 800.0),
    "p_axis":        (45.0,  -180.0, 180.0),
    "qrs_axis":      (40.0,  -180.0, 180.0),
    "t_axis":        (45.0,  -180.0, 180.0),
    "qrst_angle":    (30.0,   0.0,   180.0),
}

# flag -> regex over the UPPERCASED " | "-joined statement text. Order is the
# feature order. NORMAL ECG needs a guard: it substring-matches inside
# "ABNORMAL ECG", hence the negative lookbehind (tested).
STATEMENT_PATTERNS = {
    "sinus_rhythm":   r"SINUS RHYTHM",
    "sinus_brady":    r"SINUS BRADYCARDIA",
    "sinus_tachy":    r"SINUS TACHYCARDIA",
    "afib":           r"ATRIAL FIB",
    "rvr":            r"RAPID VENTRICULAR",
    "aflutter":       r"ATRIAL FLUTTER",
    "svt_other":      r"SUPRAVENTRICULAR (TACH|RHYTHM)|\bSVT\b|JUNCTIONAL (RHYTHM|TACH)|ECTOPIC ATRIAL|ATRIAL TACH",
    "vtach":          r"VENTRICULAR TACH|IDIOVENTRICULAR|ACCELERATED VENTRICULAR|WIDE COMPLEX",
    "av_block_1":     r"1ST DEGREE A-?V BLOCK",
    "av_block_high":  r"(2ND|3RD) DEGREE|COMPLETE (HEART|A-?V) BLOCK|MOBITZ|HIGH[- ]GRADE",
    "pvc":            r"\bPVC",
    "pac":            r"\bPAC\b|\bPACS\b|\bPAC\(S\)",
    "paced":          r"PACED\b|PACING\b|ELECTRONIC.*PACEMAKER|\bA-?V SEQUENTIAL",
    "lbbb":           r"LEFT BUNDLE",
    "rbbb":           r"RIGHT BUNDLE",
    "ivcd":           r"INTRAVENTRICULAR CONDUCTION",
    "lvh_rvh":        r"(LEFT|RIGHT) VENTRICULAR HYPERTROPHY|\bLVH\b",
    # [^|]* keeps the match inside ONE " | "-joined statement, so "Acute
    # anterior infarct" matches but ACUTE in one statement cannot pair with
    # INFARCT in a different one.
    "acute_ischemia": r"CONSIDER ACUTE|ACUTE MI\b|\bSTEMI\b|ACUTE[^|]*(INFARCT|ISCHEMIA)|INJURY",
    "st_t_abnormal":  r"ST (DEPRESSION|ELEVATION|ABNORMALIT)|T WAVE (INVERSION|ABNORMALIT)|NONSPECIFIC ST",
    "old_infarct":    r"INFARCT",
    "prolonged_qt":   r"PROLONGED QT",
    "low_voltage":    r"LOW (QRS )?VOLTAGE",
    "normal_ecg":     r"(?<![A-Z])NORMAL ECG",
    "abnormal_ecg":   r"ABNORMAL ECG",
    "has_urgent":     r"\*\*\*",
    "quality_warning": r"MISSING|DISCONNECT|(POOR|DATA) QUALITY|BASELINE (WANDER|SWAY)|ARTIFACT|MOTION|LEAD REVERSAL|SUSPECT",
}

# The rate-carrying features dropped by the --no-rate ablation (HR redundancy
# with the vitals agent -- see the training script).
RATE_FEATURES = ("hr", "rr", "sinus_tachy", "sinus_brady", "rvr")

FEATURE_NAMES = (
    list(NUMERIC_SPEC)
    + ["n_statements"]
    + [f"{k}_valid" for k in NUMERIC_SPEC]
    + list(STATEMENT_PATTERNS)
)


def _clean(col, sentinels):
    """float array with sentinels -> NaN."""
    v = pd.to_numeric(col, errors="coerce").astype(float).to_numpy()
    return np.where(np.isin(v, list(sentinels)), np.nan, v)


def _bounded(v, lo, hi):
    """NaN out anything outside plausibility bounds."""
    return np.where((v >= lo) & (v <= hi), v, np.nan)


def join_statements(df):
    """Per row: ' | '-joined, uppercased, whitespace-collapsed statement text."""
    txt = df[REPORT_COLS].fillna("").astype(str).agg(" | ".join, axis=1)
    return txt.str.upper().str.replace(r"\s+", " ", regex=True)


def build_meas_features(df):
    """(study_ids (N,), features (N, F) float32, feature_names) from raw rows.

    Rows whose numerics are ALL invalid and whose statements match nothing but
    quality warnings (e.g. "All 12 leads are missing") are dropped -- the
    agent's cache-lookup miss then treats those hours as no-ECG, exactly like a
    study that was never downloaded.
    """
    n = len(df)

    # -- numerics ---------------------------------------------------------
    rr = _bounded(_clean(df["rr_interval"], {0} | AXIS_SENTINELS), *NUMERIC_SPEC["rr"][1:])
    p_on = _clean(df["p_onset"], ONSET_SENTINELS)
    p_end = _clean(df["p_end"], ONSET_SENTINELS)
    q_on = _clean(df["qrs_onset"], ONSET_SENTINELS)
    q_end = _clean(df["qrs_end"], ONSET_SENTINELS)
    t_end = _clean(df["t_end"], ONSET_SENTINELS)

    with np.errstate(invalid="ignore", divide="ignore"):
        raw = {
            "hr": 60000.0 / rr,
            "rr": rr,
            "p_dur": p_end - p_on,
            "pr": q_on - p_on,
            "qrs_dur": q_end - q_on,
            "qt": t_end - q_on,
        }
        raw["qtc_bazett"] = raw["qt"] / np.sqrt(rr / 1000.0)
        raw["qtc_fridericia"] = raw["qt"] / np.cbrt(rr / 1000.0)
        for ax in ("p_axis", "qrs_axis", "t_axis"):
            raw[ax] = _bounded(_clean(df[ax], AXIS_SENTINELS), -180.0, 180.0)
        # angle between QRS and T axes, wrapped so 350 degrees apart == 10
        diff = np.abs(raw["qrs_axis"] - raw["t_axis"]) % 360.0
        raw["qrst_angle"] = np.where(diff > 180.0, 360.0 - diff, diff)

    values, valid_flags = [], []
    for name, (normal, lo, hi) in NUMERIC_SPEC.items():
        v = _bounded(raw[name], lo, hi)
        ok = ~np.isnan(v)
        values.append(np.where(ok, v, normal))
        valid_flags.append(ok.astype(float))

    # -- statements -------------------------------------------------------
    text = join_statements(df)
    n_statements = (df[REPORT_COLS].notna() & (df[REPORT_COLS] != "")).sum(axis=1)
    stmt_flags = [
        text.str.contains(pat, regex=True).to_numpy().astype(float)
        for pat in STATEMENT_PATTERNS.values()
    ]
    # old infarct means "infarct mentioned but not acutely"
    idx = list(STATEMENT_PATTERNS).index
    stmt_flags[idx("old_infarct")] = np.clip(
        stmt_flags[idx("old_infarct")] - stmt_flags[idx("acute_ischemia")], 0.0, 1.0)

    X = np.column_stack(
        values + [n_statements.to_numpy(dtype=float)] + valid_flags + stmt_flags
    ).astype(np.float32)
    assert X.shape == (n, len(FEATURE_NAMES))

    # -- drop fully-uninformative studies ---------------------------------
    any_valid = np.column_stack(valid_flags).any(axis=1)
    quality_only = ~np.column_stack(
        [f for name, f in zip(STATEMENT_PATTERNS, stmt_flags)
         if name not in ("quality_warning", "has_urgent")]
    ).any(axis=1)
    keep = any_valid | ~quality_only
    study_ids = df["study_id"].astype("int64").to_numpy()
    return study_ids[keep], X[keep], list(FEATURE_NAMES)


def match_census(df):
    """Per-flag positive counts + the most frequent UNMATCHED statements.

    Printed by the build script so a typo'd regex shows up as a visible gap
    instead of a mysteriously weak agent.
    """
    text = join_statements(df)
    counts = {name: int(text.str.contains(pat, regex=True).sum())
              for name, pat in STATEMENT_PATTERNS.items()}
    matched_any = np.zeros(len(df), dtype=bool)
    for pat in STATEMENT_PATTERNS.values():
        matched_any |= text.str.contains(pat, regex=True).to_numpy()
    unmatched = (df.loc[~matched_any, "report_0"].dropna()
                 .str.upper().str.strip().value_counts().head(20))
    return counts, unmatched
