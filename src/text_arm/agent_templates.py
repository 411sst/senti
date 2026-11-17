"""Templated sentences for the five agents the August text arm never covered.

ECG (``ecg_meas``), clinician cadence, fluids, respiratory support and the
tabular GBM. The rules are those of ``summaries.py``:

* deterministic: a pure function of one row of the agent's own inputs and its
  own calibrated risk, so the agents are identical across the two arms;
* strictly before t: every input is a feature the agent itself reads at hour
  t, which its builder computes from data recorded before t (tests rewrite
  the future to prove it);
* verbal precision: values are binned or rounded the way a clinician would
  say them, so the message space stays small enough to encode, and the
  risk is the same ``risk_word`` binning the other templates use.

Each template maps a frame of rows to a frame of small integer FIELDS; rows are
deduplicated in field space and only the unique field tuples are rendered, so
the encoder sees each distinct sentence once (``sentences``).
"""

import numpy as np
import pandas as pd

from text_arm.summaries import RISK_BANDS, RISK_TOP

TEMPLATE_VERSION = "agent-templates-v1"

RISK_WORDS = [w for _, w in RISK_BANDS] + [RISK_TOP, "unknown"]


def risk_code(risk):
    """Vectorised ``summaries.risk_word``: index into ``RISK_WORDS``."""
    r = np.asarray(risk, dtype=float)
    code = np.searchsorted([edge for edge, _ in RISK_BANDS], r, side="right")
    return np.where(np.isnan(r), len(RISK_WORDS) - 1, code).astype(np.int64)


def _risk_tail(code):
    return f"Risk: {RISK_WORDS[int(code)]}."


# --- counts (cadence) -----------------------------------------------------------

COUNT_EDGES = [1, 3, 6, 11, 21, 51]
COUNT_WORDS = ["no", "1-2", "3-5", "6-10", "11-20", "21-50", "over 50"]


def count_bin(log1p_counts):
    counts = np.rint(np.expm1(np.asarray(log1p_counts, dtype=float)))
    return np.searchsorted(COUNT_EDGES, counts, side="right").astype(np.int64)


# --- ECG ------------------------------------------------------------------------

ECG_FINDINGS = [
    ("vtach", "ventricular tachycardia"), ("afib", "atrial fibrillation"),
    ("rvr", "rapid ventricular response"), ("aflutter", "atrial flutter"),
    ("av_block_high", "high-grade AV block"), ("acute_ischemia", "acute ischaemia"),
    ("st_t_abnormal", "ST-T changes"), ("prolonged_qt", "prolonged QT"),
    ("lbbb", "left bundle branch block"), ("rbbb", "right bundle branch block"),
    ("paced", "paced rhythm"), ("sinus_tachy", "sinus tachycardia"),
    ("sinus_brady", "sinus bradycardia"), ("low_voltage", "low voltage"),
]
ECG_MAX_FINDINGS = 3
ECG_AGE_EDGES = [6.0, 24.0]
ECG_AGE_WORDS = ["under 6 h", "6-24 h", "1-3 days"]


def _first_k(flags, k):
    """Indices of the first ``k`` set columns per row (priority order), -1 pad."""
    flags = np.asarray(flags) > 0.5
    rank = np.cumsum(flags, axis=1)
    out = np.full((flags.shape[0], k), -1, dtype=np.int64)
    for j in range(1, k + 1):
        hit = flags & (rank == j)
        any_hit = hit.any(axis=1)
        out[any_hit, j - 1] = hit[any_hit].argmax(axis=1)
    return out


def ecg_fields(frame, risk):
    """``frame`` holds ``ecg_<name>`` machine-read columns plus ecg_has/ecg_age."""
    has = (frame["ecg_has"].to_numpy(dtype=float) > 0.5).astype(np.int64)

    def valid(col):
        return (frame[f"ecg_{col}_valid"].to_numpy(dtype=float) > 0.5) & (has > 0)

    hr = frame["ecg_hr"].to_numpy(dtype=float)
    qrs = frame["ecg_qrs_dur"].to_numpy(dtype=float)
    qtc = frame["ecg_qtc_bazett"].to_numpy(dtype=float)
    first = _first_k(np.stack([frame[f"ecg_{c}"].to_numpy(dtype=float)
                               for c, _ in ECG_FINDINGS], axis=1), ECG_MAX_FINDINGS)
    first[has == 0] = -1
    f = {
        "has": has,
        "age": np.where(has > 0, np.searchsorted(
            ECG_AGE_EDGES, frame["ecg_age"].to_numpy(dtype=float), side="right"), -1),
        "rate": np.where(valid("hr"), np.rint(hr / 10.0) * 10, -1).astype(np.int64),
        "wide_qrs": np.where(valid("qrs_dur"), (qrs >= 120).astype(np.int64), -1),
        "qtc": np.where(valid("qtc_bazett"), np.rint(qtc / 20.0) * 20, -1).astype(np.int64),
        "normal": ((frame["ecg_normal_ecg"].to_numpy(dtype=float) > 0.5) & (has > 0)).astype(np.int64),
        "risk": risk_code(risk),
    }
    for j in range(ECG_MAX_FINDINGS):
        f[f"finding{j}"] = first[:, j]
    return pd.DataFrame(f)


def render_ecg(f):
    if not f["has"]:
        return f"ECG: none in the last 72 h. {_risk_tail(f['risk'])}"
    parts = [f"rate {f['rate']}" if f["rate"] >= 0 else "rate unmeasured",
             {1: "wide QRS", 0: "narrow QRS"}.get(f["wide_qrs"], "QRS unmeasured"),
             f"QTc {f['qtc']}" if f["qtc"] >= 0 else "QTc unmeasured"]
    named = [ECG_FINDINGS[f[f"finding{j}"]][1] for j in range(ECG_MAX_FINDINGS)
             if f[f"finding{j}"] >= 0]
    findings = (", ".join(named) if named
                else ("read as normal" if f["normal"] else "no named abnormality"))
    return (f"ECG {ECG_AGE_WORDS[f['age']]} ago: {', '.join(parts)}; {findings}. "
            f"{_risk_tail(f['risk'])}")


# --- clinician cadence ------------------------------------------------------------

CADENCE_RISING, CADENCE_FALLING = 1.25, 0.75
CADENCE_TREND_WORDS = ["steady", "rising", "falling", "quiet for 24 h"]


def cadence_fields(frame, risk):
    accel = np.mean([frame[c].to_numpy(dtype=float) for c in
                     ("n_lab_draws_accel", "n_chart_rows_accel", "n_orders_med_accel")], axis=0)
    quiet = ((frame["n_lab_draws_24h"].to_numpy(dtype=float) == 0)
             & (frame["n_chart_rows_24h"].to_numpy(dtype=float) == 0)
             & (frame["n_orders_med_24h"].to_numpy(dtype=float) == 0))
    trend = np.where(quiet, 3, np.where(accel > CADENCE_RISING, 1,
                                        np.where(accel < CADENCE_FALLING, 2, 0)))
    return pd.DataFrame({
        "labs6": count_bin(frame["n_lab_draws_6h"]),
        "meds6": count_bin(frame["n_orders_med_6h"]),
        "chart6": count_bin(frame["n_chart_rows_6h"]),
        "micro24": count_bin(frame["n_micro_24h"]),
        "trend": trend.astype(np.int64),
        "risk": risk_code(risk),
    })


def render_cadence(f):
    return (f"Care activity: {COUNT_WORDS[f['labs6']]} lab draws, "
            f"{COUNT_WORDS[f['meds6']]} medication orders and "
            f"{COUNT_WORDS[f['chart6']]} chart entries in the last 6 h; "
            f"{COUNT_WORDS[f['micro24']]} culture draws in 24 h; activity "
            f"{CADENCE_TREND_WORDS[f['trend']]}. {_risk_tail(f['risk'])}")


# --- fluids -----------------------------------------------------------------------

URINE_RATE_EDGES = [10, 20, 30, 50, 100, 200]
URINE_RATE_WORDS = ["under 10", "10-20", "20-30", "30-50", "50-100", "100-200", "over 200"]
NET_STEP_L, NET_CAP_L = 0.5, 5.0


def fluid_fields(frame, risk):
    rate = np.expm1(frame["urine_6h"].to_numpy(dtype=float)) / 6.0
    net = np.clip(frame["net_24h"].to_numpy(dtype=float), -NET_CAP_L, NET_CAP_L)
    return pd.DataFrame({
        "has": (frame["has_urine_history"].to_numpy(dtype=float) > 0.5).astype(np.int64),
        "recent": (frame["hrs_since_urine_record"].to_numpy(dtype=float) <= 6).astype(np.int64),
        "rate": np.searchsorted(URINE_RATE_EDGES, rate, side="right").astype(np.int64),
        "oliguric": (frame["oliguria_6h"].to_numpy(dtype=float) > 0.5).astype(np.int64),
        "net_half_l": np.rint(net / NET_STEP_L).astype(np.int64),
        "risk": risk_code(risk),
    })


def render_fluid(f):
    net = f"net balance {f['net_half_l'] * NET_STEP_L:+.1f} L over 24 h"
    if not f["has"]:
        body = f"no urine output recorded yet; {net}"
    elif not f["recent"]:
        body = f"no urine charted in the last 6 h; {net}"
    else:
        olig = ", oliguric" if f["oliguric"] else ""
        body = f"urine {URINE_RATE_WORDS[f['rate']]} ml/h over 6 h{olig}; {net}"
    return f"Fluids: {body}. {_risk_tail(f['risk'])}"


# --- respiratory support -----------------------------------------------------------

RESP_LEVEL_WORDS = ["no oxygen device", "nasal cannula", "face mask",
                    "non-rebreather mask", "high-flow nasal cannula",
                    "non-invasive ventilation"]
FLOW_STEP, FLOW_CAP = 2.0, 60.0
FIO2_STEP = 10.0


def resp_fields(frame, risk):
    flow = np.clip(frame["resp_o2_flow"].to_numpy(dtype=float), 0, FLOW_CAP)
    fio2 = frame["resp_fio2"].to_numpy(dtype=float)
    return pd.DataFrame({
        "seen": (frame["resp_seen"].to_numpy(dtype=float) > 0.5).astype(np.int64),
        "level": np.clip(np.rint(frame["resp_level"].to_numpy(dtype=float)), 0, 5).astype(np.int64),
        "airway": (frame["resp_airway"].to_numpy(dtype=float) > 0.5).astype(np.int64),
        "escalated": (frame["resp_escalated_6h"].to_numpy(dtype=float) > 0.5).astype(np.int64),
        "flow": (np.rint(flow / FLOW_STEP) * FLOW_STEP).astype(np.int64),
        "fio2": np.where(fio2 > 21.0, np.rint(fio2 / FIO2_STEP) * FIO2_STEP, 21).astype(np.int64),
        "risk": risk_code(risk),
    })


def render_resp(f):
    if not f["seen"]:
        return f"Breathing support: none recorded. {_risk_tail(f['risk'])}"
    parts = [RESP_LEVEL_WORDS[f["level"]]]
    if f["airway"]:
        parts[0] += " with an artificial airway"
    if f["flow"] > 0:
        parts.append(f"{f['flow']} L/min")
    if f["fio2"] > 21:
        parts.append(f"FiO2 {f['fio2']}%")
    change = "escalated in the last 6 h" if f["escalated"] else "unchanged over 6 h"
    return f"Breathing support: {', '.join(parts)}; {change}. {_risk_tail(f['risk'])}"


# --- tabular GBM ------------------------------------------------------------------

def gbm_fields(frame, risk):
    """The GBM agent's state IS its one-number risk; the sentence says only that."""
    return pd.DataFrame({"risk": risk_code(risk)})


def render_gbm(f):
    return f"Tabular risk model: {RISK_WORDS[int(f['risk'])]} risk."


TEMPLATES = {
    "ecg_meas": (ecg_fields, render_ecg),
    "cadence": (cadence_fields, render_cadence),
    "fluid": (fluid_fields, render_fluid),
    "resp": (resp_fields, render_resp),
    "gbm": (gbm_fields, render_gbm),
}


def sentences(agent, frame, risk):
    """``(unique sentences, codes)`` for one agent over ``frame``'s rows.

    ``codes[i]`` indexes the sentence for row i. Rows are deduplicated on their
    integer fields before rendering; two field tuples that render the same
    words share one sentence.
    """
    fields_fn, render = TEMPLATES[agent]
    fields = fields_fn(frame, risk)
    keys, inverse = np.unique(fields.to_numpy(dtype=np.int64), axis=0,
                              return_inverse=True)
    cols = list(fields.columns)
    rendered = [render(dict(zip(cols, (int(v) for v in row)))) for row in keys]
    uniq, remap = np.unique(np.asarray(rendered, dtype=object).astype(str),
                            return_inverse=True)
    return list(uniq), remap[np.asarray(inverse).reshape(-1)].astype(np.int64)
