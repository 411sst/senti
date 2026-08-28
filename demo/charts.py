"""Altair charts for the demo dashboard and the leave-behind (altair + pandas only)."""

import altair as alt
import pandas as pd

MODEL = "#2a78d6"      # Sentinel / the fused risk
NEWS2 = "#eb6834"      # the deployed comparator
EVENT = "#e34948"      # the deterioration event
GRAY = "#8a8f98"       # thresholds, comparators
INK = "#2b2f36"        # the selected hour

_HOUR = alt.X("hour:Q", title="hour of stay")


def _x(domain):
    return alt.X("hour:Q", title="hour of stay",
                 scale=alt.Scale(domain=list(domain), nice=False),
                 axis=alt.Axis(tickMinStep=2, labelFlush=True))


def _hrule(y, label, color=GRAY, dash=(5, 4), align="left", x=None):
    df = pd.DataFrame({"y": [y], "label": [label]})
    rule = alt.Chart(df).mark_rule(color=color, strokeDash=list(dash)).encode(y="y:Q")
    text = alt.Chart(df).mark_text(align=align, baseline="bottom", dx=4, dy=-3,
                                   color=color, fontSize=11).encode(
        y="y:Q", text="label:N", x=alt.value(x if x is not None else 4))
    return rule + text


def _vrule(x, label, color, domain, dash=None):
    df = pd.DataFrame({"hour": [x], "label": [label]})
    kw = {"color": color}
    if dash:
        kw["strokeDash"] = list(dash)
    rule = alt.Chart(df).mark_rule(**kw).encode(x=_x(domain))
    # label sits on whichever side of the rule has room
    near_right = (x - domain[0]) > 0.8 * (domain[1] - domain[0])
    text = alt.Chart(df).mark_text(align="right" if near_right else "left",
                                   dx=-5 if near_right else 5, color=color,
                                   fontSize=11, fontWeight="bold").encode(
        x=_x(domain), text="label:N", y=alt.value(6))
    return rule + text


def _domain(stay):
    hours = stay["hours"]
    hi = hours[-1]
    if stay.get("event_h") is not None:
        hi = max(hi, stay["event_h"])
    return hours[0], hi + 0.5


def risk_chart(stay, thr_model, hour, unit_thr=None, height=250):
    """Composite risk over the stay: threshold, alarms, the event, the cursor."""
    domain = _domain(stay)
    df = pd.DataFrame({"hour": stay["hours"], "risk": stay["risk"]["composite"],
                       "alarm": stay["alarm"]["model"]})
    y = alt.Y("risk:Q", title="fused risk (two-head composite)",
              scale=alt.Scale(domain=[0, 1]))
    line = alt.Chart(df).mark_line(color=MODEL, strokeWidth=2).encode(
        x=_x(domain), y=y,
        tooltip=[alt.Tooltip("hour:Q", title="hour"),
                 alt.Tooltip("risk:Q", title="risk", format=".3f")])
    layers = [line, _hrule(thr_model, "alarm threshold · 2% budget")]
    if unit_thr is not None:
        layers.append(_hrule(unit_thr, "per-unit standby threshold", color="#b9bec7",
                             dash=(2, 3), align="right", x=None).properties())
    alarms = alt.Chart(df[df["alarm"]]).mark_point(color=MODEL, filled=True, size=80).encode(
        x=_x(domain), y=y, tooltip=[alt.Tooltip("hour:Q", title="model alarm at hour")])
    layers.append(alarms)
    if stay.get("event_h") is not None:
        word = {"vent": "ventilation", "vaso": "vasopressors", "death": "death"}.get(
            stay.get("event_type"), "event")
        layers.append(_vrule(stay["event_h"], word, EVENT, domain))
    layers.append(_vrule(hour, "", INK, domain, dash=(1, 3)))
    return alt.layer(*layers).properties(height=height, width="container")


def news2_chart(stay, thr_news2, hour, height=120):
    """NEWS2 total as a step line with its own alarm rule and alarm hours."""
    domain = _domain(stay)
    df = pd.DataFrame({"hour": stay["hours"], "news2": stay["news2"]["total"],
                       "alarm": stay["alarm"]["news2"]})
    y = alt.Y("news2:Q", title="NEWS2", scale=alt.Scale(domain=[0, 20]))
    line = alt.Chart(df).mark_line(color=NEWS2, strokeWidth=2, interpolate="step-after").encode(
        x=_x(domain), y=y, tooltip=[alt.Tooltip("hour:Q"), alt.Tooltip("news2:Q")])
    alarms = alt.Chart(df[df["alarm"]]).mark_point(color=NEWS2, filled=True, size=70).encode(
        x=_x(domain), y=y, tooltip=[alt.Tooltip("hour:Q", title="NEWS2 alarm at hour")])
    layers = [line, _hrule(thr_news2, f"NEWS2 alarm at ≥ {thr_news2:g}"), alarms]
    if stay.get("event_h") is not None:
        layers.append(_vrule(stay["event_h"], "", EVENT, domain))
    layers.append(_vrule(hour, "", INK, domain, dash=(1, 3)))
    return alt.layer(*layers).properties(height=height, width="container")


SHORT_LABELS = {"fluids and urine output": "fluids & urine", "respiratory support": "resp. support",
                "ECG measurements": "ECG measures", "charting cadence": "charting cadence",
                "GBM risk model": "GBM risk model"}


def attention_chart(stay, hour, agents, labels, height=270):
    """Where the fusion looked at this hour; each agent's calibrated risk at the bar end."""
    i = stay["hours"].index(hour)
    names = [SHORT_LABELS.get(labels.get(a, a), labels.get(a, a)) for a in agents]
    df = pd.DataFrame({
        "agent": names,
        "attention": [stay["attention"][a][i] for a in agents],
        "risk": [stay["agent_risk"][a][i] for a in agents],
    })
    df["label"] = df["risk"].map(lambda r: f"risk {r:.2f}")
    base = alt.Chart(df).encode(
        y=alt.Y("agent:N", sort=names, title=None,
                axis=alt.Axis(labelLimit=220, labelOverlap=False, labelPadding=6)))
    bars = base.mark_bar(color=MODEL, size=16).encode(
        x=alt.X("attention:Q", title="attention weight",
                scale=alt.Scale(domain=[0, max(0.5, float(df["attention"].max()) * 1.45)]),
                axis=alt.Axis(tickCount=5)),
        tooltip=[alt.Tooltip("agent:N"), alt.Tooltip("attention:Q", format=".2f"),
                 alt.Tooltip("risk:Q", title="calibrated risk", format=".3f")])
    text = base.mark_text(align="left", dx=5, color="#7d8792", fontSize=11).encode(
        x="attention:Q", text="label:N")
    return (bars + text).properties(height=height, width="container")


def scatter_chart(points, height=300):
    """Ranking (AUPRC) against earliness (median lead vs NEWS2 at matched volume)."""
    df = pd.DataFrame(points)
    df["kind"] = df["is_flagship"].map({True: "Sentinel", False: "comparator"})
    df["win_pct"] = (df["win_rate"] * 100).round(0)
    base = alt.Chart(df).encode(
        x=alt.X("auprc:Q", title="ranking quality (AUPRC, held-out test)",
                scale=alt.Scale(domain=[0.08, 0.34])),
        y=alt.Y("median_lead_h:Q", title="median hours earlier than NEWS2",
                scale=alt.Scale(domain=[-0.3, 1.5])),
        tooltip=[alt.Tooltip("name:N", title="model"),
                 alt.Tooltip("auprc:Q", format=".3f"), alt.Tooltip("auroc:Q", format=".3f"),
                 alt.Tooltip("median_lead_h:Q", title="median lead (h)"),
                 alt.Tooltip("win_pct:Q", title="win rate %"),
                 alt.Tooltip("n_pair:Q", title="paired stays")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=GRAY, strokeDash=[5, 4]).encode(y="y:Q")
    pts = base.mark_point(filled=True, size=140).encode(
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Sentinel", "comparator"],
                                                  range=[MODEL, GRAY]),
                        legend=alt.Legend(title=None, orient="top-left")))
    flag = df[df["is_flagship"]]
    label = alt.Chart(flag).mark_text(align="right", dx=-10, dy=-2, color=MODEL,
                                      fontWeight="bold", fontSize=12).encode(
        x="auprc:Q", y="median_lead_h:Q", text="name:N")
    return alt.layer(zero, pts, label).properties(height=height, width="container")


def burden_chart(units, budget, height=250):
    """Realised alarm rate per unit at one global threshold vs the unit's own."""
    rows = []
    for u in units:
        rows.append({"unit": u["unit"], "threshold": "one global threshold",
                     "rate": 100 * u["alarm_global"], "hours": u["hours"]})
        rows.append({"unit": u["unit"], "threshold": "per-unit threshold",
                     "rate": 100 * u["alarm_standby"], "hours": u["hours"]})
    df = pd.DataFrame(rows)
    order = [u["unit"] for u in units]
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("unit:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("threshold:N", sort=["one global threshold", "per-unit threshold"]),
        y=alt.Y("rate:Q", title="alarm rate (% of patient-hours)"),
        color=alt.Color("threshold:N",
                        scale=alt.Scale(domain=["one global threshold", "per-unit threshold"],
                                        range=[GRAY, MODEL]),
                        legend=alt.Legend(title=None, orient="top-right")),
        tooltip=[alt.Tooltip("unit:N"), alt.Tooltip("threshold:N"),
                 alt.Tooltip("rate:Q", title="alarm rate %", format=".2f"),
                 alt.Tooltip("hours:Q", title="test patient-hours")])
    rule = _hrule(100 * budget, f"{100 * budget:g}% budget")
    return (bars + rule).properties(height=height, width="container")
