"""Altair charts for the demo dashboard and the leave-behind (altair + pandas only)."""

import altair as alt
import pandas as pd

MODEL = "#2a78d6"      # Sentinel / the fused risk
NEWS2 = "#eb6834"      # the deployed comparator
EVENT = "#e34948"      # the deterioration event
GRAY = "#8a8f98"       # thresholds, comparators
INK = "#2b2f36"        # the selected hour
COMP2 = "#5c6470"      # the second ML comparator lane
TIE = "#c9cdd3"        # same-hour segment in the race bars

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


def risk_chart(stay, thr_model, hour, unit_thr=None, height=250, domain=None):
    """Composite risk over the stay: threshold, alarms, the event, the cursor.

    ``domain`` pins the x axis (the replay draws a growing series on the
    full-stay domain); by default it is derived from the stay itself.
    """
    domain = tuple(domain) if domain is not None else _domain(stay)
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


def auprc_bar_chart(points, height=240):
    """Ranking quality of the six systems, one bar each; the flagship in blue."""
    df = pd.DataFrame(points).sort_values("auprc", ascending=False)
    df["color"] = df["is_flagship"].map({True: MODEL, False: GRAY})
    df["label"] = df["auprc"].map(lambda v: f"{v:.3f}")
    order = df["name"].tolist()
    base = alt.Chart(df).encode(
        y=alt.Y("name:N", sort=order, title=None,
                axis=alt.Axis(labelLimit=260, labelOverlap=False, labelPadding=6)),
        tooltip=[alt.Tooltip("name:N", title="model"),
                 alt.Tooltip("auprc:Q", format=".3f"),
                 alt.Tooltip("auroc:Q", format=".3f")])
    bars = base.mark_bar(size=16).encode(
        x=alt.X("auprc:Q", title="ranking quality (AUPRC, held-out test)",
                scale=alt.Scale(domain=[0, 0.36])),
        color=alt.Color("color:N", scale=None))
    text = base.mark_text(align="left", dx=5, color="#7d8792", fontSize=11).encode(
        x="auprc:Q", text="label:N")
    return (bars + text).properties(height=height, width="container")


def funnel_chart(rows, height=170):
    """Catch rate over the event stays, one bar per model at the same budget.

    ``rows``: ``[{"model", "n", "pct", "flagship"}]`` in display order.
    """
    df = pd.DataFrame(rows)
    df["rate"] = 100 * df["pct"]
    df["color"] = df["flagship"].map({True: MODEL, False: GRAY})
    df["label"] = [f"{r.n} stays · {100 * r.pct:.1f}%" for r in df.itertuples()]
    order = df["model"].tolist()
    base = alt.Chart(df).encode(
        y=alt.Y("model:N", sort=order, title=None,
                axis=alt.Axis(labelLimit=260, labelPadding=6)),
        tooltip=[alt.Tooltip("model:N"), alt.Tooltip("n:Q", title="event stays caught"),
                 alt.Tooltip("rate:Q", title="% of event stays", format=".1f")])
    bars = base.mark_bar(size=20).encode(
        x=alt.X("rate:Q", title="% of event stays caught",
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("color:N", scale=None))
    text = base.mark_text(align="left", dx=5, color="#7d8792", fontSize=11).encode(
        x="rate:Q", text="label:N")
    return (bars + text).properties(height=height, width="container")


def race_chart(race_rows, labels, height=150):
    """Who fires first among jointly-caught stays, one stacked bar per opponent."""
    segs = ("Sentinel first", "same hour", "comparator first")
    colors = {segs[0]: MODEL, segs[1]: TIE, segs[2]: COMP2}
    rows = []
    for r in race_rows:
        opp = labels.get(r["opponent"], r["opponent"])
        parts = [(segs[0], r["flagship_first"], r["flagship_first_pct"]),
                 (segs[1], r["tie"], r["tie_pct"]),
                 (segs[2], r["opponent_first"], r["opponent_first_pct"])]
        left = 0.0
        for seg, n, pct in parts:
            rows.append({"opponent": f"vs {opp}", "segment": seg, "n": n,
                         "pct": 100 * pct, "mid": 100 * (left + pct / 2),
                         "label": f"{100 * pct:.0f}%" if pct >= 0.06 else ""})
            left += pct
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df["opponent"]))
    y = alt.Y("opponent:N", sort=order, title=None,
              axis=alt.Axis(labelLimit=260, labelPadding=6))
    bars = alt.Chart(df).mark_bar(size=26).encode(
        y=y, x=alt.X("pct:Q", title="% of jointly-caught event stays", stack="zero",
                     scale=alt.Scale(domain=[0, 100])),
        order=alt.Order("segment:N", sort="ascending"),
        color=alt.Color("segment:N",
                        scale=alt.Scale(domain=list(segs),
                                        range=[colors[s] for s in segs]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("opponent:N"), alt.Tooltip("segment:N"),
                 alt.Tooltip("n:Q", title="stays"),
                 alt.Tooltip("pct:Q", title="%", format=".1f")])
    text = alt.Chart(df).mark_text(color="white", fontSize=11, fontWeight="bold").encode(
        y=y, x="mid:Q", text="label:N")
    return (bars + text).properties(height=height, width="container")


def sweep_chart(sweep, labels, height=260):
    """Catch rate vs alarm budget, one line per model; the operating point ruled.

    ``sweep``: the cohort's ``h2h["sweep"]`` block — ``models``, ``operating``
    (the frozen matched volume), and ``rows`` of ``{budget, <model>: {catch,
    quiet_fp}}``.
    """
    names = [labels.get(m, m) for m in sweep["models"]]
    colors = dict(zip(names, [MODEL, GRAY, COMP2]))
    rows = []
    for r in sweep["rows"]:
        for m, name in zip(sweep["models"], names):
            rows.append({"budget": 100 * r["budget"], "model": name,
                         "catch": 100 * r[m]["catch"],
                         "quiet_fp": 100 * r[m]["quiet_fp"]})
    df = pd.DataFrame(rows)
    x = alt.X("budget:Q", title="alarm budget (% of patient-hours, frozen on validation)",
              scale=alt.Scale(domain=[0, 100 * sweep["rows"][-1]["budget"] * 1.04],
                              nice=False))
    y = alt.Y("catch:Q", title="% of event stays caught",
              scale=alt.Scale(domain=[0, 100]))
    color = alt.Color("model:N",
                      scale=alt.Scale(domain=names, range=[colors[n] for n in names]),
                      legend=alt.Legend(title=None, orient="top"))
    tooltip = [alt.Tooltip("model:N"),
               alt.Tooltip("budget:Q", title="budget %", format=".2f"),
               alt.Tooltip("catch:Q", title="caught %", format=".1f"),
               alt.Tooltip("quiet_fp:Q", title="quiet-stay FP %", format=".1f")]
    lines = alt.Chart(df).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
    pts = alt.Chart(df).mark_point(filled=True, size=60).encode(
        x=x, y=y, color=color, tooltip=tooltip)
    op = pd.DataFrame({"budget": [100 * sweep["operating"]],
                       "label": ["operating point"]})
    rule = alt.Chart(op).mark_rule(color=GRAY, strokeDash=[5, 4]).encode(x=x)
    text = alt.Chart(op).mark_text(align="left", dx=5, color=GRAY, fontSize=11).encode(
        x=x, text="label:N", y=alt.value(10))
    return alt.layer(lines, pts, rule, text).properties(height=height, width="container")


def race_strip(lanes, domain, sim_hour, event_h=None, height=140):
    """The replay's alarm race: one lane per model, a dot the moment it fires.

    ``lanes``: ``[{"lane", "alarm_h", "color"}]`` in display order; a lane's dot
    appears once ``sim_hour`` has reached its ``alarm_h``.
    """
    order = [l["lane"] for l in lanes]
    colors = {l["lane"]: l["color"] for l in lanes}
    y = alt.Y("lane:N", sort=order, title=None,
              axis=alt.Axis(labelLimit=220, labelPadding=6))
    base_df = pd.DataFrame({"lane": order})
    track = alt.Chart(base_df).mark_rule(color="#e3e5e8", strokeWidth=2).encode(y=y)
    fired = pd.DataFrame([
        {"lane": l["lane"], "hour": l["alarm_h"], "label": f"h{l['alarm_h']:g}"}
        for l in lanes if l["alarm_h"] is not None and l["alarm_h"] <= sim_hour])
    layers = [track]
    if len(fired):
        pts = alt.Chart(fired).mark_point(filled=True, size=160).encode(
            x=_x(domain), y=y,
            color=alt.Color("lane:N", scale=alt.Scale(domain=order,
                                                      range=[colors[k] for k in order]),
                            legend=None),
            tooltip=[alt.Tooltip("lane:N"), alt.Tooltip("hour:Q", title="alarm at hour")])
        txt = alt.Chart(fired).mark_text(align="left", dx=10, dy=-1, fontSize=11,
                                         fontWeight="bold", color=INK).encode(
            x=_x(domain), y=y, text="label:N")
        layers += [pts, txt]
    else:
        # keep the x scale on screen even before the first alarm
        layers.append(alt.Chart(pd.DataFrame({"hour": [domain[0]], "lane": [order[0]]}))
                      .mark_point(opacity=0).encode(x=_x(domain), y=y))
    if event_h is not None:
        layers.append(_vrule(event_h, "event", EVENT, domain))
    layers.append(_vrule(sim_hour, "", INK, domain, dash=(1, 3)))
    return alt.layer(*layers).properties(height=height, width="container")


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
