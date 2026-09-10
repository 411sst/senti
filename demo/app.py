"""Sentinel demo dashboard -- minimal, three tabs, reads two JSON files.

    .venv/bin/streamlit run demo/app.py

Reads data/cache/demo/bundle.json (patient-level, this laptop only) and
cohort.json, both written by scripts/build_demo_bundle.py. Never touches the
cohort parquet, the agent bank or torch, so it opens in a couple of seconds.
"""

import json
import sys
import time
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "data" / "cache" / "demo"
for p in (REPO / "src", REPO / "demo"):
    sys.path.insert(0, str(p))

from charts import (  # noqa: E402
    COMP2, GRAY, MODEL, attention_chart, auprc_bar_chart, burden_chart,
    funnel_chart, news2_chart, race_chart, race_strip, risk_chart, sweep_chart,
)
from demo_bundle import (  # noqa: E402
    alarm_announcements, ollama_alive, stay_label, truncate_stay,
)
from explain.drivers import AGENT_LABELS, COMPONENT_LABELS  # noqa: E402

st.set_page_config(page_title="Sentinel", page_icon="🫀", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {max-width: 1180px; padding-top: 1.4rem; padding-bottom: 3rem;}
[data-testid="stMetricValue"] {font-size: 2.3rem; font-weight: 700;}
[data-testid="stMetricLabel"] {font-size: .85rem; opacity: .75;}
.chip {display:inline-block; padding: .15rem .55rem; margin: 0 .3rem .35rem 0;
       border-radius: 999px; border: 1px solid rgba(128,128,128,.45); font-size: .85rem; opacity: .6;}
.chip.on {border-color: #eb6834; color: #eb6834; font-weight: 600; opacity: 1;}
.card {border-left: 3px solid #2a78d6; padding: .7rem 1rem; margin: .3rem 0 .4rem;
       background: rgba(42,120,214,.07); font-size: 1.05rem; line-height: 1.5;}
.quiet {font-size: .85rem; opacity: .7;}
table.vit {width: 100%; max-width: 640px; border-collapse: collapse; font-size: .9rem; margin: .2rem 0 .5rem;}
.card {max-width: 760px;}
table.vit th, table.vit td {padding: .3rem .5rem; border-bottom: 1px solid rgba(128,128,128,.25);
       text-align: left; white-space: nowrap;}
table.vit th {font-weight: 500; font-size: .75rem; letter-spacing: .04em; text-transform: uppercase; opacity: .65;}
table.vit .num {text-align: right; font-variant-numeric: tabular-nums;}
table.vit .dim {opacity: .6;}
table.vit td.rising {color: #eb6834;} table.vit td.falling {color: #2a78d6;}
.alarmbanner {border-left: 4px solid #e34948; background: rgba(227,73,72,.08);
       padding: .55rem .9rem; margin: .25rem 0; font-size: 1rem; max-width: 760px;}
.alarmbanner.won {border-left-color: #2a78d6; background: rgba(42,120,214,.08);}
.bigread {font-size: 2.5rem; font-weight: 700; font-variant-numeric: tabular-nums;
       line-height: 1.1; padding-top: .2rem;}
.bigread .sub {font-size: .85rem; font-weight: 400; opacity: .7; display: block;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_json(path):
    return json.loads(Path(path).read_text())


if not (DEMO / "bundle.json").exists() or not (DEMO / "cohort.json").exists():
    st.error("No demo bundle found. Build it first:  "
             "`.venv/bin/python scripts/build_demo_bundle.py`")
    st.stop()

bundle = load_json(str(DEMO / "bundle.json"))
cohort = load_json(str(DEMO / "cohort.json"))
AGENTS = bundle["agents"]
COMPONENTS = bundle["news2_components"]
THR = bundle["thresholds"]
HAS_H2H = "h2h" in cohort
HAS_COMP = bool(bundle["stays"]) and "comp_risk" in bundle["stays"][0]
REBUILD_MSG = ("This bundle predates the head-to-head build. Rebuild it:  "
               "`.venv/bin/python scripts/build_demo_bundle.py`")


def vitals_table_html(stay, vitals, i, hour):
    """The five-row model-view vitals table; shared by Patient and Live."""
    trends = stay["contexts"].get(str(hour), {}).get("trends", {})
    rows = []
    for v in vitals:
        t = trends.get(v, {})
        age = stay["vitals_hrs_since"][v][i]
        digits = 1 if v == "temperature" else 0
        word = {"unknown": "no 4 h history"}.get(t.get("word"), t.get("word", "—"))
        rows.append(
            f"<tr><td>{COMPONENT_LABELS.get(v, v)}</td>"
            f"<td class='num'>{'n/a' if t.get('now') is None else f'{t['now']:.{digits}f}'}</td>"
            f"<td class='num dim'>{'—' if age is None else f'{age:.0f} h ago'}</td>"
            f"<td class='num'>{'—' if t.get('before') is None else f'{t['before']:.{digits}f}'}</td>"
            f"<td class='{word}'>{word}</td></tr>")
    return ("<table class='vit'><thead><tr><th>vital</th><th class='num'>value</th>"
            "<th class='num'>charted</th><th class='num'>4 h earlier</th><th>trend</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>")

st.markdown("## Sentinel · ICU early warning")
st.markdown('<div class="quiet">Eight small agents, one ~25k-parameter fusion, one local '
            'language model for the explanation. Every number here is held-out MIMIC-IV test '
            'data under frozen thresholds.</div>', unsafe_allow_html=True)

tab_live, tab_patient, tab_h2h, tab_cohort, tab_footprint = st.tabs(
    ["Live", "Patient", "Head-to-head", "Cohort", "Footprint"])


# --- Live replay -----------------------------------------------------------------

SPEEDS = {"1 s / h": 1.0, "2 s / h": 2.0, "4 s / h": 4.0}
LANES = (("Sentinel", "sentinel", MODEL),
         ("eCART-style GBM", "ecart", GRAY),
         ("YAIB LightGBM", "yaib", COMP2))

with tab_live:
    if not HAS_COMP:
        st.error(REBUILD_MSG)
    else:
        stays = bundle["stays"]
        live_labels = [stay_label(s) for s in stays]
        live_choice = st.selectbox("Stay", live_labels, label_visibility="collapsed",
                                   key="live_stay")
        l_stay = stays[live_labels.index(live_choice)]
        l_hours = l_stay["hours"]
        ss = st.session_state

        if ss.get("live_sid") != str(l_stay["stay_id"]):
            ss["live_sid"] = str(l_stay["stay_id"])
            ss["live_state"], ss["live_i"] = "idle", -1
            ss["live_fired"], ss["live_last_tick"] = set(), 0.0

        def _play():
            if ss["live_state"] == "finished":
                ss["live_i"], ss["live_fired"] = -1, set()
            ss["live_state"] = "playing"
            ss["live_last_tick"] = 0.0          # first l_tick advances immediately

        def _pause():
            ss["live_state"] = "paused"

        def _restart():
            ss["live_state"], ss["live_i"] = "idle", -1
            ss["live_fired"] = set()

        c_speed, c_play, c_pause, c_restart, c_note = st.columns([1.6, .7, .8, .9, 3])
        with c_speed:
            st.segmented_control("pace", list(SPEEDS), default="2 s / h",
                                 key="live_speed", label_visibility="collapsed")
        playing = ss["live_state"] == "playing"
        c_play.button("▶ Play", on_click=_play, disabled=playing)
        c_pause.button("⏸ Pause", on_click=_pause, disabled=not playing)
        c_restart.button("↺ Restart", on_click=_restart)
        with c_note:
            st.markdown('<div class="quiet">a replay of precomputed held-out scores at '
                        'accelerated clock speed — not a live model run; all three models '
                        'alarm at the same frozen volume</div>', unsafe_allow_html=True)

        l_tick = SPEEDS.get(ss.get("live_speed") or "2 s / h", 2.0)
        l_announce = alarm_announcements(
            l_stay, {**bundle.get("comparator_labels", {}), "sentinel": "Sentinel"})
        l_dom = (l_hours[0], max(l_hours[-1], l_stay.get("event_h") or 0) + 0.5)
        run_every = l_tick if ss["live_state"] == "playing" else None

        @st.fragment(run_every=run_every)
        def live_panel():
            now = time.monotonic()
            if (ss["live_state"] == "playing"
                    and now - ss["live_last_tick"] >= 0.8 * l_tick):
                ss["live_i"] = min(ss["live_i"] + 1, len(l_hours) - 1)
                ss["live_last_tick"] = now
            if ss["live_state"] == "playing" and ss["live_i"] >= len(l_hours) - 1:
                ss["live_state"] = "finished"
                st.rerun()                      # re-applies the decorator: timer off

            started = ss["live_i"] >= 0
            i = max(ss["live_i"], 0)
            sim_h = l_hours[i]
            t = truncate_stay(l_stay, i + 1)

            for h_a, name, msg in l_announce:
                if started and h_a <= sim_h and name not in ss["live_fired"]:
                    st.toast(msg, icon="🚨", duration="long")
                    ss["live_fired"].add(name)

            r1, r2, r3 = st.columns([1, 1.2, 2.2])
            r1.markdown(f'<div class="bigread">h{sim_h}<span class="sub">simulated '
                        f'hour</span></div>', unsafe_allow_html=True)
            r2.markdown(f'<div class="bigread">{l_stay["risk"]["composite"][i]:.3f}'
                        f'<span class="sub">Sentinel fused risk</span></div>',
                        unsafe_allow_html=True)
            status = " · ".join(
                f"<b style='color:{color}'>{label} ALARM</b>"
                if key in ss["live_fired"] else
                f"<span style='opacity:.55'>{label} quiet</span>"
                for label, key, color in LANES)
            r3.markdown(f'<div style="padding-top:1.1rem">{status}</div>',
                        unsafe_allow_html=True)

            st.altair_chart(risk_chart(t, THR["thr_model"], sim_h, domain=l_dom),
                            width="stretch")
            lanes = [{"lane": label,
                      "alarm_h": (l_stay["first_alarm"]["model"] if key == "sentinel"
                                  else l_stay["comp_first_alarm"][key]),
                      "color": color} for label, key, color in LANES]
            st.altair_chart(race_strip(lanes, l_dom,
                                       sim_h if started else l_hours[0] - 1,
                                       event_h=t.get("event_h")), width="stretch")

            for h_a, name, msg in l_announce:
                if name in ss["live_fired"]:
                    cls = "alarmbanner won" if name == "sentinel" else "alarmbanner"
                    st.markdown(f'<div class="{cls}">🚨 {msg}</div>',
                                unsafe_allow_html=True)

            if "sentinel" in ss["live_fired"]:
                card = l_stay["cards"].get(str(l_stay["first_alarm"]["model"]), "")
                if card:
                    st.markdown("**Why it fired — the model's reading at its alarm hour**")
                    st.markdown(f'<div class="card">{card}</div>', unsafe_allow_html=True)

            if ss["live_state"] == "finished":
                event_h = l_stay.get("event_h")
                if event_h is None:
                    st.success("Stay complete — no deterioration event, and no alarm "
                               "from any model. Silence is also a prediction.")
                else:
                    word = {"vent": "ventilation", "vaso": "vasopressors",
                            "death": "death"}.get(l_stay.get("event_type"), "event")
                    parts = []
                    for label, key, _ in LANES:
                        fa_h = (l_stay["first_alarm"]["model"] if key == "sentinel"
                                else l_stay["comp_first_alarm"][key])
                        if fa_h is None or fa_h > event_h:
                            parts.append(f"{label} missed")
                        else:
                            parts.append(f"{label} warned {event_h - fa_h:.1f} h early")
                    st.error(f"Event — {word} at hour {event_h:.1f} · " + " · ".join(parts))
            elif not started:
                st.caption("Press ▶ Play to replay this stay at the chosen pace. "
                           "Alarms pop the moment each model's frozen threshold is crossed.")

            st.markdown(vitals_table_html(l_stay, bundle["vitals"], i, sim_h),
                        unsafe_allow_html=True)

        live_panel()


# --- Patient -------------------------------------------------------------------

with tab_patient:
    stays = bundle["stays"]
    labels = [stay_label(s) for s in stays]
    choice = st.selectbox("Stay", labels, label_visibility="collapsed")
    stay = stays[labels.index(choice)]
    with st.expander("What you are looking at"):
        st.markdown(
            "- **Stay** — one held-out test ICU stay, labelled by what happened and how far "
            "ahead of the ML baselines Sentinel alarmed. Four are curated: two clear wins "
            "(ventilation, vasopressors), one narrow race, and a stay with no event.\n"
            "- **Risk chart** — Sentinel's fused risk each hour (two-head composite), computed only "
            "from information before that hour. Dashed line: the alarm threshold, frozen on "
            "validation at a 2% alarm budget, volume-matched to NEWS2. Blue dots: hours Sentinel is "
            "alarming. Red line: the event. Dotted black line: the selected hour.\n"
            "- **NEWS2 chart** — the deployed bedside score as charted, alarming at ≥ 11 (its own "
            "2%-budget threshold). Orange dots: its alarm hours. The gap between the first blue dot "
            "and the first orange dot is the lead.\n"
            "- **Hour slider** — scrubs the selected hour; it opens on Sentinel's first alarm.\n"
            "- **Where the fusion looked** — the attention weight the fusion put on each of the "
            "eight agents at that hour (they sum to 1), with each agent's own calibrated risk.\n"
            "- **NEWS2 sub-scores** — the seven components behind the NEWS2 total; the non-zero "
            "ones are highlighted. This is what a bedside score can and cannot see.\n"
            "- **Vital signs** — the last charted value before this hour (with its age), the value "
            "4 h earlier, and the trend word. Exactly what the model and the card were given.\n"
            "- **Explanation card** — the local language model's one-sentence reading of the "
            "numbers above. It is never told the risk or whether the alarm fired, so it can only "
            "name drivers. *Regenerate live* runs it on the laptop GPU.")
    hours = stay["hours"]
    fa = stay["first_alarm"]
    ut = stay["unit_thresholds"]

    left, right = st.columns([3, 1])
    with right:
        show_unit = st.toggle("per-unit standby threshold", value=False)
    with left:
        st.markdown(
            f'<div class="quiet">stay {stay["stay_id"]} · {stay["unit_group"]} ICU · '
            f'hours {hours[0]}–{hours[-1]} · alarm threshold {THR["thr_model"]:.3f} '
            f'(volume-matched to NEWS2 ≥ {THR["thr_news2"]:g}) · this unit would run at '
            f'{ut["thr_model_budget"]:.3f} ({ut["source"]})</div>', unsafe_allow_html=True)

    default_hour = fa["model"] if fa["model"] is not None else hours[0]
    hour = st.session_state.get("hour_" + str(stay["stay_id"]), default_hour)
    if hour not in hours:
        hour = default_hour

    st.altair_chart(risk_chart(stay, THR["thr_model"], hour,
                               unit_thr=ut["thr_model_budget"] if show_unit else None),
                    width="stretch")
    st.altair_chart(news2_chart(stay, THR["thr_news2"], hour), width="stretch")
    hour = st.slider("hour", min_value=hours[0], max_value=hours[-1], value=int(hour),
                     step=1, key="hour_" + str(stay["stay_id"]))
    i = hours.index(hour)

    status_m = "**ALARM**" if stay["alarm"]["model"][i] else "quiet"
    status_n = "**ALARM**" if stay["alarm"]["news2"][i] else "quiet"
    st.markdown(f"Hour {hour} — Sentinel {status_m} (risk {stay['risk']['composite'][i]:.3f}) · "
                f"NEWS2 {status_n} (score {stay['news2']['total'][i]:.0f})")

    # --- the evidence at this hour, stacked ---------------------------------
    st.markdown("**Where the fusion looked**")
    att_col, _ = st.columns([1.15, 1])
    with att_col:
        st.altair_chart(attention_chart(stay, hour, AGENTS, AGENT_LABELS), width="stretch")

    st.markdown("**NEWS2 sub-scores**")
    chips = "".join(
        f'<span class="chip{" on" if stay["news2"]["subscores"][c][i] > 0 else ""}">'
        f'{COMPONENT_LABELS.get(c, c)} {stay["news2"]["subscores"][c][i]:.0f}</span>'
        for c in COMPONENTS)
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown("**Vital signs the model saw**")
    ctxh = stay["contexts"].get(str(hour), {})
    st.markdown(vitals_table_html(stay, bundle["vitals"], i, hour),
                unsafe_allow_html=True)
    st.markdown('<div class="quiet">the last value charted before this hour — the only '
                'thing the model is allowed to see — with its age; "4 h earlier" is the '
                'trend anchor the card refers to</div>', unsafe_allow_html=True)

    st.markdown("**Why — the explanation card**")
    text = stay["cards"].get(str(hour), "")
    slot = st.empty()
    slot.markdown(f'<div class="card">{text or "no card cached for this hour"}</div>',
                  unsafe_allow_html=True)
    st.markdown(f'<div class="quiet">{bundle["llm"]} · plain prompt (the benchmarked one) · '
                f'the model is never told the risk or the alarm · attention from the '
                f'single-head ALL-8 refit, seed 42</div>', unsafe_allow_html=True)
    if st.button("Regenerate live (GPU)", key="regen_" + str(stay["stay_id"]) + "_" + str(hour)):
        if not ollama_alive():
            st.warning("ollama is not reachable — showing the cached card.")
        elif not ctxh:
            st.warning("no evidence packet cached for this hour.")
        else:
            from explain.rationales import SYSTEM_PROMPT, build_prompt, clean_reply
            from text_arm.llm_orchestrator import OllamaRunner
            runner = OllamaRunner(model=bundle["llm"], system=SYSTEM_PROMPT,
                                  num_predict=90, think=False)
            t0 = time.perf_counter()
            live = clean_reply(runner.generate(build_prompt(ctxh)))
            dt = time.perf_counter() - t0
            slot.markdown(f'<div class="card">{live or "(empty reply)"}</div>',
                          unsafe_allow_html=True)
            same = " · identical to the cached card" if live == text else ""
            st.caption(f"generated live in {dt:.1f} s on the local GPU{same}")


# --- Head-to-head ---------------------------------------------------------------

with tab_h2h:
    if not HAS_H2H:
        st.error(REBUILD_MSG)
    else:
        h2 = cohort["h2h"]
        lab = h2["labels"]
        models = h2["models"]
        vol = h2["alarm_rate_test"]
        st.markdown(
            '<div class="quiet">Every model spends the same alarm budget — thresholds '
            f'frozen on the validation freeze half at a {h2["matched"]:.2%} alarm volume. '
            'Achieved test volumes: '
            + " · ".join(f"{lab.get(m, m)} {vol[m]:.2%}" for m in models)
            + ".</div>", unsafe_allow_html=True)

        st.markdown("**How often is the anomaly caught**")
        mode_label = st.segmented_control(
            "catch definition", ["any catch", "≥ 6 h warning", "mid-stay only"],
            default="any catch", key="h2h_validity", label_visibility="collapsed")
        mode = {"any catch": "any", "≥ 6 h warning": "lead6",
                "mid-stay only": "midstay"}[mode_label or "any catch"]
        st.altair_chart(funnel_chart(
            [{"model": lab.get(m, m), "n": h2["funnel"][mode][m]["n"],
              "pct": h2["funnel"][mode][m]["pct"], "flagship": m == models[0]}
             for m in models]), width="stretch")
        defs = {
            "any": "first alarm at or before the event",
            "lead6": f"first alarm at least {h2['min_lead_h']:g} h before the event",
            "midstay": ("first alarm at or before the event AND strictly after the "
                        "stay's first scored hour — a change detection, not an "
                        "arrival-profile flag"),
        }
        st.caption(f"{h2['n_event_stays']:,} event stays · catch = {defs[mode]} · "
                   f"{h2['n_missed_by_all']:,} event stays are missed by all three models")

        st.markdown("**If it is caught, who catches it first** — jointly-caught stays")
        st.altair_chart(race_chart(h2["race"], lab), width="stretch")
        for r in h2["race"]:
            differ = (f" · when they differ ({r['n_differ']} stays) Sentinel is earlier "
                      f"**{r['earlier_pct']:.0%}** of the time, median gap "
                      f"{r['median_abs_gap_h']:.1f} h, mean {r['mean_gap_h']:+.1f} h"
                      if "earlier_pct" in r else
                      f" · median gap when they differ **{r['median_gap_h']:+.1f} h**")
            st.markdown(
                f"- vs **{lab.get(r['opponent'], r['opponent'])}** — exclusive catches "
                f"**{r['only_flagship']} : {r['only_opponent']}** (only Sentinel : only "
                f"the comparator){differ} · {r['n_joint']:,} jointly-caught stays")
        if "lead" in h2:
            depth = " · ".join(
                f"{lab.get(m, m)} median **{h2['lead'][m]['median_h']:.1f} h** "
                f"(mean {h2['lead'][m]['mean_h']:.1f}) over {h2['lead'][m]['n']:,} catches"
                for m in models)
            seed_note = ""
            if "seeds" in h2:
                sd = h2["seeds"]
                lm = sd["lead_median_h"]
                seed_note = (
                    f" Across {sd['n_seeds']} training seeds the flagship's median "
                    f"warning is {lm['mean']:.1f} h (range {lm['min']:.1f}–"
                    f"{lm['max']:.1f}) and earlier-when-differ holds at sign p<0.05 on "
                    + ", ".join(
                        f"{sd['earlier'][n]['sig_seeds']}/{sd['n_seeds']} seeds vs "
                        f"{lab.get(n, n)} (mean {sd['earlier'][n]['mean_pct']:.0%})"
                        for n in sd["earlier"]) + ".")
            st.caption(f"Warning depth on the stays each model catches: {depth}. "
                       f"The depth is the same for everyone — Sentinel's edge is giving "
                       f"that warning on more events, and being the earlier alarm when "
                       f"the two disagree (the means are tail-driven by long "
                       f"arrival-profile leads; quote the medians).{seed_note}")

        st.markdown("**Is it a valid catch — and what does it cost**")
        cost = h2["cost"]
        cols = st.columns(len(models))
        for col, m in zip(cols, models):
            col.metric(lab.get(m, m), f"{100 * cost['quiet_fp_rate'][m]:.1f}%")
            col.caption(f"of never-deteriorating stays get ≥ 1 alarm · "
                        f"{100 * cost['stay_precision'][m]:.0f}% of its alarmed stays "
                        f"truly deteriorate")

        if "sweep" in h2:
            st.markdown("**Does it hold at other budgets?**")
            st.altair_chart(sweep_chart(h2["sweep"], lab), width="stretch")
            sw = h2["sweep"]["rows"]
            gaps = [r[models[0]]["catch"] - max(r[m]["catch"] for m in models[1:])
                    for r in sw]
            st.caption(
                f"The same frozen-threshold protocol at {len(sw)} alarm budgets from "
                f"{100 * sw[0]['budget']:g}% to {100 * sw[-1]['budget']:g}% of "
                f"patient-hours. The gap over the best baseline never inverts "
                f"(smallest {100 * min(gaps):+.0f} points, largest "
                f"{100 * max(gaps):+.0f}); the dashed rule is the demo's operating "
                f"point. AUPRC and AUROC are threshold-free — the budget choice is "
                f"operational, not tuned.")

        st.markdown(f'<div class="quiet">{h2["caveat"]}</div>', unsafe_allow_html=True)


# --- Cohort -----------------------------------------------------------------------

with tab_cohort:
    h = cohort["headline"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUPRC · 5-seed mean", f"{h['auprc_mean']:.3f}")
    m1.caption(f"equal-information GBM {h['fair_traj']['auprc']:.3f} · NEWS2 {h['news2']['auprc']:.3f} "
               f"· CI clear of every comparator, 5/5 seeds")
    m2.metric("AUROC", f"{h['auroc_mean']:.3f}")
    m2.caption(f"death head alone {h['dhead_death_auroc_mean']:.3f} AUROC on death")
    if HAS_H2H:
        f_any = cohort["h2h"]["funnel"]["any"]
        best_comp = max(f_any[m]["pct"] for m in cohort["h2h"]["models"][1:])
        m3.metric("Event stays caught · same alarm volume",
                  f"{100 * f_any['sentinel']['pct']:.0f}%")
        m3.caption(f"best published-spec GBM catches {100 * best_comp:.0f}% at the same "
                   f"volume · see Head-to-head")
    else:
        m3.metric("Lead vs NEWS2 · 2% budget", f"+{h['lead_2pct_mean_h']:.1f} h")
        m3.caption(f"mean over five seeds · positive in {h['lead_positive_seeds']} · "
                   f"win rate {100 * h['win_rate_mean']:.0f}%")
    b = cohort["burden"]
    m4.metric("Alarm-burden spread across units", f"{b['spread_global_x']:.1f}× → {b['spread_standby_x']:.1f}×")
    m4.caption("one global threshold → seven per-unit thresholds, same model")

    st.markdown("**Ranking quality, six systems**")
    st.altair_chart(auprc_bar_chart(cohort["scatter"]), width="stretch")
    st.caption(f"{h['n_test_hours']:,} held-out patient-hours, {h['n_stays']:,} stays, prevalence "
               f"{h['prevalence_pct']:.2f}%. Thresholds frozen on validation; seed 42 shown for "
               f"the flagship, the five-seed mean is {h['auprc_mean']:.3f}.")

    st.markdown("**Same model, seven thresholds**")
    st.altair_chart(burden_chart(b["units"], b["budget"]), width="stretch")
    st.caption("Realised alarm rate per ICU type on the test split. Per-unit thresholds are frozen "
               "on each unit's own validation half; thin units fall back to the global point.")

    with st.expander("Per-seed detail"):
        st.dataframe([{
            "seed": s["seed"], "AUPRC": f"{s['auprc']:.4f}", "AUROC": f"{s['auroc']:.4f}",
            "death-head AUPRC (death)": f"{s['dhead_death_auprc']:.3f}",
            "lead median (h)": f"{s['lead_median_h']:+.1f} [{s['lead_median_ci'][0]:+.1f}, {s['lead_median_ci'][1]:+.1f}]",
            "win rate": f"{100 * s['win_rate']:.0f}%",
            "paired stays": s["n_pair"],
            "ΔAUPRC vs GBM, 95% CI": f"[{s['delta_vs_fair_ci'][0]:+.4f}, {s['delta_vs_fair_ci'][1]:+.4f}]",
        } for s in cohort["seeds"]], hide_index=True, width="stretch")

    st.markdown("**What we do not claim**")
    for c in cohort["caveats"]:
        st.markdown(f"- {c}")


# --- Footprint ------------------------------------------------------------------

with tab_footprint:
    f = cohort["footprint"]
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Fusion parameters · per head", f"{f['n_params_per_head']:,}")
    a2.metric("Two-head total", f"{f['n_params_two_head']:,}")
    a3.metric("Checkpoint on disk", f"{f['checkpoint_bytes'] / 1024:.0f} KB")
    a4.metric("Batch scorer peak memory (incl. 2 GB cached latents)", f"{f['batch_rss_mb'] / 1024:.1f} GB")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric(f"CPU time per patient-hour · {f['torch_threads']} thread", f"{f['us_per_patient_hour']:.1f} µs")
    b2.metric("Patient-hours per second", f"{f['patient_hours_per_s']:,.0f}")
    b3.metric(f"One stay, {f['single_stay_hours']} hours", f"{f['single_stay_ms']:.1f} ms")
    card = f["llm_card_s_mean"]
    b4.metric("Explanation card · local GPU", "n/a" if card is None else f"{card:.2f} s")
    st.caption(f"Measured on this laptop at build time ({cohort['built_at']}, commit "
               f"{cohort['git_commit']}); {f['batch_patient_hours']:,} patient-hours scored in "
               f"{f['batch_wall_s']:.1f} s.")

    st.markdown("**What runs where**")
    st.markdown(
        "Each agent is a small network (MLP / causal CNN, thousands of parameters) that reads one "
        "modality and emits a latent and a calibrated risk per patient-hour; those emissions are "
        "cached, so scoring is a lookup plus the fusion. The fusion — attention across agents "
        "within the hour and a causal temporal convolution over ~29 h — runs on **CPU, one thread, "
        "bit-identically**, as a treatment head and a death head composed by noisy-OR. The alarm "
        "policy is **seven numbers keyed by ICU type**, frozen on validation and never touched by "
        "the model. The one language model, `" + bundle["llm"] + "`, runs locally under ollama on "
        "the GPU and does one thing: turn the fusion's own attention, agent risks and NEWS2 "
        "sub-scores into a sentence. It is never told the fused risk or whether the alarm fired, "
        "so it cannot alter a score, a threshold or an alarm — the decision path is LLM-free.")
