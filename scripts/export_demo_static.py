"""Render the aggregate-only leave-behind page from data/cache/demo/cohort.json.

    python scripts/export_demo_static.py [--out PATH]

This script can only open cohort.json -- never the patient-level bundle -- and
refuses to write a page that carries an id or an id-like number. The charts
are the dashboard's own Altair specs, embedded as JSON and drawn by vega-embed
from cdn.jsdelivr.net (the one chart CDN the artifact host allows).
"""

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "demo"))

from charts import auprc_bar_chart, burden_chart, funnel_chart, race_chart  # noqa: E402
from demo_bundle import assert_aggregate_only, assert_no_identifiers  # noqa: E402

COHORT = REPO / "data" / "cache" / "demo" / "cohort.json"
DEFAULT_OUT = Path("/tmp/claude-1000/-home-rahul-code-sentinel/"
                   "99cb1c7b-12cf-46b0-92d7-e7a3679b0cfb/scratchpad/sentinel-in-numbers.html")
BRIEF_URL = "https://claude.ai/code/artifact/38906e0e-3eb2-428e-aeb6-7a24fded5628"

CHART_CONFIG = {
    "background": "transparent",
    "axis": {"labelColor": "#7d8792", "titleColor": "#7d8792", "gridColor": "#80808033",
             "domainColor": "#80808066", "tickColor": "#80808066",
             "labelFont": "Fira Code, monospace", "titleFont": "Spline Sans, sans-serif",
             "labelFontSize": 11, "titleFontSize": 12},
    "legend": {"labelColor": "#7d8792", "labelFont": "Spline Sans, sans-serif"},
    "view": {"stroke": None},
    "font": "Spline Sans, sans-serif",
}

CSS = """
:root{color-scheme:light;
  --paper:#F4F6F7; --surface:#FBFCFC; --ink:#1B2230; --ink-2:#3A4452; --muted:#66717F;
  --rule:#D6DCE2; --rule-soft:#E6EAEE; --model:#2a78d6; --model-soft:#2a78d614;
  --news2:#eb6834; --event:#e34948;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
  --paper:#0F1419; --surface:#151B22; --ink:#E6EAEE; --ink-2:#C3CAD2; --muted:#8D98A5;
  --rule:#273038; --rule-soft:#1E262E; --model:#5B9BE8; --model-soft:#5B9BE81F;
  --news2:#F08A5E; --event:#EC6B6A;}}
:root[data-theme="dark"]{color-scheme:dark;
  --paper:#0F1419; --surface:#151B22; --ink:#E6EAEE; --ink-2:#C3CAD2; --muted:#8D98A5;
  --rule:#273038; --rule-soft:#1E262E; --model:#5B9BE8; --model-soft:#5B9BE81F;
  --news2:#F08A5E; --event:#EC6B6A;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"Spline Sans","Segoe UI",system-ui,sans-serif;font-size:16px;line-height:1.55;
  -webkit-font-smoothing:antialiased}
.page{max-width:1040px;margin:0 auto;padding:0 24px 72px}
h1,h2{font-family:Archivo,"Helvetica Neue",Arial,sans-serif;margin:0;text-wrap:balance;letter-spacing:-.015em}
h1{font-size:clamp(1.9rem,4.2vw,2.7rem);font-weight:800;line-height:1.05}
h2{font-size:1.25rem;font-weight:700;margin-bottom:10px}
p{margin:0}
a{color:var(--model)}
:focus-visible{outline:2px solid var(--model);outline-offset:3px}
.mono{font-family:"Fira Code",ui-monospace,monospace}
.mast{padding:52px 0 26px;border-bottom:2px solid var(--ink)}
.eyebrow{font-family:"Fira Code",monospace;font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin-bottom:16px}
.deck{margin-top:14px;color:var(--ink-2);max-width:66ch;font-size:1.02rem}
.deck b{color:var(--ink)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;background:var(--rule);border-bottom:1px solid var(--rule)}
.stat{background:var(--paper);padding:20px 18px}
.stat .n{font-family:Archivo,sans-serif;font-weight:800;font-size:2rem;line-height:1;display:block;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.stat .n.g{color:var(--model)}
.stat .k{display:block;margin-top:8px;font-size:.8rem;line-height:1.45;color:var(--muted)}
section{margin-top:48px}
.panel{background:var(--surface);border:1px solid var(--rule-soft);border-radius:10px;padding:18px 16px 10px;overflow-x:auto}
.vis{width:100%;min-height:260px}
.cap{margin-top:10px;font-size:.84rem;line-height:1.5;color:var(--muted);max-width:80ch}
.twrap{overflow-x:auto;border:1px solid var(--rule-soft);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.86rem;min-width:640px}
th,td{padding:8px 12px;text-align:right;border-bottom:1px solid var(--rule-soft);font-variant-numeric:tabular-nums;font-family:"Fira Code",monospace;font-size:.8rem}
th:first-child,td:first-child{text-align:left}
thead th{font-size:.68rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:500;border-bottom:1.5px solid var(--rule)}
tbody tr:last-child td{border-bottom:none}
.caveats{display:flex;flex-direction:column;gap:12px;max-width:72ch}
.caveat{border-left:3px solid var(--event);padding:8px 14px;color:var(--ink-2);font-size:.95rem}
.caveat b{color:var(--ink)}
.prose{max-width:72ch;color:var(--ink-2)}
.prose b{color:var(--ink)}
.prose code{font-family:"Fira Code",monospace;font-size:.85em;background:var(--model-soft);padding:1px 5px;border-radius:4px}
footer{margin-top:60px;padding-top:18px;border-top:2px solid var(--ink);font-size:.8rem;color:var(--muted);line-height:1.6}
@media (max-width:640px){.page{padding:0 16px 56px}.mast{padding-top:36px}}
"""


def stat(n, k, green=False):
    return (f'<div class="stat"><span class="n{" g" if green else ""}">{html.escape(n)}</span>'
            f'<span class="k">{html.escape(k)}</span></div>')


def render(c):
    h, b, f = c["headline"], c["burden"], c["footprint"]
    ranking = auprc_bar_chart(c["scatter"], height=240).configure(**CHART_CONFIG).to_json()
    burden = burden_chart(b["units"], b["budget"], height=250).configure(**CHART_CONFIG).to_json()
    date = datetime.now(timezone.utc).strftime("%d %b %Y")

    h2 = c.get("h2h")
    h2h_section, h2h_embeds = "", ""
    if h2:
        lab, models, cost = h2["labels"], h2["models"], h2["cost"]
        f_any = h2["funnel"]["any"]
        funnel_json = funnel_chart(
            [{"model": lab.get(m, m), "n": f_any[m]["n"], "pct": f_any[m]["pct"],
              "flagship": m == models[0]} for m in models],
            height=170).configure(**CHART_CONFIG).to_json()
        race_json = race_chart(h2["race"], lab, height=150).configure(**CHART_CONFIG).to_json()
        excl = " and ".join(
            f"{r['only_flagship']}:{r['only_opponent']} vs the "
            f"{html.escape(lab.get(r['opponent'], r['opponent']))}" for r in h2["race"])
        gaps = "; ".join(
            f"median gap when they differ {r['median_gap_h']:+.1f} h vs the "
            f"{html.escape(lab.get(r['opponent'], r['opponent']))}" for r in h2["race"])
        comp_fp = " · ".join(
            f"{html.escape(lab.get(m, m))} {100 * cost['quiet_fp_rate'][m]:.1f}%"
            for m in models[1:])
        comp_prec = " · ".join(
            f"{html.escape(lab.get(m, m))} {100 * cost['stay_precision'][m]:.0f}%"
            for m in models[1:])
        h2h_section = f"""
<section>
  <h2>Head-to-head at the same alarm volume</h2>
  <p class="prose">Sentinel against the two published-spec ML baselines retrained on this
  cohort. Every model's threshold is frozen on the validation freeze half at the same
  {h2['matched']:.2%} alarm volume — no model spends more alerts than another.</p>
  <div class="stats" style="margin-top:16px">
    {stat(f"{100 * f_any[models[0]]['pct']:.0f}%",
          f"of {h2['n_event_stays']:,} event stays caught by Sentinel — first alarm at or "
          f"before the event · {h2['n_missed_by_all']:,} stays are missed by all three models",
          green=True)}
    {stat(f"{h2['race'][0]['only_flagship']} : {h2['race'][0]['only_opponent']}",
          f"exclusive catches vs the {lab.get(h2['race'][0]['opponent'], '')} "
          "(only Sentinel : only the comparator)", green=True)}
    {stat(f"{100 * cost['quiet_fp_rate'][models[0]]:.1f}%",
          f"of never-deteriorating stays get ≥ 1 Sentinel alarm · {comp_fp}")}
    {stat(f"{100 * cost['stay_precision'][models[0]]:.0f}%",
          f"of Sentinel's alarmed stays truly deteriorate · {comp_prec}")}
  </div>
  <div class="panel" style="margin-top:16px"><div id="funnel" class="vis" style="min-height:190px"></div></div>
  <div class="panel" style="margin-top:16px"><div id="race" class="vis" style="min-height:170px"></div></div>
  <p class="cap">Top: share of event stays caught at the same alarm volume. Bottom: among
  jointly-caught stays, who fires first ({gaps}; positive = Sentinel earlier). Exclusive
  catches are {excl}. {html.escape(h2['caveat'].capitalize())}.</p>
</section>
"""
        h2h_embeds = (f"  var funnel = {funnel_json};\n  var race = {race_json};\n"
                      "  vegaEmbed(\"#funnel\", funnel, opts);\n"
                      "  vegaEmbed(\"#race\", race, opts);\n")

    seed_rows = "".join(
        f"<tr><td>{s['seed']}</td><td>{s['auprc']:.4f}</td><td>{s['auroc']:.4f}</td>"
        f"<td>{s['dhead_death_auprc']:.3f}</td>"
        f"<td>{s['lead_median_h']:+.1f} [{s['lead_median_ci'][0]:+.1f}, {s['lead_median_ci'][1]:+.1f}]</td>"
        f"<td>{100 * s['win_rate']:.0f}%</td><td>{s['n_pair']}</td>"
        f"<td>[{s['delta_vs_fair_ci'][0]:+.4f}, {s['delta_vs_fair_ci'][1]:+.4f}]</td></tr>"
        for s in c["seeds"])
    caveats = []
    for t in c["caveats"]:
        lead, _, body = t.partition(":")
        if body and len(lead) <= 48:
            caveats.append(f'<div class="caveat"><b>{html.escape(lead)}:</b>'
                           f'{html.escape(body)}</div>')
        else:
            caveats.append(f'<div class="caveat">{html.escape(t)}</div>')
    caveats = "".join(caveats)
    card = f["llm_card_s_mean"]

    return f"""<title>Sentinel in Numbers</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@700;800&family=Spline+Sans:wght@400;600&family=Fira+Code:wght@400;600&display=swap">
<script src="https://cdn.jsdelivr.net/npm/vega@5.30.0/build/vega.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5.21.0/build/vega-lite.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.26.0/build/vega-embed.min.js"></script>
<style>{CSS}</style>
<div class="page">
<header class="mast">
  <div class="eyebrow">Sentinel &middot; ICU early warning &middot; demo companion &middot; {date}</div>
  <h1>What the demo showed, in numbers</h1>
  <p class="deck">Eight small agents, one ~25k-parameter fusion, one local language model for the
  explanation card. Everything below is <b>held-out MIMIC-IV test data under frozen thresholds</b>;
  the patient-level view from the live walkthrough is not reproduced here by design.</p>
</header>

<div class="stats">
  {stat(f"{h['auprc_mean']:.3f}", f"AUPRC, five-seed mean · equal-information GBM {h['fair_traj']['auprc']:.3f}, NEWS2 {h['news2']['auprc']:.3f} · paired CI clear of every comparator, 5/5 seeds")}
  {stat(f"{h['auroc_mean']:.3f}", f"AUROC · the death head alone reaches {h['dhead_death_auroc_mean']:.3f} on death")}
  {stat(f"{100 * h2['funnel']['any'][h2['models'][0]]['pct']:.0f}%", f"of event stays caught at the same alarm volume · best published-spec GBM {100 * max(h2['funnel']['any'][m]['pct'] for m in h2['models'][1:]):.0f}% · thresholds frozen on validation", green=True) if h2 else stat(f"+{h['lead_2pct_mean_h']:.1f} h", f"mean first-alarm lead vs NEWS2 at a 2% budget · positive in {h['lead_positive_seeds']} seeds · win rate {100 * h['win_rate_mean']:.0f}%", green=True)}
  {stat(f"{b['spread_global_x']:.1f}× → {b['spread_standby_x']:.1f}×", "alarm-burden spread across ICU types: one global threshold → seven per-unit thresholds, same model", green=True)}
</div>

<section>
  <h2>Ranking quality, six systems</h2>
  <div class="panel"><div id="ranking" class="vis" style="min-height:260px"></div></div>
  <p class="cap">{h['n_test_hours']:,} held-out patient-hours, {h['n_stays']:,} stays, prevalence
  {h['prevalence_pct']:.2f}%. Thresholds frozen on validation. Seed 42 shown for the flagship —
  the five-seed mean is {h['auprc_mean']:.3f}, paired CI clear of every comparator on all five.</p>
</section>
{h2h_section}
<section>
  <h2>Same model, seven thresholds</h2>
  <div class="panel"><div id="burden" class="vis"></div></div>
  <p class="cap">Realised alarm rate per ICU type on the test split. Per-unit thresholds are frozen on
  each unit's own validation half; thin units keep the global operating point. Choosing the budgets is
  governance, not modelling — the model does not change.</p>
</section>

<section>
  <h2>Five training seeds</h2>
  <div class="twrap"><table>
    <thead><tr><th>seed</th><th>AUPRC</th><th>AUROC</th><th>death-head AUPRC</th><th>lead median, h [95% CI]</th><th>win</th><th>paired stays</th><th>ΔAUPRC vs GBM, 95% CI</th></tr></thead>
    <tbody>{seed_rows}</tbody>
  </table></div>
</section>

<section>
  <h2>What we do not claim</h2>
  <div class="caveats">{caveats}</div>
</section>

<section>
  <h2>Footprint</h2>
  <div class="stats">
    {stat(f"{f['n_params_per_head']:,}", "fusion parameters per head; two-head total " + f"{f['n_params_two_head']:,}" + f" · checkpoint {f['checkpoint_bytes'] / 1024:.0f} KB")}
    {stat(f"{f['us_per_patient_hour']:.1f} µs", f"CPU time per patient-hour at {f['torch_threads']} thread · {f['patient_hours_per_s']:,.0f} patient-hours per second")}
    {stat(f"{f['single_stay_ms']:.1f} ms", f"one stay of {f['single_stay_hours']} hours, warm, on CPU")}
    {stat("n/a" if card is None else f"{card:.2f} s", "one explanation card, local GPU (ollama), never told the risk or the alarm")}
  </div>
  <p class="prose" style="margin-top:16px">Each agent is a small network that reads one modality and emits a
  latent and a calibrated risk per patient-hour; the emissions are cached, so scoring is a lookup plus the
  fusion. The fusion — attention across agents within the hour, a causal temporal convolution over ~29 h —
  runs on <b>CPU, one thread, bit-identically</b>, as a treatment head and a death head composed by
  noisy-OR. The alarm policy is <b>seven numbers keyed by ICU type</b>. The one language model,
  <code>{html.escape(f['llm_model'])}</code>, runs locally and only turns the fusion's own attention, agent
  risks and NEWS2 sub-scores into a sentence; it cannot alter a score, a threshold or an alarm.
  Measured {html.escape(c['built_at'])} at commit {html.escape(c['git_commit'])}.</p>
</section>

<footer>
  <p>Aggregate results only — no patient-level data appears on this page. MIMIC-IV v3.1 under the
  PhysioNet credentialed-access DUA. The full attack-by-attack defence of the architecture, with
  references, is the companion <a href="{BRIEF_URL}">Sentinel Defence Brief</a>.</p>
</footer>
</div>
<script>
(function(){{
  var opts = {{actions:false, renderer:"svg"}};
  var ranking = {ranking};
  var burden = {burden};
  vegaEmbed("#ranking", ranking, opts);
  vegaEmbed("#burden", burden, opts);
{h2h_embeds}}})();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    cohort = json.loads(COHORT.read_text())
    assert_aggregate_only(cohort)
    page = render(cohort)
    assert_no_identifiers(page)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"[export] {out} ({len(page.encode()) / 1024:.0f} KB, aggregate-only, id-free)")


if __name__ == "__main__":
    main()
