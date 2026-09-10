# Sentinel demo dashboard — every component, what it is, what it indicates

The app (`demo/app.py`) reads two files written once by `scripts/build_demo_bundle.py`:
`data/cache/demo/bundle.json` (patient-level, four curated held-out test stays — stays
on the laptop) and `data/cache/demo/cohort.json` (aggregate-only). Nothing on the page is
computed live except the optional explanation-card regeneration. Every number is
held-out MIMIC-IV test data under thresholds frozen on validation.

Five tabs: **Live** (a replay of one stay with alarms popping in real time), **Patient**
(why an alarm fired), **Head-to-head** (Sentinel vs the two published-spec ML baselines at
the same alarm volume), **Cohort** (does it hold up), **Footprint** (what it costs to run).

---

## Live tab

A replay of one curated stay at accelerated clock speed — **precomputed held-out scores,
not a live model run** (the honesty line sits next to the controls). All three models —
Sentinel, the eCART-style GBM and the YAIB LightGBM — alarm at thresholds frozen on the
validation freeze half at the same alarm volume, so the race is budget-fair by construction.
NEWS2 does not appear on this tab.

### Controls
Stay selector (same labels as the Patient tab), a pace control (1 / 2 / 4 seconds per
simulated hour), and **Play / Pause / Restart**. The replay has four states — idle,
playing, paused, finished — and a stay change or Restart resets everything. Pressing Play
after a finished run starts over.

### What happens while it plays
Each tick advances one simulated hour: the fused-risk line grows on a fixed full-stay
axis, the big readouts (simulated hour, current risk, per-model alarm status) update, and
the model-view vitals table tracks the current hour. The moment a model's frozen threshold
is crossed, a toast pops, a persistent banner appears (blue-edged for Sentinel, red-edged
for the comparators), and its dot lands on the **race strip** — three lanes, one per
model, with the alarm hour labelled. When Sentinel fires, its cached explanation card for
that exact hour slides in ("why it fired", no click needed). Each alarm announces exactly
once per run, tracked in session state, so pausing, resuming, or touching another widget
never re-fires a toast.

### End of stay
On an event stay the replay ends with the event banner: the event type and hour, and per
model either "warned N h early" or "missed". On the quiet stay it ends with "no
deterioration event, and no alarm from any model — silence is also a prediction."

### One operational note
The tick timer belongs to the page, not the tab: if the replay is left playing while
another tab is selected, it keeps advancing and its toasts still pop. **Pause before
switching tabs** during a presentation.

---

## Patient tab

### Stay selector
**What it is.** A drop-down of four held-out test ICU stays, labelled by what happened
and who alarmed first, e.g. *Ventilation — model alarm 16 h before NEWS2 · surgical ICU*.
Stay ids never appear in the label.

**How they were chosen.** Deterministically, from the whole test split at the global 2%
thresholds (`src/demo_bundle.py: select_demo_stays`):

| role | rule |
|---|---|
| ventilation win, vasopressor win | both systems alarm before the event; Sentinel at least 2 h ahead; at least 12 h of stay before the event and 5 h before the alarm (so the alarm hour has a 4 h trend); alarm inside the 24 h label horizon. The **clearest** win in the pool is taken (largest lead). |
| NEWS2-first | mirror image (NEWS2 at least 2 h ahead). The **most typical** loss is taken — the one closest to the pool's median lead — not the mildest. |
| no event | no deterioration event, and neither system ever alarms; 24–72 h long. |

**What it indicates.** Three of the four are the cases the cohort numbers are made of:
a win, a loss, a true negative. The label states the lead, so a showcase win is presented
as exactly that.

### "What you are looking at" expander
A one-line-per-component reminder of this document, for the presenter.

### Stay caption
`stay <id> · <unit> ICU · hours h0–h1 · alarm threshold 0.281 (volume-matched to NEWS2 ≥ 11) ·
this unit would run at 0.301 (unit)`

- **alarm threshold 0.281** — the model score above which Sentinel alarms. It is the
  quantile of the model's validation scores that matches NEWS2's *achieved* alarm volume
  at its own 2% budget (NEWS2 is an integer score, so "≥ 11" fires on 2.6% of validation
  hours; the model is held to the same 2.6%). Frozen on the V_freeze half of validation,
  never touched by test data.
- **this unit would run at 0.301 (unit)** — the per-ICU-type deployment threshold from the
  standby layer (`data/cache/standby_report.json`): the model's own 2% quantile inside this
  unit. `(unit)` means the unit had enough validation hours to freeze its own number;
  `(global-fallback)` means it inherits the global one.

### Per-unit standby threshold (toggle)
Draws that per-unit threshold as a second, lighter dashed line on the risk chart. It
indicates how the alarm hours would move if this unit ran at its own operating point
instead of the global one — the Cohort tab's "same model, seven thresholds" idea, seen on
one patient.

### Risk chart
**What it is.** Sentinel's fused risk at every hour of the stay — the two-head composite,
seed 42, read from `data/cache/two_head.npz`. The y-axis is a calibrated probability
(0–1) of deterioration (invasive ventilation, vasopressor start, or death) within the next
24 h, computed only from information strictly before that hour.

| mark | meaning |
|---|---|
| blue line | the fused risk, hour by hour |
| grey dashed rule "alarm threshold · 2% budget" | the frozen global threshold (0.281) |
| blue dots | hours at which Sentinel is alarming (risk ≥ threshold) |
| red vertical rule, labelled *ventilation* / *vasopressors* / *death* | the hour the event happened (`event_h`); the trajectory ends there |
| dotted black vertical rule | the hour currently selected on the slider |
| lighter dashed rule (toggle) | the per-unit standby threshold |

**What it indicates.** The *first* blue dot is Sentinel's first alarm. Its distance to the
red rule is the warning time before the event. The shape of the line is the trajectory the
causal temporal convolution reads (~29 h receptive field), so a rise that begins hours
before the threshold is crossed is itself informative. Gaps between dots (risk dipping
below threshold and returning) are real: the alarm is per hour, not latched.

### NEWS2 chart
**What it is.** The deployed bedside score, as charted, hour by hour (0–20, step line),
with its own alarm rule at ≥ 11 — its 2%-budget threshold on the same validation half.
Orange dots mark its alarm hours.

**What it indicates.** The comparator every timing claim is made against. The lead
counted in the paper is the first orange dot minus the first blue dot, on stays where
*both* systems alarm before the event (`src/fair_capability.py: paired_first_alarm_diffs`).
Same x-axis as the risk chart, so the two first alarms can be read off directly.

### Hour slider
Selects the hour the four panels below describe. It opens on Sentinel's first alarm hour
(or the first hour of a quiet stay). Everything below is a snapshot at that hour.

### Status line
`Hour 23 — Sentinel ALARM (risk 0.317) · NEWS2 quiet (score 9)` — what each system is
doing at the selected hour, with the number behind it. This is the sentence the demo
turns on: an alarm from one system and silence from the other at the same hour.

### Where the fusion looked (attention chart)
**What it is.** The attention weight the fusion put on each of the eight agents at the
selected hour (`Orchestrator.attention_weights`). The weights sum to 1. Bars are in fixed
roster order; the small grey label at each bar end is that agent's **own** calibrated
risk for this hour (each agent is an independent model that emits a risk before fusion).

| agent | what it reads |
|---|---|
| vitals-TCN | a causal CNN over the five NEWS2 vital signs |
| NEWS2 | the standard score and its seven sub-scores |
| labs | 24 laboratory analytes |
| ECG measures | machine-read ECG measurements |
| GBM risk model | a 145-feature gradient-boosted model, used as an agent |
| charting cadence | how often the chart is being written to |
| fluids & urine | urine output and fluid balance |
| resp. support | the oxygen-support ladder (device, flow, FiO2) |

**What it indicates.** Which modality drove the fused risk this hour — the model's
structure exposed before any language model is involved. Attention is re-derived every
hour from the patient's own data, so it shifts as modalities appear or go stale. A tall
bar on an agent with a *low* own risk means the fusion is reading that agent's latent
trajectory, not its headline number.

**Caveat (also printed under the card).** The plotted risk is the two-head composite; the
attention comes from a single-head ALL-8 fusion refit on the same agent bank at the same
seed (the composite's shared substrate), because the two-head net has no saved checkpoint.
This is the same caveat the T17 explanation benchmark carries.

### NEWS2 sub-scores (chips)
**What it is.** The seven components NEWS2 sums: respiratory rate, SpO2, temperature,
systolic BP, heart rate (each 0–3), supplemental oxygen (0 or 2), consciousness (0 or 3).
Non-zero ones are highlighted; the chips add up to the NEWS2 total on the status line.

**What it indicates.** Exactly what the bedside score can and cannot see at this hour. A
Sentinel alarm with mostly-zero chips is the interesting case: the risk is coming from
something NEWS2 does not measure (labs, cadence, respiratory support, trajectory).

### Vital signs the model saw (table)
| column | meaning |
|---|---|
| value | the last value charted **before** this hour — the only value the model is allowed to see (strictly-before-t rule; carried forward, dropped if older than the channel's staleness bound) |
| charted | how old that value is (`<vital>_hrs_since`); "2 h ago" means it was charted two hours before the selected hour and carried forward since |
| 4 h earlier | the value the model saw four hours before — the anchor for the trend |
| trend | rising / falling / stable, by a per-vital change threshold over those 4 h; "no 4 h history" in the first hours of a stay |

**What it indicates.** The raw evidence behind the vitals agent and the NEWS2 chips, with
its freshness. It is also the exact wording the explanation card is given ("heart rate
rising (90->105)"), so a trend the card names can be checked here.

### Why — the explanation card
**What it is.** One sentence from `gemma4:12b`, running locally under ollama, generated
from the numbers above — the NEWS2 total and sub-scores, each agent's risk, the attention
weights, the 4 h trends — with the plain prompt that the T17 faithfulness benchmark scored.
Cards for every hour of the four stays were generated at build time and cached.

**What it is never given.** The fused risk and whether the alarm fired. That omission is a
design rule, not a limitation: it means the only way the sentence can be informative is by
naming the drivers, which is what the benchmark measures (it names the top-attended agent
in 97% of cards, invents an unsupported driver in 0.13 per card, and a surrogate reading
only the card text recovers the alarm decision at 0.836 AUROC).

**Regenerate live (GPU).** Re-runs the same prompt on the laptop GPU (about a second) and
reports the time. With temperature 0 and a fixed seed the sentence normally comes back
identical to the cached one — the caption says so — which is the reproducibility point.
If ollama is not running, the cached card stays and a warning says so.

**What it indicates.** That the alarm ships with a justification tied to the model's own
mechanics — display only, outside the decision path. It cannot alter a score, a threshold,
or an alarm.

---

## Head-to-head tab

Sentinel against the two published-spec ML baselines — the eCART-style GBM and the YAIB
LightGBM, both retrained on this cohort from their published feature specifications
(`src/lit_gbm.py`) — with **no NEWS2 anywhere on the tab**. Every number is recomputed at
build time into `cohort.json["h2h"]` and is aggregate-only.

### The budget banner
Every model's threshold is frozen on the validation freeze half at the same alarm volume
(the demo's matched 2.62% of patient-hours), so no model spends more alerts than another.
The achieved test volumes are printed (2.8% / 2.6% / 2.4%) rather than hidden — thresholds
are frozen on validation, so realised test volume drifts slightly.

### How often is the anomaly caught (funnel + validity selector)
**What it is.** Share of the 1,540 event stays each model catches, under a selectable
definition of "catch": **any catch** (first alarm at or before the event: 65.5% / 53.6% /
51.8%), **≥ 6 h warning** (34.9% / 28.2% / 27.1%), **mid-stay only** (an any-catch whose
first alarm falls strictly after the stay's first scored hour — a change detection, not an
arrival-profile flag: 23.2% / 24.7% / 24.5%). 449 event stays are missed by all three.

**What it indicates.** The headline gap — Sentinel catches ~12 points more of the event
stays than either GBM at the same alarm volume — holds under the ≥ 6 h definition too. The
mid-stay view is the honest one to know cold: there the three models are equal (the GBMs a
hair ahead), i.e. Sentinel's extra catches come largely from flagging high-risk arrival
profiles earlier, not from better mid-stay change detection.

### If it is caught, who catches it first (race bars)
**What it is.** Among stays both models catch: Sentinel first / same hour / comparator
first — vs the eCART-style GBM 28% / 62% / 10% (762 joint stays), vs YAIB 26% / 62% / 13%
(743). Under each bar: the exclusive catches (only Sentinel : only the comparator —
**246 : 63** vs eCART-style, **265 : 55** vs YAIB) and the median gap when they differ
(+2.0 h, positive = Sentinel earlier).

**What it indicates.** Most joint catches are same-hour ties; when the models disagree
Sentinel is earlier roughly 3:1 (73% vs eCART over 292 differing pairs, 67% vs YAIB over
285; median gap ~5 h), and the exclusive-catch ratio is ~4:1 — the asymmetry is the
persuasive number, not the median. A caption under the bullets gives **warning depth** on
each model's own catches: median 7.3 / 6.9 / 6.7 h — the same depth for everyone, so the
edge is *how many* events get that warning, not how deep it is. The means (45–49 h, also
shown) are tail-driven by arrival-hour flags on stays that deteriorate days later — quote
the medians.

### Is it a valid catch — and what does it cost (metrics row)
**What it is.** Per model: the share of never-deteriorating stays that get at least one
alarm (11.8% / 6.8% / 6.2% of 6,835 quiet stays) and stay-level precision — of the stays
a model alarms on, how many truly deteriorate (56% / 64% / 65%).

**What it indicates.** The price of the extra catches, stated plainly: Sentinel spends
more of its budget on stays that never deteriorate. Same budget, different spending
strategy — more catches, lower per-alarm precision.

### Does it hold at other budgets? (sweep lines)
**What it is.** The same frozen-threshold head-to-head at eight alarm budgets from 0.5%
to 10% of patient-hours (the demo's matched operating point included as the dashed rule):
one catch-rate line per model, quiet-stay FP in the tooltip. Computed at build time into
`cohort.json["h2h"]["sweep"]` (`src/demo_bundle.py: budget_sweep_row`).

**What it indicates.** The answer to "why 2%?": it doesn't matter. The catch-rate gap
over the best baseline runs +7 to +13 points across the whole sweep and never inverts,
and AUPRC/AUROC are threshold-free anyway — the budget is an operational choice (the
matched volume is what NEWS2 ≥ 11, the deployed standard, spends in this cohort; for a
20-bed ICU, 2% ≈ 10 alarm-hours a day), not a tuned parameter.

### The caveat line
"Published feature specs retrained on our cohort, not the commercial products" — the
comparison is to the published eCART/YAIB recipes under our protocol, not to the deployed
commercial eCART score.

---

## Cohort tab

### The four headline numbers
| metric | what it is | what it indicates |
|---|---|---|
| **AUPRC · 5-seed mean** (0.308) | area under the precision–recall curve on 361,229 held-out patient-hours, averaged over five fusion training seeds; caption gives the equal-information GBM (0.287) and NEWS2 (0.114) | ranking quality at 4.7% prevalence — the claim that is robust: the paired margin over the GBM has its CI clear of zero on 5 of 5 seeds |
| **AUROC** (0.822) | area under the ROC curve; caption gives the death head's own AUROC on death (0.946) | the conventional discrimination number; the death head is the strongest single ranking in the system |
| **Event stays caught · same alarm volume** (65%) | share of the 1,540 event stays with a first alarm at or before the event, at the frozen matched volume; caption gives the best published-spec GBM (54%) and points to the Head-to-head tab | the capability claim in one number — same budget, ~12 points more of the deteriorations caught |
| **Alarm-burden spread** (5.3× → 1.7×) | the ratio of the highest to the lowest realised alarm rate across the seven ICU types, at one global threshold versus seven per-unit thresholds | one model serves every unit once the operating point is per unit |

### Ranking quality, six systems (AUPRC bars)
**What it is.** One bar per system, sorted by AUPRC (`data/cache/lit_gbm_report.json`):
Sentinel (blue), our 145-feature GBM, three gradient-boosted models rebuilt from published
specifications (Sadanandan 2026, eCART-style, YAIB LightGBM), and NEWS2 — all gray.
Tooltip shows AUROC.

**What it indicates.** The ranking ladder at a glance: Sentinel clears every comparator,
and NEWS2 sits at the bottom as context, not as a race — the timing story now lives on the
Head-to-head tab, against the ML baselines only.

### Same model, seven thresholds (burden bars)
**What it is.** Per ICU type, the realised alarm rate on the test split at one global
threshold (grey) versus that unit's own threshold (blue), with the 2% budget as a rule
(`data/cache/standby_report.json`). Units are ordered by test patient-hours.

**What it indicates.** At a global threshold some units are starved (neuro at 0.8%) and
some over-alarmed (med-surg at 4%); per-unit thresholds pull every unit toward the budget
without retraining anything. Choosing the budgets is governance, not modelling.

### Per-seed detail (expander)
One row per training seed: AUPRC, AUROC, death-head AUPRC on death, median lead with its
patient-clustered 95% CI, win rate, number of paired stays, and the CI of the AUPRC margin
over the GBM. It indicates which claims are stable across seeds (ranking, death head) and
which are not (lead: two seeds at 0.0 h).

### What we do not claim
Three caveats, stated on the page so they are never a surprise: the timing edge is
seed-sensitive; on death stays with no documented care-limitation decision the lead is null
(early death alarms largely track decisions already made); single centre, retrospective,
no external validation, no outcome evidence.

---

## Footprint tab

All measured on the demo laptop at build time (`scripts/build_demo_bundle.py: footprint`).

| metric | what it is | what it indicates |
|---|---|---|
| Fusion parameters · per head (26,018) | parameters of one `LatentFusionNet` (attention + causal TCN) | the fusion is tiny |
| Two-head total (52,036) | the treatment head plus the death head | the whole deployed fusion |
| Checkpoint on disk (121 KB) | the saved single-head net, calibrator included | ships in a config file's worth of space |
| Batch scorer peak memory (~5.8 GB) | peak RSS of the process that scored the whole test split — dominated by the 2.46M-row cached agent bank held in RAM, not the model | the honest batch number; a single stay needs only its own latents |
| CPU time per patient-hour · 1 thread (2.3 µs) | wall time to score all 361,229 test hours through the fusion, divided by hours, one CPU thread | hourly scoring for a hospital is trivial on one core |
| Patient-hours per second (~450k) | the same number inverted | throughput headroom |
| One stay, N hours (1.7 ms) | median of three warm scorings of one demo stay | bedside latency |
| Explanation card · local GPU (~1.3 s) | mean generation time per card over the build's generation loop, gemma4:12b under ollama | the one GPU cost, and it is display-side only |

**What runs where.** Agents emit cached latents; the fusion scores on CPU, one thread,
bit-identically; the alarm policy is seven numbers keyed by ICU type; the language model
runs locally on the GPU for the card text and is never told the risk or the alarm — the
decision path is LLM-free end to end.

---

## Where each number comes from

| component | source |
|---|---|
| risk line, T/D heads | `data/cache/two_head.npz` (`test_s42`, `dhead_test_s42`) |
| thresholds | `src/fair_capability.py: frozen_thresholds` on the V_freeze half; per-unit from `standby_report.json` |
| attention, single-head risk | `models/orchestrator/all8_s42.pt` via `Orchestrator.attention_weights` / `emit` |
| agent risks, NEWS2 sub-scores | the cached agent bank (`data/cache/embeddings/*_test.npz`) |
| vitals, staleness, trends | `data/cache/mimic_cohort_hourly.parquet` + `src/explain/drivers.py: build_contexts` |
| explanation cards | `src/explain/rationales.py` with `gemma4:12b`, cached in `data/cache/demo/` |
| ranking bars | `data/cache/lit_gbm_report.json` |
| comparator traces, alarms, thresholds | `data/cache/lit_gbm_scores.npz`, frozen on the V_freeze half at the matched volume (`scripts/build_demo_bundle.py`) |
| head-to-head block | `src/demo_bundle.py: first_alarm_table` + `h2h_summary` over all 8,375 test stays, written to `cohort.json["h2h"]` |
| burden | `data/cache/standby_report.json` |
| seeds, headline | `data/cache/metrics_table.json` + `two_head.npz` metadata |

## What is never shown
No patient-level data leaves the laptop: the Patient tab exists only in `bundle.json`
(gitignored) and on the presenter's screen. The leave-behind page is rendered from
`cohort.json` alone, and two guards (`assert_aggregate_only`, `assert_no_identifiers`)
refuse any id or id-like number. Discharge notes are never shown or used as model input
anywhere in the system.
