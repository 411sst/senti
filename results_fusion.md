# Fusion & LatentCast — can Sentinel have both ranking and earliness?

Branch `experiment`. Companion to `results.md` (corrected GBM-vs-orchestrator
results) and `results_ecg.md` (ECG integration). Method provenance: "The
Method Bench" memo — Bundle 4 (fusion / gating / distillation) and Bundle 3
(LatentCast, forecast-then-alarm).

**The question.** The corrected results read as a tradeoff: the fair GBM
(FAIR+TRAJ, 145 tabular features) wins ranking — test AUPRC 0.287 vs the
orchestrator's 0.232 — but has *zero* earliness edge over NEWS2, while the
latent orchestrator alarms median +2h earlier at matched 1/2/5% alarm
budgets. Nobody has tested whether that tradeoff is real: stacking papers
report only ranking metrics, distillation papers never report lead time.

## Headline findings

1. **The tradeoff is real, and it is one shared frontier.** Score fusion
   (T1), objective/target distillation (T2), latent forecasting (T3), and
   feature-level integration in both directions (T4) all trace the same
   accuracy↔lead curve between the two parents. No method found a point
   that keeps the GBM's 0.287 *and* the orchestrator's full +2h.
2. **Earliness is representational** — a property of the architecture that
   produces the score, not of the label, the loss, or the input features.
   Four independent demonstrations: (a) label widening and loss shaping move
   the GBM's lead by exactly zero hours; (b) only importing the
   orchestrator's *targets* moves it, at proportional accuracy cost; (c) a
   mean forecaster of the latents smooths away the very spikes that cross
   thresholds early; (d) the same information at the same accuracy gives +1h
   (win 57%) when the orchestrator produces the score and 0h (win 41%) when
   the GBM does.
3. **GBM + our agents cannot be given lead** (the Target-2 answer): with
   clean out-of-fold fused-risk features the GBM stays accuracy-tied
   (0.280–0.284, ΔvsFAIR CI spans zero 3/3 seeds) and gains nothing — 0.0h
   at every budget.
4. **Both frontier endpoints improved.** LOGIT3 (3-parameter stack) is the
   project's new best ranking score: test AUPRC 0.291 > 0.287, paired CI
   clear of zero 3/3 seeds. The earliness-constrained stack Pareto-improves
   the orchestrator (0.262 at the full +1.7h@2%).
5. **The team-target deliverables** (GBM-level accuracy, lead ≥ +1h@2%):
   **CONSTR-1H** (0.283, GBM-tied 3/3, +1h 3/3, win 51–54%) and
   **ORCH+GBM** — the GBM as a fifth agent inside the orchestrator — 0.277
   (+0.030 over the parent ✓3/3) with the more robust lead (+1h at 2% *and*
   5%, win 55–57%, sign-p ≤ 1e-15). Lead-first recommendation: ORCH+GBM.
6. **Lead at fixed alarm volume is zero-sum** (T5): even a spike-preserving
   exceedance ensemble OR-ed with the parent reproduces the parent
   cell-for-cell — the parent's crossings already fill the budget, and
   prediction-only alarms get cut by the volume-matched threshold. The
   +2h@1–2% is this representation family's frontier ceiling; more lead
   requires more early signal in the representation, not a better trigger.
7. **Two methodology catches** now baked into the harness: isotonic
   calibration is only weakly monotone (a plateau on the threshold silently
   inflates alarm volume ~2×; fixed by rank-tiebreak strictification), and
   presence-guard constant columns explode z-scoring (std floor + clipping).

## Protocol (all experiments)

- **Parents.** Earliness parent = core-3+meas orchestrator (`vitals_tcn`,
  `news2`, `labs`, `ecg_meas` — the V4 config, results_ecg.md). Ranking
  parent = FAIR+TRAJ (FAIR+NOTES is not rebuildable on this clone — the note
  embedding caches are absent; FAIR+TRAJ is the better variant anyway).
  Fair cache regenerated on GPU: test AUPRC 0.287 [0.264, 0.311] — reproduces
  the published number; winner depth 4 / spw 1.0 / 375 trees.
- **V_fit / V_freeze.** Val split once, patient-level 50/50 (seed 42):
  V_fit (192,557 rows) fits everything with free label-reading parameters
  (per-parent isotonic calibrators, stack/gate weights, student early
  stopping); V_freeze (186,206 rows) freezes the NEWS2-anchored thresholds
  (fair_capability protocol) and hosts all model *selection*. Test is touched
  once per reported rung. Parents are re-scored under this same protocol, so
  their rows here differ slightly from results.md (which froze on full val).
- **Seeds.** Orchestrator seed noise exceeds typical fusion deltas
  (results_ecg.md), so every orchestrator-involving quantity runs at seeds
  {42, 1, 2} (core-3+meas val AUPRC 0.2368 / 0.2415 / 0.2279), with all
  fusion rules re-fit within seed; tables report mean + per-seed sign
  consistency. The GBM is seed-fixed.
- **Raw scores into fusion; fusion calibrates.** Blending needs a common
  probability scale, so each experiment fits its own isotonics on V_fit.
  **Isotonic plateaus are strictified** with a 1e-9 rank tiebreak in
  raw-score order: a plateau sitting on the val threshold otherwise fires
  wholesale past the matched alert volume (observed: 1.97% achieved at a
  1.03% match — the "+1h extra lead" that produced was an artifact, caught
  and killed before any table below was produced).
- Harness: `src/fair_capability.py` unchanged — volume matched to NEWS2's
  achieved val alert rate, jointly-caught stays, patient-clustered bootstrap
  (2000 reps timing / 1000 reps AUPRC).

## Pre-registered success / null criteria

Written before the full multi-seed tables were produced.

- **4a fusion ladder — success:** some rung with (i) test AUPRC within or
  above FAIR+TRAJ's paired CI, and (ii) median lead > 0 with win-rate CI
  clear of 50% at ≥ 2 budgets, both sign-consistent 3/3 seeds. **Null:**
  every rung gives back one of the two edges → "ranking and earliness are
  carried by incompatible score shapes" (publishable, sharpened by 4b).
- **4b gate:** if even ORACLE-GATE (true time-to-event decides the parent)
  fails to beat the best 4a rung on either axis, gating is dead.
- **Bundle 3 LatentCast — success:** some horizon k where median lead
  exceeds the parent orchestrator's with AUPRC not below the parent CI, and
  both controls fail to match it (shifted-label GBM gains lead only by
  shedding AUPRC; raw-vitals NEWS2 forecast inferior on both axes). **Null:**
  the shifted-label GBM matches LatentCast's frontier → forecasting adds
  nothing beyond label widening.
- **4c distillation — success:** a grid point strictly Pareto-dominating
  FAIR+TRAJ (lead > 0 at ≤ 0.01 AUPRC cost) → earliness is distillable into
  a tabular model. **Null:** the frontier is a hard tradeoff line through the
  two parents → earliness is representational (what the channel decomposition
  predicts).

## T1 — Fusion ladder + gate (4a/4b): the tradeoff is real, but both ends of the frontier moved

Test, mean over seeds {42,1,2}; ΔvsFAIR = paired patient-clustered AUPRC
delta vs FAIR+TRAJ; sign = seeds agreeing. Full per-seed rows with CIs in
`data/cache/fusion_ladder_scores.npz` meta and `logs/fusion_ladder_full.log`.

| rung | AUPRC | ΔvsFAIR | sign | lead@1% | lead@2% | lead@5% | win@2% |
|---|---|---|---|---|---|---|---|
| ORCH-meas (parent) | 0.247 | −0.0397 | 0/3 | +2.0 ✓3/3 | +1.7 ✓3/3 | +1.0 ✓3/3 | 58–63% |
| FAIR+TRAJ (parent) | 0.287 | — | — | 0.0 | 0.0 | 0.0 | 38% |
| MAX | 0.282 | −0.0047 | 0/3 | 0.0 | +0.7 (2/3) | +0.2 (1/3) | 49–54% |
| **LOGIT3** | **0.291** | **+0.0037** | **✓3/3** | 0.0 | 0.0 | 0.0 | 44–45% |
| CONSTRAINED | 0.262 | −0.0249 | 0/3 | +0.7 (2/3) | +1.7 ✓3/3 | +0.7 (2/3) | 51–63% |
| CONSTR-1H | 0.283 | −0.0038 | 0/3* | 0.0 | +1.0 ✓3/3 | +0.3 (1/3) | 51–54% |
| GATED | 0.291 | +0.0034 | ✓3/3 | 0.0 | 0.0 | 0.0 | 45% |
| ORACLE-GATE † | 0.233 | −0.0540 | 0/3 | 0.0 | +0.3 | 0.0 | 47–51% |

† label-using headroom bound, not a model.

**Verdict on the pre-registered criterion: null — no rung holds FAIR-level
AUPRC and a clear earliness win simultaneously.** Ranking and earliness are
carried by score shapes that a monotone-per-hour combination cannot merge:
every rung slides along a frontier between the two parents. But the ladder
produced two genuine improvements at the frontier's *endpoints*:

1. **LOGIT3 is the project's new best-ranking model.** A 3-parameter logistic
   stack of the two calibrated parents reaches test AUPRC 0.291 — above
   FAIR+TRAJ (0.287) with the paired delta CI clear of zero on all three
   seeds ([+0.0008,+0.0074] / [+0.0004,+0.0072] / [+0.0002,+0.0062]). The
   orchestrator's latent signal adds ranking value the 145-feature GBM does
   not carry, even though its timing dies in the stack.
2. **The earliness-constrained stack Pareto-improves the orchestrator.**
   Under a hard "give up no median lead on V_freeze" constraint, seeds 1/2
   accepted GBM mass (w=0.6/0.9) and gained +0.031/+0.014 test AUPRC at
   unchanged lead; seed 42 refused to trade (w=1.0). Mean: 0.262 vs the
   parent's 0.247 at the same +1.7h@2% ✓3/3. If the team ships an
   earliness-first model, this stack is strictly better than the raw
   orchestrator.
3. **MAX is the honest midpoint** — statistically ties FAIR+TRAJ on ranking
   (CI spans zero, 2 of 3 seeds) while keeping about half the lead at 2%.
4. **CONSTR-1H is the team-target rung** ("GBM-level AUPRC at lead ≥ +1h"):
   the same stack with the selection floor relaxed to an absolute +1h picks
   w=0.5–0.6 and lands at AUPRC 0.283 with the ΔvsFAIR CI *spanning zero on
   all three seeds* (−0.0038 mean — statistically GBM-tied) and median lead
   +1.0h@2% on 3/3 seeds. Honest caveats: the win rates (51–54%) are not
   clear of coin flip and the +1h holds only at the 2% budget — this is a
   real frontier point, softer evidence than the parent's +2h/61%.

**4b verdict: gating is dead, by its own oracle.** ORACLE-GATE — which
*knows* the true time-to-event and hands the orchestrator control inside the
24h window — is worse than both parents on AUPRC (0.233) and shows no
earliness. Hard-switching between two score scales mid-stay produces
incomparable first-crossings: an alarm surface has to be one consistent
score. The learned GATED collapsed to the ranking corner (mean gate weight
~0.14 → mostly GBM; numerically ≡ LOGIT3). Per the pre-registered gate
criterion, no further gating work is warranted.

## T3 — LatentCast (Bundle 3): forecasting rides the frontier, never past it

Forecaster: 1-layer GRU (107→128, ~160k params), direct heads k∈{2,4,6,8,12},
masked SmoothL1 in standardized space, trained on train stays only, early
stop on val forecast loss (label-free); best val loss 0.1139. Engineering
note: presence-guard constant columns (std ~4e-6) must have their z-scores
floored/clipped or training NaNs — std floor 1e-2, z-clip ±20, grad-clip 5.
Substitution semantics: whole-sequence forecast substitution under the frozen
causal head; pinned by the oracle unit test (perfect forecaster ⇒
risk_hat_k(t) == parent risk(t+k) wherever both sequences give the TCN a full
receptive field). k=0 pass-through drift 0.0 on all three seeds.

Test, mean over frozen-head seeds {42,1,2} (per-seed leads in brackets):

| model | AUPRC | lead@1% | lead@2% | win@2% | n_pair@2% |
|---|---|---|---|---|---|
| parent ORCH-meas | 0.250 | +2.0 ✓3/3 | +2.0 ✓3/3 | 61% | 434 |
| LatentCast k=2 | 0.2504 | +1.0 [1,1,1] | +1.0 [1,1,1] | 55% | ~396 |
| LatentCast k=4 | 0.2369 | [1,0,1] | [1,0,1] | 50% | ~375 |
| LatentCast k=6 | 0.2138 | [0,0,1] | 0.0 | 46% | ~351 |
| LatentCast k=8 | 0.1985 | 0.0 | 0.0 | 42% | ~332 |
| LatentCast k=12 | 0.1873 | [−.5,−1,0] | 0.0 | 35% | ~308 |
| control: shifted-GBM k=2..12 | 0.285→0.276 | 0.0, win 34–36% | 0.0, win 38–40% | — | — |
| control: raw-NEWS2-forecast k=2..12 | 0.118→0.109 | 0 to −3 | ≤0, win 12–23% | — | — |

**Verdict: the pre-registered null, with a mechanism.** Alarming on the
predicted-future latent does NOT extend the +2h — it trades along the same
frontier the fusion ladder traces: k=2 holds parent-level AUPRC (0.2504 vs
0.250) at *half* the lead, and both axes decay monotonically with k. The
mechanistic read: the SmoothL1-optimal forecast is a conditional mean, and
regression-to-the-mean smooths exactly the sharp early risk rises that buy
early threshold crossings at matched volume. The controls sharpen the claim
in both directions: label widening gives the GBM ranking-with-zero-lead at
every horizon (the GBM's zero earliness is NOT a label-horizon artifact),
and forecasting raw vitals gives neither (NEWS2-level AUPRC, negative leads)
— so the latent channel is still where the timing information lives; a mean
forecaster is just the wrong way to cash it.

The natural follow-up — the exceedance-probability ensemble, which preserves
spike mass instead of averaging it away — was subsequently run: see T5.

## T2 — Distillation grid (4c): earliness is representational; distillation walks the frontier

2 families × 4 objectives (identical definitions), students early-stopped on
V_fit, selection on V_freeze, Pareto-frontier points only on test. Full grid
in `logs/distill_grid.log` / `data/cache/distill_grid.npz`.

| family | objective | best cell | freeze AUPRC | freeze lead@2% | test (frontier only) |
|---|---|---|---|---|---|
| gbm | bce (parent) | — | 0.2545 | 0.0 | 0.287 / 0h |
| gbm | tls | H=6 | 0.2573 | **0.0** | 0.2845 / 0h |
| gbm | utility | β=0.5 | 0.2555 | **0.0** | — |
| gbm | distill | λ=0.75 (s42) | 0.2443 | **+1.0** | **0.2755 / +1.0h (win 50%)** |
| gbm | distill | λ=1.0 | ~0.230 | +1.0 ✓3/3 | — (orchestrator-level AUPRC) |
| orch | bce (parent) | s42 | 0.2312 | +2.0 | 0.2502 / +2.0h (win 61%) |
| orch | tls / utility | all | 0.223–0.239 | +1 to +2 | — (no gain on either axis) |
| orch | distill (GBM teacher) | λ=0.5 (s42) | 0.2418 | +1.0 | 0.2593 / +1.0h (win 51%) |
| orch | distill (GBM teacher) | λ=0.75 (s1) | 0.2483 | **0.0** | 0.2639 / 0h |

**Verdict: the pre-registered null — no grid point Pareto-dominates
FAIR+TRAJ — plus the sharp version of the mechanism.** Pure objective shaping
(temporal label smoothing, utility re-weighting) does not move lead by even
one hour in EITHER family: the GBM stays at zero at every H and β, the
orchestrator keeps its lead but gains no accuracy. The ONLY lever that moves
either model along the accuracy↔lead axis is importing the other model's
out-of-fold targets, and the λ sweep then traces the same frontier every
other method found: GBM λ 0→1 walks (0.287, 0h) → (≈0.23, +1h); the
orchestrator walks it in reverse (accuracy up, lead collapsing to 0 by
λ=0.75). Earliness is a property of the representation, not the objective —
exactly what the channel decomposition predicted, now demonstrated with the
strongest available counterfactual.

Usable frontier point: **GBM-distill λ=0.75 = a single deployable XGBoost at
test 0.2755 (−0.012 vs FAIR+TRAJ) with +1h median lead** — though its win
rate (50%) is not clear of coin-flip; see T4 for whether feature-level agent
integration beats it.

## T4 — The bridge experiments: who eats whom decides the timing

Team targets (set after T1–T3): **(1)** our architecture at GBM-level AUPRC
with lead ≥ +1h; **(2)** the GBM with our agents integrated, keeping its
accuracy but gaining lead.

**Target 1 — GBM as a fifth agent inside the orchestrator**
(`scripts/orch_gbm_agent.py`). Pseudo-agent cache `gbm_{split}.npz`
(latent = [logit(raw)], risk = isotonic-on-val), train rows OUT-OF-FOLD so
the fusion never learns to trust in-sample-sharp GBM scores. Config
core-3+meas+gbm, seeds {42,1,2}:

| seed | AUPRC | Δ vs parent (CI) | Δ vs FAIR (CI) | lead@2% | win@2% (CI) | lead@5% | win@5% |
|---|---|---|---|---|---|---|---|
| 42 | 0.2766 | [+0.0199, +0.0327] | [−0.0198, −0.0013] | +1.0 | 57% [52, 62] | +1.0 | 56% |
| 1 | 0.2801 | [+0.0224, +0.0357] | [−0.0157, +0.0021] | +1.0 | 55% [50, 59] | +1.0 | 52% |
| 2 | 0.2738 | [+0.0257, +0.0397] | [−0.0220, −0.0049] | +1.0 | 57% [53, 62] | +1.0 | 57% |

The biggest single accuracy jump for the architecture in the program:
**+0.030 AUPRC over the core-3+meas parent, CI clear of zero 3/3 seeds**,
while keeping **+1h median lead at BOTH the 2% and 5% budgets** (sign-test
p ≤ 1e-15 @2%; win CIs clear of 50% on 2/3 seeds). Attention allocates
34–46% to the GBM agent — the fusion genuinely arbitrates. Against Target 1's
letter: AUPRC 0.277 is *slightly* below GBM-tied (Δ vs FAIR below zero on
2/3 seeds); CONSTR-1H (T1) is GBM-tied 3/3 but with weaker, 2%-only lead.
The two are complementary frontier endpoints of the target region.

**Target 2 — the agents as features inside the GBM** (`src/gbm_agents.py`).
FAIR+TRAJ + per-agent risks, risk-trajectory features (Δ/min/max/mean over
6/12/24h), latents, and the OOF orchestrator risk + trajectory:

| variant | features | test AUPRC | lead@2% | win@2% |
|---|---|---|---|---|
| FAIR+TRAJ (parent) | 145 | 0.287 | 0.0 | 38% |
| +RISKS | 149 | 0.2742 | 0.0 | 42% |
| +RISKTRAJ | 197 | 0.2757 | 0.0 | 39% |
| +LATENT | 300 | 0.2778 | 0.0 | 41% |
| +ORCH (per seed, OOF, + agent cols) | 210 | 0.2738–0.2797 | 0.0 3/3 | 38–44% |
| +ORCHONLY (per seed, OOF, clean) | 158 | 0.2797–0.2840 | 0.0 3/3 | 41–43% |

**Feature-level integration cannot buy the GBM lead.** Two-step diagnosis:
the per-agent variants cost accuracy because the agents' train-split
emissions are in-sample (agents trained on train) — distribution-shifted
features the trees latch onto and lose at test. The clean +ORCHONLY variant
(FAIR+TRAJ + OUT-OF-FOLD orchestrator risk and its 6/12/24h trajectory, no
contaminated columns) recovers the accuracy — 0.280–0.284, ΔvsFAIR CI
spanning zero on all three seeds — **and still moves lead by exactly
nothing** (0.0 at every budget, win 38–45%). A tree ensemble handed the
timing signal as features reproduces its BCE-optimal per-hour score shape;
the information enters the level, never the temporal shape.

**The symmetry that summarises the whole program:** ORCH+GBM (our
architecture consuming the GBM) and GBM+agents (the GBM consuming our
agents) land at the *same accuracy* (~0.277) with *opposite timing* (+1h,
win 57% vs 0h, win 40%). Same information on both sides — whichever
representation PRODUCES the score determines the earliness. Earliness is
architectural, full stop: shown by objectives (T2), by forecasting (T3), by
features and by fusion direction (T4).

**Deliverable ranking against the team targets** (all at the 2% budget):

| model | AUPRC | vs FAIR | lead | win | deploy |
|---|---|---|---|---|---|
| CONSTR-1H stack | 0.283 | ties 3/3 | +1h 3/3 | 51–54% | 2 models + blend |
| **ORCH+GBM (5th agent)** | 0.277 | −0.010 | +1h 3/3 (also @5%) | 55–57% | one orchestrator |
| GBM-distill λ=0.75 | 0.2755 | −0.012 | +1h | 50% | single XGBoost |
| ORCH-meas (parent) | 0.247 | −0.040 | +2h 3/3 | 58–63% | one orchestrator |

The honest statement of what was achieved: the +2h/zero-accuracy-cost unicorn
does not exist in this design space — but the frontier now has three usable
new points, and the architecture story ("earliness is representational") is
demonstrated four independent ways.

## T5 — Exceedance ensemble: at matched volume, lead is zero-sum

The strongest remaining attack on extending the +2h: a 5-seed forecaster
ensemble (val losses 0.1132–0.1139) voting on *"will the risk cross the
parent's own volume-matched alarm threshold within the next K hours"*
(max over horizons ≤ K — spike-preserving where the T3 mean forecast was
spike-destroying), scored two ways at every budget and frozen-head seed:

- **EXCEED-K** (predicted-to-cross alone): +1h median at 1–2% (win 51–57%),
  AUPRC(ensemble-mean) ≈ 0.248–0.255 — the *same frontier point* plain
  LatentCast k=2 found. K-invariant from 2 to 8 h: short-horizon forecasts
  dominate the max (longer horizons regress harder to the mean), so the
  extra windows add nothing.
- **PARENT∨EXCEED** (alarm now OR predicted-to-alarm; provably never later
  than the parent): **cell-for-cell identical to the parent** on every seed
  and every budget — same n_pair (±0), same lead, same win rate, same
  achieved volume. E.g. seed 42 @2%: +2.0h/61%/n434 both; seed 2 @1%:
  +3.0h/64%/n244 both.

**Why, mechanistically — the finding that closes the lead-extension
question:** at a matched alert volume the parent's own crossings already
fill the entire budget, so the volume-matched threshold on the OR score
lands inside the parent-crossing level and cuts every prediction-only
alarm. Adding earlier alarms at fixed volume requires *dropping* existing
alarms (which is what EXCEED-alone does, and it costs win rate and AUPRC).
**Lead at fixed alarm volume is zero-sum: an alarm rule can only re-order
the budget the representation earns.** The +2h@1–2% is the ceiling of this
architecture family's frontier; exceeding it requires a representation with
more early signal (better encoders, new modalities — e.g. the ECG-CPC /
S4-waveform directions in results_ecg.md), not a cleverer trigger.

## Reproduce

```bash
# Phase 0 — substrate
.venv/bin/python src/baselines_fair.py --variants FAIR,FAIR+TRAJ --device cuda --reps 1000
.venv/bin/python scripts/train_orch_meas_seeds.py --seeds 42,1,2 --oof

# Bundle 4a/4b
.venv/bin/python src/fusion_ladder.py --stage all --reps 2000 --auprc-reps 1000

# Bundle 3
.venv/bin/python src/latentcast.py train --epochs 20 --device cuda
.venv/bin/python src/latentcast.py eval --reps 2000
.venv/bin/python src/latentcast.py control-shifted-gbm --device cuda
.venv/bin/python src/latentcast.py control-raw-news2 --device cuda

# Bundle 4c
.venv/bin/python src/distill_lead.py --seeds 42,1,2 --device cuda

# T4 bridge experiments
.venv/bin/python scripts/orch_gbm_agent.py --seeds 42,1,2 --device cuda
.venv/bin/python src/gbm_agents.py --variants risks,risktraj,latent,orch --seeds 42,1,2 --device cuda
.venv/bin/python src/gbm_agents.py --variants orchonly --seeds 42,1,2 --device cuda

# T5 exceedance ensemble
for s in 1 2 3 4; do .venv/bin/python src/latentcast.py train --seed $s --device cuda; done
for s in 42 1 2; do .venv/bin/python src/latentcast.py exceed --seeds $s \
  --out data/cache/latentcast_exceed_s$s.npz --device cuda; done

# tests
.venv/bin/python -m pytest -q   # 524 passing
```
