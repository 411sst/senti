# Results

**Cohort.** MIMIC-IV **in-ICU** hourly deterioration: composite label = invasive-
ventilation onset OR vasoactive-agent onset OR in-hospital death, within the
next 24h, for ICU stays >= 12h that were event-free through hour 6 (there is no
ICU-transfer component — the population is already in the ICU; see
`scripts/mimic_cohort.sql`). Hours 0–5 enter the extract as LOCF **history**
rows so the first prediction hour (h6) carries six hours of real context — the
former "warm-up rule" mask is retired at the source (see "h6 root-cause fix"
below). Test split: 361,229 patient-hours, 8,375 ICU stays, 6,335 patients,
prevalence 4.74%.

**Protocol.** Patient-level train/val/test splits (SEED=42). Every selection
decision uses **val only**: hyperparameters and early stopping, isotonic
calibration, alert-budget thresholds (frozen on val, applied once to test), and
the shipped agent configuration (`scripts/modality_ablation.py --split val`).
All headline numbers carry **patient-clustered bootstrap 95% CIs** (hours
within a stay and stays within a patient are not independent). Shipped
configuration: **core-3** (vitals-TCN + NEWS2 + labs).

Reproduce with:

```
.venv/bin/python scripts/fetch_mimic_cohort.py --force   # BigQuery -> raw extract
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import mimic_cohort; mimic_cohort.build()"
.venv/bin/python scripts/results_table.py                # agents + orchestrator (canonical caches)
.venv/bin/python src/baselines_fair.py --reps 1000       # fair GBMs + clustered CIs (--from-cache to refresh)
.venv/bin/python scripts/modality_ablation.py --split val# config selection (val, never test)
.venv/bin/python src/fair_capability.py                  # paired lead time + calibration
.venv/bin/python src/news2_band.py                       # NEWS2 fairness band
```

## Discrimination (test; patient-clustered 95% CIs; 1,000 resamples)

| model | AUPRC | 95% CI | AUROC | 95% CI | vent | vaso | death |
|---|---|---|---|---|---|---|---|
| FAIR+TRAJ (GBM, 145 feat) | **0.287** | [0.264, 0.310] | 0.805 | [0.791, 0.820] | 0.163 | 0.115 | 0.170 |
| FAIR+NOTES (GBM, 211 feat) | 0.285 | [0.264, 0.309] | 0.808 | [0.794, 0.822] | 0.158 | 0.113 | 0.181 |
| FAIR (GBM, vitals+labs, 61 feat) | 0.277 | [0.258, 0.300] | 0.799 | [0.785, 0.812] | 0.160 | 0.120 | 0.154 |
| **ORCHESTRATOR (core-3)** | 0.232 | [0.215, 0.251] | 0.786 | [0.771, 0.801] | 0.157 | 0.120 | 0.073 |
| GBM13 (vitals-only, untuned) | 0.177 | [0.160, 0.195] | 0.748 | [0.734, 0.762] | 0.087 | 0.052 | 0.189 |
| NEWS2 (as-charted) | 0.114 | [0.102, 0.128] | 0.683 | [0.668, 0.700] | 0.063 | 0.046 | 0.060 |

Single agents (point estimates, raw): vitals-TCN 0.192 / 0.755 · labs 0.171 /
0.728 · random = prevalence 0.048. Orchestrator raw AUPRC is 0.237 (isotonic is
monotone; the CI row above uses the canonical calibrated scores).

**Read this table honestly: a single equal-information GBM beats the
orchestrator on discrimination.** Paired deltas (same resamples): ORCH −
FAIR+TRAJ = **−0.055 [−0.068, −0.040]**; ORCH − FAIR = −0.045 [−0.058, −0.033];
ORCH − GBM13 = +0.055 [+0.040, +0.069]. The orchestrator's previously published
AUPRC win over the vitals-only GBM was modality access, not architecture: give
one XGBoost the same inputs (labs + vitals + staleness; deltas/window
aggregates; PCA-64 of the ClinicalBERT note embedding) with val-only tuning and
it ranks risk better. The fair GBMs also erase the orchestrator's death
deficit (0.154–0.181 vs 0.073). The orchestrator's value is **not**
discrimination — it is timing (next section).

## Timing: paired, volume-matched first alarm vs NEWS2

Protocol (`src/fair_capability.py`): NEWS2 threshold at each budget's val
quantile; the comparison model matched to NEWS2's **achieved val volume**;
thresholds frozen, applied once to test; jointly-caught stays; diff = NEWS2
first-alarm hour − model first-alarm hour (positive = model earlier); sign test
plus 2,000-rep patient-clustered CIs on the median and win rate.

| budget | model | n_pair | median (h) | 95% CI | win | 95% CI | sign p |
|---|---|---|---|---|---|---|---|
| 1% | **ORCH** | 243 | **+2.0** | [1.0, 3.0] | 59% | [52%, 65%] | 8.9e-13 |
| 1% | FAIR+NOTES | 223 | 0.0 | [0.0, 0.0] | 40% | [33%, 46%] | 0.66 |
| 2% | **ORCH** | 436 | **+2.0** | [1.0, 3.0] | 66% | [62%, 70%] | 5.9e-44 |
| 2% | FAIR+NOTES | 379 | 0.0 | [0.0, 0.0] | 38% | [33%, 43%] | 0.52 |
| 5% | **ORCH** | 606 | **+2.0** | [1.0, 2.0] | 62% | [58%, 66%] | 4.6e-61 |
| 5% | FAIR+NOTES | 549 | 0.0 | [0.0, 0.0] | 38% | [34%, 42%] | 0.34 |

**Discrimination is not earliness.** The GBM that wins AUPRC cannot beat NEWS2
on first-alarm timing at any budget (median 0h, win rate below coin-flip,
sign test null). The orchestrator — whose causal TCN reasons over the
trajectory of agent latents — fires a median 2h earlier than NEWS2 on 59–66%
of jointly-caught stays at every budget. This contrast, against a *strong*
null control, is the paper's central capability result.

### NEWS2 fairness band (`src/news2_band.py`)

The as-charted NEWS2 is conservatively degraded (missing vitals score 0,
unknown consciousness = alert, uncharted O2 = room air). The claim must hold
against NEWS2's strongest form, so the race is re-run against **best-LOCF**
NEWS2: unbounded within-stay carry-forward of every input — once observed, a
value stays current until re-measured.

| variant | budget | n_pair | median (h) | 95% CI | ORCH win | sign p |
|---|---|---|---|---|---|---|
| best-LOCF | 1% | 275 | **+3.0** | [1.0, 4.0] | 62% [56%, 67%] | 3.5e-17 |
| best-LOCF | 2% | 454 | **+2.0** | [2.0, 3.0] | 66% [62%, 70%] | 1.6e-43 |
| best-LOCF | 5% | 652 | **+2.0** | [2.0, 3.0] | 66% [62%, 69%] | 2.1e-79 |

The advantage **holds and slightly strengthens** against best-case NEWS2
(more pairs, same-or-larger medians, all CIs excluding zero).

**Clinical operating points.** The RCP escalation triggers are inoperable in
an ICU population: aggregate ≥5 fires on 45–52% of all test hours, ≥7 on
23–27%, single-parameter-3 on 46%, with 61–89% of non-event stays alarming at
least once. At those volumes NEWS2's near-continuous alarming trivially wins
any first-alarm race (orchestrator matched to those volumes wins only 27–47%
of pairs); this is alarm fatigue, not early warning, and it is why the
alert-budget regime (1–5%) is the evaluation that matters clinically.

## Calibration (test; isotonic-on-val; clinical-standard metrics; 500-rep clustered CIs)

| model | ECE (equal-mass, 15 bins) | slope (target 1) | intercept (target 0) | ICI |
|---|---|---|---|---|
| ORCH (core-3) | 0.0035 [0.0024, 0.0057] | 0.988 [0.934, 1.045] | −0.040 [−0.177, 0.118] | 0.0068 [0.0062, 0.0088] |
| FAIR+NOTES | 0.0023 [0.0018, 0.0048] | 0.989 [0.941, 1.049] | −0.033 [−0.166, 0.137] | 0.0074 [0.0067, 0.0095] |

Both models are essentially perfectly calibrated after isotonic-on-val; no CI
separates them. Calibration is a property of the shared protocol, not a
differentiator. (The old equal-width 10-bin ECE was retired: at 4.7%
prevalence ~88% of scores fall in one bin, and `metrics.ece` previously
accepted out-of-range scores silently — it now raises; the historical NEWS2
"ECE raw 0.0845" was an artifact of that bug.)

## Configuration selection (val only — `modality_ablation.py --split val`)

| config | val AUPRC | Δ vs core-3 (patient-clustered 95% CI) | P(>0) |
|---|---|---|---|
| **core-3** (shipped) | **0.230** | reference | — |
| core-3 + notes | 0.226 | −0.0038 [−0.0081, +0.0007] | 6% |
| core-3 + ECG | 0.219 | −0.0101 [−0.0181, −0.0029] | 0% |
| all-5 | 0.218 | −0.0111 [−0.0188, −0.0035] | 0% |

Switching the shipped config to core-3 cost nothing on test (0.232 cal vs the
4-agent 0.231) and every config beats NEWS2 on timing (+2h, 60–62% on val), so
the timing capability is not notes-dependent.

**Notes and ECG are retained modalities, not removed.** (On the `paper` branch
the notes agent and the waveform ECG agent's training code were removed as
unused by the flagship. Their rows here, and FAIR+NOTES, reproduce from branch
`experiment-demo`.) Their agents, caches,
and named configs (`core-3+notes`, `all-5`) stay maintained and evaluated; each
rejoins the primary the moment it clears the bar — **a positive fusion
contribution over the current primary on val with the 95% CI clear of zero**.
Future-work directions for getting them there: a header-stripped,
contamination-excluded note encoding (the INDICATION-header suspicion channel
is a known confound), and an ECG encoder that survives its coverage bias
(standalone AUROC 0.607 on covered hours; attention already downweights it to
~0.04 but downweighting only mitigates the harm).

## h6 root-cause fix (was: "warm-up rule")

The SQL grid previously started at hour 6, discarding hours 0–5 observations,
so every stay's first row was all-NaN by construction — NEWS2 pinned to 0
while learned models emitted their base-rate prior, a free first-alarm head
start that had to be masked out of alerting evaluation. The fix emits hours
0–5 as history rows (mirroring the labs 24h pre-admission lookback);
`mimic_cohort.build()` drops them after the strictly-before-t shift.

Verified on rebuild: **identical keys, labels, and splits** (2,460,983
prediction rows); h6 heart-rate coverage 0% → 97.7%; h6 NEWS2 no longer pinned
(mean 4.53, 8.7% zeros). The mask (`alertable_mask`) is retired and the 169
acute event stays it excluded are back in the lead-time analysis. Impact —
every model improved (restored real data, not leakage): NEWS2 AUROC 0.657 →
0.683, vitals-TCN AUPRC 0.159 → 0.192, GBM13 0.170 → 0.177, orchestrator
0.234 → 0.237 raw. The orchestrator's 5% timing row shrank honestly
(3h/74% → 2h/62%) and the fair GBM's pre-fix 5% timing edge (+2h/65%)
vanished — it had been riding the weak-NEWS2 artifact.

## Notes

- Pre-fix artifacts are preserved under `data/cache/pre_h6fix/` for the delta
  record.
- `data/cache/fair_gbm_scores.npz` carries the fair-GBM test/val scores, chosen
  hyperparameters, and bootstrap results; `baselines_fair.py --from-cache`
  refreshes the orchestrator comparison without retraining.
- Small run-to-run noise under the fixed seed remains (early stopping can land
  on an adjacent epoch); determinism pinning across all train scripts is
  tracked as open work from the methodology audit, alongside cache
  fingerprinting.
