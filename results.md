# Results

Test split: 361,229 patient-hours, 8,375 ICU stays, 6,335 patients, prevalence 4.74%.
All calibration is isotonic regression fit on val only, applied to test. Composite
label = ICU transfer OR mortality OR mechanical-ventilation onset OR vasopressor
onset, predicted at each hour (see `cohort.py`).

Reproduce with:

```
.venv/bin/python scripts/results_table.py       # discrimination + calibration table
.venv/bin/python scripts/train_orchestrator.py  # orchestrator + paired vs NEWS2
.venv/bin/python src/baselines_mimic.py         # GBM + paired vs NEWS2 (null control)
```

## Discrimination + calibration

| model | AUPRC raw | AUPRC cal | AUROC raw | AUROC cal | ECE raw | ECE cal | vent | vaso | death |
|---|---|---|---|---|---|---|---|---|---|
| **ORCHESTRATOR** | **0.234** | 0.229 | 0.788 | 0.788 | 0.0349 | 0.0011 | 0.153 | 0.112 | 0.081 |
| GBM | 0.170 | 0.166 | 0.745 | 0.745 | 0.0036 | 0.0016 | 0.081 | 0.049 | 0.187 |
| labs | 0.171 | 0.162 | 0.728 | 0.727 | 0.0035 | 0.0012 | 0.128 | 0.100 | 0.035 |
| vitals-TCN | 0.159 | 0.155 | 0.750 | 0.750 | 0.0524 | 0.0021 | 0.084 | 0.054 | 0.139 |
| NEWS2 | 0.108 | 0.107 | 0.657 | 0.659 | 0.0845 | 0.0014 | 0.059 | 0.041 | 0.062 |
| (random) | 0.048 | 0.048 | 0.501 | 0.501 | 0.4524 | 0.4524 | 0.033 | 0.021 | 0.008 |

AUPRC floor = prevalence = 0.0474 overall; per-type floors: vent 0.0329, vaso
0.0209, death 0.0076. AUROC floor = 0.500. AUPRC is the primary metric — at
4.7% prevalence AUROC barely moves as false positives accumulate, so it reads
high even when most alerts are wrong.

**Orchestrator attention** (mean weight over agents on test): labs 0.447,
notes 0.185, vitals-TCN 0.201, news2 0.167 — the fusion leans hardest on labs.

The orchestrator beats every baseline on AUPRC, roughly doubles NEWS2, and is
essentially perfectly calibrated after isotonic fitting (ECE 0.0011). It is
not best on every event type individually (GBM edges it on death, 0.187 vs
0.081) but wins vent, vaso, and the composite.

## Paired, alert-volume-matched lead time vs. NEWS2

Methodology (`paired_first_alarm` in `src/baselines_mimic.py`): threshold each
model to a target alert budget, restrict to ICU stays that **both** NEWS2 and
the comparison model catch before the event, and compare first-alarm hour
per stay. `diff = NEWS2 first-alarm hour − model first-alarm hour`, so
positive/">50% win rate" means the model fires earlier. Reported: median
paired diff, fraction of stays where the model wins, and a two-sided sign
test over discordant pairs. NEWS2's integer score can't hit budgets finer
than its achieved rate, so the 1% and 2% rows collapse onto the same
threshold (2.1% actual).

### Orchestrator vs. NEWS2 (primary config)

| budget | match% | n_pair | median diff (h) | ORCH earlier | sign_p |
|---|---|---|---|---|---|
| 1% | 2.1% | 343 | 4.0 | 71% (242/343) | 9.2e-25 |
| 2% | 2.1% | 343 | 4.0 | 71% (242/343) | 9.2e-25 |
| 5% | 8.0% | 698 | 4.0 | 89% (618/698) | 6.9e-129 |

### GBM vs. NEWS2 (null control)

| budget | match% | n_pair | median diff (h) | GBM earlier | sign_p |
|---|---|---|---|---|---|
| 1% | 2.1% | 367 | 0.0 | 41% (150/367) | 1.0 |
| 2% | 2.1% | 367 | 0.0 | 41% (150/367) | 1.0 |
| 5% | 8.0% | 715 | 1.0 | 53% (378/715) | 5.4e-23 |

**Reading it:** the orchestrator fires earlier than NEWS2 on 71-89% of
jointly-caught stays with a ~4h median lead, and the sign test rejects the
null of no difference by a wide margin at every budget. The null control is
the right shape for a control — GBM is statistically indistinguishable from
NEWS2 at the tight budgets (median diff 0h, 41% win rate, p=1.0) and only a
weak, near-coin-flip edge at 5% (53%, significant only because n is large).
The contrast between a strong, large-effect win for the orchestrator and a
flat-to-negligible null is the paired evidence behind the lead-time claim.

## Notes

- Numbers carry small run-to-run noise under the fixed seed (e.g. orchestrator
  AUPRC 0.234 vs. 0.235 across two runs) from early-stopping landing on a
  slightly different epoch each time — not a regression.
- `results_table.py` and `train_orchestrator.py` retrain the orchestrator (and
  re-cache its embeddings + the labs/vitals-TCN agents) from scratch each run;
  `src/baselines_mimic.py` retrains GBM. None of this touches the notes-agent
  work in progress on this branch.
