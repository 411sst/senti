# Fine-tuning the flagship on eICU-CRD

T22 scored the MIMIC-trained flagship on 208 unseen eICU hospitals with nothing
fitted there. This adds the follow-up experiment: let a new site train on its
own labelled data, and measure what fine-tuning adds over the frozen model and
what it costs on MIMIC.

## What it does

eICU is split into train / val / test **by hospital** (60/20/20 by stay count,
seed 42). Test hospitals are never seen during fine-tuning, so every eICU test
number is still an unseen-hospital number and stays comparable to T22. If a
patient (`uniquepid`) appears in more than one hospital, their rows are kept
only in the earliest split, and the drop is counted in the report.

Five arms are scored on the same eICU test rows:

| arm | agents | heads | calibrators |
|---|---|---|---|
| `zero_shot` | MIMIC | MIMIC | MIMIC (the T22 setup) |
| `recal` | MIMIC | MIMIC | head isotonics refit on eICU val |
| `heads_ft` | MIMIC (frozen) | warm-started, fine-tuned on eICU | eICU val |
| `full_ft` | warm-started, fine-tuned on eICU | warm-started, fine-tuned on eICU | eICU val |
| `scratch` (opt-in) | eICU only | eICU only | eICU val |

How each component adapts:

- **Torch agents** (vitals TCN, labs, fluid, resp) and the **two heads**: start
  from the MIMIC weights at a 10x lower learning rate. The normaliser fitted on
  MIMIC train is kept, because the weights were learned against that
  z-scoring. Early stopping on eICU val is seeded with the pretrained weights'
  own score, so a component that never improves comes back unchanged.
- **XGBoost agent**: continues boosting from the MIMIC booster (`eta` 0.02, up
  to 500 extra rounds) and is kept only if it beats the MIMIC booster on eICU
  val. Its eICU-train emissions are out-of-fold, as on MIMIC.
- **NEWS2**: deterministic; only its isotonic calibrator is refit.
- **ecg_meas**: frozen. eICU has no ECGs, so every eICU row takes the agent's
  learned no-ECG path.

The fine-tuned stack is also scored on **MIMIC test** as a forgetting check.

## Settings

Settings live in [`configs/eicu_finetune.json`](../configs/eicu_finetune.json).
Any key can be overridden on the command line (`--agent-ft-lr 3e-4`), and
unknown keys are rejected.

| setting | default | meaning |
|---|---|---|
| `agents` | flagship 7 | ALL-8 minus cadence, as in T22 |
| `split_by` | `hospital` | `patient` gives a within-hospital split instead |
| `split_fracs` | `[0.6, 0.2, 0.2]` | train / val / test share of eICU stays |
| `train_patients` | `0` (all) | subsample eICU train, for learning curves |
| `seed` | `42` | MIMIC source stack (fitted once, shared by every seed) |
| `seeds` | `[42, 1, 2, 3, 4]` | fine-tuning seeds; each re-rolls everything trained on eICU |
| `agent_ft_lr`, `agent_ft_epochs` | `1e-4`, `6` | torch agent fine-tuning |
| `head_ft_lr`, `head_ft_epochs` | `1e-4`, `8` | head fine-tuning |
| `gbm_ft_eta`, `gbm_ft_rounds` | `0.02`, `500` | continued boosting |
| `gbm_device` | `cpu` | `cuda` is faster but not bit-reproducible |
| `freeze_agents` | `["ecg_meas"]` | agents kept at their MIMIC weights |
| `scratch` | `false` | also train an eICU-only stack |
| `bootstrap_reps` | `300` | patient-clustered bootstrap for the deltas |

## Steps

1. **Data access.** On PhysioNet, link your Google account (Profile → Cloud)
   and request BigQuery access for MIMIC-IV v3.1, eICU-CRD v2.0 and
   MIMIC-IV-ECG v1.0 (each is a separate request). `ecg_meas` is the only
   agent that needs MIMIC-IV-ECG; without it, drop that agent (step 4).
2. **Python environment.**
   ```bash
   python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
   pip install -r requirements.txt torch
   ```
3. **Google Cloud auth and project.**
   ```bash
   gcloud auth application-default login
   export GOOGLE_CLOUD_PROJECT=<your-project>
   export SENTINEL_BQ_STORAGE=0   # only on a sandbox project with no billing
   # optional: a separate project per dataset
   export MIMIC_PROJECT=... ECG_PROJECT=... EICU_PROJECT=...
   ```
   All extraction queries together scan roughly 50 GB, which fits in the
   BigQuery sandbox's free monthly tier.
4. **Run everything.** This fetches the MIMIC and eICU caches (each fetch skips
   if its cache already exists) and then fine-tunes:
   ```bash
   bash scripts/run_eicu_finetune.sh --label flagship
   # without MIMIC-IV-ECG access:
   SKIP_ECG=1 bash scripts/run_eicu_finetune.sh --label all6 \
       --agents vitals_tcn news2 labs gbm fluid resp
   ```
   Or run only the fine-tuning once the caches exist:
   ```bash
   python scripts/finetune_eicu.py --config configs/eicu_finetune.json --label flagship
   ```
5. **Learning curve (optional).** How much local data does a site need?
   ```bash
   for n in 500 2000 10000; do
     python scripts/finetune_eicu.py --config configs/eicu_finetune.json \
         --train-patients $n --label n$n
   done
   ```

The fused agent bank holds MIMIC and eICU together (about 8.5M rows), so plan
for 16 GB of RAM or more. No GPU is needed. On a 20-thread laptop CPU, a
single-seed run takes about 75 minutes after the fetches and each extra seed
about 55 more, mostly the XGBoost out-of-fold refits. `--resume` lets a
long run be stopped and picked up again.

## Outputs

All outputs are gitignored, because they are derived from credentialed data.

| path | contents |
|---|---|
| `data/cache/eicu_finetune_<label>.json` | settings, split report, every metric |
| `data/cache/eicu_finetune_<label>.npz` | per-arm eICU test scores, keyed by (stay_id, hour) |
| `data/cache/eicu_ft_split_<label>.parquet` | stay → split assignment |
| `data/cache/embeddings_eicu_ft/<label>/` | agent emissions per arm |
| `models/eicu_finetune/<label>/` | fine-tuned agent weights, `gbm.json`, `head_T.pt`, `head_D.pt` |

The report prints pooled and matched-density (all five vitals fresh ≤ 2 h)
AUPRC, lift, AUROC and equal-mass ECE for every arm, the MIMIC test
forgetting check, and paired ΔAUPRC against `zero_shot` with 95%
patient-clustered bootstrap intervals.

## Results: eICU transfer flagship, 5 fine-tuning seeds

Agents: `vitals_tcn news2 labs ecg_meas gbm fluid resp`. That is ALL-8 minus
cadence, the T22 eICU configuration. The MIMIC source stack and the split are
fitted at seed 42; seeds 42, 1, 2, 3 and 4 re-roll everything trained on eICU.

eICU test is 49 held-out hospitals: 23,668 stays and 990,683 hours, 3.76% of
them positive. Fine-tuning used 129 hospitals, and early stopping and
calibration used another 30. 1,303 patients with stays in more than one split
were dropped from val/test.

| arm | eICU AUPRC | lift | AUROC | ECE | ΔAUPRC vs zero-shot [95% CI] |
|---|---|---|---|---|---|
| zero_shot | 0.1293 | 3.44 | 0.7516 | 0.0338 | — |
| recal | 0.1435 | 3.82 | 0.7517 | 0.0023 | +0.0143 [+0.0114, +0.0175] |
| heads_ft, 5-seed mean | 0.1742 ± 0.0007 | 4.64 | 0.7753 | 0.0052–0.0057 | +0.0450 (range +0.0439 to +0.0457) |
| full_ft, 5-seed mean | **0.1888 ± 0.0004** | **5.03** | **0.7950** | 0.0068–0.0073 | **+0.0596 (range +0.0590 to +0.0599)** |

- **Intervals:** all 10 fine-tuned seeds clear zero (5/5 heads_ft, 5/5
  full_ft). The narrowest full_ft interval is seed 42 at +0.0590
  [+0.0524, +0.0665].
- **Seed noise:** the widest full_ft seed gap is +0.0009 [−0.0004, +0.0019]
  (s1 − s42). That spans zero, so the seeds are indistinguishable from each
  other, and it is about 65× smaller than the gain.
- **Bootstrap:** patient-clustered, 300 reps. Every arm of every seed is scored
  on identical resamples. ECE is equal-mass.

**Matched charting density** (all five vitals fresh ≤ 2 h, 283,543 hours):

| arm | AUPRC | AUROC |
|---|---|---|
| zero_shot | 0.1618 | 0.7610 |
| recal | 0.1643 | 0.7617 |
| heads_ft, mean | 0.1804 | 0.7734 |
| full_ft, mean | 0.1921 | 0.7856 |

**Forgetting (MIMIC test).**

| arm | AUPRC | AUROC |
|---|---|---|
| zero_shot | 0.3050 | 0.8214 |
| heads_ft, mean | 0.2751 | 0.8182 |
| full_ft, mean | 0.2708 | 0.8120 |

A site that fine-tunes ends up with a site-specific model, and the MIMIC weights
should be kept for MIMIC.

**ecg_meas ablation (seed 42, single run).** Dropping `ecg_meas` changes almost
nothing:

| arm | 7 agents | 6 agents |
|---|---|---|
| zero_shot, eICU | 0.1293 | 0.1299 |
| full_ft, eICU | 0.1883 | 0.1878 |
| zero_shot, MIMIC test | 0.3050 | 0.3051 |

This is expected, since eICU has no ECGs.

**Reading.**
- Fine-tuning on a site's own hospitals adds about +0.060 AUPRC (+46%
  relative) and +0.043 AUROC over the frozen MIMIC model.
- About three-quarters of that gain comes from the two ~25k-parameter heads
  alone. A site that can only afford to retrain the heads still gets most of
  it.
- `recal` moves AUPRC as well as ECE. The noisy-OR combines two calibrated
  probabilities, so refitting the head calibrators changes the combined
  ranking, not just its scale.
- On eICU val, continued boosting lifts the XGBoost agent from 0.1521 to
  0.1614–0.1617 AUPRC across the five seeds.
