"""eICU-CRD in-ICU hourly deterioration cohort: features, NEWS2, no splits.

External-validation twin of :mod:`mimic_cohort`. Consumes the raw hourly bins
fetched by ``scripts/fetch_eicu_cohort.py`` and produces the same compact
per-hour table, so a MIMIC-trained model can be scored on it unchanged.

Deliberately thin: every transformation is IMPORTED from :mod:`mimic_cohort`
rather than reimplemented. If the two datasets were processed by two different
code paths, a transfer result would confound "different hospital" with
"different preprocessing", which is exactly the thing the experiment is meant
to rule out. The only eICU-specific logic lives here:

* **Column naming.** ``subject_key`` (eICU's ``uniquepid``) becomes
  ``patient_id``; ``hospitalid`` and ``unittype`` are carried through so
  results can be broken down by site.
* **No split.** Every row is ``test``. The model is not trained, calibrated or
  thresholded on any of this -- that is the whole point. The column exists so
  downstream loaders that filter on it keep working.
* **gcs_unable.** eICU has no "verbal not assessable" flag, so it arrives NULL
  and every row takes :func:`mimic_cohort.alert_flags`' non-intubated branch
  (NaN == 1 is False). eICU charts ``gcsverbal = 1`` for intubated patients
  instead, which depresses the total and makes NEWS2's consciousness point
  fire slightly more often than it would in MIMIC. Documented, not corrected:
  correcting it would mean inventing a flag the source does not have.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from mimic_cohort import (
    CARRIED,
    EVENT_TYPES,
    HOUR,
    LABEL,
    PATIENT_ID,
    STAY_ID,
    VITALS,
    add_event_labels,
    build_before_t_features,
    compute_news2,
)

RAW_PATH = "data/cache/eicu_raw_hourly.parquet"
OUT_PATH = "data/cache/eicu_cohort_hourly.parquet"

# Site columns unique to eICU: 208 hospitals means transfer can be reported as
# a distribution over sites, not just one pooled number.
SITE = ["hospitalid", "unittype"]


def build(raw_path=RAW_PATH, out_path=OUT_PATH):
    """Build the compact per-hour eICU table and write it to Parquet."""
    raw = pd.read_parquet(raw_path)
    raw = raw.rename(columns={"subject_key": PATIENT_ID})

    # BigQuery hands back NUMERIC as decimal.Decimal and INT64 as pandas Int64;
    # coerce every measurement channel to plain float so arithmetic, NaN
    # handling and downstream model libraries all behave.
    for col in CARRIED + ["event_h", "death_h", "vent_h", "vaso_h"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").astype(float)

    feat = build_before_t_features(raw)
    feat = add_event_labels(feat)
    news2, alert, on_oxygen = compute_news2(feat)
    feat["news2"] = news2.astype(np.int16)
    feat["is_alert"] = alert
    feat["on_oxygen"] = on_oxygen

    # History rows [0, STAB_HOURS) exist so the ffill+shift above could carry
    # their observations INTO the first prediction hour; they are not
    # prediction rows themselves and leave the table here.
    if "is_history" in feat.columns:
        feat = feat[pd.to_numeric(feat["is_history"]) == 0].reset_index(drop=True)

    # Held out by construction: nothing here was ever trained or calibrated on.
    feat["split"] = "test"

    keep = ([PATIENT_ID, STAY_ID, HOUR] + SITE + VITALS
            + [f"{v}_hrs_since" for v in VITALS]
            + ["gcs", "gcs_unable", "is_alert", "on_oxygen", "news2", "event_h",
               "event_type", LABEL]
            + [f"{LABEL}_{n}" for n in EVENT_TYPES] + ["split"])
    out = feat[keep]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out


if __name__ == "__main__":
    df = build()
    print(f"[done] {len(df):,} prediction hours, {df[STAY_ID].nunique():,} stays, "
          f"{df[PATIENT_ID].nunique():,} patients, "
          f"{df.hospitalid.nunique():,} hospitals")
    print(f"       {100*df[LABEL].mean():.2f}% positive hours, "
          f"{df.loc[df.event_h.notna(), STAY_ID].nunique():,} event stays")
