"""Sadanandan 2026 (arXiv:2603.14719v1) cohort, samples and label, for T49.

Two rule sets. ``include``/``samples`` follow the paper's text, which does not
reproduce the published counts. ``code_cohort``/``code_samples`` follow their
public code (github.com/thedatasense/Patient_Early_Deterioration_Risk_Prediction
at 785d8d7: src/data/cohort.py, labels.py, samples.py), which reproduces 74,822
stays and 5,720,904 samples exactly on MIMIC-IV 3.1; T49 uses the code rules
(agreed 2026-11-19). The code differs from the text: no vital-sign or lab
filter; age is ``anchor_age``; samples run from hour 6 to ``int(los) - 24``;
vasopressors include dobutamine (``scripts/t49_therapy_code.sql``).

The paper's text:

* Section 2.2, inclusion: ICU stays of at least 24 h; age 18 or over; vital
  signs recorded within the first 6 h. Exclusion: death within 6 h of ICU
  admission; missing admission or discharge times; no recorded vital signs or
  laboratory values. Repeat stays are kept as separate stays.
* Section 2.3, label: positive at hour t if ICU death, vasopressor initiation
  or mechanical-ventilation initiation occurs in (t, t + 24 h]. ICU death is the
  hospital expiry flag with the death time inside the ICU stay. "Initiation" is
  the first administration in the stay, so a therapy that has already started
  by t no longer counts; a therapy not yet started, or death, still does.
* Section 2.5, samples: one per hour from hour 6 through hour 48 or discharge,
  whichever comes first.

``stays`` is the frame from ``scripts/t49_stays.sql`` and ``therapy`` the one
from ``scripts/t49_therapy.sql``; all times are hours from ICU admission.
"""

import numpy as np
import pandas as pd

FIRST_HOUR = 6
LAST_HOUR = 48
MIN_LOS_H = 24.0
MIN_AGE = 18
VITALS_WINDOW_H = 6.0
EARLY_DEATH_H = 6.0
HORIZON_H = 24.0
LATE_YEAR_GROUP = "2020 - 2022"      # beyond the paper's 2008-2019


def age_at_icu(stays):
    """Age at ICU admission from MIMIC's anchor fields."""
    return stays["anchor_age"] + (stays["intime_year"] - stays["anchor_year"])


def icu_death_h(stays):
    """Hours from ICU admission to death when the death is inside the ICU stay,
    else NaN."""
    d = stays["death_h"].astype(float)
    inside = ((stays["hospital_expire_flag"] == 1) & (d >= 0)
              & (d <= stays["los_h"].astype(float)))
    return d.where(inside)


def include(stays, restrict_years=True, labs_in="icu"):
    """Apply Section 2.2 step by step.

    Returns ``(kept, steps)``: the included stays and a list of
    ``(step, stays_left, patients_left)``. ``labs_in`` is ``"icu"`` (labs
    recorded during the ICU stay) or ``"hadm"`` (anywhere in the admission).
    """
    s = stays.copy()
    steps = [("all ICU stays", len(s), s["subject_id"].nunique())]

    def keep(mask, name):
        nonlocal s
        s = s.loc[mask.loc[s.index]]
        steps.append((name, len(s), s["subject_id"].nunique()))

    keep(~(stays["icu_time_missing"].astype(bool)
           | stays["hosp_time_missing"].astype(bool)),
         "admission and discharge times present")
    keep(stays["los_h"] >= MIN_LOS_H, "ICU stay >= 24 h")
    keep(age_at_icu(stays) >= MIN_AGE, "age >= 18")
    keep(stays["first_vital_h"].notna() & (stays["first_vital_h"] < VITALS_WINDOW_H),
         "vital signs within the first 6 h")
    early = (stays["hospital_expire_flag"] == 1) & (stays["death_h"] < EARLY_DEATH_H)
    keep(~early, "no death within 6 h")
    labs = stays["n_labs_icu"] if labs_in == "icu" else stays["n_labs_hadm"]
    keep((stays["n_vitals"] > 0) & (labs > 0), "vital signs and labs recorded")
    if restrict_years:
        keep(stays["anchor_year_group"] != LATE_YEAR_GROUP,
             "anchor_year_group 2008-2019")
    return s, steps


def code_cohort(stays):
    """Their code's cohort (``build_icu_cohort``): ICU times present, ICU stay
    of 24 h or more, ``anchor_age`` 18 or over, no ICU death within 6 h.
    Returns ``(kept, steps)`` like :func:`include`."""
    s = stays.copy()
    steps = [("all ICU stays", len(s), s["subject_id"].nunique())]

    def keep(mask, name):
        nonlocal s
        s = s.loc[mask.loc[s.index]]
        steps.append((name, len(s), s["subject_id"].nunique()))

    keep(~stays["icu_time_missing"].astype(bool), "ICU times present")
    keep(stays["los_h"] >= MIN_LOS_H, "ICU stay >= 24 h")
    keep(stays["anchor_age"] >= MIN_AGE, "anchor_age >= 18")
    death = icu_death_h(stays)
    keep(~(death.notna() & (death < EARLY_DEATH_H)), "no ICU death within 6 h")
    return s, steps


def code_samples(stays, starts, first_hour=FIRST_HOUR):
    """Their code's samples (``build_hourly_samples``): hours ``first_hour`` to
    ``int(los) - 24``, so every sample has its whole 24 h horizon inside the
    stay. Same columns and label as :func:`samples`."""
    los = np.floor(stays["los_h"].to_numpy(float)).astype(int)
    return samples(stays, starts, first_hour=first_hour, last_hour=None,
                   ends=los - int(HORIZON_H))


def first_starts(therapy):
    """``stay_id -> (vaso_h, vent_h)``: the first start of each therapy."""
    t = therapy.dropna(subset=["start_h"])
    first = t.groupby(["stay_id", "kind"])["start_h"].min().unstack("kind")
    for kind in ("vaso", "vent"):
        if kind not in first:
            first[kind] = np.nan
    return first.rename(columns={"vaso": "vaso_h", "vent": "vent_h"})[["vaso_h", "vent_h"]]


def samples(stays, starts, first_hour=FIRST_HOUR, last_hour=LAST_HOUR, ends=None):
    """One row per (stay, hour) with the label and its parts.

    Hours run from ``first_hour`` to ``min(last_hour, floor(los_h))``;
    ``last_hour=None`` runs to discharge; ``ends`` (one per stay, in ``stays``
    order) overrides the last hour. Columns: ``stay_id``, ``subject_id``,
    ``hour``, ``label``, its two parts ``label_t`` (a first vasopressor or
    ventilation start in the next 24 h) and ``label_d`` (ICU death in the next
    24 h), with ``label = label_t | label_d``, and ``started`` (vasopressors or
    ventilation already started at or before the hour).
    """
    s = stays.set_index("stay_id")
    los = np.floor(s["los_h"].to_numpy(float)).astype(int)
    end = los if last_hour is None else np.minimum(los, last_hour)
    if ends is not None:
        end = np.asarray(ends, dtype=int)
    n = np.clip(end - first_hour + 1, 0, None)
    stay = np.repeat(s.index.to_numpy(), n)
    offset = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    hour = first_hour + offset
    out = pd.DataFrame({"stay_id": stay,
                        "subject_id": np.repeat(s["subject_id"].to_numpy(), n),
                        "hour": hour})
    st = starts.reindex(s.index)
    death = np.repeat(icu_death_h(s.reset_index()).to_numpy(float), n)
    vaso = np.repeat(st["vaso_h"].to_numpy(float), n)
    vent = np.repeat(st["vent_h"].to_numpy(float), n)
    h = hour.astype(float)

    def within(t):
        return (t > h) & (t <= h + HORIZON_H)       # NaN compares False

    t_part = within(vaso) | within(vent)
    d_part = within(death)
    out["label"] = (t_part | d_part).astype(np.int8)
    out["label_t"] = t_part.astype(np.int8)
    out["label_d"] = d_part.astype(np.int8)
    out["started"] = ((vaso <= h) | (vent <= h)).astype(np.int8)
    return out


def stay_outcomes(stays, starts):
    """Stay-level outcome counts comparable with the paper's Table 3."""
    st = starts.reindex(stays["stay_id"])
    death = icu_death_h(stays).notna().to_numpy()
    vaso = st["vaso_h"].notna().to_numpy()
    vent = st["vent_h"].notna().to_numpy()
    return {"icu_death": int(death.sum()), "vasopressor": int(vaso.sum()),
            "ventilation": int(vent.sum()),
            "any_event": int((death | vaso | vent).sum())}
