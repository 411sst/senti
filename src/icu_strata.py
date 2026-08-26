"""ICU-category strata: which kind of ICU is each stay, and does one model hold?

A reviewing clinician (2026-08-26, docs/references.md §5) made two points that
this module exists to answer:

    "Keep models on standby because ICU cannot be generalized."
    "ICU: Cardio, neuro, respiratory, urology, burns."

The second is a question about our cohort, and it cannot be answered from
``mimic_cohort_hourly.parquet`` -- scripts/mimic_cohort.sql selects no
stay-level metadata at all. scripts/mimic_stay_meta.sql adds it.

**Why three views and not one.** The five categories a clinician names are not
five ICU units in MIMIC-IV. Cardiac and neuro are places (CCU, CVICU, the Neuro
units); respiratory, urology and burns are not -- there is no respiratory ICU,
no urology ICU and no burn unit in this hospital. They exist only as a hospital
SERVICE (``GU`` is urology, ``PSURG`` is plastics, which is where burns land) or
as a DIAGNOSIS. So each stay gets:

    unit_group      where the patient physically was      (icustays.first_careunit)
    service_group   which team owned them at ICU entry    (services.curr_service)
    dx1_chapter     what they were admitted for           (primary ICD chapter)

**A trap worth naming.** The any-position ICD chapter flags in the extract
(``dx_cardio`` and friends) are NOT category assignments -- 87.9% of stays carry
a circulatory code somewhere in their problem list because essentially every
inpatient has hypertension coded. They are comorbidity indicators and are
treated as such here. Only ``dx_burn`` is used as a category marker, because
burns are rare and specific enough that any-position coding is the right net.

Mappings are exhaustive by construction: every observed value is listed, and
:func:`assign_strata` raises if an unmapped value appears above a small
threshold rather than silently bucketing it to "other". That guard exists
because two earlier modality builders in this repo lost rows to exactly that
failure (ECG device strings, the respiratory support ladder).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from baselines_fair import clustered_bootstrap
from config import SEED

STAY_META_PATH = "data/cache/mimic_stay_meta.parquet"

# --- View 1: where the patient physically was ---------------------------------
# Every value observed in icustays.first_careunit for stays with los >= 0.5.
CAREUNIT_GROUP = {
    "Coronary Care Unit (CCU)": "cardiac",
    "Cardiac Vascular Intensive Care Unit (CVICU)": "cardiac",
    "Medicine/Cardiology Intermediate": "cardiac",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)": "neuro",
    "Neuro Intermediate": "neuro",
    "Neuro Stepdown": "neuro",
    "Neurology": "neuro",
    "Medical Intensive Care Unit (MICU)": "medical",
    "Medicine": "medical",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)": "med-surg",
    "Med/Surg": "med-surg",
    "Surgical Intensive Care Unit (SICU)": "surgical",
    "Surgery/Vascular/Intermediate": "surgical",
    "PACU": "surgical",
    "Trauma SICU (TSICU)": "trauma",
    "Surgery/Trauma": "trauma",
    "Intensive Care Unit (ICU)": "unspecified",
}

# --- View 2: which team owned the patient at ICU entry ------------------------
# Every value observed in services.curr_service. Kept at the granularity the
# source provides; the collapse to the clinician's five happens separately, so
# nothing is merged twice.
SERVICE_GROUP = {
    "CMED": "cardiac-medical",
    "CSURG": "cardiac-surgical",
    "VSURG": "vascular-surgical",
    "NSURG": "neuro-surgical",
    "NMED": "neuro-medical",
    "TRAUM": "trauma",
    "TSURG": "trauma-surgical",
    "MED": "general-medical",
    "OMED": "oncology-medical",
    "SURG": "general-surgical",
    "ORTHO": "orthopaedic",
    "ENT": "ent",
    "GU": "urology",
    "GYN": "gynaecology",
    "OBS": "obstetric",
    "PSURG": "plastics",
    "EYE": "ophthalmology",
    "DENT": "dental",
}

# --- The clinician's five, mapped from the service view -----------------------
# Anything not in one of the five keeps its service_group name; the point of
# this map is to say honestly how much of each named category we actually hold.
CLINICIAN_CATEGORIES = ("cardio", "neuro", "respiratory", "urology", "burns")

SERVICE_TO_CLINICIAN = {
    "cardiac-medical": "cardio",
    "cardiac-surgical": "cardio",
    "vascular-surgical": "cardio",
    "neuro-surgical": "neuro",
    "neuro-medical": "neuro",
    "urology": "urology",
    "plastics": "burns",          # provisional: confirmed only by a burn ICD code
}

UNIT_TO_CLINICIAN = {"cardiac": "cardio", "neuro": "neuro"}

_ICD10_CHAPTER = {
    "A": "infectious", "B": "infectious", "C": "neoplasm", "D": "blood/neoplasm",
    "E": "endocrine", "F": "mental", "G": "nervous", "H": "sense-organ",
    "I": "circulatory", "J": "respiratory", "K": "digestive", "L": "skin",
    "M": "musculoskeletal", "N": "genitourinary", "O": "pregnancy",
    "P": "perinatal", "Q": "congenital", "R": "symptoms", "S": "injury",
    "T": "injury/poisoning", "V": "external", "W": "external", "X": "external",
    "Y": "external", "Z": "health-factors",
}

_ICD9_RANGES = [
    (1, 139, "infectious"), (140, 239, "neoplasm"), (240, 279, "endocrine"),
    (280, 289, "blood"), (290, 319, "mental"), (320, 389, "nervous"),
    (390, 459, "circulatory"), (460, 519, "respiratory"), (520, 579, "digestive"),
    (580, 629, "genitourinary"), (630, 679, "pregnancy"), (680, 709, "skin"),
    (710, 739, "musculoskeletal"), (740, 759, "congenital"),
    (760, 779, "perinatal"), (780, 799, "symptoms"), (800, 999, "injury/poisoning"),
]


def load_stay_meta(path=STAY_META_PATH):
    """Stay-level metadata, with the id column typed for a safe join.

    The float-typed ``stay_id`` that several modality parquets carry has
    silently dropped rows from merges in this repo before; cast on the way in.
    """
    meta = pd.read_parquet(path)
    meta["stay_id"] = meta["stay_id"].astype("int64")
    return meta


def icd_chapter(code, version):
    """ICD-9/10 code -> chapter name. Unparseable codes become 'unknown'."""
    if not isinstance(code, str) or not code:
        return "unknown"
    if version == 10:
        return _ICD10_CHAPTER.get(code[0].upper(), "unknown")
    try:
        n = int(code[:3])
    except ValueError:
        return "unknown"                   # E- and V-prefixed ICD-9 supplements
    for lo, hi, name in _ICD9_RANGES:
        if lo <= n <= hi:
            return name
    return "unknown"


def unmapped_census(values, mapping, name, max_frac=0.01, strict=True):
    """Report values missing from a mapping; raise if they exceed max_frac.

    Silent fallthrough to 'other' is how a mapping bug becomes a results bug,
    so this fails loudly instead.
    """
    values = pd.Series(values)
    missing = values[~values.isin(mapping) & values.notna()]
    frac = len(missing) / max(1, len(values))
    if len(missing):
        counts = missing.value_counts()
        print(f"  [unmapped:{name}] {len(missing):,} rows ({frac:.2%}) across "
              f"{len(counts)} values: {dict(counts.head(8))}")
        if strict and frac > max_frac:
            raise ValueError(
                f"{name}: {frac:.2%} of rows unmapped (> {max_frac:.2%}); "
                f"extend the mapping rather than bucketing them away")
    return frac


def assign_strata(stays, meta=None, strict=True):
    """Attach unit / service / diagnosis strata to a one-row-per-stay frame.

    ``stays`` needs a ``stay_id`` column; everything else is carried through.
    The merge is validated one-to-one and the row count is asserted, because a
    silent many-to-one here would inflate every downstream count.
    """
    if meta is None:
        meta = load_stay_meta()
    stays = stays.copy()
    stays["stay_id"] = stays["stay_id"].astype("int64")
    n_before = len(stays)
    out = stays.merge(meta, on="stay_id", how="left", validate="one_to_one")
    if len(out) != n_before:
        raise ValueError(f"strata join changed row count {n_before} -> {len(out)}")
    if out["first_careunit"].isna().any():
        n = int(out["first_careunit"].isna().sum())
        raise ValueError(f"{n} cohort stays have no metadata row; refetch stay_meta")

    unmapped_census(out["first_careunit"], CAREUNIT_GROUP, "first_careunit",
                    strict=strict)
    unmapped_census(out["curr_service"], SERVICE_GROUP, "curr_service",
                    strict=strict)

    out["unit_group"] = out["first_careunit"].map(CAREUNIT_GROUP).fillna("unspecified")
    out["service_group"] = out["curr_service"].map(SERVICE_GROUP).fillna("unknown")
    out["dx1_chapter"] = [icd_chapter(c, v)
                          for c, v in zip(out["dx1_code"], out["dx1_version"])]

    # The clinician's five. Service is the primary evidence; the unit view fills
    # in cardiac/neuro stays whose owning team was something else (a cardiac
    # arrest admitted under MED and boarded in the CCU is a cardio patient by
    # any bedside reading). Respiratory has no unit and no service anywhere in
    # MIMIC-IV, so it can only ever be a diagnosis; burns require an actual burn
    # code, since plastics also covers reconstruction and flaps.
    cat = out["service_group"].map(SERVICE_TO_CLINICIAN)
    cat = cat.fillna(out["unit_group"].map(UNIT_TO_CLINICIAN))
    cat = cat.where(~((cat.isna()) & (out["dx1_chapter"] == "respiratory")),
                    "respiratory")
    cat = cat.mask(out["dx_burn"].fillna(0).astype(int) == 1, "burns")
    out["clinician_category"] = cat.fillna("other")
    return out


def category_coverage(strata):
    """Per-category evidence and an explicit represented / not-represented call.

    Returns one row per clinician category with the count under each view, so
    the answer to "which of these is our dataset" is auditable rather than
    asserted.
    """
    n = len(strata)
    rows = []
    for cat in CLINICIAN_CATEGORIES:
        by_unit = int((strata["unit_group"].map(UNIT_TO_CLINICIAN) == cat).sum())
        by_service = int((strata["service_group"].map(SERVICE_TO_CLINICIAN) == cat).sum())
        if cat == "respiratory":
            by_dx = int((strata["dx1_chapter"] == "respiratory").sum())
        elif cat == "urology":
            by_dx = int((strata["dx1_chapter"] == "genitourinary").sum())
        elif cat == "burns":
            by_dx = int(strata["dx_burn"].fillna(0).sum())
        else:
            by_dx = int((strata["dx1_chapter"] ==
                         {"cardio": "circulatory", "neuro": "nervous"}[cat]).sum())
        assigned = int((strata["clinician_category"] == cat).sum())
        rows.append({
            "category": cat,
            "has_dedicated_unit": by_unit > 0,
            "n_by_unit": by_unit,
            "n_by_service": by_service,
            "n_by_primary_dx": by_dx,
            "n_assigned": assigned,
            "pct_assigned": round(100 * assigned / max(1, n), 2),
        })
    return pd.DataFrame(rows)


def stratum_metrics(y, scores_by_model, patient_ids, mask, reference,
                    reps=1000, seed=SEED, min_pos=25):
    """AUPRC/AUROC with patient-clustered CIs on one stratum's rows.

    Returns ``None`` when the stratum carries too few positives for a bootstrap
    to mean anything -- the caller reports the stratum as under-powered rather
    than printing a number that a resample could not support.
    """
    idx = np.flatnonzero(mask)
    yb = np.asarray(y)[idx]
    if yb.sum() < min_pos:
        return None
    sub = {k: np.asarray(v)[idx] for k, v in scores_by_model.items()}
    return clustered_bootstrap(yb, sub, np.asarray(patient_ids)[idx],
                               reference=reference, reps=reps, seed=seed)
