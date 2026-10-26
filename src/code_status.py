"""Charted limitation of care: when the decision was documented, not inferred.

T7 closed with a caveat it could not act on:

    "A death-only model is a LOWER bound on physiological signal: death is
    itself treatment-mediated (DNR/withdrawal); comfort-care exclusion is
    documented future work."

That analysis was blocked on knowing *when* a limitation of care was
documented. MIMIC-IV charts it directly -- ``itemid`` 223758 is a timestamped
ICU flowsheet item with seven distinct values, not free text -- so the honest
first move is to read the field rather than infer it from notes with a language
model. An LLM earns a place here only where this field is silent, and
:func:`coverage_report` is what decides whether that gap is worth the trouble.

**Two components, not one.** The obvious confound is death: a DNR order changes
who dies. The less obvious one is ventilation. Our composite label counts
invasive ventilation onset, and a **DNI order forecloses exactly that event** --
so a stay under DNI is one where the vent arm of the label was decided by a
person, not by physiology. Both are tracked separately here for that reason.

**This is never a model feature.** A code-status entry is a clinician's decision
about the outcome we predict; feeding it forward would be predicting the order
set, which is the precise failure T7 exists to rule out. It is used only to
PARTITION stays for a sensitivity analysis -- the same audit-only posture as the
discharge-note audit in T16, and it carries the
same asymmetry: if the lead survives the exclusion, that is strong evidence; if
it drops, the reading is ambiguous because the excluded stays differ in more
ways than one.
"""

import re

import numpy as np
import pandas as pd

CODE_STATUS_PATH = "data/cache/mimic_code_status.parquet"

# Every distinct value in the extract (7 of them, all enumerated below).
# Written as patterns rather than an exact-match table so a re-fetch that adds a
# spacing variant does not silently fall through -- and `classify_census` fails
# loudly if anything does.
_FULL = re.compile(r"^\s*full code\s*$", re.I)
_CMO = re.compile(r"comfort measures", re.I)
_CPR = re.compile(r"\bDNR\b|\bDNAR\b|do not resuscitate", re.I)
_INTUB = re.compile(r"\bDNI\b|do not intubate", re.I)


def classify(value, strict=True):
    """One charted value -> what it forecloses.

    Comfort measures only implies both limits: a patient on comfort measures is
    neither resuscitated nor intubated, and coding it as CMO alone would let a
    stay count as unrestricted for the ventilation arm.

    A non-empty value that matches no pattern raises under ``strict`` (the
    default): silent fallthrough would file an unrecognised DNR variant under
    "no limitation". Missing values (None/"") are not unknown -- they classify
    to all-False without raising. ``classify_census`` passes ``strict=False``
    so it can aggregate every unmapped value before raising its own error.
    """
    v = "" if value is None else str(value)
    cmo = bool(_CMO.search(v))
    full = bool(_FULL.match(v))
    flags = {
        "is_full": full,
        "is_cmo": cmo,
        "limits_cpr": cmo or bool(_CPR.search(v)),
        "limits_intubation": cmo or bool(_INTUB.search(v)),
    }
    if strict and v.strip() and not any(flags.values()):
        raise ValueError(f"unmapped code-status value: {value!r}")
    return flags


# Palliative-plan rows are a different KIND of evidence: an open goals-of-care
# care plan, not a statement of what will not be done. They are never folded
# into the limitation flags -- a palliative consult is not a DNR -- but they do
# mark a stay where the conversation was documented, which is the weaker signal
# worth reporting beside the stronger one.
PALLIATIVE_KIND = "palliative_plan"
CODE_STATUS_KIND = "code_status"


def classify_census(values, strict=True):
    """Report any value that matches no pattern; raise if strict.

    Silent fallthrough to "no limitation" would bias the sensitivity analysis in
    the flattering direction -- an unrecognised DNR string would leave the stay
    in the unrestricted group.
    """
    unknown = {}
    for v, n in pd.Series(values).value_counts().items():
        f = classify(v, strict=False)
        if not any(f.values()):
            unknown[v] = int(n)
    if unknown:
        print(f"  [unmapped:code_status] {len(unknown)} value(s): {unknown}")
        if strict:
            raise ValueError(f"unmapped code-status values: {list(unknown)}")
    return unknown


def load_code_status(path=CODE_STATUS_PATH, strict=True):
    """Raw charted rows, typed and classified, one row per charted entry.

    Only ``code_status`` rows are classified into limitation flags; palliative
    care-plan rows carry ``is_palliative_plan`` and nothing else, so an open
    care plan can never be counted as a limitation of care.
    """
    df = pd.read_parquet(path)
    df["stay_id"] = df["stay_id"].astype("int64")
    df["hour"] = df["hour"].astype(float)
    if "kind" not in df:
        df["kind"] = CODE_STATUS_KIND          # pre-palliative cache
    is_cs = df["kind"].eq(CODE_STATUS_KIND)

    classify_census(df.loc[is_cs, "value"], strict=strict)
    # strict=False here: the census above already enforced the caller's policy
    # with the full aggregated list, which beats dying on the first bad row.
    flags = pd.DataFrame(
        [classify(v, strict=False) if k
         else {"is_full": False, "is_cmo": False,
               "limits_cpr": False, "limits_intubation": False}
         for v, k in zip(df["value"], is_cs)], index=df.index)
    flags["is_palliative_plan"] = ~is_cs
    return pd.concat([df, flags], axis=1)


def stay_timeline(records):
    """Per stay: when each kind of limitation was first documented.

    Hours are offsets from ICU intime and may be negative -- a code status set
    before the patient reached the unit is still a real, earlier decision, so it
    is kept rather than clipped, and counted in :func:`coverage_report`.
    """
    def first_hour(g, col):
        h = g.loc[g[col], "hour"]
        return float(h.min()) if len(h) else np.nan

    rows = []
    for stay, g in records.groupby("stay_id", sort=False):
        g = g.sort_values("hour")
        cs = g[~g["is_palliative_plan"]]
        any_limit = cs["limits_cpr"] | cs["limits_intubation"]
        rows.append({
            "stay_id": stay,
            "n_status_records": int(len(cs)),
            "has_code_status": bool(len(cs)),
            "has_palliative_plan": bool(g["is_palliative_plan"].any()),
            "first_status_h": float(cs["hour"].iloc[0]) if len(cs) else np.nan,
            "first_status_is_limit": bool(any_limit.iloc[0]) if len(cs) else False,
            "first_limit_h": float(cs.loc[any_limit, "hour"].min()) if any_limit.any() else np.nan,
            "first_cpr_limit_h": first_hour(cs, "limits_cpr"),
            "first_intub_limit_h": first_hour(cs, "limits_intubation"),
            "first_cmo_h": first_hour(cs, "is_cmo"),
            "first_palliative_h": first_hour(g, "is_palliative_plan"),
            "ever_full": bool(cs["is_full"].any()) if len(cs) else False,
            "ever_limited": bool(any_limit.any()),
            "last_value": str(cs["value"].iloc[-1]) if len(cs) else "",
        })
    return pd.DataFrame(rows)


def attach_to_stays(stays, timeline):
    """Left-join the timeline onto a one-row-per-stay frame, safely.

    ``stays`` must carry ``stay_id`` and ``event_h``. Adds ``has_code_status``
    and, for stays with an event, whether a limitation was documented strictly
    BEFORE that event and how many hours before -- the quantity the T7 caveat
    turns on.
    """
    stays = stays.copy()
    stays["stay_id"] = stays["stay_id"].astype("int64")
    n = len(stays)
    out = stays.merge(timeline, on="stay_id", how="left", validate="one_to_one")
    if len(out) != n:
        raise ValueError(f"code-status join changed row count {n} -> {len(out)}")

    for c in ("has_code_status", "has_palliative_plan", "ever_limited",
              "first_status_is_limit", "ever_full"):
        out[c] = out[c].astype("object").where(out[c].notna(), False).astype(bool)
    out["has_any_goc"] = out["has_code_status"] | out["has_palliative_plan"]
    ev = out["event_h"]
    for kind, col in (("limit", "first_limit_h"), ("cmo", "first_cmo_h"),
                      ("intub_limit", "first_intub_limit_h"),
                      ("palliative", "first_palliative_h")):
        before = out[col].notna() & ev.notna() & (out[col] < ev)
        out[f"{kind}_before_event"] = before
        out[f"hours_{kind}_to_event"] = np.where(before, ev - out[col], np.nan)
    return out


def coverage_report(attached, by=("event_type",)):
    """Who has a charted code status, and who does not.

    This is the number that decides whether a language model is worth running
    over notes: the field is only worth supplementing where it is silent, and
    silence concentrated in the stays that matter is a different problem from
    silence spread evenly.
    """
    rows = []
    for key, g in attached.groupby(list(by), dropna=False):
        label = key if isinstance(key, str) else "/".join(str(k) for k in np.atleast_1d(key))
        ev = g["event_h"].notna()
        rows.append({
            "group": "event-free" if label in ("nan", "None", "") else label,
            "stays": len(g),
            "with_code_status": int(g["has_code_status"].sum()),
            "coverage_pct": round(100 * g["has_code_status"].mean(), 1),
            "with_palliative": int(g["has_palliative_plan"].sum()),
            "any_goc_pct": round(100 * g["has_any_goc"].mean(), 1),
            "ever_limited": int(g["ever_limited"].sum()),
            "limit_before_event": int(g["limit_before_event"].sum()) if ev.any() else 0,
            "cmo_before_event": int(g["cmo_before_event"].sum()) if ev.any() else 0,
            "median_h_limit_to_event": (round(float(g["hours_limit_to_event"].median()), 1)
                                        if g["hours_limit_to_event"].notna().any() else np.nan),
        })
    return pd.DataFrame(rows).sort_values("stays", ascending=False).reset_index(drop=True)
