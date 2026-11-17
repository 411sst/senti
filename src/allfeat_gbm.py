"""All-features GBM (T30): one XGBoost on the union of the eight agents' inputs.

The flagship fuses eight agents. A reviewer's fair question is whether a single
well-tuned tree model, handed every one of those inputs, does as well. This
module builds that design matrix with each agent's OWN feature code, so the GBM
sees what the agents see at hour t and nothing recorded at or after t:

==========  =======  ==================================================  ==========
block       columns  builder                                              agent(s)
==========  =======  ==================================================  ==========
FAIR+TRAJ   145      ``baselines_fair.build_design``                      vitals_tcn,
                                                                          labs, gbm
cadence     32       ``modality_features.build_cadence_features``         cadence
fluid       13       ``modality_features.build_fluid_features``           fluid
resp        12       ``modality_features.build_resp_features`` (level 5)  resp
ECG         51 + 2   ``agents.ecg_agent.assemble_ecg_features`` (72 h)    ecg_meas
NEWS2       7        ``mimic_cohort.news2_component_scores``              news2
==========  =======  ==================================================  ==========

Hours without a usable ECG get exactly what the ECG agent gets: zeros, with
``ecg_has = 0`` and ``ecg_age = 0``. The ECG agent standardises its inputs; a
tree model is invariant to that, so the raw machine-read values are used.

The no-cadence variant drops the 32 cadence columns (the clinician-behaviour
block), for the comparison without clinician-behaviour inputs.
"""

import numpy as np
import pandas as pd

from agents.ecg_agent import ECG_INDEX, STALENESS_HOURS, assemble_ecg_features
from agents.ecg_meas_agent import MEAS_CACHE
from baselines_fair import build_design, variant_columns
from mimic_cohort import NEWS2_COMPONENTS, news2_component_scores
from modality_features import (
    CADENCE_FEATURES,
    FLUID_FEATURES,
    RESP_FEATURES,
    build_cadence_features,
    build_fluid_features,
    build_resp_features,
    load_cadence,
    load_fluids,
    load_labs,
    load_resp,
)

NEWS2_SUBSCORES = [f"news2_{c}" for c in NEWS2_COMPONENTS]
RESP_MAX_LEVEL = 5            # the shipped resp agent (not the cap-4 variant)
ARMS = ("allfeat", "allfeat_nocad")


def ecg_columns(names):
    return [f"ecg_{n}" for n in names] + ["ecg_has", "ecg_age"]


def column_sets(ecg_names):
    """``{arm: column list}`` for the full and the no-cadence design."""
    full = (list(variant_columns()["FAIR+TRAJ"]) + list(CADENCE_FEATURES)
            + list(FLUID_FEATURES) + list(RESP_FEATURES)
            + ecg_columns(ecg_names) + NEWS2_SUBSCORES)
    if len(set(full)) != len(full):
        raise ValueError("duplicate column names across agent blocks")
    cadence = set(CADENCE_FEATURES)
    return {"allfeat": full,
            "allfeat_nocad": [c for c in full if c not in cadence]}


def load_sources():
    """Every source table the eight agents read, from the local caches."""
    z = np.load(MEAS_CACHE)
    return {
        "labs": load_labs(),
        "cadence": load_cadence(),
        "fluid": load_fluids(),
        "resp": load_resp(),
        "ecg": {"index": pd.read_parquet(ECG_INDEX),
                "values": z["embeddings"].astype(np.float32),
                "study_ids": z["study_ids"],
                "names": [str(n) for n in z["feature_names"]]},
    }


def _check_aligned(frame, keys, what):
    same = ((frame["stay_id"].to_numpy() == keys["stay_id"].to_numpy()).all()
            and (frame["hour"].to_numpy() == keys["hour"].to_numpy()).all())
    if not same:
        raise ValueError(f"{what} features are misaligned with the cohort rows")


def build_allfeat_design(split_df, sources):
    """``(X, y, keys)`` for one split, rows in sorted ``(stay_id, hour)`` order.

    ``X`` holds the full column set (``column_sets(...)["allfeat"]``) as
    float32; the no-cadence arm is a column subset of it.
    """
    X_fair, y, keys = build_design(split_df, sources["labs"], note_pack=None)
    ordered = split_df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    cohort_keys = ordered[["stay_id", "hour"]]

    blocks = {c: X_fair[c].to_numpy(np.float32)
              for c in variant_columns()["FAIR+TRAJ"]}
    del X_fair

    tabular = (
        ("cadence", lambda: build_cadence_features(cohort_keys, sources["cadence"]),
         CADENCE_FEATURES),
        ("fluid", lambda: build_fluid_features(cohort_keys, sources["fluid"]),
         FLUID_FEATURES),
        ("resp", lambda: build_resp_features(cohort_keys, sources["resp"],
                                             max_level=RESP_MAX_LEVEL),
         RESP_FEATURES),
    )
    for name, build, cols in tabular:
        frame = build()
        _check_aligned(frame, keys, name)
        for c in cols:
            blocks[c] = frame[c].to_numpy(np.float32)

    ecg = sources["ecg"]
    values, has, age, ecg_keys = assemble_ecg_features(
        cohort_keys, ecg["index"], ecg["values"], ecg["study_ids"],
        staleness=STALENESS_HOURS)
    _check_aligned(ecg_keys, keys, "ECG")
    for k, n in enumerate(ecg["names"]):
        blocks[f"ecg_{n}"] = values[:, k]
    blocks["ecg_has"] = has
    blocks["ecg_age"] = age

    components, _, _ = news2_component_scores(ordered)
    for k, c in enumerate(NEWS2_SUBSCORES):
        blocks[c] = components[:, k].astype(np.float32)

    cols = column_sets(ecg["names"])["allfeat"]
    X = pd.DataFrame({c: blocks[c] for c in cols})
    return X, y, keys
