"""Sources for the all-features design on eICU rows (T35).

The no-cadence all-features GBM is fit on MIMIC and scored zero-shot on eICU.
Its features come from ``allfeat_gbm.build_allfeat_design`` unchanged; only
the source tables move to their eICU twins, as ``score_eicu.point_at_eicu``
does for the agents:

* labs, fluids, respiratory support: the eICU hourly extracts;
* cadence: an empty table with the MIMIC schema. The no-cadence arm drops
  these columns, and eICU has no faithful cadence mapping (see
  ``score_eicu.build_agent``);
* ECG: eICU has no MIMIC-IV-ECG records. The ECG index is emptied, so every
  eICU hour gets what the ECG agent gets there: ``ecg_has = 0``, zero values.

Nothing is refit or recalibrated on eICU.
"""

import numpy as np

EICU_LABS = "data/cache/eicu_labs_hourly.parquet"
EICU_FLUIDS = "data/cache/eicu_fluids_hourly.parquet"
EICU_RESP = "data/cache/eicu_resp_hourly.parquet"


def eicu_sources(mimic_sources, labs, fluid, resp, eicu_stay_ids):
    """``mimic_sources`` re-pointed at eICU tables for ``build_allfeat_design``.

    Raises if any eICU stay id appears in the MIMIC ECG index, since an
    overlap would hand eICU rows a MIMIC patient's ECG.
    """
    ecg = mimic_sources["ecg"]
    overlap = np.intersect1d(np.asarray(eicu_stay_ids),
                             ecg["index"]["stay_id"].to_numpy())
    if overlap.size:
        raise ValueError(f"{overlap.size} eICU stay ids collide with the MIMIC ECG index")
    return {
        "labs": labs,
        "fluid": fluid,
        "resp": resp,
        "cadence": mimic_sources["cadence"].iloc[:0].copy(),
        "ecg": {"index": ecg["index"].iloc[:0].copy(),
                "values": ecg["values"][:0],
                "study_ids": ecg["study_ids"][:0],
                "names": list(ecg["names"])},
    }
