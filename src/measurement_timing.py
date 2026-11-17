"""Measurement-timing columns of the all-features design (T32 sensitivity).

These columns describe WHEN something was measured or recorded rather than
what it showed: time since the last value, and record presence or counts.
They are clinician behaviour in the same sense as cadence, but they live in
the labs, vitals, ECG, fluid and resp blocks, so dropping cadence keeps them.

Kept out of ``allfeat_gbm.py`` on purpose: the T30/T31 score caches
fingerprint that file, and this selector does not change the design.
"""

MEASUREMENT_TIMING_EXTRA = ["ecg_has", "ecg_age", "n_urine_records_24h",
                            "hrs_since_urine_record", "has_urine_history",
                            "resp_seen", "resp_hrs_since_record"]


def measurement_timing_columns(cols):
    """The measurement-timing subset of ``cols``, in ``cols`` order."""
    extra = set(MEASUREMENT_TIMING_EXTRA)
    return [c for c in cols if c.endswith("_hrs_since") or c in extra]
