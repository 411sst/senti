"""T49: Sadanandan cohort rules, samples and label. Synthetic stays only."""

import numpy as np
import pandas as pd

from sadanandan_cohort import (first_starts, icu_death_h, include, samples,
                               stay_outcomes)


def _stay(stay_id, **kw):
    row = dict(stay_id=stay_id, subject_id=stay_id // 10, hadm_id=stay_id + 5,
               first_careunit="MICU", los_h=60.0, intime_year=2150,
               anchor_age=60, anchor_year=2150, anchor_year_group="2011 - 2013",
               gender="F", hospital_expire_flag=0, icu_time_missing=False,
               hosp_time_missing=False, death_h=np.nan, first_vital_h=0.5,
               n_vitals=50, n_labs_icu=10, n_labs_hadm=20)
    row.update(kw)
    return row


def _frame(rows):
    return pd.DataFrame(rows)


def test_each_inclusion_rule_removes_only_its_stay():
    rows = [
        _stay(1010),                                         # kept
        _stay(1020, los_h=23.9),                             # too short
        _stay(1030, anchor_age=17),                          # under 18
        _stay(1040, first_vital_h=6.0),                      # no vitals in first 6 h
        _stay(1050, hospital_expire_flag=1, death_h=5.5),    # died within 6 h
        _stay(1060, n_labs_icu=0),                           # no labs
        _stay(1070, hosp_time_missing=True),                 # missing times
        _stay(1080, anchor_year_group="2020 - 2022"),        # outside 2008-2019
        _stay(1090, first_vital_h=np.nan, n_vitals=0),       # no vitals at all
    ]
    kept, steps = include(_frame(rows))
    assert kept["stay_id"].tolist() == [1010]
    # 1070 times, 1020 LOS, 1030 age, 1040 and 1090 vitals in 6 h, 1050 early
    # death, 1060 labs, 1080 years
    assert [n for _, n, _ in steps] == [9, 8, 7, 6, 4, 3, 2, 1]


def test_year_restriction_is_optional_and_age_uses_anchor_fields():
    rows = [_stay(1010, anchor_year_group="2020 - 2022"),
            _stay(1020, anchor_age=17, intime_year=2152, anchor_year=2150)]  # 19 at ICU
    kept, _ = include(_frame(rows), restrict_years=False)
    assert sorted(kept["stay_id"]) == [1010, 1020]


def test_labs_anywhere_in_the_admission_is_a_separate_option():
    rows = [_stay(1010, n_labs_icu=0, n_labs_hadm=3)]
    assert include(_frame(rows))[0].empty
    assert len(include(_frame(rows), labs_in="hadm")[0]) == 1


def test_icu_death_needs_the_flag_and_a_time_inside_the_stay():
    s = _frame([_stay(1010, hospital_expire_flag=1, death_h=40.0),
                _stay(1020, hospital_expire_flag=1, death_h=70.0),   # after ICU discharge
                _stay(1030, hospital_expire_flag=0, death_h=40.0)])
    d = icu_death_h(s)
    assert d.iloc[0] == 40.0 and np.isnan(d.iloc[1]) and np.isnan(d.iloc[2])


def test_sample_hours_run_from_6_to_48_or_discharge():
    s = _frame([_stay(1010, los_h=30.7), _stay(1020, los_h=100.0)])
    starts = first_starts(pd.DataFrame(columns=["stay_id", "kind", "start_h"]))
    out = samples(s, starts)
    assert out.loc[out.stay_id == 1010, "hour"].tolist() == list(range(6, 31))
    assert out.loc[out.stay_id == 1020, "hour"].tolist() == list(range(6, 49))
    full = samples(s, starts, last_hour=None)
    assert full.loc[full.stay_id == 1020, "hour"].tolist() == list(range(6, 101))


def test_label_is_any_event_in_the_next_24_hours():
    s = _frame([_stay(1010, los_h=40.0, hospital_expire_flag=1, death_h=40.0)])
    starts = first_starts(pd.DataFrame(columns=["stay_id", "kind", "start_h"]))
    out = samples(s, starts).set_index("hour")["label"]
    assert out.loc[6:15].eq(0).all()          # death more than 24 h away
    assert out.loc[16:39].eq(1).all()         # death in (h, h + 24]
    assert out.loc[40] == 0                   # death at h itself is not ahead of h


def test_a_started_therapy_no_longer_counts_but_one_not_yet_started_does():
    s = _frame([_stay(1010, los_h=60.0)])
    therapy = pd.DataFrame({"stay_id": [1010, 1010, 1010],
                            "kind": ["vaso", "vaso", "vent"],
                            "start_h": [20.0, 35.0, 3.0]})   # vent before hour 6
    out = samples(s, first_starts(therapy)).set_index("hour")
    assert out.loc[6:19, "label"].eq(1).all()   # first vasopressor start ahead
    assert out.loc[20:48, "label"].eq(0).all()  # vasopressor already started; vent started at 3 h
    assert out["started"].eq(1).all()           # ventilation started before hour 6


def test_stay_outcomes_counts_each_event_once_per_stay():
    s = _frame([_stay(1010, hospital_expire_flag=1, death_h=30.0),
                _stay(1020), _stay(1030)])
    therapy = pd.DataFrame({"stay_id": [1010, 1020, 1020],
                            "kind": ["vaso", "vent", "vent"],
                            "start_h": [10.0, 12.0, 30.0]})
    o = stay_outcomes(s, first_starts(therapy))
    assert o == {"icu_death": 1, "vasopressor": 1, "ventilation": 1, "any_event": 2}


def test_code_cohort_keeps_stays_the_paper_text_would_drop():
    from sadanandan_cohort import code_cohort
    rows = [
        _stay(1010),
        _stay(1040, first_vital_h=np.nan, n_vitals=0, n_labs_icu=0),   # kept by the code
        _stay(1020, los_h=23.9),                                        # too short
        _stay(1030, anchor_age=17, intime_year=2160),                   # anchor_age decides
        _stay(1050, hospital_expire_flag=1, death_h=5.5),               # ICU death within 6 h
        _stay(1060, hospital_expire_flag=1, death_h=40.0, los_h=30.0),  # died after leaving ICU: kept
        _stay(1070, icu_time_missing=True),
        _stay(1080, anchor_year_group="2020 - 2022"),                   # no year rule
    ]
    kept, steps = code_cohort(_frame(rows))
    assert sorted(kept["stay_id"]) == [1010, 1040, 1060, 1080]
    assert [n for _, n, _ in steps] == [8, 7, 6, 5, 4]


def test_code_samples_stop_24_hours_before_discharge():
    from sadanandan_cohort import code_samples
    s = _frame([_stay(1010, los_h=30.9), _stay(1020, los_h=29.9), _stay(1030, los_h=100.2)])
    starts = first_starts(pd.DataFrame(columns=["stay_id", "kind", "start_h"]))
    out = code_samples(s, starts)
    assert out.loc[out.stay_id == 1010, "hour"].tolist() == [6]
    assert (out.stay_id == 1020).sum() == 0
    assert out.loc[out.stay_id == 1030, "hour"].tolist() == list(range(6, 77))


def test_code_samples_label_counts_dobutamine_like_any_vasopressor():
    from sadanandan_cohort import code_samples
    s = _frame([_stay(1010, los_h=80.0)])
    therapy = pd.DataFrame({"stay_id": [1010], "kind": ["vaso"], "itemid": [221653],
                            "start_h": [40.0]})
    out = code_samples(s, first_starts(therapy)).set_index("hour")["label"]
    assert out.loc[16:39].eq(1).all() and out.loc[6:15].eq(0).all() and out.loc[40:].eq(0).all()
