"""Unit tests for the NEWS2 scorer.

Covers each parameter's scoring bands (including band boundaries and extreme
values) and the aggregate wrapper.
"""

import math

import pytest

from news2 import (
    clean_fio2_percent,
    clean_o2_flow,
    degraded_news2_score,
    full_news2_score,
    is_alert_from_gcs,
    news2_score,
    score_consciousness,
    supplemental_oxygen_flag,
    score_heart_rate,
    score_oxygen,
    score_respiratory_rate,
    score_spo2,
    score_systolic_bp,
    score_temperature,
)


# --- Individual parameter bands -------------------------------------------

@pytest.mark.parametrize(
    "respiratory_rate, expected",
    [
        (4, 3),    # extreme low
        (8, 3),    # <=8 upper boundary
        (9, 1),    # 9-11 lower boundary
        (11, 1),   # 9-11 upper boundary
        (12, 0),   # 12-20 lower boundary
        (16, 0),   # normal
        (20, 0),   # 12-20 upper boundary
        (21, 2),   # 21-24 lower boundary
        (24, 2),   # 21-24 upper boundary
        (25, 3),   # >=25 lower boundary
        (60, 3),   # extreme high
    ],
)
def test_score_respiratory_rate(respiratory_rate, expected):
    assert score_respiratory_rate(respiratory_rate) == expected


@pytest.mark.parametrize(
    "spo2, expected",
    [
        (70, 3),    # extreme low
        (91, 3),    # <=91 upper boundary
        (92, 2),    # 92-93 lower boundary
        (93, 2),    # 92-93 upper boundary
        (94, 1),    # 94-95 lower boundary
        (95, 1),    # 94-95 upper boundary
        (96, 0),    # >=96 lower boundary
        (100, 0),   # extreme high
    ],
)
def test_score_spo2(spo2, expected):
    assert score_spo2(spo2) == expected


@pytest.mark.parametrize(
    "on_oxygen, expected",
    [
        (False, 0),  # breathing air
        (True, 2),   # supplemental oxygen
    ],
)
def test_score_oxygen(on_oxygen, expected):
    assert score_oxygen(on_oxygen) == expected


@pytest.mark.parametrize(
    "temperature, expected",
    [
        (34.0, 3),   # extreme low (hypothermia)
        (35.0, 3),   # <=35.0 upper boundary
        (35.1, 1),   # 35.1-36.0 lower boundary
        (36.0, 1),   # 35.1-36.0 upper boundary
        (36.1, 0),   # 36.1-38.0 lower boundary
        (37.0, 0),   # normal
        (38.0, 0),   # 36.1-38.0 upper boundary
        (38.1, 1),   # 38.1-39.0 lower boundary
        (39.0, 1),   # 38.1-39.0 upper boundary
        (39.1, 2),   # >=39.1 lower boundary
        (41.0, 2),   # extreme high
    ],
)
def test_score_temperature(temperature, expected):
    assert score_temperature(temperature) == expected


@pytest.mark.parametrize(
    "systolic_bp, expected",
    [
        (70, 3),     # extreme low
        (90, 3),     # <=90 upper boundary
        (91, 2),     # 91-100 lower boundary
        (100, 2),    # 91-100 upper boundary
        (101, 1),    # 101-110 lower boundary
        (110, 1),    # 101-110 upper boundary
        (111, 0),    # 111-219 lower boundary
        (120, 0),    # normal
        (219, 0),    # 111-219 upper boundary
        (220, 3),    # >=220 lower boundary
        (260, 3),    # extreme high
    ],
)
def test_score_systolic_bp(systolic_bp, expected):
    assert score_systolic_bp(systolic_bp) == expected


@pytest.mark.parametrize(
    "heart_rate, expected",
    [
        (30, 3),     # extreme low
        (40, 3),     # <=40 upper boundary
        (41, 1),     # 41-50 lower boundary
        (50, 1),     # 41-50 upper boundary
        (51, 0),     # 51-90 lower boundary
        (70, 0),     # normal
        (90, 0),     # 51-90 upper boundary
        (91, 1),     # 91-110 lower boundary
        (110, 1),    # 91-110 upper boundary
        (111, 2),    # 111-130 lower boundary
        (130, 2),    # 111-130 upper boundary
        (131, 3),    # >=131 lower boundary
        (200, 3),    # extreme high
    ],
)
def test_score_heart_rate(heart_rate, expected):
    assert score_heart_rate(heart_rate) == expected


@pytest.mark.parametrize(
    "consciousness, expected",
    [
        (True, 0),             # Alert (boolean)
        ("Alert", 0),          # Alert (string)
        ("alert", 0),          # case-insensitive
        (False, 3),            # not-Alert (boolean)
        ("Confusion", 3),      # new confusion
        ("Voice", 3),          # responds to voice
        ("Pain", 3),           # responds to pain
        ("Unresponsive", 3),   # unresponsive
    ],
)
def test_score_consciousness(consciousness, expected):
    assert score_consciousness(consciousness) == expected


# --- Aggregate wrapper -----------------------------------------------------

def test_healthy_patient_scores_zero():
    """A fully healthy patient scores 0."""
    assert (
        news2_score(
            respiratory_rate=16,
            spo2=98,
            on_oxygen=False,
            temperature=37.0,
            systolic_bp=120,
            heart_rate=70,
            consciousness="Alert",
        )
        == 0
    )


def test_maximum_score_is_twenty():
    """The worst-case combination sums to the maximum score of 20."""
    assert (
        news2_score(
            respiratory_rate=4,        # 3
            spo2=70,                   # 3
            on_oxygen=True,            # 2
            temperature=34.0,          # 3 (hypothermia)
            systolic_bp=70,            # 3
            heart_rate=200,            # 3
            consciousness="Unresponsive",  # 3
        )
        == 20
    )


def test_mixed_case_sums_sub_scores():
    """A mixed-severity patient sums to the expected aggregate."""
    # 2 + 2 + 2 + 1 + 1 + 1 + 0 = 9
    assert (
        news2_score(
            respiratory_rate=22,   # 2
            spo2=93,               # 2
            on_oxygen=True,        # 2
            temperature=38.5,      # 1
            systolic_bp=105,       # 1
            heart_rate=95,         # 1
            consciousness="Alert",  # 0
        )
        == 9
    )


# --- Degraded NEWS2 (PhysioNet-2019 stand-in) ------------------------------

class TestDegradedNews2:
    def test_all_normal_scores_zero(self):
        assert (
            degraded_news2_score(
                respiratory_rate=16, spo2=98, temperature=37.0,
                systolic_bp=120, heart_rate=70, fio2=None,
            )
            == 0
        )

    def test_hand_computed_score(self):
        # 2 + 2 + 2 + 2 + 2 (five vitals) + 2 (O2 from FiO2) + 0 (Alert) = 12
        assert (
            degraded_news2_score(
                respiratory_rate=22,   # 2
                spo2=93,               # 2
                temperature=39.5,      # 2
                systolic_bp=95,        # 2
                heart_rate=130,        # 2
                fio2=0.5,              # > 0.21 -> on oxygen -> 2
            )
            == 12
        )

    @pytest.mark.parametrize(
        "fio2, expected_oxygen_points",
        [
            (None, 0),    # not charted -> assume room air
            (0.21, 0),    # exactly room air -> not on oxygen
            (0.30, 2),    # above room air -> on oxygen
            (0.50, 2),
        ],
    )
    def test_oxygen_inferred_from_fio2(self, fio2, expected_oxygen_points):
        # All five vitals normal (0) and consciousness pinned Alert (0), so the
        # total is exactly the inferred oxygen contribution.
        total = degraded_news2_score(
            respiratory_rate=16, spo2=98, temperature=37.0,
            systolic_bp=120, heart_rate=70, fio2=fio2,
        )
        assert total == expected_oxygen_points

    def test_missing_vital_scored_zero_by_default(self):
        # Respiratory rate never measured; the rest normal -> stays 0, not 3.
        assert (
            degraded_news2_score(
                respiratory_rate=float("nan"), spo2=98, temperature=37.0,
                systolic_bp=120, heart_rate=70,
            )
            == 0
        )

    def test_missing_vital_score_is_configurable(self):
        # Same missing vital, but count absence as 3.
        assert (
            degraded_news2_score(
                respiratory_rate=float("nan"), spo2=98, temperature=37.0,
                systolic_bp=120, heart_rate=70, missing_vital_score=3,
            )
            == 3
        )

    def test_all_nan_is_zero_not_worst_case(self):
        # Critical: NaN must be intercepted, not fall through the band helpers
        # to the worst (3) band. All-missing under the default rule -> 0, not 15.
        nan = float("nan")
        assert degraded_news2_score(nan, nan, nan, nan, nan, fio2=nan) == 0


# --- MIMIC derivations: FiO2 / O2 flow cleaning -----------------------------

class TestOxygenCleaning:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (0.21, 21.0),    # fraction -> percent
            (0.5, 50.0),
            (1.0, 100.0),    # 1.0 is a fraction (1% would be impossible)
            (21, 21.0),      # already a percentage
            (40, 40.0),
            (100, 100.0),
            (None, None),    # missing
            (float("nan"), None),
        ],
    )
    def test_clean_fio2_valid(self, raw, expected):
        got = clean_fio2_percent(raw)
        if expected is None:
            assert math.isnan(got)
        else:
            assert got == pytest.approx(expected)

    @pytest.mark.parametrize("junk", [40100, 101, 0.0, 0.2, 15, -1])
    def test_clean_fio2_rejects_implausible(self, junk):
        # includes the 1.0-21 dead zone: too high for a fraction, below room
        # air as a percentage.
        assert math.isnan(clean_fio2_percent(junk))

    @pytest.mark.parametrize("raw, expected", [(0, 0.0), (2, 2.0), (60, 60.0), (100, 100.0)])
    def test_clean_o2_flow_valid(self, raw, expected):
        assert clean_o2_flow(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("junk", [-1, 1210, 101])
    def test_clean_o2_flow_rejects_implausible(self, junk):
        assert math.isnan(clean_o2_flow(junk))


class TestSupplementalOxygenFlag:
    def test_nothing_charted_is_room_air(self):
        assert supplemental_oxygen_flag() is False

    def test_device_implies_oxygen(self):
        assert supplemental_oxygen_flag(o2_device=1) is True

    def test_flow_implies_oxygen(self):
        assert supplemental_oxygen_flag(o2_flow=2) is True

    def test_zero_flow_is_not_oxygen(self):
        assert supplemental_oxygen_flag(o2_flow=0) is False

    def test_fio2_above_room_air_implies_oxygen(self):
        assert supplemental_oxygen_flag(fio2=0.4) is True
        assert supplemental_oxygen_flag(fio2=40) is True

    def test_room_air_fio2_is_not_oxygen(self):
        assert supplemental_oxygen_flag(fio2=0.21) is False
        assert supplemental_oxygen_flag(fio2=21) is False

    def test_junk_values_do_not_manufacture_oxygen(self):
        # The whole point of clipping: absurd charting must not read as O2.
        assert supplemental_oxygen_flag(o2_flow=-1) is False
        assert supplemental_oxygen_flag(fio2=40100) is False


# --- MIMIC derivations: consciousness from GCS (the ETT edge case) ---------

class TestIsAlertFromGcs:
    def test_not_intubated_full_gcs_is_alert(self):
        assert is_alert_from_gcs(gcs=15, gcs_verbal=5, gcs_motor=6, gcs_eyes=4,
                                 gcs_unable=0) is True

    def test_not_intubated_reduced_gcs_is_not_alert(self):
        assert is_alert_from_gcs(gcs=12, gcs_verbal=4, gcs_motor=5, gcs_eyes=3,
                                 gcs_unable=0) is False

    def test_intubated_but_awake_is_alert(self):
        """THE edge case: verbal=0 only because of the tube, not obtundation.

        Eyes 4 + motor 6 means awake and obeying commands -- must NOT be
        penalised 3 points just for being intubated.
        """
        assert is_alert_from_gcs(gcs=10, gcs_verbal=0, gcs_motor=6, gcs_eyes=4,
                                 gcs_unable=1) is True

    def test_intubated_and_obtunded_is_not_alert(self):
        assert is_alert_from_gcs(gcs=4, gcs_verbal=0, gcs_motor=3, gcs_eyes=1,
                                 gcs_unable=1) is False

    def test_intubated_eyes_open_but_not_obeying_is_not_alert(self):
        assert is_alert_from_gcs(gcs=8, gcs_verbal=0, gcs_motor=4, gcs_eyes=4,
                                 gcs_unable=1) is False

    def test_unknown_when_nothing_recorded(self):
        assert math.isnan(is_alert_from_gcs())


# --- Full 7-parameter NEWS2 ------------------------------------------------

class TestFullNews2:
    def test_healthy_alert_room_air_scores_zero(self):
        assert full_news2_score(
            respiratory_rate=16, spo2=98, temperature=37.0, systolic_bp=120,
            heart_rate=70, gcs=15, gcs_verbal=5, gcs_motor=6, gcs_eyes=4,
            gcs_unable=0,
        ) == 0

    def test_hand_computed(self):
        # 2 + 2 + 2 + 2 + 2 vitals, +2 oxygen (device), +3 not-alert = 15
        assert full_news2_score(
            respiratory_rate=22,   # 2
            spo2=93,               # 2
            temperature=39.5,      # 2
            systolic_bp=95,        # 2
            heart_rate=130,        # 2
            gcs=9, gcs_verbal=2, gcs_motor=4, gcs_eyes=3, gcs_unable=0,  # 3
            o2_device=1,           # 2
        ) == 15

    def test_intubated_awake_does_not_get_consciousness_points(self):
        """An awake intubated patient scores only the oxygen points, not +3."""
        assert full_news2_score(
            respiratory_rate=16, spo2=98, temperature=37.0, systolic_bp=120,
            heart_rate=70,
            gcs=10, gcs_verbal=0, gcs_motor=6, gcs_eyes=4, gcs_unable=1,
            o2_device=1,
        ) == 2   # oxygen only

    def test_unknown_consciousness_assumed_alert_by_default(self):
        assert full_news2_score(
            respiratory_rate=16, spo2=98, temperature=37.0, systolic_bp=120,
            heart_rate=70,
        ) == 0
