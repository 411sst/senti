"""NEWS2 (National Early Warning Score 2) scoring.

Implements the aggregate NEWS2 score from seven physiological parameters using
the Royal College of Physicians scoring bands. Each parameter is scored 0-3 by
its own helper; :func:`news2_score` sums the seven sub-scores.

Oxygen saturation uses **Scale 1** (the default scale, for patients without
known hypercapnic respiratory failure). Reference:
https://www.rcp.ac.uk/improving-care/resources/national-early-warning-score-news-2/

Notes
-----
The score ranges 0-20: a fully healthy patient scores 0, and the worst-case
additive maximum is 20 (temperature contributes 3 via hypothermia).
"""

import math

# Score contributed by each parameter, keyed by helper for readability. The
# individual helpers are the authoritative source of the bands below.
_MAX_PER_PARAM = {
    "respiratory_rate": 3,
    "spo2": 3,
    "oxygen": 2,
    "temperature": 2,
    "systolic_bp": 3,
    "heart_rate": 3,
    "consciousness": 3,
}


def score_respiratory_rate(respiratory_rate):
    """Score respiratory rate (breaths per minute).

    Bands: <=8 -> 3, 9-11 -> 1, 12-20 -> 0, 21-24 -> 2, >=25 -> 3.
    """
    if respiratory_rate <= 8:
        return 3
    if respiratory_rate <= 11:
        return 1
    if respiratory_rate <= 20:
        return 0
    if respiratory_rate <= 24:
        return 2
    return 3


def score_spo2(spo2):
    """Score oxygen saturation on Scale 1 (percent).

    Bands: <=91 -> 3, 92-93 -> 2, 94-95 -> 1, >=96 -> 0.
    """
    if spo2 <= 91:
        return 3
    if spo2 <= 93:
        return 2
    if spo2 <= 95:
        return 1
    return 0


def score_oxygen(on_oxygen):
    """Score supplemental oxygen use.

    Breathing air -> 0, receiving supplemental oxygen -> 2. ``on_oxygen`` is
    interpreted as a boolean flag.
    """
    return 2 if on_oxygen else 0


def score_temperature(temperature):
    """Score temperature (degrees Celsius).

    Bands: <=35.0 -> 3, 35.1-36.0 -> 1, 36.1-38.0 -> 0, 38.1-39.0 -> 1,
    >=39.1 -> 2.
    """
    if temperature <= 35.0:
        return 3
    if temperature <= 36.0:
        return 1
    if temperature <= 38.0:
        return 0
    if temperature <= 39.0:
        return 1
    return 2


def score_systolic_bp(systolic_bp):
    """Score systolic blood pressure (mmHg).

    Bands: <=90 -> 3, 91-100 -> 2, 101-110 -> 1, 111-219 -> 0, >=220 -> 3.
    """
    if systolic_bp <= 90:
        return 3
    if systolic_bp <= 100:
        return 2
    if systolic_bp <= 110:
        return 1
    if systolic_bp <= 219:
        return 0
    return 3


def score_heart_rate(heart_rate):
    """Score heart rate / pulse (beats per minute).

    Bands: <=40 -> 3, 41-50 -> 1, 51-90 -> 0, 91-110 -> 1, 111-130 -> 2,
    >=131 -> 3.
    """
    if heart_rate <= 40:
        return 3
    if heart_rate <= 50:
        return 1
    if heart_rate <= 90:
        return 0
    if heart_rate <= 110:
        return 1
    if heart_rate <= 130:
        return 2
    return 3


def score_consciousness(consciousness):
    """Score level of consciousness (ACVPU).

    Alert -> 0; anything else (new Confusion, or responding only to Voice,
    Pain, or Unresponsive) -> 3.

    Accepts either a boolean (``True`` = Alert) or a string level such as
    ``"Alert"``, ``"Voice"``, ``"Pain"``, ``"Unresponsive"``, ``"Confusion"``
    (case-insensitive).
    """
    if isinstance(consciousness, str):
        is_alert = consciousness.strip().lower() == "alert"
    else:
        is_alert = bool(consciousness)
    return 0 if is_alert else 3


def news2_score(
    respiratory_rate,
    spo2,
    on_oxygen,
    temperature,
    systolic_bp,
    heart_rate,
    consciousness,
):
    """Compute the aggregate NEWS2 score (0-20).

    Parameters
    ----------
    respiratory_rate : number
        Breaths per minute.
    spo2 : number
        Oxygen saturation percentage (Scale 1).
    on_oxygen : bool
        ``True`` if the patient is receiving supplemental oxygen.
    temperature : number
        Body temperature in degrees Celsius.
    systolic_bp : number
        Systolic blood pressure in mmHg.
    heart_rate : number
        Heart rate / pulse in beats per minute.
    consciousness : bool or str
        Level of consciousness; ``True``/``"Alert"`` is Alert, otherwise
        not-Alert (see :func:`score_consciousness`).

    Returns
    -------
    int
        The summed NEWS2 score.
    """
    return (
        score_respiratory_rate(respiratory_rate)
        + score_spo2(spo2)
        + score_oxygen(on_oxygen)
        + score_temperature(temperature)
        + score_systolic_bp(systolic_bp)
        + score_heart_rate(heart_rate)
        + score_consciousness(consciousness)
    )


# Room-air FiO2 is 0.21; anything charted above it implies supplemental oxygen.
ROOM_AIR_FIO2 = 0.21


def _is_missing(value):
    """True if ``value`` is None or NaN (works for python/numpy floats)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def degraded_news2_score(
    respiratory_rate,
    spo2,
    temperature,
    systolic_bp,
    heart_rate,
    fio2=None,
    missing_vital_score=0,
):
    """A **degraded** NEWS2 for the PhysioNet-2019 stand-in data.

    This dataset only carries five of the seven NEWS2 inputs directly
    (respiratory rate, SpO2, temperature, systolic BP, heart rate). The scoring
    *bands* are the exact standard ones (via the ``score_*`` helpers); what is
    degraded is the two missing inputs and how never-measured vitals are
    handled:

    * **Supplemental oxygen** -- no flag exists, so it is *inferred* from FiO2:
      ``FiO2 > 0.21`` (above room air) is treated as on-oxygen (+2). When FiO2
      is not charted (``None``/NaN, ~95% of hours) we assume room air (0).
    * **Consciousness** -- absent entirely on this dataset (no GCS/ACVPU), so it
      is pinned to Alert (0) for everyone.
    * **Never-measured vital** -- a vital still NaN after forward-fill (never
      recorded before the prediction hour) is scored ``missing_vital_score``.
      This DEFAULTS TO 0, i.e. "assume normal until measured" -- the common
      early-warning-score convention -- but it is an explicit, named choice, not
      a silent NaN-as-normal. Note NaN must be intercepted here: passing NaN
      into the band helpers would fall through to the worst (3) band, so we
      never do that. Set ``missing_vital_score`` higher to make absence count
      against the patient instead.

    Valid ONLY as a baseline on the 2019 stand-in data. On MDS-ED we expect a
    real oxygen flag and a consciousness/GCS field, so the full
    :func:`news2_score` should be used there instead.

    Returns
    -------
    int
        The degraded NEWS2 score.
    """
    def scored(value, score_fn):
        # Explicit missing-branch: never feed NaN to the band helpers.
        return missing_vital_score if _is_missing(value) else score_fn(value)

    on_oxygen = (not _is_missing(fio2)) and (fio2 > ROOM_AIR_FIO2)

    return (
        scored(respiratory_rate, score_respiratory_rate)
        + scored(spo2, score_spo2)
        + scored(temperature, score_temperature)
        + scored(systolic_bp, score_systolic_bp)
        + scored(heart_rate, score_heart_rate)
        + score_oxygen(on_oxygen)          # inferred from FiO2 (see docstring)
        + score_consciousness(True)        # pinned Alert (0) on this dataset
    )


# =========================================================================
# MIMIC-IV derivations: the two inputs that need interpretation, not lookup
# =========================================================================

# Physiologically plausible bounds. MIMIC charts contain out-of-range values
# (FiO2 seen up to 40100, O2 flow from -1 to 1210), so both are clipped before
# they are allowed to influence the supplemental-oxygen decision.
FIO2_ROOM_AIR_PCT = 21.0
O2_FLOW_MAX_LPM = 100.0   # high-flow nasal cannula tops out around 60-80 L/min


def clean_fio2_percent(fio2):
    """Normalise a charted FiO2 to a percentage, or NaN if implausible.

    MIMIC charts FiO2 two ways -- as a fraction (0.21-1.0) and as a percentage
    (21-100) -- plus outright junk. We map each to a percentage and reject the
    rest, rather than letting a value like 40100 masquerade as supplemental
    oxygen:

    * ``0.21 <= v <= 1.0``  -> fraction, returned as ``v * 100``
    * ``21 <= v <= 100``    -> already a percentage, returned as-is
    * anything else (including the 1.0-21 dead zone, which is below room air as
      a percentage and above 1.0 as a fraction) -> ``nan``
    """
    if _is_missing(fio2):
        return float("nan")
    v = float(fio2)
    if 0.21 <= v <= 1.0:
        return v * 100.0
    if FIO2_ROOM_AIR_PCT <= v <= 100.0:
        return v
    return float("nan")


def clean_o2_flow(o2_flow):
    """Clip a charted O2 flow (L/min) to a physiological range, else NaN.

    Negative flows and absurd values (MIMIC contains both) must not be read as
    evidence of supplemental oxygen.
    """
    if _is_missing(o2_flow):
        return float("nan")
    v = float(o2_flow)
    if 0.0 <= v <= O2_FLOW_MAX_LPM:
        return v
    return float("nan")


def supplemental_oxygen_flag(o2_device=None, o2_flow=None, fio2=None):
    """Is the patient on supplemental oxygen? (the NEWS2 +2 parameter)

    True if ANY of three independent signals says so, after cleaning:

    * a delivery device other than "None" was charted (``o2_device`` truthy),
    * a cleaned O2 flow strictly above 0 L/min,
    * a cleaned FiO2 strictly above room air (21%).

    **Documented assumption:** MIMIC's ``oxygen_delivery`` table largely omits
    room-air rows, so "nothing charted" is treated as breathing air (0 points)
    rather than unknown. That is the conservative choice -- it never invents an
    oxygen requirement -- but it does mean unrecorded oxygen reads as room air.
    """
    if not _is_missing(o2_device) and float(o2_device) > 0:
        return True
    flow = clean_o2_flow(o2_flow)
    if not _is_missing(flow) and flow > 0:
        return True
    pct = clean_fio2_percent(fio2)
    if not _is_missing(pct) and pct > FIO2_ROOM_AIR_PCT:
        return True
    return False


def is_alert_from_gcs(gcs=None, gcs_verbal=None, gcs_motor=None, gcs_eyes=None,
                      gcs_unable=None):
    """Map GCS to the NEWS2 consciousness parameter (Alert vs not-Alert).

    Normally Alert means a full GCS of 15. The complication is intubation:
    MIMIC encodes "No Response-ETT" as ``gcs_verbal = 0`` with
    ``gcs_unable = 1`` (35.4% of GCS rows). Scoring that verbal 0 naively would
    make every intubated patient not-Alert and hand them a permanent +3 --
    penalising the *presence of a tube* rather than any change in
    consciousness, and swamping the real signal.

    **Our rule:** when verbal is not assessable (``gcs_unable``), judge
    alertness from the two components that ARE assessable --

        Alert  <=>  eyes == 4 (spontaneous) AND motor == 6 (obeys commands)

    An awake, cooperative, intubated patient therefore scores 0, while a
    genuinely obtunded one still scores 3. When not intubated we use the
    standard ``GCS == 15``; if components are missing we fall back to the total.
    """
    intubated = (not _is_missing(gcs_unable)) and float(gcs_unable) == 1

    if intubated:
        # Verbal is meaningless here -- use eye opening + motor response only.
        if not _is_missing(gcs_eyes) and not _is_missing(gcs_motor):
            return float(gcs_eyes) == 4 and float(gcs_motor) == 6
        return float("nan") if _is_missing(gcs) else float(gcs) >= 15

    if (not _is_missing(gcs_eyes) and not _is_missing(gcs_verbal)
            and not _is_missing(gcs_motor)):
        return (float(gcs_eyes) + float(gcs_verbal) + float(gcs_motor)) >= 15
    if not _is_missing(gcs):
        return float(gcs) >= 15
    return float("nan")   # unknown -- caller decides how to treat it


def full_news2_score(
    respiratory_rate, spo2, temperature, systolic_bp, heart_rate,
    gcs=None, gcs_verbal=None, gcs_motor=None, gcs_eyes=None, gcs_unable=None,
    o2_device=None, o2_flow=None, fio2=None,
    missing_vital_score=0, assume_alert_if_unknown=True,
):
    """The complete 7-parameter NEWS2 for MIMIC-IV.

    Unlike :func:`degraded_news2_score`, nothing here is pinned or assumed
    away: consciousness comes from GCS (via :func:`is_alert_from_gcs`, which
    handles the intubated case) and supplemental oxygen from the charted
    device/flow/FiO2 (via :func:`supplemental_oxygen_flag`, which clips both
    numeric inputs first).

    A vital never measured before the prediction hour scores
    ``missing_vital_score`` (default 0, "assume normal until measured") -- the
    same explicit rule used on PhysioNet-2019. When consciousness cannot be
    determined at all, ``assume_alert_if_unknown`` decides whether that is
    treated as Alert (0) or as the missing-vital score.
    """
    def scored(value, score_fn):
        return missing_vital_score if _is_missing(value) else score_fn(value)

    alert = is_alert_from_gcs(gcs, gcs_verbal, gcs_motor, gcs_eyes, gcs_unable)
    if _is_missing(alert):
        consciousness_points = 0 if assume_alert_if_unknown else missing_vital_score
    else:
        consciousness_points = score_consciousness(bool(alert))

    return (
        scored(respiratory_rate, score_respiratory_rate)
        + scored(spo2, score_spo2)
        + scored(temperature, score_temperature)
        + scored(systolic_bp, score_systolic_bp)
        + scored(heart_rate, score_heart_rate)
        + score_oxygen(supplemental_oxygen_flag(o2_device, o2_flow, fio2))
        + consciousness_points
    )
