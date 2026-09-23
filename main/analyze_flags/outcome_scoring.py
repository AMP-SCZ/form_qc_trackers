"""Small, testable scoring helpers used by the outcome calculations."""

import numpy as np
import pandas as pd

from outcome_inputs import OutcomeInputError


PPS_MISSING_CODES = frozenset({999.0, -900.0, -9.0})
PPS_NOT_APPLICABLE_CODES = frozenset({-300.0, -99.0, -3.0})


def _single_numeric_value(value, label: str) -> float:
    values = np.asarray(value).reshape(-1)
    if values.size == 0:
        return float("nan")
    if values.size != 1:
        raise OutcomeInputError(
            f"{label} must contain exactly one value; count={values.size}"
        )
    return pd.to_numeric(pd.Series(values), errors="coerce").iloc[0]


def pps_age_sex_score(age, sex: str) -> float:
    """Calculate the PPS age/sex component from exactly one age value."""
    age_values = np.asarray(age).reshape(-1)
    if age_values.size != 1:
        raise OutcomeInputError(
            "PPS age must contain exactly one value; "
            f"count={age_values.size}"
        )
    age_value = _single_numeric_value(age_values, "PPS age")
    if (
        pd.isna(age_value)
        or age_value in PPS_MISSING_CODES
        or sex not in {"male", "female"}
    ):
        return -900
    if age_value in PPS_NOT_APPLICABLE_CODES:
        return -300
    if 24 < age_value < 36 and sex == "male":
        return 2
    # ``>= 36``: an exact 36 previously fell through every branch and was
    # reported missing (review 2026-08-22, C8).
    if (0 < age_value < 25) or age_value >= 36 or sex == "female":
        return 0
    return -900


def subject_sex(values: pd.Series) -> str:
    """Return one consistent coded sex without substring-matching sentinels."""
    numeric = pd.to_numeric(
        values.replace(
            {
                r"^\s*1903-03-03(?:[T\s].*)?\s*$": "-300",
                r"^\s*1909-09-09(?:[T\s].*)?\s*$": "-900",
            },
            regex=True,
        ),
        errors="coerce",
    )
    observed = set(numeric[numeric.isin([1, 2])].tolist())
    if observed == {1}:
        return "male"
    if observed == {2}:
        return "female"
    return "unknown"


def valid_sips_date_comparison(frame: pd.DataFrame) -> pd.Series:
    """Identify rows whose two SIPS dates are real, comparable dates."""
    date_columns = ["conversion_date", "conversion_sips_group_date"]
    dates = frame[date_columns].apply(pd.to_datetime, errors="coerce")
    sentinel_dates = [pd.Timestamp("1903-03-03"), pd.Timestamp("1909-09-09")]
    return dates.notna().all(axis=1) & ~dates.isin(sentinel_dates).any(axis=1)


def pps_focc_score(value) -> float:
    """Score paternal occupation while preserving canonical missing codes."""
    numeric = _single_numeric_value(value, "PPS paternal occupation")
    if pd.isna(numeric) or numeric in PPS_MISSING_CODES:
        return -900
    if numeric in PPS_NOT_APPLICABLE_CODES:
        return -300
    return 1 if numeric > 6 else 0


def pps_immigration_score(frame: pd.DataFrame) -> float:
    """Score immigration while requiring every field used by the branch."""
    columns = [
        "chrdemo_cob",
        "chrdemo_country_situation",
        "chrpps_mcob",
        "chrpps_fcob",
    ]
    if frame.empty:
        return -900
    if len(frame) != 1:
        raise OutcomeInputError(
            "PPS immigration must contain exactly one baseline row; "
            f"count={len(frame)}"
        )
    numeric = (
        frame.reindex(columns=columns)
        .replace(
            {
                r"^\s*1903-03-03(?:[T\s].*)?\s*$": "-300",
                r"^\s*1909-09-09(?:[T\s].*)?\s*$": "-900",
            },
            regex=True,
        )
        .apply(pd.to_numeric, errors="coerce")
        .iloc[0]
    )

    situation = numeric["chrdemo_country_situation"]
    if pd.isna(situation) or situation in PPS_MISSING_CODES:
        return -900
    if situation in PPS_NOT_APPLICABLE_CODES:
        return -300

    target_countries = {3, 98, 109, 117, 179}
    if situation in {2, 3}:
        country = numeric["chrdemo_cob"]
        if pd.isna(country) or country in PPS_MISSING_CODES:
            return -900
        if country in PPS_NOT_APPLICABLE_CODES:
            return -300
        return 3 if country in target_countries else 2

    if situation == 4:
        parents = numeric[["chrpps_fcob", "chrpps_mcob"]]
        if parents.isin(target_countries).any():
            return 2.5
        if parents.isna().any() or parents.isin(PPS_MISSING_CODES).any():
            return -900
        if parents.isin(PPS_NOT_APPLICABLE_CODES).any():
            return -300
        return 1.5

    return -0.5


def pps_assist_score(
    frequency,
    use,
    *,
    risk_threshold: float,
    risk_value: float,
    default_value: float,
    exact_threshold: bool,
) -> float:
    """Score a PPS ASSIST component without treating 999 as high exposure."""
    use_value = _single_numeric_value(use, "PPS ASSIST use")
    if pd.isna(use_value) or use_value in PPS_MISSING_CODES:
        return -900
    if use_value in PPS_NOT_APPLICABLE_CODES:
        return -300
    if use_value == 0:
        return default_value

    frequency_value = _single_numeric_value(frequency, "PPS ASSIST frequency")
    if pd.isna(frequency_value) or frequency_value in PPS_MISSING_CODES:
        return -900
    if frequency_value in PPS_NOT_APPLICABLE_CODES:
        return -300
    is_risk = (
        frequency_value == risk_threshold
        if exact_threshold
        else frequency_value > risk_threshold
    )
    return risk_value if is_risk else default_value


def pps_family_history_score(values) -> float:
    """Score FIGS family history with an exhaustive missing-code fallback."""
    numeric = pd.to_numeric(
        pd.Series(np.asarray(values).reshape(-1)), errors="coerce"
    )
    if numeric.empty or numeric.isna().any():
        return -900
    if numeric.eq(3).any():
        return 5.5
    if numeric.isin([0, 1, 2]).all():
        return -2
    if numeric.isin(PPS_NOT_APPLICABLE_CODES).any():
        return -300
    if numeric.isin(PPS_MISSING_CODES.union({9.0})).any():
        return -900
    return -900


def subject_age(frame: pd.DataFrame, group: str) -> float:
    """Select one baseline age without using NumPy array truth values."""
    if frame.empty:
        return -900
    if len(frame) != 1:
        raise OutcomeInputError(
            "Subject age must contain exactly one baseline row; "
            f"count={len(frame)}"
        )
    if group not in {"chr", "hc"}:
        raise OutcomeInputError(f"Unsupported subject group for age: {group!r}")

    row = frame.iloc[0]
    primary_years = (
        "chrdemo_age_yrs_chr" if group == "chr" else "chrdemo_age_yrs_hc"
    )
    primary_months = (
        "chrdemo_age_mos_chr" if group == "chr" else "chrdemo_age_mos_hc"
    )

    def normalize(value, *, months: bool) -> float:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric) or numeric in PPS_MISSING_CODES:
            return -900
        if numeric in PPS_NOT_APPLICABLE_CODES:
            return -300
        if numeric < 0:
            return -900
        return float(numeric / 12 if months else numeric)

    candidates = [
        normalize(row[primary_years], months=False),
        normalize(row[primary_months], months=True),
        normalize(row["chrdemo_age_mos3"], months=False),
        normalize(row["chrdemo_age_mos2"], months=True),
    ]
    for candidate in candidates:
        if candidate >= 0:
            return candidate
    if -300 in candidates:
        return -300
    return -900


def pps_paternal_age_score(paternal_age, age) -> float:
    """Score paternal age using scalar values and canonical missing codes."""
    paternal_value = _single_numeric_value(paternal_age, "PPS paternal age")
    age_value = _single_numeric_value(age, "PPS subject age")
    if (
        pd.isna(paternal_value)
        or pd.isna(age_value)
        or paternal_value in PPS_MISSING_CODES
        or age_value in PPS_MISSING_CODES
        or paternal_value > 110
        or paternal_value < 0
        or age_value < 0
    ):
        if (
            paternal_value in PPS_NOT_APPLICABLE_CODES
            or age_value in PPS_NOT_APPLICABLE_CODES
        ):
            return -300
        return -900

    age_difference = paternal_value - age_value
    if age_difference > 45:
        return 3.5
    if age_difference > 35:
        return 0.5
    return -0.5


def pps_life_event_score(frame: pd.DataFrame) -> float:
    """Score six-month life events with exhaustive sentinel handling."""
    primary_columns = [
        *(f"chrpps_sixmo___{index}" for index in range(1, 9)),
        *(f"chrpps_sixmo___{index}" for index in range(10, 15)),
    ]
    columns = [*primary_columns, "chrpps_sixmo___15"]
    if frame.empty:
        return -900
    if len(frame) != 1:
        raise OutcomeInputError(
            "PPS life events must contain exactly one baseline row; "
            f"count={len(frame)}"
        )
    numeric = (
        frame.reindex(columns=columns)
        .apply(pd.to_numeric, errors="coerce")
        .iloc[0]
    )
    primary = numeric[primary_columns]
    if primary.eq(1).any():
        return 5.5
    if primary.eq(0).all() and numeric["chrpps_sixmo___15"] == 1:
        return -2
    if numeric.isna().any() or numeric.isin(PPS_MISSING_CODES).any():
        return -900
    if numeric.isin(PPS_NOT_APPLICABLE_CODES).any():
        return -300
    return -900

