import numpy as np
import pandas as pd
import pytest

from outcome_calculations import _pps_age_sex_score
from outcome_scoring import (
    pps_assist_score,
    pps_family_history_score,
    pps_focc_score,
    pps_immigration_score,
    subject_sex,
)


@pytest.mark.parametrize(
    ("age", "sex", "expected"),
    [
        ([-900], "female", -900),
        ([999], "male", -900),
        ([-99], "male", -300),
        ([20], "unknown", -900),
    ],
)
def test_pps_age_sex_checks_sentinels_before_demographic_rules(
    age, sex, expected
) -> None:
    assert _pps_age_sex_score(age, sex) == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["1", np.nan], "male"),
        (["2", np.nan], "female"),
        (["1", "2"], "unknown"),
        (["1903-03-03T00:00:00"], "unknown"),
        (["1909-09-09"], "unknown"),
        (["10"], "unknown"),
    ],
)
def test_subject_sex_uses_exact_codes(values, expected) -> None:
    assert subject_sex(pd.Series(values)) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (999, -900),
        (-900, -900),
        (-9, -900),
        (-99, -300),
        (-300, -300),
        (-3, -300),
        (7, 1),
        (6, 0),
    ],
)
def test_pps_focc_preserves_canonical_codes(value, expected) -> None:
    assert pps_focc_score([value]) == expected


def test_pps_focc_missing_baseline_is_missing() -> None:
    assert pps_focc_score([]) == -900


def _immigration_row(**updates) -> pd.DataFrame:
    values = {
        "chrdemo_cob": 10,
        "chrdemo_country_situation": 1,
        "chrpps_mcob": 10,
        "chrpps_fcob": 10,
    }
    values.update(updates)
    return pd.DataFrame([values])


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (pd.DataFrame(columns=[
            "chrdemo_cob",
            "chrdemo_country_situation",
            "chrpps_mcob",
            "chrpps_fcob",
        ]), -900),
        (_immigration_row(chrdemo_country_situation=999), -900),
        (_immigration_row(chrdemo_country_situation=-99), -300),
        (_immigration_row(chrdemo_country_situation=2, chrdemo_cob=999), -900),
        (_immigration_row(chrdemo_country_situation=2, chrdemo_cob=-99), -300),
        (_immigration_row(chrdemo_country_situation=2, chrdemo_cob=3), 3),
        (_immigration_row(chrdemo_country_situation=2, chrdemo_cob=10), 2),
        (_immigration_row(chrdemo_country_situation=4, chrpps_fcob=3), 2.5),
        (_immigration_row(
            chrdemo_country_situation=4,
            chrpps_fcob=999,
            chrpps_mcob=10,
        ), -900),
        (_immigration_row(
            chrdemo_country_situation=4,
            chrpps_fcob=-99,
            chrpps_mcob=10,
        ), -300),
    ],
)
def test_pps_immigration_requires_branch_inputs(frame, expected) -> None:
    assert pps_immigration_score(frame) == expected


@pytest.mark.parametrize(
    ("frequency", "use", "exact", "expected"),
    [
        (999, 1, False, -900),
        (-9, 1, False, -900),
        (-99, 1, False, -300),
        (-3, 1, False, -300),
        (6, 999, True, -900),
        (6, -99, True, -300),
        (999, 0, False, 0),
        (6, 1, True, 7),
        (4, 1, False, 7),
    ],
)
def test_pps_assist_does_not_score_missing_as_high_exposure(
    frequency, use, exact, expected
) -> None:
    assert pps_assist_score(
        frequency,
        use,
        risk_threshold=6 if exact else 3,
        risk_value=7,
        default_value=0,
        exact_threshold=exact,
    ) == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], -900),
        ([[999] + [0] * 9], -900),
        ([[-99] + [0] * 9], -300),
        ([[3, 999] + [0] * 8], 5.5),
        ([[0] * 10], -2),
        ([[4] + [0] * 9], -900),
    ],
)
def test_pps_family_history_is_exhaustive(values, expected) -> None:
    assert pps_family_history_score(values) == expected
