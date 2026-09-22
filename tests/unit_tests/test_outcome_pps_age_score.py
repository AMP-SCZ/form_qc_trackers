import numpy as np
import pytest

from outcome_calculations import _pps_age_sex_score
from outcome_inputs import OutcomeInputError


@pytest.mark.parametrize(
    ('age', 'sex', 'expected'),
    [
        (np.array([30.0]), 'male', 2),
        ([20.0], 'male', 0),
        (40.0, 'male', 0),
        (np.array([-900.0]), 'male', -900),
        (np.array([-9.0]), 'male', -900),
        (np.array([-300.0]), 'male', -300),
        (np.array([-3.0]), 'male', -300),
        (np.array([30.0]), 'unknown', -900),
    ],
)
def test_pps_age_sex_score_handles_scalar_and_missing_values(
    age, sex, expected
) -> None:
    assert _pps_age_sex_score(age, sex) == expected


@pytest.mark.parametrize('age', [[], [20, 21]])
def test_pps_age_sex_score_rejects_non_scalar_age(age) -> None:
    with pytest.raises(OutcomeInputError, match=r'exactly one value; count='):
        _pps_age_sex_score(age, 'male')
