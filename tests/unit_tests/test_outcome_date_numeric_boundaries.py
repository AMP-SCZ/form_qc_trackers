from datetime import date

import numpy as np
import pandas as pd
import pytest

from outcome_calculations import _paternal_age_values, create_min_date


@pytest.mark.parametrize(
    ('raw_value', 'expected'),
    [
        ('2025-01-02T00:00:00', date(2025, 1, 2)),
        ('1903-03-03T00:00:00', date(1903, 3, 3)),
        ('1909-09-09 00:00:00', date(1909, 9, 9)),
    ],
)
def test_create_min_date_accepts_combined_csv_iso_timestamps(
    raw_value, expected
):
    frame = pd.DataFrame(
        {
            'redcap_event_name': ['baseline_arm_1'],
            'test_date': [raw_value],
        }
    )

    result = create_min_date(
        'test_min_date',
        frame.copy(deep=True),
        frame.copy(deep=True),
        ['test_date'],
        'baseline',
        ['baseline_arm_1'],
        'str',
    )

    assert result.loc[0, 'value'] == expected


def test_paternal_age_fallback_runs_before_numeric_missing_coercion():
    frame = pd.DataFrame(
        {
            'chrpps_fage': [np.nan, '-3', '1903-03-03T00:00:00', '55'],
            'chrpps_fdob': [
                '1980-01-01T00:00:00',
                '1985-06-15',
                '1980-01-01',
                '1980-01-01',
            ],
            'chrpps_interview_date': [
                '2020-01-01T00:00:00',
                '2025-06-15',
                '2020-01-01',
                '2020-01-01',
            ],
        }
    )

    result = _paternal_age_values(frame)

    assert result.iloc[0] == pytest.approx(40.0, abs=0.01)
    assert result.iloc[1] == pytest.approx(40.0, abs=0.01)
    assert result.iloc[2] == -300.0
    assert result.iloc[3] == 55.0


def test_paternal_age_fallback_keeps_missing_when_dates_are_unusable():
    frame = pd.DataFrame(
        {
            'chrpps_fage': [np.nan, np.nan],
            'chrpps_fdob': ['not-a-date', '1909-09-09T00:00:00'],
            'chrpps_interview_date': [
                '2025-01-01T00:00:00',
                '2000-01-01T00:00:00',
            ],
        }
    )

    result = _paternal_age_values(frame)

    assert result.tolist() == [-900.0, -900.0]
