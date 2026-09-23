"""Temporal values must compare consistently across pandas versions."""
from types import SimpleNamespace

import pandas as pd
import pytest

import qc_types.proposed_checks as proposed


def _checker(raw, validation=""):
    checker = object.__new__(proposed.ProposedChecks)
    checker._raw = raw.to_frame()
    checker.catalog = SimpleNamespace(fields={
        raw.name: SimpleNamespace(validation=validation),
    })
    return checker


@pytest.mark.parametrize("dtype", ["object", "string"])
def test_unvalidated_dates_preserve_recorded_wall_time(dtype):
    values = [
        "2025-01-01T00:30:00+14:00",
        "2025-01-01T23:30:00-12:00",
        "2025-01-01T12:00:00Z",
        "2025-01-01 04:05:06",
        "2025-01-01",
        "January 1, 2025 06:07:08",
        "bad date", None, pd.NA, "", "-9",
        "1909-09-09T12:00:00Z",
    ]
    raw = pd.Series(values, dtype=dtype, name="stop",
                    index=range(4, 4 + 2 * len(values), 2))
    checker = _checker(raw)
    original = checker._raw.copy(deep=True)

    actual = checker._parsed_series("stop")

    expected = pd.Series(
        [pd.Timestamp("2025-01-01 " + clock)
         for clock in ["00:30:00", "23:30:00", "12:00:00",
                       "04:05:06", "00:00:00", "06:07:08"]]
        + [pd.NaT] * 6,
        index=raw.index, name="stop", dtype="datetime64[ns]")
    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_frame_equal(checker._raw, original)
    # Mixing offset-bearing, naive and legacy text must be row-order independent.
    reverse = _checker(raw.iloc[::-1])._parsed_series("stop").iloc[::-1]
    pd.testing.assert_series_equal(reverse, expected)


@pytest.mark.parametrize("dtype", ["object", "string"])
@pytest.mark.parametrize("values", [[], [None, pd.NA, "-9", "1901-01-01"]],
                         ids=["empty", "all-missing"])
def test_missing_date_columns_remain_naive_datetime(dtype, values):
    raw = pd.Series(values, dtype=dtype, name="stop")
    actual = _checker(raw)._parsed_series("stop")
    expected = pd.Series(pd.NaT, index=raw.index, name=raw.name,
                         dtype="datetime64[ns]")
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize("validation, valid, invalid, expected", [
    ("date_ymd", "2025-01-01", "2025-01-01T00:00:00Z", "2025-01-01"),
    ("datetime_ymd", "2025-01-01 12:30", "2025-01-01T12:30:00Z", "2025-01-01 12:30"),
    ("time", "12:30", "12:30+01:00", "1900-01-01 12:30"),
])
def test_declared_temporal_validation_remains_strict(validation, valid, invalid, expected):
    raw = pd.Series([valid, invalid, None], name="date_value")
    actual = _checker(raw, validation)._parsed_series(raw.name)
    wanted = pd.Series([pd.Timestamp(expected), pd.NaT, pd.NaT],
                       name=raw.name, dtype="datetime64[ns]")
    pd.testing.assert_series_equal(actual, wanted)


def test_aware_timestamp_input_can_compare_with_missing_or_naive_column():
    raw = pd.Series(pd.date_range("2025-01-01 23:30", periods=2, tz="UTC"),
                    name="stop", index=[2, 5])
    checker = _checker(raw)
    actual = checker._parsed_series("stop")
    missing = checker._parsed_series("absent")
    expected = pd.Series([pd.Timestamp("2025-01-01 23:30"),
                          pd.Timestamp("2025-01-02 23:30")],
                         name="stop", index=raw.index, dtype="datetime64[ns]")
    pd.testing.assert_series_equal(actual, expected)
    assert (actual - missing).isna().all()
