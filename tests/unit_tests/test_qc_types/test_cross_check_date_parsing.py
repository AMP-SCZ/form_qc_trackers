"""Cross-form date parsing must support deployed and current pandas versions."""

import pandas as pd
import pytest

import test_cross_checks as source


@pytest.mark.parametrize("dtype", ["object", "string"])
def test_calendar_dates_preserve_valid_values_and_mask_missing_codes(dtype):
    # Mixing a real date with a sentinel reproduced the pandas 1.4.2
    # StringArray failure. Nonconsecutive indices also check row alignment.
    values = [
        " 2025-04-03 ",
        "2025-04-04T23:59:00-04:00",
        "April 5, 2025 12:34:00",
        "not a date",
        None, pd.NA, float("nan"), "",
        "-3", "-9", "-99", "999",
        "1901-01-01", "1903-03-03T12:34:56", "1909-09-09 00:00:00",
    ]
    raw = pd.Series(
        values, dtype=dtype, index=range(7, 7 + 3 * len(values), 3),
        name="chrchs_interview_date")
    original = raw.copy(deep=True)

    parsed = source.prepare_dataframe(
        raw.to_frame(), [raw.name])[raw.name]

    expected = pd.Series(
        [pd.Timestamp("2025-04-03"), pd.Timestamp("2025-04-04"),
         pd.Timestamp("2025-04-05")] + [pd.NaT] * (len(values) - 3),
        index=raw.index, name=raw.name, dtype="datetime64[ns]")
    pd.testing.assert_series_equal(parsed, expected)
    pd.testing.assert_series_equal(raw, original)


@pytest.mark.parametrize("dtype", ["object", "string"])
@pytest.mark.parametrize("values", [[], [None, pd.NA, "", "-9", "1901-01-01"]],
                         ids=["empty", "all-missing"])
def test_calendar_dates_accept_empty_or_all_missing_columns(dtype, values):
    raw = pd.Series(values, dtype=dtype, name="chrchs_interview_date")

    parsed = source.prepare_dataframe(
        raw.to_frame(), [raw.name])[raw.name]

    expected = pd.Series(
        pd.NaT, index=raw.index, name=raw.name, dtype="datetime64[ns]")
    pd.testing.assert_series_equal(parsed, expected)
