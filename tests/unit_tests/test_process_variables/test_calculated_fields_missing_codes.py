"""Missing-code contracts for the calculated-field simulation runtime."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from process_variables.transform_calculated_fields import (
    CALCULATION_MISSING_CODES,
    evaluate_converted_calculation,
    is_calculation_missing_code,
    redcap_value,
    translate_redcap_calculation,
)


def _evaluate(source, row=None, event_data=None):
    converted, _ = translate_redcap_calculation(source)
    result = evaluate_converted_calculation(
        converted, {} if row is None else row, event_data)
    assert result.status == "ok", result.error
    return result.value


def test_every_configured_missing_code_is_recognized():
    assert all(
        is_calculation_missing_code(value)
        for value in CALCULATION_MISSING_CODES
    )
    datetime_codes = [
        value for value in CALCULATION_MISSING_CODES
        if isinstance(value, str) and len(value) == 16 and value[10] == " "
    ]
    assert len(datetime_codes) == 3 * 24 * 60


@pytest.mark.parametrize(
    "missing_code",
    (
        -3, "-9.00", "-99.00", "999.00",
        "1901-01-01", "1903-03-03 12:34", "1909-09-09 23:59",
    ),
)
def test_representative_missing_codes_are_blank_on_each_field_lookup(
        missing_code):
    row = {"source": missing_code}

    assert _evaluate("[source] = ''", row) == 1
    assert _evaluate("[source] + 1", row) == ""
    assert _evaluate("sum([source], 2)", row) == 2


@pytest.mark.parametrize("missing_code", (-9, "-99.00", "1901-01-01 08:17", 999.0))
def test_cross_event_and_repeating_lookups_also_blank_missing_codes(
        missing_code):
    nested_event_data = {
        "baseline_arm_1": {"source": missing_code},
    }
    tuple_event_data = {
        ("baseline_arm_1", "source"): missing_code,
    }
    repeating_event_data = {
        "baseline_arm_1": {2: {"source": missing_code}},
    }

    assert _evaluate(
        "[baseline_arm_1][source] = ''", {}, nested_event_data) == 1
    assert _evaluate(
        "[baseline_arm_1][source] = ''", {}, tuple_event_data) == 1
    assert _evaluate(
        "[baseline_arm_1][source][2] = ''",
        {}, repeating_event_data) == 1


@pytest.mark.parametrize(
    "missing_code", (" -3 ", "\t-9.0", "999\n", " 1903-03-03 "))
def test_whitespace_around_a_missing_code_remains_data(missing_code):
    assert _evaluate("[source] = ''", {"source": missing_code}) == 0


def test_series_and_direct_runtime_lookups_use_the_same_normalizer():
    row = pd.Series({"source": "999.0"})

    assert redcap_value(row, "source") == ""
    assert _evaluate("[source] = ''", row) == 1


def test_missing_date_code_cannot_participate_in_datediff():
    assert _evaluate(
        "datediff([source], '2025-01-02', 'd')",
        {"source": "1909-09-09"}) == ""


@pytest.mark.parametrize(
    "missing_date",
    (
        date(1901, 1, 1),
        "1901-01-01 00:00",
        "1903-03-03 12:30",
        "1909-09-09 23:59",
    ),
)
def test_exact_date_and_minute_sentinels_are_blank(missing_date):
    assert _evaluate("[source] = ''", {"source": missing_date}) == 1


@pytest.mark.parametrize(
    "ordinary_value",
    (
        "-90",
        "9999",
        "value -9",
        "1901-01-010",
        datetime(1903, 3, 3, 0, 0),
        pd.Timestamp("1909-09-09"),
        "1901-01-01 00:00:00",
        "1903-03-03T12:30:00Z",
        "1903-03-03 12:30:00",
        "1909-09-09 24:00",
        "1909-09-09 23:60",
    ),
)
def test_only_whole_exact_codes_are_blank(ordinary_value):
    assert _evaluate("[source] = ''", {"source": ordinary_value}) == 0


def test_formula_literals_keep_expression_semantics_before_result_policy():
    assert _evaluate("999 + 1") == 1000
    assert _evaluate("-9 + 1") == -8
    assert _evaluate(
        "if([source] = '', 999, 0) + 1", {"source": -9}) == 1000
    assert _evaluate(
        "if([source] = '', 999, 0)", {"source": -9}) == ""
    assert _evaluate("'1901-01-01'") == ""
    assert _evaluate("'1903-03-03 12:30'") == ""
    assert _evaluate(
        "'1903-03-03 12:30:00'") == "1903-03-03 12:30:00"


def test_boolean_values_are_not_numeric_missing_codes():
    assert _evaluate("[source] = ''", {"source": True}) == 0
    assert _evaluate("[source] = ''", {"source": False}) == 0
