"""Regression tests for the 2026-07-19 calculated-field translator fixes.

Covers docs/TRANSLATOR_SYNTAX_REVIEW_2026-07-19.md items: not() binding
(CP-1), checkbox choice-code export normalization (CP-2), comment
handling (CP-3/CP-4), datediff 'today'/'now' (CE-1) and numeric
returnSigned flags (CR-06), huge-integer results (CR-03/CR-04), the dead
'==' operator (CP-8), and the core order-of-operations contract.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from process_variables.transform_calculated_fields import (
    CalculationTranslationError,
    evaluate_converted_calculation,
    translate_redcap_calculation,
)


def _run(expression: str, row: dict | None = None):
    converted, _ = translate_redcap_calculation(expression)
    return evaluate_converted_calculation(converted, row or {})


# --- CP-1: not() is a one-argument function, not a loose unary operator -------

def test_not_function_does_not_absorb_following_operators() -> None:
    assert _run("not([a]=1) + not([b]=1)", {"a": 2, "b": 2}).value == 2
    assert _run("not([a]) - 1", {"a": 2}).value == -1
    assert _run("not([a]=1) * 2 + not([b]=1)", {"a": 2, "b": 2}).value == 3


def test_bare_not_without_parentheses_fails_closed() -> None:
    with pytest.raises(CalculationTranslationError):
        translate_redcap_calculation("not [a]")


# --- CP-3 / CP-4: comment markers ---------------------------------------------

def test_whitespace_delimited_comments_are_stripped() -> None:
    assert _run("[a] / [b] // note", {"a": 10, "b": 2}).value == 5
    assert _run("# adjusted per protocol\n[a] + 1", {"a": 1}).value == 2
    assert _run("[a] + 1 # note", {"a": 1}).value == 2


def test_mid_expression_double_slash_fails_closed() -> None:
    # "[a]//[b]" used to silently truncate to just "[a]" with status ok
    with pytest.raises(CalculationTranslationError):
        translate_redcap_calculation("[a]//[b]")


# --- CP-2: checkbox choice codes use REDCap export column naming --------------

def test_negative_checkbox_choice_maps_to_export_column() -> None:
    converted, _ = translate_redcap_calculation("[race(-2)] = 1")
    assert "race____2" in converted
    assert evaluate_converted_calculation(
        converted, {"race____2": "1"}).value == 1


def test_letter_checkbox_choice_is_lowercased() -> None:
    converted, _ = translate_redcap_calculation("[f(A)] = 1")
    assert "f___a" in converted


# --- CE-1 / CR-06: datediff dynamic keywords and signed flags ------------------

def test_datediff_today_returns_a_number_not_blank() -> None:
    result = _run("datediff([dob],'today','y')", {"dob": "2000-01-01"})
    assert result.status == "ok"
    years = float(result.value)
    expected = (datetime.now() - datetime(2000, 1, 1)).days / 365.2425
    assert abs(years - expected) < 0.02


def test_datediff_now_returns_a_number_not_blank() -> None:
    result = _run("datediff('now','2020-01-01 00:00','h','ymd h:m')")
    assert result.status == "ok"
    assert float(result.value) > 0


def test_datediff_numeric_return_signed_flags() -> None:
    row = {"a": "2020-02-01", "b": "2020-01-01"}
    assert _run("datediff([a],[b],'d',1)", row).value == -31
    assert _run("datediff([a],[b],'d',0)", row).value == 31
    assert _run("datediff([a],[b],'d',true)", row).value == -31


# --- CR-03 / CR-04: huge integral results stay compact -------------------------

def test_huge_numeric_literal_returns_compact_text() -> None:
    result = _run("9E+200000")
    assert result.status == "ok"
    assert isinstance(result.value, str)
    # small enough for any consumer's str()/repr under the Python 3.11+
    # 4300-digit integer-string limit
    assert len(result.value) < 30


def test_ordinary_integral_results_remain_ints() -> None:
    assert _run("2 + 3").value == 5
    assert isinstance(_run("2 + 3").value, int)
    assert _run("(10)^(50)").value == 10 ** 50


# --- CP-8: '==' tokenizes -------------------------------------------------------

def test_double_equals_is_accepted() -> None:
    assert _run("[a] == 1", {"a": 1}).value == 1


# --- Order-of-operations contract (pins the verified precedence core) ----------

@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2+3*4", 14),
        ("(2+3)*4", 20),
        ("10-3-2", 5),
        ("100/10/5", 2),
        ("2^3^2", 512),   # right-associative
        ("-2^2", -4),     # exponent binds tighter than the leading sign
        ("2*-3", -6),
        ("6/2*3", 9),
        ("2+3*4^2", 50),
        ("if(1=1 or 1=2 and 2=3, 1, 0)", 1),  # and binds tighter than or
    ],
)
def test_order_of_operations(expression: str, expected: int) -> None:
    assert _run(expression).value == expected
