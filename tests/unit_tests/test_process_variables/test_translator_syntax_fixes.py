"""Regression tests for the 2026-07-19 branching-logic translator fixes.

Covers the syntax/order-of-operations defects from
docs/TRANSLATOR_SYNTAX_REVIEW_2026-07-19.md: constant-False ordered
variable comparisons (BLPREC-1), unparenthesized top-level ``or`` in the
equality expansion regrouping under source-level ``and`` (O1), crashing
``<>`` states (BLPREC-2), absent-vs-blank equality (BLPREC-3), mangled
quoted date literals (K1), unguarded non-numeric literals (BLPREC-5),
cross-event references (BLX-1), the pharm edit's constant-False root
gates (BLX-4/O3), mixed-case boolean operators (BLSEM-4), and the
chrpsychs_scr_e24 division-by-zero manual rule (BLPREC-6).

Every generated expression is evaluated through the shared runtime
validator, so these tests also pin that the templates stay inside the
production evaluator's allowlist.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from process_variables.transform_branching_logic import TransformBranchingLogic
from utils.branching_logic_eval import evaluate_branching_logic


VARIABLE = "Variable / Field Name"
BRANCHING = "Branching Logic (Show field only if...)"
FIELD_TYPE = "Field Type"


class _TestUtils:
    absolute_path = str(Path(__file__).resolve().parents[3])
    missing_code_list = [-3, -9, -99, 999, "-3", "-9", "-99", "999"]

    @staticmethod
    def can_be_float(value: object) -> bool:
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        return True


def _transformer(rows: list[dict] | None = None) -> TransformBranchingLogic:
    dictionary = pd.DataFrame(rows or [{
        VARIABLE: "placeholder",
        BRANCHING: "",
        FIELD_TYPE: "text",
    }])
    with patch(
            "process_variables.transform_branching_logic.Utils", _TestUtils):
        return TransformBranchingLogic(dictionary)


def _evaluate(
        transform: TransformBranchingLogic, redcap_expression: str,
        **values: object) -> bool:
    converted = transform.branching_logic_redcap_to_python(redcap_expression)
    return evaluate_branching_logic(
        converted,
        curr_row=SimpleNamespace(**values),
        instance=transform,
    )


@pytest.fixture(name="transform")
def _transform_fixture() -> TransformBranchingLogic:
    return _transformer()


# --- BLPREC-1 / K5: ordered variable-to-variable comparisons -----------------

@pytest.mark.parametrize(
    ("expression", "values", "expected"),
    [
        ("[intdate] > [entdate]",
         {"intdate": "2023-05-01", "entdate": "2023-01-01"}, True),
        ("[intdate] > [entdate]",
         {"intdate": "2023-01-01", "entdate": "2023-05-01"}, False),
        # numeric values must keep numeric ordering ('10' < '9' as strings)
        ("[a] > [b]", {"a": "10", "b": "9"}, True),
        # blank or absent operands compare False in both directions
        ("[intdate] > [entdate]",
         {"intdate": "2023-05-01", "entdate": ""}, False),
        ("[intdate] > [entdate]", {"intdate": "2023-05-01"}, False),
        # whitespace around the operator must not corrupt the hasattr guard
        ("[a] >= [b]", {"a": "2023-05-01", "b": "2023-01-01"}, True),
    ],
)
def test_ordered_var_comparisons_string_fallback_and_guards(
        transform: TransformBranchingLogic, expression: str,
        values: dict, expected: bool) -> None:
    assert _evaluate(transform, expression, **values) is expected


# --- O1 / BLPREC-3: equality expansion grouping and blank handling -----------

@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"a": "2", "b": "2", "c": "1"}, True),
        # a==b but c wrong: the unparenthesized "or" used to make this True
        ({"a": "2", "b": "2", "c": "0"}, False),
        ({"a": "2", "b": "3", "c": "1"}, False),
        ({"c": "1"}, True),
        ({"c": "0"}, False),
    ],
)
def test_equality_expansion_keeps_source_grouping_under_and(
        transform: TransformBranchingLogic, values: dict,
        expected: bool) -> None:
    assert _evaluate(
        transform, "[a] = [b] and [c] = '1'", **values) is expected


def test_equality_treats_absent_and_blank_as_equal(
        transform: TransformBranchingLogic) -> None:
    assert _evaluate(transform, "[a] = [b]", b="") is True
    assert _evaluate(transform, "[a] = [b]", a="1", b="1.0") is True


# --- BLPREC-2: not-equal expansion must not crash or misjudge mixed types ----

@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"b": ""}, False),          # absent vs blank: used to AttributeError
        ({}, False),                 # both absent: used to AttributeError
        ({"a": "1", "b": ""}, True),
        ({"a": "1", "b": "abc"}, True),  # mixed types: used to be False
        ({"a": "2", "b": "2.0"}, False),
    ],
)
def test_not_equal_expansion_truth_table(
        transform: TransformBranchingLogic, values: dict,
        expected: bool) -> None:
    assert _evaluate(transform, "[a] <> [b]", **values) is expected


# --- K1 / BLPREC-5: quoted literals -------------------------------------------

def test_quoted_date_literal_survives_as_guarded_string_comparison(
        transform: TransformBranchingLogic) -> None:
    # '1903-03-03' used to be split around its leading digits into
    # invalid Python (617 expressions in the shipped corpus)
    assert _evaluate(transform, "[d] <> '1903-03-03'") is True
    assert _evaluate(
        transform, "[d] <> '1903-03-03'", d="1903-03-03") is False
    assert _evaluate(
        transform, "[d] = '1903-03-03'", d="1903-03-03") is True


def test_non_numeric_literal_comparisons_are_guarded(
        transform: TransformBranchingLogic) -> None:
    # bare curr_row.a == 'A11' used to raise AttributeError when absent
    assert _evaluate(transform, "[a] = 'A11'") is False
    assert _evaluate(transform, "[a] = 'A11'", a="A11") is True
    assert _evaluate(transform, "[a] <> 'NaN'") is True


def test_double_zero_semantics_unchanged(
        transform: TransformBranchingLogic) -> None:
    assert _evaluate(transform, "[a] = '00'", a="00") is True
    assert _evaluate(transform, "[a] = '00'") is False
    assert _evaluate(transform, "[a] <> '00'") is True


def test_quoted_numeric_literals_still_compare_as_floats(
        transform: TransformBranchingLogic) -> None:
    assert _evaluate(transform, "[a] >= '0.9'", a="0.95") is True
    assert _evaluate(transform, "[a] = -1", a="-1.0") is True


# --- BLSEM-3/4: boolean operator casing ---------------------------------------

def test_mixed_case_boolean_operators_convert(
        transform: TransformBranchingLogic) -> None:
    assert _evaluate(
        transform, "[a] = '1' Or [b] = '2'", a="1", b="0") is True


def test_uppercase_fragments_inside_quoted_values_survive(
        transform: TransformBranchingLogic) -> None:
    converted = transform.branching_logic_redcap_to_python(
        "[status] = 'FORMER'")
    assert "'FORMER'" in converted


# --- BLX-1: cross-event references fail closed --------------------------------

def test_cross_event_reference_is_excluded_not_converted() -> None:
    transform = _transformer([
        {VARIABLE: "chrguid_dateerror2",
         BRANCHING: ("[screening_arm_1][chric_consent_date] > "
                     "[chrguid_interview_date]"),
         FIELD_TYPE: "text"},
    ])
    converted = transform.convert_all_branching_logic()
    entry = converted["chrguid_dateerror2"]
    assert entry["converted_branching_logic"] == ""
    assert "chrguid_dateerror2" in transform.excluded_conversions


# --- BLX-4 / O3: pharm edit ----------------------------------------------------

def test_pharm_root_and_medstop_variables_are_not_gated(
        transform: TransformBranchingLogic) -> None:
    for variable in (
            "chrpharm_med", "chrpharm_med_past",
            "chrpharm_medstop_list", "chrpharm_medstop_list_2"):
        assert transform.edit_past_pharm_branch_logic(variable, "") == ""


def test_pharm_medication_fields_keep_gate_without_empty_tuple(
        transform: TransformBranchingLogic) -> None:
    edited = transform.edit_past_pharm_branch_logic("chrpharm_med2_dose", "")
    assert edited == ("([chrpharm_med2_name] <> '999' and"
                      " [chrpharm_med1_add] = '1')")
    assert "()" not in transform.branching_logic_redcap_to_python(edited)

    gated = transform.edit_past_pharm_branch_logic(
        "chrpharm_med2_dose", "[x] = '1'")
    assert gated == ("([chrpharm_med2_name] <> '999' and"
                     " [chrpharm_med1_add] = '1') and ([x] = '1')")


def test_pharm_two_digit_medication_numbers_chain_correctly(
        transform: TransformBranchingLogic) -> None:
    edited = transform.edit_past_pharm_branch_logic(
        "chrpharm_med10_add_past", "[y] = '1'")
    assert "[chrpharm_med10_name_past] <> '999'" in edited
    assert "[chrpharm_med9_add_past] = '1'" in edited


def test_pharm_onset_offset_variables_stay_unedited(
        transform: TransformBranchingLogic) -> None:
    assert transform.edit_past_pharm_branch_logic(
        "chrpharm_med2_onset", "[z] = '1'") == "[z] = '1'"


# --- BLPREC-6: manual chrpsychs_scr_e24 rule -----------------------------------

def test_e24_manual_rule_survives_zero_premorbid_score(
        transform: TransformBranchingLogic) -> None:
    expression = transform.manual_conversions["chrpsychs_scr_e24"]
    row = SimpleNamespace(
        chrpsychs_scr_ac1="0", chrpsychs_scr_e4="1", chrpsychs_scr_e21="0",
        chrsofas_currscore="55", chrsofas_premorbid="0")
    # used to raise ZeroDivisionError
    assert evaluate_branching_logic(
        expression, curr_row=row, instance=transform) is False
    row.chrsofas_premorbid = "60"
    assert evaluate_branching_logic(
        expression, curr_row=row, instance=transform) is True
