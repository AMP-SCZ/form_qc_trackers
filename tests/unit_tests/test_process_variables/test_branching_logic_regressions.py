"""Regression tests for audited REDCap branching-logic defects.

The legacy regex/evaluation architecture is intentionally left in place. These
tests constrain the requested rule corrections and the new compile-validation
boundary around its generated Python.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from process_variables.transform_branching_logic import TransformBranchingLogic
from utils.branching_logic_eval import (
    BranchingLogicValidationError,
    evaluate_branching_logic,
)


VARIABLE = "Variable / Field Name"
BRANCHING = "Branching Logic (Show field only if...)"
FIELD_TYPE = "Field Type"


class _TestUtils:
    """Minimal deterministic Utils surface used by transformation rules."""

    absolute_path = str(Path(__file__).resolve().parents[3])
    missing_code_list = [-3, -9, -99, 999, "-3", "-9", "-99", "999"]

    @staticmethod
    def can_be_float(value: object) -> bool:
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def collect_digit(value: str) -> str:
        match = re.search(r"\d+", value)
        return match.group(0) if match else ""


def _transformer() -> TransformBranchingLogic:
    dictionary = pd.DataFrame([{
        VARIABLE: "placeholder",
        BRANCHING: "",
        FIELD_TYPE: "text",
    }])
    with patch(
            "process_variables.transform_branching_logic.Utils", _TestUtils):
        return TransformBranchingLogic(dictionary)


def _legacy_eval(
        transform: TransformBranchingLogic, redcap_expression: str,
        **values: object) -> bool:
    converted = transform.branching_logic_redcap_to_python(redcap_expression)
    row = SimpleNamespace(**values)
    return bool(eval(converted, {  # noqa: S307 - tests the retained legacy path
        "curr_row": row,
        "instance": transform,
    }))


def test_scid_followup_logic_includes_b24_as_an_independent_trigger() -> None:
    transform = _transformer()

    edited = transform.apply_branching_logic_edits(
        "chrscid_b45", "[original_gate] = '1'")

    assert "[chrscid_b24] = 3" in edited
    assert _legacy_eval(
        transform,
        edited,
        original_gate="1",
        chrscid_b24="3",
    )


def test_tbi_fourth_injury_fields_remain_applicable_when_count_is_five() -> None:
    transform = _transformer()

    edited = transform.apply_branching_logic_edits(
        "chrtbi_subject_age4", "[original_gate] = '1'")

    assert "[chrtbi_number_injs] >= '4'" in edited
    assert "[chrtbi_subject_times] = '3'" in edited
    assert _legacy_eval(
        transform,
        edited,
        chrtbi_number_injs="5",
        chrtbi_subject_times="3",
    )


def test_tbi_parent_detail_uses_parent_times_and_at_least_injury_count() -> None:
    transform = _transformer()

    edited = transform.apply_branching_logic_edits(
        "chrtbi_parent_length4", "ignored")

    assert "[chrtbi_number_injs] >= '4'" in edited
    assert "[chrtbi_parent_times] = '3'" in edited


def test_tbi_numeric_suffix_does_not_rewrite_unrelated_error_field() -> None:
    transform = _transformer()
    original = "[chrtbi_interview_date] <> ''"

    edited = transform.apply_branching_logic_edits(
        "chrtbi_dateerror4", original)

    assert edited == original


def test_iq_picture_completion_manual_rules_no_longer_accept_code_five() -> None:
    transform = _transformer()

    for variable in (
        "chriq_pic_completion_raw",
        "chriq_scaled_pic_completion",
    ):
        converted = transform.manual_conversions[variable]
        assert "float(5)" not in converted
        assert bool(eval(converted, {  # noqa: S307 - retained legacy rule
            "curr_row": SimpleNamespace(chriq_assessment="4"),
            "instance": transform,
        }))
        assert not bool(eval(converted, {  # noqa: S307 - retained legacy rule
            "curr_row": SimpleNamespace(chriq_assessment="5"),
            "instance": transform,
        }))


def test_compile_validation_excludes_and_diagnoses_invalid_python() -> None:
    transform = _transformer()
    converted = {
        "valid": {
            "variable": "valid",
            "original_branching_logic": "[source] = 1",
            "converted_branching_logic": (
                "hasattr(curr_row, 'source') and curr_row.source == 1"),
        },
        "invalid": {
            "variable": "invalid",
            "original_branching_logic": "[source] =",
            "converted_branching_logic": "curr_row.source ==",
        },
        "blank": {
            "variable": "blank",
            "original_branching_logic": "",
            "converted_branching_logic": "",
        },
    }

    diagnostics = transform.validate_converted_branching_logic(converted)

    assert [item["variable"] for item in diagnostics] == ["invalid"]
    diagnostic = diagnostics[0]
    assert diagnostic["original_branching_logic"] == "[source] ="
    assert diagnostic["converted_branching_logic"] == "curr_row.source =="
    assert diagnostic["error_type"] in {
        "SyntaxError", "BranchingLogicValidationError"}
    assert diagnostic["error"]
    assert [item["variable"] for item in transform.invalid_conversions] == [
        "invalid"]
    assert transform.excluded_conversions["invalid"] == "[source] ="
    assert "valid" not in transform.excluded_conversions
    assert "blank" not in transform.excluded_conversions


@pytest.mark.parametrize(
    "expression",
    [
        "open('should_not_exist.txt', 'w')",
        "__import__('os')",
        "curr_row.__class__",
    ],
)
def test_shared_branching_evaluator_rejects_unsafe_python(
        expression: str) -> None:
    with pytest.raises(BranchingLogicValidationError):
        evaluate_branching_logic(
            expression,
            curr_row=SimpleNamespace(),
            instance=SimpleNamespace(),
        )


def test_shared_branching_evaluator_accepts_an_ordinary_generated_rule() -> None:
    transform = _transformer()
    expression = transform.branching_logic_redcap_to_python(
        "[gate] = 1 and [other_gate] <> 2")

    assert evaluate_branching_logic(
        expression,
        curr_row=SimpleNamespace(gate="1", other_gate="3"),
        instance=transform,
    )


def test_shared_branching_evaluator_accepts_current_scid_self_getattr_rule(
        ) -> None:
    transform = _transformer()
    project_root = Path(__file__).resolve().parents[3]
    scid_rules = json.loads(
        (project_root.parent / "dependencies" / "scid_diagnosis_vars.json").read_text(
            encoding="utf-8"))
    expression = scid_rules["chrscid_a129"]["extra_conditionals"][0]

    assert evaluate_branching_logic(
        expression,
        curr_row=SimpleNamespace(chrscid_a122="3"),
        instance=transform,
    )
