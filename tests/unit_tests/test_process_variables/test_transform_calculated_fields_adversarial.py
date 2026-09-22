"""Adversarial regression tests for the REDCap calculation translator.

These cases cover syntax and runtime behavior that is easy to miss when the
current study dictionary does not happen to exercise it.  The expectations are
based on lossless numeric translation, REDCap's documented field notation, and
the documented character-position behavior of the text helpers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from process_variables import transform_calculated_fields as calculated


def _evaluate(source: str, curr_row=None, event_data=None):
    converted, _ = calculated.translate_redcap_calculation(source)
    result = calculated.evaluate_converted_calculation(
        converted, {} if curr_row is None else curr_row, event_data)
    assert result.status == "ok", result.error
    return result.value


def _dictionary(*rows: tuple[str, str, str]) -> pd.DataFrame:
    """Build a minimal dictionary from variable/type/calculation triples."""
    return pd.DataFrame(
        [
            {
                calculated.VARIABLE_COLUMN: variable,
                calculated.FORM_COLUMN: "test_form",
                calculated.FIELD_TYPE_COLUMN: field_type,
                calculated.CALCULATION_COLUMN: expression,
            }
            for variable, field_type, expression in rows
        ]
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("9007199254740993.0 = 9007199254740992.0", 0),
        (
            "12345678901234567890.1 - 12345678901234567890",
            0.1,
        ),
    ],
)
def test_decimal_literals_are_not_rounded_through_binary_float(
        source: str, expected) -> None:
    assert _evaluate(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("left(' abc', 2)", " a"),
        ("right('abc ', 2)", "c "),
        ("mid(' abc', 1, 2)", " a"),
    ],
)
def test_character_functions_preserve_significant_outer_whitespace(
        source: str, expected: str) -> None:
    assert _evaluate(source) == expected


def test_power_rejects_pathological_static_exponent_without_large_result(
        ) -> None:
    """A static guard may reject translation or safely yield REDCap blank."""
    try:
        converted, _ = calculated.translate_redcap_calculation("2 ^ 999999")
    except calculated.UnsupportedCalculationError:
        return

    result = calculated.evaluate_converted_calculation(converted, {})
    assert result.status == "ok", result.error
    assert result.value == ""


def test_power_rejects_pathological_dynamic_exponent_at_runtime() -> None:
    converted, _ = calculated.translate_redcap_calculation("[base] ^ [power]")
    result = calculated.evaluate_converted_calculation(
        converted, {"base": 2, "power": 999999})

    assert result.status == "ok", result.error
    assert result.value == ""
    assert _evaluate("[base] ^ [power]", {"base": 2, "power": 10}) == 1024


@pytest.mark.parametrize(
    "expression",
    [
        "curr_row",
        "event_data",
        "redcap_sum",
        "redcap_sum(1) if curr_row else event_data",
        "(curr_row and event_data)",
    ],
)
def test_generated_python_validator_rejects_noncompiler_shapes(
        expression: str) -> None:
    with pytest.raises(calculated.CalculationTranslationError):
        calculated.validate_generated_python(expression)


def test_repeating_instance_references_parse_and_resolve() -> None:
    """Support current-event, cross-event, and checkbox instance notation."""
    source = (
        "[score][2] + "
        "[month_6_arm_1][score][2] + "
        "[month_6_arm_1][symptom(3)][5]"
    )
    converted, node = calculated.translate_redcap_calculation(source)
    references = list(calculated.iter_field_references(node))

    assert [reference.variable for reference in references] == [
        "score", "score", "symptom"]
    assert [reference.event for reference in references] == [
        None, "month_6_arm_1", "month_6_arm_1"]
    assert [reference.choice for reference in references] == [None, None, "3"]
    assert [reference.instance for reference in references] == [2, 2, 5]

    result = calculated.evaluate_converted_calculation(
        converted,
        {},
        {
            ("score", 2): 1,
            ("month_6_arm_1", "score", 2): 2,
            ("month_6_arm_1", "symptom___3", 5): 3,
        },
    )
    assert result.status == "ok", result.error
    assert result.value == 6


def test_reference_metadata_preserves_repeat_instance() -> None:
    frame = _dictionary(
        ("score", "text", ""),
        (
            "repeat_total",
            "calc",
            "[month_6_arm_1][score][2] + [month_6_arm_1][score][3]",
        ),
    )
    entry = calculated.TransformCalculatedFields(frame)()["repeat_total"]

    assert entry["status"] == "converted"
    assert entry["references"] == [
        {
            "variable": "score",
            "event": "month_6_arm_1",
            "choice": None,
            "instance": 2,
            "kind": "field",
            "source_field_type": "text",
            "source_form": "test_form",
            "display": "[month_6_arm_1][score][2]",
        },
        {
            "variable": "score",
            "event": "month_6_arm_1",
            "choice": None,
            "instance": 3,
            "kind": "field",
            "source_field_type": "text",
            "source_form": "test_form",
            "display": "[month_6_arm_1][score][3]",
        },
    ]


def test_line_comments_are_ignored_outside_string_literals() -> None:
    assert _evaluate("1 + 2 // first subtotal\n + 3") == 6
    assert _evaluate("1 + 2 // comment at end of calculation") == 3
    assert _evaluate("left('a//b', 4) = 'a//b'") == 1


def test_checkbox_reference_requires_checkbox_source_and_declared_choice(
        ) -> None:
    frame = _dictionary(
        ("plain", "text", ""),
        ("symptom", "checkbox", "1, Yes | 2, No"),
        ("valid_choice", "calc", "[symptom(2)]"),
        ("choice_on_text", "calc", "[plain(2)]"),
        ("unknown_choice", "calc", "[symptom(999)]"),
    )
    entries = calculated.TransformCalculatedFields(frame)()

    assert entries["valid_choice"]["status"] == "converted"
    for variable in ("choice_on_text", "unknown_choice"):
        assert entries[variable]["status"] == "invalid"
        assert entries[variable]["converted_calculation"] == ""
        assert "checkbox" in entries[variable]["error"].casefold()
