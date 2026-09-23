"""Evidence-backed REDCap calculated-field runtime semantics.

The blank arithmetic contracts are documented by the Yale REDCap calculated
field FAQ and the University of Florida REDCap calculated-fields guide.  The
``datediff`` contracts are documented by Yale's date-interval FAQ.  The local
February 2026 recalculation export is covered separately by the opt-in
``blank_result_as_zero`` compatibility test because that export's blank and
three-argument ``datediff`` behavior conflicts with the published contracts.
"""

from __future__ import annotations

import pytest

from process_variables import transform_calculated_fields as calculated


def _evaluate(
        source: str, row: dict[str, object] | None = None, *,
        blank_result_as_zero: bool = False):
    converted, _ = calculated.translate_redcap_calculation(source)
    result = calculated.evaluate_converted_calculation(
        converted,
        {} if row is None else row,
        blank_result_as_zero=blank_result_as_zero,
    )
    assert result.status == "ok", result.error
    return result.value


@pytest.mark.parametrize(
    "source",
    [
        "+[missing]",
        "-[missing]",
        "[missing] + 2",
        "2 - [missing]",
        "[missing] * 2",
        "2 / [missing]",
        "[missing] ^ 2",
        "2 ^ [missing]",
    ],
)
def test_direct_arithmetic_propagates_blank(source: str) -> None:
    assert _evaluate(source, {"missing": ""}) == ""


@pytest.mark.parametrize(
    ("source", "row", "expected"),
    [
        ("sum([a], [b])", {"a": "", "b": "2.5"}, 2.5),
        ("sum([a], [b])", {"a": "", "b": ""}, 0),
        ("sum([a], [b], 3)", {"a": "1", "b": ""}, 4),
    ],
)
def test_sum_ignores_blank_arguments(
        source: str, row: dict[str, str], expected: int | float) -> None:
    assert _evaluate(source, row) == expected


def test_datediff_default_is_documented_unsigned_absolute_value() -> None:
    assert _evaluate(
        "datediff('2024-01-03', '2024-01-01', 'd')") == 2


@pytest.mark.parametrize("signed", ["true", "'true'"])
def test_datediff_fourth_boolean_argument_selects_signed_result(
        signed: str) -> None:
    assert _evaluate(
        f"datediff('2024-01-03', '2024-01-01', 'd', {signed})"
    ) == -2


def test_datediff_fourth_false_argument_remains_unsigned() -> None:
    assert _evaluate(
        "datediff('2024-01-03', '2024-01-01', 'd', false)") == 2


def test_datediff_five_argument_legacy_signature_remains_supported() -> None:
    assert _evaluate(
        "datediff('03/01/2024', '01/01/2024', 'd', 'dmy', true)"
    ) == -2


def test_blank_persistence_compatibility_is_explicit_and_narrow() -> None:
    source = "if([answer] = '', '', 5)"
    assert _evaluate(source, {"answer": ""}) == ""
    assert _evaluate(
        source, {"answer": ""}, blank_result_as_zero=True) == 0
    assert _evaluate(
        source, {"answer": "1"}, blank_result_as_zero=True) == 5


def test_blank_persistence_mode_does_not_rewrite_nan_sentinel() -> None:
    source = "if([answer] = '', 'NaN', 5)"
    assert _evaluate(
        source, {"answer": ""}, blank_result_as_zero=True) == "NaN"

