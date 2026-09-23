"""Black-box contracts for the standalone calculated-field mismatch scanner."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

import pandas as pd
import pytest

from discover_errors import calculated_field_mismatches as scanner


DICTIONARY_COLUMNS = {
    "form": "Form Name",
    "type": "Field Type",
    "variable": "Variable / Field Name",
    "calculation": "Choices, Calculations, OR Slider Labels",
}

CORE_MISMATCH_COLUMNS = {
    "subjectid",
    "redcap_event_name",
    "variable",
    "stored_value",
    "recalculated_value",
    "status",
    "error",
}


def _write_dictionary(path: Path, *rows: tuple[str, str, str]) -> Path:
    pd.DataFrame([
        {
            DICTIONARY_COLUMNS["variable"]: variable,
            DICTIONARY_COLUMNS["form"]: "test_form",
            DICTIONARY_COLUMNS["type"]: field_type,
            DICTIONARY_COLUMNS["calculation"]: calculation,
        }
        for variable, field_type, calculation in rows
    ]).to_csv(path, index=False)
    return path


def _write_data(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run(
        dictionary: Path, inputs: list[Path], output: Path,
        diagnostics: Path, *extra: str) -> int:
    arguments = [
        "--data-dictionary", str(dictionary),
        "--output", str(output),
        "--diagnostics", str(diagnostics),
    ]
    for path in inputs:
        arguments.extend(("--input", str(path)))
    arguments.extend(extra)
    return scanner.main(arguments)


def _read_output(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert CORE_MISMATCH_COLUMNS.issubset(frame.columns)
    return frame


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (1, "1.0"),
        (" 01.000 ", Decimal("1")),
        ("1e0", "1.0000000000"),
        ("-0", 0),
        (None, ""),
        (pd.NA, "   "),
        (float("nan"), None),
    ],
)
def test_values_match_normalizes_numeric_representations_and_real_missing(
        left, right) -> None:
    assert scanner.values_match(left, right)
    assert scanner.values_match(right, left)


@pytest.mark.parametrize("literal", ("NaN", "nan", "NAN", " nAn "))
@pytest.mark.parametrize("blank", (None, "", "   ", pd.NA, float("nan")))
def test_literal_nan_matches_blank_for_report_comparison(
        literal, blank) -> None:
    assert scanner.values_match(literal, blank)
    assert scanner.values_match(blank, literal)


@pytest.mark.parametrize(
    "ordinary", ("sNaN", "NaN value", "nanosecond", "Infinity"))
def test_only_exact_trimmed_literal_nan_matches_blank(ordinary) -> None:
    assert not scanner.values_match(ordinary, "")


@pytest.mark.parametrize(
    ("eligibility", "stored", "recalculated", "reason"),
    (
        ("visible", "", "1", ""),
        ("visible", "1", "", ""),
        ("visible", "", "", "both_values_blank"),
        ("visible", " \t", "NaN", "both_values_blank"),
        ("visible", "nAn", "   ", "both_values_blank"),
        ("unknown", "1", "2", "branching_not_visible"),
        ("hidden", "1", "2", "branching_not_visible"),
    ),
)
def test_output_suppression_requires_visible_and_nonblank_value(
        eligibility, stored, recalculated, reason) -> None:
    result = {
        "branching_eligibility": eligibility,
        "stored_value": stored,
        "recalculated_value": recalculated,
    }
    assert scanner._output_suppression_reason(result) == reason


def test_numeric_tolerance_has_an_explicit_inclusive_boundary() -> None:
    tolerance = Decimal("1e-10")
    assert scanner.values_match(
        "1", "1.0000000001", numeric_tolerance=tolerance)
    assert not scanner.values_match(
        "1", "1.00000000010000000001", numeric_tolerance=tolerance)
    with pytest.raises(ValueError, match="tolerance"):
        scanner.values_match("1", "1", numeric_tolerance=Decimal("-1"))


def test_cli_flags_only_material_numeric_mismatches(tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        ("calc_score", "calc", "[source_score] + 1"),
    )
    data = _write_data(tmp_path / "input.csv", [
        {
            "subjectid": "MATCH_FORMAT",
            "redcap_event_name": "baseline_arm_1",
            "source_score": "0",
            "calc_score": "01.000",
        },
        {
            "subjectid": "MATCH_TOLERANCE",
            "redcap_event_name": "baseline_arm_1",
            "source_score": "0",
            "calc_score": "1.0000000001",
        },
        {
            "subjectid": "MISMATCH",
            "redcap_event_name": "baseline_arm_1",
            "source_score": "0",
            "calc_score": "1.000000001",
        },
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [data], output, diagnostics) == 0
    result = _read_output(output)
    assert result["subjectid"].tolist() == ["MISMATCH"]
    assert result.loc[0, "variable"] == "calc_score"
    assert result.loc[0, "status"] == "mismatch"


def test_blank_root_persistence_mode_is_explicit_and_does_not_rewrite_nan(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("answer", "text", ""),
        ("blank_calc", "calc", "if([answer] = '', '', 5)"),
    )
    data = _write_data(tmp_path / "input.csv", [
        {
            "subjectid": "BLANK",
            "redcap_event_name": "baseline_arm_1",
            "answer": "",
            "blank_calc": "",
        },
        {
            "subjectid": "ZERO",
            "redcap_event_name": "baseline_arm_1",
            "answer": "",
            "blank_calc": "0",
        },
        {
            "subjectid": "LITERAL_NAN",
            "redcap_event_name": "baseline_arm_1",
            "answer": "",
            "blank_calc": "NaN",
        },
    ])

    ordinary = tmp_path / "ordinary.csv"
    assert _run(
        dictionary, [data], ordinary, tmp_path / "ordinary.json") == 0
    assert _read_output(ordinary)["subjectid"].tolist() == ["ZERO"]

    compatible = tmp_path / "compatible.csv"
    assert _run(
        dictionary,
        [data],
        compatible,
        tmp_path / "compatible.json",
        "--blank-result-as-zero",
    ) == 0
    assert _read_output(compatible)["subjectid"].tolist() == [
        "BLANK", "LITERAL_NAN"]


def test_exact_duplicate_rows_collapse_to_one_mismatch(tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        ("calc_score", "calc", "[source_score] + 1"),
    )
    row = {
        "subjectid": "DUPLICATE",
        "redcap_event_name": "baseline_arm_1",
        "source_score": "1",
        "calc_score": "99",
    }
    data = _write_data(tmp_path / "input.csv", [row, dict(row)])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [data], output, diagnostics) == 0
    result = _read_output(output)
    assert len(result) == 1
    assert result.loc[0, "subjectid"] == "DUPLICATE"


def test_conflicting_duplicate_event_rows_are_quarantined_order_independently(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        (
            "cross_calc",
            "calc",
            "[baseline_arm_1][source_score] + 1",
        ),
    )
    rows = [
        {
            "subjectid": "CONFLICT",
            "redcap_event_name": "baseline_arm_1",
            "source_score": value,
            "cross_calc": "",
        }
        for value in ("1", "2")
    ] + [{
        "subjectid": "CONFLICT",
        "redcap_event_name": "month_1_arm_1",
        "source_score": "",
        "cross_calc": "2",
    }]
    data = tmp_path / "input.csv"

    artifacts: list[tuple[bytes, str]] = []
    for index, ordered_rows in enumerate((rows, list(reversed(rows)))):
        _write_data(data, ordered_rows)
        output = tmp_path / f"mismatches_{index}.csv"
        diagnostics = tmp_path / f"diagnostics_{index}.json"
        assert _run(dictionary, [data], output, diagnostics) == 0
        assert _read_output(output).empty
        artifacts.append((output.read_bytes(), diagnostics.read_text(
            encoding="utf-8")))

    assert artifacts[0] == artifacts[1]
    assert "duplicate" in artifacts[0][1].casefold()
    assert "CONFLICT" in artifacts[0][1]


def test_repeat_columns_are_part_of_row_identity(tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        ("calc_score", "calc", "[source_score] + 1"),
    )
    data = _write_data(tmp_path / "input.csv", [
        {
            "subjectid": "REPEAT",
            "redcap_event_name": "baseline_arm_1",
            "redcap_repeat_instrument": "test_form",
            "redcap_repeat_instance": instance,
            "source_score": source,
            "calc_score": calculated_value,
        }
        for instance, source, calculated_value in (
            (1, 1, 2),
            (2, 2, 3),
        )
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [data], output, diagnostics) == 0
    assert _read_output(output).empty
    assert "duplicate" not in diagnostics.read_text(
        encoding="utf-8").casefold()


def test_missing_dependency_is_diagnosed_once_not_fabricated_as_blank(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("required_input", "text", ""),
        ("calc_score", "calc", "[required_input] + 1"),
    )
    data = _write_data(tmp_path / "input.csv", [{
        "subjectid": "MISSING_COLUMN",
        "redcap_event_name": "baseline_arm_1",
        "calc_score": "1",
    }])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [data], output, diagnostics) == 0
    assert _read_output(output).empty
    text = diagnostics.read_text(encoding="utf-8")
    assert "required_input" in text
    assert "missing" in text.casefold()


def test_unsupported_formula_is_a_calculation_diagnostic_not_row_flags(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        ("bad_calc", "calc", "mystery([source_score])"),
    )
    data = _write_data(tmp_path / "input.csv", [{
        "subjectid": "ONE",
        "redcap_event_name": "baseline_arm_1",
        "source_score": "1",
        "bad_calc": "2",
    }])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [data], output, diagnostics) == 0
    assert _read_output(output).empty
    text = diagnostics.read_text(encoding="utf-8")
    assert "bad_calc" in text
    assert "unsupported" in text.casefold()


def test_output_is_byte_deterministic_across_input_and_row_order(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        ("calc_score", "calc", "[source_score] + 1"),
    )
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    _write_data(first, [{
        "subjectid": "Z_SUBJECT",
        "redcap_event_name": "month_1_arm_1",
        "source_score": "1",
        "calc_score": "90",
    }])
    _write_data(second, [
        {
            "subjectid": subject,
            "redcap_event_name": "baseline_arm_1",
            "source_score": "1",
            "calc_score": stored,
        }
        for subject, stored in (("B_SUBJECT", "80"), ("A_SUBJECT", "70"))
    ])

    outputs: list[bytes] = []
    for index, paths in enumerate(((first, second), (second, first))):
        output = tmp_path / f"out_{index}.csv"
        assert _run(
            dictionary,
            list(paths),
            output,
            tmp_path / f"diag_{index}.json",
        ) == 0
        outputs.append(output.read_bytes())

    assert outputs[0] == outputs[1]


def test_input_directory_discovery_is_narrow_and_excludes_own_outputs(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        ("calc_score", "calc", "[source_score] + 1"),
    )
    input_dir = tmp_path / "combined"
    input_dir.mkdir()
    _write_data(
        input_dir / "AMPSCZ-combined-redcap_baseline_ProNET-day1to1.csv",
        [{
            "subjectid": "DISCOVERED",
            "redcap_event_name": "baseline_arm_1",
            "source_score": "1",
            "calc_score": "9",
        }],
    )
    # These CSVs must not become scanner inputs on the first or resumed run.
    (input_dir / "current_data_dictionary.csv").write_text(
        "not,a,data,file\n", encoding="utf-8")
    output = input_dir / "calculated_field_mismatches.csv"
    diagnostics = input_dir / "calculated_field_mismatches_diagnostics.json"

    arguments = [
        "--data-dictionary", str(dictionary),
        "--input-dir", str(input_dir),
        "--allow-partial-input-dir",
        "--output", str(output),
        "--diagnostics", str(diagnostics),
    ]
    assert scanner.main(arguments) == 0
    first = output.read_bytes()
    assert _read_output(output)["subjectid"].tolist() == ["DISCOVERED"]
    assert scanner.main(arguments) == 0
    assert output.read_bytes() == first


def test_missing_explicit_input_and_empty_discovery_fail_helpfully(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        ("calc_score", "calc", "[source_score] + 1"),
    )
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"
    missing = tmp_path / "does_not_exist.csv"
    assert _run(dictionary, [missing], output, diagnostics) == 2
    assert "does_not_exist" in capsys.readouterr().err

    empty = tmp_path / "empty"
    empty.mkdir()
    assert scanner.main([
        "--data-dictionary", str(dictionary),
        "--input-dir", str(empty),
        "--output", str(output),
        "--diagnostics", str(diagnostics),
    ]) == 2
    assert "input" in capsys.readouterr().err.casefold()


def test_missing_subject_identity_fails_without_partial_output(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        ("calc_score", "calc", "[source_score] + 1"),
    )
    data = _write_data(tmp_path / "input.csv", [{
        "redcap_event_name": "baseline_arm_1",
        "source_score": "1",
        "calc_score": "2",
    }])
    output = tmp_path / "mismatches.csv"

    assert _run(
        dictionary,
        [data],
        output,
        tmp_path / "diagnostics.json",
    ) == 2
    assert "subjectid" in capsys.readouterr().err.casefold()
    assert not output.exists()


def test_subject_arm_propagates_to_sparse_later_timepoint_rows(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        (
            "cross_calc",
            "calc",
            "[baseline_arm_1][source_score] + 1",
        ),
    )
    screening = _write_data(
        tmp_path / "AMPSCZ-combined-redcap_screening_ProNET-day1to1.csv",
        [{"subjectid": "SPARSE_ARM", "chrcrit_part": "1"}],
    )
    baseline = _write_data(
        tmp_path / "AMPSCZ-combined-redcap_baseline_ProNET-day1to1.csv",
        [{"subjectid": "SPARSE_ARM", "source_score": "5"}],
    )
    month_1 = _write_data(
        tmp_path / "AMPSCZ-combined-redcap_month_1_ProNET-day1to1.csv",
        [{"subjectid": "SPARSE_ARM", "cross_calc": "99"}],
    )
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(
        dictionary,
        [screening, baseline, month_1],
        output,
        diagnostics,
    ) == 0
    result = _read_output(output)
    assert result[[
        "subjectid", "redcap_event_name", "variable", "recalculated_value",
    ]].to_dict("records") == [{
        "subjectid": "SPARSE_ARM",
        "redcap_event_name": "month_1_arm_1",
        "variable": "cross_calc",
        "recalculated_value": "6",
    }]


def test_mapped_missing_button_suppresses_calculated_target(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        ("calc_score", "calc", "[source_score] + 1"),
    )
    form_map = tmp_path / "important_form_vars.json"
    form_map.write_text(json.dumps({
        "test_form": {
            "missing_var": "test_form_missing",
            "completion_var": "test_form_complete",
        },
    }), encoding="utf-8")
    data = _write_data(
        tmp_path / "AMPSCZ-combined-redcap_baseline_ProNET-day1to1.csv",
        [{
            "subjectid": "FORM_MISSING",
            "redcap_event_name": "baseline_arm_1",
            "source_score": "1",
            "calc_score": "99",
            "test_form_missing": "1",
            "test_form_complete": "2",
        }],
    )
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(
        dictionary,
        [data],
        output,
        diagnostics,
        "--important-form-vars", str(form_map),
    ) == 0
    assert _read_output(output).empty
    summary = json.loads(diagnostics.read_text(encoding="utf-8"))["summary"]
    assert summary["inactive_target_skipped__explicit_missing_button"] == 1


def test_prescient_rpms_missing_completion_codes_suppress_calculated_targets(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        ("calc_score", "calc", "[source_score] + 1"),
    )
    form_map = tmp_path / "important_form_vars.json"
    form_map.write_text(json.dumps({
        "test_form": {
            "missing_var": "test_form_missing",
            "completion_var": "test_form_complete",
        },
    }), encoding="utf-8")
    data = _write_data(
        tmp_path / "AMPSCZ-combined-redcap_baseline_PRESCIENT-day1to1.csv",
        [
            {
                "subjectid": f"RPMS_MISSING_{code}",
                "redcap_event_name": "baseline_arm_1",
                "source_score": "1",
                "calc_score": "99",
                "test_form_missing": "0",
                "test_form_complete_rpms": code,
            }
            for code in ("3", "4")
        ],
    )
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(
        dictionary,
        [data],
        output,
        diagnostics,
        "--important-form-vars", str(form_map),
    ) == 0
    assert _read_output(output).empty
    summary = json.loads(diagnostics.read_text(encoding="utf-8"))["summary"]
    assert summary[
        "inactive_target_skipped__prescient_missing_completion_code"] == 2


def test_repeat_row_only_evaluates_calculations_for_its_instrument(
        tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.csv"
    pd.DataFrame([
        {
            DICTIONARY_COLUMNS["variable"]: "source_a",
            DICTIONARY_COLUMNS["form"]: "form_a",
            DICTIONARY_COLUMNS["type"]: "text",
            DICTIONARY_COLUMNS["calculation"]: "",
        },
        {
            DICTIONARY_COLUMNS["variable"]: "calc_a",
            DICTIONARY_COLUMNS["form"]: "form_a",
            DICTIONARY_COLUMNS["type"]: "calc",
            DICTIONARY_COLUMNS["calculation"]: "[source_a] + 1",
        },
        {
            DICTIONARY_COLUMNS["variable"]: "source_b",
            DICTIONARY_COLUMNS["form"]: "form_b",
            DICTIONARY_COLUMNS["type"]: "text",
            DICTIONARY_COLUMNS["calculation"]: "",
        },
        {
            DICTIONARY_COLUMNS["variable"]: "calc_b",
            DICTIONARY_COLUMNS["form"]: "form_b",
            DICTIONARY_COLUMNS["type"]: "calc",
            DICTIONARY_COLUMNS["calculation"]: "[source_b] + 1",
        },
    ]).to_csv(dictionary, index=False)
    data = _write_data(tmp_path / "input.csv", [{
        "subjectid": "REPEAT_FORM",
        "redcap_event_name": "baseline_arm_1",
        "redcap_repeat_instrument": "form_a",
        "redcap_repeat_instance": "1",
        "source_a": "1",
        "calc_a": "99",
        "source_b": "1",
        "calc_b": "99",
    }])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [data], output, diagnostics) == 0
    result = _read_output(output)
    assert result["variable"].tolist() == ["calc_a"]
    assert result.loc[0, "recalculated_value"] == "2"
    summary = json.loads(diagnostics.read_text(encoding="utf-8"))["summary"]
    assert summary["other_repeat_instrument_targets_skipped"] == 1


def test_explicit_event_reference_resolves_across_separate_canonical_files(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("baseline_source", "text", ""),
        (
            "cross_calc",
            "calc",
            "[baseline_arm_1][baseline_source] + 1",
        ),
    )
    baseline = _write_data(
        tmp_path / (
            "AMPSCZ-combined-redcap_baseline_ProNET-day1to1.csv"),
        [{
            "subjectid": "CROSS_FILE",
            "chrcrit_part": "1",
            "baseline_source": "4",
        }],
    )
    month1 = _write_data(
        tmp_path / (
            "AMPSCZ-combined-redcap_month_1_ProNET-day1to1.csv"),
        [{
            "subjectid": "CROSS_FILE",
            "chrcrit_part": "1",
            "test_form_complete": "2",
            "cross_calc": "5",
        }],
    )
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(
        dictionary, [month1, baseline], output, diagnostics) == 0
    assert _read_output(output).empty
    diagnostic_text = diagnostics.read_text(encoding="utf-8").casefold()
    assert "missing_cross_event" not in diagnostic_text


def test_same_subject_id_never_shares_cross_event_values_between_networks(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("baseline_source", "text", ""),
        (
            "cross_calc",
            "calc",
            "[baseline_arm_1][baseline_source] + 1",
        ),
    )
    paths: list[Path] = []
    for network, baseline_value, calculated_value in (
            ("ProNET", "4", "5"),
            ("PRESCIENT", "9", "10")):
        paths.append(_write_data(
            tmp_path / (
                f"AMPSCZ-combined-redcap_baseline_{network}-day1to1.csv"),
            [{
                "subjectid": "SHARED_ID",
                "chrcrit_part": "1",
                "baseline_source": baseline_value,
            }],
        ))
        paths.append(_write_data(
            tmp_path / (
                f"AMPSCZ-combined-redcap_month_1_{network}-day1to1.csv"),
            [{
                "subjectid": "SHARED_ID",
                "chrcrit_part": "1",
                "test_form_complete": "2",
                "cross_calc": calculated_value,
            }],
        ))
    output = tmp_path / "mismatches.csv"

    assert _run(
        dictionary,
        list(reversed(paths)),
        output,
        tmp_path / "diagnostics.json",
    ) == 0
    assert _read_output(output).empty


def test_direct_and_propagated_modes_have_documented_calc_dependency_semantics(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("raw_score", "text", ""),
        ("first_calc", "calc", "[raw_score] + 1"),
        ("second_calc", "calc", "[first_calc] + 1"),
    )
    data = _write_data(tmp_path / "input.csv", [
        {
            "subjectid": "A_STORED_DOWNSTREAM_CORRECT",
            "redcap_event_name": "baseline_arm_1",
            "test_form_complete": "2",
            "raw_score": "1",
            "first_calc": "99",
            "second_calc": "3",
        },
        {
            "subjectid": "B_SNAPSHOT_DOWNSTREAM_CORRECT",
            "redcap_event_name": "baseline_arm_1",
            "test_form_complete": "2",
            "raw_score": "1",
            "first_calc": "99",
            "second_calc": "100",
        },
    ])

    direct_output = tmp_path / "direct.csv"
    assert _run(
        dictionary,
        [data],
        direct_output,
        tmp_path / "direct.json",
        "--comparison-mode", "direct",
    ) == 0
    direct = _read_output(direct_output)
    assert list(zip(direct["subjectid"], direct["variable"])) == [
        ("A_STORED_DOWNSTREAM_CORRECT", "first_calc"),
        ("A_STORED_DOWNSTREAM_CORRECT", "second_calc"),
        ("B_SNAPSHOT_DOWNSTREAM_CORRECT", "first_calc"),
    ]

    propagated_output = tmp_path / "propagated.csv"
    assert _run(
        dictionary,
        [data],
        propagated_output,
        tmp_path / "propagated.json",
        "--comparison-mode", "propagated",
    ) == 0
    propagated = _read_output(propagated_output)
    assert list(zip(
        propagated["subjectid"], propagated["variable"])) == [
        ("A_STORED_DOWNSTREAM_CORRECT", "first_calc"),
        ("B_SNAPSHOT_DOWNSTREAM_CORRECT", "first_calc"),
        ("B_SNAPSHOT_DOWNSTREAM_CORRECT", "second_calc"),
    ]


def test_blank_target_form_completion_is_inactive_but_complete_form_is_checked(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        ("calc_score", "calc", "[source_score] + 1"),
    )
    data = _write_data(tmp_path / "input.csv", [
        {
            "subjectid": "INACTIVE_FORM",
            "redcap_event_name": "baseline_arm_1",
            "test_form_complete": "",
            "source_score": "1",
            "calc_score": "",
        },
        {
            "subjectid": "ACTIVE_FORM",
            "redcap_event_name": "baseline_arm_1",
            "test_form_complete": "2",
            "source_score": "1",
            "calc_score": "",
        },
        {
            "subjectid": "INACTIVE_WITH_STORED_CALC",
            "redcap_event_name": "baseline_arm_1",
            "test_form_complete": "",
            "source_score": "1",
            "calc_score": "99",
        },
    ])
    output = tmp_path / "mismatches.csv"

    assert _run(
        dictionary,
        [data],
        output,
        tmp_path / "diagnostics.json",
    ) == 0
    result = _read_output(output)
    assert list(zip(result["subjectid"], result["variable"])) == [
        ("ACTIVE_FORM", "calc_score")]


def test_custom_config_owns_relative_dictionary_and_testing_output_defaults(
        tmp_path: Path) -> None:
    """A selected config must not borrow paths from the repository config."""
    project = tmp_path / "custom_project"
    (project / "dependencies" / "data_dictionary").mkdir(parents=True)
    dictionary = _write_dictionary(
        project / "dependencies" / "data_dictionary"
        / "current_data_dictionary.csv",
        ("source_score", "text", ""),
        ("custom_config_calc", "calc", "[source_score] + 1"),
    )
    data = _write_data(project / "input.csv", [{
        "subjectid": "CUSTOM_CONFIG",
        "redcap_event_name": "baseline_arm_1",
        "source_score": "1",
        "custom_config_calc": "99",
    }])
    config = project / "config.json"
    config.write_text(json.dumps({
        "paths": {
            "dependencies_path": "dependencies",
            "combined_csv_path": "combined",
            "output_path": "outputs",
        },
        "testing_enabled": "True",
    }), encoding="utf-8")

    assert scanner.main([
        "--config", str(config),
        "--input", str(data),
    ]) == 0

    output = project / "outputs" / "testing" / (
        "calculated_field_mismatches.csv")
    diagnostics = project / "outputs" / "testing" / (
        "calculated_field_mismatches_diagnostics.json")
    assert output.is_file()
    assert diagnostics.is_file()
    assert _read_output(output)["variable"].tolist() == [
        "custom_config_calc"]
    metadata = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert Path(metadata["source_data_dictionary"]) == dictionary.resolve()
    assert not (project / "outputs" / (
        "calculated_field_mismatches.csv")).exists()


@pytest.mark.parametrize("colliding_option", ["--output", "--diagnostics"])
def test_config_file_cannot_be_used_as_an_output_path(
        tmp_path: Path, capsys: pytest.CaptureFixture[str],
        colliding_option: str) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        ("calc_score", "calc", "[source_score] + 1"),
    )
    data = _write_data(tmp_path / "input.csv", [{
        "subjectid": "CONFIG_COLLISION",
        "redcap_event_name": "baseline_arm_1",
        "source_score": "1",
        "calc_score": "99",
    }])
    config = tmp_path / "config.json"
    original_config = json.dumps({
        "paths": {"output_path": str(tmp_path / "outputs")},
        "testing_enabled": "False",
    })
    config.write_text(original_config, encoding="utf-8")
    ordinary_output = tmp_path / "ordinary.csv"
    ordinary_diagnostics = tmp_path / "ordinary.json"
    arguments = [
        "--config", str(config),
        "--data-dictionary", str(dictionary),
        "--input", str(data),
        "--output", str(ordinary_output),
        "--diagnostics", str(ordinary_diagnostics),
    ]
    option_index = arguments.index(colliding_option) + 1
    arguments[option_index] = str(config)

    assert scanner.main(arguments) == 2
    assert "config" in capsys.readouterr().err.casefold()
    assert config.read_text(encoding="utf-8") == original_config
    assert not ordinary_output.exists()
    assert not ordinary_diagnostics.exists()


def test_form_schedule_skips_unscheduled_event_and_audits_scheduled_event(
        tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.csv"
    pd.DataFrame([
        {
            DICTIONARY_COLUMNS["variable"]: "source_score",
            DICTIONARY_COLUMNS["form"]: "scheduled_form",
            DICTIONARY_COLUMNS["type"]: "text",
            DICTIONARY_COLUMNS["calculation"]: "",
        },
        {
            DICTIONARY_COLUMNS["variable"]: "calc_score",
            DICTIONARY_COLUMNS["form"]: "scheduled_form",
            DICTIONARY_COLUMNS["type"]: "calc",
            DICTIONARY_COLUMNS["calculation"]: "[source_score] + 1",
        },
    ]).to_csv(dictionary, index=False)
    schedule = tmp_path / "forms_per_timepoint.json"
    schedule.write_text(json.dumps({
        "chr": {
            "baseline": ["another_form"],
            "month1": ["scheduled_form"],
        },
        "hc": {},
    }), encoding="utf-8")
    data = _write_data(tmp_path / "input.csv", [
        {
            "subjectid": "SCHEDULEDNESS",
            "redcap_event_name": "baseline_arm_1",
            "source_score": "1",
            "calc_score": "99",
        },
        {
            "subjectid": "SCHEDULEDNESS",
            "redcap_event_name": "month_1_arm_1",
            "source_score": "1",
            "calc_score": "99",
        },
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(
        dictionary,
        [data],
        output,
        diagnostics,
        "--forms-per-timepoint", str(schedule),
    ) == 0
    result = _read_output(output)
    assert result[[
        "subjectid", "redcap_event_name", "variable", "recalculated_value",
    ]].to_dict("records") == [{
        "subjectid": "SCHEDULEDNESS",
        "redcap_event_name": "month_1_arm_1",
        "variable": "calc_score",
        "recalculated_value": "2",
    }]
    summary = json.loads(diagnostics.read_text(encoding="utf-8"))["summary"]
    assert summary["inactive_target_skipped__unscheduled_form"] == 1


def test_noncalculated_branching_logic_false_suppresses_calculated_target(
        tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.csv"
    pd.DataFrame([
        {
            DICTIONARY_COLUMNS["variable"]: "show_calc",
            DICTIONARY_COLUMNS["form"]: "test_form",
            DICTIONARY_COLUMNS["type"]: "text",
            DICTIONARY_COLUMNS["calculation"]: "",
            "Branching Logic (Show field only if...)": "",
        },
        {
            DICTIONARY_COLUMNS["variable"]: "source_score",
            DICTIONARY_COLUMNS["form"]: "test_form",
            DICTIONARY_COLUMNS["type"]: "text",
            DICTIONARY_COLUMNS["calculation"]: "",
            "Branching Logic (Show field only if...)": "",
        },
        {
            DICTIONARY_COLUMNS["variable"]: "calc_score",
            DICTIONARY_COLUMNS["form"]: "test_form",
            DICTIONARY_COLUMNS["type"]: "calc",
            DICTIONARY_COLUMNS["calculation"]: "[source_score] + 1",
            "Branching Logic (Show field only if...)": "[show_calc] = 1",
        },
    ]).to_csv(dictionary, index=False)
    data = _write_data(tmp_path / "input.csv", [{
        "subjectid": "BRANCH_FALSE",
        "redcap_event_name": "baseline_arm_1",
        "test_form_complete": "2",
        "show_calc": "0",
        "source_score": "1",
        "calc_score": "99",
    }])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [data], output, diagnostics) == 0
    assert _read_output(output).empty
    metadata = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert metadata["safe_branching_rule_count"] == 1
    assert metadata["summary"][
        "inactive_target_skipped__branching_logic_false"] == 1


def test_calc_dependent_branching_logic_is_suppressed_from_output(
        tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.csv"
    branching_column = "Branching Logic (Show field only if...)"
    pd.DataFrame([
        {
            DICTIONARY_COLUMNS["variable"]: "source_score",
            DICTIONARY_COLUMNS["form"]: "test_form",
            DICTIONARY_COLUMNS["type"]: "text",
            DICTIONARY_COLUMNS["calculation"]: "",
            branching_column: "",
        },
        {
            DICTIONARY_COLUMNS["variable"]: "gate_calc",
            DICTIONARY_COLUMNS["form"]: "test_form",
            DICTIONARY_COLUMNS["type"]: "calc",
            DICTIONARY_COLUMNS["calculation"]: "[source_score]",
            branching_column: "",
        },
        {
            DICTIONARY_COLUMNS["variable"]: "self_calc",
            DICTIONARY_COLUMNS["form"]: "test_form",
            DICTIONARY_COLUMNS["type"]: "calc",
            DICTIONARY_COLUMNS["calculation"]: "[source_score] + 1",
            branching_column: "[self_calc] = 0",
        },
        {
            DICTIONARY_COLUMNS["variable"]: "dependent_calc",
            DICTIONARY_COLUMNS["form"]: "test_form",
            DICTIONARY_COLUMNS["type"]: "calc",
            DICTIONARY_COLUMNS["calculation"]: "[source_score] + 2",
            branching_column: "[gate_calc] = 0",
        },
    ]).to_csv(dictionary, index=False)
    data = _write_data(tmp_path / "input.csv", [{
        "subjectid": "CALC_BRANCH",
        "redcap_event_name": "baseline_arm_1",
        "test_form_complete": "2",
        "source_score": "1",
        "gate_calc": "1",
        "self_calc": "99",
        "dependent_calc": "99",
    }])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [data], output, diagnostics) == 0
    assert _read_output(output).empty
    metadata = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert metadata["safe_branching_rule_count"] == 0
    assert metadata["branching_rule_diagnostics"][
        "calculated_dependency_ignored"] == ["dependent_calc", "self_calc"]
    summary = metadata["summary"]
    assert summary["evaluated"] == 3
    assert summary["matches"] == 1
    assert summary["mismatches"] == 2
    assert summary["output_row_count"] == 0
    assert summary["output_suppressed__branching_not_visible"] == 2


def test_mixed_arm_history_is_accepted_and_explicit_events_remain_authoritative(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("baseline_source", "text", ""),
        (
            "cross_calc",
            "calc",
            "[baseline_arm_1][baseline_source] + 1",
        ),
    )
    baseline = _write_data(
        tmp_path / "AMPSCZ-combined-redcap_baseline_PRESCIENT-day1to1.csv",
        [{
            "subjectid": "ARM_TRANSITION",
            "redcap_event_name": "baseline_arm_1",
            "baseline_source": "5",
        }],
    )
    month_1 = _write_data(
        tmp_path / "AMPSCZ-combined-redcap_month_1_PRESCIENT-day1to1.csv",
        [{
            "subjectid": "ARM_TRANSITION",
            "redcap_event_name": "month_1_arm_2",
            "cross_calc": "99",
        }],
    )
    output = tmp_path / "mismatches.csv"

    assert _run(
        dictionary,
        [baseline, month_1],
        output,
        tmp_path / "diagnostics.json",
    ) == 0
    result = _read_output(output)
    assert result[[
        "subjectid", "redcap_event_name", "variable", "recalculated_value",
    ]].to_dict("records") == [{
        "subjectid": "ARM_TRANSITION",
        "redcap_event_name": "month_1_arm_2",
        "variable": "cross_calc",
        "recalculated_value": "6",
    }]


def test_extreme_decimal_exponents_fail_closed_without_crashing(
        tmp_path: Path) -> None:
    """Decimal overflows are mismatches, not fatal evaluation errors."""
    assert scanner.values_match("1e999999999", "1e999999999")
    assert not scanner.values_match("1e999999999", "2e999999999")

    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("constant_calc", "calc", "1"),
    )
    data = _write_data(tmp_path / "input.csv", [{
        "subjectid": "EXTREME_EXPONENT",
        "redcap_event_name": "baseline_arm_1",
        "test_form_complete": "2",
        "constant_calc": "1e999999999",
    }])
    output = tmp_path / "mismatches.csv"

    assert _run(
        dictionary,
        [data],
        output,
        tmp_path / "diagnostics.json",
    ) == 0
    result = _read_output(output)
    assert list(zip(result["subjectid"], result["variable"])) == [
        ("EXTREME_EXPONENT", "constant_calc")]
    assert result.loc[0, "numeric_difference"] == "unrepresentable"


@pytest.mark.parametrize(
    "bad_repeat_instance",
    ["NaN", "Infinity", "0", "-1", "1.5", "not-a-number"],
)
def test_invalid_repeat_instance_fails_cleanly_without_outputs(
        tmp_path: Path, capsys: pytest.CaptureFixture[str],
        bad_repeat_instance: str) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        ("calc_score", "calc", "[source_score] + 1"),
    )
    data = _write_data(tmp_path / "input.csv", [{
        "subjectid": "BAD_REPEAT",
        "redcap_event_name": "baseline_arm_1",
        "redcap_repeat_instrument": "test_form",
        "redcap_repeat_instance": bad_repeat_instance,
        "source_score": "1",
        "calc_score": "2",
    }])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [data], output, diagnostics) == 2
    error = capsys.readouterr().err.casefold()
    assert "repeat instance" in error
    assert bad_repeat_instance.casefold() in error
    assert not output.exists()
    assert not diagnostics.exists()


def test_directory_output_collision_cannot_overwrite_canonical_input(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        ("calc_score", "calc", "[source_score] + 1"),
    )
    input_dir = tmp_path / "combined"
    input_dir.mkdir()
    canonical_input = _write_data(
        input_dir / "AMPSCZ-combined-redcap_baseline_ProNET-day1to1.csv",
        [{
            "subjectid": "COLLISION",
            "redcap_event_name": "baseline_arm_1",
            "source_score": "1",
            "calc_score": "2",
        }],
    )
    original_input = canonical_input.read_bytes()
    diagnostics = tmp_path / "diagnostics.json"

    assert scanner.main([
        "--data-dictionary", str(dictionary),
        "--input-dir", str(input_dir),
        "--allow-partial-input-dir",
        "--output", str(canonical_input),
        "--diagnostics", str(diagnostics),
    ]) == 2
    assert "collid" in capsys.readouterr().err.casefold()
    assert canonical_input.read_bytes() == original_input
    assert not diagnostics.exists()


def test_incomplete_canonical_directory_requires_explicit_opt_in(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dictionary = _write_dictionary(
        tmp_path / "dictionary.csv",
        ("source_score", "text", ""),
        ("calc_score", "calc", "[source_score] + 1"),
    )
    input_dir = tmp_path / "combined"
    input_dir.mkdir()
    _write_data(
        input_dir / "AMPSCZ-combined-redcap_baseline_ProNET-day1to1.csv",
        [{
            "subjectid": "PARTIAL_DIRECTORY",
            "redcap_event_name": "baseline_arm_1",
            "source_score": "1",
            "calc_score": "2",
        }],
    )
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    base_arguments = [
        "--data-dictionary", str(dictionary),
        "--input-dir", str(input_dir),
        "--output", str(output),
        "--diagnostics", str(diagnostics),
    ]
    assert scanner.main(base_arguments) == 2
    error = capsys.readouterr().err.casefold()
    assert "incomplete" in error
    assert "--allow-partial-input-dir" in error
    assert not output.exists()
    assert not diagnostics.exists()

    assert scanner.main(
        base_arguments + ["--allow-partial-input-dir"]) == 0
    assert output.is_file()
    assert diagnostics.is_file()
    assert _read_output(output).empty
