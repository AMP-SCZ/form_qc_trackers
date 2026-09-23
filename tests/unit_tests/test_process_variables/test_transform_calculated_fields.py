"""Focused contract tests for the REDCap calculated-field translator."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from process_variables import transform_calculated_fields as calculated


def _dictionary(*rows: dict[str, str]) -> pd.DataFrame:
    """Build the smallest data dictionary that satisfies the public schema."""
    normalized = []
    for row in rows:
        normalized.append({
            calculated.VARIABLE_COLUMN: row["variable"],
            calculated.FORM_COLUMN: row.get("form", "test_form"),
            calculated.FIELD_TYPE_COLUMN: row.get("field_type", "text"),
            calculated.CALCULATION_COLUMN: row.get("calculation", ""),
            calculated.BRANCHING_LOGIC_COLUMN: row.get(
                "branching_logic", ""),
            calculated.IDENTIFIER_COLUMN: row.get("identifier", ""),
        })
    return pd.DataFrame(normalized)


def _row(variable: str, *, field_type: str = "text",
         calculation: str = "", form: str = "test_form",
         identifier: str = "") -> dict[str, str]:
    return {
        "variable": variable,
        "field_type": field_type,
        "calculation": calculation,
        "form": form,
        "identifier": identifier,
    }


def _evaluate(source: str, curr_row=None, event_data=None):
    converted, _ = calculated.translate_redcap_calculation(source)
    result = calculated.evaluate_converted_calculation(
        converted, {} if curr_row is None else curr_row, event_data)
    assert result.status == "ok", result.error
    return result.value


def test_iso_date_literal_remains_one_token_and_evaluates() -> None:
    source = "datediff('2024-01-09', '2024-02-10', 'd')"
    tokens = calculated.CalculationTokenizer().tokenize(source)

    date_tokens = [token for token in tokens if token.kind == "STRING"]
    assert [token.value for token in date_tokens] == [
        "2024-01-09", "2024-02-10", "d"]
    assert date_tokens[0].raw == "'2024-01-09'"
    assert _evaluate(source) == 32


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("2 + 3 * 4 ^ 2", 50),
        ("(2 + 3) * 4", 20),
        ("2 ^ 3 ^ 2", 512),
        ("1 + 2 * 3 = 7 and 4 > 3", True),
    ],
)
def test_operator_precedence_and_associativity(source: str, expected) -> None:
    assert _evaluate(source) == expected


def test_nested_if_is_lazy_and_does_not_evaluate_unchosen_division() -> None:
    source = (
        "if([flag] = 1, if([second_flag] = 1, 42, 1 / 0), 1 / 0)"
    )
    assert _evaluate(source, {"flag": 1, "second_flag": 1}) == 42
    assert _evaluate("if([flag] = 1, 1 / 0, 7)", {"flag": 0}) == 7


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("sum(1, '', 2.5, 3)", 6.5),
        ("round(2.675, 2)", 2.68),
        ("datediff('2024-01-01', '2024-01-03', 'd')", 2),
        ("datediff('2024-01-03', '2024-01-01', 'd', 'ymd', true)", -2),
        ("left('abcdef', 2)", "ab"),
        ("mid('abcdef', 2, 3)", "bcd"),
        ("right('abcdef', 2)", "ef"),
    ],
)
def test_supported_function_evaluation(source: str, expected) -> None:
    assert _evaluate(source) == expected


def test_field_checkbox_and_event_references_have_metadata_and_evaluate() -> None:
    frame = _dictionary(
        _row("amount"),
        _row(
            "symptom", field_type="checkbox",
            calculation="1, Absent | 2, Present"),
        _row("score"),
        _row(
            "total",
            field_type="calc",
            calculation=(
                "[amount] + [symptom(2)] + [month_6_arm_1][score]"),
        ),
    )
    entries = calculated.TransformCalculatedFields(frame)()
    total = entries["total"]

    assert total["status"] == "converted"
    assert total["referenced_variables"] == ["amount", "symptom", "score"]
    assert [(item["event"], item["choice"], item["kind"])
            for item in total["references"]] == [
                (None, None, "field"),
                (None, "2", "checkbox"),
                ("month_6_arm_1", None, "field"),
            ]
    result = calculated.evaluate_converted_calculation(
        total["converted_calculation"],
        {"amount": "2", "symptom___2": "3"},
        {"month_6_arm_1": {"score": "4"}},
    )
    assert result.status == "ok"
    assert result.value == 9


def test_unknown_function_field_and_trailing_tokens_get_correct_statuses() -> None:
    frame = _dictionary(
        _row("known"),
        _row("valid", field_type="calc", calculation="[known] + 1"),
        _row("unknown_function", field_type="calc", calculation="mean([known])"),
        _row("unknown_field", field_type="calc", calculation="[absent] + 1"),
        _row("trailing", field_type="calc", calculation="[known] + 1 2"),
    )
    transformer = calculated.TransformCalculatedFields(frame)
    entries = transformer()

    assert entries["valid"]["status"] == "converted"
    assert entries["unknown_function"]["status"] == "unsupported"
    assert entries["unknown_function"]["error_code"] == (
        "unsupported_calculation")
    assert entries["unknown_field"]["status"] == "invalid"
    assert entries["unknown_field"]["error_code"] == "invalid_calculation"
    assert "absent" in entries["unknown_field"]["error"]
    assert entries["trailing"]["status"] == "invalid"
    assert "trailing token" in entries["trailing"]["error"].lower()
    assert set(transformer.excluded_conversions) == {
        "unknown_function", "unknown_field", "trailing"}


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "redcap_value(curr_row, 'x', None, event_data).__class__",
        "[value for value in curr_row]",
        "redcap_add(1, 2); redcap_add(3, 4)",
    ],
)
def test_generated_python_security_validator_rejects_non_allowlisted_code(
        expression: str) -> None:
    with pytest.raises(calculated.CalculationTranslationError):
        calculated.validate_generated_python(expression)


def test_redcap_source_rejects_injection_syntax() -> None:
    with pytest.raises(calculated.UnsupportedCalculationError):
        calculated.translate_redcap_calculation(
            "[amount] + __import__('os').system('echo unsafe')")


def test_schema_rejects_duplicate_variable_names() -> None:
    frame = _dictionary(_row("duplicate"), _row("duplicate"))
    with pytest.raises(calculated.CalculationSchemaError, match="Duplicate"):
        calculated.TransformCalculatedFields(frame)


def test_same_event_dependencies_are_topologically_ordered() -> None:
    # Dictionary order is intentionally the reverse of dependency order.
    frame = _dictionary(
        _row("third", field_type="calc", calculation="[second] + 1"),
        _row("second", field_type="calc", calculation="[first] + 1"),
        _row("first", field_type="calc", calculation="1"),
        _row(
            "cross_event",
            field_type="calc",
            calculation="[month_6_arm_1][third] + 1",
        ),
    )
    transformer = calculated.TransformCalculatedFields(frame)
    entries = transformer()

    assert transformer.evaluation_order == [
        "first", "cross_event", "second", "third"]
    assert entries["second"]["same_event_calc_dependencies"] == ["first"]
    assert entries["third"]["same_event_calc_dependencies"] == ["second"]
    assert entries["cross_event"]["same_event_calc_dependencies"] == []
    assert entries["cross_event"]["cross_event_dependencies"][0][
        "variable"] == "third"


def test_same_event_cycle_is_excluded_from_evaluation_order() -> None:
    frame = _dictionary(
        _row("independent", field_type="calc", calculation="1"),
        _row("cycle_a", field_type="calc", calculation="[cycle_b] + 1"),
        _row("cycle_b", field_type="calc", calculation="[cycle_a] + 1"),
    )
    transformer = calculated.TransformCalculatedFields(frame)
    entries = transformer()

    assert transformer.evaluation_order == ["independent"]
    for variable in ("cycle_a", "cycle_b"):
        assert entries[variable]["status"] == "invalid"
        assert entries[variable]["error_code"] == (
            "calculation_dependency_cycle")
        assert entries[variable]["converted_calculation"] == ""


def test_failed_upstream_blocks_same_event_but_not_cross_event_dependency(
        ) -> None:
    frame = _dictionary(
        _row(
            "unsupported_upstream",
            field_type="calc",
            calculation="mystery(1)",
        ),
        _row(
            "same_event_downstream",
            field_type="calc",
            calculation="[unsupported_upstream] + 1",
        ),
        _row(
            "cross_event_downstream",
            field_type="calc",
            calculation="[month_6_arm_1][unsupported_upstream] + 1",
        ),
    )
    transformer = calculated.TransformCalculatedFields(frame)
    entries = transformer()

    assert entries["unsupported_upstream"]["status"] == "unsupported"
    same_event = entries["same_event_downstream"]
    assert same_event["status"] == "invalid"
    assert same_event["error_code"] == "blocked_calculation_dependency"
    assert "unsupported_upstream" in same_event["error"]
    assert same_event["converted_calculation"] == ""

    cross_event = entries["cross_event_downstream"]
    assert cross_event["status"] == "converted"
    assert cross_event["same_event_calc_dependencies"] == []
    assert cross_event["cross_event_dependencies"][0]["variable"] == (
        "unsupported_upstream")
    assert transformer.evaluation_order == ["cross_event_downstream"]


def test_only_actual_cycle_members_are_cycles_and_downstream_is_blocked(
        ) -> None:
    frame = _dictionary(
        _row("cycle_a", field_type="calc", calculation="[cycle_b] + 1"),
        _row("cycle_b", field_type="calc", calculation="[cycle_a] + 1"),
        _row("after_cycle", field_type="calc", calculation="[cycle_a] + 1"),
        _row("independent", field_type="calc", calculation="1"),
    )
    transformer = calculated.TransformCalculatedFields(frame)
    entries = transformer()

    for variable in ("cycle_a", "cycle_b"):
        assert entries[variable]["status"] == "invalid"
        assert entries[variable]["error_code"] == (
            "calculation_dependency_cycle")
    assert entries["after_cycle"]["status"] == "invalid"
    assert entries["after_cycle"]["error_code"] == (
        "blocked_calculation_dependency")
    assert "cycle_a" in entries["after_cycle"]["error"]
    assert entries["independent"]["status"] == "converted"
    assert transformer.evaluation_order == ["independent"]


def test_pathological_literals_and_nesting_are_isolated_as_unsupported(
        ) -> None:
    oversized_number = "9" * (calculated.MAX_NUMERIC_LITERAL_LENGTH + 1)
    deep_expression = (
        "(" * (calculated.MAX_PARSE_DEPTH + 100)
        + "1"
        + ")" * (calculated.MAX_PARSE_DEPTH + 100)
    )
    frame = _dictionary(
        _row("normal", field_type="calc", calculation="1 + 1"),
        _row(
            "oversized_number",
            field_type="calc",
            calculation=oversized_number,
        ),
        _row(
            "deep_nesting",
            field_type="calc",
            calculation=deep_expression,
        ),
    )

    entries = calculated.TransformCalculatedFields(frame)()

    assert entries["normal"]["status"] == "converted"
    for variable in ("oversized_number", "deep_nesting"):
        assert entries[variable]["status"] == "unsupported"
        assert entries[variable]["error_code"] == "unsupported_calculation"
        assert entries[variable]["converted_calculation"] == ""
    assert "Numeric literal exceeds" in entries["oversized_number"]["error"]
    assert "nesting" in entries["deep_nesting"]["error"].casefold()


def test_schema_rejects_variable_name_surrounding_whitespace() -> None:
    frame = _dictionary(_row("valid"), _row(" valid "))
    with pytest.raises(
            calculated.CalculationSchemaError,
            match="leading/trailing whitespace"):
        calculated.TransformCalculatedFields(frame)


def test_schema_rejects_case_insensitive_variable_name_collision() -> None:
    frame = _dictionary(_row("SameName"), _row("samename"))
    with pytest.raises(calculated.CalculationSchemaError, match="Duplicate"):
        calculated.TransformCalculatedFields(frame)


def test_pandas_series_size_and_name_labels_resolve_cells_not_attributes(
        ) -> None:
    curr_row = pd.Series(
        {"size": "2", "name": "3"},
        name="series_metadata_name",
    )

    assert _evaluate("[size] + [name]", curr_row) == 5


@pytest.mark.parametrize(
    ("source", "row", "expected"),
    [
        ("[value] = 2", {"value": 2}, 1),
        ("[value] = 2", {"value": 3}, 0),
        ("[left] = 1 and [right] = 1", {"left": 1, "right": 1}, 1),
        # bare "not" (no parentheses) is undocumented REDCap and now fails
        # closed; the function-call form keeps exact one-argument semantics
        ("not([value])", {"value": 1}, 0),
    ],
)
def test_root_logical_result_is_numeric_one_or_zero(
        source: str, row: dict[str, int], expected: int) -> None:
    value = _evaluate(source, row)
    assert type(value) is int
    assert value == expected


def test_identifier_dependencies_propagate_directly_and_transitively() -> None:
    frame = _dictionary(
        _row("patient_name", identifier="y"),
        _row("direct", field_type="calc", calculation="[patient_name] = ''"),
        _row("indirect", field_type="calc", calculation="[direct] + 1"),
        _row("twice_removed", field_type="calc", calculation="[indirect] + 1"),
    )
    entries = calculated.TransformCalculatedFields(frame)()

    assert entries["direct"]["direct_identifier_dependencies"] == [
        "patient_name"]
    assert entries["direct"]["transitive_identifier_dependencies"] == []
    assert entries["indirect"]["direct_identifier_dependencies"] == []
    assert entries["indirect"]["transitive_identifier_dependencies"] == [
        "patient_name"]
    assert entries["twice_removed"][
        "transitive_identifier_dependencies"] == ["patient_name"]


def test_only_calculated_field_rows_are_emitted() -> None:
    frame = _dictionary(
        _row("ordinary_text"),
        _row("ordinary_checkbox", field_type="checkbox"),
        _row("derived", field_type=" CALC ", calculation="1 + 1"),
    )
    assert set(calculated.TransformCalculatedFields(frame)()) == {"derived"}


def test_cli_strict_writes_complete_atomic_outputs_for_small_dictionary(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dictionary_path = tmp_path / "dictionary.csv"
    output_dir = tmp_path / "outputs"
    _dictionary(
        _row("source"),
        _row("good_calc", field_type="calc", calculation="[source] + 1"),
        _row("bad_calc", field_type="calc", calculation="mystery([source])"),
    ).to_csv(dictionary_path, index=False)

    exit_code = calculated.main([
        "--data-dictionary", str(dictionary_path),
        "--output-dir", str(output_dir),
        "--strict",
    ])

    assert exit_code == 1
    artifact_path = output_dir / "converted_calculated_fields.json"
    csv_path = output_dir / "converted_calculated_fields.csv"
    diagnostics_path = output_dir / "excluded_calculated_field_vars.json"
    assert artifact_path.is_file()
    assert csv_path.is_file()
    assert diagnostics_path.is_file()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    audit = pd.read_csv(csv_path, keep_default_na=False)
    assert artifact["summary"] == {
        "calculated_field_count": 2,
        "converted_count": 1,
        "excluded_count": 1,
        "same_event_evaluation_order_count": 1,
    }
    assert set(audit["variable"]) == {"good_calc", "bad_calc"}
    assert set(diagnostics) == {"bad_calc"}
    assert not list(output_dir.glob("*.tmp"))
    assert "Converted 1 of 2 calculated fields; 1 excluded." in (
        capsys.readouterr().out)


def test_atomic_json_failure_preserves_destination_and_removes_temporary_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(source, target) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(calculated.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        calculated._atomic_json_write({"new": True}, destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_rejects_colliding_output_paths_without_writing(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dictionary_path = tmp_path / "dictionary.csv"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    sentinel = output_dir / "same.json"
    sentinel.write_text('{"old": true}\n', encoding="utf-8")
    _dictionary(
        _row("source"),
        _row("derived", field_type="calc", calculation="[source] + 1"),
    ).to_csv(dictionary_path, index=False)

    exit_code = calculated.main([
        "--data-dictionary", str(dictionary_path),
        "--output-dir", str(output_dir),
        "--output-json", "same.json",
        "--output-csv", "same.json",
        "--diagnostics", "same.json",
    ])

    assert exit_code == 2
    assert json.loads(sentinel.read_text(encoding="utf-8")) == {"old": True}
    assert "three distinct files" in capsys.readouterr().err


def test_artifact_set_rolls_back_all_outputs_when_final_commit_fails(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    json_path = tmp_path / "artifact.json"
    csv_path = tmp_path / "audit.csv"
    diagnostics_path = tmp_path / "diagnostics.json"
    json_path.write_text('{"generation": "old"}\n', encoding="utf-8")
    csv_path.write_text("generation\nold\n", encoding="utf-8")
    diagnostics_path.write_text(
        '{"generation": "old"}\n', encoding="utf-8")

    real_replace = calculated.os.replace
    failed = False

    def fail_final_manifest_once(source, target) -> None:
        nonlocal failed
        if (not failed and Path(target) == json_path
                and str(source).endswith(".stage")):
            failed = True
            raise OSError("simulated final commit failure")
        real_replace(source, target)

    monkeypatch.setattr(calculated.os, "replace", fail_final_manifest_once)
    with pytest.raises(OSError, match="simulated final commit failure"):
        calculated._write_artifact_set(
            {"generation": "new"},
            {"derived": {
                "variable": "derived",
                "references": [],
                "referenced_variables": [],
                "same_event_calc_dependencies": [],
                "cross_event_dependencies": [],
                "direct_identifier_dependencies": [],
                "transitive_identifier_dependencies": [],
                "functions": [],
            }},
            {"generation": "new"},
            json_path, csv_path, diagnostics_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "generation": "old"}
    assert csv_path.read_text(encoding="utf-8") == "generation\nold\n"
    assert json.loads(diagnostics_path.read_text(encoding="utf-8")) == {
        "generation": "old"}
    assert not list(tmp_path.glob(".*.stage"))
    assert not list(tmp_path.glob(".*.backup"))


def test_dictionary_discovery_resolves_relative_to_config_directory(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    dictionary_dir = project / "relative_dependencies" / "data_dictionary"
    dictionary_dir.mkdir(parents=True)
    dictionary_path = dictionary_dir / "current_data_dictionary.csv"
    dictionary_path.write_text("placeholder\n", encoding="utf-8")
    (project / "config.json").write_text(json.dumps({
        "paths": {"dependencies_path": "relative_dependencies"},
    }), encoding="utf-8")
    monkeypatch.chdir(project)

    assert calculated._discover_data_dictionary() == dictionary_path.resolve()


def test_dictionary_discovery_rejects_ambiguous_candidates(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    dictionary_dir = project / "dependencies" / "data_dictionary"
    dictionary_dir.mkdir(parents=True)
    for name in (
            "site_a_current_data_dictionary_2026.csv",
            "site_b_current_data_dictionary_2026.csv"):
        (dictionary_dir / name).write_text("placeholder\n", encoding="utf-8")
    (project / "config.json").write_text(json.dumps({
        "paths": {"dependencies_path": "dependencies"},
    }), encoding="utf-8")
    monkeypatch.chdir(project)

    with pytest.raises(FileNotFoundError, match="Multiple current"):
        calculated._discover_data_dictionary()
