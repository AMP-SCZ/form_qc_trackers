"""Regression contracts for issues found in the calculated-field audit.

These tests intentionally live outside ``test_calculated_field_mismatches.py``
so that the audit-remediation work remains independently reviewable.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd
import pytest

from discover_errors import calculated_field_mismatches as scanner


VARIABLE = "Variable / Field Name"
FORM = "Form Name"
FIELD_TYPE = "Field Type"
CALCULATION = "Choices, Calculations, OR Slider Labels"
BRANCHING = "Branching Logic (Show field only if...)"


def _write_dictionary(path: Path, rows: list[dict[str, str]]) -> Path:
    normalized = []
    for row in rows:
        normalized.append({
            VARIABLE: row["variable"],
            FORM: row.get("form", "test_form"),
            FIELD_TYPE: row.get("field_type", "text"),
            CALCULATION: row.get("calculation", ""),
            BRANCHING: row.get("branching_logic", ""),
        })
    pd.DataFrame(normalized).to_csv(path, index=False)
    return path


def _field(variable: str, *, form: str = "test_form") -> dict[str, str]:
    return {"variable": variable, "form": form, "field_type": "text"}


def _calc(
        variable: str, calculation: str, *, form: str = "test_form",
        branching_logic: str = "") -> dict[str, str]:
    return {
        "variable": variable,
        "form": form,
        "field_type": "calc",
        "calculation": calculation,
        "branching_logic": branching_logic,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run(
        dictionary: Path, inputs: list[Path], output: Path,
        diagnostics: Path, *extra: str) -> int:
    argv = [
        "--data-dictionary", str(dictionary),
        "--output", str(output),
        "--diagnostics", str(diagnostics),
    ]
    for input_path in inputs:
        argv.extend(("--input", str(input_path)))
    argv.extend(extra)
    return scanner.main(argv)


def _read_output(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _base_row(subjectid: str = "S1", event: str = "baseline_arm_1",
              **values: object) -> dict[str, object]:
    return {
        "subjectid": subjectid,
        "redcap_event_name": event,
        **values,
    }


def test_representative_missing_codes_are_blank_at_ingestion_and_comparison(
        tmp_path: Path) -> None:
    missing_codes = (
        -3, "-9.00", "-99.00", "999.00",
        "1901-01-01", "1903-03-03", "1909-09-09",
        "1901-01-01 00:00", "1903-03-03 12:34", "1909-09-09 23:59",
    )
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source"),
        _calc("target", "[source] + 1"),
    ])
    data = _write_csv(tmp_path / "input.csv", [
        _base_row(
            subjectid=f"MISSING_{index:02d}",
            source=missing_code,
            target=missing_code,
        )
        for index, missing_code in enumerate(missing_codes)
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [data], output, diagnostics) == 0
    assert _read_output(output).empty
    details = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert details["inputs"][0][
        "missing_code_cells_replaced_with_blank"] == (
            2 * len(missing_codes))
    assert details["summary"][
        "missing_code_cells_replaced_with_blank"] == (
            2 * len(missing_codes))


def test_blank_and_missing_code_duplicate_rows_merge_without_conflict(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source"),
        _calc("target", "[source] + 1"),
    ])
    blank = _write_csv(tmp_path / "blank.csv", [
        _base_row(source="", target=""),
    ])
    coded = _write_csv(tmp_path / "coded.csv", [
        _base_row(source="999.0", target="-9"),
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [blank, coded], output, diagnostics) == 0
    assert _read_output(output).empty
    details = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert not any(
        item.get("type") == "conflicting_duplicate_identity"
        for item in details.get("row_identity_diagnostics", []))


def test_cross_event_missing_code_is_blank_in_snapshot_calculation(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source"),
        _calc("target", "[baseline_arm_1][source] + 1"),
    ])
    data = _write_csv(tmp_path / "input.csv", [
        _base_row(event="baseline_arm_1", source="1903-03-03"),
        _base_row(event="month_1_arm_1", target="-99.0"),
    ])
    output = tmp_path / "mismatches.csv"

    assert _run(
        dictionary, [data], output, tmp_path / "diagnostics.json") == 0
    assert _read_output(output).empty


def test_branching_calculation_sees_missing_code_as_blank(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("gate"),
        _calc("target", "5", branching_logic="[gate] = ''"),
    ])
    data = _write_csv(tmp_path / "input.csv", [
        _base_row(gate="-9.0", target="0"),
    ])
    output = tmp_path / "mismatches.csv"

    assert _run(
        dictionary, [data], output, tmp_path / "diagnostics.json") == 0
    result = _read_output(output)
    assert result[["variable", "recalculated_value"]].to_dict(
        "records") == [{"variable": "target", "recalculated_value": "5"}]


@pytest.mark.parametrize("identity_value", ("999", "999.00"))
def test_structural_identity_value_999_is_not_treated_as_response_missing(
        tmp_path: Path, identity_value: str) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source"),
        _calc("target", "[source] + 1"),
    ])
    data = _write_csv(tmp_path / "input.csv", [{
        **_base_row(subjectid=identity_value, source="1", target="2"),
        "redcap_repeat_instrument": "test_form",
        "redcap_repeat_instance": identity_value,
    }])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [data], output, diagnostics) == 0
    assert _read_output(output).empty
    details = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert details["summary"][
        "missing_code_cells_replaced_with_blank"] == 0


def test_direct_mode_marks_downstream_calculation_as_snapshot_uncertain(
        tmp_path: Path) -> None:
    """A stale upstream calc must not make a correct downstream value actionable."""
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source"),
        _calc("upstream_calc", "[source] + 1"),
        _calc("downstream_calc", "[upstream_calc] + 1"),
    ])
    data = _write_csv(tmp_path / "input.csv", [
        _base_row(source="1", upstream_calc="99", downstream_calc="3"),
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(
        dictionary, [data], output, diagnostics,
        "--comparison-mode", "direct") == 0
    rows = _read_output(output).set_index("variable")

    assert rows.loc["upstream_calc", "actionability"] == "actionable"
    downstream = rows.loc["downstream_calc"]
    assert downstream["actionability"] == "semantic_uncertain"
    assert (
        "same_event_calc_dependency_uses_stored_snapshot"
        in downstream["semantic_risk"].split("|")
    )


@pytest.mark.parametrize(
    ("variable", "branching_logic"),
    [
        ("self_gated_calc", "[self_gated_calc] = 2"),
        ("unsupported_gate_calc", "contains([source], '1')"),
        ("missing_gate_calc", "[missing_gate] = 1"),
    ],
)
@pytest.mark.parametrize("comparison_mode", ("direct", "propagated", "multipass"))
def test_unassessed_branching_eligibility_is_suppressed_from_output(
        tmp_path: Path, variable: str, branching_logic: str,
        comparison_mode: str) -> None:
    """Unknown branching remains evaluated but is not written to output."""
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source"),
        _field("missing_gate"),
        _calc(
            variable, "[source] + 1",
            branching_logic=branching_logic,
        ),
    ])
    data = _write_csv(tmp_path / "input.csv", [
        _base_row(source="1", **{variable: "99"}),
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(
        dictionary, [data], output, diagnostics,
        "--comparison-mode", comparison_mode) == 0
    assert _read_output(output).empty

    summary = json.loads(
        diagnostics.read_text(encoding="utf-8"))["summary"]
    assert summary["evaluated"] == 1
    assert summary["mismatches"] == 1
    assert summary["output_row_count"] == 0
    assert summary["output_suppressed__branching_not_visible"] == 1


def test_numerically_equivalent_duplicate_rows_collapse_without_quarantine(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source"),
        _calc("target", "[source] + 1"),
    ])
    first = _write_csv(tmp_path / "first.csv", [
        _base_row(source="1", target="99"),
    ])
    second = _write_csv(tmp_path / "second.csv", [
        _base_row(source="1.0", target="99.0"),
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [first, second], output, diagnostics) == 0
    rows = _read_output(output)
    details = json.loads(diagnostics.read_text(encoding="utf-8"))

    assert rows["variable"].tolist() == ["target"]
    assert not any(
        item.get("type") == "conflicting_duplicate_identity"
        for item in details.get("row_identity_diagnostics", [])
    )


def test_duplicate_schema_union_does_not_lose_blank_calculated_target(
        tmp_path: Path) -> None:
    """Target selection must use the merged schema, not the winning filename."""
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source"),
        _calc("target", "[source] + 1"),
    ])
    without_target = _write_csv(tmp_path / "a_without_target.csv", [
        _base_row(source="1"),
    ])
    with_target = _write_csv(tmp_path / "z_with_target.csv", [
        _base_row(source="1", target=""),
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(
        dictionary, [without_target, with_target], output, diagnostics) == 0
    rows = _read_output(output)

    assert rows[["variable", "stored_value", "recalculated_value"]].to_dict(
        "records") == [{
            "variable": "target",
            "stored_value": "",
            "recalculated_value": "2",
        }]


def test_generic_and_canonical_explicit_inputs_share_actual_network_context(
        tmp_path: Path) -> None:
    """Filename classification must not split rows from the same network."""
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source"),
        _calc("target", "[baseline_arm_1][source] + 1"),
    ])
    baseline = _write_csv(
        tmp_path / "AMPSCZ-combined-redcap_baseline_ProNET-day1to1.csv",
        [_base_row(source="2")],
    )
    month_one = _write_csv(tmp_path / "custom_month_one.csv", [
        {
            **_base_row(event="month_1_arm_1", target="99"),
            "network": "PRONET",
            "timepoint": "month1",
        },
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [baseline, month_one], output, diagnostics) == 0
    row = _read_output(output).iloc[0]

    assert row["network"] == "PRONET"
    assert row["variable"] == "target"
    assert row["recalculated_value"] == "3"


def test_explicit_opposite_arm_reference_uses_present_mixed_arm_event(
        tmp_path: Path) -> None:
    """An explicit arm-2 event is not blank merely because target is arm 1."""
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source"),
        _calc("target", "[baseline_arm_2][source] + 1"),
    ])
    data = _write_csv(tmp_path / "input.csv", [
        {
            **_base_row(event="baseline_arm_2", source="4"),
            "network": "PRESCIENT",
            "timepoint": "baseline",
        },
        {
            **_base_row(event="month_1_arm_1", target="99"),
            "network": "PRESCIENT",
            "timepoint": "month1",
        },
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [data], output, diagnostics) == 0
    row = _read_output(output).iloc[0]

    assert row["variable"] == "target"
    assert row["recalculated_value"] == "5"
    assert row["structural_blank_dependencies"] == ""


def test_conflict_in_one_repeat_instance_does_not_quarantine_another_instance(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("repeat_source", form="repeat_form"),
        _calc(
            "target", "[baseline_arm_1][repeat_source][2] + 1",
            form="target_form",
        ),
    ])
    data = _write_csv(tmp_path / "input.csv", [
        {
            **_base_row(event="baseline_arm_1", repeat_source="1"),
            "network": "PRONET",
            "timepoint": "baseline",
            "redcap_repeat_instrument": "repeat_form",
            "redcap_repeat_instance": "1",
        },
        {
            **_base_row(event="baseline_arm_1", repeat_source="9"),
            "network": "PRONET",
            "timepoint": "baseline",
            "redcap_repeat_instrument": "repeat_form",
            "redcap_repeat_instance": "1",
        },
        {
            **_base_row(event="baseline_arm_1", repeat_source="2"),
            "network": "PRONET",
            "timepoint": "baseline",
            "redcap_repeat_instrument": "repeat_form",
            "redcap_repeat_instance": "2",
        },
        {
            **_base_row(event="month_1_arm_1", target="99"),
            "network": "PRONET",
            "timepoint": "month1",
            "redcap_repeat_instrument": "",
            "redcap_repeat_instance": "",
        },
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [data], output, diagnostics) == 0
    rows = _read_output(output)

    assert rows["variable"].tolist() == ["target"]
    assert rows.iloc[0]["recalculated_value"] == "3"


def test_repeating_calc_reads_base_event_and_is_not_checked_on_base_row(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("base_source", form="base_form"),
        _calc("repeat_target", "[base_source] + 1", form="repeat_form"),
    ])
    data = _write_csv(tmp_path / "input.csv", [
        {
            **_base_row(base_source="4", repeat_target=""),
            "redcap_repeat_instrument": "",
            "redcap_repeat_instance": "",
        },
        {
            **_base_row(base_source="", repeat_target="99"),
            "redcap_repeat_instrument": "repeat_form",
            "redcap_repeat_instance": "1",
        },
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [data], output, diagnostics) == 0
    rows = _read_output(output)
    summary = json.loads(diagnostics.read_text(encoding="utf-8"))["summary"]

    assert rows[[
        "variable", "redcap_repeat_instrument", "redcap_repeat_instance",
        "recalculated_value",
    ]].to_dict("records") == [{
        "variable": "repeat_target",
        "redcap_repeat_instrument": "repeat_form",
        "redcap_repeat_instance": "1",
        "recalculated_value": "5",
    }]
    assert summary["base_row_repeat_instrument_targets_skipped"] == 1


@pytest.mark.parametrize(
    ("flag", "missing_name"),
    [
        ("--config", "missing_config.json"),
        ("--important-form-vars", "missing_important_form_vars.json"),
        ("--forms-per-timepoint", "missing_forms_per_timepoint.json"),
    ],
)
def test_explicit_missing_configuration_inputs_fail_without_outputs(
        tmp_path: Path, capsys: pytest.CaptureFixture[str],
        flag: str, missing_name: str) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source"),
        _calc("target", "[source] + 1"),
    ])
    data = _write_csv(tmp_path / "input.csv", [
        _base_row(source="1", target="2"),
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"
    missing = tmp_path / missing_name

    exit_code = _run(
        dictionary, [data], output, diagnostics, flag, str(missing))

    assert exit_code == 2
    assert missing_name in capsys.readouterr().err
    assert not output.exists()
    assert not diagnostics.exists()


def test_complete_production_directory_requires_both_applicability_maps(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dictionary_dir = tmp_path / "dictionary"
    dictionary_dir.mkdir()
    dictionary = _write_dictionary(
        dictionary_dir / "current_data_dictionary.csv", [
            _field("source"),
            _calc("target", "[source] + 1"),
        ])
    combined = tmp_path / "combined"
    combined.mkdir()
    for network in ("PRONET", "PRESCIENT"):
        filename_network = "ProNET" if network == "PRONET" else network
        for timepoint in scanner.TIMEPOINTS:
            filename_timepoint = (
                "floating_forms" if timepoint == "floating"
                else re.sub(r"^month(\d+)$", r"month_\1", timepoint)
            )
            (combined / (
                f"AMPSCZ-combined-redcap_{filename_timepoint}_"
                f"{filename_network}-day1to1.csv"
            )).write_text("subjectid\nS1\n", encoding="utf-8")
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    exit_code = scanner.main([
        "--data-dictionary", str(dictionary),
        "--input-dir", str(combined),
        "--output", str(output),
        "--diagnostics", str(diagnostics),
    ])

    assert exit_code == 2
    error = capsys.readouterr().err.casefold()
    assert "important-form variable map does not exist" in error
    assert "important_form_vars.json" in error
    assert not output.exists()
    assert not diagnostics.exists()


def test_diagnostics_report_target_coverage_and_actual_branching_mode(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source"),
        _calc("covered_calc", "[source] + 1"),
        _calc("uncovered_calc", "[source] + 2"),
    ])
    data = _write_csv(tmp_path / "input.csv", [
        _base_row(source="1", covered_calc="99"),
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [data], output, diagnostics) == 0
    details = json.loads(diagnostics.read_text(encoding="utf-8"))
    coverage = details["converted_target_coverage"]

    assert details["branching_logic_mode"] == (
        "safe_noncalculated_tri_state_eligibility")
    assert coverage["covered_count"] == 1
    assert coverage["uncovered_count"] == 1
    assert coverage["uncovered_variables"] == [{
        "variable": "uncovered_calc",
        "form": "test_form",
    }]


def test_network_spools_are_removed_if_final_commit_fails(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source"),
        _calc("target", "[source] + 1"),
    ])
    data = _write_csv(tmp_path / "input.csv", [
        _base_row(source="1", target="99"),
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    def fail_commit(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated final commit failure")

    monkeypatch.setattr(
        scanner, "_write_spooled_artifacts_transactionally", fail_commit)

    assert _run(dictionary, [data], output, diagnostics) == 2
    assert not list(tmp_path.glob("*.network-spool"))
    assert not output.exists()
    assert not diagnostics.exists()


def test_multipass_is_default_and_propagates_explicit_event_calculations(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source", form="baseline_form"),
        _calc("baseline_calc", "[source] + 1", form="baseline_form"),
        _calc(
            "month_calc", "[baseline_arm_1][baseline_calc] + 1",
            form="month_form"),
    ])
    baseline = _write_csv(tmp_path / "baseline.csv", [
        _base_row(event="baseline_arm_1", source="1", baseline_calc="99"),
    ])
    month = _write_csv(tmp_path / "month.csv", [
        _base_row(event="month_1_arm_1", month_calc="3"),
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [baseline, month], output, diagnostics) == 0
    rows = _read_output(output)
    details = json.loads(diagnostics.read_text(encoding="utf-8"))

    assert rows["variable"].tolist() == ["baseline_calc"]
    assert details["comparison_mode"] == "multipass"
    assert details["cross_event_mode"] == "iterative_fixpoint"
    assert details["summary"]["multipass_converged"] == 1
    assert details["summary"]["multipass_passes"] >= 2

    direct_output = tmp_path / "direct.csv"
    assert _run(
        dictionary, [baseline, month], direct_output,
        tmp_path / "direct.json", "--comparison-mode", "direct") == 0
    direct = _read_output(direct_output).set_index("variable")
    assert direct.loc["month_calc", "recalculated_value"] == "100"


def test_multipass_reports_cross_event_nonconvergence(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _calc(
            "baseline_calc", "1 - [month_1_arm_1][month_calc]",
            form="baseline_form"),
        _calc(
            "month_calc", "[baseline_arm_1][baseline_calc]",
            form="month_form"),
    ])
    baseline = _write_csv(tmp_path / "baseline.csv", [
        _base_row(event="baseline_arm_1", baseline_calc="0"),
    ])
    month = _write_csv(tmp_path / "month.csv", [
        _base_row(event="month_1_arm_1", month_calc="0"),
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    assert _run(dictionary, [baseline, month], output, diagnostics) == 1
    rows = _read_output(output)
    details = json.loads(diagnostics.read_text(encoding="utf-8"))

    assert set(rows["variable"]) == {"baseline_calc", "month_calc"}
    assert set(rows["status"]) == {"nonconvergent"}
    assert set(rows["actionability"]) == {"not_evaluable"}
    assert details["summary"]["multipass_converged"] == 0
    assert details["summary"]["multipass_nonconvergent_targets"] == 2
    assert any(
        item["type"] == "multipass_nonconvergence"
        for item in details["structural_diagnostics"])


def test_multipass_blocks_cross_event_dependents_of_evaluation_errors(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source", form="baseline_form"),
        _calc(
            "baseline_calc", "[source] / 0", form="baseline_form"),
        _calc(
            "month_calc", "[baseline_arm_1][baseline_calc] + 1",
            form="month_form"),
    ])
    baseline = _write_csv(tmp_path / "baseline.csv", [
        _base_row(
            event="baseline_arm_1", source="1", baseline_calc="9"),
    ])
    month = _write_csv(tmp_path / "month.csv", [
        _base_row(event="month_1_arm_1", month_calc="10"),
    ])
    output = tmp_path / "mismatches.csv"
    diagnostics = tmp_path / "diagnostics.json"

    real_evaluator = scanner.evaluate_converted_calculation

    def injected_failure(
            expression: str, *args: object, **kwargs: object) -> object:
        result = real_evaluator(expression, *args, **kwargs)
        if "redcap_divide" in expression:
            return result.__class__(
                value="", status="error", error="injected upstream failure")
        return result

    monkeypatch.setattr(
        scanner, "evaluate_converted_calculation", injected_failure)

    assert _run(dictionary, [baseline, month], output, diagnostics) == 0
    rows = _read_output(output).set_index("variable")
    details = json.loads(diagnostics.read_text(encoding="utf-8"))

    assert rows.loc["baseline_calc", "status"] == "evaluation_error"
    assert rows.loc["month_calc", "status"] == "blocked_dependency"
    assert "baseline_calc" in rows.loc["month_calc", "error"]
    assert details["summary"]["multipass_converged"] == 1
    assert details["summary"]["evaluation_errors"] == 1
    assert details["summary"]["blocked_dependencies"] == 1


def test_parser_defaults_to_multipass_but_keeps_legacy_modes() -> None:
    parser = scanner.build_parser()

    assert parser.parse_args([]).comparison_mode == "multipass"
    assert parser.parse_args([
        "--comparison-mode", "direct"]).comparison_mode == "direct"
    assert parser.parse_args([
        "--comparison-mode", "propagated"]).comparison_mode == "propagated"


def test_multipass_state_equality_preserves_normalized_representation() -> None:
    assert scanner._multipass_state_key(1) == scanner._multipass_state_key("1")
    assert scanner._multipass_state_key("1.0") != scanner._multipass_state_key("1")


def test_multipass_inactive_calculation_is_blank_not_stale(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source", form="baseline_form"),
        _field("baseline_form_complete", form="baseline_form"),
        _calc("inactive_calc", "[source] + 1", form="baseline_form"),
        _calc(
            "month_calc", "[baseline_arm_1][inactive_calc] + 1",
            form="month_form"),
    ])
    baseline = _write_csv(tmp_path / "baseline.csv", [
        _base_row(
            event="baseline_arm_1", source="1", inactive_calc="99",
            baseline_form_complete=""),
    ])
    month = _write_csv(tmp_path / "month.csv", [
        _base_row(event="month_1_arm_1", month_calc=""),
    ])
    output = tmp_path / "mismatches.csv"

    assert _run(
        dictionary, [baseline, month], output,
        tmp_path / "diagnostics.json") == 0
    assert _read_output(output).empty


def test_multipass_normalizes_root_missing_code_before_cross_event_use(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _calc("root_calc", "999", form="baseline_form"),
        _calc(
            "month_calc", "[baseline_arm_1][root_calc] + 1",
            form="month_form"),
    ])
    baseline = _write_csv(tmp_path / "baseline.csv", [
        _base_row(event="baseline_arm_1", root_calc="5"),
    ])
    month = _write_csv(tmp_path / "month.csv", [
        _base_row(event="month_1_arm_1", month_calc=""),
    ])
    output = tmp_path / "mismatches.csv"

    assert _run(
        dictionary, [baseline, month], output,
        tmp_path / "diagnostics.json") == 0
    rows = _read_output(output)
    assert rows[["variable", "recalculated_value"]].to_dict("records") == [{
        "variable": "root_calc", "recalculated_value": "",
    }]


def test_comparison_blank_equivalence_does_not_blank_evaluator_inputs(
        tmp_path: Path) -> None:
    dictionary = _write_dictionary(tmp_path / "dictionary.csv", [
        _field("source"),
        _calc("target", "if([source] = '', 1, 2)"),
    ])
    data = _write_csv(tmp_path / "input.csv", [
        _base_row(subjectid="WHITESPACE", source=" ", target="2"),
        _base_row(subjectid="NAN_MIXED", source="NaN", target="2"),
        _base_row(subjectid="NAN_LOWER", source="nan", target="2"),
    ])
    output = tmp_path / "mismatches.csv"

    assert _run(
        dictionary, [data], output, tmp_path / "diagnostics.json") == 0
    assert _read_output(output).empty


def test_multipass_unresolved_and_ambiguous_calc_cells_block_safely() -> None:
    target_row = scanner.LoadedRow(
        network="PRONET", timepoint="month1", subjectid="S1",
        event_name="month_1_arm_1", repeat_instrument="",
        repeat_instance="", values={"target": "9"},
        available_columns=frozenset({"target"}), source_file="input.csv",
        source_row=2)
    entry = {
        "status": "converted", "form": "target_form",
        "converted_calculation": "0", "references": [{
            "variable": "source_calc", "event": "baseline_arm_1",
            "instance": 1, "source_form": "", "kind": "calc",
            "display": "[baseline_arm_1][source_calc][1]",
        }], "functions": [],
    }
    plan = scanner._PropagationPlan(
        row_index=0, row=target_row, variable="target", entry=entry,
        structural_blanks=(), branching_eligibility="visible",
        branching_reason="", extra_semantic_risks=())

    for extra_rows in ([], [
        scanner.LoadedRow(
            network="PRONET", timepoint="baseline", subjectid="S1",
            event_name="baseline_arm_1", repeat_instrument=form,
            repeat_instance="1", values={"source_calc": "3"},
            available_columns=frozenset({"source_calc"}),
            source_file="input.csv", source_row=index + 3)
        for index, form in enumerate(("repeat_a", "repeat_b"))
    ]):
        loaded = scanner.LoadedInputs(
            rows=[target_row, *extra_rows], input_details=[], file_schemas={},
            event_schemas={}, timepoint_schemas={}, network_schemas={},
            quarantined_identities=frozenset(),
            quarantined_values=frozenset(), row_identity_diagnostics=[])
        artifacts = scanner.ScanArtifacts()
        scanner._scan_multipass_calculated_fields(
            loaded, [plan], {
                "target": entry,
                "source_calc": {"status": "converted"},
            }, {}, {}, inactive_cells=set(), artifacts=artifacts,
            numeric_tolerance=scanner.DEFAULT_NUMERIC_TOLERANCE,
            blank_result_as_zero=False, mismatch_sink=None)

        assert artifacts.mismatches[0]["status"] == "blocked_dependency"
        assert "unresolved or ambiguous" in artifacts.mismatches[0]["error"]


def test_merge_scan_metadata_uses_multipass_partition_semantics() -> None:
    first = scanner.ScanArtifacts()
    first.counters.update({
        "multipass_converged": 1, "multipass_passes": 2,
        "multipass_nonconvergent_targets": 0,
    })
    first.add_diagnostic(("same",), {"network": "PRONET"})
    second = scanner.ScanArtifacts()
    second.counters.update({
        "multipass_converged": 0, "multipass_passes": 5,
        "multipass_nonconvergent_targets": 3,
    })
    second.add_diagnostic(("same",), {"network": "PRESCIENT"})

    merged = scanner.merge_scan_metadata([first, second])

    assert merged.counters["multipass_converged"] == 0
    assert merged.counters["multipass_passes"] == 5
    assert merged.counters["multipass_nonconvergent_targets"] == 3
    assert len(merged.structural_diagnostics) == 2
