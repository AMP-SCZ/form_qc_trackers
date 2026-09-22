"""Interface regressions for the standalone branching-logic checker."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from harmonization_qc.qc_branching_logic import (
    TestTransformBranchingLogic as BranchingLogicChecker,
)
from utils.utils import Utils


def _dictionary() -> pd.DataFrame:
    return pd.DataFrame([{
        "Variable / Field Name": "right_field",
        "Form Name": "test_form",
        "Field Type": "text",
        "Branching Logic (Show field only if...)": "[gate] = 1",
    }])


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_dictionary(path: Path, variable: str) -> None:
    pd.DataFrame([{
        "Variable / Field Name": variable,
        "Form Name": "test_form",
        "Field Type": "text",
        "Branching Logic (Show field only if...)": "",
    }]).to_csv(path, index=False)


def test_utils_dictionary_discovery_prefers_exact_canonical_file(
        tmp_path: Path) -> None:
    dictionary_dir = tmp_path / "data_dictionary"
    dictionary_dir.mkdir()
    _write_dictionary(
        dictionary_dir / "current_data_dictionary.csv", "canonical")
    _write_dictionary(
        dictionary_dir / "backup_current_data_dictionary_2099.csv", "backup")
    utils = Utils.__new__(Utils)
    utils.config_info = {"paths": {"dependencies_path": str(tmp_path)}}

    selected = Utils.read_data_dictionary(utils)

    assert selected["Variable / Field Name"].tolist() == ["canonical"]


def test_utils_dictionary_discovery_rejects_ambiguous_variants(
        tmp_path: Path) -> None:
    dictionary_dir = tmp_path / "data_dictionary"
    dictionary_dir.mkdir()
    _write_dictionary(
        dictionary_dir / "a_current_data_dictionary_2098.csv", "a")
    _write_dictionary(
        dictionary_dir / "b_current_data_dictionary_2099.csv", "b")
    utils = Utils.__new__(Utils)
    utils.config_info = {"paths": {"dependencies_path": str(tmp_path)}}

    with pytest.raises(FileNotFoundError, match="Multiple data dictionary"):
        Utils.read_data_dictionary(utils)


def _checker(tmp_path: Path) -> BranchingLogicChecker:
    dependencies = tmp_path / "dependencies"
    combined = tmp_path / "combined"
    output = tmp_path / "output"
    dependencies.mkdir()
    combined.mkdir()
    output.mkdir()

    pd.DataFrame([{
        "var": "right_field",
        "affected_col": "branching_logic",
        "affected_col_val": "[gate] = 1",
    }]).to_csv(dependencies / "identifier_effects.csv", index=False)
    # A same-named output artifact must never be used as a dependency input.
    pd.DataFrame([{
        "var": "wrong_output_field",
        "affected_col": "branching_logic",
        "affected_col_val": "[gate] = 1",
    }]).to_csv(output / "identifier_effects.csv", index=False)
    _write_json(dependencies / "excluded_branching_logic_vars.json", {})
    _write_json(dependencies / "grouped_variables.json", {
        "var_forms": {"right_field": "test_form", "gate": "test_form"},
    })
    _write_json(dependencies / "important_form_vars.json", {
        "test_form": {
            "interview_date_var": "test_interview_date",
            "missing_var": "test_missing",
            "completion_var": "test_form_complete",
        },
    })
    _write_json(dependencies / "forms_per_timepoint.json", {
        "chr": {"baseline": ["test_form"], "month1": ["test_form"]},
        "hc": {"baseline": ["test_form"], "month1": ["test_form"]},
    })

    return BranchingLogicChecker(
        config_info={
            "paths": {
                "dependencies_path": str(dependencies),
                "combined_csv_path": str(combined),
                "output_path": str(output),
            },
        },
        data_dictionary_df=_dictionary(),
    )


def test_checker_uses_dependency_path_for_identifier_effects(
        tmp_path: Path) -> None:
    checker = _checker(tmp_path)

    assert checker.ident_bl_vars == {"right_field"}


def test_checker_discovers_current_canonical_filenames_with_optional_suffix(
        tmp_path: Path) -> None:
    checker = _checker(tmp_path)
    combined = Path(checker.combined_csv_path)
    expected = [
        combined / "AMPSCZ-combined-redcap_baseline_ProNET-day1to1.csv",
        combined / "AMPSCZ-combined-redcap_month_1_PRESCIENT.csv",
    ]
    ignored = combined / "combined-PRONET-baseline-day1to1.csv"
    for path in [*expected, ignored]:
        path.write_text("subjectid\nS1\n", encoding="utf-8")

    specs = checker.discover_combined_csvs()

    assert [(spec.path, spec.network, spec.timepoint) for spec in specs] == [
        (expected[0].resolve(), "PRONET", "baseline"),
        (expected[1].resolve(), "PRESCIENT", "month1"),
    ]


def test_identifier_filter_is_explicitly_network_scoped(
        tmp_path: Path) -> None:
    checker = _checker(tmp_path)
    row = SimpleNamespace(
        subjectid="S1",
        chrcrit_part="1",
        redcap_repeat_instrument="",
        test_missing="0",
        test_form_complete="2",
        test_form_complete_rpms="2",
        gate="1",
        right_field="",
    )

    assert not checker.filter_rows(
        row, "right_field", "PRONET", "baseline")
    assert checker.filter_rows(
        row, "right_field", "PRESCIENT", "baseline")


def test_checker_honors_schedule_missing_rpms_and_repeat_identity(
        tmp_path: Path) -> None:
    checker = _checker(tmp_path)
    base = {
        "subjectid": "S1",
        "chrcrit_part": "1",
        "redcap_repeat_instrument": "test_form",
        "test_missing": "0",
        "test_form_complete_rpms": "2",
        "gate": "1",
        "right_field": "",
    }

    assert checker.filter_rows(
        SimpleNamespace(**base), "right_field", "PRESCIENT", "baseline",
        frozenset({"test_form"}))
    assert not checker.filter_rows(
        SimpleNamespace(**{**base, "redcap_repeat_instrument": "other_form"}),
        "right_field", "PRESCIENT", "baseline",
        frozenset({"test_form", "other_form"}))
    assert not checker.filter_rows(
        SimpleNamespace(**{**base, "test_missing": "1"}),
        "right_field", "PRESCIENT", "baseline",
        frozenset({"test_form"}))
    assert not checker.filter_rows(
        SimpleNamespace(**{**base, "test_form_complete_rpms": "3"}),
        "right_field", "PRESCIENT", "baseline",
        frozenset({"test_form"}))
    assert not checker.filter_rows(
        SimpleNamespace(**base), "right_field", "PRESCIENT", "month2",
        frozenset({"test_form"}))


def _prescient_row(subjectid: str) -> dict[str, object]:
    return {
        "subjectid": subjectid,
        "chrcrit_part": "1",
        "redcap_repeat_instrument": "",
        "test_missing": "0",
        "test_form_complete_rpms": "2",
        "gate": "1",
        "right_field": "",
    }


def test_checker_accumulates_all_timepoints_before_writing_network_output(
        tmp_path: Path) -> None:
    checker = _checker(tmp_path)
    combined = Path(checker.combined_csv_path)
    pd.DataFrame([_prescient_row("BASE")]).to_csv(
        combined / "AMPSCZ-combined-redcap_baseline_PRESCIENT-day1to1.csv",
        index=False,
    )
    pd.DataFrame([_prescient_row("M1")]).to_csv(
        combined / "AMPSCZ-combined-redcap_month_1_PRESCIENT-day1to1.csv",
        index=False,
    )

    checker.compare_bl_to_database()

    output = pd.read_csv(
        Path(checker.output_path) / "bl_mismatches_PRESCIENT.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert output[["network", "timepoint", "variable"]].to_dict(
        "records") == [
            {
                "network": "PRESCIENT",
                "timepoint": "baseline",
                "variable": "right_field",
            },
            {
                "network": "PRESCIENT",
                "timepoint": "month1",
                "variable": "right_field",
            },
        ]


def test_checker_records_runtime_evaluation_failures_instead_of_silencing_them(
        tmp_path: Path) -> None:
    checker = _checker(tmp_path)
    combined = Path(checker.combined_csv_path)
    pd.DataFrame([_prescient_row("ERROR")]).to_csv(
        combined / "AMPSCZ-combined-redcap_baseline_PRESCIENT-day1to1.csv",
        index=False,
    )
    checker.converted_bl["right_field"]["converted_branching_logic"] = "1 / 0"

    checker.compare_bl_to_database()

    errors = pd.read_csv(
        Path(checker.output_path) / "branching_logic_eval_errors.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert errors[[
        "network", "timepoint", "variable", "error_type", "count",
    ]].to_dict("records") == [{
        "network": "PRESCIENT",
        "timepoint": "baseline",
        "variable": "right_field",
        "error_type": "ZeroDivisionError",
        "count": "1",
    }]
