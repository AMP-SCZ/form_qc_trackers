"""Black-box contracts for the standalone form-value match scanner."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from discover_errors import form_value_matches as scanner


OUTPUT_COLUMNS = [
    "form_1",
    "timepoint_1",
    "form_2",
    "timepoint_2",
    "median_match_percentage",
    "mean_match_percentage",
    "record_pairs_compared",
    "unique_participants_compared",
    "participant_records_form_1",
    "participant_records_form_2",
    "median_matching_values",
    "median_usable_values_form_1",
    "median_usable_values_form_2",
    "networks_represented",
]


def _write_form_map(path: Path, mapping: dict[str, str]) -> Path:
    """Write the same ``var_forms`` envelope used by grouped_variables.json."""
    path.write_text(
        json.dumps({"var_forms": mapping}, indent=2),
        encoding="utf-8",
    )
    return path


def _write_input(
    directory: Path,
    *,
    timepoint: str,
    network: str,
    rows: list[dict[str, object]],
) -> Path:
    network_token = {
        "PRONET": "ProNET",
        "PRESCIENT": "PRESCIENT",
    }[network]
    path = directory / (
        f"AMPSCZ-combined-redcap_{timepoint}_{network_token}-day1to1.csv"
    )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run(
    form_map: Path,
    inputs: list[Path],
    output: Path,
    *extra: str,
) -> int:
    arguments = ["--form-map", str(form_map)]
    for path in inputs:
        arguments.extend(("--input", str(path)))
    arguments.extend(("--output", str(output)))
    arguments.extend(extra)
    return scanner.main(arguments)


def _read_output(path: Path) -> pd.DataFrame:
    result = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert result.columns.tolist() == OUTPUT_COLUMNS
    return result


def _identity_pair(row) -> frozenset[tuple[str, str]]:
    return frozenset({
        (row.form_1, row.timepoint_1),
        (row.form_2, row.timepoint_2),
    })


def _row_for(
    result: pd.DataFrame,
    identity_1: tuple[str, str],
    identity_2: tuple[str, str],
) -> pd.Series:
    """Return the unique row for an unordered pair, including a self-pair."""
    form_1, timepoint_1 = identity_1
    form_2, timepoint_2 = identity_2
    direct = (
        (result["form_1"] == form_1)
        & (result["timepoint_1"] == timepoint_1)
        & (result["form_2"] == form_2)
        & (result["timepoint_2"] == timepoint_2)
    )
    reverse = (
        (result["form_1"] == form_2)
        & (result["timepoint_1"] == timepoint_2)
        & (result["form_2"] == form_1)
        & (result["timepoint_2"] == timepoint_1)
    )
    matches = result.loc[direct | reverse]
    assert len(matches) == 1
    return matches.iloc[0]


def test_multiset_dice_scores_are_aggregated_with_true_medians(
    tmp_path: Path,
) -> None:
    """A value occurrence can match at most one occurrence in the other form."""
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "a_1": "test_form",
        "a_2": "test_form",
        "a_3": "test_form",
        "b_1": "test_form",
        "b_2": "test_form",
    })
    baseline = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[
            {"subjectid": "S1", "a_1": " x ", "a_2": "x", "a_3": "y"},
            {"subjectid": "S2", "a_1": "x", "a_2": "y", "a_3": ""},
            {"subjectid": "S3", "a_1": "a", "a_2": "b", "a_3": "c"},
        ],
    )
    month_2 = _write_input(
        tmp_path,
        timepoint="month_2",
        network="PRONET",
        rows=[
            {"subjectid": "S1", "b_1": "x", "b_2": "z"},
            {"subjectid": "S2", "b_1": "x", "b_2": "y"},
            {"subjectid": "S3", "b_1": "z", "b_2": ""},
        ],
    )
    output = tmp_path / "matches.csv"

    assert _run(
        form_map, [baseline, month_2], output, "--chunk-size", "1"
    ) == 0

    result = _read_output(output)
    assert len(result) == 3
    row = _row_for(
        result,
        ("test_form", "baseline"),
        ("test_form", "month2"),
    )
    # Per-participant scores are 40%, 100%, and 0%; the true median is 40%.
    assert float(row["median_match_percentage"]) == pytest.approx(40.0)
    assert int(row["record_pairs_compared"]) == 3
    assert float(row["median_matching_values"]) == pytest.approx(1.0)
    assert float(row["median_usable_values_form_1"]) == pytest.approx(3.0)
    assert float(row["median_usable_values_form_2"]) == pytest.approx(2.0)
    assert row["networks_represented"] == "PRONET"


def test_only_same_source_forms_are_emitted_across_and_within_timepoints(
    tmp_path: Path,
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "blood_value": "blood_form",
        "mood_value": "mood_form",
    })
    baseline = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[
            {
                "subjectid": "S1",
                "blood_value": "a",
                "mood_value": "same",
            },
            {
                "subjectid": "S2",
                "blood_value": "b",
                "mood_value": "same",
            },
        ],
    )
    month_2 = _write_input(
        tmp_path,
        timepoint="month_2",
        network="PRONET",
        rows=[
            {"subjectid": "S1", "blood_value": "a"},
            {"subjectid": "S2", "blood_value": "c"},
        ],
    )
    output = tmp_path / "matches.csv"

    assert _run(form_map, [baseline, month_2], output) == 0

    result = _read_output(output)
    expected_pairs = {
        frozenset({("blood_form", "baseline")}),
        frozenset({("blood_form", "baseline"), ("blood_form", "month2")}),
        frozenset({("blood_form", "month2")}),
        frozenset({("mood_form", "baseline")}),
    }
    actual_pairs = {_identity_pair(row) for row in result.itertuples(index=False)}
    assert actual_pairs == expected_pairs
    assert len(result) == 4
    assert (result["form_1"] == result["form_2"]).all()

    cross_timepoint = _row_for(
        result,
        ("blood_form", "baseline"),
        ("blood_form", "month2"),
    )
    # Cross-timepoint comparisons continue to align records by participant.
    assert int(cross_timepoint["record_pairs_compared"]) == 2
    assert int(cross_timepoint["unique_participants_compared"]) == 2
    assert float(cross_timepoint["median_match_percentage"]) == pytest.approx(50.0)

    baseline_self = _row_for(
        result,
        ("blood_form", "baseline"),
        ("blood_form", "baseline"),
    )
    # A same-timepoint self-pair compares the two distinct participant records.
    assert int(baseline_self["record_pairs_compared"]) == 1
    assert int(baseline_self["unique_participants_compared"]) == 2
    assert float(baseline_self["median_match_percentage"]) == pytest.approx(0.0)


def test_same_form_instance_compares_every_distinct_participant_pair(
    tmp_path: Path,
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "blood_1": "blood_form",
        "blood_2": "blood_form",
    })
    pronet = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[
            {
                "subjectid": "S1",
                "blood_1": "a",
                "blood_2": "b",
            },
            {
                "subjectid": "S2",
                "blood_1": "a",
                "blood_2": "b",
            },
        ],
    )
    prescient = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRESCIENT",
        rows=[{
            # The participant key includes the network; this is a third form
            # instance even though its subject text duplicates PRONET S1.
            "subjectid": "S1",
            "blood_1": "a",
            "blood_2": "c",
        }],
    )
    output = tmp_path / "matches.csv"

    assert _run(form_map, [pronet, prescient], output) == 0

    result = _read_output(output)
    assert len(result) == 1
    row = _row_for(
        result,
        ("blood_form", "baseline"),
        ("blood_form", "baseline"),
    )
    # The three n-choose-2 scores are 100%, 50%, and 50%.  Comparing each
    # participant's row with itself would incorrectly produce three 100%s.
    assert float(row["median_match_percentage"]) == pytest.approx(50.0)
    assert float(row["mean_match_percentage"]) == pytest.approx(66.666667)
    assert int(row["record_pairs_compared"]) == 3
    assert int(row["unique_participants_compared"]) == 3
    assert int(row["participant_records_form_1"]) == 3
    assert int(row["participant_records_form_2"]) == 3
    assert float(row["median_matching_values"]) == pytest.approx(1.0)
    assert float(row["median_usable_values_form_1"]) == pytest.approx(2.0)
    assert float(row["median_usable_values_form_2"]) == pytest.approx(2.0)
    assert row["networks_represented"] == "PRESCIENT|PRONET"


def test_self_pair_profile_multiplicity_weights_every_record_pair(
    tmp_path: Path,
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "blood_value": "blood_form",
    })
    source = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[
            {"subjectid": "X1", "blood_value": "x"},
            {"subjectid": "X2", "blood_value": "x"},
            {"subjectid": "Y1", "blood_value": "y"},
            {"subjectid": "Y2", "blood_value": "y"},
        ],
    )
    output = tmp_path / "matches.csv"

    assert _run(form_map, [source], output) == 0

    row = _row_for(
        _read_output(output),
        ("blood_form", "baseline"),
        ("blood_form", "baseline"),
    )
    # Six participant pairs: two within-profile matches and four cross-profile
    # mismatches. Profile grouping must retain all six pairs as separate votes.
    assert int(row["record_pairs_compared"]) == 6
    assert int(row["unique_participants_compared"]) == 4
    assert float(row["median_match_percentage"]) == pytest.approx(0.0)
    assert float(row["mean_match_percentage"]) == pytest.approx(33.333333)



def test_default_output_keeps_unshared_pairs_and_flag_can_omit_them(
    tmp_path: Path,
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "value": "shared_form",
    })
    baseline = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[
            {"subjectid": "BASELINE_1", "value": "x"},
            {"subjectid": "BASELINE_2", "value": "x"},
        ],
    )
    month_2 = _write_input(
        tmp_path,
        timepoint="month_2",
        network="PRONET",
        rows=[{"subjectid": "MONTH2_ONLY", "value": "x"}],
    )

    complete_output = tmp_path / "all_pairs.csv"
    assert _run(form_map, [baseline, month_2], complete_output) == 0
    complete = _read_output(complete_output)
    assert len(complete) == 3
    assert sorted(complete["record_pairs_compared"].astype(int)) == [0, 0, 1]

    shared_output = tmp_path / "shared_pairs.csv"
    assert _run(
        form_map,
        [baseline, month_2],
        shared_output,
        "--omit-unshared-pairs",
    ) == 0
    shared = _read_output(shared_output)
    assert len(shared) == 1
    baseline_self = _row_for(
        shared,
        ("shared_form", "baseline"),
        ("shared_form", "baseline"),
    )
    assert int(baseline_self["record_pairs_compared"]) == 1


def test_checkbox_columns_map_by_base_and_missing_and_completion_values_are_opt_in(
    tmp_path: Path,
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "check_answer": "check_form",
        "check_form_complete": "check_form",
        "target_answer": "check_form",
        "target_missing": "check_form",
        "target_form_complete": "check_form",
    })
    baseline = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[{
            "subjectid": "S1",
            "check_answer___1": "shared",
            "check_answer___2": "-9",
            "check_form_complete": "2",
        }],
    )
    month_2 = _write_input(
        tmp_path,
        timepoint="month_2",
        network="PRONET",
        rows=[{
            "subjectid": "S1",
            "target_answer": "shared",
            "target_missing": "-9",
            "target_form_complete": "2",
        }],
    )
    inputs = [baseline, month_2]
    identities = (("check_form", "baseline"), ("check_form", "month2"))

    default_output = tmp_path / "default.csv"
    assert _run(form_map, inputs, default_output) == 0
    default = _row_for(_read_output(default_output), *identities)
    assert float(default["median_matching_values"]) == pytest.approx(1.0)
    assert float(default["median_usable_values_form_1"]) == pytest.approx(1.0)
    assert float(default["median_usable_values_form_2"]) == pytest.approx(1.0)

    missing_output = tmp_path / "with_missing.csv"
    assert _run(
        form_map, inputs, missing_output, "--include-missing-codes"
    ) == 0
    with_missing = _row_for(_read_output(missing_output), *identities)
    assert float(with_missing["median_matching_values"]) == pytest.approx(2.0)
    assert float(with_missing["median_usable_values_form_1"]) == pytest.approx(2.0)
    assert float(with_missing["median_usable_values_form_2"]) == pytest.approx(2.0)

    completion_output = tmp_path / "with_completion.csv"
    assert _run(
        form_map, inputs, completion_output, "--include-completion-fields"
    ) == 0
    with_completion = _row_for(_read_output(completion_output), *identities)
    assert float(with_completion["median_matching_values"]) == pytest.approx(2.0)
    assert float(with_completion["median_usable_values_form_1"]) == pytest.approx(2.0)
    assert float(with_completion["median_usable_values_form_2"]) == pytest.approx(2.0)


def test_numeric_equivalence_is_disabled_by_default_and_explicitly_opt_in(
    tmp_path: Path,
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "a_value": "numeric_form",
        "b_value": "numeric_form",
    })
    baseline = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[{"subjectid": "S1", "a_value": "1"}],
    )
    month_2 = _write_input(
        tmp_path,
        timepoint="month_2",
        network="PRONET",
        rows=[{"subjectid": "S1", "b_value": "1.0"}],
    )
    inputs = [baseline, month_2]
    identities = (("numeric_form", "baseline"), ("numeric_form", "month2"))

    exact_output = tmp_path / "exact.csv"
    assert _run(form_map, inputs, exact_output) == 0
    exact = _row_for(_read_output(exact_output), *identities)
    assert float(exact["median_match_percentage"]) == pytest.approx(0.0)
    assert float(exact["median_matching_values"]) == pytest.approx(0.0)

    numeric_output = tmp_path / "numeric.csv"
    assert _run(
        form_map, inputs, numeric_output, "--numeric-equivalence"
    ) == 0
    numeric = _row_for(_read_output(numeric_output), *identities)
    assert float(numeric["median_match_percentage"]) == pytest.approx(100.0)
    assert float(numeric["median_matching_values"]) == pytest.approx(1.0)


def test_same_subject_text_in_two_networks_is_two_participants(
    tmp_path: Path,
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "a_value": "shared_form",
        "b_value": "shared_form",
    })
    pronet_baseline = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[{"subjectid": "DUPLICATED_TEXT", "a_value": "same"}],
    )
    pronet_month_2 = _write_input(
        tmp_path,
        timepoint="month_2",
        network="PRONET",
        rows=[{"subjectid": "DUPLICATED_TEXT", "b_value": "same"}],
    )
    prescient_baseline = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRESCIENT",
        rows=[{"subjectid": "DUPLICATED_TEXT", "a_value": "same"}],
    )
    prescient_month_2 = _write_input(
        tmp_path,
        timepoint="month_2",
        network="PRESCIENT",
        rows=[{"subjectid": "DUPLICATED_TEXT", "b_value": "different"}],
    )
    output = tmp_path / "matches.csv"

    assert _run(
        form_map,
        [pronet_baseline, pronet_month_2, prescient_baseline, prescient_month_2],
        output,
    ) == 0

    row = _row_for(
        _read_output(output),
        ("shared_form", "baseline"),
        ("shared_form", "month2"),
    )
    assert int(row["record_pairs_compared"]) == 2
    assert float(row["median_match_percentage"]) == pytest.approx(50.0)
    assert float(row["median_matching_values"]) == pytest.approx(0.5)
    represented = {
        network
        for network in ("PRONET", "PRESCIENT")
        if network in row["networks_represented"].upper()
    }
    assert represented == {"PRONET", "PRESCIENT"}


def test_noncanonical_input_fails_cleanly_without_writing_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "a_value": "form_a",
        "b_value": "form_b",
    })
    source = tmp_path / "not-a-canonical-combined-file.csv"
    pd.DataFrame([{
        "subjectid": "S1",
        "a_value": "x",
        "b_value": "x",
    }]).to_csv(source, index=False)
    output = tmp_path / "matches.csv"

    assert _run(form_map, [source], output) == 2
    error = capsys.readouterr().err.casefold()
    assert "canonical" in error or "filename" in error
    assert not output.exists()


def test_missing_subject_identifier_fails_cleanly_without_writing_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "a_value": "form_a",
        "b_value": "form_b",
    })
    source = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[{
            "participant": "S1",
            "a_value": "x",
            "b_value": "x",
        }],
    )
    output = tmp_path / "matches.csv"

    assert _run(form_map, [source], output) == 2
    assert "subjectid" in capsys.readouterr().err.casefold()
    assert not output.exists()


def test_output_path_cannot_replace_an_input_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "a_value": "form_a",
        "b_value": "form_b",
    })
    source = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[{
            "subjectid": "S1",
            "a_value": "x",
            "b_value": "x",
        }],
    )
    original = source.read_bytes()

    assert _run(form_map, [source], source) == 2
    error = capsys.readouterr().err.casefold()
    assert "output" in error
    assert "input" in error
    assert source.read_bytes() == original


def test_output_path_cannot_replace_the_form_map(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "a_value": "form_a",
        "b_value": "form_b",
    })
    source = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[{
            "subjectid": "S1",
            "a_value": "x",
            "b_value": "x",
        }],
    )
    original = form_map.read_bytes()

    assert _run(form_map, [source], form_map) == 2
    error = capsys.readouterr().err.casefold()
    assert "output" in error
    assert "form" in error or "map" in error
    assert form_map.read_bytes() == original


def test_duplicate_participant_error_does_not_disclose_identifier(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "a_value": "form_a",
        "b_value": "form_b",
    })
    sensitive_id = "TOP_SECRET_SUBJECT_123"
    source = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[
            {
                "subjectid": sensitive_id,
                "a_value": "x",
                "b_value": "x",
            },
            {
                "subjectid": sensitive_id,
                "a_value": "y",
                "b_value": "y",
            },
        ],
    )
    output = tmp_path / "matches.csv"

    assert _run(form_map, [source], output) == 2
    error = capsys.readouterr().err
    assert "duplicate" in error.casefold()
    assert sensitive_id not in error
    assert not output.exists()


def test_explicit_config_rejects_noncanonical_testing_enabled(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "a_value": "form_a",
        "b_value": "form_b",
    })
    source = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[{
            "subjectid": "S1",
            "a_value": "x",
            "b_value": "x",
        }],
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"testing_enabled": "true"}),
        encoding="utf-8",
    )
    output = tmp_path / "matches.csv"

    assert _run(
        form_map,
        [source],
        output,
        "--config",
        str(config),
    ) == 2
    assert "testing_enabled" in capsys.readouterr().err.casefold()
    assert not output.exists()


def test_partial_input_directory_requires_explicit_opt_in(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "a_value": "form_a",
        "b_value": "form_b",
    })
    input_dir = tmp_path / "combined"
    input_dir.mkdir()
    _write_input(
        input_dir,
        timepoint="baseline",
        network="PRONET",
        rows=[{
            "subjectid": "S1",
            "a_value": "x",
            "b_value": "x",
        }],
    )
    output = tmp_path / "matches.csv"
    arguments = [
        "--form-map",
        str(form_map),
        "--input-dir",
        str(input_dir),
        "--output",
        str(output),
    ]

    assert scanner.main(arguments) == 2
    error = capsys.readouterr().err.casefold()
    assert "incomplete" in error
    assert "--allow-partial-input-dir" in error
    assert not output.exists()

    assert scanner.main(arguments + ["--allow-partial-input-dir"]) == 0
    assert output.is_file()
    result = _read_output(output)
    assert len(result) == 2
    assert (result["form_1"] == result["form_2"]).all()


def test_nat_and_pandas_na_text_are_missing_by_default(
    tmp_path: Path,
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "a_shared": "shared_form",
        "a_nat": "shared_form",
        "a_na": "shared_form",
        "b_shared": "shared_form",
        "b_nat": "shared_form",
        "b_na": "shared_form",
    })
    baseline = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[{
            "subjectid": "S1",
            "a_shared": "shared",
            "a_nat": "NaT",
            "a_na": "<NA>",
        }],
    )
    month_2 = _write_input(
        tmp_path,
        timepoint="month_2",
        network="PRONET",
        rows=[{
            "subjectid": "S1",
            "b_shared": "shared",
            "b_nat": "nat",
            "b_na": "<NA>",
        }],
    )
    output = tmp_path / "matches.csv"

    assert _run(form_map, [baseline, month_2], output) == 0

    row = _row_for(
        _read_output(output),
        ("shared_form", "baseline"),
        ("shared_form", "month2"),
    )
    assert float(row["median_match_percentage"]) == pytest.approx(100.0)
    assert float(row["median_matching_values"]) == pytest.approx(1.0)
    assert float(row["median_usable_values_form_1"]) == pytest.approx(1.0)
    assert float(row["median_usable_values_form_2"]) == pytest.approx(1.0)


def _write_important_form_vars(
    path: Path,
    forms: dict[str, dict[str, str]],
) -> Path:
    path.write_text(json.dumps(forms, indent=2), encoding="utf-8")
    return path



def test_admin_only_form_is_globally_pruned_but_existing_missing_instance_counts(
    tmp_path: Path,
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "a_value": "shared_form",
        "b_value": "shared_form",
        "ghost_value": "ghost_form",
        "ghost_missing": "ghost_form",
        "ghost_missing_spec": "ghost_form",
        "ghost_form_complete": "ghost_form",
    })
    important = _write_important_form_vars(
        tmp_path / "important_form_vars.json",
        {
            "shared_form": {
                "missing_var": "",
                "missing_spec_var": "",
                "completion_var": "shared_form_complete",
            },
            "ghost_form": {
                "missing_var": "ghost_missing",
                "missing_spec_var": "ghost_missing_spec",
                "completion_var": "ghost_form_complete",
            },
        },
    )
    baseline = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[
            {
                "subjectid": "S1",
                "a_value": "x",
                "ghost_value": "-9",
                "ghost_missing": "1",
                "ghost_missing_spec": "M2",
                "ghost_form_complete": "2",
            },
            {
                "subjectid": "S2",
                "a_value": "x",
                "ghost_value": "",
                "ghost_missing": "1",
                "ghost_missing_spec": "M2",
                "ghost_form_complete": "2",
            },
        ],
    )
    month_2 = _write_input(
        tmp_path,
        timepoint="month_2",
        network="PRONET",
        rows=[
            {"subjectid": "S1", "b_value": "x"},
            {"subjectid": "S2", "b_value": ""},
        ],
    )
    output = tmp_path / "matches.csv"

    assert _run(
        form_map,
        [baseline, month_2],
        output,
        "--important-form-vars",
        str(important),
    ) == 0

    result = _read_output(output)
    assert len(result) == 3
    row = _row_for(
        result,
        ("shared_form", "baseline"),
        ("shared_form", "month2"),
    )
    assert "ghost_form" not in set(result["form_1"]) | set(result["form_2"])
    assert int(row["record_pairs_compared"]) == 2
    assert float(row["median_match_percentage"]) == pytest.approx(50.0)
    assert float(row["median_matching_values"]) == pytest.approx(0.5)
    assert float(row["median_usable_values_form_1"]) == pytest.approx(1.0)
    assert float(row["median_usable_values_form_2"]) == pytest.approx(1.0)


def test_paired_missing_slots_cancel_but_one_sided_excess_stays_in_denominator(
    tmp_path: Path,
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "a_1": "shared_form",
        "a_2": "shared_form",
        "b_1": "shared_form",
        "b_2": "shared_form",
    })
    baseline_rows = [
        {"subjectid": "PAIRED_MISSING", "a_1": "x", "a_2": ""},
        {"subjectid": "ONE_SIDED_MISSING", "a_1": "x", "a_2": ""},
        {"subjectid": "NO_COMPARABLE_VALUES", "a_1": "", "a_2": "-9"},
    ]
    month_2_rows = [
        {"subjectid": "PAIRED_MISSING", "b_1": "x", "b_2": "-9"},
        {"subjectid": "ONE_SIDED_MISSING", "b_1": "x", "b_2": "y"},
        {"subjectid": "NO_COMPARABLE_VALUES", "b_1": "", "b_2": "999"},
    ]
    baseline = _write_input(
        tmp_path, timepoint="baseline", network="PRONET", rows=baseline_rows
    )
    month_2 = _write_input(
        tmp_path, timepoint="month_2", network="PRONET", rows=month_2_rows
    )
    output = tmp_path / "matches.csv"

    assert _run(form_map, [baseline, month_2], output) == 0

    row = _row_for(
        _read_output(output),
        ("shared_form", "baseline"),
        ("shared_form", "month2"),
    )
    # PAIRED_MISSING is 100%; ONE_SIDED_MISSING is 50%. The third
    # participant has a zero effective denominator and is not compared.
    assert int(row["record_pairs_compared"]) == 2
    assert float(row["median_match_percentage"]) == pytest.approx(75.0)
    assert float(row["median_matching_values"]) == pytest.approx(1.0)
    assert float(row["median_usable_values_form_1"]) == pytest.approx(1.5)
    assert float(row["median_usable_values_form_2"]) == pytest.approx(1.5)


def test_prescient_missing_completion_codes_do_not_seed_a_ghost_form(
    tmp_path: Path,
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "a_value": "shared_form",
        "b_value": "shared_form",
        "ghost_value": "ghost_form",
        "ghost_missing": "ghost_form",
        "ghost_missing_spec": "ghost_form",
        "ghost_form_complete_rpms": "ghost_form",
    })
    important = _write_important_form_vars(
        tmp_path / "important_form_vars.json",
        {
            "shared_form": {
                "missing_var": "",
                "missing_spec_var": "",
                "completion_var": "shared_form_complete",
            },
            "ghost_form": {
                "missing_var": "ghost_missing",
                "missing_spec_var": "ghost_missing_spec",
                "completion_var": "ghost_form_complete",
            },
        },
    )
    baseline = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRESCIENT",
        rows=[
            {
                "subjectid": "P3",
                "a_value": "same",
                "ghost_value": "",
                "ghost_missing": "0",
                "ghost_missing_spec": "",
                "ghost_form_complete_rpms": "3",
            },
            {
                "subjectid": "P4",
                "a_value": "same",
                "ghost_value": "-9",
                "ghost_missing": "0",
                "ghost_missing_spec": "",
                "ghost_form_complete_rpms": "4",
            },
        ],
    )
    month_2 = _write_input(
        tmp_path,
        timepoint="month_2",
        network="PRESCIENT",
        rows=[
            {"subjectid": "P3", "b_value": "same"},
            {"subjectid": "P4", "b_value": "same"},
        ],
    )
    output = tmp_path / "matches.csv"

    assert _run(
        form_map,
        [baseline, month_2],
        output,
        "--important-form-vars",
        str(important),
    ) == 0

    result = _read_output(output)
    assert len(result) == 3
    row = _row_for(
        result,
        ("shared_form", "baseline"),
        ("shared_form", "month2"),
    )
    assert "ghost_form" not in set(result["form_1"]) | set(result["form_2"])
    assert int(row["record_pairs_compared"]) == 2
    assert float(row["median_match_percentage"]) == pytest.approx(100.0)
    assert row["networks_represented"] == "PRESCIENT"


def test_cross_timepoint_same_form_compares_only_matching_participant_keys(
    tmp_path: Path,
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "a_value": "shared_form",
        "b_value": "shared_form",
    })
    baseline = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[
            {"subjectid": "S1", "a_value": "alpha"},
            {"subjectid": "S2", "a_value": "beta"},
        ],
    )
    month_2 = _write_input(
        tmp_path,
        timepoint="month_2",
        network="PRONET",
        rows=[
            {"subjectid": "S1", "b_value": "alpha"},
            {"subjectid": "S2", "b_value": "beta"},
        ],
    )
    output = tmp_path / "matches.csv"

    assert _run(form_map, [baseline, month_2], output) == 0

    row = _row_for(
        _read_output(output),
        ("shared_form", "baseline"),
        ("shared_form", "month2"),
    )
    # Same-participant comparisons are both 100%. A Cartesian product across
    # participants would add two 0% comparisons and produce a 50% median.
    assert float(row["median_match_percentage"]) == pytest.approx(100.0)
    assert int(row["record_pairs_compared"]) == 2
    assert int(row["unique_participants_compared"]) == 2
    assert int(row["participant_records_form_1"]) == 2
    assert int(row["participant_records_form_2"]) == 2


def test_self_pair_penalizes_one_sided_missing_and_skips_both_missing(
    tmp_path: Path,
) -> None:
    form_map = _write_form_map(tmp_path / "grouped_variables.json", {
        "blood_1": "blood_form",
        "blood_2": "blood_form",
    })
    source = _write_input(
        tmp_path,
        timepoint="baseline",
        network="PRONET",
        rows=[
            {
                "subjectid": "USABLE",
                "blood_1": "x",
                "blood_2": "",
            },
            {
                "subjectid": "ALL_MISSING_1",
                "blood_1": "",
                "blood_2": "-9",
            },
            {
                "subjectid": "ALL_MISSING_2",
                "blood_1": "-9",
                "blood_2": "999",
            },
        ],
    )
    output = tmp_path / "matches.csv"

    assert _run(form_map, [source], output) == 0

    row = _row_for(
        _read_output(output),
        ("blood_form", "baseline"),
        ("blood_form", "baseline"),
    )
    # USABLE versus each all-missing record is a one-sided-missing 0%.
    # The all-missing versus all-missing pair has no denominator and is skipped.
    assert float(row["median_match_percentage"]) == pytest.approx(0.0)
    assert int(row["record_pairs_compared"]) == 2
    assert int(row["unique_participants_compared"]) == 3
    assert int(row["participant_records_form_1"]) == 3
    assert int(row["participant_records_form_2"]) == 3
    assert float(row["median_matching_values"]) == pytest.approx(0.0)
    assert float(row["median_usable_values_form_1"]) == pytest.approx(1.0)
    assert float(row["median_usable_values_form_2"]) == pytest.approx(1.0)
    assert row["networks_represented"] == "PRONET"
