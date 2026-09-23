"""Regression coverage for the calc-diff before/after pair builder."""

import csv

import pytest

from extract_calc_diff_value_pairs import (
    DIFF_COLUMNS,
    build_calc_diff_pairs,
    build_network_result,
)
from merge_blank_flag_pairs import PAIR_COLUMNS, PairMergeError


def _write_diff(diff_dir, network_lower, rows):
    path = diff_dir / f"diffs_test_revised_new_module_{network_lower}.csv"
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(DIFF_COLUMNS)
        writer.writerows(rows)


def _write_combined(combined_dir, filename, rows, columns):
    with (combined_dir / filename).open("w", encoding="utf-8",
                                        newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def _pair_map(result):
    return {
        row[PAIR_COLUMNS.index("variable")]: dict(zip(PAIR_COLUMNS, row))
        for row in result.rows
    }


@pytest.fixture()
def setup_dirs(tmp_path):
    diff_dir = tmp_path / "diffs"
    combined_dir = tmp_path / "combined"
    diff_dir.mkdir()
    combined_dir.mkdir()
    _write_combined(
        combined_dir,
        "AMPSCZ-combined-redcap_baseline_ProNET-day1to1.csv",
        [["S1", "12", "", "5"]],
        ["subjectid", "calc_a", "calc_b", "calc_c"],
    )
    return diff_dir, combined_dir


def test_calc_diff_pairs_use_comb_value_as_before_and_live_as_current(
        setup_dirs):
    diff_dir, combined_dir = setup_dirs
    _write_diff(diff_dir, "pronet", [
        # before 10, live now 12 -> a real correction pair.
        ["S1", "baseline", "calc_a", "10", "11"],
        # duplicate rows with the same before value collapse.
        ["S1", "baseline", "calc_b", "7", "8"],
        ["S1", "baseline", "calc_b", "7", "9"],
        # conflicting before values fail closed into the audit.
        ["S1", "baseline", "calc_c", "1", "2"],
        ["S1", "baseline", "calc_c", "3", "2"],
        # identity columns and unknown subjects are skipped.
        ["S1", "baseline", "subjectid", "X", "Y"],
        ["S9", "baseline", "calc_a", "4", "5"],
    ])

    result = build_network_result(diff_dir, combined_dir, "PRONET")
    pairs = _pair_map(result)

    assert set(pairs) == {"calc_a", "calc_b"}
    assert pairs["calc_a"]["before_value"] == "10"
    assert pairs["calc_a"]["current_value"] == "12"
    assert pairs["calc_a"]["status"] == "filled"
    assert pairs["calc_b"]["before_value"] == "7"
    assert pairs["calc_b"]["current_value"] == ""
    assert pairs["calc_b"]["status"] == "still_blank"
    assert [row[2] for row in result.conflicts] == ["calc_c"]
    assert result.counts["skipped_identity_column_target"] == 1
    assert result.counts["skipped_subject_missing"] == 1
    # recalc_val must never leak into the pair artifact.
    flattened = {value for row in result.rows for value in row}
    assert {"11", "8", "9"}.isdisjoint(flattened)


def test_after_source_recalc_uses_export_value_as_current(setup_dirs):
    diff_dir, combined_dir = setup_dirs
    _write_diff(diff_dir, "pronet", [
        ["S1", "baseline", "calc_a", "10", "11"],
        # Missing from the live CSVs -> still dropped (staging needs the
        # target cell to exist), even in recalc mode.
        ["S9", "baseline", "calc_a", "4", "5"],
    ])

    result = build_network_result(
        diff_dir, combined_dir, "PRONET", after_source="recalc")
    pairs = _pair_map(result)

    assert set(pairs) == {"calc_a"}
    assert pairs["calc_a"]["before_value"] == "10"
    assert pairs["calc_a"]["current_value"] == "11"
    assert result.counts["skipped_subject_missing"] == 1


def test_only_changed_drops_noop_pairs(setup_dirs):
    diff_dir, combined_dir = setup_dirs
    _write_diff(diff_dir, "pronet", [
        ["S1", "baseline", "calc_a", "10", "11"],   # live 12 -> changed
        ["S1", "baseline", "calc_c", "5", "6"],     # live 5 -> no-op
    ])

    result = build_network_result(
        diff_dir, combined_dir, "PRONET", only_changed=True)
    pairs = _pair_map(result)

    assert set(pairs) == {"calc_a"}
    assert result.counts["pairs_skipped_as_noop"] == 1


def test_publish_refuses_existing_output_directory(setup_dirs, tmp_path):
    diff_dir, combined_dir = setup_dirs
    _write_diff(diff_dir, "pronet", [["S1", "baseline", "calc_a", "10", "11"]])
    _write_diff(diff_dir, "prescient", [["S1", "baseline", "calc_a", "2", "3"]])
    _write_combined(
        combined_dir,
        "AMPSCZ-combined-redcap_baseline_PRESCIENT-day1to1.csv",
        [["S1", "4"]],
        ["subjectid", "calc_a"],
    )
    output_dir = tmp_path / "pairs_out"

    results = build_calc_diff_pairs(diff_dir, combined_dir, output_dir)

    assert {result.network for result in results} == {"PRONET", "PRESCIENT"}
    published = sorted(path.name for path in output_dir.iterdir())
    assert "resolved_before_after_values_PRONET.csv" in published
    assert "calc_diff_pair_summary.json" in published

    with pytest.raises(PairMergeError):
        build_calc_diff_pairs(diff_dir, combined_dir, output_dir)
