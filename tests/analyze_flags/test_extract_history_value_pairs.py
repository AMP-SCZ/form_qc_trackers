"""Regression coverage for the history-wide before/after pair harvest."""

import csv

import pandas as pd
import pytest

from extract_history_value_pairs import (
    HISTORY_BASENAME,
    build_history_value_pairs,
    harvest_network,
)
from merge_blank_flag_pairs import PAIR_COLUMNS, PairMergeError


HISTORY_COLUMNS = [
    "Subject", "Timepoint", "General_Flag", "variable", "message",
    "canonical_template", "episode_id", "Earliest_seen", "Latest_seen",
    "Resolution_observed", "source", "is_currently_open",
    "message_variable", "message_source", "message_value_conflict",
    "resolution_eligible",
]


def _history_row(subject, variable, message, *, timepoint="baseline",
                 latest_seen="2026-01-05 00:00:00+00:00",
                 message_variable="", conflict="False"):
    return {
        "Subject": subject,
        "Timepoint": timepoint,
        "General_Flag": "example_form",
        "variable": variable,
        "message": message,
        "canonical_template": "<var> : template",
        "episode_id": "ep_x",
        "Earliest_seen": "2026-01-01 00:00:00+00:00",
        "Latest_seen": latest_seen,
        "Resolution_observed": "",
        "source": "V1",
        "is_currently_open": "False",
        "message_variable": message_variable or variable,
        "message_source": "V1",
        "message_value_conflict": conflict,
        "resolution_eligible": "True",
    }


def _write_history(history_dir, network, rows):
    frame = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    frame.to_csv(history_dir / HISTORY_BASENAME.format(network=network),
                 index=False)


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
def pronet_setup(tmp_path):
    history_dir = tmp_path / "history"
    combined_dir = tmp_path / "combined"
    history_dir.mkdir()
    combined_dir.mkdir()
    _write_combined(
        combined_dir,
        "AMPSCZ-combined-redcap_baseline_ProNET-day1to1.csv",
        [["S1", "5", "2", "", "1"]],
        ["subjectid", "field_blank", "field_latest", "field_conflict",
         "field_open"],
    )
    return history_dir, combined_dir


def test_harvest_recovers_all_provable_families_once_per_cell(pronet_setup):
    history_dir, combined_dir = pronet_setup
    rows = [
        # Blank family: recovered as an exact empty string.
        _history_row("S1", "field_blank", "Value is empty"),
        # Latest observation wins over an older different value.
        _history_row("S1", "field_latest", "field_latest is equal to 7.",
                     latest_seen="2026-01-02 00:00:00+00:00"),
        _history_row("S1", "field_latest", "field_latest is equal to 9.",
                     latest_seen="2026-01-06 00:00:00+00:00"),
        # Distinct values tied at the same latest timestamp fail closed.
        _history_row("S1", "field_conflict", "field_conflict is equal to 1."),
        _history_row("S1", "field_conflict", "field_conflict is equal to 2."),
        # A message-level conflict marker is skipped outright.
        _history_row("S1", "field_open", "field_open is equal to 3.",
                     conflict="True"),
        # Unprovable prose is counted but never paired.
        _history_row("S1", "field_open", "Related fields disagree."),
        # Cross-timepoint entries cannot map to one canonical file.
        _history_row("S1", "field_blank", "Value is empty",
                     timepoint="multiple_timepoints"),
        # Subjects absent from the combined CSV are counted and skipped.
        _history_row("S9", "field_blank", "Value is empty"),
    ]
    _write_history(history_dir, "PRONET", rows)

    result = harvest_network(history_dir, combined_dir, "PRONET")
    pairs = _pair_map(result)

    assert set(pairs) == {"field_blank", "field_latest"}
    assert pairs["field_blank"]["before_value"] == ""
    assert pairs["field_blank"]["before_value_source"] == "blank_flag"
    assert pairs["field_blank"]["current_value"] == "5"
    assert pairs["field_blank"]["status"] == "filled"
    assert pairs["field_latest"]["before_value"] == "9"
    assert pairs["field_latest"]["current_value"] == "2"
    assert [row[2] for row in result.conflicts] == ["field_conflict"]

    counts = result.counts
    assert counts["history_entries"] == len(rows)
    assert counts["skipped_message_value_conflict"] == 1
    assert counts["skipped_multiple_timepoints"] == 1
    assert counts["entries_without_provable_value"] == 1
    assert counts["cells_with_conflicting_latest_values"] == 1
    assert counts["skipped_subject_missing"] == 1
    assert counts["pairs_written"] == 2


def test_legacy_history_schema_without_message_columns_still_harvests(
        pronet_setup):
    history_dir, combined_dir = pronet_setup
    legacy_columns = [
        "Subject", "Timepoint", "General_Flag", "variable", "message",
        "canonical_template", "Earliest_seen", "Latest_seen", "source",
        "is_currently_open",
    ]
    row = _history_row("S1", "field_blank", "Value is empty")
    frame = pd.DataFrame([row], columns=HISTORY_COLUMNS)[legacy_columns]
    frame.to_csv(history_dir / HISTORY_BASENAME.format(network="PRONET"),
                 index=False)

    result = harvest_network(history_dir, combined_dir, "PRONET")
    pairs = _pair_map(result)

    assert pairs["field_blank"]["before_value"] == ""
    assert pairs["field_blank"]["message_variable"] == "field_blank"


def test_unchanged_values_still_pair_and_are_counted(pronet_setup):
    history_dir, combined_dir = pronet_setup
    _write_history(history_dir, "PRONET", [
        _history_row("S1", "field_latest", "field_latest is equal to 2."),
    ])

    result = harvest_network(history_dir, combined_dir, "PRONET")

    assert result.counts["pairs_written"] == 1
    assert result.counts["pairs_where_before_equals_current"] == 1


def test_publish_refuses_existing_output_directory(pronet_setup, tmp_path):
    history_dir, combined_dir = pronet_setup
    for network in ("PRONET", "PRESCIENT"):
        _write_history(history_dir, network, [
            _history_row("S1", "field_blank", "Value is empty"),
        ])
    _write_combined(
        combined_dir,
        "AMPSCZ-combined-redcap_baseline_PRESCIENT-day1to1.csv",
        [["S1", ""]],
        ["subjectid", "field_blank"],
    )
    output_dir = tmp_path / "pairs_out"

    results = build_history_value_pairs(history_dir, combined_dir, output_dir)

    assert {result.network for result in results} == {"PRONET", "PRESCIENT"}
    published = sorted(path.name for path in output_dir.iterdir())
    assert "resolved_before_after_values_PRONET.csv" in published
    assert "history_value_pair_summary.json" in published

    with pytest.raises(PairMergeError):
        build_history_value_pairs(history_dir, combined_dir, output_dir)
