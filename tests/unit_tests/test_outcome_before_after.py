"""Focused synthetic coverage for the before/current outcome experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

import outcome_before_after as before_after
from outcome_checkpoints import CHECKPOINT_COLUMNS
from outcome_inputs import OUTCOME_TIMEPOINTS, OutcomeInputError, combined_csv_file


NETWORK = "PRONET"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair_row(**updates: str) -> dict[str, str]:
    row = {
        "Subject": "S1",
        "Timepoint": "baseline",
        "General_Flag": "example_form",
        "variable": "field_a",
        "message_variable": "field_a",
        "canonical_template": "example template",
        "before_value_available": "True",
        "before_value": "old",
        "before_value_source": "message_pattern",
        "Latest_seen": "2026-01-01",
        "status": "filled",
        "current_value": "new",
    }
    row.update(updates)
    return row


def _write_pairs(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=list(before_after.REQUIRED_PAIR_COLUMNS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_combined_slices(
    root: Path,
    *,
    baseline_a: str = "new",
    month1_b: str = "month-one-current",
) -> None:
    root.mkdir(parents=True)
    for timepoint in OUTCOME_TIMEPOINTS:
        path = combined_csv_file(root, NETWORK, timepoint)
        with path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(
                ["subjectid", "chrcrit_part", "field_a", "field_b", "note"]
            )
            writer.writerow(
                [
                    "S1",
                    "1",
                    baseline_a if timepoint == "baseline" else f"S1-{timepoint}",
                    "S1-b",
                    "untouched",
                ]
            )
            writer.writerow(
                [
                    "S2",
                    "1",
                    "S2-a",
                    month1_b if timepoint == "month1" else f"S2-{timepoint}",
                    "untouched",
                ]
            )


def _read_cell(path: Path, subject: str, variable: str) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source)
        matches = [row for row in rows if row["subjectid"] == subject]
    assert len(matches) == 1
    return matches[0][variable]


def _source_hashes(root: Path) -> dict[str, str]:
    return {
        timepoint: _sha256(combined_csv_file(root, NETWORK, timepoint))
        for timepoint in OUTCOME_TIMEPOINTS
    }


def _write_outcomes(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(CHECKPOINT_COLUMNS)
        writer.writerows(rows)


def test_load_value_replacements_preserves_exact_values_and_deduplicates(
    tmp_path: Path,
) -> None:
    pair_path = tmp_path / "resolved_before_after_values_PRONET.csv"
    exact_before = "  0012, 'quoted'\nnext\\line NA  "
    exact_current = " current,\"quoted\"\nsecond line "
    duplicate = _pair_row(
        before_value=exact_before,
        current_value=exact_current,
    )
    _write_pairs(
        pair_path,
        [
            duplicate,
            dict(duplicate, canonical_template="another flag episode"),
            _pair_row(
                Subject="S2",
                Timepoint="month_1",
                variable="field_b",
                message_variable="field_b",
                status="still_blank",
                before_value="",
                current_value="",
            ),
        ],
    )

    replacements = before_after.load_value_replacements(pair_path, NETWORK)

    assert len(replacements) == 2
    first, second = replacements
    assert first.key == ("baseline", "S1", "field_a")
    assert first.before_value == exact_before
    assert first.current_value == exact_current
    assert first.source_entry_count == 2
    assert not first.is_noop
    assert second.key == ("month1", "S2", "field_b")
    assert second.before_value == ""
    assert second.current_value == ""
    assert second.is_noop


@pytest.mark.parametrize(
    ("changed_column", "sensitive_value"),
    [
        ("before_value", "PRIVATE_CONFLICTING_ORIGINAL"),
        ("current_value", "PRIVATE_CONFLICTING_CURRENT"),
    ],
)
def test_load_value_replacements_rejects_conflicting_duplicate_cells_count_only(
    tmp_path: Path,
    changed_column: str,
    sensitive_value: str,
) -> None:
    sensitive_subject = "PRIVATE_SUBJECT_CONFLICT"
    pair_path = tmp_path / "resolved_before_after_values_PRONET.csv"
    first = _pair_row(Subject=sensitive_subject)
    second = dict(first, **{changed_column: sensitive_value})
    _write_pairs(pair_path, [first, second])

    with pytest.raises(OutcomeInputError) as exc_info:
        before_after.load_value_replacements(pair_path, NETWORK)

    diagnostic = str(exc_info.value)
    assert "1 conflicting duplicate target row(s)" in diagnostic
    assert sensitive_subject not in diagnostic
    assert sensitive_value not in diagnostic


def test_load_value_replacements_reports_unsafe_rows_without_private_fields(
    tmp_path: Path,
) -> None:
    sensitive_subject = "PRIVATE_UNSAFE_SUBJECT"
    sensitive_value = "PRIVATE_UNSAFE_VALUE"
    pair_path = tmp_path / "resolved_before_after_values_PRONET.csv"
    _write_pairs(
        pair_path,
        [
            _pair_row(
                Subject=sensitive_subject,
                variable="subjectid",
                message_variable="subjectid",
                before_value=sensitive_value,
            ),
            _pair_row(
                Subject="",
                Timepoint="multiple_timepoints",
                before_value_available="False",
                status="subject_missing",
            ),
        ],
    )

    with pytest.raises(OutcomeInputError) as exc_info:
        before_after.load_value_replacements(pair_path, NETWORK)

    diagnostic = str(exc_info.value)
    assert "unsafe rows" in diagnostic
    assert "1 identity-column target row(s)" in diagnostic
    assert "1 unsupported timepoint row(s)" in diagnostic
    assert sensitive_subject not in diagnostic
    assert sensitive_value not in diagnostic


def test_build_value_scenarios_stages_all_slices_without_mutating_source(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "combined"
    scenario_root = tmp_path / "experiment"
    pair_path = tmp_path / "resolved_before_after_values_PRONET.csv"
    month1_current = "  0007, value\nwith newline  "
    _write_combined_slices(
        source_root,
        baseline_a="new",
        month1_b=month1_current,
    )
    _write_pairs(
        pair_path,
        [
            _pair_row(before_value="old", current_value="new"),
            _pair_row(
                Subject="S2",
                Timepoint="month1",
                variable="field_b",
                message_variable="field_b",
                before_value="",
                current_value=month1_current,
            ),
        ],
    )
    hashes_before = _source_hashes(source_root)
    replacements = before_after.load_value_replacements(pair_path, NETWORK)

    layout = before_after.build_value_scenarios(
        source_root,
        pair_path,
        scenario_root,
        NETWORK,
        replacements,
    )

    assert _source_hashes(source_root) == hashes_before
    assert len(list(layout.original_combined.glob("*.csv"))) == len(
        OUTCOME_TIMEPOINTS
    )
    assert len(list(layout.current_combined.glob("*.csv"))) == len(
        OUTCOME_TIMEPOINTS
    )
    for timepoint in OUTCOME_TIMEPOINTS:
        source_path = combined_csv_file(source_root, NETWORK, timepoint)
        current_path = combined_csv_file(
            layout.current_combined, NETWORK, timepoint
        )
        assert current_path.read_bytes() == source_path.read_bytes()
        if timepoint not in {"baseline", "month1"}:
            original_path = combined_csv_file(
                layout.original_combined, NETWORK, timepoint
            )
            assert original_path.read_bytes() == source_path.read_bytes()

    assert _read_cell(
        combined_csv_file(layout.original_combined, NETWORK, "baseline"),
        "S1",
        "field_a",
    ) == "old"
    assert _read_cell(
        combined_csv_file(layout.original_combined, NETWORK, "month1"),
        "S2",
        "field_b",
    ) == ""
    assert _read_cell(
        combined_csv_file(layout.current_combined, NETWORK, "month1"),
        "S2",
        "field_b",
    ) == month1_current

    manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    assert len(manifest["source_files"]) == len(OUTCOME_TIMEPOINTS)
    assert len(
        manifest["scenario_files"][before_after.ORIGINAL_SCENARIO]
    ) == len(OUTCOME_TIMEPOINTS)
    assert manifest["counts"] == {
        "affected_subjects": 2,
        "by_timepoint": {"baseline": 1, "month1": 1},
        "no_op_pairs": 0,
        "outcome_roster": 2,
        "source_flag_entries": 2,
        "unique_value_pairs": 2,
    }


def test_build_value_scenarios_after_from_pairs_overlays_both_sides(
    tmp_path: Path,
) -> None:
    # The live cell deliberately matches NEITHER side of the pair: classic
    # mode would abort on the stale current-value check, while the
    # before-vs-after mode overlays each side onto the live base.
    source_root = tmp_path / "combined"
    scenario_root = tmp_path / "experiment"
    pair_path = tmp_path / "resolved_before_after_values_PRONET.csv"
    _write_combined_slices(source_root, baseline_a="live-now")
    _write_pairs(
        pair_path,
        [_pair_row(before_value="old", current_value="corrected")],
    )

    layout = before_after.build_value_scenarios(
        source_root,
        pair_path,
        scenario_root,
        NETWORK,
        after_from_pairs=True,
    )

    assert _read_cell(
        combined_csv_file(layout.original_combined, NETWORK, "baseline"),
        "S1",
        "field_a",
    ) == "old"
    assert _read_cell(
        combined_csv_file(layout.current_combined, NETWORK, "baseline"),
        "S1",
        "field_a",
    ) == "corrected"
    # Untouched cells still come from the live base on both sides.
    assert _read_cell(
        combined_csv_file(layout.current_combined, NETWORK, "baseline"),
        "S2",
        "field_a",
    ) == "S2-a"
    manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    assert manifest["after_mode"] == "pair_current_values"


def test_build_value_scenarios_stale_current_value_is_atomic_and_private(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "combined"
    scenario_root = tmp_path / "experiment"
    sensitive_subject = "PRIVATE_STALE_SUBJECT"
    sensitive_before = "PRIVATE_STALE_BEFORE"
    sensitive_current = "PRIVATE_STALE_CURRENT"
    pair_path = tmp_path / "resolved_before_after_values_PRONET.csv"
    _write_combined_slices(source_root, baseline_a="actual-current")
    _write_pairs(
        pair_path,
        [
            _pair_row(
                Subject=sensitive_subject,
                before_value=sensitive_before,
                current_value=sensitive_current,
            )
        ],
    )
    hashes_before = _source_hashes(source_root)

    with pytest.raises(OutcomeInputError) as exc_info:
        before_after.build_value_scenarios(
            source_root, pair_path, scenario_root, NETWORK
        )

    diagnostic = str(exc_info.value)
    assert "1 missing or duplicate subject target(s)" in diagnostic
    assert sensitive_subject not in diagnostic
    assert sensitive_before not in diagnostic
    assert sensitive_current not in diagnostic
    assert not scenario_root.exists()
    assert not list(tmp_path.glob(".experiment.*"))
    assert _source_hashes(source_root) == hashes_before


def test_build_value_scenarios_stale_cell_mismatch_is_count_only(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "combined"
    scenario_root = tmp_path / "experiment"
    sensitive_before = "PRIVATE_BEFORE_VALUE"
    sensitive_current = "PRIVATE_EXPECTED_CURRENT"
    pair_path = tmp_path / "resolved_before_after_values_PRONET.csv"
    _write_combined_slices(source_root, baseline_a="different-current")
    _write_pairs(
        pair_path,
        [
            _pair_row(
                before_value=sensitive_before,
                current_value=sensitive_current,
            )
        ],
    )

    with pytest.raises(OutcomeInputError) as exc_info:
        before_after.build_value_scenarios(
            source_root, pair_path, scenario_root, NETWORK
        )

    diagnostic = str(exc_info.value)
    assert "1 stale current-value mismatch(es)" in diagnostic
    assert sensitive_before not in diagnostic
    assert sensitive_current not in diagnostic
    assert not scenario_root.exists()
    assert not list(tmp_path.glob(".experiment.*"))


def test_run_value_scenarios_wires_distinct_inputs_outputs_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = before_after.ScenarioLayout.for_root(tmp_path / "experiment")
    layout.root.mkdir(parents=True)
    layout.original_combined.mkdir(parents=True)
    layout.current_combined.mkdir(parents=True)
    layout.affected_subject_roster.parent.mkdir(parents=True)
    layout.affected_subject_roster.write_text("Subject\nS1\n", encoding="utf-8")
    checkpoint = (
        layout.original_outcomes
        / ".outcome_checkpoints"
        / "run"
        / "manifest.json"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}\n", encoding="utf-8")
    layout.manifest.write_text(
        json.dumps(
            {
                "state": {
                    before_after.ORIGINAL_SCENARIO: "prepared",
                    before_after.CURRENT_SCENARIO: "prepared",
                    "comparison": "pending",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        before_after,
        "_completed_outcome_run_record",
        lambda *_args: {"test_record": True},
    )
    monkeypatch.setattr(
        before_after,
        "_checkpoint_input_files_for_scenario",
        lambda *_args: (),
    )
    reserve_checks: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        before_after,
        "_require_disk_reserve",
        lambda path, required: reserve_checks.append((path, required)),
    )
    commands: list[list[str]] = []

    before_after.run_value_scenarios(
        layout,
        NETWORK,
        workers=7,
        resume=True,
        command_runner=lambda command: commands.append(list(command)),
    )

    assert len(commands) == 2
    assert reserve_checks == [
        (layout.root, before_after.DEFAULT_OUTCOME_DISK_RESERVE_BYTES),
        (layout.root, before_after.DEFAULT_OUTCOME_DISK_RESERVE_BYTES),
    ]
    original_command, current_command = commands
    assert original_command[:4] == [
        sys.executable,
        str(Path(before_after.__file__).with_name("outcome_calculations.py")),
        NETWORK,
        "run_outcome",
    ]
    assert original_command[original_command.index("--combined-csv-path") + 1] == str(
        layout.original_combined
    )
    assert original_command[original_command.index("--output-dir") + 1] == str(
        layout.original_outcomes
    )
    assert original_command[-3:] == ["--workers", "7", "--resume"]
    assert current_command[current_command.index("--combined-csv-path") + 1] == str(
        layout.current_combined
    )
    assert current_command[current_command.index("--output-dir") + 1] == str(
        layout.current_outcomes
    )
    assert current_command[-2:] == ["--workers", "7"]
    assert "--resume" not in current_command
    state = json.loads(layout.manifest.read_text(encoding="utf-8"))["state"]
    assert state[before_after.ORIGINAL_SCENARIO] == "complete"
    assert state[before_after.CURRENT_SCENARIO] == "complete"


def test_compare_outcome_files_streams_aligned_rows_and_exact_values(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.csv"
    current = tmp_path / "current.csv"
    output = tmp_path / "comparison"
    exact_before = "before, value\nsecond line"
    exact_current = "after, value\nsecond line"
    _write_outcomes(
        original,
        [
            ["Integer", "baseline_arm_1", "1", "same_score", "S1"],
            ["Text", "month_1_arm_1", exact_before, "changed_score", "S1"],
            ["Integer", "month_2_arm_1", "-900", "changed_score", "S2"],
        ],
    )
    _write_outcomes(
        current,
        [
            ["Integer", "baseline_arm_1", "1", "same_score", "S1"],
            ["Text", "month_1_arm_1", exact_current, "changed_score", "S1"],
            ["Integer", "month_2_arm_1", "2", "changed_score", "S2"],
        ],
    )

    result = before_after.compare_outcome_files(original, current, output)

    assert result == {
        "total_outcome_rows": 3,
        "changed_outcome_rows": 2,
        "added_outcome_rows": 0,
        "removed_outcome_rows": 0,
        "changed_subjects": 2,
    }
    with (output / "outcome_value_changes.csv").open(
        "r", encoding="utf-8", newline=""
    ) as source:
        changes = list(csv.DictReader(source))
    assert len(changes) == 2
    assert [row["change_type"] for row in changes] == ["changed", "changed"]
    assert changes[0]["original_outcome_value"] == exact_before
    assert changes[0]["current_outcome_value"] == exact_current
    assert changes[1]["original_outcome_value"] == "-900"
    assert changes[1]["current_outcome_value"] == "2"
    with (output / "outcome_change_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as source:
        summary = list(csv.DictReader(source))
    assert summary == [
        {
            "data_type": "Integer",
            "variable": "changed_score",
            "changed_outcome_rows": "1",
            "added_outcome_rows": "0",
            "removed_outcome_rows": "0",
        },
        {
            "data_type": "Text",
            "variable": "changed_score",
            "changed_outcome_rows": "1",
            "added_outcome_rows": "0",
            "removed_outcome_rows": "0",
        },
    ]
    marker = json.loads(
        (output / "outcome_comparison_summary.json").read_text(encoding="utf-8")
    )
    assert marker["status"] == "complete"
    assert {name: marker[name] for name in result} == result


def test_compare_outcome_files_alignment_failure_is_atomic_and_private(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.csv"
    current = tmp_path / "current.csv"
    output = tmp_path / "comparison"
    output.mkdir()
    existing = output / "outcome_value_changes.csv"
    existing.write_text("last-good\n", encoding="utf-8")
    private_before = "PRIVATE_BEFORE_OUTCOME"
    private_after = "PRIVATE_AFTER_OUTCOME"
    private_subject = "PRIVATE_SUBJECT_OUTCOME"
    private_current_subject = "PRIVATE_CURRENT_SUBJECT_OUTCOME"
    _write_outcomes(
        original,
        [["Text", "baseline_arm_1", private_before, "score_a", private_subject]],
    )
    _write_outcomes(
        current,
        [
            [
                "Text",
                "baseline_arm_1",
                private_after,
                "score_b",
                private_current_subject,
            ]
        ],
    )

    with pytest.raises(OutcomeInputError) as exc_info:
        before_after.compare_outcome_files(original, current, output)

    diagnostic = str(exc_info.value)
    assert "different ordered subject rosters" in diagnostic
    assert private_before not in diagnostic
    assert private_after not in diagnostic
    assert private_subject not in diagnostic
    assert private_current_subject not in diagnostic
    assert existing.read_text(encoding="utf-8") == "last-good\n"
    assert not list(output.glob(".*.tmp"))
    assert not (output / "outcome_change_summary.csv").exists()
    marker_path = output / "outcome_comparison_summary.json"
    assert marker_path.exists()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["status"] == "running"
    assert "started_at" in marker
