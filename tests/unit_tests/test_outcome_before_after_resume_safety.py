"""Additional fail-closed tests for the before/current outcome experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import outcome_before_after as before_after
from outcome_checkpoints import CHECKPOINT_COLUMNS
from outcome_inputs import OUTCOME_TIMEPOINTS, OutcomeInputError, combined_csv_file


NETWORK = "PRONET"


def _pair_row(**updates: str) -> dict[str, str]:
    row = {
        "Subject": "S1",
        "Timepoint": "baseline",
        "General_Flag": "example_form",
        "variable": "field_a",
        "message_variable": "field_a",
        "canonical_template": "example template",
        "before_value_available": "True",
        "before_value": "original",
        "before_value_source": "message_pattern",
        "Latest_seen": "2026-01-01",
        "status": "filled",
        "current_value": "current",
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


def _write_combined_slices(root: Path) -> None:
    root.mkdir(parents=True)
    for timepoint in OUTCOME_TIMEPOINTS:
        path = combined_csv_file(root, NETWORK, timepoint)
        with path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(["subjectid", "chrcrit_part", "field_a"])
            writer.writerow(["S1", "1", "current"])


def _write_outcomes(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(CHECKPOINT_COLUMNS)
        writer.writerows(rows)


def _replacement(
    *,
    subject_id: str = "S1",
    timepoint: str = "baseline",
    variable: str = "field_a",
) -> before_after.ValueReplacement:
    return before_after.ValueReplacement(
        network=NETWORK,
        subject_id=subject_id,
        timepoint=timepoint,
        variable=variable,
        before_value="original",
        current_value="current",
    )


def test_scenario_manifest_binds_program_identity_for_resume(tmp_path: Path) -> None:
    source_root = tmp_path / "combined"
    scenario_root = tmp_path / "experiment"
    pair_path = tmp_path / "resolved_before_after_values_PRONET.csv"
    _write_combined_slices(source_root)
    _write_pairs(pair_path, [_pair_row()])

    layout = before_after.build_value_scenarios(
        source_root,
        pair_path,
        scenario_root,
        NETWORK,
    )
    manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    assert manifest["program"] == before_after._program_identity()
    assert set(manifest["program"]) == {
        "module_sha256",
        "python_implementation",
        "python_version",
    }
    assert (
        before_after.load_existing_scenarios(
            source_root, pair_path, scenario_root, NETWORK
        )
        == layout
    )

    manifest["program"]["module_sha256"] = "changed-program"
    layout.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(OutcomeInputError, match="does not match this request"):
        before_after.load_existing_scenarios(
            source_root, pair_path, scenario_root, NETWORK
        )


@pytest.mark.parametrize(
    "replacement",
    [
        _replacement(subject_id=""),
        _replacement(variable=""),
        _replacement(variable="subjectid"),
        _replacement(timepoint="multiple_timepoints"),
    ],
)
def test_build_revalidates_custom_replacement_targets(
    tmp_path: Path, replacement: before_after.ValueReplacement
) -> None:
    with pytest.raises(OutcomeInputError, match="unsafe targets"):
        before_after._validate_replacement_targets([replacement], NETWORK)


def test_build_rejects_duplicate_custom_replacement_keys(tmp_path: Path) -> None:
    replacement = _replacement()
    with pytest.raises(OutcomeInputError, match="duplicate replacement key"):
        before_after._validate_replacement_targets([replacement, replacement], NETWORK)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                ["Integer", "baseline_arm_1", "1", "score", "S1"],
                ["Integer", "baseline_arm_1", "2", "score", "S1"],
            ],
            "duplicate canonical key",
        ),
        (
            [
                ["Integer", "baseline_arm_1", "1", "score", "S1"],
                ["Integer", "baseline_arm_1", "2", "score", "S2"],
                ["Integer", "month_1_arm_1", "3", "other", "S1"],
            ],
            "subject rows are not contiguous",
        ),
    ],
)
def test_compare_rejects_invalid_subject_blocks_atomically(
    tmp_path: Path, rows: list[list[str]], message: str
) -> None:
    original = tmp_path / "original.csv"
    current = tmp_path / "current.csv"
    output = tmp_path / "comparison"
    output.mkdir()
    existing = output / "outcome_value_changes.csv"
    existing.write_text("last-good\n", encoding="utf-8")
    _write_outcomes(original, rows)
    _write_outcomes(current, rows)

    with pytest.raises(OutcomeInputError, match=message):
        before_after.compare_outcome_files(original, current, output)

    assert existing.read_text(encoding="utf-8") == "last-good\n"
    assert not list(output.glob(".*.tmp"))

def test_multiple_timepoints_rows_are_audited_without_blocking_canonical_rows(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "combined"
    scenario_root = tmp_path / "experiment"
    pair_path = tmp_path / "resolved_before_after_values_PRONET.csv"
    _write_combined_slices(source_root)
    _write_pairs(
        pair_path,
        [
            _pair_row(),
            _pair_row(
                Subject="S2",
                Timepoint="multiple_timepoints",
                variable="field_b",
                message_variable="field_b",
                before_value="before-many",
                current_value="after-many",
            ),
        ],
    )
    live_hash = before_after._sha256_file(pair_path)

    plan = before_after.load_value_replacement_plan(pair_path, NETWORK)
    assert tuple(item.key for item in plan.replacements) == (
        ("baseline", "S1", "field_a"),
    )
    assert len(plan.unmapped) == 1
    assert plan.unmapped[0].timepoint == "multiple_timepoints"
    assert before_after.load_value_replacements(pair_path, NETWORK) == (
        plan.replacements
    )

    layout = before_after.build_value_scenarios(
        source_root, pair_path, scenario_root, NETWORK
    )

    assert before_after._sha256_file(pair_path) == live_hash
    assert layout.frozen_before_after.read_bytes() == pair_path.read_bytes()
    with layout.unmapped_replacements.open(
        "r", encoding="utf-8", newline=""
    ) as source:
        audit_rows = list(csv.DictReader(source))
    assert len(audit_rows) == 1
    assert audit_rows[0]["Subject"] == "S2"
    assert audit_rows[0]["Timepoint"] == "multiple_timepoints"
    assert audit_rows[0]["variable"] == "field_b"
    assert audit_rows[0]["before_value"] == "before-many"
    assert audit_rows[0]["current_value"] == "after-many"
    assert audit_rows[0]["skip_reason"] == (
        "multiple_timepoints_not_mappable_to_canonical_slice"
    )
    manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    assert manifest["counts"]["skipped_multiple_timepoint_pairs"] == 1
    assert manifest["before_after_sha256"] == live_hash
    assert manifest["frozen_before_after_sha256"] == live_hash
    assert (
        before_after.load_existing_scenarios(
            source_root, pair_path, scenario_root, NETWORK
        )
        == layout
    )

@pytest.mark.parametrize(
    "bad_timepoint",
    ["multiple_timepoint", "Multiple_timepoints", "multiple_timepoints "],
)
def test_only_exact_multiple_timepoints_is_explicitly_unmappable(
    tmp_path: Path, bad_timepoint: str
) -> None:
    pair_path = tmp_path / "resolved_before_after_values_PRONET.csv"
    _write_pairs(pair_path, [_pair_row(Timepoint=bad_timepoint)])

    with pytest.raises(OutcomeInputError, match="unsupported timepoint"):
        before_after.load_value_replacement_plan(pair_path, NETWORK)


def test_build_detects_pair_change_during_frozen_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "combined"
    scenario_root = tmp_path / "experiment"
    pair_path = tmp_path / "resolved_before_after_values_PRONET.csv"
    _write_combined_slices(source_root)
    _write_pairs(pair_path, [_pair_row()])
    real_copy2 = before_after.shutil.copy2

    def copy_then_change(source: Path, target: Path, *args: object, **kwargs: object):
        result = real_copy2(source, target, *args, **kwargs)
        if Path(source).resolve() == pair_path.resolve():
            _write_pairs(pair_path, [_pair_row(before_value="changed-during-copy")])
        return result

    monkeypatch.setattr(before_after.shutil, "copy2", copy_then_change)

    with pytest.raises(OutcomeInputError, match="changed while being staged"):
        before_after.build_value_scenarios(
            source_root, pair_path, scenario_root, NETWORK
        )
    assert not scenario_root.exists()
    assert not list(tmp_path.glob(".experiment.*"))


def test_build_rejects_supplied_replacements_from_a_stale_live_parse(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "combined"
    scenario_root = tmp_path / "experiment"
    pair_path = tmp_path / "resolved_before_after_values_PRONET.csv"
    _write_combined_slices(source_root)
    _write_pairs(pair_path, [_pair_row()])
    stale_replacements = before_after.load_value_replacements(pair_path, NETWORK)
    _write_pairs(pair_path, [_pair_row(before_value="newer-original")])

    with pytest.raises(OutcomeInputError, match="do not exactly match"):
        before_after.build_value_scenarios(
            source_root,
            pair_path,
            scenario_root,
            NETWORK,
            stale_replacements,
        )
    assert not scenario_root.exists()


def test_build_rechecks_live_pair_immediately_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "combined"
    scenario_root = tmp_path / "experiment"
    pair_path = tmp_path / "resolved_before_after_values_PRONET.csv"
    _write_combined_slices(source_root)
    _write_pairs(pair_path, [_pair_row()])
    real_atomic_write = before_after._atomic_write_json

    def write_then_change(path: Path, payload: dict[str, object]) -> None:
        real_atomic_write(path, payload)
        if path.name == "experiment_manifest.json":
            _write_pairs(pair_path, [_pair_row(before_value="changed-before-publish")])

    monkeypatch.setattr(before_after, "_atomic_write_json", write_then_change)

    with pytest.raises(OutcomeInputError, match="changed while scenarios"):
        before_after.build_value_scenarios(
            source_root, pair_path, scenario_root, NETWORK
        )
    assert not scenario_root.exists()
    assert not list(tmp_path.glob(".experiment.*"))


def test_main_fresh_path_does_not_parse_the_live_artifact_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root = tmp_path / "combined"
    scenario_root = tmp_path / "experiment"
    pair_path = tmp_path / "resolved_before_after_values_PRONET.csv"
    _write_combined_slices(source_root)
    _write_pairs(
        pair_path,
        [
            _pair_row(),
            _pair_row(
                Subject="S2",
                Timepoint="multiple_timepoints",
                variable="field_b",
                message_variable="field_b",
            ),
        ],
    )

    def forbidden_live_wrapper(*args: object, **kwargs: object) -> None:
        raise AssertionError("fresh main must parse only the frozen copy")

    monkeypatch.setattr(
        before_after, "load_value_replacements", forbidden_live_wrapper
    )

    assert (
        before_after.main(
            [
                NETWORK,
                "--combined-csv-path",
                str(source_root),
                "--before-after-csv",
                str(pair_path),
                "--experiment-dir",
                str(scenario_root),
                "--prepare-only",
                    "--acknowledge-protected-output",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Skipped 1 explicitly unmappable multiple_timepoints pair(s)" in output
