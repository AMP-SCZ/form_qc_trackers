"""Full-outer comparison regressions for before/current outcome scenarios."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from outcome_before_after import compare_outcome_files
from outcome_checkpoints import CHECKPOINT_COLUMNS
from outcome_inputs import OutcomeInputError


def _row(
    subject: str,
    variable: str,
    value: str,
    *,
    event: str = "baseline_arm_1",
    data_type: str = "Integer",
) -> list[str]:
    return [data_type, event, value, variable, subject]


def _write_outcomes(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(CHECKPOINT_COLUMNS)
        writer.writerows(rows)


def _read_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def test_full_outer_comparison_allows_reordered_added_and_removed_keys(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.csv"
    current = tmp_path / "current.csv"
    output = tmp_path / "comparison"
    _write_outcomes(
        original,
        [
            _row("S1", "same_score", "1"),
            _row("S1", "changed_score", "old"),
            _row("S1", "removed_score", "removed-value", event="month_1_arm_1"),
            _row("S2", "same_score", "2"),
        ],
    )
    _write_outcomes(
        current,
        [
            _row("S1", "added_score", "added-value", event="month_2_arm_1"),
            _row("S1", "changed_score", "new"),
            _row("S1", "same_score", "1"),
            _row("S2", "same_score", "2"),
        ],
    )

    result = compare_outcome_files(original, current, output)

    assert result == {
        "total_outcome_rows": 5,
        "changed_outcome_rows": 1,
        "added_outcome_rows": 1,
        "removed_outcome_rows": 1,
        "changed_subjects": 1,
    }
    changes = _read_dicts(output / "outcome_value_changes.csv")
    assert {row["change_type"] for row in changes} == {
        "changed",
        "added",
        "removed",
    }
    by_type = {row["change_type"]: row for row in changes}
    assert by_type["changed"]["variable"] == "changed_score"
    assert by_type["changed"]["original_outcome_value"] == "old"
    assert by_type["changed"]["current_outcome_value"] == "new"
    assert by_type["added"]["variable"] == "added_score"
    assert by_type["added"]["original_outcome_value"] == ""
    assert by_type["added"]["current_outcome_value"] == "added-value"
    assert by_type["removed"]["variable"] == "removed_score"
    assert by_type["removed"]["original_outcome_value"] == "removed-value"
    assert by_type["removed"]["current_outcome_value"] == ""

    summary = {
        row["variable"]: row
        for row in _read_dicts(output / "outcome_change_summary.csv")
    }
    assert summary["changed_score"] == {
        "data_type": "Integer",
        "variable": "changed_score",
        "changed_outcome_rows": "1",
        "added_outcome_rows": "0",
        "removed_outcome_rows": "0",
    }
    assert summary["added_score"]["added_outcome_rows"] == "1"
    assert summary["added_score"]["changed_outcome_rows"] == "0"
    assert summary["added_score"]["removed_outcome_rows"] == "0"
    assert summary["removed_score"]["removed_outcome_rows"] == "1"
    assert summary["removed_score"]["changed_outcome_rows"] == "0"
    assert summary["removed_score"]["added_outcome_rows"] == "0"


@pytest.mark.parametrize("duplicate_side", ["original", "current"])
def test_duplicate_canonical_key_within_subject_fails_closed_and_private(
    tmp_path: Path,
    duplicate_side: str,
) -> None:
    original = tmp_path / "original.csv"
    current = tmp_path / "current.csv"
    output = tmp_path / "comparison"
    output.mkdir()
    last_good = output / "outcome_value_changes.csv"
    last_good.write_text("last-good\n", encoding="utf-8")
    private_subject = "PRIVATE_DUPLICATE_SUBJECT"
    private_value = "PRIVATE_DUPLICATE_VALUE"
    one_row = _row(private_subject, "score", private_value)
    _write_outcomes(
        original,
        [one_row, list(one_row)] if duplicate_side == "original" else [one_row],
    )
    _write_outcomes(
        current,
        [one_row, list(one_row)] if duplicate_side == "current" else [one_row],
    )

    with pytest.raises(OutcomeInputError) as exc_info:
        compare_outcome_files(original, current, output)

    diagnostic = str(exc_info.value)
    assert "duplicate canonical keys" in diagnostic
    assert private_subject not in diagnostic
    assert private_value not in diagnostic
    assert last_good.read_text(encoding="utf-8") == "last-good\n"
    assert not list(output.glob(".*.tmp"))


def test_noncontiguous_subject_block_is_rejected_without_private_id(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.csv"
    current = tmp_path / "current.csv"
    private_subject = "PRIVATE_NONCONTIGUOUS_SUBJECT"
    rows = [
        _row(private_subject, "first", "1"),
        _row("S2", "middle", "2"),
        _row(private_subject, "last", "3"),
    ]
    _write_outcomes(original, rows)
    _write_outcomes(current, rows)

    with pytest.raises(OutcomeInputError) as exc_info:
        compare_outcome_files(original, current, tmp_path / "comparison")

    diagnostic = str(exc_info.value)
    assert "subject rows are not contiguous" in diagnostic
    assert private_subject not in diagnostic


def test_different_ordered_subject_rosters_fail_closed_and_private(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.csv"
    current = tmp_path / "current.csv"
    private_first = "PRIVATE_ROSTER_FIRST"
    private_second = "PRIVATE_ROSTER_SECOND"
    _write_outcomes(
        original,
        [
            _row(private_first, "score", "1"),
            _row(private_second, "score", "2"),
        ],
    )
    _write_outcomes(
        current,
        [
            _row(private_second, "score", "2"),
            _row(private_first, "score", "1"),
        ],
    )

    with pytest.raises(OutcomeInputError) as exc_info:
        compare_outcome_files(original, current, tmp_path / "comparison")

    diagnostic = str(exc_info.value)
    assert "different ordered subject rosters" in diagnostic
    assert private_first not in diagnostic
    assert private_second not in diagnostic
