import csv
import json
from pathlib import Path

import pytest

from merge_blank_flag_pairs import (
    BLANK_HISTORY_COLUMNS,
    CONFLICT_COLUMNS,
    PAIR_COLUMNS,
    PairMergeError,
    _targeted_row_values,
    build_blank_merged_pairs,
    main,
)
from outcome_before_after import load_value_replacement_plan
from outcome_inputs import combined_csv_file


def _write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _read_dicts(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source, strict=True))


def _pair_row(
    subject: str,
    timepoint: str,
    variable: str,
    before: str,
    current: str,
    *,
    status: str = "filled",
):
    values = {
        "Subject": subject,
        "Timepoint": timepoint,
        "General_Flag": "synthetic_form",
        "variable": variable,
        "message_variable": variable,
        "canonical_template": "synthetic_template",
        "before_value_available": "True",
        "before_value": before,
        "before_value_source": "synthetic_test",
        "Latest_seen": "2026-01-01",
        "status": status,
        "current_value": current,
    }
    return [values[name] for name in PAIR_COLUMNS]


def _history_row(
    network: str,
    subject: str,
    variable: str,
    timepoint: str,
    *,
    stale_current: str = "ignored-history-value",
    status: str = "filled",
):
    values = {
        "network": network,
        "subject": subject,
        "variable": variable,
        "timepoint": timepoint,
        "current_value": stale_current,
        "status": status,
    }
    return [values[name] for name in BLANK_HISTORY_COLUMNS]


def _base_inputs(tmp_path: Path):
    pair_dir = tmp_path / "existing"
    combined = tmp_path / "combined"
    history = tmp_path / "blank_flag_history.csv"
    _write_csv(
        pair_dir / "resolved_before_after_values_PRONET.csv",
        PAIR_COLUMNS,
        [_pair_row("P-KEEP", "baseline", "keep_var", "old", "current")],
    )
    _write_csv(
        pair_dir / "resolved_before_after_values_PRESCIENT.csv",
        PAIR_COLUMNS,
        [_pair_row("C-KEEP", "baseline", "keep_var", "old", "current")],
    )
    return history, pair_dir, combined


def test_build_merges_blank_pairs_refreshes_exact_values_and_is_outcome_compatible(
    tmp_path,
):
    history, pair_dir, combined = _base_inputs(tmp_path)
    _write_csv(
        pair_dir / "resolved_before_after_values_PRONET.csv",
        PAIR_COLUMNS,
        [
            _pair_row("P-ONE", "baseline", "score", "historical", "stale"),
            _pair_row("P-KEEP", "baseline", "keep_var", "old", "current"),
        ],
    )
    _write_csv(
        history,
        BLANK_HISTORY_COLUMNS,
        [
            _history_row("PRONET", "P-ONE", "score", "baseline"),
            _history_row(
                "PRONET", "P-TWO", "blank_now", "baseline", status="still_blank"
            ),
            _history_row("PRESCIENT", "C-ONE", "missing_now", "month_1"),
        ],
    )
    _write_csv(
        combined_csv_file(combined, "PRONET", "baseline"),
        ["subjectid", "score", "blank_now", "keep_var"],
        [
            ["P-ONE", " exact current ", "not-targeted", "unused"],
            ["P-TWO", "not-targeted", "", "unused"],
            ["P-KEEP", "not-targeted", "not-targeted", "current"],
        ],
    )
    _write_csv(
        combined_csv_file(combined, "PRESCIENT", "month1"),
        ["subjectid", "missing_now"],
        [["C-ONE", "999"]],
    )

    output = tmp_path / "new_pairs"
    results = build_blank_merged_pairs(history, pair_dir, combined, output)

    assert [result.network for result in results] == ["PRONET", "PRESCIENT"]
    pronet_rows = _read_dicts(
        output / "resolved_before_after_values_PRONET.csv"
    )
    by_target = {(row["Subject"], row["variable"]): row for row in pronet_rows}
    assert by_target[("P-ONE", "score")]["before_value"] == ""
    assert by_target[("P-ONE", "score")]["current_value"] == " exact current "
    assert by_target[("P-ONE", "score")]["before_value_source"] == "blank_flag_history"
    assert by_target[("P-TWO", "blank_now")]["before_value"] == ""
    assert by_target[("P-TWO", "blank_now")]["current_value"] == ""
    assert by_target[("P-TWO", "blank_now")]["status"] == "still_blank"
    assert by_target[("P-KEEP", "keep_var")]["before_value"] == "old"

    prescient_rows = _read_dicts(
        output / "resolved_before_after_values_PRESCIENT.csv"
    )
    missing_row = next(row for row in prescient_rows if row["variable"] == "missing_now")
    assert missing_row["before_value"] == ""
    assert missing_row["current_value"] == "999"
    assert missing_row["status"] == "missing_code"

    conflict_rows = _read_dicts(output / "blank_pair_conflicts_PRONET.csv")
    assert len(conflict_rows) == 1
    assert conflict_rows[0]["conflict_reason"] == "blank_before_precedence"
    assert tuple(conflict_rows[0]) == CONFLICT_COLUMNS

    pronet_plan = load_value_replacement_plan(
        output / "resolved_before_after_values_PRONET.csv", "PRONET"
    )
    prescient_plan = load_value_replacement_plan(
        output / "resolved_before_after_values_PRESCIENT.csv", "PRESCIENT"
    )
    assert len(pronet_plan.replacements) == 3
    assert len(prescient_plan.replacements) == 2

    summary = json.loads((output / "blank_pair_build_summary.json").read_text())
    assert summary["conflict_policy"] == "blank_history_precedence_with_protected_audit"
    assert summary["networks"]["PRONET"]["blank_pairs_refreshed"] == 2


def test_duplicate_subject_is_accepted_only_when_exact_values_agree(tmp_path):
    history, pair_dir, combined = _base_inputs(tmp_path)
    _write_csv(
        history,
        BLANK_HISTORY_COLUMNS,
        [
            _history_row("PRONET", "P-AGREE", "value", "baseline"),
            _history_row("PRONET", "P-DIFFER", "value", "baseline"),
            _history_row("PRONET", "P-MISSING-VAR", "absent", "baseline"),
        ],
    )
    _write_csv(
        combined_csv_file(combined, "PRONET", "baseline"),
        ["subjectid", "value"],
        [
            ["P-AGREE", "same"],
            ["P-AGREE", "same"],
            ["P-DIFFER", "first"],
            ["P-DIFFER", "second"],
            ["P-MISSING-VAR", "unrelated"],
        ],
    )

    output = tmp_path / "new_pairs"
    results = build_blank_merged_pairs(history, pair_dir, combined, output)
    pronet = next(result for result in results if result.network == "PRONET")

    assert pronet.counts["blank_pairs_refreshed"] == 1
    assert pronet.counts["skipped_subject_ambiguous"] == 1
    assert pronet.counts["skipped_variable_missing"] == 1
    rows = _read_dicts(output / "resolved_before_after_values_PRONET.csv")
    assert any(row["Subject"] == "P-AGREE" for row in rows)
    assert not any(row["Subject"] == "P-DIFFER" for row in rows)


def test_targeted_row_values_does_not_capture_subject_variable_cross_product():
    unrelated_sensitive = "PRIVATE-UNRELATED-SYNTHETIC"
    captured = dict(
        _targeted_row_values(
            "P-ONE",
            ["P-ONE", "requested", unrelated_sensitive],
            {"subjectid": 0, "for_one": 1, "for_two": 2},
            {"P-ONE": {"for_one"}, "P-TWO": {"for_two"}},
        )
    )

    assert captured == {("P-ONE", "for_one"): "requested"}
    assert unrelated_sensitive not in captured.values()


def test_unresolved_known_blank_target_cannot_preserve_existing_nonblank_pair(
    tmp_path,
):
    history, pair_dir, combined = _base_inputs(tmp_path)
    _write_csv(
        history,
        BLANK_HISTORY_COLUMNS,
        [_history_row("PRONET", "P-DIFFER", "value", "baseline")],
    )
    _write_csv(
        pair_dir / "resolved_before_after_values_PRONET.csv",
        PAIR_COLUMNS,
        [
            _pair_row(
                "P-DIFFER", "baseline", "value", "known-wrong-before", "stale"
            ),
            _pair_row("P-KEEP", "baseline", "keep_var", "old", "current"),
        ],
    )
    _write_csv(
        combined_csv_file(combined, "PRONET", "baseline"),
        ["subjectid", "value"],
        [["P-DIFFER", "first"], ["P-DIFFER", "second"]],
    )

    output = tmp_path / "new_pairs"
    results = build_blank_merged_pairs(history, pair_dir, combined, output)
    pronet = next(result for result in results if result.network == "PRONET")

    rows = _read_dicts(output / "resolved_before_after_values_PRONET.csv")
    assert not any(
        row["Subject"] == "P-DIFFER" and row["variable"] == "value"
        for row in rows
    )
    audit = _read_dicts(output / "blank_pair_conflicts_PRONET.csv")
    dropped = next(
        row
        for row in audit
        if row["conflict_reason"] == "unresolved_current_target_dropped"
    )
    assert dropped["selected_pair_published"] == "False"
    assert pronet.counts["overlapping_unresolved_targets_dropped"] == 1


def test_unrelated_existing_value_conflict_fails_closed_without_output(tmp_path):
    history, pair_dir, combined = _base_inputs(tmp_path)
    _write_csv(history, BLANK_HISTORY_COLUMNS, [])
    _write_csv(
        pair_dir / "resolved_before_after_values_PRONET.csv",
        PAIR_COLUMNS,
        [
            _pair_row("P-CONFLICT", "baseline", "score", "one", "current"),
            _pair_row("P-CONFLICT", "baseline", "score", "two", "current"),
        ],
    )
    output = tmp_path / "new_pairs"

    with pytest.raises(PairMergeError, match="conflicting non-blank-history"):
        build_blank_merged_pairs(history, pair_dir, combined, output)

    assert not output.exists()


def test_strict_blank_history_schema_fails_before_publication(tmp_path):
    history, pair_dir, combined = _base_inputs(tmp_path)
    _write_csv(history, [*BLANK_HISTORY_COLUMNS, "unexpected"], [])
    output = tmp_path / "new_pairs"

    with pytest.raises(PairMergeError, match="exact required schema"):
        build_blank_merged_pairs(history, pair_dir, combined, output)

    assert not output.exists()


def test_combined_parse_error_diagnostic_does_not_expose_values(tmp_path):
    history, pair_dir, combined = _base_inputs(tmp_path)
    sensitive_value = "PRIVATE-SYNTHETIC-VALUE"
    _write_csv(
        history,
        BLANK_HISTORY_COLUMNS,
        [_history_row("PRONET", "P-ONE", "score", "baseline")],
    )
    path = combined_csv_file(combined, "PRONET", "baseline")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'subjectid,score\nP-ONE,"{sensitive_value}\n', encoding="utf-8"
    )
    output = tmp_path / "new_pairs"

    with pytest.raises(PairMergeError) as caught:
        build_blank_merged_pairs(history, pair_dir, combined, output)

    assert sensitive_value not in str(caught.value)
    assert not output.exists()


def test_refuses_existing_output_and_cli_requires_protected_acknowledgement(tmp_path):
    history, pair_dir, combined = _base_inputs(tmp_path)
    _write_csv(history, BLANK_HISTORY_COLUMNS, [])
    output = tmp_path / "new_pairs"
    output.mkdir()

    with pytest.raises(PairMergeError, match="already exists"):
        build_blank_merged_pairs(history, pair_dir, combined, output)

    with pytest.raises(PairMergeError, match="acknowledge-protected-output"):
        main(
            [
                "--blank-history",
                str(history),
                "--existing-pair-dir",
                str(pair_dir),
                "--combined-csv-root",
                str(combined),
                "--output-dir",
                str(tmp_path / "another_output"),
            ]
        )
