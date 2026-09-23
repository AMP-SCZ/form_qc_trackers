from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import outcome_before_after as before_after
from outcome_checkpoints import CHECKPOINT_COLUMNS
from outcome_inputs import OutcomeInputError


NETWORK = "PRONET"
AGGREGATE_NAME = "pronet_affected.csv"


def _write_outcomes(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(CHECKPOINT_COLUMNS)
        writer.writerow(["Integer", "baseline_arm_1", value, "score", "S1"])


def _identity_record(path: Path, *, code: str = "a" * 64) -> dict[str, object]:
    identity: dict[str, object] = {
        "checkpoint_format": 1,
        "network": NETWORK,
        "version": "run_outcome",
        "roster_mode": "affected_only",
        "code_sha256": code,
        "runtime": {
            "python_implementation": "CPython",
            "python_version": "3.test",
            "pandas_version": "3.test",
            "numpy_version": "2.test",
        },
        "roster_count": 1,
        "roster_sha256": before_after._sha256_json(["S1"]),
    }
    return {
        **identity,
        "calculation_identity_sha256": before_after._sha256_json(identity),
        "signature": "c" * 64,
        "run_id": "run-test",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "aggregate_file": AGGREGATE_NAME,
        "aggregate_sha256": before_after._sha256_file(path),
    }


def test_completed_run_record_binds_checkpoint_runtime_roster_and_aggregate(
    tmp_path: Path,
) -> None:
    combined = tmp_path / "combined"
    output = tmp_path / "outcomes"
    combined.mkdir()
    aggregate = output / AGGREGATE_NAME
    _write_outcomes(aggregate, "1")
    payload = {
        "checkpoint_format": 1,
        "network": NETWORK,
        "version": "run_outcome",
        "roster_mode": "affected_only",
        "combined_csv_path": str(combined.resolve()),
        "subject_ids": ["S1"],
        "input_files": [],
        "code_sha256": "a" * 64,
        "runtime": {
            "python_implementation": "CPython",
            "python_version": "3.test",
            "pandas_version": "3.test",
            "numpy_version": "2.test",
        },
    }
    signature = before_after._sha256_json(payload)
    checkpoint = (
        output
        / ".outcome_checkpoints"
        / "pronet"
        / "run_outcome"
        / signature
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text(
        json.dumps({"signature": signature, "run_id": "run-1", "payload": payload}),
        encoding="utf-8",
    )
    completed_at = datetime.now(timezone.utc).isoformat()
    (checkpoint / "state.json").write_text(
        json.dumps(
            {
                "signature": signature,
                "run_id": "run-1",
                "status": "complete",
                "updated_at": completed_at,
            }
        ),
        encoding="utf-8",
    )

    record = before_after._completed_outcome_run_record(
        output,
        combined,
        NETWORK,
        payload["input_files"],
        ["S1"],
    )

    assert record["code_sha256"] == payload["code_sha256"]
    assert record["runtime"] == payload["runtime"]
    assert record["roster_count"] == 1
    assert record["roster_sha256"] == before_after._sha256_json(["S1"])
    assert record["signature"] == signature
    assert record["run_id"] == "run-1"
    assert record["aggregate_sha256"] == before_after._sha256_file(aggregate)
    with pytest.raises(
        OutcomeInputError,
        match="without a matching completed checkpoint",
    ):
        before_after._completed_outcome_run_record(
            output,
            combined,
            NETWORK,
            [
                {"name": "unexpected.csv", "size": 0, "mtime_ns": 0, "sha256": "0" * 64}
            ],
            ["S1"],
        )


def test_scenario_comparison_rejects_different_calculation_code(
    tmp_path: Path,
) -> None:
    layout = before_after.ScenarioLayout.for_root(tmp_path / "experiment")
    layout.root.mkdir(parents=True)
    original = layout.original_outcomes / AGGREGATE_NAME
    current = layout.current_outcomes / AGGREGATE_NAME
    _write_outcomes(original, "1")
    _write_outcomes(current, "2")
    layout.manifest.write_text(
        json.dumps(
            {
                "state": {
                    before_after.ORIGINAL_SCENARIO: "complete",
                    before_after.CURRENT_SCENARIO: "complete",
                    "comparison": "pending",
                },
                "outcome_runs": {
                    before_after.ORIGINAL_SCENARIO: _identity_record(original),
                    before_after.CURRENT_SCENARIO: _identity_record(
                        current, code="b" * 64
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        OutcomeInputError, match="calculation identities do not match"
    ):
        before_after.compare_scenario_outcomes(layout, NETWORK)

    assert not (layout.root / "outcome_value_changes.csv").exists()
    state = json.loads(layout.manifest.read_text(encoding="utf-8"))["state"]
    assert state["comparison"] == "failed"


def test_comparison_marker_hashes_every_input_and_output_and_invalidates_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "original.csv"
    current = tmp_path / "current.csv"
    output = tmp_path / "comparison"
    _write_outcomes(original, "1")
    _write_outcomes(current, "2")
    before_after.compare_outcome_files(original, current, output)

    marker_path = output / "outcome_comparison_summary.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["status"] == "complete"
    assert marker["original_aggregate_sha256"] == before_after._sha256_file(original)
    assert marker["current_aggregate_sha256"] == before_after._sha256_file(current)
    assert marker["outcome_value_changes_sha256"] == before_after._sha256_file(
        output / "outcome_value_changes.csv"
    )
    assert marker["outcome_change_summary_sha256"] == before_after._sha256_file(
        output / "outcome_change_summary.csv"
    )

    _write_outcomes(current, "3")
    real_replace = before_after.os.replace

    def fail_summary_commit(source: str | Path, target: str | Path) -> None:
        if Path(target) == output / "outcome_change_summary.csv":
            raise OSError("injected summary commit failure")
        real_replace(source, target)

    monkeypatch.setattr(before_after.os, "replace", fail_summary_commit)
    with pytest.raises(OSError, match="injected summary commit failure"):
        before_after.compare_outcome_files(original, current, output)

    interrupted_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert interrupted_marker["status"] == "running"
    assert interrupted_marker["status"] != "complete"
