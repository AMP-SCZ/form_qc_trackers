import json
from pathlib import Path

import pandas as pd
import pytest

import outcome_checkpoints
from outcome_checkpoints import (
    CHECKPOINT_COLUMNS,
    RunCheckpointContext,
    _sha256_file,
    build_run_manifest_payload,
    load_subject_checkpoint,
    outcome_run_lock,
    prepare_run_checkpoint,
    write_aggregate_from_checkpoints,
    write_subject_checkpoint,
)
from outcome_inputs import OutcomeInputError


def _context(
    output_dir: Path, subject_ids: tuple[str, ...] = ("S1",)
) -> RunCheckpointContext:
    return prepare_run_checkpoint(
        output_dir,
        {
            "checkpoint_format": 1,
            "network": "PRONET",
            "version": "run_outcome",
            "subject_ids": list(subject_ids),
            "synthetic_input": "unit-test",
        },
        resume=False,
    )


def _frame(subject_id: str, variable: str = "score") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "data_type": ["Integer"],
            "redcap_event_name": ["baseline_arm_1"],
            "value": [1.2346],
            "variable": [variable],
            "ID": [subject_id],
        }
    )


def _checkpoint_csv(context: RunCheckpointContext) -> Path:
    paths = list((context.root / "subjects").rglob("*.csv"))
    assert len(paths) == 1
    return paths[0]


def test_code_fingerprint_changes_when_outcome_scoring_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for filename in outcome_checkpoints._CODE_FINGERPRINT_FILENAMES:
        (tmp_path / filename).write_text(
            f"synthetic contents for {filename}\n", encoding="utf-8"
        )
    monkeypatch.setattr(
        outcome_checkpoints, "__file__", str(tmp_path / "outcome_checkpoints.py")
    )

    before = outcome_checkpoints._code_fingerprint()
    (tmp_path / "outcome_scoring.py").write_text(
        "synthetic scoring change\n", encoding="utf-8"
    )

    assert outcome_checkpoints._code_fingerprint() != before


def test_run_manifest_payload_binds_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = (
        {
            "name": "synthetic.csv",
            "size": 10,
            "mtime_ns": 20,
            "sha256": "input-sha256",
        },
    )
    first_runtime = {
        "python_implementation": "CPython",
        "python_version": "3.12.0",
        "pandas_version": "3.0.0",
        "numpy_version": "2.4.0",
    }
    monkeypatch.setattr(
        outcome_checkpoints,
        "snapshot_combined_inputs",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        outcome_checkpoints, "_code_fingerprint", lambda: "code-sha256"
    )
    monkeypatch.setattr(
        outcome_checkpoints, "_runtime_identity", lambda: first_runtime
    )

    first_payload = build_run_manifest_payload(
        tmp_path, "PRONET", "run_outcome", ["S1"], snapshot
    )
    prepare_run_checkpoint(tmp_path / "output", first_payload, resume=False)
    assert first_payload["runtime"] == first_runtime

    monkeypatch.setattr(
        outcome_checkpoints,
        "_runtime_identity",
        lambda: {**first_runtime, "pandas_version": "3.1.0"},
    )
    changed_payload = build_run_manifest_payload(
        tmp_path, "PRONET", "run_outcome", ["S1"], snapshot
    )

    assert changed_payload != first_payload
    with pytest.raises(OutcomeInputError, match="no compatible checkpoint"):
        prepare_run_checkpoint(
            tmp_path / "output", changed_payload, resume=True
        )


def test_manifest_resume_requires_the_exact_stored_payload(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    context = _context(output_dir)
    payload = {
        "checkpoint_format": 1,
        "network": "PRONET",
        "version": "run_outcome",
        "subject_ids": ["S1"],
        "synthetic_input": "unit-test",
    }

    assert prepare_run_checkpoint(output_dir, payload, resume=True) == context
    manifest_path = context.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload"]["subject_ids"] = ["S2"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(OutcomeInputError, match="manifest does not match"):
        prepare_run_checkpoint(output_dir, payload, resume=True)


def test_output_root_lock_blocks_other_networks_and_versions(
    tmp_path: Path,
) -> None:
    with outcome_run_lock(tmp_path, "PRONET", "run_outcome"):
        with pytest.raises(OutcomeInputError, match="this output root"):
            with outcome_run_lock(tmp_path, "PRESCIENT", "single_subject"):
                pytest.fail("a second output-root writer acquired the lock")

    with outcome_run_lock(tmp_path, "PRESCIENT", "single_subject"):
        pass
    lock_files = list((tmp_path / ".outcome_checkpoints" / "locks").glob("*"))
    assert [path.name for path in lock_files] == ["output-root.lock"]


def test_subject_checkpoint_round_trip_normalizes_schema_and_values(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    result = write_subject_checkpoint(context, "S1", _frame("S1"))

    assert result.status == "frame"
    assert result.frame is not None
    assert tuple(result.frame.columns) == CHECKPOINT_COLUMNS
    loaded = load_subject_checkpoint(context, "S1", strict=True)
    assert loaded is not None and loaded.frame is not None
    assert loaded.frame["value"].tolist() == ["1.235"]
    assert loaded.frame["ID"].tolist() == ["S1"]


def test_checkpoint_metadata_carries_no_measure_output_contract(
    tmp_path: Path,
) -> None:
    # The per-subject per-measure CSV tree was removed on 2026-08-22; the
    # combined per-network aggregate is the only outcome deliverable. The
    # checkpoint metadata therefore must not reference measure outputs, and
    # nothing may be written outside the checkpoint root.
    context = _context(tmp_path)
    write_subject_checkpoint(context, "S1", _frame("S1"))

    metadata = json.loads(
        _checkpoint_csv(context).with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    assert "measure_outputs" not in metadata
    assert not (tmp_path / "subject_outcomes").exists()


def test_none_checkpoint_is_a_resumable_completion(tmp_path: Path) -> None:
    context = _context(tmp_path)
    write_subject_checkpoint(context, "S1", None)

    loaded = load_subject_checkpoint(context, "S1", strict=True)
    assert loaded is not None
    assert loaded.status == "none"
    assert loaded.frame is None


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame(columns=CHECKPOINT_COLUMNS), "checkpoint is empty"),
        (_frame("S1").drop(columns="ID"), "unexpected schema"),
        (_frame("ANOTHER"), "checkpoint ID does not match"),
    ],
)
def test_checkpoint_write_rejects_invalid_subject_frames(
    tmp_path: Path, frame: pd.DataFrame, message: str
) -> None:
    context = _context(tmp_path)
    with pytest.raises(OutcomeInputError, match=message):
        write_subject_checkpoint(context, "S1", frame)


def test_failed_checkpoint_write_preserves_previous_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    write_subject_checkpoint(context, "S1", _frame("S1", "original"))

    def fail_after_partial_write(
        self: pd.DataFrame, target, **_: object
    ) -> None:
        target.write("partial")
        raise OSError("synthetic checkpoint failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_after_partial_write)
    with pytest.raises(OSError, match="synthetic checkpoint failure"):
        write_subject_checkpoint(context, "S1", _frame("S1", "replacement"))

    loaded = load_subject_checkpoint(context, "S1", strict=True)
    assert loaded is not None and loaded.frame is not None
    assert loaded.frame["variable"].tolist() == ["original"]
    assert not list(_checkpoint_csv(context).parent.glob("*.tmp"))


def test_corrupt_checkpoint_is_pending_unless_strict(tmp_path: Path) -> None:
    context = _context(tmp_path)
    write_subject_checkpoint(context, "S1", _frame("S1"))
    _checkpoint_csv(context).write_text("corrupt\n", encoding="utf-8")

    assert load_subject_checkpoint(context, "S1", strict=False) is None
    with pytest.raises(OutcomeInputError, match="integrity check"):
        load_subject_checkpoint(context, "S1", strict=True)


def test_aggregate_streams_checkpoints_in_manifest_roster_order(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, ("S1", "S2"))
    for subject_id in ("S2", "S1"):
        write_subject_checkpoint(
            context, subject_id, _frame(subject_id, f"score_{subject_id}")
        )

    count = write_aggregate_from_checkpoints(
        context,
        tmp_path,
        "pronet_all.csv",
    )
    aggregate = pd.read_csv(tmp_path / "pronet_all.csv", dtype=str)
    assert count == 2
    assert aggregate["ID"].tolist() == ["S1", "S2"]
    assert aggregate["variable"].tolist() == ["score_S1", "score_S2"]


def test_failed_aggregate_stream_preserves_previous_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    write_subject_checkpoint(context, "S1", _frame("S1"))
    target = tmp_path / "pronet_all.csv"
    target.write_text("previous aggregate\n", encoding="utf-8")

    def fail_after_partial_write(
        self: pd.DataFrame, stream, **_: object
    ) -> None:
        stream.write("partial")
        raise OSError("synthetic aggregate failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_after_partial_write)
    with pytest.raises(OSError, match="synthetic aggregate failure"):
        write_aggregate_from_checkpoints(
            context,
            tmp_path,
            target.name,
        )

    assert target.read_text(encoding="utf-8") == "previous aggregate\n"
    assert not list(tmp_path.glob(".pronet_all.csv.*.tmp"))


def test_checkpoint_fingerprint_error_omits_artifact_path(
    tmp_path: Path,
) -> None:
    sensitive_subject = "PRIVATE_SUBJECT_FINGERPRINT"
    sensitive_token = "PRIVATE_CHECKPOINT_TOKEN"
    artifact_path = (
        tmp_path / f"subject-{sensitive_subject}" / f"{sensitive_token}.csv"
    )

    with pytest.raises(
        OutcomeInputError, match="validate checkpoint artifact"
    ) as exc_info:
        _sha256_file(artifact_path)

    diagnostic = str(exc_info.value)
    assert sensitive_subject not in diagnostic
    assert sensitive_token not in diagnostic
    assert str(artifact_path) not in diagnostic
    assert exc_info.value.__suppress_context__


def test_corrupt_checkpoint_error_omits_subject_and_token(
    tmp_path: Path,
) -> None:
    sensitive_subject = "PRIVATE_SUBJECT_CHECKPOINT"
    context = _context(tmp_path, (sensitive_subject,))
    write_subject_checkpoint(context, sensitive_subject, _frame(sensitive_subject))
    checkpoint_path = _checkpoint_csv(context)
    checkpoint_path.write_text("corrupt\n", encoding="utf-8")

    with pytest.raises(
        OutcomeInputError, match="integrity check"
    ) as exc_info:
        load_subject_checkpoint(context, sensitive_subject, strict=True)

    diagnostic = str(exc_info.value)
    assert sensitive_subject not in diagnostic
    assert checkpoint_path.name not in diagnostic
    assert context.signature not in diagnostic
    assert str(checkpoint_path) not in diagnostic


def test_manifest_read_error_omits_run_path_and_roster(
    tmp_path: Path,
) -> None:
    sensitive_subject = "PRIVATE_SUBJECT_MANIFEST"
    payload = {
        "checkpoint_format": 1,
        "network": "PRONET",
        "version": "run_outcome",
        "subject_ids": [sensitive_subject],
        "synthetic_input": "unit-test",
    }
    context = prepare_run_checkpoint(tmp_path, payload, resume=False)
    manifest_path = context.root / "manifest.json"
    manifest_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(
        OutcomeInputError, match="Could not read checkpoint manifest"
    ) as exc_info:
        prepare_run_checkpoint(tmp_path, payload, resume=True)

    diagnostic = str(exc_info.value)
    assert sensitive_subject not in diagnostic
    assert context.signature not in diagnostic
    assert str(manifest_path) not in diagnostic
    assert exc_info.value.__suppress_context__
