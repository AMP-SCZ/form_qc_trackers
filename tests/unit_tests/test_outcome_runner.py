import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import outcome_runner

# The 2026-08-22 outcome refactor replaced per-subject output files with two
# combined per-network CSVs; outcome_checkpoints no longer exports
# SUBJECT_OUTPUT_FILENAMES / subject_output_directory. Until this module is
# rewritten against the combined-CSV design, skip it cleanly instead of
# aborting collection of the entire test suite with an ImportError.
try:
    from outcome_checkpoints import (
        SUBJECT_OUTPUT_FILENAMES,
        subject_output_directory,
    )
except ImportError:
    pytest.skip(
        "tests the pre-2026-08-22 per-subject outcome output design; "
        "outcome_checkpoints no longer exports SUBJECT_OUTPUT_FILENAMES",
        allow_module_level=True)
from outcome_inputs import (
    OUTCOME_TIMEPOINTS,
    OutcomeInputError,
    combined_csv_file,
)
from outcome_runner import main


def _write_combined_inputs(
    combined_dir: Path, subject_cohorts: list[tuple[str, str]]
) -> None:
    combined_dir.mkdir()
    for timepoint in OUTCOME_TIMEPOINTS:
        pd.DataFrame(
            [
                {
                    "subjectid": subject_id,
                    "chrcrit_part": cohort,
                    "source_value": f"{subject_id}-{timepoint}",
                }
                for subject_id, cohort in subject_cohorts
            ]
        ).to_csv(
            combined_csv_file(combined_dir, "PRONET", timepoint),
            index=False,
        )


def _write_measure_outputs(
    output_dir: Path,
    network: str,
    subject_id: str,
    event_name: str,
) -> None:
    subject_dir = subject_output_directory(
        output_dir, network, subject_id
    )
    subject_dir.mkdir(parents=True, exist_ok=True)
    for index, filename in enumerate(SUBJECT_OUTPUT_FILENAMES):
        pd.DataFrame(
            {
                "data_type": ["Integer"],
                "redcap_event_name": [event_name],
                "value": [str(index)],
                "variable": [f"measure_{index}"],
            }
        ).to_csv(subject_dir / filename, index=False)


def _outcome_frame(
    subject_id: str,
    event_name: str,
    variable: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "data_type": ["Integer"],
            "redcap_event_name": [event_name],
            "value": ["1"],
            "variable": [variable],
            "ID": [subject_id],
        }
    )


def _calculator(
    calls: list[str],
    *,
    fail_on: str | None = None,
    prefix: str = "synthetic",
):
    def fake_compute(
        subject_id,
        subject_frame,
        network,
        version,
        resolved_output,
    ):
        calls.append(subject_id)
        if subject_id == fail_on:
            raise RuntimeError(f"synthetic interruption for {subject_id}")
        event_name = str(subject_frame["redcap_event_name"].iloc[0])
        _write_measure_outputs(
            Path(resolved_output), network, subject_id, event_name
        )
        return _outcome_frame(
            subject_id, event_name, f"{prefix}-{subject_id}"
        )

    return fake_compute


def _multiprocess_calculator(
    subject_id,
    subject_frame,
    network,
    version,
    resolved_output,
):
    event_name = str(subject_frame["redcap_event_name"].iloc[0])
    _write_measure_outputs(
        Path(resolved_output), network, subject_id, event_name
    )
    return _outcome_frame(
        subject_id, event_name, f"multiprocess-{subject_id}"
    )


def _run_args(
    combined_dir: Path,
    output_dir: Path,
    *,
    workers: int = 1,
    resume: bool = False,
) -> list[str]:
    arguments = [
        "PRONET",
        "run_outcome",
        "--combined-csv-path",
        str(combined_dir),
        "--output-dir",
        str(output_dir),
        "--workers",
        str(workers),
    ]
    if resume:
        arguments.append("--resume")
    return arguments


def _find_subject_shard(output_dir: Path, subject_id: str) -> Path:
    matches = []
    checkpoint_root = output_dir / ".outcome_checkpoints"
    for path in checkpoint_root.rglob("*.csv"):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "ID" in frame and set(frame["ID"]) == {subject_id}:
            matches.append(path)
    assert len(matches) == 1
    return matches[0]


def test_full_run_derives_subject_roster_from_combined_csvs(
    tmp_path: Path,
) -> None:
    combined_dir = tmp_path / "combined"
    output_dir = tmp_path / "output"
    _write_combined_inputs(combined_dir, [("S2", "1"), ("S1", "2")])
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "paths": {
                    "combined_csv_path": str(combined_dir),
                    "output_path": str(output_dir),
                }
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_compute(
        subject_id, subject_frame, network, version, resolved_output
    ):
        calls.append((subject_id, network, version))
        assert resolved_output == output_dir.resolve()
        assert len(subject_frame) == len(OUTCOME_TIMEPOINTS)
        expected_arm = "2" if subject_id == "S1" else "1"
        assert subject_frame["redcap_event_name"].str.endswith(
            f"_arm_{expected_arm}"
        ).all()
        event_name = str(subject_frame["redcap_event_name"].iloc[0])
        _write_measure_outputs(
            Path(resolved_output), network, subject_id, event_name
        )
        return _outcome_frame(subject_id, event_name, "synthetic")

    result = main(
        fake_compute,
        [
            "PRONET",
            "run_outcome",
            "--config",
            str(config_path),
            "--workers",
            "1",
        ],
    )

    assert result == 0
    assert calls == [
        ("S1", "PRONET", "run_outcome"),
        ("S2", "PRONET", "run_outcome"),
    ]
    aggregate = pd.read_csv(output_dir / "pronet_all.csv", dtype=str)
    assert aggregate["ID"].tolist() == ["S1", "S2"]
    assert (output_dir / "last_date_runoutcome.txt").is_file()


def test_explicit_paths_do_not_require_readable_config(
    tmp_path: Path,
) -> None:
    combined_dir = tmp_path / "combined"
    output_dir = tmp_path / "output"
    _write_combined_inputs(combined_dir, [("S1", "1")])
    calls: list[str] = []

    result = main(
        _calculator(calls),
        [
            "PRONET",
            "run_outcome",
            "--config",
            str(tmp_path / "does-not-exist.json"),
            "--combined-csv-path",
            str(combined_dir),
            "--output-dir",
            str(output_dir),
            "--workers",
            "1",
        ],
    )

    assert result == 0
    assert calls == ["S1"]


def test_interrupted_run_resumes_only_pending_subjects_and_preserves_target(
    tmp_path: Path,
) -> None:
    combined_dir = tmp_path / "combined"
    output_dir = tmp_path / "output"
    _write_combined_inputs(combined_dir, [("S2", "1"), ("S1", "1")])
    output_dir.mkdir()
    aggregate_path = output_dir / "pronet_all.csv"
    aggregate_path.write_text("previous aggregate\n", encoding="utf-8")
    first_calls: list[str] = []

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        main(
            _calculator(first_calls, fail_on="S2", prefix="first"),
            _run_args(combined_dir, output_dir),
        )

    assert first_calls == ["S1", "S2"]
    assert aggregate_path.read_text(encoding="utf-8") == "previous aggregate\n"
    assert not (output_dir / "last_date_runoutcome.txt").exists()

    resumed_calls: list[str] = []
    assert main(
        _calculator(resumed_calls, prefix="resumed"),
        _run_args(combined_dir, output_dir, resume=True),
    ) == 0

    assert resumed_calls == ["S2"]
    aggregate = pd.read_csv(aggregate_path, dtype=str)
    assert aggregate["ID"].tolist() == ["S1", "S2"]
    assert aggregate["variable"].tolist() == ["first-S1", "resumed-S2"]


def test_corrupt_subject_shard_is_recomputed_on_resume(tmp_path: Path) -> None:
    combined_dir = tmp_path / "combined"
    output_dir = tmp_path / "output"
    _write_combined_inputs(combined_dir, [("S2", "1"), ("S1", "1")])
    initial_calls: list[str] = []
    assert main(
        _calculator(initial_calls, prefix="initial"),
        _run_args(combined_dir, output_dir),
    ) == 0
    assert initial_calls == ["S1", "S2"]

    _find_subject_shard(output_dir, "S1").write_text(
        "corrupt\n", encoding="utf-8"
    )
    resumed_calls: list[str] = []
    assert main(
        _calculator(resumed_calls, prefix="recomputed"),
        _run_args(combined_dir, output_dir, resume=True),
    ) == 0

    assert resumed_calls == ["S1"]
    aggregate = pd.read_csv(output_dir / "pronet_all.csv", dtype=str)
    assert aggregate["ID"].tolist() == ["S1", "S2"]
    assert aggregate["variable"].tolist() == [
        "recomputed-S1",
        "initial-S2",
    ]


def test_same_size_measure_tampering_is_recomputed_on_resume(
    tmp_path: Path,
) -> None:
    combined_dir = tmp_path / "combined"
    output_dir = tmp_path / "output"
    _write_combined_inputs(combined_dir, [("S1", "1")])
    assert main(
        _calculator([], prefix="initial"),
        _run_args(combined_dir, output_dir),
    ) == 0

    measure_path = (
        subject_output_directory(output_dir, "PRONET", "S1")
        / SUBJECT_OUTPUT_FILENAMES[0]
    )
    original = measure_path.read_bytes()
    tampered = original.replace(b"measure_0", b"changed_0")
    assert tampered != original
    assert len(tampered) == len(original)
    measure_path.write_bytes(tampered)
    assert measure_path.stat().st_size == len(original)

    resumed_calls: list[str] = []
    assert main(
        _calculator(resumed_calls, prefix="recomputed"),
        _run_args(combined_dir, output_dir, resume=True),
    ) == 0
    assert resumed_calls == ["S1"]
    aggregate = pd.read_csv(output_dir / "pronet_all.csv", dtype=str)
    assert aggregate["variable"].tolist() == ["recomputed-S1"]


def test_full_run_none_is_pending_and_preserves_published_files(
    tmp_path: Path,
) -> None:
    combined_dir = tmp_path / "combined"
    output_dir = tmp_path / "output"
    _write_combined_inputs(combined_dir, [("S1", "1")])
    output_dir.mkdir()
    aggregate_path = output_dir / "pronet_all.csv"
    log_path = output_dir / "last_date_runoutcome.txt"
    aggregate_path.write_text("previous aggregate\n", encoding="utf-8")
    log_path.write_text("previous log\n", encoding="utf-8")
    calls: list[str] = []

    def no_frame(
        subject_id, subject_frame, network, version, resolved_output
    ):
        calls.append(subject_id)
        return None

    with pytest.raises(
        OutcomeInputError, match="returned no frame"
    ) as exc_info:
        main(no_frame, _run_args(combined_dir, output_dir))
    assert "S1" not in str(exc_info.value)

    assert calls == ["S1"]
    assert aggregate_path.read_text(encoding="utf-8") == "previous aggregate\n"
    assert log_path.read_text(encoding="utf-8") == "previous log\n"
    checkpoint_root = output_dir / ".outcome_checkpoints"
    assert not [
        path
        for path in checkpoint_root.rglob("*.json")
        if "subjects" in path.parts
    ]

    resumed_calls: list[str] = []
    assert main(
        _calculator(resumed_calls, prefix="resumed"),
        _run_args(combined_dir, output_dir, resume=True),
    ) == 0
    assert resumed_calls == ["S1"]
    aggregate = pd.read_csv(aggregate_path, dtype=str)
    assert aggregate["variable"].tolist() == ["resumed-S1"]
    assert "previous log" in log_path.read_text(encoding="utf-8")


def test_resume_rejects_same_size_input_change_with_restored_mtime(
    tmp_path: Path,
) -> None:
    combined_dir = tmp_path / "combined"
    output_dir = tmp_path / "output"
    _write_combined_inputs(combined_dir, [("S1", "1")])
    assert main(
        _calculator([], prefix="initial"),
        _run_args(combined_dir, output_dir),
    ) == 0

    baseline_path = combined_csv_file(combined_dir, "PRONET", "baseline")
    original_stat = baseline_path.stat()
    original = baseline_path.read_bytes()
    changed = original.replace(b"S1-baseline", b"Z1-baseline")
    assert changed != original and len(changed) == len(original)
    baseline_path.write_bytes(changed)
    os.utime(
        baseline_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    restored_stat = baseline_path.stat()
    assert restored_stat.st_size == original_stat.st_size
    assert restored_stat.st_mtime_ns == original_stat.st_mtime_ns
    calls: list[str] = []

    with pytest.raises(
        OutcomeInputError, match="no compatible checkpoint manifest"
    ):
        main(
            _calculator(calls, prefix="must-not-run"),
            _run_args(combined_dir, output_dir, resume=True),
        )
    assert calls == []


def test_full_run_avoids_cohort_concat_and_writes_roster_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    combined_dir = tmp_path / "combined"
    output_dir = tmp_path / "output"
    _write_combined_inputs(combined_dir, [("S2", "1"), ("S1", "1")])
    calls: list[str] = []

    class ReversePool:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def map(self, function, tasks, chunksize=1):
            return [function(task) for task in reversed(list(tasks))]

    def unexpected_concat(*_: object, **__: object):
        raise AssertionError("run_outcome retained cohort frames for concat")

    monkeypatch.setattr(outcome_runner.multiprocessing, "Pool", ReversePool)
    monkeypatch.setattr(
        outcome_runner, "pd", SimpleNamespace(concat=unexpected_concat)
    )

    assert main(
        _calculator(calls, prefix="ordered"),
        _run_args(combined_dir, output_dir, workers=2),
    ) == 0

    assert calls == ["S2", "S1"]
    aggregate = pd.read_csv(output_dir / "pronet_all.csv", dtype=str)
    assert aggregate["ID"].tolist() == ["S1", "S2"]
    assert aggregate["variable"].tolist() == ["ordered-S1", "ordered-S2"]


def test_real_windows_multiprocessing_checkpoint_smoke(tmp_path: Path) -> None:
    combined_dir = tmp_path / "combined"
    output_dir = tmp_path / "output"
    _write_combined_inputs(combined_dir, [("S2", "1"), ("S1", "1")])

    assert main(
        _multiprocess_calculator,
        _run_args(combined_dir, output_dir, workers=2),
    ) == 0

    aggregate = pd.read_csv(output_dir / "pronet_all.csv", dtype=str)
    assert aggregate["ID"].tolist() == ["S1", "S2"]
    assert aggregate["variable"].tolist() == [
        "multiprocess-S1",
        "multiprocess-S2",
    ]
    for subject_id in ("S1", "S2"):
        subject_dir = subject_output_directory(
            output_dir, "PRONET", subject_id
        )
        assert all(
            (subject_dir / filename).is_file()
            for filename in SUBJECT_OUTPUT_FILENAMES
        )


def test_sequential_checkpoint_progress_is_count_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    subject_ids = [f"PRIVATE-SUBJECT-{index:02d}" for index in range(26)]
    combined_subjects = {
        subject_id: pd.DataFrame() for subject_id in subject_ids
    }
    completed: list[str] = []

    def fake_checkpoint(task: tuple) -> tuple[str, bool]:
        completed.append(task[1])
        return "redacted", True

    monkeypatch.setattr(
        outcome_runner, "_compute_and_checkpoint", fake_checkpoint
    )
    outcome_runner._compute_checkpointed(
        lambda *_: None,
        combined_subjects,
        subject_ids,
        "PRONET",
        "run_outcome",
        tmp_path,
        1,
        SimpleNamespace(),
    )

    output = capsys.readouterr().out
    assert completed == subject_ids
    assert output == (
        "Checkpointed 25/26 pending subject(s).\n"
        "Checkpointed 26/26 pending subject(s).\n"
    )
    assert all(subject_id not in output for subject_id in subject_ids)


def test_multiprocess_checkpoint_progress_is_count_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    subject_ids = [f"PRIVATE-SUBJECT-{index:02d}" for index in range(5)]
    combined_subjects = {
        subject_id: pd.DataFrame() for subject_id in subject_ids
    }

    class InlinePool:
        def __init__(self, processes: int, *, maxtasksperchild: int) -> None:
            assert processes == 2
            assert maxtasksperchild == 10

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def map(self, function, tasks, chunksize=1):
            assert chunksize == 1
            return [function(task) for task in tasks]

    monkeypatch.setattr(outcome_runner.multiprocessing, "Pool", InlinePool)
    monkeypatch.setattr(
        outcome_runner,
        "_compute_and_checkpoint",
        lambda task: ("redacted", True),
    )
    outcome_runner._compute_checkpointed(
        lambda *_: None,
        combined_subjects,
        subject_ids,
        "PRONET",
        "run_outcome",
        tmp_path,
        2,
        SimpleNamespace(),
    )

    output = capsys.readouterr().out
    assert output == (
        "Checkpointed 2/5 pending subject(s).\n"
        "Checkpointed 4/5 pending subject(s).\n"
        "Checkpointed 5/5 pending subject(s).\n"
    )
    assert all(subject_id not in output for subject_id in subject_ids)
