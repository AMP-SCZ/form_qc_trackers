"""Command-line orchestration for the AMP-SCZ outcome calculations."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
from typing import Callable, Optional, Sequence
from uuid import uuid4

import pandas as pd

from outcome_checkpoints import (
    RunCheckpointContext,
    build_run_manifest_payload,
    load_subject_checkpoint,
    outcome_run_lock,
    prepare_run_checkpoint,
    snapshot_combined_inputs,
    write_aggregate_from_checkpoints,
    write_run_state,
    write_subject_checkpoint,
)
from outcome_inputs import (
    OutcomeInputError,
    load_combined_subjects,
    normalize_network,
)


TEST_SUBJECTS = {
    "PRONET": (
        "YA16606", "YA01508", "LA00145", "LA00834",
        "OR00697", "PI01355", "HA04408",
    ),
    "PRESCIENT": (
        "ME00772", "ME78581", "BM90491", "ME33634",
        "ME20845", "BM73097", "ME21922",
    ),
}

SINGLE_SUBJECTS = {
    "PRONET": ("SI63982",),
    "PRESCIENT": ("BM58352", "ME31725", "CP08326"),
}

RUN_VERSIONS = ("test", "create_control", "single_subject", "run_outcome")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate AMP-SCZ outcomes from the canonical combined REDCap "
            "CSVs for one network."
        )
    )
    parser.add_argument("network", help="PRONET or PRESCIENT")
    parser.add_argument("version", choices=RUN_VERSIONS)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.json"),
        help="config JSON containing paths.combined_csv_path",
    )
    parser.add_argument(
        "--combined-csv-path",
        type=Path,
        help="override config paths.combined_csv_path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override config paths.output_path for aggregate outputs and logs",
    )
    parser.add_argument(
        "--subject-id",
        action="append",
        dest="subject_ids",
        help="limit a non-full run to this subject; may be repeated",
    )
    parser.add_argument(
        "--subject-id-file",
        type=Path,
        help=(
            "protected one-column Subject CSV used for an explicitly "
            "affected-only full-checkpoint run"
        ),
    )
    parser.add_argument(
        "--allow-partial-roster",
        action="store_true",
        help="permit run_outcome to publish an explicitly affected-only aggregate",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="worker count (default: SLURM_CPUS_PER_TASK or 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume an interrupted full run only when its input, roster, and "
            "calculation-code manifest match exactly"
        ),
    )
    return parser


def _load_config(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise OutcomeInputError(f"Could not read config JSON {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise OutcomeInputError(f"Config root must be an object: {path}")
    return config


def _configured_path(config: dict, config_path: Path, key: str) -> Optional[Path]:
    try:
        raw_path = config["paths"][key]
    except (KeyError, TypeError):
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def requested_subject_ids(
    network: str, version: str, explicit_ids: Optional[Sequence[str]] = None
) -> Optional[list[str]]:
    """Return requested IDs, or None when all combined-CSV IDs should run."""
    normalized_network = normalize_network(network)
    if explicit_ids:
        cleaned = [str(subject_id).strip() for subject_id in explicit_ids]
        if any(not subject_id for subject_id in cleaned):
            raise OutcomeInputError("--subject-id cannot be blank")
        return list(dict.fromkeys(cleaned))
    if version in {"test", "create_control"}:
        return list(TEST_SUBJECTS[normalized_network])
    if version == "single_subject":
        return list(SINGLE_SUBJECTS[normalized_network])
    return None


def load_subject_id_file(path: str | Path) -> list[str]:
    """Load a private, exact one-column affected-subject roster."""

    roster_path = Path(path).expanduser().resolve()
    try:
        with roster_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            try:
                header = next(reader)
            except StopIteration:
                raise OutcomeInputError("Affected-subject roster is empty") from None
            if header != ["Subject"]:
                raise OutcomeInputError(
                    "Affected-subject roster must have exactly one Subject column"
                )
            subject_ids: list[str] = []
            for row in reader:
                if len(row) != 1:
                    raise OutcomeInputError(
                        "Affected-subject roster has a malformed logical row"
                    )
                subject_id = row[0].strip()
                if not subject_id:
                    raise OutcomeInputError(
                        "Affected-subject roster contains a blank subject"
                    )
                subject_ids.append(subject_id)
    except OutcomeInputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise OutcomeInputError("Affected-subject roster could not be read") from exc
    if not subject_ids:
        raise OutcomeInputError("Affected-subject roster contains no subjects")
    if len(set(subject_ids)) != len(subject_ids):
        raise OutcomeInputError("Affected-subject roster contains duplicate subjects")
    return subject_ids


def _unique_temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = _unique_temporary(path)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_last_run_log(
    output_dir: Path, network: str, version: str, subject_ids: Sequence[str]
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    display_network = normalize_network(network).capitalize()
    if version == "single_subject":
        message = (
            f"Single-subject outcome calculation ({len(subject_ids)} subject(s)) "
            f"for {display_network} was last completed on: {now}"
        )
    else:
        message = (
            f"{display_network}: outcome calculations were run for the last "
            f"time on: {now}"
        )

    log_path = output_dir / "last_date_runoutcome.txt"
    try:
        existing = log_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = ""
    except OSError as exc:
        raise OutcomeInputError(f"Could not read outcome run log {log_path}") from exc
    _atomic_write_text(log_path, f"{message}\n{existing}")
    print(f"Log written to {log_path}")


def _aggregate_filename(network: str, version: str) -> Optional[str]:
    normalized_network = normalize_network(network).lower()
    if version == "test":
        return f"{normalized_network}_new_subjects.csv"
    if version == "create_control":
        return f"{normalized_network}_test_subjects.csv"
    if version == "run_outcome":
        return f"{normalized_network}_all.csv"
    return None


def _write_aggregate(
    frame: pd.DataFrame, output_dir: Path, network: str, version: str
) -> None:
    filename = _aggregate_filename(network, version)
    if filename is None:
        return
    target_path = output_dir / filename
    temporary = _unique_temporary(target_path)
    try:
        with temporary.open("x", encoding="utf-8", newline="") as target:
            frame.to_csv(
                target,
                index=False,
                header=True,
                float_format="%.3f",
                lineterminator="\n",
            )
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, target_path)
    finally:
        temporary.unlink(missing_ok=True)
    if version == "test":
        print("Wrote the test subjects outcome file.")
    elif version == "run_outcome":
        print("Wrote the full outcome file.")


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    config_path = args.config.expanduser().resolve()
    config = {}
    if args.combined_csv_path is None or args.output_dir is None:
        config = _load_config(config_path)

    combined_path = args.combined_csv_path
    if combined_path is None:
        combined_path = _configured_path(
            config, config_path, "combined_csv_path"
        )
    if combined_path is None:
        raise OutcomeInputError(
            "No combined CSV directory supplied and config is missing "
            "paths.combined_csv_path"
        )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = _configured_path(config, config_path, "output_path")
    if output_dir is None:
        output_dir = config_path.parent
    return (
        combined_path.expanduser().resolve(),
        output_dir.expanduser().resolve(),
    )


def _resolve_workers(requested_workers: Optional[int]) -> int:
    workers = requested_workers
    if workers is None:
        try:
            workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        except ValueError as exc:
            raise OutcomeInputError(
                "SLURM_CPUS_PER_TASK must be a positive integer"
            ) from exc
    if workers < 1:
        raise OutcomeInputError("--workers must be a positive integer")
    return workers


def _compute_bounded(
    compute_outcomes: Callable[..., Optional[pd.DataFrame]],
    combined_subjects,
    subject_ids: Sequence[str],
    network: str,
    version: str,
    output_dir: Path,
    workers: int,
) -> list[pd.DataFrame]:
    """Calculate the deliberately small non-full-run rosters."""
    outcomes: list[pd.DataFrame] = []
    if workers == 1:
        for subject_id in subject_ids:
            outcome = compute_outcomes(
                subject_id,
                combined_subjects[subject_id],
                network,
                version,
                output_dir,
            )
            if outcome is not None:
                outcomes.append(outcome)
        return outcomes

    pool_workers = min(workers, len(subject_ids))
    with multiprocessing.Pool(
        pool_workers, maxtasksperchild=10
    ) as pool:
        for start in range(0, len(subject_ids), pool_workers):
            batch_ids = subject_ids[start : start + pool_workers]
            tasks = [
                (
                    subject_id,
                    combined_subjects[subject_id],
                    network,
                    version,
                    output_dir,
                )
                for subject_id in batch_ids
            ]
            outcomes.extend(
                outcome
                for outcome in pool.starmap(compute_outcomes, tasks)
                if outcome is not None
            )
    return outcomes


def _compute_and_checkpoint(task: tuple) -> tuple[str, bool]:
    """Picklable worker wrapper that returns only a tiny completion result."""
    (
        compute_outcomes,
        subject_id,
        subject_frame,
        network,
        version,
        output_dir,
        context,
        require_frame,
    ) = task
    outcome = compute_outcomes(
        subject_id,
        subject_frame,
        network,
        version,
        output_dir,
    )
    if require_frame and outcome is None:
        raise OutcomeInputError(
            "Full outcome calculation returned no frame"
        )
    write_subject_checkpoint(context, subject_id, outcome)
    return str(subject_id), outcome is not None


def _compute_checkpointed(
    compute_outcomes: Callable[..., Optional[pd.DataFrame]],
    combined_subjects,
    subject_ids: Sequence[str],
    network: str,
    version: str,
    output_dir: Path,
    workers: int,
    context: RunCheckpointContext,
) -> None:
    """Compute full-run subjects while retaining no returned cohort frames."""
    if not subject_ids:
        return
    require_frame = version == "run_outcome"
    if workers == 1:
        for completed, subject_id in enumerate(subject_ids, start=1):
            _compute_and_checkpoint(
                (
                    compute_outcomes,
                    subject_id,
                    combined_subjects[subject_id],
                    network,
                    version,
                    output_dir,
                    context,
                    require_frame,
                )
            )
            if completed % 25 == 0 or completed == len(subject_ids):
                print(
                    f"Checkpointed {completed}/{len(subject_ids)} "
                    "pending subject(s).",
                    flush=True,
                )
        return

    pool_workers = min(workers, len(subject_ids))
    with multiprocessing.Pool(
        pool_workers, maxtasksperchild=10
    ) as pool:
        for start in range(0, len(subject_ids), pool_workers):
            batch_ids = subject_ids[start : start + pool_workers]
            tasks = [
                (
                    compute_outcomes,
                    subject_id,
                    combined_subjects[subject_id],
                    network,
                    version,
                    output_dir,
                    context,
                    require_frame,
                )
                for subject_id in batch_ids
            ]
            pool.map(_compute_and_checkpoint, tasks, chunksize=1)
            completed = start + len(batch_ids)
            print(
                f"Checkpointed {completed}/{len(subject_ids)} "
                "pending subject(s).",
                flush=True,
            )


def _load_subject_roster(
    combined_path: Path,
    network: str,
    version: str,
    explicit_ids: Optional[Sequence[str]],
    *, require_all_requested: bool = False,
):
    requested_ids = requested_subject_ids(network, version, explicit_ids)
    combined_subjects = load_combined_subjects(
        combined_path,
        network,
        subject_ids=requested_ids,
    )
    if not combined_subjects:
        raise OutcomeInputError(
            "None of the requested subjects were found in the combined CSVs"
        )

    if requested_ids is None:
        subject_ids = sorted(combined_subjects)
    else:
        missing_ids = [
            subject_id
            for subject_id in requested_ids
            if subject_id not in combined_subjects
        ]
        if missing_ids and require_all_requested:
            raise OutcomeInputError(
                f"Affected-subject roster is missing {len(missing_ids)} subject(s)"
            )
        if missing_ids and not require_all_requested:
            print(
                f"WARNING: {len(missing_ids)} requested subject(s) were absent "
                "from the combined CSVs and will be skipped."
            )
        subject_ids = [
            subject_id
            for subject_id in requested_ids
            if subject_id in combined_subjects
        ]
    if not subject_ids:
        raise OutcomeInputError("No combined-CSV subjects remain to calculate")
    return combined_subjects, subject_ids


def _run_full_outcome(
    compute_outcomes: Callable[..., Optional[pd.DataFrame]],
    combined_subjects,
    subject_ids: Sequence[str],
    combined_path: Path,
    output_dir: Path,
    network: str,
    workers: int,
    input_snapshot: Sequence[dict[str, object]],
    *,
    resume: bool,
    roster_mode: str = "full",
) -> None:
    payload = build_run_manifest_payload(
        combined_path,
        network,
        "run_outcome",
        subject_ids,
        input_snapshot,
        roster_mode=roster_mode,
    )
    context = prepare_run_checkpoint(output_dir, payload, resume=resume)
    write_run_state(context, "running")
    pending_ids: list[str] = []
    reused = 0
    for subject_id in subject_ids:
        checkpoint = load_subject_checkpoint(
            context,
            subject_id,
            strict=False,
        )
        if checkpoint is None or checkpoint.status != "frame":
            pending_ids.append(subject_id)
        else:
            reused += 1
    if resume:
        print(
            f"Verified {reused} completed checkpoint(s); "
            f"{len(pending_ids)} subject(s) remain."
        )

    try:
        _compute_checkpointed(
            compute_outcomes,
            combined_subjects,
            pending_ids,
            network,
            "run_outcome",
            output_dir,
            workers,
            context,
        )
        if roster_mode == "affected_only":
            filename = f"{normalize_network(network).lower()}_affected.csv"
        else:
            filename = _aggregate_filename(network, "run_outcome")
            assert filename is not None
        completed_frames = write_aggregate_from_checkpoints(
            context,
            output_dir,
            filename,
        )
        _write_last_run_log(output_dir, network, "run_outcome", subject_ids)
        write_run_state(
            context,
            "complete",
            detail=f"{completed_frames} subject frame(s)",
        )
        print("Wrote the outcome aggregate.")
    except BaseException as exc:
        try:
            write_run_state(context, "failed", detail=type(exc).__name__)
        except Exception:
            pass
        raise


def _run_locked(
    compute_outcomes: Callable[..., Optional[pd.DataFrame]],
    args: argparse.Namespace,
    network: str,
    combined_path: Path,
    output_dir: Path,
    workers: int,
) -> int:
    input_snapshot = None
    if args.version == "run_outcome":
        input_snapshot = snapshot_combined_inputs(combined_path, network)
    combined_subjects, subject_ids = _load_subject_roster(
        combined_path,
        network,
        args.version,
        args.subject_ids,
        require_all_requested=args.allow_partial_roster,
    )
    print(f"Number of processes: {min(workers, len(subject_ids))}")

    if args.version == "run_outcome":
        assert input_snapshot is not None
        _run_full_outcome(
            compute_outcomes,
            combined_subjects,
            subject_ids,
            combined_path,
            output_dir,
            network,
            workers,
            input_snapshot,
            resume=args.resume,
            roster_mode="affected_only" if args.allow_partial_roster else "full",
        )
        return 0

    outcomes = _compute_bounded(
        compute_outcomes,
        combined_subjects,
        subject_ids,
        network,
        args.version,
        output_dir,
        workers,
    )
    if not outcomes:
        raise OutcomeInputError(
            "Outcome calculations produced no subject output frames"
        )
    concatenated = pd.concat(outcomes, ignore_index=True, sort=False)
    _write_aggregate(concatenated, output_dir, network, args.version)
    if args.version == "single_subject":
        _write_last_run_log(
            output_dir, network, args.version, subject_ids
        )
    return 0


def main(
    compute_outcomes: Callable[..., Optional[pd.DataFrame]],
    argv: Optional[Sequence[str]] = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.resume and args.version != "run_outcome":
        raise OutcomeInputError("--resume is supported only for run_outcome")
    if args.subject_ids and args.subject_id_file is not None:
        raise OutcomeInputError(
            "--subject-id and --subject-id-file cannot be used together"
        )
    file_subject_ids = None
    if args.subject_id_file is not None:
        file_subject_ids = load_subject_id_file(args.subject_id_file)
    if args.version == "run_outcome":
        if args.subject_ids:
            raise OutcomeInputError(
                "--subject-id cannot be used with run_outcome because a subset "
                "must not overwrite the full-network aggregate"
            )
        if args.subject_id_file is not None and not args.allow_partial_roster:
            raise OutcomeInputError(
                "--subject-id-file requires --allow-partial-roster for run_outcome"
            )
        if args.allow_partial_roster and args.subject_id_file is None:
            raise OutcomeInputError(
                "--allow-partial-roster requires --subject-id-file"
            )
    elif args.subject_id_file is not None or args.allow_partial_roster:
        raise OutcomeInputError(
            "affected-subject roster options are supported only for run_outcome"
        )
    if file_subject_ids is not None:
        args.subject_ids = file_subject_ids

    network = normalize_network(args.network)
    combined_path, output_dir = _resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = _resolve_workers(args.workers)
    with outcome_run_lock(output_dir, network, args.version):
        return _run_locked(
            compute_outcomes,
            args,
            network,
            combined_path,
            output_dir,
            workers,
        )
