"""Atomic, bounded checkpoints for full outcome-calculation runs."""

from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Iterator, Optional, Sequence
from urllib.parse import quote
from uuid import uuid4

import numpy as np
import pandas as pd

from outcome_inputs import (
    OUTCOME_TIMEPOINTS,
    OutcomeInputError,
    combined_csv_file,
    normalize_network,
)


CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_COLUMNS = (
    "data_type",
    "redcap_event_name",
    "value",
    "variable",
    "ID",
)
_CODE_FINGERPRINT_FILENAMES = (
    "outcome_calculations.py",
    "outcome_aggregation.py",
    "outcome_scoring.py",
    "outcome_inputs.py",
    "outcome_runner.py",
    "outcome_checkpoints.py",
)


@dataclass(frozen=True)
class RunCheckpointContext:
    """Immutable identity for one resumable run generation."""

    root: Path
    network: str
    version: str
    signature: str
    run_id: str
    subject_ids: tuple[str, ...]


@dataclass(frozen=True)
class SubjectCheckpoint:
    """A verified subject completion marker and its optional output frame."""

    status: str
    frame: Optional[pd.DataFrame]


def _safe_component(value: str) -> str:
    return quote(str(value), safe="").replace(".", "%2E")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise OutcomeInputError("Could not validate checkpoint artifact") from None
    return digest.hexdigest()


def _unique_temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary(path)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, f"{_canonical_json(value)}\n")


def snapshot_combined_inputs(
    combined_csv_path: str | Path, network: str
) -> tuple[dict[str, object], ...]:
    """Fingerprint canonical inputs by metadata and file content."""
    root = Path(combined_csv_path).expanduser().resolve()
    normalized_network = normalize_network(network)
    snapshot: list[dict[str, object]] = []
    for timepoint in OUTCOME_TIMEPOINTS:
        path = combined_csv_file(root, normalized_network, timepoint)
        try:
            metadata = path.stat()
        except FileNotFoundError:
            raise OutcomeInputError(f"Missing combined CSV: {path}") from None
        except OSError as exc:
            raise OutcomeInputError(
                f"Could not inspect combined CSV metadata: {path}"
            ) from exc
        if not path.is_file():
            raise OutcomeInputError(f"Combined CSV path is not a file: {path}")
        snapshot.append(
            {
                "name": path.name,
                "size": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
                "sha256": _sha256_file(path),
            }
        )
    return tuple(snapshot)


def _code_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for filename in _CODE_FINGERPRINT_FILENAMES:
        path = root / filename
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise OutcomeInputError(
                f"Could not fingerprint outcome source file: {path}"
            ) from exc
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_identity() -> dict[str, str]:
    """Return calculation-runtime versions that can affect outcome results."""
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "pandas_version": str(pd.__version__),
        "numpy_version": str(np.__version__),
    }


def build_run_manifest_payload(
    combined_csv_path: str | Path,
    network: str,
    version: str,
    subject_ids: Sequence[str],
    input_snapshot: Sequence[dict[str, object]],
    *,
    roster_mode: str = "full",
) -> dict[str, object]:
    """Build the immutable resume identity for a loaded input snapshot."""
    normalized_network = normalize_network(network)
    normalized_ids = tuple(str(subject_id).strip() for subject_id in subject_ids)
    if not normalized_ids or any(not subject_id for subject_id in normalized_ids):
        raise OutcomeInputError("A checkpoint run requires nonblank subject IDs")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise OutcomeInputError("A checkpoint run requires unique subject IDs")
    if roster_mode not in {"full", "affected_only"}:
        raise OutcomeInputError("A checkpoint run has an invalid roster mode")
    expected_snapshot = snapshot_combined_inputs(combined_csv_path, normalized_network)
    if tuple(input_snapshot) != expected_snapshot:
        raise OutcomeInputError(
            "Combined CSV metadata changed while the inputs were being loaded; "
            "restart the outcome run"
        )
    return {
        "checkpoint_format": CHECKPOINT_FORMAT_VERSION,
        "network": normalized_network,
        "version": str(version),
        "roster_mode": roster_mode,
        "combined_csv_path": str(Path(combined_csv_path).expanduser().resolve()),
        "subject_ids": list(normalized_ids),
        "input_files": list(expected_snapshot),
        "code_sha256": _code_fingerprint(),
        "runtime": _runtime_identity(),
    }


def prepare_run_checkpoint(
    output_dir: str | Path,
    payload: dict[str, object],
    *,
    resume: bool,
) -> RunCheckpointContext:
    """Create a fresh run generation or reopen its exact resume manifest."""
    signature = _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    network = normalize_network(str(payload.get("network", "")))
    version = str(payload.get("version", ""))
    subject_ids = tuple(str(value) for value in payload.get("subject_ids", []))
    root = (
        Path(output_dir)
        / ".outcome_checkpoints"
        / network.lower()
        / _safe_component(version)
        / signature
    )
    manifest_path = root / "manifest.json"

    if resume:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise OutcomeInputError(
                "--resume found no compatible checkpoint manifest for the "
                "current inputs, roster, and outcome code"
            ) from None
        except (OSError, json.JSONDecodeError) as exc:
            raise OutcomeInputError(
                "Could not read checkpoint manifest"
            ) from None
        if (
            manifest.get("signature") != signature
            or manifest.get("payload") != payload
            or not str(manifest.get("run_id", "")).strip()
        ):
            raise OutcomeInputError(
                "Checkpoint manifest does not match the current outcome run"
            )
        run_id = str(manifest["run_id"])
    else:
        run_id = uuid4().hex
        manifest = {
            "signature": signature,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        _atomic_write_json(manifest_path, manifest)

    return RunCheckpointContext(
        root=root,
        network=network,
        version=version,
        signature=signature,
        run_id=run_id,
        subject_ids=subject_ids,
    )


def write_run_state(
    context: RunCheckpointContext,
    status: str,
    *,
    detail: Optional[str] = None,
) -> None:
    state: dict[str, object] = {
        "signature": context.signature,
        "run_id": context.run_id,
        "status": str(status),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if detail:
        state["detail"] = str(detail)
    _atomic_write_json(context.root / "state.json", state)


def _subject_paths(
    context: RunCheckpointContext, subject_id: str
) -> tuple[Path, Path]:
    token = _sha256_bytes(str(subject_id).encode("utf-8"))
    root = context.root / "subjects" / token[:2]
    return root / f"{token}.csv", root / f"{token}.json"


def _read_csv_header(path: Path, *, label: str) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            header = next(csv.reader(source, strict=True))
    except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
        raise OutcomeInputError(f"Could not validate {label}") from None
    if not header or any(not name for name in header):
        raise OutcomeInputError(f"Invalid {label} header")
    if len(header) != len(set(header)):
        raise OutcomeInputError(f"Duplicate columns in {label}")
    return tuple(header)


def _normalize_checkpoint_frame(
    frame: pd.DataFrame, subject_id: str, _source: Path
) -> pd.DataFrame:
    if frame.empty:
        raise OutcomeInputError("Outcome checkpoint is empty")
    if frame.columns.duplicated().any():
        raise OutcomeInputError("Outcome checkpoint has duplicate columns")
    if set(frame.columns) != set(CHECKPOINT_COLUMNS):
        raise OutcomeInputError(
            "Outcome checkpoint has an unexpected schema"
        )
    normalized = frame.loc[:, CHECKPOINT_COLUMNS].copy()
    observed_ids = set(normalized["ID"].astype(str).str.strip())
    if observed_ids != {str(subject_id)}:
        raise OutcomeInputError(
            "Outcome checkpoint ID does not match its subject token"
        )
    return normalized


def write_subject_checkpoint(
    context: RunCheckpointContext,
    subject_id: str,
    frame: Optional[pd.DataFrame],
) -> SubjectCheckpoint:
    """Atomically commit one subject result; metadata is the commit marker."""
    checkpoint_path, metadata_path = _subject_paths(context, subject_id)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "checkpoint_format": CHECKPOINT_FORMAT_VERSION,
        "signature": context.signature,
        "run_id": context.run_id,
        "subject_id": str(subject_id),
    }
    if frame is None:
        metadata["status"] = "none"
        _atomic_write_json(metadata_path, metadata)
        return SubjectCheckpoint(status="none", frame=None)

    normalized = _normalize_checkpoint_frame(frame, subject_id, checkpoint_path)
    temporary = _unique_temporary(checkpoint_path)
    try:
        with temporary.open("x", encoding="utf-8", newline="") as target:
            normalized.to_csv(
                target,
                index=False,
                header=True,
                float_format="%.3f",
                lineterminator="\n",
            )
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, checkpoint_path)
    finally:
        temporary.unlink(missing_ok=True)

    metadata.update(
        {
            "status": "frame",
            "rows": len(normalized),
            "columns": list(CHECKPOINT_COLUMNS),
            "csv_sha256": _sha256_file(checkpoint_path),
        }
    )
    _atomic_write_json(metadata_path, metadata)
    return SubjectCheckpoint(status="frame", frame=normalized)


def _load_subject_checkpoint_strict(
    context: RunCheckpointContext,
    subject_id: str,
) -> Optional[SubjectCheckpoint]:
    checkpoint_path, metadata_path = _subject_paths(context, subject_id)
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutcomeInputError(
            "Could not read subject checkpoint metadata"
        ) from None
    if (
        metadata.get("checkpoint_format") != CHECKPOINT_FORMAT_VERSION
        or metadata.get("signature") != context.signature
        or metadata.get("run_id") != context.run_id
        or metadata.get("subject_id") != str(subject_id)
    ):
        return None
    status = metadata.get("status")
    if status == "none":
        return SubjectCheckpoint(status="none", frame=None)
    if status != "frame":
        raise OutcomeInputError(
            "Subject checkpoint has an invalid status"
        )
    if not checkpoint_path.is_file():
        raise OutcomeInputError(
            "Subject checkpoint data is missing"
        )
    if metadata.get("csv_sha256") != _sha256_file(checkpoint_path):
        raise OutcomeInputError(
            "Subject checkpoint data failed its integrity check"
        )
    header = _read_csv_header(checkpoint_path, label="subject checkpoint")
    if header != CHECKPOINT_COLUMNS:
        raise OutcomeInputError(
            "Subject checkpoint has an unexpected header"
        )
    try:
        frame = pd.read_csv(
            checkpoint_path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            on_bad_lines="error",
            low_memory=False,
        )
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise OutcomeInputError(
            "Could not read subject checkpoint"
        ) from None
    normalized = _normalize_checkpoint_frame(frame, subject_id, checkpoint_path)
    if metadata.get("rows") != len(normalized):
        raise OutcomeInputError(
            "Subject checkpoint row count does not match metadata"
        )
    if metadata.get("columns") != list(CHECKPOINT_COLUMNS):
        raise OutcomeInputError(
            "Subject checkpoint column metadata is invalid"
        )
    return SubjectCheckpoint(status="frame", frame=normalized)


def load_subject_checkpoint(
    context: RunCheckpointContext,
    subject_id: str,
    *,
    strict: bool = False,
) -> Optional[SubjectCheckpoint]:
    """Load a verified completion, or treat corruption as pending on resume."""
    try:
        return _load_subject_checkpoint_strict(context, subject_id)
    except OutcomeInputError:
        if strict:
            raise
        return None


def write_aggregate_from_checkpoints(
    context: RunCheckpointContext,
    output_dir: str | Path,
    filename: str,
) -> int:
    """Stream verified shards in roster order and atomically promote output."""
    target_path = Path(output_dir) / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary(target_path)
    frame_count = 0
    try:
        with temporary.open("x", encoding="utf-8", newline="") as target:
            for subject_id in context.subject_ids:
                checkpoint = load_subject_checkpoint(
                    context,
                    subject_id,
                    strict=True,
                )
                if checkpoint is None:
                    raise OutcomeInputError(
                        "A requested subject has no completed checkpoint"
                    )
                if checkpoint.status == "none":
                    continue
                assert checkpoint.frame is not None
                checkpoint.frame.to_csv(
                    target,
                    index=False,
                    header=frame_count == 0,
                    float_format="%.3f",
                    lineterminator="\n",
                )
                frame_count += 1
            target.flush()
            os.fsync(target.fileno())
        if frame_count == 0:
            raise OutcomeInputError(
                "Outcome calculations produced no subject output frames"
            )
        os.replace(temporary, target_path)
    finally:
        temporary.unlink(missing_ok=True)
    return frame_count


@contextmanager
def outcome_run_lock(
    output_dir: str | Path, network: str, version: str
) -> Iterator[None]:
    """Serialize every outcome writer sharing one output root."""
    lock_path = (
        Path(output_dir)
        / ".outcome_checkpoints"
        / "locks"
        / "output-root.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    try:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OutcomeInputError(
                "Another outcome run is already writing this output root"
            ) from exc
        acquired = True
        yield
    finally:
        if acquired:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()
