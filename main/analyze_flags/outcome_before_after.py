"""Run outcome calculations against available original and latest values.

The input value pairs come from Analyze Flags'
``resolved_before_after_values_<NETWORK>.csv`` artifact.  Analyze Flags can
only recover an original value when the raw flag evidence proves it, and its
``current_value`` is the value in the latest combined export at analytics
time.  Consequently these scenarios are deliberately named
``original_available_values`` and ``latest_current_values``; they are not
claimed to be complete historical snapshots.

The protected combined CSVs are never modified.  This program validates each
pair against the source cell, publishes two immutable staged input trees, and
then invokes the existing outcome runner once per tree in separate output
directories.  A streaming comparison reports only outcome values that differ.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import groupby, zip_longest
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Optional, Sequence
from uuid import uuid4

from outcome_checkpoints import CHECKPOINT_COLUMNS
from outcome_inputs import (
    OUTCOME_TIMEPOINTS,
    OutcomeInputError,
    combined_csv_file,
    load_combined_subjects,
    normalize_network,
    normalize_timepoint,
)
from outcome_runner import load_subject_id_file


MANIFEST_FORMAT_VERSION = 3
DEFAULT_OUTCOME_DISK_RESERVE_GB = 10.0
DEFAULT_OUTCOME_DISK_RESERVE_BYTES = int(
    DEFAULT_OUTCOME_DISK_RESERVE_GB * 1024**3
)
STAGING_OVERHEAD_BYTES = 256 * 1024 * 1024
ORIGINAL_SCENARIO = "original_available_values"
CURRENT_SCENARIO = "latest_current_values"
PAIR_STATUS_ALLOWLIST = frozenset({"filled", "still_blank", "missing_code"})
REQUIRED_PAIR_COLUMNS = (
    "Subject",
    "Timepoint",
    "General_Flag",
    "variable",
    "message_variable",
    "canonical_template",
    "before_value_available",
    "before_value",
    "before_value_source",
    "Latest_seen",
    "status",
    "current_value",
)
IMMUTABLE_IDENTITY_COLUMNS = frozenset(
    {
        "subjectid",
        "redcap_event_name",
        "event_name",
        "redcap_repeat_instrument",
        "redcap_repeat_instance",
        "network",
        "timepoint",
    }
)
OUTCOME_KEY_COLUMNS = ("ID", "data_type", "redcap_event_name", "variable")
SNAPSHOT_RECORD_KEYS = frozenset(
    {"timepoint", "filename", "size", "mtime_ns", "sha256"}
)


@dataclass(frozen=True)
class ValueReplacement:
    """One unambiguous Analyze Flags value pair mapped to a combined cell."""

    network: str
    subject_id: str
    timepoint: str
    variable: str
    before_value: str
    current_value: str
    source_entry_count: int = 1

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.timepoint, self.subject_id, self.variable)

    @property
    def is_noop(self) -> bool:
        return self.before_value == self.current_value

@dataclass(frozen=True)
class UnmappedValuePair:
    """One safe pair that cannot map to a single canonical timepoint slice."""

    network: str
    subject_id: str
    timepoint: str
    variable: str
    before_value: str
    current_value: str
    before_value_source: str
    latest_seen: str
    status: str
    general_flag: str
    message_variable: str
    canonical_template: str


@dataclass(frozen=True)
class ValueReplacementPlan:
    """Applicable replacements plus safe, explicitly unmappable audit rows."""

    replacements: tuple[ValueReplacement, ...]
    unmapped: tuple[UnmappedValuePair, ...]


@dataclass(frozen=True)
class ScenarioLayout:
    """Filesystem layout for one immutable before/current experiment."""

    root: Path
    original_combined: Path
    current_combined: Path
    original_outcomes: Path
    current_outcomes: Path
    manifest: Path
    normalized_replacements: Path
    unmapped_replacements: Path
    frozen_before_after: Path
    affected_subject_roster: Path

    @classmethod
    def for_root(cls, root: str | Path) -> "ScenarioLayout":
        resolved = Path(root).expanduser().resolve()
        return cls(
            root=resolved,
            original_combined=resolved / "inputs" / ORIGINAL_SCENARIO,
            current_combined=resolved / "inputs" / CURRENT_SCENARIO,
            original_outcomes=resolved / "outcomes" / ORIGINAL_SCENARIO,
            current_outcomes=resolved / "outcomes" / CURRENT_SCENARIO,
            manifest=resolved / "experiment_manifest.json",
            normalized_replacements=resolved / "applied_value_pairs.csv",
            unmapped_replacements=resolved / "unmapped_value_pairs.csv",
            frozen_before_after=(
                resolved / "source_artifacts" / "before_after_values.csv"
            ),
            affected_subject_roster=(
                resolved / "source_artifacts" / "affected_subject_roster.csv"
            ),
        )


def _raise_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_field_size_limit()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _program_identity() -> dict[str, str]:
    """Return the exact program/runtime identity required for safe resume."""
    return {
        "module_sha256": _sha256_file(Path(__file__).resolve()),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }


def _unique_temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary(path)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as target:
            json.dump(payload, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)



def _warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def _validated_reserve_bytes(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise OutcomeInputError("Outcome disk reserve must be a nonnegative integer")
    return value


def _require_disk_reserve(path: Path, required_bytes: int) -> None:
    required = _validated_reserve_bytes(required_bytes)
    try:
        available = shutil.disk_usage(path).free
    except OSError as exc:
        raise OutcomeInputError("Could not inspect experiment disk space") from exc
    if available < required:
        raise OutcomeInputError(
            "Insufficient free disk space for the configured protected "
            "outcome-output reserve"
        )


@contextmanager
def experiment_root_lock(scenario_root: str | Path) -> Iterator[None]:
    """Serialize every writer for one before/current experiment root."""

    root = Path(scenario_root).expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{root.name}.outcome-experiment.lock"
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
                "Another before/current outcome experiment is already using "
                "this experiment directory"
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

def _strict_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read a modest audit CSV without losing blank/string values."""

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            try:
                header = next(reader)
            except StopIteration:
                raise OutcomeInputError(
                    "Analyze Flags before/after artifact is empty"
                ) from None
            duplicates = [
                name for name, count in Counter(header).items() if count > 1
            ]
            if duplicates:
                raise OutcomeInputError(
                    "Analyze Flags before/after artifact has duplicate headers"
                )
            expected = len(header)
            rows: list[list[str]] = []
            for row in reader:
                if len(row) != expected:
                    raise OutcomeInputError(
                        "Analyze Flags before/after artifact has a malformed "
                        f"logical row at physical line {reader.line_num}"
                    )
                rows.append(row)
            return header, rows
    except FileNotFoundError:
        raise OutcomeInputError(
            "Analyze Flags before/after artifact was not found"
        ) from None
    except csv.Error as exc:
        raise OutcomeInputError(
            "Analyze Flags before/after artifact is not valid CSV"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise OutcomeInputError(
            "Analyze Flags before/after artifact could not be read"
        ) from exc


def _validate_pair_filename(path: Path, network: str) -> None:
    expected_name = f"resolved_before_after_values_{network}.csv"
    if path.name.casefold() != expected_name.casefold():
        raise OutcomeInputError(
            "Analyze Flags artifact filename does not match the selected network; "
            f"expected {expected_name}"
        )


def _parse_value_replacement_plan(
    path: Path, normalized_network: str
) -> ValueReplacementPlan:
    """Parse a frozen paired-value artifact into applicable and audit rows.

    Duplicate flag episodes may target the same cell.  They collapse only
    when both values agree exactly; conflicting histories fail closed because
    one all-original scenario cannot represent two different prior values.
    """

    header, rows = _strict_csv_rows(path)
    missing_headers = [name for name in REQUIRED_PAIR_COLUMNS if name not in header]
    unexpected_headers = [name for name in header if name not in REQUIRED_PAIR_COLUMNS]
    if missing_headers or unexpected_headers:
        raise OutcomeInputError(
            "Analyze Flags before/after artifact does not have the exact "
            "paired-value schema"
        )

    positions = {name: index for index, name in enumerate(header)}
    parsed: list[ValueReplacement] = []
    unmapped: list[UnmappedValuePair] = []
    issue_counts: Counter[str] = Counter()
    for row in rows:
        subject_id = row[positions["Subject"]].strip()
        variable = row[positions["variable"]].strip()
        available = row[positions["before_value_available"]].strip().casefold()
        status = row[positions["status"]].strip().casefold()
        raw_timepoint = row[positions["Timepoint"]]
        target_is_safe = bool(
            subject_id
            and variable
            and variable.casefold() not in IMMUTABLE_IDENTITY_COLUMNS
            and available in {"true", "1", "yes"}
            and status in PAIR_STATUS_ALLOWLIST
        )

        if not subject_id:
            issue_counts["blank subject"] += 1
        if not variable:
            issue_counts["blank variable"] += 1
        elif variable.casefold() in IMMUTABLE_IDENTITY_COLUMNS:
            issue_counts["identity-column target"] += 1
        if available not in {"true", "1", "yes"}:
            issue_counts["unavailable original value"] += 1
        if status not in PAIR_STATUS_ALLOWLIST:
            issue_counts["unsupported status"] += 1

        if raw_timepoint == "multiple_timepoints" and target_is_safe:
            unmapped.append(
                UnmappedValuePair(
                    network=normalized_network,
                    subject_id=subject_id,
                    timepoint=raw_timepoint,
                    variable=variable,
                    before_value=row[positions["before_value"]],
                    current_value=row[positions["current_value"]],
                    before_value_source=row[positions["before_value_source"]],
                    latest_seen=row[positions["Latest_seen"]],
                    status=row[positions["status"]],
                    general_flag=row[positions["General_Flag"]],
                    message_variable=row[positions["message_variable"]],
                    canonical_template=row[positions["canonical_template"]],
                )
            )
            continue

        try:
            timepoint = normalize_timepoint(raw_timepoint)
        except OutcomeInputError:
            issue_counts["unsupported timepoint"] += 1
            timepoint = ""

        if target_is_safe and timepoint:
            parsed.append(
                ValueReplacement(
                    network=normalized_network,
                    subject_id=subject_id,
                    timepoint=timepoint,
                    variable=variable,
                    before_value=row[positions["before_value"]],
                    current_value=row[positions["current_value"]],
                )
            )

    if issue_counts:
        detail = ", ".join(
            f"{count} {label} row(s)" for label, count in sorted(issue_counts.items())
        )
        raise OutcomeInputError(
            "Analyze Flags before/after artifact contains unsafe rows: " + detail
        )

    replacements_by_key: dict[tuple[str, str, str], ValueReplacement] = {}
    conflicts = 0
    for item in parsed:
        prior = replacements_by_key.get(item.key)
        if prior is None:
            replacements_by_key[item.key] = item
        elif (
            prior.before_value != item.before_value
            or prior.current_value != item.current_value
        ):
            conflicts += 1
        else:
            replacements_by_key[item.key] = replace(
                prior, source_entry_count=prior.source_entry_count + 1
            )
    if conflicts:
        raise OutcomeInputError(
            "Analyze Flags before/after artifact has "
            f"{conflicts} conflicting duplicate target row(s)"
        )
    if not replacements_by_key and not unmapped:
        raise OutcomeInputError(
            "Analyze Flags before/after artifact has no applicable value pairs"
        )

    timepoint_order = {value: index for index, value in enumerate(OUTCOME_TIMEPOINTS)}
    replacements = tuple(
        sorted(
            replacements_by_key.values(),
            key=lambda item: (
                timepoint_order[item.timepoint],
                item.subject_id,
                item.variable,
            ),
        )
    )
    return ValueReplacementPlan(replacements=replacements, unmapped=tuple(unmapped))


def load_value_replacement_plan(
    before_after_csv: str | Path, network: str
) -> ValueReplacementPlan:
    """Load applicable replacements and explicitly unmappable audit rows."""

    normalized_network = normalize_network(network)
    path = Path(before_after_csv).expanduser().resolve()
    _validate_pair_filename(path, normalized_network)
    return _parse_value_replacement_plan(path, normalized_network)


def load_value_replacements(
    before_after_csv: str | Path, network: str
) -> tuple[ValueReplacement, ...]:
    """Compatibility wrapper returning only applicable canonical replacements."""

    return load_value_replacement_plan(before_after_csv, network).replacements


def _validate_replacement_targets(
    replacements: Sequence[ValueReplacement], network: str
) -> None:
    issue_counts: Counter[str] = Counter()
    replacement_keys: set[tuple[str, str, str]] = set()
    for item in replacements:
        if item.network != network:
            issue_counts["wrong network"] += 1
        if item.key in replacement_keys:
            issue_counts["duplicate replacement key"] += 1
        replacement_keys.add(item.key)
        if not str(item.subject_id).strip():
            issue_counts["blank subject"] += 1
        variable = str(item.variable).strip()
        if not variable:
            issue_counts["blank variable"] += 1
        elif variable.casefold() in IMMUTABLE_IDENTITY_COLUMNS:
            issue_counts["identity-column target"] += 1
        if item.timepoint not in OUTCOME_TIMEPOINTS:
            issue_counts["unsupported timepoint"] += 1
    if issue_counts:
        detail = ", ".join(
            f"{count} {label}(s)" for label, count in sorted(issue_counts.items())
        )
        raise OutcomeInputError("Value replacements contain unsafe targets: " + detail)


def _combined_snapshot(root: Path, network: str) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    missing = 0
    for timepoint in OUTCOME_TIMEPOINTS:
        path = combined_csv_file(root, network, timepoint)
        try:
            stat = path.stat()
        except FileNotFoundError:
            missing += 1
            continue
        if not path.is_file():
            missing += 1
            continue
        records.append(
            {
                "timepoint": timepoint,
                "filename": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256_file(path),
            }
        )
    if missing:
        raise OutcomeInputError(
            f"Combined input directory is missing {missing} canonical file(s)"
        )
    return tuple(records)


def _validated_snapshot_records(
    value: object, network: str, *, label: str
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != len(OUTCOME_TIMEPOINTS):
        raise OutcomeInputError(f"Scenario manifest has invalid {label} metadata")
    records: list[dict[str, object]] = []
    for expected_timepoint, record in zip(OUTCOME_TIMEPOINTS, value):
        if not isinstance(record, dict) or set(record) != SNAPSHOT_RECORD_KEYS:
            raise OutcomeInputError(f"Scenario manifest has invalid {label} metadata")
        size = record.get("size")
        mtime_ns = record.get("mtime_ns")
        sha256 = record.get("sha256")
        expected_filename = combined_csv_file(
            Path("."), network, expected_timepoint
        ).name
        if (
            record.get("timepoint") != expected_timepoint
            or record.get("filename") != expected_filename
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(mtime_ns, int)
            or isinstance(mtime_ns, bool)
            or mtime_ns < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise OutcomeInputError(f"Scenario manifest has invalid {label} metadata")
        records.append(dict(record))
    return tuple(records)


def _snapshots_equal(
    first: Sequence[Mapping[str, object]], second: Sequence[Mapping[str, object]]
) -> bool:
    return tuple(dict(record) for record in first) == tuple(
        dict(record) for record in second
    )


def _snapshot_contents_equal(
    first: Sequence[Mapping[str, object]], second: Sequence[Mapping[str, object]]
) -> bool:
    keys = ("timepoint", "filename", "size", "sha256")
    return tuple(tuple(record[key] for key in keys) for record in first) == tuple(
        tuple(record[key] for key in keys) for record in second
    )


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _write_original_slice(
    source_path: Path,
    target_path: Path,
    replacements: Sequence[ValueReplacement],
) -> None:
    """Stream one source slice into its original-value counterfactual."""

    _write_overlay_slice(
        source_path,
        target_path,
        replacements,
        value_attr="before_value",
        verify_current=True,
    )


def _write_overlay_slice(
    source_path: Path,
    target_path: Path,
    replacements: Sequence[ValueReplacement],
    *,
    value_attr: str,
    verify_current: bool,
) -> None:
    """Stream one source slice with ``value_attr`` overlaid per target cell.

    ``verify_current`` preserves the classic contract that the live cell must
    equal the pair's ``current_value`` byte-for-byte; the pair-vs-pair mode
    (``--after-from-pairs``) overlays both sides onto the live base and
    deliberately skips that check.
    """

    by_subject: dict[str, list[ValueReplacement]] = defaultdict(list)
    for item in replacements:
        by_subject[item.subject_id].append(item)

    try:
        with source_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            try:
                header = next(reader)
            except StopIteration:
                raise OutcomeInputError("A canonical combined CSV is empty") from None
            if len(header) != len(set(header)):
                raise OutcomeInputError(
                    "A canonical combined CSV has duplicate headers"
                )
            if "subjectid" not in header:
                raise OutcomeInputError(
                    "A canonical combined CSV has no subjectid column"
                )
            header_positions = {name: index for index, name in enumerate(header)}
            missing_variables = {
                item.variable
                for item in replacements
                if item.variable not in header_positions
            }
            if missing_variables:
                raise OutcomeInputError(
                    "A canonical combined CSV is missing "
                    f"{len(missing_variables)} targeted variable column(s)"
                )

            expected_fields = len(header)
            subject_position = header_positions["subjectid"]
            seen_targets: Counter[tuple[str, str, str]] = Counter()
            source_mismatches = 0
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("x", encoding="utf-8", newline="") as target:
                writer = csv.writer(target, lineterminator="\n")
                writer.writerow(header)
                for row in reader:
                    if len(row) != expected_fields:
                        raise OutcomeInputError(
                            "A canonical combined CSV has a malformed logical "
                            f"row at physical line {reader.line_num}"
                        )
                    subject_id = row[subject_position].strip()
                    subject_changes = by_subject.get(subject_id, ())
                    if subject_changes:
                        row = list(row)
                        for item in subject_changes:
                            value_position = header_positions[item.variable]
                            if (
                                verify_current
                                and row[value_position] != item.current_value
                            ):
                                source_mismatches += 1
                            row[value_position] = getattr(item, value_attr)
                            seen_targets[item.key] += 1
                    writer.writerow(row)
                target.flush()
                os.fsync(target.fileno())

            wrong_match_count = sum(
                seen_targets[item.key] != 1 for item in replacements
            )
            if source_mismatches or wrong_match_count:
                target_path.unlink(missing_ok=True)
                details = []
                if source_mismatches:
                    details.append(
                        f"{source_mismatches} stale current-value mismatch(es)"
                    )
                if wrong_match_count:
                    details.append(
                        f"{wrong_match_count} missing or duplicate subject target(s)"
                    )
                raise OutcomeInputError(
                    "Combined values do not match the Analyze Flags snapshot: "
                    + ", ".join(details)
                )
    except csv.Error as exc:
        target_path.unlink(missing_ok=True)
        raise OutcomeInputError("A canonical combined CSV is not valid CSV") from exc
    except (OSError, UnicodeError) as exc:
        target_path.unlink(missing_ok=True)
        raise OutcomeInputError(
            "A canonical combined CSV could not be staged"
        ) from exc


def _write_normalized_replacements(
    path: Path, replacements: Sequence[ValueReplacement]
) -> None:
    with path.open("x", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(
            [
                "network",
                "Subject",
                "Timepoint",
                "variable",
                "before_value",
                "current_value",
                "source_entry_count",
                "is_noop",
            ]
        )
        for item in replacements:
            writer.writerow(
                [
                    item.network,
                    item.subject_id,
                    item.timepoint,
                    item.variable,
                    item.before_value,
                    item.current_value,
                    item.source_entry_count,
                    str(item.is_noop),
                ]
            )
        target.flush()
        os.fsync(target.fileno())


def _write_unmapped_replacements(
    path: Path, replacements: Sequence[UnmappedValuePair]
) -> None:
    with path.open("x", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(
            [
                "network",
                "Subject",
                "Timepoint",
                "variable",
                "before_value",
                "current_value",
                "before_value_source",
                "Latest_seen",
                "status",
                "General_Flag",
                "message_variable",
                "canonical_template",
                "skip_reason",
            ]
        )
        for item in replacements:
            writer.writerow(
                [
                    item.network,
                    item.subject_id,
                    item.timepoint,
                    item.variable,
                    item.before_value,
                    item.current_value,
                    item.before_value_source,
                    item.latest_seen,
                    item.status,
                    item.general_flag,
                    item.message_variable,
                    item.canonical_template,
                    "multiple_timepoints_not_mappable_to_canonical_slice",
                ]
            )
        target.flush()
        os.fsync(target.fileno())


def _affected_subject_ids(
    replacements: Sequence[ValueReplacement],
) -> tuple[str, ...]:
    subject_ids = tuple(sorted({item.subject_id for item in replacements}))
    if not subject_ids:
        raise OutcomeInputError(
            "No mappable affected subjects remain for outcome comparison"
        )
    return subject_ids


def _write_affected_subject_roster(
    path: Path, subject_ids: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(["Subject"])
        for subject_id in subject_ids:
            writer.writerow([subject_id])
        target.flush()
        os.fsync(target.fileno())


def _freeze_pair_artifact(live_path: Path, staged_path: Path) -> str:
    """Freeze one stable paired artifact and return its verified digest."""

    staged_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        live_before = _sha256_file(live_path)
        shutil.copy2(live_path, staged_path)
        staged_hash = _sha256_file(staged_path)
        live_after = _sha256_file(live_path)
    except (OSError, UnicodeError) as exc:
        staged_path.unlink(missing_ok=True)
        raise OutcomeInputError(
            "Analyze Flags before/after artifact could not be frozen"
        ) from exc
    if live_before != staged_hash or staged_hash != live_after:
        staged_path.unlink(missing_ok=True)
        raise OutcomeInputError(
            "Analyze Flags before/after artifact changed while being staged"
        )
    return staged_hash


def _verify_live_pair_hash(live_path: Path, expected_hash: str) -> None:
    try:
        actual_hash = _sha256_file(live_path)
    except OSError as exc:
        raise OutcomeInputError(
            "Analyze Flags before/after artifact could not be verified"
        ) from exc
    if actual_hash != expected_hash:
        raise OutcomeInputError(
            "Analyze Flags before/after artifact changed while scenarios "
            "were being staged"
        )


def validate_scenario_inputs(layout: ScenarioLayout, network: str, subject_ids: Sequence[str]) -> int:
    """Validate full staged files while materializing only affected subjects."""

    original_store = load_combined_subjects(layout.original_combined, network, subject_ids=subject_ids)
    original_subjects = set(original_store)
    del original_store
    gc.collect()

    current_store = load_combined_subjects(layout.current_combined, network, subject_ids=subject_ids)
    current_subjects = set(current_store)
    del current_store
    gc.collect()
    if original_subjects != set(subject_ids) or current_subjects != set(subject_ids):
        raise OutcomeInputError(
            "An affected-subject roster is missing from a staged scenario"
        )
    return len(subject_ids)


def build_value_scenarios(
    source_combined_dir: str | Path,
    before_after_csv: str | Path,
    scenario_root: str | Path,
    network: str,
    replacements: Optional[Sequence[ValueReplacement]] = None,
    outcome_disk_reserve_bytes: int = DEFAULT_OUTCOME_DISK_RESERVE_BYTES,
    after_from_pairs: bool = False,
) -> ScenarioLayout:
    """Atomically publish immutable original/current combined input trees.

    With ``after_from_pairs`` the "current" scenario is also an overlay: each
    pair's ``current_value`` is written onto the live base, and the live-cell
    equality check is skipped, so the comparison becomes before-value versus
    after-value rather than before-value versus live data.
    """

    normalized_network = normalize_network(network)
    source_root = Path(source_combined_dir).expanduser().resolve()
    pair_path = Path(before_after_csv).expanduser().resolve()
    destination = Path(scenario_root).expanduser().resolve()
    _validate_pair_filename(pair_path, normalized_network)
    if destination.exists():
        raise OutcomeInputError(
            "Scenario directory already exists; use --resume to reuse it"
        )
    if _paths_overlap(source_root, destination):
        raise OutcomeInputError(
            "Scenario directory and source combined directory must not overlap"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    temporary_layout = ScenarioLayout.for_root(temporary_root)
    primary_error: Optional[BaseException] = None
    try:
        outcome_disk_reserve_bytes = _validated_reserve_bytes(outcome_disk_reserve_bytes)
        staged_pair_hash = _freeze_pair_artifact(
            pair_path, temporary_layout.frozen_before_after
        )
        plan = _parse_value_replacement_plan(
            temporary_layout.frozen_before_after, normalized_network
        )
        parsed_replacements = plan.replacements
        _validate_replacement_targets(parsed_replacements, normalized_network)
        if replacements is not None:
            supplied_replacements = tuple(replacements)
            _validate_replacement_targets(
                supplied_replacements, normalized_network
            )
            if supplied_replacements != parsed_replacements:
                raise OutcomeInputError(
                    "Caller-supplied value replacements do not exactly match "
                    "the frozen Analyze Flags artifact"
                )
        replacements = parsed_replacements

        source_before = _combined_snapshot(source_root, normalized_network)
        source_bytes = sum(int(record["size"]) for record in source_before)
        required_free = (
            source_bytes * 2
            + STAGING_OVERHEAD_BYTES
            + outcome_disk_reserve_bytes
        )
        if shutil.disk_usage(destination.parent).free < required_free:
            raise OutcomeInputError(
                "Insufficient free disk space for staged protected inputs and "
                "the configured outcome-output reserve"
            )

        temporary_layout.original_combined.mkdir(parents=True)
        temporary_layout.current_combined.mkdir(parents=True)
        _write_normalized_replacements(
            temporary_layout.normalized_replacements, replacements
        )
        _write_unmapped_replacements(
            temporary_layout.unmapped_replacements, plan.unmapped
        )
        affected_subject_ids = _affected_subject_ids(replacements)
        _write_affected_subject_roster(
            temporary_layout.affected_subject_roster, affected_subject_ids
        )

        changes_by_timepoint: dict[str, list[ValueReplacement]] = defaultdict(list)
        for item in replacements:
            changes_by_timepoint[item.timepoint].append(item)

        for timepoint in OUTCOME_TIMEPOINTS:
            source_path = combined_csv_file(
                source_root, normalized_network, timepoint
            )
            original_path = combined_csv_file(
                temporary_layout.original_combined,
                normalized_network,
                timepoint,
            )
            current_path = combined_csv_file(
                temporary_layout.current_combined,
                normalized_network,
                timepoint,
            )
            timepoint_changes = changes_by_timepoint.get(timepoint, ())
            if after_from_pairs and timepoint_changes:
                _write_overlay_slice(
                    source_path,
                    current_path,
                    timepoint_changes,
                    value_attr="current_value",
                    verify_current=False,
                )
            else:
                shutil.copy2(source_path, current_path)
            if timepoint_changes:
                _write_overlay_slice(
                    source_path,
                    original_path,
                    timepoint_changes,
                    value_attr="before_value",
                    verify_current=not after_from_pairs,
                )
            else:
                shutil.copy2(source_path, original_path)

        source_after = _combined_snapshot(source_root, normalized_network)
        if not _snapshots_equal(source_before, source_after):
            raise OutcomeInputError(
                "Source combined CSVs changed while scenarios were being staged"
            )

        roster_count = validate_scenario_inputs(
            temporary_layout, normalized_network, affected_subject_ids
        )
        original_snapshot = _combined_snapshot(
            temporary_layout.original_combined, normalized_network
        )
        current_snapshot = _combined_snapshot(
            temporary_layout.current_combined, normalized_network
        )
        if not after_from_pairs and not _snapshot_contents_equal(
            source_before, current_snapshot
        ):
            raise OutcomeInputError(
                "Staged current-value inputs do not match the source snapshot"
            )
        by_timepoint = Counter(item.timepoint for item in replacements)
        counts: dict[str, object] = {
            "unique_value_pairs": len(replacements),
            "source_flag_entries": sum(
                item.source_entry_count for item in replacements
            ),
            "no_op_pairs": sum(item.is_noop for item in replacements),
            "affected_subjects": len(affected_subject_ids),
            "outcome_roster": roster_count,
            "by_timepoint": dict(sorted(by_timepoint.items())),
        }
        if plan.unmapped:
            counts["skipped_multiple_timepoint_pairs"] = len(plan.unmapped)
        manifest_payload: dict[str, object] = {
            "manifest_format": MANIFEST_FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "program": _program_identity(),
            "network": normalized_network,
            "source_combined_dir": str(source_root),
            "before_after_csv": str(pair_path),
            "before_after_sha256": staged_pair_hash,
            "frozen_before_after_file": str(
                temporary_layout.frozen_before_after.relative_to(temporary_root)
            ),
            "frozen_before_after_sha256": staged_pair_hash,
            "affected_subject_roster_file": str(
                temporary_layout.affected_subject_roster.relative_to(temporary_root)
            ),
            "affected_subject_roster_sha256": _sha256_file(
                temporary_layout.affected_subject_roster
            ),
            "roster_mode": "affected_only",
            "applied_value_pairs_sha256": _sha256_file(
                temporary_layout.normalized_replacements
            ),
            "unmapped_value_pairs_sha256": _sha256_file(
                temporary_layout.unmapped_replacements
            ),
            "source_files": list(source_before),
            "scenario_files": {
                ORIGINAL_SCENARIO: list(original_snapshot),
                CURRENT_SCENARIO: list(current_snapshot),
            },
            "counts": counts,
            "after_mode": (
                "pair_current_values" if after_from_pairs else "live_combined"
            ),
            "semantics": (
                (
                    "Counterfactual overlay of the pairs' before values versus "
                    "an overlay of the pairs' after values on the same live "
                    "base; the live-cell equality check is skipped. Outcome "
                    "calculations and comparisons are limited to the frozen "
                    "affected-subject roster."
                )
                if after_from_pairs
                else (
                    "Counterfactual overlay of conservatively recovered original "
                    "values versus latest combined-CSV values; not complete "
                    "resolution-time snapshots. Outcome calculations and comparisons "
                    "are limited to the frozen affected-subject roster."
                )
            ),
            "state": {
                ORIGINAL_SCENARIO: "prepared",
                CURRENT_SCENARIO: "prepared",
                "comparison": "pending",
            },
        }
        _atomic_write_json(temporary_layout.manifest, manifest_payload)
        _verify_live_pair_hash(pair_path, staged_pair_hash)
        if destination.exists():
            raise OutcomeInputError(
                "Scenario directory appeared while scenarios were being staged"
            )
        os.replace(temporary_root, destination)
        return ScenarioLayout.for_root(destination)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if temporary_root.exists():
            try:
                shutil.rmtree(temporary_root)
            except OSError as cleanup_exc:
                if primary_error is None:
                    raise OutcomeInputError(
                        "Temporary protected staging data could not be removed"
                    ) from cleanup_exc
                note = (
                    "Temporary protected staging cleanup also failed; a hidden "
                    "temporary directory may remain beside the experiment."
                )
                try:
                    primary_error.add_note(note)
                except AttributeError:
                    pass
                _warn(note)


def load_existing_scenarios(
    source_combined_dir: Optional[str | Path],
    before_after_csv: Optional[str | Path],
    scenario_root: str | Path,
    network: str,
) -> ScenarioLayout:
    """Validate staged artifacts as authority; live paths are advisory."""

    normalized_network = normalize_network(network)
    layout = ScenarioLayout.for_root(scenario_root)
    try:
        payload = json.loads(layout.manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise OutcomeInputError("Scenario manifest was not found") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise OutcomeInputError("Scenario manifest could not be read") from exc
    if not isinstance(payload, dict):
        raise OutcomeInputError("Scenario manifest could not be read")

    program = payload.get("program")
    valid_program = bool(
        isinstance(program, dict)
        and set(program) == {
            "module_sha256",
            "python_implementation",
            "python_version",
        }
        and isinstance(program.get("module_sha256"), str)
        and len(program["module_sha256"]) == 64
        and all(
            character in "0123456789abcdef"
            for character in program["module_sha256"]
        )
        and isinstance(program.get("python_implementation"), str)
        and isinstance(program.get("python_version"), str)
    )
    if (
        payload.get("manifest_format") != MANIFEST_FORMAT_VERSION
        or payload.get("network") != normalized_network
        or not valid_program
        or payload.get("roster_mode") != "affected_only"
        or payload.get("affected_subject_roster_file")
        != str(layout.affected_subject_roster.relative_to(layout.root))
        or payload.get("frozen_before_after_file")
        != str(layout.frozen_before_after.relative_to(layout.root))
    ):
        raise OutcomeInputError(
            "Existing scenario manifest does not match this request"
        )
    if program != _program_identity():
        _warn(
            "The orchestration program changed after staging; validated staged "
            "artifacts remain the resume authority."
        )

    try:
        frozen_pair_hash = _sha256_file(layout.frozen_before_after)
        applied_audit_hash = _sha256_file(layout.normalized_replacements)
        unmapped_audit_hash = _sha256_file(layout.unmapped_replacements)
        affected_roster_hash = _sha256_file(layout.affected_subject_roster)
    except OSError as exc:
        raise OutcomeInputError(
            "A frozen scenario artifact could not be verified"
        ) from exc
    if (
        payload.get("frozen_before_after_sha256") != frozen_pair_hash
        or payload.get("before_after_sha256") != frozen_pair_hash
        or payload.get("affected_subject_roster_sha256") != affected_roster_hash
        or payload.get("applied_value_pairs_sha256") != applied_audit_hash
        or payload.get("unmapped_value_pairs_sha256") != unmapped_audit_hash
    ):
        raise OutcomeInputError(
            "A frozen scenario artifact changed after scenario creation"
        )

    frozen_plan = _parse_value_replacement_plan(
        layout.frozen_before_after, normalized_network
    )
    expected_subject_ids = _affected_subject_ids(frozen_plan.replacements)
    stored_subject_ids = tuple(load_subject_id_file(layout.affected_subject_roster))
    counts = payload.get("counts")
    if (
        stored_subject_ids != expected_subject_ids
        or not isinstance(counts, dict)
        or counts.get("affected_subjects") != len(stored_subject_ids)
        or counts.get("outcome_roster") != len(stored_subject_ids)
    ):
        raise OutcomeInputError("The frozen affected-subject roster is inconsistent")

    expected_source = _validated_snapshot_records(
        payload.get("source_files"), normalized_network, label="source input"
    )
    scenario_files = payload.get("scenario_files")
    if not isinstance(scenario_files, dict) or set(scenario_files) != {
        ORIGINAL_SCENARIO,
        CURRENT_SCENARIO,
    }:
        raise OutcomeInputError("Scenario manifest has invalid input metadata")
    for label, directory in (
        (ORIGINAL_SCENARIO, layout.original_combined),
        (CURRENT_SCENARIO, layout.current_combined),
    ):
        expected = _validated_snapshot_records(
            scenario_files.get(label), normalized_network, label=f"{label} input"
        )
        actual = _combined_snapshot(directory, normalized_network)
        if not _snapshots_equal(expected, actual):
            raise OutcomeInputError(
                f"Staged {label} combined inputs changed after creation"
            )

    if source_combined_dir is not None:
        try:
            live_source = _combined_snapshot(
                Path(source_combined_dir).expanduser().resolve(), normalized_network
            )
        except (OutcomeInputError, OSError):
            _warn(
                "The optional live combined source could not be verified; "
                "validated staged inputs will be used."
            )
        else:
            if not _snapshot_contents_equal(expected_source, live_source):
                _warn(
                    "The optional live combined source has drifted; validated "
                    "staged inputs will be used."
                )
    if before_after_csv is not None:
        try:
            live_pair_hash = _sha256_file(
                Path(before_after_csv).expanduser().resolve()
            )
        except OSError:
            _warn(
                "The optional live before/after artifact could not be verified; "
                "the validated frozen artifact will be used."
            )
        else:
            if live_pair_hash != frozen_pair_hash:
                _warn(
                    "The optional live before/after artifact has drifted; the "
                    "validated frozen artifact will be used."
                )
    return layout


def _update_experiment_state(
    layout: ScenarioLayout, label: str, status: str
) -> None:
    try:
        payload = json.loads(layout.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutcomeInputError("Scenario manifest could not be updated") from exc
    state = payload.setdefault("state", {})
    if not isinstance(state, dict):
        raise OutcomeInputError("Scenario manifest has invalid state metadata")
    state[label] = status
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(layout.manifest, payload)


def _checkpoint_manifest_exists(output_dir: Path) -> bool:
    checkpoint_root = output_dir / ".outcome_checkpoints"
    return checkpoint_root.is_dir() and any(
        checkpoint_root.rglob("manifest.json")
    )


def _default_command_runner(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def run_value_scenarios(
    layout: ScenarioLayout,
    network: str,
    *,
    workers: Optional[int] = None,
    resume: bool = False,
    restart_outcomes: bool = False,
    outcome_disk_reserve_bytes: int = DEFAULT_OUTCOME_DISK_RESERVE_BYTES,
    command_runner: Callable[[Sequence[str]], None] = _default_command_runner,
) -> None:
    """Run the existing full outcome engine sequentially for both scenarios."""

    normalized_network = normalize_network(network)
    expected_subject_ids = tuple(load_subject_id_file(layout.affected_subject_roster))
    if restart_outcomes and not resume:
        raise OutcomeInputError("restart_outcomes requires an existing experiment")
    if workers is not None and workers < 1:
        raise OutcomeInputError("workers must be a positive integer")
    calculation_script = Path(__file__).with_name("outcome_calculations.py")
    for label, combined_dir, output_dir in (
        (
            ORIGINAL_SCENARIO,
            layout.original_combined,
            layout.original_outcomes,
        ),
        (
            CURRENT_SCENARIO,
            layout.current_combined,
            layout.current_outcomes,
        ),
    ):
        expected_input_files = _checkpoint_input_files_for_scenario(
            layout, label, combined_dir, normalized_network
        )
        _update_experiment_state(layout, label, "running")
        _invalidate_comparison_marker(layout, label)
        command = [
            sys.executable,
            str(calculation_script),
            normalized_network,
            "run_outcome",
            "--combined-csv-path",
            str(combined_dir),
            "--output-dir",
            str(output_dir),
            "--subject-id-file",
            str(layout.affected_subject_roster),
            "--allow-partial-roster",
        ]
        if workers is not None:
            command.extend(["--workers", str(workers)])
        if resume and not restart_outcomes and _checkpoint_manifest_exists(output_dir):
            command.append("--resume")
        try:
            _require_disk_reserve(layout.root, outcome_disk_reserve_bytes)
            command_runner(command)
            completed_run = _completed_outcome_run_record(
                output_dir,
                combined_dir,
                normalized_network,
                expected_input_files,
                expected_subject_ids,
            )
        except BaseException:
            _update_experiment_state(layout, label, "failed")
            raise
        _record_completed_outcome_run(layout, label, completed_run)


def _read_outcome_header(reader: csv.reader, label: str) -> list[str]:
    try:
        header = next(reader)
    except StopIteration:
        raise OutcomeInputError(f"{label} outcome aggregate is empty") from None
    if len(header) != len(set(header)):
        raise OutcomeInputError(
            f"{label} outcome aggregate has duplicate headers"
        )
    if tuple(header) != tuple(CHECKPOINT_COLUMNS):
        raise OutcomeInputError(
            f"{label} outcome aggregate has an unexpected schema"
        )
    return header


def _iter_outcome_subjects(
    reader: csv.reader,
    header: Sequence[str],
    label: str,
) -> Iterable[tuple[str, dict[tuple[str, ...], str]]]:
    """Yield one bounded key/value map per contiguous outcome subject."""

    positions = {name: index for index, name in enumerate(header)}
    subject_position = positions["ID"]
    value_position = positions["value"]
    key_positions = tuple(positions[name] for name in OUTCOME_KEY_COLUMNS)

    def validated_rows() -> Iterable[list[str]]:
        for row in reader:
            if len(row) != len(header):
                raise OutcomeInputError(
                    f"{label} outcome aggregate has a malformed logical row"
                )
            yield row

    seen_subjects: set[str] = set()
    for subject_id, subject_rows in groupby(
        validated_rows(), key=lambda row: row[subject_position]
    ):
        if subject_id in seen_subjects:
            raise OutcomeInputError(
                f"{label} outcome aggregate subject rows are not contiguous"
            )
        seen_subjects.add(subject_id)
        values_by_key: dict[tuple[str, ...], str] = {}
        for row in subject_rows:
            key = tuple(row[index] for index in key_positions)
            if key in values_by_key:
                raise OutcomeInputError(
                    f"{label} outcome aggregate has duplicate canonical keys "
                    "within a subject"
                )
            values_by_key[key] = row[value_position]
        yield subject_id, values_by_key


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _read_json_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutcomeInputError(f"{label} could not be read") from exc
    if not isinstance(value, dict):
        raise OutcomeInputError(f"{label} has invalid metadata")
    return value


def _checkpoint_input_files_for_scenario(
    layout: ScenarioLayout,
    label: str,
    combined_dir: Path,
    network: str,
) -> tuple[dict[str, object], ...]:
    normalized_network = normalize_network(network)
    manifest = _read_json_mapping(layout.manifest, "Scenario manifest")
    scenario_files = manifest.get("scenario_files")
    if not isinstance(scenario_files, dict) or set(scenario_files) != {
        ORIGINAL_SCENARIO,
        CURRENT_SCENARIO,
    }:
        raise OutcomeInputError("Scenario manifest has invalid input metadata")
    expected = _validated_snapshot_records(
        scenario_files.get(label),
        normalized_network,
        label=f"{label} input",
    )
    actual = _combined_snapshot(combined_dir, normalized_network)
    if not _snapshots_equal(expected, actual):
        raise OutcomeInputError(
            f"Staged {label} combined inputs changed after creation"
        )
    return tuple(
        {
            "name": record["filename"],
            "size": record["size"],
            "mtime_ns": record["mtime_ns"],
            "sha256": record["sha256"],
        }
        for record in expected
    )


def _hash_required_file(path: Path, label: str) -> str:
    try:
        if not path.is_file():
            raise OSError
        return _sha256_file(path)
    except OSError:
        raise OutcomeInputError(f"{label} is missing or unreadable") from None


def _outcome_aggregate_roster(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            header = _read_outcome_header(reader, "Affected")
            return tuple(subject_id for subject_id, _ in _iter_outcome_subjects(reader, header, "Affected"))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise OutcomeInputError("Affected outcome aggregate could not be validated") from exc


def _completed_outcome_run_record(
    output_dir: Path,
    combined_dir: Path,
    network: str,
    expected_input_files: Sequence[Mapping[str, object]],
    expected_subject_ids: Sequence[str],
) -> dict[str, object]:
    normalized_network = normalize_network(network)
    expected_checkpoint_inputs = [dict(record) for record in expected_input_files]
    expected_subjects = list(expected_subject_ids)
    expected_combined = combined_dir.expanduser().resolve()
    checkpoint_root = output_dir / ".outcome_checkpoints" / normalized_network.lower() / "run_outcome"
    candidates: list[tuple[datetime, dict[str, object]]] = []
    for manifest_path in sorted(checkpoint_root.glob("*/manifest.json")):
        try:
            manifest = _read_json_mapping(manifest_path, "Outcome checkpoint manifest")
        except OutcomeInputError:
            continue
        payload = manifest.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("input_files") != expected_checkpoint_inputs:
            continue
        combined_value = payload.get("combined_csv_path")
        if not isinstance(combined_value, str):
            continue
        try:
            payload_combined = Path(combined_value).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if payload_combined != expected_combined or payload.get("network") != normalized_network or payload.get("version") != "run_outcome":
            continue
        signature = manifest.get("signature")
        run_id = manifest.get("run_id")
        code_sha256 = payload.get("code_sha256")
        runtime = payload.get("runtime")
        subject_ids = payload.get("subject_ids")
        if (
            not _is_sha256(signature)
            or signature != _sha256_json(payload)
            or manifest_path.parent.name != signature
            or not isinstance(run_id, str) or not run_id.strip()
            or not _is_sha256(code_sha256)
            or not isinstance(runtime, dict) or not runtime
            or any(not isinstance(key, str) or not isinstance(value, str) for key, value in runtime.items())
            or payload.get("roster_mode") != "affected_only"
            or not isinstance(subject_ids, list) or not subject_ids
            or subject_ids != expected_subjects
            or any(not isinstance(subject_id, str) or not subject_id.strip() for subject_id in subject_ids)
            or len(set(subject_ids)) != len(subject_ids)
        ):
            continue
        try:
            state = _read_json_mapping(manifest_path.parent / "state.json", "Outcome checkpoint state")
        except OutcomeInputError:
            continue
        completed_at = state.get("updated_at")
        if state.get("status") != "complete" or state.get("signature") != signature or state.get("run_id") != run_id or not isinstance(completed_at, str):
            continue
        try:
            completed_time = datetime.fromisoformat(completed_at)
        except ValueError:
            continue
        if completed_time.tzinfo is None:
            continue
        identity = {
            "checkpoint_format": payload.get("checkpoint_format"),
            "network": normalized_network,
            "version": "run_outcome",
            "roster_mode": "affected_only",
            "code_sha256": code_sha256,
            "runtime": dict(runtime),
            "roster_count": len(subject_ids),
            "roster_sha256": _sha256_json(subject_ids),
        }
        candidates.append((completed_time, {
            **identity,
            "calculation_identity_sha256": _sha256_json(identity),
            "signature": signature,
            "run_id": run_id,
            "completed_at": completed_at,
        }))
    if not candidates:
        raise OutcomeInputError("Outcome subprocess returned without a matching completed checkpoint")
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and candidates[-2][0] == candidates[-1][0]:
        raise OutcomeInputError("Outcome subprocess completion checkpoint is ambiguous")
    record = dict(candidates[-1][1])
    aggregate_name = f"{normalized_network.lower()}_affected.csv"
    aggregate_path = output_dir / aggregate_name
    if _outcome_aggregate_roster(aggregate_path) != tuple(expected_subjects):
        raise OutcomeInputError("Affected outcome aggregate roster does not match")
    record["aggregate_file"] = aggregate_name
    record["aggregate_sha256"] = _hash_required_file(aggregate_path, "Outcome aggregate")
    return record


def _record_completed_outcome_run(layout: ScenarioLayout, label: str, record: Mapping[str, object]) -> None:
    payload = _read_json_mapping(layout.manifest, "Scenario manifest")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise OutcomeInputError("Scenario manifest has invalid state metadata")
    outcome_runs = payload.setdefault("outcome_runs", {})
    if not isinstance(outcome_runs, dict):
        raise OutcomeInputError("Scenario manifest has invalid outcome metadata")
    outcome_runs[label] = dict(record)
    state[label] = "complete"
    state["comparison"] = "pending"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(layout.manifest, payload)


def _invalidate_comparison_marker(layout: ScenarioLayout, label: str) -> None:
    _atomic_write_json(layout.root / "outcome_comparison_summary.json", {
        "status": "invalidated",
        "reason": "outcome_scenario_running",
        "scenario": label,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    _update_experiment_state(layout, "comparison", "pending")


def _validated_scenario_outcome_runs(layout: ScenarioLayout, network: str, aggregate_name: str) -> None:
    payload = _read_json_mapping(layout.manifest, "Scenario manifest")
    outcome_runs = payload.get("outcome_runs")
    if not isinstance(outcome_runs, dict):
        raise OutcomeInputError("Scenario outcomes do not have completed calculation identities")
    normalized_network = normalize_network(network)
    identities: list[dict[str, object]] = []
    for label, output_dir in ((ORIGINAL_SCENARIO, layout.original_outcomes), (CURRENT_SCENARIO, layout.current_outcomes)):
        raw_record = outcome_runs.get(label)
        if not isinstance(raw_record, dict):
            raise OutcomeInputError("Scenario outcomes do not have completed calculation identities")
        record = dict(raw_record)
        runtime = record.get("runtime")
        identity = {
            "checkpoint_format": record.get("checkpoint_format"),
            "network": record.get("network"),
            "version": record.get("version"),
            "roster_mode": record.get("roster_mode"),
            "code_sha256": record.get("code_sha256"),
            "runtime": runtime,
            "roster_count": record.get("roster_count"),
            "roster_sha256": record.get("roster_sha256"),
        }
        if (
            identity["network"] != normalized_network or identity["version"] != "run_outcome"
            or identity["roster_mode"] != "affected_only"
            or not _is_sha256(identity["code_sha256"])
            or not isinstance(runtime, dict) or not runtime
            or not isinstance(identity["roster_count"], int) or isinstance(identity["roster_count"], bool) or identity["roster_count"] < 1
            or not _is_sha256(identity["roster_sha256"])
            or record.get("calculation_identity_sha256") != _sha256_json(identity)
            or not _is_sha256(record.get("signature"))
            or not isinstance(record.get("run_id"), str) or not str(record["run_id"]).strip()
            or record.get("aggregate_file") != aggregate_name
            or not _is_sha256(record.get("aggregate_sha256"))
        ):
            raise OutcomeInputError("Scenario outcome calculation identity is invalid")
        if _hash_required_file(output_dir / aggregate_name, "Scenario outcome aggregate") != record["aggregate_sha256"]:
            raise OutcomeInputError("Scenario outcome aggregate changed after calculation")
        identities.append(identity)
    if identities[0] != identities[1]:
        raise OutcomeInputError("Original/current outcome calculation identities do not match")


def compare_outcome_files(
    original_path: str | Path,
    current_path: str | Path,
    output_dir: str | Path,
) -> dict[str, int]:
    """Full-outer compare outcomes one contiguous subject block at a time."""

    before_path = Path(original_path).expanduser().resolve()
    after_path = Path(current_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    changes_path = destination / "outcome_value_changes.csv"
    summary_path = destination / "outcome_change_summary.csv"
    comparison_path = destination / "outcome_comparison_summary.json"
    _atomic_write_json(comparison_path, {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    original_aggregate_sha256 = _hash_required_file(before_path, "Original outcome aggregate")
    current_aggregate_sha256 = _hash_required_file(after_path, "Current outcome aggregate")
    changes_temporary = _unique_temporary(changes_path)
    summary_temporary = _unique_temporary(summary_path)
    change_counts_by_variable: dict[
        tuple[str, str], Counter[str]
    ] = defaultdict(Counter)
    changed_subjects: set[str] = set()
    total_rows = 0
    changed_rows = 0
    added_rows = 0
    removed_rows = 0
    try:
        with (
            before_path.open("r", encoding="utf-8-sig", newline="") as before,
            after_path.open("r", encoding="utf-8-sig", newline="") as after,
            changes_temporary.open("x", encoding="utf-8", newline="") as target,
        ):
            before_reader = csv.reader(before, strict=True)
            after_reader = csv.reader(after, strict=True)
            before_header = _read_outcome_header(before_reader, "Original")
            after_header = _read_outcome_header(after_reader, "Current")
            if before_header != after_header:
                raise OutcomeInputError(
                    "Original/current outcome aggregate headers differ"
                )

            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(
                [
                    *OUTCOME_KEY_COLUMNS,
                    "change_type",
                    "original_outcome_value",
                    "current_outcome_value",
                ]
            )
            before_subjects = _iter_outcome_subjects(
                before_reader, before_header, "Original"
            )
            after_subjects = _iter_outcome_subjects(
                after_reader, after_header, "Current"
            )
            for before_subject, after_subject in zip_longest(
                before_subjects, after_subjects, fillvalue=None
            ):
                if before_subject is None or after_subject is None:
                    raise OutcomeInputError(
                        "Original/current outcome aggregates have different "
                        "ordered subject rosters"
                    )
                before_subject_id, before_values = before_subject
                after_subject_id, after_values = after_subject
                if before_subject_id != after_subject_id:
                    raise OutcomeInputError(
                        "Original/current outcome aggregates have different "
                        "ordered subject rosters"
                    )

                ordered_keys = list(before_values)
                ordered_keys.extend(
                    key for key in after_values if key not in before_values
                )
                total_rows += len(ordered_keys)
                for key in ordered_keys:
                    in_original = key in before_values
                    in_current = key in after_values
                    original_value = before_values.get(key, "")
                    current_value = after_values.get(key, "")
                    if in_original and in_current:
                        if original_value == current_value:
                            continue
                        change_type = "changed"
                        changed_rows += 1
                    elif in_current:
                        change_type = "added"
                        added_rows += 1
                    else:
                        change_type = "removed"
                        removed_rows += 1

                    changed_subjects.add(before_subject_id)
                    key_values = dict(zip(OUTCOME_KEY_COLUMNS, key))
                    change_counts_by_variable[
                        (key_values["data_type"], key_values["variable"])
                    ][change_type] += 1
                    writer.writerow(
                        [
                            *key,
                            change_type,
                            original_value,
                            current_value,
                        ]
                    )
            target.flush()
            os.fsync(target.fileno())

        with summary_temporary.open(
            "x", encoding="utf-8", newline=""
        ) as summary_target:
            writer = csv.writer(summary_target, lineterminator="\n")
            writer.writerow(
                [
                    "data_type",
                    "variable",
                    "changed_outcome_rows",
                    "added_outcome_rows",
                    "removed_outcome_rows",
                ]
            )
            for (data_type, variable), counts in sorted(
                change_counts_by_variable.items(),
                key=lambda item: (-sum(item[1].values()), item[0]),
            ):
                writer.writerow(
                    [
                        data_type,
                        variable,
                        counts["changed"],
                        counts["added"],
                        counts["removed"],
                    ]
                )
            summary_target.flush()
            os.fsync(summary_target.fileno())

        if (
            _hash_required_file(before_path, "Original outcome aggregate") != original_aggregate_sha256
            or _hash_required_file(after_path, "Current outcome aggregate") != current_aggregate_sha256
        ):
            raise OutcomeInputError("An outcome aggregate changed during comparison")
        os.replace(changes_temporary, changes_path)
        os.replace(summary_temporary, summary_path)
        result = {
            "total_outcome_rows": total_rows,
            "changed_outcome_rows": changed_rows,
            "added_outcome_rows": added_rows,
            "removed_outcome_rows": removed_rows,
            "changed_subjects": len(changed_subjects),
        }
        _atomic_write_json(comparison_path, {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "original_aggregate_sha256": original_aggregate_sha256,
            "current_aggregate_sha256": current_aggregate_sha256,
            "outcome_value_changes_sha256": _hash_required_file(changes_path, "Outcome value-change comparison"),
            "outcome_change_summary_sha256": _hash_required_file(summary_path, "Outcome change summary"),
            **result,
        })
        return result
    except csv.Error as exc:
        raise OutcomeInputError("An outcome aggregate is not valid CSV") from exc
    finally:
        changes_temporary.unlink(missing_ok=True)
        summary_temporary.unlink(missing_ok=True)


def compare_scenario_outcomes(
    layout: ScenarioLayout, network: str
) -> dict[str, int]:
    normalized_network = normalize_network(network).lower()
    aggregate_name = f"{normalized_network}_affected.csv"
    _update_experiment_state(layout, "comparison", "running")
    try:
        _validated_scenario_outcome_runs(layout, network, aggregate_name)
        result = compare_outcome_files(
            layout.original_outcomes / aggregate_name,
            layout.current_outcomes / aggregate_name,
            layout.root,
        )
    except BaseException:
        _update_experiment_state(layout, "comparison", "failed")
        raise
    _update_experiment_state(layout, "comparison", "complete")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage Analyze Flags original/current values and run the outcome "
            "calculations against both immutable snapshots."
        )
    )
    parser.add_argument("network", help="PRONET or PRESCIENT")
    parser.add_argument(
        "--combined-csv-path",
        type=Path,
        help="Live canonical combined CSVs (required only for a fresh experiment).",
    )
    parser.add_argument(
        "--before-after-csv",
        type=Path,
        help="Live Analyze Flags paired artifact (required only when fresh).",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="New directory for staged inputs, outcomes, and comparison files.",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stage and validate both input snapshots without running outcomes.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an exact staged experiment and resume interrupted outcomes.",
    )
    parser.add_argument(
        "--restart-outcomes",
        action="store_true",
        help="Reuse staged inputs but start fresh outcome checkpoint generations.",
    )
    parser.add_argument(
        "--outcome-disk-reserve-gb",
        type=float,
        default=DEFAULT_OUTCOME_DISK_RESERVE_GB,
        help="Free GiB reserved for both protected outcome runs.",
    )
    parser.add_argument(
        "--acknowledge-protected-output",
        action="store_true",
        help="Confirm the experiment directory is approved for protected data.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Run both scenarios but do not produce the streaming outcome diff.",
    )
    parser.add_argument(
        "--after-from-pairs",
        action="store_true",
        help=(
            "Overlay the pairs' current_value onto the live base for the "
            "'current' scenario (before-vs-after comparison) instead of "
            "requiring current_value to equal the live cell."
        ),
    )
    return parser


def _experiment_counts(layout: ScenarioLayout) -> Mapping[str, object]:
    try:
        payload = json.loads(layout.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutcomeInputError("Scenario manifest could not be read") from exc
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise OutcomeInputError("Scenario manifest has invalid count metadata")
    for name in ("unique_value_pairs", "skipped_multiple_timepoint_pairs"):
        value = counts.get(name, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise OutcomeInputError(
                "Scenario manifest has invalid count metadata"
            )
    return counts


def _main_locked(args: argparse.Namespace) -> int:
    network = normalize_network(args.network)
    if args.workers is not None and args.workers < 1:
        raise OutcomeInputError("--workers must be a positive integer")

    if args.resume:
        layout = load_existing_scenarios(
            args.combined_csv_path,
            args.before_after_csv,
            args.experiment_dir,
            network,
        )
        action = "Verified an existing staged experiment with"
    else:
        layout = build_value_scenarios(
            args.combined_csv_path,
            args.before_after_csv,
            args.experiment_dir,
            network,
            outcome_disk_reserve_bytes=args.outcome_disk_reserve_bytes,
            after_from_pairs=args.after_from_pairs,
        )
        action = "Staged and validated"

    counts = _experiment_counts(layout)
    replacement_count = int(counts.get("unique_value_pairs", 0))
    skipped_count = int(counts.get("skipped_multiple_timepoint_pairs", 0))
    if args.resume:
        print(
            f"{action} {replacement_count} unique value pair(s)."
        )
    else:
        print(
            f"{action} {replacement_count} unique value pair(s) without "
            "modifying the source combined CSVs."
        )
    if skipped_count:
        print(
            f"Skipped {skipped_count} explicitly unmappable "
            "multiple_timepoints pair(s); details are in unmapped_value_pairs.csv."
        )

    if args.prepare_only:
        print("Preparation complete; outcome calculations were not started.")
        return 0

    run_value_scenarios(
        layout,
        network,
        workers=args.workers,
        resume=args.resume,
        restart_outcomes=args.restart_outcomes,
        outcome_disk_reserve_bytes=args.outcome_disk_reserve_bytes,
    )
    if not args.skip_comparison:
        result = compare_scenario_outcomes(layout, network)
        print(
            "Outcome comparison complete: "
            f"{result['changed_outcome_rows']} changed row(s) across "
            f"{result['changed_subjects']} subject(s)."
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.acknowledge_protected_output:
        raise OutcomeInputError(
            "--acknowledge-protected-output is required because the experiment "
            "directory will contain protected inputs and subject-level outputs"
        )
    if not args.resume and (
        args.combined_csv_path is None or args.before_after_csv is None
    ):
        raise OutcomeInputError(
            "Fresh experiments require --combined-csv-path and --before-after-csv"
        )
    if args.restart_outcomes and not args.resume:
        raise OutcomeInputError("--restart-outcomes requires --resume")
    if args.restart_outcomes and args.prepare_only:
        raise OutcomeInputError("--restart-outcomes cannot be used with --prepare-only")
    reserve_gb = args.outcome_disk_reserve_gb
    if (
        not isinstance(reserve_gb, (int, float))
        or isinstance(reserve_gb, bool)
        or not math.isfinite(reserve_gb)
        or reserve_gb < 0
    ):
        raise OutcomeInputError("--outcome-disk-reserve-gb must be nonnegative")
    args.outcome_disk_reserve_bytes = int(reserve_gb * 1024**3)
    _warn(
        "The experiment directory will contain protected combined snapshots, "
        "raw value pairs, subject outputs, checkpoints, and outcome differences."
    )
    with experiment_root_lock(args.experiment_dir):
        return _main_locked(args)

if __name__ == "__main__":
    raise SystemExit(main())

