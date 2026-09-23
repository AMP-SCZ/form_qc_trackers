"""Merge known-blank flag history into Analyze Flags before/after pairs.

The blank-history artifact is used only as a set of
``(network, subject, timepoint, variable)`` keys.  For each key, the exact
current value is reread from the supplied canonical combined CSV.  The known
historical value is the exact empty string.  Existing paired-value artifacts
are preserved unless the blank history identifies the same target cell; in
that case the known-blank row takes precedence and the displaced protected
values are written to a per-network conflict audit in the new output tree.

This program never modifies its inputs and refuses to publish into an existing
output directory.  Operator diagnostics contain counts and categories only,
not subjects, variables, or values.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from outcome_inputs import (
    OUTCOME_TIMEPOINTS,
    OutcomeInputError,
    combined_csv_file,
    normalize_network,
    normalize_timepoint,
)


NETWORKS = ("PRONET", "PRESCIENT")
PAIR_COLUMNS = (
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
BLANK_HISTORY_COLUMNS = (
    "network",
    "subject",
    "variable",
    "timepoint",
    "current_value",
    "status",
)
CONFLICT_COLUMNS = (
    "Subject",
    "Timepoint",
    "variable",
    "existing_before_value",
    "existing_current_value",
    "selected_before_value",
    "selected_current_value",
    "selected_pair_published",
    "conflict_reason",
)
PAIR_STATUS_ALLOWLIST = frozenset({"filled", "still_blank", "missing_code"})
HISTORY_STATUS_ALLOWLIST = frozenset(
    {
        "filled",
        "still_blank",
        "missing_code",
        "subject_missing",
        "subject_ambiguous",
        "variable_missing",
        "csv_missing",
    }
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
MISSING_CODE_VALUES = frozenset(
    {
        "-3",
        "-3.0",
        "-9",
        "-9.0",
        "-99",
        "-99.0",
        "999",
        "999.0",
        "1909-09-09",
        "1903-03-03",
        "1901-01-01",
    }
)
TIMEPOINT_ORDER = {value: index for index, value in enumerate(OUTCOME_TIMEPOINTS)}


class PairMergeError(ValueError):
    """An input cannot safely produce a paired-value artifact."""


@dataclass(frozen=True, order=True)
class TargetKey:
    network: str
    timepoint: str
    subject: str
    variable: str


@dataclass(frozen=True)
class PairRow:
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(PAIR_COLUMNS):
            raise ValueError("PairRow has the wrong field count")

    def get(self, name: str) -> str:
        return self.values[PAIR_COLUMNS.index(name)]

    def as_mapping(self) -> dict[str, str]:
        return dict(zip(PAIR_COLUMNS, self.values))


@dataclass(frozen=True)
class NetworkBuildResult:
    network: str
    rows: tuple[PairRow, ...]
    conflicts: tuple[tuple[str, ...], ...]
    counts: Mapping[str, int]


def _raise_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_field_size_limit()


def _safe_normalize_network(value: str) -> str:
    try:
        return normalize_network(value)
    except OutcomeInputError as exc:
        raise PairMergeError("Blank history contains an unsupported network") from exc


def _safe_normalize_timepoint(value: str, *, source: str) -> str:
    try:
        return normalize_timepoint(value)
    except OutcomeInputError as exc:
        raise PairMergeError(f"{source} contains an unsupported timepoint") from exc


def _strict_rows(path: Path, expected_columns: Sequence[str], *, label: str) -> list[tuple[str, ...]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            try:
                header = next(reader)
            except StopIteration:
                raise PairMergeError(f"{label} is empty") from None
            if tuple(header) != tuple(expected_columns):
                raise PairMergeError(f"{label} does not have the exact required schema")
            rows: list[tuple[str, ...]] = []
            for row in reader:
                if len(row) != len(expected_columns):
                    raise PairMergeError(f"{label} contains a malformed logical row")
                rows.append(tuple(row))
            return rows
    except FileNotFoundError:
        raise PairMergeError(f"{label} was not found") from None
    except csv.Error as exc:
        raise PairMergeError(f"{label} is not valid CSV") from exc
    except UnicodeError as exc:
        raise PairMergeError(f"{label} is not valid UTF-8 CSV") from exc
    except OSError as exc:
        raise PairMergeError(f"{label} could not be read") from exc


def load_blank_history(path: str | Path) -> dict[str, set[TargetKey]]:
    """Load and validate blank-history keys without trusting stored values."""

    source_path = Path(path).expanduser().resolve()
    rows = _strict_rows(source_path, BLANK_HISTORY_COLUMNS, label="Blank history artifact")
    positions = {name: index for index, name in enumerate(BLANK_HISTORY_COLUMNS)}
    by_network: dict[str, set[TargetKey]] = {network: set() for network in NETWORKS}
    issues: Counter[str] = Counter()
    for row in rows:
        try:
            network = _safe_normalize_network(row[positions["network"]])
        except PairMergeError:
            issues["unsupported network"] += 1
            continue
        try:
            timepoint = _safe_normalize_timepoint(
                row[positions["timepoint"]], source="Blank history artifact"
            )
        except PairMergeError:
            issues["unsupported timepoint"] += 1
            continue
        subject = row[positions["subject"]].strip()
        variable = row[positions["variable"]].strip()
        status = row[positions["status"]].strip().casefold()
        if not subject:
            issues["blank subject"] += 1
        if not variable:
            issues["blank variable"] += 1
        elif variable.casefold() in IMMUTABLE_IDENTITY_COLUMNS:
            issues["identity-column target"] += 1
        if status not in HISTORY_STATUS_ALLOWLIST:
            issues["unsupported status"] += 1
        if (
            subject
            and variable
            and variable.casefold() not in IMMUTABLE_IDENTITY_COLUMNS
            and status in HISTORY_STATUS_ALLOWLIST
        ):
            by_network[network].add(TargetKey(network, timepoint, subject, variable))
    if issues:
        detail = ", ".join(
            f"{count} {category} row(s)" for category, count in sorted(issues.items())
        )
        raise PairMergeError("Blank history artifact contains unsafe rows: " + detail)
    return by_network


def _pair_key(row: PairRow, network: str) -> TargetKey:
    timepoint_raw = row.get("Timepoint")
    if timepoint_raw == "multiple_timepoints":
        timepoint = timepoint_raw
    else:
        timepoint = _safe_normalize_timepoint(
            timepoint_raw, source="Existing paired-value artifact"
        )
    return TargetKey(
        network,
        timepoint,
        row.get("Subject").strip(),
        row.get("variable").strip(),
    )


def load_existing_pairs(
    pair_dir: str | Path,
) -> dict[str, dict[TargetKey, tuple[PairRow, ...]]]:
    """Load exact paired schemas and group duplicates without resolving them."""

    root = Path(pair_dir).expanduser().resolve()
    result: dict[str, dict[TargetKey, tuple[PairRow, ...]]] = {}
    for network in NETWORKS:
        path = root / f"resolved_before_after_values_{network}.csv"
        raw_rows = _strict_rows(
            path, PAIR_COLUMNS, label=f"Existing {network} paired-value artifact"
        )
        groups: dict[TargetKey, list[PairRow]] = defaultdict(list)
        issues: Counter[str] = Counter()
        for values in raw_rows:
            row = PairRow(values)
            subject = row.get("Subject").strip()
            variable = row.get("variable").strip()
            available = row.get("before_value_available").strip().casefold()
            status = row.get("status").strip().casefold()
            if not subject:
                issues["blank subject"] += 1
            if not variable:
                issues["blank variable"] += 1
            elif variable.casefold() in IMMUTABLE_IDENTITY_COLUMNS:
                issues["identity-column target"] += 1
            if available not in {"true", "1", "yes"}:
                issues["unavailable original value"] += 1
            if status not in PAIR_STATUS_ALLOWLIST:
                issues["unsupported status"] += 1
            try:
                key = _pair_key(row, network)
            except PairMergeError:
                issues["unsupported timepoint"] += 1
                continue
            if (
                subject
                and variable
                and variable.casefold() not in IMMUTABLE_IDENTITY_COLUMNS
                and available in {"true", "1", "yes"}
                and status in PAIR_STATUS_ALLOWLIST
            ):
                groups[key].append(row)
        if issues:
            detail = ", ".join(
                f"{count} {category} row(s)"
                for category, count in sorted(issues.items())
            )
            raise PairMergeError(
                f"Existing {network} paired-value artifact contains unsafe rows: "
                + detail
            )
        result[network] = {
            key: tuple(sorted(rows, key=lambda item: item.values))
            for key, rows in groups.items()
        }
    return result


def _targeted_row_values(
    subject: str,
    row: Sequence[str],
    positions: Mapping[str, int],
    variables_by_subject: Mapping[str, set[str]],
) -> tuple[tuple[tuple[str, str], str], ...]:
    """Return only explicitly requested cells from one matched subject row."""

    return tuple(
        ((subject, variable), row[positions[variable]])
        for variable in sorted(variables_by_subject.get(subject, ()))
        if variable in positions
    )


def _read_current_values(
    combined_root: Path,
    network: str,
    keys: Iterable[TargetKey],
) -> tuple[dict[TargetKey, str], Counter[str]]:
    """Read exact target values while retaining no unrelated combined data."""

    by_timepoint: dict[str, list[TargetKey]] = defaultdict(list)
    for key in keys:
        by_timepoint[key.timepoint].append(key)
    resolved: dict[TargetKey, str] = {}
    skipped: Counter[str] = Counter()
    for timepoint in sorted(by_timepoint, key=TIMEPOINT_ORDER.__getitem__):
        target_keys = sorted(
            by_timepoint[timepoint], key=lambda item: (item.subject, item.variable)
        )
        target_subjects = {item.subject for item in target_keys}
        target_variables = {item.variable for item in target_keys}
        variables_by_subject: dict[str, set[str]] = defaultdict(set)
        for item in target_keys:
            variables_by_subject[item.subject].add(item.variable)
        path = combined_csv_file(combined_root, network, timepoint)
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.reader(source, strict=True)
                try:
                    header = next(reader)
                except StopIteration:
                    raise PairMergeError("A required combined CSV is empty") from None
                if len(header) != len(set(header)):
                    raise PairMergeError("A required combined CSV has duplicate headers")
                if "subjectid" not in header:
                    raise PairMergeError(
                        "A required combined CSV has no subjectid column"
                    )
                positions = {name: index for index, name in enumerate(header)}
                missing_variables = target_variables.difference(positions)
                skipped["variable_missing"] += sum(
                    item.variable in missing_variables for item in target_keys
                )
                subject_position = positions["subjectid"]
                values_by_target: dict[tuple[str, str], list[str]] = defaultdict(list)
                expected_fields = len(header)
                blank_subject_rows = 0
                for row in reader:
                    if len(row) != expected_fields:
                        raise PairMergeError(
                            "A required combined CSV contains a malformed logical row"
                        )
                    subject = row[subject_position].strip()
                    if not subject:
                        blank_subject_rows += 1
                        continue
                    if subject not in target_subjects:
                        continue
                    for target, value in _targeted_row_values(
                        subject, row, positions, variables_by_subject
                    ):
                        values_by_target[target].append(value)
                if blank_subject_rows:
                    raise PairMergeError(
                        "A required combined CSV contains blank subject identifiers"
                    )

                for key in target_keys:
                    if key.variable in missing_variables:
                        continue
                    values = values_by_target.get((key.subject, key.variable), [])
                    if not values:
                        skipped["subject_missing"] += 1
                    elif len(set(values)) != 1:
                        skipped["subject_ambiguous"] += 1
                    else:
                        resolved[key] = values[0]
        except FileNotFoundError:
            raise PairMergeError("A required combined CSV was not found") from None
        except csv.Error as exc:
            raise PairMergeError("A required combined CSV is not valid CSV") from exc
        except UnicodeError as exc:
            raise PairMergeError("A required combined CSV is not valid UTF-8 CSV") from exc
        except OSError as exc:
            raise PairMergeError("A required combined CSV could not be read") from exc
    return resolved, skipped


def _status_for_current_value(value: str) -> str:
    normalized = value.strip()
    if normalized == "":
        return "still_blank"
    if normalized in MISSING_CODE_VALUES:
        return "missing_code"
    return "filled"


def _blank_pair(key: TargetKey, current_value: str) -> PairRow:
    values = {
        "Subject": key.subject,
        "Timepoint": key.timepoint,
        "General_Flag": "blank_flag_history",
        "variable": key.variable,
        "message_variable": key.variable,
        "canonical_template": "Variable is blank.",
        "before_value_available": "True",
        "before_value": "",
        "before_value_source": "blank_flag_history",
        "Latest_seen": "",
        "status": _status_for_current_value(current_value),
        "current_value": current_value,
    }
    return PairRow(tuple(values[name] for name in PAIR_COLUMNS))


def _row_sort_key(row: PairRow, network: str) -> tuple[object, ...]:
    key = _pair_key(row, network)
    timepoint_order = TIMEPOINT_ORDER.get(key.timepoint, len(TIMEPOINT_ORDER))
    return (timepoint_order, key.subject, key.variable, row.values)


def build_network_result(
    network: str,
    blank_keys: Iterable[TargetKey],
    existing_groups: Mapping[TargetKey, Sequence[PairRow]],
    combined_root: str | Path,
) -> NetworkBuildResult:
    """Merge one network with explicit blank precedence and safe conflicts."""

    normalized_network = _safe_normalize_network(network)
    blank_key_set = set(blank_keys)
    current_values, skipped = _read_current_values(
        Path(combined_root).expanduser().resolve(),
        normalized_network,
        blank_key_set,
    )
    blank_rows = {
        key: _blank_pair(key, current_value)
        for key, current_value in current_values.items()
    }
    # Include every blank-history key, even when its refreshed current value
    # could not be determined. Otherwise an older nonblank-before row for the
    # same cell could incorrectly survive merely because refresh failed.
    all_keys = set(existing_groups).union(blank_key_set)
    selected: list[PairRow] = []
    conflicts: list[tuple[str, ...]] = []
    unrelated_conflicts = 0
    duplicate_rows_removed = 0
    for key in sorted(
        all_keys,
        key=lambda item: (
            TIMEPOINT_ORDER.get(item.timepoint, len(TIMEPOINT_ORDER)),
            item.subject,
            item.variable,
        ),
    ):
        existing = tuple(existing_groups.get(key, ()))
        blank = blank_rows.get(key)
        if blank is not None:
            selected.append(blank)
            for old in existing:
                if (
                    old.get("before_value") == blank.get("before_value")
                    and old.get("current_value") == blank.get("current_value")
                ):
                    duplicate_rows_removed += 1
                    continue
                reason = (
                    "blank_before_precedence"
                    if old.get("before_value") != ""
                    else "refreshed_current_precedence"
                )
                conflicts.append(
                    (
                        key.subject,
                        key.timepoint,
                        key.variable,
                        old.get("before_value"),
                        old.get("current_value"),
                        "",
                        blank.get("current_value"),
                        "True",
                        reason,
                    )
                )
            continue

        if key in blank_key_set:
            # The historical blank is known, but no exact refreshed current
            # value exists. Do not let any older same-cell pair survive: it
            # cannot pass the exact-current contract required by staging.
            for old in existing:
                conflicts.append(
                    (
                        key.subject,
                        key.timepoint,
                        key.variable,
                        old.get("before_value"),
                        old.get("current_value"),
                        "",
                        "",
                        "False",
                        "unresolved_current_target_dropped",
                    )
                )
            continue

        value_pairs = {
            (row.get("before_value"), row.get("current_value")) for row in existing
        }
        if len(value_pairs) > 1:
            unrelated_conflicts += 1
            continue
        if existing:
            selected.append(min(existing, key=lambda item: item.values))
            duplicate_rows_removed += max(0, len(existing) - 1)
    if unrelated_conflicts:
        raise PairMergeError(
            "Existing paired-value artifacts contain "
            f"{unrelated_conflicts} conflicting non-blank-history target(s)"
        )

    selected.sort(key=lambda row: _row_sort_key(row, normalized_network))
    conflicts.sort(
        key=lambda row: (
            TIMEPOINT_ORDER.get(row[1], len(TIMEPOINT_ORDER)),
            row[0],
            row[2],
            row[8],
            row,
        )
    )
    counts: dict[str, int] = {
        "blank_history_unique_keys": len(blank_key_set),
        "blank_pairs_refreshed": len(blank_rows),
        "existing_unique_targets": len(existing_groups),
        "merged_unique_targets": len(selected),
        "blank_precedence_conflicts": len(conflicts),
        "overlapping_unresolved_targets_dropped": sum(
            row[-1] == "unresolved_current_target_dropped" for row in conflicts
        ),
        "duplicate_rows_removed": duplicate_rows_removed,
        "skipped_subject_missing": skipped["subject_missing"],
        "skipped_subject_ambiguous": skipped["subject_ambiguous"],
        "skipped_variable_missing": skipped["variable_missing"],
    }
    return NetworkBuildResult(
        network=normalized_network,
        rows=tuple(selected),
        conflicts=tuple(conflicts),
        counts=counts,
    )


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
        target.flush()
        os.fsync(target.fileno())


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def publish_results(
    output_dir: str | Path,
    results: Sequence[NetworkBuildResult],
    *,
    source_roots: Sequence[str | Path] = (),
) -> Path:
    """Publish a complete new directory only after every file is staged."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise PairMergeError("Output directory already exists; refusing to overwrite it")
    for source in source_roots:
        source_path = Path(source).expanduser().resolve()
        if _paths_overlap(destination, source_path):
            raise PairMergeError("Output directory overlaps an input location")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        summary: dict[str, object] = {
            "format_version": 1,
            "before_value_semantics": "exact_empty_string_from_blank_flag_history",
            "current_value_semantics": "exact_value_refreshed_from_combined_csv",
            "conflict_policy": "blank_history_precedence_with_protected_audit",
            "networks": {},
        }
        for result in results:
            _write_csv(
                staging / f"resolved_before_after_values_{result.network}.csv",
                PAIR_COLUMNS,
                (row.values for row in result.rows),
            )
            _write_csv(
                staging / f"blank_pair_conflicts_{result.network}.csv",
                CONFLICT_COLUMNS,
                result.conflicts,
            )
            summary["networks"][result.network] = dict(result.counts)
        summary_path = staging / "blank_pair_build_summary.json"
        with summary_path.open("x", encoding="utf-8", newline="\n") as target:
            json.dump(summary, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def build_blank_merged_pairs(
    blank_history: str | Path,
    existing_pair_dir: str | Path,
    combined_csv_root: str | Path,
    output_dir: str | Path,
) -> tuple[NetworkBuildResult, ...]:
    """Build both network artifacts and publish them into a new directory."""

    blank_path = Path(blank_history).expanduser().resolve()
    existing_root = Path(existing_pair_dir).expanduser().resolve()
    combined_root = Path(combined_csv_root).expanduser().resolve()
    blank_by_network = load_blank_history(blank_path)
    existing_by_network = load_existing_pairs(existing_root)
    results = tuple(
        build_network_result(
            network,
            blank_by_network[network],
            existing_by_network[network],
            combined_root,
        )
        for network in NETWORKS
    )
    publish_results(
        output_dir,
        results,
        source_roots=(blank_path, existing_root, combined_root),
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge known-blank flag history with existing Analyze Flags value "
            "pairs after refreshing exact current values from combined CSVs."
        )
    )
    parser.add_argument("--blank-history", type=Path, required=True)
    parser.add_argument("--existing-pair-dir", type=Path, required=True)
    parser.add_argument("--combined-csv-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--acknowledge-protected-output",
        action="store_true",
        help="Confirm the new output directory is approved for protected values.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.acknowledge_protected_output:
        raise PairMergeError(
            "--acknowledge-protected-output is required because outputs contain "
            "protected subject-level values"
        )
    results = build_blank_merged_pairs(
        args.blank_history,
        args.existing_pair_dir,
        args.combined_csv_root,
        args.output_dir,
    )
    for result in results:
        counts = result.counts
        print(
            f"{result.network}: wrote {counts['merged_unique_targets']} merged "
            f"pair(s), including {counts['blank_pairs_refreshed']} refreshed "
            f"known-blank pair(s); {counts['blank_precedence_conflicts']} "
            "conflict audit row(s)."
        )
        skipped = sum(
            counts[name]
            for name in (
                "skipped_subject_missing",
                "skipped_subject_ambiguous",
                "skipped_variable_missing",
            )
        )
        if skipped:
            print(
                f"{result.network}: skipped {skipped} blank-history target(s) "
                "whose current value was not uniquely determinable; category "
                "counts are in blank_pair_build_summary.json."
            )
    print("Published protected paired-value artifacts to the new output directory.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PairMergeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
