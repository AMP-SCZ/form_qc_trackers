"""Harvest provable before/after value pairs from the merged tracker history.

Reads ``open_tracker_row_history_{NETWORK}.csv`` (the entry-level history
built by ``analyze_flags.estimate_resolved``), recovers each entry's exact
observed value with ``analyze_flags.value_extraction.infer_before_value``,
rereads the exact current value from the canonical combined CSVs, and
publishes ``resolved_before_after_values_{NETWORK}.csv`` artifacts in the
exact schema required by ``outcome_before_after.py`` and
``merge_blank_flag_pairs.py``.

Coverage decisions (deliberate):

* Every history entry is considered — open or resolved, jump-included or
  not.  A message's interpolated value is a factual observation of the data
  at sighting time regardless of whether the flag's later disappearance
  counts as a real resolution, mirroring how the blank-flag walk treats
  "ever flagged" keys.
* One row per (network, timepoint, subject, variable) cell.  When several
  entries prove values for the same cell, the observation with the latest
  ``Latest_seen`` wins; distinct values tied at the same latest timestamp
  fail closed and land in the conflict audit instead.
* Entries whose message carries the value-conflict marker, identity-column
  targets, unsupported timepoints, and ``multiple_timepoints`` entries are
  counted and skipped.
* Cells whose current value cannot be uniquely determined are counted and
  skipped — the outcome overlay requires the exact live value to verify
  before applying a replacement.

This program never modifies its inputs and refuses to publish into an
existing output directory.  Operator diagnostics contain counts only, not
subjects, variables, or values.
"""

from __future__ import annotations

import main  # Make deployed sibling packages available to this CLI.

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd

from analyze_flags.value_extraction import infer_before_value
from merge_blank_flag_pairs import (
    IMMUTABLE_IDENTITY_COLUMNS,
    PAIR_COLUMNS,
    TIMEPOINT_ORDER,
    PairMergeError,
    TargetKey,
    _paths_overlap,
    _read_current_values,
    _status_for_current_value,
    _write_csv,
)
from outcome_inputs import OutcomeInputError, normalize_network
from outcome_inputs import normalize_timepoint as normalize_outcome_timepoint


NETWORKS = ("PRONET", "PRESCIENT")
HISTORY_BASENAME = "open_tracker_row_history_{network}.csv"
REQUIRED_HISTORY_COLUMNS = (
    "Subject",
    "Timepoint",
    "General_Flag",
    "variable",
    "message",
    "canonical_template",
    "Latest_seen",
)
CONFLICT_COLUMNS = (
    "Subject",
    "Timepoint",
    "variable",
    "n_tied_observations",
    "distinct_before_values",
)
TRUTHY = frozenset({"true", "1", "yes"})


@dataclass(frozen=True)
class Observation:
    """One provable value sighting for a target cell."""

    latest_seen_ts: pd.Timestamp
    before_value: str
    before_value_source: str
    subject: str
    timepoint: str
    general_flag: str
    variable: str
    message_variable: str
    canonical_template: str
    latest_seen_raw: str


@dataclass(frozen=True)
class NetworkHarvestResult:
    network: str
    rows: tuple[tuple[str, ...], ...]
    conflicts: tuple[tuple[str, ...], ...]
    counts: Mapping[str, int]


def _load_history_frame(history_dir: Path, network: str) -> pd.DataFrame:
    path = history_dir / HISTORY_BASENAME.format(network=network)
    try:
        frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    except FileNotFoundError:
        raise PairMergeError(
            f"History CSV for {network} was not found in the history "
            "directory"
        ) from None
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError,
            UnicodeError) as exc:
        raise PairMergeError(
            f"History CSV for {network} could not be read"
        ) from exc
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [c for c in REQUIRED_HISTORY_COLUMNS if c not in frame.columns]
    if missing:
        raise PairMergeError(
            f"History CSV for {network} is missing columns {missing}"
        )
    # Histories written before the message-audit columns existed carry the
    # same compatibility defaults flag_analytics applies: the raw message
    # variable is the entry variable, and no conflict was recorded.
    if "message_variable" not in frame.columns:
        frame["message_variable"] = frame["variable"]
    if "message_value_conflict" not in frame.columns:
        frame["message_value_conflict"] = "False"
    return frame


def collect_observations(
        frame: pd.DataFrame, network: str,
) -> tuple[dict[TargetKey, list[Observation]], Counter]:
    """Group provable value observations by target cell, counting drops."""

    counts: Counter[str] = Counter()
    by_cell: dict[TargetKey, list[Observation]] = defaultdict(list)
    latest_ts = pd.to_datetime(
        frame["Latest_seen"], errors="coerce", utc=True)
    for position, row in enumerate(frame.itertuples(index=False)):
        counts["history_entries"] += 1
        subject = str(row.Subject).strip()
        variable = str(row.variable).strip()
        if not subject or not variable:
            counts["skipped_blank_subject_or_variable"] += 1
            continue
        if variable.casefold() in IMMUTABLE_IDENTITY_COLUMNS:
            counts["skipped_identity_column_target"] += 1
            continue
        if str(row.message_value_conflict).strip().casefold() in TRUTHY:
            counts["skipped_message_value_conflict"] += 1
            continue

        timepoint_raw = str(row.Timepoint).strip()
        if timepoint_raw == "multiple_timepoints":
            counts["skipped_multiple_timepoints"] += 1
            continue
        try:
            timepoint = normalize_outcome_timepoint(timepoint_raw)
        except OutcomeInputError:
            counts["skipped_unsupported_timepoint"] += 1
            continue

        message_variable = str(row.message_variable).strip() or variable
        before = infer_before_value(message_variable, row.message)
        if before is None:
            counts["entries_without_provable_value"] += 1
            continue
        counts["entries_with_provable_value"] += 1

        ts = latest_ts.iloc[position]
        if pd.isna(ts):
            # Unparseable Latest_seen sorts before any real sighting, so it
            # can only win a cell when no dated observation exists.
            ts = pd.Timestamp(0, tz="UTC")
        by_cell[TargetKey(network, timepoint, subject, variable)].append(
            Observation(
                latest_seen_ts=ts,
                before_value=before.value,
                before_value_source=before.source,
                subject=subject,
                timepoint=timepoint,
                general_flag=str(row.General_Flag).strip(),
                variable=variable,
                message_variable=message_variable,
                canonical_template=str(row.canonical_template),
                latest_seen_raw=str(row.Latest_seen),
            )
        )
    return by_cell, counts


def resolve_cells(
        by_cell: Mapping[TargetKey, Sequence[Observation]],
        counts: Counter,
) -> tuple[dict[TargetKey, Observation], list[tuple[str, ...]]]:
    """Latest observation wins; distinct values at the tied latest fail closed."""

    chosen: dict[TargetKey, Observation] = {}
    conflicts: list[tuple[str, ...]] = []
    for key in sorted(by_cell):
        observations = by_cell[key]
        newest = max(observation.latest_seen_ts
                     for observation in observations)
        candidates = [observation for observation in observations
                      if observation.latest_seen_ts == newest]
        distinct_values = sorted({candidate.before_value
                                  for candidate in candidates})
        if len(distinct_values) != 1:
            counts["cells_with_conflicting_latest_values"] += 1
            conflicts.append((
                key.subject,
                key.timepoint,
                key.variable,
                str(len(candidates)),
                "; ".join(distinct_values),
            ))
            continue
        # Deterministic representative among same-value ties.
        chosen[key] = min(
            candidates,
            key=lambda observation: (
                observation.canonical_template,
                observation.general_flag,
                observation.message_variable,
                observation.latest_seen_raw,
            ),
        )
    counts["cells_with_provable_value"] = len(chosen) + sum(
        1 for _ in conflicts)
    return chosen, conflicts


def harvest_network(
        history_dir: Path, combined_root: Path, network: str,
) -> NetworkHarvestResult:
    normalized_network = normalize_network(network)
    frame = _load_history_frame(history_dir, normalized_network)
    by_cell, counts = collect_observations(frame, normalized_network)
    chosen, conflicts = resolve_cells(by_cell, counts)

    current_values, skipped = _read_current_values(
        combined_root, normalized_network, chosen.keys())
    counts["skipped_subject_missing"] = skipped["subject_missing"]
    counts["skipped_subject_ambiguous"] = skipped["subject_ambiguous"]
    counts["skipped_variable_missing"] = skipped["variable_missing"]

    rows: list[tuple[str, ...]] = []
    for key in sorted(
            chosen,
            key=lambda item: (
                TIMEPOINT_ORDER.get(item.timepoint, len(TIMEPOINT_ORDER)),
                item.subject,
                item.variable,
            )):
        current_value = current_values.get(key)
        if current_value is None:
            continue
        observation = chosen[key]
        values = {
            "Subject": observation.subject,
            "Timepoint": observation.timepoint,
            "General_Flag": observation.general_flag,
            "variable": observation.variable,
            "message_variable": observation.message_variable,
            "canonical_template": observation.canonical_template,
            "before_value_available": "True",
            "before_value": observation.before_value,
            "before_value_source": observation.before_value_source,
            "Latest_seen": observation.latest_seen_raw,
            "status": _status_for_current_value(current_value),
            "current_value": current_value,
        }
        rows.append(tuple(values[name] for name in PAIR_COLUMNS))
        counts["pairs_written"] += 1
        counts[f"pairs_source_{observation.before_value_source}"] += 1
        if observation.before_value == current_value:
            counts["pairs_where_before_equals_current"] += 1

    return NetworkHarvestResult(
        network=normalized_network,
        rows=tuple(rows),
        conflicts=tuple(conflicts),
        counts=dict(counts),
    )


def publish_results(
        output_dir: str | Path,
        results: Sequence[NetworkHarvestResult],
        *,
        source_roots: Sequence[str | Path] = (),
) -> Path:
    """Publish a complete new directory only after every file is staged."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise PairMergeError(
            "Output directory already exists; refusing to overwrite it")
    for source in source_roots:
        source_path = Path(source).expanduser().resolve()
        if _paths_overlap(destination, source_path):
            raise PairMergeError("Output directory overlaps an input location")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        summary: dict[str, object] = {
            "format_version": 1,
            "before_value_semantics": (
                "exact_value_proven_by_flag_message_latest_observation"),
            "current_value_semantics": (
                "exact_value_refreshed_from_combined_csv"),
            "coverage": (
                "all_history_entries_regardless_of_resolution_or_jump_state"),
            "networks": {},
        }
        for result in results:
            _write_csv(
                staging / f"resolved_before_after_values_{result.network}.csv",
                PAIR_COLUMNS,
                result.rows,
            )
            _write_csv(
                staging / f"value_pair_conflicts_{result.network}.csv",
                CONFLICT_COLUMNS,
                result.conflicts,
            )
            summary["networks"][result.network] = dict(result.counts)
        summary_path = staging / "history_value_pair_summary.json"
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


def build_history_value_pairs(
        history_dir: str | Path,
        combined_csv_root: str | Path,
        output_dir: str | Path,
) -> tuple[NetworkHarvestResult, ...]:
    history_root = Path(history_dir).expanduser().resolve()
    combined_root = Path(combined_csv_root).expanduser().resolve()
    results = tuple(
        harvest_network(history_root, combined_root, network)
        for network in NETWORKS
    )
    publish_results(
        output_dir,
        results,
        source_roots=(history_root, combined_root),
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Harvest provable before/after value pairs for every flag type "
            "from the merged tracker history, refreshing exact current "
            "values from combined CSVs."
        )
    )
    parser.add_argument("--history-dir", type=Path, required=True)
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
            "--acknowledge-protected-output is required because outputs "
            "contain protected subject-level values"
        )
    results = build_history_value_pairs(
        args.history_dir,
        args.combined_csv_root,
        args.output_dir,
    )
    for result in results:
        counts = result.counts
        print(
            f"{result.network}: {counts.get('history_entries', 0)} history "
            f"entries -> {counts.get('entries_with_provable_value', 0)} "
            f"provable observations -> "
            f"{counts.get('pairs_written', 0)} pair(s) written "
            f"({counts.get('pairs_where_before_equals_current', 0)} where "
            "the value is unchanged); "
            f"{counts.get('cells_with_conflicting_latest_values', 0)} "
            "conflicting cell(s) in the audit."
        )
        skipped_current = sum(
            counts.get(name, 0)
            for name in (
                "skipped_subject_missing",
                "skipped_subject_ambiguous",
                "skipped_variable_missing",
            )
        )
        if skipped_current:
            print(
                f"{result.network}: {skipped_current} cell(s) skipped "
                "because the current value was not uniquely determinable; "
                "category counts are in history_value_pair_summary.json."
            )
    print("Published protected paired-value artifacts to the new output "
          "directory.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PairMergeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
