"""Build before/after value pairs from the calculated-field diff artifacts.

Reads ``diffs_test_revised_new_module_{pronet,prescient}.csv`` (written by
``discover_errors/search_calc_errs.py``: one row per (subject, timepoint,
variable) whose stored combined-CSV value disagreed with the newer REDCap
export). The ``comb_csv_val`` column is the stored value BEFORE the
calculated field was corrected; it becomes the pair's ``before_value``. The
exact current value is reread from the supplied canonical combined CSVs —
the outcome experiment verifies the live cell against ``current_value``
byte-for-byte and aborts otherwise, so the export-side ``recalc_val`` column
is deliberately not used.

One pair per (network, timepoint, subject, variable) cell; duplicate rows
for a cell collapse only when their before values agree exactly, otherwise
the cell fails closed into a conflict audit. Identity-column targets,
unsupported timepoints, and cells whose current value is not uniquely
determinable are counted and skipped.

This program never modifies its inputs and refuses to publish into an
existing output directory. Operator diagnostics contain counts only, not
subjects, variables, or values.
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
from typing import Mapping, Optional, Sequence

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
DIFF_BASENAME = "diffs_test_revised_new_module_{network_lower}.csv"
DIFF_COLUMNS = ("subject", "timepoint", "var", "comb_csv_val", "recalc_val")
CONFLICT_COLUMNS = (
    "Subject",
    "Timepoint",
    "variable",
    "n_rows",
    "distinct_before_values",
)


@dataclass(frozen=True)
class NetworkBuildResult:
    network: str
    rows: tuple[tuple[str, ...], ...]
    conflicts: tuple[tuple[str, ...], ...]
    counts: Mapping[str, int]


def _read_diff_rows(path: Path) -> list[tuple[str, ...]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            try:
                header = next(reader)
            except StopIteration:
                raise PairMergeError("Calc-diff artifact is empty") from None
            if tuple(header) != DIFF_COLUMNS:
                raise PairMergeError(
                    "Calc-diff artifact does not have the exact expected schema"
                )
            rows: list[tuple[str, ...]] = []
            for row in reader:
                if len(row) != len(DIFF_COLUMNS):
                    raise PairMergeError(
                        "Calc-diff artifact contains a malformed logical row"
                    )
                rows.append(tuple(row))
            return rows
    except FileNotFoundError:
        raise PairMergeError("Calc-diff artifact was not found") from None
    except csv.Error as exc:
        raise PairMergeError("Calc-diff artifact is not valid CSV") from exc
    except UnicodeError as exc:
        raise PairMergeError("Calc-diff artifact is not valid UTF-8 CSV") from exc
    except OSError as exc:
        raise PairMergeError("Calc-diff artifact could not be read") from exc


def collect_before_values(
    rows: Sequence[tuple[str, ...]], network: str,
    *, after_source: str = "live",
) -> tuple[dict[TargetKey, tuple[str, str]], list[tuple[str, ...]], Counter]:
    """Group (before, recalc) values per target cell; conflicts fail closed.

    In live mode only the before value must be unique per cell (recalc is
    unused there); in recalc mode both sides must be unique.
    """

    counts: Counter[str] = Counter()
    by_cell: dict[TargetKey, list[tuple[str, str]]] = defaultdict(list)
    positions = {name: index for index, name in enumerate(DIFF_COLUMNS)}
    for row in rows:
        counts["diff_rows"] += 1
        subject = row[positions["subject"]].strip()
        variable = row[positions["var"]].strip()
        if not subject or not variable:
            counts["skipped_blank_subject_or_variable"] += 1
            continue
        if variable.casefold() in IMMUTABLE_IDENTITY_COLUMNS:
            counts["skipped_identity_column_target"] += 1
            continue
        try:
            timepoint = normalize_outcome_timepoint(row[positions["timepoint"]])
        except OutcomeInputError:
            counts["skipped_unsupported_timepoint"] += 1
            continue
        by_cell[TargetKey(network, timepoint, subject, variable)].append(
            (row[positions["comb_csv_val"]], row[positions["recalc_val"]])
        )

    chosen: dict[TargetKey, tuple[str, str]] = {}
    conflicts: list[tuple[str, ...]] = []
    for key in sorted(by_cell):
        if after_source == "recalc":
            distinct = sorted(set(by_cell[key]))
        else:
            # Live mode never reads the recalc side, so rows differing only
            # in recalc_val are not a conflict.
            distinct = [
                (before, "")
                for before in sorted({before for before, _ in by_cell[key]})
            ]
        if len(distinct) != 1:
            counts["cells_with_conflicting_before_values"] += 1
            conflicts.append((
                key.subject,
                key.timepoint,
                key.variable,
                str(len(by_cell[key])),
                "; ".join(before for before, _ in distinct),
            ))
            continue
        chosen[key] = distinct[0]
    counts["unique_cells"] = len(chosen) + len(conflicts)
    return chosen, conflicts, counts


def build_network_result(
    diff_dir: Path, combined_root: Path, network: str,
    *, only_changed: bool = False, after_source: str = "live",
) -> NetworkBuildResult:
    if after_source not in {"live", "recalc"}:
        raise PairMergeError("after_source must be 'live' or 'recalc'")
    normalized_network = normalize_network(network)
    path = diff_dir / DIFF_BASENAME.format(
        network_lower=normalized_network.lower()
    )
    rows = _read_diff_rows(path)
    chosen, conflicts, counts = collect_before_values(
        rows, normalized_network, after_source=after_source)

    # The live read doubles as the existence gate either way: the outcome
    # experiment requires each target cell to exist exactly once in the
    # staged base, whichever value ends up in current_value.
    current_values, skipped = _read_current_values(
        combined_root, normalized_network, chosen.keys()
    )
    counts["skipped_subject_missing"] = skipped["subject_missing"]
    counts["skipped_subject_ambiguous"] = skipped["subject_ambiguous"]
    counts["skipped_variable_missing"] = skipped["variable_missing"]

    pair_rows: list[tuple[str, ...]] = []
    for key in sorted(
        chosen,
        key=lambda item: (
            TIMEPOINT_ORDER.get(item.timepoint, len(TIMEPOINT_ORDER)),
            item.subject,
            item.variable,
        ),
    ):
        live_value = current_values.get(key)
        if live_value is None:
            continue
        before_value, recalc_value = chosen[key]
        current_value = recalc_value if after_source == "recalc" else live_value
        if only_changed and before_value == current_value:
            # A no-op overlay writes the same byte back and cannot change
            # any outcome, but it would still put the subject on the
            # experiment's compute roster; drop it when requested.
            counts["pairs_skipped_as_noop"] += 1
            continue
        values = {
            "Subject": key.subject,
            "Timepoint": key.timepoint,
            "General_Flag": "calculated_field_diff",
            "variable": key.variable,
            "message_variable": key.variable,
            "canonical_template": (
                "Stored combined value differs from the newer REDCap export."
            ),
            "before_value_available": "True",
            "before_value": before_value,
            "before_value_source": "calc_diff_comb_csv_val",
            "Latest_seen": "",
            "status": _status_for_current_value(current_value),
            "current_value": current_value,
        }
        pair_rows.append(tuple(values[name] for name in PAIR_COLUMNS))
        counts["pairs_written"] += 1
        if before_value == current_value:
            counts["pairs_where_before_equals_current"] += 1

    return NetworkBuildResult(
        network=normalized_network,
        rows=tuple(pair_rows),
        conflicts=tuple(conflicts),
        counts=dict(counts),
    )


def publish_results(
    output_dir: str | Path,
    results: Sequence[NetworkBuildResult],
    *,
    source_roots: Sequence[str | Path] = (),
    after_source: str = "live",
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise PairMergeError(
            "Output directory already exists; refusing to overwrite it"
        )
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
                "stored_combined_value_before_calculated_field_correction"
            ),
            "current_value_semantics": (
                "recalc_val_from_diff_artifact"
                if after_source == "recalc"
                else "exact_value_refreshed_from_combined_csv"
            ),
            "source": "diffs_test_revised_new_module (search_calc_errs)",
            "networks": {},
        }
        for result in results:
            _write_csv(
                staging / f"resolved_before_after_values_{result.network}.csv",
                PAIR_COLUMNS,
                result.rows,
            )
            _write_csv(
                staging / f"calc_diff_pair_conflicts_{result.network}.csv",
                CONFLICT_COLUMNS,
                result.conflicts,
            )
            summary["networks"][result.network] = dict(result.counts)
        summary_path = staging / "calc_diff_pair_summary.json"
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


def build_calc_diff_pairs(
    diff_dir: str | Path,
    combined_csv_root: str | Path,
    output_dir: str | Path,
    *, only_changed: bool = False, after_source: str = "live",
) -> tuple[NetworkBuildResult, ...]:
    diff_root = Path(diff_dir).expanduser().resolve()
    combined_root = Path(combined_csv_root).expanduser().resolve()
    results = tuple(
        build_network_result(
            diff_root, combined_root, network,
            only_changed=only_changed, after_source=after_source)
        for network in NETWORKS
    )
    publish_results(
        output_dir,
        results,
        source_roots=(diff_root, combined_root),
        after_source=after_source,
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build before/after value pairs from the calculated-field diff "
            "artifacts, refreshing exact current values from combined CSVs."
        )
    )
    parser.add_argument("--diff-dir", type=Path, required=True)
    parser.add_argument("--combined-csv-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--acknowledge-protected-output",
        action="store_true",
        help="Confirm the new output directory is approved for protected values.",
    )
    parser.add_argument(
        "--only-changed",
        action="store_true",
        help=(
            "Keep only pairs whose before value differs from the live "
            "combined value (no-op pairs cannot change any outcome)."
        ),
    )
    parser.add_argument(
        "--after-source",
        choices=("live", "recalc"),
        default="live",
        help=(
            "Where current_value comes from: 'live' rereads the combined "
            "CSVs; 'recalc' uses the diff artifact's recalc_val (pair the "
            "result with outcome_before_after --after-from-pairs)."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.acknowledge_protected_output:
        raise PairMergeError(
            "--acknowledge-protected-output is required because outputs "
            "contain protected subject-level values"
        )
    results = build_calc_diff_pairs(
        args.diff_dir,
        args.combined_csv_root,
        args.output_dir,
        only_changed=args.only_changed,
        after_source=args.after_source,
    )
    for result in results:
        counts = result.counts
        print(
            f"{result.network}: {counts.get('diff_rows', 0)} diff row(s) -> "
            f"{counts.get('unique_cells', 0)} unique cell(s) -> "
            f"{counts.get('pairs_written', 0)} pair(s) written "
            f"({counts.get('pairs_where_before_equals_current', 0)} where "
            "the value is unchanged); "
            f"{counts.get('cells_with_conflicting_before_values', 0)} "
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
                f"{result.network}: {skipped_current} cell(s) skipped because "
                "the current value was not uniquely determinable; category "
                "counts are in calc_diff_pair_summary.json."
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
