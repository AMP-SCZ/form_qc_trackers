"""Count unique (comb_csv_val, recalc_value) combinations for each var."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = "diffs_test_revised_new_module_prescient.csv"
OUTPUT_COLUMNS = ("var", "comb_csv_val", "recalc_value")
COLUMN_ALIASES = {
    "var": ("var",),
    "comb_csv_val": ("comb_csv_val",),
    "recalc_value": ("recalc_value", "recalc_val"),
}
Combination = tuple[str, str, str]


def resolve_source_columns(fieldnames: list[str] | None) -> tuple[str, str, str]:
    """Map the requested output columns to their actual source headers."""
    header_lookup = {
        column.strip().casefold(): column
        for column in (fieldnames or [])
        if column is not None
    }
    resolved: list[str] = []
    missing: list[str] = []

    for output_column in OUTPUT_COLUMNS:
        source_column = next(
            (
                header_lookup[alias.casefold()]
                for alias in COLUMN_ALIASES[output_column]
                if alias.casefold() in header_lookup
            ),
            None,
        )
        if source_column is None:
            missing.append(output_column)
        else:
            resolved.append(source_column)

    if missing:
        missing_list = ", ".join(repr(column) for column in missing)
        raise ValueError(f"Required column(s) {missing_list} were not found")

    return tuple(resolved)  # type: ignore[return-value]


def count_combinations(input_path: Path) -> Counter[Combination]:
    """Return exact, case-sensitive counts for each three-column combination."""
    counts: Counter[Combination] = Counter()

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        source_columns = resolve_source_columns(reader.fieldnames)

        for row in reader:
            combination = tuple(
                "" if row.get(column) is None else row[column]
                for column in source_columns
            )
            counts[combination] += 1

    return counts


def write_counts(output_path: Path, counts: Counter[Combination]) -> None:
    """Group by var and sort each var's combinations by descending count."""
    ordered = sorted(
        counts.items(),
        key=lambda item: (
            item[0][0].casefold(),
            item[0][0],
            -item[1],
            item[0][1].casefold(),
            item[0][1],
            item[0][2].casefold(),
            item[0][2],
        ),
    )

    with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow([*OUTPUT_COLUMNS, "count"])
        writer.writerows((*combination, count) for combination, count in ordered)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / DEFAULT_INPUT

    parser = argparse.ArgumentParser(
        description=(
            "Count unique (comb_csv_val, recalc_value) combinations for each var."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=default_input,
        help=f"source CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output CSV (default: <input name>_var_combo_counts.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_var_combo_counts.csv")
    )

    if output_path == input_path:
        raise ValueError("The output path must be different from the input path")

    counts = count_combinations(input_path)
    write_counts(output_path, counts)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    main()
