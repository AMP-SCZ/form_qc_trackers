"""Count unique value combinations per ``var`` and write a sorted CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = "diffs_test_revised_new_module_prescient.csv"
GROUP_COLUMNS = ("var", "comb_csv_val", "recalc_value")
Combination = tuple[str, str, str]


def count_combinations(input_path: Path) -> Counter[Combination]:
    """Return exact, case-sensitive counts for the three grouping columns."""
    counts: Counter[Combination] = Counter()

    # utf-8-sig accepts ordinary UTF-8 and also removes an Excel-style BOM.
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = set(reader.fieldnames or [])
        missing = [column for column in GROUP_COLUMNS if column not in fieldnames]
        if missing:
            missing_list = ", ".join(repr(column) for column in missing)
            raise ValueError(
                f"Required column(s) {missing_list} not found in {input_path.name}"
            )

        for row in reader:
            values = tuple(
                "" if row.get(column) is None else row[column]
                for column in GROUP_COLUMNS
            )
            counts[values] += 1

    return counts


def write_counts(output_path: Path, counts: Counter[Combination]) -> None:
    """Group by var and order each var's combinations from most to least common."""
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

    # utf-8-sig makes the result open cleanly in Excel while preserving Unicode.
    with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow([*GROUP_COLUMNS, "count"])
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
