"""Count unique values in a CSV's ``var`` column and write a sorted CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = "diffs_test_revised_new_module_prescient.csv"


def count_var_values(input_path: Path) -> Counter[str]:
    """Return exact, case-sensitive counts from the input file's ``var`` column."""
    counts: Counter[str] = Counter()

    # utf-8-sig accepts ordinary UTF-8 and also removes an Excel-style BOM.
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "var" not in reader.fieldnames:
            raise ValueError(f"Required column 'var' was not found in {input_path.name}")

        for row in reader:
            value = row.get("var")
            counts["" if value is None else value] += 1

    return counts


def write_counts(output_path: Path, counts: Counter[str]) -> None:
    """Write counts from most to least common, then alphabetically for ties."""
    ordered = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0].casefold(), item[0]),
    )

    # utf-8-sig makes the result open cleanly in Excel while preserving Unicode.
    with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["var", "count"])
        writer.writerows(ordered)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / DEFAULT_INPUT

    parser = argparse.ArgumentParser(
        description="Count unique values in the 'var' column of a CSV file."
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
        help="output CSV (default: <input name>_var_counts.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_var_counts.csv")
    )

    if output_path == input_path:
        raise ValueError("The output path must be different from the input path")

    counts = count_var_values(input_path)
    write_counts(output_path, counts)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    main()
