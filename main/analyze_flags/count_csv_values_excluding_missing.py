#!/usr/bin/env python3
"""Count populated CSV cells while excluding this project's missing codes."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path


# Canonical missing codes from utils/utils.py. Numeric forms are normalized so
# equivalent CSV spellings such as 999 and 999.0 are treated the same way.
NUMERIC_MISSING_CODES = {
    Decimal("-3"): "-3",
    Decimal("-9"): "-9",
    Decimal("-99"): "-99",
    Decimal("999"): "999",
}
DATE_MISSING_CODES = frozenset({"1901-01-01", "1903-03-03", "1909-09-09"})


def missing_code_label(value: str) -> str | None:
    """Return a canonical label when *value* is one of the missing codes."""
    if value in DATE_MISSING_CODES:
        return value

    try:
        numeric_value = Decimal(value)
    except InvalidOperation:
        return None

    return NUMERIC_MISSING_CODES.get(numeric_value)


def count_usable_cells(
    csv_path: Path,
) -> tuple[int, int, int, int, Counter[str]]:
    """Count nonblank cells, excluding missing codes from data rows."""
    data_rows = 0
    widest_row = 0
    populated_header_cells = 0
    usable_data_cells = 0
    missing_code_counts: Counter[str] = Counter()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)

        if header is not None:
            widest_row = len(header)
            populated_header_cells = sum(bool(cell.strip()) for cell in header)

        for row in reader:
            data_rows += 1
            widest_row = max(widest_row, len(row))
            for cell in row:
                value = cell.strip()
                if not value:
                    continue

                code = missing_code_label(value)
                if code is None:
                    usable_data_cells += 1
                else:
                    missing_code_counts[code] += 1

    return (
        data_rows,
        widest_row,
        populated_header_cells,
        usable_data_cells,
        missing_code_counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count nonblank CSV cells while excluding the project's missing codes."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path("wide_data_updated.csv"),
        help="CSV file to inspect (default: wide_data_updated.csv)",
    )
    args = parser.parse_args()

    data_rows, widest_row, header_cells, data_cells, missing_counts = (
        count_usable_cells(args.csv_path)
    )
    missing_cells = sum(missing_counts.values())

    print(f"File: {args.csv_path.resolve()}")
    print(f"Data rows: {data_rows:,}")
    print(f"Columns in widest row: {widest_row:,}")
    print(f"Populated header cells: {header_cells:,}")
    print(f"Non-empty data cells before exclusions: {data_cells + missing_cells:,}")
    print(f"Missing-code cells excluded: {missing_cells:,}")
    print(f"Usable data cells (header excluded): {data_cells:,}")
    print(f"Usable cells total (header included): {header_cells + data_cells:,}")
    print("Excluded missing-code breakdown:")
    for code in ("-3", "-9", "-99", "999", *sorted(DATE_MISSING_CODES)):
        print(f"  {code}: {missing_counts[code]:,}")


if __name__ == "__main__":
    main()
