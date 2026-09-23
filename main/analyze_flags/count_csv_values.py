#!/usr/bin/env python3
"""Count populated cells in a CSV file without loading it all into memory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def count_populated_cells(csv_path: Path) -> tuple[int, int, int, int]:
    """Return data rows, widest row, populated header cells, and populated data cells."""
    data_rows = 0
    widest_row = 0
    populated_header_cells = 0
    populated_data_cells = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)

        if header is not None:
            widest_row = len(header)
            populated_header_cells = sum(bool(cell.strip()) for cell in header)

        for row in reader:
            data_rows += 1
            widest_row = max(widest_row, len(row))
            populated_data_cells += sum(bool(cell.strip()) for cell in row)

    return data_rows, widest_row, populated_header_cells, populated_data_cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count non-empty, non-whitespace cells in a CSV file."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path("wide_data_updated.csv"),
        help="CSV file to inspect (default: wide_data_updated.csv)",
    )
    args = parser.parse_args()

    data_rows, widest_row, header_cells, data_cells = count_populated_cells(
        args.csv_path
    )

    print(f"File: {args.csv_path.resolve()}")
    print(f"Data rows: {data_rows:,}")
    print(f"Columns in widest row: {widest_row:,}")
    print(f"Populated header cells: {header_cells:,}")
    print(f"Populated data cells (header excluded): {data_cells:,}")
    print(f"Populated cells total (header included): {header_cells + data_cells:,}")


if __name__ == "__main__":
    main()
