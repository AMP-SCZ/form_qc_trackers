"""Summarize both forms involved in Date Report flags and create a graph.

The input may be:

* a QC tracker workbook containing a ``Date Report`` sheet;
* a CSV/TSV export of that sheet; or
* the pipeline's raw CSV/Parquet flag table.

Each individual flag contributes once to every distinct form involved: the
later/current form and the earlier reference form. Tracker workbooks can merge
several flags into one displayed row, so the program separates the ``Flags``
messages and uses ``dependencies/important_form_vars.json`` to map the earlier
interview-date variable back to its form. Raw flag tables use ``affected_forms``
directly. If the same form appears at both timepoints, it is counted once for
that flag because only one distinct form is involved.

Examples
--------
    python date_report_form_summary.py PRONET_Output_V2.xlsx
    python date_report_form_summary.py combined_qc_flags.parquet --network PRONET
    python date_report_form_summary.py date_report.csv --output-dir summaries
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import textwrap
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SHEET = "Date Report"
FORM_COLUMN_NAMES = ("displayed_form", "Form")
FLAG_COUNT_COLUMN_NAMES = ("Flag Count", "flag_count")
FLAG_TEXT_COLUMN_NAMES = ("Flags", "error_message")
REPORT_COLUMN_NAMES = ("reports", "report", "Report")
NETWORK_COLUMN_NAMES = ("network", "Network")
SUPPORTED_EXCEL_SUFFIXES = frozenset({".xlsx", ".xlsm", ".xls", ".xlsb"})
PREVIOUS_DATE_VARIABLE_RE = re.compile(
    r"\bday\(s\)\s+before\s+"
    r"(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"\(\d{4}-\d{2}-\d{2}\)\s+at\s+the\s+earlier\s+timepoint\b",
    re.IGNORECASE,
)


class DateReportSummaryError(ValueError):
    """Raised when an input cannot be interpreted as Date Report data."""


def _normalized_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    by_normalized_name = {
        _normalized_column_name(column): column for column in frame.columns
    }
    for candidate in candidates:
        match = by_normalized_name.get(_normalized_column_name(candidate))
        if match is not None:
            return match
    return None


def _route_contains_date_report(value: object) -> bool:
    if pd.isna(value):
        return False
    return any(
        part.strip().casefold() == DEFAULT_SHEET.casefold()
        for part in str(value).split("|")
    )


def _read_excel_date_report(path: Path, sheet_name: str) -> pd.DataFrame:
    try:
        workbook = pd.ExcelFile(path)
    except Exception as exc:
        raise DateReportSummaryError(
            f"Could not open workbook '{path}': {exc}") from exc

    matching_sheet = next(
        (name for name in workbook.sheet_names
         if name.strip().casefold() == sheet_name.strip().casefold()),
        None,
    )
    if matching_sheet is None:
        available = ", ".join(workbook.sheet_names) or "<none>"
        raise DateReportSummaryError(
            f"Workbook '{path}' has no '{sheet_name}' sheet. "
            f"Available sheets: {available}")

    try:
        return pd.read_excel(workbook, sheet_name=matching_sheet)
    except Exception as exc:
        raise DateReportSummaryError(
            f"Could not read sheet '{matching_sheet}' from '{path}': {exc}"
        ) from exc


def read_date_report(path: Path, sheet_name: str = DEFAULT_SHEET,
                     network: str | None = None) -> pd.DataFrame:
    """Read a tracker sheet or raw flag table and retain Date Report rows."""
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise DateReportSummaryError(f"Input path is not a file: {path}")

    suffix = path.suffix.casefold()
    try:
        if suffix in SUPPORTED_EXCEL_SUFFIXES:
            frame = _read_excel_date_report(path, sheet_name)
        elif suffix == ".csv":
            frame = pd.read_csv(path, keep_default_na=False)
        elif suffix in {".tsv", ".txt"}:
            frame = pd.read_csv(path, sep="\t", keep_default_na=False)
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        else:
            raise DateReportSummaryError(
                "Unsupported input type. Use an Excel workbook, CSV, TSV, "
                "or Parquet file.")
    except DateReportSummaryError:
        raise
    except Exception as exc:
        raise DateReportSummaryError(
            f"Could not read '{path}': {exc}") from exc

    # Workbooks are already restricted by sheet. Raw combined flag files need
    # an exact route-token filter so names such as "Old Date Report" do not
    # match accidentally.
    if suffix not in SUPPORTED_EXCEL_SUFFIXES:
        report_column = _find_column(frame, REPORT_COLUMN_NAMES)
        if report_column is not None:
            frame = frame[
                frame[report_column].map(_route_contains_date_report)
            ].copy()

    if network:
        network_column = _find_column(frame, NETWORK_COLUMN_NAMES)
        if network_column is None:
            raise DateReportSummaryError(
                "--network was supplied, but the input has no network column.")
        frame = frame[
            frame[network_column].astype(str).str.strip().str.casefold()
            == network.strip().casefold()
        ].copy()

    return frame


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return ""
    return str(value).strip()


def _split_distinct_forms(value: object) -> list[str]:
    """Read a raw affected_forms cell while preserving first-seen order."""
    if (not isinstance(value, (str, bytes, dict))
            and pd.api.types.is_list_like(value)):
        candidates = [_text(item) for item in value]
    else:
        raw = _text(value)
        parsed = None
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                parsed = None
        if (parsed is not None
                and not isinstance(parsed, (str, bytes, dict))
                and pd.api.types.is_list_like(parsed)):
            candidates = [_text(item) for item in parsed]
        else:
            candidates = [_text(item) for item in raw.split("|")]
    return list(dict.fromkeys(item for item in candidates if item))


def load_interview_date_form_map(path: Path) -> dict[str, str]:
    """Load date-variable and legacy form tokens -> canonical form names."""
    if not path.exists():
        raise FileNotFoundError(f"Form mapping file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DateReportSummaryError(
            f"Could not read form mapping '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise DateReportSummaryError(
            f"Form mapping '{path}' must contain a JSON object.")

    mapping: dict[str, str] = {}
    for form, info in payload.items():
        if not isinstance(info, dict):
            continue
        form_name = _text(form)
        if not form_name:
            continue
        variable = _text(info.get("interview_date_var", ""))
        # DateChecks normally prints the previous interview-date variable. Its
        # legacy fallback may print the previous form name instead, so accept
        # both tokens without guessing.
        aliases = [form_name]
        if variable:
            aliases.insert(0, variable)
        for alias in aliases:
            key = alias.casefold()
            existing = mapping.get(key)
            if existing is not None and existing != form_name:
                raise DateReportSummaryError(
                    f"Mapping token '{alias}' resolves to both '{existing}' "
                    f"and '{form_name}' in '{path}'.")
            mapping[key] = form_name
    if not mapping:
        raise DateReportSummaryError(
            f"No interview-date mappings were found in '{path}'.")
    return mapping


def count_individual_flags(frame: pd.DataFrame) -> int:
    """Count raw flags or exploded workbook messages for CLI reporting."""
    if frame.empty:
        return 0
    if _find_column(frame, ("affected_forms",)) is not None:
        return len(frame)
    form_column = _find_column(frame, FORM_COLUMN_NAMES)
    flag_text_column = _find_column(frame, FLAG_TEXT_COLUMN_NAMES)
    is_formatted_tracker = (
        form_column is not None
        and _normalized_column_name(form_column)
        == _normalized_column_name("Form")
    )
    if is_formatted_tracker and flag_text_column is not None:
        return sum(
            len([part for part in _text(value).split(" | ") if part.strip()])
            for value in frame[flag_text_column]
        )
    return len(frame)


def _resolve_form_map_path(explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "dependencies" / "important_form_vars.json",
        script_dir.parent / "dependencies" / "important_form_vars.json",
        script_dir.parent.parent / "dependencies" / "important_form_vars.json",
        Path.cwd() / "dependencies" / "important_form_vars.json",
        "/home/ob001/refactored_qc/dependencies/important_form_vars.json"
    ]
    checked: list[Path] = []
    for candidate in candidates:
        resolved = Path(candidate).expanduser().resolve()
        if resolved in checked:
            continue
        checked.append(resolved)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "Could not locate dependencies/important_form_vars.json. Pass its "
        "path with --important-form-vars. Checked: "
        + ", ".join(str(path) for path in checked)
    )


def _expected_merged_flag_count(value: object, column: str,
                                workbook_row: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = math.nan
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise DateReportSummaryError(
            f"'{column}' must contain a non-negative whole number at "
            f"workbook row {workbook_row}.")
    return int(numeric)


def _previous_form_from_message(message: str,
                                variable_to_form: dict[str, str],
                                workbook_row: int) -> str:
    matches = list(PREVIOUS_DATE_VARIABLE_RE.finditer(message))
    if len(matches) != 1:
        raise DateReportSummaryError(
            f"Could not identify exactly one earlier date variable in the "
            f"flag at workbook row {workbook_row}: {message!r}")
    variable = matches[0].group("variable")
    previous_form = variable_to_form.get(variable.casefold())
    if previous_form is None:
        raise DateReportSummaryError(
            f"Earlier date variable '{variable}' at workbook row "
            f"{workbook_row} is not mapped in important_form_vars.json.")
    return previous_form


def count_flags_by_form(
        frame: pd.DataFrame,
        variable_to_form: dict[str, str] | None = None) -> pd.DataFrame:
    """Count every distinct form involved in each individual date flag."""
    if frame.empty:
        return pd.DataFrame({
            "form": pd.Series(dtype="string"),
            "flag_count": pd.Series(dtype="int64"),
        })

    form_occurrences: list[str] = []
    affected_forms_column = _find_column(frame, ("affected_forms",))
    if affected_forms_column is not None:
        # Canonical raw output already carries the exact distinct forms for one
        # flag. This is more reliable than reverse-parsing its message.
        for position, value in enumerate(
                frame[affected_forms_column], start=2):
            forms = _split_distinct_forms(value)
            if not forms:
                raise DateReportSummaryError(
                    f"'{affected_forms_column}' is blank at data row "
                    f"{position}; both forms cannot be counted accurately.")
            form_occurrences.extend(forms)
    else:
        form_column = _find_column(frame, FORM_COLUMN_NAMES)
        flag_text_column = _find_column(frame, FLAG_TEXT_COLUMN_NAMES)
        if form_column is None or flag_text_column is None:
            raise DateReportSummaryError(
                "Counting both forms requires either raw 'affected_forms', "
                "or tracker columns 'Form' and 'Flags'.")
        if variable_to_form is None:
            raise DateReportSummaryError(
                "An interview-date variable mapping is required to recover "
                "the earlier form from tracker flag messages.")

        is_formatted_tracker = (
            _normalized_column_name(form_column)
            == _normalized_column_name("Form")
        )
        flag_count_column = (
            _find_column(frame, FLAG_COUNT_COLUMN_NAMES)
            if is_formatted_tracker else None
        )

        for position, (_, row) in enumerate(frame.iterrows(), start=2):
            current_form = _text(row[form_column])
            merged_message = _text(row[flag_text_column])
            if not current_form or not merged_message:
                raise DateReportSummaryError(
                    f"Current form or flag message is blank at data row "
                    f"{position}; both forms cannot be counted accurately.")
            messages = (
                [part.strip() for part in merged_message.split(" | ")
                 if part.strip()]
                if is_formatted_tracker else [merged_message]
            )
            if flag_count_column is not None:
                expected = _expected_merged_flag_count(
                    row[flag_count_column], flag_count_column, position)
                if expected != len(messages):
                    raise DateReportSummaryError(
                        f"'{flag_count_column}' says {expected} flag(s), but "
                        f"{len(messages)} message(s) were found at workbook "
                        f"row {position}.")
            for message in messages:
                previous_form = _previous_form_from_message(
                    message, variable_to_form, position)
                form_occurrences.extend(dict.fromkeys(
                    (current_form, previous_form)))

    working = pd.DataFrame({"form": form_occurrences})
    return (
        working.value_counts("form", sort=False)
        .rename("flag_count")
        .reset_index()
        .sort_values(["flag_count", "form"], ascending=[False, True],
                     kind="stable")
        .reset_index(drop=True)
    )


def save_bar_graph(summary: pd.DataFrame, output_path: Path,
                   title: str = "Date Report flags involving each form") -> None:
    """Save an accessible horizontal bar graph of the per-form counts."""
    form_count = len(summary)
    figure_height = max(4.0, 1.6 + 0.46 * max(form_count, 1))
    figure, axis = plt.subplots(figsize=(12, figure_height))

    if summary.empty:
        axis.text(
            0.5, 0.5, "No Date Report flags found",
            ha="center", va="center", transform=axis.transAxes, fontsize=13,
        )
        axis.set_axis_off()
    else:
        plotted = summary.iloc[::-1]
        labels = [
            "\n".join(textwrap.wrap(form, width=42, break_long_words=True))
            for form in plotted["form"]
        ]
        bars = axis.barh(
            labels, plotted["flag_count"], color="#3B6EA8", edgecolor="none")
        maximum = max(int(plotted["flag_count"].max()), 1)
        axis.set_xlim(0, maximum * 1.14)
        axis.set_xlabel("Number of date flags involving form")
        axis.grid(axis="x", color="#D0D0D0", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for bar, value in zip(bars, plotted["flag_count"]):
            axis.text(
                bar.get_width() + maximum * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{int(value):,}",
                va="center", ha="left", fontsize=9,
            )

    axis.set_title(title, pad=12)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count both distinct forms involved in every Date Report flag and "
            "save a CSV plus horizontal bar graph."),
    )
    parser.add_argument(
        "input_file",
        help=(
            "QC tracker workbook, Date Report CSV/TSV, or raw flag Parquet "
            "file."),
    )
    parser.add_argument(
        "--sheet", default=DEFAULT_SHEET,
        help=f"Excel sheet to read (default: {DEFAULT_SHEET!r}).",
    )
    parser.add_argument(
        "--network", choices=("PRONET", "PRESCIENT"),
        help="Optionally restrict a raw combined flag table to one network.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: the input file's directory).",
    )
    parser.add_argument(
        "--important-form-vars",
        help=(
            "Path to important_form_vars.json. Needed for tracker workbooks; "
            "by default the program searches nearby dependencies folders."),
    )
    parser.add_argument(
        "--title", default="Date Report flags involving each form",
        help="Title shown above the graph.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input_file).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else input_path.parent
    )

    try:
        frame = read_date_report(
            input_path, sheet_name=args.sheet, network=args.network)
        variable_to_form = None
        if (not frame.empty
                and _find_column(frame, ("affected_forms",)) is None):
            form_map_path = _resolve_form_map_path(args.important_form_vars)
            variable_to_form = load_interview_date_form_map(form_map_path)
        summary = count_flags_by_form(frame, variable_to_form)
        source_flag_count = count_individual_flags(frame)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_stem = f"{input_path.stem}_date_flags_by_form"
        csv_path = output_dir / f"{output_stem}.csv"
        graph_path = output_dir / f"{output_stem}.png"
        summary.rename(columns={
            "form": "Form", "flag_count": "Flag Count",
        }).to_csv(csv_path, index=False)
        save_bar_graph(summary, graph_path, title=args.title)
    except (FileNotFoundError, DateReportSummaryError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Processed {source_flag_count:,} date flag(s); counted "
          f"{int(summary['flag_count'].sum()):,} form involvement(s) across "
          f"{len(summary):,} form(s).")
    print(f"CSV: {csv_path}")
    print(f"Bar graph: {graph_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
