"""Collect participants whose current pharm entries contain only code 999.

The current pharmaceutical treatment instrument is split across two floating
forms: medication slots 1-25 and 26-50.  This program reads the canonical
floating-form combined CSV for each network, considers only
``chrpharm_med{N}_name`` (never ``*_name_past``), and writes one row per
participant when:

* at least one current medication-name field is exactly code 999 after
  normalizing numeric/string representations such as ``999.0``; and
* no other nonblank value is present in any current medication-name field.

Other sentinels (for example 777, 888, or -9) are deliberately disqualifying:
they do not establish that the participant has no other medication.

Combined exports may legitimately omit current-medication columns.  Missing
slots are reported as a warning but do not stop the analysis; every available
current medication-name field is still checked.

Run from the project root with::

    python analyze_dataset/collect_current_pharm_999_only.py

By default, input and output directories come from ``config.json``.  Use
``--help`` for path and network overrides.

Form completion fields are not used as a gate: the current floating pharm
instrument is cumulative, and this utility classifies the medication values
that are actually present in its two form blocks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, MutableMapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = "/home/ob001/formqc_new_copy_6_24/form_qc_trackers/config.json"
DEFAULT_OUTPUT_FILENAME = "current_pharm_999_only_participants.csv"

NETWORKS = ("PRONET", "PRESCIENT")
NETWORK_FILE_LABELS = {
    "PRONET": "ProNET",
    "PRESCIENT": "PRESCIENT",
}

CURRENT_MEDICATION_NAME_RE = re.compile(
    r"^chrpharm_med(?P<med_number>\d+)_name$"
)
EXPECTED_CURRENT_MEDICATION_COLUMNS = tuple(
    f"chrpharm_med{med_number}_name" for med_number in range(1, 51)
)

OUTPUT_COLUMNS = (
    "subjectid",
    "network",
    "medication_code",
    "n_999_entries",
    "medication_fields",
    "source_file",
)

_BLANK_TEXT_VALUES = frozenset({"", "nan", "nat", "none", "null", "<na>"})


def normalize_med_code(value) -> str:
    """Return a stable string representation of one medication code.

    Blank/NA values become ``""``.  Integral numeric representations are
    normalized without a decimal suffix, so 999, 999.0, and ``" 999.0 "``
    all become ``"999"``.  Non-numeric text is stripped but otherwise left
    unchanged; this prevents values such as ``"1999"`` or ``"999mg"`` from
    being mistaken for code 999.
    """

    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        # The analysis passes scalar cells, but treating an unusual object as
        # nonblank is safer than silently dropping it.
        pass

    text = str(value).strip()
    if text.lower() in _BLANK_TEXT_VALUES:
        return ""

    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return text

    if number.is_finite() and number == number.to_integral_value():
        return str(int(number))
    return text


def current_medication_name_columns(columns: Iterable[object]) -> list[str]:
    """Return current medication-name columns in medication-slot order.

    The two current forms presently define slots 1 through 50.  Exact
    matching excludes past-form fields, dosage fields, and similarly named
    columns.
    """

    matched = []
    for column in columns:
        column_name = str(column)
        match = CURRENT_MEDICATION_NAME_RE.fullmatch(column_name)
        if match is None:
            continue
        med_number = int(match.group("med_number"))
        if 1 <= med_number <= 50:
            matched.append((med_number, column_name))
    return [column_name for _, column_name in sorted(matched)]


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame(columns=list(OUTPUT_COLUMNS))


def _normalize_subjectid(value) -> str:
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in _BLANK_TEXT_VALUES else text


def _accumulate_frame(
    combined_df: pd.DataFrame,
    *,
    network: str,
    source_file: str,
    participant_state: MutableMapping[tuple[str, str], dict],
) -> None:
    if "subjectid" not in combined_df.columns:
        raise ValueError("combined data must contain a 'subjectid' column")

    medication_columns = current_medication_name_columns(combined_df.columns)
    if not medication_columns:
        return

    selected = combined_df.loc[:, ["subjectid", *medication_columns]]
    for values in selected.itertuples(index=False, name=None):
        subjectid = _normalize_subjectid(values[0])
        if not subjectid:
            continue

        key = (network, subjectid)
        state = participant_state.setdefault(
            key,
            {
                "codes": set(),
                "n_999_entries": 0,
                "medication_fields": set(),
                "source_files": set(),
            },
        )
        if source_file:
            state["source_files"].add(str(source_file))

        for medication_field, raw_value in zip(medication_columns, values[1:]):
            code = normalize_med_code(raw_value)
            if not code:
                continue
            state["codes"].add(code)
            if code == "999":
                state["n_999_entries"] += 1
                state["medication_fields"].add(medication_field)


def _format_participant_state(participant_state: MutableMapping) -> pd.DataFrame:
    rows = []
    for (network, subjectid), state in participant_state.items():
        # Exact set equality simultaneously requires a 999 observation and
        # rejects every real, unknown, no-information, or missing-code value.
        if state["codes"] != {"999"}:
            continue
        rows.append(
            {
                "subjectid": subjectid,
                "network": network,
                "medication_code": "999",
                "n_999_entries": state["n_999_entries"],
                "medication_fields": ";".join(
                    sorted(
                        state["medication_fields"],
                        key=lambda name: int(
                            CURRENT_MEDICATION_NAME_RE.fullmatch(name).group(
                                "med_number"
                            )
                        ),
                    )
                ),
                "source_file": ";".join(sorted(state["source_files"])),
            }
        )

    if not rows:
        return _empty_output()
    return (
        pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
        .sort_values(["network", "subjectid"], kind="stable")
        .reset_index(drop=True)
    )


def collect_999_only_participants(
    combined_df: pd.DataFrame,
    network: str,
    source_file: str = "",
) -> pd.DataFrame:
    """Collect qualifying participants from one combined data frame.

    Multiple rows for the same participant are aggregated before filtering,
    so a 999-only duplicate row cannot hide a real medication on another row.
    """

    normalized_network = str(network).strip().upper()
    if not normalized_network:
        raise ValueError("network must be a nonblank value")

    participant_state: dict[tuple[str, str], dict] = {}
    _accumulate_frame(
        combined_df,
        network=normalized_network,
        source_file=str(source_file),
        participant_state=participant_state,
    )
    return _format_participant_state(participant_state)


class CurrentPharm999OnlyCollector:
    """Sweep canonical current-pharm floating CSVs for both networks."""

    def __init__(
        self,
        combined_csv_path: str | Path,
        networks: Iterable[str] = NETWORKS,
        *,
        require_all_medication_columns: bool = False,
        require_all_networks: bool = True,
    ) -> None:
        self.combined_csv_path = Path(combined_csv_path)
        self.networks = tuple(
            dict.fromkeys(str(network).strip().upper() for network in networks)
        )
        unsupported = sorted(set(self.networks) - set(NETWORKS))
        if unsupported:
            raise ValueError(f"unsupported network(s): {', '.join(unsupported)}")
        self.require_all_medication_columns = require_all_medication_columns
        self.require_all_networks = require_all_networks

        self.n_files_read = 0
        self.n_rows_scanned = 0
        self.n_participants_collected = 0
        self.unavailable_networks: list[str] = []

    def input_path_for_network(self, network: str) -> Path:
        network_label = NETWORK_FILE_LABELS[network]
        return self.combined_csv_path / (
            "AMPSCZ-combined-redcap_floating_forms_"
            f"{network_label}-day1to1.csv"
        )

    @staticmethod
    def _read_input(csv_path: Path) -> pd.DataFrame:
        return pd.read_csv(
            csv_path,
            dtype=str,
            keep_default_na=False,
            usecols=lambda column: (
                column == "subjectid"
                or CURRENT_MEDICATION_NAME_RE.fullmatch(str(column)) is not None
            ),
            on_bad_lines="error",
        )

    def collect(self) -> pd.DataFrame:
        participant_state: dict[tuple[str, str], dict] = {}
        self.n_files_read = 0
        self.n_rows_scanned = 0
        self.unavailable_networks = []

        expected_columns = set(EXPECTED_CURRENT_MEDICATION_COLUMNS)
        for network in self.networks:
            csv_path = self.input_path_for_network(network)
            try:
                combined_df = self._read_input(csv_path)
            except FileNotFoundError:
                self.unavailable_networks.append(network)
                print(
                    f"[current_pharm_999_only] input not found for {network}: "
                    f"{csv_path}. Skipping.",
                    file=sys.stderr,
                )
                continue
            except pd.errors.EmptyDataError:
                self.unavailable_networks.append(network)
                print(
                    f"[current_pharm_999_only] input is empty for {network}: "
                    f"{csv_path}. Skipping.",
                    file=sys.stderr,
                )
                continue

            if "subjectid" not in combined_df.columns:
                self.unavailable_networks.append(network)
                print(
                    f"[current_pharm_999_only] {csv_path} has no subjectid "
                    "column. Skipping.",
                    file=sys.stderr,
                )
                continue
            if combined_df.empty:
                self.unavailable_networks.append(network)
                print(
                    f"[current_pharm_999_only] {csv_path} has headers but no "
                    "participant rows. Skipping.",
                    file=sys.stderr,
                )
                continue

            found_columns = set(current_medication_name_columns(combined_df.columns))
            missing_columns = expected_columns - found_columns
            if not found_columns:
                self.unavailable_networks.append(network)
                print(
                    f"[current_pharm_999_only] {csv_path} has no current "
                    "medication-name columns. Skipping.",
                    file=sys.stderr,
                )
                continue
            if missing_columns:
                message = (
                    f"[current_pharm_999_only] {csv_path} is missing "
                    f"{len(missing_columns)} of 50 expected current medication-name "
                    "columns."
                )
                if self.require_all_medication_columns:
                    raise ValueError(message)
                print(f"{message} Using the available columns.", file=sys.stderr)

            self.n_files_read += 1
            self.n_rows_scanned += len(combined_df)
            _accumulate_frame(
                combined_df,
                network=network,
                source_file=str(csv_path),
                participant_state=participant_state,
            )

        if self.require_all_networks and self.unavailable_networks:
            raise ValueError(
                "no usable current-pharm floating input for selected "
                "network(s): " + ", ".join(self.unavailable_networks)
            )

        output = _format_participant_state(participant_state)
        self.n_participants_collected = len(output)
        return output

    def run_script(self, output_path: str | Path) -> pd.DataFrame:
        output = self.collect()
        if self.n_files_read == 0:
            raise ValueError(
                "no readable current-pharm floating files were found; "
                "refusing to write an empty result"
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(
            f".{output_path.name}.{os.getpid()}.tmp"
        )
        try:
            output.to_csv(temp_path, index=False)
            os.replace(temp_path, output_path)
        finally:
            # If serialization fails, preserve any prior valid output and
            # remove only this run's incomplete temporary file.
            temp_path.unlink(missing_ok=True)
        print(
            f"[current_pharm_999_only] scanned {self.n_rows_scanned} row(s) "
            f"from {self.n_files_read} file(s); wrote "
            f"{self.n_participants_collected} participant(s) to {output_path}"
        )
        return output


def _load_config(config_path: Path) -> dict:
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError as exc:
        raise ValueError(f"config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in config file {config_path}: {exc}") from exc

    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"config file {config_path} is missing the 'paths' object")
    return config


def _resolve_config_path(raw_path: str, config_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else config_path.parent / path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"configuration JSON (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--combined-csv-path",
        type=Path,
        help="override config paths.combined_csv_path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "output CSV (default: paths.output_path/"
            f"{DEFAULT_OUTPUT_FILENAME})"
        ),
    )
    parser.add_argument(
        "--network",
        nargs="+",
        choices=NETWORKS,
        default=list(NETWORKS),
        help="network(s) to scan (default: both)",
    )
    parser.add_argument(
        "--require-all-50-columns",
        action="store_true",
        help=(
            "fail when a floating CSV lacks one or more med1-med50 name "
            "fields (default: warn and use the available fields)"
        ),
    )
    parser.add_argument(
        "--allow-missing-networks",
        action="store_true",
        help=(
            "write results when a selected network file is unavailable "
            "(default: fail closed)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()

    try:
        config = _load_config(config_path)
        paths = config["paths"]

        if args.combined_csv_path is not None:
            combined_csv_path = args.combined_csv_path.expanduser()
        else:
            raw_combined_path = paths.get("combined_csv_path")
            if not raw_combined_path:
                raise ValueError(
                    f"config file {config_path} is missing paths.combined_csv_path"
                )
            combined_csv_path = _resolve_config_path(
                raw_combined_path, config_path
            )

        if args.output is not None:
            output_path = args.output.expanduser()
        else:
            raw_output_path = paths.get("output_path")
            if not raw_output_path:
                raise ValueError(
                    f"config file {config_path} is missing paths.output_path"
                )
            output_path = (
                _resolve_config_path(raw_output_path, config_path)
                / DEFAULT_OUTPUT_FILENAME
            )

        collector = CurrentPharm999OnlyCollector(
            combined_csv_path,
            networks=args.network,
            require_all_medication_columns=args.require_all_50_columns,
            require_all_networks=not args.allow_missing_networks,
        )
        collector.run_script(output_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
