"""Recover explicitly manually resolved flags from retained Dropbox revisions.

Run from the repository root::

    python main/analyze_flags/extract_manually_resolved_flags.py
    python main/analyze_flags/extract_manually_resolved_flags.py --network PRONET

Uses config.json and dependencies_path/dropbox_credentials.json, like the other
history scripts, without starting the QC pipeline or requiring its input CSVs.
Defaults to both networks' combined V1/V2 trackers, including PRONET V2 under
refactoring_tests. --tracker NETWORK=/absolute/dropbox/path.xlsx (repeatable)
replaces those defaults, for example to recover a site-specific tracker.

Every retained revision and authoritative report sheet is checked, with no date
sampling. Only an explicit manual marker qualifies; Date Resolved and missing
rows are not evidence of manual resolution. Display cells follow the tracker's
nonblank-text convention (even "no" is a marker); raw manually_resolved boolean
columns use boolean semantics. Blank, NaN and [clear] are not manual marks.

The CSV has one row per network/subject/normalized timepoint/form/variable/exact
message/check ID. It retains distinct messages instead of collapsing values into
templates. Multi-flag rows are split with the shared transport parser; form-level
or unparseable messages are retained with a blank variable. Evidence columns are
JSON arrays. First/last timestamps mean observations of a manual mark in Dropbox,
not the exact time a person resolved it; revision_count counts marked snapshots,
not separate human decisions. A later unmarked revision does not remove a flag.

Output defaults to <pipeline output>/combined_outputs/analyze_flags/
manually_resolved_flag_history.csv. Dropbox is read-only. Any failed download,
malformed report, or incomplete pagination aborts without replacing a previous
CSV. Coverage is limited to history that Dropbox still retains at these paths.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import dropbox
import pandas as pd

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(1, str(MAIN_DIR))

from analyze_flags.canonicalize import (  # noqa: E402
    normalize_general_flag,
    normalize_timepoint,
    split_entry,
    split_specific_flag_entries,
    unescape_specific_flag_text,
)
from analyze_flags.paths import (  # noqa: E402
    analyze_flags_artifact_dir,
    pipeline_output_path_from_config,
)

NETWORKS = ("PRESCIENT", "PRONET")
OUTPUT_BASENAME = "manually_resolved_flag_history.csv"
# Same generated review surfaces as extract_blank_flags and estimate_resolved.
REPORT_SHEETS = frozenset({
    "Main Report", "Secondary Report", "Date Report", "Cross Checks",
    "Proposed Checks", "Medication Flags", "Non Team Forms", "Cognition Report",
    "Blood Report", "Fluids Report", "Scid Report", "MRI Report", "EEG Report",
    "Digital Report", "Incomplete Forms", "Missingness Report", "Conversion Report",
})
ALIASES = {
    "subject": ("Subject", "Participant", "ID", "subject"),
    "timepoint": ("Timepoint", "Timepoint of Error", "displayed_timepoint"),
    "form": ("General Flag", "Form", "displayed_form"),
    "flags": ("Specific Flags", "Flags", "error_message"),
    "manual": ("Manually Marked as Resolved", "Manually Resolved", "manually_resolved"),
    "comments": ("Additional Comments", "Site Comments", "Network Comments", "Comments", "comments"),
    "date_resolved": ("Date Resolved", "date_resolved"),
    "check_id": ("Check ID", "check_id"),
}
KEY_COLUMNS = ("network", "subject", "timepoint", "form", "variable", "message", "check_id")
EVIDENCE_COLUMNS = {
    "raw_entries": "raw_entry",
    "manual_markers": "manual_marker",
    "comments": "comments",
    "dates_resolved": "date_resolved",
    "original_forms": "original_form",
    "original_timepoints": "original_timepoint",
    "source_paths": "source_path",
    "sheets": "sheet",
}
OUTPUT_COLUMNS = [
    *KEY_COLUMNS, "first_seen_manual_utc", "last_seen_manual_utc",
    "first_revision", "last_revision", "first_source", "last_source",
    "revision_count", *EVIDENCE_COLUMNS,
]


@dataclass(frozen=True)
class TrackerSource:
    network: str
    path: str


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def _coalesce(row, field):
    return next((_text(row[c]) for c in ALIASES[field]
                 if c in row and _text(row[c])), "")


def _manual_marker(row):
    for column in ALIASES["manual"]:
        value = _text(row.get(column, ""))
        if not value or value.casefold() == "nan":
            continue
        if value.casefold() == "[clear]":
            return ""
        if column == "manually_resolved" and value.casefold() in {
            "false", "0", "0.0", "no", "f", "none", "nat",
        }:
            return ""
        return value
    return ""


def collect_manual_flags(sheets, source):
    """Extract marked entries, rejecting unreadable authoritative schemas."""
    records = []
    recognized = 0
    for sheet, frame in sheets.items():
        if sheet not in REPORT_SHEETS:
            continue
        recognized += 1
        missing = [field for field in ("subject", "timepoint", "form", "flags", "manual")
                   if not any(c in frame.columns for c in ALIASES[field])]
        if missing:
            raise ValueError(f"{source.path}: {sheet!r} lacks columns for {missing}")
        for row_number, row in enumerate(frame.to_dict("records"), start=2):
            marker = _manual_marker(row)
            if not marker:
                continue
            subject = _coalesce(row, "subject")
            if not subject:
                raise ValueError(f"{source.path}: {sheet!r} row {row_number} is marked but has no subject")
            form = _coalesce(row, "form")
            timepoint = _coalesce(row, "timepoint")
            # Preserve even a manually marked row with no specific flag text.
            for entry in split_specific_flag_entries(_coalesce(row, "flags")) or [""]:
                variable, message = split_entry(entry)
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable):
                    variable, message = "", unescape_specific_flag_text(entry)
                records.append({
                    "network": source.network, "subject": subject,
                    "timepoint": normalize_timepoint(timepoint),
                    "form": normalize_general_flag(form),
                    "variable": variable, "message": message,
                    "check_id": _coalesce(row, "check_id"),
                    "raw_entry": entry, "manual_marker": marker,
                    "comments": _coalesce(row, "comments"),
                    "date_resolved": _coalesce(row, "date_resolved"),
                    "original_form": form, "original_timepoint": timepoint,
                    "source_path": source.path, "sheet": sheet,
                })
    if not recognized:
        raise ValueError(f"{source.path}: no recognized report sheets")
    return records


def _requester(dbx):
    # Older SDKs expose the raw requester on a transport instead of the client.
    for candidate in [dbx] + [getattr(dbx, name, None) for name in (
        "_transport", "_client", "_request", "_session",
        "_Dropbox__transport", "_Dropbox__client",
    )]:
        if callable(getattr(candidate, "request_json_object", None)):
            return candidate
    raise RuntimeError("Dropbox client does not expose request_json_object; upgrade the dropbox package")


def iter_revisions(dbx, path, page_size=100):
    """Follow the same raw before_rev pagination as the other history scripts."""
    # These HTTP fields are not exposed by the pinned Dropbox 12.0.2 SDK:
    # https://docs.dropboxapi.com/dropbox-api/api-reference/user-endpoints/files/list-revisions
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")
    if hasattr(dbx, "check_and_refresh_access_token"):
        dbx.check_and_refresh_access_token()
    requester = _requester(dbx)
    auth = getattr(dropbox.dropbox_client, "USER_AUTH", None)
    if auth is None:
        auth = getattr(getattr(dropbox.dropbox_client, "AuthType", None), "USER", "user")
    before_rev = None
    seen = set()
    while True:
        request = {"path": path, "mode": "path", "limit": page_size}
        if before_rev:
            request["before_rev"] = before_rev
        data = requester.request_json_object(
            "api", "files/list_revisions", "rpc", request, auth, None,
        )
        # Some SDK versions also return the HTTP response object.
        if isinstance(data, tuple):
            data = data[0]
        entries = data.get("entries", [])
        has_more = data.get("has_more", len(entries) == page_size)
        if not entries:
            if has_more or not seen:
                raise RuntimeError(f"{path}: empty/incomplete revision listing")
            return
        new_entries = [entry for entry in entries if entry["rev"] not in seen]
        if not new_entries:
            raise RuntimeError(f"{path}: revision pagination made no progress")
        for entry in new_entries:
            if entry["rev"] in seen:
                continue
            seen.add(entry["rev"])
            yield entry
        if not has_more:
            return
        cursor = entries[-1]["rev"]
        if cursor == before_rev:
            raise RuntimeError(f"{path}: revision pagination cursor did not advance")
        before_rev = cursor


def read_tracker_workbook(content):
    """Keep identifiers as strings and skip unrelated tabs when parsing Excel."""
    engines = ["openpyxl"]
    try:
        import python_calamine  # noqa: F401
        if tuple(int(x) for x in pd.__version__.split(".")[:2]) >= (2, 2):
            engines.insert(0, "calamine")
    except (ImportError, ValueError):
        pass
    for engine in engines:
        try:
            with pd.ExcelFile(BytesIO(content), engine=engine) as workbook:
                return {sheet: workbook.parse(sheet, dtype=str, keep_default_na=False)
                        for sheet in workbook.sheet_names if sheet in REPORT_SHEETS}
        except Exception:
            if engine == "openpyxl":
                raise
            print("Calamine could not read this revision; retrying with openpyxl.")


def _write_csv_atomic(frame, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp",
                                     dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def recover_flags(dbx, sources, output_path):
    """Publish the deduplicated list only after every source walk succeeds."""
    sources = list(dict.fromkeys(sources))
    if not sources:
        raise ValueError("At least one tracker source is required")
    recovered = {}
    for source in sources:
        print(f"Scanning every retained revision: {source.network} {source.path}", flush=True)
        # Reuse consecutive identical workbooks, but still count and date every
        # revision. Keep only one parsed snapshot to bound cache memory.
        last_hash, last_records = None, None
        for revision_number, revision in enumerate(iter_revisions(dbx, source.path), start=1):
            rev = revision["rev"]
            try:
                when = datetime.fromisoformat(revision["server_modified"].replace("Z", "+00:00"))
                when = when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)
                content_hash = revision.get("content_hash")
                if not content_hash or content_hash != last_hash:
                    _, response = dbx.files_download(path=source.path, rev=rev)
                    try:
                        last_records = collect_manual_flags(read_tracker_workbook(response.content), source)
                    finally:
                        if callable(getattr(response, "close", None)):
                            response.close()
                    last_hash = content_hash
                revision_keys = set()
                for record in last_records:
                    key = tuple(record[column] for column in KEY_COLUMNS)
                    if key not in recovered:
                        recovered[key] = {
                            **dict(zip(KEY_COLUMNS, key)),
                            "first_seen_manual_utc": when, "last_seen_manual_utc": when,
                            "first_revision": rev, "last_revision": rev,
                            "first_source": source.path, "last_source": source.path,
                            "revision_count": 0,
                            **{column: set() for column in EVIDENCE_COLUMNS},
                        }
                    saved = recovered[key]
                    for bound, comparison in (("first", when < saved["first_seen_manual_utc"]),
                                              ("last", when > saved["last_seen_manual_utc"])):
                        if comparison:
                            saved[f"{bound}_seen_manual_utc"] = when
                            saved[f"{bound}_revision"] = rev
                            saved[f"{bound}_source"] = source.path
                    if key not in revision_keys:
                        saved["revision_count"] += 1
                        revision_keys.add(key)
                    for column, field in EVIDENCE_COLUMNS.items():
                        if record[field]:
                            saved[column].add(record[field])
            except Exception as exc:
                raise RuntimeError(f"{source.path}: could not recover revision {rev}: {exc}") from exc
            print(f"  Revision {revision_number}: {rev} ({when.isoformat()}); "
                  f"{len(recovered)} unique marked flags so far", flush=True)
    rows = []
    for key in sorted(recovered):
        row = recovered[key]
        for column in ("first_seen_manual_utc", "last_seen_manual_utc"):
            row[column] = row[column].isoformat()
        for column in EVIDENCE_COLUMNS:
            row[column] = json.dumps(sorted(row[column]), ensure_ascii=False)
        rows.append(row)
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    _write_csv_atomic(frame, output_path)
    print(f"Wrote {len(frame)} unique manually resolved flags to {output_path}")
    return frame


def default_sources(config, networks=NETWORKS):
    testing = config.get("testing_enabled") == "True"
    production = "/Apps/Automated QC Trackers/"
    test_base = production + "refactoring_tests/"
    base = test_base if testing else production
    sources = []
    for network in networks:
        sources.append(TrackerSource(network, f"{base}{network}/combined/{network}_Output.xlsx"))
        v2_base = test_base if network == "PRONET" else base
        sources.append(TrackerSource(network, f"{v2_base}{network}/combined/{network}_Output_V2.xlsx"))
    return sources


def _tracker_argument(value):
    network, separator, path = value.partition("=")
    network = network.upper()
    if not separator or network not in NETWORKS or not path.startswith("/"):
        raise argparse.ArgumentTypeError("Use PRESCIENT=/path/file.xlsx or PRONET=/path/file.xlsx")
    return TrackerSource(network, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=MAIN_DIR.parent / "config.json")
    parser.add_argument("--output", type=Path, help="Output CSV; defaults to the configured analyze_flags artifacts folder")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--network", nargs="+", choices=NETWORKS, default=NETWORKS)
    selection.add_argument("--tracker", action="append", type=_tracker_argument,
                           help="Explicit NETWORK=/Dropbox/path.xlsx; repeat to scan multiple files")
    args = parser.parse_args(argv)
    try:
        with args.config.open(encoding="utf-8-sig") as stream:
            config = json.load(stream)
        output = args.output or Path(analyze_flags_artifact_dir(pipeline_output_path_from_config(config))) / OUTPUT_BASENAME
        sources = args.tracker or default_sources(config, args.network)
        credentials_path = Path(config["paths"]["dependencies_path"]) / "dropbox_credentials.json"
        with credentials_path.open(encoding="utf-8-sig") as stream:
            credentials = json.load(stream)
        # Same refresh-token authentication as Utils.collect_dropbox_credentials;
        # avoid Utils initialization, which loads unrelated QC dependencies.
        with dropbox.Dropbox(app_key=credentials["app_key"], app_secret=credentials["app_secret"],
                             oauth2_refresh_token=credentials["refresh_token"]) as dbx:
            recover_flags(dbx, sources, output)
    except Exception as exc:
        print(f"Recovery failed; existing output was not replaced. {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
