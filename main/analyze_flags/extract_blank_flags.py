"""
Walk every Dropbox revision of the combined network tracker(s) and emit
a deduplicated CSV of every (subject, variable, timepoint) that was
*ever* flagged as blank — the current "Variable is blank." wording or
the pre-V2 "Value is empty" wording — regardless of whether the flag
was later resolved.

Both the V2 tracker (``{network}_Output_V2.xlsx``) and the original
pre-V2 tracker (``{network}_Output.xlsx``) are walked. The two files
use different column schemas and slightly different timepoint
vocabularies ('baseln' / 'screen' in the original); detection is
per-revision and timepoints are normalized at collection time so the
union dedupes correctly.

Mirrors the iteration / pagination structure of analyze_flags.estimate_resolved
so the same Dropbox auth + revision walk works here. Two deliberate
deviations from that script:
  * No ``sample_every_days`` skip — full coverage, so a flag that
    appeared and was resolved between two sampled revisions is still
    captured.
  * No "drop resolved rows" filter — "ever flagged" includes resolved
    ones.

Optional comparison mode (``compare_to_current=True``): after the
revision walk, look up each (subject, variable, timepoint) in the
current AMPSCZ-combined-redcap CSVs and emit a status column saying
whether the value is now filled, still blank, a missing-code, or
unresolvable (subject / variable / CSV missing). The lookup uses the
same path convention as qc_forms_main.py:102.
"""

import pandas as pd
import os
import sys
import json
import tempfile
import dropbox
from collections import defaultdict
from io import BytesIO
from datetime import datetime, timedelta, timezone

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)
from utils.utils import Utils
from analyze_flags.canonicalize import (
    normalize_timepoint,
    split_entry,
    split_specific_flag_entries,
)
from analyze_flags.paths import (
    ensure_analyze_flags_artifact_dir,
    pipeline_output_path_from_config,
)


class BlankFlagExtractor:
    # Exact blank-value message contracts, compared casefolded with the
    # trailing period stripped. "Variable is blank." is the current producer
    # wording (qc_types/general_checks.py); "Value is empty" is the dominant
    # pre-V2 spelling of the same assertion in the walked tracker history.
    # Substring matching stays unsafe because malformed free-text values can
    # themselves contain either phrase.
    BLANK_MESSAGES = frozenset({"variable is blank", "value is empty"})

    OUTPUT_BASENAME = "blank_flag_history.csv"

    NETWORKS = ['PRESCIENT', 'PRONET']

    # Only generated review surfaces are authoritative. A backup/operator tab
    # can retain an old blank flag indefinitely and must not enter history just
    # because it happens to have tracker-like columns.
    AUTHORITATIVE_REPORT_SHEETS = frozenset({
        'Main Report', 'Secondary Report', 'Date Report', 'Cross Checks',
        'Proposed Checks', 'Medication Flags', 'Non Team Forms',
        'Cognition Report', 'Blood Report', 'Fluids Report', 'Scid Report',
        'MRI Report', 'EEG Report', 'Digital Report', 'Incomplete Forms',
        'Missingness Report', 'Conversion Report',
    })

    # PRONET's V2 tracker still lives under refactoring_tests/ — it has
    # not been promoted to the production path yet. Its pre-V2 tracker
    # (and both PRESCIENT trackers) are at the production base.
    PRONET_V2_BASE_OVERRIDE = '/Apps/Automated QC Trackers/refactoring_tests/'

    def __init__(self, compare_to_current=True):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json', 'r') as file:
            self.config_info = json.load(file)
        self.output_path = pipeline_output_path_from_config(self.config_info)
        if self.config_info["testing_enabled"] == "True":
            self.dropbox_path = '/Apps/Automated QC Trackers/refactoring_tests/'
        else:
            self.dropbox_path = '/Apps/Automated QC Trackers/'
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.analyze_flags_dir = ensure_analyze_flags_artifact_dir(self.output_path)
        self.compare_to_current = compare_to_current
        # Tuple shape: (network, subject, variable, timepoint). Network
        # is tracked even in non-compare mode so the comparison branch
        # can build the right combined-CSV path when enabled.
        self.blank_records = set()
        # Excel engine resolved lazily on the first read (calamine if the
        # package is installed, else openpyxl) and cached here. See
        # _read_tracker_excel.
        self._excel_engine = None

    def run_script(self):
        if not self.loop_dropbox():
            print(
                "WARNING: blank-flag history was not refreshed because at "
                "least one configured tracker walk was incomplete. Preserving "
                "the last-good output."
            )
            return False
        if self.compare_to_current:
            comparison_rows = self.compare_against_combined_csvs()
            self.write_comparison_output(comparison_rows)
        else:
            self.write_history_output()
        return True

    def loop_dropbox(self):
        staged_records = set()
        complete = True
        for network_name in self.NETWORKS:
            for base, basename in self._tracker_paths(network_name):
                combined_output = f'{base}{network_name}/combined/{basename}'
                source_records = set()
                try:
                    source_complete = self.extract_blank_flags(
                        combined_output,
                        network_name,
                        days_back=None,
                        target_records=source_records,
                    )
                except Exception as e:
                    print(f"WARNING: failed walking revisions for {combined_output}: "
                          f"{type(e).__name__}: {str(e)[:200]}")
                    complete = False
                    continue
                if not source_complete:
                    print(
                        f"WARNING: rejecting partial revision walk for "
                        f"{combined_output}."
                    )
                    complete = False
                    continue
                staged_records.update(source_records)

        if complete:
            # Replace rather than union so a successful authoritative empty run
            # can clear records which no longer exist. On any failure, leave
            # the in-memory and on-disk last-good result untouched.
            self.blank_records = staged_records
        return complete

    def _tracker_paths(self, network_name):
        """(base, basename) pairs to walk for a given network.

        PRONET's V2 file lives only under refactoring_tests/ until it is
        promoted, so we override the base for that one combination. When
        testing_enabled is True we already point self.dropbox_path at
        refactoring_tests/, in which case the override is a no-op.
        """
        base = self.dropbox_path
        pairs = [(base, f'{network_name}_Output.xlsx')]
        v2_base = base
        if network_name == 'PRONET' and self.config_info["testing_enabled"] != "True":
            v2_base = self.PRONET_V2_BASE_OVERRIDE
        pairs.append((v2_base, f'{network_name}_Output_V2.xlsx'))
        return pairs

    def extract_blank_flags(self, path, network_name, days_back=None,
                            page_limit=100, target_records=None):
        """
        Walk every revision of ``path`` (paging beyond 100 via before_rev)
        and add (subject, variable, timepoint) tuples to
        ``self.blank_records`` for every blank-variable flag observed.

        The raw-RPC plumbing is copied from estimate_resolved.py so the
        same older-SDK fallback behavior holds.
        """
        staged_records = set() if target_records is None else target_records
        dbx = self.utils.collect_dropbox_credentials()
        if hasattr(dbx, "check_and_refresh_access_token"):
            dbx.check_and_refresh_access_token()

        def find_requester(client):
            if hasattr(client, "request_json_object"):
                return client
            for attr in (
                "_transport", "_client", "_request", "_session",
                "_Dropbox__transport", "_Dropbox__client",
            ):
                obj = getattr(client, attr, None)
                if obj is not None and hasattr(obj, "request_json_object"):
                    return obj
            for name in dir(client):
                if name.startswith("__"):
                    continue
                try:
                    obj = getattr(client, name)
                except Exception:
                    continue
                if obj is not None and hasattr(obj, "request_json_object"):
                    return obj
            return None

        requester = find_requester(dbx)
        if requester is None:
            raise RuntimeError(
                "Couldn't find an internal requester on the Dropbox client "
                "with request_json_object(). Upgrade the `dropbox` package "
                "or call list_revisions with requests directly."
            )

        USER_AUTH = getattr(dropbox.dropbox_client, "USER_AUTH", None)
        if USER_AUTH is None:
            USER_AUTH = getattr(dropbox.dropbox_client, "AuthType", None)
            USER_AUTH = getattr(USER_AUTH, "USER", None) if USER_AUTH else None
        if USER_AUTH is None:
            USER_AUTH = "user"

        cutoff = None
        if days_back is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
            print(f"Retrieving revisions from the last {days_back} days (cutoff: {cutoff})")
        else:
            print("Retrieving all available revisions (no time limit)")

        before_rev = None
        has_more = True
        kept = 0
        skipped_dupes = 0
        # content_hash of every revision already parsed in THIS file's walk.
        # Scoped per (network, tracker) call so the per-network tag baked into
        # each tuple can never be lost to a cross-file hash collision.
        seen_hashes = set()
        oldest_revision = None
        newest_revision = None
        newest_processed = None
        complete = True

        new_cols = ["Subject", "Timepoint", "Specific Flags"]
        old_cols = ["Participant", "Timepoint", "Flags"]

        while has_more:
            req = {"path": path, "mode": "path", "limit": page_limit}
            if before_rev:
                req["before_rev"] = before_rev

            data = requester.request_json_object(
                "api",
                "files/list_revisions",
                "rpc",
                req,
                USER_AUTH,
                None,
            )

            entries = data.get("entries", [])
            has_more = bool(data.get("has_more", False))
            if not entries:
                break

            for entry in entries:
                rev = entry["rev"]
                when = datetime.fromisoformat(entry["server_modified"].replace("Z", "+00:00"))

                if oldest_revision is None or when < oldest_revision:
                    oldest_revision = when
                if newest_revision is None or when > newest_revision:
                    newest_revision = when

                if cutoff is not None and when < cutoff:
                    has_more = False
                    print(f"Reached cutoff date ({cutoff}), stopping retrieval")
                    break

                # Skip the download+parse for a content_hash already
                # successfully processed in this walk: byte-identical bytes
                # yield identical (subject, variable, timepoint) tuples, so the
                # output is unchanged. The hash is only recorded after a
                # successful parse (below), so a transient download/parse
                # failure on the first copy never suppresses an identical
                # later revision.
                content_hash = entry.get("content_hash")
                if content_hash is not None and content_hash in seen_hashes:
                    skipped_dupes += 1
                    continue

                try:
                    md2, resp = dbx.files_download(path=path, rev=rev)
                    workbook_sheets = self._read_tracker_excel(resp.content)
                    revision_records = set()
                    recognized_sheets = 0
                    malformed_sheets = []
                    for sheet_name, df in workbook_sheets.items():
                        if sheet_name not in self.AUTHORITATIVE_REPORT_SHEETS:
                            continue
                        if all(c in df.columns for c in new_cols):
                            subject_col, tp_col, flags_col = new_cols
                        elif all(c in df.columns for c in old_cols):
                            subject_col, tp_col, flags_col = old_cols
                        else:
                            malformed_sheets.append(sheet_name)
                            continue
                        recognized_sheets += 1
                        self._collect_blank_records(
                            df, subject_col, tp_col, flags_col, network_name,
                            target_records=revision_records,
                        )

                    if malformed_sheets or not recognized_sheets:
                        complete = False
                        detail = (
                            f"malformed authoritative sheets {malformed_sheets}"
                            if malformed_sheets else
                            "no recognized authoritative report sheets"
                        )
                        print(
                            f"WARNING: Skipping revision {rev} from {when} - "
                            f"{detail}."
                        )
                        continue

                    before = len(staged_records)
                    staged_records.update(revision_records)
                    added = len(staged_records) - before
                    if content_hash is not None:
                        seen_hashes.add(content_hash)
                    if newest_processed is None or when > newest_processed:
                        newest_processed = when

                    kept += 1
                    print(f"{path} - Revision {kept}: {rev} from {when} (+{added} new blank rows; total {len(staged_records)})")
                except Exception as e:
                    complete = False
                    print(f"WARNING: Skipping revision {rev} from {when} - file error: {type(e).__name__}: {str(e)[:100]}")
                    continue

            before_rev = entries[-1]["rev"]

        if oldest_revision and newest_revision:
            days_span = (newest_revision - oldest_revision).days
            print(f"Done with {network_name}: scanned {kept} revisions "
                  f"(skipped {skipped_dupes} duplicate-content) spanning {days_span} days "
                  f"({oldest_revision} → {newest_revision}); "
                  f"{len(staged_records)} unique (subject, variable, timepoint) so far.")
        else:
            print(f"Done with {network_name}: scanned {kept} revisions "
                  f"(skipped {skipped_dupes} duplicate-content); "
                  f"{len(staged_records)} unique (subject, variable, timepoint) so far.")

        if newest_revision is None or newest_processed is None:
            complete = False
        elif newest_processed < newest_revision:
            complete = False
            print(
                f"WARNING: newest listed revision of {path} "
                f"({newest_revision}) failed to process; the source is stale."
            )

        if target_records is None and complete:
            self.blank_records.update(staged_records)
        return complete

    def _collect_blank_records(self, df, subject_col, tp_col, flags_col,
                               network_name, target_records=None):
        """
        Parse the Specific Flags / Flags column, splitting per
        form_check.py:353 ("{var} : {message}") joined with " | "
        (create_trackers.py:342). Adds a tuple per blank-variable flag
        and returns how many *new* tuples were added in this call.
        """
        records = self.blank_records if target_records is None else target_records
        before = len(records)
        for row in df[[subject_col, tp_col, flags_col]].itertuples(index=False, name=None):
            subject, timepoint, flags = row
            subject_s = "" if pd.isna(subject) else str(subject).strip()
            if not subject_s:
                continue
            if not isinstance(flags, str) or not flags.strip():
                continue
            for entry in split_specific_flag_entries(flags):
                variable, message = split_entry(entry)
                if not variable:
                    continue
                base_message = message.strip()
                if base_message.startswith("["):
                    closing_bracket = base_message.find("]")
                    if closing_bracket >= 0:
                        base_message = base_message[closing_bracket + 1:].strip()
                normalized_message = base_message.rstrip(".").strip().casefold()
                if normalized_message in self.BLANK_MESSAGES:
                    records.add((
                        network_name,
                        subject_s,
                        variable,
                        normalize_timepoint(timepoint),
                    ))
        return len(records) - before

    def _read_tracker_excel(self, content):
        """Read every sheet of one tracker revision into a mapping.

        Prefers the Rust-backed ``calamine`` engine — typically several times
        faster than openpyxl on these multi-MB workbooks, and it releases the
        GIL — and falls back to openpyxl otherwise so the walk still runs
        everywhere. ``keep_default_na=False`` matches the original read so empty
        cells stay empty strings (not NaN); the two engines were verified
        equivalent on representative synthetic data under it. The engine is
        resolved once (see ``_resolve_excel_engine``) and cached on the instance.

        If calamine chokes on a single old/oddly-formatted revision that
        openpyxl can still read, that one file is retried with openpyxl (engine
        choice unchanged) so coverage is never worse than an openpyxl-only run.
        Re-parsing identical bytes can't diverge — the engines agree on any
        file both can parse — it only recovers a revision we'd otherwise lose.
        """
        if self._excel_engine is None:
            self._excel_engine = self._resolve_excel_engine()
        try:
            return pd.read_excel(
                BytesIO(content),
                keep_default_na=False,
                engine=self._excel_engine,
                sheet_name=None,
            )
        except Exception as e:
            if self._excel_engine == "openpyxl":
                raise
            # "Unknown engine: calamine" means calamine is fundamentally
            # unusable in this runtime (e.g. a pandas that slipped past the
            # version probe) — downgrade permanently so we don't retry-and-fail
            # on every revision. Any other error is just this one odd file:
            # keep calamine for the rest and retry only this file with openpyxl.
            if "Unknown engine" in str(e):
                print(f"NOTE: calamine engine unavailable ({type(e).__name__}: "
                      f"{str(e)[:80]}); switching to openpyxl for the rest of "
                      f"the run.")
                self._excel_engine = "openpyxl"
            else:
                print(f"NOTE: calamine failed to parse a revision "
                      f"({type(e).__name__}: {str(e)[:80]}); retrying with "
                      f"openpyxl")
            return pd.read_excel(
                BytesIO(content),
                keep_default_na=False,
                engine="openpyxl",
                sheet_name=None,
            )

    def _resolve_excel_engine(self):
        """Pick the Excel engine once. Prefer ``calamine``, but only when it is
        actually usable: ``python-calamine`` must be importable AND pandas must
        be new enough to know the engine (added in pandas 2.2). Checking the
        pandas version — not just the import — is what avoids selecting an
        engine that ``read_excel`` rejects with "Unknown engine: calamine" on
        every revision. Falls back to openpyxl with a one-line reason."""
        try:
            import python_calamine  # noqa: F401
        except ImportError:
            print("NOTE: python-calamine not installed; using openpyxl "
                  "(slower). Run `pip install python-calamine` to speed up the "
                  "revision walk.")
            return "openpyxl"
        try:
            major, minor = (int(x) for x in pd.__version__.split(".")[:2])
            supported = (major, minor) >= (2, 2)
        except (ValueError, AttributeError):
            supported = False
        if not supported:
            print(f"NOTE: pandas {pd.__version__} predates the calamine engine "
                  f"(needs >= 2.2); using openpyxl. Upgrade pandas in this "
                  f"environment to speed up the revision walk.")
            return "openpyxl"
        return "calamine"

    def write_history_output(self):
        """Just (subject, variable, timepoint) — drop network so the
        original 3-column ask is preserved when compare_to_current is off."""
        out_csv = os.path.join(self.analyze_flags_dir, self.OUTPUT_BASENAME)
        rows = sorted({(subject, variable, tp) for (_, subject, variable, tp) in self.blank_records})
        out_df = pd.DataFrame(rows, columns=["subject", "variable", "timepoint"])
        self._write_csv_atomic(out_df, out_csv)
        print(f"Wrote {len(out_df)} unique (subject, variable, timepoint) rows to: {out_csv}")

    def write_comparison_output(self, comparison_rows):
        out_csv = os.path.join(self.analyze_flags_dir, self.OUTPUT_BASENAME)
        out_df = pd.DataFrame(
            comparison_rows,
            columns=["network", "subject", "variable", "timepoint", "current_value", "status"],
        )
        out_df = out_df.sort_values(["network", "subject", "timepoint", "variable"]).reset_index(drop=True)
        self._write_csv_atomic(out_df, out_csv)
        # Quick status breakdown for the operator (no values, just counts).
        if len(out_df):
            counts = out_df["status"].value_counts().to_dict()
            print(f"Status breakdown: {counts}")
        print(f"Wrote {len(out_df)} comparison rows to: {out_csv}")

    @staticmethod
    def _write_csv_atomic(df, out_csv):
        """Replace a CSV only after its complete contents are on disk."""
        directory = os.path.dirname(os.path.abspath(out_csv))
        os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(out_csv)}.", suffix=".csv",
            dir=directory,
        )
        os.close(fd)
        try:
            df.to_csv(temporary_path, index=False)
            os.replace(temporary_path, out_csv)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def compare_against_combined_csvs(self):
        """
        For every (subject, variable, timepoint) blank record, look up
        the current value in the per-network/per-timepoint combined CSV
        and produce a row tagged with one of:
          * filled         — non-empty, not a missing-code
          * still_blank    — empty string
          * missing_code   — value matches utils.missing_code_set
          * subject_missing  — subject not in that CSV
          * subject_ambiguous — duplicate subject rows disagree on this value
          * variable_missing — column not in that CSV
          * csv_missing      — combined CSV file not found / unreadable

        CSV path follows qc_forms_main.py:102. Each combined CSV is
        loaded at most once; subjects are indexed into a dict for O(1)
        lookup.
        """
        by_csv = defaultdict(list)
        for record in self.blank_records:
            network, subject, variable, timepoint = record
            by_csv[(network, timepoint)].append(record)

        rows = []
        for (network, timepoint), records in by_csv.items():
            csv_path = self._combined_csv_path(network, timepoint)
            cdf, csv_error = self._load_combined_csv(csv_path)

            if cdf is None:
                # Same status for every record in this (network, tp) bucket.
                print(f"WARNING: combined CSV unreadable for {network}/{timepoint}: {csv_error}")
                for (_, subject, variable, _) in records:
                    rows.append((network, subject, variable, timepoint, "", "csv_missing"))
                continue

            if "subjectid" not in cdf.columns:
                print(f"WARNING: combined CSV at {csv_path} has no 'subjectid' column; treating as csv_missing")
                for (_, subject, variable, _) in records:
                    rows.append((network, subject, variable, timepoint, "", "csv_missing"))
                continue

            # Duplicate subjects are not resolved by row order.  A current
            # value remains determinable only when every matching row agrees
            # for the requested variable.
            subject_positions = defaultdict(list)
            for i, sid in enumerate(cdf["subjectid"]):
                sid_s = "" if pd.isna(sid) else str(sid).strip()
                if sid_s:
                    subject_positions[sid_s].append(i)
            missing_codes = {
                str(code).strip()
                for code in (
                    set(self.utils.missing_code_set)
                    | set(self.utils.missing_code_list)
                )
            }

            for (_, subject, variable, _) in records:
                if subject not in subject_positions:
                    rows.append((network, subject, variable, timepoint, "", "subject_missing"))
                    continue
                if variable not in cdf.columns:
                    rows.append((network, subject, variable, timepoint, "", "variable_missing"))
                    continue
                values = []
                for position in subject_positions[subject]:
                    value = cdf.iloc[position][variable]
                    values.append("" if pd.isna(value) else str(value).strip())
                distinct_values = set(values)
                if len(distinct_values) != 1:
                    rows.append((
                        network, subject, variable, timepoint, "",
                        "subject_ambiguous",
                    ))
                    continue
                value_s = values[0]
                if value_s == "":
                    status = "still_blank"
                elif value_s in missing_codes:
                    status = "missing_code"
                else:
                    status = "filled"
                rows.append((network, subject, variable, timepoint, value_s, status))

        return rows

    def _combined_csv_path(self, network, timepoint):
        """Match the path convention in qc_forms_main.py:102.

        Tracker timepoints come from FormCheck.timepoint (e.g. 'screening',
        'baseline', 'month1'..'month12', 'month18', 'month24', 'floating',
        'conversion'). The combined CSV uses 'month_N' / 'floating_forms'
        and 'ProNET' for PRONET.
        """
        tp_munged = timepoint.replace("month", "month_").replace("floating", "floating_forms")
        net_munged = network.replace("PRONET", "ProNET")
        return (
            f"{self.comb_csv_path}AMPSCZ-combined-redcap_"
            f"{tp_munged}_{net_munged}-day1to1.csv"
        )

    def _load_combined_csv(self, csv_path):
        """Read a combined CSV with the same hardening as qc_forms_main:
        keep_default_na=False, utf-8, on_bad_lines='error'. Returns
        (df, None) on success or (None, error_string) on failure so the
        caller can tag every dependent record with csv_missing."""
        if not os.path.isfile(csv_path):
            return None, f"file not found: {csv_path}"
        try:
            df = pd.read_csv(
                csv_path,
                keep_default_na=False,
                encoding="utf-8",
                on_bad_lines="error",
                dtype=str,
            )
            return df, None
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
            return None, f"{type(e).__name__}: {str(e)[:200]}"


if __name__ == '__main__':
    # Flip compare_to_current=False to skip the combined-CSV lookup and
    # just emit the (subject, variable, timepoint) history.
    BlankFlagExtractor(compare_to_current=True).run_script()
