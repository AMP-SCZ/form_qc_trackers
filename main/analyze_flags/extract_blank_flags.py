"""
Walk every Dropbox revision of the combined network tracker(s) and emit
a deduplicated CSV of every (subject, variable, timepoint) that was
*ever* flagged with "Variable is blank.", regardless of whether the flag
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
import re
import sys
import json
import dropbox
from collections import defaultdict
from io import BytesIO
from datetime import datetime, timedelta, timezone
from xml.etree.ElementTree import ParseError as XMLParseError

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
from utils.utils import Utils
from analyze_flags.canonicalize import normalize_timepoint
from analyze_flags.paths import (
    ensure_analyze_flags_artifact_dir,
    pipeline_output_path_from_config,
)


class BlankFlagExtractor:
    # Substring (case-insensitive) used to identify blank-variable flags.
    # Only canonical phrasing in the current codebase is "Variable is blank."
    # (qc_types/general_checks.py:114). Substring keeps it tolerant to
    # tense / punctuation drift in older Dropbox revisions.
    BLANK_MSG_NEEDLE = "variable is blank"

    OUTPUT_BASENAME = "blank_flag_history.csv"

    NETWORKS = ['PRESCIENT', 'PRONET']

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
        self.loop_dropbox()
        if self.compare_to_current:
            comparison_rows = self.compare_against_combined_csvs()
            self.write_comparison_output(comparison_rows)
        else:
            self.write_history_output()

    def loop_dropbox(self):
        for network_name in self.NETWORKS:
            for base, basename in self._tracker_paths(network_name):
                combined_output = f'{base}{network_name}/combined/{basename}'
                try:
                    self.extract_blank_flags(combined_output, network_name, days_back=None)
                except Exception as e:
                    # Missing pre-V2 tracker path / API errors shouldn't
                    # kill the whole run — log and move on so the V2 file
                    # (or the next network) still gets processed.
                    print(f"WARNING: failed walking revisions for {combined_output}: "
                          f"{type(e).__name__}: {str(e)[:200]}")
                    continue

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

    def extract_blank_flags(self, path, network_name, days_back=None, page_limit=100):
        """
        Walk every revision of ``path`` (paging beyond 100 via before_rev)
        and add (subject, variable, timepoint) tuples to
        ``self.blank_records`` for every blank-variable flag observed.

        The raw-RPC plumbing is copied from estimate_resolved.py so the
        same older-SDK fallback behavior holds.
        """
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
                    df = self._read_tracker_excel(resp.content)

                    if all(c in df.columns for c in new_cols):
                        subject_col, tp_col, flags_col = new_cols
                    elif all(c in df.columns for c in old_cols):
                        subject_col, tp_col, flags_col = old_cols
                    else:
                        print(f"WARNING: Skipping revision {rev} from {when} - unrecognized columns: {list(df.columns)}")
                        continue

                    added = self._collect_blank_records(df, subject_col, tp_col, flags_col, network_name)
                    if content_hash is not None:
                        seen_hashes.add(content_hash)

                    kept += 1
                    print(f"{path} - Revision {kept}: {rev} from {when} (+{added} new blank rows; total {len(self.blank_records)})")
                except (ValueError, XMLParseError, Exception) as e:
                    print(f"WARNING: Skipping revision {rev} from {when} - file error: {type(e).__name__}: {str(e)[:100]}")
                    continue

            before_rev = entries[-1]["rev"]

        if oldest_revision and newest_revision:
            days_span = (newest_revision - oldest_revision).days
            print(f"Done with {network_name}: scanned {kept} revisions "
                  f"(skipped {skipped_dupes} duplicate-content) spanning {days_span} days "
                  f"({oldest_revision} → {newest_revision}); "
                  f"{len(self.blank_records)} unique (subject, variable, timepoint) so far.")
        else:
            print(f"Done with {network_name}: scanned {kept} revisions "
                  f"(skipped {skipped_dupes} duplicate-content); "
                  f"{len(self.blank_records)} unique (subject, variable, timepoint) so far.")

    def _collect_blank_records(self, df, subject_col, tp_col, flags_col, network_name):
        """
        Parse the Specific Flags / Flags column, splitting per
        form_check.py:353 ("{var} : {message}") joined with " | "
        (create_trackers.py:342). Adds a tuple per blank-variable flag
        and returns how many *new* tuples were added in this call.
        """
        before = len(self.blank_records)
        for row in df[[subject_col, tp_col, flags_col]].itertuples(index=False, name=None):
            subject, timepoint, flags = row
            if not isinstance(flags, str) or not flags.strip():
                continue
            for entry in flags.split('|'):
                entry = entry.strip()
                if not entry:
                    continue
                parts = entry.split(':', 1)
                if len(parts) != 2:
                    continue
                variable, message = parts[0].strip(), parts[1].strip()
                if not variable:
                    continue
                if self.BLANK_MSG_NEEDLE in message.lower():
                    self.blank_records.add((
                        network_name,
                        str(subject).strip(),
                        variable,
                        normalize_timepoint(timepoint),
                    ))
        return len(self.blank_records) - before

    def _read_tracker_excel(self, content):
        """Read one tracker revision's bytes into a DataFrame.

        Prefers the Rust-backed ``calamine`` engine — typically several times
        faster than openpyxl on these multi-MB workbooks, and it releases the
        GIL — and falls back to openpyxl if ``python-calamine`` isn't installed
        so the walk still runs everywhere. ``keep_default_na=False`` matches the
        original read so empty cells stay empty strings (not NaN); the two
        engines were verified equivalent on representative synthetic data under
        it. The engine is resolved once and cached on the instance.

        If calamine chokes on a single old/oddly-formatted revision that
        openpyxl can still read, that one file is retried with openpyxl (engine
        choice unchanged) so coverage is never worse than an openpyxl-only run.
        Re-parsing identical bytes can't diverge — the engines agree on any
        file both can parse — it only recovers a revision we'd otherwise lose.
        """
        if self._excel_engine is None:
            try:
                import python_calamine  # noqa: F401
                self._excel_engine = "calamine"
            except ImportError:
                self._excel_engine = "openpyxl"
                print("NOTE: python-calamine not installed; using openpyxl "
                      "(slower). Run `pip install python-calamine` to speed up "
                      "the revision walk.")
        try:
            return pd.read_excel(
                BytesIO(content),
                keep_default_na=False,
                engine=self._excel_engine,
            )
        except Exception as e:
            if self._excel_engine == "openpyxl":
                raise
            print(f"NOTE: calamine failed to parse a revision "
                  f"({type(e).__name__}: {str(e)[:80]}); retrying with openpyxl")
            return pd.read_excel(
                BytesIO(content),
                keep_default_na=False,
                engine="openpyxl",
            )

    def write_history_output(self):
        """Just (subject, variable, timepoint) — drop network so the
        original 3-column ask is preserved when compare_to_current is off."""
        out_csv = os.path.join(self.analyze_flags_dir, self.OUTPUT_BASENAME)
        rows = sorted({(subject, variable, tp) for (_, subject, variable, tp) in self.blank_records})
        out_df = pd.DataFrame(rows, columns=["subject", "variable", "timepoint"])
        out_df.to_csv(out_csv, index=False)
        print(f"Wrote {len(out_df)} unique (subject, variable, timepoint) rows to: {out_csv}")

    def write_comparison_output(self, comparison_rows):
        out_csv = os.path.join(self.analyze_flags_dir, self.OUTPUT_BASENAME)
        out_df = pd.DataFrame(
            comparison_rows,
            columns=["network", "subject", "variable", "timepoint", "current_value", "status"],
        )
        out_df = out_df.sort_values(["network", "subject", "timepoint", "variable"]).reset_index(drop=True)
        out_df.to_csv(out_csv, index=False)
        # Quick status breakdown for the operator (no values, just counts).
        if len(out_df):
            counts = out_df["status"].value_counts().to_dict()
            print(f"Status breakdown: {counts}")
        print(f"Wrote {len(out_df)} comparison rows to: {out_csv}")

    def compare_against_combined_csvs(self):
        """
        For every (subject, variable, timepoint) blank record, look up
        the current value in the per-network/per-timepoint combined CSV
        and produce a row tagged with one of:
          * filled         — non-empty, not a missing-code
          * still_blank    — empty string
          * missing_code   — value matches utils.missing_code_set
          * subject_missing  — subject not in that CSV
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

            # Index subjectid -> first matching positional index. Combined
            # CSVs should be 1 row per subject after day1to1, but if a
            # duplicate sneaks in we just take the first match.
            subject_index = {}
            for i, sid in enumerate(cdf["subjectid"].astype(str)):
                if sid not in subject_index:
                    subject_index[sid] = i

            for (_, subject, variable, _) in records:
                if subject not in subject_index:
                    rows.append((network, subject, variable, timepoint, "", "subject_missing"))
                    continue
                if variable not in cdf.columns:
                    rows.append((network, subject, variable, timepoint, "", "variable_missing"))
                    continue
                value = cdf.iloc[subject_index[subject]][variable]
                value_s = "" if pd.isna(value) else str(value).strip()
                if value_s == "":
                    status = "still_blank"
                elif value_s in self.utils.missing_code_set or value_s in {str(c) for c in self.utils.missing_code_list}:
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
