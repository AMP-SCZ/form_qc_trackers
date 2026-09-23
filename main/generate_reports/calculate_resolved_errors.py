import pandas as pd

import os
import sys
import json
import random
import re
import shutil
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from generate_reports.tracker_paths import sanitize_ra_folder_name
from generate_reports.dropbox_reads import download_workbook, get_metadata
from datetime import datetime
from io import BytesIO
import numpy as np
from dropbox.exceptions import ApiError

"""
cols to match : subject, displayed timepoint, 
displayed form, displayed variable, error message
if row exists in current output and not at all in old
    append date detected as today

if row exists in old output and not new
    if the most recent date resolved is after the most recent date
    detected and neither are blank, ignore it
    if the most recent date detected is after the most recent date resolved
    and neither are blank, append today's date to the date resolved, set currently_resolved to False
    if date detected is not blank and date resolved is blank, append
    today's date to date resolved, set currently_resolved to True

if row exists in both outputs
    if it is currently resolved in the old one,
    append today's date to dates detected
    and change currently_resolved to False
    If it is currently not resolved in the old one,
    replace it with the new one (so other columns like
    subject's current timepoint get updated)
"""

class CalculateResolvedErrors():
    DATE_REPORT_NAME = 'Date Report'
    CROSS_CHECK_REPORT_NAME = 'Cross Checks'
    PROPOSED_CHECKS_REPORT_NAME = 'Proposed Checks'
    MEDICATION_FLAGS_REPORT_NAME = 'Medication Flags'
    CONVERSION_REPORT_NAME = 'Conversion Report'
    # Empty cells in historical workbooks are ambiguous: they can mean either
    # "the reviewer did not edit this field" or "clear the saved value".  Keep
    # blanks as no-ops and provide one explicit, case-insensitive command for
    # the latter.  The token is consumed during import and is never persisted
    # to current_output or the next generated workbook.
    CLEAR_VALUE_TOKEN = '[CLEAR]'
    CROSS_CHECK_ID_PATTERN = r'CROSS-QC-\d{3}'
    # Optional numeric evidence columns must remain numeric across the
    # new -> current -> old reconciliation cycle. A blanket ``fillna('')``
    # converts them to object columns containing both floats and strings, which
    # PyArrow cannot serialize (e.g. date_gap_days: 10.0 plus '').
    NUMERIC_OUTPUT_COLUMNS = frozenset({
        'date_gap_days',
        'time_since_last_detection',
    })
    
    def __init__(self,formatted_col_names):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.output_path = self.config_info['paths']['output_path']
        if self.config_info["testing_enabled"] == "True":
            self.output_path += "testing/"
            self.dropbox_path = f'/Apps/Automated QC Trackers/refactoring_tests/'
        else:
            self.dropbox_path = f'/Apps/Automated QC Trackers/'

        self.out_paths = {}
        for path_pref in ['old','new','current']:
            directory = f"{self.output_path}combined_outputs/{path_pref}_output/"
            if not os.path.exists(directory):
                os.makedirs(os.path.dirname(directory), exist_ok=True)

            # Parquet handoff: ~3-5x smaller than CSV, preserves dtypes
            # (no more bool-as-string round trips), and substantially faster
            # to read/write at ~1M rows. Requires `pyarrow` to be installed.
            # `directory` already ends in '/', so don't add another.
            self.out_paths[path_pref] = f"{directory}combined_qc_flags.parquet"
        self.new_output = []
        # Dropbox tracker workbooks that could not be parsed this run
        # (torn/truncated zip, 0-byte, or a save interrupted mid-write).
        # Populated in read_dropbox_data, summarized at the end of
        # loop_dropbox_files so one bad file out of ~456 is identifiable.
        self._corrupt_workbooks = []
        self.old_output_csv_path = f'{self.output_path}combined_outputs/old_output/combined_qc_flags.parquet'
        self.new_output_csv_path = f'{self.output_path}combined_outputs/new_output/combined_qc_flags.parquet'
        self.dropbox_data_folder = f'{self.output_path}formatted_outputs/dropbox_files/'

        self.formatted_column_names = formatted_col_names
        self.melbourne_ras = self.utils.load_dependency_json('melbourne_ra_subs.json')

    def run_script(self):
        # determine which errors no longer exist in the new output
        self.determine_resolved_rows()
        print('stage 1 done')
        # read specified columns from dropbox to new output
        self.loop_dropbox_files()
        print('stage 2 done')

    @classmethod
    def _normalize_output_frame(cls, df):
        """Preserve canonical Parquet dtypes while filling text nulls.

        Merge/state logic uses ``''`` for absent text, but numeric evidence
        must use nullable numeric NA. Only columns whose populated values are
        genuinely strings receive the empty-string fill. Known numeric columns
        are validated and coerced explicitly so invalid text fails with a clear
        error instead of a low-level ArrowTypeError at write time.
        """
        if df is None:
            return pd.DataFrame()
        out = df.copy()
        for col in cls.NUMERIC_OUTPUT_COLUMNS.intersection(out.columns):
            ser = out[col]
            if pd.api.types.is_numeric_dtype(ser.dtype):
                # Idempotent path: _normalize_output_frame is intentionally
                # called on read AND immediately before write. pandas 1.x can
                # crash inside Series.replace('', pd.NA) when ``ser`` is
                # already a nullable Int64 column (scalar bool mask has no
                # ``to_numpy``). Numeric columns need no string replacement.
                raw = ser
            else:
                # Avoid Series.replace for old-pandas compatibility and handle
                # only scalar blank strings. Object columns can contain mixed
                # legacy scalars, where vectorized ``arr == ''`` may itself
                # return a scalar False and trigger the same pandas bug.
                raw = ser.map(
                    lambda value: pd.NA
                    if isinstance(value, str) and value.strip() == ''
                    else value)
            numeric = pd.to_numeric(raw, errors='coerce')
            invalid = raw.notna() & numeric.isna()
            if invalid.any():
                examples = raw[invalid].astype(str).drop_duplicates().head(5).tolist()
                raise ValueError(
                    f"Column {col} must be numeric or blank; invalid values: {examples}")
            out[col] = numeric.astype('Int64')

        for col in out.columns:
            if col in cls.NUMERIC_OUTPUT_COLUMNS:
                continue
            ser = out[col]
            if not (pd.api.types.is_object_dtype(ser.dtype)
                    or pd.api.types.is_string_dtype(ser.dtype)):
                continue
            populated = ser.dropna()
            if populated.empty or populated.map(lambda v: isinstance(v, str)).all():
                out[col] = ser.fillna('')
        return out

    def _read_current(self):
        return self._normalize_output_frame(
            pd.read_parquet(self.out_paths['current']))

    def _write_current(self, df):
        # Atomic write of canonical current_output. PID-stamped tmp
        # avoids concurrent-run collisions on the same .tmp path;
        # os.replace is atomic on POSIX and effectively atomic on
        # Windows. Without this, a crash mid-write produces a corrupt
        # parquet that the next run cannot read; the consumer-side
        # empty-df guard would then falsely mark every previously-open
        # flag as resolved.
        current_path = self.out_paths['current']
        tmp_path = f"{current_path}.{os.getpid()}.tmp"
        try:
            df = self._normalize_output_frame(df)
            df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, current_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    @classmethod
    def _stable_cross_check_ids(cls, df):
        """Return a normalized stable ID for each Cross Checks row.

        New files carry ``check_id`` explicitly.  Legacy parquet/workbooks do
        not, but Cross Checks messages have always begun with
        ``[CROSS-QC-NNN]``; extracting that prefix supplies a safe migration
        fallback.  If an explicit ID and message prefix conflict, use neither
        rather than risk applying reviewer state to the wrong check.
        """
        index = df.index
        if 'check_id' in df.columns:
            explicit = df['check_id'].fillna('').astype(str).str.strip().str.upper()
        else:
            explicit = pd.Series('', index=index, dtype='object')
        explicit_valid = explicit.str.fullmatch(
            cls.CROSS_CHECK_ID_PATTERN, case=False, na=False)
        explicit = explicit.where(explicit_valid, '')

        if 'error_message' in df.columns:
            from_message = (
                df['error_message'].fillna('').astype(str)
                .str.extract(
                    rf'\[({cls.CROSS_CHECK_ID_PATTERN})\]',
                    flags=re.IGNORECASE, expand=False)
                .fillna('').str.upper()
            )
        else:
            from_message = pd.Series('', index=index, dtype='object')

        conflict = explicit.ne('') & from_message.ne('') & explicit.ne(from_message)
        result = explicit.where(explicit.ne(''), from_message)
        return result.mask(conflict, '')

    @classmethod
    def _add_cross_workbook_identity(cls, df):
        """Add temporary keys that prefer check_id and fall back to legacy keys."""
        out = df.copy()
        stable_id = cls._stable_cross_check_ids(out)
        has_id = stable_id.ne('')
        out['__cross_check_id'] = stable_id
        out['__cross_legacy_form'] = out['displayed_form'].where(~has_id, '')
        out['__cross_legacy_message'] = out['error_message'].where(~has_id, '')
        return out

    @staticmethod
    def _row_belongs_to_report(row, report):
        """True when a merged reconciliation row is routed to ``report``.

        The old/new outer merge suffixes both sides, so consult the new-side
        routing first (that is this run's truth for a flag that is still
        firing) and fall back to an unsuffixed legacy column, then the old
        side. The first non-blank routing string decides; `reports` is a
        ' | '-joined token list, so match whole tokens rather than a
        substring of a longer report name.
        """
        for attr in ('reports_new', 'reports', 'reports_old'):
            value = getattr(row, attr, '')
            # `_normalize_output_frame` fills text nulls with '', but guard
            # against a raw NaN reaching here anyway: `str(nan)` is 'nan',
            # which is non-blank and would stop the fallback at a side that
            # carries no routing at all.
            if value is None or pd.isna(value) or str(value) == '':
                continue
            return report in str(value).split(' | ')
        return False

    @classmethod
    def _add_reconciliation_identity(cls, df):
        """Add stable old/new merge keys without changing persisted columns."""
        out = df.copy()
        stable_id = cls._stable_cross_check_ids(out)
        has_id = stable_id.ne('')
        # Keep the original ``network`` column out of the merge keys because
        # compare_old_new_outputs deliberately inspects network_old/network_new
        # to distinguish old-only, new-only, and continuing rows.  A temporary
        # copy provides network isolation without changing that state-machine
        # contract.  Subject IDs are not guaranteed to remain globally unique
        # across PRONET and PRESCIENT.
        out['__history_network'] = out['network']
        out['__history_check_id'] = stable_id
        out['__history_legacy_form'] = out['displayed_form'].where(~has_id, '')
        out['__history_legacy_variable'] = out['displayed_variable'].where(~has_id, '')
        out['__history_legacy_message'] = out['error_message'].where(~has_id, '')
        return out

    @classmethod
    def _explode_report_flags(cls, report_df):
        """Explode grouped tracker flags while keeping Check ID pairs aligned."""
        expanded = []
        has_id_column = 'check_id' in report_df.columns
        for _, source_row in report_df.iterrows():
            message_parts = str(source_row['error_message']).split(' | ')
            if has_id_column:
                raw_id = source_row.get('check_id', '')
                id_parts = str(raw_id).split(' | ') if str(raw_id).strip() else []
            else:
                id_parts = []
            ids_align = len(id_parts) == len(message_parts)
            for position, message in enumerate(message_parts):
                row = source_row.copy()
                row['error_message'] = message
                row['check_id'] = id_parts[position] if ids_align else ''
                expanded.append(row)
        if not expanded:
            return report_df.iloc[0:0].assign(check_id='')
        return pd.DataFrame(expanded).reset_index(drop=True)

    @staticmethod
    def _strip_proposed_evidence(message):
        """Return canonical flag text from a decorated workbook cell."""

        return re.sub(
            r"\r?\nVariables & values(?: v[23])?:.*\Z", "", str(message),
            flags=re.DOTALL)

    @classmethod
    def _deduplicate_workbook_rows(
            cls, report_df, merge_keys, columns_to_read,
            dropbox_path, report):
        """Coalesce unambiguous duplicate edits and reject conflicting ones.

        Keeping the first copy silently loses a valid edit when a reviewer
        copy/pastes a row, edits the copy, and leaves the original blank.
        Choosing the last copy is equally unsafe when two copies conflict.
        Merge complementary or identical actions; omit an ambiguous identity
        entirely so canonical state remains unchanged for that flag.
        """
        duplicate_mask = report_df.duplicated(subset=merge_keys, keep=False)
        if not duplicate_mask.any():
            return report_df

        pieces = [report_df.loc[~duplicate_mask].copy()]
        conflicts = 0
        duplicate_rows = report_df.loc[duplicate_mask]
        for _, group in duplicate_rows.groupby(
                merge_keys, sort=False, dropna=False):
            merged_row = group.iloc[0].copy()
            ambiguous = False
            for column in columns_to_read:
                if column not in group.columns:
                    continue
                actions = {}
                for value in group[column].tolist():
                    if pd.isna(value):
                        continue
                    text = str(value)
                    stripped = text.strip()
                    if not stripped:
                        continue
                    canonical = stripped
                    if (stripped.casefold()
                            == cls.CLEAR_VALUE_TOKEN.casefold()):
                        canonical = cls.CLEAR_VALUE_TOKEN
                    actions.setdefault(canonical, value)
                if len(actions) > 1:
                    ambiguous = True
                    break
                merged_row[column] = (
                    next(iter(actions.values())) if actions else '')
            if ambiguous:
                conflicts += 1
                continue
            pieces.append(pd.DataFrame([merged_row], columns=report_df.columns))

        if conflicts:
            print(
                f"[calculate_resolved_errors] WARNING: skipped {conflicts} "
                f"conflicting duplicate flag identity/identities in sheet "
                f"{report!r} of {dropbox_path}; canonical reviewer state was "
                f"left unchanged for those flags.")
        return pd.concat(pieces, ignore_index=True)
            

    def loop_dropbox_files(self):
        """Reconcile edits from the established V2 workbooks for both networks."""
        dbx = self.utils.collect_dropbox_credentials()
        self._corrupt_workbooks = []
        current_df = self._read_current()
        for network in self.utils.pipeline_networks:
            if (network not in self.formatted_column_names
                    or network not in self.utils.all_sites):
                continue
            network_dir = self.dropbox_path.rstrip('/') + '/' + network
            combined_output = (
                network_dir + f'/combined/{network}_Output_V2.xlsx')
            combined_cols = self.formatted_column_names[network]['combined']
            if self.check_dbx_file_exists(dbx, combined_output):
                _, data = download_workbook(dbx, combined_output)
                authoritative_reports = [
                    'Main Report', 'Non Team Forms', self.CONVERSION_REPORT_NAME,
                    self.DATE_REPORT_NAME]
                current_df = self.read_dropbox_data(
                    current_df, combined_cols, ['manually_resolved', 'comments'],
                    combined_output, dbx, network, ['Main Report'],
                    preloaded_data=data)
                # Date has its own reviewer surface for both networks. Read
                # network comments and manual resolutions explicitly; the
                # generic pass below cannot override these Date-owned fields.
                current_df = self.read_dropbox_data(
                    current_df, combined_cols, ['manually_resolved', 'comments'],
                    combined_output, dbx, network, [self.DATE_REPORT_NAME],
                    preloaded_data=data)
                current_df = self.read_dropbox_data(
                    current_df, combined_cols, ['manually_resolved'],
                    combined_output, dbx, network, ['Non Team Forms'],
                    preloaded_data=data)
                current_df = self.read_dropbox_data(
                    current_df, combined_cols, ['comments'],
                    combined_output, dbx, network, ['Missingness Report'],
                    protected_reports=authoritative_reports,
                    preloaded_data=data)
                current_df = self.read_dropbox_data(
                    current_df, combined_cols, ['manually_resolved'],
                    combined_output, dbx, network, [self.CONVERSION_REPORT_NAME],
                    preloaded_data=data)
                current_df = self.read_dropbox_data(
                    current_df, combined_cols, ['manually_resolved'],
                    combined_output, dbx, network, [], excl_report=False,
                    protected_reports=authoritative_reports,
                    preloaded_data=data)
            for site_abr in self.utils.all_sites[network]:
                site = self.utils.site_full_name_translations.get(site_abr, site_abr)
                site_output = (
                    network_dir + f'/{site}/{network}_{site_abr}_Output_V2.xlsx')
                site_cols = self.formatted_column_names[network]['sites']
                reports_to_read = (
                    ['Non Team Forms'] if site_abr == 'ME' else ['Main Report'])
                reports_to_read.append(self.DATE_REPORT_NAME)
                # The Melbourne site workbook is the fallback for unassigned
                # participants; a specific RA workbook overrides it afterwards.
                current_df = self.read_dropbox_data(
                    current_df, site_cols, ['site_comments'],
                    site_output, dbx, network, reports_to_read)
                if site_abr == 'ME':
                    for ra in self.melbourne_ras:
                        safe_ra = sanitize_ra_folder_name(ra)
                        ra_output = (
                            network_dir + f'/{site}/{safe_ra}/'
                            + f'{network}_Melbourne_Output_V2.xlsx')
                        current_df = self.read_dropbox_data(
                            current_df, site_cols, ['site_comments'],
                            ra_output, dbx, network, reports_to_read)
        if self._corrupt_workbooks:
            distinct = sorted(set(self._corrupt_workbooks))
            print(
                f"[calculate_resolved_errors] WARNING: skipped "
                f"{len(distinct)} corrupt/unreadable tracker workbook(s); "
                f"reviewer edits in them were not merged: {distinct}")
        self._write_current(current_df)
        
    def check_dbx_file_exists(self, dbx, dropbox_path):
        """
        Lightweight existence probe via metadata only (no full file download).

        - Path not found -> False (caller skips merge for that tracker).
        - Temporary transport failures are retried with bounded backoff.
        - Other failures propagate; none are treated as a missing file.
        """
        try:
            get_metadata(dbx, dropbox_path)
            return True
        except ApiError as e:
            if e.error.is_path():
                lookup = e.error.get_path()
                if lookup is not None and lookup.is_not_found():
                    print(
                        f"[calculate_resolved_errors] Dropbox path not found "
                        f"(skipping comment merge for this file): {dropbox_path}"
                    )
                    return False
            raise RuntimeError(
                f"FATAL: Dropbox API error while checking {dropbox_path!r}: {e!r}. "
                f"Refusing to treat this as a missing file — fix credentials, "
                f"network, or Dropbox status and re-run."
            ) from e
        
    def read_dropbox_data(self,
        prev_output_df, col_names, columns_to_read,
        dropbox_path, dbx, network, reports_to_read,
        excl_report = True, protected_reports = None,
        preloaded_data = None
    ):
        """
        Merge reviewer-edited columns (manual-resolution, comments) from a
        Dropbox tracker workbook back into the canonical current_output.

        protected_reports : list | None
            When supplied, the values read from this workbook's sheets may
            only overwrite a flag's column if that flag does NOT belong to
            any of the named reports. Used so the individual / team report
            sheets cannot change the manual-resolution state of a flag that
            also lives in an authoritative report (Main Report / Non Team
            Forms) — those reports own their flags and can only be edited in
            their own sheets. When None (the default), every non-blank value
            is applied unconditionally, preserving the prior behaviour.
        preloaded_data : bytes | None
            Raw workbook bytes already downloaded by the caller. Lets a
            caller download a workbook once and run several read passes over
            it (different sheets / columns) without re-downloading. When None
            the workbook is downloaded here as before.
        """
        reversed_col_translations = self.utils.reverse_dictionary(col_names)
        if preloaded_data is not None:
            data = preloaded_data
        else:
            if self.check_dbx_file_exists(dbx, dropbox_path) == False:
                return prev_output_df
            _, data = download_workbook(dbx, dropbox_path)
        try:
            excel_data = pd.ExcelFile(BytesIO(data))
        except Exception as e:
            # The bytes for this tracker are not a readable workbook. The
            # real shapes (each raises a different exception, all handled the
            # same way — skip THIS file, do not abort the whole reports stage
            # for the other ~456 trackers):
            #   - torn / truncated zip  (PK magic present, central directory
            #     missing — killed/disk-full/power-loss or truncated transfer
            #     mid-write)  -> zipfile.BadZipFile "File is not a zip file"
            #   - 0-byte or non-zip bytes -> ValueError "Excel file format
            #     cannot be determined"
            #   - structurally-valid zip missing xl/workbook.xml (a save
            #     interrupted by a Python exception) -> OptionError / KeyError
            # Log the path + magic bytes + length so the one bad file is
            # identifiable in the log (a zip signature and a byte count carry
            # no PHI). The end-of-run summary in loop_dropbox_files repeats
            # the list. Returning prev_output_df leaves this run's edits from
            # the file unmerged; the next create_trackers run regenerates it
            # as a fresh valid workbook.
            magic = bytes(data[:4]) if data else b''
            print(
                f"[calculate_resolved_errors] CORRUPT WORKBOOK — skipping "
                f"comment/resolution merge for {dropbox_path} "
                f"({type(e).__name__}: {e}; first4={magic!r}, "
                f"bytes={len(data) if data is not None else 0})"
            )
            self._corrupt_workbooks.append(dropbox_path)
            return prev_output_df
        sheet_names = excel_data.sheet_names
        for report in sheet_names:
            if report not in reports_to_read and excl_report == True:
                continue
            try:
                report_df = pd.read_excel(
                    BytesIO(data), sheet_name=report, keep_default_na=False)
            except Exception as e:
                # A valid workbook container can still contain one truncated or
                # malformed worksheet XML part. ExcelFile() only reads workbook
                # metadata, so that damage may surface here rather than in the
                # outer workbook-open guard. Quarantine this sheet/file and let
                # the remaining tracker workbooks continue.
                print(
                    f"[calculate_resolved_errors] CORRUPT WORKSHEET — skipping "
                    f"sheet {report!r} in {dropbox_path} "
                    f"({type(e).__name__}: {e})")
                self._corrupt_workbooks.append(dropbox_path)
                continue
            report_df.rename(columns=reversed_col_translations, inplace=True)
            # ``Check ID`` is intentionally added by CreateTrackers without
            # changing the long-lived formatted-column JSON contract.  Accept
            # it explicitly, while continuing to read old workbooks that do
            # not have the column.
            if 'Check ID' in report_df.columns and 'check_id' not in report_df.columns:
                report_df.rename(columns={'Check ID': 'check_id'}, inplace=True)
            # Restrict report_df to the merge keys + the columns we
            # actually intend to read back. Without this, every Excel
            # column not present in current_output (e.g., flag_count,
            # date_resolved) leaks into current_df via the left-merge:
            # such columns arrive with NO `_dbx` suffix (they only
            # exist on the report side), so the suffix-based cleanup
            # at the end of the loop never removes them. They then
            # persist into _write_current as object-dtype columns
            # mixing real values (matched rows) with '' (unmatched,
            # post-fillna) — which pyarrow rejects on to_parquet.
            # ``current_output`` contains both networks at once. The workbook
            # itself does not expose a network column, so stamp the network
            # implied by its Dropbox path onto every imported row and include
            # it in the merge identity. Subject IDs are normally network-
            # distinct, but relying on that convention could transfer a
            # comment/resolution to the other network if an ID is ever reused.
            report_df['network'] = network
            workbook_merge_keys = [
                'displayed_form', 'displayed_timepoint',
                'subject', 'error_message']
            merge_keys = ['network'] + workbook_merge_keys
            # These workbooks are hand-edited in Dropbox, and the all-sheets
            # pass (excl_report=False) now visits every tab — including any a
            # reviewer added (scratch / pivot / notes) or whose headers they
            # renamed. Such a sheet won't carry the pipeline column schema, so
            # quarantine it (log + skip) instead of letting a KeyError on the
            # merge keys abort the entire reports stage for that network.
            if not all(k in report_df.columns for k in workbook_merge_keys):
                print(
                    f"[calculate_resolved_errors] skipping sheet {report!r} in "
                    f"{dropbox_path}: missing expected column(s) "
                    f"{[k for k in workbook_merge_keys if k not in report_df.columns]} "
                    f"— not a standard report sheet."
                )
                continue
            keep = [c for c in (merge_keys + ['check_id'] + list(columns_to_read))
                    if c in report_df.columns]
            report_df = report_df[keep]
            # Coerce error_message to str before splitting: keep_default_na=
            # False leaves a hand-typed bare number (or other non-string cell)
            # as a non-str, which would AttributeError on .split.
            report_df['error_message'] = report_df['error_message'].astype(str)
            report_df = self._explode_report_flags(report_df)
            if report == self.PROPOSED_CHECKS_REPORT_NAME:
                # Proposed workbook flags show current variable/value evidence,
                # but the canonical message remains the stable identity used
                # for lifecycle and reviewer-state reconciliation.
                report_df['error_message'] = report_df['error_message'].map(
                    self._strip_proposed_evidence)

            merge_keys_for_report = merge_keys
            if report == self.CROSS_CHECK_REPORT_NAME:
                # Stable Cross Checks identity deliberately ignores display
                # form/message text.  Labels may be clarified without making a
                # still-open flag look resolved and without losing comments.
                # Rows from legacy workbooks fall back to the exact keys used
                # before Check ID existed.
                prev_output_df = self._add_cross_workbook_identity(prev_output_df)
                report_df = self._add_cross_workbook_identity(report_df)
                merge_keys_for_report = [
                    'network', 'displayed_timepoint', 'subject', '__cross_check_id',
                    '__cross_legacy_form', '__cross_legacy_message']
            # Resolve duplicate merge keys on the hand-edited report side
            # before the left merge. Safe complementary edits are coalesced;
            # contradictory copies are quarantined rather than multiplying a
            # canonical row or arbitrarily choosing one reviewer's action.
            report_df = self._deduplicate_workbook_rows(
                report_df, merge_keys_for_report, columns_to_read,
                dropbox_path, report)

            prev_output_df = pd.merge(prev_output_df, report_df,
                on=merge_keys_for_report,
                how='left', suffixes=('', '_dbx'))
            # The canonical frame can contain nullable integer evidence (for
            # example ``date_gap_days``). A frame-wide ``fillna('')`` attempts
            # to put an empty string into Int64 and raises on pandas 1.x/2.x.
            # Normalize by schema instead: text nulls become '', while numeric
            # nulls remain pd.NA and retain their Parquet-safe dtype.
            prev_output_df = self._normalize_output_frame(prev_output_df)

            # When reading a non-authoritative (individual / team) report,
            # an override may only touch flags that do NOT also belong to a
            # protected report. A flag's `reports` is a ' | '-joined string
            # (e.g. "Main Report | Cognition Report"); pad with the delimiter
            # so each protected name is matched as a whole entry, never as a
            # substring of a longer report name.
            override_allowed = None
            if protected_reports:
                if 'reports' in prev_output_df.columns:
                    delim = ' | '
                    padded = delim + prev_output_df['reports'].astype(str) + delim
                    protected_mask = pd.Series(False, index=prev_output_df.index)
                    for pr in protected_reports:
                        protected_mask = protected_mask | padded.str.contains(
                            delim + pr + delim, regex=False)
                    override_allowed = ~protected_mask
                else:
                    # Fail CLOSED. Protection was requested but there is no
                    # `reports` column to identify which flags belong to an
                    # authoritative report (Main Report / Non Team Forms).
                    # Blocking every override this pass is the safe default:
                    # applying them unconditionally could let a team sheet
                    # change an authoritative flag's manual-resolution, which
                    # the authority rule forbids. `reports` is always present
                    # in a well-formed current_output, so this only trips on a
                    # degraded/abnormal schema — surface it loudly.
                    print(
                        "[calculate_resolved_errors] WARNING: protected_reports "
                        f"set but 'reports' column missing in current_output for "
                        f"{dropbox_path} — blocking all overrides this pass to "
                        f"preserve report authority."
                    )
                    override_allowed = pd.Series(False, index=prev_output_df.index)

            # Date flags belong to their dedicated tab. Older workbooks can
            # contain duplicate Date rows on Main or now-excluded report tabs;
            # those copies must not overwrite Date edits (including explicit
            # clears or blank cells that preserve existing reviewer state).
            # Conversely, a stale row copied onto Date cannot edit another
            # report's canonical flag. Match complete routing tokens only.
            if 'reports' in prev_output_df.columns:
                date_owned = prev_output_df['reports'].fillna('').astype(str).map(
                    lambda value: self.DATE_REPORT_NAME in value.split(' | '))
            else:
                date_owned = pd.Series(False, index=prev_output_df.index)
            date_override_allowed = (
                date_owned if report == self.DATE_REPORT_NAME else ~date_owned)
            override_allowed = (
                date_override_allowed if override_allowed is None
                else override_allowed & date_override_allowed)

            for col_to_read in columns_to_read:
                dbx_col = f"{col_to_read}_dbx"
                if dbx_col not in prev_output_df.columns:
                    continue
                col_values = prev_output_df[dbx_col]
                if isinstance(col_values, pd.DataFrame):
                    col_values = col_values.iloc[:, 0]
                text_values = col_values.fillna('').astype(str)
                clear_requested = (
                    text_values.str.strip().str.casefold()
                    == self.CLEAR_VALUE_TOKEN.casefold())
                # Whitespace-only cells are still blank/unedited cells.
                # ``[CLEAR]`` remains the sole explicit erase instruction.
                has_new = (text_values.str.strip().str.len() > 0) | clear_requested
                if override_allowed is not None:
                    has_new = has_new & override_allowed
                values_to_apply = col_values.copy()
                values_to_apply.loc[clear_requested] = ''
                prev_output_df.loc[has_new, col_to_read] = values_to_apply[has_new]

            # Drop merge artifacts so the next iteration / next dropbox file
            # starts clean — otherwise these accumulate and a later merge
            # tries to create a `_dbx` column that already exists.
            drop_cols = [c for c in prev_output_df.columns
                         if (c.endswith('_dbx')
                             or c.startswith('__cross_'))]
            if drop_cols:
                prev_output_df = prev_output_df.drop(columns=drop_cols)

        return prev_output_df

    def determine_resolved_rows(self):
        # Guard against a missing new_output. Two scenarios with very
        # different correct responses:
        #
        #  - True first run: no prior history → no current, no old →
        #    treating new as empty is benign (no rows to merge).
        #
        #  - Interrupted Stage 1, deleted file, or "reports-only"
        #    rerun against an existing baseline: old/current exist
        #    but new is absent → treating new as empty would mark
        #    every previously-open flag as resolved with today's
        #    date, silently destroying QC history.
        #
        # Distinguish by checking for prior history. If old_output OR
        # current_output exists with non-zero size, missing new is
        # almost certainly a mistake; halt with FATAL.
        if not os.path.exists(self.out_paths['new']):
            has_prior_history = any(
                os.path.exists(self.out_paths[k])
                and os.path.getsize(self.out_paths[k]) > 0
                for k in ('old', 'current')
            )
            if has_prior_history:
                raise FileNotFoundError(
                    f"FATAL: new_output missing "
                    f"({self.out_paths['new']}) but prior history "
                    f"exists in old_output and/or current_output. "
                    f"Refusing to merge — proceeding would mark every "
                    f"previously-open flag as resolved with today's "
                    f"date, destroying QC history. Restore new_output "
                    f"(rerun qc_forms_main); if you really intend a "
                    f"reset, delete old_output and current_output "
                    f"explicitly first."
                )
            new_df = pd.DataFrame()
        else:
            new_df = self._normalize_output_frame(
                pd.read_parquet(self.out_paths['new']))
        selected_networks = set(self.utils.pipeline_networks)
        if not new_df.empty:
            if 'network' not in new_df.columns:
                raise RuntimeError(
                    "FATAL: scoped QC new_output has no network column.")
            unexpected = sorted(
                set(new_df['network'].dropna().astype(str))
                - selected_networks)
            if unexpected:
                raise RuntimeError(
                    "FATAL: scoped QC produced rows outside the selected "
                    f"network(s): {unexpected}.")
        untouched_old = pd.DataFrame()
        if os.path.exists(self.out_paths['current']):
            # Promote previous "current" to "old" before merging — old then
            # represents the canonical state at the end of the previous run,
            # which is what the resolved/reopened comparison needs.
            #
            # First validate that current is a readable parquet. A corrupt
            # current (e.g., from an interrupted prior run before the
            # atomic-write fix existed) would otherwise be copied verbatim
            # into old, leaving the user with NO recoverable history. Halt
            # with a clear message instead.
            try:
                pd.read_parquet(self.out_paths['current'], columns=[]).head(0)
            except Exception as e:
                raise RuntimeError(
                    f"current_output is unreadable ({self.out_paths['current']}): {e}."
                    " Refusing to copy corrupt current → old, which would"
                    " destroy recoverable history. Restore from backup or"
                    " delete current_output to start fresh.") from e
            # Stage to .old.tmp first then atomically replace, so a crash
            # mid-copy doesn't leave `old` half-written. (`shutil.copyfile`
            # writes incrementally; `os.replace` on the same volume is
            # atomic on POSIX and effectively atomic on Windows.)
            # PID-stamped tmp so concurrent runs don't collide on the same
            # `.tmp` path. Without this, a second run's copyfile could
            # truncate the first's in-flight write.
            old_tmp = f"{self.out_paths['old']}.{os.getpid()}.tmp"
            shutil.copyfile(self.out_paths['current'], old_tmp)
            os.replace(old_tmp, self.out_paths['old'])
        legacy_cols_to_merge = [
            'subject', 'displayed_form', 'displayed_timepoint',
            'displayed_variable', 'error_message']
        if os.path.exists(self.out_paths['old']):
            old_df = self._normalize_output_frame(
                pd.read_parquet(self.out_paths['old']))
            if not old_df.empty and 'network' not in old_df.columns:
                raise RuntimeError(
                    "FATAL: prior QC history has no network column and "
                    "cannot be preserved safely during a scoped run.")
            if 'network' in old_df.columns:
                in_scope = old_df['network'].isin(selected_networks)
                untouched_old = old_df.loc[~in_scope].copy()
                old_df = old_df.loc[in_scope].copy()
            # Skip the merge entirely if old has no usable schema (e.g., a
            # prior zero-flag run wrote a 0-row, 0-col parquet). Trying to
            # merge on missing keys would KeyError. Treat as no-old-history.
            old_has_schema = all(
                c in old_df.columns for c in legacy_cols_to_merge)
            if old_has_schema:
                # If new_df is empty (no flags this run) it may have no columns,
                # which would KeyError on the merge. Reuse old_df's schema so
                # the merge keys exist; the outer-merge then yields only old
                # rows (each will be marked resolved by compare_old_new_outputs).
                if new_df.empty:
                    new_df = pd.DataFrame(columns=old_df.columns)
                orig_columns = list(new_df.columns)
                old_for_merge = self._add_reconciliation_identity(old_df)
                new_for_merge = self._add_reconciliation_identity(new_df)
                cols_to_merge = [
                    '__history_network', 'subject', 'displayed_timepoint',
                    '__history_check_id',
                    '__history_legacy_form', '__history_legacy_variable',
                    '__history_legacy_message']
                merged = old_for_merge.merge(
                    new_for_merge, on=cols_to_merge, how='outer',
                    suffixes=('_old','_new'))
                merged_df = self._normalize_output_frame(merged)
                new_df = self.compare_old_new_outputs(orig_columns, cols_to_merge, merged_df)
        else:
            # Legitimate for a fresh install or a new experiment directory,
            # catastrophic if prior history should exist (the Aug-2026
            # production state reset silently restarted years of detection
            # history from zero). Loud, but not blocking.
            print(
                f"[calculate_resolved_errors] NOTICE: no prior QC state at "
                f"{self.out_paths['old']} — treating this as a FIRST RUN. "
                f"Every flag will be written as newly detected with no "
                f"resolution history. If prior history should exist, stop "
                f"and restore old_output before the next run overwrites "
                f"detection dates.")
        if not untouched_old.empty:
            new_df = pd.concat(
                [untouched_old, new_df],
                axis=0, ignore_index=True, sort=False)
        # Atomic write of the canonical current_output. Without this, a
        # crash mid-`to_parquet` produces a corrupt parquet that the next
        # stage cannot read — and my consumer-side guard then treats it as
        # zero flags, falsely marking every previously-open flag as resolved.
        self._write_current(new_df)

    def compare_old_new_outputs(self, orig_columns, cols_to_merge, merged_df):
        # Release-hardening (RH-2): reset accumulator at top of method.
        # Currently safe because GenerateReports instantiates
        # CalculateResolvedErrors fresh per run, but explicit reset is
        # a defensive belt against the same instance being reused
        # across runs (e.g. if a future caller calls run_script twice).
        # Without this, the second call would emit duplicate rows.
        self.new_output = []
        curr_date = str(datetime.today().date())
        # Count detections and auto-resolutions per (network, timepoint)
        # so changes in reconciliation volume remain visible in diagnostics.
        newly_detected = {}
        auto_resolved = {}
        prev_open = {}
        # The merged frame's column set is constant for the whole method.
        # Precompute it once so append_all_cols does an O(1) set lookup
        # instead of a `(col + suffix) in merged_df.columns` pandas-Index
        # membership test on every one of the ~2M rows × n_cols.
        present_cols = set(merged_df.columns)
        # CSV round-trip turns booleans into the strings 'True'/'False', so
        # `row.currently_resolved_old == True` was previously almost always
        # False even for resolved rows, which broke re-opening logic.
        truthy = (True, 'True', 'true', 'TRUE', 1, '1')
        for row in merged_df.itertuples():
            curr_row_output = {}
            resolved_old = getattr(row, 'currently_resolved_old', '') in truthy
            # if col only exists in new output
            if row.network_new != '' and row.network_old == '':
                detected_key = (row.network_new, row.displayed_timepoint)
                newly_detected[detected_key] = (
                    newly_detected.get(detected_key, 0) + 1)
                # sets resolved to false
                curr_row_output['currently_resolved'] = False
                # adds current date to dates_detected
                curr_row_output['dates_detected'] = self.append_formatted_list(
                row.dates_detected_old, curr_date)
                # adds all other columns of the new output
                curr_row_output = self.append_all_cols(
                row, curr_row_output, orig_columns, '_new', cols_to_merge,present_cols)
            elif row.network_old != '':
                # if exists in old and not new
                if row.network_new == '':
                    if resolved_old:
                        # if it is still resolved, add all columns from old
                        curr_row_output = self.append_all_cols(row, curr_row_output,
                        orig_columns, '_old',cols_to_merge,present_cols)
                    else:
                        resolved_key = (
                            row.network_old, row.displayed_timepoint)
                        auto_resolved[resolved_key] = (
                            auto_resolved.get(resolved_key, 0) + 1)
                        prev_open[row.network_old] = (
                            prev_open.get(row.network_old, 0) + 1)
                        # if it was not resolved, set it to resolved
                        curr_row_output['currently_resolved'] = True
                        # adds current date to dates resolved
                        curr_row_output['dates_resolved'] = self.append_formatted_list(
                        row.dates_resolved_old, curr_date)
                        # adds all other columns from old output
                        curr_row_output = self.append_all_cols(row, curr_row_output,
                        orig_columns, '_old', cols_to_merge,present_cols)
                # if exists in old and new
                elif row.network_new != '':
                    if not resolved_old:
                        prev_open[row.network_old] = (
                            prev_open.get(row.network_old, 0) + 1)
                    # sets currently_resolved to false
                    curr_row_output['currently_resolved'] = False
                    if resolved_old:
                        # if it was resolved in the old output
                        # add current date to dates_detected
                        curr_row_output['dates_detected'] = self.append_formatted_list(
                        row.dates_detected_old, curr_date)
                    # Reviewer state is canonical state, not regenerated QC
                    # evidence.  Preserve it from the old row when a flag is
                    # still present; Dropbox imports then apply explicit edits.
                    # This also makes blank legacy cells true no-ops, while the
                    # [CLEAR] token remains the unambiguous way to erase state.
                    old_state_columns = [
                        'dates_resolved', 'dates_detected',
                        'comments', 'site_comments']
                    # A continuously-open flag keeps its review disposition.
                    # If a previously resolved condition genuinely reappears,
                    # require a fresh manual resolution instead of silently
                    # carrying a stale one onto the new occurrence.
                    #
                    # Conversion Report flags are exempt. They fire off the
                    # floating row and depend on cross-timepoint booleans
                    # stamped onto subject_info upstream; conversion_checks
                    # SKIPS (rather than fires) whenever those keys are absent,
                    # so a conversion flag can drop out for one run -- being
                    # auto-resolved -- and return the next, with nothing about
                    # the underlying data having changed. Resetting the
                    # reviewer's mark on that round trip silently reopens work
                    # that was genuinely resolved, so carry it forward.
                    if (not resolved_old
                            or self._row_belongs_to_report(
                                row, self.CONVERSION_REPORT_NAME)):
                        old_state_columns.append('manually_resolved')
                    for old_col in old_state_columns:
                        if old_col not in curr_row_output.keys():
                            # if dates_resolved and dates_detected not in curr output
                            # then will add them from the old output.
                            # NOTE: for the still-open (NOT resolved_old) path
                            # dates_detected is intentionally NOT appended with
                            # today — `time_since_last_detection` is then
                            # "days since first/most-recent reopen detection"
                            # which is the staleness signal reviewers use to
                            # rank long-open issues. Appending today every run
                            # would collapse that signal to 0.
                            old_name = old_col + '_old'
                            if old_name in present_cols:
                                curr_row_output[old_col] = getattr(row, old_name)
                    # adds all remaining columns from new output
                    curr_row_output = self.append_all_cols(row,
                    curr_row_output, orig_columns, '_new',cols_to_merge,present_cols)
            dates_detected = str(curr_row_output.get('dates_detected', '')).split(' | ')
            most_recent_detection = dates_detected[-1]
            # `days_since_today` calls strptime and crashes on '' or any
            # non-`%Y-%m-%d` string. This can happen if `dates_detected_old`
            # was empty (e.g., a row carried over from before this column
            # was populated) and got copied through via `append_all_cols`.
            # Default to '' to mean "unknown" rather than crashing the run.
            if (most_recent_detection
                and self.utils.check_if_val_date_format(most_recent_detection)):
                curr_row_output['time_since_last_detection'] = self.utils.days_since_today(
                str(most_recent_detection))
            else:
                curr_row_output['time_since_last_detection'] = ''
            self.new_output.append(curr_row_output)
        networks_seen = sorted(
            {key[0] for key in newly_detected}
            | {key[0] for key in auto_resolved}
            | set(prev_open))
        for network in networks_seen:
            detected_total = sum(
                count for key, count in newly_detected.items()
                if key[0] == network)
            resolved_total = sum(
                count for key, count in auto_resolved.items()
                if key[0] == network)
            open_before = prev_open.get(network, 0)
            print(
                f"[calculate_resolved_errors] {network}: "
                f"{detected_total} newly detected, "
                f"{resolved_total} auto-resolved, "
                f"{open_before} previously open")
            for (net, tp), count in sorted(auto_resolved.items()):
                if (net == network and count >= 50
                        and (network, tp) not in newly_detected):
                    print(
                        f"[calculate_resolved_errors] ERROR: {network}/{tp}: "
                        f"{count} flag(s) auto-resolved with ZERO new "
                        f"detections at this timepoint — the signature of a "
                        f"timepoint that was skipped or produced no checker "
                        f"output this run.")
        new_df = pd.DataFrame(self.new_output)
        return new_df
    
    def append_formatted_list(self, curr_list_string, item_to_append):
        new_list = []
        if curr_list_string == '':
            return item_to_append
        else:
            new_list = curr_list_string.split(' | ')
            # Dedupe against the last entry — re-running on the same day
            # would otherwise grow the list unboundedly with the same date.
            if new_list and new_list[-1] == item_to_append:
                return curr_list_string
            new_list.append(item_to_append)
            new_list_string = ' | '.join(new_list)

            return new_list_string
            
    def append_all_cols(self, row, curr_row_output, all_cols, suffix, merged_cols, present_cols):
        for col in all_cols:
            if col not in curr_row_output.keys():
                if col not in merged_cols:
                    if (col + suffix) in present_cols:
                        curr_row_output[col] = getattr(row, col + suffix)
                    # A column present in only ONE side of the old/new outer
                    # merge arrives UN-suffixed (bare `col`, no _old/_new,
                    # because pandas only suffixes columns present in BOTH
                    # frames). Without this fallback a freshly added field —
                    # present in new_output but not yet in old_output, e.g.
                    # `cohort` — is silently dropped and never reaches
                    # current_output (permanently, since current is promoted
                    # to old each run). Only fires for single-side columns;
                    # in steady state both sides carry `col`, it is suffixed,
                    # and the branch above wins — so this is a no-op then.
                    # (Lifts the append_all_cols limitation noted in
                    # generate_reports/date_report_sort.py.)
                    elif col in present_cols:
                        curr_row_output[col] = getattr(row, col)
                else:
                    curr_row_output[col] =  getattr(row, col)

        return curr_row_output
