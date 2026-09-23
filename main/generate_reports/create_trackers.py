import pandas as pd
import os
import sys
import json
import dropbox
from openpyxl.styles import Border, Side, PatternFill, Font, Alignment, colors, Protection,Color
from openpyxl.worksheet.dimensions import ColumnDimension
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils import range_boundaries,get_column_letter
from openpyxl import load_workbook,Workbook
import numpy as np
from openpyxl.styles.differential import DifferentialStyle
from openpyxl.formatting.rule import Rule
import openpyxl
import dropbox
import webbrowser
import base64
import requests
import json
from io import BytesIO
import ast
from openpyxl.formatting.formatting import ConditionalFormattingList
import re
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from generate_reports.tracker_paths import sanitize_ra_folder_name
from generate_reports.date_report_sort import sort_date_report_rows
from qc_types.date_check_logic import (
    is_date_report_excluded_form,
    is_date_report_excluded_variable,
)
import time
from functools import wraps
import tempfile
import shutil
from io import BytesIO

class CreateTrackers():

    # Report (tab) name for the cross-timepoint backward-date flags.
    # Must match DateChecks.DATE_REPORT in qc_types/date_checks.py (the
    # report name is the contract between the qc_type and this tab, the
    # same way 'Main Report' / 'Blood Report' are shared literals).
    DATE_REPORT_NAME = 'Date Report'
    CROSS_CHECK_REPORT_NAME = 'Cross Checks'
    PROPOSED_CHECKS_REPORT_NAME = 'Proposed Checks'
    MEDICATION_FLAGS_REPORT_NAME = 'Medication Flags'
    PROPOSED_EVIDENCE_LABEL = 'Variables & values v3:'
    TRACKER_FILENAME_SUFFIX = '_Output_V2.xlsx'
    # Date findings have a dedicated reviewer tab in the deployed V2 files.
    # The other excluded report families remain canonical-data-only.
    EXCLUDED_REPORT_NAMES = frozenset({
        CROSS_CHECK_REPORT_NAME, PROPOSED_CHECKS_REPORT_NAME,
        MEDICATION_FLAGS_REPORT_NAME})
    ALWAYS_PRESENT_REPORTS = frozenset({DATE_REPORT_NAME})
    KNOWN_REPORT_TABS = frozenset({
        'Main Report', 'Secondary Report', 'Non Team Forms',
        'Missingness Report', 'Conversion Report', 'Incomplete Forms',
        'Scid Report', 'Cognition Report', 'Digital Report',
        'Blood Report', 'Fluids Report', 'MRI Report', 'EEG Report',
        DATE_REPORT_NAME})

    @staticmethod
    def _routed_row_mask(df):
        """Rows intentionally routed to at least one reviewer surface.

        QC history retains rows for withdrawn/excluded/non-recruited subjects,
        but FormCheck clears their reports field. The PRONET family whitelist
        must not accidentally make those suppressed rows visible.
        """
        if 'reports' not in df.columns:
            return pd.Series(False, index=df.index, dtype=bool)
        return df['reports'].fillna('').astype(str).str.strip().ne('')

    @staticmethod
    def _report_token_mask(df, report):
        """Rows routed to one exact pipe-delimited report name."""
        if 'reports' not in df.columns:
            return pd.Series(False, index=df.index, dtype=bool)
        return df['reports'].fillna('').astype(str).map(
            lambda value: report in value.split(' | '))


    @staticmethod
    def _normalize_tracker_frame(df):
        """Fill text nulls without corrupting nullable numeric columns.

        ``current_output`` includes optional integer evidence such as
        ``date_gap_days`` and ``time_since_last_detection``. pandas rejects a
        frame-wide ``fillna('')`` when one of those columns uses nullable Int64
        (``ValueError: invalid literal for int() with base 10: ''``). Tracker
        logic still expects absent text values to be empty strings, so fill only
        genuinely textual columns and leave numeric missing values as pd.NA.
        """
        out = df.copy()
        for col in out.columns:
            ser = out[col]
            if not (pd.api.types.is_object_dtype(ser.dtype)
                    or pd.api.types.is_string_dtype(ser.dtype)):
                continue
            populated = ser.dropna()
            if populated.empty or populated.map(
                    lambda value: isinstance(value, str)).all():
                out[col] = ser.fillna('')
        return out

    @staticmethod
    def _date_report_metadata_items(value):
        """Read list-like or serialized tracker metadata into exact tokens."""
        if (not isinstance(value, (str, bytes, dict))
                and pd.api.types.is_list_like(value)):
            return [str(item).strip() for item in value]

        raw = '' if pd.isna(value) else str(value).strip()
        if not raw:
            return []

        # NumPy stringification omits commas (['a' 'b']), which ast treats as
        # one pair of adjacent string literals. Recover quoted items first so
        # historical CSV/Parquet transitions cannot concatenate form names.
        if ((raw.startswith('[') and raw.endswith(']'))
                or (raw.startswith('(') and raw.endswith(')'))):
            quoted = [left or right for left, right in re.findall(
                r"'([^']*)'|\"([^\"]*)\"", raw)]
            if quoted:
                return [item.strip() for item in quoted]

        parsed = None
        if raw[:1] in '[(' and raw[-1:] in '])':
            try:
                parsed = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                parsed = None
        if (parsed is not None
                and not isinstance(parsed, (str, bytes, dict))
                and pd.api.types.is_list_like(parsed)):
            return [str(item).strip() for item in parsed]
        return [item.strip() for item in raw.split('|')]

    @classmethod
    def _contains_excluded_date_report_form(cls, value):
        """Whether an affected/displayed-form cell names an excluded form."""
        return any(
            is_date_report_excluded_form(form)
            for form in cls._date_report_metadata_items(value))

    @classmethod
    def _contains_excluded_date_report_variable(cls, value):
        """Whether variable metadata identifies an excluded date source."""
        return any(
            (is_date_report_excluded_variable(variable)
             or is_date_report_excluded_form(variable))
            for variable in cls._date_report_metadata_items(value))

    @staticmethod
    def _message_contains_excluded_date_report_variable(value):
        """Find excluded field/form tokens retained only in legacy messages."""
        raw = '' if pd.isna(value) else str(value)
        return any(
            (is_date_report_excluded_variable(token)
             or is_date_report_excluded_form(token))
            for token in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', raw))

    @classmethod
    def _exclude_date_report_form_rows(cls, df):
        """Drop historical/current Date Report rows involving target forms."""
        if df.empty:
            return df.copy()
        excluded = pd.Series(False, index=df.index, dtype=bool)
        for column in ('affected_forms', 'displayed_form'):
            if column in df.columns:
                excluded |= df[column].map(
                    cls._contains_excluded_date_report_form)
        for column in ('affected_variables', 'displayed_variable'):
            if column in df.columns:
                excluded |= df[column].map(
                    cls._contains_excluded_date_report_variable)
        if 'error_message' in df.columns:
            excluded |= df['error_message'].map(
                cls._message_contains_excluded_date_report_variable)
        return df.loc[~excluded].copy()

    @classmethod
    def _date_report_row_has_exact_variable_match(cls, row):
        """Whether a Date Report row proves both dates use one exact field.

        Current rows name the earlier field in the message and retain both
        fields in ``affected_variables``. Require the message proof and reject
        any contradictory metadata. Historical rows whose earlier field is
        unknowable fail closed.
        """
        displayed = [item for item in cls._date_report_metadata_items(
            row.get('displayed_variable', '')) if item]
        affected = [item for item in cls._date_report_metadata_items(
            row.get('affected_variables', '')) if item]
        current_variable = displayed[0] if displayed else (
            affected[0] if affected else '')

        message = row.get('error_message', '')
        message = '' if pd.isna(message) else str(message)
        previous_match = re.search(
            r'\bbefore\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', message)
        distinct_affected = set(affected)
        if not current_variable:
            return False
        if distinct_affected and distinct_affected != {current_variable}:
            return False
        # A singleton legacy metadata field can identify only the current
        # source. Require the standard message's earlier-field token as
        # independent proof; unknowable historical rows fail closed.
        return bool(
            previous_match is not None
            and previous_match.group(1) == current_variable)

    @classmethod
    def _exclude_cross_variable_date_report_rows(cls, df):
        """Hide current or historical date rows that compare two fields."""
        if df.empty:
            return df.copy()
        keep = df.apply(
            cls._date_report_row_has_exact_variable_match, axis=1)
        return df.loc[keep].copy()

    @classmethod
    def _select_date_report_rows(cls, df):
        """Select the exact rows eligible for the Date Report surface."""
        selected = df.loc[
            cls._report_token_mask(df, cls.DATE_REPORT_NAME)].copy()
        selected = cls._exclude_date_report_form_rows(selected)
        return cls._exclude_cross_variable_date_report_rows(selected)


    @classmethod
    def _select_main_report_rows(cls, network_df):
        """Keep Date Report findings out of Main, including old dual routes."""
        return network_df.loc[
            cls._report_token_mask(network_df, 'Main Report')
            & ~cls._report_token_mask(network_df, cls.DATE_REPORT_NAME)
        ].copy()

    @classmethod
    def _read_combined_tracker(cls, path):
        return cls._normalize_tracker_frame(pd.read_parquet(path))

    def __init__(self, formatted_col_names):
        self.utils = Utils()
        with open(f'{self.utils.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.output_path = self.config_info['paths']['output_path']
        if self.config_info["testing_enabled"] == "True":
            self.output_path += "testing/"
            self.dropbox_path = f'/Apps/Automated QC Trackers/refactoring_tests/'
        else:
            self.dropbox_path = f'/Apps/Automated QC Trackers/'
        self.all_reports = ['Main Report', 'Secondary Report', self.DATE_REPORT_NAME]
        self.site_reports = ['Main Report', self.DATE_REPORT_NAME]
        self.all_report_df = {}
        self.all_pronet_sites = self.utils.all_pronet_sites
        self.all_prescient_sites = self.utils.all_prescient_sites
        self.all_sites = self.utils.all_sites
        self.site_translations = self.utils.site_full_name_translations
        self.old_output_csv_path = f'{self.output_path}combined_outputs/old_output/combined_qc_flags.parquet'
        self.curr_output_csv_path = f'{self.output_path}combined_outputs/current_output/combined_qc_flags.parquet'
        self.formatted_outputs_path = f'{self.output_path}formatted_outputs/'
        if not os.path.exists(self.formatted_outputs_path):
            os.makedirs(self.formatted_outputs_path)
        self.dropbox_output_path =  f'{self.output_path}formatted_outputs/dropbox_files/'
        self.colors = {"green" : PatternFill(start_color='b9d9b4', end_color='b9d9b4', fill_type='gray125'),
                       "yellow" : PatternFill(start_color='e8e7b0', end_color='e8e7b0', fill_type='gray125'),
                       "blue" : PatternFill(start_color='d8cffc', end_color='d8cffc', fill_type='gray125'),
                       "orange" : PatternFill(start_color='ebba83', end_color='ebba83', fill_type='gray125'),
                       "red" : PatternFill(start_color='de9590', end_color='de9590', fill_type='gray125'),
                       "grey" : PatternFill(start_color='ededed', end_color='ededed', fill_type='gray125'),
                       "pink" : PatternFill(start_color='F0A1F0', end_color='F0A1F0', fill_type='gray125')}
        self.thin_border = Border(left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'),bottom=Side(style='thin'))
        self.formatted_column_names = formatted_col_names
        self.melbourne_ras = self.utils.load_dependency_json('melbourne_ra_subs.json')

        self.master = pd.DataFrame()
        # Workbook path -> set of sheet names written this run; consumed by
        # blank_stale_sheets so no pipeline-owned tab can ship stale.
        self._sheets_written_by_path = {}

    def run_script(self):
        self.combined_tracker = self._read_combined_tracker(
            self.curr_output_csv_path)
        print('stage 1')
        self.collect_new_reports()
        print('stage 2')
        self.generate_reports()
        self.blank_stale_sheets()
        print('stage 3')
        self.upload_trackers()
        #self.append_recovered_comments('PRESCIENT')
 
    def collect_new_reports(self):
        scoped_tracker = self.combined_tracker[
            self.combined_tracker['network'].isin(self.utils.pipeline_networks)]
        for reports in scoped_tracker['reports'].fillna('').astype(str):
            for report in reports.split(' | '):
                if (report and report not in self.EXCLUDED_REPORT_NAMES
                        and report not in self.all_reports):
                    self.all_reports.append(report)

    def generate_reports(self):
        report_plan = list(dict.fromkeys([
            *self.all_reports, *sorted(self.ALWAYS_PRESENT_REPORTS)]))
        for network in self.utils.pipeline_networks:
            network_df = self.combined_tracker[
                self.combined_tracker['network'] == network]
            for report in report_plan:
                if report in self.EXCLUDED_REPORT_NAMES:
                    continue
                if report == self.DATE_REPORT_NAME:
                    report_df = self._select_date_report_rows(network_df)
                elif report == 'Main Report':
                    report_df = self._select_main_report_rows(network_df)
                else:
                    report_df = network_df.loc[
                        self._report_token_mask(network_df, report)]
                self.all_report_df[report] = report_df
                raw_row_count = len(report_df)
                if report_df.empty:
                    if report not in self.ALWAYS_PRESENT_REPORTS:
                        continue
                    # A clean run replaces stale findings with a header-only tab.
                    report_df = self._header_only_report_df(network, report)
                else:
                    report_df = self.convert_to_shared_format(report_df, network)
                if report == self.DATE_REPORT_NAME:
                    report_df = sort_date_report_rows(report_df)
                    print(
                        f"[create_trackers] {network} Date Report: "
                        f"{raw_row_count} routed QC row(s), "
                        f"{len(report_df)} workbook row(s)")
                combined_path = (
                    f'{self.dropbox_output_path}{network}/combined/')
                os.makedirs(combined_path, exist_ok=True)
                self.format_excl_sheet(
                    report_df, report, combined_path,
                    f'{network}_Output_V2.xlsx')
                self.loop_sites(network, report, report_df)

    def _header_only_report_df(self, network, report):
        report_columns = list(
            self.formatted_column_names[network]["combined"].values())
        if (report == self.DATE_REPORT_NAME
                and 'Days Apart' not in report_columns):
            report_columns.append('Days Apart')
        if (report == self.CROSS_CHECK_REPORT_NAME
                and 'Check ID' not in report_columns):
            report_columns.append('Check ID')
        return pd.DataFrame(columns=report_columns)

    def blank_stale_sheets(self):
        """Clear stale pipeline-owned tabs in workbooks regenerated this run."""
        for full_path in sorted(self._sheets_written_by_path):
            written = self._sheets_written_by_path[full_path]
            filename = os.path.basename(full_path)
            if not filename.endswith(self.TRACKER_FILENAME_SUFFIX):
                continue
            network = filename.split('_', 1)[0]
            if (network not in self.formatted_column_names
                    or not os.path.exists(full_path)):
                continue
            workbook = load_workbook(full_path, read_only=True)
            existing_sheets = set(workbook.sheetnames)
            workbook.close()
            folder = full_path[:len(full_path) - len(filename)]
            for report in sorted(
                    (existing_sheets & self.KNOWN_REPORT_TABS) - written):
                print(
                    f"[create_trackers] WARNING: blanking stale sheet "
                    f"{report!r} in {full_path}; no rows were routed to it.")
                self.format_excl_sheet(
                    self._header_only_report_df(network, report),
                    report, folder, filename)

    def loop_sites(self, network, report, report_df):
        # Pre-validate the Participant column once: any subject whose first
        # two characters do not match a known site code in this network would
        # silently land in the WRONG site's tracker (a PHI access-control
        # concern: data-entry typos route a subject's flags to RAs at a
        # different site). Log them and exclude from per-site trackers; they
        # still appear in the network combined tracker.
        valid_site_codes = set(self.all_sites[network])
        prefixes = report_df['Participant'].astype(str).str[:2]
        unknown_prefix_mask = ~prefixes.isin(valid_site_codes)
        if unknown_prefix_mask.any():
            unknown_subjects = sorted(
                report_df.loc[unknown_prefix_mask, 'Participant'].astype(str).unique()
            )
            print(f"[create_trackers] WARNING: {len(unknown_subjects)} subject(s)"
                  f" in network {network} have a Participant prefix that does"
                  f" not match any known site code; excluding from per-site"
                  f" trackers (still in network combined): {unknown_subjects[:10]}"
                  f"{'...' if len(unknown_subjects) > 10 else ''}")

        for site_abr in self.all_sites[network]:
            if site_abr in self.utils.site_full_name_translations.keys():
                site = self.utils.site_full_name_translations[site_abr]
            else:
                site = site_abr
            # Date findings are available to each site's reviewers, including
            # Melbourne and its RA workbooks; other report routing is unchanged.
            allowed_reports = (
                {'Non Team Forms', self.DATE_REPORT_NAME} if site_abr == 'ME'
                else {'Main Report', self.DATE_REPORT_NAME})
            if report not in allowed_reports:
                continue
            if site_abr == 'ME':
                self.loop_ras(network, site, report, report_df)
            site_path = f'{self.dropbox_output_path}{network}/{site}/'
            if not os.path.exists(site_path):
                os.makedirs(site_path)
            # Exact-prefix match against THIS site's code only — combined
            # with the unknown-prefix log above, this guarantees a subject
            # only ever lands in the correct site's file.
            site_df = report_df[prefixes == site_abr]
            site_filename = f'{network}_{site_abr}_Output_V2.xlsx'
            self.format_excl_sheet(
                site_df, report, site_path, site_filename)

    def loop_ras(self, network, site, report, report_df):
        for ra, subjects in self.melbourne_ras.items():
            # Sanitize RA name before using as a path component. The source
            # is `RAname` from raw PRESCIENT CSVs (free-text). Without
            # stripping path separators, a value containing `/` or `..`
            # could escape the intended directory and overwrite a sibling
            # RA's tracker. Allow only word chars + dashes; cap length.
            safe_ra = sanitize_ra_folder_name(ra)
            ra_path = f'{self.dropbox_output_path}{network}/{site}/{safe_ra}/'
            ra_df = report_df[report_df['Participant'].isin(subjects)]
            self.format_excl_sheet(ra_df,
            report, ra_path,
            f'{network}_Melbourne_Output_V2.xlsx')

    def upload_trackers(self):
        fullpath = os.path.join(
            self.output_path, 'formatted_outputs', 'dropbox_files')
        dbx = self.utils.collect_dropbox_credentials()
        upload_failures = []
        for root, dirs, files in os.walk(fullpath):
            for filename in files:
                full_path = os.path.join(root, filename)
                local_path = os.path.relpath(
                    full_path, fullpath).replace(os.sep, '/')
                network = local_path.split('/', 1)[0]
                # Only this run's selected networks and deployed V2 filenames
                # are eligible. Never upload stale V1 sheet-only donor files.
                if (network not in self.utils.pipeline_networks
                        or not filename.endswith(self.TRACKER_FILENAME_SUFFIX)
                        or not filename.startswith(network + '_')):
                    continue
                try:
                    self.save_to_dropbox(full_path, local_path, dbx=dbx)
                except Exception as exc:
                    upload_failures.append((local_path, str(exc)))
                    print(f"[create_trackers] upload failed for {local_path}: {exc}")
        if upload_failures:
            raise RuntimeError(
                f"[create_trackers] FATAL: {len(upload_failures)} tracker(s) "
                f"failed to upload. Dropbox is partially updated; rerun the "
                f"reports pipeline after investigating: {upload_failures[:5]}")

    def format_excl_sheet(self, df, report, folder, filename):
        print('formatting')
        full_path = folder + filename
        print(folder + filename)
        # Lazy-init: tests construct this class via object.__new__ and call
        # format_excl_sheet directly, bypassing __init__.
        if not hasattr(self, '_sheets_written_by_path'):
            self._sheets_written_by_path = {}
        self._sheets_written_by_path.setdefault(full_path, set()).add(report)
        if not os.path.exists(folder):
            os.makedirs(folder)

        if not os.path.exists(folder + filename):
            tmp_new = f"{full_path}.{os.getpid()}.new.tmp.xlsx"
            try:
                df.to_excel(tmp_new, sheet_name=report, index=False)
                os.replace(tmp_new, full_path)
            except Exception:
                if os.path.exists(tmp_new):
                    try:
                        os.remove(tmp_new)
                    except OSError:
                        pass
                raise

        with pd.ExcelWriter(full_path, mode='a',\
        engine='openpyxl',if_sheet_exists = 'replace') as writer:                
            df.to_excel(writer, sheet_name=report, index=False)

        # Load from an in-memory copy so openpyxl never retains a handle to the
        # canonical path. A path-backed workbook prevented the atomic
        # os.replace below on Windows even after Workbook.close().
        with open(full_path, 'rb') as source_workbook:
            workbook_buffer = BytesIO(source_workbook.read())
        workbook = load_workbook(workbook_buffer)
        worksheet = workbook[report]
        worksheet = self.change_excel_colors(worksheet)
        worksheet = self.change_excel_column_sizes(worksheet)
        if report == self.PROPOSED_CHECKS_REPORT_NAME:
            flags_col = self.find_col_letter(worksheet, 'Flags')
            if flags_col is not None:
                worksheet.column_dimensions[flags_col].width = 60
                for cell in worksheet[flags_col][1:]:
                    cell.alignment = Alignment(
                        wrap_text=True, vertical='top')
        # Check ID is a machine identity used to preserve reviewer state when
        # a Cross Checks label/form is clarified.  Keep it in the workbook for
        # round-trip safety but hide it from the normal reviewer surface so it
        # is not mistaken for an editable field.
        check_id_col = self.find_col_letter(worksheet, 'Check ID')
        if check_id_col is not None:
            worksheet.column_dimensions[check_id_col].hidden = True
        # Atomic publish: save to PID-stamped tmp then replace, so a crash
        # mid-write does not leave a truncated xlsx as the canonical tracker.
        tmp_wb = f"{full_path}.{os.getpid()}.tmp.xlsx"
        try:
            workbook.save(tmp_wb)
            # Release the source workbook before replacing it. POSIX permits
            # replacing an open file, but Windows raises PermissionError and
            # leaves the freshly generated tracker unpublished.
            workbook.close()
            workbook_buffer.close()
            os.replace(tmp_wb, full_path)
        except Exception:
            try:
                workbook.close()
            except Exception:
                pass
            try:
                workbook_buffer.close()
            except Exception:
                pass
            if os.path.exists(tmp_wb):
                try:
                    os.remove(tmp_wb)
                except OSError:
                    pass
            raise

        #if not os.path.exists(folder + filename):
        #df.to_excel(folder + filename, sheet_name = report, index = False)

    def change_excel_colors(self, worksheet):
        # Build the {column_index: header_value} map ONCE per worksheet.
        # Previously this loop and each of the four color helpers re-fetched
        # the row-1 header cell per data cell via worksheet.cell(row=1, ...),
        # ~5x per data cell across the whole sheet × ~456 tracker files.
        header_by_col = {cell.column: cell.value for cell in worksheet[1]}
        for row in worksheet.iter_rows():
            cell_color = self.colors['grey']
            # the order of this list determines which colors
            # override others
            for color in [self.time_based_color(row,worksheet,header_by_col),
            self.color_priority_items(row,worksheet,header_by_col),
            self.determine_resolved_color(row,worksheet,'Date Resolved','green',header_by_col),
            self.determine_resolved_color(row,worksheet,'Manually Resolved','blue',header_by_col)]:
                if color != None:
                    cell_color = color
            for cell in row:
                cell.border = self.thin_border
                header_value = header_by_col.get(cell.column)
                if header_value in ['Flag Count','Flags','Form']:
                    cell.fill = cell_color
                else:
                    cell.fill = self.colors['grey']

        return worksheet

    def time_based_color(self, excel_row, worksheet, header_by_col):
        for cell in excel_row:
            header_value = header_by_col.get(cell.column)
            cell_val = str(cell.value)
            if cell_val == 'None':
                cell_val = ''
            if header_value == 'Days Since Detected':
                if self.utils.can_be_float(cell_val):
                    days_since_detected = int(float(cell_val))
                    if days_since_detected < 7:
                        return self.colors['yellow']
                    elif 7 <= days_since_detected < 14:
                        return self.colors['orange']
                    else:
                        return self.colors['red']
        return None

    def color_priority_items(self, excel_row, worksheet, header_by_col):
        for cell in excel_row:
            header_value = header_by_col.get(cell.column)
            cell_val = str(cell.value)
            if cell_val == 'None':
                cell_val = ''
            if header_value == 'Priority Item':
                if cell_val == 'True':
                    return self.colors['pink']
        return None

    def determine_resolved_color(self, excel_row,
    worksheet, col_to_check, color_to_return, header_by_col):
        for cell in excel_row:
            cell.fill = self.colors['grey']
            header_value = header_by_col.get(cell.column)
            cell_val = str(cell.value)
            if cell_val == 'None':
                cell_val = ''
            if header_value == col_to_check:
                if cell_val !='' and cell_val != header_value:
                    color = self.colors[color_to_return]
                    return color

        return None

    def change_excel_column_sizes(self,worksheet):
        columns_sizes = {
            'Participant' : 10,
            'Cohort' : 10,
            'Timepoint' : 10,
            'Flag Count' : 10,
            'Form' : 35,
            'Flags' : 35,
            'Translations' : 35,
            'Days Since Detected' : 25,
            'Date Resolved': 20,
            'Manually Resolved' : 20,
            'Comments' : 20,
            'Priority Item' : 20,
            # "Date Report" tab only (harmless elsewhere — find_col_letter
            # skips headers not present on a sheet).
            'Days Apart' : 12
        }
        for header, length in columns_sizes.items():
            col_letter = self.find_col_letter(worksheet, header)
            if col_letter != None:  
                worksheet.column_dimensions[col_letter].width = length

        return worksheet

    def find_col_letter(self, worksheet, col_name):
        for cell in worksheet[1]:  
            if cell.value == col_name:
                column_letter = get_column_letter(cell.column)
                return column_letter
        return None
        
    def upload_files_to_dropbox(self):
        pass

    def merge_rows(self, values):
        unique_values  = list(dict.fromkeys(values))

        if len(unique_values) == 1:
            return values.iloc[0]
        else:
            return ' | '.join(unique_values)
        
    def first_true(self,series):
        for value in series:
            if value==True:
                return value
        # If no value satisfies the condition, return the first value
        return series.iloc[0]

    def convert_to_shared_format(self, raw_df, network):
        # Work with a copy: adding the Cross Checks machine identifier must not
        # mutate the shared combined/sites mapping or current_col_names.json.
        columns_names = dict(
            self.formatted_column_names[network]["combined"])
        # Guarantee a `cohort` column exists BEFORE the groupby/agg and the
        # hard column-selection (`merged_df[list(columns_names.values())]`)
        # below. current_output normally carries cohort, but a run against a
        # pre-cohort current_output, or a zero-flag / stale new_output run
        # (where reconciliation rebuilds the frame from the cohort-less old
        # schema), would otherwise lack it and KeyError on 'Cohort' — crashing
        # the whole reports stage. Placed above the agg_args loop so cohort is
        # aggregated ('first'); assign() (not in-place) avoids SettingWithCopy
        # on a possible slice.
        if 'cohort' not in raw_df.columns:
            raw_df = raw_df.assign(cohort='')
        is_cross_checks_frame = (
            'reports' in raw_df.columns
            and not raw_df.empty
            and raw_df['reports'].map(
                lambda value: self.CROSS_CHECK_REPORT_NAME
                in str(value).split(' | ')).all())
        if is_cross_checks_frame:
            extracted_check_id = (
                raw_df['error_message'].astype(str).str.extract(
                    r'\[(CROSS-QC-\d{3})\]', expand=False).fillna(''))
            if 'check_id' in raw_df.columns:
                existing_check_id = (
                    raw_df['check_id'].fillna('').astype(str).str.strip())
                raw_df = raw_df.assign(
                    check_id=existing_check_id.where(
                        existing_check_id.ne(''), extracted_check_id))
            else:
                raw_df = raw_df.assign(check_id=extracted_check_id)
            columns_names['check_id'] = 'Check ID'
        is_proposed_checks_frame = (
            'reports' in raw_df.columns
            and not raw_df.empty
            and raw_df['reports'].map(
                lambda value: self.PROPOSED_CHECKS_REPORT_NAME
                in str(value).split(' | ')).all())
        if (is_proposed_checks_frame
                and 'proposed_variable_values' in raw_df.columns):
            evidence = (
                raw_df['proposed_variable_values']
                .fillna('').astype(str).str.strip())
            messages = raw_df['error_message'].fillna('').astype(str)
            raw_df = raw_df.assign(
                error_message=messages.where(
                    evidence.eq(''),
                    messages + '\n' + self.PROPOSED_EVIDENCE_LABEL + ' '
                    + evidence))
        columns_to_match = ['subject','displayed_timepoint','displayed_form',
                            'currently_resolved','manually_resolved']
        # Legacy CSV/Parquet history can contain either booleans or their
        # string forms. `== True` misses the string `'True'` form, so match the
        # truthy pattern used by resolved-state reconciliation as well.
        truthy = (True, 'True', 'true', 'TRUE', 1, '1')
        is_resolved = raw_df['currently_resolved'].isin(truthy)
        raw_df.loc[:, 'date_resolved'] = ''
        raw_df.loc[is_resolved, 'date_resolved'] = (
            raw_df.loc[is_resolved, 'dates_resolved']
            .apply(lambda x: str(x).split(' | ')[-1])
        )
        agg_args = {}
        for col in raw_df.columns:
            if col in columns_to_match:
                continue
            agg_args[col] = 'first'
        for splt_col in ['var_translations','error_message']:
            agg_args[splt_col] = self.merge_rows
        if is_cross_checks_frame:
            agg_args['check_id'] = self.merge_rows
        agg_args['time_since_last_detection'] = 'max'
        agg_args['priority_item'] = self.first_true
        #merged_df = raw_df.groupby(columns_to_match).agg(self.merge_rows).reset_index()
        merged_df = raw_df.groupby(columns_to_match).agg(agg_args).reset_index()

        merged_df['flag_count'] = merged_df['error_message'].str.count(r'\|') + 1
        #merged_df['displayed_form'] = merged_df['displayed_form'].str.title().str.replace('_',' ')
        merged_df.rename(columns=columns_names, inplace=True)
        merged_df = merged_df[list(columns_names.values())]
        #move manually resolved to the bottom
        merged_df = self.move_rows_to_bottom('Manually Resolved',None, merged_df)
        merged_df = self.move_rows_to_bottom('Date Resolved','Manually Resolved', merged_df)
        
        return merged_df
    
    def move_rows_to_bottom(self, incl_col_name, excl_col_name, df):
        if excl_col_name != None:
            moving_df = df[(df[incl_col_name] != '') & (df[excl_col_name]=='')]
            df = df[(df[incl_col_name] == '') | (df[excl_col_name]!='')]
        else:
            moving_df = df[df[incl_col_name] != '']
            df = df[df[incl_col_name] == '']

        result = pd.concat([df, moving_df], ignore_index=True)
        return result

    def save_to_dropbox(self, fullpath, local_path, dbx=None):
        """Publish a complete V2 workbook with only deployed report surfaces."""
        with open(fullpath, 'rb') as workbook_file:
            payload = workbook_file.read()
        if os.path.basename(fullpath).endswith(self.TRACKER_FILENAME_SUFFIX):
            # Existing local workbooks may contain tabs from a previous run
            # with a different report policy. Filter the final payload too:
            # even files not regenerated this run must not revive those tabs.
            workbook = load_workbook(BytesIO(payload))
            try:
                excluded = self.EXCLUDED_REPORT_NAMES.intersection(
                    workbook.sheetnames)
                if excluded:
                    if len(excluded) == len(workbook.sheetnames):
                        print(
                            f"[create_trackers] skipping {local_path}: "
                            "workbook contains only excluded report tabs.")
                        return
                    for report in excluded:
                        del workbook[report]
                    cleaned = BytesIO()
                    workbook.save(cleaned)
                    payload = cleaned.getvalue()
            finally:
                workbook.close()
        if dbx is None:
            dbx = self.utils.collect_dropbox_credentials()
        dbx.files_upload(
            payload,
            self.dropbox_path.rstrip('/') + '/' + local_path.lstrip('/'),
            mode=dropbox.files.WriteMode.overwrite)
            #self.recover_comments(self.dropbox_path + local_path)
            
    def recover_comments(self, path):
        dbx = self.utils.collect_dropbox_credentials()
        md = dbx.files_get_metadata(path)  
        file_id = md.id                          
        rev_result = dbx.files_list_revisions(
            path=file_id,
            mode=dropbox.files.ListRevisionsMode.id,
            limit=100,  
        )

        for idx, entry in enumerate(rev_result.entries, start=1):
            if idx > 40:
                break
            rev = entry.rev
            when = entry.server_modified
            md2, resp = dbx.files_download(path=path, rev=rev)
            df = pd.read_excel(BytesIO(resp.content), keep_default_na = False)  
            #print('RECOVER COMMENTS TEST')
            #print(f"[{idx}] rev={rev}  modified={when}  shape={df.shape}")
            #print(df)
            #print(path)
            cols = ['Site Comments','Network Comments', 'Manually Resolved']
            tmp = df[cols].replace(r"^\s*$", pd.NA, regex=True)
            mask_any_nonblank = tmp.notna().any(axis=1)
            df_filtered = df[mask_any_nonblank]
            self.master = pd.concat([self.master, df_filtered], ignore_index=True)
            #print(self.master)
            #print(path)
            #print(idx)

        self.master = self.master.drop_duplicates(subset=["Participant", "Timepoint","Form"])

        self.master.to_csv(f'{self.output_path}recovered_comments.csv', index = False)
        #self.append_recovered_comments('PRESCIENT')

    def recover_old_flags(self, path):
        """function to recover history of specified row over time"""
        dbx = self.utils.collect_dropbox_credentials()
        md = dbx.files_get_metadata(path)  
        file_id = md.id                          
        rev_result = dbx.files_list_revisions(
            path=file_id,
            mode=dropbox.files.ListRevisionsMode.id,
            limit=100,  
        )

        for idx, entry in enumerate(rev_result.entries, start=1):
            if idx > 40:
                break
            rev = entry.rev
            when = entry.server_modified
            md2, resp = dbx.files_download(path=path, rev=rev)
            df = pd.read_excel(BytesIO(resp.content), keep_default_na = False)  
            #print('RECOVER COMMENTS TEST')
            #print(f"[{idx}] rev={rev}  modified={when}  shape={df.shape}")
            #print(df)
            #print(path)
            cols = ['Participant','Site Comments','Network Comments',
            'Manually Resolved','Date Resolved','Form','Flags']
            tmp = df[cols].replace(r"^\s*$", pd.NA, regex=True)
            mask_any_nonblank = tmp.notna().any(axis=1)
            df_filtered = df[mask_any_nonblank]
            self.master = pd.concat([self.master, df_filtered], ignore_index=True)
            print(self.master)
            print(path)
            print(idx)
        
        self.master = self.master.drop_duplicates(subset=["Participant",
        "Timepoint","Form","Flags"])
        self.master.to_csv('recovered_flags.csv', index = False)

    def append_recovered_comments(self, network):
        # 1. load recovered comments
        comments_df = pd.read_csv(f'{self.output_path}recovered_comments.csv',
                                keep_default_na=False)

        # 2. map formatted -> raw, but safely
        reversed_dict = self.utils.reverse_dictionary(
            self.formatted_column_names[network]['combined']
        )
        comments_df = comments_df.rename(
            columns={c: reversed_dict.get(c, c) for c in comments_df.columns}
        )

        # 3. define the key we actually use to identify a row
        # adjust these names to match your *raw* combined_qc_flags.csv
        key_cols = ['subject', 'displayed_timepoint', 'displayed_form']

        # keep only the columns we care about from recovered comments
        # (so we don't accidentally merge on comment columns)
        comment_cols = ['site_comments', 'network_comments', 'manually_resolved']
        comment_cols = [c for c in comment_cols if c in comments_df.columns]

        comments_df = comments_df[key_cols + comment_cols].drop_duplicates()

        # 4. merge onto the current tracker by key only
        merged = self.combined_tracker.merge(
            comments_df,
            on=key_cols,
            how='left',
            suffixes=('', '_rec')
        )

        # 5. for each comment-like column: if current is blank, fill from recovered
        for col in comment_cols:
            rec_col = f'{col}_rec'
            if rec_col in merged.columns:
                # treat '' as blank
                merged[col] = merged[col].where(merged[col] != '', merged[rec_col])
                merged = merged.drop(columns=[rec_col])

        # 6. write back (atomic — same pattern as calculate_resolved_errors)
        path = self.curr_output_csv_path
        tmp_path = f"{path}.{os.getpid()}.tmp"
        try:
            merged.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
