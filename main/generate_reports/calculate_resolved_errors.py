import pandas as pd

import os
import sys
import json
import random
import shutil
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from datetime import datetime
from io import BytesIO
import numpy as np 

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

    def _read_current(self):
        return pd.read_parquet(self.out_paths['current']).fillna('')

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
            df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, current_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
            
    def loop_dropbox_files(self):
        # Load the current tracker once, accumulate dropbox-side comments
        # in memory across all networks/sites/RAs, then write back once.
        # Previously each source caused a full read + write of this
        # ~1M-row file.
        dbx = self.utils.collect_dropbox_credentials()
        current_df = self._read_current()
        for network in dbx.files_list_folder(self.dropbox_path).entries:
            if network.name not in ['PRONET','PRESCIENT']:
                continue
            network_dir = self.dropbox_path + f'{network.name}'
            combined_output = network_dir + f'/combined/{network.name}_Output_V2.xlsx'
            current_df = self.read_dropbox_data(
                current_df,
                self.formatted_column_names[network.name]["combined"],
                ['manually_resolved','comments'],
                combined_output, dbx, network.name, ['Main Report'])
            for site_abr in self.utils.all_sites[network.name]:
                site = self.utils.site_full_name_translations[site_abr]
                site_output = network_dir + f'/{site}/{network.name}_{site_abr}_Output_V2.xlsx'
                site_cols = self.formatted_column_names[network.name]["sites"]
                if site_abr == 'ME':
                    reports_to_read = ['Non Team Forms']
                    for ra in self.melbourne_ras:
                        ra_output = network_dir + f'/{site}/{ra}/{network.name}_Melbourne_Output_V2.xlsx'
                        current_df = self.read_dropbox_data(
                            current_df, site_cols,
                            ['site_comments','comments'],
                            ra_output, dbx, network.name, reports_to_read)
                else:
                    reports_to_read = ['Main Report']
                    current_df = self.read_dropbox_data(
                        current_df, site_cols,
                        ['site_comments','comments'],
                        site_output, dbx, network.name, reports_to_read)
        self._write_current(current_df)
        return
        
    def check_dbx_file_exists(self,dbx, dropbox_path):
        try:
            print('checking if dbx exists')
            print(dropbox_path)
            _, res = dbx.files_download(dropbox_path)
            data = res.content
            print('exists')
            return True
        except Exception as e:
            print('does not exist')
            print(e)
            return False
        
    def read_dropbox_data(self,
        prev_output_df, col_names, columns_to_read,
        dropbox_path, dbx, network, reports_to_read,
        excl_report = True
    ):
        reversed_col_translations = self.utils.reverse_dictionary(col_names)
        if self.check_dbx_file_exists(dbx, dropbox_path) == False:
            return prev_output_df
        _, res = dbx.files_download(dropbox_path)
        data = res.content
        excel_data = pd.ExcelFile(BytesIO(data))
        sheet_names = excel_data.sheet_names
        for report in sheet_names:
            if report not in reports_to_read and excl_report == True:
                continue
            report_df = pd.read_excel(BytesIO(data),
                sheet_name=report, keep_default_na=False)
            report_df.rename(columns=reversed_col_translations, inplace=True)
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
            merge_keys = ['displayed_form', 'displayed_timepoint',
                          'subject', 'error_message']
            keep = [c for c in (merge_keys + list(columns_to_read))
                    if c in report_df.columns]
            report_df = report_df[keep]
            report_df = report_df.assign(
                error_message=report_df['error_message'].apply(lambda x: x.split(' | '))
            )
            report_df = report_df.explode('error_message').reset_index(drop=True)
            report_df['current_report'] = report
            prev_output_df['current_report'] = np.where(
                prev_output_df['reports'].astype(str).str.contains(report, case=False, regex=False, na=False),
                report, '')

            prev_output_df = pd.merge(prev_output_df, report_df, on=[
                'displayed_form','displayed_timepoint',
                'subject','error_message'],
                how='left', suffixes=('', '_dbx'))
            prev_output_df = prev_output_df.fillna('')

            for col_to_read in columns_to_read:
                dbx_col = f"{col_to_read}_dbx"
                if dbx_col not in prev_output_df.columns:
                    continue
                col_values = prev_output_df[dbx_col]
                if isinstance(col_values, pd.DataFrame):
                    col_values = col_values.iloc[:, 0]
                has_new = col_values.astype(str).str.len() > 0
                prev_output_df.loc[has_new, col_to_read] = col_values[has_new]

            # Drop merge artifacts so the next iteration / next dropbox file
            # starts clean — otherwise these accumulate and a later merge
            # tries to create a `_dbx` column that already exists.
            drop_cols = [c for c in prev_output_df.columns
                         if c.endswith('_dbx') or c == 'current_report']
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
            new_df = pd.read_parquet(self.out_paths['new']).fillna('')
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
        cols_to_merge = ['subject', 'displayed_form', 'displayed_timepoint',
        'displayed_variable','error_message']
        if os.path.exists(self.out_paths['old']):
            old_df = pd.read_parquet(self.out_paths['old']).fillna('')
            # Skip the merge entirely if old has no usable schema (e.g., a
            # prior zero-flag run wrote a 0-row, 0-col parquet). Trying to
            # merge on missing keys would KeyError. Treat as no-old-history.
            old_has_schema = all(c in old_df.columns for c in cols_to_merge)
            if old_has_schema:
                # If new_df is empty (no flags this run) it may have no columns,
                # which would KeyError on the merge. Reuse old_df's schema so
                # the merge keys exist; the outer-merge then yields only old
                # rows (each will be marked resolved by compare_old_new_outputs).
                if new_df.empty:
                    new_df = pd.DataFrame(columns=old_df.columns)
                orig_columns = list(new_df.columns)
                merged = old_df.merge(new_df,
                on=cols_to_merge, how='outer', suffixes=('_old','_new'))
                merged_df = merged.fillna('')
                new_df = self.compare_old_new_outputs(orig_columns, cols_to_merge, merged_df)
        # Atomic write of the canonical current_output. Without this, a
        # crash mid-`to_parquet` produces a corrupt parquet that the next
        # stage cannot read — and my consumer-side guard then treats it as
        # zero flags, falsely marking every previously-open flag as resolved.
        current_path = self.out_paths['current']
        tmp_path = f"{current_path}.{os.getpid()}.tmp"
        try:
            new_df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, current_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def compare_old_new_outputs(self, orig_columns, cols_to_merge, merged_df):
        curr_date = str(datetime.today().date())
        # CSV round-trip turns booleans into the strings 'True'/'False', so
        # `row.currently_resolved_old == True` was previously almost always
        # False even for resolved rows, which broke re-opening logic.
        truthy = (True, 'True', 'true', 'TRUE', 1, '1')
        for row in merged_df.itertuples():
            curr_row_output = {}
            resolved_old = getattr(row, 'currently_resolved_old', '') in truthy
            # if col only exists in new output
            if row.network_new != '' and row.network_old == '':
                # sets resolved to false
                curr_row_output['currently_resolved'] = False
                # adds current date to dates_detected
                curr_row_output['dates_detected'] = self.append_formatted_list(
                row.dates_detected_old, curr_date)
                # adds all other columns of the new output
                curr_row_output = self.append_all_cols(
                row, curr_row_output, orig_columns, '_new', cols_to_merge,merged_df)
            elif row.network_old != '':
                # if exists in old and not new
                if row.network_new == '':
                    if resolved_old:
                        # if it is still resolved, add all columns from old
                        curr_row_output = self.append_all_cols(row, curr_row_output,
                        orig_columns, '_old',cols_to_merge,merged_df)
                    else:
                        # if it was not resolved, set it to resolved
                        curr_row_output['currently_resolved'] = True
                        # adds current date to dates resolved
                        curr_row_output['dates_resolved'] = self.append_formatted_list(
                        row.dates_resolved_old, curr_date)
                        # adds all other columns from old output
                        curr_row_output = self.append_all_cols(row, curr_row_output,
                        orig_columns, '_old', cols_to_merge,merged_df)
                # if exists in old and new
                elif row.network_new != '':
                    # sets currently_resolved to false
                    curr_row_output['currently_resolved'] = False
                    if resolved_old:
                        # if it was resolved in the old output
                        # add current date to dates_detected
                        curr_row_output['dates_detected'] = self.append_formatted_list(
                        row.dates_detected_old, curr_date)
                    for old_col in ['dates_resolved', 'dates_detected']:
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
                            curr_row_output[old_col] = getattr(row, old_col + '_old')
                    # adds all remaining columns from new output
                    curr_row_output = self.append_all_cols(row,
                    curr_row_output, orig_columns, '_new',cols_to_merge,merged_df)
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
            
    def append_all_cols(self, row, curr_row_output, all_cols, suffix, merged_cols,df):
        for col in all_cols:
            if col not in curr_row_output.keys():
                if col not in merged_cols:
                    if (col + suffix) in df.columns:
                        curr_row_output[col] = getattr(row, col + suffix)
                else:
                    curr_row_output[col] =  getattr(row, col)

        return curr_row_output
