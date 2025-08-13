import pandas as pd

import os
import sys
import json
import random
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
        self.dropbox_path = f'/Apps/Automated QC Trackers/refactoring_tests/'
        if self.config_info["testing_enabled"] == "True":
            self.output_path += 'testing/'
            self.dropbox_path += 'testing/'

        self.old_path = '/PHShome/ob001/anaconda3/refactored_qc/output/combined_outputs/old_output/combined_qc_flags.csv'
        self.new_path = '/PHShome/ob001/anaconda3/refactored_qc/output/combined_outputs/new_output/combined_qc_flags.csv'
        self.out_paths = {}
        for path_pref in ['old','new','current']:
            self.out_paths[path_pref] = f"{self.output_path}combined_outputs/{path_pref}_output/combined_qc_flags.csv"
            
        self.new_output = []
        self.old_output_csv_path = f'{self.output_path}combined_outputs/old_output/combined_qc_flags.csv'
        self.new_output_csv_path = f'{self.output_path}combined_outputs/new_output/combined_qc_flags.csv'
        self.dropbox_data_folder = f'{self.output_path}formatted_outputs/dropbox_files/'

        self.formatted_column_names = formatted_col_names
        self.melbourne_ras = self.utils.load_dependency_json('melbourne_ra_subs.json')

    def run_script(self):
        # determine which errors no longer exist in the new output
        self.determine_resolved_rows() 
        # read specified columns from dropbox to new output
        self.loop_dropbox_files()
            
    def loop_dropbox_files(self):
        # define columns to read over
        # match formatted spreadsheet to old one by the sheet name,
        # subject','displayed_timepoint', and displayed_form
        # for any rows in the old df that match those, pull the necessary columns 
        # then save this as the old df again and compare to the new df
        # when comparing to the new df, make sure those columns from the old df are preserved in all conditions

        dbx = self.utils.collect_dropbox_credentials()
        for network in dbx.files_list_folder(self.dropbox_path).entries:
            if network.name in ['PRONET','PRESCIENT']:
                print(network.name)
                print(self.dropbox_path + f'{network.name}')
                network_dir = self.dropbox_path + f'{network.name}'
                #for network_entry in dbx.files_list_folder(network_dir).entries:
                combined_output = network_dir + f'/combined/{network.name}_combined_Output.xlsx'
                all_reports = self.utils.collect_new_reports(pd.read_csv(self.out_paths['current'],
                keep_default_na = False))
                self.read_dropbox_data(self.formatted_column_names[network.name]["combined"],
                ['manually_resolved','comments'], combined_output, dbx, network.name, all_reports)
                for site_abr in self.utils.all_sites[network.name]:
                    site = self.utils.site_full_name_translations[site_abr]
                    site_output = network_dir + f'/{site}/{network.name}_{site_abr}_Output.xlsx'
                    site_cols = self.formatted_column_names[network.name]["sites"]
                    if site_abr == 'ME':
                        # melbourne not working
                        reports_to_read = ['Non Team Forms']
                        for ra in self.melbourne_ras:
                            ra_output = network_dir + f'/{site}/{ra}/{network.name}_Melbourne_RA_Output.xlsx'
                            self.read_dropbox_data(site_cols, ['site_comments'], ra_output, dbx, network.name, reports_to_read)
                    else:
                        reports_to_read = ['Main Report']
                        self.read_dropbox_data(site_cols,['site_comments'], site_output, dbx, network.name, reports_to_read)
        return 
        
    def check_dbx_file_exists(self,dbx, dropbox_path):
        try:
            _, res = dbx.files_download(dropbox_path)
            data = res.content
            return True
        except Exception as e:
            return False
        
    def read_dropbox_data(self,
        col_names,columns_to_read,
        dropbox_path, dbx, network, reports_to_read,
        excl_report = True
    ):  
        print(columns_to_read)
        reversed_col_translations = self.utils.reverse_dictionary(col_names)
        if self.check_dbx_file_exists(dbx, dropbox_path) == False:
            return
        _, res = dbx.files_download(dropbox_path)
        data = res.content
        excel_data = pd.ExcelFile(BytesIO(data))
        sheet_names = excel_data.sheet_names
        prev_output_df = pd.read_csv(self.out_paths['current'],
         keep_default_na = False)
        orig_columns = prev_output_df.columns
        for report in sheet_names:
            if report not in reports_to_read and excl_report == True:
                continue
            report_df = pd.read_excel(BytesIO(data),\
                sheet_name=report, keep_default_na = False)
            
            report_df.rename(columns=reversed_col_translations, inplace=True)
            report_df = report_df.assign(
                error_message=report_df['error_message'].apply(lambda x: x.split(' | '))
            )

            subjects_to_merge = report_df['subject'].tolist()

            report_df = report_df.explode('error_message').reset_index(drop=True)
            report_df['current_report'] = report
            prev_output_df['current_report'] = np.where(
            prev_output_df['reports'].str.contains(report, case=False), report, '')

            prev_output_df = pd.merge(prev_output_df, report_df, on=[
            'displayed_form','displayed_timepoint',
            'subject','error_message'],
            how = 'left', suffixes=('', '_dbx'))
            prev_output_df = prev_output_df.fillna('')
            
            for col_to_read in columns_to_read:
                print(col_to_read)
                dbx_col = col_to_read + '_dbx'
                prev_output_df.loc[
                prev_output_df['subject'].isin(subjects_to_merge), col_to_read] = prev_output_df[dbx_col]
            prev_output_df = prev_output_df[orig_columns]  
        prev_output_df.to_csv(self.out_paths['current'], index = False)

    def determine_resolved_rows(self):
        new_df = pd.read_csv(self.out_paths['new'], keep_default_na = False)
        #new_df = new_df[new_df['currently_resolved'] == False]
        #new_df = new_df.drop('NDA Excluder', axis=1)
        #old_df = old_df.drop('NDA Excluder', axis=1)
        if os.path.exists(self.out_paths['current']):
            curr_df = pd.read_csv(self.out_paths['current'],keep_default_na = False)
            curr_df.to_csv(self.out_paths['old'],index = False)

        if os.path.exists(self.out_paths['old']):
            old_df = pd.read_csv(self.out_paths['old'], keep_default_na = False)

            orig_columns = list(new_df.columns)

            cols_to_merge = ['subject', 'displayed_form', 'displayed_timepoint',
            'displayed_variable','error_message']

            merged = old_df.merge(new_df,
            on = cols_to_merge, how='outer', suffixes = ('_old','_new'))

            merged_df = merged.fillna('')
            new_df = self.compare_old_new_outputs(orig_columns,cols_to_merge, merged_df)

        output_dir = os.path.dirname(self.out_paths['current'])
        os.makedirs(output_dir, exist_ok=True)

        new_df.to_csv(self.out_paths['current'], index = False)

    def compare_old_new_outputs(self, orig_columns, cols_to_merge, merged_df):
        curr_date = str(datetime.today().date())
        for row in merged_df.itertuples():
            curr_row_output = {}
            # if col only exists in new output
            if row.network_new != '' and row.network_old == '':
                # sets resolved to false
                curr_row_output['currently_resolved'] = False
                # adds current date to dates_detected
                curr_row_output['dates_detected'] = self.append_formatted_list(
                row.dates_detected_old, curr_date)
                # adds all other columns of the new output 
                curr_row_output = self.append_all_cols(
                row, curr_row_output, orig_columns, '_new',cols_to_merge,merged_df)
            elif row.network_old != '':
                # if exists in old and not new
                if row.network_new == '':
                    if self.config_info["remove_flags"] == "True":
                        continue
                    if row.currently_resolved_old == True:
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
                        orig_columns, '_old',cols_to_merge,merged_df)
                # if exists in old and new
                elif row.network_new != '':
                    # sets currently_resolved to false
                    curr_row_output['currently_resolved'] = False
                    if row.currently_resolved_old == True:
                        # if it was resolved in the old output
                        # add current date to dates_detected
                        curr_row_output['dates_detected'] = self.append_formatted_list(
                        row.dates_detected_old, curr_date)   
                    for old_col in ['dates_resolved', 'dates_detected']:
                        if old_col not in curr_row_output.keys():
                            # if dates_resolved and dates_detected not in curr output
                            # then will add them from the old output
                            curr_row_output[old_col] = getattr(row, old_col + '_old')
                    # adds all remaining columns from new output
                    curr_row_output = self.append_all_cols(row,
                    curr_row_output, orig_columns, '_new',cols_to_merge,merged_df)
            dates_detected = curr_row_output['dates_detected'].split(' | ')
            most_recent_detection = dates_detected[-1]
            curr_row_output['time_since_last_detection'] = self.utils.days_since_today(
            str(most_recent_detection))
            print(curr_row_output['time_since_last_detection'])

            self.new_output.append(curr_row_output)
                #if row.currently_resolved
                #curr_row_output['currently_resolved'] = True
        new_df = pd.DataFrame(self.new_output)
        return new_df
    

    def append_formatted_list(self, curr_list_string, item_to_append):
        new_list = []
        if curr_list_string == '':
            return item_to_append
        else:
            new_list = curr_list_string.split(' | ')
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
