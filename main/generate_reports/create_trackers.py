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

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class CreateTrackers():

    def __init__(self):
        self.utils = Utils()
        with open(f'{self.utils.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.output_path = self.config_info['paths']['output_path']
        if self.config_info["testing_enabled"] == "True":
            self.output_path += "testing/"
        
        self.all_reports = ['Main Report','Secondary Report']
        self.site_reports = ['Main Report']
        self.all_report_df = {}
        self.all_pronet_sites = self.utils.all_pronet_sites
        self.all_prescient_sites = self.utils.all_prescient_sites
        self.all_sites = self.utils.all_sites
        self.site_translations = self.utils.site_full_name_translations
        self.old_output_csv_path = f'{self.output_path}combined_outputs/old_output/combined_qc_flags.csv'
        self.new_output_csv_path = f'{self.output_path}combined_outputs/new_output/combined_qc_flags.csv'
        self.formatted_outputs_path = f'{self.output_path}formatted_outputs/'
        if not os.path.exists(self.formatted_outputs_path):
            os.makedirs(self.formatted_outputs_path)
        self.dropbox_output_path =  f'{self.output_path}formatted_outputs/dropbox_files/'
        self.colors = {"green" : PatternFill(start_color='b9d9b4', end_color='b9d9b4', fill_type='gray125'),
                       "yellow" : PatternFill(start_color='bab466', end_color='bab466', fill_type='gray125'),
                       "blue" : PatternFill(start_color='d8cffc', end_color='d8cffc', fill_type='gray125'),
                       "orange" : PatternFill(start_color='ebba83', end_color='ebba83', fill_type='gray125'),
                       "red" : PatternFill(start_color='de9590', end_color='de9590', fill_type='gray125')}
        
        self.formatted_column_names = {
            "PRONET" : {
            "subject":"Participant",
            "displayed_timepoint":"Timepoint",
            "displayed_form" : "Form",
            "flag_count" : "Flag Count",
            "error_message" : "Flags",
            "var_translations":"Translations",
            "time_since_last_detection": "Days Since Detected",
            "date_resolved":"Date Resolved",
            "manually_resolved": "Manually Resolved",
            "comments": "Comments"},

            "PRESCIENT" :{
            "subject":"Participant",
            "displayed_timepoint":"Timepoint",
            "displayed_form" : "Form",
            "flag_count" : "Flag Count",
            "error_message" : "Flags",
            "var_translations":"Translations",
            "time_since_last_detection": "Days Since Detected",
            "date_resolved":"Date Resolved",
            "manually_resolved": "Manually Resolved",
            "comments": "Comments"},
        }

    def __call__(self):
        pass

    def run_script(self):
        self.combined_tracker = pd.read_csv(self.new_output_csv_path, keep_default_na= False)
        self.collect_new_reports()
        self.generate_reports()
        #self.save_to_dropbox()

    def collect_new_reports(self):
        for row in self.combined_tracker.itertuples():
            if row.reports == '':
                continue
            reports = (row.reports).split(' | ')
            for report in reports:
                if report not in self.all_reports:
                    self.all_reports.append(report)
                    print(self.all_reports)

    def generate_reports(self):
        for network in ['PRONET','PRESCIENT']:
            for report in self.all_reports[:1]:
                network_df = self.combined_tracker[self.combined_tracker['network']==network]
                report_df = network_df[
                network_df['reports'].str.contains(report)]
                self.all_report_df[report] = report_df
                if report_df.empty:
                    continue
                report_df = self.convert_to_shared_format(report_df, network)
                if report in self.all_reports: 
                    print(network)
                    if not os.path.exists(f'{self.dropbox_output_path}{network}'):
                        os.makedirs(f'{self.dropbox_output_path}{network}')
                    self.format_excl_sheet(report_df,report,
                    f'{self.dropbox_output_path}{network}/',
                    f'{network}_combined_Output.xlsx')
                    """for site in self.all_sites[network]:
                        print(site)
                        if report != 'Main Report' and site != 'ME':
                            continue 
                        if report != 'Non Team Forms' and site == 'ME':
                            continue 
                        print(report)
                        if not os.path.exists(f'{self.dropbox_output_path}{site}'):
                            os.makedirs(f'{self.dropbox_output_path}{site}')
                        site_df = report_df[report_df['Participant'].str[:2] == site]"""
                        #self.format_excl_sheet(site_df,
                        #report,f'{self.dropbox_output_path}{site}/',
                        #f'{network}_{site}_Output.xlsx')

                        #site_df.to_csv(f'{self.dropbox_output_path}{site}/{network}_{site}_Output.csv')

    def format_excl_sheet(self, df, report, folder, filename):
        full_path = folder + filename
        if not os.path.exists(folder):
            os.makedirs(folder)

        if not os.path.exists(folder + filename):
            df.to_excel(folder + filename, sheet_name = report, index = False)

        with pd.ExcelWriter(full_path, mode='a',\
        engine='openpyxl',if_sheet_exists = 'replace') as writer:                
            df.to_excel(writer, sheet_name=report, index=False)

        workbook = load_workbook(full_path)
        worksheet = workbook[report]
        self.change_excel_colors(worksheet)

        #if not os.path.exists(folder + filename):
        #df.to_excel(folder + filename, sheet_name = report, index = False)
    
    def change_excel_colors(self, worksheet):
        for row in worksheet.iter_rows():
            for cell in row:
                header_value = worksheet.cell(row=1, column=cell.column).value
                cell_val = str(cell.value)   
                if cell_val == 'None':
                    cell_val = ''
                print(header_value)
                print(cell_val)

    def change_excel_column_sizes(self,path,sheet):
        pass
    def upload_files_to_dropbox(self):
        pass

    def merge_rows(self, values):
        unique_values  = list(dict.fromkeys(values))

        if len(unique_values) == 1:
            return values.iloc[0]
        else:
            return ' | '.join(unique_values)

    def convert_to_shared_format(self, raw_df, network):
        columns_names = self.formatted_column_names[network]
        columns_to_match = ['subject','displayed_timepoint','displayed_form']
        raw_df['date_resolved'] = ''
        raw_df.loc[raw_df['currently_resolved'] == True, 'date_resolved'] = (
            raw_df.loc[raw_df['currently_resolved'] == True, 'dates_resolved']
            .apply(lambda x: x.split(' | ')[-1])
        )

        #raw_df = raw_df.sort_values(by='manually_resolved', key=lambda x: x == "", ascending=True)
        print(raw_df)
        #raw_df = raw_df.sort_values(by='date_resolved', key=lambda x: x == "", ascending=True)
        #raw_df['is_blank'] = raw_df['date_resolved'] == ''
        #raw_df = raw_df.sort_values(by='is_blank', ascending=False).drop(columns=['is_blank']).reset_index(drop=True)

        print(raw_df)
        merged_df = raw_df.groupby(columns_to_match).agg(self.merge_rows).reset_index()
        merged_df['flag_count'] = merged_df['error_message'].str.count(r'\|') + 1
        merged_df['displayed_form'] = merged_df['displayed_form'].str.title().str.replace('_',' ')
        merged_df.rename(columns=columns_names, inplace=True)
        merged_df = merged_df[list(columns_names.values())]
        #move manually resolved to the bottom
        merged_df = merged_df.sort_values(by='Manually Resolved', key=lambda x: x.str.strip() == "", ascending=False)
        # move date resolved below manually resolved
        merged_df = merged_df.sort_values(by='Date Resolved', key=lambda x: x.str.strip() == "", ascending=False)
        

        merged_df.to_csv(f'{self.formatted_outputs_path}AMPSCZ_Output.csv',index = False)

        return merged_df

    def save_to_dropbox(self):
        dbx = self.utils.collect_dropbox_credentials()
        path = f'{self.formatted_outputs_path}AMPSCZ_Output.csv'
        dropbox_path = f'/Apps/Automated QC Trackers/refactoring_tests'
        with open(path, 'rb') as f:
            dbx.files_upload(f.read(), dropbox_path + f'/AMPSCZ_Output.csv',\
            mode=dropbox.files.WriteMode.overwrite)

if __name__ == '__main__':
    CreateTrackers().run_script()