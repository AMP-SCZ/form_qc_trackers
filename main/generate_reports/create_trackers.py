import pandas as pd
import os
import sys
import json
import dropbox
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class CreateTrackers():

    def __init__(self):
        self.utils = Utils()
        with open(f'{self.utils.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.output_path = self.config_info['paths']['output_path']
        
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
            for report in self.all_reports:
                report_df = self.combined_tracker[
                self.combined_tracker['reports'].str.contains(report)]
                self.all_report_df[report] = report_df
                report_df = self.convert_to_shared_format(report_df)
                if report in self.all_reports: 
                    print(report)
                    print(network)
                    for site in self.all_sites[network]:
                        print(site)
                        if report != 'Main Report' and site != 'ME':
                            continue 
                        if report != 'Non Team Forms' and site == 'ME':
                            continue 
                        print(report)
                        if not os.path.exists(f'{self.dropbox_output_path}{site}'):
                            os.makedirs(f'{self.dropbox_output_path}{site}')
                        site_df = report_df[report_df['Participant'].str[:2] == site]
                        site_df.to_csv(f'{self.dropbox_output_path}{site}/{network}_{site}_Output.csv')
    def format_to_excl(self, df, report, filename):
        pass

    def merge_rows(self, values):
        unique_values  = list(dict.fromkeys(values))

        if len(unique_values) == 1:
            return values.iloc[0]
        else:
            return ' | '.join(unique_values)

    def convert_to_shared_format(self, raw_df):
        columns_names = self.formatted_column_names['PRONET']
        columns_to_match = ['subject','displayed_timepoint','displayed_form']
        raw_df['date_resolved'] = ''
        raw_df.loc[raw_df['currently_resolved'] == True, 'date_resolved'] = (
            raw_df.loc[raw_df['currently_resolved'] == True, 'dates_resolved']
            .apply(lambda x: x.split(' | ')[-1])
        )
        merged_df = raw_df.groupby(columns_to_match).agg(self.merge_rows).reset_index()
        merged_df['flag_count'] = merged_df['error_message'].str.count(r'\|') + 1
        merged_df['displayed_form'] = merged_df['displayed_form'].str.title().str.replace('_',' ')
        merged_df.rename(columns=columns_names, inplace=True)
        merged_df = merged_df[list(columns_names.values())]

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