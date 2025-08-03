import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import datetime
import matplotlib.dates as mdates
import json 
import subprocess

class PlotDataVolume():
    def __init__(self):
        self.output = {'PRONET':{},'PRESCIENT':{}}
        
        self.all_timepoints = ['screen','baseln',\
        'month1','month2','month3', 'month4',\
        'month5','month6','month7','month8','month9',\
        'month10','month11','month12','month18','month24']

        self.missing_code_list =  ['-3','-9',-3,-9,-3.0,-9.0,'-3.0','-9.0',\
        '1909-09-09','1903-03-03','1901-01-01','-99',-99,-99.0,'-99.0']

        unique_form_vars_path = '/PHShome/ob001/anaconda3/new_forms_qc/QC/unique_form_vars.json'

        self.combined_csv_path = '/data/predict1/data_from_nda/formqc/'
        self.combined_csv_path = '/data/predict1/data_from_nda/formsdb/generated_outputs/combined/PROTECTED/'
        self.old_csv_path = '/PHShome/ob001/anaconda3/email_notifications/old_combined_csvs/'

        self.absolute_path = '/PHShome/ob001/anaconda3/email_notifications/'

        self.change_his_path = '/PHShome/ob001/anaconda3/email_notifications/change_history.csv'
        self.data_dict_path = ('/PHShome/ob001/anaconda3/'
        'new_forms_qc/QC/data_dictionaries/CloneOfYaleRealRecords_DataDictionary_2023-06-13.csv')
        if os.path.exists(self.change_his_path):
            self.update_history = pd.read_csv(self.change_his_path, \
            keep_default_na = False)
        else:
            self.update_history = pd.DataFrame([])

        self.final_output = []

        with open(unique_form_vars_path, 'r') as file:
            self.unique_form_variables = json.load(file)
        self.email_list = ["oborders@mgh.harvard.edu"]
        
        self.email_list = ["oborders@mgh.harvard.edu"]

    def create_var_list(self):
        data_dict = pd.read_csv(self.data_dict_path, \
        encoding = 'latin-1',keep_default_na=False)
        var_col_name =  'ï»¿"Variable / Field Name"'
        self.variable_list = data_dict[var_col_name].tolist()
    
    def run_script(self):
        self.create_var_list()
        self.compare_csvs()
        self.check_data_changes()
        self.send_emails(f'PIPELINE QC COMPLETED: {str(self.final_output)}',\
        str(datetime.datetime.today().date()), "oborders@mgh.harvard.edu")

    def compare_csvs(self):
        for network in ['PRONET','PRESCIENT']:
            timepoint_str = 'screen'
            for timepoint in self.all_timepoints:
                print(timepoint)
                if timepoint == 'screen':
                    timepoint_str = 'screening'
                elif timepoint == 'baseln':
                    timepoint_str = 'baseline'
                else:
                    timepoint_str = timepoint
                print(f'{self.combined_csv_path}combined-{network}-{timepoint_str}-day1to1.csv')
                if os.path.exists(f'{self.combined_csv_path}combined-{network}-{timepoint_str}-day1to1.csv'):
                    csv_name = f'combined-{network}-{timepoint_str}-day1to1.csv'
                    old_df = pd.read_csv(f'{self.old_csv_path}{csv_name}',\
                    keep_default_na = False)
                    current_df = pd.read_csv(f'{self.combined_csv_path}{csv_name}',\
                    keep_default_na = False)
                    var_list = [var for var in self.variable_list if var in current_df.columns]
                    var_list = [var for var in self.variable_list if var in old_df.columns]
                    old_df = old_df[var_list]
                    current_df = current_df[var_list]
                    if old_df.equals(current_df):
                        equals_val = 'Yes'
                    else:
                        equals_val = 'No'
                    self.final_output.append({'CSV':csv_name,'Equals':equals_val,\
                    'Date':datetime.datetime.today().date()})
                    print(f'{self.combined_csv_path}combined-{network}-{timepoint_str}-day1to1.csv')
                    if os.path.exists(f'{self.combined_csv_path}combined-{network}-{timepoint_str}-day1to1.csv'):
                        print('pass')
                        db = pd.read_csv(f'{self.combined_csv_path}combined-{network}-{timepoint_str}-day1to1.csv',\
                        keep_default_na = False)
                        db.to_csv(f'{self.old_csv_path}combined-{network}-{timepoint_str}-day1to1.csv',index = False)

        self.final_output = pd.DataFrame(self.final_output)
        df_combined = pd.concat([self.update_history, self.final_output], ignore_index=True)

        df_combined.to_csv(f'{self.absolute_path}change_history.csv', index = False)

    def find_time_diff(self, first_date, second_date):
        time_diff = (second_date - first_date).days
        return time_diff
    
    def send_emails(self, subject_line, message, recipient):
        result = subprocess.run(['/PHShome/ob001/anaconda3/email_notifications/email_test.sh'] + \
        [message,recipient,subject_line], capture_output=True, text=True)

    def check_data_changes(self):
        change_df = pd.read_csv(self.change_his_path, \
        keep_default_na = False)
        today = datetime.datetime.today().date()
        change = False
        for network in ['PRONET','PRESCIENT']:
            change = False
            for row in change_df.itertuples():
                print(network)
                if network not in row.CSV:
                    continue
                date = datetime.datetime.strptime(row.Date, '%Y-%m-%d').date()
                diff = self.find_time_diff(date,today)
                if diff > 2:
                    continue
                if row.Equals == 'No':
                    change = True
            if change == False:
                for recipient in self.email_list:
                    self.send_emails('Potential Data Influx Issue Detected',\
                    f'Data from the combined CSVs for {network} has not changed for at least 2 days.', recipient)
            
if __name__ == '__main__':
    PlotDataVolume().run_script()