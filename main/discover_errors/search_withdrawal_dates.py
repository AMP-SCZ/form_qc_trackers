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
from matplotlib import pyplot as plt

class AnalyzeWithdrawalDates():

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.output_path = self.config_info['paths']['output_path']
        depen_path = self.config_info['paths']['dependencies_path']
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.date_vars = self.utils.collect_all_type_vars()
        self.withdrawn_subs = {}
        self.dates_after_withdrawal = []
        self.grouped_variables = self.utils.load_dependency_json(
            f'grouped_variables.json')
        self.forms_per_var = self.grouped_variables['var_forms']
        self.important_form_vars = self.utils.load_dependency_json(
        'important_form_vars.json')

        self.excluded_strings = ['removed_date','entry_date']

        self.count_per_form = {}
    
    def run_script(self):
        self.loop_csvs()

    def loop_csvs(self):
        tp_list = self.utils.create_timepoint_list()
        for network in ['PRESCIENT','PRONET']:
            combined_df_floating = pd.read_csv(
            (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
            f'{"floating_forms"}_{network}-day1to1.csv'),
            keep_default_na = False)
            self.withdrawn_subs = self.collect_withdrawn_subs(combined_df_floating, network)
            print(self.withdrawn_subs)
            for tp in tp_list:
                print(tp)
                combined_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}_{network}-day1to1.csv'),
                keep_default_na = False)
                self.check_withdrawal_date(combined_df, tp,network)

        count_per_form_list = []
        for form, count in self.count_per_form.items():
            count_per_form_list.append({'form':form,'count':count})
        count_per_form_df = pd.DataFrame(count_per_form_list)
        count_per_form_df.to_csv(
        f'{self.output_path}withdrawal_errors_per_form.csv',
        index = False)

    def collect_withdrawn_subs(self, floating_df, network):
        withdrawal_var = 'removed_date'
        filtered_df = floating_df[~floating_df[
        withdrawal_var].isin(self.utils.missing_code_list + [''])] 
        if network == 'PRONET':
            filtered_df = filtered_df[~filtered_df[
            'chr_statusform_reentry'].isin(self.utils.all_dtype([1]))] 

        print(filtered_df)
        withdrawal_dict = filtered_df.set_index("subjectid")[withdrawal_var].to_dict()
    
        return withdrawal_dict

    def check_withdrawal_date(self, combined_df, tp, network):
        withdrawal_var = 'removed_date'
        for row in combined_df.itertuples():
            subject = getattr(row,'subjectid')
            if subject in self.withdrawn_subs.keys():
                withdrawal_date = self.withdrawn_subs[subject]
                if network == 'PRESCIENT':
                    withdrawal_date = self.convert_prescient_date(withdrawal_date)
                withdrawal_date = str(withdrawal_date).split(' ')[0]
                if not (self.utils.check_if_val_date_format(withdrawal_date)):
                    continue
                for date_var in self.date_vars:
                    if any(excl_str in date_var for excl_str in self.excluded_strings):
                        continue
                    if date_var.startswith('chrmiss'):
                        continue
                    if date_var in self.forms_per_var.keys():
                        form = self.forms_per_var[date_var]
                        missing_var = self.important_form_vars[form]['missing_var']
                        if (missing_var != "" and hasattr(row,missing_var) 
                        and getattr(row,missing_var) in [1,1.0,'1','1.0']):
                            continue
                    if hasattr(row, date_var):
                        date_val = getattr(row, date_var)
                        date_val = str(date_val).split(' ')[0]
                        if (date_val not in (self.utils.missing_code_list + [''])
                        and self.utils.check_if_val_date_format(date_val)):
                            withdrawn_date_formatted = datetime.strptime(withdrawal_date, "%Y-%m-%d")  
                            compared_date_formatted = datetime.strptime(date_val, "%Y-%m-%d")  
                            if compared_date_formatted > withdrawn_date_formatted:
                                self.count_per_form.setdefault(form,0)
                                self.count_per_form[form] += 1
                                time_between = self.utils.find_days_between(withdrawal_date, date_val)
                                self.dates_after_withdrawal.append({'subject':row.subjectid, 'network':network,
                                'timepoint':tp,'form':form,'form_date_var':date_var,
                                'form_date':date_val,'withdrawal_date':withdrawal_date,
                                "days_between_dates":time_between, 'withdrawal_info_source':row.removed_info_source})
        withdrawal_df = pd.DataFrame(self.dates_after_withdrawal)
        withdrawal_df.to_csv(
        f'{self.output_path}dates_after_withdrawal.csv',
        index = False)

    def convert_prescient_date(self, inp_date):
        formatted_date = str(inp_date).replace('/','-')
        if formatted_date.count("-") < 2:
            return inp_date

        formatted_date = formatted_date.split('-')
        formatted_date = [formatted_date[2],formatted_date[1],formatted_date[0]]
        formatted_date = "-".join(formatted_date)

        return formatted_date


if __name__ == '__main__':
    AnalyzeWithdrawalDates().run_script()
