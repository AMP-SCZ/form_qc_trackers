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


class AnalyzeWithdrawalDates():

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.output_path = self.config_info['paths']['output_path']
        print(self.output_path)
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.date_vars = self.utils.collect_all_type_vars()
        self.withdrawn_subs = {}
    

    def run_script(self):
        self.loop_csvs()


    def loop_csvs(self):
        tp_list = self.utils.create_timepoint_list()
        for network in ['PRONET']:
            combined_df_floating = pd.read_csv(
            (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
            f'{"floating_forms"}_{network}-day1to1.csv'),
            keep_default_na = False)
            self.withdrawn_subs = self.collect_withdrawn_subs(combined_df_floating)
            for tp in tp_list:
                print(tp)
                combined_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}_{network}-day1to1.csv'),
                keep_default_na = False)
                self.check_withdrawal_date(combined_df, tp)

    def collect_withdrawn_subs(self, floating_df):
        withdrawal_var = 'chr_subject_date_of_withdrawal'
        filtered_df = floating_df[~floating_df[
        withdrawal_var].isin(self.utils.missing_code_list + [''])] 
        print(filtered_df)
        withdrawal_dict = filtered_df.set_index("subjectid")[withdrawal_var].to_dict()
    
        return withdrawal_dict

    def check_withdrawal_date(self, combined_df, tp):
        withdrawal_var = 'chr_subject_date_of_withdrawal'
        for row in combined_df.itertuples():
            subject = getattr(row,'subjectid')
            if subject in self.withdrawn_subs.keys():
                withdrawal_date = self.withdrawn_subs[subject]
                if not (self.utils.check_if_val_date_format(withdrawal_date)):
                    continue
                for date_var in self.date_vars:
                    if hasattr(row, date_var):
                        date_val = getattr(row, date_var)
                        if (date_val not in (self.utils.missing_code_list + [''])
                        and self.utils.check_if_val_date_format(date_val)):
                            withdrawn_date_formatted = datetime.strptime(withdrawal_date, "%Y-%m-%d")  
                            compared_date_formatted = datetime.strptime(date_val, "%Y-%m-%d")  
                            if compared_date_formatted > withdrawn_date_formatted:
                                print(row.subjectid)
                                print(compared_date_formatted)
                                print(withdrawn_date_formatted)
            

        
if __name__ == '__main__':
    AnalyzeWithdrawalDates().run_script()
