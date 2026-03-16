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

class FormatIssueFinder():
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
        for network in ['PRESCIENT','PRONET']:3
            print(self.withdrawn_subs)
            for tp in tp_list:
                print(tp)
                combined_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}_{network}-day1to1.csv'),
                keep_default_na = False)
                self.find_whitespace(combined_df, tp)

    def find_whitespace(self, combined_df, tp):
        all_vars = list(combined_df.columns)
        for row in combined_df.itertuples():
            for var in all_vars:
                val = getattr(row,var)
                if row.subjectid == 'SL15064' and var == 'chrsofas_currscore12mo':
                    print('-----------------------')
                    print(val)
                    print(var)
                    print(len(val))
                    print(getattr(row,'subjectid'))

                if str(val) != (str(val)).strip():
                    print('-----------------------')
                    print(val)
                    print(var)
                    print(getattr(row,'subjectid'))

if __name__ == '__main__':
    FormatIssueFinder().run_script()
