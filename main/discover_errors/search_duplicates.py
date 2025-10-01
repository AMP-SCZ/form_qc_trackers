import pandas as pd

import sys
import os
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
from utils.utils import Utils
import json

class DuplicateSearcher():

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.output_path = self.config_info['paths']['output_path']

        self.final_output = {}

    def run_script(self):
        self.loop_csvs() 

    def loop_csvs(self):
        tp_list = self.utils.create_timepoint_list()
        for network in ['PRONET']:
           for tp in tp_list:
                print(tp)
                combined_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}_{network}-day1to1.csv'),
                keep_default_na = False)
                self.create_var_distributions(combined_df)
                self.format_output()

    def create_var_distributions(self, combined_df):
        all_vars = combined_df.columns
        dups = []
        subs = []
        for row in combined_df.itertuples():
            for var in all_vars:
                self.final_output.setdefault(var, 
                {'unique_val_count':0, 'all_vals':[],'duplicates':0})
                val = getattr(row,var)
                if val not in self.final_output[var]['all_vals']:
                    self.final_output[var]['all_vals'].append(val)
                    self.final_output[var]['unique_val_count'] = len(
                        self.final_output[var]['all_vals'])
                else:
                    self.final_output[var]['duplicates'] +=1
                    if var == 'chric_ampscz_id':
                        dups.append(getattr(row, var))
                        #print(dups)
                if var == 'chric_ampscz_id':
                    if getattr(row, var) in ['HA16198', 'yk586', 'yk586',
                     'yk586', 'yk586', 'yk586', 'WU54597']:
                        if row.subjectid not in subs:
                            subs.append(row.subjectid)
                            print(subs)

    def format_output(self):
        formatted_list = []
        for var, vals in self.final_output.items():
            formatted_list.append({'var':var,
            'unique_vals':vals['unique_val_count'],'duplicates':vals['duplicates']})
        df = pd.DataFrame(formatted_list)

        df.to_csv(f'{self.output_path}all_var_unique_vals.csv',
        index = False)


if __name__ == '__main__':
    DuplicateSearcher().run_script()