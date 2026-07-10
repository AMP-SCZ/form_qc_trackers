import pandas as pd 
import json
import os
import sys

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class DataMerger():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        self.earliest_latest_dates_per_tp = {}
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.depen_path = self.config_info['paths']['dependencies_path']
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.data_dict = self.utils.read_data_dictionary()
        self.merged_df = pd.DataFrame()
        
    def run_script(self):
        self.load_csvs() 

    def collect_vars_to_check(self):
        categs_to_keep = ['calc','radio','yesno']
        filtered_df = self.data_dict[self.data_dict[
        'Field Type'].isin(categs_to_keep)]
        vars_to_preserve = filtered_df['Variable / Field Name'].tolist()
        vars_to_preserve.extend(['subjectid','event_name',
        'visit_status','visit_status_string'])

        return vars_to_preserve

    def load_csvs(self):
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        cols_to_keep = self.collect_vars_to_check()
        for network in ['PRONET','PRESCIENT']:
            for tp in tp_list:
                combined_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}'
                f'_{network.replace("PRONET","ProNET")}-day1to1.csv'),
                keep_default_na = False)
                combined_df = combined_df[[c for c in cols_to_keep
                if c in combined_df.columns]]
                if self.merged_df.empty:
                    # Previously this set merged_df = combined_df.copy() and
                    # then unconditionally concatenated combined_df again,
                    # duplicating the first chunk on every fresh merge.
                    self.merged_df = combined_df.copy()
                else:
                    self.merged_df = pd.concat([self.merged_df,
                    combined_df], axis=0, ignore_index=True)

        self.merged_df.to_csv(f'{self.depen_path}merged_data.csv', index = False)
                
if __name__ == '__main__':
    DataMerger().run_script()