import pandas as pd
import os
import sys
import json
import re
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)
from main.utils.utils import Utils
from logging_config import logger  

class REDCapSimulator():
    def __init__(self):
        self.utils = Utils()
        self.data_dictionary_df = self.utils.read_data_dictionary()
        with open(f'{self.utils.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.combined_csv_path = self.config_info['paths']['combined_csv_path']
        depend_path = self.config_info['paths']['dependencies_path']
        self.output_path = self.config_info['paths']['output_path']
        self.excl_bl = self.utils.load_dependency_json('excluded_branching_logic_vars.json')
        self.miss_codes = self.utils.missing_code_list
        self.exported_redcap_data = pd.read_csv(
        f'{depend_path}PRONETTESTALLRECORDS_DATA_2026-02-19_1443.csv',
        keep_default_na = False)
        print(self.exported_redcap_data.shape)
        self.exported_redcap_data.rename(columns={"chric_record_id": "subjectid"}, inplace=True)
        

    def run_script(self):
        self.loop_combined_csv()

    def loop_combined_csv(self):
        # TODO: split checks by ones that will only be checked 
        # if a form in compl and no
        # t missing and ones 
        # that will be checked regardless
        vals_only_in_redcap = []
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        for network in ['PRONET']:
            for tp in tp_list:
                tp = tp.replace("month","month_").replace("floating","floating_forms")
                combined_df = pd.read_csv(
                (f'{self.combined_csv_path}AMPSCZ-combined-redcap_'
                f'{tp}_{network}-day1to1.csv'),
                keep_default_na = False, on_bad_lines='skip')
                
                filtered_redcap_df = self.exported_redcap_data[
                self.exported_redcap_data["redcap_event_name"].str.contains(f"{tp}_")]
                out_diffs = f"{self.output_path}differences_{tp}.csv"
                out_only_1 = f"{self.output_path}only_in_file1_{tp}.csv"
                out_only_2 = f"{self.output_path}only_in_file2_{tp}.csv"

                self.utils.compare_dataframes(
                combined_df,filtered_redcap_df,
                out_diffs,out_only_1,out_only_2)
                print(tp)
                print(filtered_redcap_df.shape)
                filtered_cols = list(filtered_redcap_df.columns)
                comb_cols = list(combined_df.columns)

                only_in_filtered = [col for col in filtered_cols if col not in comb_cols]
                only_in_combined = [col for col in comb_cols if col not in filtered_cols]
                for row in filtered_redcap_df.itertuples():
                    for col in only_in_filtered:
                        if ('___' in col or col.endswith('_complete')
                        or col in ['redcap_event_name']):
                            continue
                        val = getattr(row,col)
                        if val not in (self.miss_codes + ['','nan']):
                            subject = getattr(row,'subjectid')
                            print(val)
                            print(col)
                            print(subject)
                            vals_only_in_redcap.append({'subject':subject,
                            'timepoint':tp,'variable':col,'value':val})
                vals_only_in_redcap_df = pd.DataFrame(vals_only_in_redcap)
                vals_only_in_redcap_df.to_csv(
                f'{self.output_path}values_only_in_redcap.csv',
                index = False)
                
                #print(only_in_filtered)
                #print(only_in_combined)



if __name__ =='__main__':
    REDCapSimulator().run_script()
