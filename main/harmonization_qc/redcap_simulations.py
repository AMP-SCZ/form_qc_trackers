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
        output_path = self.config_info['paths']['output_path']
        self.excl_bl = self.utils.load_dependency_json('excluded_branching_logic_vars.json')
        self.miss_codes = self.utils.missing_code_list

    def run_script(self):
        self.loop_combined_csv()

    def loop_combined_csv(self):
        # TODO: split checks by ones that will only be checked 
        # if a form in compl and no
        # t missing and ones 
        # that will be checked regardless
        final_output = []
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        for network in ['PRESCIENT']:
            for tp in tp_list:
                combined_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}_{network}-day1to1.csv'),
                keep_default_na = False, on_bad_lines='skip')
                #combined_df = combined_df.iloc[80:120]
                #combined_df = combined_df.sample(n=20)
                #combined_df = combined_df.sample(n=100, random_state=42)
                for row in combined_df.itertuples(): 
                    #print(row.Index)
                    #TODO: Add tracker for all subjects not existing here 
                if len(final_output) > 0:
                    combined_output_df = pd.DataFrame(final_output)
                    print(combined_df)
                    if combined_output_df.shape[0] > 3000000:
                        print(f"output rows is {combined_output_df.shape[0]}")
                        sys.exit()
                    combined_flags_path = f'{self.output_path}combined_outputs'
                    os.makedirs(combined_flags_path,exist_ok=True)  # Creates the folder and any necessary parent directories
                    new_out_path =f'{combined_flags_path}/new_output/'
                    os.makedirs(new_out_path,exist_ok=True)
                    try:
                        combined_output_df.to_csv(f'{new_out_path}combined_qc_flags.csv', index=False)
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        sys.exit()




if __name__ =='__main__':
    REDCapSimulator().run_script()
