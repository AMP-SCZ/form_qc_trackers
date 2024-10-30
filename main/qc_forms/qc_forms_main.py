import pandas as pd

import os
import sys
import json
import random
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

from qc_forms.qc_types.general_checks import GeneralChecks

class CheckFormsMain():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        depen_path = self.config_info['paths']['dependencies_path']
        with open(f'{depen_path}converted_branching_logic.json','r') as file:
            self.conv_bl = json.load(file)

        self.final_output_list = []

    def run_script(self):
        self.iterate_combined_dfs()

    def iterate_combined_dfs(self):
        #TODO: split checks by ones that will only be checked 
        # if a form in compl and not missing and ones 
        # that will be checked regardless
        test_output=[]
        tp_list = self.utils.create_timepoint_list()
        for network in ['PRONET','PRESCIENT']:
            for tp in tp_list:
                combined_df = pd.read_csv(
                f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv',
                keep_default_na = False)
                #combined_df = combined_df.iloc[:10]
                combined_df = combined_df.sample(n=10)
                combined_df = combined_df.sample(n=10, random_state=42)
                for row in combined_df.itertuples():
                    print(row.Index)
                    gen_checks = GeneralChecks(row,tp,network)
                    test_output.extend(gen_checks())
                    print(tp)
                    print(len(test_output))
                    #print(test_output[random.randint(0,len(test_output)-1)])
                test_df = pd.DataFrame(test_output)
                test_df.to_csv('refactored_qc.csv',index = False)

if __name__ == '__main__':
    CheckFormsMain().run_script()