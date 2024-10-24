import pandas as pd

import os
import sys
import json

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
        tp_list = self.utils.create_timepoint_list()
        for network in ['PRONET','PRESCIENT']:
            for tp in tp_list:
                combined_df = pd.read_csv(
                f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv')
                test_output = GeneralChecks(combined_df,tp,network)

                print(test_output())

        
if __name__ == '__main__':
    CheckFormsMain().run_script()