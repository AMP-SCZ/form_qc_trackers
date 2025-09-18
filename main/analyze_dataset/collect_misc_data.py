import pandas as pd
import os
import sys 
import json
import numpy as np

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
print(parent_dir)
from utils.utils import Utils

class CollectData():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.depend_path = self.config_info['paths']['dependencies_path']
        self.output_path = self.config_info['paths']['output_path']
        self.form_vars = self.utils.load_dependency_json('important_form_vars.json')
        self.all_results = []
        self.final_output = []
    
    def run_script(self):
        self.loop_timepoints()


    def loop_timepoints(self):
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        for network in ['PRONET']:
            for tp in tp_list:
                if tp != 'screening':
                    continue
                combined_df = pd.read_csv(
                f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv')
                for row in combined_df.itertuples():
                    for form, all_vars in self.form_vars.items():
                        compl_var = all_vars['completion_var']
                        if compl_var != '':
                            print(compl_var)
                        

if __name__ == '__main__':
    CollectData().run_script()
