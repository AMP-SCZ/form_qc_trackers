import pandas as pd
import os
import sys
import json
import unittest

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)
from main.process_variables.transform_branching_logic import TransformBranchingLogic
from main.utils.utils import Utils
from logging_config import logger  

class TestTransformBranchingLogic(unittest.TestCase):

    def __init__(self):
        self.utils = Utils()
        self.data_dictionary_df = self.utils.read_data_dictionary()
        self.transform_branching_logic = TransformBranchingLogic(self.data_dictionary_df)
        self.convert_bl = self.transform_branching_logic.convert_all_branching_logic
        with open(f'{self.utils.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.combined_csv_path = self.config_info['paths']['combined_csv_path']
        self.miss_codes = self.utils.missing_code_list

    def run_script(self):
        self.compare_bl_to_database()

    def compare_bl_to_database(self):
        #TODO: check for indentifiers in branching logic
        converted_bl = self.convert_bl()
        exceptions = []
        count = 0
        tp_list = self.utils.create_timepoint_list()
        for network in ['PRONET','PRESCIENT']:
            for tp in tp_list:
                combined_df_path = f'{self.combined_csv_path}combined-{network}-{tp}-day1to1.csv'
                combined_df = pd.read_csv(combined_df_path,keep_default_na=False)
                for row in combined_df.itertuples():
                    for col in combined_df.columns:
                        if col not in converted_bl.keys():
                            continue
                        if any(x in col for x in [
                        'error','chrsaliva_flag','chrchs_flag','_err','invalid','notes']):
                            continue
                        bl = converted_bl[col]['converted_branching_logic']
                        if bl == '':
                            continue
                        #if 'psychs' in col:
                        #    continue
                        if '_info' in bl:
                            continue
                        #if 'scid' in col:
                        #    continue
                        #if 'chrap' in col:
                        #    continue
                        #if 'tbi' in col:
                        #    continue
                        try:
                            """if getattr(row,col) not in (self.miss_codes + ['',0,'0']) and eval(bl) == False:
                                print(row.subjectid)
                                print(tp)
                                print(col)
                                print(getattr(row,col))
                                print(bl)  
                                print('---------------')"""
                        except Exception as e:
                            if 'arm' not in str(e):
                                print(col)
                                print(bl)
                                print(e)
                            continue
                            


if __name__ =='__main__':
    TestTransformBranchingLogic().run_script()
