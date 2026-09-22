import os 
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
from process_variables.transform_branching_logic import TransformBranchingLogic

from utils.utils import Utils
import re
import pandas as pd 

class BranchingLogicTester():  
    """
    General QC checks that are applied
    to all forms
    """      
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path

        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.depend_path = self.config_info['paths']['dependencies_path']
        print(self.depend_path)
        ct_data_dict_path = f'{self.depend_path}data_dictionary/data_dict_clinical_trial.csv'
        self.ct_data_dict = pd.read_csv(ct_data_dict_path, 
        keep_default_na = False, encoding ='latin-1')
  
    def run_script(self):
        self.convert_branching_logic()

    def convert_branching_logic(self):
        transform_bl = TransformBranchingLogic(self.ct_data_dict)
        converted_branching_logic = transform_bl()
        self.utils.save_dictionary_as_csv(converted_branching_logic,
        f"{self.config_info['paths']['dependencies_path']}converted_branching_logic_clinical_trial.csv")
    
if __name__ == '__main__':
    BranchingLogicTester().run_script()