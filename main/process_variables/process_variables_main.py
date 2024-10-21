import pandas as pd
import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from define_important_variables import DefineImportantVariables

class ProcessVariables():
    """
    Class to process and organize
    different variables in the data dictionary
    that will be used for the QC. This includes 
    collecting important variable names (interview dates,
    missing variables, etc), organizing variables into
    their respective reports,and grouping together 
    variables for different types of QC checks.
    """

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

    def run_script(self):
        data_dict_df = self.utils.read_data_dictionary()
        important_form_vars = DefineImportantVariables(data_dict_df)
        self.utils.save_dictionary_as_csv(important_form_vars(),
        f"{self.config_info['paths']['dependencies_path']}important_form_vars.csv")

 
if __name__ == '__main__':
    ProcessVariables().run_script()

        