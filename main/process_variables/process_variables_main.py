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
        data_dict_df = self.read_data_dictionary()
        self.define_imp_variables = DefineImportantVariables(data_dict_df)
        important_form_vars = self.define_variables()

    def read_data_dictionary(
        self, match_str : str = 'current_data_dictionary'
    ) -> pd.DataFrame:
        """
        Finds the current data dictionary
        in the data_dictionary dependencies 
        folder. 

        Parameters
        ------------------
        match_str : str
            String that must be in the 
            name of the data dictionary
            file that will be used
        
        Returns 
        -----------------------
        data_dictionary_df : pd.DataFrame
            Pandas dataframe of the entire
            REDCap data dictionary
        """
    
        depend_path = self.config_info['paths']['dependencies_path']
        for file in os.listdir(f"{depend_path}data_dictionary"):
            # loops through directory to search for current data dictionary
            if match_str in file:
                data_dictionary_df = pd.read_csv(
                f"{depend_path}data_dictionary/{file}",
                keep_default_na=False) # setting this to false preserves empty strings

        return data_dictionary_df
 
if __name__ == '__main__':
    ProcessVariables().run_script()

        