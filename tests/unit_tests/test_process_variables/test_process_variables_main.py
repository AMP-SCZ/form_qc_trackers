import pandas as pd
import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-4])
sys.path.insert(1, parent_dir)

from main.utils.utils import Utils
from tests.unit_tests.test_process_variables.test_define_important_variables import TestDefineVariables

class TestProcessVariables():

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        print(self.absolute_path)

    def run_script(self):
        data_dict_df = self.read_data_dictionary()
        self.test_define_variables = TestDefineVariables(data_dict_df)
        self.test_define_variables.run_script()

    def read_data_dictionary(self):
        depend_path = self.config_info['paths']['dependencies_path']
        for file in os.listdir(f"{depend_path}data_dictionary"):
            if 'current_data_dictionary' in file:
                data_dictionary_df = pd.read_csv(\
                f"{depend_path}data_dictionary/{file}",\
                keep_default_na=False)

        return data_dictionary_df
 
if __name__ == '__main__':
    TestProcessVariables().run_script()

        