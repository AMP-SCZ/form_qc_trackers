import pandas as pd
import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class PrepareDateChecks():

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.data_dict_df = self.utils.read_data_dictionary()

        self.date_chronologies = {}

    def run_script(self):
        self.organize_variable_checks()

    def append_date_chronologies(self):
        """
        Creates file to define when dates should be 
        """
        pass
        


if __name__ == '__main__':
    PrepareDateChecks().run_script()

        