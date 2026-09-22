import pandas as pd
import os
import sys
import json
import re

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class MedNameCollector():
    """
    Class to collect all med names
    for each participants
    """


    def __init__(self, data_dict_df=None):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.comb_csv_path = self.config_info['paths']['combined_csv_path']

        if data_dict_df is None:
            data_dict_df = self.utils.read_data_dictionary()

        self.plus_dosage_meds = {}

    def __call__(self):
        self.collect_plus_dosage_meds()

        return self.format_output()

    def loop_csvs(self):
        pass 


if __name__ == '__main__':
    print(json.dumps(PlusDosageMedCollector()(), indent=4))
