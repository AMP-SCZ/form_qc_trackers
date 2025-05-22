import pandas as pd

import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-4])
sys.path.insert(1, parent_dir)
from utils.utils import Utils

class ThresholdCheck():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path

        with open(f'{self.absolute_path}/config.json', 'r') as file:
            self.config_info = json.load(file)

        self.comb_csv_path = self.config_info['paths']['combined_csv_path']

    def loop_combined_df(self):
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating', 'conversion'])

        for network in ['PRONET', 'PRESCIENT']:
            for tp in tp_list:
                print(tp)
                print('-------')
                combined_df = pd.read_csv(
                    f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv',
                    keep_default_na=False)


if __name__ == '__main__':
    ThresholdCheck().loop_combined_df()
