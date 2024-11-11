import pandas as pd

import os
import sys
import json
from datetime import datetime
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils


class HarmonizationQC():

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.dep_path = self.config_info['paths']['dependencies_path']
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.miss_codes = self.utils.missing_code_list

        self.network_means = {}
        self.data_dict_df = self.utils.read_data_dictionary()
        self.field_types = self.data_dict_df.set_index('Variable / Field Name')['Field Type'].to_dict()


    def run_script(self):
        self.collect_all_var_means()

    def collect_all_var_means(self):
        timepoints = self.utils.create_timepoint_list()
        for network in ['PRONET','PRESCIENT']:
            for tp in timepoints:
                self.network_means.setdefault(tp, {})
                print(tp)
                combined_df = pd.read_csv(
                f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv',
                keep_default_na = False)
                self.find_num_var_means(combined_df, network, tp)

        df = self.dictionary_to_df(self.network_means)

        df.to_csv('network_mean_comparison.csv', index = False)


    def find_num_var_means(self, df, network, tp):
        print(self.miss_codes)
        for column in df.columns:
            curr_col_df = df[[column]]
            curr_col_df[column] = curr_col_df[column].astype(str).str.strip()

            curr_col_df = curr_col_df[~curr_col_df[column].isin(self.miss_codes)]
            numeric_column = pd.to_numeric(curr_col_df[column], errors='coerce')
            column_mean = numeric_column.mean()
            self.network_means[tp].setdefault(column,{'PRONET':'','PRESCIENT':''})
            self.network_means[tp][column][network] = column_mean
            
    def dictionary_to_df(self, inp_dict):
        df_list = []
        for tp, variables in inp_dict.items():
            for var, network_means in variables.items():
                presc_mean = network_means['PRESCIENT']
                pron_mean = network_means['PRONET']
                perc_diff = self.percentage_difference(presc_mean,pron_mean)
                if var in self.field_types.keys():
                    field_type = self.field_types[var]
                else:
                    field_type = ''
                df_list.append({'Timepoint':tp,
                'Variable':var, 'Pronet_Mean':pron_mean,
                'Prescient_Mean':presc_mean,
                'percentage_difference':perc_diff,'field_type':field_type})
        df = pd.DataFrame(df_list)
        return df

    def percentage_difference(self,num1, num2):
        if not all(self.utils.can_be_float(num) for num in [num1,num2]):
            return 0
        average = (num1 + num2) / 2
        if average == 0:
            return 0
        diff = abs(num1 - num2)
        percentage_diff = (diff / average) * 100
        return percentage_diff

if __name__ == '__main__':
    HarmonizationQC().run_script()