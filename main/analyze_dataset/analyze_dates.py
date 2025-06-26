import pandas as pd
import os
import sys
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
import json

class AnalyzeDates():

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.depen_path = self.config_info['paths']['dependencies_path']

        self.date_vars = self.utils.collect_all_type_vars()
        self.data_dict_df = self.utils.read_data_dictionary()
        vars_not_marked = []
        self.date_df = self.data_dict_df[self.data_dict_df[
        'Text Validation Type OR Show Slider Number'].isin(['date_ymd',
        'datetime_ymd'])]
        self.data_dict_dates = self.date_df['Variable / Field Name'].tolist()
        all_vars = self.data_dict_df['Variable / Field Name'].tolist()
        for var in self.date_vars:
            if var not in self.data_dict_dates and var in all_vars:
                vars_not_marked.append(var)
        print(vars_not_marked)
        self.final_output_list = []

        self.date_gt_comparisons = {}

    def run_script(self):
        self.analyze_dates()
        self.format_output()

    def analyze_dates(self):
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        for network in ['PRONET','PRESCIENT']:
            for tp in tp_list:
                combined_df = pd.read_csv(
                f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv',
                keep_default_na = False)
                #combined_df = combined_df.iloc[80:120]
                #combined_df = combined_df.sample(n=20)
                #combined_df = combined_df.sample(n=100, random_state=42)
                for row in combined_df.itertuples(): 
                    for initial_var in self.date_vars:
                        if hasattr(row, initial_var):
                            initial_date_val = str(getattr(row,initial_var)).split(' ')[0]
                            if self.utils.check_if_val_date_format(initial_date_val):
                                for secondary_var in self.date_vars:
                                    if secondary_var == initial_var:
                                        continue
                                    if hasattr(row, secondary_var):
                                        secondary_date_val = str(
                                        getattr(row, secondary_var)).split(' ')[0]
                                        if self.utils.check_if_val_date_format(secondary_date_val):
                                            self.date_gt_comparisons.setdefault(initial_var + '_' + secondary_var,
                                            {'gt' : 0, 'total' : 0})
                                            self.date_gt_comparisons[initial_var + '_' + secondary_var]['total'] += 1
                                            if initial_date_val > secondary_date_val:
                                                self.date_gt_comparisons[initial_var + '_' + secondary_var]['gt'] += 1
                 
    def format_output(self):
        for var_combo, vals in self.date_gt_comparisons.items():
            if vals['gt'] != 0 and vals['total'] != 0:
                percent = (vals['gt'] / vals['total'] ) * 100
            else:
                percent = 0
            self.final_output_list.append({'dates':var_combo,
            'gt_percent':percent,'total':vals['total']})

        df = pd.DataFrame(self.final_output_list)
        df.to_csv('date_test.csv', index = False)
        print(self.date_gt_comparisons)
        

if __name__ == '__main__':
    AnalyzeDates().run_script()