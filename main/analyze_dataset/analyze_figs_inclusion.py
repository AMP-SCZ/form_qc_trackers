import pandas as pd
import os
import sys
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
from utils.utils import Utils
import json
import re

# if([chrpsychs_scr_ac7]=1 or [chrpsychs_scr_ac27]=1,1,if(([chrpsychs_scr_ac7]<>''
# and [chrpsychs_scr_ac7]=0) and ([chrpsychs_scr_ac27]<>'' and [chrpsychs_scr_ac27]=0),0,''))

# psychs vars that pull from figs
# chrpsychs_scr_e2
# chrpsychs_fu_e2
# hcpsychs_fu_e2

# exclusion criteria chrcrit_excl5 pulls from chrpsychs_scr_ac1

class AnalyzeFigs():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.depen_path = self.config_info['paths']['dependencies_path']
        self.output_path = self.config_info['paths']['output_path']
        self.data_dict_df = self.utils.read_data_dictionary()

    def run_script(self):
        self.recursive_filter_df(self.data_dict_df)
        #self.filter_inclusion_status()

    def filter_inclusion_status(self):
        for network in ['PRONET','PRESCIENT']:
            screening_df =  pd.read_csv((f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
            f'screening_{network}-day1to1.csv'), keep_default_na = False)
            print(screening_df)

    def search_figs_vars(self):
        pass

    def filter_df_by_list(self, filter_list, inp_df, filtered_col):
        output_df = inp_df[inp_df[
        filtered_col].str.contains("|".join(map(re.escape, filter_list)))]

        return output_df

    def recursive_filter_df(self, inp_df):
        # find any variables that chrpsychs_scr_e2
        # feeds into that is connected to
        # chrcrit_inc3 in any way
        filtered_data_dict = self.data_dict_df[self.data_dict_df[
        'Choices, Calculations, OR Slider Labels'].str.contains('chrpsychs_scr_e2')]

        print(filtered_data_dict)

        psychs_vars = filtered_data_dict['Variable / Field Name'].tolist()
        print('-------')
        print(psychs_vars)

        filtered_df = self.filter_df_by_list(psychs_vars, self.data_dict_df,
        'Choices, Calculations, OR Slider Labels')

        branched_vars = filtered_df['Variable / Field Name'].tolist()
        count = 0

        while not any('chrcrit_inc3' in var for var in branched_vars):
        # while count < 100:
            count += 1
            print(count)
            print(branched_vars)
            filtered_df = self.filter_df_by_list(branched_vars, self.data_dict_df,
            'Choices, Calculations, OR Slider Labels')
            branched_vars = filtered_df['Variable / Field Name'].tolist()
            print(branched_vars)

if __name__ == '__main__':
    AnalyzeFigs().run_script()