import pandas as pd
import os
import sys 
import json
import numpy as np

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
print(parent_dir)
from utils.utils import Utils

class ConversionSearcher():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.depend_path = self.config_info['paths']['dependencies_path']
        self.output_path = self.config_info['paths']['output_path']
        
        
        self.var_list = ['chrbprs_bprs_somc',
        'chrbprs_bprs_guil', 'chrbprs_bprs_gran',
        'chrbprs_bprs_susp', 'chrbprs_bprs_hall',
        'chrbprs_bprs_unus', 'chrbprs_bprs_bizb',
        'chrbprs_bprs_conc', 'chrpsychs_fu_1c0', 'chrpsychs_fu_1d0',
        'chrpsychs_fu_2c0', 'chrpsychs_fu_2d0', 'chrpsychs_fu_3c0',
        'chrpsychs_fu_3d0', 'chrpsychs_fu_4c0', 'chrpsychs_fu_4d0',
        'chrpsychs_fu_5c0', 'chrpsychs_fu_5d0', 'chrpsychs_fu_6c0',
        'chrpsychs_fu_6d0', 'chrpsychs_fu_7c0', 'chrpsychs_fu_7d0',
        'chrpsychs_fu_8c0', 'chrpsychs_fu_8d0', 'chrpsychs_fu_9c0',
        'chrpsychs_fu_9d0', 'chrpsychs_fu_10c0', 'chrpsychs_fu_10d0',
        'chrpsychs_fu_11c0', 'chrpsychs_fu_11d0', 'chrpsychs_fu_12c0',
        'chrpsychs_fu_12d0', 'chrpsychs_fu_13c0', 'chrpsychs_fu_13d0',
        'chrpsychs_fu_14c0', 'chrpsychs_fu_14d0', 'chrpsychs_fu_15c0',
        'chrpsychs_fu_15d0', 'chrscid_c10', 'chrscid_c26', 'chrscid_c14',
        'chrscid_c37', 'chrscid_c44', 'chrscid_d47_d52', 'chrscid_d63',
        'chrscid_c71', 'chrscid_c78', 'chrscid_c11', 'chrscid_c21',
        'chrscid_c47', 'chrscid_c28', 'chrscid_c50', 'chrscid_c51']

        self.all_results = []
    
    def run_script(self):
        self.loop_timepoints()

    def find_relevant_vars(self):
        data_dict_df = self.utils.read_data_dictionary()
        text_num_df = data_dict_df[
        data_dict_df['Field Type'].isin(['calc','text'])]
        text_num_vars = text_num_df['Variable / Field Name'].tolist()
        mult_choice_df = data_dict_df[
        data_dict_df['Field Type'].isin(['radio'])]
        mask = (
            mult_choice_df["Choices, Calculations, OR Slider Labels"]                    
            .astype(str)                 
            .str.count(r"\|")              # count the number of "|" characters
            .gt(3)                         # keep rows where the count > 3
        )

        mult_choice_df = mult_choice_df[mask].copy()        
        mult_choice_vars = mult_choice_df[
        'Variable / Field Name'].tolist()

        all_vars = text_num_vars + mult_choice_vars

        return all_vars

    def merge(self):
        data_dict_df = self.utils.read_data_dictionary()
        conv_df = pd.read_csv('/home/ob001/refactored_qc/form_qc_trackers/main/analyze_dataset/conv_corr.csv')
        data_dict_df.rename(columns={"Variable / Field Name": "Variable"}, inplace=True)

        data_dict_df = data_dict_df[['Variable','Field Label','Choices, Calculations, OR Slider Labels']]
        merged = pd.merge(conv_df, data_dict_df, on ='Variable',how = 'left')
        merged.to_csv('merged_conv_corrs.csv',index = False)
        print(len(conv_df), "rows in conv_df")
        print(len(merged),  "rows after merge") 



    def loop_timepoints(self):
        tp_list = self.utils.create_timepoint_list()
        vars_to_compare = self.find_relevant_vars()
        tp_list.extend(['floating','conversion'])
        for network in ['PRONET','PRESCIENT']:
            for tp in tp_list:
                combined_df = pd.read_csv(
                f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv')
                if any(var in combined_df.columns for var in self.var_list):
                    com_vars = [var for var in self.var_list if var in combined_df.columns]
                    #print(com_vars)
                    self.search_correlations(combined_df, self.var_list,vars_to_compare, tp)
    
    def search_correlations(self, df, target_vars, compared_vars, tp):
        print(tp)
        results = {}
        df_numeric = df.select_dtypes(include=[np.number])
        print('-------')
        print(df_numeric)
        # Drop rows with missing values (optional, or use pairwise correlations)
        
        for target in target_vars:
            if target not in df_numeric.columns:
                print(f"Skipping {target}: not found or not numeric.")
                continue

            # Pairwise correlations using pandas corr() (automatically handles NaNs pairwise)
            corrs = df_numeric.corr()[target].drop(target).dropna()

            for var, r in corrs.items():
                if var not in compared_vars:
                    continue
                self.all_results.append({
                    'Target': target,
                    'Variable': var,
                    'R': r,
                    'AbsR': abs(r)
                })

        if len(self.all_results) > 0:
            summary_df = pd.DataFrame(self.all_results).sort_values(by='AbsR', ascending=False)
            summary_df = summary_df[['Target', 'Variable', 'R']]
            print(summary_df)
            summary_df.to_csv(f'{self.output_path}conv_corr.csv', index = False)

if __name__ == '__main__':
    ConversionSearcher().run_script()
