import pandas as pd
import os
import sys
import json
import re
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)
from main.process_variables.transform_branching_logic import TransformBranchingLogic
from main.utils.utils import Utils
from logging_config import logger  

class TestTransformBranchingLogic():
    def __init__(self):
        self.utils = Utils()
        self.data_dictionary_df = self.utils.read_data_dictionary()
        self.transform_branching_logic = TransformBranchingLogic(self.data_dictionary_df)
        self.convert_bl = self.transform_branching_logic.convert_all_branching_logic
        with open(f'{self.utils.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.combined_csv_path = self.config_info['paths']['combined_csv_path']
        output_path = self.config_info['paths']['output_path']
        identifier_df = pd.read_csv(f"{output_path}identifier_effects.csv")
        self.ident_bl_vars = identifier_df[
        identifier_df['affected_col'] == 'branching_logic']['var'].tolist()
        self.excl_bl = self.utils.load_dependency_json('excluded_branching_logic_vars.json')
        print(self.ident_bl_vars)
        self.miss_codes = self.utils.missing_code_list

        self.var_forms = self.utils.load_dependency_json(
        f"grouped_variables.json")
        self.forms_per_var = self.var_forms['var_forms']
        self.important_form_vars = self.utils.load_dependency_json(
        'important_form_vars.json')
        

    def run_script(self):
        self.compare_bl_to_database()

    def compare_bl_to_database(self):
        #TODO: check for indentifiers in branching logic
        converted_bl = self.convert_bl()
        exceptions = []
        output_list = []
        # condensed by removing and counting duplicates
        dupl_removed_output = {}
        dupl_removed_output_list = []
        count = 0
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        for network in ['PRONET','PRESCIENT']:
            dupl_removed_output = {}
            for tp in tp_list:
                print(tp)
                dupl_removed_output_list = []
                combined_df_path = f'{self.combined_csv_path}combined-{network}-{tp}-day1to1.csv'
                combined_df = pd.read_csv(combined_df_path,keep_default_na=False)
                for curr_row in combined_df.itertuples():
                    for col in combined_df.columns:
                        if col not in self.forms_per_var.keys():
                            continue
                        form = self.forms_per_var[col]
                        if form in self.important_form_vars.keys():
                            interview_date_var = self.important_form_vars[form]['interview_date_var']
                            missing_var = self.important_form_vars[form]['missing_var']
                            complete_var = self.important_form_vars[form]['completion_var']
                        if (hasattr(curr_row, missing_var)
                        and getattr(curr_row, missing_var)
                        in self.utils.all_dtype([1])):
                            continue
                        if (hasattr(curr_row, complete_var)
                        and getattr(curr_row, complete_var)
                        not in self.utils.all_dtype([2])):
                            continue
                        if col not in converted_bl.keys():
                            continue
                        if col in self.excl_bl.keys():
                            continue
                        if any(x in col for x in [
                        'error','chrsaliva_flag','chrchs_flag','_err','invalid','notes']):
                            continue
                        bl = converted_bl[col]['converted_branching_logic']
                        bl = bl.replace('instance','self')
                        if bl == '':
                            continue
                        if '_info' in bl:
                            continue
                        if col in self.ident_bl_vars and network == "PRONET":
                            continue
                        try:
                            if ((getattr(curr_row, col) == '' or (
                            network =='PRESCIENT' and getattr(curr_row, col)
                            in self.missing_code_list)) and eval(bl) == True):
                                match_key =  network + tp + col + str(getattr(curr_row,col)) + 'True'
                                dupl_removed_output.setdefault(match_key, {
                                'subjects':[],'network':network,'timepoint':tp,
                                'variable':col,'variable_value':getattr(curr_row,col), 'count': 0,
                                'pronet_branching_logic':converted_bl[col]['original_branching_logic'],
                                'converted_branching_logic': bl,'BL_cond':True})
                                dupl_removed_output[match_key]['count'] +=1
                                if len(dupl_removed_output[match_key]['subjects']) < 50:
                                    dupl_removed_output[match_key]['subjects'].append(curr_row.subjectid)
                                output_list.append({'subject':curr_row.subjectid,'network':network,'timepoint':tp,
                                'variable':col,'variable_value':getattr(curr_row,col),
                                'pronet_branching_logic':converted_bl[col]['original_branching_logic'],
                                'BL_Cond':True})

                            elif (getattr(curr_row, col) not in
                            (self.missing_code_list +['']) and eval(bl) == False):
                                match_key =  network + tp + col + str(getattr(curr_row,col)) + 'False'
                                dupl_removed_output.setdefault(match_key, {
                                'subjects':[],'network':network,'timepoint':tp,
                                'variable':col,'variable_value':getattr(curr_row,col), 'count': 0,
                                'pronet_branching_logic':converted_bl[col]['original_branching_logic'],
                                'converted_branching_logic': bl,'BL_cond':False})
                                dupl_removed_output[match_key]['count'] +=1
                                if len(dupl_removed_output[match_key]['subjects']) < 50:
                                    dupl_removed_output[match_key]['subjects'].append(curr_row.subjectid)
                                output_list.append({'subject':curr_row.subjectid,'network':network,'timepoint':tp,
                                'variable':col,'variable_value':getattr(curr_row,col),
                                'pronet_branching_logic':converted_bl[col]['original_branching_logic'],
                                'BL_Cond':False})

                        except Exception as e:
                            """if 'arm' not in str(e):
                                print('error')
                                print(col)
                                print(bl)
                                print(e)"""
                            continue

                for key,val in dupl_removed_output.items():
                    dupl_removed_output_list.append(val)
                output_df = pd.DataFrame(dupl_removed_output_list)
                output_df.to_csv(
                f"{self.config_info['paths']['output_path']}bl_mismatches_{network}.csv",
                index = False)

if __name__ =='__main__':
    TestTransformBranchingLogic().run_script()
