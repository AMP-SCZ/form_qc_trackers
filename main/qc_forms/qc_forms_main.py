import pandas as pd

import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from error_compiler import ErrorCompiler

class CheckFormsMain():
    """
    Function Args
        1. subject
        2. variable
        3. affected forms
        5. affected timepoints
        6. network
        7. error
        8. reports
        9. nda exclusion
        10. withdrawn enabled
        11. excluded enabled
        12. error rewordings
    """

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        
        with open(f'{self.absolute_path}/subject_info.json','r') as file:
            self.subject_info = json.load(file)

        with open(f'{self.absolute_path}/general_check_vars.json','r') as file:
            self.general_check_vars = json.load(file)

        with open(f'{self.absolute_path}/important_form_vars.json','r') as file:
            self.important_form_vars = json.load(file)

        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        depen_path = self.config_info['paths']['dependencies_path']
        with open(f'{depen_path}forms_per_timepoint.json','r') as file:
            self.forms_per_tp = json.load(file)

        with open(f'{depen_path}converted_branching_logic.json','r') as file:
            self.conv_bl = json.load(file)

        self.current_row_output = {
            "Network" : "",
            "Subject" :  "",
            "Timepoint" : "",
            "Subject's Current Timepoint" : "",
            "Affected Timepoints" : "",
            "Affected Forms": "",
            "Affected Variables" : "",
            "Displayed Form" : "",
            "Displayed Timepoint" : "",
            "Displayed Variable" : "",
            "Error Message" : "",
            "Error Rewordings" : "",
            "Error Removed" : "",
            "Reports" : "",
            "Withdrawn Status" : "",
            "Inclusion Status" : "",
            "Excluded Enabled" : "",
            "Withdrawn Enabled" : "",
            "NDA Excluder" : "",    
        }

    def run_script(self):
        self.iterate_combined_dfs()

    def iterate_combined_dfs(self):
        tp_list = self.utils.create_timepoint_list()
        for network in ['PRONET','PRESCIENT']:
            for tp in tp_list:
                combined_df = pd.read_csv(
                f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv')

    def filter_qc_check(self, 
        curr_row : tuple, network : str, timepoint : str,
        filtered_forms : list, all_vars : list, bl_filtered_vars : list = [],
        filter_excl_vars : bool = True
    ):
        def apply_filter(func):
            def qc_check(*args, **kwargs):
                cohort = self.subject_info[curr_row.subjectid]['cohort']
                curr_tp_forms = self.forms_per_tp[cohort][timepoint]
                if not (all(form in curr_tp_forms for form in filtered_forms)):
                    return 
                if (not all(self.check_compl_not_missing(
                curr_row, form) for form in filtered_forms)):
                    return
                if not (hasattr(curr_row, var) for var in all_vars):
                    return
                if bl_filtered_vars != []:
                    for var in bl_filtered_vars:
                        bl = self.conv_bl[var]
                        if bl != "" and eval(bl) == False:
                            return
                if filter_excl_vars == True:
                    excl_vars = self.general_check_vars['excluded_vars'][network]
                    if any(var in excl_vars for var in all_vars):
                        return
                                    
                func()

            return qc_check
        
        return apply_filter

    def check_compl_not_missing(self, curr_row : tuple, form):
        compl_var = self.important_form_vars["completion_var"]
        missing_var = self.important_form_vars["completion_var"]

        if (compl_var == "" or not hasattr(curr_row, compl_var)):
            return False
        
        elif missing_var != "" and not hasattr(curr_row, missing_var):
            return False
        
        elif getattr(curr_row, compl_var) not in self.utils.all_dtype([2]):
            return True
        
        elif missing_var == "":
            return True
            
        elif getattr(curr_row, missing_var) not in self.utils.all_dtype([1]):
            return True

        return False


    def qc_timepoint(self):
        pass
                
                
        
if __name__ == '__main__':
    CheckFormsMain().run_script()