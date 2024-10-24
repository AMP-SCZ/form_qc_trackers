import pandas as pd

import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class FormCheck():

    def __init__(self, timepoint, network): 
        self.utils = Utils()
        self.timepoint = timepoint
        self.network = network   
        self.absolute_path = self.utils.absolute_path

        self.final_output_list = []

        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        depend_path = self.config_info['paths']['dependencies_path']
        with open(f'{depend_path}/subject_info.json','r') as file:
            self.subject_info = json.load(file)

        with open(f'{depend_path}/general_check_vars.json','r') as file:
            self.general_check_vars = json.load(file)

        with open(f'{depend_path}/important_form_vars.json','r') as file:
            self.important_form_vars = json.load(file)

        with open(f'{depend_path}forms_per_timepoint.json','r') as file:
            self.forms_per_tp = json.load(file)


    def call_checks(self):
        pass
    
    @classmethod
    def filter_qc_check(cls, func):
        def qc_check(instance, curr_row, filtered_forms, all_vars,changed_output_vals={}, bl_filtered_vars=[],
        filter_excl_vars=True, *args, **kwargs):
            cohort = instance.subject_info[curr_row.subjectid]['cohort']
            if cohort.lower() not in ["hc", "chr"]:
                return
            curr_tp_forms = instance.forms_per_tp[cohort][instance.timepoint]
            print(filtered_forms)
            if not (all(form in curr_tp_forms for form in filtered_forms)):
                return
            print('pass')
            if not (all(instance.check_compl_not_missing(
            curr_row, form) for form in filtered_forms)):
                return
            if not all(hasattr(curr_row, var) for var in all_vars):
                return
            if bl_filtered_vars:
                for var in bl_filtered_vars:
                    bl = instance.conv_bl[var]
                    if bl != "" and not eval(bl):
                        return
            if filter_excl_vars:
                excl_vars = instance.general_check_vars['excluded_vars'][instance.network]
                if any(var in excl_vars for var in all_vars):
                    return

            error_message = func(instance,curr_row,
            filtered_forms,all_vars,changed_output_vals={},
            bl_filtered_vars=[],filter_excl_vars=True, *args, **kwargs)
            
            error_output = instance.create_def_row_output(
            curr_row,filtered_forms,all_vars,error_message)

            if error_message != '':
        
                if changed_output_vals:
                    for key, val in changed_output_vals.items():
                        error_output[key] = val
                if error_message != '':
                    instance.final_output_list.append(error_output)
                print(instance.final_output_list)

        return qc_check

    def check_compl_not_missing(self, curr_row : tuple, form):
        compl_var = self.important_form_vars[form]["completion_var"]
        missing_var = self.important_form_vars[form]["completion_var"]

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
    
    def create_def_row_output(
        self, curr_row : tuple, forms: list, variables : list, error_message : str
    ):
        subject = curr_row.subjectid
        row_output = {
            "Network" : self.network,
            "Subject" : subject,
            "Affected Timepoints" : [self.timepoint],
            "Subject's Current Timepoint" : self.subject_info[subject]["visit_status"],
            "Affected Forms": forms,
            "Affected Variables" : variables,
            "Displayed Form" : forms[0],
            "Displayed Timepoint" : self.timepoint,
            "Displayed Variable" : variables[0],
            "Error Message" : error_message,
            "Error Rewordings" : [],
            "Error Removed" : False,
            "Reports" : ["Main Report"],
            "Withdrawn Status" : "",
            "Inclusion Status" : self.subject_info[subject]["inclusion_status"],
            "Excluded Enabled" : False,
            "Withdrawn Enabled" : False,
            "NDA Excluder" : "",    
        }

        if "Main Report" in row_output["Reports"]:
            row_output["NDA Excluder"] = True

        return row_output
    
    def format_lists(self, dict_to_format):
        for key in dict_to_format.keys():
            if isinstance(dict_to_format[key], list):
                if len(dict_to_format[key]) > 0:
                    dict_to_format[key] = self.list_to_string(dict_to_format[key])
                else:
                    dict_to_format[key] =''

        return dict_to_format

    def list_to_string(self, inp_list):
        inp_list = [str(item) for item in inp_list]

        inp_list = '|'.join(inp_list)

        return inp_list











