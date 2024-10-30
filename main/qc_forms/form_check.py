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

        self.tp_list = self.utils.create_timepoint_list()
        
        self.subject_info = self.utils.load_dependency_json('subject_info.json')
        self.general_check_vars = self.utils.load_dependency_json('general_check_vars.json')
        self.important_form_vars = self.utils.load_dependency_json('important_form_vars.json')
        self.forms_per_tp = self.utils.load_dependency_json('forms_per_timepoint.json')
        self.var_info = self.utils.load_dependency_json('var_info.json')
        self.conv_bl = self.utils.load_dependency_json('converted_branching_logic.json')
        self.excl_bl = self.utils.load_dependency_json('excluded_branching_logic_vars.json')
        self.forms_per_report = self.utils.load_dependency_json('team_report_forms.json')

    def call_checks(self):
        pass
    
    @classmethod
    def standard_qc_check_filter(cls, func):
        def qc_check(instance, curr_row, filtered_forms, all_vars,changed_output_vals={}, bl_filtered_vars=[],
        filter_excl_vars=True, *args, **kwargs):
            cohort = instance.subject_info[curr_row.subjectid]['cohort']
            if cohort.lower() not in ["hc", "chr"]:
                return
            curr_tp_forms = instance.forms_per_tp[cohort][instance.timepoint]
            if not (all(form in curr_tp_forms for form in filtered_forms)):
                return
            if not (all(instance.standard_form_filter(
            curr_row, form) for form in filtered_forms)):
                return
            if not all(hasattr(curr_row, var) for var in all_vars):
                return
            
            if filter_excl_vars:
                excl_vars = instance.general_check_vars['excluded_vars'][instance.network]
                if any(var in excl_vars for var in all_vars):
                    return
                
            error_message = func(instance,curr_row,
            filtered_forms,all_vars,changed_output_vals={},
            bl_filtered_vars=[],filter_excl_vars=True, *args, **kwargs)

            if error_message == None:
                return
        
            if bl_filtered_vars != []:
                for var in bl_filtered_vars:
                    if var in instance.excl_bl.keys():
                        return 
                    bl = instance.conv_bl[var]["converted_branching_logic"]
                    if bl != "" and eval(bl) == False:
                        return
                    
            error_output = instance.create_row_output(
            curr_row,filtered_forms,all_vars,error_message, changed_output_vals)

            instance.final_output_list.append(error_output)

        return qc_check

    def standard_form_filter(self, curr_row : tuple, form):
        compl_var = self.important_form_vars[form]["completion_var"]
        missing_var = self.important_form_vars[form]["missing_var"]
        non_bl_vars = self.important_form_vars[form]["non_branch_logic_vars"]
        non_bl_vars_filled_out = 0

        formatted_visit_status = (curr_row.visit_status).replace('_','')
        next_tp = False
        if self.tp_list.index(formatted_visit_status) > self.tp_list.index(self.timepoint):
            next_tp = True

        completion_filter = False
        # will not check the form if it is not marked as complete
        # or the subject has not moved onto the next timepoint (prescient only)
        if (compl_var == "" or not hasattr(curr_row, compl_var)):
            return False

        if ((self.network == 'PRESCIENT' and next_tp == True)
        or getattr(curr_row, compl_var) in self.utils.all_dtype([2])):
            completion_filter = True
        
        if completion_filter == False:
            return False
        
        elif (missing_var != "" and not hasattr(curr_row, missing_var)):
            return False
        
        if missing_var == "":
            for non_bl_var in non_bl_vars:
                if (hasattr(curr_row,non_bl_var)
                and getattr(curr_row,non_bl_var) != ''):
                    non_bl_vars_filled_out +=1
            if non_bl_vars_filled_out < (len(non_bl_vars)/2):
                return False
            else:
                return True
            
        elif getattr(curr_row, missing_var) not in self.utils.all_dtype([1]):
            return True

        return False
    
    def create_row_output(
        self, curr_row : tuple, forms: list,
        variables : list, error_message : str,
        output_changes : dict = {}
    ):
        subject = curr_row.subjectid
        if curr_row.visit_status_string == 'removed':
            removed_status = True
        else:
            removed_status = False

        row_output = {
            "network" : self.network,
            "subject" : subject,
            "affected_timepoints" : [self.timepoint],
            "subject_current_timepoint" : self.subject_info[subject]["visit_status"],
            "affected_forms": forms,
            "affected_variables" : variables,
            "displayed_form" : forms[0],
            "displayed_timepoint" : self.timepoint,
            "displayed_variable" : variables[0],
            "error_message" : error_message,
            "error_rewordings" : [],
            "error_removed" : False,
            "reports" : ["Main Report"],
            "withdrawn_status" : removed_status,
            "inclusion_status" : self.subject_info[subject]["inclusion_status"],
            "excluded_enabled" : False,
            "withdrawn_enabled" : False,
            "nda_excluder" : False,    
        }

        if "Main Report" in row_output["reports"]:
            row_output["NDA Excluder"] = True

        if output_changes != {}:
            for key, val in output_changes.items():
                row_output[key] = val
                
        for key in row_output.keys():
            if isinstance(row_output[key], list): 
                row_output[key] = ' | '.join(row_output[key]) 

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











