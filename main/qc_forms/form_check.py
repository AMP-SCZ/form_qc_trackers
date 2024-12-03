import pandas as pd

import os
import sys
import json
from datetime import datetime
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class FormCheck():

    def __init__(self, timepoint, network,form_check_info): 
        self.utils = Utils()
        self.timepoint = timepoint
        self.network = network   
        self.absolute_path = self.utils.absolute_path

        self.final_output_list = []

        self.tp_list = self.utils.create_timepoint_list()
        
        self.subject_info = form_check_info['subject_info'] 
        self.general_check_vars = form_check_info['general_check_vars'] 
        self.important_form_vars = form_check_info['important_form_vars'] 
        self.forms_per_tp = form_check_info['forms_per_timepoint'] 
        self.conv_bl = form_check_info['converted_branching_logic']
        self.excl_bl = form_check_info['excluded_branching_logic_vars']
        self.forms_per_report = form_check_info['team_report_forms']
        self.grouped_vars = form_check_info['grouped_variables']
        self.missing_code_list = self.utils.missing_code_list

        self.prescient_forms_no_compl_status = [
        'family_interview_for_genetic_studies_figs']

        self.priority_forms = ['guid_form','bprs','psychs_p1p8','psychs_p9ac32',
        'psychs_p1p8_fu','psychs_p9ac32_fu','psychs_p1p8_fu_hc','psychs_p9ac32_fu_hc',
        'sociodemographics','scid5_psychosis_mood_substance_abuse']

        self.priority_timepoints = ['screening']

    def call_checks(self):
        pass
    
    @classmethod
    def standard_qc_check_filter(cls, func):
        def qc_check(instance, curr_row, filtered_forms,
        all_vars,changed_output_vals={}, bl_filtered_vars=[],
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
                    print(len(excl_vars))
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

    def check_if_next_tp(self, curr_row):
        if self.timepoint in ['floating','conversion']:
            return False
        formatted_visit_status = (curr_row.visit_status).replace('_','')
        if self.tp_list.index(formatted_visit_status) > self.tp_list.index(self.timepoint):
            return True
        
        return False

    def standard_form_filter(self, curr_row : tuple, form):
        compl_var = self.important_form_vars[form]["completion_var"]
        missing_var = self.important_form_vars[form]["missing_var"]
        non_bl_vars = self.important_form_vars[form]["non_branch_logic_vars"]
        
        formatted_visit_status = (curr_row.visit_status).replace('_','')
        completion_filter = False
        # will not check the form if it is not marked as complete
        # or the subject has not moved onto the next timepoint (prescient only)
        if (compl_var == "" or not hasattr(curr_row, compl_var)):
            return False
        
        if ((self.network == 'PRESCIENT' and self.check_if_next_tp(curr_row) == True)
        or getattr(curr_row, compl_var) in self.utils.all_dtype([2])):
            completion_filter = True
        if (self.network == 'PRESCIENT' and form
        in self.prescient_forms_no_compl_status 
        and self.check_if_next_tp(curr_row) == False):
            completion_filter = False
        if completion_filter == False:
            return False
        
        if self.check_if_missing(curr_row, form) == True:
            return False
                   
        return True
    
    def check_if_missing(self,curr_row : tuple, form):
        compl_var = self.important_form_vars[form]["completion_var"]
        missing_var = self.important_form_vars[form]["missing_var"]
        non_bl_vars = self.important_form_vars[form]["non_branch_logic_vars"]
        non_bl_vars_filled_out = 0

        if missing_var != "":
            if not hasattr(curr_row, missing_var):
                return False
            # prescient missingness can also be indicated by the completion var
            if (self.network == 'PRESCIENT' and
            getattr(curr_row, compl_var) in self.utils.all_dtype([3,4])):
                return True
            if getattr(curr_row, missing_var) not in self.utils.all_dtype([1]):
                return False
            else:
                return True
        
        elif missing_var == "":                
            for non_bl_var in non_bl_vars:
                if (hasattr(curr_row,non_bl_var)
                and getattr(curr_row,non_bl_var) != ''):
                    non_bl_vars_filled_out +=1
            if non_bl_vars_filled_out < (len(non_bl_vars)/2):
                return True
            else:
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

        incl_status = self.subject_info[subject]["inclusion_status"]

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
            "var_translations" : [],
            "error_message" : error_message,
            "error_rewordings" : [],
            "error_removed" : False,
            "reports" : ["Main Report"],
            "withdrawn_status" : removed_status,
            "inclusion_status" : incl_status,
            "excluded_enabled" : False,
            "withdrawn_enabled" : False,
            "nda_excluder" : False,  
            "priority_item" : False,
            "dates_detected" : str(datetime.today().date()).split(' ')[0],
            "time_since_last_detection":"",
            "dates_resolved" : "",
            "currently_resolved": False,
            "manually_resolved" : "",
            "comments" : "",
            "site_comments" : ""
        }

        if "Main Report" in row_output["reports"]:
            row_output["nda_excluder"] = True

        var_translations = self.grouped_vars['var_translations']

        for var in variables:
            if var in var_translations.keys():
                row_output["var_translations"].append(
                self.grouped_vars['var_translations'][var]) 

        if self.timepoint == 'screening':
            row_output["priority"] = True
        
        row_output['error_message'] = row_output[
        'displayed_variable'] + ' : ' + row_output['error_message']

        if output_changes != {}:
            for key, val in output_changes.items():
                row_output[key] = val

        # if priority form or tp set priority_item to True 
        if (any(form in self.priority_forms
        for form in row_output['affected_forms'])
        or any(tp in self.priority_timepoints
        for tp in row_output['affected_timepoints'])):
            row_output['priority_item'] = True

        if (row_output['withdrawn_enabled'] == False
        and removed_status == True):
            row_output['reports'] = []

        if (row_output['excluded_enabled'] == False
        and incl_status.lower() != 'included'):
            row_output['reports'] = []
                
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



