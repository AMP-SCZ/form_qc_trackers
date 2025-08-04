import pandas as pd

import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck
import re

class GeneralChecks(FormCheck):  
    """
    General QC checks that are applied
    to all forms
    """      
    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network,form_check_info)
        self.test_val = 0
        self.call_checks(row)
        
    def __call__(self):

        return self.final_output_list

    def call_checks(self, row):
        self.row = row
        self.check_blank_values(row)
        self.check_missing_code_values(row)
        self.call_spec_val_check(row)
        self.check_form_completion(row)
        self.call_range_checks(row)
        for guid_var in ['chrguid_guid','chrguid_pseudoguid']:
            self.guid_format_check(row, ['guid_form'],
            [guid_var],{'reports':['Main Report','Non Team Forms'],
            "withdrawn_enabled" : True},
            bl_filtered_vars=[guid_var],filter_excl_vars=True,
            checked_guid_var=guid_var)

        #self.missing_code_check(row)

    def call_range_checks(self, row):
        for var, range_dict in self.variable_ranges[self.network].items():
            min = range_dict['min']
            max = range_dict['max']
            self.range_check(row, [range_dict['form']],[var],
            {"reports" : ['Main Report']},bl_filtered_vars=[],
            filter_excl_vars=True,range_var = var,
            lower = min,upper = max)

    def check_blank_values(self, row):
        #TODO:optimize performance of this part
        for report in ['Main Report', 'Secondary Report']:
            blank_check_forms = self.general_check_vars["blank_check_vars"][
            self.network][report]
            for form in blank_check_forms:
                report_list = [report]
                for team, forms in self.forms_per_report.items():
                    if form in forms:
                        report_list.append(team)
                if self.standard_form_filter(row, form):
                    for var in blank_check_forms[form]:
                        #if ('pharm' in var and 'past' not in var):
                        #    print(var)                            
                        if self.prescient_scid_filter(var, row) == True:
                            continue

                        self.check_if_blank(row, [form], [var],
                        {"reports" : report_list},[var])

    def check_missing_code_values(self, row):
        #TODO:optimize performance of this part
        for report in ['Main Report']:
            missing_code_forms = self.general_check_vars["missing_code_vars"][
            self.network][report]
            for form in missing_code_forms:
                report_list = [report]
                for team, forms in self.forms_per_report.items():
                    if form in forms:
                        report_list.append(team)
                if self.standard_form_filter(row, form):
                    for var in missing_code_forms[form]:
                        if self.prescient_scid_filter(var, row) == True:
                            continue
                        self.check_if_missing_code(row, [form], [var],
                        {"reports" : report_list},[var])

    def prescient_scid_filter(self, var, row):
        """
        Filters out modules
        not used for Prescient 
        """
        if var in self.module_b_vars or var in self.module_c_vars:
            if self.network == 'PRESCIENT':
                if (hasattr(row,'chrscid_b1') and row.chrscid_b1 in (
                self.utils.missing_code_list + [''])
                or hasattr(row,'chrscid_b16') and row.chrscid_b16 in (
                self.utils.missing_code_list + [''])):
                    return True
        return False

    @FormCheck.standard_qc_check_filter
    def check_if_blank(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):  
        """
        Standard check applied across all
        forms to see if form is blank
        """
        if hasattr(row,all_vars[0]) and getattr(row, all_vars[0]) == '':
            return "Variable is blank."
        return 

    @FormCheck.standard_qc_check_filter
    def check_if_missing_code(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):  
        if  (getattr(row, all_vars[0]) 
        in self.utils.missing_code_list or
        str(getattr(row, all_vars[0])).replace(' ','') == 'NaN'):
            return "Variable is a missing code."
        return 
    
    def call_spec_val_check(self,row):
        for var, conditions in self.general_check_vars["specific_val_check_vars"].items():
            if var in self.grouped_vars['var_forms'].keys():
                form = self.grouped_vars['var_forms'][var]
                if self.standard_form_filter(row, form):
                    self.specific_value_checks(row,[form],[var],
                    {'reports':conditions['report']},[],True,conditions=conditions)

    @FormCheck.standard_qc_check_filter
    def specific_value_checks(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, conditions={}
    ):
        var = conditions['correlated_variable']
        if hasattr(row, var): 
            if conditions["negative"] == 'False':
                if getattr(row, var) in conditions["checked_value_list"]:
                    return conditions['message']
            elif conditions["negative"] == 'True':
                if getattr(row, var) not in conditions["checked_value_list"]:
                    return conditions['message']
                
    def check_form_completion(self,row):
        cohort = self.subject_info[row.subjectid]['cohort']
        if cohort.lower() not in ["hc", "chr"]:
            return
        curr_tp_forms = self.forms_per_tp[cohort][self.timepoint]
        if (self.check_if_next_tp(row) == True ):
            for form in curr_tp_forms:
                if form in self.prescient_forms_no_compl_status:
                    continue
                compl_var = self.important_form_vars[form]['completion_var']   
                if self.network == 'PRESCIENT':
                    compl_var += '_rpms'
                    # rpms compl variables same for both cohorts
                    compl_var = compl_var.replace('_hc','') 
                    if 'informed_consent' in compl_var:
                        return
                if (hasattr(row, compl_var) and 
                getattr(row, compl_var) not in self.utils.all_dtype([2,3,4])):
                    error_message = f"{form} not marked as complete, but subject has started the next timepoint"
                    if self.network == 'PRONET':
                        output_changes = {'reports' : ['Incomplete Forms']}
                    else:
                        output_changes = {'reports' : ['Main Report','Incomplete Forms']}
                    error_output = self.create_row_output(
                    row,[form],[compl_var], error_message, output_changes)

                    self.final_output_list.append(error_output)

    def guid_format_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, checked_guid_var = 'chrguid_guid'
    ):
        """Checks if GUID is in proper format"""
        if not hasattr(row,checked_guid_var):
            return
        guid = str(getattr(row,checked_guid_var))
        if guid == '' or guid in self.utils.missing_code_list:
            return
        if not re.search(r"^NDA[A-Z0-9]+$", guid):
            error_message = f"GUID in incorrect format. GUID was reported to be {guid}."
            error_output = self.create_row_output(
            row,filtered_forms,[checked_guid_var], error_message,
            changed_output_vals)
            self.final_output_list.append(error_output)

    def range_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, range_var = '', lower = 0, upper = 100
    ):  
        if not hasattr(row, range_var):
            return
        var_val = getattr(row, range_var)
        if (self.utils.can_be_float(var_val) and
        var_val not in self.utils.missing_code_list):
            if float(var_val) < lower or float(var_val) > upper:
                error_message = f'{range_var} value ({var_val}) is out of range'
                error_output = self.create_row_output(
                row, filtered_forms, [range_var], error_message, 
                changed_output_vals)
                self.final_output_list.append(error_output)

    def age_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        """
        Checks all ages to make sure
         they are in the correct range.
        """
        age = self.subject_info[row.subjectid]["age"]
        if age == "unknown":
            return "Age is Unknown."
        elif self.utils.can_be_float(age) and age < 12 or age > 30:
            return f"Age ({age}) is out of range."

    def missing_code_check(self, row):
        for form, vars in self.important_form_vars.items():
            if 'missing_spec_var' in list(vars.keys()) and vars['missing_spec_var'] != '':                
                if (hasattr(row, vars['missing_spec_var'])
                and getattr(row, vars['missing_var']) in self.utils.all_dtype([1])
                and getattr(row, vars['missing_spec_var']) == ''):
                    form = self.grouped_vars['var_forms'][vars['missing_spec_var']]
                    error_message = f'Missing data button clicked, but reason not specified.'
                    error_output = self.create_row_output(
                    row, [form], [vars['missing_spec_var'],vars['missing_var']], error_message, 
                     {"reports" : ['Main Report']})
                    self.final_output_list.append(error_output)
