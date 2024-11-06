import pandas as pd

import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck

class GeneralChecks(FormCheck):
    
    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network,form_check_info)
        self.test_val = 0
        self.call_checks(row)
        
        
    def __call__(self):

        return self.final_output_list


    def call_checks(self, row):
        self.row = row
        self.check_blank_values(row)
        self.call_spec_val_check(row)
        self.check_form_completion(row)
    

    def check_blank_values(self, row):
        #TODO:optimize performance of this part
        for report in ['Main Report', 'Secondary Report']:
            blank_check_forms = self.general_check_vars["blank_check_vars"][report]
            for form in blank_check_forms:
                report_list = [report]
                for team, forms in self.forms_per_report.items():
                    if form in forms:
                        report_list.append(team)
                if self.standard_form_filter(row, form):
                    for var in blank_check_forms[form]:
                        self.check_if_blank(row, [form], [var],{"reports": report_list},[var])


    @FormCheck.standard_qc_check_filter
    def check_if_blank(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ): 
        if getattr(row, all_vars[0]) == '':
            return "Variable is Blank"
        return 
    
    def call_spec_val_check(self,row):
        for var, conditions in self.general_check_vars["specific_val_check_vars"].items():
            if var in self.var_info['var_forms'].keys():
                form = self.var_info['var_forms'][var]
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

        if self.check_if_next_tp(row) == True:
            for form in curr_tp_forms:
                compl_var = self.important_form_vars[form]['completion_var']                
                if (hasattr(row, compl_var) and 
                getattr(row, compl_var) not in self.utils.all_dtype([2])):
                    error_message = f"{form} not marked as complete, but subject has started the next timepoint"
                    output_changes = {'reports' : ['Incomplete Forms']}
                    error_output = self.create_row_output(
                    row,[form],[compl_var], error_message, output_changes)

                    self.final_output_list.append(error_output)



                



        


        








        