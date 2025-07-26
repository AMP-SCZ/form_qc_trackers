import pandas as pd
import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-4])
sys.path.insert(1, parent_dir)
from utils.utils import Utils
from qc_forms.form_check import FormCheck
from qc_forms.qc_types.clinical_checks.scid_checks import ScidChecks
from datetime import datetime,timedelta

class MultiTPChecks(FormCheck): 
    """
    QC Checks that compare 
    forms from different timepoints 
    using the dataframe that combines 
    each timepoint for specified variables
    """  
   
    def __init__(self, row, timepoint, network, form_check_info, all_vars):
        super().__init__(timepoint, network, form_check_info)
        self.timepoint = timepoint
        self.blood_vars = self.grouped_vars['blood_vars']
        self.all_col_names = all_vars
        self.call_checks(row)
               
    def __call__(self):
        return self.final_output_list

    def call_checks(self, row):
        print('RUNNING MULTI TP CHECK')
        self.check_blood_id_duplicates(row)
        figs_pps_ages = {'chrpps_mage_baseline':'chrfigs_mother_age_screening', 
        'chrpps_fage_baseline':'chrfigs_father_age_screening'}
        forms = ['family_interview_for_genetic_studies_figs',
        'psychosis_polyrisk_score']
        reports = {"reports" : ['Main Report']}

        for figs_var, pps_var in figs_pps_ages.items():
            self.figs_pps_age_check(curr_row = row, filtered_forms = forms, 
            all_vars = [figs_var, pps_var], changed_output_vals = reports,
            bl_filtered_vars = [], filter_excl_vars = True,
            pps_age_var = pps_var, figs_age_var = figs_var)

    @FormCheck.standard_qc_check_filter
    def figs_pps_age_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, pps_age_var = '', figs_age_var = ''
    ):
        if any((getattr(row,var) in self.utils.missing_code_list
        or not self.utils.can_be_float(getattr(row,var)))
        for var in [pps_age_var,figs_age_var]):
            return 
        
        if float(getattr(row,pps_age_var)) != float(getattr(row, figs_age_var)):
            return f"{pps_age_var} is equal to {getattr(row,pps_age_var)},"
            f" but {figs_age_var} is equal to {float(getattr(row, figs_age_var))}"

    def check_blood_id_duplicates(self, row):
        all_id_vals = {}
        for var in self.blood_vars['id_variables']:
            for col in self.all_col_names:
                if var in col:
                    all_id_vals[col] = getattr(row,col)

        for init_var, init_val in all_id_vals.items():
            for second_var, second_val in all_id_vals.items():
                if init_var != second_var:
                    if init_val == second_val:
                        error_message = (f"Duplicate IDs found between different"
                        f" variables ({init_var} = {init_val} / {second_var} = {second_val})")
                        error_output = self.create_row_output(
                        row, filtered_forms, [scores['raw'], scores['scaled']],
                        error_message, reports)
                        self.final_output_list.append(error_output)



                    

        
        