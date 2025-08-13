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
   
    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network, form_check_info)
        self.timepoint = timepoint
               
    def __call__(self):
        return self.final_output_list

    def call_checks(self, row):
        figs_pps_ages = {'chrpps_mage_baseline':'chrfigs_mother_age_screening', 
        'chrpps_fage_baseline':'chrfigs_father_age_screening'}
        forms = ['family_interview_for_genetic_studies_figs',
        'psychosis_polyrisk_score']
        reports = {"reports" : ['Main Report']}

        for figs_var, pps_var in figs_pps_ages.items():
            self.figs_pps_age_check(row = row, filtered_forms = forms, 
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
        for var in [pps_age_var, figs_age_var]):
            return 
        
        if float(getattr(row,pps_age_var)) != float(getattr(row, figs_age_var)):
            return f"{pps_age_var} is equal to {getattr(row, pps_age_var)},"
            f" but {figs_age_var} is equal to {float(getattr(row, figs_age_var))}"
        