import datetime
import pandas as pd

import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck

class CognitionChecks(FormCheck):
    """
    QC Checks related to cognitive forms
    """
    
    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network, form_check_info)
        self.scid_diagnosis_dict = self.utils.load_dependency_json(
        'scid_diagnosis_vars.json')
        self.assessment_translations = {1.0 : 'wasi', 2.0 : 'wais'}
        self.timepoint = timepoint
        self.call_checks(row)
        
    def call_checks(self, row):
        self.call_cognition_checks(row)

    def __call__(self):
        return self.final_output_list

    def call_cognition_checks(self,row):
        reports = {'reports' : ['Main Report','Cognition Report']}
        """self.temp_cognition_fix(row, 
        ['iq_assessment_wasiii_wiscv_waisiv'], 
        ['chriq_fsiq'], reports)"""

        if self.timepoint in ["baseline","month24"]:
            if ('age' in self.subject_info[row.subjectid].keys() and
            'demographics_date' in self.subject_info[row.subjectid].keys()):
                self.age = self.subject_info[row.subjectid]['age']
                self.demo_date = self.subject_info[row.subjectid]['demographics_date']
                if all(self.utils.check_if_val_date_format(str(date_val)) for
                date_val in [self.demo_date, row.chriq_interview_date]):
                    iq_age_diff = self.utils.find_days_between(self.demo_date, row.chriq_interview_date)
                    if all(self.utils.can_be_float(age_val) for age_val in [iq_age_diff,self.age]):
                        self.iq_age = float(self.age) + (float(iq_age_diff) / 365)
                        if (hasattr(row, 'chriq_assessment') and 
                        row.chriq_assessment not in (self.utils.missing_code_list + ['']) and
                        self.utils.can_be_float(row.chriq_assessment) and
                        float(row.chriq_assessment) in [1.0,2.0]):
                            assessment = self.assessment_translations[float(row.chriq_assessment)]
    
    def fsiq_conversion_check(self, row, iq_type):
        pass

    @FormCheck.standard_qc_check_filter 
    def temp_cognition_fix(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False
    ):
        prescient_output_path = '/PHShome/ob001/anaconda3/new_forms_qc/QC/site_outputs/PRESCIENT/combined/PRESCIENT_Output.xlsx'

        for cog_row in self.cog_df.itertuples():
            if (getattr(cog_row, 'Subject') == row.subjectid and
            getattr(cog_row, 'Timepoint').replace('screen','screening').replace('baseln','baseline') == self.timepoint):
                if 'IQ Assessment Wasii' in getattr(cog_row, 'General_Flag') and 'Conversion' in getattr(cog_row,'Specific_Flags'):
                    flags = getattr(cog_row,'Specific_Flags').split('|')
                    print(flags)
                    for flag in flags:
                        print(flag)
                        if 'Conversion' in flag:
                            print(flag)
                            return flag

