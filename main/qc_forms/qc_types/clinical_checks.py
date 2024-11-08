
import pandas as pd

import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck
from datetime import datetime
class ClinicalChecks(FormCheck):
    
    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network,form_check_info)
        self.test_val = 0
        self.call_checks(row)
               
    def __call__(self):
        return self.final_output_list

    def call_checks(self, row):
        self.call_clinical_checks(row)

    def call_clinical_checks(self,row):
        pass
        
    def call_global_function_checks(self):
        """Checks for contradictions in the
        global functioning forma"""

        try:
            if self.variable == 'chrgfs_gf_social_low':
                if self.row.chrgfs_gf_social_low not in self.missing_code_list\
                and (((float(self.row.chrgfs_gf_social_low) \
                > float(self.row.chrgfs_gf_social_scale) and self.row.chrgfs_gf_social_scale\
                not in self.missing_code_list) or\
                ((float(self.row.chrgfs_gf_social_low) > float(self.row.chrgfs_gf_social_high)\
                and self.row.chrgfs_gf_social_high not in self.missing_code_list)))):
                    self.compile_errors.append_error\
                    (self.row,(f"Social Scale low score ({self.row.chrgfs_gf_social_low}) is not the lowest score"\
                     f"(current score = {self.row.chrgfs_gf_social_scale}, high score = {self.row.chrgfs_gf_social_high})."),\
                    self.variable,self.form,['Main Report','Non Team Forms'])
            elif self.variable == 'chrgfs_gf_social_scale':
                if self.row.chrgfs_gf_social_scale not in self.missing_code_list\
                and (float(self.row.chrgfs_gf_social_scale)\
                > float(self.row.chrgfs_gf_social_high) and\
                self.row.chrgfs_gf_social_high not in self.missing_code_list):
                    self.compile_errors.append_error\
                    (self.row,(f"Social Scale current score ({self.row.chrgfs_gf_social_scale}) is greater",\
                    f"than the high score ({self.row.chrgfs_gf_social_high})."),\
                    self.variable,self.form,['Main Report','Non Team Forms'])
            elif self.variable == 'chrgfs_gf_role_low':
                if self.row.chrgfs_gf_role_low not in self.missing_code_list\
                and ((float(self.row.chrgfs_gf_role_low)\
                > float(self.row.chrgfs_gf_role_scale) and\
                self.row.chrgfs_gf_role_scale not in self.missing_code_list)\
                or (float(self.row.chrgfs_gf_role_low) > float(self.row.chrgfs_gf_role_high)\
                and self.row.chrgfs_gf_role_high not in self.missing_code_list)):
                    self.compile_errors.append_error\
                    (self.row,(f"Role Scale low score ({self.row.chrgfs_gf_role_low}) is not the lowest score",\
                    f"(current score = {self.row.chrgfs_gf_role_scale}, high score = {self.row.chrgfs_gf_role_high})."),\
                    self.variable,self.form,['Main Report','Non Team Forms'])
            elif self.variable == 'chrgfs_gf_role_scale':
                if self.row.chrgfs_gf_role_scale not in self.missing_code_list\
                and float(self.row.chrgfs_gf_role_scale) > float(self.row.chrgfs_gf_role_high)\
                and self.row.chrgfs_gf_role_high not in self.missing_code_list:
                    self.compile_errors.append_error\
                    (self.row,(f"Role Scale current score ({self.row.chrgfs_gf_role_scale}) is greater",
                    f"than the high score ({self.row.chrgfs_gf_role_high})."),\
                    self.variable,self.form,['Main Report','Non Team Forms'])
        except Exception as e:
            print(e)

    def lowest_score_checks(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, low_score = '', other_scores = []
    ):
        
        