
import pandas as pd

import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck
from datetime import datetime
class FluidChecks(FormCheck):
    
    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network,form_check_info)
        self.test_val = 0
        self.call_checks(row)
               
    def __call__(self):

        return self.final_output_list

    def call_checks(self, row):
        self.call_blood_checks(row)
        
    def call_blood_checks(self,row):
        form = 'blood_sample_preanalytic_quality_assurance'
        blood_reports =  ['Main Report','Blood Report','Fluids Report']
        all_vol_vars = self.grouped_vars['blood_vars']['volume_variables']
        for vol_var in all_vol_vars:
            self.blood_vol_check(row,[form],[vol_var],{"reports":blood_reports})
        proc_time_vars = ['chrblood_wholeblood_freeze','chrblood_serum_freeze',
        'chrblood_plasma_freeze','chrblood_buffy_freeze']
        for proc_time_var in proc_time_vars:
            self.check_blood_freeze(row,[form],[proc_time_var],{"reports":blood_reports})
        self.cbc_differential_check(row)

    @FormCheck.standard_qc_check_filter    
    def blood_vol_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):  
        vol_var = all_vars[0]
        if hasattr(row ,vol_var):
            vol = getattr(row,vol_var)
            if vol not in (self.missing_code_list+['']) and\
            self.utils.can_be_float(vol) and float(vol) <= 0:
                print(f'Volume ({vol_var} = {vol}) is less than or equal to 0')
                return  f'Volume ({vol_var} = {vol}) is less than or equal to 0'
    
    @FormCheck.standard_qc_check_filter 
    def check_blood_freeze(self,row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):    
        proc_time_var = all_vars[0]
        proc_time = getattr(row, proc_time_var)
        if (proc_time not in (self.missing_code_list+[''])
        and self.utils.can_be_float(proc_time) and float(proc_time) > 480):
            return  f'Processing time ({proc_time_var} = {proc_time}) is greater than 480'

    
    def cbc_differential_check(self,row):
        """Checks for additional errors in
        CBC with differntial form"""
        forms = ['cbc_with_differential']
        blood_reports =  ['Main Report','Blood Report','Fluids Report']

        self.white_bc_check(row,forms,['chrcbc_wbcinrange','chrcbc_wbcsum','chrcbc_wbc'],{"reports":blood_reports})
        self.cbc_compl_check(row,['cbc_with_differential'],
        ['cbc_with_differential_complete','chrblood_cbc'],{"reports":blood_reports,"affected_forms":
        ['cbc_with_differential','blood_sample_preanalytic_quality_assurance']})


    @FormCheck.standard_qc_check_filter 
    def white_bc_check(self,row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ): 
        if not any(x in (self.missing_code_list+[''])\
        for x in [row.chrcbc_wbcsum,row.chrcbc_wbc]):
            if (self.utils.can_be_float(row.chrcbc_wbcsum)
            and self.utils.can_be_float(row.chrcbc_wbc)):
                wbc_sum = float(self.row.chrcbc_wbcsum)
                wbc = float(self.row.chrcbc_wbc)
                if (wbc_sum < (wbc-(0.1*wbc))) or wbc_sum > (wbc+(0.1*wbc)):
                    return (f'Sum of absolute neutrophils, lymphocytes,'
                        f' monocytes, basophils, and eosinophils'
                        f' ({row.chrcbc_wbcsum}) is not within 10% of '
                        f' WBC count ({row.chrcbc_wbc}).')

    @FormCheck.standard_qc_check_filter
    def edta_(self,row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ): 
        if getattr(row, 'cbc_with_differential_complete') in self.utils.all_dtype([2]):
            if (row.chrblood_cbc in self.utils.all_dtype([1]) and
            row.chrblood_interview_date not in (self.missing_code_list+[''])):
                time_since_blood = self.utils.find_days_between(str(self.row.chrblood_interview_date),
                str(datetime.datetime.today()))
                if time_since_blood > 5:
                    return ('Blood form indicates EDTA tube was sent to lab for CBC'
                    f', but CBC form has not been completed.')          
    
    @FormCheck.standard_qc_check_filter
    def cbc_compl_check(self,row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ): 
        if getattr(row, 'cbc_with_differential_complete') in self.utils.all_dtype([2]):
            if (row.chrblood_cbc in self.utils.all_dtype([1]) and
            row.chrblood_interview_date not in (self.missing_code_list+[''])):
                time_since_blood = self.utils.find_days_between(str(self.row.chrblood_interview_date),
                str(datetime.datetime.today()))
                if time_since_blood > 5:
                    return ('Blood form indicates EDTA tube was sent to lab for CBC'
                    f', but CBC form has not been completed.')
                
        elif getattr(row, 'cbc_with_differential_complete') not in self.utils.all_dtype([2]):
            if self.check_if_missing(row,'cbc_with_differential_complete') != True:
                if row.chrblood_cbc not in self.utils.all_dtype([1]):
                    return ('Blood form indicates EDTA tube was not sent to lab for CBC'
                    f', but CBC form has been completed and not marked as missing.')
