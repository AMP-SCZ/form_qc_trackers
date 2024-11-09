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
        self.gf_score_check_vars = {'high':{'global_functioning_social_scale':{
        'chrgfs_gf_social_high':['chrgfs_gf_social_scale','chrgfs_gf_social_low']},
        'global_functioning_role_scale':{'chrgfr_gf_role_low chrgfr_gf_role_high': [
        'chrgfr_gf_role_scole','chrgfr_gf_role_low']}},
                            
        'low': {'global_functioning_social_scale':{
        'chrgfs_gf_social_low':['chrgfs_gf_social_scale','chrgfs_gf_social_high']},
        'global_functioning_role_scale':{'chrgfr_gf_role_low' : [
        'chrgfr_gf_role_scole','chrgfr_gf_role_high']}}}

        self.excluded_21_day_forms = ['cbc_with_differential',
        'gcp_cbc_with_differential','gcp_current_health_status',
        'psychs_p1p8_fu','psychs_p1p8_fu_hc'
        'chrpred_interview_date','sociodemographics']

        self.call_checks(row)
               
    def __call__(self):
        return self.final_output_list

    def call_checks(self, row):
        self.call_global_function_checks(row)
        self.call_oasis_checks(row)
        self.call_cssrs_checks(row)
        self.call_twenty_one_day_check(row)

    def call_global_function_checks(self,row):
        """
        Checks for contradictions in the
        global functioning forms
        """
        report_list = ['Main Report','Non Team Forms']
        for score_type, score_vals in self.gf_score_check_vars.items():
            if score_type == 'low':
                gt_bool = True
            else:
                gt_bool = False
            for form, vars  in score_vals.items():
                for low_score, other_scores in vars.items():
                    var_list = other_scores + [low_score]
                    self.functioning_score_check(row, [form],
                    var_list, {'reports':report_list},
                    bl_filtered_vars=[],filter_excl_vars=True,
                    compared_score_var = low_score, 
                    other_score_vars = other_scores, gt = gt_bool)

    def call_oasis_checks(self, row):
        forms = ['oasis']
        report_list = ['Main Report','Non Team Forms']
        for x in range(2,6):
            oasis_anx_var = f'chroasis_oasis_{x}'
            self.oasis_anxiety_check(row, forms, ['chroasis_oasis_1',oasis_anx_var],
            {'reports': report_list},bl_filtered_vars=[],
                filter_excl_vars=True, anx_var = oasis_anx_var)
        oasis_lifestyle_vars = {'chroasis_oasis_4': 1 , 'chroasis_oasis_5' : 0}
        for oasis_lifestyle_var, cutoff_val in oasis_lifestyle_vars.items():
            self.oasis_lifestyle_check(row, forms, ['chroasis_oasis_3',oasis_lifestyle_var],
            {'reports': report_list}, bl_filtered_vars=[],
            filter_excl_vars=True, lifestyle_var = oasis_lifestyle_var, cutoff = cutoff_val)

    def call_cssrs_checks(self,row):
        forms = ['cssrs_baseline']
        report_list = ['Main Report','Non Team Forms']
        cssrs_unequal_vals_dict = {'chrcssrsb_cssrs_actual':'chrcssrsb_sb1l',
        'chrcssrsb_cssrs_nssi':'chrcssrsb_sbnssibl','chrcssrsb_cssrs_yrs_ia':'chrcssrsb_sb3l',
        'chrcssrsb_cssrs_yrs_aa':'chrcssrsb_sb4l',
        'chrcssrsb_cssrs_yrs_pab':'chrcssrsb_sb5l','chrcssrsb_cssrs_yrs_sb':'chrcssrsb_sb6l'}
        cssr_greater_vals_dict = {'chrcssrsb_idintsvl':'chrcssrsb_css_sipmms',\
        'chrcssrsb_snmacatl':'chrcssrsb_cssrs_num_attempt',\
        'chrcssrsb_nminatl':'chrcssrsb_cssrs_yrs_nia','chrcssrsb_nmabatl':'chrcssrsb_cssrs_yrs_naa'}
        for x in range(1,6):
            cssrs_unequal_vals_dict[f'chrcssrsb_si{x}l'] =  f'chrcssrsb_css_sim{x}'

        for cssrs_dict, method in [
            (cssrs_unequal_vals_dict, self.cssrs_unequal_vals_check),
            (cssr_greater_vals_dict, self.cssrs_greater_vals_check)
        ]:
            for lifetime_cssrs_var, recent_cssrs_var in cssrs_dict.items():
                method(row, forms, [lifetime_cssrs_var, recent_cssrs_var],{'reports':report_list}, [], True, 
                    lifetime_var=lifetime_cssrs_var, recent_var=recent_cssrs_var)
            
    @FormCheck.standard_qc_check_filter 
    def cssrs_unequal_vals_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, lifetime_var = '', recent_var = ''
    ):
        lifetime_var_val = getattr(row,lifetime_var)
        recent_var_val = getattr(row,recent_var)
        for var_val in [lifetime_var_val, recent_var_val]:
            if (var_val in self.utils.missing_code_list 
            or not self.utils.can_be_float(var_val)):
                return
        if (lifetime_var_val in self.utils.all_dtype([2])
        and recent_var_val in self.utils.all_dtype([1])):
            return (f'Lifetime variable ({lifetime_var}) was answered as yes,'
                    f' but recent variable ({recent_var}) was answered as no.')
        
    @FormCheck.standard_qc_check_filter 
    def cssrs_greater_vals_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, lifetime_var = '', recent_var = ''
    ):
        lifetime_var_val = getattr(row,lifetime_var)
        recent_var_val = getattr(row,recent_var)
        for var_val in [lifetime_var_val, recent_var_val]:
            if (var_val in self.utils.missing_code_list 
            or not self.utils.can_be_float(var_val)):
                return
        if float(lifetime_var_val) < float(recent_var_val):
            return (f'Lifetime variable ({lifetime_var} = {lifetime_var_val}) has lower intensity,'
                    f' than recent variable ({recent_var} = {recent_var_val}).')

    @FormCheck.standard_qc_check_filter
    def oasis_anxiety_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, anx_var = ''
    ):
        if row.chroasis_oasis_1 in self.utils.all_dtype([0]):
            anx_var_val = getattr(row,anx_var)
            if (anx_var_val not in (self.utils.missing_code_list + [''])
            and self.utils.can_be_float(anx_var_val)):
                if float(anx_var_val) > 0:
                    return (f'Marked as having'
                            f' no anxiety in chroasis_oasis_1,'
                            f' but {anx_var} is equal to {anx_var_val}')
                
    @FormCheck.standard_qc_check_filter       
    def oasis_lifestyle_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, lifestyle_var = '', cutoff = 0
    ):
        compared_oasis_val = getattr(row,'chroasis_oasis_3')
        lifestyle_var_val = getattr(row,lifestyle_var)
        for var_val in [compared_oasis_val,lifestyle_var_val]:
            if (var_val in self.missing_code_list 
            or not self.utils.can_be_float(var_val)):
                return
        if float(compared_oasis_val) < 2 and float(lifestyle_var_val) > cutoff:
            return (f"chroasis_oasis_3 states that lifestyle"
            f" was not affected, but {lifestyle_var} is greater than {cutoff}")

    @FormCheck.standard_qc_check_filter
    def functioning_score_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, compared_score_var = '',
        other_score_vars = [], gt = True
    ):  
        compared_score_val = getattr(row, compared_score_var)
        if not self.utils.can_be_float(compared_score_val):
            return
        
        for score_var in other_score_vars:
            other_score_val = getattr(row,score_var)
            if self.utils.can_be_float(other_score_val):
                if gt == True and float(compared_score_val) > float(other_score_val):
                    return (f'{compared_score_var} ({compared_score_val}) is'
                    f' not the lowest score ({score_var} = {other_score_val})')
                elif gt == False and float(compared_score_val) < float(other_score_val):
                    return (f'{compared_score_var} ({compared_score_val}) is'
                    f' not the highest score ({score_var} = {other_score_val})')
        return 
    
    def call_twenty_one_day_check(self, row):   
        if self.timepoint != 'baseline':
            return 
        cohort = self.subject_info[row.subjectid]['cohort']
        if cohort.lower() not in ["hc", "chr"]:
            return
        curr_tp_forms = self.forms_per_tp[cohort][self.timepoint]
        missing_spec_vars = {'psychs_p1p8_fu':'chrpsychs_fu_missing_spec_fu',
                             'psychs_p1p8_fu_hc': 'hcpsychs_fu_missing_spec_fu'}
        scr_int_date = str(self.subject_info[row.subjectid]['psychs_screen_date'])
        if (not self.utils.check_if_val_date_format(scr_int_date)
        or scr_int_date in self.utils.missing_code_list):
            return
        for form in curr_tp_forms:
            if form in ['psychs_p1p8_fu_hc','psychs_p1p8_fu']:
                curr_psychs_form = form
        missing_spec_var = missing_spec_vars[curr_psychs_form]
        self.check_if_over_21_days(row,missing_spec_var, scr_int_date,curr_tp_forms)

    def check_if_over_21_days(self,
        row,missing_spec_var, scr_int_date,
        curr_tp_forms,curr_psychs_form
    ):
        reports = ['Main Report', 'Non Team Forms']
        if (hasattr(row, missing_spec_var)
        and str(getattr(row, missing_spec_var)) == 'M6'):
            for form in curr_tp_forms:
                if form in self.excluded_21_day_forms:
                    continue
                if self.check_if_missing(row,form) == True:
                    continue
                int_date_var = self.important_form_vars[form]['interview_date_var']
                if hasattr(row, int_date_var):
                    int_date = str(getattr(row,int_date_var))
                    if (not self.utils.check_if_val_date_format(int_date)
                    or int_date in self.utils.missing_code_list):
                        continue
                    days_between = self.utils.find_days_between(int_date,scr_int_date)
                    if days_between > 21:
                        error_message = ("M6 clicked on baseline psychs,"
                        " but there are more than 21 days between"
                        f" {form} (date = {int_date}) and Screening Psychs (date = {scr_int_date})")
                        output_changes = {'reports':reports}
                        error_output = self.create_row_output(
                        row,[curr_psychs_form],[missing_spec_var],
                        error_message, output_changes)
                        self.final_output_list.append(error_output)


                    

                
