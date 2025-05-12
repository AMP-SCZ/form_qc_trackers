import pandas as pd
import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)
from utils.utils import Utils
from qc_forms.form_check import FormCheck
from datetime import datetime
    

class ScidChecks(FormCheck):
    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network, form_check_info)
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

        self.scid_diagnosis_dict = self.utils.load_dependency_json(
        'scid_diagnosis_vars.json')

        self.call_checks(row)
    

    def call_scid_diagnosis_check(self,row):
        for checked_variable,conditions in self.scid_diagnosis_dict.items(): 
            self.scid_diagnosis_check(
            row,checked_variable, conditions['diagnosis_variables'],
            conditions['disorder'], True, conditions['extra_conditionals'])
            self.scid_diagnosis_check(row,checked_variable,conditions[
            'diagnosis_variables'], conditions['disorder'],
            False, conditions['extra_conditionals'])

    def scid_diagnosis_check(
        self, curr_row, variable, conditional_variables,
        disorder, fulfilled, extra_conditionals
    ):
        form = 'scid5_psychosis_mood_substance_abuse'
        affected_vars = conditional_variables
        changed_output = {'reports': ['Main Report','Scid Report']}
        affected_vars.append(variable)
        if fulfilled == True:
            for condition in conditional_variables:
                if (hasattr(curr_row,condition) and 
                getattr(curr_row, condition) not in [3,3.0,'3','3.0']):
                    return 
            if extra_conditionals != '':
                for conditional in extra_conditionals:
                    if not eval(conditional):
                        return 
            self.scid_diagnostic_criteria_check(curr_row, [form],
            affected_vars,changed_output, bl_filtered_vars=[],filter_excl_vars=False, 
            diagnostic_variable=variable, disorder=disorder, fulfilled=fulfilled)                    
        else:
            for condition in conditional_variables:
                if (hasattr(curr_row, condition) and 
                getattr(curr_row, condition) not in [3,3.0,'3','3.0']):
                    self.scid_diagnostic_criteria_check(curr_row, [form],
                    affected_vars, changed_output,bl_filtered_vars=[],filter_excl_vars=False, 
                    diagnostic_variable=variable, disorder=disorder, fulfilled=fulfilled)      
            if extra_conditionals != '':
                for conditional in extra_conditionals:
                    if not eval(conditional):
                        self.scid_diagnostic_criteria_check(curr_row, [form],
                        affected_vars,changed_output,bl_filtered_vars=[],filter_excl_vars=False, 
                        diagnostic_variable=variable, disorder=disorder, fulfilled=fulfilled) 

    @FormCheck.standard_qc_check_filter 
    def bprs_val_comparisons(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False, var_comps = {}
    ):
        all_vars = list(var_comps.keys())
        if (all(self.utils.can_be_float(getattr(row,var)) for var in all_vars)):
            if all(float(getattr(row,var)) in var_comps[var] for var in all_vars):
                return (f'{all_vars[0]} is'
                f' {getattr(row, all_vars[0])}, but'
                f' {all_vars[1]} is {getattr(row,all_vars[1])}')
            
    @FormCheck.standard_qc_check_filter 
    def chrchs_weight_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False
    ):
        weight = getattr(row, 'chrchs_weightkg')
        if self.utils.can_be_float(weight) and float(weight) <0:
            return f'chrchs_weightkg is {weight}'

    @FormCheck.standard_qc_check_filter 
    def scid_diagnostic_criteria_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False, diagnostic_variable = '',
        disorder = '', fulfilled = True
    ):
        if fulfilled and getattr(row, diagnostic_variable) not in self.utils.all_dtype([3]):
            return f'{disorder} criteria are fulfilled, but it is not indicated.'
        elif not fulfilled and getattr(row, diagnostic_variable) in self.utils.all_dtype([3]):
            return f'{disorder} criteria are NOT fulfilled, but it is indicated.'
    
    @FormCheck.standard_qc_check_filter 
    def depressive_manic_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False
    ):
        var_list = ['chrscid_a54','chrscid_a92','chrscid_a132']
        if (all(getattr(row,var) in self.utils.all_dtype([1]) for var in var_list)
        and all(getattr(row,var) in self.utils.all_dtype([3])
        for var in ['chrscid_d26','chrscid_d27'])):
            if getattr(row,'chrscid_d28') not in self.utils.all_dtype([3]):
                return ('subject does not fulfill any'
                ' manic or hypomanic episode but depression is excluded due to this symptom.')
        
    @FormCheck.standard_qc_check_filter 
    def major_depressive_episode_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False
    ):
        error_message = (f"Contradiction between variables"
                        " related to major depressive episode"
                        f" (chrscid_a25 = {row.chrscid_a25},"
                        f" chrscid_a51 = {row.chrscid_a51}, chrscid_d26 = {row.chrscid_d26})")
        
        if (row.chrscid_a25 not in self.utils.all_dtype([3])
        and row.chrscid_a51 not in self.utils.all_dtype([3])):
            if row.chrscid_d26 in self.utils.all_dtype([3]):
                return error_message
        elif (row.chrscid_a25 in self.utils.all_dtype([3])
        or row.chrscid_a51 in self.utils.all_dtype([3])):
            if (not any(getattr(row,var) in self.utils.all_dtype([3])
            for var in ['chrscid_d3','chrscid_d9','chrscid_d11','chrscid_d23'])):
                if row.chrscid_d26 not in self.utils.all_dtype([3]):
                    return error_message
            
    @FormCheck.standard_qc_check_filter 
    def manic_episode_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False
    ):
        error_message = (f"Contradiction between variables"
                        " related to manic episode"
                        f" (chrscid_a108 = {row.chrscid_a108},"
                        f" chrscid_a70 = {row.chrscid_a70}, chrscid_d2 = {row.chrscid_d2})")
        
        if (row.chrscid_a70 not in self.utils.all_dtype([3])
        and row.chrscid_a108 not in self.utils.all_dtype([3])):
            if row.chrscid_d2 in self.utils.all_dtype([3]):
                return error_message
        elif (row.chrscid_a70 in self.utils.all_dtype([3])
        or row.chrscid_a108 in self.utils.all_dtype([3])):
            if row.chrscid_d2 not in self.utils.all_dtype([3]):
                return error_message

    @FormCheck.standard_qc_check_filter 
    def hypomanic_episode_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False
    ):
        error_message = (f"Contradiction between variables"
                        " related to hypomanic episode"
                        f" (chrscid_a91 = {row.chrscid_a91},"
                        f" chrscid_a129 = {row.chrscid_a129}, chrscid_d5 = {row.chrscid_d5})")
        
        if (row.chrscid_a91 not in self.utils.all_dtype([3])
        and row.chrscid_a129 not in self.utils.all_dtype([3])):
            if row.chrscid_d5 in self.utils.all_dtype([3]):
                return error_message
        elif (row.chrscid_a91 in self.utils.all_dtype([3])
        or row.chrscid_a129 in self.utils.all_dtype([3])):
            if row.chrscid_d5 not in self.utils.all_dtype([3]):
                return error_message
            
    @FormCheck.standard_qc_check_filter 
    def mania_not_fulfilled_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False
    ):  
        for var in ['chrscid_a70','chrscid_a91','chrscid_a108',
        'chrscid_a129','chrscid_a138']:
            if getattr(row,var) in self.utils.all_dtype([3]):
                return
            
        if row.chrscid_d28 not in self.utils.all_dtype([3]):
            return 'chrscid_d28 has to be 3 since no manic or hypomanic episode was fulfilled.'
    
    @FormCheck.standard_qc_check_filter 
    def depressed_mood_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False
    ):
        if row.chrscid_a27 in [3,3.0,'3','3.0'] and row.chrscid_a28 in [3,3.0,'3','3.0'] and\
        (row.chrscid_a48_1 not in [2,2.0,'2','2.0']): 
            return 'Fulfills both main criteria but was counted incorrectly, a27, a28, a48_1'
        elif ((row.chrscid_a27 in [3,3.0,'3','3.0'] and (row.chrscid_a28 in
        [2,2.0,'2','2.0'] or row.chrscid_a28 in [1,1.0,'1','1.0'])) or
        ((row.chrscid_a28 in [3,3.0,'3','3.0'] and (row.chrscid_a27
        in [2,2.0,'2','2.0'] or row.chrscid_a27 in [1,1.0,'1','1.0']))) and
        (row.chrscid_a48_1 not in [1,1.0,'1','1.0'])):
            return 'Fulfills main criteria but further value was wrong, a27, a28, a48_1'
        
    @FormCheck.standard_qc_check_filter 
    def major_depressive_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False
    ):
        if (self.utils.can_be_float(row.chrscid_a26_53) and row.chrscid_a26_53
        not in self.utils.missing_code_list and 
        float(row.chrscid_a26_53) < 1 and (row.chrscid_a25
        in [3,3.0,'3','3.0'] or row.chrscid_a51 in [3,3.0,'3','3.0'])):
            return ('has no indication of total mde episodes'
            ' fulfilled in life even though fulfills current major depression. a26_53, a51')
        elif (self.utils.can_be_float(row.chrscid_a26_53) and row.chrscid_a26_53
        not in self.utils.missing_code_list and float(row.chrscid_a26_53) > 0
        and (row.chrscid_a25 not in [3,3.0,'3','3.0']
        and row.chrscid_a51 not in [3,3.0,'3','3.0'])):
            return ('fulfills more manic episodes than 0 but there'
            ' is no indication of past or current depressive episode. a26_53, a25')
        
    @FormCheck.standard_qc_check_filter 
    def curr_major_depressive_threshold_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False
    ):
        if (row.chrscid_a1 not in self.utils.all_dtype([3]) 
        and row.chrscid_a2 not in self.utils.all_dtype([3])):
            if row.chrscid_a25 in self.utils.all_dtype([3]):
                return ('main criteria for major depressive episode'
                ' both below threshold but major depressive episode fulfilled.')
            
    @FormCheck.standard_qc_check_filter 
    def past_major_depressive_threshold_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False
    ):  
        if (row.chrscid_a27 not in self.utils.all_dtype([3]) 
        and row.chrscid_a28 not in self.utils.all_dtype([3])):
            if row.chrscid_a51 in self.utils.all_dtype([3]):
                return ('main criteria for past major depressive episode'
                ' both below threshold but past major depressive episode fulfilled.')

    @FormCheck.standard_qc_check_filter
    def mood_symptoms_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False
    ):  
        if ((any(getattr(row,var) in self.utils.all_dtype([3]) for var in ['chrscid_a25',
        'chrscid_a51','chrscid_a70','chrscid_a91','chrscid_a108','chrscid_a129',
        'chrscid_a138','chrscid_a153','chrscid_a170'])) 
        and row.chrscid_c26 not in self.utils.all_dtype([3])):
            if row.chrscid_d1 not in self.utils.all_dtype([1]):
                return ('chrscid_d1 needs to be checked'
                ' if there are any clinically significant mood symptoms.')
        
    @FormCheck.standard_qc_check_filter  
    def mood_episode_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False
    ):
        if (row.chrscid_a22 not in (self.utils.missing_code_list+['']) and
        row.chrscid_a22_1 not in (self.utils.missing_code_list+[''])):
            if (self.utils.can_be_float(row.chrscid_a22) and float(row.chrscid_a22) > 4 and
            self.utils.can_be_float(row.chrscid_a22_1) and float(row.chrscid_a22_1) > 0  and 
            row.chrscid_a25 in (self.utils.missing_code_list+[''])):
                return ('A. MOOD EPISODES: subject fulfills more than 4'
                ' criteria of depression but further questions are not asked: start checking a22, a22_1, a25')
