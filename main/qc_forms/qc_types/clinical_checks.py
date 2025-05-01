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
               
    def __call__(self):
        return self.final_output_list

    def call_checks(self, row):
        self.call_global_function_checks(row)
        self.call_oasis_checks(row)
        self.call_cssrs_checks(row)
        self.call_twenty_one_day_check(row)
        self.call_tbi_checks(row)
        self.call_scid_checks(row)
        self.call_bprs_checks(row)
        self.call_chrchs_checks(row)
        self.call_conversion_check(row)
        self.check_onset_date(row, 
        ['current_pharmaceutical_treatment_floating_med_125'],
        ['chrpharm_date_mod'], {'reports': ['Main Report']})

    def call_conversion_check(self,row):
        gt_var_val_pairs = {'chrbprs_bprs_somc': 5,
        'chrbprs_bprs_guil':5,'chrbprs_bprs_gran':5,
        'chrbprs_bprs_susp':5,'chrbprs_bprs_hall':5,
        'chrbprs_bprs_unus':5,'chrbprs_bprs_bizb':5,
        'chrbprs_bprs_conc':5}
        
        eq_var_val_pairs = {'chrpsychs_fu_1c0':6,
        'chrpsychs_fu_1d0':6,'chrpsychs_fu_2c0':6,
        'chrpsychs_fu_2d0':6,'chrpsychs_fu_3c0':6,
        'chrpsychs_fu_3d0':6,'chrpsychs_fu_4c0':6,
        'chrpsychs_fu_4d0':6,'chrpsychs_fu_5c0':6,
        'chrpsychs_fu_5d0':6,'chrpsychs_fu_6c0':6,
        'chrpsychs_fu_6d0':6,'chrpsychs_fu_7c0':6,
        'chrpsychs_fu_7d0':6,'chrpsychs_fu_8c0':6,
        'chrpsychs_fu_8d0':6,'chrpsychs_fu_9c0':6,
        'chrpsychs_fu_9d0':6,'chrpsychs_fu_10c0':6,
        'chrpsychs_fu_10d0':6,'chrpsychs_fu_11c0':6,
        'chrpsychs_fu_11d0':6,'chrpsychs_fu_12c0':6,
        'chrpsychs_fu_12d0':6,'chrpsychs_fu_13c0':6,
        'chrpsychs_fu_13d0':6,'chrpsychs_fu_14c0':6,
        'chrpsychs_fu_14d0':6,'chrpsychs_fu_15c0':6,
        'chrpsychs_fu_15d0':6,'chrscid_c10':3,
        'chrscid_c26':3,'chrscid_c14':3,'chrscid_c37':3,
        'chrscid_c44':3,'chrscid_d47_d52':1,'chrscid_d63':1,
        'chrscid_c71':3,'chrscid_c78':3,'chrscid_c11':1,
        'chrscid_c21':1,'chrscid_c47':1,'chrscid_c28':1,
        'chrscid_c50':3,'chrscid_c51':8}

        for var, threshold in gt_var_val_pairs.items():
            form = self.grouped_vars['var_forms'][var]
            if not hasattr(row,var):
                continue
            var_val = getattr(row, var) 
            if self.utils.can_be_float(var_val) and float(var_val) > threshold:
                self.conversion_criteria_check(row, [form],
                [var],{'reports': ['Main Report']})

        for var, threshold in eq_var_val_pairs.items():
            form = self.grouped_vars['var_forms'][var]
            if not hasattr(row,var):
                continue
            var_val = getattr(row, var) 
            if self.utils.can_be_float(var_val) and float(var_val) == threshold:
                self.conversion_criteria_check(row, [form],
                [var],{'reports': ['Main Report']})


    def call_chrchs_checks(self, row):
        changed_output = {'reports': ['Main Report']}
        form = 'bprs'
        vars = ['chrchs_weightkg']
        self.chrchs_weight_check(row, [form],
        vars, changed_output)
        
    def call_bprs_checks(self,row):
        changed_output = {'reports': ['Main Report']}
        form = 'bprs'
        initial_vars = ['chrbprs_bprs_somc',
        'chrbprs_bprs_guil','chrbprs_bprs_gran']
        var_val_pairs = []
        for var in initial_vars:
            val_checks = {}
            val_checks[var] = [6,7]
            val_checks['chrbprs_bprs_unus'] = [1,2,3]
            var_val_pairs.append(val_checks)

        var_val_pairs.append({'chrbprs_bprs_susp':[4,5,6,7],
                              'chrbprs_bprs_unus':[1]})
        
        for pair in var_val_pairs:
            vars = list(pair.keys())
            self.bprs_val_comparisons(row, [form],
            all_vars =vars,changed_output_vals = changed_output,
            bl_filtered_vars = [], filter_excl_vars = False,
            var_comps = pair)

    def call_scid_checks(self, row):
        changed_output = {'reports': ['Main Report','Scid Report']}
        form = 'scid5_psychosis_mood_substance_abuse'
        self.call_scid_diagnosis_check(row)
        self.depressed_mood_check(row, [form], 
        ['chrscid_a27','chrscid_a28','chrscid_a48_1'],
        changed_output)   
        self.major_depressive_check(row, [form], 
        ['chrscid_a26_53','chrscid_a25','chrscid_a51'],
        changed_output)  
        self.mood_episode_check(row, [form], 
        ['chrscid_a25','chrscid_a22','chrscid_a22_1'],
        changed_output)  
        self.mood_symptoms_check(row, [form], 
        ['chrscid_d1','chrscid_a25','chrscid_a51','chrscid_a70',
        'chrscid_a91','chrscid_a108','chrscid_a129',
        'chrscid_a138','chrscid_a153','chrscid_a170'],
        changed_output)   
        self.curr_major_depressive_threshold_check(row, [form], 
        ['chrscid_a1','chrscid_a2','chrscid_a25'],
        changed_output)  
        self.past_major_depressive_threshold_check(row, [form], 
        ['chrscid_a27','chrscid_a28','chrscid_a51'],
        changed_output)       
        self.mania_not_fulfilled_check(row, [form], 
        ['chrscid_d28','chrscid_a70','chrscid_a91','chrscid_a108',
        'chrscid_a129','chrscid_a138'],
        changed_output, bl_filtered_vars=['chrscid_d28'],filter_excl_vars=False)  
        self.major_depressive_episode_check(row, [form], 
        ['chrscid_d26','chrscid_a51','chrscid_a25',
        'chrscid_d3','chrscid_d9','chrscid_d11','chrscid_d23'],
        changed_output, bl_filtered_vars=[],filter_excl_vars=False)  
        self.manic_episode_check(row, [form], 
        ['chrscid_a108','chrscid_a70','chrscid_d2'],
        changed_output, bl_filtered_vars=[],filter_excl_vars=False)

        self.hypomanic_episode_check(row, [form], 
        ['chrscid_d5','chrscid_a91','chrscid_a129'],
        changed_output, bl_filtered_vars=[],filter_excl_vars=False)  

        self.depressive_manic_check(row, [form], 
        ['chrscid_d28','chrscid_a54','chrscid_a92','chrscid_a132'],
        changed_output, bl_filtered_vars=[],filter_excl_vars=False)


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
                    var_list, {'reports' : report_list},
                    bl_filtered_vars=[], filter_excl_vars=True,
                    compared_score_var = low_score, 
                    other_score_vars = other_scores, gt = gt_bool)

    def call_oasis_checks(self, row):
        forms = ['oasis']
        report_list = ['Main Report', 'Non Team Forms']
        for x in range(2, 6):
            oasis_anx_var = f'chroasis_oasis_{x}'
            self.oasis_anxiety_check(row, forms, ['chroasis_oasis_1',oasis_anx_var],
            {'reports': report_list}, bl_filtered_vars=[],
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
        cssr_greater_vals_dict = {'chrcssrsb_idintsvl':'chrcssrsb_css_sipmms',
        'chrcssrsb_snmacatl':'chrcssrsb_cssrs_num_attempt',
        'chrcssrsb_nminatl':'chrcssrsb_cssrs_yrs_nia',
        'chrcssrsb_nmabatl':'chrcssrsb_cssrs_yrs_naa'}

        for x in range(1, 6):
            cssrs_unequal_vals_dict[f'chrcssrsb_si{x}l'] =  f'chrcssrsb_css_sim{x}'

        for cssrs_dict, method in [
            (cssrs_unequal_vals_dict, self.cssrs_unequal_vals_check),
            (cssr_greater_vals_dict, self.cssrs_greater_vals_check)
        ]:
            for lifetime_cssrs_var, recent_cssrs_var in cssrs_dict.items():
                method(row, forms, [lifetime_cssrs_var, recent_cssrs_var],
                {'reports':report_list}, [], True, 
                    lifetime_var=lifetime_cssrs_var, recent_var=recent_cssrs_var)
    
    def call_tbi_checks(self, row):
        forms = ['traumatic_brain_injury_screen']
        reports = ['Main Report','Non Team Forms']
        self.tbi_inj_mismatch_check(row, forms, ['chrtbi_parent_headinjury',
        'chrtbi_subject_head_injury','chrtbi_sourceinfo'],
        {'reports': ['Main Report','Non Team Forms']})

        self.tbi_info_source_check(row, forms, ['chrtbi_parent_headinjury',
        'chrtbi_subject_head_injury','chrtbi_sourceinfo'],
        {'reports': ['Secondary Report']})

    @FormCheck.standard_qc_check_filter 
    def tbi_inj_mismatch_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        sub_inj = row.chrtbi_subject_head_injury
        par_inj = row.chrtbi_parent_headinjury
        if row.chrtbi_sourceinfo in self.utils.all_dtype([3]):
            if (self.utils.can_be_float(sub_inj) and self.utils.can_be_float(par_inj)
            and not any(val in self.utils.missing_code_list for val in [sub_inj, par_inj])):
                if float(sub_inj) != float(par_inj):
                    return ("Subject and parent answered differently to whether"
                    " or not the subject has ever had a head injury.")
                
    @FormCheck.standard_qc_check_filter         
    def tbi_info_source_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        if row.chrtbi_sourceinfo in self.utils.all_dtype([1,2]):
            if (row.chrtbi_subject_head_injury in self.utils.all_dtype([1,0])
            and row.chrtbi_parent_headinjury in self.utils.all_dtype([1,0])):
                return ("Subject and parent not both selected as source of information,"
                " but answers appear to be provided by both the subject and parent.")

    @FormCheck.standard_qc_check_filter 
    def cssrs_unequal_vals_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, lifetime_var = '', recent_var = ''
    ):
        lifetime_var_val = getattr(row, lifetime_var)
        recent_var_val = getattr(row, recent_var)
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
        lifetime_var_val = getattr(row, lifetime_var)
        recent_var_val = getattr(row, recent_var)
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
        for var_val in [compared_oasis_val, lifestyle_var_val]:
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
        self.check_if_over_21_days(row,missing_spec_var, scr_int_date,
        curr_tp_forms,curr_psychs_form)

    @FormCheck.standard_qc_check_filter
    def check_onset_date(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True):
        chrpharm_med2_onset
        date_list = []
        for x in range(0,10):
            date_list.append(f'chrpharm_med{x}_onset')

        for date_var in date_list:
            if hasattr(row, date_var):
                date_val = str(getattr(row,date_var))
                data_entry_val = str(getattr(row,chrpharm_date_mod))

                if (self.utils.check_if_val_date_format(
                self, date_val, date_format="%Y-%m-%d")
                and self.utils.check_if_val_date_format(
                self, data_entry_val, date_format="%Y-%m-%d")):
                    days_btwn = self.utils.find_days_between(
                    date_val,data_entry_val)
                    
        if days_btwn > 10:
            return f'There are {days_btwn} days between the most recent medication date and medication mod date'

    def check_if_over_21_days(self,
        row, missing_spec_var, scr_int_date,
        curr_tp_forms, curr_psychs_form
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
                    days_between = self.utils.find_days_between(int_date, scr_int_date)
                    if days_between > 21:
                        error_message = ("M6 clicked on baseline psychs,"
                        " but there are more than 21 days between"
                        f" {form} (date = {int_date}) and Screening Psychs (date = {scr_int_date})")
                        output_changes = {'reports' : reports}
                        error_output = self.create_row_output(
                        row, [curr_psychs_form], [missing_spec_var],
                        error_message, output_changes)
                        self.final_output_list.append(error_output)

    @FormCheck.standard_qc_check_filter   
    def conversion_criteria_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        if row.visit_status_string != 'converted':
            return f'{all_vars[0]} is {getattr(row, all_vars[0])}, but participant is not marked as converted.'
