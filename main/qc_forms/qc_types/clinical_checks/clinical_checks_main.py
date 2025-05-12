import pandas as pd
import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-4])
sys.path.insert(1, parent_dir)
from utils.utils import Utils
from qc_forms.form_check import FormCheck
from qc_forms.qc_types.clinical_checks.scid_checks import ScidChecks
from datetime import datetime

class ClinicalChecksMain(FormCheck):    
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
        'psychs_p1p8_fu','psychs_p1p8_fu_hc',
        'chrpred_interview_date','sociodemographics']

        scid_checks = ScidChecks(row, timepoint, network,
        form_check_info)

        self.final_output_list = scid_checks()

        self.call_checks(row)
        
        #self.dates_per_tp = self.utils 
               
    def __call__(self):
        return self.final_output_list

    def call_checks(self, row):
        self.call_global_function_checks(row)
        self.call_oasis_checks(row)
        self.call_cssrs_checks(row)
        self.call_twenty_one_day_check(row)
        self.call_tbi_checks(row)
        self.call_bprs_checks(row)
        self.call_chrchs_checks(row)
        self.call_conversion_check(row)
        self.check_onset_date(row, 
        ['current_pharmaceutical_treatment_floating_med_125'],
        ['chrpharm_date_mod'], {'reports': ['Main Report']})

        self.pharm_firstdose_check(row, 
        ['current_pharmaceutical_treatment_floating_med_125'],
        ['chrpharm_med1_onset','chrpharm_firstdose_med1'],
        {'reports': ['Main Report']})

        self.pharm_date_mod_check(row, 
        ['current_pharmaceutical_treatment_floating_med_125'],
        ['chrpharm_date_mod'],
        {'reports': ['Main Report']})

        name_vars = self.grouped_vars['pharm_vars']['name_vars']
        self.pharm_med_name_check(row, 
        ['current_pharmaceutical_treatment_floating_med_125'],
        name_vars, {'reports': ['Main Report']})


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
        print(days_btwn)
        print(row.subjectid)   
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

    @FormCheck.standard_qc_check_filter   
    def pharm_firstdose_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        if (row.chrpharm_firstdose_med1 not in (self.utils.missing_code_list + [''])
        and row.chrpharm_med1_onset not in (self.utils.missing_code_list + [''])
        and str(row.chrpharm_med1_onset) != str(row.chrpharm_firstdose_med1)):
            return (f'chrpharm_med1_onset ({row.chrpharm_med1_onset}) does'
            f' not equal chrpharm_firstdose_med1 ({row.chrpharm_firstdose_med1})')

    @FormCheck.standard_qc_check_filter
    def pharm_date_mod_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        if row.subjectid in self.tp_date_ranges.keys():
            print(row.subjectid)
            pharm_date_mod = str(row.chrpharm_date_mod).split(' ')[0]
            tp_list = self.utils.create_timepoint_list()
            curr_tp = row.visit_status_string.replace(
            'screen','screening').replace('baseln','baseline')
            if curr_tp in tp_list:
                prev_visit_ind = tp_list.index(curr_tp) - 1
                prev_visit = tp_list[prev_visit_ind]  
                if (all(vis in self.tp_date_ranges[
                row.subjectid].keys() for vis in [curr_tp,prev_visit])):
                    if self.utils.check_if_val_date_format(pharm_date_mod):
                        all_date_comp_dates = []
                        for vis in [curr_tp,prev_visit]:
                            for date in self.tp_date_ranges[row.subjectid][vis].values():
                                all_date_comp_dates.append(datetime.strptime(date, "%Y-%m-%d"))
                        pharm_datetime = datetime.strptime(pharm_date_mod, "%Y-%m-%d")
                        if (all(pharm_datetime > vis_date for vis_date in all_date_comp_dates)
                        or all(pharm_datetime < vis_date for vis_date in all_date_comp_dates)):
                            return f"pharm date mode {pharm_date_mod} is out of range of current and previous visits" 


    @FormCheck.standard_qc_check_filter
    def pharm_med_name_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        pharm_vars = self.grouped_vars['parm_vars']['name_vars']
        pharm_var_vals = {}
        for var in pharm_vars:
            pharm_var_vals[var] = getattr(row,var) 
        duplicates = []
        for prim_var, prim_val in pharm_var_vals.items():
            print(prim_var)
            for second_var, second_val in pharm_var_vals.items():
                if prim_val == second_val:
                    duplicates.append(
                    f'{prim_var} (prim_val) = {second_var} (second_val)')
        if len(duplicates) > 0:
            return f"Duplicates found in pharm name vars ({duplicates})"

            


