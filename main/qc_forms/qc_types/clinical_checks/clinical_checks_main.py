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

class ClinicalChecksMain(FormCheck):    
    """
    QC Checks related to clinical forms
    """   
    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network, form_check_info)
        self.test_val = 0
        self.gf_score_check_vars = {'high':{'global_functioning_social_scale':{
        'chrgfs_gf_social_high' : ['chrgfs_gf_social_scale','chrgfs_gf_social_low']},
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

        # subjects should be converted if 
        # these variables exceed their respective values
        self.gt_var_val_pairs = {'chrbprs_bprs_somc': 5,
        'chrbprs_bprs_guil':5,'chrbprs_bprs_gran':5,
        'chrbprs_bprs_susp':5,'chrbprs_bprs_hall':5,
        'chrbprs_bprs_unus':5,'chrbprs_bprs_bizb':5,
        'chrbprs_bprs_conc':5}
        
        # subjects should be converted if 
        # these variables equal their respective values
        self.eq_var_val_pairs = {'chrpsychs_fu_1c0':6,
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

        scid_checks = ScidChecks(row, timepoint, network,
        form_check_info)

        #self.final_output_list = scid_checks()

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
        self.call_pharm_checks(row)
        self.call_premorbid_adjustment_checks(row)
        self.call_age_comparisons(row)
        self.call_pps_checks(row)
    
    def call_pps_checks(self,row):
        forms = ['psychosis_polyrisk_score']
        reports = {'reports' : ['Main Report']}
        for dob_var, pps_age_var in {'chrpps_fdob':'chrpps_fage',
        'chrpps_mdob':'chrpps_mage'}.items():
            vars = ['chrpps_interview_date',pps_age_var]
            self.pps_dob_age_range_check(row, forms,
            ['chrpps_interview_date', dob_var], reports)
            for age_var in [pps_age_var, 'demographics_age']:
                self.pps_age_comp(row,forms,['chrpps_interview_date',
                dob_var, age_var], bl_filtered_vars =[],
                filter_excl_vars=False, age_var = age_var,
                pps_dob_var = dob_var)

    def call_age_comparisons(self,row):
        for var in ['chrpps_fage','chrpps_mage',
        'chrfigs_mother_age','chrfigs_father_age']:
            forms = [self.grouped_vars['var_forms'][var]]
            reports = {'reports' : ['Main Report']}
            self.age_comparison(row, forms, [var],reports, 
            bl_filtered_vars = [], filter_excl_vars = True,
            compared_age_var = var, diff_min = 10, diff_max = 85)

        for var in ['chrscid_c56_c65','chrscid_a52',
        'chrscid_a109','chrscid_c58_c66',
        'chrscid_d45_d49_d54_d56','chrscid_d61_d65',
        'chrscid_e19_37','chrscid_e164_302',
        'chrscid_e168_306','chrscid_e172_310',
        'chrscid_e176_314','chrscid_e180_318','chrscid_e184_322',
        'chrscid_e188_326','chrscid_e192_330']:
            forms = [self.grouped_vars['var_forms'][var]]
            reports = {'reports' : ['Main Report']}
            self.age_comparison(row, forms, [var], reports, 
            bl_filtered_vars = [], filter_excl_vars = True,
            compared_age_var = var, diff_min = 0, diff_max = 0)

    def call_premorbid_adjustment_checks(self, row):
        forms = ['premorbid_adjustment_scale']

        reports = {'reports' : ['Main Report']}

        self.pas_marriage_check(row, forms, ['chrpas_pmod_adult3v3',
        'chrpas_pmod_adult3v1'], reports)

    def call_pharm_checks(self, row):
        past_pharm_form = ['past_pharmaceutical_treatment']
        curr_pharm_forms = ['current_pharmaceutical_treatment_floating_med_125',
        'current_pharmaceutical_treatment_floating_med_2650']

        reports = {'reports' : ['Main Report', 'Non Teams Forms']}

        self.check_onset_date(row, curr_pharm_forms,
        ['chrpharm_date_mod'], reports)

        self.pharm_firstdose_check(row, curr_pharm_forms,
        ['chrpharm_med1_onset','chrpharm_firstdose_med1'], reports)

        self.pharm_date_mod_check(row, curr_pharm_forms,
        ['chrpharm_date_mod'], reports)

        name_vars = self.grouped_vars['pharm_vars']['name_vars']
        for forms in [past_pharm_form, curr_pharm_forms]:
            self.pharm_med_name_check(row, 
            curr_pharm_forms, name_vars, reports)
        
        self.medication_blinded_check(row, name_vars, curr_pharm_forms)
        self.check_onset_offset_dates(row)

        for initial_med_count in range(0,50):
            for secondary_med_count in range(0,50):
                if initial_med_count == secondary_med_count:
                    continue
                med_vars = [
                f'chrpharm_med{initial_med_count}_name',
                f'chrpharm_med{initial_med_count}_onset',
                f'chrpharm_lastuse_med{initial_med_count}',
                f'chrpharm_med{initial_med_count}_use',
                f'chrpharm_med{secondary_med_count}_name',
                f'chrpharm_med{secondary_med_count}_onset',
                f'chrpharm_lastuse_med{secondary_med_count}',
                f'chrpharm_med{secondary_med_count}_use'
                ]  
                self.pharm_overlapping_days(row, forms, med_vars, reports)  

    def medication_blinded_check(
        self, row, med_name_vars, form
    ):
        """
        Checks if med statuses are appropriate
        """
        for name_var in med_name_vars:
            if hasattr(row, name_var):
                if (getattr(row, name_var) in
                self.utils.all_dtype([573,542,538,539])):
                    error_message = (f"Medication is currently"
                    f" blinded ({name_var} = {getattr(row, name_var)})")
                    output_changes = {'reports' : ['Main Report']}
                    error_output = self.create_row_output(
                    row, [form], [name_var],
                    error_message, output_changes)
                    self.final_output_list.append(error_output)
                elif getattr(row, name_var) in self.utils.all_dtype([888]):
                    pharm_count = self.utils.collect_digit(str(getattr(row,name_var)))
                    comp_vars_with_vals = []
                    comp_vars = [
                    f'chrpharm_med{pharm_count}_onset',
                    f'chrpharm_med{pharm_count}_offset',
                    f'chrpharm_interm_meds_{pharm_count}',
                    f'chrpharm_med{pharm_count}_dosage',
                    f'chrpharm_med{pharm_count}_dosage_2',
                    f'chrpharm_med{pharm_count}_comp',
                    f'chrpharm_med{pharm_count}_comp_2'
                    ]
                    for tp_count in range(0, 9):
                        comp_vars.append(f'chrpharm_med{pharm_count}_mo{tp_count}')
                    for comp_var in comp_vars:
                        if (hasattr(row, comp_var) and
                        getattr(row, comp_var) not in 
                        (self.utils.missing_code_list + [''])):
                            comp_vars_with_vals.append(comp_var)
                    if len(comp_vars_with_vals) > 0:
                        output_changes = {'reports' : reports}
                        error_output = self.create_row_output(
                        row, [curr_psychs_form], [missing_spec_var],
                        error_message, output_changes)
                        self.final_output_list.append(error_output)
                        error_message = (f"The med_name indicates that there"
                        " is no information on whether the subject took any"
                        f" medication, but other fields ({comp_vars_with_vals})"
                        " indicate that 777 is more plausible") 
                        
    def call_conversion_check(self,row):
        for var, threshold in self.gt_var_val_pairs.items():
            form = self.grouped_vars['var_forms'][var]
            if not hasattr(row,var):
                continue
            var_val = getattr(row, var) 
            if self.utils.can_be_float(var_val) and float(var_val) > threshold:
                self.conversion_criteria_check(row, [form],
                [var], {'reports': ['Main Report']})

        for var, threshold in self.eq_var_val_pairs.items():
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
            self.oasis_anxiety_check(row, forms, ['chroasis_oasis_1', oasis_anx_var],
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
        cohort = self.subject_info[row.subjectid]['cohort'].lower()
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
        filter_excl_vars=True
    ):
        date_list = []
        for x in range(0,10):
            date_list.append(f'chrpharm_med{x}_onset')
        for date_var in date_list:
            if hasattr(row, date_var):
                date_val = str(getattr(row,date_var))
                data_entry_val = str(getattr(row,'chrpharm_date_mod'))
                if (date_val in self.utils.missing_code_list or 
                data_entry_val in self.utils.missing_code_list):
                    continue
                if (self.utils.check_if_val_date_format(
                date_val, date_format="%Y-%m-%d")
                and self.utils.check_if_val_date_format(
                data_entry_val, date_format="%Y-%m-%d")):
                    days_btwn = self.utils.find_days_between(
                    date_val,data_entry_val)      
                    if days_btwn > 10:
                        return (f'There are {days_btwn} days between'
                        f' the most recent medication date ({date_val}) and'
                        f' medication mod date ({data_entry_val})')
        
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
            return (f'{all_vars[0]} is {getattr(row, all_vars[0])},'
            ' but participant is not marked as converted.')

    @FormCheck.standard_qc_check_filter   
    def pas_marriage_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False
    ):
        """
        These variables are excluded,
        so need to set filter_excl_vars to False
        """

        if (getattr(row, 'chrpas_pmod_adult3v1') in self.utils.all_dtype([0,1,2,3]) and
        getattr(row, 'chrpas_pmod_adult3v3') in self.utils.all_dtype([0,1,2,3,4,5,6])):
            return (f"chrpas_pmod_adult3v1 is equal to {getattr(row, 'chrpas_pmod_adult3v1')},"
            f" but chrpas_pmod_adult3v3 is equal to {getattr(row, 'chrpas_pmod_adult3v3')}")
            
    @FormCheck.standard_qc_check_filter   
    def pharm_firstdose_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        if (str(row.chrpharm_med1_onset).split(' ')[0]
        not in (self.utils.missing_code_list + [''])
        and str(row.chrpharm_firstdose_med1).split(' ')[0]
        not in (self.utils.missing_code_list + [''])
        and str(row.chrpharm_med1_onset).split(' ')[0]
        != str(row.chrpharm_firstdose_med1).split(' ')[0]):
            return (f'chrpharm_med1_onset ({row.chrpharm_med1_onset}) does'
            f' not equal chrpharm_firstdose_med1 ({row.chrpharm_firstdose_med1})')

    @FormCheck.standard_qc_check_filter
    def pharm_date_mod_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        """
        Checks that pharm form has been
        during the current or previous timepoint

        #TODO find out if both pharm forms need
        to be filled out at each timepoint
        """
        
        if row.subjectid in self.tp_date_ranges.keys():
            mod_vars_out_of_range = 0
            for mod_var in ['chrpharm_date_mod','chrpharm_date_mod_2']:
                pharm_date_mod = str(getattr(row, mod_var)).split(' ')[0]
                if pharm_date_mod in self.utils.missing_code_list:
                    continue
                tp_list = self.utils.create_timepoint_list()
                curr_tp = row.visit_status_string.replace(
                'screen','screening').replace('baseln','baseline')
                if curr_tp in tp_list:
                    prev_visit_ind = tp_list.index(curr_tp) - 1
                    if prev_visit_ind >= 0:
                        prev_visit = tp_list[prev_visit_ind]  
                        vis_list =  [curr_tp, prev_visit]
                    else:
                        vis_list =  [curr_tp]
                    if (all(vis in self.tp_date_ranges[
                    row.subjectid].keys() for vis in vis_list)):
                        if self.utils.check_if_val_date_format(pharm_date_mod):
                            all_date_comp_dates = []
                            for vis in vis_list:
                                for date in self.tp_date_ranges[row.subjectid][vis].values():
                                    all_date_comp_dates.append(datetime.strptime(date, "%Y-%m-%d"))
                            pharm_datetime = datetime.strptime(pharm_date_mod, "%Y-%m-%d")
                            if (all(pharm_datetime > vis_date for vis_date in all_date_comp_dates)
                            or all(pharm_datetime < vis_date for vis_date in all_date_comp_dates)):
                                mod_vars_out_of_range +=1
            if mod_vars_out_of_range == 2:
                return (f"pharm modification dates ({getattr(row,'chrpharm_date_mod')}"
                f" and {getattr(row,'chrpharm_date_mod')}) are out of range of current and previous visits") 

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
    
    @FormCheck.standard_qc_check_filter
    def pharm_overlapping_days(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        date_vars = []
        nums = []
        for var in all_vars:
            if 'onset' in var or 'lastuse' in var:
                date_vars.append(var)
            var_num = self.utils.collect_digit(var)
            if self.utils.collect_digit(var_num) not in nums:
                nums.append(var_num)
        lower_num = min(nums)
        upper_num = max(nums)
        # makes sure all date variables are valid
        if any((self.utils.check_if_val_date_format(
        str(getattr(row, date_var)).split(' ')[0]) == False 
        or str(getattr(row, date_var)).split(' ')[0] in
        self.utils.missing_code_list)
        for date_var in date_vars):
            return
        if self.det_if_overlapping_pharm_dates(row,nums) == True:
            med_use_lower = getattr(row,f'chrpharm_med{lower_num}_use')
            med_use_upper = getattr(row,f'chrpharm_med{upper_num}_use')
            if (not((med_use_lower in self.utils.all_dtype([1])
            and med_use_upper in self.utils.all_dtype([2]))
            or (med_use_lower in self.utils.all_dtype([2]) 
            and med_use_upper in self.utils.all_dtype([1])))):
                lower_name = getattr(row,f'chrpharm_med{lower_num}_name')
                upper_name = getattr(row,f'chrpharm_med{upper_num}_name')
                if lower_name == upper_name:
                    return (f"Medications have the same names and overlapping dates"
                    f" (chrpharm_med{lower_num}_name = {lower_name} and"
                    f" chrpharm_med{upper_num}_name = {upper_name} )")
                    
    def check_onset_offset_dates(self, row):
        for med_count in range(0,61):
            for pharm_tp_str in ["","_past"]
            onset_var = f"chrpharm_med{med_count}_onset{pharm_tp_str}"
            offset_var = f"chrpharm_med{med_count}_offset{pharm_tp_str}"
            if (hasattr(row, onset_var) and hasattr(row,offset_var)):
                onset_val = str(getattr(row, onset_var)).split(' ')[0]
                offset_val = str(getattr(row, offset_var)).split(' ')[0]
                if all(self.utils.check_if_val_date_format(date_val)
                for date_val in [onset_val, offset_val]):
                    if (datetime.strptime(onset_val, '%Y-%m-%d')
                    > datetime.strptime(offset_val, '%Y-%m-%d')):
                        error_message = (f"Onset date ({onset_var} = {onset_val})"
                        f" is later than offset date ({offset_var} = {offset_val})")
                        output_changes = {'reports' : ['Main Report']}
                        error_output = self.create_row_output(
                        row, [form], [name_var],
                        error_message, output_changes)
                        self.final_output_list.append(error_output)




    def det_if_overlapping_pharm_dates(self,row, nums):
        ranges_iterated_daily = []
        ranges_per_var_num = []
        for num in nums:
            onset_var = f'chrpharm_med{num}_onset'
            lastuse_var = f'chrpharm_lastuse_med{num}'
            onset_val = datetime.strptime(str(getattr(
            row,onset_var)).split(' ')[0],"%Y-%m-%d") 
            lastuse_val = datetime.strptime(str(getattr(
            row, lastuse_var)).split(' ')[0],"%Y-%m-%d") 
            ranges_per_var_num.append(
            f'{onset_val} to {lastuse_val}')
            if onset_val == lastuse_val:
                ranges_iterated_daily.append(onset_val)
                continue
            if onset_val > lastuse_val:
                return False
            add_days = True
            while add_days:
                ranges_iterated_daily.append(onset_val)
                if onset_val == lastuse_val:
                    add_days = False
                onset_val += timedelta(days=1)
        ranges_set = (ranges_iterated_daily)
        # checks for duplicates
        if len(ranges_set) == len(ranges_iterated_daily):
            return True
        else:
            return False

    @FormCheck.standard_qc_check_filter
    def pps_dob_age_range_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        if (any((getattr(row,var) in self.utils.missing_code_list or not 
        self.utils.check_if_val_date_format(str(getattr(row,var)).split(' ')[0]))
        for var in all_vars)):
            return 
        pps_int_date = str(getattr(row,all_vars[0])).split(' ')[0]
        pps_date = str(getattr(row,all_vars[1])).split(' ')[0]
        days_between = self.utils.find_days_between(pps_int_date, pps_date)
        yrs_between = days_between/365
        if yrs_between < 10 or yrs_between > 85:
            return f"Difference between {all_vars[0]} and {all_vars[1]} is {yrs_between}."

    @FormCheck.standard_qc_check_filter
    def pps_age_comp(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, age_var = '', pps_dob_var = ''
    ):
        # if date variables are missing codes or improper dates
        if (any((getattr(row,var) in self.utils.missing_code_list or not 
        self.utils.check_if_val_date_format(str(getattr(row,var)).split(' ')[0]))
        for var in [pps_dob_var,'chrpps_interview_date'])):
            return 
        if age_var == 'demographics_age':
            age = self.subject_info[row.subjectid]['age']
        else:
            age = getattr(row, age_var)
        # if age variable is missing codes or not a number
        if (age in self.utils.missing_code_list 
        or not self.utils.can_be_float(age)):
            return
        pps_int_date = str(getattr(row,'chrpps_interview_date')).split(' ')[0]
        pps_dob = str(getattr(row,pps_dob_var)).split(' ')[0]
        days_between = self.utils.find_days_between(pps_int_date, pps_dob)
        yrs_between = days_between/365
        age_diffs = yrs_between - float(age)
        if age_diffs >= 1:
            return f"Age ({age_var} = {age}) does not align with date of birth ({pps_dob})."

    @FormCheck.standard_qc_check_filter
    def age_comparison(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, compared_age_var = '',
        diff_min = 0, diff_max = 0
    ):
        if (any(getattr(row,var) in self.utils.missing_code_list
        for var in all_vars)):
            return
        age = self.subject_info[row.subjectid]['age']
        compared_age = getattr(row,compared_age_var)
        if self.utils.can_be_float(age) and self.utils.can_be_float(compared_age):
            diff = round(abs(float(age) - float(compared_age)), 2)
            if diff < diff_min or diff > diff_max:
                return (f"Difference between demographics"
                f" age ({age}) and {compared_age_var} ({compared_age}) is {diff}")