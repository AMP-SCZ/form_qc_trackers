import pandas as pd
import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class OrganizeReports():

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.data_dict_df = self.utils.read_data_dictionary()

        self.all_psychs_forms = ['psychs_p1p8_fu_hc','psychs_p9ac32_fu_hc',
        'psychs_p1p8','psychs_p9ac32','psychs_p1p8_fu','psychs_p9ac32_fu']

        self.self_report_forms = ['pubertal_developmental_scale',\
        'psychosis_polyrisk_score','oasis','item_promis_for_sleep','pgis',\
        'perceived_stress_scale','perceived_discrimination_scale']

        self.rel_psychs_vars = self.collect_psychs_variables()

    def run_script(self):
        self.organize_variable_checks()

    def organize_variable_checks(self):
        qc_var_info = {
            "blank_check_vars" : self.organize_blank_check_vars(),
            "specific_val_check_vars" : self.organize_spec_val_check_vars(),
            "checkbox_vars" : self.organize_checkbox_vars(),
            "excluded_vars" : self.define_excluded_variables()
        }

        self.utils.save_dependency_json(
        qc_var_info,'general_check_vars.json')

        team_report_forms = self.define_team_report_forms()
        self.utils.save_dependency_json(
        team_report_forms, 'team_report_forms.json')

    def organize_blank_check_vars(self):

        # applies filters that are relevant to both reports
        filtered_df = self.filter_blank_check_df()
        
        main_report_df = filtered_df[(
        ~filtered_df['Field Type'].isin(
        ['notes','descriptive'])) | (filtered_df[
        'Variable / Field Name'].isin(self.define_additional_blank_check_vars()))]

        main_report_df = main_report_df[
        ~main_report_df['Form Name'].isin(self.self_report_forms)]

        secondary_report_df = filtered_df[(
        filtered_df['Field Type'].isin(
        ['notes','descriptive'])) | (filtered_df['Form Name'].isin(
        self.self_report_forms))]

        blank_check_vars = {
        'Main Report':main_report_df.groupby(
        'Form Name')['Variable / Field Name'].apply(list).to_dict(),

        'Secondary Report':secondary_report_df.groupby(
        'Form Name')['Variable / Field Name'].apply(list).to_dict()
        }

        return blank_check_vars
    
    def filter_blank_check_df(self):

        filtered_df = self.data_dict_df[self.data_dict_df['Identifier?']!='y']
        additional_blank_check_vars = self.define_additional_blank_check_vars()

        filtered_df = filtered_df[
        ((filtered_df['Required Field?']=='y') & (filtered_df[
        'Field Type']!='checkbox') & (filtered_df[
        'Identifier?']!='y') &(~filtered_df['Form Name'].isin(self.all_psychs_forms))) | (filtered_df[
        'Variable / Field Name'].isin(additional_blank_check_vars))]

        return filtered_df

    def define_additional_blank_check_vars(self):
        additional_blank_check_vars = [
        'chrpsychs_av_dev_desc', 'chrcrit_included',
        'chrchs_timeslept','chrdemo_age_mos_chr',
        'chrdemo_age_mos_hc','chrdemo_age_mos2','chroasis_oasis_1',
        'chroasis_oasis_3','chrblood_rack_barcode','chrcrit_inc3']
        
        pharm_vars_df = self.data_dict_df[
        (self.data_dict_df['Variable / Field Name'].str.contains('chrpharm_med')
        & self.data_dict_df['Variable / Field Name'].str.contains('name_past'))]

        pharm_vars = pharm_vars_df['Variable / Field Name'].tolist()
        scid_df = self.data_dict_df[
        self.data_dict_df['Form Name'] == 'scid5_psychosis_mood_substance_abuse']
        all_scid_vars = scid_df['Variable / Field Name'].tolist()
        additional_blank_check_vars.extend(pharm_vars)

        additional_blank_check_vars.extend(self.collect_psychs_variables())
        additional_blank_check_vars.extend(all_scid_vars)

        return additional_blank_check_vars
    
    def organize_spec_val_check_vars(self):
        specific_value_check_dictionary = {'chrspeech_upload':
            {'correlated_variable':'chrspeech_upload',
            'checked_value_list':[0,0.0,'0','0.0'],
            'branching_logic':"",'negative':'False',
            'message':'Speech sample not uploaded to Box',
            'report':['Secondary Report']},
            'chrpenn_complete':{'correlated_variable':'chrpenn_complete',
            'checked_value_list':[2,2.0,'2','2.0',3,3.0,'3','3.0'],
            'branching_logic':"",'negative':'False',
            'message': f'Penncnb not completed (value is either 2 or 3).',
            'report':['Secondary Report','Cognition Report']}}
                
        for var in self.rel_psychs_vars:
            if 'app' in var:
                specific_value_check_dictionary[var] ={
                'correlated_variable':var,'checked_value_list':[0,0.0,'0','0.0'],
                'branching_logic':"",'negative':'False',
                'message': f'value is 0','report':['Main Report','Non Team Forms']}

        return specific_value_check_dictionary

    def organize_checkbox_vars(self):
        checkbox_df = self.data_dict_df[self.data_dict_df['Required Field?']=='y']
        checkbox_df = checkbox_df[checkbox_df[
        'Field Type'] == 'checkbox']
        checkbox_vars = checkbox_df['Variable / Field Name'].tolist()

        return checkbox_vars

    def define_excluded_variables(self):
        excluded_vars = {}
        pronet_excl_strings = ['comment', 'note','chrcssrsb_cssrs_yrs_sb',
        'chrcssrsb_sb6l','chrcssrsb_nmapab','chrpas_pmod_adult3v3',
        'chrpas_pmod_adult3v1','chrmri_t2_ge','chrcssrsfu_skip_aa',
        'chric_surveys','chrdbb_phone_model','chrdbb_phone_software','chrblood_pl1id_2',
        'chrdemo_parent_fa','chrdemo_parent_mo','chrmri_dmri176_qc']

        excluded_strings =  {'PRONET':pronet_excl_strings,
        
        'PRESCIENT':(pronet_excl_strings + ['chrdemo_racial','chrsaliva_food',
        'chrscid_overview_version','chrblood_freezerid',
        'chrdbb_phone_model','chrdbb_phone_software'])}

        for network in ['PRONET','PRESCIENT']:
            filtered_df = self.utils.apply_df_str_filter(
            self.data_dict_df, excluded_strings[network], 'Variable / Field Name')
            excluded_vars[network] = filtered_df['Variable / Field Name'].tolist()
        
        return excluded_vars

    def collect_psychs_variables(self):
        """
        Function to collect all of the Psychs
        variables that will be checked and
        adds them to the additional variables list
        """
        essential_psychs_vars = []
        letter_match_dict = {'b': ['5','6','19'],'d': ['16','30']}
        for key in letter_match_dict.keys():
            for x in range (1,16):
                for number in letter_match_dict[key]:
                    essential_psychs_vars.append('chrpsychs_scr_'
                    + str(x) + key + number)
                    essential_psychs_vars.append('chrpsychs_scr_'
                    + str(x) + key + number + '_app')
                    if key == 'd':
                        essential_psychs_vars.extend(['chrpsychs_fu_'+
                        str(x) + key + number,
                        'chrpsychs_fu_'+ str(x) + key + number+'_app',
                        'hcpsychs_fu_'+ str(x) + key + number,'hcpsychs_fu_'+
                        str(x) + key + number+'_app'])
        essential_psychs_vars.extend(['chrpsychs_scr_e11',
            'chrpsychs_scr_e27','chrpsychs_scr_e11_app',
            'chrpsychs_scr_e27_app','chrpsychs_fu_e27',
            'chrpsychs_fu_e27_app','hcpsychs_fu_e27','hcpsychs_fu_e27_app'])
        
        print(essential_psychs_vars)

        return essential_psychs_vars
    
    def define_team_report_forms(self):
        team_reports = {
        'Blood Report':['blood_sample_preanalytic_quality_assurance','cbc_with_differential'],
        'Cognition Report':['penncnb','premorbid_iq_reading_accuracy',
        'iq_assessment_wasiii_wiscv_waisiv'],
        'Scid Report':['scid5_psychosis_mood_substance_abuse'],
        'MRI Report':['mri_run_sheet'],'EEG Report':['eeg_run_sheet'],
        'Digital Report':['digital_biomarkers_mindlamp_onboarding',
        'digital_biomarkers_axivity_onboarding',
        'digital_biomarkers_mindlamp_checkin','digital_biomarkers_axivity_checkin',
        'digital_biomarkers_axivity_end_of_12month_study_pe',
        'digital_biomarkers_mindlamp_end_of_12month_study_p'],
        'Fluids Report':['gcp_cbc_with_differential','gcp_current_health_status',
        'daily_activity_and_saliva_sample_collection','blood_sample_preanalytic_quality_assurance',
        'cbc_with_differential', 'current_health_status']}

        non_team_forms = []
        all_team_forms = []
        all_forms  = self.data_dict_df['Form Name'].unique().tolist()

        for key,value in team_reports.items():
            for form in value:
                if form not in all_team_forms:
                    all_team_forms.append(form)
        for form in all_forms:
            if form not in all_team_forms:
                if form not in non_team_forms:
                    non_team_forms.append(form)

        # all forms not yet defined in the above dictionary
        team_reports['Non Team Forms'] = non_team_forms 


        return team_reports


if __name__ == '__main__':
    OrganizeReports().run_script()

        