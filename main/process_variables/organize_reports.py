import pandas as pd
import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from define_important_variables import DefineImportantVariables
from transform_branching_logic import TransformBranchingLogic


class OrganizeReports():

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.data_dict_df = self.utils.read_data_dictionary()

        self.all_psychs_forms = ['psychs_p1p8_fu_hc','psychs_p9ac32_fu_hc',
        'psychs_p1p8','psychs_p9ac32','psychs_p1p8_fu','psychs_p9ac32_fu']

    def run_script(self):
        pass

    def define_excluded_variables(self):
        excluded_vars = {}
        pronet_excl_strings = ['comment', 'note','chrcssrsb_cssrs_yrs_sb',\
        'chrcssrsb_sb6l','chrcssrsb_nmapab','chrpas_pmod_adult3v3',\
        'chrpas_pmod_adult3v1','chrmri_t2_ge','chrcssrsfu_skip_aa',\
        'chric_surveys','chrdbb_phone_model','chrdbb_phone_software']

        excluded_strings =  {'PRONET':pronet_excl_strings,
        
        'PRESCIENT':pronet_excl_strings.extend(['chrdemo_racial','chrsaliva_food',
        'chrscid_overview_version','chrblood_freezerid',
        'chrdbb_phone_model','chrdbb_phone_software'] )}

        for network in ['PRONET','PRESCIENT']:
            filtered_df = self.utils.apply_df_str_filter(
            self.data_dict_df, excluded_strings[network], 'Variable / Field Name')
            excluded_vars[network] = filtered_df['Variable / Field Name'].tolist()
        
        return excluded_vars


    def organize_variable_checks(self):
        filtered_df = self.data_dict_df[self.data_dict_df['Required Field?']=='y']
        filtered_df = filtered_df[filtered_df['identifier?']!='y']

        blank_check_vars = self.organize_blank_check_vars(filtered_df)

    def organize_blank_check_vars(self,filtered_df):
        # remove psychs from blank check
        # these variables will be specified later
        filtered_df = filtered_df
        [~filtered_df['Form Name'].isin(self.all_psychs_forms)]

        blank_check_vars = {
        'Main Report':filtered_df[
        ~filtered_df['Field Type'].isin(
        ['notes','descriptive'])]['Variable / Field Name'].tolist(),
        
        'Secondary Report':filtered_df[
        filtered_df['Field Type'].isin(
        ['notes','descriptive'])]['Variable / Field Name'].tolist()
        }

        return blank_check_vars
    
    def organize_spec_val_check_vars(self):
    
        self.specific_value_check_dictionary = {'chrspeech_upload':\
            {'correlated_variable':'chrspeech_upload','checked_value_list':[0,0.0,'0','0.0'],\
            'branching_logic':"",'negative':False,'message':'Speech sample not uploaded to Box',\
            'report':['Secondary Report']},
            'chrpenn_complete':{'correlated_variable':'chrpenn_complete',\
            'checked_value_list':[2,2.0,'2','2.0',3,3.0,'3','3.0'],\
            'branching_logic':"",'negative':False,\
            'message': f'Penncnb not completed (value is either 2 or 3).',\
            'report':['Secondary Report','Cognition Report']}}


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
                    essential_psychs_vars.append('chrpsychs_scr_'\
                    + str(x) + key + number)
                    essential_psychs_vars.append('chrpsychs_scr_'\
                    + str(x) + key + number + '_app')
                    if key == 'd':
                        essential_psychs_vars.extend(['chrpsychs_fu_'+\
                        str(x) + key + number,\
                        'chrpsychs_fu_'+ str(x) + key + number+'_app',\
                        'hcpsychs_fu_'+ str(x) + key + number,'hcpsychs_fu_'+\
                        str(x) + key + number+'_app'])
        essential_psychs_vars.extend(['chrpsychs_scr_e11',\
            'chrpsychs_scr_e27','chrpsychs_scr_e11_app',\
            'chrpsychs_scr_e27_app','chrpsychs_fu_e27',\
            'chrpsychs_fu_e27_app','hcpsychs_fu_e27','hcpsychs_fu_e27_app'])
        for x in essential_psychs_vars:
            if 'app' in x and 'psychs' in x:
                self.specific_value_check_dictionary[x] =\
                {'correlated_variable':x,'checked_value_list':[0,0.0,'0','0.0'],\
                'branching_logic':"",'negative':False,\
                'message': f'value is 0','report':['Main Report']}



if __name__ == '__main__':
    OrganizeReports().run_script()

        