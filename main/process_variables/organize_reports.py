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

    def run_script(self):
        pass

    def organize_blank_check_vars(self):
        filtered_df = self.data_dict_df[self.data_dict_df['Required Field?']=='y']
        filtered_df = filtered_df[filtered_df['identifier?']!='y']

        self.blank_check_vars = {
        'Main Report':filtered_df[
        ~filtered_df['Field Type'].isin(['notes','descriptive'])].tolist(),
        
        'Secondary Report':filtered_df[
        filtered_df['Field Type'].isin(['notes','descriptive'])].tolist()
        }


    def collect_psychs_variables(self):
        """
        Function to collect all of the Psychs
        variables that will be checked and
        adds them to the additional variables list
        """

        letter_match_dict = {'b': ['5','6','19'],'d': ['16','30']}
        for key in letter_match_dict.keys():
            for x in range (1,16):
                for number in letter_match_dict[key]:
                    self.additional_variables.append('chrpsychs_scr_'\
                    + str(x) + key + number)
                    self.additional_variables.append('chrpsychs_scr_'\
                    + str(x) + key + number + '_app')
                    if key == 'd':
                        self.additional_variables.extend(['chrpsychs_fu_'+\
                        str(x) + key + number,\
                        'chrpsychs_fu_'+ str(x) + key + number+'_app',\
                        'hcpsychs_fu_'+ str(x) + key + number,'hcpsychs_fu_'+\
                        str(x) + key + number+'_app'])
        self.additional_variables.extend(['chrpsychs_scr_e11',\
            'chrpsychs_scr_e27','chrpsychs_scr_e11_app',\
            'chrpsychs_scr_e27_app','chrpsychs_fu_e27',\
            'chrpsychs_fu_e27_app','hcpsychs_fu_e27','hcpsychs_fu_e27_app'])
        for x in self.additional_variables:
            if 'app' in x and 'psychs' in x:
                self.specific_value_check_dictionary[x] =\
                {'correlated_variable':x,'checked_value_list':[0,0.0,'0','0.0'],\
                'branching_logic':"",'negative':False,\
                'message': f'value is 0','report':['Main Report']}



if __name__ == '__main__':
    OrganizeReports().run_script()

        