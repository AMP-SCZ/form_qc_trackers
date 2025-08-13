import pandas as pd
import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
import re

class AnalyzeIdentifiers():
    """
    The purpose of this class is to 
    analyze the effects of identifiers on 
    branching logic and calculated fields.
    Identifiers are not exported to DPACC,
    at least by ProNET. Any programs
    that rely on calculated fields or branching logic
    need to factor this in.
    """

    def __init__(self):
        self.utils = Utils()
        with open(f'{self.utils.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.data_dict_df = self.utils.read_data_dictionary()
        identifiers_df = self.data_dict_df[self.data_dict_df[
        'Identifier?']=='y']
        self.identifiers = identifiers_df['Variable / Field Name'].tolist()

    def run_script(self):
        affected_vars = self.collect_calcs_with_identifiers('calc')
        affected_vars.extend(self.collect_calcs_with_identifiers('branching_logic'))
        output_df = pd.DataFrame(affected_vars)
        depend_path = self.config_info['paths']['dependencies_path']
        output_df.to_csv(f'{depend_path}identifier_effects.csv',index = False)
    
    def regex_ident_search(self, pattern, inp_str):
        match = re.search(pattern, inp_str)
        if match != None:
            return True
    
        return False

    def collect_calcs_with_identifiers(self,
        col_to_check : str
    ):
        affected_fields = []
        df = self.data_dict_df
        df = df.rename(
        columns={'Variable / Field Name': 'variable',
        'Choices, Calculations, OR Slider Labels': 'calc',
        'Branching Logic (Show field only if...)': 'branching_logic'})
        for row in df.itertuples():
            str_to_check = getattr(row, col_to_check)
            for ident_var in self.identifiers:
                if ident_var in str_to_check:
                    pattern = fr"\[{ident_var}\]"
                    if self.regex_ident_search(pattern,str_to_check):
                        affected_fields.append(
                        {'var': row.variable,'affected_col':col_to_check,
                        'affected_col_val':str_to_check,'identifier':ident_var})
                        
        return affected_fields
