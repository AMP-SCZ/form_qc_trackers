import pandas as pd
import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils



class CollectSubjectInfo():
    """
    Class to collect basic information
    about each subject, such as their cohort,
    sex, age, and inclusion status
    """

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.comb_csv_path = self.config_info['paths']['combined_csv_path']

        self.subject_info = {}

        self.var_translations = {
            'chrcrit_included': {1:'included',0: 'excluded'},
            'chrcrit_part' : {1: 'CHR', 2: 'HC'},
            'chrdemo_sexassigned' : {1 : 'Male', 2 : 'Female'}
        }

    def __call__(self):
        self.collect_screening_info()
        self.collect_baseline_info()

        return self.subject_info

    def translate_var_vals(self, translation_dict, var_val):
        for orig_val, translated_val in translation_dict.items():
            if var_val in self.utils.all_dtype([orig_val]):
                return translated_val
        
        return 'unknown'
    
    def collect_age(self,row):
        for age_var in ['chrdemo_age_mos_chr','chrdemo_age_mos_hc', 'chrdemo_age_mos2']:
            if not hasattr(row,age_var):
                continue
            age_val = getattr(row,age_var)
            if (age_val not in 
            (self.utils.missing_code_list + [""]) and 
            self.utils.can_be_float(age_val) and
            self.utils.is_a_num(str(age_val).replace('.',''))):
                age = round(float(age_val) / 12, 2)
                return age

        return 'unknown'

    def collect_screening_info(self):
        tp = 'screening'
        for network in ['PRONET','PRESCIENT']:
            combined_df = pd.read_csv(
                f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv')
            col_list = ['subjectid','visit_status_string',
            'chrcrit_part', 'chrcrit_included']
            col_list = [col for col in col_list if col in combined_df.columns]
            combined_df = combined_df[['subjectid','visit_status_string',
            'chrcrit_part', 'chrcrit_included']]
            for row in combined_df.itertuples():
                sub = row.subjectid
                self.subject_info.setdefault(sub, {})
                self.subject_info[sub][
                'inclusion_status'] = self.translate_var_vals(
                self.var_translations['chrcrit_included'], row.chrcrit_included)
                self.subject_info[sub][
                'cohort'] = self.translate_var_vals(
                self.var_translations['chrcrit_part'], row.chrcrit_part)
                self.subject_info[sub][
                'visit_status'] = row.visit_status_string

    def collect_baseline_info(self):
        tp = 'baseline'
        for network in ['PRONET','PRESCIENT']:
            combined_df = pd.read_csv(
                f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv')
            col_list = ['subjectid','chrdemo_age_mos_chr',
            'chrdemo_age_mos_hc', 'chrdemo_age_mos2', 'chrdemo_sexassigned']
            col_list = [col for col in col_list if col in combined_df.columns]
            combined_df = combined_df[col_list]
            for row in combined_df.itertuples():
                sub = row.subjectid
                self.subject_info.setdefault(sub, {})
                self.subject_info[sub][
                'sex'] = self.translate_var_vals(
                self.var_translations['chrdemo_sexassigned'], row.chrdemo_sexassigned)
                self.subject_info[sub][
                'age'] = self.collect_age(row)

    
