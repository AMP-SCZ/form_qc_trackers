import pandas as pd
import os
import sys
import json
import re

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class APMedMapper():
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
        self.data_dict = self.utils.read_data_dictionary()

        self.subject_info = {}

    def __call__(self):
        return self.map_ap_meds_to_pharm_codes() 


    def map_ap_meds_to_pharm_codes(self):
        filtered_df = self.data_dict[self.data_dict[
        'Field Label'].str.contains('Have you ever taken')]
        filtered_df = filtered_df[filtered_df[
        'Variable / Field Name'].str.contains('chrap')]

        med_name_df = self.data_dict[self.data_dict[
        'Variable / Field Name'] 
        == 'chrpharm_med1_name_past']

        med_options = med_name_df["Choices, Calculations, OR Slider Labels"].iloc[0]
        med_options = med_options.split('|')

        med_option_dict ={}
        for med in med_options:
            med_num = med.split(',')[0].strip()
            med_name = med.split(',')[1].strip()
            med_option_dict[med_name] = med_num


        filtered_df = filtered_df.rename(columns={
            "Variable / Field Name": "variable",
            "Field Label": "field_label",
        })

        ap_form_meds = {}
        ap_med_list = []

        for row in filtered_df.itertuples():
           # print(row.variable)
           # print(row.field_label)
            field_label = row.field_label
            ap_med_str = field_label.split('Have you ever taken ')[-1]
            ap_med_list.extend(re.findall(r"\b\w+\b", ap_med_str))
            
            #print(ap_med_list)

        for ap_med in ap_med_list:
            ap_form_meds.setdefault(ap_med, [])
            for med, med_num in med_option_dict.items():
                med_list =  med.lower().replace('/',' ').replace('_',' ')
                med_list = med_list.split(' ')
                if ap_med.lower() in med_list and 'special' not in med.lower():
                    ap_form_meds[ap_med].append(med_num)

        return ap_form_meds
        
