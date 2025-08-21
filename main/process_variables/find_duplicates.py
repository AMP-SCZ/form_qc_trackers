import pandas as pd
from main.utils.utils import Utils
from collections import Counter
import json

class DuplicateFinder():
    """
    Class to find duplicates between multiples
    timepoints or multiple participants 
    """

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        self.earliest_latest_dates_per_tp = {}
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        depen_path = self.config_info['paths']['dependencies_path']

        self.multi_tp_df = pd.read_csv(f'{depen_path}multi_tp_PRESCIENT_combined.csv',
        keep_default_na = False) 
        self.grouped_vars = self.utils.load_dependency_json(
        f"grouped_variables.json")
        self.blood_vars = self.grouped_vars["blood_vars"]
        self.find_blood_id_duplicates()

    def __call__(self):
        self.find_blood_id_duplicates()

    def find_blood_id_duplicates(self):
        """
        Finds duplicate IDS between different 
        participants  
        """
        id_vars = self.blood_vars['id_variables']
        filtered_df = self.multi_tp_df
        print(filtered_df.columns)
        id_vars = [var for var in filtered_df.columns 
        if any(id in var for id in id_vars)]
        new_cols = id_vars + ['subjectid']
        filtered_df = filtered_df[new_cols]
        dup_count = 0
        duplicates = {}
        for id_var in id_vars:
            missing_removed_df = filtered_df[
            ~filtered_df[id_var].isin(self.utils.missing_code_list + [''])]
            id_dict = missing_removed_df.set_index('subjectid')[id_var].to_dict()
            id_values = list(id_dict.values())
            seen = set()
            duplicates = [x for x in id_values if x in seen or seen.add(x)]
            if len(duplicates)> 0:
                dup_count += 1
                print(dup_count)
            #print(duplicates)


            
        
