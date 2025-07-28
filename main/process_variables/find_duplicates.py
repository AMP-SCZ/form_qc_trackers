import pandas as pd
from main.utils.utils import Utils

class DuplicateFinder():
    """
    Class to find duplicates between multiples
    timepoints or multiple participants 
    """

    def __init__(self):
        self.utils = Utils()
        self.multi_tp_df = pd.read_csv('/home/ob001/dependencies/multi_tp_PRESCIENT_combined.csv',
        keep_default_na = False) 
        self.grouped_vars = self.utils.load_dependency_json(
        f"grouped_variables.json")
        self.blood_vars = self.grouped_vars["blood_vars"]
        self.find_blood_id_duplicates()

    def __call__(self):
        pass

    def find_blood_id_duplicates(self):
        """
        Finds duplicate IDS between different 
        participants or between different 
        variables of the same participant 
        """
        id_vars = self.blood_vars['id_variables']
        filtered_df = self.multi_tp_df
        id_vars = [var for var in filtered_df.columns if any(id in var for id in id_vars)]

        
        new_cols = id_vars + ['subjectid']
        print(new_cols)
        filtered_df = filtered_df[new_cols]
        print(filtered_df.columns)
        


        



    

        
