import pandas as pd
import sys
import os
from itertools import combinations

parent_dir_path = os.path.abspath(\
os.path.join(os.path.dirname(__file__),\
'..', '/PHShome/ob001/anaconda3/new_forms_qc/QC/'))
if parent_dir_path not in sys.path:
    sys.path.append(parent_dir_path)

from utils.utils import Utils
import json
class VariableMismatches():
    """
    Class to discover variables 
    that are typically equal to each other.
    """
    def __init__(self):
        self.final_output = {}
        self.utils = Utils()
        self.absolute_path = '/PHShome/ob001/anaconda3/new_forms_qc/QC/'
        self.combined_df_folder = '/data/predict1/data_from_nda/formqc/'
        self.excluded_field_types =\
        ['checkbox','dropdown','radio','yesno']

        with open(f'{self.absolute_path}unique_form_vars.json', 'r') as file:
            self.unique_form_vars = json.load(file)
        self.all_forms = list(self.unique_form_vars.keys())
    
    def run_script(self) -> None:
        self.var_field_types =\
        self.utils.collect_var_data(['field_type'])
        self.loop_timepoints()


    def loop_timepoints(self) -> None:
        """
        Loops through each timepoint 
        for each network and reads the 
        corresponding combined dataframe.
        """

        self.timepoint_list = \
        self.utils.create_timepoint_list()
        for network in ['PRONET','PRESCIENT']:
            for timepoint in self.timepoint_list:
                combined_df = pd.read_csv((f'{self.combined_df_folder}'
                f'combined-{network}-{timepoint}-day1to1.csv'),\
                keep_default_na=False) 
                similar_columns_df = self.find_matching_variables(combined_df)
                print(similar_columns_df)
                similar_columns_df.to_csv('sim_col_test.csv')


    def find_matching_variables(
        self,df : pd.DataFrame,
        threshold : float = 0.95
    ) -> pd.DataFrame:
        """
        Finds combinations 
        of two variables that are
        typically equal to each other.

        Parameters
        ---------------
        df : pd.DataFrame
            current dataframe being checked
        threshold : float
            How often the two variables 
            have to be equal. 1 is equal
            to 100% of the time.
        """
        similar_columns = []
        df = df.loc[:, df.apply(lambda col: (col != "").sum()) > 30]
        df = df.loc[:, [col for col in df.columns if col not in self.all_forms]]
        df = df.loc[:, [col for col in df.columns if '__' not in col]]
        for col1, col2 in combinations(df.columns, 2):
            try:
                equality_percentage = (df[col1].astype(float) == df[col2].astype(float)).mean()
            except Exception:
                equality_percentage = (df[col1].astype(str) == df[col2].astype(str)).mean()
            if equality_percentage > threshold:
                print(equality_percentage)
                print(col1)
                print(col2)
                similar_columns.append((col1, col2, equality_percentage))
        similar_columns_df = pd.DataFrame(similar_columns,\
        columns=['Column 1', 'Column 2', 'Equality Percentage'])
        
        return similar_columns_df



if __name__=='__main__':
    VariableMismatches().run_script()
