import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils import Utils
class DiscoverErrors():
    def __init__(self):
        self.absolute_path = '/PHShome/ob001/anaconda3/new_forms_qc/QC/'
        self.combined_df_folder = '/data/predict1/data_from_nda/formqc/'

        self.utils = Utils()

        self.data_dictionary_df = pd.read_csv(\
        f'{self.absolute_path}data_dictionaries/CloneOfYaleRealRecords_DataDictionary_2023-05-19.csv',\
        encoding = 'latin-1',keep_default_na=False)

        self.dtype_count = {}
        

    def run_script(self):
        self.loop_timepoints()


    def loop_timepoints(self):
        self.timepoint_list = \
        self.utils.create_timepoint_list()
        for network in ['PRONET','PRESCIENT']:
            for timepoint in self.timepoint_list:
                self.combined_df = pd.read_csv((f'{self.combined_df_folder}'
                    f'combined-{network}-{timepoint}-day1to1.csv'),\
                    keep_default_na=False) 
                self.discover_numerical_variables(self.combined_df)

    def discover_numerical_variables(self,df):
        for row in df.itertuples():
            for col in df.columns:
                self.dtype_count.setdefault(\
                    col,{'number':0,'nan':0})
                if getattr(row,col) not in\
                (self.utils.missing_code_list + ['']):
                    if self.utils.check_if_number(\
                    getattr(row,col)) == True:
                        self.dtype_count[col]['number'] +=1
                    else:
                        self.dtype_count[col]['nan'] +=1 

        self.create_numerical_var_list(self.dtype_count)

    def create_numerical_var_list(self,var_dict):
        num_var_list = []
        for var, values in var_dict.items():
            if values['number'] > 0 or\
            values['nan'] > 0:
                if values['number'] / (values['number'] + values['nan']) > 0.9:
                    if values['number'] / (values['number'] + values['nan']) < 1:
                        print(values['number'] / (values['number'] + values['nan']) )
                        print(var)
                    num_var_list.append(var)

                
                           


if __name__ == '__main__':
    DiscoverErrors().run_script()
