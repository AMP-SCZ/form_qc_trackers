import pandas as pd
import pandas as pd
import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)
from main.utils.utils import Utils
from logging_config import logger  


class TestTransition():

    def __init__(self):
        self.utils = Utils()
        self.orig_combined_csv_path = '/data/predict1/data_from_nda/formqc/'
        self.new_combined_csv_path = '/data/predict1/data_from_nda/formsdb/generated_outputs/combined/PROTECTED/'
        self.diff_output = []
        with open(f'{self.utils.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

    def run_script(self):
        self.compare_non_blank_changes()

    def compare_non_blank_changes(self):
        tp_list = self.utils.create_timepoint_list()
        for network in ['PRONET','PRESCIENT']:
            for tp in tp_list:
                print(tp)
                print(network)
                orig_path = f'{self.orig_combined_csv_path}combined-{network}-{tp}-day1to1.csv'
                orig_combined_df = pd.read_csv(orig_path, keep_default_na=False)
                new_path = f'{self.new_combined_csv_path}combined-{network}-{tp}-day1to1.csv'
                new_combined_df = pd.read_csv(new_path, keep_default_na=False)

                common_columns = orig_combined_df.columns.intersection(new_combined_df.columns)

                orig_combined_df = orig_combined_df[common_columns]
                new_combined_df = new_combined_df[common_columns]

                merged_df = orig_combined_df.merge(new_combined_df, on='subjectid', suffixes=('_df1', '_df2'))


                for row in merged_df.itertuples():
                    print(row.Index)
                    for col in orig_combined_df.columns:
                        if col not in ['subjectid','visit_status']:
                            orig_val= getattr(row, f"{col}_df1")
                            new_val = getattr(row, f"{col}_df2")
                            if str(orig_val) not in ['',str(new_val)]:
                                if self.utils.can_be_float(new_val) and self.utils.can_be_float(orig_val)\
                                and float(new_val) == float(orig_val):
                                    continue
                                self.diff_output.append({'network':network,
                                'timepoint':tp,'subject':row.subjectid,
                                'var_changed':col,'orig_val':getattr(row, f"{col}_df1"),
                                'new_val':getattr(row, f"{col}_df2")})
                output_df = pd.DataFrame(self.diff_output)   
                output_df.to_csv(
                f"{self.config_info['paths']['output_path']}csv_transition_qc.csv",
                index = False)                         


if __name__ == '__main__':
    TestTransition().run_script()
