import pandas as pd

import sys
import os
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
from utils.utils import Utils
import json

class DuplicateSearcher():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.output_path = self.config_info['paths']['output_path']
        self.forms_per_tp = self.utils.load_dependency_json(f"forms_per_timepoint.json")
        self.sub_info = self.utils.load_dependency_json(f"subject_info.json")
        self.grouped_vars = self.utils.load_dependency_json(f"grouped_variables.json")
        self.form_per_var = self.grouped_vars["var_forms"]
        
        self.final_output = []

    def run_script(self):
        self.loop_csvs() 

    def loop_csvs(self):
        tp_list = self.utils.create_timepoint_list()
        for network in ['PRONET','PRESCIENT']:
           for tp in tp_list:
                print(tp)
                combined_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}'
                f'_{network}-day1to1.csv'),
                keep_default_na = False)
                self.check_forms(combined_df, tp, network)

    def check_forms(self, df, tp, network):
        all_cols = list(df.columns)
        for row in df.itertuples():
            subject = getattr(row, "subjectid")
            if subject not in self.sub_info.keys():
                continue
            if "cohort" not in self.sub_info[subject].keys():
                continue
            cohort = self.sub_info[subject]["cohort"]
            if cohort not in ["CHR","HC"]:
                continue
            curr_tp_forms = self.forms_per_tp[cohort][tp]
            for var in all_cols:
                if var not in self.form_per_var.keys():
                    continue
                var_form = self.form_per_var[var]
                if var_form in curr_tp_forms:
                    continue
                else:
                    if hasattr(row, var):
                        var_val = getattr(row, var)
                        if str(var_val) not in self.utils.missing_code_list + (['','nan']):
                            self.final_output.append({'subject':subject,'cohort':cohort,
                            'timepoint':tp,'network':network,
                            'form':var_form, 'var': var, 'var_value': var_val})

        df = pd.DataFrame(self.final_output)
        df.to_csv(f'{self.output_path}forms_wrong_tp.csv',
        index = False)

if __name__ == '__main__':
    DuplicateSearcher().run_script()