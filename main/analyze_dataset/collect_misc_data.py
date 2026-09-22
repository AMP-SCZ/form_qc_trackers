import pandas as pd
import os
import sys 
import json
import numpy as np

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
print(parent_dir)
from utils.utils import Utils

class CollectData():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.depend_path = self.config_info['paths']['dependencies_path']
        self.output_path = self.config_info['paths']['output_path']
        self.form_vars = self.utils.load_dependency_json('important_form_vars.json')
        self.all_results = []
        self.final_output = []
        self.forms_per_tp = self.utils.load_dependency_json('forms_per_timepoint.json')
        self.subject_info = self.utils.load_dependency_json('subject_info.json')
    
    def run_script(self):
        self.loop_timepoints()

    def loop_timepoints(self):
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        for network in ['PRONET']:
            for tp in tp_list:
                if tp not in ['screening','baseline']:
                    continue
                combined_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}_{network}-day1to1.csv'),
                keep_default_na = False)
                for row in combined_df.itertuples():
                    subject = row.subjectid
                    cohort = 'unknown'
                    if subject in self.subject_info.keys():
                        if (self.subject_info[subject]['cohort'].lower() 
                        in ['hc','chr']):
                            cohort = self.subject_info[subject]['cohort']
                    if cohort == 'unknown':
                        continue
                    for form, all_vars in self.form_vars.items():
                        if form not in self.forms_per_tp[cohort.upper()]:
                            continue
                        compl_var = all_vars['completion_var']
                        if row.recruitment_status == 'recruited':
                            if (hasattr(row, compl_var) and
                            getattr(row, compl_var) not in self.utils.all_dtype([2])):
                                self.final_output.append({'subject': row.subjectid,
                                'timepoint': tp, 'form':form})
            df = pd.DataFrame(self.final_output)
            df.to_csv('incomplete_forms.csv', index = False)
            print('saved')

if __name__ == '__main__':
    CollectData().run_script()
