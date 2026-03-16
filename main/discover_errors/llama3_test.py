import requests

import pandas as pd
import os
import sys
import json
import traceback
import random
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
from utils.utils import Utils
from qc_forms.qc_types.general_checks import GeneralChecks
from qc_forms.qc_types.fluid_checks import FluidChecks
from qc_forms.qc_types.clinical_checks.clinical_checks_main import ClinicalChecksMain
from qc_forms.qc_types.cognition_checks import CognitionChecks
from qc_forms.qc_types.SOP_checks import SOPChecks
from qc_forms.qc_types.multi_tp_checks import MultiTPChecks

class PIIGenerator():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.depen_path = self.config_info['paths']['dependencies_path']
        self.output_path = self.config_info['paths']['output_path']
        if self.config_info['testing_enabled'] == "True":
            self.output_path += "testing/"

        self.data_dict = self.utils.read_data_dictionary()

        self.filtered_df = self.data_dict[self.data_dict['Field Type'].isin(['text','notes'])]
        self.pii_vars = self.filtered_df['Variable / Field Name'].tolist()
        self.prompt_intro = (
            "Task: Determine whether the input text contains Personally Identifiable Information (PII). "
            "Return ONLY one of these exact outputs:\n"
            "1) This contains PII\n"
            "2) This does not contain PII\n\n"
            "Count as PII: personal names; email addresses; phone numbers; full or partial street addresses; "
            "postal codes when paired with other identifying info; medical record numbers (MRN); "
            "social security numbers; driver's license numbers; passport numbers; account numbers; "
            "IP addresses; full-face photo references; and any unique subject/participant identifiers "
            "(e.g., 'subjectid=12345', 'PID', 'record_id') if they could identify a person.\n\n"
            "Do NOT count as PII: empty/placeholder values (e.g., '', 'N/A', 'NA', 'nan', 'none'); "
            "dates of any kind (including DOB) should be treated as NOT PII for this task; "
            "general locations (country/state/large city) unless it is a specific address or facility+unit "
            "that could identify someone.\n\n"
            "Input text:\n"
        )
        self.final_output_list = []

    def run_script(self):
        #self.loop_csvs()
        self.check_data_dictionary_rows()


    def loop_csvs(self):
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        for network in ['PRONET','PRESCIENT']:
            for tp in tp_list:
                combined_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}_{network}-day1to1.csv'),
                keep_default_na = False, on_bad_lines='skip')
                self.check_df_for_pii(combined_df, network, tp)
    
    def check_df_for_pii(self, df, network, tp):
        print(tp)
        print(network)
        for row in df.itertuples():
            print(row.Index)
            for var in self.pii_vars:
                if hasattr(row, var):
                    val = str(getattr(row, var))
                    if len(val) < 3: 
                        continue
                    resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "llama3.3:latest", "prompt": self.prompt_intro + val, "stream": False},
                    timeout=300,
                    )
                    data = resp.json()
                    response = data["response"]
                    if "This contains PII" in response:
                        print(var)
                        print(val)
                        print(response)
                        self.final_output_list.append({'Subject': row.subjectid,
                        "network": network,"timepoint":tp,"variable": var,
                        "variable_value":val,"llama_response":response})
            output_df = pd.DataFrame(self.final_output_list)
            output_df.to_csv(f'{self.output_path}potential_pii_detected.csv',index = False)

    def check_data_dictionary_rows(self):
        output_list = []
        df = self.data_dict
        df.columns = df.columns.str.replace(" ", "_")
        prompt_intro = ("I am going to send you"
        " a row from a redcap data dictionary. This"
        " is for a large-scale, multisite, longitudinal study on psychosis."
        " Can you inspect this row for any potential issues. If there are issues"
        " in it, please include 'ISSUES FOUND IN ROW', in your response."
        " If there are none, make sure that string is not present. Do not worry about empty values."
        " Mainly focus on potential typos or mistakes in branching logic or equations. "
        "Then, explicitly state what the issues are and how to fix them. Here is the row: ")
        for row in df.to_numpy():
            row_val = dict(zip(df.columns, row))
            var = row_val['Variable_/_Field_Name']
            prompt = prompt_intro + str(row_val)
            resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.3:latest", "prompt": prompt, "stream": False},
            timeout=300,
            )
            data = resp.json()
            response = data["response"]
            print(response)
            if "ISSUES FOUND IN ROW" in response:
                output_list.append({'variable':var,'response':response})
                output_df = pd.DataFrame(output_list)
                output_df.to_csv(
                f'{self.output_path}data_dictionary_issues.csv', 
                index = False)

if __name__ == '__main__':
    PIIGenerator().run_script()