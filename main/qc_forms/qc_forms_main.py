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
class QCFormsMain():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.depen_path = self.config_info['paths']['dependencies_path']
        with open(f'{self.depen_path}converted_branching_logic.json','r') as file:
            self.conv_bl = json.load(file)

        self.output_path = self.config_info['paths']['output_path']
        if self.config_info['testing_enabled'] == "True":
            self.output_path += "testing/"

        self.final_output_list = []

        self.combined_flags_path = f'{self.output_path}combined_outputs/'

        self.form_check_info = {'cognition_csvs':{}}

        for filename in ['subject_info','general_check_vars',
        'important_form_vars','forms_per_timepoint',
        'converted_branching_logic','excluded_branching_logic_vars',
        'team_report_forms','grouped_variables','variables_added_later',
        'raw_csv_conversions', 'variable_ranges','earliest_latest_dates_per_tp']:
            self.form_check_info[filename] = self.utils.load_dependency_json(f"{filename}.json")

        for iq_type in ['wais','wasi']:
            for conv_type in ['iq_raw','fsiq']:
                self.form_check_info['cognition_csvs'][
                f'{conv_type}_conversion_{iq_type}'] = pd.read_csv(
                f'{self.depen_path}cognition/{conv_type}_conversion_{iq_type}.csv',
                keep_default_na = False) 

        self.auxiliary_files_new_tabs = ['']

        self.scid_subs = []
        self.scid_subs_df = []

    def run_script(self):
        self.move_previous_output()
        self.iterate_combined_dfs()

    def move_previous_output(self):
        for path in [f"{self.combined_flags_path}new_output",
        f"{self.combined_flags_path}old_output",self.combined_flags_path]:
            if not os.path.exists(path):
                os.makedirs(path) 
        new_path =  f'{self.combined_flags_path}new_output/combined_qc_flags.csv'
        old_path = new_path.replace('new_output','old_output')
        if os.path.exists(new_path):
            try:
                df = pd.read_csv(new_path, keep_default_na=False)
                df.to_csv(old_path, index = False)
            except pd.errors.EmptyDataError:
                return

    def iterate_combined_dfs(self):
        #TODO: split checks by ones that will only be checked 
        # if a form in compl and no
        # t missing and ones 
        # that will be checked regardless
        final_output=[]
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        for network in ['PRESCIENT']:
            multi_tp_path = f"{self.depen_path}multi_tp_{network}_combined.csv"
            multi_tp_df = pd.read_csv(multi_tp_path,
            keep_default_na = False)
            for row in multi_tp_df.itertuples():
                multi_tp_vars = multi_tp_df.columns
                multi_tp_checks = MultiTPChecks(row,
                'multiple_timepoints', network, 
                self.form_check_info,multi_tp_vars)
                final_output.extend(multi_tp_checks())
            for tp in tp_list:
                combined_df = pd.read_csv(
                f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv',
                keep_default_na = False)
                #combined_df = combined_df.iloc[80:120]
                #combined_df = combined_df.sample(n=20)
                #combined_df = combined_df.sample(n=100, random_state=42)
                for row in combined_df.itertuples(): 
                    #print(tp)
                    print(row.Index)
                    #TODO: Add tracker for all subjects not existing here 
                    if (row.subjectid not
                    in self.form_check_info['subject_info']):
                        continue
                    #print(row.Index)
                    """gen_checks = GeneralChecks(row, tp,
                    network, self.form_check_info)
                    fluid_checks = FluidChecks(row, tp,
                    network, self.form_check_info)
                    clinical_checks = ClinicalChecksMain(row,
                    tp, network, self.form_check_info)
                    cognition_checks = CognitionChecks(row,
                    tp, network, self.form_check_info)
                    sop_checks = SOPChecks(row,
                    tp, network, self.form_check_info)
                    final_output.extend(gen_checks())
                    final_output.extend(fluid_checks())
                    final_output.extend(clinical_checks())
                    final_output.extend(sop_checks())
                    final_output.extend(cognition_checks())"""
                if len(final_output) > 0:
                    combined_output_df = pd.DataFrame(final_output)
                    if combined_output_df.shape[0] > 1000000:
                        print(f"output rows is {combined_output_df.shape[0]}")
                        sys.exit()
                    combined_flags_path = f'{self.output_path}combined_outputs'
                    os.makedirs(combined_flags_path,exist_ok=True)  # Creates the folder and any necessary parent directories
                    new_out_path =f'{combined_flags_path}/new_output/'
                    os.makedirs(new_out_path,exist_ok=True)
                    try:
                        print(combined_output_df.shape[0])
                        print(f'{new_out_path}combined_qc_flags.csv')
                        combined_output_df.to_csv(f'{new_out_path}combined_qc_flags.csv', index=False)
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        sys.exit()
