import pandas as pd
import os
import sys
import json
import random
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
from utils.utils import Utils
from qc_forms.qc_types.general_checks import GeneralChecks
from qc_forms.qc_types.fluid_checks import FluidChecks
from qc_forms.qc_types.clinical_checks.clinical_checks_main import ClinicalChecksMain
from qc_forms.qc_types.SOP_checks import SOPChecks
from qc_forms.qc_types.multi_tp_checks import MultiTPChecks
from qc_forms.qc_types.cognition_checks import CognitionChecks
from logging_config import logger

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

        self.combined_flags_path = f'{self.output_path}combined_outputs/'

        self.form_check_info = {}
        self.cognition_csvs = {}

        for filename in ['subject_info','general_check_vars',
        'important_form_vars','forms_per_timepoint',
        'converted_branching_logic','excluded_branching_logic_vars',
        'team_report_forms','grouped_variables','variables_added_later',
        'raw_csv_conversions', 'variable_ranges','earliest_latest_dates_per_tp']:
            self.form_check_info[filename] = self.utils.load_dependency_json(f"{filename}.json")

        for filename in ['fsiq_conversion_wais','fsiq_conversion_wasi',
        'iq_raw_conversion_wais','iq_raw_conversion_wasi']:
            self.cognition_csvs[filename] = pd.read_csv(f'{self.depen_path}cognition/{filename}.csv', 
            keep_default_na = False)

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
        main_output = []
        withdrawn_status_list = []

        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        for network in ['PRONET','PRESCIENT']:
            multi_tp_path = f"{self.depen_path}multi_tp_{network}_combined.csv"
            multi_tp_df = pd.read_csv(multi_tp_path,
            keep_default_na = False)
            for row in multi_tp_df.itertuples():
                multi_tp_checks = MultiTPChecks(row,
                'multiple_timepoints', network, 
                self.form_check_info)
                main_output.extend(multi_tp_checks())
            for tp in tp_list:
                print(tp)
                print('-------')
                combined_df = pd.read_csv(
                f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv',
                keep_default_na = False)
                #combined_df = combined_df.iloc[80:120]
                #combined_df = combined_df.sample(n=20)
                #combined_df = combined_df.sample(n=100, random_state=42)
                for row in combined_df.itertuples(): 
                    #print(tp)
                    #print(row.Index)
                    #TODO: Add tracker for all subjects not existing here 
                    if (row.subjectid not
                    in self.form_check_info['subject_info']):
                        continue
                    #print(row.Index)
                    gen_checks = GeneralChecks(row, tp,
                    network, self.form_check_info)
                    fluid_checks = FluidChecks(row, tp,
                    network, self.form_check_info)
                    clinical_checks = ClinicalChecksMain(row,
                    tp, network, self.form_check_info)
                    sop_checks = SOPChecks(row,
                    tp, network, self.form_check_info)
                    cognition_checks = CognitionChecks(row,
                    tp, network, self.form_check_info)
                    main_output.extend(gen_checks())
                    main_output.extend(fluid_checks())
                    main_output.extend(clinical_checks())
                    main_output.extend(sop_checks()[0])
                    withdrawn_status_list.extend(sop_checks()[1])
                    main_output.extend(cognition_checks())
                combined_output_df = pd.DataFrame(main_output)
                if combined_output_df.shape[0] > 2000000:
                    logger.error(f"output rows is {combined_output_df.shape[0]}")
                    break
                combined_flags_path = f'{self.output_path}combined_outputs'
                if not os.path.exists(combined_flags_path):
                    os.makedirs(combined_flags_path)  # Creates the folder and any necessary parent directories
                combined_output_df.to_csv(
                f'{combined_flags_path}/new_output/combined_qc_flags.csv',
                index = False)
                withdrawn_df = pd.DataFrame(withdrawn_status_list)
                withdrawn_df.to_csv(f'{self.utils.config_info["paths"]["output_path"]}withdrawn_status.csv',
                index = False)


