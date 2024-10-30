import pandas as pd
import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from define_important_variables import DefineEssentialFormVars, CollectMiscVariables
from transform_branching_logic import TransformBranchingLogic
from organize_reports import OrganizeReports
from collect_subject_info import CollectSubjectInfo

from analyze_identifier_effects import AnalyzeIdentifiers


class ProcessVariables():
    """
    Class to process and organize
    different variables in the data dictionary
    that will be used for the QC. This includes 
    collecting important variable names (interview dates,
    missing variables, etc), organizing variables into
    their respective reports,and grouping together 
    variables for different types of QC checks.
    """

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

    def run_script(self):
        data_dict_df = self.utils.read_data_dictionary()
        important_form_vars = DefineEssentialFormVars(data_dict_df)

        # needs to run before branching logic conversion
        self.identifier_effects = AnalyzeIdentifiers()
        self.identifier_effects.run_script()

        self.utils.save_dependency_json(important_form_vars(), 'important_form_vars.json')

        grouped_variables = CollectMiscVariables(data_dict_df)

        self.utils.save_dependency_json(grouped_variables(), 'var_info.json')

        transform_bl = TransformBranchingLogic(data_dict_df)
        converted_branching_logic = transform_bl()
        self.utils.save_dictionary_as_csv(converted_branching_logic,
        f"{self.config_info['paths']['dependencies_path']}converted_branching_logic.csv")

        self.utils.save_dependency_json(converted_branching_logic, 'converted_branching_logic.json')

        self.utils.save_dependency_json(grouped_variables(), 'grouped_variables.json')

        subject_info = CollectSubjectInfo()

        self.utils.save_dependency_json(subject_info(), 'subject_info.json')

        organize_reports = OrganizeReports()
        organize_reports.run_script()
        

if __name__ == '__main__':
    ProcessVariables().run_script()

        