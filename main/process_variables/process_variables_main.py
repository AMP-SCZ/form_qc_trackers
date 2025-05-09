import pandas as pd
import os
import sys
import json

#parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
#sys.path.insert(1, parent_dir)

from main.utils.utils import Utils
from main.process_variables.define_important_variables import DefineEssentialFormVars, CollectMiscVariables
from main.process_variables.transform_branching_logic import TransformBranchingLogic
from main.process_variables.organize_reports import OrganizeReports
from main.process_variables.collect_subject_info import CollectSubjectInfo

from main.process_variables.analyze_identifier_effects import AnalyzeIdentifiers
from main.process_variables.collect_ra_subjects import RaSubjects

from main.process_variables.collect_multi_timepoint_data import MultiTPDataCollector

from main.process_variables.collect_raw_csv_info import RawCSVCollector

from main.process_variables.define_ranges import RangeDefiner


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
        grouped_variables = CollectMiscVariables(data_dict_df)
        self.utils.save_dependency_json(grouped_variables(), 'grouped_variables.json')
        important_form_vars = DefineEssentialFormVars(data_dict_df)
        self.identifier_effects = AnalyzeIdentifiers()
        self.identifier_effects.run_script()
        self.utils.save_dependency_json(important_form_vars(),
        'important_form_vars.json')
        transform_bl = TransformBranchingLogic(data_dict_df)
        converted_branching_logic = transform_bl()
        self.utils.save_dictionary_as_csv(converted_branching_logic,
        f"{self.config_info['paths']['dependencies_path']}converted_branching_logic.csv")

        self.utils.save_dependency_json(converted_branching_logic,
        'converted_branching_logic.json')
        subject_info = CollectSubjectInfo()

        self.utils.save_dependency_json(subject_info(),
        'subject_info.json')

        organize_reports = OrganizeReports()
        organize_reports.run_script()

        ra_subs = RaSubjects()
        self.utils.save_dependency_json(ra_subs(), 'melbourne_ra_subs.json')

        raw_csv_conversions = RawCSVCollector()
        self.utils.save_dependency_json(raw_csv_conversions(), 'raw_csv_conversions.json')

        var_ranges = RangeDefiner()
        self.utils.save_dependency_json(var_ranges(), 'variable_ranges.json')

        # must be called last as it uses dependencies 
        # from preceding classes
        self.multi_tp_data = MultiTPDataCollector()
        


        