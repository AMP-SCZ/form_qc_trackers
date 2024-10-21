import pandas as pd
import os
import sys
import json
import unittest

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-4])
sys.path.insert(1, parent_dir)
print(parent_dir)
from main.process_variables.define_important_variables import DefineImportantVariables
from main.utils.utils import Utils
from logging_config import logger  

class TestDefineVariables(unittest.TestCase):
    def __init__(self):
        self.utils = Utils()
        data_dictionary_df = self.utils.read_data_dictionary()
        self.important_form_vars = DefineImportantVariables(data_dictionary_df)
        self.data_dictionary_df = data_dictionary_df

        self.forms_without_missing_variables = ['revised_status_form',
        'scid5_psychosis_mood_substance_abuse','status_form',
        'current_pharmaceutical_treatment_floating_med_2650',
        'family_interview_for_genetic_studies_figs',
        'current_pharmaceutical_treatment_floating_med_125',
        'past_pharmaceutical_treatment', 'adverse_events','missing_data',
        'enrollment_note','conversion_form', 'resource_use_log',
        'gcp_cbc_with_differential', 'gcp_current_health_status']

        self.forms_without_interview_dates = ['']

    def run_script(self):
        self.test_assign_variables_to_forms()

    def test_assign_variables_to_forms(self):
        all_unique_vars = self.important_form_vars.collect_important_vars() 
        all_forms = list(set(self.data_dictionary_df['Form Name'].tolist()))
        important_form_vars = self.important_form_vars.assign_variables_to_forms(all_unique_vars)

        for form, var_types in important_form_vars.items():
            for var_type, var_value in var_types.items():
                try:
                    self.assertNotEqual(var_value, '',
                    f"Value for {form}'s {var_type} is an empty string")

                except AssertionError as e:
                    logger.error(f"Blank value found for {form}'s {var_type}: {e}")

if __name__ == '__main__':
    TestDefineVariables().run_script()

