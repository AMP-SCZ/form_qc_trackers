import pandas as pd
import os
import sys
import json
import unittest

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-4])
sys.path.insert(1, parent_dir)
from main.process_variables.define_important_variables import DefineEssentialFormVars
from main.utils.utils import Utils
from logging_config import logger  

class TestDefineVariables(unittest.TestCase):
    def __init__(self):
        self.utils = Utils()
        data_dictionary_df = self.utils.read_data_dictionary()
        self.important_form_vars = DefineEssentialFormVars(data_dictionary_df)
        self.data_dictionary_df = data_dictionary_df

        # NOTE: Current pharm treatment 125 floating forms has two entry dates
        # One entry date is for date of entry at screening and the other is for the most recent one
        # Current pharm treatment 125 only has date of last data entry
        # They are being included in this list because they do not function like the other entry dates

        self.forms_without_vars = {'missing_var':['revised_status_form', # forms without missing variables
        'scid5_psychosis_mood_substance_abuse','status_form',
        'current_pharmaceutical_treatment_floating_med_2650',
        'family_interview_for_genetic_studies_figs',
        'current_pharmaceutical_treatment_floating_med_125',
        'past_pharmaceutical_treatment', 'adverse_events','missing_data',
        'enrollment_note','conversion_form', 'resource_use_log',
        'gcp_cbc_with_differential', 'gcp_current_health_status'],

        'interview_date_var':['psychosocial_treatment_form', # forms without interview dates
        'revised_status_form','psychs_p9ac32','psychs_p9ac32_fu_hc','status_form','psychs_p9ac32_fu',
        'adverse_events','health_conditions_medical_historypsychiatric_histo',
        'current_pharmaceutical_treatment_floating_med_2650',
        'current_pharmaceutical_treatment_floating_med_125','resource_use_log','coenrollment_form'],
        
        'entry_date_var':['psychosocial_treatment_form', # forms without entry date variables
        'psychs_p9ac32','psychs_p9ac32_fu_hc','status_form','psychs_p9ac32_fu',
        'adverse_events','health_conditions_medical_historypsychiatric_histo',
        'enrollment_note','current_pharmaceutical_treatment_floating_med_2650',
        'current_pharmaceutical_treatment_floating_med_125','resource_use_log',
        'conversion_form','coenrollment_form','missing_data','informed_consent_run_sheet',
        'gcp_current_health_status','lifetime_ap_exposure_screen','gcp_cbc_with_differential',
        'inclusionexclusion_criteria_review','informed_reconsent','mri_run_sheet']} 

    def run_script(self):
        self.test_assign_variables_to_forms()

    def test_assign_variables_to_forms(self):
        all_unique_vars = self.important_form_vars.collect_important_vars() 
        important_form_vars = self.important_form_vars.assign_variables_to_forms(all_unique_vars)

        for form, var_types in important_form_vars.items():
            for var_type, var_value in var_types.items():
                try:
                    if (var_type in self.forms_without_vars.keys() and
                    form not in self.forms_without_vars[var_type]):  
                        self.assertNotEqual(var_value, '',
                        f"Value for {form}'s {var_type} is an empty string")
                except AssertionError as e:
                    logger.error(f"Blank value found for {form}'s {var_type}: {e}")
                try:
                    if (var_type in self.forms_without_vars.keys() and
                    form in self.forms_without_vars[var_type]):
                        self.assertTrue(var_value == '',
                        f"Value for {form}'s {var_type} is an empty string")
                except AssertionError as e:
                    logger.error(f"{form}'s {var_type} is no longer blank: {e}")


if __name__ == '__main__':
    TestDefineVariables().run_script()

