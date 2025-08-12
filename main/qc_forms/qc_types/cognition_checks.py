import datetime
import pandas as pd
import sys
import os
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck

class CognitionChecks(FormCheck):
    """
    QC Checks related to cognitive forms
    """
    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network, form_check_info)
        self.scid_diagnosis_dict = self.utils.load_dependency_json(
        'scid_diagnosis_vars.json')
        self.assessment_translations = {1.0 : 'wasi', 2.0 : 'wais'}
        self.timepoint = timepoint
        self.call_checks(row)
        
    def call_checks(self, row):
        pass
        #self.call_cognition_checks(row)

    def __call__(self):
        return self.final_output_list

    def call_cognition_checks(self,row):
        reports = {'reports' : ['Main Report','Cognition Report']}
        """
        self.temp_cognition_fix(row, 
        ['iq_assessment_wasiii_wiscv_waisiv'], 
        ['chriq_fsiq'], reports)
        """
        forms =  ['iq_assessment_wasiii_wiscv_waisiv']
        reports = ['Main Report', 'Cognition Report']
        if self.timepoint in ["baseline","month24"]:
            if ('age' in self.subject_info[row.subjectid].keys() and
            'demographics_date' in self.subject_info[row.subjectid].keys()):
                self.age = self.subject_info[row.subjectid]['age']
                self.demo_date = self.subject_info[row.subjectid]['demographics_date']
                if all(self.utils.check_if_val_date_format(str(date_val)) for
                date_val in [self.demo_date, row.chriq_interview_date]):
                    iq_age_diff = self.utils.find_days_between(self.demo_date, row.chriq_interview_date)
                    if all(self.utils.can_be_float(age_val) for age_val in [iq_age_diff, self.age]):
                        iq_age = float(self.age) + (float(iq_age_diff) / 365)
                        if (hasattr(row, 'chriq_assessment') and 
                        row.chriq_assessment not in (self.utils.missing_code_list + ['']) and
                        self.utils.can_be_float(row.chriq_assessment) and
                        float(row.chriq_assessment) in [1.0,2.0]):
                            assessment = self.assessment_translations[float(row.chriq_assessment)]
                            if assessment == 'wasi':
                                self.fsiq_conversion_check(row, forms,
                                ['chriq_fsiq'], {"reports" : ['Main Report']},
                                bl_filtered_vars = [], filter_excl_vars = False,
                                assessment = assessment, iq_age = iq_age)
                            self.standardized_score_check(row, forms,
                            ['chriq_fsiq'], {"reports" : ['Main Report']},
                            bl_filtered_vars = [], filter_excl_vars = False,
                            assessment = assessment, iq_age = iq_age)

    @FormCheck.standard_qc_check_filter 
    def fsiq_conversion_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False, assessment = '', iq_age = ''
    ):   
        conversion_vars = {'wasi' : 'chriq_tscore_sum',
        'wais': 'chriq_scaled_sum'}
        var_to_convert = conversion_vars[assessment]
        if (self.utils.can_be_float(iq_age) and hasattr(row, var_to_convert)):
            redcap_fsiq = getattr(row, 'chriq_fsiq')
            redcap_raw_score = getattr(row, var_to_convert)
            if all(iq_var not in (self.utils.missing_code_list + [''])
            and self.utils.can_be_float(iq_var) for iq_var
            in [redcap_fsiq, redcap_raw_score]):
                fsiq_table = self.cognition_csvs[f'fsiq_conversion_{assessment}'] 
                for iq_row in fsiq_table.itertuples():
                    if (float(redcap_raw_score) == float(
                    getattr(iq_row, var_to_convert))):
                        python_fsiq = float(getattr(iq_row, 'chriq_fsiq'))
                        if python_fsiq != float(redcap_fsiq):
                            return (f"FSIQ Miscalculated" 
                            f" (Recorded as {redcap_fsiq}, but should be {python_fsiq})")
                        
    def standardized_score_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False, assessment = '', iq_age = ''
    ):
        conversion_vars = {'wasi' : 'chriq_tscore_sum',
        'wais': 'chriq_scaled_sum'}
        iq_col_names =  {'wasi' : 't_score',
        'wais': 'scaled_score'}
        reports = {'reports' : ['Main Report', 'Cognition Report']}
        standardized_score_table = self.cognition_csvs[f'iq_raw_conversion_{assessment}'] 
        unique_ranges = standardized_score_table.iloc[0].unique().tolist()
        range_to_use = ''
        iq_age_mos = int(float(iq_age) * 12)
        for range_str in unique_ranges:
            if iq_age_mos in self.utils.convert_range_to_list(str(range_str)):
                range_to_use = range_str
        if range_to_use != '':
            filtered_table = standardized_score_table.loc[:,
            standardized_score_table.iloc[0] == range_to_use]
            filtered_table = filtered_table.iloc[1:]
            filtered_table.columns = filtered_table.iloc[0]  # first row becomes column names
            score_dict = {'vocab':{'raw':'chriq_vocab_raw',
            'scaled':'chriq_scaled_vocab','col_name':'vc'},
            'matrix':{'raw':'chriq_matrix_raw',
            'scaled':'chriq_scaled_matrix','col_name':'mr'}}
            vocab_raw = getattr(row,'chriq_vocab_raw')
            matrix_raw = getattr(row, 'chriq_matrix_raw')
            for iq_row in filtered_table.itertuples():
                for test_type, scores in score_dict.items():
                    if assessment != 'wais':
                        continue
                    conversion_sheet_score = self.utils.convert_range_to_list(
                    getattr(iq_row, scores['col_name']))
                    redcap_score = getattr(row, scores['raw'])
                    if redcap_score == conversion_sheet_score:
                        redcap_scaled = getattr(row, scores['scaled'])
                        qc_scaled = getattr(iq_row, iq_col_names[assessment])
                        if (redcap_scaled != qc_scaled and redcap_scaled
                        not in (self.utils.missing_code_list + [''])):
                            error_message = (f"Check scaled conversion for {test_type} IQ score."
                            f" Recorded as {redcap_scaled}, but should potentially be"
                            f" {qc_scaled} (may be false flag due to age estimates)")
                            error_output = self.create_row_output(
                            row, filtered_forms, [scores['raw'], scores['scaled']],
                            error_message, reports)
                            self.final_output_list.append(error_output)

