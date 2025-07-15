import pandas as pd

import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)
from utils.utils import Utils
from qc_forms.form_check import FormCheck
import re
withdrawn_status_list = []

class SOPChecks(FormCheck):
    """
    QC Checks that compare 
    forms from different timepoints 
    using the dataframe that combines 
    each timepoint for specified variables
    """  

    def __init__(self,
        row, timepoint, network, form_check_info
    ):
        super().__init__(timepoint, network, form_check_info)
        self.test_val = 0
        self.call_checks(row)
        self.conversion_subs = self.raw_csv_converters.keys()
        
    def __call__(self):
        return self.final_output_list
    
    def call_checks(self, row):
        self.call_conversion_checks(row)
        #self.withdrawn_check(row)

    def call_conversion_checks(self, row):
        changed_output = {'reports': ['Main Report']}
        self.compare_raw_csv(row, ['conversion_form'],
        ['chrconv_interview_date'], changed_output)

    @FormCheck.standard_qc_check_filter 
    def compare_raw_csv(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=False, var_comps = {}
    ):
    
        """
        Finds subjects that are converted
        but did not fill out the conversion form.
        """
        if (row.subjectid in self.conversion_subs
        and getattr(row, 'chrconv_interview_date')
        in (self.utils.missing_code_list + [''])):
            return "Subject converted, but date not filled out in conversion form."
    
    def withdrawn_check(self, row):
        """
        Calculates days between most recent visit 
        (with a form that has a date 
        and is not marked missing)
        and when the next visit should be
        """
        if row.subjectid in self.subject_info.keys():
            cohort = self.subject_info[row.subjectid]['cohort']
            inclusion = self.subject_info[row.subjectid]['inclusion_status']
            if self.network == 'PRONET':
                screen_fail = self.subject_info[row.subjectid]['screenfail']
                completed_study = self.subject_info[row.subjectid]['completed_study']
            else:
                screen_fail = 'false'
                completed_study = 'false'
        else:
            return
        if (cohort.lower() == 'unknown' or 
        inclusion != 'included' or screen_fail == 'true'
        or completed_study == 'true'):
            return
        
        if row.subjectid in self.tp_date_ranges.keys():
            for tp, dates in self.tp_date_ranges[row.subjectid].items():
                if tp in ['floating','conversion']:
                    continue
                most_recent_date = dates['latest']
                visit = tp
            if visit != self.timepoint:
                return
            days_until_next_tp = self.utils.time_to_next_visit(visit,cohort)
            if days_until_next_tp != None:
                days_since_form = self.utils.days_since_today(most_recent_date)
                days_over_expected = days_until_next_tp - days_since_form
                if (any(row.subjectid == entry['subjectid']
                for entry in withdrawn_status_list)):
                    return
                withdrawn_status_list.append({'subjectid':row.subjectid,
                'network' : self.network,'days_until_expected_next_visit':days_over_expected,
                'current_timepoint':visit, 'most_recent_date':most_recent_date,
                'withdrawn_status':row.removed})
                df = pd.DataFrame(withdrawn_status_list)
                df.to_csv(f'{self.utils.config_info["paths"]["output_path"]}withdrawn_status.csv',
                index = False)

        return 

        


        
