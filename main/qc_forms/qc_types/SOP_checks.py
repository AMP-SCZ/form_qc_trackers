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
        visit = row.visit_status
        visit = visit.replace('_','')
        cohort = self.subject_info[row.subjectid]['cohort']
        if self.timepoint == visit and cohort.lower() != 'unknown':
            all_tp_forms = self.forms_per_tp[cohort.upper()][self.timepoint]
            form_vars = self.important_form_vars
            all_dates = {}
            for form in all_tp_forms:
                if form in form_vars.keys():
                    date_var = ''
                    if 'interview_date_var' in form_vars[form].keys():
                        date_var = form_vars[form]['interview_date_var']
                    elif 'entry_date_var' in form_vars[form].keys():
                        date_var = form_vars[form]['entry_date_var']
                    if (date_var != '' and hasattr(row, date_var) and
                    self.utils.check_if_val_date_format(getattr(row, date_var))):
                        all_dates[date_var] = getattr(row, date_var) 
        else:
            return
        
        if all_dates != {}:
            most_recent_date_var = self.utils.recent_date_from_dict(all_dates)
            days_until_next_tp = self.utils.time_to_next_visit(visit)
            print(days_until_next_tp)
            if days_until_next_tp != None and most_recent_date_var != '':
                days_since_form = self.utils.days_since_today(all_dates[most_recent_date_var])
                days_over_expected = days_until_next_tp - days_since_form
                print('-----------')
                print(days_over_expected)
                print(row.subjectid)
                withdrawn_status_list.append({'subjectid':row.subjectid,
                'days_until_expected_next_visit':days_over_expected,'current_timepoint':visit,
                'most_recent_date':all_dates[most_recent_date_var],
                'most_recent_date_var':most_recent_date_var, 'withdrawn_status':row.removed})
                df = pd.DataFrame(withdrawn_status_list)
                df.to_csv('withdrawn_status.csv',index = False)


        
