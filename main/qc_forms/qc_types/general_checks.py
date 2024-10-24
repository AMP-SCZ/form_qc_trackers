import pandas as pd

import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck

class GeneralChecks(FormCheck):
    
    def __init__(self, combined_df, timepoint, network):
        super().__init__(combined_df, timepoint, network)

        self.test_val = 0
        
        self.loop_dataframe()

    def __call__(self):

        return self.test_val

    def call_checks(self, row):
        self.row = row
        self.test_val +=1 
        self.test_check(row)
        

    @FormCheck.filter_qc_check(filtered_forms=['psychs_p1p8'], 
    all_vars=['chrfigs_sibling6_dmiss2'], 
    changed_output_vals={"Reports": ["Main Report"]})
    def test_check(self, row):
        if "1" not in row.subjectid:
            return "1 not in subject"
        return "" 
        



