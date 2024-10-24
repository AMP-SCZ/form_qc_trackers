import pandas as pd

import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck

class GeneralChecks(FormCheck):
    
    def __init__(self, row, timepoint, network):
        super().__init__(timepoint, network)

        self.test_val = 0

        self.call_checks(row)
        

    def __call__(self):

        return self.final_output_list

    def call_checks(self, row):
        self.row = row
        self.test_val +=1 
        self.test_check(row, ['psychs_p1p8'],
        ['chrfigs_sibling6_dmiss2'],
        {"Reports": ["Main Report"]},False, False )
        

    @FormCheck.filter_qc_check
    def test_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        if "1" not in row.subjectid:
            return "1 not in subject"
        return "" 
        



