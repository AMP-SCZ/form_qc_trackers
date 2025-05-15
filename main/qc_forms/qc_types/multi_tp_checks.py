import pandas as pd
import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-4])
sys.path.insert(1, parent_dir)
from utils.utils import Utils
from qc_forms.form_check import FormCheck
from qc_forms.qc_types.clinical_checks.scid_checks import ScidChecks
from datetime import datetime,timedelta

class MultiTPChecks(FormCheck):    
    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network, form_check_info)
               
    def __call__(self):
        return self.final_output_list

    def call_checks(self, row):
        pass
