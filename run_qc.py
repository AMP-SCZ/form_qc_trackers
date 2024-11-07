
import os
import pandas as pd

import pandas as pd

import os
import sys
import json
import random
#parent_dir = os.path.realpath(__file__)
#sys.path.insert(1, parent_dir)
#print(parent_dir)
from main.process_variables.process_variables_main import ProcessVariables
from main.qc_forms.qc_forms_main import QCFormsMain
from main.generate_reports.generate_reports_main import GenerateReports

"""
QC ORDER
1. move combined output from new to old output folder
2. rerun qc to generate new combined output
3. determine resolved errors in new output by comparing to old, 
then merge them with the appropriate errors marked resolved.
4. save merged df as current output. 
5. pull data (manually resolved and comments columns) from combined formatted 
dropbox outputs and add it to file in old output folder
6. loop through each network, report, site, and RA (for melbourne) to create
formatted outputs for each. for the sites only
include the main report (for melbourne, non team form report)
7. save all formatted outputs to folder and upload them to dropbox
"""

class RunQC():

    def __init__(self):
        self.process_vars = ProcessVariables()
        self.qc_forms = QCFormsMain()
        self.generate_reports = GenerateReports()
    
    def run_script(self):
        #self.process_vars.run_script()
        #self.qc_forms.run_script()
        self.generate_reports.run_script()
if __name__ == '__main__':
    RunQC().run_script()