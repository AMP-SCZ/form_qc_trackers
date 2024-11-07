import pandas as pd
import os
import sys
import json
import dropbox
#parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
#sys.path.insert(1, parent_dir)

from main.utils.utils import Utils
from main.generate_reports.create_trackers import CreateTrackers
from main.generate_reports.calculate_resolved_errors import CalculateResolvedErrors

class GenerateReports():

    def __init__(self):
        self.utils = Utils()
        self.calc_resolved = CalculateResolvedErrors()
        self.create_trackers = CreateTrackers()

    def run_script(self):
        self.calc_resolved.run_script()
        self.create_trackers.run_script()
   