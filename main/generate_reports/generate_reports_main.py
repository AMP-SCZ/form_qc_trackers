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
                
        self.formatted_column_names = {
            "PRONET" : {
            "subject":"Participant",
            "displayed_timepoint":"Timepoint",
            "displayed_form" : "Form",
            "flag_count" : "Flag Count",
            "error_message" : "Flags",
            "var_translations":"Translations",
            "time_since_last_detection": "Days Since Detected",
            "date_resolved":"Date Resolved",
            "manually_resolved": "Manually Resolved",
            "comments" : "Network Comments",
            "site_comments": "Site Comments"},

            "PRESCIENT" :{
            "subject":"Participant",
            "displayed_timepoint":"Timepoint",
            "displayed_form" : "Form",
            "flag_count" : "Flag Count",
            "error_message" : "Flags",
            "var_translations":"Translations",
            "time_since_last_detection": "Days Since Detected",
            "date_resolved":"Date Resolved",
            "manually_resolved": "Manually Resolved",
            "comments" : "Network Comments",
            "site_comments": "Site Comments"},
        }

        """self.formatted_column_names['sites'] = {'PRONET' : {}, 'PRESCIENT' : {}}
        for network in ['PRONET','PRESCIENT']:
            for orig, trans in self.formatted_column_names[network].items():
                if orig == 'comments':
                    self.formatted_column_names['sites'][network]['site_comments'] = 

                self.formatted_column_names['sites'][network][orig] = trans

            self.formatted_column_names['sites'][network] = self.formatted_column_names[network]
            self.formatted_column_names['sites'][network][] """


        self.utils = Utils()
        self.calc_resolved = CalculateResolvedErrors(self.formatted_column_names)
        self.create_trackers = CreateTrackers(self.formatted_column_names)

    def run_script(self):
        self.calc_resolved.run_script()
        self.create_trackers.run_script()
   