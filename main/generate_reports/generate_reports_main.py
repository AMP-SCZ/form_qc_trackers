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
from utils.utils import Utils

class GenerateReports():

    def __init__(self):
        self.utils = Utils()

        with open(f'{self.utils.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.output_path = self.config_info['paths']['output_path']
        self.dependencies_path = self.config_info['paths']['dependencies_path']
        col_names_json = 'current_col_names.json'
        
        # any new columns added have to 
        self.formatted_column_names = {
            "PRONET" : {"combined": {
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
            "site_comments": "Site Comments",
            "priority_item" : "Priority Item"}},

            "PRESCIENT" :{"combined":{
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
            "site_comments": "Site Comments",
            "priority_item" : "Priority Item"}},
        }

        self.formatted_column_names['PRONET']['sites'] = self.formatted_column_names['PRONET']['combined']
        self.formatted_column_names['PRESCIENT']['sites'] = self.formatted_column_names['PRESCIENT']['combined']

        if not os.path.exists(col_names_json):
            self.utils.save_dependency_json(self.formatted_column_names, col_names_json)

        old_col_names = self.utils.load_dependency_json(col_names_json)

        if old_col_names != self.formatted_column_names:
            dbx_col_names = old_col_names
        else:
            dbx_col_names = self.formatted_column_names
        
        """self.formatted_column_names['sites'] = {'PRONET' : {}, 'PRESCIENT' : {}}
        for network in ['PRONET','PRESCIENT']:
            for orig, trans in self.formatted_column_names[network].items():
                if orig == 'comments':
                    self.formatted_column_names['sites'][network]['site_comments'] = 

                self.formatted_column_names['sites'][network][orig] = trans

            self.formatted_column_names['sites'][network] = self.formatted_column_names[network]
            self.formatted_column_names['sites'][network][] """

        self.utils = Utils()
        self.calc_resolved = CalculateResolvedErrors(dbx_col_names)
        self.create_trackers = CreateTrackers(self.formatted_column_names)

    def run_script(self):
        self.calc_resolved.run_script()
        self.create_trackers.run_script()
   