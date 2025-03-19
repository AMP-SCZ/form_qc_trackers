import pandas as pd
import os
from datetime import datetime
import json

class RaSubjects():

    def __init__(self):
        self.ra_output = {}
        self.prescient_raw_csv_path = \
        '/data/predict1/data_from_nda/Prescient/PHOENIX/PROTECTED/'
        self.all_dates_list = []
        self.ra_assignments = {}

    def __call__(self):
        ra_assignments = self.loop_csv_files()
        return ra_assignments

    def loop_csv_files(self):
        ra_assignments = {} 
        for site_folder in os.listdir(self.prescient_raw_csv_path):
            site_path = self.prescient_raw_csv_path + '/' + site_folder
            if site_folder.startswith('Prescient') and os.path.isdir(site_path):
                raw_path = site_path + '/raw'
                for subject in os.listdir(raw_path):
                    surveys_path = raw_path + '/' + subject + '/surveys'
                    for file in os.listdir(surveys_path):
                        client_file =  f'{subject}_ClientListWithDpaccID.csv'
                        if file == client_file:
                            df = pd.read_csv(surveys_path + '/' + client_file,\
                                            keep_default_na=False)
                            for row in df.itertuples():
                                if row.RAname != '':
                                    ra_assignments.setdefault(row.RAname, [])
                                    if row.subjectkey != '' and row.subjectkey \
                                    not in ra_assignments[row.RAname]:
                                        ra_assignments[row.RAname].append(row.subjectkey)
        return ra_assignments




if __name__ == '__main__':
    RaSubjects().loop_csv_files()