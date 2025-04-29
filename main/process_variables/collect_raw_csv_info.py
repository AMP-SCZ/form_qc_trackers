import pandas as pd
import sys 
import os
import re

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class RawCSVCollector():

    def __init__(self):
        self.utils = Utils()
        self.conversion_dates = {}
        self.loop_folders()

    def __call__(self):
        return self.conversion_dates

    def loop_folders(self):
        for network in ['Prescient']:
            raw_csv_folder = f'/data/predict1/data_from_nda/{network}/PHOENIX/PROTECTED'
            for site in os.listdir(raw_csv_folder):
                if re.search(rf'{network}[a-zA-Z]', site):
                    sub_folder = f'{raw_csv_folder}/{site}/raw/'
                    for subject in os.listdir(sub_folder):
                        date_filepath = (f'{sub_folder}{subject}/surveys/'
                        f'{subject}_ClientStatus_AllDates.csv')
                        self.collect_consent_dates(subject, date_filepath)

    def collect_consent_dates(self, subject, filepath):
        if os.path.exists(filepath):
            df = pd.read_csv(filepath,
            keep_default_na = False)
            conv_date = str(df.loc[0,'Transition / Conversion'])
            if (conv_date != '' and self.utils.check_if_val_date_format(
            string=conv_date, date_format="%d/%m/%Y")):
                conv_date = self.convert_prescient_dates(conv_date)
                self.conversion_dates[subject] = conv_date
                
    def convert_prescient_dates(self, date_str):
        """
        converts prescient 
        dates to standard format

        Parameters 
        ---------------
        date_str : str
            input date 

        Returns 
        ---------------
        modified_date : 
        """
        
        splt_date = date_str.split('/')
        modified_date = (splt_date[2] + '-' +
        splt_date[1] + '-' + splt_date[0])

        return modified_date
