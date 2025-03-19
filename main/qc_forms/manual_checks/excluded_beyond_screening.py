import pandas as pd
import os 
import json
import sys
from datetime import datetime
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class ExcludedChecks():

    def __init__(self):
        self.utils = Utils()
        self.abs_path = '/data/predict1/collect_data/' 
        self.comb_csv_dir = ('/data/predict1/data_from_nda'
        '/formsdb/generated_outputs/combined/PROTECTED/')

        self.missing_code_list = [
        '-3','-9',-3,-9,-3.0,-9.0,'-3.0','-9.0',
        '1909-09-09','1903-03-03','1901-01-01',
        '-99',-99,-99.0,'-99.0',999,999.0,'999','999.0','NA']

        self.final_output = []
        self.all_dates = {}

        self.form_vars = self.utils.load_dependency_json('important_form_vars.json')

    def create_tp_list(self):
        """
        creates a list of all timepoints 
        except floating
        """
        output_per_timepoint = ['screening','baseline']
        for x in range(1,13):
            output_per_timepoint.append('month'+f'{x}')
        output_per_timepoint.extend(['month18','month24','conversion'])

        return output_per_timepoint

    def run_script(self):
        self.loop_dataframes()
        self.final_output_df = pd.DataFrame(self.final_output)
        self.final_output_df.to_csv('excluded_beyond_screen.csv', index = False)

    def loop_dataframes(self):
        """
        loops through each combined csv
        and collects the relevant data 
        for that timepoint
        """
        final_output_df = pd.DataFrame()
        tp_list = self.create_tp_list()
        for timepoint in tp_list:
            print(timepoint)
            for network in ['PRONET','PRESCIENT']:
                comb_csv_path = (f"{self.comb_csv_dir}"
                f"combined-{network}-{timepoint}-day1to1.csv")
                comb_df = pd.read_csv(comb_csv_path,
                keep_default_na = False)
                self.collect_all_dates(comb_df,timepoint)

        self.collect_most_recent_dates()

        for network in ['PRONET','PRESCIENT']:
            comb_csv_path = (f"{self.comb_csv_dir}"
                f"combined-{network}-screening-day1to1.csv")
            comb_df = pd.read_csv(comb_csv_path,
            keep_default_na = False)
            self.check_excluded(comb_df,network)

    def collect_most_recent_dates(self):
        self.most_rec_dates = {}
        for sub, forms in self.all_dates.items():
            self.most_rec_dates.setdefault(sub, {'date':'','form':''})
            for form, date in forms.items():
                if form =='inclusionexclusion_criteria_review_screening':
                    continue
                if self.utils.check_if_val_date_format(date):
                    if self.most_rec_dates[sub]['date'] == '':
                        self.most_rec_dates[sub]['date'] = date
                        self.most_rec_dates[sub]['form'] = form
                    else:
                        if datetime.strptime(date, '%Y-%m-%d') > datetime.strptime(
                        self.most_rec_dates[sub]['date'], '%Y-%m-%d'):
                            self.most_rec_dates[sub]['date'] = date
                            self.most_rec_dates[sub]['form'] = form
        print(self.most_rec_dates)
                            
    def collect_all_dates(self, df, tp):
        for row in df.itertuples():
            for form in self.form_vars.keys():
                date_var = self.form_vars[form]['interview_date_var']
                if (hasattr(row,date_var) and getattr(row,date_var)
                not in (self.missing_code_list+ [''])):
                    date_val = str(getattr(row,date_var)).split(' ')[0]
                    self.all_dates.setdefault(row.subjectid,{})
                    self.all_dates[row.subjectid][form+'_'+tp] = date_val

    def check_excluded(self,df,network):
        for row in df.itertuples():
            if (row.chrcrit_included in [0,0.0,'0','0.0']
            and row.visit_status_string not in 'screening'):
                self.final_output.append({
                'subject':row.subjectid,
                'curr_visit':row.visit_status_string,'network':network,
                'most_recent_date':self.most_rec_dates[row.subjectid]})

if __name__ == '__main__':
    ExcludedChecks().run_script()