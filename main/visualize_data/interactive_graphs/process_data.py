import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import os
import json 


class ProcessData():

    def __init__(self):
        self.qc_output_path = '/data/predict1/formqc/output/'
        self.dependencies_path = '/PHShome/ob001/anaconda3/refactored_qc/dependencies/'
        self.tracker_abs_path = '/PHShome/ob001/anaconda3/new_forms_qc/QC/site_outputs/'
        self.absolute_path = '/PHShome/ob001/anaconda3/new_forms_qc/QC/'
        self.site_errors = {}
        self.stacked_line_data = {}

    def run_script(self):
        """
        Loops through both networks
        and calls functions to visualize
        the number of errors per site.
        """
        
        for network in ['PRESCIENT','PRONET']:
            tracker_path = f'{self.tracker_abs_path}/{network}/combined/{network}_Output.xlsx'
            tracker_df = pd.read_excel(tracker_path, keep_default_na=False)
            self.create_stacked_line_graph(tracker_df, network)        

        """with open((f"{self.dependencies_path}"
        "stacked_line_data.json"), "w") as json_file:
            json.dump(self.stacked_line_data, json_file)"""

    def create_stacked_line_graph(self, df, network):
        start_date = datetime(2025, 7, 15)
        end_date = datetime.today()
        current_date = start_date

        error_categories = {categ: {} for categ in [ 'outstanding']}

        while current_date <= end_date:
            cutoff_date = pd.to_datetime(current_date)
            next_day = cutoff_date + timedelta(days=1)

            outstanding = df[
                (pd.to_datetime(df['Date Resolved'], errors='coerce') > cutoff_date) | 
                (df['Date Resolved'] == '')
            ]
            outstanding = outstanding[
                pd.to_datetime(outstanding['Date Flag Detected'],
                errors='coerce') <= cutoff_date
            ]
            for name, cat_df in zip(['outstanding'],
                                    [outstanding]):
                counts = cat_df['Timepoint'].value_counts().to_dict()
                error_categories[name][cutoff_date] = counts

            current_date += timedelta(days=1)

        for category, data in error_categories.items():
            df_plot = pd.DataFrame(data).T.fillna(0).astype(int).sort_index()
            if df_plot.empty:
                continue
            timepoints = df_plot.columns.tolist()
            timepoints = [tp.replace('screening','screen').replace(
            'baseline','baseln') for tp in timepoints]
            timepoints = [tp for tp in timepoints if tp in df_plot.columns]
            values = [df_plot[tp].values for tp in timepoints]

            x = df_plot.index
            y = values
            labels = timepoints 
            self.stacked_line_data.setdefault(
            network, {'x_axis':x,
            'y_axis':y,'labels':labels})
            print(self.stacked_line_data)
                

if __name__ == '__main__':
    ProcessData().run_script()