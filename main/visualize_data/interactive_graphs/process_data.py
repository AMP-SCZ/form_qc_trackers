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
        self.stacked_line_data = []

    def __call__(self):
        for network in ['PRONET','PRESCIENT']:
            tracker_path = f'{self.tracker_abs_path}/{network}/combined/{network}_Output.xlsx'
            tracker_df = pd.read_excel(tracker_path, keep_default_na=False)
            self.create_stacked_line_graph(tracker_df, network) 
        stacked_df = pd.DataFrame(self.stacked_line_data)
        stacked_df.to_csv('stacked_data.csv',index = False)
        return stacked_df

    def create_stacked_line_graph(self, df, network):
        start_date = datetime(2025, 1, 15)
        end_date = datetime.today()
        current_date = start_date

        error_categories = {categ: {} for categ in ['outstanding']}

        while current_date <= end_date:
            cutoff_date = pd.to_datetime(current_date)

            outstanding = df[
                (pd.to_datetime(df['Date Resolved'], errors='coerce') > cutoff_date) |
                (df['Date Resolved'] == '')
            ]
            outstanding = outstanding[
                pd.to_datetime(outstanding['Date Flag Detected'], errors='coerce') <= cutoff_date
            ]

            date_key = cutoff_date.strftime('%Y-%m-%d')  
            counts = outstanding['Timepoint'].value_counts().to_dict()
            error_categories['outstanding'][date_key] = counts

            current_date += timedelta(days=1)

        for category, data in error_categories.items():
            df_plot = pd.DataFrame(data).T.fillna(0).astype(int).sort_index()

            if df_plot.empty:
                continue

            if isinstance(df_plot.index, pd.DatetimeIndex):
                df_plot.index = df_plot.index.strftime('%Y-%m-%d')
            else:
                df_plot.index = df_plot.index.astype(str)

            original_timepoints = df_plot.columns.tolist()
            labels = [tp.replace('screening', 'screen').replace('baseline', 'baseln')
                    for tp in original_timepoints]
            values = [df_plot[tp].to_numpy() for tp in original_timepoints]

            x = df_plot.index.tolist()
            for count in range(0,len(x)):
                for val_count in range(0,len(values)):
                    self.stacked_line_data.append(
                        {'group':network,'date': x[count], 
                        'measure1': values[val_count][count], 'category': labels[val_count]}
                    )

            
    def create_stacked_bar_graph(self):
        pass

            
