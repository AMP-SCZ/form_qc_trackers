import sys
import os
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class VisualizeDataInflux():

    def __init__(self):
        self.data_change_path = '/PHShome/ob001/anaconda3/refactored_qc/output/daily_changes/'

        self.changes_list = []
        self.date_list = []

    def run_script(self):
        self.collect_lengths('PRESCIENT')
        self.create_line_graph()

    def collect_lengths(self, network : str):
        for file in os.listdir(self.data_change_path):
            fullpath = self.data_change_path + file
            if os.path.getsize(fullpath) <= 1:
                continue

            df = pd.read_csv(self.data_change_path + file,
            keep_default_na = False)
            df = df[df['network'] == network]
            change_count = df.shape[0]
            print(change_count)

            self.changes_list.append(change_count)
            date = file.split('_')[-1]
            date = date.replace('.csv','')
            self.date_list.append(date)
            print(date)

    def create_line_graph(self):
        dates = pd.to_datetime(self.date_list)
        print(dates)
        fig, ax = plt.subplots()
        ax.plot(dates, self.changes_list)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        plt.xticks(rotation=45, ha='right')
        plt.title('PRESCIENT Data Changes Over Time')
        plt.xlabel('Date')
        plt.ylabel('Variables Changed')
        plt.tight_layout()
        plt.savefig('data_changes_prescient.png',dpi = 600)
        
if __name__ == '__main__':
    VisualizeDataInflux().run_script()