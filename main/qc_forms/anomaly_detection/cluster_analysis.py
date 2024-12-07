import pandas as pd

import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import numpy as np 
import math

class ClusterAnalysis():
    """
    class to calculate how much each 
    data point deviates from the others 
    in scatter plots of pairs of variables
    """

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.all_distances = []

    def run_script(self):
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        for network in ['PRONET','PRESCIENT']:
            for tp in tp_list[1:2]:
                print(tp)
                combined_df = pd.read_csv(
                f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv',
                keep_default_na = False)
                self.cluster_cols(combined_df,'chriq_tscore_sum','chriq_fsiq')

    def cluster_cols(self, df : pd.DataFrame,
        col1 : str, col2 : str
    ):
        """
        Collects all numerical rows of
        two specified columns and calculates
        scores based on how much each point would
        from the rest in a scatter plot of the two
        columns

        Parameters
        ---------------
            df : pd.DataFrame
                dataframe being analyzed
            col1 : str
                name of first column of interest
            col2 : str
                name of second column of interest
        """
        df = df.replace(self.utils.missing_code_list, '')
        for col in [col1, col2]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        filtered_df = df.dropna(subset=[col1, col2])
        scatter_axes = {'x':filtered_df[col1].tolist(),
        'y':filtered_df[col2].tolist()}
        subjects = filtered_df['subjectid'].tolist()
        x = scatter_axes['x']
        y = scatter_axes['y']
        
        for ind in range(0,len(x)):
            curr_point_distances = []
            for sub_ind in range(0,len(x)):
                p1 = [x[ind],y[ind]]
                p2 = [x[sub_ind],y[sub_ind]]
                distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                curr_point_distances.append(distance)
            self.all_distances.append({'subject':subjects[ind],
            'point_deviation_global_score':sum(curr_point_distances),
            'point_deviation_local_score':sum(sorted(curr_point_distances)[:5]),
            'x_val':x[ind],'y_val':y[ind]})
        output_df = pd.DataFrame(self.all_distances)
        output_df.to_csv('outlier_test.csv',index = False)


if __name__ == '__main__':
    ClusterAnalysis().run_script()