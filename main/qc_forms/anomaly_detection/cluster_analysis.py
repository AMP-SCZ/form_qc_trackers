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
        self.output_path = f"{self.config_info['paths']['output_path']}variable_relation_data/"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.all_distances = []
        data_dict = self.utils.read_data_dictionary()

        filtered_data_dict = data_dict[data_dict['Field Type'].isin(['text'])]

        self.vars_to_check = filtered_data_dict['Variable / Field Name'].tolist()
        self.grouped_vars = self.utils.load_dependency_json(f"grouped_variables.json")
        self.forms_vars = self.utils.load_dependency_json(f"important_form_vars.json")
        self.melbourne_ra_subjects = self.utils.load_dependency_json(f"melbourne_ra_subs.json")
        blood_vars = self.grouped_vars['blood_vars']
        self.excluded_vars = (blood_vars['id_variables']
        + blood_vars['barcode_variables'] + blood_vars['position_variables'])
        print(self.excluded_vars)
        self.excluded_vars.extend(['chrsaliva_box1a','chrsaliva_box2a',
        'chrsaliva_box3a','chrsaliva_box1b','chrsaliva_box2b','chrsaliva_box3b',
        'chrsaliva_id1a','chrsaliva_bid2a','chrsaliva_id3a','chrsaliva_id1b',
        'chrsaliva_id2b','chrsaliva_id3b','chrdbb_phone_software','chrpps_postal',
        'chrax_device_id','chrpps_postalch'])

        self.vars_to_check = [var for var in self.vars_to_check
        if var not in self.excluded_vars]
        
        self.numerical_stds = {}

        self.numerical_outlier_scores = []
        
        self.melbourne_sub_ra = {}
        for ra, subs in self.melbourne_ra_subjects.items():
            for sub in subs:
                self.melbourne_sub_ra[sub] = ra

    def run_script(self):
        self.analyze_clusters()

    def generate_numerical_outliers(self, df, cols, timepoint, network):
        self.numerical_stds.setdefault(network, {})
        self.numerical_stds[network].setdefault(timepoint, {})
        print(cols)
        
        for col in cols:
            print(col)
            numeric_df = df[[col]].apply(pd.to_numeric, errors='coerce')

            self.numerical_stds[network][timepoint][col] = {'median':numeric_df[col].median(),
                                                            'std':numeric_df[col].std()}
        print(self.numerical_stds)
        for row in df.itertuples():
            for col in cols:
                if col in self.excluded_vars:
                    continue
                if self.utils.can_be_float(getattr(row, col)):
                    val = float(getattr(row, col))
                    std_info = self.numerical_stds[network][timepoint][col]
                    form = self.grouped_vars['var_forms'][col]
                    redcap_user_var = self.forms_vars[form]['redcap_user_var']
                    date_var = self.forms_vars[form]['interview_date_var']
                    redcap_user = ''
                    form_date = ''
                    if redcap_user_var != '' and hasattr(row, redcap_user_var):
                        redcap_user = getattr(row, redcap_user_var)
                    if date_var != '' and hasattr(row, date_var):
                        form_date = getattr(row, date_var)
                        form_date = str(form_date)
                        form_date = form_date.replace('/','-')
                        form_date = form_date.split(' ')[0]
                        form_date = form_date.split('T')[0]
                    if network == 'PRESCIENT':
                        if row.subjectid in self.melbourne_sub_ra.keys():
                            redcap_user = self.melbourne_sub_ra[row.subjectid]
                    median = std_info['median']
                    std = std_info['std']
                    if std == 0:
                        continue
                    curr_var_std = abs((median - val) / std)
                    self.numerical_outlier_scores.append({
                    'subject':row.subjectid,'variable':col,
                    'network':network,'timepoint':timepoint,'variable_val':val,
                    'stds_from_median':curr_var_std,'redcap_user':redcap_user,
                    'form_date':form_date})
            
        outlier_df = pd.DataFrame(self.numerical_outlier_scores)
        outlier_df.to_csv('outlier_score_test.csv',index = False)
    
    def analyze_rater_outliers(self):
        outlier_df = pd.read_csv('outlier_score_test.csv',
                                 keep_default_na=False)
        rater_outliers = {}
        output_list = []
        rater_sub_form_combos = {}
        rater_sub_var_combos = {}
        rater_var_combos = {}
        for row in outlier_df.itertuples():
            if row.redcap_user in ['','[survey respondent]']:
                continue
            rater_sub_form_combos.setdefault(row.redcap_user,{})
            rater_var_combos.setdefault(row.redcap_user, {})
            rater_sub_var_combos.setdefault(row.redcap_user, [])
            rater_outliers.setdefault(row.redcap_user, {'user':row.redcap_user,
                                                    'total':0,'sum':0,'score':0})
            
            form = self.grouped_vars['var_forms'][row.variable]
            sub_form = row.subject + '_' + form
            sub_var = row.subject + '_' + row.variable
            if sub_form in rater_sub_form_combos[row.redcap_user].keys():
                form_multiplier = rater_sub_form_combos[row.redcap_user][sub_form]
            else:
                form_multiplier = 1
            if row.variable in rater_var_combos[row.redcap_user].keys():
                var_multiplier = rater_var_combos[row.redcap_user][row.variable]
            else:
                var_multiplier = 1
            rater_outliers[row.redcap_user]['total']+= 1
            
            rater_outliers[row.redcap_user]['sum'] += ((row.stds_from_median) ** 0.75)* form_multiplier * var_multiplier
            rater_outliers[row.redcap_user]['score'] = (
            rater_outliers[row.redcap_user]['sum'] /
            rater_outliers[row.redcap_user]['total'])
            if row.stds_from_median > 2:
                if sub_var not in rater_sub_var_combos[row.redcap_user]:
                    rater_sub_var_combos[row.redcap_user].append(sub_var)
                    rater_var_combos[row.redcap_user].setdefault(row.variable, 1)
                    rater_var_combos[row.redcap_user][row.variable] *= (row.stds_from_median  ** 0.25)

                rater_sub_form_combos[row.redcap_user].setdefault(sub_form, 1)
                rater_sub_form_combos[row.redcap_user][sub_form] *= 0.5

        for rater, values in rater_outliers.items():
            output_list.append(values)

        output_df = pd.DataFrame(output_list)
        output_df.to_csv('rater_analysis.csv',index = False)

        print(rater_outliers)

    def analyze_clusters(self):
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        self.completed_pairs = {}
        self.numerical_vars = {}
        for network in ['PRONET','PRESCIENT']:
            for tp in tp_list:
                output_df = pd.DataFrame()
                self.completed_pairs.setdefault(network,{})
                self.completed_pairs[network].setdefault(tp,[])
                self.numerical_vars.setdefault(network,{})
                self.numerical_vars[network].setdefault(tp,[])
                self.timepoint = tp
                self.network = network
                combined_df = pd.read_csv(
                f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv',
                keep_default_na = False)
                combined_df = combined_df.replace(self.utils.missing_code_list + [99,99.0,'99','99.0'], '')
                all_columns = list(combined_df.columns)
                self.collect_numerical_vars(combined_df, all_columns, tp, network)
                all_columns = [var for var in all_columns if var in
                            self.vars_to_check and var in self.numerical_vars[network][tp]
                            and var not in self.excluded_vars]
                self.generate_numerical_outliers(combined_df,all_columns, tp, network)
                """for var_ind in range(0,len(all_columns)):
                    if all_columns[var_ind] not in self.numerical_vars[network][tp]:
                        continue
                    print(tp)
                    print(network)  
                    print(var_ind)
                    for sub_var_ind in range(0,len(all_columns)):
                        pairs = [all_columns[var_ind],all_columns[sub_var_ind]]
                        if (pairs in self.completed_pairs[network][tp]
                        or [pairs[1],pairs[0]] in self.completed_pairs[network][tp]):
                            continue
                        elif pairs[0] == pairs[1]:
                            continue
                        else:
                            self.completed_pairs[network][tp].append(pairs)
                        if (all_columns[var_ind] in self.vars_to_check 
                        and all_columns[sub_var_ind] in self.vars_to_check):
                            if (all_columns[sub_var_ind] in self.numerical_vars[network][tp]):     
                                curr_pair_output = self.cluster_cols(
                                combined_df,all_columns[var_ind], all_columns[sub_var_ind])
                                if curr_pair_output.empty:
                                    continue
                                if output_df.empty:
                                    output_df = curr_pair_output
                                else:
                                    output_df = pd.concat([output_df, curr_pair_output], ignore_index=True)
                                output_df = output_df[
                                ((output_df['normalized_deviation_local_score'] > 1) | 
                                (output_df['normalized_deviation_global_score'] > 1))]

                                output_df.to_csv(f'{self.output_path}{network}_{tp}_var_relation_data.csv', index = False)"""


    def collect_numerical_vars(self, df, var_list, tp, network):
        for var in var_list:
            if self.determine_numerical_val_count(df,var) > 50:
                self.numerical_vars[network][tp].append(var)

        print(self.numerical_vars)

    def determine_numerical_val_count(self, df, col):
        numeric_df = df[[col]].apply(pd.to_numeric, errors='coerce')

        result = numeric_df.count()

        return int(result)


    def cluster_cols(self, df : pd.DataFrame,
        col1 : str, col2 : str, 
        filter_outliers : bool = True
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
        all_distances = []
        df = df.replace(self.utils.missing_code_list + [99,99.0,'99','99.0'], '')
        for col in [col1, col2]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if filter_outliers:
                median = df[col].median()
                std =df[col].std()
                if std == 0:
                    continue
                df[col + '_std'] = abs(df[col] - median / std)
                df = df[df[col+'_std']< 2.5] 
            
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
            if len(curr_point_distances) < 50:
                continue
            all_distances.append({'subject':subjects[ind], 'timepoint':self.timepoint,
            'network' : self.network,'point_deviation_global_score':sum(curr_point_distances),
            'point_deviation_local_score':sum(sorted(curr_point_distances)[:5]),
            'x_val':x[ind],'y_val':y[ind]})
        output_df = pd.DataFrame(all_distances)
        if output_df.empty:
            return output_df
        for categ in ['global','local']:
            median = output_df[f'point_deviation_{categ}_score'].median()
            std_dev = output_df[f'point_deviation_{categ}_score'].std()

            output_df[f'normalized_deviation_{categ}_score'] = abs(
            (output_df[f'point_deviation_{categ}_score'] - median) / std_dev)
        output_df['column_pairs'] = f"X : {col1} | Y: {col2}"

        return output_df
    
    def create_final_scores(self):
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        self.qc_scores = {}
        self.qc_output_list = []
        for network in ['PRONET','PRESCIENT']:
            for tp in tp_list:
                self.qc_scores.setdefault(network,{})
                self.qc_scores[network].setdefault(tp,{})

                output_df_path = f'{self.output_path}{network}_{tp}_var_relation_data.csv'
                if not os.path.exists(output_df_path):
                    continue
                output_df = pd.read_csv(output_df_path, keep_default_na=False)
                for row in output_df.itertuples():
                    sub = row.subject
                    self.qc_scores[network][tp].setdefault(sub, {})
                    var_list = row.column_pairs
                    var_list = var_list.replace('X : ','').replace('Y: ','') 
                    var_list = var_list.split(' | ')
                    
                    for var in var_list:
                        self.qc_scores[network][tp][sub].setdefault(var,
                        {'local_scores':[],'global_scores' : [], "variable_pairs": []})
                        self.qc_scores[network][tp][sub][var][
                        'local_scores'].append(row.normalized_deviation_local_score)
                        self.qc_scores[network][tp][sub][var][
                        'global_scores'].append(row.normalized_deviation_global_score)
                        other_var = [alt_var for alt_var in var_list if alt_var != var]
                        other_var = other_var[0]
                        self.qc_scores[network][tp][sub][var][
                        'variable_pairs'].append({var_list[0]:row.x_val,var_list[1]: row.y_val})
                        
                        
                    self.qc_scores[network][tp][sub][var_list[0]]['var_val'] = row.x_val
                    self.qc_scores[network][tp][sub][var_list[1]]['var_val'] = row.y_val
                for sub, vars in self.qc_scores[network][tp].items():
                    for var, scores in vars.items():
                        scores['global_scores'] = [score for score in scores['global_scores'] if score !='']
                        scores['local_scores'] = [score for score in scores['local_scores'] if score !='']
                        
                        if len(scores['global_scores']) > 5 and len(scores['local_scores']) > 5:
                            scores['global_scores'] = [float(score) for score in scores['global_scores']]
                            scores['local_scores'] = [float(score) for score in scores['local_scores']]
                            try:
                                #paired = sorted(zip(scores['global_scores'], scores['variable_pairs']))
                                #sorted_list1, sorted_list2 = zip(*paired)

                                #scores['variable_pairs'] = list(sorted_list2)[:5]

                                self.qc_output_list.append({'subject':sub,
                                'timepoint':tp,'network':network,'variable':var, 'var_val': scores['var_val'],
                                'total_global':sum(scores['global_scores']),
                                'top_5_global':sum(sorted(scores['global_scores'])[:5]),
                                'total_local':sum(scores['local_scores']),
                                'top_5_local':sum(sorted(scores['local_scores'])[:5]),
                                'most_impactful_vars':scores['variable_pairs']},
                                )
                            except Exception as e:
                                print(e)
                                print(scores)
                                continue
                outlier_df = pd.read_csv('outlier_score_test.csv',keep_default_na=False)
                
                output_df = pd.DataFrame(self.qc_output_list)
                #merged = pd.merge(outlier_df, output_df,how = 'inner', on =['subject','timepoint','variable'])
                #print(merged.columns)
                #merged['global_score_outlier_ratio'] =merged['total_global'] /merged['stds_from_median']
                #merged.to_csv('merged_qc_Scores.csv',index = False)
                output_df.to_csv('qc_output_test.csv',index = False)

if __name__ == '__main__':
    ClusterAnalysis().analyze_rater_outliers()
    #ClusterAnalysis().create_final_scores()