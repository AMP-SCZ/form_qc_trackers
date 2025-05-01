import pandas as pd
import os
import sys
import json
from datetime import datetime
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class MultiTPDataCollector():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.combined_csv_path = self.config_info['paths']['combined_csv_path']
        self.grouped_vars = self.utils.load_dependency_json(f"grouped_variables.json")
        self.loop_networks()
        self.earliest_date_per_var = {}
        self.var_forms = self.utils.load_dependency_json(
        f"grouped_variables.json")
        self.forms_per_var = self.var_forms['var_forms']
        self.important_form_vars = self.utils.load_dependency_json(
        'important_form_vars.json')
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.loop_csvs()

    def __call__(self):
        self.loop_csvs()

    def loop_csvs(self):
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        for network in ['PRESCIENT','PRONET']:
            for tp in tp_list:
                combined_df = pd.read_csv(
                f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv',
                keep_default_na = False)
                self.collect_earliest_date(combined_df)

    def collect_earliest_date(self, combined_df : pd.DataFrame):
        all_cols = combined_df.columns
        for row in combined_df.itertuples():
            for col in all_cols:
                if col in self.forms_per_var.keys():
                    form = self.forms_per_var[col]
                    if form in self.important_form_vars.keys():
                        interview_date_var = self.important_form_vars[form]['interview_date_var']
                        missing_var = self.important_form_vars[form]['missing_var']
                        missing_spec_var = self.important_form_vars[form]['missing_spec_var']
                        if interview_date_var != '' and interview_date_var in all_cols:
                            interview_date_val = getattr(row, interview_date_var)
                            if (interview_date_val not in (self.utils.missing_code_list + ['']) and
                            self.utils.check_if_val_date_format(str(interview_date_val))):
                                datetime_format_val = datetime.strptime(str(interview_date_val), "%Y-%m-%d")
                                if datetime_format_val < datetime.strptime("2022-01-01", "%Y-%m-%d"):
                                    continue
                                self.earliest_date_per_var.setdefault(col, str(interview_date_val))
                                if (datetime_format_val < datetime.strptime(
                                self.earliest_date_per_var[col], "%Y-%m-%d") ):
                                    self.earliest_date_per_var[col] = str(interview_date_val)

        sorted_date_dict = dict(
            sorted(self.earliest_date_per_var.items(),
            key=lambda item: datetime.strptime(item[1], "%Y-%m-%d"), reverse=True)
        )
 
        self.utils.save_dependency_json(sorted_date_dict,
        'earliest_dates_per_var.json')

    def search_timepoint_dates(self, row, forms):
        date_list = []
        
    def loop_networks(self):
        for network in ['PRONET']:
            self.collect_blood_duplicates(network)
            
    def collect_blood_duplicates(self, network):
        baseln_df = pd.read_csv(
        f'{self.combined_csv_path}combined-{network}-baseline-day1to1.csv',
        keep_default_na = False)
        preserved_cols = [col for col in baseln_df.columns if 'chrblood' in col]
        preserved_cols.append('subjectid')
        baseln_df = baseln_df[preserved_cols] 

        month2_df = pd.read_csv(
        f'{self.combined_csv_path}combined-{network}-month2-day1to1.csv',
        keep_default_na = False)
        preserved_cols = [col for col in preserved_cols if col in month2_df.columns]
        month2_df = month2_df[preserved_cols]

        merged_blood_df = pd.merge(baseln_df, month2_df,
        on='subjectid', how = 'outer', suffixes=('_baseln','_month2'))

        self.check_position_duplicates(merged_blood_df)

    def check_id_duplicates(self, merged_blood_df):
        id_vars = self.grouped_vars['blood_vars']['id_variables']
        merged_id_vars = []
        for var in id_vars:
            for suffix in ['_baseln','_month2']:
                merged_id_vars.append(var + suffix)
        merged_id_vars = [var for var in merged_id_vars
        if var in merged_blood_df.columns]

        preserved_cols = merged_id_vars + ['subjectid']
        id_df = merged_blood_df[preserved_cols]
        excluded_val_list = (self.utils.missing_code_list + [''])

        id_df = id_df.melt(id_vars='subjectid',
        value_vars=merged_id_vars, var_name='id_name', value_name='id_val')
        id_df = id_df[~id_df['id_val'].isin(excluded_val_list)]
        id_df = id_df[id_df.duplicated(subset=['id_val'], keep=False) & 
        (id_df.duplicated(subset=['id_val', 'subjectid'], keep=False) == False)]

    def check_position_duplicates(self, merged_blood_df):
        pos_vars = self.grouped_vars['blood_vars']['position_variables']
        merged_pos_vars = []
        merged_barcode_vars = []
        for suffix in ['_baseln','_month2']:
            merged_barcode_vars.append('chrblood_rack_barcode' + suffix)
            for var in pos_vars:
                merged_pos_vars.append(var + suffix)
        
        preserved_cols = merged_pos_vars + ['subjectid'] + merged_barcode_vars
        preserved_cols = [var for var in preserved_cols
        if var in merged_blood_df.columns]

        pos_df = merged_blood_df[preserved_cols]

        merged_barc_pos = []
        excluded_val_list = (self.utils.missing_code_list + [''])

        for barcode_var in merged_barcode_vars:
            for pos_var in merged_pos_vars:
                diff_tp = False
                for suffix in ['_baseln','_month2']:
                    if suffix in pos_var and suffix not in barcode_var:
                        diff_tp = True
                if diff_tp:
                    continue      
                pos_df.loc[(~pos_df[
                barcode_var].isin(excluded_val_list)) & (
                ~pos_df[pos_var].isin(
                excluded_val_list)), barcode_var + '_' + pos_var] = pos_df[
                barcode_var].astype(str) + '_' + pos_df[pos_var].astype(str)
                merged_barc_pos.append(barcode_var + '_' + pos_var)

        pos_df = pos_df[merged_barc_pos + ['subjectid']]
        pos_df = pos_df.melt(id_vars='subjectid', value_vars=merged_barc_pos,
        var_name='id_name', value_name='barc_pos_val')
        pos_df = pos_df.fillna('')
        pos_df = pos_df[~pos_df['barc_pos_val'].isin(excluded_val_list)]
        pos_df = pos_df[pos_df.duplicated(subset=['barc_pos_val'], keep=False) & 
        (pos_df.duplicated(subset=['barc_pos_val', 'subjectid'], keep=False) == False)]

                                    


