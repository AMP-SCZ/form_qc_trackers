import pandas as pd
import os
import sys
import json
from datetime import datetime
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class MultiTPDataCollector():
    """
    Class to collect data
    from multiple timepoints 

    """
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        self.earliest_latest_dates_per_tp = {}
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.combined_csv_path = self.config_info['paths']['combined_csv_path']
        self.grouped_vars = self.utils.load_dependency_json(f"grouped_variables.json")
        #self.loop_networks()
        self.earliest_date_per_var = {}
        self.var_forms = self.utils.load_dependency_json(
        f"grouped_variables.json")
        self.forms_per_var = self.var_forms['var_forms']
        self.important_form_vars = self.utils.load_dependency_json(
        'important_form_vars.json')
        self.forms_per_tp = self.utils.load_dependency_json(
        'forms_per_timepoint.json')
        self.subject_info = self.utils.load_dependency_json(
        'subject_info.json')
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.multitp_output = pd.DataFrame()
        self.multi_tp_vars = ['subjectid', 'visit_status_string',
        'chrblood_wb2id']
        self.loop_csvs()

    def __call__(self):
        self.loop_csvs()

    def loop_csvs(self):
        """
        loops through each combined csv
        to collect multi timepoint data
        """
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        multi_tp_df = pd.DataFrame()
        for network in ['PRESCIENT','PRONET']:
            for tp in tp_list:
                combined_df = pd.read_csv(
                f'{self.comb_csv_path}combined-{network}-{tp}-day1to1.csv',
                keep_default_na = False)
                self.collect_earliest_date(combined_df)
                self.collect_earliest_latest_dates(combined_df, tp)
                """modified_df = self.utils.append_suffix_to_cols(combined_df, tp)
                if multi_tp_df.empty:
                    multi_tp_df = modified_df
                else:
                    multi_tp_df = multi_tp_data.merge(modified_df,
                    how = 'outer')"""

        #multi_tp_df.to_csv('multi_tp_df_test.csv',
        #index = False)
                self.utils.save_dependency_json(self.earliest_latest_dates_per_tp,
                'earliest_latest_dates_per_tp.json')

    def collect_earliest_latest_dates(self,
        combined_df : pd.DataFrame, tp: str
    ):
        """
        Collects the earliest and
        latest dates of each visit
        for each subject 

        Parameters
        --------
        combined_df : pd.DataFrame
            current timepoint's dataframe
        tp : str
            current tp
        """
        forms_per_tp = self.forms_per_tp
        #tp = tp.replace('screening','screen').replace('baseline','baseln')
        for row in combined_df.itertuples():
            subject = row.subjectid
            if (subject in self.subject_info.keys()
            and self.subject_info[subject]['cohort'] != 'unknown'):
                cohort = self.subject_info[subject]['cohort']
                all_forms = forms_per_tp[cohort][tp]
                for form in all_forms:
                    if self.utils.check_if_missing(row, form) == True:
                        continue
                    interview_date_var = self.important_form_vars[form]['interview_date_var']
                    if interview_date_var != '' and hasattr(row,interview_date_var):
                        int_date = getattr(row, interview_date_var)
                        if (int_date not in (self.utils.missing_code_list + [''])
                        and self.utils.check_if_val_date_format(int_date)):
                            int_date_str = str(int_date).split(' ')[0]
                            int_date_datetime = datetime.strptime(str(int_date).split(' ')[0], "%Y-%m-%d")
                            self.earliest_latest_dates_per_tp.setdefault(subject, {})
                            self.earliest_latest_dates_per_tp[subject].setdefault(tp,
                            {'earliest':int_date_str,'latest':int_date_str})
                            curr_early = self.earliest_latest_dates_per_tp[
                            subject][tp]['earliest']
                            curr_late = curr_early = self.earliest_latest_dates_per_tp[
                            subject][tp]['latest']
                            if int_date_datetime < datetime.strptime(curr_early, "%Y-%m-%d"):
                                self.earliest_latest_dates_per_tp[subject][tp]['earliest'] = int_date_str
                            if int_date_datetime > datetime.strptime(curr_late, "%Y-%m-%d"):
                                self.earliest_latest_dates_per_tp[subject][tp]['latest'] = int_date_str

    def collect_earliest_date(self, 
        combined_df : pd.DataFrame
    ):
        """
        Collects earliest date a 
        variable was used if there 
        is an interview date
        """
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

    def search_timepoint_dates(self, 
        row : tuple,
        forms : list
    ):
        date_list = []
        
    def loop_networks(self):
        for network in ['PRONET', 'PRESCIENT']:
            self.collect_blood_duplicates(network)
            
    def collect_blood_duplicates(self, 
        network : str
    ):
        """
        creates dataframe of 
        blood variables 
        and calls functions to 
        check for duplicates

        parameters
        -------------------
        network : str
            current network (Pronet or Prescient) 
        """
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

    def check_id_duplicates(self, 
        merged_blood_df : pd.DataFrame
    ):
        """
        collects duplicate IDs

        Parameters 
        -------------
        merged_blood_df : pd.DataFrame
            Dataframe of blood variables

        """
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

    def check_position_duplicates(self, 
        merged_blood_df : pd.DataFrame
    ):  
        """
        collects duplicate
        blood vial positions

        Parameters 
        -------------
        merged_blood_df : pd.DataFrame
            Dataframe of blood variables
        """
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


                                    