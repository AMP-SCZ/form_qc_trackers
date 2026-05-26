import pandas as pd
import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class CollectSubjectInfo():
    """
    Class to collect basic information
    about each subject, such as their cohort,
    sex, age, and inclusion status
    """

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.comb_csv_path = self.config_info['paths']['combined_csv_path']

        self.subject_info = {}

        self.var_translations = {
            'chrcrit_included': {1:'included',0: 'excluded'},
            'chrcrit_part' : {1: 'CHR', 2: 'HC'},
            'chrdemo_sexassigned' : {1 : 'Male', 2 : 'Female'},
            'chr_statusform_screenfail' : {1 : 'true', 0 : 'false'},
            'chr_subject_eos' : {1 : 'true', 0 : 'false'}
        }

    def __call__(self):
        self.collect_screening_info()
        self.collect_baseline_info()
        self.collect_floating_info()
        self.collect_conversion_info()
        self.collect_conversion_criteria_info()

        return self.subject_info

    def translate_var_vals(self, translation_dict, var_val):
        for orig_val, translated_val in translation_dict.items():
            if var_val in self.utils.all_dtype([orig_val]):
                return translated_val
        
        return 'unknown'
    
    def collect_age(self,row):
        for age_var in ['chrdemo_age_mos_chr',
        'chrdemo_age_mos_hc', 'chrdemo_age_mos2']:
            if not hasattr(row,age_var):
                continue
            age_val = getattr(row,age_var)
            if (age_val not in 
            (self.utils.missing_code_list + [""]) and 
            self.utils.can_be_float(age_val) and
            self.utils.is_a_num(str(age_val).replace('.',''))):
                age = round(float(age_val) / 12, 2)
                return age

        return 'unknown'

    def collect_screening_info(self):
        tp = 'screening'
        for network in ['PRONET','PRESCIENT']:
            combined_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}'
                f'_{network.replace("PRONET","ProNET")}-day1to1.csv'),
                keep_default_na = False)
            col_list = ['subjectid','visit_status_string',
            'chrcrit_part', 'chrcrit_included',
            'chrpsychs_scr_interview_date',
            'chric_actigraphy','chric_passive','chrpharm_interview_date']
            col_list = [col for col in col_list if col in combined_df.columns]
            combined_df = combined_df[col_list]
            for row in combined_df.itertuples():
                sub = row.subjectid
                self.subject_info.setdefault(sub, {})
                self.subject_info[sub][
                'inclusion_status'] = self.translate_var_vals(
                self.var_translations['chrcrit_included'], row.chrcrit_included)
                self.subject_info[sub][
                'cohort'] = self.translate_var_vals(
                self.var_translations['chrcrit_part'], row.chrcrit_part)
                self.subject_info[sub][
                'visit_status'] = row.visit_status_string
                self.subject_info[sub][
                'psychs_screen_date'] = row.chrpsychs_scr_interview_date
                self.subject_info[sub][
                'axivity_opt'] = row.chric_actigraphy 
                self.subject_info[sub][
                'mindlamp_opt'] = row.chric_passive
                self.subject_info[sub][
                'past_pharm_date'] = row.chrpharm_interview_date

    def collect_baseline_info(self):
        tp = 'baseline'
        for network in ['PRONET','PRESCIENT']:
            combined_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}'
                f'_{network.replace("PRONET","ProNET")}-day1to1.csv'),
                keep_default_na = False)
            col_list = ['subjectid','chrdemo_age_mos_chr',
            'chrdemo_age_mos_hc', 'chrdemo_age_mos2', 
            'chrdemo_sexassigned','chrdemo_interview_date']
            col_list = [col for col in col_list if col in combined_df.columns]
            combined_df = combined_df[col_list]
            for row in combined_df.itertuples():
                sub = row.subjectid
                self.subject_info.setdefault(sub, {})
                self.subject_info[sub][
                'sex'] = self.translate_var_vals(
                self.var_translations['chrdemo_sexassigned'], row.chrdemo_sexassigned)
                self.subject_info[sub][
                'age'] = self.collect_age(row)
                self.subject_info[sub][
                'demographics_date'] = getattr(row,'chrdemo_interview_date')

    def collect_floating_info(self):
        tp = 'floating'
        for network in ['PRONET']:
            combined_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}'
                f'_{network.replace("PRONET","ProNET")}-day1to1.csv'),
                keep_default_na = False)
            col_list = ['subjectid','chr_statusform_screenfail',
            'chr_subject_eos','chrpharm_date_first','chrpharm_date_mod',
            'chrpharm_date_mod_2']
            col_list = [col for col in col_list if col in combined_df.columns]
            combined_df = combined_df[col_list]
            for row in combined_df.itertuples():
                sub = row.subjectid
                self.subject_info.setdefault(sub, {})
                self.subject_info[sub][
                'screenfail'] = self.translate_var_vals(
                self.var_translations['chr_statusform_screenfail'],
                row.chr_statusform_screenfail)
                self.subject_info[sub][
                'completed_study'] = self.translate_var_vals(
                self.var_translations['chr_subject_eos'],
                row.chr_subject_eos)
                self.subject_info[sub][
                'curr_pharm_date'] = row.chrpharm_date_first
                self.subject_info[sub][
                'chrpharm_date_mod'] = row.chrpharm_date_mod
                self.subject_info[sub][
                'chrpharm_date_mod_2'] = row.chrpharm_date_mod_2

    def collect_conversion_info(self):
        """
        Collect per-subject conversion status from the floating-forms CSV
        and stamp it onto subject_info[sub]['converted'].

        The conversion form (chrconv_conv) is filed at the floating
        timepoint, but the resulting status is consumed by per-row checks
        (`conversion_criteria_check` and `marked_converted_no_criteria_check`
        in clinical_checks_main.py) at every timepoint. Collecting it here
        lets those checks read it directly off subject_info without
        re-opening the floating CSV.

        chrconv_conv == 1 → marked converted. Anything else (blank,
        missing-coded, 0, etc.) → not converted. Subjects without a
        floating-CSV row simply don't get the key; the consumer uses
        `subject_info.get(sid, {}).get('converted', False)` as default.

        Loops over both networks so PRESCIENT subjects are covered too;
        wraps the read in try/FileNotFoundError so a missing floating
        CSV for one network just skips that network with a warning rather
        than aborting all of process_variables.
        """
        tp = 'floating'
        converted_codes = self.utils.all_dtype([1])
        for network in ['PRONET', 'PRESCIENT']:
            csv_path = (
                f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}'
                f'_{network.replace("PRONET","ProNET")}-day1to1.csv'
            )
            try:
                combined_df = pd.read_csv(csv_path, keep_default_na=False)
            except FileNotFoundError:
                print(
                    f"[collect_subject_info] floating CSV not found for "
                    f"{network}: {csv_path}. Skipping conversion-status "
                    f"collection for this network."
                )
                continue
            if 'chrconv_conv' not in combined_df.columns:
                print(
                    f"[collect_subject_info] {network} floating CSV has no "
                    f"chrconv_conv column. Skipping conversion-status "
                    f"collection for this network."
                )
                continue
            col_list = [c for c in ['subjectid', 'chrconv_conv']
                        if c in combined_df.columns]
            combined_df = combined_df[col_list]
            for row in combined_df.itertuples():
                sub = row.subjectid
                self.subject_info.setdefault(sub, {})
                val = getattr(row, 'chrconv_conv', '')
                if val in converted_codes:
                    self.subject_info[sub]['converted'] = True
                else:
                    self.subject_info[sub].setdefault('converted', False)

    def collect_conversion_criteria_info(self):
        """
        Aggregate cross-timepoint psychs / scid conversion-criteria
        flags onto subject_info. Consumed by
        `clinical_checks_main.marked_converted_no_criteria_check` at
        the floating row, where chrconv_method picks which set of
        criteria the operator used to determine conversion.

        The criteria values themselves can appear at any timepoint
        (psychs / scid forms aren't floating-only), so we sweep every
        per-tp combined CSV for both networks and OR-accumulate
        booleans onto:
          subject_info[sub]['has_psychs_criteria']
          subject_info[sub]['has_scid_criteria']

        We additionally track conversion-tp-only versions, which only
        flip True from rows in the `conversion` tp CSV, consumed by
        `marked_converted_no_conv_tp_criteria_check`:
          subject_info[sub]['has_psychs_criteria_at_conversion_tp']
          subject_info[sub]['has_scid_criteria_at_conversion_tp']

        Threshold spec is sourced from
        dependencies/conversion_criteria_thresholds.json so the
        consumer in clinical_checks_main reads the same file.
        """
        criteria = self.utils.load_dependency_json(
            'conversion_criteria_thresholds.json'
        )
        psychs_thresholds = criteria['psychs']
        scid_thresholds = criteria['scid']

        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating', 'conversion'])

        for network in ['PRONET', 'PRESCIENT']:
            for tp in tp_list:
                csv_path = (
                    f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                    f'{tp.replace("month","month_").replace("floating","floating_forms")}'
                    f'_{network.replace("PRONET","ProNET")}-day1to1.csv'
                )
                try:
                    combined_df = pd.read_csv(csv_path, keep_default_na=False)
                except FileNotFoundError:
                    continue
                if 'subjectid' not in combined_df.columns:
                    continue

                psychs_cols = [c for c in psychs_thresholds
                               if c in combined_df.columns]
                scid_cols = [c for c in scid_thresholds
                             if c in combined_df.columns]
                col_list = ['subjectid'] + psychs_cols + scid_cols
                combined_df = combined_df[col_list]

                for row in combined_df.itertuples():
                    sub = row.subjectid
                    self.subject_info.setdefault(sub, {})
                    self.subject_info[sub].setdefault(
                        'has_psychs_criteria', False)
                    self.subject_info[sub].setdefault(
                        'has_scid_criteria', False)
                    self.subject_info[sub].setdefault(
                        'has_psychs_criteria_at_conversion_tp', False)
                    self.subject_info[sub].setdefault(
                        'has_scid_criteria_at_conversion_tp', False)

                    psychs_hit = False
                    if not self.subject_info[sub]['has_psychs_criteria'] \
                            or (tp == 'conversion' and not
                                self.subject_info[sub][
                                    'has_psychs_criteria_at_conversion_tp']):
                        for var in psychs_cols:
                            var_val = getattr(row, var, '')
                            if var_val in self.utils.missing_code_set:
                                continue
                            if (self.utils.can_be_float(var_val)
                                and float(var_val) == psychs_thresholds[var]):
                                psychs_hit = True
                                break
                    if psychs_hit:
                        self.subject_info[sub]['has_psychs_criteria'] = True
                        if tp == 'conversion':
                            self.subject_info[sub][
                                'has_psychs_criteria_at_conversion_tp'] = True

                    scid_hit = False
                    if not self.subject_info[sub]['has_scid_criteria'] \
                            or (tp == 'conversion' and not
                                self.subject_info[sub][
                                    'has_scid_criteria_at_conversion_tp']):
                        for var in scid_cols:
                            var_val = getattr(row, var, '')
                            if var_val in self.utils.missing_code_set:
                                continue
                            if (self.utils.can_be_float(var_val)
                                and float(var_val) == scid_thresholds[var]):
                                scid_hit = True
                                break
                    if scid_hit:
                        self.subject_info[sub]['has_scid_criteria'] = True
                        if tp == 'conversion':
                            self.subject_info[sub][
                                'has_scid_criteria_at_conversion_tp'] = True
