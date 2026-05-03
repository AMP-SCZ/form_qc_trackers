import os
import sys
import json
import csv
import traceback
from pathlib import Path
from collections import Counter

import pandas as pd
from frictionless import validate, Schema, system

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class CalcQCer:
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f"{self.absolute_path}/config.json", "r") as file:
            self.config_info = json.load(file)
        self.comb_csv_path = "/home/ob001/refactored_qc/dependencies/comb_csvs_3_30_2026/PROTECTED/"
        self.output_path = self.config_info["paths"]["output_path"]
        self.depend_path = self.config_info["paths"]["dependencies_path"]
        self.grouped_vars = self.utils.load_dependency_json(f"grouped_variables.json")
        self.form_per_var = self.grouped_vars["var_forms"]
        self.important_form_vars = self.utils.load_dependency_json(
        'important_form_vars.json')
        self.all_data_path = f"{self.depend_path}PNLYaleExtractTEST_DATA_2026-04-11_1732.csv"
        self.all_data_df = pd.read_csv(
        self.all_data_path,
        keep_default_na = False)
        self.recalc_path = f"{self.depend_path}rule_h_like_discrepancies.csv"
        self.reclacs_df = pd.read_csv(self.recalc_path,
        keep_default_na = False)
        subs = self.reclacs_df['record_id'].tolist()
        print(len(list(set(subs))))
        self.final_output_list = []
        self.revised_final_output_list = []
        self.diff_parts = []
        self.excluded_strings = ['__','complete']
        self.data_dict = self.utils.read_data_dictionary()

        self.calcs_only = self.data_dict[
        self.data_dict['Field Type'] == 'calc']

        self.all_calcs = self.calcs_only['Variable / Field Name'].tolist()


    def run_script(self):
        self.loop_csvs() 

    def loop_csvs(self):
        tp_list = self.utils.create_timepoint_list()
        for network in ['PRONET','PRESCIENT']:
           for tp in tp_list:
                tp = tp.replace("month","month_").replace("floating","floating_forms")
                combined_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp}'
                f'_{network.replace("PRONET","ProNET")}-day1to1.csv'),
                keep_default_na = False)
                self.compare_all_data_revised(combined_df, tp)

                #self.merge_tps(combined_df, tp)

    def compare_all_data_revised(self,comb_df,tp):
        for row in self.all_data_df.itertuples():
            sub = getattr(row,'chric_record_id')
            if tp != getattr(row,'redcap_event_name').replace('_arm_1','').replace('_arm_2',''):
                continue
            print(sub)
            print(row.Index)
            filtered_df = comb_df[comb_df['subjectid'] == sub]
            for comb_row in filtered_df.itertuples():
                for col in self.all_calcs:
                    if col in comb_df.columns and col in self.all_data_df.columns:
                        skip = False
                        if col not in self.form_per_var.keys():
                            continue
                        form = self.form_per_var[col]
                        for imp_var in ['missing_var','completion_var']:
                            if imp_var not in self.important_form_vars[form].keys():
                                skip = True
                                continue
                            imp_var_name = self.important_form_vars[form][imp_var]
                            if imp_var_name =='':
                                skip = True
                                continue

                            if (imp_var == 'missing_var' and
                            hasattr(comb_row, imp_var_name) and
                            getattr(comb_row, imp_var_name) in [1,1.0,'1','1.0']):
                                skip = True
                                continue

                            if (imp_var == 'completion_var' and 
                            hasattr(comb_row, imp_var_name) and
                            getattr(comb_row, imp_var_name) not in [2,2.0,'2','2.0']):
                                skip = True
                                continue

                        if skip == True:
                            continue

                        comb_val = str(getattr(comb_row, col))
                        recalc_val = str(getattr(row, col))
                        if comb_val.strip() != recalc_val.strip():
                            if (not any(string in col for string in self.excluded_strings) and
                            comb_val not in ['nan'] and len(comb_val) < 100):
                                self.revised_final_output_list.append({
                                'subject':sub,'timepoint':tp,'var':col,'comb_csv_val':
                                comb_val,'recalc_val':recalc_val})

            output_df = pd.DataFrame(self.revised_final_output_list)
            output_df.to_csv('diffs_test_revised_new_module.csv',index = False)


    def compare_all_data(self, comb_df, tp):
        key_col = "subjectid"

        filtered_df = self.all_data_df[
            self.all_data_df["redcap_event_name"].str.contains(tp, na=False)
        ].copy()

        filtered_df = filtered_df.rename(columns={"chric_record_id": key_col}).copy()
        comb_df = comb_df.copy()

        # Normalize blank strings to missing
        filtered_df = filtered_df.replace(r"^\s*$", pd.NA, regex=True)
        comb_df = comb_df.replace(r"^\s*$", pd.NA, regex=True)

        # Keep only columns shared by both dfs
        common_cols = [c for c in filtered_df.columns if c in comb_df.columns and c != key_col]

        # Merge once
        merged = filtered_df[[key_col] + common_cols].merge(
            comb_df[[key_col] + common_cols],
            on=key_col,
            how="outer",
            suffixes=("_df1", "_df2"),
            indicator=True
        )

        def normalize_pair(s1, s2):
            """
            Normalize two series to a common comparable type.
            """
            # Try numeric comparison first
            n1 = pd.to_numeric(s1, errors="coerce")
            n2 = pd.to_numeric(s2, errors="coerce")

            if n1.notna().sum() > 0 or n2.notna().sum() > 0:
                both_numeric_or_na = (
                    (n1.notna() | s1.isna()) &
                    (n2.notna() | s2.isna())
                )
                if both_numeric_or_na.all():
                    return n1, n2

            # Try datetime comparison
            d1 = pd.to_datetime(s1, errors="coerce")
            d2 = pd.to_datetime(s2, errors="coerce")
            if d1.notna().sum() > 0 or d2.notna().sum() > 0:
                both_datetime_or_na = (
                    (d1.notna() | s1.isna()) &
                    (d2.notna() | s2.isna())
                )
                if both_datetime_or_na.all():
                    return d1, d2

            # Fallback to stripped string comparison
            s1_norm = s1.astype("string").str.strip()
            s2_norm = s2.astype("string").str.strip()
            return s1_norm, s2_norm

        def should_skip_col(row, col):
            if col not in self.form_per_var:
                return False

            form = self.form_per_var[col]
            if form not in self.important_form_vars:
                return False

            form_meta = self.important_form_vars[form]

            # Skip if marked missing
            missing_var_name = form_meta.get("missing_var", "")
            if missing_var_name:
                missing_val = row.get(missing_var_name, pd.NA)
                if pd.notna(missing_val) and str(missing_val).strip() in {"1", "1.0"}:
                    return True

            # Skip if form is not complete
            completion_var_name = form_meta.get("completion_var", "")
            if completion_var_name:
                completion_val = row.get(completion_var_name, pd.NA)
                if pd.isna(completion_val) or str(completion_val).strip() not in {"2", "2.0"}:
                    return True

            return False

        for _, row in merged.iterrows():
            for col in common_cols:
                if should_skip_col(row, col):
                    continue

                left = pd.Series([row[f"{col}_df1"]])
                right = pd.Series([row[f"{col}_df2"]])

                left_norm, right_norm = normalize_pair(left, right)

                same_mask = (
                    left_norm.eq(right_norm) |
                    (left_norm.isna() & right_norm.isna())
                )

                if not same_mask.iloc[0]:
                    temp = pd.DataFrame({
                        key_col: [row[key_col]],
                        "column": [col],
                        "df1_value": [row[f"{col}_df1"]],
                        "df2_value": [row[f"{col}_df2"]],
                        "_merge": [row["_merge"]],
                    })
                    self.diff_parts.append(temp)

        if self.diff_parts:
            diff_df = pd.concat(self.diff_parts, ignore_index=True)
        else:
            diff_df = pd.DataFrame(columns=[key_col, "column",
            "df1_value", "df2_value", "_merge"])

        print(diff_df)
        diff_df.to_csv("diffs_test.csv", index=False)

    def merge_tps(self, comb_df, tp):
        print(tp)
        tp = tp.replace('month', 'month_')

        filtered_df = self.reclacs_df[
            self.reclacs_df['redcap_event_name'].str.contains(tp,
            regex=False, na=False)
        ].copy()

        filtered_df = filtered_df.rename(columns={'record_id': 'subjectid'})
        filtered_vals = {}
        for row in filtered_df.itertuples():
            filtered_vals.setdefault(row.subjectid,{})
            var = getattr(row,'field_name')
            val = getattr(row,'recalc_value')
            filtered_vals[row.subjectid][var]=val

        all_cols = list(comb_df.columns)
        for row in comb_df.itertuples():
            sub = row.subjectid
            if sub not in filtered_vals.keys():
                continue
            for col in all_cols:
                if col not in filtered_vals[sub].keys():
                    continue
                if col not in self.form_per_var.keys():
                    continue
                form = self.form_per_var[col]
                skip = False
                for imp_var in ['missing_var','completion_var']:
                    if imp_var not in self.important_form_vars[form].keys():
                        skip = True
                        continue
                    imp_var_name = self.important_form_vars[form][imp_var]
                    if imp_var_name =='':
                        skip = True
                        continue

                    if (imp_var == 'missing_var' and
                    getattr(row, imp_var_name) in [1,1.0,'1','1.0']):
                        skip = True
                        continue

                    if (imp_var == 'complete_var' and 
                    getattr(row, imp_var_name) not in [2,2.0,'2','2.0']):
                        skip = True
                        continue

                if skip == True:
                    continue
                
                comb_val = getattr(row,col)
                recalc_val = filtered_vals[row.subjectid][col]
                if str(comb_val) != str(recalc_val):
                    self.final_output_list.append({
                    'sub':sub,'tp':tp,'var':col,
                    'comb_csv_val':comb_val,'recalc_val':recalc_val})

        output_df = pd.DataFrame(self.final_output_list)
        output_df.to_csv(
        f'{self.output_path}calculation_mismatches.csv',
        index = False)
        print(f'saved to {self.output_path} ')

if __name__ == '__main__':
    CalcQCer().run_script()