import os
import sys
import json

import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from utils.branching_logic_eval import evaluate_branching_logic
from process_variables.transform_branching_logic import TransformBranchingLogic

class CalcQCer:
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f"{self.absolute_path}/config.json", "r") as file:
            self.config_info = json.load(file)
        self.output_path = self.config_info["paths"]["output_path"]
        self.depend_path = self.config_info["paths"]["dependencies_path"]
        # was hardcoded /home/ob001/refactored_qc/dependencies/...; now routed
        # through the configured dependencies path (same location on the host)
        self.comb_csv_path = f"/data/predict1/data_from_nda/formsdb/generated_outputs/combined/PROTECTED/"
        self.grouped_vars = self.utils.load_dependency_json(f"grouped_variables.json")
        self.form_per_var = self.grouped_vars["var_forms"]
        self.important_form_vars = self.utils.load_dependency_json(
        'important_form_vars.json')
        self.all_data_path = f"{self.depend_path}all_recalcs_combined.csv" # replace with new name of exported data

        self.all_data_df = pd.read_csv(
        self.all_data_path,
        keep_default_na = False)
        self.recalc_path = f"{self.depend_path}rule_h_like_discrepancies.csv"
        self.reclacs_df = pd.read_csv(self.recalc_path,
        keep_default_na = False)
        self.final_output_list = []
        self.revised_final_output_list = []
        self.diff_parts = []
        self.excluded_strings = ['__','complete']
        self.data_dict = self.utils.read_data_dictionary()

        self.calcs_only = self.data_dict[
        self.data_dict['Field Type'] == 'calc']

        self.all_calcs = self.calcs_only['Variable / Field Name'].tolist()

        raw_converted_bl = TransformBranchingLogic(
        self.data_dict).convert_all_branching_logic()
        self.converted_bl = {
            var: {
                'original_branching_logic': entry['original_branching_logic'],
                'converted_branching_logic': entry['converted_branching_logic'],
            }
            for var, entry in raw_converted_bl.items()
        }
        self.excl_bl = self.utils.load_dependency_json(
        'excluded_branching_logic_vars.json')

        # index the extract by arm-stripped event name once, instead of
        # re-scanning the full dataframe for every (network, timepoint) pair
        stripped_events = (self.all_data_df['redcap_event_name'].astype(str)
        .str.replace('_arm_1', '', regex=False)
        .str.replace('_arm_2', '', regex=False))
        self.all_data_by_tp = {tp: group for tp, group
        in self.all_data_df.groupby(stripped_events, sort=False)}

        self._column_plan_cache = {}
        self.bl_eval_failures = 0


    def run_script(self):
        self.loop_csvs() 

    def loop_csvs(self):
        tp_list = self.utils.create_timepoint_list()
        # create_timepoint_list only covers screening/baseline/month*;
        # floating and conversion combined CSVs exist for both networks
        # and were previously never compared
        tp_list.extend(['floating', 'conversion'])
        for network in ['PRESCIENT']:
           for tp in tp_list:
                tp = tp.replace("month","month_").replace("floating","floating_forms")
                print(f'{network} {tp}')
                comb_csv_file = (
                f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp}'
                f'_{network.replace("PRONET","ProNET")}-day1to1.csv')
                try:
                    combined_df = pd.read_csv(
                    comb_csv_file,
                    keep_default_na = False)
                except FileNotFoundError:
                    print(f'combined csv not found, skipping: {comb_csv_file}')
                    continue
                self.compare_all_data_revised(combined_df, tp, network)

                #self.merge_tps(combined_df, tp)
        if self.bl_eval_failures:
            print(f'branching logic eval raised {self.bl_eval_failures} '
            'times (those mismatches were dropped)')

    def _decimal_places(self, val):
        if 'e' in val.lower():
            # scientific notation: treat as fully precise
            return 10
        if '.' not in val:
            return 0
        return len(val.split('.', 1)[1])

    def _values_match(self, val_a, val_b):
        try:
            num_a = float(val_a)
            num_b = float(val_b)
        except (ValueError, TypeError):
            return val_a == val_b
        if abs(num_a - num_b) <= 1e-4:
            return True
        # ignore rounding/precision artifacts: the two exports can store
        # the same quantity at different decimal precision (46.7 vs
        # 46.66666667, 47 vs 46.5). Values that agree to within half a
        # unit in the last decimal place of the LESS precise side are
        # formatting differences, not calculation discrepancies.
        places = min(self._decimal_places(val_a), self._decimal_places(val_b))
        return abs(num_a - num_b) <= 0.5 * 10 ** -places + 1e-9

    def _branching_logic_passes(self, curr_row, col):
        # Lenient: keep mismatches when BL can't be evaluated reliably (no BL,
        # _info pattern, excluded var, var missing); drop only on definitive
        # False or eval-raise.
        if col not in self.converted_bl:
            return True
        bl = self.converted_bl[col]['converted_branching_logic']
        if bl == '' or '_info' in bl or col in self.excl_bl:
            return True
        try:
            return evaluate_branching_logic(
                bl, curr_row=curr_row, instance=self)
        except Exception:
            self.bl_eval_failures += 1
            return False

    def _network_completion_var(self, completion_var, network):
        # PRESCIENT records completion in the *_rpms variable; same
        # transform as utils.check_if_missing and
        # calculated_field_mismatches._prescient_completion_variable
        if network != 'PRESCIENT' or completion_var == '':
            return completion_var
        return ((completion_var + '_rpms').replace('_hc', '')
        .replace('onboarding', 'checkin')
        .replace('end_of_12month_study_pe', 'checkin')
        .replace('end_of_12month_study_p', 'checkin'))

    def _column_plan(self, comb_df, network):
        # per-column constants (form metadata, name filters) resolved once
        # per combined-CSV schema instead of once per row pair
        key = (network, tuple(comb_df.columns))
        plan = self._column_plan_cache.get(key)
        if plan is not None:
            return plan
        comb_cols = set(comb_df.columns)
        all_data_cols = set(self.all_data_df.columns)
        plan = []
        for col in self.all_calcs:
            if col not in comb_cols or col not in all_data_cols:
                continue
            if col not in self.form_per_var:
                continue
            if any(string in col for string in self.excluded_strings):
                continue
            form = self.form_per_var[col]
            form_meta = self.important_form_vars.get(form)
            if form_meta is None:
                # no metadata for this form (previously a KeyError):
                # compare without missing/completion gating
                plan.append((col, '', ''))
                continue
            missing_var = form_meta.get('missing_var', '')
            completion_var = self._network_completion_var(
            form_meta.get('completion_var', ''), network)
            plan.append((col, missing_var, completion_var))
        self._column_plan_cache[key] = plan
        return plan

    def _form_skipped(self, comb_row, tp, missing_var, completion_var):
        # missing button uses the plain variable name on both networks;
        # floating forms are never marked missing (utils.check_if_missing).
        # An empty missing_var now just skips the missing check instead of
        # excluding the whole form (matches should_skip_col / the pipeline).
        if (missing_var != '' and tp != 'floating_forms'
        and hasattr(comb_row, missing_var)
        and getattr(comb_row, missing_var) in [1,1.0,'1','1.0']):
            return True
        # completion_var is already network-resolved (_rpms on PRESCIENT,
        # where values 3/4 signal missingness and also fail the ==2 test)
        if (completion_var != ''
        and hasattr(comb_row, completion_var)
        and getattr(comb_row, completion_var) not in [2,2.0,'2','2.0']):
            return True
        return False

    def _write_revised_output(self):
        output_df = pd.DataFrame(self.revised_final_output_list)
        output_df.to_csv(
        f'{self.output_path}calc_discrepancies_redcap_r_code_prescient.csv',
        index = False)

    def compare_all_data_revised(self,comb_df,tp,network):
        # exact duplicate rows would double-report every mismatch;
        # conflicting duplicates are still all compared
        comb_df = comb_df.drop_duplicates()
        plan = self._column_plan(comb_df, network)
        tp_rows = self.all_data_by_tp.get(tp)
        if tp_rows is None or not plan:
            self._write_revised_output()
            return
        comb_by_sub = {sub: group for sub, group in comb_df.groupby('subjectid', sort=False)}
        for row in tp_rows.itertuples():
            sub = getattr(row,'chric_record_id')
            if sub not in comb_by_sub:
                continue
            filtered_df = comb_by_sub[sub]
            for comb_row in filtered_df.itertuples():
                # the skip decision is per (comb_row, form), not per column
                skip_by_form = {}
                for col, missing_var, completion_var in plan:
                    form_key = (missing_var, completion_var)
                    skip = skip_by_form.get(form_key)
                    if skip is None:
                        skip = self._form_skipped(
                        comb_row, tp, missing_var, completion_var)
                        skip_by_form[form_key] = skip
                    if skip:
                        continue

                    comb_val = str(getattr(comb_row, col))
                    recalc_val = str(getattr(row, col))
                    if self._values_match(comb_val.strip(), recalc_val.strip()):
                        continue
                    if comb_val in ['nan'] or len(comb_val) >= 100:
                        continue
                    if not self._branching_logic_passes(comb_row, col):
                        continue
                    self.revised_final_output_list.append({
                    'subject':sub,'timepoint':tp,'var':col,'comb_csv_val':
                    comb_val,'recalc_val':recalc_val})

        # write once per (network, timepoint) call, not once per row
        self._write_revised_output()


    def compare_all_data(self, comb_df, tp):
        key_col = "subjectid"

        # exact match on the arm-stripped event name; str.contains(tp)
        # made 'month_1' also match month_10/11/12/18
        stripped_events = (self.all_data_df["redcap_event_name"].astype(str)
        .str.replace('_arm_1', '', regex=False)
        .str.replace('_arm_2', '', regex=False))
        filtered_df = self.all_data_df[stripped_events == tp].copy()

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
        diff_df.to_csv(f"{self.output_path}diffs_test.csv", index=False)

    def merge_tps(self, comb_df, tp):
        print(tp)
        # tp arrives already transformed by loop_csvs ('month_1', ...);
        # re-replacing 'month' here produced 'month__1' and matched nothing.
        # Exact match on the arm-stripped event name; str.contains made
        # 'month_1' also match month_10/11/12/18
        stripped_events = (self.reclacs_df['redcap_event_name'].astype(str)
        .str.replace('_arm_1', '', regex=False)
        .str.replace('_arm_2', '', regex=False))
        filtered_df = self.reclacs_df[stripped_events == tp].copy()

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
                    hasattr(row, imp_var_name) and
                    getattr(row, imp_var_name) in [1,1.0,'1','1.0']):
                        skip = True
                        continue

                    # was 'complete_var', which never matched the loop
                    # values, so the completion skip never fired
                    if (imp_var == 'completion_var' and
                    hasattr(row, imp_var_name) and
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
