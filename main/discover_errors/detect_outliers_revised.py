import os
import sys
import re
import pandas as pd

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils


class NumericalOutliers:
    """
    Detect potential numerical outliers using the IQR rule and
    output continuous severity scores.

    Output includes:
    - iqr_score: distance beyond Tukey fence / IQR, 0 if not beyond fence
    - median_iqr_score: absolute distance from median / IQR for all values
    """

    def __init__(self):
        self.absolute_path = '/PHShome/ob001/anaconda3/new_forms_qc/QC/'
        self.combined_df_folder = '/data/predict1/data_from_nda/formsdb/generated_outputs/combined/PROTECTED/'

        self.utils = Utils()

        self.likely_missing_vals = [
            '', 'NA', 'na', 'N/A', 'n/a',
            'nan', 'NAN', 'NaN', 'nil'
        ]

        self.data_dict = self.utils.read_data_dictionary()
        self.data_dict.columns = self.data_dict.columns.str.replace(' ', '_')
        self.var_field_types = self.data_dict.set_index(
            "Variable_/_Field_Name"
        )["Field_Type"].to_dict()

        self.form_info = self.utils.load_dependency_json('grouped_variables.json')
        self.forms_per_var = self.form_info['var_forms']
        self.blood_vars = self.form_info['blood_vars']
        self.excl_blood_vars = (
        self.blood_vars['position_variables'] + 
        self.blood_vars['barcode_variables'] + 
        self.blood_vars['id_variables'])
 


        self.excluded_field_types = ['checkbox', 'dropdown', 'radio', 'yesno']

        self.output_list = []

    def run_script(self) -> None:
        self.loop_timepoints()

    def loop_timepoints(self) -> None:
        """
        Loop through each timepoint for each network, detect outliers,
        and save all results to one CSV.
        """
        timepoint_list = self.utils.create_timepoint_list()

        for network in ['PRONET', 'PRESCIENT']:
            for timepoint in timepoint_list:
                print(network)
                print(timepoint)

                file_path = (
                    f'{self.combined_df_folder}AMPSCZ-combined-redcap_'
                    f'{timepoint.replace("month", "month_").replace("floating", "floating_forms")}'
                    f'_{network.replace("PRONET", "ProNET")}-day1to1.csv'
                )

                try:
                    df = pd.read_csv(
                        file_path,
                        keep_default_na=False,
                        on_bad_lines='skip'
                    )
                except Exception as exc:
                    print(f'Could not read {file_path}: {exc}')
                    continue

                numeric_vars = self.discover_numerical_variables(df)
                iqr_stats = self.collect_iqr_stats(df, numeric_vars)
                self.detect_outliers_iqr(df, timepoint, network, iqr_stats)

        output_df = pd.DataFrame(self.output_list)
        output_df.to_csv('outliers_iqr_full.csv', index=False)
        print(f'Saved {len(output_df)} rows to outliers_iqr.csv')

    def is_missing_value(self, value) -> bool:
        return value in (self.utils.missing_code_list + self.likely_missing_vals)

    def parse_numeric_value(self, value):
        """
        Try to extract a numeric value from a cell.
        Returns float or None.
        """
        if self.is_missing_value(value):
            return None

        value_str = str(value).strip()
        if not value_str:
            return None

        #value_str = value_str.replace(',', '')

        if self.utils.check_if_number(value_str):
            return float(value_str)

        first_token = value_str.split(' ')[0]
        if self.utils.check_if_number(first_token):
            return float(first_token)

        #match = re.match(r'^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$', first_token)
        #if match:
        #    try:
        #        return float(first_token)
        #    except ValueError:
        #        return None

        return None

    def discover_numerical_variables(self, df: pd.DataFrame, threshold: float = 0.9) -> list:
        """
        Identify columns that are usually numeric.
        """
        dtype_count = {}

        for col in df.columns:
            if (
                (col in self.var_field_types and self.var_field_types[col] in self.excluded_field_types)
                or col not in self.var_field_types
            ):
                continue

            dtype_count[col] = {'number': 0, 'non_number': 0}

            for value in df[col]:
                if self.is_missing_value(value):
                    continue

                parsed = self.parse_numeric_value(value)
                if parsed is not None:
                    dtype_count[col]['number'] += 1
                else:
                    dtype_count[col]['non_number'] += 1

        numeric_vars = []
        for var, counts in dtype_count.items():
            total = counts['number'] + counts['non_number']
            if total == 0:
                continue

            numeric_fraction = counts['number'] / total
            if numeric_fraction >= threshold:
                numeric_vars.append(var)

        return numeric_vars

    def collect_iqr_stats(self, df: pd.DataFrame, var_list: list, min_n: int = 10) -> dict:
        """
        For each numeric variable, compute robust distribution stats.
        """
        iqr_stats = {}

        for var in var_list:
            numeric_values = df[var].apply(self.parse_numeric_value).dropna()

            if len(numeric_values) < min_n:
                continue

            q1 = numeric_values.quantile(0.25)
            median = numeric_values.median()
            q3 = numeric_values.quantile(0.75)
            iqr = q3 - q1

            if pd.isna(iqr) or iqr == 0:
                continue

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            iqr_stats[var] = {
                'n': int(len(numeric_values)),
                'q1': float(q1),
                'median': float(median),
                'q3': float(q3),
                'iqr': float(iqr),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
            }

        return iqr_stats

    def calculate_iqr_scores(self, x: float, stats: dict) -> tuple:
        """
        Calculate continuous outlier severity measures.

        Returns
        -------
        iqr_score : float
            Distance beyond Tukey fence / IQR.
            0.0 if x is inside the Tukey fences.

        median_iqr_score : float
            Absolute distance from median / IQR.
            Continuous for all values, even non-outliers.
        """
        iqr = stats['iqr']
        median = stats['median']
        lower_bound = stats['lower_bound']
        upper_bound = stats['upper_bound']

        if iqr == 0:
            return 0.0, 0.0

        if x < lower_bound:
            iqr_score = (lower_bound - x) / iqr
        elif x > upper_bound:
            iqr_score = (x - upper_bound) / iqr
        else:
            iqr_score = 0.0

        median_iqr_score = abs(x - median) / iqr

        return float(iqr_score), float(median_iqr_score)

    def detect_outliers_iqr(
        self,
        df: pd.DataFrame,
        timepoint: str,
        network: str,
        iqr_stats: dict
    ) -> None:
        """
        Score each row for each variable using IQR-based statistics.

        Writes every numeric value for variables with valid IQR stats.
        Adds:
        - is_outlier_iqr
        - iqr_score
        - median_iqr_score
        """
        seen_records = set()

        for var, stats in iqr_stats.items():
            if var not in df.columns:
                continue
            if var in self.excl_blood_vars:
                continue

            for idx, raw_value in df[var].items():
                parsed_value = self.parse_numeric_value(raw_value)
                if parsed_value is None:
                    continue

                iqr_score, median_iqr_score = self.calculate_iqr_scores(parsed_value, stats)
                is_outlier_iqr = parsed_value < stats['lower_bound'] or parsed_value > stats['upper_bound']

                subject = df.at[idx, 'subjectid'] if 'subjectid' in df.columns else None
                form = self.forms_per_var.get(var)

                record_key = (subject, timepoint, network, var, parsed_value, idx)
                if record_key in seen_records:
                    continue
                seen_records.add(record_key)

                if is_outlier_iqr:
                    print('--------------------')
                    print(f'Variable: {var}')
                    print(f'Value: {parsed_value}')
                    print(f'Median: {stats["median"]}')
                    print(f'Q1: {stats["q1"]}')
                    print(f'Q3: {stats["q3"]}')
                    print(f'IQR: {stats["iqr"]}')
                    print(f'Lower bound: {stats["lower_bound"]}')
                    print(f'Upper bound: {stats["upper_bound"]}')
                    print(f'IQR score: {iqr_score}')
                    print(f'Median IQR score: {median_iqr_score}')

                self.output_list.append({
                    'subject': subject,
                    'timepoint': timepoint,
                    'network': network,
                    'variable': var,
                    'form': form,
                    'row_index': idx,
                    'raw_value': raw_value,
                    'var_value': parsed_value,
                    'n': stats['n'],
                    'q1': stats['q1'],
                    'median': stats['median'],
                    'q3': stats['q3'],
                    'iqr': stats['iqr'],
                    'lower_bound': stats['lower_bound'],
                    'upper_bound': stats['upper_bound'],
                    'is_outlier_iqr': is_outlier_iqr,
                    'iqr_score': iqr_score,
                    'median_iqr_score': median_iqr_score,
                })


if __name__ == '__main__':
    NumericalOutliers().run_script()