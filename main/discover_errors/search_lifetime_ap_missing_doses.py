"""Searches the lifetime AP exposure form in the screening combined CSVs
for dosage fields (dose/day and days/course) that hold an explicit missing
code while the medication is marked as having been taken at some point
(chrap_<med> = 1). Blank values are excluded on purpose: with branched
fields a blank only means the branch never opened, and the pipeline's
general blank check already covers entered-but-blank fields. Note 999 is in
the project-wide missing code list but can be a real mg dose for a few high
dose antipsychotics, so review flagged 999 values before acting on them.
The same caveat applies to 999 in the days/course fields, where it could be
a real duration. Complements the AP-001/AP-002/AP-003 pipeline rules, which
cover data present when Ever = No, invalid course counts, data beyond the
declared course count, and non-positive doses.
"""
import pandas as pd

import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(1, parent_dir)

from utils.utils import Utils


class SearchLifetimeAPMissingDoses():
    def __init__(self, comb_csv_path=None, output_path=None):
        self.utils = Utils()
        self.config_info = self.utils.config_info
        self.comb_csv_path = (comb_csv_path if comb_csv_path is not None
        else self.config_info['paths']['combined_csv_path'])
        self.output_path = (output_path if output_path is not None
        else self.config_info['paths']['output_path'])
        self.important_form_vars = self.utils.load_dependency_json(
        'important_form_vars.json')
        self.form = 'lifetime_ap_exposure_screen'
        self.form_info = self.important_form_vars[self.form]
        self.numeric_missing_codes = {
        float(code) for code in self.utils.missing_code_list
        if self.utils.can_be_float(code)}
        self.med_blocks = self.collect_med_blocks()
        self.flagged_values = []
        self.count_per_med = {}

    def run_script(self):
        self.loop_csvs()
        self.write_output()

    def collect_med_blocks(self):
        data_dict_df = self.utils.read_data_dictionary()
        ap_form_rows = data_dict_df[
        data_dict_df['Form Name'] == self.form]
        ap_form_vars = set(ap_form_rows['Variable / Field Name'])
        label_prefix = 'have you ever taken'
        med_blocks = []
        for _, dd_row in ap_form_rows.iterrows():
            label = str(dd_row['Field Label']).strip()
            if (dd_row['Field Type'] != 'yesno'
            or not label.lower().startswith(label_prefix)):
                continue
            yes_var = dd_row['Variable / Field Name']
            med_name = label[len(label_prefix):].strip().rstrip('?').strip()
            course_var = f'{yes_var}course'
            if course_var not in ap_form_vars:
                continue
            course_choices = ap_form_rows.loc[
            ap_form_rows['Variable / Field Name'] == course_var,
            'Choices, Calculations, OR Slider Labels'].iloc[0]
            max_courses = self.parse_max_courses(course_var, course_choices)
            dosage_vars_per_course = {}
            for course_num in range(1, max_courses + 1):
                dosage_vars_per_course[course_num] = [
                f'{yes_var}{dosage_str}{course_num}'
                for dosage_str in ['dose', 'days']
                if f'{yes_var}{dosage_str}{course_num}' in ap_form_vars]
            med_blocks.append({'med_name': med_name, 'yes_var': yes_var,
            'course_var': course_var, 'max_courses': max_courses,
            'dosage_vars_per_course': dosage_vars_per_course})
        if med_blocks == []:
            raise ValueError(
            f'No medication blocks found for {self.form} '
            'in the data dictionary.')
        return med_blocks

    def parse_max_courses(self, course_var, course_choices):
        course_codes = [choice.split(',')[0].strip()
        for choice in str(course_choices).split('|')]
        course_codes = [int(float(code)) for code in course_codes
        if self.utils.can_be_float(code)]
        if course_codes == []:
            raise ValueError(
            f'Could not parse course choices for {course_var}: '
            f'{course_choices}')
        return max(course_codes)

    def loop_csvs(self):
        for network in ['PRONET', 'PRESCIENT']:
            screening_df = pd.read_csv(
            (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
            f'screening_{network.replace("PRONET","ProNET")}-day1to1.csv'),
            keep_default_na = False, low_memory = False,
            encoding = 'utf-8-sig')
            self.search_screening_df(screening_df, network)

    def search_screening_df(self, screening_df, network):
        missing_var = self.form_info['missing_var']
        completion_var = self.form_info['completion_var']
        if network == 'PRESCIENT':
            completion_var += '_rpms'
        for row in screening_df.itertuples():
            if (hasattr(row, missing_var) and
            getattr(row, missing_var) in self.utils.all_dtype([1])):
                continue
            # prescient missingness can also be indicated by the completion var
            if (network == 'PRESCIENT' and hasattr(row, completion_var)
            and getattr(row, completion_var) in self.utils.all_dtype([3, 4])):
                continue
            for med_block in self.med_blocks:
                self.check_med_block(row, med_block, network, completion_var)

    def check_med_block(self, row, med_block, network, completion_var):
        yes_var = med_block['yes_var']
        if not hasattr(row, yes_var):
            return
        if getattr(row, yes_var) not in self.utils.all_dtype([1]):
            return
        course_var = med_block['course_var']
        course_val = (getattr(row, course_var)
        if hasattr(row, course_var) else '')
        declared_courses = self.parse_declared_courses(
        course_val, med_block['max_courses'])
        for course_num, dosage_vars in med_block[
        'dosage_vars_per_course'].items():
            for dosage_var in dosage_vars:
                if not hasattr(row, dosage_var):
                    continue
                dosage_val = getattr(row, dosage_var)
                if not self.is_missing_code(dosage_val):
                    continue
                self.count_per_med.setdefault(
                (network, med_block['med_name']), 0)
                self.count_per_med[(network, med_block['med_name'])] += 1
                self.flagged_values.append({'subject': row.subjectid,
                'network': network, 'timepoint': 'screening',
                'form': self.form, 'medication': med_block['med_name'],
                'var': dosage_var, 'var_value': dosage_val,
                'ever_taken_var': yes_var,
                'course_var_value': course_val,
                'course_number': course_num,
                'within_declared_courses': (declared_courses is not None
                and course_num <= declared_courses),
                'completion_var_value': (getattr(row, completion_var)
                if hasattr(row, completion_var) else '')})

    def parse_declared_courses(self, course_val, max_courses):
        if not self.utils.can_be_float(course_val):
            return None
        try:
            declared_courses = int(float(course_val))
        # can_be_float accepts 'nan'/'inf', which int() cannot convert
        except (ValueError, OverflowError):
            return None
        if 1 <= declared_courses <= max_courses:
            return declared_courses
        return None

    def is_missing_code(self, value):
        if value in self.utils.missing_code_set:
            return True
        value_str = str(value).strip()
        if value_str in self.utils.missing_code_set:
            return True
        if self.utils.can_be_float(value_str):
            return float(value_str) in self.numeric_missing_codes
        return False

    def write_output(self):
        flagged_df = pd.DataFrame(self.flagged_values)
        flagged_df.to_csv(
        f'{self.output_path}lifetime_ap_dose_missing_codes.csv',
        index = False)
        count_per_med_list = []
        for (network, med_name), count in self.count_per_med.items():
            count_per_med_list.append({'network': network,
            'medication': med_name, 'count': count})
        count_per_med_df = pd.DataFrame(count_per_med_list)
        count_per_med_df.to_csv(
        f'{self.output_path}lifetime_ap_dose_missing_codes_per_med.csv',
        index = False)
        print(f'{len(flagged_df.index)} missing code dosage values written '
        f'to {self.output_path}lifetime_ap_dose_missing_codes.csv')


if __name__ == '__main__':
    SearchLifetimeAPMissingDoses().run_script()
