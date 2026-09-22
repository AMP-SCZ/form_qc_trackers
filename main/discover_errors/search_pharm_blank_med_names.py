"""Pulls the comments from the pharmaceutical treatment forms (past
pharmaceutical treatment in the screening combined CSVs, both current
floating med forms in the floating forms combined CSVs) for any medication
entry whose medication name is blank. A name is treated as blank when it is
an empty/NaN string or an explicit numeric missing code (-3/-9/-99), but
NOT when it is one of the meaningful dropdown codes (999 = no meds, 888 =
no information, 777/333/444/555/666 = name unknown; MED-QC-11/12 already
cover 888/999 paired with data). Only entries with actual comment text are
pulled, since the comment is the deliverable - comments that are blank or
hold only a missing code are skipped. The name value and the slot's
add-radio value are included so blank-string and coded-missing names can
be told apart.
"""
import pandas as pd

import os
import re
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(1, parent_dir)

from utils.utils import Utils


class SearchPharmBlankMedNames():
    def __init__(self, comb_csv_path=None, output_path=None):
        self.utils = Utils()
        self.config_info = self.utils.config_info
        self.comb_csv_path = (comb_csv_path if comb_csv_path is not None
        else self.config_info['paths']['combined_csv_path'])
        self.output_path = (output_path if output_path is not None
        else self.config_info['paths']['output_path'])
        self.important_form_vars = self.utils.load_dependency_json(
        'important_form_vars.json')
        self.pharm_forms = [
        {'form': 'past_pharmaceutical_treatment',
        'timepoint': 'screening', 'past': True},
        {'form': 'current_pharmaceutical_treatment_floating_med_125',
        'timepoint': 'floating', 'past': False},
        {'form': 'current_pharmaceutical_treatment_floating_med_2650',
        'timepoint': 'floating', 'past': False}]
        self.blank_strings = ['', 'nan', 'nat', 'none', 'null']
        # 999 is a meaningful name code (no meds), not a missing code here
        self.non_blank_codes = [str(val)
        for val in self.utils.all_dtype([999])]
        self.collect_med_slots()
        self.flagged_comments = []
        self.count_per_form = {}

    def run_script(self):
        self.loop_csvs()
        self.write_output()

    def med_var(self, med_num, suffix, past):
        base_var = f'chrpharm_med{med_num}_{suffix}'
        return f'{base_var}_past' if past else base_var

    def get_slot_open_var(self, med_num, past):
        if med_num <= 1:
            return 'chrpharm_med_past' if past else 'chrpharm_med'
        return self.med_var(med_num - 1, 'add', past)

    def collect_med_slots(self):
        data_dict_df = self.utils.read_data_dictionary()
        name_var_format = re.compile(r'chrpharm_med(\d+)_name(_past)?$')
        for form_info in self.pharm_forms:
            print(self.pharm_forms)
            form_rows = data_dict_df[
            data_dict_df['Form Name'] == form_info['form']]
            form_vars = set(form_rows['Variable / Field Name'])
            med_nums = []
            for form_var in form_vars:
                name_match = name_var_format.fullmatch(form_var)
                if name_match is None:
                    continue

                med_num = int(name_match.group(1))

                if (form_var == self.med_var(med_num, 'name',
                form_info['past']) and self.med_var(med_num, 'comments',
                form_info['past']) in form_vars):
                    med_nums.append(med_num)
            if med_nums == []:
                raise ValueError(
                f'No medication name variables found for '
                f'{form_info["form"]} in the data dictionary.')
            form_info['med_nums'] = sorted(med_nums)

    def loop_csvs(self):
        for network in ['PRONET', 'PRESCIENT']:
            for timepoint in ['screening', 'floating']:
                combined_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{timepoint.replace("floating","floating_forms")}'
                f'_{network.replace("PRONET","ProNET")}-day1to1.csv'),
                keep_default_na = False, dtype = str,
                encoding = 'utf-8-sig')
                tp_forms = [form_info for form_info in self.pharm_forms
                if form_info['timepoint'] == timepoint]
                self.search_combined_df(
                combined_df, network, timepoint, tp_forms)

    def search_combined_df(self, combined_df, network, timepoint, tp_forms):
        completion_vars = {}
        for form_info in tp_forms:
            completion_var = self.important_form_vars[
            form_info['form']]['completion_var']
            if network == 'PRESCIENT':
                completion_var += '_rpms'
            completion_vars[form_info['form']] = completion_var
        for row in combined_df.itertuples():
            for form_info in tp_forms:
                for med_num in form_info['med_nums']:
                    self.check_med_slot(row, med_num, form_info,
                    network, timepoint, completion_vars[form_info['form']])

    def check_med_slot(self, row, med_num, form_info, network,
    timepoint, completion_var):
        past = form_info['past']
        name_var = self.med_var(med_num, 'name', past)
        comment_var = self.med_var(med_num, 'comments', past)
        if not hasattr(row, name_var) or not hasattr(row, comment_var):
            return
        name_val = getattr(row, name_var)
        if not self.is_blank_med_name(name_val):
            return
        comment_val = getattr(row, comment_var)
        if not self.has_comment_text(comment_val):
            return
        slot_open_var = self.get_slot_open_var(med_num, past)
        self.count_per_form.setdefault((network, form_info['form']), 0)
        self.count_per_form[(network, form_info['form'])] += 1
        self.flagged_comments.append({'subject': row.subjectid,
        'network': network, 'timepoint': timepoint,
        'form': form_info['form'], 'med_number': med_num,
        'name_var': name_var, 'name_value': name_val,
        'slot_open_var': slot_open_var,
        'slot_open_value': (getattr(row, slot_open_var)
        if hasattr(row, slot_open_var) else ''),
        'completion_var_value': (getattr(row, completion_var)
        if hasattr(row, completion_var) else ''),
        'comment_var': comment_var, 'comment': comment_val})

    def is_blank_med_name(self, name_val):
        name_str = str(name_val).strip()
        if name_str.lower() in self.blank_strings:
            return True
        if name_str in self.non_blank_codes:
            return False
        return name_str in self.utils.missing_code_set

    def has_comment_text(self, comment_val):
        comment_str = str(comment_val).strip()
        if comment_str.lower() in self.blank_strings:
            return False
        return comment_str not in self.utils.missing_code_set

    def write_output(self):
        flagged_df = pd.DataFrame(self.flagged_comments)
        flagged_df.to_csv(
        f'{self.output_path}pharm_blank_med_name_comments.csv',
        index = False)
        count_per_form_list = []
        for (network, form), count in self.count_per_form.items():
            count_per_form_list.append({'network': network,
            'form': form, 'count': count})
        count_per_form_df = pd.DataFrame(count_per_form_list)
        count_per_form_df.to_csv(
        f'{self.output_path}pharm_blank_med_name_comments_per_form.csv',
        index = False)
        print(f'{len(flagged_df.index)} blank medication name comments '
        f'written to {self.output_path}pharm_blank_med_name_comments.csv')


if __name__ == '__main__':
    SearchPharmBlankMedNames().run_script()
