import pandas as pd
import os
import sys
import json
import re

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class PlusDosageMedCollector():
    """
    Class to collect every unique medication name
    in the current/past pharmaceutical treatment forms
    whose recorded dosage value contains a '+' symbol
    (combination-style dosing, e.g. '25+4'). Sweeps the
    per-timepoint combined CSVs for both networks and
    translates each course's medication dropdown code
    into its human readable name using the choices
    defined in the data dictionary.
    """

    # chrpharm_med{N}_name(_past) holds the medication dropdown code;
    # chrpharm_med{N}_dosage(_past) and chrpharm_med{N}_dosage_2(_past)
    # hold the dose recorded for that course.
    name_var_format = re.compile(r'chrpharm_med(\d+)_name(_past)?$')
    dosage_var_format = re.compile(r'chrpharm_med(\d+)_dosage(?:_2)?(_past)?$')

    def __init__(self, data_dict_df=None):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.comb_csv_path = self.config_info['paths']['combined_csv_path']

        if data_dict_df is None:
            data_dict_df = self.utils.read_data_dictionary()
        self.med_code_translations = self.collect_med_code_translations(
            data_dict_df)

        self.plus_dosage_meds = {}

    def __call__(self):
        self.collect_plus_dosage_meds()

        return self.format_output()

    def collect_med_code_translations(self, data_dict_df):
        """
        Builds a translation dictionary from medication
        dropdown codes to their human readable names,
        using the choices defined in the data dictionary
        for each chrpharm_med{N}_name(_past) variable.

        Parameters
        ------------
        data_dict_df : pd.DataFrame
            REDCap data dictionary

        Returns
        ------------
        med_code_translations : dict
            medication code mapped to its name
        """
        med_code_translations = {}
        col_renames = {'Variable / Field Name':'var',
        'Choices, Calculations, OR Slider Labels':'choices'}
        filtered_data_dict = data_dict_df[list(col_renames.keys())]
        filtered_data_dict = filtered_data_dict.rename(columns=col_renames)
        for row in filtered_data_dict.itertuples():
            if not self.name_var_format.match(str(row.var)):
                continue
            # REDCap choices format: "code, label | code, label | ..."
            for choice in str(row.choices).split('|'):
                if ',' not in choice:
                    continue
                code, med_name = choice.split(',', 1)
                med_code_translations.setdefault(
                    code.strip(), med_name.strip())

        return med_code_translations

    def collect_plus_dosage_meds(self):
        """
        Loops through every per-timepoint combined CSV
        for both networks and collects each medication
        course whose dosage value contains a '+' symbol.
        """
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        for network in ['PRONET','PRESCIENT']:
            for tp in tp_list:
                csv_path = (
                    f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                    f'{tp.replace("month","month_").replace("floating","floating_forms")}'
                    f'_{network.replace("PRONET","ProNET")}-day1to1.csv'
                )
                # dtype=str so dosage values keep their literal text --
                # the default parser would read a value like '+10' as
                # the number 10 and drop the '+' before it can be seen.
                try:
                    combined_df = pd.read_csv(
                        csv_path, keep_default_na=False, dtype=str,
                        usecols=lambda c: (
                            self.dosage_var_format.match(c) is not None
                            or self.name_var_format.match(c) is not None))
                except FileNotFoundError:
                    continue
                self.collect_csv_plus_dosages(combined_df)

    def collect_csv_plus_dosages(self, combined_df):
        """
        Records every medication course in one combined
        CSV whose dosage value contains a '+' symbol.

        Parameters
        ------------
        combined_df : pd.DataFrame
            combined CSV filtered down to the
            medication name and dosage columns
        """
        dosage_name_pairs = []
        for col in combined_df.columns:
            dosage_match = self.dosage_var_format.match(col)
            if dosage_match is None:
                continue
            med_num = dosage_match.group(1)
            past_suffix = dosage_match.group(2) or ''
            dosage_name_pairs.append(
                (col, f'chrpharm_med{med_num}_name{past_suffix}'))
        if not dosage_name_pairs:
            return
        for row in combined_df.itertuples():
            for dosage_var, name_var in dosage_name_pairs:
                dosage_val = str(getattr(row, dosage_var)).strip()
                if '+' not in dosage_val:
                    continue
                med_code = str(getattr(row, name_var, '')).strip()
                self.record_plus_dosage(med_code, dosage_var, dosage_val)

    def record_plus_dosage(self, med_code, dosage_var, dosage_val):
        """
        Adds one '+' dosage medication course to the
        collected output, keyed by the medication's
        human readable name.

        Parameters
        ------------
        med_code : str
            value of the course's name variable
        dosage_var : str
            dosage variable that contains the '+'
        dosage_val : str
            recorded dosage value
        """
        med_code = self.normalize_med_code(med_code)
        if med_code == '':
            med_name = 'no medication name recorded'
        elif med_code in self.med_code_translations:
            med_name = self.med_code_translations[med_code]
        else:
            med_name = f'unrecognized medication code ({med_code})'
        med_entry = self.plus_dosage_meds.setdefault(med_name, {
            'med_codes' : set(), 'dosage_vars' : set(),
            'dosage_vals' : set(), 'n_courses' : 0})
        if med_code != '':
            med_entry['med_codes'].add(med_code)
        med_entry['dosage_vars'].add(dosage_var)
        med_entry['dosage_vals'].add(dosage_val)
        med_entry['n_courses'] += 1

    def normalize_med_code(self, med_code):
        """
        Strips a trailing '.0' left by float round
        trips (e.g. '146.0' -> '146') so codes match
        the data dictionary choices.

        Parameters
        ------------
        med_code : str
            value of the course's name variable

        Returns
        ------------
        med_code : str
            normalized medication code
        """
        if med_code.endswith('.0') and med_code[:-2].isdigit():
            return med_code[:-2]

        return med_code

    def format_output(self):
        """
        Converts the collected sets into sorted lists
        so the output is JSON serializable and
        deterministic across runs.

        Returns
        ------------
        formatted_output : dict
            each unique medication name mapped to the
            codes, dosage variables, and dosage values
            it was collected with
        """
        formatted_output = {}
        for med_name in sorted(self.plus_dosage_meds):
            med_entry = self.plus_dosage_meds[med_name]
            formatted_output[med_name] = {
            'med_codes' : sorted(med_entry['med_codes']),
            'dosage_vars' : sorted(med_entry['dosage_vars']),
            'dosage_vals' : sorted(med_entry['dosage_vals']),
            'n_courses' : med_entry['n_courses']}

        return formatted_output


if __name__ == '__main__':
    print(json.dumps(PlusDosageMedCollector()(), indent=4))
