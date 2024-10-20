import pandas as pd
import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from main.process_variables.define_important_variables import DefineVariables

class TestDefineVariables():
    def __init__(self,data_dictionary_df):
        self.define_variables = DefineVariables(data_dictionary_df)
        self.data_dictionary_df = data_dictionary_df

    def run_script(self):
        self.test_collect_important_vars()

    def test_collect_important_vars(self):
        self.define_variables.run_script()
        all_forms = list(set(self.data_dictionary_df['Form Name'].tolist()))
        for var_list in [self.define_variables.missing_form_variables,\
        self.define_variables.interview_date_variables,\
        self.define_variables.entry_date_variables]:
            for form in all_forms:
                if form not in var_list:
                    print(form)
                    print(var_list)



