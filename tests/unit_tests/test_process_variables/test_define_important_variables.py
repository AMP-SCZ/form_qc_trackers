import pandas as pd
import os
import sys
import json
import unittest

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from main.process_variables.define_important_variables import DefineImportantVariables
from main.utils.utils import Utils
class TestDefineVariables(unittest.TestCase):
    def __init__(self,data_dictionary_df):
        important_form_vars = DefineImportantVariables(data_dictionary_df)
        self.data_dictionary_df = data_dictionary_df

    def run_script(self):
        self.test_collect_important_vars()

    def test_assign_variables_to_forms(self):
        all_unique_vars = self.important_form_vars.collect_important_vars() 
        all_forms = list(set(self.data_dictionary_df['Form Name'].tolist()))
        important_form_vars = self.important_form_vars.assign_variables_to_forms(all_unique_vars)




