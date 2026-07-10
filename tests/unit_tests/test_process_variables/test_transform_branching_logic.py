import pandas as pd
import os
import sys
import json
import unittest

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-4])
sys.path.insert(1, parent_dir)
from main.process_variables.transform_branching_logic import TransformBranchingLogic
from main.utils.utils import Utils
from logging_config import logger  
import re

class TestTransformBranchingLogic(unittest.TestCase):

    def __init__(self):
        self.utils = Utils()
        self.data_dictionary_df = self.utils.read_data_dictionary()
        self.transform_branching_logic = TransformBranchingLogic(self.data_dictionary_df)
        self.convert_bl = self.transform_branching_logic.convert_all_branching_logic

        self.miss_codes = self.utils.missing_code_list

    def run_script(self):
        self.test_convert_all_branching_logic()

    def test_convert_all_branching_logic(self):
        #TODO: check for indentifiers in branching logic
        converted_bl = self.convert_bl()
     

if __name__ =='__main__':
    TestTransformBranchingLogic().run_script()
