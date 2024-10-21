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

class TestTransformBranchingLogic(unittest.TestCase):

    def __init__(self):
        self.utils = Utils()
        self.data_dictionary_df = self.utils.read_data_dictionary()
        self.transform_branching_logic = TransformBranchingLogic(self.data_dictionary_df)

    def run_script(self):
        self.test_branching_logic_redcap_to_python()

    def test_branching_logic_redcap_to_python(self):
        self.data_dictionary_df = self.data_dictionary_df.rename(columns={'Variable / Field Name': 'variable',
        'Branching Logic (Show field only if...)': 'branching_logic'})

        for row in self.data_dictionary_df.itertuples():
            var = getattr(row, 'variable')
            branching_logic = getattr(row, 'branching_logic')
            converted_bl = ''
            if branching_logic !='':
                converted_bl = self.transform_branching_logic.branching_logic_redcap_to_python(branching_logic)
            
            if 'float' not in converted_bl:
                print(converted_bl)
            else:
                print(converted_bl.replace(' ','').split('float'))
    

if __name__ =='__main__':
    TestTransformBranchingLogic().run_script()
