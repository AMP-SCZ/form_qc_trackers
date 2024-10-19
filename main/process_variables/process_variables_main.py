import pandas as pd
import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class ProcessVariables():
    
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        print(self.absolute_path)

    def run_script(self):
        pass



if __name__ == '__main__':
    ProcessVariables().run_script()

        