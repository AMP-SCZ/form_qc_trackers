import pandas as pd
import os 
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
from process_variables.transform_branching_logic import TransformBranchingLogic

from utils.utils import Utils
import re
import pandas as pd 


class DataGenerator():

    