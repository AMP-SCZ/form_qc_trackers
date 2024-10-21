import logging
import sys
import json
import os
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-1])
sys.path.insert(1, parent_dir)
print('-----')
print(parent_dir)
from main.utils.utils import Utils

utils = Utils()
absolute_path = utils.absolute_path

with open(f'{absolute_path}/config.json','r') as file:
    config_info = json.load(file)

logger = logging.getLogger('unit_test_logger')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
f'{config_info["paths"]["error_log_path"]}test_errors.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
