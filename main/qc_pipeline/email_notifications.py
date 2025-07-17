import pandas as pd
import os 
import sys
import json
from datetime import datetime
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class EmailNotifications():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        print(self.absolute_path)
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.depen_path = self.config_info['paths']['dependencies_path']
        self.output_path = self.config_info['paths']['output_path']

    def run_script(self):
        self.check_output_files()

    def send_email(self, subject_line, message, recipient):
        result = subprocess.run([
        f'{self.absolute_path}/main/qc_pipeline/send_email.sh'] + \
        [message,recipient,subject_line], capture_output=True, text=True)

    def check_output_files(self):
        for root, dirs, files in os.walk(self.output_path, topdown=False):
            for name in files:
                if name.endswith('.csv') or name.endswith('.xlsx'):
                    fullpath = os.path.join(root, name)
                    mod_time = os.path.getmtime(fullpath)
                    mod_datetime = datetime.fromtimestamp(mod_time)
                    print(mod_datetime)
                    days = self.utils.days_since_today(str(mod_datetime).split(' ')[0])
                    if days > 1:
                        print(days)

    def check_dependencies(self):
        for file in os.listdir(self.depen_path):
            print(file)
            mod_time = os.path.getmtime(self.depen_path + file)
            mod_datetime = datetime.fromtimestamp(mod_time)
            print(mod_datetime)
        
if __name__ =='__main__':
    EmailNotifications().run_script()