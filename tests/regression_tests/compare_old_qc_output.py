import pandas as pd
import os 
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)
from main.utils.utils import Utils
from logging_config import logger  

class CompareOldQC():

    def __init__(self):
        self.old_output_dir = '/PHShome/ob001/anaconda3/new_forms_qc/QC/site_outputs/'
        self.new_output_dir = ('/PHShome/ob001/anaconda3/refactored_qc/'
        'output/formatted_outputs/dropbox_files/')
        self.old_output_fullpaths = {}
        self.new_output_fullpaths = {}
        self.networks = ['PRONET','PRESCIENT']
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.output_path = self.config_info['paths']['output_path']

        
        for network in self.networks:
            self.old_output_fullpaths[network] = (f'{self.old_output_dir}{network}'
                                                  f'/combined/{network}_Output.xlsx')
            self.new_output_fullpaths[network] = (f'{self.new_output_dir}{network}'
                                                  f'/combined/{network}_Output_v2.xlsx')
    def run_script(self):
        for network in self.networks:
            old_df = pd.read_excel(
            self.old_output_fullpaths[network],
            keep_default_na = True)
            old_df = self.format_old_df_forms(old_df)

            # modifies old output to be in same format
            old_df.rename(columns={'Subject': 'Participant'},
            inplace=True)

            new_df = pd.read_excel(
            self.new_output_fullpaths[network],
            keep_default_na = True)

            self.compare_outputs(old_df, new_df, network)

    def compare_outputs(self, old_df, new_df, network):
        merged = pd.merge(new_df, old_df,
        on = ['Participant','Timepoint','Form'], how = 'outer')
        merged.to_csv(f'{self.output_path}{network}_merged.csv', index = False)
    
    def format_old_df_forms(self, df):
        df['Form'] = df['General Flag'].apply(
        lambda x: (x.split(' :')[0]).strip().replace(' ','_').lower())
        df['Timepoint'] = df['Timepoint'].apply(
        lambda x: (x.replace('baseln','baseline').replace('screen','screening')))
        print(df['Form'])

        return df


if __name__=='__main__':
    CompareOldQC().run_script()