import pandas as pd
import os
import sys
import json

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class RangeDefiner():
    def __init__(self):
        self.utils = Utils()
        self.ranges_dict = { "PRONET":
            {'chrpps_mage': {'min':10,'max':85, 'form':'psychosis_polyrisk_score'},
            'chrpps_fage': {'min':10,'max':85, 'form':'psychosis_polyrisk_score'},
            'chrfigs_mother_age': {'min':10,'max':85,
            'form':'family_interview_for_genetic_studies_figs'},
            'chrfigs_father_age': {'min':10,'max':85,
            'form':'family_interview_for_genetic_studies_figs'},
            'chrchs_weightkg': {'min':0,'max':300,
            'form':'current_health_status'}},

            "PRESCIENT": {'chrpps_mage': {'min':10,'max':85, 'form':'psychosis_polyrisk_score'},
            'chrpps_fage': {'min':10,'max':85, 'form':'psychosis_polyrisk_score'},
            'chrfigs_mother_age': {'min':10,'max':85,
            'form':'family_interview_for_genetic_studies_figs'},
            'chrchs_weightkg': {'min':0,'max':300,
            'form':'current_health_status'}}
        }
        
        self.comparative_values_dict = {}
        self.collect_ranges()

    def __call__(self):
        return self.ranges_dict

    def collect_ranges(self):
        """
        Collects ranges for 
        """
        data_dict_df = self.utils.read_data_dictionary()
        filtered_df = data_dict_df[((data_dict_df[
        'Text Validation Min'] != '') & (data_dict_df[
        'Text Validation Max'] !=''))]
        cols = ['Text Validation Min',
        'Text Validation Max','Variable / Field Name','Form Name']
        filtered_df = filtered_df[cols]
        filtered_df = filtered_df.rename(columns={
        'Variable / Field Name': 'variable',
        'Text Validation Min':'min',
        'Text Validation Max':'max','Form Name':'form'})
        filtered_df = filtered_df.replace(
        self.utils.missing_code_list,0)
        for row in filtered_df.itertuples():
            if (self.utils.can_be_float(row.min)
            and self.utils.can_be_float(row.max)):
                self.ranges_dict["PRONET"][row.variable] = {
                'min':float(row.min),'max':float(row.max),'form':row.form}
                # modify prescient values
                presc_min, presc_max = self.modify_prescient_vals(row.variable, row.min, row.max)
                self.ranges_dict["PRESCIENT"][row.variable] = {
                'min':float(presc_min),'max':float(presc_max),'form':row.form}

    def modify_prescient_vals(self, var, orig_min, orig_max):
        mod_min = orig_min
        mod_max = orig_max
        if 'saliva' in var and 'vol' in var:
            mod_min = orig_min
            mod_max = 2.1
        if var == 'chrpps_ausgrade':
            mod_min = orig_min
            mod_max = 100
        return mod_min, mod_max

            
