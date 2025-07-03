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
        self.ranges_dict = {
            'chrpps_mage': {'min':10,'max':85, 'form':'psychosis_polyrisk_score'},
            'chrpps_fage': {'min':10,'max':85, 'form':'psychosis_polyrisk_score'},
            'chrfigs_mother_age': {'min':10,'max':85,
            'form':'family_interview_for_genetic_studies_figs'},
            'chrfigs_father_age': {'min':10,'max':85,
            'form':'family_interview_for_genetic_studies_figs'},
        }
        
        self.comparative_values_dict = {}
        self.collect_ranges()
        self.negative_values_list = ['chrpharm_med2_dosage_past',
                                     'chrblood_spintemp', 'chrscid_e207_note',
                                     'chrscid_b11_note', 'chrgfss_prompt1', 'chrchs_bodytemp',
                                     'chrchs_bmi', 'chrscid_e8_note', 'chrpsychs_av_audio_expl',
                                     'chrblood_spintempc', 'chrchs_weightkg', 'chriq_tscore_sum',
                                     'chrpps_mage', 'chrspeech_comments', 'chrpharm_med4_dosage_past',
                                     'chriq_scaled_sum', 'chrpps_postalch', 'chrsaliva_freezer_id', 'chrchs_bodytempf',
                                     'chrblood_serum_freeze', 'chrcssrsb_comments', 'chrpharm_med3_dosage_past', 'chrchs_heightcm',
                                     'chrscid_d61_d65', 'chrblood_wb4pos', 'chrpps_fage', 'chrblood_wholeblood_freeze', 'chrgfr_prompt3c',
                                     'chrblood_plasma_freeze', 'chrap_qtpdoseeq1', 'chrap_qtpdosecourse1', 'chrblood_freezerid',
                                     'chrgfssfu_prompt1', 'chrpharm_med1_dosage_past', 'chrap_total']

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
                self.ranges_dict[row.variable] = {
                'min':float(row.min),'max':float(row.max),'form':row.form}

    def collect_comparative_values(self):
        pass

            
