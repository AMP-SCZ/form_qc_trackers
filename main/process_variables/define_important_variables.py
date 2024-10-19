class DefineVariables():
    

    def __init__(self):
        pass

    def run_script(self):
        self.initialize_unique_variable_names()
        
    def initialize_unique_variable_names(self):
        """Defines variables with names that don't follow the common
        naming conventions of those variable types"""

        self.unique_missing_variable_names = ['chrpsychs_scr_missing_2',\
        'hcpsychs_scr_missing_2','chrpsychs_fu_missing_fu',\
        'hcpsychs_fu_missing_fu','chrpsychs_fu_missing_fu_2',\
        'hcpsychs_fu_missing_fu_2','chrsofas_missing_fu','chrdig_missing_all']
        self.forms_with_unique_missing_variables = ['psychs_p1p8_fu_hc',\
        'psychs_p9ac32_fu_hc','psychs_p9ac32','psychs_p1p8_fu','psychs_p9ac32_fu',\
        'digital_biomarkers_mindlamp_checkin','sofas_followup']
        self.unique_date_variable_names = ['chrmri_entry_date',\
        'chrsofas_interview_date_fu','chrcrit_date',\
        'chric_consent_date','chrcbccs_review_date',\
        'chrgpc_date','chrsofas_interview_date_fu','chrap_date']
