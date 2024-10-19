import pandas as pd

class DefineVariables():
    
    def __init__(self, data_dictionary_df):
        self.data_dictionary_df = data_dictionary_df

    def run_script(self):
        self.collect_missing_form_variables()

    def collect_missing_form_variables(self):
        missing_var_df = self.data_dictionary_df[\
        self.data_dictionary_df['Choices, Calculations, OR Slider Labels'].str.contains(\
        'Please click if this form is missing all of its data')]
        missing_form_variables = missing_var_df.set_index(\
        'Form Name')['Variable / Field Name'].to_dict()
        
        return missing_form_variables

