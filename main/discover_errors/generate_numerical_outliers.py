import pandas as pd

import os
import sys
import json
import random
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class NumericalOutliers():
    """
    Class to discover new potential
    errors to be added to the QC program
    """
    def __init__(self):
        self.combined_df_folder = '/data/predict1/data_from_nda/formsdb/generated_outputs/combined/PROTECTED/'

        self.utils = Utils()
        self.dtype_count = {}

        self.likely_missing_vals =\
        ['','NA','na','N/A','n/a',\
        'nan','NAN','NaN','nil']

        self.data_dict = self.utils.read_data_dictionary()
        self.data_dict.columns = self.data_dict.columns.str.replace(' ','_')
        self.var_field_types = self.data_dict.set_index("Variable_/_Field_Name")["Field_Type"].to_dict()

        self.form_info = self.utils.load_dependency_json('grouped_variables.json')
        self.forms_per_var = self.form_info['var_forms']
        self.blood_vars = self.form_info['blood_vars']
        self.excl_blood_vars = (
        self.blood_vars['position_variables'] + 
        self.blood_vars['barcode_variables'] + 
        self.blood_vars['id_variables'])
 
        self.excluded_field_types =\
        ['checkbox','dropdown','radio','yesno']

        self.det_var_test = []
        self.output_list = []

    def run_script(self) -> None:
        self.loop_timepoints()

    def loop_timepoints(self) -> None:
        """
        Loops through each timepoint 
        for each network and reads the 
        corresponding combined dataframe.
        """

        self.timepoint_list = \
        self.utils.create_timepoint_list()
        for network in ['PRONET','PRESCIENT']:
            for timepoint in self.timepoint_list:
                print(network)
                print(timepoint)
                self.combined_df = pd.read_csv(
                (f'{self.combined_df_folder}AMPSCZ-combined-redcap_'
                f'{timepoint.replace("month","month_").replace("floating","floating_forms")}'
                f'_{network.replace("PRONET","ProNET")}-day1to1.csv'),
                keep_default_na = False, on_bad_lines='skip')
                self.discover_numerical_variables(self.combined_df)
                self.detect_deviations_from_mean(self.combined_df,timepoint)

    def discover_numerical_variables(self,df) -> pd.DataFrame:
        """
        Finds any variable that is typically 
        a numerical value

        Parameters
        -----------
        df: current combined dataframe
        """
        
        for row in df.itertuples():
            for col in df.columns:
                if (col in self.var_field_types.keys()\
                and self.var_field_types[col] in\
                self.excluded_field_types) or\
                col not in self.var_field_types.keys():
                    continue
                self.dtype_count.setdefault(\
                    col,{'number':0,'nan':0})
                if getattr(row,col) not in\
                (self.utils.missing_code_list\
                + self.likely_missing_vals):
                    if self.utils.check_if_number(\
                    str(getattr(row,col)).split(' ')[0]) == True:
                        self.dtype_count[col]['number'] +=1
                    else:
                        self.dtype_count[col]['nan'] +=1 

        self.filter_lists()

    def filter_lists(self) -> None:
        """
        Filters lists to exclude
        values that are unlikely
        to be meaningful
        """

        self.num_var_list = \
        self.create_numerical_var_list(self.dtype_count)
        self.var_means,self.var_totals,self.all_var_values =\
        self.collect_numerical_var_means(self.num_var_list)
        self.var_means = \
        self.utils.filter_numerical_dict(self.var_means, 2)
        
    def detect_deviations_from_mean(self,df,tp) -> pd.DataFrame:
        """
        Detects any values for any variables
        that deviate significantly from the mean
        value for that variable.

        Parameters
        ---------------
        df : pd.DataFrame
            Dataframe that contains the variables
        """
        for row in df.itertuples():
            for var, val_list in self.all_var_values.items():
                if hasattr(row,var):
                    var_val = str(getattr(row,var)).split(' ')[0]
                    if self.utils.check_if_number(var_val) == True:
                        if len(val_list) < 10 or (sum(val_list) / len(val_list)) < 3:
                            continue
                        dev,std_dev =\
                        self.utils.deviation_from_list(float(var_val),val_list)
                        if abs(dev) > (3 * std_dev) \
                        and var_val not in (self.utils.missing_code_list)\
                        and var not in self.det_var_test:
                        #and not any(x in var for x in ['figs','psychs','blood','cbc']):
                            #self.det_var_test.append(var)
                            print('--------------------')
                            print(f'Variable:{var}')
                            print(f'value: {var_val}')
                            print(f'deviation: {dev}')
                            print(f'std:{std_dev}')
                            print(f'Mean: {sum(val_list) / len(val_list)}')
                            form = self.forms_per_var[var]
                            self.output_list.append({'subject':row.subjectid,'timepoint':tp,
                            'variable':var,'form':form,
                            'var_value':var_val,'dev':dev,'std':std_dev,
                            'mean':sum(val_list) / len(val_list)})
        output_df = pd.DataFrame(self.output_list)
        output_df.to_csv('outliers.csv',index = False)


    def collect_numerical_var_means(self,var_list):
        """
        Collect mean values for each numerical
        variable.

        Parameters
        -----------
        var_list: list of numerical variables

        Return
        -------------
        mean_dict: dictionary of every 
        variable with their means
        """

        total_sums = {}
        all_values = {}
        for row in self.combined_df.itertuples():
            for var in var_list:
                if hasattr(row,var) and\
                getattr(row,var) not in \
                (self.utils.missing_code_list\
                + self.likely_missing_vals) and\
                self.utils.check_if_number(\
                str(getattr(row,var)).split(' ')[0]) ==\
                True:
                    total_sums.setdefault(var,{'sum':0,'n':0})
                    all_values.setdefault(var,[])
                    var_val = float(str(getattr(row,var)).split(' ')[0])
                    all_values[var].append(var_val)          
                    total_sums[var]['sum'] +=\
                    var_val
                    total_sums[var]['n'] +=1
        mean_dict = self.utils.calculate_dictionary_means(total_sums)
        return mean_dict,total_sums, all_values
                   
    def create_numerical_var_list(self,var_dict,threshold = 0.9):
        """
        Creates a list of every variable
        that is typically a number.

        Parameters
        -------------------
        Threshold: Determines the frequency
        that a variable has to be a number
        in order to be added to the list 

        Returns
        -------------------
        num_var_list: list of numerical variables
        """

        num_var_list = []
        self.sometimes_nums = []
        for var, values in var_dict.items():
            if values['number'] > 0 or \
            values['nan'] > 0:
                if values['number'] / (values['number'] + \
                 values['nan']) > threshold:
                    if values['number'] / (values['number'] \
                    + values['nan']) < 1:
                        self.sometimes_nums.append(var)
                    num_var_list.append(var)

        #self.analyze_partial_numerical_vars()
        return num_var_list

    def analyze_partial_numerical_vars(self): 
        """
        Analyzes any variables that are usually
        numbers, but not always
        """       
        for row in self.combined_df.itertuples():
            for var in self.sometimes_nums:
                if self.utils.check_if_number(\
                str(getattr(row,var)).split(' ')[0])== False\
                and getattr(row,var) not in\
                (self.utils.missing_code_list \
                + self.likely_missing_vals):
                    print(var)
                    print(getattr(row,var))


if __name__ == '__main__':
    NumericalOutliers().run_script()
