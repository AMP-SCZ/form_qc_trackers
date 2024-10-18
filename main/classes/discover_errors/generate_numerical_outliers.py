import pandas as pd
import sys
import os
parent_dir_path = os.path.abspath(\
os.path.join(os.path.dirname(__file__),\
'..', '/PHShome/ob001/anaconda3/new_forms_qc/QC/'))
if parent_dir_path not in sys.path:
    sys.path.append(parent_dir_path)

from classes.utils import Utils
class NumericalOutliers():
    """
    Class to discover new potential
    errors to be added to the QC program
    """
    def __init__(self):
        self.absolute_path = '/PHShome/ob001/anaconda3/new_forms_qc/QC/'
        self.combined_df_folder = '/data/predict1/data_from_nda/formqc/'

        self.utils = Utils()
        self.dtype_count = {}

        self.likely_missing_vals =\
        ['','NA','na','N/A','n/a',\
        'nan','NAN','NaN','nil']

        self.excluded_field_types =\
        ['checkbox','dropdown','radio','yesno']

        self.det_var_test = []

    def run_script(self) -> None:
        self.var_field_types =\
        self.utils.collect_var_data(['field_type'])
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
                self.combined_df = pd.read_csv((f'{self.combined_df_folder}'
                    f'combined-{network}-{timepoint}-day1to1.csv'),\
                    keep_default_na=False) 
                self.discover_numerical_variables(self.combined_df)
                self.detect_deviations_from_mean(self.combined_df)

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
                and self.var_field_types[col]['field_type'] in\
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
        
    def detect_deviations_from_mean(self,df) -> pd.DataFrame:
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
                        and var not in self.det_var_test\
                        and not any(x in var for x in ['figs','psychs','blood','cbc']):
                            #self.det_var_test.append(var)
                            print('--------------------')
                            print(f'Variable:{var}')
                            print(f'value: {var_val}')
                            print(f'deviation: {dev}')
                            print(f'std:{std_dev}')
                            print(f'Mean: {sum(val_list) / len(val_list)}')


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
