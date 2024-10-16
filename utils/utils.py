import pandas as pd

class Utils():
    def __init__(self):
        self.missing_code_list = \
        ['-3','-9',-3,-9,-3.0,-9.0,'-3.0','-9.0',\
        '1909-09-09','1903-03-03','1901-01-01','-99',-99,-99.0,\
        '-99.0',999,999.0,'999','999.0'] 

        self.absolute_path = '/PHShome/ob001/anaconda3/new_forms_qc/QC/'

        self.data_dictionary_df = pd.read_csv(\
        (f'{self.absolute_path}data_dictionaries/'
        'CloneOfYaleRealRecords_DataDictionary_2023-05-19.csv'),\
        encoding = 'latin-1',keep_default_na=False)

    def create_timepoint_list(self):
        """
        Organizes every timepoint
        as list

        Returns
        ------------
        timepoint_list: list of timepoints
        """

        tp_list = ['screening','baseline']
        for x in range(1,13):
            tp_list.append('month'+f'{x}') 
        tp_list.append('month18')
        tp_list.append('month24')

        return tp_list
    
    def check_if_number(self,input):
        """
        Checks if the input is 
        a number or not.

        Parameters
        -------------
        input: input that will be checked

        Returns
        ------------
        True if a it input is 
        a number and False if
        it is not.
        """
        try:
            float(input)  
            return True
        except ValueError:
            return False
        
    def collect_var_data(self,headers):
        """
        Collects data from the specified
        column headers for each variable
        in the data dictionary

        Parameters
        ---------------
        headers: headers from the 
        data dictionary that will be added
        to output

        Returns
        ---------------
        var_info_dict: dictionary 
        of the information from each
        specified column header for
        each variable.
        """

        var_info_dict = {}
        for row in self.data_dictionary_df.itertuples():
            data_dictionary_col_names = {'form':'Form Name',\
            'variable':'ï»¿"Variable / Field Name"','field_type':'Field Type',\
            'required_field': 'Required Field?','identifier':'Identifier?',\
            'branching_logic':'Branching Logic (Show field only if...)',\
            'field_annotation':'Field Annotation','field_label':'Field Label',\
            'choices': 'Choices, Calculations, OR Slider Labels'}
            col_values = {}
            for key,value in data_dictionary_col_names.items():
                col_values[key] = self.data_dictionary_df.at[row.Index, value] 
            
            for header in headers:
                var_info_dict.setdefault(col_values['variable'],{})
                var_info_dict[col_values['variable']][header]\
                = col_values[header]

        return var_info_dict
    
    def calculate_dictionary_means(self,total_sums_dict):
        """
        Converts dictionary with the sums 
        and sample sizes of each variable
        into a dictionary with the means.

        Parameters
        --------------
        total_sums_dict: dictionary with
        each variable's sum and sample size

        Returns
        -----------------
        mean_dict: dictionary with
        each variable's mean
        """

        mean_dict = {}
        for var, values in total_sums_dict.items():
            if values['n'] > 0:
                mean_dict[var]\
                = values['sum']/values['n']
        return mean_dict
    
    
    def filter_numerical_dict(self,num_dict,threshold,
                         upper = True):
        
        """
        Function to filter a dictionary 
        of numbers as values to only include the items 
        with values above or below a certain threshold

        Parameters
        ------------
        num_dict: Input dictionary, which 
        should consist of numbers as the values.

        threshold: threshold that the value has 
        to be above or below

        upper: determines if the number has to be
        above or below the threshold

        Returns
        --------------
        filtered_num_dict: dictionary with only the 
        numbers that were not filtered out by 
        the threshold
        """
        
        filtered_num_dict = {}
        for key,num in num_dict.items():
            if (upper == True and num > threshold) or\
            (upper == False and num < threshold):
                filtered_num_dict[key] = num

        return filtered_num_dict

    def deviation_from_list(self,single_number, numbers):
        # Calculate the mean of the numbers
        mean = sum(numbers) / len(numbers)
        
        # Calculate the variance
        variance = sum((x - mean) ** 2 for x in numbers) / (len(numbers) - 1)
        
        # Calculate the standard deviation
        std_dev = variance ** 0.5
        
        # Calculate the deviation of the single number from the mean
        deviation = single_number - mean

        return deviation,std_dev

