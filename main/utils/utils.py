import pandas as pd
import os
import json
class Utils():
    def __init__(self):
        self.missing_code_list = \
        ['-3','-9',-3,-9,-3.0,-9.0,'-3.0','-9.0',\
        '1909-09-09','1903-03-03','1901-01-01','-99',-99,-99.0,\
        '-99.0',999,999.0,'999','999.0'] 

        self.absolute_path  = "/".join(os.path.realpath(__file__).split("/")[0:-3])


        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)


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
    
    def save_dictionary_as_csv(
        self, inp_dictionary : dict, output_path : str
    ):
        """
        Function to save a dictionary
        as a csv file. Will append its
        values to a list, convert the list
        to a dataframe, then save the dataframe
        as a csv.


        Parameters
        -------------------
        inp_dictionary : dict
            dictionary to save
        output_path : str
            path where the csv file will
            be saved
        """
        
        output_list = list(inp_dictionary.values())
        output_df = pd.DataFrame(output_list)
        output_df.to_csv(output_path, index = False)
    
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

    def read_data_dictionary(
        self, match_str : str = 'current_data_dictionary'
    ) -> pd.DataFrame:
        """
        Finds the current data dictionary
        in the data_dictionary dependencies 
        folder. 

        Parameters
        ------------------
        match_str : str
            String that must be in the 
            name of the data dictionary
            file that will be used
        
        Returns 
        -----------------------
        data_dictionary_df : pd.DataFrame
            Pandas dataframe of the entire
            REDCap data dictionary
        """
    
        depend_path = self.config_info['paths']['dependencies_path']
        for file in os.listdir(f"{depend_path}data_dictionary"):
            # loops through directory to search for current data dictionary
            if match_str in file:
                data_dictionary_df = pd.read_csv(
                f"{depend_path}data_dictionary/{file}",
                keep_default_na=False) # setting this to false preserves empty strings

        return data_dictionary_df
    
    def can_be_float(self,value):
        try:
            float(value)  
            return True
        except (ValueError, TypeError):
            return False
        
    def apply_df_str_filter(self, df, filter_list, filter_col):
        excluded_str_filter = '|'.join(filter_list)

        filtered_df = df[
        self.data_dict_df[filter_col].str.contains(
        excluded_str_filter)]

        return filtered_df



        




