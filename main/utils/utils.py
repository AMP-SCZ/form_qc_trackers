import pandas as pd
import os
import json
import re
import dropbox
from datetime import datetime
class Utils():
    def __init__(self):
        self.missing_code_list = \
        ['-3','-9',-3,-9,-3.0,-9.0,'-3.0','-9.0',\
        '1909-09-09','1903-03-03','1901-01-01','-99',-99,-99.0,\
        '-99.0',999,999.0,'999','999.0'] 

        self.absolute_path  = "/".join(os.path.realpath(__file__).split("/")[0:-3])

        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.all_pronet_sites = ["KC", "BI", "SD", "NL", "OR", "CA", "IR", "MU","YA", "HA",\
        "MA", "PI", "PV", "MT", "SF", "NC",'NN','PA','WU',"LA",'GA','TE','CM','SL','SI','SH','UR','OH']
        self.all_prescient_sites = ['BM', 'CG', 'CP', 'GW', 'HK', 'JE', 'LS', 'ME', 'SG', 'ST']

        self.all_sites = {'PRONET' :["KC", "BI", "SD", "NL",
        "OR", "CA", "IR", "MU","YA", "HA","MA", "PI", "PV",
        "MT", "SF", "NC",'NN','PA','WU',"LA",'GA','TE','CM',
        'SL','SI','SH','UR','OH'] ,'PRESCIENT' : ['BM', 'CG', 'CP',
        'GW', 'HK', 'JE', 'LS', 'ME', 'SG', 'ST']}
        
        self.site_full_name_translations = {'BI': 'Beth Israel (Harvard) (BI)',\
                'CA': 'Calgary, CA (CA)', 'CM': 'Cambridge (CM)', 'GA': 'Georgia (GA)',\
                'HA': 'Hartford (Institute of Living) (HA)', 'IR': 'UC Irvine (IR)',\
                'KC': "King's College, UK (KC)", 'LA': 'UCLA (LA)', 'MA': 'Madrid, Spain (MA)',\
                'MT': 'Montreal, CA (MT)', 'MU': 'Munich, Germany (MU)', 'NC': 'UNC (North Carolina) (NC)',\
                'NL': 'Northwell (NL)', 'NN': 'Northwestern (NN)', 'OR': 'Oregon (OR)',
                'PA': 'University of Pennsylvania (PA)', 'PI': 'Pittsburgh (UPMC) (PI)',\
                'PV': 'Pavia, Italy (PV)', 'SD': 'UCSD (SD)', 'SF': 'UCSF (Mission Bay) (SF)',\
                'SH': 'Shanghai, China (SH)', 'SI': 'Mt. Sinai (SI)', 'SL': 'Seoul, South Korea (SL)',\
                'TE': 'Temple (TE)', 'WU': 'Washington University (WU)', 'YA': 'Yale (YA)','UR':'University of Rochester (UR)',\
                'OH':'Ohio (OH)', 'BM': 'Birmingham, UK (BM)', 'CG': 'Cologne, DE (CG)', \
                'CP': 'Copenhagen, DK (CP)', 'GW': 'Gwangju, KR (GW)', 'HK': 'Hong Kong (HK)',\
                'JE': 'Jena, DE (JE)', 'LS': 'Lausanne, CH (LS)', 'ME': 'Melbourne (ME)',\
                'SG': 'Singapore (SG)', 'ST': 'Santiago (ST)',
                'PRONET':'PRONET','PRESCIENT':'PRESCIENT','AMPSCZ':'AMPSCZ'}
        
        self.withdrawn_status_list = []

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
        
    def is_a_num(self,value):
        if re.fullmatch(r'\d+', str(value)):
            return True
        else:
            return False
        
    def apply_df_str_filter(self, df, filter_list, filter_col):
        excluded_str_filter = '|'.join(filter_list)

        filtered_df = df[
        df[filter_col].str.contains(
        excluded_str_filter)]

        return filtered_df

    def save_dependency_json(self, data, filename):
        dep_path = self.config_info["paths"]["dependencies_path"]
        with open(f'{dep_path}{filename}',
        'w') as json_file:
            json.dump(data, json_file, indent=4)  

    def load_dependency_json(self, filename):
        dep_path = self.config_info["paths"]["dependencies_path"]

        try:
            with open(f'{dep_path}{filename}','r') as json_file:
                json_data = json.load(json_file) 
                return json_data 
        except json.JSONDecodeError:
            return {}


    def all_dtype(self, inp_list):
        output_list = []
        for number in inp_list:
            if self.can_be_float(number):
                int_num = int(number)
                output_list.extend([
                int_num,str(int_num),float(int_num),
                str(float(int_num))])
        return output_list

    def collect_digit(self,string):
        """Collects digit in current string
        
        Parameters
        ------------
        string: string containing digit
        """
        number_str = ''
        for char in string:
            if char.isdigit():
                number_str += char
            elif number_str != '':
                return number_str
        return number_str

    def collect_dropbox_credentials(self):
        """reads dropbox credentials from
        JSON file"""

        depend_path = self.config_info['paths']['dependencies_path']

        with open(depend_path +'dropbox_credentials.json', 'r') as file:
            json_data = json.load(file)
        APP_KEY = json_data['app_key']
        APP_SECRET = json_data['app_secret']
        REFRESH_TOKEN = json_data['refresh_token']

        dbx = dropbox.Dropbox(
            app_key = APP_KEY,
            app_secret = APP_SECRET,
            oauth2_refresh_token = REFRESH_TOKEN
        )

        return dbx

    def days_since_today(self,date_str, date_format="%Y-%m-%d"):
        date_str = date_str.replace('/','-')
        input_date = datetime.strptime(date_str, date_format)
        
        today = datetime.today()
        
        delta = today - input_date
        
        return delta.days

    def reverse_dictionary(self, inp_dict):
        reversed_dict = {}
        for key, value in inp_dict.items():
            reversed_dict[value] = key

        return reversed_dict

    def find_days_between(self,d1,d2):
        """
        finds the days between two dates
        
        Parameters
        -----------------
        d1: first date
        d2: second date
        """

        date_format = "%Y-%m-%d"
        date1 = datetime.strptime(\
        d1.split(' ')[0], date_format)
        date2 = datetime.strptime(\
        d2.split(' ')[0], date_format)
        date_difference = date2 - date1
        days_between = date_difference.days

        return abs(days_between)
    
    def check_if_val_date_format(self,string, date_format="%Y-%m-%d"):
        try:
            datetime.strptime(string, date_format)
            return True
        except ValueError:
            return False
        
    def convert_prescient_compl_var(
        self, inp_compl_var : str
    ) -> str:
        output_compl_var = inp_compl_var.replace('_hc','').replace('onboarding','checkin') 

    def recent_date_from_dict(self,
        dates : dict
    ) -> str:
        """
        returns which date variables has 
        the most recent date, excluding missing
        codes

        Parameters
        --------------
        dates : dict
            dictionary with 
            the variable names as the
            keys and dates as the values

        Returns
        ------------
        recent_date_var : str
            name of the variable that 
            corresponds to the most recent
            date in the dictionary
        """
        recent_date_var = ''
        for date_var, date in dates.items():
            if date in self.missing_code_list:
                continue
            curr_date = datetime.strptime(date, "%Y-%m-%d")
            if curr_date > datetime.today():
                continue
            if recent_date_var == '':
                recent_date_var = date_var
            elif self.check_if_val_date_format(date):
                if curr_date > datetime.strptime(dates[recent_date_var], "%Y-%m-%d"):
                    recent_date_var = date_var

        return recent_date_var
    
    def time_to_next_visit(
        self, curr_tp : str
    ) -> int:
        """
        Calculates the number of days that there
        should be until the next timepoint
        """
        timepoints = self.create_timepoint_list()

        if curr_tp in ['month24','conversion']:
            return

        if curr_tp in ['screening','baseline']:
            days_btwn = 30
        else:
            curr_tp_ind = timepoints.index(curr_tp)
            next_tp = timepoints[curr_tp_ind + 1]
            months_btwn = int(next_tp.replace('month',''))-int(curr_tp.replace('month',''))
            days_btwn = months_btwn * 30
        
        return days_btwn

    def append_suffix_to_cols(self, 
        input_df : pd.DataFrame, suffix, incl_cols,
        excl_cols = ['subjectid']
    ) -> pd.DataFrame:
        """
        adds tp suffix to column names, 
        except for those in excluded list

        Parameters
        ----------------
        input_df : pd.DataFrame
            dataframe being modified 
        excl_cols : list
            columns not being modified
        """

        modified_cols = list(input_df.columns)
        # columns that will not have timepoint suffix added
        modified_col = [col + f'_{suffix}' for col in modified_cols if col not in excl_cols]
        modified_cols = {}
        for col in input_df.columns:
            if col in excl_cols:
                modified_cols[col] = col 
            else:
                modified_cols[col] = col + '_tp'
        modified_df = input_df.rename(columns=modified_cols)

        return modified_df 


        

            





