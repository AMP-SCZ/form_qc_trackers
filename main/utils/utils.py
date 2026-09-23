import pandas as pd
import os
import json
import re
import dropbox
from datetime import datetime

# Module-level cache for dependency JSONs. The pipeline instantiates Utils()
# (and thus loads `important_form_vars.json` + `variables_added_later.json`)
# inside every per-row FormCheck subclass — at ~10K subjects × 14 timepoints
# × 2 networks × 5 checker classes that's ~1.4M Utils() constructions per
# run, each previously doing 2 fresh disk reads + JSON parses. With this
# cache, subsequent loads are O(1) dict lookups. Keyed by (dep_path, filename)
# so test environments with a different dependencies dir don't get crossed.
_DEPENDENCY_JSON_CACHE = {}

# config.json is loaded read-only at runtime. Without this cache, every
# Utils() construction re-opens and re-parses it. Keyed by absolute_path
# so a different project root (test harness) recomputes correctly.
_CONFIG_CACHE = {}


def _project_root():
    """Return the repository root using platform-native path semantics."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def _validate_testing_enabled(config_info, source_path):
    """
    Central validation for config.testing_enabled. Several call sites
    compare the value to the literal string "True" exactly; a typo like
    "true", "True " (trailing space), JSON boolean true, or 1 would
    silently route a test run to production output paths and the live
    Dropbox folder. Reject anything other than "True" or "False" with
    a clear fatal error. Missing key is treated as "False" — that is
    the established default at every existing call site (they fall
    through to production path on absence).
    """
    if 'testing_enabled' not in config_info:
        config_info['testing_enabled'] = "False"
        return
    val = config_info['testing_enabled']
    if val not in ("True", "False"):
        raise RuntimeError(
            f"FATAL: config.testing_enabled at {source_path} must be "
            f"exactly the string 'True' or 'False'; got {val!r} "
            f"(type {type(val).__name__}). Refusing to run — silently "
            f"accepting this typo would route outputs to "
            f"{'production' if val != 'True' else 'testing'} paths."
        )


def _load_config(absolute_path):
    cached = _CONFIG_CACHE.get(absolute_path)
    if cached is None:
        config_path = f'{absolute_path}/config.json'
        with open(config_path, 'r') as file:
            cached = json.load(file)
        _validate_testing_enabled(cached, config_path)
        _CONFIG_CACHE[absolute_path] = cached
    return cached


# Process-wide constant singletons (roadmap #9, safe subset). Utils is
# constructed ~8x per row on the QC hot path; rebuilding these ~40-entry
# dicts/lists on every construction was pure per-row allocation. Built once
# at import and shared. All verified read-only at call sites: concatenations
# like `missing_code_list + ['']` create new lists, and the site dicts are
# only read via [] (never reassigned). The mutable `withdrawn_status_list`
# is intentionally NOT hoisted (stays a fresh per-instance []).
_MISSING_CODE_LIST = \
['-3','-9',-3,-9,-3.0,-9.0,'-3.0','-9.0',
'1909-09-09','1903-03-03','1901-01-01','-99',-99,-99.0,
'-99.0',999,999.0,'999','999.0']
_MISSING_CODE_SET = frozenset(_MISSING_CODE_LIST)
_ALL_PRONET_SITES = ["KC", "BI", "SD", "NL", "OR", "CA", "IR", "MU","YA", "HA",
"MA", "PI", "PV", "MT", "SF", "NC",'NN','PA','WU',"LA",'GA','TE','CM','SL','SI','SH','UR','OH']
_ALL_PRESCIENT_SITES = ['BM', 'CG', 'CP', 'GW', 'HK', 'JE', 'LS', 'ME', 'SG', 'ST']
_ALL_SITES = {'PRONET': list(_ALL_PRONET_SITES), 'PRESCIENT': list(_ALL_PRESCIENT_SITES)}
_SITE_FULL_NAME_TRANSLATIONS = {'BI': 'Beth Israel (Harvard) (BI)',
        'CA': 'Calgary, CA (CA)', 'CM': 'Cambridge (CM)', 'GA': 'Georgia (GA)',
        'HA': 'Hartford (Institute of Living) (HA)', 'IR': 'UC Irvine (IR)',
        'KC': "King's College, UK (KC)", 'LA': 'UCLA (LA)', 'MA': 'Madrid, Spain (MA)',
        'MT': 'Montreal, CA (MT)', 'MU': 'Munich, Germany (MU)', 'NC': 'UNC (North Carolina) (NC)',
        'NL': 'Northwell (NL)', 'NN': 'Northwestern (NN)', 'OR': 'Oregon (OR)',
        'PA': 'University of Pennsylvania (PA)', 'PI': 'Pittsburgh (UPMC) (PI)',
        'PV': 'Pavia, Italy (PV)', 'SD': 'UCSD (SD)', 'SF': 'UCSF (Mission Bay) (SF)',
        'SH': 'Shanghai, China (SH)', 'SI': 'Mt. Sinai (SI)', 'SL': 'Seoul, South Korea (SL)',
        'TE': 'Temple (TE)', 'WU': 'Washington University (WU)', 'YA': 'Yale (YA)','UR':'University of Rochester (UR)',
        'OH':'Ohio (OH)', 'BM': 'Birmingham, UK (BM)', 'CG': 'Cologne, DE (CG)',
        'CP': 'Copenhagen, DK (CP)', 'GW': 'Gwangju, KR (GW)', 'HK': 'Hong Kong (HK)',
        'JE': 'Jena, DE (JE)', 'LS': 'Lausanne, CH (LS)', 'ME': 'Melbourne (ME)',
        'SG': 'Singapore (SG)', 'ST': 'Santiago (ST)',
        'PRONET':'PRONET','PRESCIENT':'PRESCIENT','AMPSCZ':'AMPSCZ'}


class Utils():
    def __init__(self):
        self.missing_code_list = _MISSING_CODE_LIST
        self.missing_code_set = _MISSING_CODE_SET

        self.absolute_path = _project_root()

        self.config_info = _load_config(self.absolute_path)

        # The persistent network scope is defined in config.json. QC_NETWORKS
        # can override it for a one-off run without changing future scheduled
        # runs. Every pipeline stage consumes this single validated list,
        # including Dropbox readback/upload.
        configured_networks = os.environ.get('QC_NETWORKS')
        if configured_networks is None:
            configured_networks = self.config_info.get('pipeline_networks')
        elif isinstance(configured_networks, str):
            configured_networks = configured_networks.split(',')
        if (not isinstance(configured_networks, (list, tuple))
                or not configured_networks):
            raise RuntimeError(
                "Set config.json pipeline_networks to a non-empty list, or "
                "provide a comma-separated QC_NETWORKS override.")
        normalized_networks = []
        for network in configured_networks:
            normalized = str(network).strip().upper()
            if normalized not in {'PRONET', 'PRESCIENT'}:
                raise RuntimeError(
                    f"Unsupported QC network {network!r}; expected PRONET "
                    "and/or PRESCIENT.")
            if normalized not in normalized_networks:
                normalized_networks.append(normalized)
        self.pipeline_networks = tuple(normalized_networks)

        self.output_path = self.config_info['paths']['output_path']

        self.all_pronet_sites = _ALL_PRONET_SITES
        self.all_prescient_sites = _ALL_PRESCIENT_SITES
        self.all_sites = _ALL_SITES
        self.site_full_name_translations = _SITE_FULL_NAME_TRANSLATIONS

        self.withdrawn_status_list = []
        self.important_form_vars = self.load_dependency_json(
        'important_form_vars.json')

        self.vars_added_later = self.load_dependency_json(
        'variables_added_later.json')


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
        # Match calculated-field discovery: prefer the exact canonical file,
        # accept one unambiguous dated variant, and fail when variants compete.
        dictionary_dir = os.path.join(depend_path, 'data_dictionary')
        matches = sorted(
            f for f in os.listdir(dictionary_dir)
            if match_str in f and f.lower().endswith('.csv')
        )
        if not matches:
            raise FileNotFoundError(
                f"No data dictionary file matching '{match_str}' found in"
                f" {dictionary_dir}")
        exact_name = f"{match_str}.csv"
        if exact_name in matches:
            chosen = exact_name
        elif len(matches) == 1:
            chosen = matches[0]
        else:
            raise FileNotFoundError(
                f"Multiple data dictionary files match '{match_str}' in "
                f"{dictionary_dir}: {matches}. Keep the canonical "
                f"{exact_name}, remove stale variants, or explicitly select "
                "one dictionary in the standalone tool.")
        data_dictionary_df = pd.read_csv(
            os.path.join(dictionary_dir, chosen),
            keep_default_na=False)  # setting this to false preserves empty strings

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
        # Atomic write: serialize to a PID-stamped tmp file then
        # os.replace into place. Without this, a crash mid-dump
        # leaves a truncated JSON file on disk; the next
        # load_dependency_json call would raise RuntimeError
        # (post-RH-1) or silently return {} (pre-RH-1, the bug
        # this avoids). Pattern matches the parquet writes
        # elsewhere in the pipeline.
        dep_path = self.config_info["paths"]["dependencies_path"]
        final_path = f'{dep_path}{filename}'
        tmp_path = f'{final_path}.{os.getpid()}.tmp'
        try:
            with open(tmp_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            os.replace(tmp_path, final_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
        # Invalidate the cache so any subsequent load_dependency_json call
        # in this process picks up the freshly-written content.
        _DEPENDENCY_JSON_CACHE.pop((dep_path, filename), None)

    def load_dependency_json(self, filename):
        dep_path = self.config_info["paths"]["dependencies_path"]
        cache_key = (dep_path, filename)
        if cache_key in _DEPENDENCY_JSON_CACHE:
            return _DEPENDENCY_JSON_CACHE[cache_key]
        # Release-hardening (RH-1): corrupt JSON now raises instead
        # of silently returning {}. The previous behavior was a
        # silent-misconfig channel — a truncated dep file produced
        # {} → downstream code saw no subjects / no forms / no
        # Melbourne RAs and proceeded with wrong data. Failing
        # loudly here forces the operator to investigate (likely
        # re-run process_variables) before any QC stage uses the
        # broken file. FileNotFoundError already propagated; no
        # change for missing files.
        full_path = f'{dep_path}{filename}'
        try:
            with open(full_path, 'r') as json_file:
                json_data = json.load(json_file)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"FATAL: dependency JSON at {full_path} is "
                f"unreadable ({type(e).__name__}: {e}). Refusing "
                f"to silently return an empty dict — downstream QC "
                f"would proceed with missing data. Restore the "
                f"file (e.g., re-run process_variables) and try "
                f"again."
            ) from e
        _DEPENDENCY_JSON_CACHE[cache_key] = json_data
        return json_data


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
    
    def check_if_val_date_format(self, string, date_format="%Y-%m-%d"):
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
            if date in self.missing_code_set:
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
        self, curr_tp : str, cohort :str,
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
            if cohort.lower() == 'hc' and curr_tp == 'month2':
                next_tp = 'month12'
                
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

        modified_cols = {}
        for col in input_df.columns:
            if col in excl_cols:
                modified_cols[col] = col 
            elif col in incl_cols:
                modified_cols[col] = col + f'_{suffix}'
        modified_df = input_df.rename(columns=modified_cols)
        modified_df = modified_df[list(modified_cols.values())]

        return modified_df 

    def check_if_after_date(self, 
        curr_row : tuple,
        form : str, date_var : str
    ) -> bool:
        """
        Checks to make sure date 
        is after it was added in 
        particularly for missing_data
        buttons that were added later

        Parameters
        --------------
        curr_row : tuple
            current dataframe row being checked \
        form : str
            current form being checked
        date_var : str
            date variable being checked
        """
        if form not in self.vars_added_later.keys():
            return True
        if date_var not in self.vars_added_later[form].keys():
            return True
        date_added = self.vars_added_later[form][date_var]
        if hasattr(curr_row, date_var):
            date_val = getattr(curr_row, date_var)
            date_val = str(date_val)
            try:
                date_val = datetime.strptime(date_val, '%Y-%m-%d')
                if date_val > datetime.strptime(date_added, '%Y-%m-%d'):
                    return True
            except Exception as e:
                return False
        
        return False

    def check_if_missing(self,
        curr_row : tuple, form : str,
        timepoint : str, network: str
    ):
        """
        Checks if a form is marked as missing 

        Parameters 
        ------------
        curr_row : tuple
            current row of dataframe
        form : str
            current form
        """
        # floating forms do not need to 
        # be marked missing
        if timepoint == 'floating':
            return False
        compl_var = self.important_form_vars[form]["completion_var"]
        date_var = self.important_form_vars[form]["interview_date_var"]
        if network == 'PRESCIENT':
            compl_var += '_rpms'
            # rpms compl variables same for both cohorts
            compl_var = compl_var.replace('_hc','').replace(
            'onboarding','checkin').replace(
            'end_of_12month_study_pe','checkin').replace('end_of_12month_study_p','checkin')  
        missing_var = self.important_form_vars[form]["missing_var"]
        non_bl_vars = self.important_form_vars[form]["non_branch_logic_vars"]
        non_bl_vars_filled_out = 0        
        if missing_var != "" and not (form in self.vars_added_later.keys()
        and missing_var in self.vars_added_later[form].keys() and
        self.check_if_after_date(curr_row, form, date_var) == False):                
            if not hasattr(curr_row, missing_var):
                return False
            # prescient missingness can also be indicated by the completion var
            if (network == 'PRESCIENT' and
                    hasattr(curr_row, compl_var) and
                    getattr(curr_row, compl_var) in self.all_dtype([3,4])):
                return True
            if getattr(curr_row, missing_var) not in self.all_dtype([1]):
                return False
            else:
                return True
        
        elif missing_var == "" or (form in self.vars_added_later.keys() and missing_var
        in self.vars_added_later[form].keys() and
        self.check_if_after_date(curr_row, form, date_var) == False):
            for non_bl_var in non_bl_vars:
                if (hasattr(curr_row,non_bl_var)
                and getattr(curr_row,non_bl_var) != ''):
                    non_bl_vars_filled_out +=1
            if non_bl_vars_filled_out < (len(non_bl_vars)/2):
                return True
            else:
                return False

    def collect_all_type_vars(self, 
        data_type : str = 'date',
        threshold : float = 0.5
    ):
        """
        Collects all variables with 
        a defined threshold of values
        being the specified data type.

        Parameters
        -----------------
        threshold : float
            threshold of values that need to 
            the be specified category

        Returns
        --------------------
        all_type_vars : list
            list of all variables that belog to 
            the sprecified category
        """

        variable_type_distributions = self.load_dependency_json(
        'variable_type_distributions.json')
        all_type_vars = []
        for var, distributions in variable_type_distributions.items():
            total = (distributions['num'] + distributions['string']
             + distributions['date'])
            if total > 0:
                if distributions[data_type]/total > threshold:
                    all_type_vars.append(var)

        return all_type_vars            
    
    def date_ranges_overlap(self, start1, end1, start2, end2):
        """
        Check if two date ranges [start1, end1] and [start2, end2] overlap.

        Parameters:
        - start1, end1: datetime objects for the first range
        - start2, end2: datetime objects for the second range

        Returns:
        - True if the ranges overlap, False otherwise
        """
        return start1 <= end2 and start2 <= end1


        
    def convert_range_to_list(self, 
        range_str, str_conv = False
    ):
        """
        Converts string with number range to a list of 
        the numbers included in that range. Used for IQ age checks.

        Parameters
        -------------
        range_str: str
            string of number range
        str_conv: bool
            whether or not input needs
            to be converted to a string first
        """
        
        range_list = []
        if '-' not in range_str:
            if str_conv ==True:
                return [str(range_str).replace(' ','')]
            else:
                return range_str
        first_item = int(range_str.split('-')[0])
        last_item = int(range_str.split('-')[1])
        for x in range(first_item, last_item+1):
            if str_conv ==True:
                new_item = str(x).replace(' ','')
            else:
                new_item = x
            range_list.append(new_item)
        return range_list

    def compare_dataframes(self, df1, df2,out_diffs,out_only_1,out_only_2):
        KEY_COL = "subjectid"
        if KEY_COL not in df1.columns or KEY_COL not in df2.columns:
            raise ValueError(f"Key column '{KEY_COL}' must exist in both files.")

        df1 = df1.set_index(KEY_COL)
        df2 = df2.set_index(KEY_COL)

        keys1 = set(df1.index)
        keys2 = set(df2.index)

        common_keys = sorted(keys1 & keys2)
        only1 = sorted(keys1 - keys2)
        only2 = sorted(keys2 - keys1)

        # compare columns (all columns except the key)
        compare_cols = sorted(set(df1.columns) | set(df2.columns))

        diffs = []
        for k in common_keys:
            r1 = df1.loc[k]
            r2 = df2.loc[k]

            if isinstance(r1, pd.DataFrame) or isinstance(r2, pd.DataFrame):
                raise ValueError(f"Duplicate key found: {k}. This simple script requires unique keys.")

            for col in compare_cols:
                v1 = r1[col] if col in r1.index else ""
                v2 = r2[col] if col in r2.index else ""
                if v1 != v2:
                    diffs.append({"key": k, "column": col, "file1_value": v1, "file2_value": v2})
        diffs_df = pd.DataFrame(diffs)
        diffs_df = diffs_df[diffs_df['column'].str.contains('figs')]       
        diffs_df.to_csv(out_diffs, index=False)
        #pd.DataFrame({"key": only1}).to_csv(out_only_1, index=False)
        #pd.DataFrame({"key": only2}).to_csv(out_only_2, index=False)
