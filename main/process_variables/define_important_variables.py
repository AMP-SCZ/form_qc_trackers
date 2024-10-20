import pandas as pd

class DefineImportantVariables():
    """
    Class to define important variables for
    each form that will be used 
    frequently throughout the QC. 

    Attributes
    -------------
    data_dictionary_df : pd.DataFrame
        Dataframe of the REDCap data dictionary
    
    Returns
    --------------
    important_form_vars : dict
        Dictionary of each form and 
        important variables for that form
        that are needed for QC
    """

    def __init__(self, data_dictionary_df):
        self.data_dictionary_df = data_dictionary_df   
        self.all_forms = list(set(self.data_dictionary_df['Form Name'].tolist()))     

    def __call__(self):
        # collect all interview dates, entry_dates, and missing form variables
        all_unique_vars = self.collect_important_vars() 

        # assign these variables to their respective forms
        important_form_vars = self.assign_variables_to_forms(all_unique_vars)
        
        # returns dictionary of each form and its variables collected in this class
        return important_form_vars

    def create_list_from_df(
        self, col_to_check : str, str_to_check : str
    ) -> list:
        """
        Creates a list from a dictionary
        of every variable where a specified
        column contains a specified string

        Parameters
        -------------
        col_to_check : str
            Column to check for containing 
            the string
        str_to_check : str
            String the the col_to_check values
            are checked for containing

        Returns
        -------------
        variable_list : list
            List of all variables
            from each row where the 
            col_to_check column contained
            the str_to_check

        """

        # filter dataframe to only include rows where
        # col_to_check contains str_to_check
        filtered_df = self.data_dictionary_df[\
        self.data_dictionary_df[col_to_check].str.contains(\
        str_to_check)]

        # create list of all variables in these rows
        variable_list = filtered_df['Variable / Field Name'].tolist()

        return variable_list
                  
    def collect_important_vars(self) -> dict:
        """
        Creates lists of all entry date,
        interview date, and missing form 
        variables

        Returns
        -------------
        all_unique_vars : dict
            dictionary that contains 
            lists for each type of
            variable being collected 
            for each form
        """

        # define dictionary to store each type of variable
        all_unique_vars = {'missing_var':[],\
        'interview_date_var':[],'entry_date_var':[]}

        # collects missing variables based on their descriptions
        all_unique_vars['missing_var'] = self.create_list_from_df(
        'Choices, Calculations, OR Slider Labels',
        'click if this form is missing all of its data')

        # collects interview dates
        all_unique_vars['interview_date_var'] = self.create_list_from_df(
        'Field Label',
        'Date of Interview')

        # adds interview date variables that had different descriptions in data dictionary
        all_unique_vars['interview_date_var'].extend(['chrcrit_date',
        'chric_consent_date','chrcbccs_review_date',
        'chrgpc_date','chrap_date'])
        
        # collects entry dates
        all_unique_vars['entry_date_var'] = self.create_list_from_df(
        'Field Label',
        'Date of Data Entry')
        
        # filters out irrelevant date variables
        for var_type in ['entry_date_var','interview_date_var']:
            all_unique_vars[var_type] = [var for var in 
            all_unique_vars[var_type] if 'err' not
            in var and 'invalid' not in var]

        return all_unique_vars

    def assign_variables_to_forms(
        self, unique_var_dict : dict
    ) -> dict: 
        """
        Organizes all important variable types into
        a dictionary with each form as the key and 
        its essential variables as the values

        Parameters
        --------------------
        unique_var_dict : dict
            Dictionary of lists for 
            each variables type (interview dates,
            missing variables, and entry dates)

        Returns
        ------------------
        important_form_vars : dict
            Dictionary with each form
            as the key and its respective
            varaibles that were collected
        """

        important_form_vars = {}
        # creates list of all forms in the data dictionary
        
        for form in self.all_forms:
            # creates df with only the rows that belong to current form
            form_df = self.data_dictionary_df[\
            self.data_dictionary_df['Form Name'] == form]
            curr_form_variables = form_df['Variable / Field Name'].tolist()
            important_form_vars.setdefault(form,{})
            for var_type, var_names in unique_var_dict.items(): 
                important_form_vars[form].setdefault(var_type,'')
                for var_name in var_names: 
                    if var_name in curr_form_variables: # if variable belongs to current form
                        important_form_vars[form][var_type] = var_name 

                # form completion variables are not in the data dictionary
                important_form_vars[form]['completion_var'] = form + '_complete' 

        return important_form_vars

        









