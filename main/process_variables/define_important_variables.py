import pandas as pd
import re

class DefineEssentialFormVars():
    """
    Class to define important variables for
    each form that will be used 
    frequently throughout the QC. 
    This includes important date variables,
    variables indicating that the form is missing,
    and variables indicating that the form is complete.

    Attributes
    -------------
    data_dictionary_df : pd.DataFrame
        Dataframe of the REDCap data dictionary
    
    Returns
    --------------
    important_form_vars : dict
        Dictionary with each form as a key and 
        important variables for that form
        that are needed for QC
    """

    def __init__(self, data_dictionary_df):
        self.data_dictionary_df = data_dictionary_df   
        self.all_forms = list(set(self.data_dictionary_df['Form Name'].tolist()))     

    def __call__(self):
        # collect all interview dates, entry_dates, and missing form variables
        all_import_vars = self.collect_important_vars() 

        # assign these variables to their respective forms
        important_form_vars = self.assign_variables_to_forms(all_import_vars)

        important_form_vars = self.collect_no_miss_cutoffs(important_form_vars)
        
        # returns dictionary of each form and its variables collected in this class
        return important_form_vars
    
    def collect_no_miss_cutoffs(self, form_vars):
        for form,var_info in form_vars.items():
            form_df = self.data_dictionary_df[
            self.data_dictionary_df['Form Name'] == form]

            form_df = form_df[
            form_df['Branching Logic (Show field only if...)'] == '']

            form_df = form_df[
            form_df['Identifier?'] == '']

            form_df = form_df[
            ~form_df['Field Type'].isin(['checkbox','notes','descriptive'])]

            non_branch_logic_vars = form_df['Variable / Field Name'].tolist()

            form_vars[form]['non_branch_logic_vars'] = non_branch_logic_vars

        return form_vars

    def create_list_from_df(
        self, col_to_check : str, strings_to_check : list
    ) -> list:
        """
        Creates a list from a dataframe
        of every variable where a specified
        column contains a specified string

        Parameters
        -------------
        col_to_check : str
            Column to check for containing 
            the string
        str_to_check : list
            list of strings the column 
            is checked for containing 

        Returns
        -------------
        variable_list : list
            List of all variables
            from each row where the 
            col_to_check column contained
            the str_to_check

        """

        # filters dataframe to only include rows where
        # col_to_check contains any string in strings_to_check
        filtered_df = self.data_dictionary_df[
            self.data_dictionary_df[col_to_check].str.contains(
                '|'.join(strings_to_check), regex=True
            )
        ]

        # create list of all variables in these rows
        variable_list = filtered_df['Variable / Field Name'].tolist()

        return variable_list
    
    def collect_user_variables(self):
        filterered_df = self.data_dictionary_df[
        (self.data_dictionary_df[
        'Variable / Field Name'].str.endswith('redcap_user')) | self.data_dictionary_df[
        'Variable / Field Name'].isin(['chrcssrsfu_redacp_user','chrpps_redcao_user'])]
        
        user_vars = filterered_df['Variable / Field Name'].tolist()

        return user_vars
   
    def collect_important_vars(self) -> dict:
        """
        Creates lists of all entry date,
        interview date, and missing form 
        variables

        Returns
        -------------
        all_import_vars : dict
            dictionary that contains 
            lists for each type of
            variable being collected 
            for each form
        """

        # define dictionary to store each type of variable
        all_import_vars = {'missing_var':[], 'missing_spec_var':[],\
        'interview_date_var':[],'entry_date_var':[]}

        # collects missing variables based on their descriptions
        all_import_vars['missing_var'] = self.create_list_from_df(
        'Choices, Calculations, OR Slider Labels',
        ['click if this form is missing all of its data'])

        all_import_vars['missing_spec_var'] = self.create_list_from_df(
        'Choices, Calculations, OR Slider Labels',
        [('M1, M1 - Measure refusal (no reason provided)'
        ' | M2, M2 - No show | M3, M3 - Research assistant forgot |'
        ' M4, M4 - Uncontrollable circumstance | M5, M5 - Participant dropped out')])
        
        # collects interview dates
        all_import_vars['interview_date_var'] = self.create_list_from_df(
        'Field Label',
        ['Date of Interview',
        'Interview date', 'Interview Date'])

        # adds interview date variables that had different descriptions in data dictionary
        # NOTE: mri_entry_date fits the role of interview date better than entry date 
        all_import_vars['interview_date_var'].extend(['chrcrit_date',
        'chric_consent_date','chrcbccs_review_date',
        'chrgpc_date','chrap_date','chrsaliva_interview_date',
        'chrblood_interview_date','chrcbc_interview_date',
        'chrchs_interview_date','chreeg_interview_date',
        'enrollmentnote_dateofconsent','chrconv_interview_date',
        'chric_reconsent_date','chrpred_interview_date','chrmri_entry_date']) 
        
        # collects entry dates
        all_import_vars['entry_date_var'] = self.create_list_from_df(
        'Field Label',
        ['Date of Data Entry'])

        all_import_vars['entry_date_var'].extend([
        'chrblood_entry_date','chrsaliva_entry_date',
        'chrpred_entry_date'])

        all_import_vars['redcap_user_var'] = self.collect_user_variables()
        
        # filters out irrelevant date variables
        for var_type in ['entry_date_var','interview_date_var']:
            all_import_vars[var_type] = [var for var in 
            all_import_vars[var_type] if 'err' not
            in var and 'invalid' not in var]
            
        return all_import_vars

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
            important_form_vars.setdefault(form,{'form':form})
            for var_type, var_names in unique_var_dict.items(): 
                important_form_vars[form].setdefault(var_type,'')
                for var_name in var_names: 
                    if var_name in curr_form_variables: # if variable belongs to current form
                        important_form_vars[form][var_type] = var_name 
                # form completion variables are not in the data dictionary
                # but they have the same naming conventions
                important_form_vars[form]['completion_var'] = form + '_complete' 

        return important_form_vars


class CollectMiscVariables():
    """
    class to organize any other
    groups of variables that will
    be needed by the QC
    """

    def __init__(self, data_dictionary_df):
        self.data_dictionary_df = data_dictionary_df   
        self.all_forms = list(set(self.data_dictionary_df['Form Name'].tolist()))     

    def __call__(self):
        var_info = {"blood_vars":self.collect_blood_var_types(),
                    "scid_vars": {"module_b_vars":self.collect_scid_module_b_vars(),
                                  "module_c_vars":self.collect_scid_module_c_vars()},
                    "var_forms" : self.collect_form_per_var(),
                    'var_translations' : self.create_variable_translations(),
                    'pharm_vars':self.collect_pharm_vars()}

        return var_info

    def collect_blood_var_types(self):
        blood_df = self.data_dictionary_df[
        self.data_dictionary_df[
        'Form Name'] == 'blood_sample_preanalytic_quality_assurance']
        all_blood_vars = blood_df['Variable / Field Name'].tolist()
        blood_var_categs = {}
        blood_var_categs['position_variables'] = [
        var for var in all_blood_vars if 'pos' in var]
        blood_var_categs['id_variables'] = [
        var for var in all_blood_vars if 'id' in var and var != 'chrblood_pl1id_2']
        blood_var_categs['volume_variables'] =  [
        var for var in all_blood_vars if 'vol' in var and 'error' not in var]

        blood_var_categs['barcode_variables'] = self.collect_blood_barcode_vars()

        return blood_var_categs
    
    def collect_blood_barcode_vars(self):
        filterered_df = self.data_dictionary_df[
        self.data_dictionary_df[
        'Form Name'] == 'blood_sample_preanalytic_quality_assurance']

        filterered_df = filterered_df[
        filterered_df['Field Label'].str.contains('barcode')]

        barcode_vars = filterered_df['Variable / Field Name'].tolist()

        return barcode_vars
    
    def collect_form_per_var(self):
        filterered_df = self.data_dictionary_df[['Variable / Field Name','Form Name']]
        filterered_df = filterered_df.rename(
        columns={'Variable / Field Name': 'variable',
        'Form Name': 'form'})

        form_per_var = filterered_df.set_index('variable')['form'].to_dict()

        return form_per_var

    def collect_scid_module_b_vars(self):
        scid_df = self.data_dictionary_df[
        self.data_dictionary_df['Form Name'] == 'scid5_psychosis_mood_substance_abuse']
        all_scid_vars = scid_df['Variable / Field Name'].tolist()
        exceptions = ['chrscid_bipolar_sub_desc',
        'chrscid_bp_current_severity',
        'chrscid_bp_recent_ep']
        
        module_b_vars = [var
        for var in all_scid_vars if 'chrscid_b'
        in var and var not in exceptions]
    
        return module_b_vars
    
    def collect_scid_module_c_vars(self):
        scid_df = self.data_dictionary_df[
        self.data_dictionary_df['Form Name'] == 'scid5_psychosis_mood_substance_abuse']
        all_scid_vars = scid_df['Variable / Field Name'].tolist()
        
        module_c_vars = []
        for var in all_scid_vars:
            for num in range(1,80):
                if f'c{num}' in var:
                    module_c_vars.append(var)
    
        return module_c_vars


    def create_variable_translations(self):
        """Removes some unwanted characters from
        branching logic and adds them to translation
        dictionary

        Parameters
        -----------
        col_values: values from
        current data dictionary row
        """  
        var_translations = {}

        col_renames = {'Variable / Field Name':'var','Field Label':'field_label',
        'Choices, Calculations, OR Slider Labels':'choices'}

        filtered_data_dict = self.data_dictionary_df[list(col_renames.keys())]
        filtered_data_dict.rename(columns=col_renames, inplace=True)
        
        for row in filtered_data_dict.itertuples():
            if str(row.field_label) != '':
                pattern = r'<.*?>'
                replacement_text = ''
                translation = row.field_label
                translation = re.sub(pattern,\
                replacement_text, translation)
                for char in ['<','>','/','\n','Â']:
                    translation =\
                    translation.replace(char,'')
                var_translations[
                row.var] = row.var + ' = ' +  translation
            else:
                var_translations[row.var] = row.var + ' = ' +  row.choices

        return var_translations

    def collect_pharm_vars(self):
        """
        Collects pharmaceutical variables
        used in later checks
        """
        
        pharm_df = self.data_dictionary_df[
        self.data_dictionary_df[
        'Form Name'].str.contains('pharmaceutical')]
        
        col_renames = {'Variable / Field Name':'var',
        'Form Name':'form'}

        pharm_df = pharm_df[list(col_renames.keys())]
        print(pharm_df)

        pharm_df = pharm_df.rename(columns=col_renames)
        print(pharm_df)

        pharm_vars_categorized = {
        'name_vars':[],
        'firstdose_vars':[],
        'med_status_vars':[]
        }
        
        pattern = r"chrpharm_med\d*_mo\d*"
        for row in pharm_df.itertuples():
            if re.match(pattern, row.var):
                pharm_vars_categorized.append(row.var)
            
        return pharm_vars_categorized

