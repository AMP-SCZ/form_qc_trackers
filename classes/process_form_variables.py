import pandas as pd,numbers,math,sys,datetime,random,re,os # pandas 1.4.2
from openpyxl import load_workbook
from openpyxl.worksheet.dimensions import ColumnDimension
from openpyxl.utils import range_boundaries,get_column_letter
import openpyxl

import json

class ProcessVariables():
    """Class to collect and organize 
    variables from the data
    dictionary"""

    def __init__(self,dataframe,timepoint, sheet_title):
        self.location = ''
        if self.location =='pnl_server':
            self.combined_df_folder = '/data/predict1/data_from_nda/formqc/'
            self.combined_cognition_folder = ''
            self.penn_path = '/data/predict1/data_from_nda/'
            self.absolute_path = '/PHShome/ob001/anaconda3/new_forms_qc/QC/'
        else:
            self.combined_df_folder =\
            'C:/formqc/AMPSCZ_QC_and_Visualization/QC/combined_csvs/'
            self.penn_path = ''
            self.combined_cognition_folder = 'cognition/'
            self.absolute_path = ''

        self.sheet_title = sheet_title

        with open(f'{self.absolute_path}data_storage.json', 'r') as file:
            self.json_data = json.load(file)

        if 'PRONET' in dataframe:
            self.prescient = False
            self.site_str = 'PRONET'
        else:
            self.site_str = 'PRESCIENT'
            self.prescient = True

        self.timepoint = timepoint
        self.missing_code_list =  ['-3','-9',-3,-9,-3.0,-9.0,'-3.0','-9.0',\
        '1909-09-09','1903-03-03','1901-01-01','-99',-99,-99.0,\
        '-99.0',999,999.0,'999','999.0'] #TODO: make sure these can never be legitimate values

        self.read_dataframes(dataframe)
        self.initialize_penn_data()
        self.create_timepoint_dict()
        self.define_form_exceptions()

        self.variable_info_dictionary = {'baseline_dates_forms': {},\
        'baseline_missing_variables':[],'forms_without_missing_variable':[],
        'total_num_form_variables':{},'every_missing_variable':[],\
        'variable_translation_dict':{},'all_date_variables':[],\
        'all_checkbox_variables':[],'variable_list_dictionary':{},
        'included_subjects':[], 'total_error_list':[],\
        'baseline_dates_variables':[],'unique_form_variables':{},
        'comments_dictionary': {}, 'all_csv_variables': []}

        self.error_dictionary = {}
        self.match_timepoint_forms_dict_chr =\
        self.json_data['match_timepoint_forms_dict_chr']
        self.match_timepoint_forms_dict_hc =\
        self.json_data['match_timepoint_forms_dict_hc']
        for col_name, col_value in self.ampscz_df.iloc[0].iteritems():
            self.variable_info_dictionary['all_csv_variables'].append(col_name)
        self.branch_logic_edit_dictionary =\
        self.json_data['branch_logic_edit_dictionary']
        self.scid_missing_code_checks = []
        self.branching_logic_qc_dict = {}
        self.report_list = ['Main Report','Secondary Report',\
        'Blood Report','Scid Report','Cognition Report']
        self.blank_check_variables_per_report = {}
        for report in self.report_list:
            self.blank_check_variables_per_report[report] = []
        self.variable_translation_dict = {}
        self.all_barcode_variables = []
        self.blood_id_collected = False
        self.all_blood_position_variables = []
        self.all_blood_id_variables = []
        self.all_blood_volume_variables = []
        self.all_forms = []


        self.organize_variables()

    def create_timepoint_dict(self):
        """Creates a dictionary
        with every timepoint."""

        self.timepoint_variable_lists = {'removed':[],'consent':[],\
        'screen':[],'baseln':[]}
        for x in range(1,13):
          self.timepoint_variable_lists['month' + str(x)] = []
        self.timepoint_variable_lists['month18'] = []
        self.timepoint_variable_lists['month24'] = []
        self.timepoint_variable_keys = []

    def read_dataframes(self,dataframe_path):
        """Reads and modifies various 
        dataframes to be used later.

        Parameters
        ---------------
        dataframe_path: path of the current
        timepoint's combined CSV
        """

        self.screening_df = pd.read_csv(\
        f'{self.combined_df_folder}combined-{self.site_str}-screening-day1to1.csv',\
        encoding = 'unicode_escape',keep_default_na = False)

        self.conversion_df = pd.read_csv(\
        f'{self.combined_df_folder}combined-{self.site_str}-conversion-day1to1.csv',\
        encoding = 'unicode_escape',keep_default_na = False)

        self.floating_df = pd.read_csv(\
        f'{self.combined_df_folder}combined-{self.site_str}-floating-day1to1.csv',\
        encoding = 'unicode_escape',keep_default_na = False)

        self.ampscz_df =\
        pd.read_csv(f'{dataframe_path}', keep_default_na=False,\
        encoding = 'unicode_escape')

        self.data_dictionary_df = pd.read_csv(\
        f'{self.absolute_path}data_dictionaries/CloneOfYaleRealRecords_DataDictionary_2023-05-19.csv',\
        encoding = 'latin-1',keep_default_na=False)

        self.prescient_entry_statuses = pd.read_csv(\
        f'{self.absolute_path}combined_prescient_completion_status.csv',\
        keep_default_na=False)

        self.csv_mismatch_df = pd.read_csv(f'{self.absolute_path}csv_differences.csv',\
        keep_default_na = False)

        self.modify_csv_mismatch_df()
        
        self.iq_conversion_df = pd.read_csv(\
        f'{self.absolute_path}cognition/iq_tscore_conversion.csv',\
        keep_default_na=False)
        self.iq_conversion_df.iloc[0] =\
        self.iq_conversion_df.iloc[0].apply(self.convert_range_to_list)
        self.all_iq_age_ranges = []

        for column, value in self.iq_conversion_df.iloc[0].iteritems():
            if value not in self.all_iq_age_ranges:
                self.all_iq_age_ranges.append(value)

        self.fsiq_conversion_df = pd.read_csv(\
        f'{self.absolute_path}cognition/fsiq_conversion.csv')

    def modify_csv_mismatch_df(self):
        """Changes visit status format 
        and initializes variables
        used for data corrections 
        from raw CSVs
        """

        if self.timepoint == 'screen':
            timepoint_str = 'screening'
        elif self.timepoint == 'baseln':
            timepoint_str = 'baseline'
        else:
            timepoint_str = self.timepoint
        self.csv_mismatch_df =  self.csv_mismatch_df[\
        (self.csv_mismatch_df['visit'] == timepoint_str)]
        self.csv_mismatch_df =  self.csv_mismatch_df[\
        (self.csv_mismatch_df['combined_csv'] == '')]
        csv_mismatch_filter_list = self.missing_code_list.copy()
        csv_mismatch_filter_list.extend([\
        'n/a','na','none','no','nan','nil','false'])
        self.csv_mismatch_df = self.csv_mismatch_df[\
        self.csv_mismatch_df['raw_csv'].isin(csv_mismatch_filter_list)]

    def organize_variables(self):
        """Function for collecting informtation 
        from the data dictionary to be used later 
        in the script."""

        self.collect_psychs_variables()
        self.initialize_scid_variables()
        self.process_data_dictionary()
        self.collect_variables_not_yet_added_to_dictioanry()
        self.collect_forms_without_missing_variables()
        self.collect_included_subjects()
        self.collect_checkbox_variables()
        print(self.variable_info_dictionary['total_num_form_variables'])

    def convert_range_to_list(self,range_str,str_conv = False):
        """Converts a range to a list of every
        value that would be in that range. Used
        for age ranges in IQ tables.

        Parameters 
        --------------
        range_str: string of the range 
        str_conv: determines whether the
        output will be a string or not
        """
        
        range_list = []
        if '-' not in range_str:
            if str_conv ==True:
                return [str(range_str).replace(' ','')]
            else:
                return range_str
        first_item = int(range_str.split('-')[0])
        last_item = int(range_str.split('-')[1])
        for x in range(first_item,last_item+1):
            if str_conv ==True:
                new_item = str(x).replace(' ','')
            else:
                new_item = x
            range_list.append(new_item)
        return range_list

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

    def initialize_penn_data(self):
        self.penn_data_summary_df = []
        self.penn_combined_data_no_split = []
        self.penn_combined_data_split = []
        self.penn_sheet_name = self.timepoint

    def define_form_exceptions(self):
        """Initializes variables associated with
        various exceptions to generalizations made
        in the code."""

        self.specific_value_check_dictionary = {'chrspeech_upload':\
        {'correlated_variable':'chrspeech_upload','checked_value_list':[0,0.0,'0','0.0'],\
        'branching_logic':"",'negative':False,'message':'Speech sample not uploaded to Box',\
        'report':['Secondary Report']},
        'chrpenn_complete':{'correlated_variable':'chrpenn_complete',\
        'checked_value_list':[2,2.0,'2','2.0',3,3.0,'3','3.0'],\
        'branching_logic':"",'negative':False,\
        'message': f'Penncnb not completed (value is either 2 or 3).',\
        'report':['Secondary Report','Cognition Report']}}
        self.initialize_unique_variable_names()
        self.excluded_21day_dates = ['chrcbc_interview_date',\
        'chrcbccs_review_date','chrgpc_date','chrpsychs_fu_interview_date',\
        'chrpred_interview_date','chrscid_interview_date','chrdemo_interview_date']
        self.excluded_variables_test = []
        self.removed_participants_forms = ['guid_form', 'sociodemographics'] 
        self.additional_variables =\
        ['chrpsychs_av_dev_desc', 'chrcrit_included',\
        'chrchs_timeslept','chrdemo_age_mos_chr',\
        'chrdemo_age_mos_hc','chrdemo_age_mos2','chroasis_oasis_1',\
        'chroasis_oasis_3','chrblood_rack_barcode']
        self.secondary_report_variables_additional_check = ['chrtbi_sourceinfo',\
        'chrspeech_uppast_pharmaceutical_treatment','family_interview_for_genetic_studies_figs',\
        'enrollment_noteload','chrpenn_complete']

        #self.initialize_scid_variables()
        self.define_excluded_data()
        
    def define_excluded_data(self):
        """Defines various forms and variables that will be 
        excluded from different parts of the checks."""

        self.excluded_from_blank_check = ['chroasis_oasis_1','chroasis_oasis_3','chrchs_timeslept']
        self.excluded_prescient_forms = ['family_interview_for_genetic_studies_figs']
        self.excluded_self_report_forms = ['pubertal_developmental_scale',\
        'psychosis_polyrisk_score','oasis','item_promis_for_sleep','pgis',\
        'perceived_stress_scale','perceived_discrimination_scale']

        self.excluded_prescient_strings = ['chrdemo_racial','chrsaliva_food',\
        'chrscid_overview_version','wb3id','se3id','se2id','wb2id',\
        'chrblood_freezerid','chrdbb_phone_model','chrdbb_phone_software'] 
     
        self.excluded_pronet_strings = ['comment', 'note','chrcssrsb_cssrs_yrs_sb',\
        'chrcssrsb_sb6l','chrcssrsb_nmapab','chrpas_pmod_adult3v3',\
        'chrpas_pmod_adult3v1','chrmri_t2_ge','chrcssrsfu_skip_aa',\
        'chric_surveys','chrdbb_phone_model','chrdbb_phone_software']

        self.missing_variable_exceptions = ['past_pharmaceutical_treatment',\
        'scid5_psychosis_mood_substance_abuse']

        self.forms_with_exceptions = ['psychs_p1p8_fu_hc','psychs_p9ac32_fu_hc',\
        'psychs_p1p8','psychs_p9ac32','psychs_p1p8_fu','psychs_p9ac32_fu']

        self.excluded_forms = []
        if not self.prescient:
            self.excluded_strings = self.excluded_pronet_strings
        else:
            self.excluded_forms.extend(self.excluded_prescient_forms)
            self.excluded_strings = self.excluded_pronet_strings
            self.excluded_strings.extend(self.excluded_prescient_strings)
            self.excluded_strings.extend(['chrpsychs_av_dev_desc','chrguid_interview'])
       
    def initialize_scid_variables(self):
        """Creates lists that will be used for 
        various SCID form checks"""

        self.scid_diagnosis_check_dictionary =\
        self.json_data['scid_diagnosis_check_dictionary']
        self.scid_additional_variables = self.json_data['scid_additional_variables']
        self.specific_value_check_scid_variables = ['chrscid_as20'\
        ,'chrscid_a51','chrscid_a70','chrscid_a91','chrscid_a108','chrscid_a129',\
        'chrscid_a138','chrscid_a153','chrscid_as124','chrscid_a170','chrscid_a192',\
        'chrscid_a198','chrscid_a206','chrscid_a213','chrscid_a221','chrscid_c10',\
        'chrscid_c14','chrscid_c20','chrscid_c26','chrscid_c37',\
        'chrscid_c44','chrscid_c71','chrscid_c78','chrscid_d9']
        for variable in self.specific_value_check_scid_variables:
            checked_value_list = [-9,-9.0,'-9','-9.0',\
            1,1.0,'1','1.0',3,3.0,'3','3.0','']
            checked_value_list.extend(self.missing_code_list)
            self.specific_value_check_dictionary[variable]\
            = {'correlated_variable':variable,\
            'checked_value_list':checked_value_list,\
            'branching_logic':"",'negative':True,\
            'message': f'Value should be 1,3, or -9/NA','report':['Scid Report']}
        for key in self.scid_diagnosis_check_dictionary.keys():
            self.additional_variables.append(key)
        self.additional_variables.extend(self.scid_additional_variables) 
        self.additional_variables.extend(self.specific_value_check_scid_variables)
        self.excluded_from_blank_check.extend(self.specific_value_check_scid_variables)

    def collect_variables_not_yet_added_to_dictioanry(self):
        """missing data variables that haven't been 
        added to data dictionary yet, so they 
        are manually being added here"""

        for key, item in {'digital_biomarkers_axivity_onboarding':'chrax_missing',\
        'digital_biomarkers_axivity_checkin':'chraxci_missing',\
        'digital_biomarkers_axivity_end_of_12month_study_pe': 'chraxe_missing',\
        'digital_biomarkers_mindlamp_checkin':'chrdig_missing_all'}.items():
            self.variable_info_dictionary['unique_form_variables'][key]['missing_variable'] = item

    def collect_blood_variables(self,variable):
        """Initializes lists used for blood form
        checks

        Parameters
        -------------
        variable: variable from current row of data dictionary
        """
        if 'blood' in variable and 'pos' in variable\
        and variable not in self.all_blood_position_variables:
            self.all_blood_position_variables.append(variable)
        if 'blood' in variable and 'id' in variable\
        and variable not in self.all_blood_id_variables and\
        'freezer' not in variable and variable != 'chrblood_pl1id_2':
            self.all_blood_id_variables.append(variable)
        if 'blood' in variable and 'vol' in variable\
        and variable not in self.all_blood_volume_variables:
            self.all_blood_volume_variables.append(variable)

    def initialize_report_variables(self, report, col_values):
        """Adds variable to corresponding report

        Parameters
        -------------
        report: report/sheet of interest
        col_values: values from current data dictionary row
        """

        col_values['branching_logic'] = self.branch_logic_edit_dictionary.get(\
        col_values['variable'], col_values['branching_logic'])
        
        self.variable_info_dictionary['total_num_form_variables'].setdefault(col_values['form'], 0)
        if not any(x in col_values['variable'] for x in self.excluded_strings):
            self.blank_check_variables_per_report[report].append(col_values['variable'])
            self.branching_logic_redcap_to_python(col_values['variable'],\
            col_values['form'],col_values['branching_logic']) 
            if report == 'Main Report' and col_values['branching_logic'] == '':
                self.variable_info_dictionary['total_num_form_variables'][col_values['form']] += 1
        if 'barcode' in str(col_values['field_label']):
            self.all_barcode_variables.append(col_values['variable'])
        if col_values['field_type'] == 'radio':
            self.scid_missing_code_checks.append(col_values['variable'])
        if col_values['field_type'] == 'checkbox':
            self.variable_info_dictionary[\
            'all_checkbox_variables'].append(col_values['variable'])

    def add_more_additional_variables(self,variable):
        if 'chrpharm_med' in variable and 'name_past' in variable:
            self.additional_variables.append(variable)

    def process_data_dictionary(self):
        """Loops through the data dictionary and uses information
        to define roles of the different variables"""

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
            if col_values['form'] not in self.all_forms:
                self.all_forms.append(col_values['form'])

            #self.add_more_additional_variables(col_values['variable'])
            self.collect_blood_variables(col_values['variable'])
            self.collect_unique_variables(col_values['form'],\
            col_values['variable'],col_values['field_type'])
            self.edit_tbi_branch_logic(col_values['variable'])
            self.edit_past_pharm_branch_logic(col_values['variable'])

            if col_values['variable'] in self.additional_variables or\
            (col_values['required_field'] == 'y' and col_values['identifier'] != 'y'\
            and ((col_values['field_type'] not in ['notes','descriptive']))\
            and col_values['form'] not in self.forms_with_exceptions and col_values['form']\
            not in self.excluded_self_report_forms):
                self.initialize_report_variables('Main Report',col_values)

            if col_values['form'] not in self.forms_with_exceptions and\
            col_values['required_field'] == 'y' and col_values['identifier'] != 'y'\
            and ((col_values['field_type'] in ['notes','descriptive'])\
            or (col_values['form'] in self.excluded_self_report_forms)):
                self.initialize_report_variables('Secondary Report',col_values)

            self.create_variable_translations(col_values)

    def create_variable_translations(self,col_values):
        """Removes some unwanted characters from
        branching loggic and adds them to translation
        dictionary

        Parameters
        -----------
        col_values: values from
        current data dictionary row
        """

        if str(col_values['field_label']) not in ['nan','']:
            pattern = r'<.*?>'
            replacement_text = ''
            col_values['field_label'] = re.sub(pattern,\
            replacement_text, col_values['field_label'])
            for char in ['<','>','/','\n','Â']:
                col_values['field_label'] =\
                col_values['field_label'].replace(char,'')
            self.variable_translation_dict[col_values[\
            'variable']] = col_values['variable']\
            + ' = ' +  col_values['field_label']
        else:
            self.variable_translation_dict[\
            col_values['variable']] = col_values['variable']\
            + ' = ' +  col_values['choices']

    def edit_tbi_branch_logic(self,variable):
        """Modifies branching logic for 
        TBI form to only check the variables 
        that correspond to the number of injuries.

        Parameters
        -----------
        variable: variable from current
        data dictionary row
        """

        if 'chrtbi' in variable:
            for injury_count in [4,5]:
                if f'{injury_count}' in variable:
                    if 'parent' not in variable:
                        self.branch_logic_edit_dictionary[str(variable)]\
                        = (f"[chrtbi_number_injs] = '{injury_count}'"
                        " and [chrtbi_subject_times] = '3'")
                    else:
                        self.branch_logic_edit_dictionary[str(variable)]\
                        = (f"[chrtbi_number_injs] = '{injury_count}'"
                        " and [chrtbi_parent_times] = '3'")

    def edit_past_pharm_branch_logic(self,variable):
        """Edits pharm branching logic to account
        for subject selecting no medication for 
        name of medication

        Parameters 
        --------------
        variable: current variable being processed
        """

        if 'chrpharm_med' in variable:
            number = self.collect_digit(variable)
            if number not in ['1','']:
                new_branching_logic = \
                (f"[chrpharm_med{number}_name_past] <> '999' and"\
                " [chrpharm_med{int(number)-1}_add_past] = '1'")
            else:
                new_branching_logic = \
                (f"[chrpharm_med{number}_name_past] <> '999'"
                " and [chrpharm_med_past] = '1'")
            if 'onset_past' in variable:
                self.branch_logic_edit_dictionary[\
                f"chrpharm_med{number}_onset_past"]\
                = new_branching_logic
            elif 'offset_past' in variable:
                self.branch_logic_edit_dictionary[\
                f"chrpharm_med{number}_offset_past"]\
                =new_branching_logic

    def collect_digit(self,string):
        """Collects digit in current string
        
        Parameters
        ------------
        string: string containing digit
        """

        for char in string:
            if char.isdigit():
                return char
        return ''

    def collect_forms_without_missing_variables(self):
        """collecting forms that do not have working missing
        data buttons, so that the work-around can be used on them later """  

        for key,value in self.variable_info_dictionary['unique_form_variables'].items():
            if 'missing_variable' not in value and key not in\
            self.variable_info_dictionary['forms_without_missing_variable']\
            and key not in self.missing_variable_exceptions:
                self.variable_info_dictionary['forms_without_missing_variable'].append(key)
        if self.prescient:  
            self.variable_info_dictionary['forms_without_missing_variable'].extend(['guid_form'])

    def collect_included_subjects(self):
        """Creates a list of all included subjects"""

        for row in self.screening_df.itertuples():
            if row.chrcrit_included in [1,1.0,'1','1.0']: 
                self.variable_info_dictionary['included_subjects'].append(row.subjectid)

    def collect_psychs_variables(self):
        """Function to collect all of the Psychs
        variables that will be checked and
        adds them to the additional variables list"""

        letter_match_dict = {'b': ['5','6','19'],'d': ['16','30']}
        for key in letter_match_dict.keys():
            for x in range (1,16):
                for number in letter_match_dict[key]:
                    self.additional_variables.append('chrpsychs_scr_'\
                    + str(x) + key + number)
                    self.additional_variables.append('chrpsychs_scr_'\
                    + str(x) + key + number + '_app')
                    if key == 'd':
                        self.additional_variables.extend(['chrpsychs_fu_'+\
                        str(x) + key + number,\
                        'chrpsychs_fu_'+ str(x) + key + number+'_app',\
                        'hcpsychs_fu_'+ str(x) + key + number,'hcpsychs_fu_'+\
                        str(x) + key + number+'_app'])
        self.additional_variables.extend(['chrpsychs_scr_e11',\
            'chrpsychs_scr_e27','chrpsychs_scr_e11_app',\
            'chrpsychs_scr_e27_app','chrpsychs_fu_e27',\
            'chrpsychs_fu_e27_app','hcpsychs_fu_e27','hcpsychs_fu_e27_app'])
        for x in self.additional_variables:
            if 'app' in x and 'psychs' in x:
                self.specific_value_check_dictionary[x] =\
                {'correlated_variable':x,'checked_value_list':[0,0.0,'0','0.0'],\
                'branching_logic':"",'negative':False,\
                'message': f'value is 0','report':['Main Report']}
                
    def branching_logic_redcap_to_python(self,variable,form,branching_logic):
        """This function focuses on converting the syntax
        from the REDCap branching logic in the data dictionary
        into Python syntax to be used as conditionals later in the code.

        Parameters
        ----------------
        variable: current variable of interest
        form: current of interest
        branching logic: redcap version of branching logic 
        """

        self.variable_info_dictionary['variable_list_dictionary'][form][variable] = {
                    'branching_logic': str(branching_logic).replace('[', '').replace(']', '').\
                    replace('<>', '!=').replace('OR', 'or').replace('AND', 'and').replace("\n", ' ')}
        patterns_replacements = [
            # Replaces single equals sign "=" with double equals sign "==" 
            (r"(?<!=)(?<![<>!])=(?!=)", r"=="),  
            # Converts numbers in single quotes to floats  by adding "float()" around it .
            (r"([=<>]\s*)(-?\d+(\.\d+)?)", r"\1float(\2)"), 
            # Converts numeric values preceded by a comparison operator (=, <, >) by adding the "float()" function.
            (r"'(-?(?!00)\d+(\.\d+)?)'", r"float(\1)"),  
            # Adds "row." to the beginning of variable names or function calls followed by a comparison operator (!=, =, <, >) 
            (r"\b([\w\.]+(?:\(\d+\))*?)\s*(!=|=|<|>)\s*", r"row.\1\2"), 
            # Replaces numbers in parentheses with "___" appended to the beginning (for checkbox variables) 
            (r"(?<!float)\((\d+)\)", r"___\1"),  
            # Adds the "float()" function to variable names starting with "row." 
            # if it is followed by a comparison operator and a float number
            (r"(row\.\w+)(?==|!=|>|<|>=|<=)(?! )(?!=='00)(?=.*?\bfloat\()", r"float(\1)")  
        ] 

        for pattern, replacement_text in patterns_replacements: 
            self.variable_info_dictionary['variable_list_dictionary']\
            [form][variable]['branching_logic'] = \
                re.sub(pattern, replacement_text,\
                self.variable_info_dictionary['variable_list_dictionary']\
                [form][variable]['branching_logic'])
        if str(self.variable_info_dictionary['variable_list_dictionary']\
            [form][variable]['branching_logic']) == 'nan':
            self.variable_info_dictionary['variable_list_dictionary']\
            [form][variable]['branching_logic'] = ''

    def collect_unique_variables(self,form,variable,field_type):
        """Function to collect various relevant variables for each form
         to be used later (complete variables, missing variables, etc.)
        
        Parameters:
        -----------------
        form: form that the variable belongs to
        variable: current variable being checked
        field_type: field type of variable
        """
        
        if form not in self.variable_info_dictionary['variable_list_dictionary']:
            self.variable_info_dictionary['variable_list_dictionary'][form] = {}
            self.variable_info_dictionary['unique_form_variables'][form] = {}
        if (variable.endswith('_missing') and form not in self.forms_with_unique_missing_variables) \
        or (variable in self.unique_missing_variable_names\
        and form in self.forms_with_unique_missing_variables):
            self.variable_info_dictionary['every_missing_variable'].append(variable)
            self.variable_info_dictionary['unique_form_variables'][form]['missing_variable'] = variable
        if variable.endswith('interview_date') or variable in self.unique_date_variable_names:
            self.variable_info_dictionary['unique_form_variables'][form]['interview_date'] = variable 
        self.variable_info_dictionary['unique_form_variables'][form]['complete_variable'] =  form + '_complete'
        if '_date' in variable and 'error' not in variable\
        and 'add' not in variable and 'invalid' not in variable\
        and 'mod' not in variable and 'first' not in variable and '_err' not in variable:
            self.variable_info_dictionary['all_date_variables'].append(variable)   
        
    def collect_checkbox_variables(self):
        """Checkbox variables are formatted
        differently in the data dictionary
        and combined csv. This will collect
        all of the variables as they appear in the CSVs"""

        self.checkbox_variable_dictionary = {}
        for x in self.variable_info_dictionary['all_checkbox_variables']:
            for y in self.variable_info_dictionary['all_csv_variables']:
                if x + '___' in y:
                    if x not in self.checkbox_variable_dictionary:
                         self.checkbox_variable_dictionary[x] = []
                    self.checkbox_variable_dictionary[x].append(y)



