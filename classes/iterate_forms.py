import sys 
import os

from classes.process_form_variables import ProcessVariables
from classes.data_checks import DataChecks

import math
import pandas as pd
import datetime
class IterateForms(ProcessVariables):
    """Class to loop through each form and filter out
    variables/forms based on data collected in parent class."""

    def __init__(self,dataframe,timepoint, sheet_title):
        super().__init__(dataframe,timepoint, sheet_title)
        self.current_prescient_complete_value = ''
        
    def assign_forms_to_timepoints(self):
        """Assigns different forms to the current timepoint,
        depending on if the subject being checked is CHR or HC"""

        self.timepoint_variable_lists = {key: [] for\
        key in self.timepoint_variable_lists}
        self.timepoint_variable_keys = []
        current_subject = self.row.subjectid 
        matching_rows = self.screening_df.loc[\
        self.screening_df['subjectid'] == current_subject, 'chrcrit_part']  
        if not matching_rows.empty:
          hc_or_chr = matching_rows.iloc[0]
          if hc_or_chr in [1,1.0,'1','1.0',2,2.0,'2','2.0']: #TODO disregard cohort for screen and baseln
              hc_chr_dict = {1.0: self.match_timepoint_forms_dict_chr,\
            2.0:self.match_timepoint_forms_dict_hc}
              for key in self.timepoint_variable_lists.keys():
                self.timepoint_variable_keys.append(key)
                for values in hc_chr_dict[float(hc_or_chr)].values():
                    if any(key.replace('month','') == x for x in values[0]):
                      self.timepoint_variable_lists[key].extend(values[1])
         
    def append_error(self, message,  variable, form,sheet_list = ['Main Report']):
        """Function to append errors to a dictionary and collect
         comments from the form that those errors appeared in.

        Parameters
        ------------------
        message: message that will appear on the output 
        variable: variable involved in the error 
        form: form currently being flagged
        sheet_list: Sheets on the final tracker that this error
        will appear on
        """

        for sheet in sheet_list:
            self.error_dictionary.setdefault(sheet,{})
            self.error_dictionary[sheet].setdefault(self.row.subjectid,{})
            self.error_dictionary[sheet][self.row.subjectid].setdefault(form,{})
            column_headers = {"Subject":self.row.subjectid,"Timepoint":self.timepoint,\
            "Subject's Current Timepoint":self.row.visit_status_string,\
            "General Flag":f"{form.replace('_',' ').title()} : 0 flags detected.","Specific Flags":[],\
            "Variable Translations":'',"Date Flag Detected":f"{datetime.date.today()}",\
            "Time since Flag Detected":'',"Date Resolved":"",\
            "Manually Marked as Resolved":"","Sent to Site":"","Additional Comments":""}

            for column_header,default_value in column_headers.items():
                self.error_dictionary[sheet][\
                self.row.subjectid][form].setdefault(column_header,default_value)

            if f"{variable} : {message}" not in self.error_dictionary[sheet][self.row.subjectid][\
            form]["Specific Flags"]:
                self.error_dictionary[sheet][self.row.subjectid][\
                form]["Specific Flags"].append(f"{variable} : {message}")
                spec_flag_list = self.error_dictionary[sheet][self.row.subjectid][form]["Specific Flags"]
                flag_str = "flags"
                if len(spec_flag_list) == 1:
                    flag_str = "flag"
                self.error_dictionary[sheet][self.row.subjectid][form]["General Flag"] =\
                f"{form.replace('_',' ').title()} : {len(spec_flag_list)} {flag_str} detected."

                self.error_dictionary[sheet][self.row.subjectid][form]["Variable Translations"]\
                =self.add_variable_translations(spec_flag_list)

    def add_variable_translations(self, flag_list):
        """Adds variable translations for any variable that is present
        in the error list.

        Parameters
        -----------------------------
        flag_list: list of errors for the current row
        """
        translation_list = []
        for flag in flag_list:
            for word in flag.split():
                if word in self.variable_translation_dict.keys():
                    translation_list.append(\
                    self.variable_translation_dict[word])

        return translation_list

    def main_loop(self):
        """Main loop to further filter out
        which variables/forms get sent down 
        the pipeline to be checked for errors"""

        for row in self.ampscz_df.itertuples(): 
            self.row = row
            self.filter_rows(self.timepoint)
          

    def filter_rows(self,timepoint):
        """Filters out certain forms or variables and 
        allows the ones that should be checked throught to
        the error check functions.

        Parameters
        --------------
        timepoint: current timepoint being checked
        """

        self.assign_forms_to_timepoints()
        if self.timepoint == 'baseln' and\
        str(self.twenty_one_day_rule()) not in ['None','']: 
             self.append_error(self.twenty_one_day_rule(),\
             '21 day rule','psychs_p1p8_fu/psychs_p9ac32_fu',['Main Report'])
        if self.timepoint_variable_keys!= [] and\
        self.row.visit_status_string not in ['consent','converted']: 
            for form, variable_list in \
            self.variable_info_dictionary['variable_list_dictionary'].items():
                self.form = form
                if self.row.visit_status_string != 'removed' or\
                form in self.removed_participants_forms: 
                    if self.prescient==True and form in\
                    self.timepoint_variable_lists[timepoint]:
                        if self.check_prescient_na_values(form) == True:
                            continue
                    self.missing_workaround_error_count = 0
                    if form in self.timepoint_variable_lists[timepoint] and\
                    form not in self.excluded_forms \
                    and ((self.timepoint_variable_keys.index(timepoint)\
                    < self.timepoint_variable_keys.index(\
                    self.row.visit_status_string) and self.prescient == True) or \
                    (hasattr(self.row,self.variable_info_dictionary['unique_form_variables']\
                    [form]['complete_variable']) and\
                    getattr(self.row, self.variable_info_dictionary['unique_form_variables']\
                    [form]['complete_variable']) in [2,2.0,'2','2.0'])\
                    or self.row.subjectid not in \
                    self.variable_info_dictionary['included_subjects']):
                        if self.additional_conditionals() == False:
                            continue
                        self.call_error_checks(variable_list,form)
                    self.check_form_completion(timepoint)

    def check_form_completion(self,timepoint):
        """Only used for Prescient. Checks
        if form has not been completed when subject
        has moved onto next timepoint"""

        if self.form != 'informed_consent_run_sheet' and self.prescient ==True and\
        self.row.subjectid in self.variable_info_dictionary['included_subjects'] and self.form in\
        self.timepoint_variable_lists[timepoint] and self.form not in self.excluded_forms\
        and (self.timepoint_variable_keys.index(timepoint)\
         < self.timepoint_variable_keys.index(self.row.visit_status_string) and \
        (self.current_prescient_complete_value not in\
        [2,2.0,'2','2.0','3','3.0',3,3.0,'4','4.0',4,4.0])) and 'figs' not in self.form:
            self.append_error(\
            'Form not marked as complete, but subject has moved onto next timepoint',\
            self.variable_info_dictionary['unique_form_variables'][self.form]['complete_variable'],\
            self.form,['Substantial Missing Data'])

    def call_error_checks(self,variable_list,form):
        """Calls error checks if form is not marked
        as missing or passes the workaround if there
        is not missing data button

        Parameters
        -------------
        variable_list: list of variable from current form
        form: current form
        """

        if form not in self.variable_info_dictionary['forms_without_missing_variable']:
            if (form in self.missing_variable_exceptions) or\
            (hasattr(self.row,self.variable_info_dictionary[\
            'unique_form_variables'][form]['missing_variable']) \
            and str(getattr(self.row,\
            self.variable_info_dictionary['unique_form_variables'][form]['missing_variable']))\
            in ['','nan','False', "0", "0.0" ,"'0'","'0.0'"]): 
                self.error_check(variable_list) 
        elif form not in self.excluded_forms and\
        (form in self.variable_info_dictionary['forms_without_missing_variable'])\
        and form in self.variable_info_dictionary['total_num_form_variables']\
        and self.variable_info_dictionary['total_num_form_variables'][form] > 0:
            self.error_check(variable_list,True)
            if form in self.variable_info_dictionary['total_num_form_variables'] and\
            (self.missing_workaround_error_count/self.variable_info_dictionary[\
            'total_num_form_variables'][form])\
            < 0.55 and self.missing_workaround_error_count!=0.0: 
                self.error_check(variable_list,False) 

    def call_specific_value_check(self,workaround=False):
        """Filters out certain variables before calling 
        specific value check function

        Parameters
        -------------
        workaround: makes sure the missing variable workaround is not occuring
        """

        for checked_variable,conditions in self.specific_value_check_dictionary.items(): 
            if workaround == False and self.variable == conditions['correlated_variable']:
                self.specific_value_check(checked_variable,\
                    self.form,conditions['checked_value_list'],conditions['negative'],\
                    conditions['message'],conditions['branching_logic'],conditions['report']) 

    def error_check(self,variable_list,workaround=False):
        """Function used by the main_loop to check a variable for any errors

        Parameters
        ----------------
        variable_list: all variables of current form
        workaround: whether or not missing variable workaround is occuring
        """

        for variable, branch in variable_list.items(): 
            row = self.row
            self.variable = variable
            if workaround == False and (self.row.subjectid\
            not in self.variable_info_dictionary['included_subjects']\
            or self.row.visit_status_string == 'removed')\
            and self.sheet_title == 'Main Report':
                if not self.prescient:
                    self.excluded_check()
            if self.row.subjectid in\
            self.variable_info_dictionary['included_subjects']:
                self.call_specific_value_check(workaround)
                if branch['branching_logic'] in ['nan','']: 
                    for string in self.excluded_strings: 
                        if string in variable:
                            break
                    else: 
                        self.check_if_blank(workaround)
                else: 
                    for string in self.excluded_strings:
                        if string in variable:
                            break
                    else: 
                        try: 
                            if eval(branch['branching_logic']):
                                self.check_if_blank(workaround)
                        except Exception as e:
                            continue

    def check_if_blank(self,workaround):
        """Function to check if a value is blank
    
        Parameters
        ------------
        workaround: whether or not missing variable workaround is occuring
        """
        self.current_report_list = []
        for report, var_list in self.blank_check_variables_per_report.items():
            if self.variable in var_list:
                self.current_report_list.append(report)
        team_report_forms = {'Blood Report':['blood_sample_preanalytic_quality_assurance'],\
        'Cognition Report':['penncnb','premorbid_iq_reading_accuracy',\
        'iq_assessment_wasiii_wiscv_waisiv'],
        'Scid Report':['scid5_psychosis_mood_substance_abuse']}
        for report, form_list in team_report_forms.items():
            if self.form in form_list:
                self.current_report_list.append(report)
        self.call_extra_checks(self.form)
        if self.variable\
        not in self.excluded_from_blank_check:
            if hasattr(self.row, self.variable):
                if (getattr(self.row, self.variable) =='' or\
                pd.isna(getattr(self.row, self.variable))\
                or (self.prescient == True and self.sheet_title == 'Scid Report'\
                and getattr(self.row, self.variable) in self.missing_code_list) ):
                    raw_csv_variable_value = self.check_prescient_csv_mismatches(self.form)
                    if raw_csv_variable_value == '' or (self.prescient == True\
                    and self.sheet_title == 'Scid Report' and\
                    raw_csv_variable_value in self.missing_code_list\
                    and self.variable in self.scid_missing_code_checks):
                        if workaround == False:
                            if raw_csv_variable_value not in self.missing_code_list:
                                error_str = f'Value is empty'
                            elif self.variable in self.scid_missing_code_checks:
                                error_str = f'Value is a missing code ({raw_csv_variable_value}).'
                            self.append_error(error_str,self.variable,\
                            self.form,self.current_report_list)
                        else:
                            self.missing_workaround_error_count+=1 
        
    def additional_conditionals(self):
        """Additional conditions added for certain forms.
        Those forms will only be checked if the conditions 
        specified here are true."""

        if self.form == 'pubertal_developmental_scale':
            age = ''
            for age_var in ['chrdemo_age_mos_chr',\
            'chrdemo_age_mos_hc','chrdemo_age_mos2']:
                if hasattr(self.row,age_var)\
                and getattr(self.row,age_var)\
                not in (self.missing_code_list + ['']):
                    age = float(getattr(self.row,age_var))/12 
            if age == '' or age >18:
                return False
            return True
        elif 'axivity' in self.form:
            matching_rows = self.screening_df.loc[\
            self.screening_df['subjectid']\
            == self.row.subjectid, 'chric_actigraphy']
            if not matching_rows.empty\
            and not math.isnan(matching_rows):
                opt_in = matching_rows.iloc[0]
                if opt_in in [1,1.0,'1','1.0']:
                    return True      
                return False
        elif 'mindlamp' in self.form:
            matching_rows = self.screening_df.loc[\
            self.screening_df['subjectid']\
            == self.row.subjectid, 'chric_passive']
            if not matching_rows.empty\
            and not math.isnan(matching_rows):
                opt_in = matching_rows.iloc[0]
                if opt_in in [1,1.0,'1','1.0',2,2.0,'2','2.0']:
                    return True      
                return False
        else: 
            return True
   
    def check_prescient_csv_mismatches(self,form):
        if not self.prescient:
            return getattr(self.row,self.variable)
        timepoint_str = self.timepoint.replace(\
        'baseln','baseline').replace('screen','screening')
        df = self.csv_mismatch_df
        current_variable = df[(df['subject'] == self.row.subjectid) & (\
        df['visit'] == timepoint_str) & (df['variable'] == self.variable)]
        if not current_variable.empty:
            for raw_csv_row in current_variable.itertuples():
                if raw_csv_row.combined_csv != raw_csv_row.raw_csv:
                    if raw_csv_row.raw_csv in self.missing_code_list\
                    or any(x in raw_csv_row.raw_csv.lower() for\
                    x in ['n/a','na','none','no','nan','nil','false']):
                        return raw_csv_row.raw_csv 
        return getattr(self.row,self.variable)

    def check_prescient_na_values(self,form):
        df = self.prescient_entry_statuses
        form_name = form.replace('_fu_hc','_fu') 
        current_form = df[(df['Subject'] == self.row.subjectid)\
        & (df['Form_Timepoint'] == \
        self.timepoint) & (df['Form_Translation'] == form_name)]
        for row in current_form.itertuples():
            self.current_prescient_complete_value =\
            row.Completion_Status
            if row.Completion_Status\
            in [3,3.0,'3','3.0','4','4.0',4,4.0]:
                return True 
            else:
                return False

    def specific_value_check(self,variable,form,checked_value_list,
    negative,message,branching_logic = '',report = 'default'):
        """Checks if a variable has a certain
        value.

        Parameters
        -----------------
        variable: current variable in form iteration
        form: current form in form iteration 
        checked_value_list: values that the variable is checked for
        negative: determines if variable is checked as NOT being a value instead
        message: error message appended to output
        branching_logic: any branching logic of the variable 
        """
        if report =='default':
            report = self.current_report_list

        try:
            if branching_logic == '':
                variable_value = self.check_prescient_csv_mismatches(self.form)
                if (negative == False and variable_value in checked_value_list) \
                or (negative == True and variable_value not in checked_value_list) :
                    self.append_error(\
                    message,self.variable,self.form,report)
            else:
                variable_value = self.check_prescient_csv_mismatches(self.form)
                if eval(branching_logic) and  (negative == False and variable_value in \
                checked_value_list) or (negative == True\
                and variable_value not in checked_value_list):
                    self.append_error(\
                    message,self.variable,self.form,report)
        except Exception as e:
            print(e)


    def excluded_check(self):
        """Additional checks for subjects
        who were excluded or removed"""

        current_subject = self.row.subjectid 
        racial_list = []
        hc_or_chr = self.screening_df.loc[self.screening_df['subjectid'] \
        == current_subject, 'chrcrit_part'].iloc[0]
        hc_chr_dict = {1.0: 'chr', 2.0:'hc'}
        incl_or_excl = self.screening_df.loc[self.screening_df['subjectid']\
        == current_subject, 'chrcrit_excluded'].iloc[0]
        consent_date = self.screening_df.loc[self.screening_df['subjectid']\
        == current_subject, 'chric_consent_date'].iloc[0]
        if incl_or_excl in ['1','1.0',1,1.0] and hc_or_chr in [1,1.0,'1','1.0','2','2.0',2,2.0]: 
            if self.variable == 'chrdemo_sexassigned' and\
            getattr(self.row,self.variable) in ['','nan','False', "0", "0.0" ,"'0'","'0.0'"]: 
                self.append_error( f'Participant has been excluded, but their sex was not recorded.',\
                self.variable,self.form,['Main Report']) 
            elif self.variable == 'chrdemo_age_mos_' +\
            hc_chr_dict[float(hc_or_chr)] and getattr(self.row,self.variable) \
            in ['','nan','False', "0", "0.0" ,"'0'","'0.0'"]\
            and getattr(self.row,'chrdemo_age_mos2') \
            in ['','nan','False', "0", "0.0" ,"'0'","'0.0'"]:
                self.append_error(\
                f'Participant has been excluded, but their age was not recorded.',\
                self.variable,self.form,['Main Report'])
            elif 'chrdemo_racial_back' in self.variable and consent_date != ''\
            and datetime.datetime.strptime(consent_date, '%Y-%m-%d')\
            < datetime.datetime.strptime('2023-01-01', '%Y-%m-%d'):
                for x in range(1,9):
                    racial_list.append(getattr(self.row,'chrdemo_racial_back___'+str(x)))
                if self.prescient == False:
                    racial_list.extend([getattr(self.row,'chrdemo_racial_back____9'),\
                        getattr(self.row,'chrdemo_racial_back____3'),\
                        getattr(self.row,'chrdemo_racial_back___1909_09_09'),\
                        getattr(self.row,'chrdemo_racial_back___1903_03_03')])
                if not any(x in racial_list for x in [1, 1.0, '1', '1.0']): 
                    self.append_error( f'Participant has been excluded, but their race was not recorded.',\
                    'chrdemo_racial_back',self.form,['Main Report'])
            if self.row.visit_status_string not in ['removed','screen'] and \
            self.form == 'inclusionexclusion_criteria_review' and\
            self.row.chrcrit_included in [0.0,'0','0.0']\
            and self.row.chrcrit_excluded in[1,1.0,'1','1.0']:
                self.append_error(\
                    f'Participant has been excluded, but moved beyond screening.',\
                    'chrcrit_included',self.form,['Main Report'])
        self.inclusion_checks()


    def inclusion_checks(self):
        """Makes sure participant is marked
        as included/excluded and that they have not 
        collected blood if included"""

        if self.variable == 'chrcrit_included' and (self.prescient == False\
        and self.row.inclusionexclusion_criteria_review_complete in [2,2.0,'2','2.0'])\
        or (self.prescient == 'True' and self.row.visit_status_string not in ['consent', 'screen']):
            if self.row.chrcrit_included == '' and self.row.chrcrit_excluded == '':
                self.append_error('Participant not marked as included or excluded',\
                    self.variable,self.form,['Main Report']) 
        if self.variable == 'chrblood_cbc':
            if getattr(self.row,self.variable) in [1,1.0,'1','1.0'] and \
            self.row.subjectid not in self.variable_info_dictionary['included_subjects']:
                self.append_error(f"Participant has not been included, but blood has been collected.",\
                self.variable,self.form,['Main Report','Blood Report'])

