class ClinicalChecks():

    def collect_twenty_one_day_rule_dates(self,row):
        """Collectes appropriate baseline
        dates to check the 21 day rule"""

        self.missing_baseline_dates = []
        self.baseline_date_list = []
        self.max_baseline_date_variable_list = []
        baseline_date_variables = []
        for form in self.timepoint_variable_lists['baseln']:
            if 'interview_date' in self.variable_info_dictionary[\
            'unique_form_variables'][form]\
            and self.variable_info_dictionary[\
            'unique_form_variables'][form]['interview_date']\
            not in self.process_variables.excluded_21day_dates:
                baseline_date_variables.append(\
                self.variable_info_dictionary[\
                'unique_form_variables'][form]['interview_date'])
        matching_rows = self.screening_df.loc[self.screening_df['subjectid'] ==\
        self.row.subjectid, 'chrpsychs_scr_interview_date']
        self.psychs_interview_date = matching_rows.iloc[0] if not matching_rows.empty else '' 
        print(self.psychs_interview_date)
        if getattr(self.row,'subjectid') in self.variable_info_dictionary['included_subjects']\
        and self.psychs_interview_date not in (self.missing_code_list+['']):
            self.psychs_interview_date = self.psychs_interview_date.replace('/', '-')
            self.psychs_interview_date = self.convert_date_format(self.psychs_interview_date)
            self.psychs_interview_date = re.search('\d{4}(-|/)\d{1,2}(-|/)\d{1,2}',\
            self.psychs_interview_date.replace('/', '-'))
            self.psychs_interview_date = \
            datetime.datetime.strptime(self.psychs_interview_date.group(), '%Y-%m-%d')
            for x, variable in enumerate(baseline_date_variables):
                extracted_baseline_date = re.search('\d{4}(-|/)\d{1,2}(-|/)\d{1,2}',\
                getattr(self.row, variable).replace('/', '-'))
                if extracted_baseline_date and\
                getattr(self.row, variable) not in self.missing_code_list:
                    self.baseline_date_list.append(datetime.datetime.strptime(\
                    extracted_baseline_date.group(), '%Y-%m-%d'))
                    self.max_baseline_date_variable_list.append(variable)
                else:
                    self.missing_baseline_dates.append(variable)
                    continue
        else:
            return False

        return True

    def twenty_one_day_rule(self,row,timepoint_variable_lists):
        """Check for baseline psychs to see if
        they are properly following the 21 day rule."""
        self.row = row
        self.timepoint_variable_lists = timepoint_variable_lists

        if self.collect_twenty_one_day_rule_dates(row) == True:
            if not self.baseline_date_list:
                return ''  
            date_of_last_baseline_assess = max(self.baseline_date_list)
            time_between = date_of_last_baseline_assess - self.psychs_interview_date
            time_since = datetime.datetime.today() - self.psychs_interview_date
            self.psychs_interview_date =\
            datetime.datetime.strftime(self.psychs_interview_date, "%Y-%m-%d")
            max_baseline_date_index = self.baseline_date_list.index(max(self.baseline_date_list))
            max_baseline_date = max(self.baseline_date_list)
            max_baseline_date_variable = self.max_baseline_date_variable_list[max_baseline_date_index]
            filtered_missing_dates = []
            for form in self.timepoint_variable_lists['baseln']:
                if 'interview_date' in self.variable_info_dictionary['unique_form_variables'][form]:
                    if self.variable_info_dictionary['unique_form_variables'][form]['interview_date']\
                    == max_baseline_date_variable:
                        max_form = form
                    if 'missing_variable' in self.variable_info_dictionary['unique_form_variables'][form]:
                        missing_var = self.variable_info_dictionary[\
                            'unique_form_variables'][form]['missing_variable']
                        for var in self.missing_baseline_dates:
                            if var == self.variable_info_dictionary[\
                            'unique_form_variables'][form]['interview_date'] and\
                            hasattr(self.row,missing_var) and\
                            getattr(self.row,missing_var)\
                            not in [1,1.0,'1','1.0']:
                                filtered_missing_dates.append(var)
            final_baseline_date = (f"{max_form}, "
            f"{datetime.datetime.strftime(max_baseline_date, '%Y-%m-%d')}")
            self.missing_baseline_dates = filtered_missing_dates 
            self.append_twenty_one_day_tracker_row(row,\
            max_form,date_of_last_baseline_assess,time_since)
            if time_between.days > 21:
                return self.append_twenty_one_day_error(time_between,final_baseline_date)

        return ''

    def append_twenty_one_day_tracker_row(self,row,max_form,date_of_last_baseline_assess,time_since):
        """Defines columns of 21 day tracker and
        appends a row.

        Parameters 
        ------------
        max_form: latest baseline assessment 
        date_of_last_baseline_assess: date of latest baseline assess 
        time_since: time between screening psychs and last baseline assess
        """

        if self.row.visit_status_string in ['baseln'] and not any(x in ['2',2,2.0,'2.0'] for x in\
        [self.row.psychs_p1p8_fu_hc_complete,self.row.psychs_p1p8_fu_complete,\
        self.row.psychs_p9ac32_fu_hc_complete,self.row.psychs_p9ac32_fu_complete]) \
        and not any(x in [1,1.0,'1','1.0'] for x in [self.row.chrpsychs_fu_missing_fu,\
        self.row.hcpsychs_fu_missing_fu,self.row.chrpsychs_fu_missing_fu_2,\
        self.row.hcpsychs_fu_missing_fu_2]):
            self.compile_errors.twentyone_day_tracker.append({'subject':self.row.subjectid,\
                    'time_since_screening_psychs':str(time_since.days) +' days',\
                    'recent_baseline_assessment':max_form,\
                    'date_of_recent_baseline_assessment':date_of_last_baseline_assess,\
                    'screening_psychs_date':self.psychs_interview_date,\
                    'Current Timepoint':self.row.visit_status_string,'sent_to_site':'',\
                    'manually_resolved':''})

    def append_twenty_one_day_error(self,time_between,final_baseline_date):
        """If current subject did not follow the 21
        day rule, will append error to final output
        
        Parameters
        --------------
        time_between: time between screening psychs and last baseline assess
        final_baseline_date: most recent baseline assessment
        """

        if len(self.missing_baseline_dates) > 1:
            sing_or_plur_str = (f"(This calculation does not include {self.missing_baseline_dates},"\
            " as the forms are missing dates and not marked as missing).")
        elif len(self.missing_baseline_dates) == 1:
            sing_or_plur_str = (f"(This calculation does not include {self.missing_baseline_dates},"\
            " as the form is missing its date and not marked as missing).")
        else:
            sing_or_plur_str = ''   
        if hasattr(self.row,'chrpsychs_fu_missing_spec_fu') and\
        (getattr(self.row,'chrpsychs_fu_missing_spec_fu')\
        in ['M6'] or getattr(self.row,'hcpsychs_fu_missing_spec_fu')\
        in ['M6'] or getattr(self.row,'chrpsychs_fu_missing_spec_fu_2') in ['M6']\
        or getattr(self.row,'hcpsychs_fu_missing_spec_fu_2') in ['M6']):
            return (f"M6 clicked, but there are {time_between.days} (> 21) days between"\
            f" screening PSYCHS ({self.psychs_interview_date}) and most recently"\
            f" completed baseline visit component ({final_baseline_date}). ") + sing_or_plur_str  

    def call_scid_diagnosis_check(self,variable,row):
        self.variable = variable
        self.row = row
        for checked_variable,conditions in\
        self.process_variables.scid_diagnosis_check_dictionary.items(): 
            if self.variable == checked_variable: 
                self.scid_diagnosis_check(\
                self.form,conditions['diagnosis_variables'],\
                conditions['disorder'],True,conditions['extra_conditionals'])
                self.scid_diagnosis_check(self.form,conditions[\
                'diagnosis_variables'],conditions['disorder'],\
                False,conditions['extra_conditionals'])
        self.scid_additional_checks(self.row,self.variable)

    def scid_additional_checks(self,row,variable):
        if self.prescient:
            report_list = ['Scid Report', 'Main Report']
        else:
            report_list = ['Scid Report']
        if variable == 'chrscid_a48_1':
            if row.chrscid_a27 in [3,3.0,'3','3.0'] and row.chrscid_a28 in [3,3.0,'3','3.0'] and\
               (row.chrscid_a48_1 not in [2,2.0,'2','2.0']): 
                self.compile_errors.append_error(self.row,\
                f'Fulfills both main criteria but was counted incorrectly, a27, a28, a48_1',\
                self.variable,self.form,report_list)
            elif ((row.chrscid_a27 in [3,3.0,'3','3.0'] and (row.chrscid_a28 in\
            [2,2.0,'2','2.0'] or row.chrscid_a28 in [1,1.0,'1','1.0'])) or\
                 (row.chrscid_a28 in [3,3.0,'3','3.0'] and (row.chrscid_a27\
                in [2,2.0,'2','2.0'] or row.chrscid_a27 in [1,1.0,'1','1.0']))) and\
                 (row.chrscid_a48_1 not in [1,1.0,'1','1.0']):
                self.compile_errors.append_error(self.row,\
                'Fulfills main criteria but further value was wrong, a27, a28, a48_1',\
                    self.variable,self.form,report_list)
        elif variable == 'chrscid_a51' and \
        row.chrscid_a26_53 not in (self.missing_code_list+['']):
            if float(row.chrscid_a26_53) <1 and (row.chrscid_a25\
            in [3,3.0,'3','3.0'] or row.chrscid_a51 in [3,3.0,'3','3.0']):
                self.compile_errors.append_error(self.row,('has no indication of total mde episodes'
                ' fulfilled in life even though fulfills current major depression. a26_53, a51'),\
                    self.variable,self.form,report_list)
            if float(row.chrscid_a26_53) > 0 and (row.chrscid_a25 not in [3,3.0,'3','3.0']\
            and row.chrscid_a51 not in [3,3.0,'3','3.0']):
                self.compile_errors.append_error(self.row,('fulfills more manic episodes than 0 but there'
                ' is no indication of past or current depressive episode. a26_53, a25'),\
                    self.variable,self.form,report_list)
        elif variable == 'chrscid_a1':
            if row.chrscid_a1 in [3,3.0,'3','3.0'] and\
            row.chrscid_a2 in [3,3.0,'3','3.0'] and\
            (row.chrscid_a22_1 not in [2,2.0,'2','2.0']):
                self.compile_errors.append_error(self.row,(f"Fulfills both main criteria"\
                " but was counted incorrectly, check a1, a2, a22_1"),\
                self.variable,self.form,report_list)
            if ((row.chrscid_a1 in [3,3.0,'3','3.0'] and (row.chrscid_a2 in [2,2.0,'2','2.0']\
            or row.chrscid_a2 in [1,1.0,'1','1.0'])) or\
                 (row.chrscid_a2 in  [3,3.0,'3','3.0'] and (row.chrscid_a1 in [2,2.0,'2','2.0']\
                 or row.chrscid_a1 in [1,1.0,'1','1.0']))) and\
                 (row.chrscid_a22_1 not in [1,1.0,'1','1.0']):
                self.compile_errors.append_error(self.row,\
                'Fulfills main criteria but further value was wrong, a1, a2, a22_1',\
                self.variable,self.form,report_list)
        elif variable == 'chrscid_a25' and row.chrscid_a22 not\
        in (self.missing_code_list+['']) and row.chrscid_a22_1 not in (self.missing_code_list+['']):
            if float(row.chrscid_a22) > 4 and float(row.chrscid_a22_1) > 0  and (row.chrscid_a25\
            == '' or row.chrscid_a25 in self.missing_code_list):
                self.compile_errors.append_error(self.row,\
                ('A. MOOD EPISODES: subject fulfills more than 4'
                ' criteria of depression but further questions are not asked: start checking a22, a22_1, a25'),\
                    self.variable,self.form,report_list)


    def scid_diagnosis_check(self,form,conditional_variables,
                             disorder,fulfilled,extra_conditionals):
        if self.prescient:
            report_list = ['Scid Report','Main Report']
        else:
            report_list = ['Scid Report']
        try:
            row = self.row
            if fulfilled == True:
                for condition in conditional_variables:
                    if hasattr(self.row,condition) and \
                    getattr(self.row,condition) not in [3,3.0,'3','3.0']:
                        return ''
                if extra_conditionals != '':
                    for conditional in extra_conditionals:
                        if not eval(conditional):
                            return ''
                if getattr(self.row,self.variable) not in [3,3.0,'3','3.0']:
                    self.compile_errors.append_error(self.row,\
                    f'{disorder} criteria are fulfilled, but it is not indicated.',\
                    self.variable,form,report_list)
            else:
                for condition in conditional_variables:
                    if hasattr(self.row,condition) and \
                    getattr(self.row,condition) not in [3,3.0,'3','3.0']:     
                        if hasattr(self.row,self.variable) and\
                        getattr(self.row,self.variable) in [3,3.0,'3','3.0']:
                            self.compile_errors.append_error(self.row,\
                            f'{disorder} criteria are NOT fulfilled, but it is indicated.',\
                            self.variable,form,report_list)
                if extra_conditionals != '':
                    for conditional in extra_conditionals:
                        if not eval(conditional):
                            if hasattr(self.row,self.variable) and\
                            getattr(self.row,self.variable) in [3,3.0,'3','3.0']:
                                if self.variable == 'chrscid_a25':
                                    print(getattr(self.row,condition))
                                    print(condition)
                                    print(getattr(self.row,self.variable))
                                    print(disorder)
                                self.compile_errors.append_error(self.row,\
                                f'{disorder} criteria are NOT fulfilled, but it is indicated.',\
                                self.variable,form,report_list)
        except Exception as e:
            print(e)

    def cssr_additional_checks(self):
        """Checks for contradictions in cssr form"""

        lifetime_pastyear_dict = {'chrcssrsb_cssrs_actual':'chrcssrsb_sb1l',\
        'chrcssrsb_cssrs_nssi':'chrcssrsb_sbnssibl','chrcssrsb_cssrs_yrs_ia':'chrcssrsb_sb3l',\
        'chrcssrsb_cssrs_yrs_aa':'chrcssrsb_sb4l',\
        'chrcssrsb_cssrs_yrs_pab':'chrcssrsb_sb5l','chrcssrsb_cssrs_yrs_sb':'chrcssrsb_sb6l'}
        lifetime_pastyear_greater_dict = {'chrcssrsb_idintsvl':'chrcssrsb_css_sipmms',\
        'chrcssrsb_snmacatl':'chrcssrsb_cssrs_num_attempt',\
        'chrcssrsb_nminatl':'chrcssrsb_cssrs_yrs_nia','chrcssrsb_nmabatl':'chrcssrsb_cssrs_yrs_naa'}
        for x in range(1,6):
            if self.variable == 'chrcssrsb_si{x}l':
                if getattr(self.row, f'chrcssrsb_si{x}l') in [2,2.0,'2','2.0']\
                and getattr(self.row, f'chrcssrsb_css_sim{x}') in [1,1.0,'1','1.0']:
                    self.compile_errors.append_error(self.row,f'Past month does not equal lifetime.',\
                    self.variable,self.form,['Main Report', 'Non Team Forms'])
        for key_past_months, value_lifetime in lifetime_pastyear_greater_dict.items():
            if self.variable == key_past_months:
                try:
                    if getattr(self.row,key_past_months) not in\
                    self.missing_code_list and \
                    getattr(self.row,value_lifetime) not in self.missing_code_list\
                    and float(getattr(self.row,key_past_months))\
                    < float(getattr(self.row,value_lifetime)):
                        self.compile_errors.append_error(self.row,\
                        (f'Lifetime ({getattr(self.row,key_past_months)}) cannot',\
                        f'be lower than past month ({getattr(self.row,value_lifetime)}).'),\
                        self.variable,self.form,['Main Report', 'Non Team Forms'])
                except Exception as e:
                    print(e)
        for key_past_months,value_lifetime in lifetime_pastyear_dict.items():
            if self.variable == key_past_months:
                if getattr(self.row,key_past_months) in [1,1.0,'1','1.0']\
                and getattr(self.row,value_lifetime) in [2,2.0,'2','2.0']:
                    self.compile_errors.append_error(self.row,\
                    f'Lifetime value cannot be different that past month value.',\
                    self.variable,self.form,['Main Report','Non Team Forms'])

    def inclusion_psychs_check(self):
        """Checks for contradictions between
        inclusion/exclusion form and psychs"""

        if self.variable == 'chrcrit_part':
            if (self.row.chrcrit_part in [1,1.0,'1','1.0']\
            and self.row.chrcrit_inc3 in [0,0.0,'0','0.0']):
                self.compile_errors.append_error(self.row,\
                f'CHR subject does not fulfill CHR-criteria on PSYCHS.',\
                self.variable,self.form,['Main Report','Non Team Forms'])
            elif (self.row.chrcrit_part in [2,2.0,'2','2.0']\
            and self.row.chrcrit_inc3 in [1,1.0,'1','1.0']):
                self.compile_errors.append_error(self.row,\
                f'HC subject fulfills CHR-criteria on PSYCHS.',\
                self.variable,self.form,['Main Report','Non Team Forms'])

    def oasis_additional_checks(self):
        """Checks for contradictions in OASIS form"""

        try:
            if self.variable == f'chroasis_oasis_1':
                for x in range(2,6):
                    if getattr(self.row,f'chroasis_oasis_{x}') not in self.missing_code_list\
                    and float(getattr(self.row,f'chroasis_oasis_{x}')) > 0 and\
                    self.row.chroasis_oasis_1 in [0,0.0,'0','0.0']:
                        self.compile_errors.append_error(self.row,\
                        (f'No anxiety at all (last week) cannot have anxiety '\
                        f'level or be influenced by anxiety (last week) (chroasis_oasis_{x}).'),\
                        self.variable,self.form,['Main Report','Non Team Forms'])
            if self.variable == 'chroasis_oasis_3':
                if self.row.chroasis_oasis_3 not in\
                self.missing_code_list and self.row.chroasis_oasis_4 not in self.missing_code_list\
                and self.row.chroasis_oasis_5 not in self.missing_code_list and \
                float(self.row.chroasis_oasis_3) <2 and \
                (float(self.row.chroasis_oasis_4 > 1) or float(self.row.chroasis_oasis_5)>0):
                    self.compile_errors.append_error(self.row,\
                    (f'If lifestyle is not affected (chroasis_oasis_3<2) no'\
                    f'lifestyle situation can be described to be affected (chroasis_oasis_4 and chroasis_oasis_5)'),\
                    self.variable,self.form,['Main Report', 'Non Team Forms'])
        except Exception as e:
            print(e)
