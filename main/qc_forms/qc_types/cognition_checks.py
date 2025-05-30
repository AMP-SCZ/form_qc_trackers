import datetime

class CognitionChecks():
    """
    QC Checks related to cognitive forms
    """
    def find_iq_age(self):
        """
        Finds age at the time of the iq
        assessment
        """

        try:
            demographics_date = datetime.datetime.strptime(
            self.row.chrdemo_interview_date, "%Y-%m-%d") 
            iq_date = datetime.datetime.strptime(
            self.row.chriq_interview_date, "%Y-%m-%d")  
            days_between = (iq_date - demographics_date).days
            months_between = int(days_between/30)
            return months_between
        except Exception as e:
            print(e)
            return 0

    def find_days_between(self,d1,d2):
        """
        finds the days between two dates
        
        Parameters
        -----------------
        d1: first date
        d2: second date
        """

        date_format = "%Y-%m-%d"
        date1 = datetime.datetime.strptime(
        d1.split(' ')[0], date_format)
        date2 = datetime.datetime.strptime(
        d2.split(' ')[0], date_format)
        date_difference = date2 - date1
        days_between = date_difference.days

        return abs(days_between)

    def collect_age(self):
        """Collects current age of 
        subject by checking each possible
        age variable"""

        age = ''
        for age_var in ['chrdemo_age_mos_chr',
        'chrdemo_age_mos_hc','chrdemo_age_mos2']:
            if (hasattr(self.row,age_var)
            and getattr(self.row,age_var)
            not in (self.missing_code_list+[''])):
                age = int(getattr(self.row,age_var))
                age = age - self.find_iq_age()
                break
        return age

    def age_range_workaround(self,age):
        """
        Workaround for ages that fall on the 
        boundaries of a range and may fall 
        in one or another depending on how
        it is rounded

        Parameters
        --------------
        age: age being checked
        """

        current_age_ranges = []
        iq_age_ranges = self.process_variables.all_iq_age_ranges
        for age_range in iq_age_ranges:
            if (age == age_range[-1] or
            age == age_range[-2]) and age < 360:
                possibly_next_range = True
                current_age_ranges.append(age_range)
                current_age_ranges.append(iq_age_ranges[
                iq_age_ranges.index(age_range) + 1])
                flag_error = False
            elif (age == age_range[0]
            or age == age_range[1]) and age > 195:
                possibly_previous_range = True
                current_age_ranges.append(age_range)
                current_age_ranges.append(iq_age_ranges[
                iq_age_ranges.index(age_range) - 1])
                flag_error = False
            elif age in age_range:
                current_age_ranges.append(age_range)
                flag_error = True
        return current_age_ranges, flag_error

    def fsiq_check(self, iq_type = 'wasi'):
        """
        Checks if FSIQ conversion was done
        properly
        """

        if self.variable == 'chriq_fsiq':
            for fsiq_row in self.process_variables.fsiq_conversions_per_test[iq_type].itertuples():
                if str(fsiq_row.t_score).replace(' ','')
                == str(self.row.chriq_tscore_sum).replace(' ',''):
                    if str(fsiq_row.fsiq).replace(' ','')
                    != str(self.row.chriq_fsiq).replace(' ',''):
                        self.compile_errors.append_error(self.row,
                        (f'FSIQ-2 Conversion not done properly.'
                        f'Entered value was {self.row.chriq_fsiq}, but should be {fsiq_row.fsiq}'),
                        self.variable,self.form,['Main Report','Cognition Report'])

    def loop_iq_table(self,age, conversion_col, t_score_col, iq_variable, iq_type = 'wasi'):
        """
        Loops through IQ table to make sure
        conversions were done properly

        Parameters
        -----------
        age: age of current subject
        conversion_col: current column being converted
        t_score_col: column of the T-scores 
        iq_variable: variable associated with conversion
        """

        current_age_ranges,flag_error = self.age_range_workaround(age)
        conversion_df = self.process_variables.iq_raw_conversions_per_test[iq_type] 
        error_message = ''
        incorrect_range_count = 0
        for range_index in range(0,len(current_age_ranges)):
            any_match = False
            columns_to_keep = conversion_df.columns[
            conversion_df.iloc[0].apply(
            lambda x: current_age_ranges[range_index] == x)]
            iq_df = conversion_df[columns_to_keep].copy()
            iq_df_filtered = iq_df.iloc[1:].copy().reset_index(drop=True)
            iq_df_filtered.columns = iq_df_filtered.iloc[0]
            for iq_row in iq_df_filtered.itertuples():
                if str(getattr(self.row,iq_variable)).replace(' ','') in
                self.convert_range_to_list(str(getattr(iq_row,conversion_col)),True):
                    if str(iq_row.t_score).replace(' ','')
                    != str(getattr(self.row, t_score_col)).replace(' ','') and 
                    str(getattr(self.row,iq_variable)).replace(' ','') not in (self.missing_code_list + ['']):
                        print(getattr(self.row, t_score_col))
                        if len(current_age_ranges) == 2 and
                        range_index == 1 and not any_match:
                            flag_error = True
                        if current_age_ranges[range_index]
                        == current_age_ranges[-1] and flag_error == True:
                            if len(current_age_ranges) == 1:
                                error_message = (f'T-Score Conversion not done properly.'
                                f'Entered value was {getattr(self.row,t_score_col)}, but should be {iq_row.t_score}')
                            else:
                                print('IQ------------------')
                                print(current_age_ranges)
                                print(self.row.subjectid)
                                print(conversion_df)
                                print(current_age_ranges[range_index])
                                print(self.convert_range_to_list(str(getattr(iq_row,conversion_col)),True))
                                incorrect_range_count +=1
                                if incorrect_range_count == 2:

                                    error_message = (f'T-Score Conversion not done properly.'
                                    f'Entered value was {getattr(self.row,t_score_col)}' 
                                    f'(cannot calculate proper value that it should be due to insufficient age rounding).')

                            if error_message != '':
                                self.compile_errors.append_error(
                                self.row,error_message,self.variable,
                                self.form,['Main Report','Cognition Report'])
                    else:
                        any_match = True

    def iq_conversion_check(self):
        """
        Loops through different IQ
        variables being converted and 
        calls functions to check them
        """

        conversion_col_names = {'wasi':{'vocab_raw':'vc','matrix_raw':'mr',
        'vocab_conversion':'chriq_tscore_vocab','matrix_conversion':'chriq_tscore_matrix'},
        'wais':{'vocab_raw':'vc','matrix_raw':'mr',
        'vocab_conversion':'chriq_tscore_vocab','matrix_conversion':
        'chriq_tscore_matrix'}, 'wisc': {}}

        iq_type_translations = {1:'wasi'}
        if not hasattr(self.row,'chriq_assessment') or self.row.chriq_assessment == '':
            return ''
        try:
            iq_type_var_val = int(self.row.chriq_assessment)
        except Exception as e:
            print(e)
            return ''
        if iq_type_var_val in iq_type_translations.keys():
        
            iq_type = iq_type_translations[iq_type_var_val]
        else:
            return ''

        for iq_variable in ['chriq_vocab_raw','chriq_matrix_raw']:
            possibly_next_range = False
            possibly_previous_range = False
            if self.variable == iq_variable and 
            self.row.chriq_assessment in [1,1.0,'1','1.0']:
                if iq_variable == 'chriq_vocab_raw':
                    conversion_col = conversion_col_names[iq_type]['vocab_raw']
                    t_score_col = conversion_col_names[iq_type]['vocab_conversion']
                else:
                    conversion_col = conversion_col_names[iq_type]['matrix_raw']
                    t_score_col = conversion_col_names[iq_type]['matrix_conversion']
                age = self.collect_age()
                if age !='' and age>191 and self.row.chriq_tscore_vocab != ''
                and self.row.chriq_vocab_raw not in self.missing_code_list:
                    self.loop_iq_table(age,conversion_col, t_score_col, iq_variable, iq_type)
        self.fsiq_check()

    def convert_range_to_list(self,range_str,str_conv = False):
        """Converts string with number range to a list of 
        the numbers included in that range. Used for IQ age checks.

        Parameters
        -------------
        range_str: string of number range
        str_conv:
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
