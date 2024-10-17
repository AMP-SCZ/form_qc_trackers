class FluidChecks():

    def blood_vol_check(self):
        if self.variable in self.all_blood_volume_variables and hasattr(self.row,self.variable):
            vol = getattr(self.row,self.variable)
            if vol not in (self.missing_code_list+['']) and\
            float(vol) <= 0:
                self.compile_errors.append_error(self.row,\
                f'Volume ({self.variable} = {vol}) is less than or equal to 0',\
                self.variable,self.form,['Main Report','Blood Report','Fluids Report'])

    def check_blood_freeze(self):
        if self.variable == 'chrblood_wbfrztime':
            for proc_time_var in ['chrblood_wholeblood_freeze',\
            'chrblood_serum_freeze','chrblood_plasma_freeze','chrblood_buffy_freeze']:
                proc_time =getattr(self.row, proc_time_var)
                if proc_time not in (self.missing_code_list+['']) and float(proc_time) > 480:
                    self.compile_errors.append_error(self.row,\
                    f'Processing time ({self.variable} = {proc_time}) is greater than 480',\
                    self.variable,self.form,['Main Report','Blood Report','Fluids Report'])

    def check_blood_duplicates(self):
        """Checks for duplicate blood positions 
        and IDs"""

        filtered_columns = [col for col in self.ampscz_df.columns if 'blood' in col]
        filtered_columns.append('subjectid')
        self.filtered_blood_df = self.ampscz_df[filtered_columns]
        if self.variable =='chrblood_rack_barcode' and hasattr(self.row,'chrblood_rack_barcode'):
            if self.prescient:
                report_list = []
            else:
                report_list = ['Main Report','Blood Report','Fluids Report']

            blood_df = self.filtered_blood_df[self.filtered_blood_df[\
            'chrblood_rack_barcode']==getattr(self.row,'chrblood_rack_barcode')]
            for blood_pos_var in self.all_blood_position_variables:
                blood_df = blood_df[blood_df[blood_pos_var]==getattr(self.row,blood_pos_var)]
                for blood_row in blood_df.itertuples():
                    if blood_row.subjectid != self.row.subjectid:
                        if getattr(blood_row,blood_pos_var) == getattr(self.row,blood_pos_var)\
                        and getattr(blood_row,blood_pos_var) not in (self.missing_code_list+[''])\
                        and getattr(blood_row,'chrblood_rack_barcode') not in (self.missing_code_list+['']):
                            self.compile_errors.append_error(self.row,(\
                            f"Duplicate positions found in two different subjects ({self.row.subjectid} and {blood_row.subjectid} "\
                            f"both have {blood_pos_var} equal to {getattr(blood_row,blood_pos_var)}"\
                            f" and barcode equal to {self.row.chrblood_rack_barcode})."),\
                            blood_pos_var,self.form,report_list)
        if self.variable in self.all_blood_id_variables:
            if self.prescient:
                report_list = []
            else:
                report_list = ['Main Report','Blood Report','Fluids Report']
            blood_df = self.filtered_blood_df[\
            self.filtered_blood_df[self.variable]==getattr(self.row,self.variable)]
            for blood_row in blood_df.itertuples():
                if blood_row.subjectid != self.row.subjectid and\
                getattr(self.row,self.variable) not in (self.missing_code_list + ['']):
                    self.compile_errors.append_error(self.row,\
                    (f"Duplicate blood ID ({self.variable} = {getattr(self.row,self.variable)})",\
                    f" found between subjects {self.row.subjectid} and {blood_row.subjectid}."),\
                    self.variable,self.form,report_list)
        elif self.variable in self.all_blood_volume_variables:
            if hasattr(self.row,self.variable) and\
            getattr(self.row,self.variable) not in (self.missing_code_list + [''])\
            and float(getattr(self.row,self.variable)) > 1.1:
                self.compile_errors.append_error(self.row,\
                f"Blood volume ({getattr(self.row,self.variable)}) is greater than 1.1.",\
                self.variable,self.form,['Blood Report','Fluids Report'])

    def check_blood_dates(self):
        """Function to check if the blood
        draw date is later than the date
        sent to lab."""

        if self.variable == 'chrblood_drawdate':
            if self.prescient:
                report_list = []
            else:
                report_list = ['Blood Report','Fluids Report']

            if any(date in (self.missing_code_list +['']) for\
            date in [self.row.chrblood_drawdate,self.row.chrblood_labdate]):
                return ''
            if datetime.datetime.strptime(\
            self.row.chrblood_drawdate,"%Y-%m-%d %H:%M")\
            > datetime.datetime.strptime(\
            self.row.chrblood_labdate,"%Y-%m-%d %H:%M"):
                self.compile_errors.append_error(self.row,\
                f"Blood draw date ({self.row.chrblood_drawdate}) is later than date sent to lab ({self.row.chrblood_labdate}).",\
                self.variable,self.form,report_list)

    def barcode_format_check(self):
        """Makes sure blood barcodes are in
        proper format"""

        if self.variable in self.all_barcode_variables\
        and 'blood' in self.variable:
            if hasattr(self.row,self.variable):
                barcode = getattr(self.row,self.variable)
                if barcode not in (self.missing_code_list + [''])\
                and 'pronet' not in barcode.lower(): 
                    if len(barcode) < 10:
                        self.compile_errors.append_error(self.row,\
                        f"Barcode ({barcode}) length is less than 10 characters.",\
                        self.variable,self.form,['Blood Report','Fluids Report'])
                    if any(not char.isdigit() for char in barcode):
                        self.compile_errors.append_error(self.row,\
                        f"Barcode ({barcode}) contains non-numeric characters.",\
                        self.variable,self.form,['Blood Report','Fluids Report'])

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
