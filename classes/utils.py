import pandas as pd

class Utils():
    def __init__(self):
        self.missing_code_list =  ['-3','-9',-3,-9,-3.0,-9.0,'-3.0','-9.0',\
        '1909-09-09','1903-03-03','1901-01-01','-99',-99,-99.0,\
        '-99.0',999,999.0,'999','999.0'] 

    def create_timepoint_list(self):
        """Organizes every timepoint as list
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
    
    def check_if_number(self,input):
        try:
            float(input)  
            return True
        except ValueError:
            return False

