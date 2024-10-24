import pandas as pd
import datetime
import os
import openpyxl
from openpyxl.utils import range_boundaries,get_column_letter
from openpyxl.worksheet.dimensions import ColumnDimension
from openpyxl import load_workbook
import copy
class ErrorCompiler():
    """Class to format errors
    into final output as they
    are detected

    DF colums
        1. subject
        2. variable
        3. timepoint of error
        4. subject's current timepoint
        5. inclusion status
        6. withdrawn status
        7. form
        8. error
        9. reports 
        10. nda exclusion


    Function Args
        1. subject
        2. variable
        3. affected forms
        5. affected timepoints
        6. network
        7. error
        8. reports
        9. nda exclusion
        10. withdrawn enabled
        11. excluded enabled
        12. error rewordings
    """

    def __init__(self):
        self.error_output = []

    def append_error(self, error_info):
        self.filter_variables()
        formatted_error = copy.deepcopy(error_info)



    def filter_variables(self, error_info):
        pass

    def format_lists(self, dict_to_format):
        for key in dict_to_format.keys():
            if isinstance(dict_to_format[key], list):
                if len(dict_to_format[key]) > 0:
                    dict_to_format[key] = self.list_to_string(dict_to_format[key])
                else:
                    dict_to_format[key] =''

        return dict_to_format

    def list_to_string(self, inp_list):
        inp_list = [str(item) for item in inp_list]

        inp_list = '|'.join(inp_list)

        return inp_list

            

