import pandas as pd
import re
import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

class TransformBranchingLogic():
    def __init__(self, data_dictionary_df):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)

        self.data_dictionary_df = data_dictionary_df

        self.all_vars = self.data_dictionary_df['Variable / Field Name'].tolist()
        self.all_converted_branching_logic = {}
        self.manual_conversions = {"chr_ae1date_dr":
        "float(row.chrae_aescreen)==float(1) and float(chrae_dr1) == float(1)",
        "chreeg_entry_date":"row.chreeg_interview_date !=''"}

    def __call__(self):
        converted_branching_logic = self.convert_all_branching_logic()
        self.find_problematic_conversions(converted_branching_logic)

        return converted_branching_logic
        
    def convert_all_branching_logic(self):
        all_converted_branching_logic = {}
        self.data_dictionary_df = self.data_dictionary_df.rename(
        columns={'Variable / Field Name': 'variable',
        'Branching Logic (Show field only if...)': 'branching_logic','Field Type':'field_type'})

        for row in self.data_dictionary_df.itertuples():
            if row.field_type =='descriptive':
                continue
            var = getattr(row, 'variable')
            branching_logic = getattr(row, 'branching_logic')
            converted_bl = ''
            if branching_logic !='':
                converted_bl = self.branching_logic_redcap_to_python(branching_logic)
            if var in self.manual_conversions.keys():
                converted_bl = self.manual_conversions[var]
            all_converted_branching_logic[var] = {'variable':var,
            'original_branching_logic': branching_logic, 'converted_branching_logic': converted_bl}
        
        return all_converted_branching_logic
            
    def branching_logic_redcap_to_python(self, branching_logic):
        """This function focuses on converting the syntax
        from the REDCap branching logic in the data dictionary
        into Python syntax to be used as conditionals later in the code.

        Parameters
        ----------------
        variable: current variable of interest chrscid_a130
        form: current of interest
        branching logic: redcap version of branching logic 
        """
        modified_branching_logic = str(branching_logic).replace('[', '').replace(
        ']', '').replace('<>', '!=').replace('OR', 'or').replace('AND', 'and').replace("\n", ' ')
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
            (r"(row\.\w+)(?==|>|<|>=|<=)(?! )(?!=='00)(?=.*?\bfloat\()", r"float(\1)"),
            (r"(row\.)(\w+)(!=)(float\(-?\d+|row\.\w+)", r"(\1\2=='' or not hasattr(row,'\2') or float(\1\2)\3\4)"),
            (r"(float\()(row\.)(\w+)(\))(\s*==\s*)(float\(-?\d+\.?\d*|row\.\w+)",
            r"(hasattr(row,'\3') and self.utils.can_be_float(\2\3)==True and \1\2\3\4\5\6)")
            #(r"(row\.\w+)(\s*!=\s*float\(\s*-?\d+\s*\))", r"(\1=='' or float(\1)\2)")  
        ]

        for pattern, replacement_text in patterns_replacements: 
            modified_branching_logic = \
                re.sub(pattern, replacement_text,\
                modified_branching_logic)
            
        return modified_branching_logic
    

    def find_problematic_conversions(self,converted_bl):
        exceptions = []
        count = 0
        
        for var, values in converted_bl.items():
            converted_bl = values['converted_branching_logic']
            if not any(
            x in var for x in ['error','chrsaliva_flag','chrchs_flag','_err','invalid']):
                if ('float' not in converted_bl and converted_bl != ''
                and "!=''" not in converted_bl and 'pharm' not in converted_bl):
                    print(var)
                    print(converted_bl)
                    count+=1
                    print(count)
                else:
                    split_bl = converted_bl.replace(' or ',' and ').split(' and ')
                    for bl_sect in split_bl:
                        if (("float" not in bl_sect) 
                        and bl_sect !='' and "!=''" not in bl_sect
                        and "='00'" not in bl_sect) :
                            exceptions.append({"var":var,"bl":split_bl,"full":converted_bl})
                            break
                    
        exceptions_df = pd.DataFrame(exceptions)
        exceptions_df.to_csv(
        f'{self.config_info["paths"]["output_path"]}branching_logic_qc.csv',
        index = False)

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

    def edit_past_pharm_branch_logic(self, variable):
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
                f" [chrpharm_med{int(number)-1}_add_past] = '1'")
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