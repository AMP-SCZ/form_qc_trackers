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
        
        # variables with branching logic
        # that can't or is not currently
        # being converted properly
        self.excluded_conversions = {}

        self.data_dictionary_df = data_dictionary_df


        self.all_vars = self.data_dictionary_df['Variable / Field Name'].tolist()
        self.all_converted_branching_logic = {}
        self.manual_conversions = {"chr_ae1date_dr":
        ("hasattr(curr_row, 'chrae_aescreen') and"
        " instance.utils.can_be_float(curr_row.chrae_aescreen)==True"
        " and float(curr_row.chrae_aescreen)==float(1) and"
        " hasattr(curr_row, 'chrae_dr1') and"
        " instance.utils.can_be_float(curr_row.chrae_dr1)==True"
        " and float(curr_row.chrae_dr1)==float(1)"),
        "chreeg_entry_date":"hasattr(curr_row,'chreeg_interview_date') and curr_row.chreeg_interview_date !=''",
        
        "chrpsychs_scr_e24":("(hasattr(curr_row,'chrpsychs_scr_ac1') and curr_row.chrpsychs_scr_ac1!=''" 
        " and instance.utils.can_be_float(curr_row.chrpsychs_scr_ac1)==True"
        " and float(curr_row.chrpsychs_scr_ac1)==float(0)) and (hasattr(curr_row,'chrpsychs_scr_e4')"
        " and instance.utils.can_be_float(curr_row.chrpsychs_scr_e4)==True and float(curr_row.chrpsychs_scr_e4)==float(1))"
        " and (hasattr(curr_row,'chrpsychs_scr_e21') and curr_row.chrpsychs_scr_e21!=''"
        " and instance.utils.can_be_float(curr_row.chrpsychs_scr_e21)==True"
        " and float(curr_row.chrpsychs_scr_e21)==float(0)) and (hasattr(curr_row,'chrsofas_currscore')"
        " and instance.utils.can_be_float(curr_row.chrsofas_currscore)==True"
        " and hasattr(curr_row,'chrsofas_premorbid') and instance.utils.can_be_float(curr_row.chrsofas_premorbid)==True)"
        " and ((float(curr_row.chrsofas_currscore)/float(curr_row.chrsofas_premorbid))"
        " >=float(0.9))"),
        
        "chriq_pic_completion_raw" :("hasattr(curr_row,'chriq_assessment')"
        " and instance.utils.can_be_float(curr_row.chriq_assessment)==True and"
        " (float(curr_row.chriq_assessment)==float(4) or float(curr_row.chriq_assessment)==float(5))"),

        "chriq_scaled_pic_completion" :("hasattr(curr_row,'chriq_assessment')"
        " and instance.utils.can_be_float(curr_row.chriq_assessment)==True and"
        " (float(curr_row.chriq_assessment)==float(4) or float(curr_row.chriq_assessment)==float(5))"),

        "chrdig_notes_5" : ("hasattr(curr_row,chrdig_reason_missing)"
        " and instance.utils.can_be_float(curr_row.chrdig_reason_missing) and float(curr_row.chrdig_reason_missing) == float(3)"
        " and hasattr(curr_row,'chrdig_motivational')"
        " and curr_row.chrdig_motivational != ''"),

        "chreeg_entry_date" : "hasattr(curr_row,'chreeg_interview_date') and curr_row.chreeg_interview_date != ''"
        }

    def __call__(self):
        converted_branching_logic = self.convert_all_branching_logic()
        self.find_problematic_conversions(converted_branching_logic)
        self.find_pattern_exceptions()
        self.exclude_identifiers()

        self.utils.save_dependency_json(self.excluded_conversions,
         'excluded_branching_logic_vars.json')

        return converted_branching_logic
        
    def convert_all_branching_logic(self):
        all_converted_branching_logic = {}
        self.data_dictionary_df = self.data_dictionary_df.rename(
        columns={'Variable / Field Name': 'variable',
        'Branching Logic (Show field only if...)': 'branching_logic','Field Type':'field_type'})

        for row in self.data_dictionary_df.itertuples():
            var = getattr(row, 'variable')
            branching_logic = getattr(row, 'branching_logic')
            branching_logic = self.edit_tbi_branch_logic(var, branching_logic)
            branching_logic = self.edit_past_pharm_branch_logic(var, branching_logic)
            branching_logic = self.edit_av_branch_logic(var, branching_logic)
            branching_logic = self.edit_scid_bl(var, branching_logic)
            converted_bl = ''
            if branching_logic !='':
                converted_bl = self.branching_logic_redcap_to_python(branching_logic)
            if var in self.manual_conversions.keys():
                converted_bl = self.manual_conversions[var]
            all_converted_branching_logic[var] = {'variable':var,
            'original_branching_logic': branching_logic, 'converted_branching_logic': converted_bl}
        
        return all_converted_branching_logic
    

    def find_pattern_exceptions(self):
        all_patterns = [ (r"(\()*(\s*)(and|or)?(\s*)(\(*)(\[\w+(\(\d+\))?\])(\s*)"
        r"(<>|=|>=|<=|>|<)(\s*)('00'|''|\[\w+(\(\d+\))?\]|"
        r"(\')?-?\d+(\.\d+)?(\')?)(\s*)(\)*)(and|or)?(\s*)(\))*")
        ]
        data_dictionary_df = self.data_dictionary_df.rename(
        columns={'Variable / Field Name': 'variable',
        'Branching Logic (Show field only if...)': 'branching_logic','Field Type':'field_type'})

        for row in data_dictionary_df.itertuples():
            if row.field_type =='descriptive':
                continue
            var = getattr(row, 'variable')
            
            branching_logic = getattr(row, 'branching_logic')
            branching_logic = self.edit_tbi_branch_logic(var, branching_logic)
            branching_logic = self.edit_past_pharm_branch_logic(var, branching_logic)
            branching_logic = self.edit_av_branch_logic(var, branching_logic)
            converted_bl = ''
            modified_bl = str(branching_logic).replace('OR', 'or').replace('AND', 'and').replace("\n", ' ').replace('"',"'")
            if modified_bl !='':
                for pattern in all_patterns:
                    modified_bl = re.sub(
                    pattern, '',
                    modified_bl)
            if modified_bl != '':
                if var not in self.manual_conversions.keys():
                    self.excluded_conversions[var] = branching_logic

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
        ']', '').replace('<>', '!=').replace('OR', 'or').replace(
        'AND', 'and').replace("\n", ' ').replace('"',"'")

        #NOTE: functions must remain in this order in the list
        for pattern_replacements in [self.format_floats_and_vars(),
        self.format_var_comparisons(), self.refine_float_comparisons(),
        self.format_double_zeroes()]:
            for pattern, replacement_text in pattern_replacements: 
                modified_branching_logic = re.sub(
                    pattern, replacement_text,
                    modified_branching_logic)
                
        return modified_branching_logic
    

    def exclude_identifiers(self):
        depend_path = self.config_info['paths']['dependencies_path']
        ident_df = pd.read_csv(f'{depend_path}identifier_effects.csv')
        for row in ident_df.itertuples():
            if row.affected_col == 'branching_logic':
                self.excluded_conversions[row.var] = row.affected_col_val
    
    def format_floats_and_vars(self):
        pattern_replacements = [
            # Replaces single equals sign "=" with double equals sign "==" 
            (r"(?<!=)(?<![<>!])=(?!=)", r"=="),  
            # Converts numbers (that do not equal '00') prececeded by comparison operators to floats.
            (r"([=<>]\s*)(\'?(?!00)-?\d+(\.\d+)?\'?)", r"\1float(\2)"), 
            # Adds "curr_row." to the beginning of variable names or function calls followed by a comparison operator (!=, =, <, >) 
            (r"([A-Za-z][A-Za-z0-9_]*)(\(\d+\))?(\s*)(!=|=|<|>)", r"curr_row.\1\2\3\4"), 
            # Adds "curr_row." to the beginning of variable names or function calls preceded by a comparison operator (!=, =, <, >) 
            (r"(!=|=|<|>)(\s*)((?!float\()[A-Za-z][A-Za-z0-9_]*)(\(\d+\))?", r"\1\2curr_row.\3\4"), 
            # Replaces numbers in parentheses with "___" appended to the beginning (for checkbox variables) 
            (r"(?<!float)\((\d+)\)", r"___\1"),  
            # Adds the "float()" function to variable names starting with "curr_row." 
            # if it is followed by a comparison operator and a float number
            (r"(curr_row\.\w+\_*)(\s*)(==|>|<|>=|<=)(\s*)(float\()", r"float(\1)\2\3\4\5"),
            # if the dataframe does not have the variable, then it is considered blank
            # otherwise, checks if it is blank or if it can be a float that is not equal
            # to the value of interest
            (r'(curr_row\.)(\w+)(\s*!=\s*)(float\((\'?(?!00)-?\d+(\.\d+)?\'?)\))',
            r"((not hasattr(curr_row,'\2') or \1\2=='') or instance.utils.can_be_float(\1\2)==False or float(\1\2)\3\4)")
            ]
        
        return pattern_replacements
    
    def format_var_comparisons(self):
        pattern_replacements = [
            # if it is checking if two variables are equal, then it checks
            # if they are both blank, both equal floats, or both equal strings
            (r'(curr_row\.)(\w+)(\s*==\s*)(curr_row\.)(\w+)',
            (r"((hasattr(curr_row,'\5') and hasattr(curr_row,'\2')) and"
            r" ((instance.utils.can_be_float(\1\2)==True and"
            r" instance.utils.can_be_float(\4\5)==True and float(\1\2) \3 float(\4\5)) or (str(\1\2) \3 str(\4\5) )) )"
            r" or (not hasattr(curr_row,'\5') and not hasattr(curr_row,'\2'))")),
            # if it is checking if two variables are greater than
            # or less than each other, checks if they are both
            # floats then compares them
            (r'(curr_row\.)(\w+\s*)(>|<|>=|<=)(\s*curr_row\.)(\w+)',
            (r"(hasattr(curr_row,'\5') and hasattr(curr_row,'\2')) and"
            r" (instance.utils.can_be_float(\1\2)==True and"
            r" instance.utils.can_be_float(\4\5)==True and float(\1\2) \3 float(\4\5))")),
            # when checking if two variables are not equal, will check if 
            # one exists and is not blank while the other one exists, or
            # if they are both floats that are not equal 
            # or if they can't be floats and their strings are not equal
            (r'(curr_row\.)(\w+)(\s*!=\s*)(curr_row\.)(\w+)',
            (r"((not hasattr(curr_row,'\5') and hasattr(curr_row,'\2') and \1\2!='') or"
            r" (not hasattr(curr_row,'\2') and hasattr(curr_row,'\5') and \4\5!='') or "
            r" (instance.utils.can_be_float(\1\2)==True and"
            r" instance.utils.can_be_float(\4\5)==True and float(\1\2) \3 float(\4\5)) or"
            r" (instance.utils.can_be_float(\1\2)==False and"
            r" instance.utils.can_be_float(\4\5)==False and str(\1\2) \3 str(\4\5)))"))
        ]

        return pattern_replacements
    
    def refine_float_comparisons(self):
        pattern_replacements =[
            # if it is comparing a variable to a float (other than a negative comparison), checks if 
            # the variable exists and that it can be a float
            (r"(float\()(curr_row\.)(\w+)(\))(\s*)(==|>=|<=|<|>)(\s*)(float\(\'?-?\d+\.?\d*\'?)",
            r"(hasattr(curr_row,'\3') and instance.utils.can_be_float(\2\3)==True and \1\2\3\4\5\6\7\8)"),
            # if it is checking if a variable does not equal a float, 
            # it can either not exist in the dataframe or be a float that 
            # does not equal the value of interest
            (r"(float\()(curr_row\.)(\w+)(\))(\s*)(!=)(\s*)(float\(\'?-?\d+\.?\d*\'?)",
            r"(not hasattr(curr_row,'\3') or (instance.utils.can_be_float(\2\3)==True and \1\2\3\4\5\6\7\8))"),
            # if it is checking if a variable is not blank
            # it needs to exist in the dataframe
            (r"(curr_row\.)(\w+)(\s*!=\s*)('')",
            r"(hasattr(curr_row,'\2') and \1\2\3\4)"),
            # if it is checking if a variable is blank
            # it can either not exist in the dataframe or be blank
            (r"(curr_row\.)(\w+)(\s*==\s*)('')",
            r"(not hasattr(curr_row,'\2') or \1\2\3\4)")
        ]

        return pattern_replacements
    
    def format_double_zeroes(self):
        pattern_replacements = [
            # if it is checking if a variable it not "00"
            # it can either not exist in the dataframe or exist and not equal "00"
            (r"(curr_row\.)(\w+)(\s*!=\s*)('00')",
            r"(not hasattr(curr_row,'\2') or \1\2\3\4)"),
            # if it is checking if a variable is "00"
            # it must exist in the dataframe
            (r"(curr_row\.)(\w+)(\s*==\s*)('00')",
            r"(hasattr(curr_row,'\2') and \1\2\3\4)"),
        ]

        return pattern_replacements

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

    def edit_tbi_branch_logic(self,variable, orig_bl):
        """Modifies branching logic for 
        TBI form to only check the variables 
        that correspond to the number of injuries.

        Parameters
        -----------
        variable: variable from current
        data dictionary row
        """
  
        branch_logic_edits =  {
            "chrtbi_parent_headinjury": 
            "[chrtbi_sourceinfo] = '3' or [chrtbi_sourceinfo] ='2'",
            "chrtbi_subject_head_injury":
            "[chrtbi_sourceinfo] ='3' or [chrtbi_sourceinfo] ='1'",
            "chrtbi_age_first_inj": 
            "[chrtbi_parent_headinjury] ='1' or [chrtbi_subject_head_injury] ='1'",
            "chrtbi_age_recent_inj":
            "[chrtbi_parent_headinjury] ='1' or [chrtbi_subject_head_injury] ='1'",
            "chrtbi_number_injs":
            "[chrtbi_parent_headinjury] ='1' or [chrtbi_subject_head_injury] ='1'",
        }

        for var, new_bl in branch_logic_edits.items():
            if variable == var:
                branching_logic = new_bl
                return branching_logic
        if 'chrtbi' in variable:
            for injury_count in [4,5]:
                if f'{injury_count}' in variable:
                    if 'parent' not in variable:
                        branching_logic = (f"[chrtbi_number_injs] = '{injury_count}'"
                        " and [chrtbi_subject_times] = '3'")
                        return branching_logic
                    else:
                        branching_logic = (f"[chrtbi_number_injs] = '{injury_count}'"
                        " and [chrtbi_parent_times] = '3'")
                        return branching_logic
                    
        return orig_bl

    def edit_past_pharm_branch_logic(self, variable, orig_bl):
        """Edits pharm branching logic to account
        for subject selecting no medication for 
        name of medication

        Parameters 
        --------------
        variable: current variable being processed
        """

        if 'chrpharm_med' in variable:
            number = self.utils.collect_digit(variable)
            if number not in ['1','']:
                new_branching_logic = \
                (f"[chrpharm_med{number}_name_past] <> '999' and"\
                f" [chrpharm_med{int(number)-1}_add_past] = '1'")
            else:
                new_branching_logic = \
                (f"[chrpharm_med{number}_name_past] <> '999'"
                " and [chrpharm_med_past] = '1'")
            if 'onset_past' in variable:
                return new_branching_logic
            elif 'offset_past' in variable:
                return new_branching_logic

        return orig_bl
    

    def edit_av_branch_logic(self, variable, orig_bl):

        if variable == 'chrpsychs_av_audio_expl':
            new_branching_logic = '[chrpsychs_av_audio_yn] = 0'
            return new_branching_logic
        if variable == 'chrpsychs_av_qual_desc':
            new_branching_logic = '[chrpsychs_av_quality] = 0'
            return new_branching_logic
        elif variable == 'chrpsychs_av_dev_desc':
            new_branching_logic = '[chrpsychs_av_deviation] = 0'
            return new_branching_logic
        elif variable == 'chrpsychs_av_pause_reason':
            new_branching_logic = '[chrpsychs_av_pause_rec] = 1'
            return new_branching_logic

        return orig_bl


    def edit_scid_bl(self, variable, orig_bl):
        """
        instructions from nora : chrscid_b45 - chrscid_b64 are blank
        only if any of the fields chrscid_b1-chrscid_b14
        or chrscid_b16-chrscid_bf21 or chrscid_b24 or chrscid_b26-chrscid_b38
        are 3 OR if chrscid_b40 is 3 AND chrscid_b41 = 3 OR chrscid_b42 = 3
        AND chrscid_b43 = 3. if all these fields are not a 3 remove the errors
        """
    
        for var_count in range(45, 65):
            if variable == f'chrscid_b{var_count}':
                new_bl = f"({orig_bl}) and ("
                new_bl = self.loop_scid_conditions(14, 1, new_bl) + " or " 
                new_bl = self.loop_scid_conditions(21, 16, new_bl) + " or "
                new_bl = self.loop_scid_conditions(38, 26, new_bl)
                new_bl += (" or ([chrscid_b40] = 3 and [chrscid_b41] = 3)"
                         " or ([chrscid_b42] = 3 and [chrscid_b43] = 3))")
                checkbox_conds = {'chrscid_b48' : 'chrscid_b49',
                'chrscid_b53' : 'chrscid_b54', 'chrscid_b58' : 'chrscid_b59',
                'chrscid_b63' : 'chrscid_b64'}
                for text_var, checkbox_var in checkbox_conds.items():
                    if variable == text_var:
                        new_bl += f" and ([{checkbox_var}(1)] <> 1)"
                return new_bl
            
        if variable == "chrscid_c56_c65":
            new_bl = f"({orig_bl})"
            new_bl += f" and ([chrscid_b23(1)] <> 1)"
            return new_bl
        
        return orig_bl

    def loop_scid_conditions(self, range_max : int, range_min : int, bl : str):
        new_bl = bl
        for cond_var_count in range(range_min, range_max):
            new_bl+= f"[chrscid_b{cond_var_count}] = 3 or "
        new_bl += f"[chrscid_b{range_max}] = 3"

        return new_bl

                
                
                    

