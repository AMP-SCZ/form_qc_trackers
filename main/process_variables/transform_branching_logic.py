import pandas as pd
import re
import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from utils.branching_logic_eval import (
    BranchingLogicValidationError,
    validate_branching_logic,
)

class TransformBranchingLogic():
    def __init__(
            self, data_dictionary_df, *, utils=None, config_info=None):
        self.utils = utils if utils is not None else Utils()
        self.absolute_path = self.utils.absolute_path
        if config_info is None:
            with open(f'{self.absolute_path}/config.json','r') as file:
                self.config_info = json.load(file)
        else:
            self.config_info = dict(config_info)
        
        # variables with branching logic
        # that can't or is not currently
        # being converted properly
        self.excluded_conversions = {}
        self.invalid_conversions = []

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
        " and float(curr_row.chrsofas_premorbid)!=float(0)"
        " and ((float(curr_row.chrsofas_currscore)/float(curr_row.chrsofas_premorbid))"
        " >=float(0.9))"),
        
        "chriq_pic_completion_raw" :("hasattr(curr_row,'chriq_assessment')"
        " and instance.utils.can_be_float(curr_row.chriq_assessment)==True and"
        " float(curr_row.chriq_assessment)==float(4)"),

        "chriq_scaled_pic_completion" :("hasattr(curr_row,'chriq_assessment')"
        " and instance.utils.can_be_float(curr_row.chriq_assessment)==True and"
        " float(curr_row.chriq_assessment)==float(4)"),

        "chrdig_notes_5" : ("hasattr(curr_row,'chrdig_reason_missing')"
        " and instance.utils.can_be_float(curr_row.chrdig_reason_missing) and float(curr_row.chrdig_reason_missing) == float(3)"
        " and hasattr(curr_row,'chrdig_motivational')"
        " and curr_row.chrdig_motivational != ''"),

        "chrfigs_father_sex" : ("hasattr(curr_row,'chrfigs_father_info')"
        " and instance.utils.can_be_float(curr_row.chrfigs_father_info)==True and"
        " (float(curr_row.chrfigs_father_info)==float(1) or float(curr_row.chrfigs_father_info)==float(1))"),

        "chrfigs_mother_sex" : ("hasattr(curr_row,'chrfigs_mother_info')"
        " and instance.utils.can_be_float(curr_row.chrfigs_mother_info)==True and"
        " (float(curr_row.chrfigs_mother_info)==float(1) or float(curr_row.chrfigs_mother_info)==float(1))")
        }

    def __call__(self):
        self.excluded_conversions = {}
        self.invalid_conversions = []
        os.makedirs(self.config_info['paths']['output_path'], exist_ok=True)
        converted_branching_logic = self.convert_all_branching_logic()
        self.find_problematic_conversions(converted_branching_logic)
        self.find_pattern_exceptions()
        self.validate_converted_branching_logic(
            converted_branching_logic, write_diagnostics=True)
        self.exclude_identifiers()

        self.utils.save_dependency_json(self.excluded_conversions,
         'excluded_branching_logic_vars.json')

        return converted_branching_logic

    def apply_branching_logic_edits(self, variable, branching_logic):
        """Apply every project-specific rewrite through one shared path."""

        for editor in (
                self.edit_tbi_branch_logic,
                self.edit_past_pharm_branch_logic,
                self.edit_av_branch_logic,
                self.edit_scid_bl,
                self.edit_figs_bl):
            branching_logic = editor(variable, branching_logic)
        return branching_logic
        
    def convert_all_branching_logic(self):
        all_converted_branching_logic = {}
        self.data_dictionary_df = self.data_dictionary_df.rename(
        columns={'Variable / Field Name': 'variable',
        'Branching Logic (Show field only if...)': 'branching_logic','Field Type':'field_type'})

        for row in self.data_dictionary_df.itertuples():
            var = getattr(row, 'variable')
            branching_logic = getattr(row, 'branching_logic')
            branching_logic = self.apply_branching_logic_edits(
                var, branching_logic)
            converted_bl = ''
            if re.search(r'\]\s*\[', str(branching_logic)):
                # cross-event [event][field] references cannot be resolved
                # from the single-row eval namespace; bracket stripping used
                # to fuse them into one attribute name that never exists
                # (constant False). Fail closed: exclude from gating instead
                # of emitting runnable output.
                self.excluded_conversions[var] = branching_logic
            elif branching_logic !='':
                converted_bl = self.branching_logic_redcap_to_python(branching_logic)
            if var in self.manual_conversions.keys():
                converted_bl = self.manual_conversions[var]
                self.excluded_conversions.pop(var, None)
            all_converted_branching_logic[var] = {'variable':var,
            'original_branching_logic': branching_logic, 'converted_branching_logic': converted_bl}
        
        return all_converted_branching_logic

    def find_pattern_exceptions(self):
        all_patterns = [ (r"(\()*(\s*)(and|or)?(\s*)(\(*)(\[\w+(\(\d+\))?\])(\s*)"
        r"(<>|=|>=|<=|>|<)(\s*)('[^']*'|\[\w+(\(\d+\))?\]|"
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
            branching_logic = self.apply_branching_logic_edits(
                var, branching_logic)
            converted_bl = ''
            modified_bl = re.sub(
                r'\b(?:or|and)\b', lambda match: match.group(0).lower(),
                str(branching_logic), flags=re.IGNORECASE
            ).replace("\n", ' ').replace('"',"'")
            if modified_bl !='':
                for pattern in all_patterns:
                    modified_bl = re.sub(
                    pattern, '',
                    modified_bl)
            if modified_bl != '':
                if var not in self.manual_conversions.keys():
                    self.excluded_conversions[var] = branching_logic

    def validate_converted_branching_logic(
            self, converted_branching_logic, write_diagnostics=False):
        """Quarantine translated expressions that cannot be evaluated safely.

        The regex translation remains unchanged. This validation is a separate
        fail-closed boundary: syntactically invalid expressions, bare event or
        field names, and unsupported calls are added to the existing exclusion
        map and emitted to a structured diagnostic CSV.
        """

        diagnostics = []
        for variable, values in converted_branching_logic.items():
            converted = values.get('converted_branching_logic', '')
            if converted == '':
                continue
            try:
                validate_branching_logic(
                    converted, allow_legacy_self=False)
            except (BranchingLogicValidationError, SyntaxError) as exc:
                original = values.get('original_branching_logic', '')
                self.excluded_conversions[variable] = original
                diagnostics.append({
                    'variable': variable,
                    'original_branching_logic': original,
                    'converted_branching_logic': converted,
                    'reason': getattr(exc, 'reason', 'syntax_error'),
                    'error_type': type(exc).__name__,
                    'error': str(exc),
                })

        self.invalid_conversions = diagnostics
        columns = [
            'variable', 'original_branching_logic',
            'converted_branching_logic', 'reason', 'error_type', 'error'
        ]
        if write_diagnostics:
            output_path = self.config_info['paths']['output_path']
            os.makedirs(output_path, exist_ok=True)
            final_path = os.path.join(
                output_path, 'branching_logic_invalid_conversions.csv')
            tmp_path = f'{final_path}.{os.getpid()}.tmp'
            try:
                pd.DataFrame(diagnostics, columns=columns).to_csv(
                    tmp_path, index=False)
                os.replace(tmp_path, final_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return diagnostics

    def branching_logic_redcap_to_python(self, branching_logic):
        """
        This function focuses on converting the syntax
        from the REDCap branching logic in the data dictionary
        into Python syntax to be used as conditionals later in the code.

        Parameters
        ----------------
        variable: current variable of interest chrscid_a130
        form: current of interest
        branching logic: redcap version of branching logic 
        """
        modified_branching_logic = str(branching_logic).replace('[', '').replace(
        ']', '').replace('<>', '!=').replace("\n", ' ').replace('"',"'")
        # REDCap boolean operators are case-insensitive; lower them only as
        # whole words so quoted values such as 'FORMER' keep their spelling
        modified_branching_logic = re.sub(
            r'\b(?:or|and)\b', lambda match: match.group(0).lower(),
            modified_branching_logic, flags=re.IGNORECASE)

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
        ident_df = pd.read_csv(os.path.join(
            depend_path, 'identifier_effects.csv'))
        for row in ident_df.itertuples():
            if row.affected_col == 'branching_logic':
                self.excluded_conversions[row.var] = row.affected_col_val
    
    def format_floats_and_vars(self):
        pattern_replacements = [
            # Replaces single equals sign "=" with double equals sign "==" 
            (r"(?<!=)(?<![<>!])=(?!=)", r"=="),  
            # Converts numbers (that do not equal '00') prececeded by comparison operators to floats.
            # A quoted value only counts as a number when the quotes wrap the
            # whole token; otherwise "'1903-03-03'" used to be split around
            # its leading digits, producing invalid Python. Non-numeric quoted
            # literals now fall through to the string-comparison guards below.
            (r"([=<>]\s*)('(?!00)-?\d+(\.\d+)?'|(?!00)-?\d+(\.\d+)?)", r"\1float(\2)"),
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
        # Every replacement below is wrapped in one outer pair of parentheses:
        # the expansions contain top-level "or" clauses, and without the outer
        # parens a neighbouring source-level "and" regroups them under
        # Python's operator precedence, changing the truth value.
        pattern_replacements = [
            # if it is checking if two variables are equal: equal when both
            # are present and non-blank with equal floats or equal strings,
            # or when both are effectively blank (absent counts as blank,
            # matching how [x]='' is converted)
            (r'(curr_row\.)(\w+)(\s*==\s*)(curr_row\.)(\w+)',
            (r"((((hasattr(curr_row,'\2') and \1\2!='') and (hasattr(curr_row,'\5') and \4\5!=''))"
            r" and ((instance.utils.can_be_float(\1\2)==True and"
            r" instance.utils.can_be_float(\4\5)==True and float(\1\2) \3 float(\4\5))"
            r" or (str(\1\2) \3 str(\4\5))))"
            r" or ((not hasattr(curr_row,'\2') or \1\2=='')"
            r" and (not hasattr(curr_row,'\5') or \4\5=='')))")),
            # if it is checking if two variables are greater than or less
            # than each other: compares floats when both are numeric and
            # falls back to string ordering (correct for ISO dates)
            # otherwise; blank or absent values compare False, matching the
            # calculated-field runtime
            (r'(curr_row\.)(\w+)(\s*)(>=|<=|>|<)(\s*)(curr_row\.)(\w+)',
            (r"((hasattr(curr_row,'\2') and hasattr(curr_row,'\7'))"
            r" and \1\2!='' and \6\7!=''"
            r" and ((instance.utils.can_be_float(\1\2)==True and"
            r" instance.utils.can_be_float(\6\7)==True and float(\1\2) \4 float(\6\7))"
            r" or ((instance.utils.can_be_float(\1\2)==False or"
            r" instance.utils.can_be_float(\6\7)==False) and str(\1\2) \4 str(\6\7))))")),
            # not-equal is the negation of the equality expansion above, so
            # mixed states (one numeric/one blank, one absent/one blank)
            # cannot crash or fall through to a wrong default
            (r'(curr_row\.)(\w+)(\s*!=\s*)(curr_row\.)(\w+)',
            (r"(not ((((hasattr(curr_row,'\2') and \1\2!='') and (hasattr(curr_row,'\5') and \4\5!=''))"
            r" and ((instance.utils.can_be_float(\1\2)==True and"
            r" instance.utils.can_be_float(\4\5)==True and float(\1\2) == float(\4\5))"
            r" or (str(\1\2) == str(\4\5))))"
            r" or ((not hasattr(curr_row,'\2') or \1\2=='')"
            r" and (not hasattr(curr_row,'\5') or \4\5==''))))"))
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
        # Guards every remaining comparison against a non-empty quoted
        # literal ('00', 'A11', '1903-03-03', ...): these are the values the
        # float pass deliberately leaves as strings. REDCap treats a missing
        # column as blank, so inequality is satisfied by an absent field
        # while equality (and ordering) requires the field to exist. Without
        # these guards the bare attribute access raises AttributeError the
        # first time the column is absent from a row batch.
        pattern_replacements = [
            (r"(curr_row\.)(\w+)(\s*!=\s*)('[^']+')",
            r"(not hasattr(curr_row,'\2') or \1\2\3\4)"),
            (r"(curr_row\.)(\w+)(\s*==\s*)('[^']+')",
            r"(hasattr(curr_row,'\2') and \1\2\3\4)"),
            # ordered comparisons against a quoted non-numeric literal
            # (e.g. ISO dates) string-compare and require a present,
            # non-blank value, matching the calculated-field runtime
            (r"(curr_row\.)(\w+)(\s*(?:>=|<=|>|<)\s*)('[^']+')",
            r"(hasattr(curr_row,'\2') and \1\2!='' and \1\2\3\4)"),
        ]

        return pattern_replacements

    def find_problematic_conversions(self,converted_bl):
        exceptions = []
        count = 0
        
        # No name-based skip list here: the previous one ('error', '_err',
        # 'invalid', ...) hid exactly the families whose conversions were
        # broken, so the QC CSV could never surface them.
        for var, values in converted_bl.items():
            converted_expression = values['converted_branching_logic']
            if ('float' not in converted_expression and converted_expression != ''
            and "!=''" not in converted_expression and 'pharm' not in converted_expression):
                count+=1
            else:
                split_bl = converted_expression.replace(' or ',' and ').split(' and ')
                for bl_sect in split_bl:
                    if (("float" not in bl_sect)
                    and bl_sect !='' and "!=''" not in bl_sect
                    and re.search(r"(?:==|!=|>=|<=|>|<)\s*'[^']*'", bl_sect) is None):
                        exceptions.append({"var":var,"bl":split_bl,"full":converted_expression})
                        break
                    
        exceptions_df = pd.DataFrame(exceptions)
        exceptions_df.to_csv(
        os.path.join(
            self.config_info["paths"]["output_path"],
            'branching_logic_qc.csv'),
        index = False)

    def edit_tbi_branch_logic(self,variable, orig_bl):
        """
        Modifies branching logic for 
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
        injury_detail = re.fullmatch(
            r"chrtbi_(subject|parent)_"
            r"(?:age|anterograde|circumstance|length|medical_findings|"
            r"med_find|retrograde|symptom_length|symptoms2_)([45])",
            variable)
        if injury_detail is not None:
            reporter, injury_count = injury_detail.groups()
            return (f"[chrtbi_number_injs] >= '{injury_count}'"
                    f" and [chrtbi_{reporter}_times] = '3'")
                    
        return orig_bl

    def edit_past_pharm_branch_logic(self, variable, orig_bl):
        """Edits pharm branching logic to account
        for subject selecting no medication for 
        name of medication

        Parameters 
        --------------
        variable: current variable being processed
        """

        med_match = re.match(r'chrpharm_med(\d+)', variable)
        if med_match is None:
            # root gates (chrpharm_med/chrpharm_med_past) and the medstop
            # list/table variables are not per-medication fields; gating
            # them on medication-number fields produced constant-False
            # logic ("... and ()") for the root medication questions
            return orig_bl
        #add onset, offset, and continue variables as exceptions here
        if (any(excluded_key in variable for
        excluded_key in ['onset','offset','interm_meds'])):
            return orig_bl
        suffix = '_past' if '_past' in variable else ''
        number = med_match.group(1)
        if number != '1':
            gate = (f"[chrpharm_med{number}_name{suffix}] <> '999' and"
            f" [chrpharm_med{int(number)-1}_add{suffix}] = '1'")
        else:
            gate = (f"[chrpharm_med{number}_name{suffix}] <> '999'"
            f" and [chrpharm_med{suffix}] = '1'")
        if str(orig_bl).strip() == '':
            return f"({gate})"
        return f"({gate}) and ({orig_bl})"
    
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

    def edit_figs_bl(
        self, variable : str, orig_bl : str
    ) -> str:
        for family_member in ['mother','father']:
            if variable in  [f'chrfigs_{family_member}_sex',
            f'chrfigs_{family_member}_age']:
                new_branching_logic = f'[chrfigs_{family_member}_info] = 1'
                return new_branching_logic
        for count in range(1,10):
            for family_member in ['child','sibling']:
                if variable in [f'chrfigs_{family_member}{count}_sex',
                f'chrfigs_{family_member}{count}_age']:
                    new_branching_logic = f'[chrfigs_{family_member}{count}_info] = 1'
                    return new_branching_logic

        return orig_bl

    def edit_scid_bl(
        self, variable : str, orig_bl : str
    ) -> str:
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
                new_bl += "[chrscid_b24] = 3 or "
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

    def loop_scid_conditions(
        self, range_max : int, range_min : int, bl : str
    ) -> str:
        new_bl = bl
        for cond_var_count in range(range_min, range_max):
            new_bl+= f"[chrscid_b{cond_var_count}] = 3 or "
        new_bl += f"[chrscid_b{range_max}] = 3"

        return new_bl

                
                
                    
