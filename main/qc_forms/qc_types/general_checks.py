import pandas as pd

import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck
import re

class GeneralChecks(FormCheck):  
    """
    General QC checks that are applied
    to all forms
    """      
    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network,form_check_info)
        self.test_val = 0
        self.call_checks(row)
        
    def __call__(self):

        return self.final_output_list

    def call_checks(self, row):
        self.row = row
        self.check_blank_values(row)
        self.check_missing_code_values(row)
        self.call_spec_val_check(row)
        self.check_form_completion(row)
        self.call_range_checks(row)
        for guid_var in ['chrguid_guid','chrguid_pseudoguid']:
            self.guid_format_check(row, ['guid_form'],
            [guid_var],{'reports':['Main Report','Non Team Forms'],
            "withdrawn_enabled" : True},
            bl_filtered_vars=[guid_var],filter_excl_vars=True,
            checked_guid_var=guid_var)
        self.missing_data_form_consistency_check(row)

        #self.missing_code_check(row)

    def call_range_checks(self, row):
        for var, range_dict in self.variable_ranges[self.network].items():
            min = range_dict['min']
            max = range_dict['max']
            self.range_check(row, [range_dict['form']],[var],
            {"reports" : ['Main Report']},bl_filtered_vars=[],
            filter_excl_vars=True,range_var = var,
            lower = min,upper = max)

    def check_blank_values(self, row):
        #TODO:optimize performance of this part
        for report in ['Main Report', 'Secondary Report']:
            blank_check_forms = self.general_check_vars["blank_check_vars"][
            self.network][report]
            for form in blank_check_forms:
                if (form in self.general_check_vars["excluded_forms"][
                self.network]):
                    continue
                report_list = [report]
                for team, forms in self.forms_per_report.items():
                    if form in forms:
                        report_list.append(team)
                if self.standard_form_filter(row, form):
                    for var in blank_check_forms[form]:
                        if self.prescient_scid_filter(var, row) == True:
                            continue
                        self.check_if_blank(row, [form], [var],
                        {"reports" : report_list},[var])

    def check_missing_code_values(self, row):
        #TODO:optimize performance of this part
        for report in ['Main Report']:
            missing_code_forms = self.general_check_vars["missing_code_vars"][
            self.network][report]
            for form in missing_code_forms:
                report_list = [report]
                for team, forms in self.forms_per_report.items():
                    if form in forms:
                        report_list.append(team)
                if self.standard_form_filter(row, form):
                    for var in missing_code_forms[form]:
                        if self.prescient_scid_filter(var, row) == True:
                            continue
                        self.check_if_missing_code(row, [form], [var],
                        {"reports" : report_list},[var])

    def prescient_scid_filter(self, var, row):
        """
        Filters out modules
        not used for Prescient 
        """
        if var in self.module_b_vars or var in self.module_c_vars:
            if self.network == 'PRESCIENT':
                if (hasattr(row,'chrscid_b1') and row.chrscid_b1 in (
                self.utils.missing_code_list + [''])
                or hasattr(row,'chrscid_b16') and row.chrscid_b16 in (
                self.utils.missing_code_list + [''])):
                    return True
        return False

    @FormCheck.standard_qc_check_filter
    def check_if_blank(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):  
        """
        Standard check applied across all
        forms to see if form is blank
        """
        if hasattr(row,all_vars[0]) and getattr(row, all_vars[0]) == '':
            return "Variable is blank."
        return 

    @FormCheck.standard_qc_check_filter
    def check_if_missing_code(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):  
        if  (getattr(row, all_vars[0])
        in self.utils.missing_code_set or
        str(getattr(row, all_vars[0])).replace(' ','') == 'NaN'):
            return "Variable is a missing code."
        return 
    
    def call_spec_val_check(self,row):
        for var, conditions in self.general_check_vars["specific_val_check_vars"].items():
            if var in self.grouped_vars['var_forms'].keys():
                form = self.grouped_vars['var_forms'][var]
                if self.standard_form_filter(row, form):
                    self.specific_value_checks(row,[form],[var],
                    {'reports':conditions['report']},[],True,conditions=conditions)

    @FormCheck.standard_qc_check_filter
    def specific_value_checks(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, conditions={}
    ):
        var = conditions['correlated_variable']
        if hasattr(row, var): 
            if conditions["negative"] == 'False':
                if getattr(row, var) in conditions["checked_value_list"]:
                    return conditions['message']
            elif conditions["negative"] == 'True':
                if getattr(row, var) not in conditions["checked_value_list"]:
                    return conditions['message']
                
    def check_form_completion(self,row):
        cohort = self.subject_info[row.subjectid]['cohort']
        if cohort.lower() not in ["hc", "chr"]:
            return
        curr_tp_forms = self.forms_per_tp[cohort][self.timepoint]
        if (self.check_if_next_tp(row) == True ):
            for form in curr_tp_forms:
                if (form in self.prescient_forms_no_compl_status
                or form in self.general_check_vars["excluded_forms"][
                self.network]):
                    continue
                compl_var = self.important_form_vars[form]['completion_var']
                if self.network == 'PRESCIENT':
                    compl_var += '_rpms'
                    # rpms compl variables same for both cohorts
                    compl_var = compl_var.replace('_hc','')
                    if 'informed_consent' in compl_var:
                        # Skip ONLY this form, not the whole loop —
                        # `return` here previously aborted check_form_completion
                        # for every form ordered after informed_consent in
                        # `curr_tp_forms`, silently dropping completion checks.
                        continue
                if (hasattr(row, compl_var) and 
                getattr(row, compl_var) not in self.utils.all_dtype([2,3,4])):
                    error_message = (f"{form} not marked as complete, but"
                    " subject has started the next timepoint")
                    if self.network == 'PRONET':
                        report_list =  ['Incomplete Forms']
                    else:
                        report_list =  ['Main Report', 'Incomplete Forms']

                    for team, forms in self.forms_per_report.items():
                        if form in forms:
                            report_list.append(team)

                    output_changes = {'reports' : report_list}
                    error_output = self.create_row_output(
                    row,[form],[compl_var], error_message, output_changes)
                    self.final_output_list.append(error_output)

    def guid_format_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, checked_guid_var = 'chrguid_guid'
    ):
        """Checks if GUID is in proper format"""
        if not hasattr(row,checked_guid_var):
            return
        guid = str(getattr(row,checked_guid_var))
        if guid == '' or guid in self.utils.missing_code_set:
            return
        if not re.search(r"^NDAR[A-Z0-9]+$", guid):
            error_message = f"GUID in incorrect format. GUID was reported to be {guid}."
            error_output = self.create_row_output(
            row,filtered_forms,[checked_guid_var], error_message,
            changed_output_vals)
            self.final_output_list.append(error_output)

    def range_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, range_var = '', lower = 0, upper = 100
    ):  
        if not hasattr(row, range_var):
            return
        var_val = getattr(row, range_var)
        if (self.utils.can_be_float(var_val) and
        var_val not in self.utils.missing_code_set):
            if float(var_val) < lower or float(var_val) > upper:
                error_message = f'{range_var} value ({var_val}) is out of range'
                error_output = self.create_row_output(
                row, filtered_forms, [range_var], error_message, 
                changed_output_vals)
                self.final_output_list.append(error_output)

    def age_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        """
        Checks all ages to make sure
        they are in the correct range.
        """
        age = self.subject_info[row.subjectid]["age"]
        if age == "unknown":
            return "Age is Unknown."
        elif self.utils.can_be_float(age) and age < 12 or age > 30:
            return f"Age ({age}) is out of range."

    # missing_data form trigger variables (per missing_data_form_info.csv).
    # All four are radio fields with single choice value 1; set to 1 means
    # missingness of that scope has been recorded on the missing_data form.
    # Time/withdrawn/discon imply the WHOLE TP (and beyond, for the latter
    # two) is missing. domain implies only specific assessment domains
    # listed in chrmiss_domain_type___1..7 are missing.
    _WHOLE_TP_TRIGGER_VARS = (
        'chrmiss_time', 'chrmiss_withdrawn', 'chrmiss_discon',
    )
    _DOMAIN_TRIGGER_VAR = 'chrmiss_domain'
    _MISSING_DATA_TRIGGER_VARS = (
        _WHOLE_TP_TRIGGER_VARS + (_DOMAIN_TRIGGER_VAR,)
    )

    def missing_data_form_consistency_check(self, row):
        """
        Flags contradictions between the missing_data form (chrmiss_*
        triggers) and individual forms' missing-data buttons
        (chr{form}_missing).

        Three flag types:
          A) Per form. chr{form}_missing=1 but none of the missing_data
             triggers are set — the RA marked the form missing without
             recording the reason on the missing_data form.
          B-strict) Per form. A whole-TP trigger (chrmiss_time /
             chrmiss_withdrawn / chrmiss_discon) is set, but the form's
             missing button is not checked. When the whole TP/participant
             is missing, every applicable form is expected to have its
             missing button set.
          B-coarse) Single summary. chrmiss_domain=1 (with no whole-TP
             trigger) but no individual form has its missing button
             checked. Domain-aware verification (matching
             chrmiss_domain_type___N to forms in that domain) requires a
             domain→form mapping not yet available; coarse "at least one"
             check is the conservative substitute.

        Gate: runs when the subject has moved past this TP OR any
        chrmiss_* trigger is set. The OR is required because withdrawn /
        discontinued participants will not progress past the withdrawal
        TP, so check_if_next_tp alone would never evaluate exactly the
        rows we most need to check.

        Form applicability filters (mirrored from check_form_completion /
        extra_form_conditions): forms in excluded_forms[network] and forms
        whose extra conditions don't apply (axivity/mindlamp opt-out,
        pubertal scale age cutoff) are excluded from Direction B-strict
        to avoid flooding withdrawal rows with irrelevant flags.
        """
        cohort = self.subject_info[row.subjectid]['cohort']
        if cohort.lower() not in ['hc', 'chr']:
            return
        # Skip if the missing_data form columns aren't present on this row
        # (TP that doesn't include the missing_data form).
        if not any(hasattr(row, v) for v in self._MISSING_DATA_TRIGGER_VARS):
            return

        triggered_chrmiss_vars = [
            v for v in self._MISSING_DATA_TRIGGER_VARS
            if hasattr(row, v)
            and getattr(row, v) in self.utils.all_dtype([1])
        ]
        any_trigger_set = len(triggered_chrmiss_vars) > 0
        whole_tp_triggered = [
            v for v in triggered_chrmiss_vars
            if v in self._WHOLE_TP_TRIGGER_VARS
        ]
        has_whole_tp_trigger = len(whole_tp_triggered) > 0
        has_domain_trigger = self._DOMAIN_TRIGGER_VAR in triggered_chrmiss_vars

        if self.check_if_next_tp(row) != True and not any_trigger_set:
            return

        excluded_forms = self.general_check_vars['excluded_forms'][self.network]
        applicable_forms = []   # (form, missing_var) pairs eligible for B-strict
        forms_marked_missing = []   # subset of applicable_forms with _missing=1
        for form, vars in self.important_form_vars.items():
            if form == 'missing_data':
                continue
            missing_var = vars.get('missing_var', '')
            if missing_var == '':
                continue
            if not hasattr(row, missing_var):
                continue
            if form in excluded_forms:
                continue
            if self.extra_form_conditions(row, form) == False:
                continue
            applicable_forms.append((form, missing_var))
            if getattr(row, missing_var) in self.utils.all_dtype([1]):
                forms_marked_missing.append((form, missing_var))

        # Direction A: per-form, _missing=1 but no triggers on missing_data.
        if not any_trigger_set and forms_marked_missing:
            for form, missing_var in forms_marked_missing:
                error_message = (
                    f"{form} marked as missing on the form, but the"
                    " missing_data form has no missing-data triggers set"
                    " for this timepoint."
                )
                error_output = self.create_row_output(
                row, [form, 'missing_data'],
                [missing_var] + list(self._MISSING_DATA_TRIGGER_VARS),
                error_message, {'reports': ['Missingness Report']})
                self.final_output_list.append(error_output)

        # Direction B-strict: whole-TP trigger set but a form's missing
        # button is not checked. One flag per applicable form.
        if has_whole_tp_trigger:
            marked_set = {f for f, _ in forms_marked_missing}
            for form, missing_var in applicable_forms:
                if form in marked_set:
                    continue
                error_message = (
                    "missing_data form indicates whole-timepoint"
                    f" missingness ({', '.join(whole_tp_triggered)} set),"
                    f" but {form} does not have its missing-data button"
                    " checked."
                )
                error_output = self.create_row_output(
                row, [form, 'missing_data'],
                [missing_var] + whole_tp_triggered,
                error_message, {'reports': ['Missingness Report']})
                self.final_output_list.append(error_output)

        # Direction B-coarse: domain-only trigger, but no form is marked
        # missing anywhere. Single summary flag attributed to missing_data.
        if (has_domain_trigger and not has_whole_tp_trigger
        and not forms_marked_missing):
            error_message = (
                "missing_data form indicates a domain is missing"
                " (chrmiss_domain set), but no individual form has its"
                " missing-data button checked for this timepoint."
            )
            error_output = self.create_row_output(
            row, ['missing_data'], [self._DOMAIN_TRIGGER_VAR],
            error_message, {'reports': ['Missingness Report']})
            self.final_output_list.append(error_output)

    def missing_code_check(self, row):
        for form, vars in self.important_form_vars.items():
            if 'missing_spec_var' in list(vars.keys()) and vars['missing_spec_var'] != '':                
                if (hasattr(row, vars['missing_spec_var'])
                and getattr(row, vars['missing_var']) in self.utils.all_dtype([1])
                and getattr(row, vars['missing_spec_var']) == ''):
                    form = self.grouped_vars['var_forms'][vars['missing_spec_var']]
                    error_message = f'Missing data button clicked, but reason not specified.'
                    error_output = self.create_row_output(
                    row, [form], [vars['missing_spec_var'],vars['missing_var']], error_message, 
                     {"reports" : ['Main Report']})
                    self.final_output_list.append(error_output)
    
    def date_inconsistency_check(self, row):
        pass