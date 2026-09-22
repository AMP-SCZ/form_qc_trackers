import pandas as pd

import os
import sys
import json
from datetime import datetime
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck, _compile_bl
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

    # Blank-check flags on these forms are surfaced as priority items so they
    # don't get buried — pharm course-level data is hand-entered and the
    # missing-field count tends to be high.
    _PRIORITY_BLANK_FORMS = {
        'current_pharmaceutical_treatment_floating_med_125',
        'current_pharmaceutical_treatment_floating_med_2650',
        'past_pharmaceutical_treatment',
    }

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
                output_changes = {"reports" : report_list}
                if form in self._PRIORITY_BLANK_FORMS:
                    output_changes["priority_item"] = True
                form_passes_filter = self.standard_form_filter(row, form)
                for var in blank_check_forms[form]:
                    # pharm variables are blank-checked even when the form
                    # is incomplete or marked missing
                    if (not form_passes_filter and
                    not var.startswith(self._PHARM_VAR_PREFIX)):
                        continue
                    if self.prescient_scid_filter(var, row) == True:
                        continue
                    self.check_if_blank(row, [form], [var],
                    output_changes,[var])

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

    # Pharm-form variables (all prefixed chrpharm_) bypass the filters in
    # standard_qc_check_filter — cohort/timepoint/form-completion exclusion
    # and excluded-variable lists — so blank pharm fields are always
    # flagged. Branching logic is still honored so that fields whose
    # branch never opened (e.g. med slots beyond the number of meds
    # entered) are not flagged.
    _PHARM_VAR_PREFIX = 'chrpharm_'

    def check_if_blank(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        """
        Standard check applied across all
        forms to see if form is blank
        """
        if all_vars and str(all_vars[0]).startswith(self._PHARM_VAR_PREFIX):
            self._check_if_blank_pharm(row, filtered_forms, all_vars,
            changed_output_vals, bl_filtered_vars)
            return
        self._check_if_blank_filtered(row, filtered_forms, all_vars,
        changed_output_vals, bl_filtered_vars, filter_excl_vars)

    def _check_if_blank_pharm(self, curr_row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[]
    ):
        if not (hasattr(curr_row, all_vars[0])
        and getattr(curr_row, all_vars[0]) == ''):
            return
        # converted branching logic expressions reference the local names
        # `curr_row` and `instance` (see standard_qc_check_filter)
        instance = self
        for var in bl_filtered_vars:
            if var in self.excl_bl.keys():
                return
            bl = self.conv_bl[var]["converted_branching_logic"]
            if bl != "" and eval(_compile_bl(bl)) == False:
                return
        error_output = self.create_row_output(
        curr_row, filtered_forms, all_vars, "Variable is blank.",
        changed_output_vals)
        self.final_output_list.append(error_output)

    @FormCheck.standard_qc_check_filter
    def _check_if_blank_filtered(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):  
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

    def _missingness_output_changes(self, check_id):
        """Common routing/identity fields for Missingness Report flags."""
        return {
            'reports': ['Missingness Report'],
            'withdrawn_enabled': True,
            'nda_excluder': False,
            'check_id': check_id,
        }

    def _append_missingness_flag(self, row, forms, variables, message,
                                 check_id):
        output = self.create_row_output(
            row, forms, variables, message,
            self._missingness_output_changes(check_id))
        self.final_output_list.append(output)

    def _has_moved_past_timepoint(self, row):
        """Safe visit-order check; terminal/noncanonical statuses are not TPs."""
        if self.timepoint not in self.tp_list:
            return False
        visit_status = getattr(
            row, 'visit_status',
            self.subject_info.get(row.subjectid, {}).get('visit_status', ''))
        formatted_status = str(visit_status).replace('_', '')
        if formatted_status not in self.tp_list:
            return False
        return (self.tp_list.index(formatted_status)
                > self.tp_list.index(self.timepoint))

    def _missing_button_available(self, row, form, form_info):
        """Whether this form's missing button existed on the assessment date."""
        missing_var = form_info.get('missing_var', '')
        added_on = self.vars_added_later.get(form, {}).get(missing_var)
        if not added_on:
            return True
        date_var = form_info.get('interview_date_var', '')
        if not date_var or not hasattr(row, date_var):
            return False
        raw_date = str(getattr(row, date_var)).strip().split(' ')[0]
        try:
            return (datetime.strptime(raw_date, '%Y-%m-%d')
                    >= datetime.strptime(added_on, '%Y-%m-%d'))
        except ValueError:
            # Do not demand a button when the record cannot establish that the
            # field existed; a separate date-format check owns malformed dates.
            return False

    def _scheduled_missingness_forms(self, row, cohort_key):
        """Scheduled, present, applicable forms whose missing button existed."""
        scheduled = self.forms_per_tp.get(cohort_key, {}).get(
            self.timepoint, [])
        excluded = set(self.general_check_vars.get(
            'excluded_forms', {}).get(self.network, []))
        applicable = []
        for form in dict.fromkeys(scheduled):
            if form == 'missing_data' or form in excluded:
                continue
            form_info = self.important_form_vars.get(form, {})
            missing_var = form_info.get('missing_var', '')
            if not missing_var or not hasattr(row, missing_var):
                continue
            if self.extra_form_conditions(row, form) is False:
                continue
            if not self._missing_button_available(row, form, form_info):
                continue
            applicable.append((form, missing_var))
        return applicable

    def _form_completion_var(self, form):
        completion_var = self.important_form_vars[form].get(
            'completion_var', '')
        if self.network == 'PRESCIENT' and completion_var:
            completion_var = (completion_var + '_rpms').replace(
                '_hc', '').replace('onboarding', 'checkin')
        return completion_var

    def _form_is_complete(self, row, form):
        completion_var = self._form_completion_var(form)
        return (completion_var and hasattr(row, completion_var)
                and getattr(row, completion_var)
                in self.utils.all_dtype([2]))

    def _selected_missing_domains(self, row):
        selected = []
        for code, info in self.missingness_domain_forms.items():
            if not str(code).isdigit() or not isinstance(info, dict):
                continue
            variable = f'chrmiss_domain_type___{code}'
            if (hasattr(row, variable)
                    and getattr(row, variable) in self.utils.all_dtype([1])):
                selected.append((str(code), info))
        return selected

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
          B-domain) Per mapped form. chrmiss_domain=1 requires at least one
             chrmiss_domain_type___N selection; every scheduled/applicable
             form in each selected domain must be recorded missing. Forms in
             unrelated domains cannot satisfy or trigger that domain check.

        Gate: runs when the subject has moved past this TP OR any
        chrmiss_* trigger is set. The OR is required because withdrawn /
        discontinued participants will not progress past the withdrawal
        TP, so check_if_next_tp alone would never evaluate exactly the
        rows we most need to check.

        Form applicability starts with forms_per_timepoint for the normalized
        cohort and current timepoint, then applies network exclusions,
        axivity/mindlamp opt-in and pubertal-age conditions, column presence,
        and missing-button availability dates.
        """
        cohort_key = str(
            self.subject_info[row.subjectid]['cohort']).strip().lower()
        if cohort_key not in ['hc', 'chr']:
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
        selected_domains = self._selected_missing_domains(row)

        if (not any_trigger_set and not selected_domains
                and not self._has_moved_past_timepoint(row)):
            return

        applicable_forms = self._scheduled_missingness_forms(row, cohort_key)
        forms_marked_missing = [
            (form, missing_var) for form, missing_var in applicable_forms
            if self.check_if_missing(row, form) is True]

        if selected_domains and not has_domain_trigger:
            selected_vars = [
                f'chrmiss_domain_type___{code}'
                for code, _ in selected_domains]
            self._append_missingness_flag(
                row, ['missing_data'],
                [self._DOMAIN_TRIGGER_VAR] + selected_vars,
                "A missing assessment domain is selected, but"
                " chrmiss_domain is not set.",
                'missingness_domain_selected_without_trigger')

        # Direction A: per-form, _missing=1 but no triggers on missing_data.
        if not any_trigger_set and forms_marked_missing:
            for form, missing_var in forms_marked_missing:
                error_message = (
                    f"{form} marked as missing on the form, but the"
                    " missing_data form has no missing-data triggers set"
                    " for this timepoint."
                )
                self._append_missingness_flag(
                    row, [form, 'missing_data'],
                    [missing_var] + list(self._MISSING_DATA_TRIGGER_VARS),
                    error_message,
                    f'missingness_form_without_trigger::{form}')

        # Direction B-strict: whole-TP trigger set but a form's missing
        # button is not checked. One flag per applicable form.
        if has_whole_tp_trigger:
            marked_set = {f for f, _ in forms_marked_missing}
            completed_forms = []
            for form, missing_var in applicable_forms:
                if form in marked_set:
                    continue
                if self._form_is_complete(row, form):
                    completed_forms.append(form)
                    continue
                error_message = (
                    "missing_data form indicates whole-timepoint"
                    " missingness,"
                    f" but {form} does not have its missing-data button"
                    " checked."
                )
                self._append_missingness_flag(
                    row, [form, 'missing_data'],
                    [missing_var] + whole_tp_triggered,
                    error_message,
                    f'missingness_whole_tp_unchecked::{form}')

            # A complete form contradicts chrmiss_time specifically. Completed
            # forms are allowed at the visit where withdrawal/discontinuation
            # begins because those triggers apply to remaining assessments.
            if ('chrmiss_time' in whole_tp_triggered and completed_forms):
                error_message = (
                    "missing_data marks the complete assessment time as"
                    " missing, but completed scheduled forms were found: "
                    + ', '.join(completed_forms) + ". Review chrmiss_time"
                    " rather than marking completed forms missing."
                )
                self._append_missingness_flag(
                    row, ['missing_data'] + completed_forms,
                    ['chrmiss_time'] + [
                        self._form_completion_var(form)
                        for form in completed_forms],
                    error_message, 'missingness_whole_tp_has_completed_forms')

        # Domain-aware direction: every scheduled form in each selected domain
        # must be recorded missing. Buttons in unrelated domains do not satisfy
        # the selected domain and administrative forms are intentionally absent
        # from the seven-domain mapping.
        if has_domain_trigger and not has_whole_tp_trigger:
            if not selected_domains:
                self._append_missingness_flag(
                    row, ['missing_data'],
                    [self._DOMAIN_TRIGGER_VAR, 'chrmiss_domain_type'],
                    "chrmiss_domain is set, but no missing assessment domain"
                    " is selected.",
                    'missingness_domain_type_not_selected')
                return

            applicable_by_form = dict(applicable_forms)
            marked_set = {form for form, _ in forms_marked_missing}
            selected_domain_forms = {
                form
                for _, domain_info in selected_domains
                for form in domain_info.get('forms', [])
            }
            for form, missing_var in forms_marked_missing:
                if form in selected_domain_forms:
                    continue
                self._append_missingness_flag(
                    row, [form, 'missing_data'],
                    [missing_var, self._DOMAIN_TRIGGER_VAR] + [
                        f'chrmiss_domain_type___{code}'
                        for code, _ in selected_domains],
                    f"{form} is recorded as missing, but it is not part of"
                    " any selected missing assessment domain.",
                    f'missingness_form_outside_selected_domain::{form}')

            for code, domain_info in selected_domains:
                domain_label = domain_info.get('label', f'Domain {code}')
                domain_forms = [
                    form for form in domain_info.get('forms', [])
                    if form in applicable_by_form]
                if not domain_forms:
                    continue

                completed_forms = []
                for form in domain_forms:
                    if form in marked_set:
                        continue
                    if self._form_is_complete(row, form):
                        completed_forms.append(form)
                        continue
                    missing_var = applicable_by_form[form]
                    self._append_missingness_flag(
                        row, [form, 'missing_data'],
                        [missing_var, self._DOMAIN_TRIGGER_VAR,
                         f'chrmiss_domain_type___{code}'],
                        f"missing_data marks {domain_label} as missing, but"
                        f" {form} is not recorded as missing.",
                        f'missingness_domain_unchecked::{code}::{form}')

                if completed_forms:
                    self._append_missingness_flag(
                        row, ['missing_data'] + completed_forms,
                        [self._DOMAIN_TRIGGER_VAR,
                         f'chrmiss_domain_type___{code}'] + [
                            self._form_completion_var(form)
                            for form in completed_forms],
                        f"missing_data marks {domain_label} as missing, but"
                        " completed forms from that domain were found: "
                        + ', '.join(completed_forms) + ".",
                        f'missingness_domain_has_completed_forms::{code}')

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
