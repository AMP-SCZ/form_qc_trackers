import pandas as pd

import os
import sys
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck
from qc_forms.qc_types.date_check_logic import (
    find_backward_visit_dates,
    is_form_marked_missing,
    is_date_report_excluded_form,
    is_date_report_excluded_variable,
)


class DateChecks(FormCheck):
    """
    QC checks for exact-field cross-timepoint date ordering (qc_type "date").

    Flags a timepoint when an interview-date field is earlier than that exact
    same REDCap variable at an earlier timepoint for the subject. Every
    configured interview-date field present on a current form not explicitly
    marked missing is checked. Earlier-timepoint per-variable maxima are
    precomputed by process_variables/collect_multi_timepoint_data.py and exposed through
    form_check_info['earliest_latest_dates_per_tp'].

    Runs per (subject, tp) like the other FormCheck subclasses and emits one
    finding for every distinct offending current form/variable/date.
    Flags route to a dedicated "Date Report", which create_trackers
    surfaces as its own tracker tab automatically (collect_new_reports),
    the same way Blood/Fluids Report tabs are produced. To also surface
    these in per-site Main Report tabs, add 'Main Report' to the route.
    """

    DATE_REPORT = 'Date Report'
    _date_fields_source = None
    _date_fields = ()

    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network, form_check_info)
        if DateChecks._date_fields_source is not self.important_form_vars:
            DateChecks._date_fields_source = self.important_form_vars
            DateChecks._date_fields = tuple(
                (form, info.get('interview_date_var', ''))
                for form, info in self.important_form_vars.items()
                if (not is_date_report_excluded_form(form)
                    and info.get('interview_date_var', '')
                    and not is_date_report_excluded_variable(
                        info.get('interview_date_var', ''))))
        self.call_checks(row)

    def __call__(self):
        return self.final_output_list

    def call_checks(self, row):
        self.check_visit_date_order(row)

    def check_visit_date_order(self, row):
        """
        Compare every interview date on this (subject, tp) row against the
        same variable at earlier canonical timepoints. The comparison and
        floating/conversion + missing-date filtering live in the pure helper;
        here we fetch prior per-variable dates and pass the full REDCap
        missing-code set so
        no missing code — including the date-shaped sentinels 1909-09-09 /
        1903-03-03 / 1901-01-01 — is ever read as a real visit date, and
        wrap any violation as an output row that names the date variable
        at each timepoint.
        """
        subject = row.subjectid
        subject_tp_dates = self.tp_date_ranges.get(subject, {})
        # String-form the whole missing-code set so numeric codes match the
        # string dates stored in tp_date_ranges; add '' for blanks.
        missing_values = {str(code) for code in self.utils.missing_code_set}
        missing_values.add('')
        current_observations = []
        for form, variable in self._date_fields:
            # Defense against a stale class-level field cache in a long-lived
            # process or test harness: excluded forms never reach the helper.
            if (is_date_report_excluded_form(form)
                    or is_date_report_excluded_variable(variable)):
                continue
            if not hasattr(row, variable):
                continue
            # Do not compare a populated date that belongs to a form whose
            # explicit missing-data button is selected.  Prior-timepoint dates
            # receive the same treatment while their network extrema are built
            # in MultiTPDataCollector.  The pure helper deliberately does not
            # infer missingness from otherwise sparse forms: only an explicit
            # form marker (or PRESCIENT completion status 3/4) qualifies.
            form_info = self.important_form_vars.get(form, {})
            if is_form_marked_missing(row, form_info, self.network):
                continue
            current_observations.append({
                'date': getattr(row, variable),
                'form': form,
                'variable': variable,
                'network': self.network,
            })

        results = find_backward_visit_dates(
            subject_tp_dates, self.tp_list, self.timepoint,
            missing_values=missing_values, network=self.network,
            current_observations=current_observations)

        for result in results:
            # Final defense for old network-extrema dependencies and a stale
            # class field cache: neither side of an emitted flag may identify
            # an excluded form or one of its stable date fields.
            if (is_date_report_excluded_form(result.get('current_form', ''))
                    or is_date_report_excluded_form(
                        result.get('prev_form', ''))
                    or is_date_report_excluded_variable(
                        result.get('current_variable', ''))
                    or is_date_report_excluded_variable(
                        result.get('prev_variable', ''))):
                continue
            # Name both source fields. Fall back to the configured form mapping
            # for v1 dependencies that do not retain variable names.
            cur_form = (result['current_form'] or result['prev_form']
                        or 'unknown_form')
            cur_var = (result.get('current_variable')
                       or self._interview_date_var(result['current_form'])
                       or cur_form)
            prev_var = (result.get('prev_variable')
                        or self._interview_date_var(result['prev_form'])
                        or result['prev_form'] or 'visit date')

            error_message = (
                f"Visit date at {self.timepoint} ({result['current_date']}) is "
                f"{result['days_apart']} day(s) before {prev_var} "
                f"({result['prev_date']}) at the earlier timepoint "
                f"{result['prev_timepoint']}.")

            forms = [form for form in dict.fromkeys(
                [cur_form, result['prev_form']]) if form]
            variables = [value for value in dict.fromkeys(
                [cur_var, prev_var]) if value] or [cur_form]
            output_changes = {
                'reports': [self.DATE_REPORT],
                'affected_timepoints': [
                    result['prev_timepoint'], self.timepoint],
                'check_id': (
                    f"date_backward_visit::{cur_form}::{cur_var}"),
                'date_gap_days': result['days_apart'],
                'nda_excluder': False,
            }
            error_output = self.create_row_output(
                row, forms, variables, error_message, output_changes)
            self.final_output_list.append(error_output)

    def _interview_date_var(self, form):
        """Interview-date variable name for a form, or '' if unknown."""
        if form and form in self.important_form_vars:
            return self.important_form_vars[form].get('interview_date_var', '')
        return ''
