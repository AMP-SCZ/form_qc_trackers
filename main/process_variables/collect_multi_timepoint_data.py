import pandas as pd
import os
import sys
import json
from collections.abc import Mapping
from datetime import datetime
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.qc_types.date_check_logic import (
    is_form_marked_missing,
    is_date_report_excluded_form,
    is_date_report_excluded_variable,
)


def _normalize_schedule_key(value):
    """Return the case-insensitive key used by schedule lookups."""
    if value is None:
        return ''
    return str(value).strip().casefold()


def _normalize_forms_per_timepoint(forms_per_timepoint):
    """Validate and canonicalize a forms-per-timepoint dependency.

    Cohort and timepoint keys are configuration identifiers and are therefore
    matched case-insensitively.  Form names remain case-sensitive because they
    are REDCap instrument identifiers.  Aliases that collapse to the same key
    are accepted only when their schedules agree; conflicting definitions
    fail loudly instead of silently disabling scheduled-form consumers.
    """
    if not isinstance(forms_per_timepoint, Mapping):
        raise TypeError('forms_per_timepoint must be a mapping')

    normalized = {}
    cohort_sources = {}
    for raw_cohort, raw_schedule in forms_per_timepoint.items():
        cohort_key = _normalize_schedule_key(raw_cohort)
        if not cohort_key:
            raise ValueError(
                'forms_per_timepoint contains a blank cohort key')
        if not isinstance(raw_schedule, Mapping):
            raise TypeError(
                'forms_per_timepoint cohort '
                f'{raw_cohort!r} must map timepoints to form lists')

        normalized_schedule = {}
        timepoint_sources = {}
        for raw_timepoint, raw_forms in raw_schedule.items():
            timepoint_key = _normalize_schedule_key(raw_timepoint)
            if not timepoint_key:
                raise ValueError(
                    'forms_per_timepoint cohort '
                    f'{raw_cohort!r} contains a blank timepoint key')
            if not isinstance(raw_forms, list):
                raise TypeError(
                    'forms_per_timepoint entry '
                    f'{raw_cohort!r}/{raw_timepoint!r} must be a list')

            forms = []
            seen_forms = set()
            for raw_form in raw_forms:
                if not isinstance(raw_form, str):
                    raise TypeError(
                        'forms_per_timepoint entry '
                        f'{raw_cohort!r}/{raw_timepoint!r} contains a '
                        'non-string form name')
                form = raw_form.strip()
                if not form:
                    raise ValueError(
                        'forms_per_timepoint entry '
                        f'{raw_cohort!r}/{raw_timepoint!r} contains a blank '
                        'form name')
                if form not in seen_forms:
                    forms.append(form)
                    seen_forms.add(form)

            if timepoint_key in normalized_schedule:
                if normalized_schedule[timepoint_key] != forms:
                    raise ValueError(
                        'Conflicting forms_per_timepoint definitions for '
                        f'timepoints {timepoint_sources[timepoint_key]!r} and '
                        f'{raw_timepoint!r} in cohort {raw_cohort!r}')
                continue
            normalized_schedule[timepoint_key] = forms
            timepoint_sources[timepoint_key] = raw_timepoint

        if cohort_key in normalized:
            if normalized[cohort_key] != normalized_schedule:
                raise ValueError(
                    'Conflicting forms_per_timepoint definitions for cohorts '
                    f'{cohort_sources[cohort_key]!r} and {raw_cohort!r}')
            continue
        normalized[cohort_key] = normalized_schedule
        cohort_sources[cohort_key] = raw_cohort

    return normalized


def _normalize_interview_date(value, missing_values):
    """Return (YYYY-MM-DD, datetime) for a usable interview-date value."""
    raw = str(value).strip()
    date_str = raw.split(' ')[0].split('T')[0]
    missing_set = (missing_values if isinstance(missing_values, set)
                   else {str(item).strip() for item in missing_values})
    if raw in missing_set or date_str in missing_set:
        return None
    try:
        return date_str, datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def _include_legacy_interview_date_field(form, variable):
    """Retain the old extrema inputs while hardening Date Report separately.

    The six target forms are still used by unrelated Pharm/SOP consumers of
    the top-level extrema. Their stable variables are therefore collected,
    but the missing-data instrument date remains omitted as before.
    """
    return bool(variable) and (
        not is_date_report_excluded_variable(variable)
        or is_date_report_excluded_form(form))


def _update_date_range(date_range, date_str, date_value, form, variable):
    """Update deterministic earliest/latest extrema in ``date_range``."""
    candidate_key = (date_value, form, variable)

    if 'earliest' not in date_range:
        date_range.update({
            'earliest': date_str,
            'latest': date_str,
            'earliest_form': form,
            'latest_form': form,
            'earliest_variable': variable,
            'latest_variable': variable,
        })
        return

    earliest_key = (
        datetime.strptime(date_range['earliest'], "%Y-%m-%d"),
        date_range.get('earliest_form', ''),
        date_range.get('earliest_variable', ''))
    latest_key = (
        datetime.strptime(date_range['latest'], "%Y-%m-%d"),
        date_range.get('latest_form', ''),
        date_range.get('latest_variable', ''))

    if candidate_key < earliest_key:
        date_range['earliest'] = date_str
        date_range['earliest_form'] = form
        date_range['earliest_variable'] = variable
    if candidate_key > latest_key:
        date_range['latest'] = date_str
        date_range['latest_form'] = form
        date_range['latest_variable'] = variable


def _update_variable_date(date_range, date_str, date_value, form, variable):
    """Retain the latest observation for one exact interview-date field.

    Date Report comparisons are field-specific. The legacy earliest/latest
    extrema are not sufficient because a field can be neither extremum at a
    timepoint. Keep one compact maximum per variable: comparing a later value
    with this maximum is equivalent to comparing it with every duplicate value
    observed for that subject/timepoint/network/field.
    """
    variable_dates = date_range.setdefault('variable_dates', {})
    existing = variable_dates.get(variable)
    candidate_key = (date_value, form)

    if isinstance(existing, dict):
        existing_date = _normalize_interview_date(
            existing.get('date', ''), set())
        if existing_date is not None:
            existing_key = (existing_date[1], existing.get('form', ''))
            if candidate_key <= existing_key:
                return

    variable_dates[variable] = {
        'date': date_str,
        'form': form,
    }


class MultiTPDataCollector():
    """
    Class to collect data
    from multiple timepoints 
    """
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        self.earliest_latest_dates_per_tp = {}
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.combined_csv_path = self.config_info['paths']['combined_csv_path']
        self.depend_path = self.config_info["paths"]["dependencies_path"]
        self.grouped_vars = self.utils.load_dependency_json(f"grouped_variables.json")
        self.forms_per_var = self.grouped_vars['var_forms']
        self.earliest_date_per_var = {}
        self.important_form_vars = self.utils.load_dependency_json(
        'important_form_vars.json')
        self.interview_date_fields = [
            (form, info.get('interview_date_var', ''))
            for form, info in self.important_form_vars.items()
            if _include_legacy_interview_date_field(
                form, info.get('interview_date_var', ''))
        ]
        self.forms_per_tp = _normalize_forms_per_timepoint(
            self.utils.load_dependency_json('forms_per_timepoint.json'))
        self.subject_info = self.utils.load_dependency_json(
        'subject_info.json')
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.multitp_output = pd.DataFrame()
        self.variable_type_distributions = {}
        self.multi_tp_vars = [
        'chrpps_fage','chrfigs_father_age',
        'chrpps_mage','chrfigs_mother_age',
        'chrblood_wb1id','chrblood_wb2id',
        'chrblood_se3id', 'visit_status','subjectid'
        ]
        for variable_grp, var_list in self.grouped_vars['blood_vars'].items():
            self.multi_tp_vars.extend(var_list)
        print(self.multi_tp_vars)

        self.loop_csvs()

    def __call__(self):
        self.loop_csvs()

    def loop_csvs(self):
        """
        loops through each combined csv
        to collect multi timepoint data
        """
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        for network in ['PRESCIENT','PRONET']:
            multi_tp_df = pd.DataFrame()
            for tp in tp_list:
                combined_df =pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}_{network.replace("PRONET","ProNET")}-day1to1.csv'),
                keep_default_na = False)
                self.collect_earliest_date(combined_df)
                self.collect_earliest_latest_dates(combined_df, tp, network)
                self.collect_variable_type_distributions(combined_df)
                modified_df = self.utils.append_suffix_to_cols(combined_df,
                tp, self.multi_tp_vars)
                print(modified_df.columns)
                if multi_tp_df.empty:
                    multi_tp_df = modified_df
                else:
                    multi_tp_df = multi_tp_df.merge(modified_df,
                    on = 'subjectid', how = 'outer')

            multi_tp_df.to_csv(
            f'{self.depend_path}multi_tp_{network}_combined.csv',
            index = False)

        # Hoisted out of the per-tp loop: these dicts accumulate across
        # every (network, tp) pass, so writing them inside the loop wrote
        # the same growing dict ~16 times per run.
        self.utils.save_dependency_json(self.earliest_latest_dates_per_tp,
        'earliest_latest_dates_per_tp.json')
        self.utils.save_dependency_json(self.variable_type_distributions,
        'variable_type_distributions.json')
        # Also hoisted: collect_earliest_date accumulates into
        # self.earliest_date_per_var across every (network, tp) call, so
        # sorting + atomically rewriting earliest_dates_per_var.json at the
        # end of each call meant ~31 redundant sort+write cycles per run.
        # Only the final, fully-accumulated state matters.
        sorted_date_dict = dict(
            sorted(self.earliest_date_per_var.items(),
            key=lambda item: datetime.strptime(item[1], "%Y-%m-%d"), reverse=True)
        )
        self.utils.save_dependency_json(sorted_date_dict,
        'earliest_dates_per_var.json')

    def collect_earliest_latest_dates(self,
        combined_df : pd.DataFrame, tp: str,
        network : str
    ):
        """
        Collect every eligible configured interview date present in this
        timepoint's CSV. Retain both legacy per-timepoint extrema and a compact
        per-variable map for exact-field Date Report comparisons. Dates on
        explicitly missing-marked forms are ineligible.

        Legacy subject/timepoint extrema remain available to existing
        consumers. ``network_extrema`` prevents PRONET/PRESCIENT records from
        contaminating the Date Report when subject identifiers overlap.

        Parameters
        --------
        combined_df : pd.DataFrame
            current timepoint's dataframe
        tp : str
            current tp
        """
        date_fields = getattr(self, 'interview_date_fields', None)
        if date_fields is None:
            date_fields = [
                (form, info.get('interview_date_var', ''))
                for form, info in self.important_form_vars.items()
                if _include_legacy_interview_date_field(
                    form, info.get('interview_date_var', ''))
            ]
        else:
            # Test doubles and long-lived callers may supply a cached field
            # list assembled before the exclusion was introduced.
            date_fields = [
                (form, variable) for form, variable in date_fields
                if _include_legacy_interview_date_field(form, variable)]
        # Restrict once per dataframe rather than checking all configured
        # fields for every participant row.
        present_date_fields = [
            (form, variable) for form, variable in date_fields
            if variable in combined_df.columns]
        # Normalize at the use boundary as well as during __init__.  This
        # keeps test doubles and long-lived callers that replace the mapping
        # from reintroducing case-sensitive schedule lookups.
        forms_per_tp = _normalize_forms_per_timepoint(self.forms_per_tp)
        self.forms_per_tp = forms_per_tp
        missing_values = {
            str(item).strip() for item in self.utils.missing_code_list}
        missing_values.add('')

        for row in combined_df.itertuples():
            subject = row.subjectid
            if subject not in self.subject_info:
                continue
            cohort_key = _normalize_schedule_key(
                self.subject_info[subject].get('cohort', ''))
            timepoint_key = _normalize_schedule_key(tp)
            scheduled_forms = set(
                forms_per_tp.get(cohort_key, {}).get(timepoint_key, []))
            for form, interview_date_var in present_date_fields:
                parsed = _normalize_interview_date(
                    getattr(row, interview_date_var), missing_values)
                if parsed is None:
                    continue
                # A populated interview date does not make a deliberately
                # missing form a real visit observation.  Exclude it before
                # updating either range: ``network_extrema`` feeds the Date
                # Report's prior-timepoint comparison, while the top-level
                # extrema remain the legacy Pharm/SOP contract.
                #
                form_info = self.important_form_vars.get(form, {})
                if is_form_marked_missing(row, form_info, network):
                    continue
                int_date_str, int_date_datetime = parsed

                tp_dates = None
                # The network-specific range feeds only the Date Report.
                # Excluded administrative/non-target forms must not become
                # either side of those comparisons.  Keep processing below,
                # however, because the legacy top-level range is consumed by
                # unrelated Pharm/SOP logic and retains its prior contract.
                if (not is_date_report_excluded_form(form)
                        and not is_date_report_excluded_variable(
                            interview_date_var)):
                    subject_dates = (
                        self.earliest_latest_dates_per_tp.setdefault(
                            subject, {}))
                    tp_dates = subject_dates.setdefault(tp, {})
                    network_dates = tp_dates.setdefault(
                        'network_extrema', {}).setdefault(network, {})
                    _update_variable_date(
                        network_dates, int_date_str, int_date_datetime,
                        form, interview_date_var)
                    _update_date_range(
                        network_dates, int_date_str, int_date_datetime,
                        form, interview_date_var)

                # Preserve the legacy extrema contract for Pharm/SOP and
                # exploratory consumers: only scheduled forms contribute to
                # the top-level range.  The explicit-missing gate above now
                # applies consistently to both this range and Date Report's
                # network-specific range.
                if form not in scheduled_forms:
                    continue
                # Retain the legacy top-level range semantics, including the
                # inferred-missing behavior for forms that have no explicit
                # button.  Date Report's network range above intentionally
                # uses only the explicit marker gate.
                if self.utils.check_if_missing(row, form, tp, network):
                    continue
                if tp_dates is None:
                    subject_dates = (
                        self.earliest_latest_dates_per_tp.setdefault(
                            subject, {}))
                    tp_dates = subject_dates.setdefault(tp, {})
                _update_date_range(
                    tp_dates, int_date_str, int_date_datetime,
                    form, interview_date_var)

    def collect_earliest_date(self, 
        combined_df : pd.DataFrame
    ):
        """
        Collects earliest date a
        variable was used if there
        is an interview date
        """
        # Restructured (roadmap #8) to drop the O(rows x columns) per-cell
        # loop. The original read each form's interview-date value once per
        # (row, column-of-that-form); a column's OWN value is never used, so
        # earliest_date_per_var[col] is purely min(valid date >= 2022) of the
        # col's form interview-date column. Group cols by their date var,
        # scan each date column ONCE, and apply the same setdefault/min merge.
        # Byte-identical to the original incl. cross-call accumulation
        # (min is commutative); verified by
        # tests/synthetic/test_collect_multitp_parity.py.
        all_cols = combined_df.columns
        # Duplicate column names make combined_df[date_var] a DataFrame and
        # diverge from the original getattr-on-itertuples semantics; fall back
        # to the exact original loop for that (non-production) case.
        if len(set(all_cols)) != len(all_cols):
            self._collect_earliest_date_rowwise(combined_df)
            return
        cutoff = datetime.strptime("2022-01-01", "%Y-%m-%d")
        excluded = self.utils.missing_code_list + ['']
        cols_by_date_var = {}
        for col in all_cols:
            if col in self.forms_per_var.keys():
                form = self.forms_per_var[col]
                if form in self.important_form_vars.keys():
                    interview_date_var = self.important_form_vars[form]['interview_date_var']
                    if interview_date_var != '' and interview_date_var in all_cols:
                        cols_by_date_var.setdefault(interview_date_var, []).append(col)
        for interview_date_var, cols in cols_by_date_var.items():
            min_str = None
            min_dt = None
            for interview_date_val in combined_df[interview_date_var]:
                if (interview_date_val not in excluded and
                self.utils.check_if_val_date_format(str(interview_date_val))):
                    dtv = datetime.strptime(str(interview_date_val), "%Y-%m-%d")
                    if dtv < cutoff:
                        continue
                    if min_dt is None or dtv < min_dt:
                        min_dt = dtv
                        min_str = str(interview_date_val)
            if min_str is None:
                continue
            for col in cols:
                self.earliest_date_per_var.setdefault(col, min_str)
                if min_dt < datetime.strptime(self.earliest_date_per_var[col], "%Y-%m-%d"):
                    self.earliest_date_per_var[col] = min_str

    def _collect_earliest_date_rowwise(self, combined_df):
        # Exact original per-(row, col) implementation; retained as the
        # fallback for frames with duplicate column names.
        all_cols = combined_df.columns
        for row in combined_df.itertuples():
            for col in all_cols:
                if col in self.forms_per_var.keys():
                    form = self.forms_per_var[col]
                    if form in self.important_form_vars.keys():
                        interview_date_var = self.important_form_vars[form]['interview_date_var']
                        if interview_date_var != '' and interview_date_var in all_cols:
                            interview_date_val = getattr(row, interview_date_var)
                            if (interview_date_val not in (self.utils.missing_code_list + ['']) and
                            self.utils.check_if_val_date_format(str(interview_date_val))):
                                datetime_format_val = datetime.strptime(str(interview_date_val), "%Y-%m-%d")
                                if datetime_format_val < datetime.strptime("2022-01-01", "%Y-%m-%d"):
                                    continue
                                self.earliest_date_per_var.setdefault(col, str(interview_date_val))
                                if (datetime_format_val < datetime.strptime(
                                self.earliest_date_per_var[col], "%Y-%m-%d") ):
                                    self.earliest_date_per_var[col] = str(interview_date_val)

    def search_timepoint_dates(self,
        row : tuple,
        forms : list
    ):
        date_list = []
        
    def loop_networks(self):
        for network in ['PRONET', 'PRESCIENT']:
            self.collect_blood_duplicates(network)
            
    def collect_blood_duplicates(self, 
        network : str
    ):
        """
        creates dataframe of 
        blood variables 
        and calls functions to 
        check for duplicates

        parameters
        -------------------
        network : str
            current network (Pronet or Prescient) 
        """
        baseln_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'baseline_{network.replace("PRONET","ProNET")}-day1to1.csv'),
                keep_default_na = False)
        preserved_cols = [col for col in baseln_df.columns if 'chrblood' in col]
        preserved_cols.append('subjectid')
        baseln_df = baseln_df[preserved_cols] 

        month2_df = pd.read_csv(
                (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'month_2_{network.replace("PRONET","ProNET")}-day1to1.csv'),
                keep_default_na = False)
        preserved_cols = [col for col in preserved_cols if col in month2_df.columns]
        month2_df = month2_df[preserved_cols]

        merged_blood_df = pd.merge(baseln_df, month2_df,
        on='subjectid', how = 'outer', suffixes=('_baseln','_month2'))

        self.check_position_duplicates(merged_blood_df)

    def check_id_duplicates(self, 
        merged_blood_df : pd.DataFrame
    ):
        """
        collects duplicate IDs

        Parameters 
        -------------
        merged_blood_df : pd.DataFrame
            Dataframe of blood variables
        """

        id_vars = self.grouped_vars['blood_vars']['id_variables']
        merged_id_vars = []
        for var in id_vars:
            for suffix in ['_baseln','_month2']:
                merged_id_vars.append(var + suffix)
        merged_id_vars = [var for var in merged_id_vars
        if var in merged_blood_df.columns]

        preserved_cols = merged_id_vars + ['subjectid']
        id_df = merged_blood_df[preserved_cols]
        excluded_val_list = (self.utils.missing_code_list + [''])

        id_df = id_df.melt(id_vars='subjectid',
        value_vars=merged_id_vars, var_name='id_name', value_name='id_val')
        id_df = id_df[~id_df['id_val'].isin(excluded_val_list)]
        id_df = id_df[id_df.duplicated(subset=['id_val'], keep=False) & 
        (id_df.duplicated(subset=['id_val', 'subjectid'], keep=False) == False)]

    def check_position_duplicates(self, 
        merged_blood_df : pd.DataFrame
    ):  
        """
        collects duplicate
        blood vial positions

        Parameters 
        -------------
        merged_blood_df : pd.DataFrame
            Dataframe of blood variables
        """

        pos_vars = self.grouped_vars['blood_vars']['position_variables']
        merged_pos_vars = []
        merged_barcode_vars = []
        for suffix in ['_baseln','_month2']:
            merged_barcode_vars.append('chrblood_rack_barcode' + suffix)
            for var in pos_vars:
                merged_pos_vars.append(var + suffix)
        
        preserved_cols = merged_pos_vars + ['subjectid'] + merged_barcode_vars
        preserved_cols = [var for var in preserved_cols
        if var in merged_blood_df.columns]

        pos_df = merged_blood_df[preserved_cols]

        merged_barc_pos = []
        excluded_val_list = (self.utils.missing_code_list + [''])

        for barcode_var in merged_barcode_vars:
            for pos_var in merged_pos_vars:
                diff_tp = False
                for suffix in ['_baseln','_month2']:
                    if suffix in pos_var and suffix not in barcode_var:
                        diff_tp = True
                if diff_tp:
                    continue      
                pos_df.loc[(~pos_df[
                barcode_var].isin(excluded_val_list)) & (
                ~pos_df[pos_var].isin(
                excluded_val_list)), barcode_var + '_' + pos_var] = pos_df[
                barcode_var].astype(str) + '_' + pos_df[pos_var].astype(str)
                merged_barc_pos.append(barcode_var + '_' + pos_var)

        pos_df = pos_df[merged_barc_pos + ['subjectid']]
        pos_df = pos_df.melt(id_vars='subjectid', value_vars=merged_barc_pos,
        var_name='id_name', value_name='barc_pos_val')
        pos_df = pos_df.fillna('')
        pos_df = pos_df[~pos_df['barc_pos_val'].isin(excluded_val_list)]
        pos_df = pos_df[pos_df.duplicated(subset=['barc_pos_val'], keep=False) & 
        (pos_df.duplicated(subset=['barc_pos_val', 'subjectid'], keep=False) == False)]
        pos_df.to_csv('duplicate_blood.csv',index = False)

    def collect_variable_type_distributions(self, 
        combined_df : pd.DataFrame
    ):
        """
        Collects distribution of different value
        categories for each variable

        Parameters
        -------------
        combined_df : pd.DataFrame
            current combined dataframe
            being looped through
        """
        # Restructured (roadmap #8): iterate column-wise instead of
        # itertuples(), which on a wide frame materializes a named tuple of
        # thousands of fields per row. Classification logic + elif precedence
        # are unchanged and setdefault stays inside the value loop (so a
        # 0-row frame creates no entries), making output byte-identical to
        # the original — verified by tests/synthetic/test_collect_multitp_parity.py.
        # Duplicate column names mangle under itertuples' getattr; fall back
        # to the exact original loop in that (non-production) case.
        cols = list(combined_df.columns)
        if len(set(cols)) != len(cols):
            self._collect_variable_type_distributions_rowwise(combined_df)
            return
        missing_code_set = self.utils.missing_code_set
        dist = self.variable_type_distributions
        for var in cols:
            for var_val in combined_df[var]:
                dist.setdefault(var, {'missing_code':0,
                'num':0,'date':0,'blank':0,'string':0, 'other':0})
                if var_val in missing_code_set:
                    dist[var]['missing_code'] +=1
                elif self.utils.can_be_float(var_val):
                    dist[var]['num'] +=1
                elif self.utils.check_if_val_date_format(
                str(var_val).split(' ')[0]):
                    dist[var]['date'] +=1
                elif var_val == '':
                    dist[var]['blank'] +=1
                elif isinstance(var_val, str):
                    dist[var]['string'] +=1
                else:
                    dist[var]['other'] +=1

    def _collect_variable_type_distributions_rowwise(self, combined_df):
        # Exact original itertuples implementation; fallback for frames with
        # duplicate column names.
        for row in combined_df.itertuples():
            for var in combined_df.columns:
                var_val = getattr(row,var)
                self.variable_type_distributions.setdefault(var, {'missing_code':0,
                'num':0,'date':0,'blank':0,'string':0, 'other':0})
                if var_val in self.utils.missing_code_set:
                    self.variable_type_distributions[var]['missing_code'] +=1
                elif self.utils.can_be_float(var_val):
                    self.variable_type_distributions[var]['num'] +=1
                elif self.utils.check_if_val_date_format(
                str(var_val).split(' ')[0]):
                    self.variable_type_distributions[var]['date'] +=1
                elif var_val == '':
                    self.variable_type_distributions[var]['blank'] +=1
                elif isinstance(var_val, str):
                    self.variable_type_distributions[var]['string'] +=1
                else:
                    self.variable_type_distributions[var]['other'] +=1









                                    
