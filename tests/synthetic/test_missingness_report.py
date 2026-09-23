import json
import os
import sys
from types import SimpleNamespace

import pandas as pd

_HERE = os.path.dirname(os.path.realpath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _qc_harness import install_shim, project_root  # noqa: E402


ROOT = install_shim()

from qc_forms.qc_types.general_checks import GeneralChecks  # noqa: E402
from qc_forms.form_check import FormCheck  # noqa: E402
from process_variables.define_important_variables import (  # noqa: E402
    DefineEssentialFormVars,
)


class _Utils:
    missing_code_list = ['-3', '-9', '-99', '999']
    missing_code_set = set(missing_code_list)

    @staticmethod
    def all_dtype(values):
        output = []
        for value in values:
            output.extend([value, float(value), str(value), str(float(value))])
        return output

    @staticmethod
    def can_be_float(value):
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def check_if_after_date(row, form, date_var):
        return str(getattr(row, date_var, '')) >= '2024-12-25'


def _form_info(form):
    return {
        'form': form,
        'missing_var': f'{form}_missing',
        'completion_var': f'{form}_complete',
        'interview_date_var': f'{form}_date',
        'non_branch_logic_vars': [f'{form}_date'],
    }


def _checker(forms, scheduled, *, network='PRONET', timepoint='month1',
             excluded=(), domains=None, vars_added_later=None):
    checker = object.__new__(GeneralChecks)
    checker.network = network
    checker.timepoint = timepoint
    checker.tp_list = ['screening', 'baseline', 'month1', 'month2']
    checker.subject_info = {
        'S1': {
            'cohort': 'CHR',
            'inclusion_status': 'included',
            'visit_status': 'month2',
            'age': 17,
            'axivity_opt': 1,
            'mindlamp_opt': 1,
        }
    }
    checker.forms_per_tp = {'chr': {timepoint: list(scheduled)}, 'hc': {}}
    checker.general_check_vars = {
        'excluded_forms': {network: list(excluded)}}
    checker.important_form_vars = {
        form: _form_info(form) for form in forms}
    checker.vars_added_later = vars_added_later or {}
    checker.missingness_domain_forms = domains or {}
    checker.utils = _Utils()
    checker.final_output_list = []

    def _create(row, affected_forms, variables, message, changes):
        return {
            'displayed_form': affected_forms[0],
            'affected_forms': affected_forms,
            'affected_variables': variables,
            'error_message': message,
            **changes,
        }

    checker.create_row_output = _create
    return checker


def _row(forms, **updates):
    values = {
        'subjectid': 'S1',
        'visit_status': 'month2',
        'chrmiss_time': '',
        'chrmiss_domain': '',
        'chrmiss_withdrawn': '',
        'chrmiss_discon': '',
    }
    for form in forms:
        values[f'{form}_missing'] = ''
        values[f'{form}_complete'] = ''
        values[f'{form}_complete_rpms'] = ''
        values[f'{form}_date'] = '2025-01-01'
    values.update(updates)
    return SimpleNamespace(**values)


def _ids(checker):
    return [output['check_id'] for output in checker.final_output_list]


def test_whole_timepoint_checks_only_scheduled_forms():
    checker = _checker(
        ['scheduled', 'unscheduled'], scheduled=['scheduled'])
    checker.missing_data_form_consistency_check(_row(
        ['scheduled', 'unscheduled'], chrmiss_time=1))
    assert _ids(checker) == ['missingness_whole_tp_unchecked::scheduled']


def test_excluded_form_is_not_required():
    checker = _checker(['scheduled'], scheduled=['scheduled'],
                       excluded=['scheduled'])
    checker.missing_data_form_consistency_check(
        _row(['scheduled'], chrmiss_time=1))
    assert checker.final_output_list == []


def test_opted_out_digital_form_is_not_required():
    form = 'digital_biomarkers_axivity_checkin'
    checker = _checker([form], scheduled=[form])
    checker.subject_info['S1']['axivity_opt'] = 0
    checker.missing_data_form_consistency_check(
        _row([form], chrmiss_time=1))
    assert checker.final_output_list == []


def test_prescient_missing_completion_state_satisfies_missingness():
    checker = _checker(
        ['scheduled'], scheduled=['scheduled'], network='PRESCIENT')
    checker.missing_data_form_consistency_check(_row(
        ['scheduled'], chrmiss_time=1, scheduled_complete_rpms=3))
    assert checker.final_output_list == []


def test_button_is_not_required_before_it_was_added():
    checker = _checker(
        ['scheduled'], scheduled=['scheduled'],
        vars_added_later={
            'scheduled': {'scheduled_missing': '2024-12-25'}})
    checker.missing_data_form_consistency_check(_row(
        ['scheduled'], chrmiss_time=1, scheduled_date='2023-01-01'))
    assert checker.final_output_list == []


def test_selected_domain_targets_only_mapped_scheduled_forms():
    domains = {
        '1': {'label': 'Clinical measures', 'forms': ['clinical']},
        '2': {'label': 'EEG', 'forms': ['eeg']},
    }
    checker = _checker(
        ['clinical', 'eeg'], scheduled=['clinical', 'eeg'], domains=domains)
    checker.missing_data_form_consistency_check(_row(
        ['clinical', 'eeg'], chrmiss_domain=1,
        chrmiss_domain_type___2=1))
    assert _ids(checker) == ['missingness_domain_unchecked::2::eeg']


def test_selected_domains_use_union_without_duplicate_or_unrelated_flags():
    domains = {
        '1': {'label': 'Clinical measures', 'forms': ['clinical']},
        '2': {'label': 'EEG', 'forms': ['eeg']},
    }
    checker = _checker(
        ['clinical', 'eeg', 'admin'],
        scheduled=['clinical', 'eeg', 'admin'], domains=domains)
    checker.missing_data_form_consistency_check(_row(
        ['clinical', 'eeg', 'admin'], chrmiss_domain=1,
        chrmiss_domain_type___1=1, chrmiss_domain_type___2=1,
        clinical_missing=1))
    assert _ids(checker) == ['missingness_domain_unchecked::2::eeg']


def test_selected_domain_with_no_scheduled_forms_is_not_flagged():
    checker = _checker(['clinical'], scheduled=['clinical'], domains={
        '2': {'label': 'EEG', 'forms': ['eeg']}})
    checker.missing_data_form_consistency_check(_row(
        ['clinical'], chrmiss_domain=1, chrmiss_domain_type___2=1))
    assert checker.final_output_list == []


def test_empty_selected_domain_does_not_hide_actionable_domain_flag():
    domains = {
        '1': {'label': 'Clinical measures', 'forms': ['clinical']},
        '2': {'label': 'EEG', 'forms': ['eeg']},
    }
    checker = _checker(
        ['clinical'], scheduled=['clinical'], domains=domains)
    checker.missing_data_form_consistency_check(_row(
        ['clinical'], chrmiss_domain=1,
        chrmiss_domain_type___1=1, chrmiss_domain_type___2=1))
    assert _ids(checker) == [
        'missingness_domain_unchecked::1::clinical']


def test_selected_domain_with_all_target_forms_missing_is_clean():
    checker = _checker(['eeg'], scheduled=['eeg'], domains={
        '2': {'label': 'EEG', 'forms': ['eeg']}})
    checker.missing_data_form_consistency_check(_row(
        ['eeg'], chrmiss_domain=1, chrmiss_domain_type___2=1,
        eeg_missing=1))
    assert checker.final_output_list == []


def test_unrelated_missing_button_does_not_hide_domain_violation():
    domains = {
        '1': {'label': 'Clinical measures', 'forms': ['clinical']},
        '2': {'label': 'EEG', 'forms': ['eeg']},
    }
    checker = _checker(
        ['clinical', 'eeg'], scheduled=['clinical', 'eeg'], domains=domains)
    checker.missing_data_form_consistency_check(_row(
        ['clinical', 'eeg'], chrmiss_domain=1,
        chrmiss_domain_type___2=1, clinical_missing=1))
    assert set(_ids(checker)) == {
        'missingness_form_outside_selected_domain::clinical',
        'missingness_domain_unchecked::2::eeg',
    }


def test_domain_trigger_without_domain_selection_is_one_summary():
    checker = _checker(['clinical'], scheduled=['clinical'], domains={
        '1': {'label': 'Clinical measures', 'forms': ['clinical']}})
    checker.missing_data_form_consistency_check(
        _row(['clinical'], chrmiss_domain=1))
    assert _ids(checker) == ['missingness_domain_type_not_selected']


def test_domain_checkbox_without_trigger_is_flagged():
    checker = _checker(['clinical'], scheduled=['clinical'], domains={
        '1': {'label': 'Clinical measures', 'forms': ['clinical']}})
    checker.missing_data_form_consistency_check(_row(
        ['clinical'], visit_status='month1', chrmiss_domain_type___1=1))
    assert _ids(checker) == ['missingness_domain_selected_without_trigger']


def test_completed_form_gets_one_trigger_summary_not_missing_button_flag():
    checker = _checker(['clinical'], scheduled=['clinical'])
    checker.missing_data_form_consistency_check(_row(
        ['clinical'], chrmiss_time=1, clinical_complete=2))
    assert _ids(checker) == ['missingness_whole_tp_has_completed_forms']


def test_terminal_visit_status_does_not_raise_and_flags_remain_visible():
    checker = _checker(['clinical'], scheduled=['clinical'])
    checker.missing_data_form_consistency_check(_row(
        ['clinical'], visit_status='removed', chrmiss_time=1))
    assert _ids(checker) == ['missingness_whole_tp_unchecked::clinical']
    assert checker.final_output_list[0]['withdrawn_enabled'] is True
    assert checker.final_output_list[0]['nda_excluder'] is False


def test_real_output_builder_keeps_removed_missingness_flag_visible():
    checker = _checker(['clinical'], scheduled=['clinical'])
    checker.grouped_vars = {'var_translations': {}}
    checker.priority_forms = []
    checker.priority_timepoints = []
    checker.config_info = {
        'withdrawn_enabled': False,
        'recruited_only': False,
    }
    row = _row(
        ['clinical'], visit_status='removed',
        visit_status_string='removed',
        recruitment_status_v2='not_recruited')
    output = FormCheck.create_row_output(
        checker, row, ['clinical', 'missing_data'],
        ['clinical_missing'], 'test',
        checker._missingness_output_changes('missingness_test'))
    assert output['reports'] == 'Missingness Report'
    assert output['withdrawn_enabled'] is True
    assert output['nda_excluder'] is False


def test_checked_in_domain_map_covers_each_scheduled_missing_form_once():
    with open(os.path.join(os.path.dirname(ROOT), 'dependencies', 'important_form_vars.json'),
              encoding='utf-8') as handle:
        important = json.load(handle)
    with open(os.path.join(os.path.dirname(ROOT), 'dependencies', 'forms_per_timepoint.json'),
              encoding='utf-8') as handle:
        schedule = json.load(handle)
    with open(os.path.join(os.path.dirname(ROOT), 'dependencies',
                           'missingness_domain_forms.json'),
              encoding='utf-8') as handle:
        mapping = json.load(handle)

    scheduled_missing_forms = {
        form
        for timepoints in schedule.values()
        for forms in timepoints.values()
        for form in forms
        if important.get(form, {}).get('missing_var', '')
    }
    all_missing_forms = {
        form for form, info in important.items()
        if info.get('missing_var', '')}
    mapped = [
        form for info in mapping.values() for form in info.get('forms', [])]
    assert {code: mapping[code]['label'] for code in map(str, range(1, 8))} == {
        '1': 'Clinical measures',
        '2': 'EEG',
        '3': 'Neuroimaging',
        '4': 'Cognition',
        '5': 'Genetics and Fluid Biomarkers',
        '6': 'Digital Biomarkers',
        '7': 'Speech Sampling',
    }
    assert len(mapped) == len(set(mapped))
    # chrfigs_missing left the data dictionary (verified absent from
    # data_dict_new.csv, 2026-08-25), so the regenerated
    # important_form_vars.json now gives FIGS an empty missing_var while
    # the domain map still lists the form. Keep FIGS mapped (harmless —
    # missingness checks skip forms without a missing_var) but exempt it
    # from the exact-cover assertion.
    figs_form = 'family_interview_for_genetic_studies_figs'
    assert set(mapped) - {figs_form} == all_missing_forms
    assert scheduled_missing_forms <= set(mapped)
    assert important[figs_form]['missing_var'] == ''


def test_missing_button_discovery_is_case_insensitive():
    definition = object.__new__(DefineEssentialFormVars)
    definition.data_dictionary_df = pd.DataFrame([{
        'Variable / Field Name': 'chrfigs_missing',
        'Choices, Calculations, OR Slider Labels':
            '1, Please click if this form is missing all of its data',
    }])
    assert definition.create_list_from_df(
        'Choices, Calculations, OR Slider Labels',
        ['click if this form is missing all of its data']) == [
            'chrfigs_missing']
