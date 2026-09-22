import pandas as pd
import pytest

from process_variables.collect_multi_timepoint_data import (
    MultiTPDataCollector,
    _normalize_forms_per_timepoint,
)
from qc_types.date_check_logic import DATE_REPORT_EXCLUDED_FORMS


EXCLUDED_DATE_FIELDS = {
    'past_pharmaceutical_treatment': 'chrpharm_interview_date',
    'psychs_av_recording_run_sheet': 'chrpsychs_av_interview_date',
    'mri_incidental_findings_run_sheet': 'chrif_interview_date',
    'gcp_current_health_status': 'chrgpc_date',
    'cbc_with_differential': 'chrcbc_interview_date',
    'gcp_cbc_with_differential': 'chrcbccs_review_date',
}


def test_forms_per_timepoint_normalization_preserves_lowercase_schedule():
    schedule = {
        'chr': {
            'screening': ['scheduled_form'],
            'floating': [],
        },
        'hc': {'screening': ['scheduled_form']},
    }

    assert _normalize_forms_per_timepoint(schedule) == schedule


def test_forms_per_timepoint_normalization_handles_case_and_whitespace():
    schedule = {
        '  CHR  ': {
            ' Screening ': [' scheduled_form ', 'second_form'],
        },
        ' Hc ': {' MONTH1 ': []},
    }

    assert _normalize_forms_per_timepoint(schedule) == {
        'chr': {'screening': ['scheduled_form', 'second_form']},
        'hc': {'month1': []},
    }


def test_forms_per_timepoint_normalization_accepts_identical_aliases():
    schedule = {
        'CHR': {'Screening': ['scheduled_form']},
        ' chr ': {' screening ': ['scheduled_form']},
    }

    assert _normalize_forms_per_timepoint(schedule) == {
        'chr': {'screening': ['scheduled_form']},
    }


def test_forms_per_timepoint_normalization_rejects_conflicting_aliases():
    schedule = {
        'CHR': {'screening': ['scheduled_form']},
        'chr': {'screening': ['different_form']},
    }

    with pytest.raises(ValueError, match='Conflicting.*cohorts.*CHR.*chr'):
        _normalize_forms_per_timepoint(schedule)


@pytest.mark.parametrize(
    ('schedule', 'error_type', 'message'),
    [
        (None, TypeError, 'must be a mapping'),
        ({'CHR': []}, TypeError, 'must map timepoints to form lists'),
        ({'CHR': {'screening': 'scheduled_form'}},
         TypeError, 'must be a list'),
        ({'': {'screening': []}}, ValueError, 'blank cohort key'),
        ({'CHR': {' ': []}}, ValueError, 'blank timepoint key'),
        ({'CHR': {'screening': [None]}},
         TypeError, 'non-string form name'),
        ({'CHR': {'screening': [' ']}}, ValueError, 'blank form name'),
    ],
)
def test_forms_per_timepoint_normalization_rejects_invalid_shape(
        schedule, error_type, message):
    with pytest.raises(error_type, match=message):
        _normalize_forms_per_timepoint(schedule)


class _Utils:
    missing_code_list = ['-3', '-9', '-99', '999',
                         '1903-03-03', '1909-09-09', '1901-01-01']

    @staticmethod
    def check_if_missing(row, form, timepoint, network):
        if form == 'unexpected_form':
            raise AssertionError(
                'unscheduled forms must not reach the legacy missingness check')
        return bool(
            getattr(row, f'{form}_missing', False)
            or getattr(row, f'{form}_inferred_missing', False))


def _collector():
    collector = object.__new__(MultiTPDataCollector)
    collector.utils = _Utils()
    collector.subject_info = {
        'S1': {'cohort': 'CHR'},
        'S2': {'cohort': 'unknown'},
    }
    collector.important_form_vars = {
        'scheduled_form': {
            'interview_date_var': 'scheduled_date',
            'missing_var': 'scheduled_form_missing',
            'completion_var': 'scheduled_form_complete',
            'non_branch_logic_vars': ['scheduled_date'],
        },
        # The collector must inspect recognized date fields even when their
        # form is absent from forms_per_timepoint.
        'unexpected_form': {
            'interview_date_var': 'unexpected_date',
            'missing_var': 'unexpected_form_missing',
            'completion_var': 'unexpected_form_complete',
            'non_branch_logic_vars': ['unexpected_date'],
        },
        'invalid_form': {
            'interview_date_var': 'invalid_date',
            'missing_var': 'invalid_form_missing',
            'completion_var': 'invalid_form_complete',
            'non_branch_logic_vars': ['invalid_date'],
        },
        'missing_data': {
            'interview_date_var': 'chrmiss_interview_date',
            'missing_var': '',
            'completion_var': 'chrmiss_complete',
            'non_branch_logic_vars': ['chrmiss_interview_date'],
        },
    }
    collector.interview_date_fields = [
        (form, info['interview_date_var'])
        for form, info in collector.important_form_vars.items()
    ]
    collector.forms_per_tp = {
        'chr': {'screening': ['scheduled_form']},
        'hc': {'screening': ['scheduled_form']},
    }
    collector.earliest_latest_dates_per_tp = {}
    return collector


def test_collects_every_present_date_without_cohort_or_schedule_lookup():
    collector = _collector()
    frame = pd.DataFrame([{
        'subjectid': 'S1',
        'scheduled_date': '2025-01-20',
        'unexpected_date': '2025-01-01 00:00:00',
        'invalid_date': '-9',
    }])

    collector.collect_earliest_latest_dates(frame, 'screening', 'PRONET')

    info = collector.earliest_latest_dates_per_tp['S1']['screening']
    # Legacy consumers retain scheduled-form semantics.
    assert info['earliest'] == '2025-01-20'
    assert info['earliest_form'] == 'scheduled_form'
    assert info['earliest_variable'] == 'scheduled_date'
    assert info['latest'] == '2025-01-20'
    assert info['latest_form'] == 'scheduled_form'
    # Date Report's network range includes every configured date field,
    # including unexpected forms and ISO timestamps.
    assert info['network_extrema']['PRONET']['earliest'] == '2025-01-01'
    assert info['network_extrema']['PRONET']['earliest_form'] == 'unexpected_form'
    assert info['network_extrema']['PRONET']['latest'] == '2025-01-20'
    assert info['network_extrema']['PRONET']['variable_dates'] == {
        'scheduled_date': {
            'date': '2025-01-20', 'form': 'scheduled_form'},
        'unexpected_date': {
            'date': '2025-01-01', 'form': 'unexpected_form'},
    }


def test_duplicate_subject_rows_keep_latest_date_for_each_variable():
    collector = _collector()
    frame = pd.DataFrame([
        {
            'subjectid': 'S1',
            'scheduled_date': '2025-01-20',
            'unexpected_date': '',
            'invalid_date': '',
        },
        {
            'subjectid': 'S1',
            'scheduled_date': '2025-01-10',
            'unexpected_date': '',
            'invalid_date': '',
        },
    ])

    collector.collect_earliest_latest_dates(frame, 'screening', 'PRONET')

    network_dates = collector.earliest_latest_dates_per_tp[
        'S1']['screening']['network_extrema']['PRONET']
    assert network_dates['variable_dates']['scheduled_date'] == {
        'date': '2025-01-20', 'form': 'scheduled_form'}


def test_uppercase_production_cohort_keys_populate_legacy_ranges():
    """Production-style ``CHR``/``HC`` schedule keys must be case-insensitive.

    The Date Report's network range is deliberately schedule-independent, so
    checking only that range would miss this regression.  MED-QC-01 consumes
    the legacy top-level ``latest`` value asserted below.
    """
    collector = _collector()
    collector.subject_info = {
        'CHR_SUBJECT': {'cohort': ' chr '},
        'HC_SUBJECT': {'cohort': 'hC'},
    }
    collector.forms_per_tp = {
        'CHR': {'screening': ['scheduled_form']},
        'HC': {'screening': ['scheduled_form']},
    }
    frame = pd.DataFrame([
        {
            'subjectid': 'CHR_SUBJECT',
            'scheduled_date': '2025-01-20',
            'unexpected_date': '',
            'invalid_date': '',
        },
        {
            'subjectid': 'HC_SUBJECT',
            'scheduled_date': '2025-01-21',
            'unexpected_date': '',
            'invalid_date': '',
        },
    ])

    collector.collect_earliest_latest_dates(frame, 'screening', 'PRONET')

    chr_info = collector.earliest_latest_dates_per_tp[
        'CHR_SUBJECT']['screening']
    hc_info = collector.earliest_latest_dates_per_tp[
        'HC_SUBJECT']['screening']
    assert chr_info['latest'] == '2025-01-20'
    assert chr_info['latest_form'] == 'scheduled_form'
    assert hc_info['latest'] == '2025-01-21'
    assert hc_info['latest_form'] == 'scheduled_form'


def test_chrmiss_interview_date_is_not_collected_for_date_report():
    collector = _collector()
    frame = pd.DataFrame([{
        'subjectid': 'S1',
        'scheduled_date': '2025-01-20',
        'unexpected_date': '',
        'invalid_date': '',
        'chrmiss_interview_date': '2025-12-31',
    }])

    collector.collect_earliest_latest_dates(frame, 'screening', 'PRONET')

    network_range = collector.earliest_latest_dates_per_tp[
        'S1']['screening']['network_extrema']['PRONET']
    assert network_range['earliest_variable'] == 'scheduled_date'
    assert network_range['latest_variable'] == 'scheduled_date'
    assert 'chrmiss_interview_date' not in network_range['variable_dates']


def test_network_extrema_are_isolated_for_same_subject_identifier():
    collector = _collector()
    pronet = pd.DataFrame([{
        'subjectid': 'S1', 'scheduled_date': '2025-01-01',
        'unexpected_date': '', 'invalid_date': ''}])
    prescient = pd.DataFrame([{
        'subjectid': 'S1', 'scheduled_date': '2025-03-01',
        'unexpected_date': '', 'invalid_date': ''}])

    collector.collect_earliest_latest_dates(pronet, 'screening', 'PRONET')
    collector.collect_earliest_latest_dates(
        prescient, 'screening', 'PRESCIENT')

    info = collector.earliest_latest_dates_per_tp['S1']['screening']
    assert info['network_extrema']['PRONET']['latest'] == '2025-01-01'
    assert info['network_extrema']['PRESCIENT']['latest'] == '2025-03-01'
    assert info['network_extrema']['PRONET']['variable_dates'][
        'scheduled_date']['date'] == '2025-01-01'
    assert info['network_extrema']['PRESCIENT']['variable_dates'][
        'scheduled_date']['date'] == '2025-03-01'
    # Existing consumers retain the legacy merged extrema.
    assert info['earliest'] == '2025-01-01'
    assert info['latest'] == '2025-03-01'


def test_subjects_not_known_to_pipeline_are_skipped():
    collector = _collector()
    frame = pd.DataFrame([{
        'subjectid': 'NOT_IN_SUBJECT_INFO',
        'scheduled_date': '2025-01-01',
        'unexpected_date': '',
        'invalid_date': '',
    }])
    collector.collect_earliest_latest_dates(frame, 'screening', 'PRONET')
    assert collector.earliest_latest_dates_per_tp == {}


def test_populated_date_on_missing_marked_form_is_excluded_from_all_ranges():
    collector = _collector()
    frame = pd.DataFrame([{
        'subjectid': 'S1',
        'scheduled_date': '2025-01-15',
        'unexpected_date': '',
        'invalid_date': '',
        'scheduled_form_missing': True,
    }])

    collector.collect_earliest_latest_dates(frame, 'screening', 'PRONET')

    assert collector.earliest_latest_dates_per_tp == {}


def test_missing_form_date_cannot_become_prior_maximum_for_date_report():
    collector = _collector()
    frame = pd.DataFrame([{
        'subjectid': 'S1',
        'scheduled_date': '2025-02-15',
        'unexpected_date': '2025-01-20',
        'invalid_date': '',
        'scheduled_form_missing': True,
        'unexpected_form_missing': False,
    }])

    collector.collect_earliest_latest_dates(frame, 'screening', 'PRONET')

    info = collector.earliest_latest_dates_per_tp['S1']['screening']
    # The only surviving observation belongs to the non-missing form.  In
    # particular, the later populated date on the missing scheduled form must
    # not become the prior maximum used by Date Report comparisons.
    network_range = info['network_extrema']['PRONET']
    assert network_range['earliest'] == '2025-01-20'
    assert network_range['latest'] == '2025-01-20'
    assert network_range['latest_form'] == 'unexpected_form'
    assert set(network_range['variable_dates']) == {'unexpected_date'}
    # The surviving form is not scheduled, so the legacy top-level extrema
    # remain absent.
    assert 'earliest' not in info


def test_prescient_unscheduled_missing_form_needs_no_completion_column():
    collector = _collector()
    # Unscheduled forms can have an interview-date + missing button in the
    # combined CSV without the PRESCIENT ``*_complete_rpms`` field.  The
    # explicit Date Report gate must handle this without calling the legacy
    # Utils check (the test double raises if it does).
    frame = pd.DataFrame([{
        'subjectid': 'S1',
        'scheduled_date': '',
        'unexpected_date': '2025-01-20',
        'invalid_date': '',
        'unexpected_form_missing': 1,
        # Deliberately no unexpected_form_complete_rpms column.
    }])

    collector.collect_earliest_latest_dates(
        frame, 'screening', 'PRESCIENT')

    assert collector.earliest_latest_dates_per_tp == {}


def test_legacy_top_level_range_keeps_inferred_missing_semantics():
    collector = _collector()
    # Forms without a missing button retain the old scheduled-form behavior:
    # the general helper may infer that the form is missing, so it stays out of
    # legacy top-level extrema.  Date Report uses the narrower explicit-only
    # rule and therefore still retains the populated observation.
    collector.important_form_vars['scheduled_form']['missing_var'] = ''
    frame = pd.DataFrame([{
        'subjectid': 'S1',
        'scheduled_date': '2025-01-20',
        'unexpected_date': '',
        'invalid_date': '',
        'scheduled_form_inferred_missing': True,
    }])

    collector.collect_earliest_latest_dates(frame, 'screening', 'PRONET')

    info = collector.earliest_latest_dates_per_tp['S1']['screening']
    assert 'earliest' not in info
    assert info['network_extrema']['PRONET']['latest'] == '2025-01-20'


def test_excluded_forms_stay_out_of_date_report_but_not_legacy_range():
    assert set(EXCLUDED_DATE_FIELDS) == set(DATE_REPORT_EXCLUDED_FORMS)
    collector = _collector()
    for form, variable in EXCLUDED_DATE_FIELDS.items():
        collector.important_form_vars[form] = {
            'interview_date_var': variable,
            'missing_var': '',
            'completion_var': f'{form}_complete',
            'non_branch_logic_vars': [variable],
        }
        # Deliberately mutate the cached field list to prove the collection
        # loop itself enforces the Date Report exclusion.
        collector.interview_date_fields.append((form, variable))
    collector.forms_per_tp['chr']['screening'].extend(
        EXCLUDED_DATE_FIELDS)

    row = {
        'subjectid': 'S1',
        'scheduled_date': '2025-01-20',
        'unexpected_date': '',
        'invalid_date': '',
    }
    row.update({variable: '2025-12-31'
                for variable in EXCLUDED_DATE_FIELDS.values()})
    collector.collect_earliest_latest_dates(
        pd.DataFrame([row]), 'screening', 'PRONET')

    info = collector.earliest_latest_dates_per_tp['S1']['screening']
    network_range = info['network_extrema']['PRONET']
    assert network_range['latest_form'] == 'scheduled_form'
    assert network_range['latest_variable'] == 'scheduled_date'
    assert set(network_range['variable_dates']) == {'scheduled_date'}
    # Preserve the separate legacy range used by Pharm/SOP consumers.
    assert info['latest_form'] in DATE_REPORT_EXCLUDED_FORMS
