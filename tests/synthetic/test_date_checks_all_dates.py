import os
import sys
from types import SimpleNamespace


_HERE = os.path.dirname(os.path.realpath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _qc_harness import seed_config  # noqa: E402
from utils import utils as utils_module  # noqa: E402


CONFIG = seed_config()

from qc_forms import form_check as form_check_module  # noqa: E402

form_check_module._CONFIG_CACHE[utils_module._project_root()] = CONFIG

from qc_forms.qc_types.date_checks import DateChecks  # noqa: E402


def _form_check_info(config_info=None):
    def date_form(form, variable):
        return {
            'interview_date_var': variable,
            'missing_var': f'{form}_missing',
            'completion_var': f'{form}_complete',
            'non_branch_logic_vars': [variable],
        }

    important = {
        'prior_form': date_form('prior_form', 'prior_date'),
        'current_a': date_form('current_a', 'current_a_date'),
        'current_b': date_form('current_b', 'current_b_date'),
        'current_clean': date_form('current_clean', 'current_clean_date'),
    }
    info = {
        'subject_info': {
            'S1': {
                'cohort': 'CHR',
                'inclusion_status': 'included',
                'visit_status': 'baseline',
            }
        },
        'general_check_vars': {},
        'important_form_vars': important,
        'forms_per_timepoint': {'chr': {}, 'hc': {}},
        'converted_branching_logic': {},
        'excluded_branching_logic_vars': {},
        'team_report_forms': {},
        'grouped_variables': {
            'var_translations': {},
            'scid_vars': {'module_b_vars': [], 'module_c_vars': []},
        },
        'variables_added_later': {},
        'raw_csv_conversions': {},
        'variable_ranges': {},
        'earliest_latest_dates_per_tp': {
            'S1': {
                'screening': {
                    'earliest': '2025-01-20',
                    'latest': '2025-01-20',
                    'earliest_form': 'prior_form',
                    'latest_form': 'prior_form',
                    'earliest_variable': 'prior_date',
                    'latest_variable': 'prior_date',
                    'network_extrema': {
                        'PRONET': {
                            'earliest': '2025-01-20',
                            'latest': '2025-01-20',
                            'earliest_form': 'prior_form',
                            'latest_form': 'prior_form',
                            'earliest_variable': 'prior_date',
                            'latest_variable': 'prior_date',
                            'variable_dates': {
                                'current_a_date': {
                                    'date': '2025-01-20',
                                    'form': 'current_a',
                                },
                                'current_b_date': {
                                    'date': '2025-01-20',
                                    'form': 'current_b',
                                },
                                'current_clean_date': {
                                    'date': '2025-01-20',
                                    'form': 'current_clean',
                                },
                            },
                        }
                    },
                }
            }
        },
        'cognition_csvs': {},
    }
    if config_info is not None:
        info['config_info'] = config_info
    return info


def test_date_checks_emits_one_report_finding_per_offending_field():
    row = SimpleNamespace(
        subjectid='S1',
        visit_status_string='baseline',
        recruitment_status_v2='not_recruited',
        current_a_date='2025-01-10',
        current_b_date='2025-01-19',
        current_clean_date='2025-01-21',
    )

    outputs = DateChecks(
        row, 'baseline', 'PRONET', _form_check_info())()

    assert len(outputs) == 2
    assert [item['displayed_variable'] for item in outputs] == [
        'current_a_date', 'current_b_date']
    assert all(item['reports'] == 'Date Report' for item in outputs)
    assert all(
        item['affected_timepoints'] == 'screening | baseline'
        for item in outputs)
    assert [item['date_gap_days'] for item in outputs] == [10, 1]
    assert all(
        item['affected_variables'] == item['displayed_variable']
        for item in outputs)
    assert all(item['nda_excluder'] is False for item in outputs)
    assert all(item['check_id'].startswith('date_backward_visit::')
               for item in outputs)


def test_date_checks_excludes_current_date_on_form_marked_missing():
    row = SimpleNamespace(
        subjectid='S1',
        visit_status_string='baseline',
        recruitment_status_v2='not_recruited',
        current_a_date='2025-01-10',
        current_a_missing=1,
        current_b_date='2025-01-19',
        current_b_missing=0,
        current_clean_date='2025-01-21',
        current_clean_missing=0,
    )

    outputs = DateChecks(
        row, 'baseline', 'PRONET', _form_check_info())()

    assert len(outputs) == 1
    assert outputs[0]['displayed_variable'] == 'current_b_date'
    assert outputs[0]['date_gap_days'] == 1


def test_date_checks_never_flags_chrmiss_interview_date():
    info = _form_check_info()
    info['important_form_vars']['missing_data'] = {
        'interview_date_var': 'chrmiss_interview_date',
        'missing_var': '',
        'completion_var': 'chrmiss_complete',
        'non_branch_logic_vars': ['chrmiss_interview_date'],
    }
    info['earliest_latest_dates_per_tp']['S1']['screening'][
        'network_extrema']['PRONET']['variable_dates'][
            'chrmiss_interview_date'] = {
                'date': '2025-01-20', 'form': 'missing_data'}
    row = SimpleNamespace(
        subjectid='S1',
        visit_status_string='baseline',
        recruitment_status_v2='not_recruited',
        current_a_date='2025-01-21',
        current_b_date='2025-01-21',
        current_clean_date='2025-01-21',
        chrmiss_interview_date='2025-01-01',
    )

    assert DateChecks(row, 'baseline', 'PRONET', info)() == []


def test_chrmiss_interview_date_cannot_be_the_prior_reference():
    info = _form_check_info()
    prior = info['earliest_latest_dates_per_tp']['S1']['screening']
    prior['network_extrema']['PRONET'].update({
        'earliest': '2025-01-20',
        'latest': '2025-01-20',
        'earliest_form': 'missing_data',
        'latest_form': 'missing_data',
        'earliest_variable': 'chrmiss_interview_date',
        'latest_variable': 'chrmiss_interview_date',
        'variable_dates': {
            'chrmiss_interview_date': {
                'date': '2025-01-20',
                'form': 'missing_data',
            },
        },
    })
    row = SimpleNamespace(
        subjectid='S1',
        visit_status_string='baseline',
        recruitment_status_v2='not_recruited',
        current_a_date='2025-01-10',
        current_b_date='2025-01-21',
        current_clean_date='2025-01-21',
    )

    assert DateChecks(row, 'baseline', 'PRONET', info)() == []


def test_date_checks_does_not_compare_different_variables():
    info = _form_check_info()
    network_dates = info['earliest_latest_dates_per_tp'][
        'S1']['screening']['network_extrema']['PRONET']
    network_dates['variable_dates'] = {
        'prior_date': {'date': '2025-01-20', 'form': 'prior_form'},
    }
    row = SimpleNamespace(
        subjectid='S1',
        visit_status_string='baseline',
        recruitment_status_v2='not_recruited',
        current_a_date='2025-01-10',
        current_b_date='2025-01-19',
        current_clean_date='2025-01-21',
    )

    assert DateChecks(row, 'baseline', 'PRONET', info)() == []


def test_recruited_only_keeps_recruited_date_flags(monkeypatch):
    """Raw rows remain present, but tracker routing must honor the option."""
    monkeypatch.setitem(CONFIG, 'recruited_only', True)
    row = SimpleNamespace(
        subjectid='S1',
        visit_status_string='baseline',
        recruitment_status_v2='recruited',
        current_a_date='2025-01-10',
        current_b_date='2025-01-19',
        current_clean_date='2025-01-21',
    )

    outputs = DateChecks(
        row, 'baseline', 'PRONET', _form_check_info())()

    assert len(outputs) == 2
    assert all(item['reports'] == 'Date Report' for item in outputs)


def test_recruited_only_suppresses_nonrecruited_tracker_routes(monkeypatch):
    """The option changes report routing, not raw-QC-row generation."""
    monkeypatch.setitem(CONFIG, 'recruited_only', True)
    row = SimpleNamespace(
        subjectid='S1',
        visit_status_string='baseline',
        recruitment_status_v2='not_recruited',
        current_a_date='2025-01-10',
        current_b_date='2025-01-19',
        current_clean_date='2025-01-21',
    )

    outputs = DateChecks(
        row, 'baseline', 'PRONET', _form_check_info())()

    assert len(outputs) == 2
    assert all(item['reports'] == '' for item in outputs)


def test_current_run_config_overrides_stale_formcheck_cache(monkeypatch):
    """A same-process toggle must use QCFormsMain's new run snapshot."""
    monkeypatch.setitem(CONFIG, 'recruited_only', False)
    current_config = dict(CONFIG, recruited_only=True)
    row = SimpleNamespace(
        subjectid='S1',
        visit_status_string='baseline',
        recruitment_status_v2='not_recruited',
        current_a_date='2025-01-10',
        current_b_date='2025-01-19',
        current_clean_date='2025-01-21',
    )

    outputs = DateChecks(
        row, 'baseline', 'PRONET',
        _form_check_info(config_info=current_config))()

    assert len(outputs) == 2
    assert all(item['reports'] == '' for item in outputs)
