"""
Tests for qc_types/date_check_logic.py (the pure helper behind the
"date" qc_type). Dependency-light: targets find_backward_visit_date with
plain dicts, so no FormCheck/Utils/config — DateChecks itself can't be
instantiated off-pipeline (FormCheck.__init__ builds Utils(), which
breaks on Windows).

Covers:
  DC-1: a later-tp visit date earlier than an earlier-tp date -> hit
  DC-2: running max -> compared against the largest earlier date
  DC-3: clean non-decreasing schedule -> None at every tp
  DC-4: floating/conversion as current_tp -> None
  DC-5: missing-coded / blank current date -> None
  DC-6: missing-coded earlier date is skipped, real earlier date used
  DC-7: equal date (not strictly earlier) -> None
  DC-8: no earlier date (first tp) -> None
  DC-9: current_tp not in tp_order -> None
"""

import os
import random
import sys
from datetime import date, timedelta
from types import SimpleNamespace

_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.realpath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc_types.date_check_logic import (  # noqa: E402
    DATE_REPORT_EXCLUDED_FORMS,
    find_backward_visit_date,
    find_backward_visit_dates,
    is_form_marked_missing,
)

TP_ORDER = ['screening', 'baseline', 'month1', 'month2', 'month3']
FULL_TP_ORDER = (
    ['screening', 'baseline']
    + [f'month{number}' for number in range(1, 13)]
    + ['month18', 'month24'])
# Includes the date-shaped REDCap sentinels so their exclusion is tested.
MISSING = ('', '-9', '-99', '999',
           '1909-09-09', '1903-03-03', '1901-01-01')


def _tp(date, form='psychs', variable='visit_date'):
    return {'earliest': date, 'latest': date,
            'earliest_form': form, 'latest_form': form,
            'earliest_variable': variable,
            'latest_variable': variable}


def test_explicit_missing_helper_handles_prescient_completion_mapping():
    info = {
        'missing_var': 'axivity_missing',
        'completion_var':
            'digital_biomarkers_axivity_end_of_12month_study_pe_complete',
    }
    row = SimpleNamespace(
        axivity_missing=0,
        digital_biomarkers_axivity_checkin_complete_rpms=3,
    )

    assert is_form_marked_missing(row, info, 'PRESCIENT') is True


def test_explicit_missing_helper_needs_no_prescient_completion_column():
    info = {
        'missing_var': 'measure_missing',
        'completion_var': 'measure_complete',
    }
    row = SimpleNamespace(measure_missing='1.0')

    assert is_form_marked_missing(row, info, 'PRESCIENT') is True


def test_prescient_completion_is_not_used_without_configured_missing_button():
    """Preserve the established check_if_missing semantics for these forms."""
    info = {'missing_var': '', 'completion_var': 'measure_complete'}
    row = SimpleNamespace(measure_complete_rpms=3)

    assert is_form_marked_missing(row, info, 'PRESCIENT') is False


# ---------------------------------------------------------------- DC-1

def test_dc1_backward_detected():
    dates = {
        'screening': _tp('2025-02-01'),
        'month1': _tp('2025-01-15', form='visit_form'),
    }
    res = find_backward_visit_date(dates, TP_ORDER, 'month1', MISSING)
    assert res is not None
    assert res['prev_timepoint'] == 'screening'
    assert res['prev_date'] == '2025-02-01'
    # prev_form is surfaced so DateChecks can NAME the earlier date.
    assert res['prev_form'] == 'psychs'
    assert res['current_date'] == '2025-01-15'
    assert res['current_form'] == 'visit_form'
    assert res['days_apart'] == 17


# ---------------------------------------------------------------- DC-2

def test_dc2_running_max():
    dates = {
        'screening': _tp('2025-01-10'),
        'baseline': _tp('2025-01-20'),
        'month1': _tp('2025-01-01'),
    }
    res = find_backward_visit_date(dates, TP_ORDER, 'month1', MISSING)
    # Compared against baseline (the latest earlier date), not screening.
    assert res['prev_timepoint'] == 'baseline'
    assert res['days_apart'] == 19
    # And a forward month2 vs that max is clean.
    dates['month2'] = _tp('2025-01-25')
    assert find_backward_visit_date(dates, TP_ORDER, 'month2', MISSING) is None


# ---------------------------------------------------------------- DC-3

def test_dc3_clean_schedule_none_at_every_tp():
    dates = {
        'screening': _tp('2025-01-01'),
        'baseline': _tp('2025-02-01'),
        'month1': _tp('2025-03-01'),
        'month2': _tp('2025-04-01'),
    }
    for tp in TP_ORDER:
        assert find_backward_visit_date(dates, TP_ORDER, tp, MISSING) is None


# ---------------------------------------------------------------- DC-4

def test_dc4_non_longitudinal_current_tp():
    dates = {
        'screening': _tp('2025-02-01'),
        'floating': _tp('2025-01-01'),
        'conversion': _tp('2025-01-02'),
    }
    assert find_backward_visit_date(dates, TP_ORDER, 'floating', MISSING) is None
    assert find_backward_visit_date(dates, TP_ORDER, 'conversion', MISSING) is None


def test_non_longitudinal_timepoints_cannot_be_prior_references():
    order = ['screening', 'floating', 'conversion', 'baseline']
    dates = {
        'screening': _tp('2025-01-01'),
        'floating': _tp('2025-04-01'),
        'conversion': _tp('2025-05-01'),
        'baseline': _tp('2025-02-01'),
    }

    assert find_backward_visit_date(
        dates, order, 'baseline', MISSING) is None


# ---------------------------------------------------------------- DC-5

def test_dc5_missing_or_blank_current_date():
    base = {'screening': _tp('2025-02-01')}
    # Blanks, numeric codes, the date-shaped sentinel family, and plain
    # junk must all be treated as "no date" -> no flag.
    for bad in ('', '-9', '-99', '999', '1909-09-09', '1903-03-03',
                '1901-01-01', 'not-a-date'):
        dates = dict(base, month1=_tp(bad))
        assert find_backward_visit_date(dates, TP_ORDER, 'month1', MISSING) is None


def test_dc5b_missing_coded_earlier_date_not_used():
    # A date-shaped sentinel at an earlier tp must not be read as a real
    # prior visit date (which would suppress or misattribute the flag).
    dates = {
        'screening': _tp('1909-09-09'),   # sentinel -> ignored as prior
        'baseline': _tp('2025-02-01'),
        'month1': _tp('2025-01-15'),
    }
    res = find_backward_visit_date(dates, TP_ORDER, 'month1', MISSING)
    assert res is not None
    assert res['prev_timepoint'] == 'baseline'   # not screening sentinel
    assert res['days_apart'] == 17


# ---------------------------------------------------------------- DC-6

def test_dc6_missing_earlier_date_skipped():
    dates = {
        'screening': _tp('2025-02-01'),
        'baseline': _tp('-9'),            # missing -> skipped as prior
        'month1': _tp('2025-01-15'),
    }
    res = find_backward_visit_date(dates, TP_ORDER, 'month1', MISSING)
    assert res is not None
    assert res['prev_timepoint'] == 'screening'  # not baseline
    assert res['days_apart'] == 17


# ---------------------------------------------------------------- DC-7

def test_dc7_equal_date_not_flagged():
    dates = {
        'screening': _tp('2025-02-01'),
        'month1': _tp('2025-02-01'),
    }
    assert find_backward_visit_date(dates, TP_ORDER, 'month1', MISSING) is None


# ---------------------------------------------------------------- DC-8

def test_dc8_first_tp_no_earlier():
    dates = {'screening': _tp('2025-02-01')}
    assert find_backward_visit_date(dates, TP_ORDER, 'screening', MISSING) is None


# ---------------------------------------------------------------- DC-9

def test_dc9_current_tp_not_in_order():
    dates = {'screening': _tp('2025-02-01'), 'monthX': _tp('2025-01-01')}
    assert find_backward_visit_date(dates, TP_ORDER, 'monthX', MISSING) is None


def test_prior_latest_date_is_used_not_prior_earliest():
    dates = {
        'screening': {
            'earliest': '2025-01-01',
            'latest': '2025-01-20',
            'earliest_form': 'screening_a',
            'latest_form': 'screening_b',
            'earliest_variable': 'visit_date',
            'latest_variable': 'visit_date',
        },
        'baseline': _tp('2025-01-15', form='baseline_form'),
    }
    res = find_backward_visit_date(dates, TP_ORDER, 'baseline', MISSING)
    assert res is not None
    assert res['prev_date'] == '2025-01-20'
    assert res['prev_form'] == 'screening_b'
    assert res['days_apart'] == 5


def test_every_offending_current_date_is_returned():
    dates = {
        'screening': {
            'dates': [
                {'date': '2025-01-20', 'form': 'current_a',
                 'variable': 'current_a_date', 'network': 'PRONET'},
                {'date': '2025-01-20', 'form': 'current_b',
                 'variable': 'current_b_date', 'network': 'PRONET'},
                {'date': '2025-01-20', 'form': 'current_c',
                 'variable': 'current_c_date', 'network': 'PRONET'},
            ],
        },
    }
    current = [
        {'date': '2025-01-10', 'form': 'current_a',
         'variable': 'current_a_date', 'network': 'PRONET'},
        {'date': '2025-01-19', 'form': 'current_b',
         'variable': 'current_b_date', 'network': 'PRONET'},
        {'date': '2025-01-21', 'form': 'current_c',
         'variable': 'current_c_date', 'network': 'PRONET'},
    ]
    results = find_backward_visit_dates(
        dates, TP_ORDER, 'baseline', MISSING,
        network='PRONET', current_observations=current)

    assert [item['current_variable'] for item in results] == [
        'current_a_date', 'current_b_date']
    assert {item['prev_date'] for item in results} == {'2025-01-20'}


def test_cross_variable_prior_date_is_not_compared():
    dates = {
        'screening': {'dates': [{
            'date': '2025-03-01', 'form': 'form_b',
            'variable': 'date_b', 'network': 'PRONET'}]},
    }
    current = [{
        'date': '2025-01-01', 'form': 'form_a',
        'variable': 'date_a', 'network': 'PRONET'}]

    assert find_backward_visit_dates(
        dates, TP_ORDER, 'baseline', MISSING,
        network='PRONET', current_observations=current) == []


def test_matching_non_extreme_variable_is_compared():
    dates = {
        'screening': {
            'earliest': '2025-01-01',
            'latest': '2025-03-01',
            'earliest_form': 'form_b',
            'latest_form': 'form_c',
            'earliest_variable': 'date_b',
            'latest_variable': 'date_c',
            'variable_dates': {
                'date_a': {'date': '2025-02-01', 'form': 'form_a'},
                'date_b': {'date': '2025-01-01', 'form': 'form_b'},
                'date_c': {'date': '2025-03-01', 'form': 'form_c'},
            },
        },
    }
    current = [{
        'date': '2025-01-15', 'form': 'form_a',
        'variable': 'date_a', 'network': 'PRONET'}]

    results = find_backward_visit_dates(
        dates, TP_ORDER, 'baseline', MISSING,
        network='PRONET', current_observations=current)

    assert len(results) == 1
    assert results[0]['prev_variable'] == 'date_a'
    assert results[0]['prev_date'] == '2025-02-01'
    assert results[0]['days_apart'] == 17


def test_variable_names_match_case_sensitively():
    dates = {'screening': _tp(
        '2025-02-01', form='form_a', variable='Date_A')}
    current = [{
        'date': '2025-01-01', 'form': 'form_a',
        'variable': 'date_a'}]

    assert find_backward_visit_dates(
        dates, TP_ORDER, 'baseline', MISSING,
        current_observations=current) == []


def test_blank_legacy_variable_metadata_is_never_guessed():
    dates = {'screening': {
        'earliest': '2025-02-01',
        'latest': '2025-02-01',
        'earliest_form': 'form_a',
        'latest_form': 'form_a',
        'earliest_variable': '',
        'latest_variable': '',
    }}
    current = [{
        'date': '2025-01-01', 'form': 'form_a',
        'variable': 'date_a'}]

    assert find_backward_visit_dates(
        dates, TP_ORDER, 'baseline', MISSING,
        current_observations=current) == []


def test_same_timepoint_dates_are_not_compared_to_each_other():
    dates = {'screening': _tp(
        '2025-01-01', variable='current_a_date')}
    current = [
        {'date': '2025-01-10', 'form': 'current_a',
         'variable': 'current_a_date'},
        {'date': '2025-02-10', 'form': 'current_b',
         'variable': 'current_b_date'},
    ]
    assert find_backward_visit_dates(
        dates, TP_ORDER, 'baseline', MISSING,
        current_observations=current) == []


def test_exact_duplicates_are_removed_but_distinct_fields_are_retained():
    dates = {'screening': {'dates': [
        {'date': '2025-01-20', 'form': 'current_a',
         'variable': 'current_a_date'},
        {'date': '2025-01-20', 'form': 'current_b',
         'variable': 'current_b_date'},
    ]}}
    first = {'date': '2025-01-10', 'form': 'current_a',
             'variable': 'current_a_date'}
    second = {'date': '2025-01-10', 'form': 'current_b',
              'variable': 'current_b_date'}
    results = find_backward_visit_dates(
        dates, TP_ORDER, 'baseline', MISSING,
        current_observations=[first, first.copy(), second])
    assert {(item['current_form'], item['current_variable'])
            for item in results} == {
                ('current_a', 'current_a_date'),
                ('current_b', 'current_b_date'),
            }


def test_distinct_current_dates_for_same_variable_are_both_returned():
    dates = {'screening': _tp(
        '2025-01-20', form='current', variable='current_date')}
    first = {'date': '2025-01-10', 'form': 'current',
             'variable': 'current_date'}
    second = {'date': '2025-01-15', 'form': 'current',
              'variable': 'current_date'}

    results = find_backward_visit_dates(
        dates, TP_ORDER, 'baseline', MISSING,
        current_observations=[first, first.copy(), second])

    assert [item['current_date'] for item in results] == [
        '2025-01-10', '2025-01-15']
    assert [item['days_apart'] for item in results] == [10, 5]


def test_configured_forms_cannot_be_current_date_report_observations():
    excluded = sorted(DATE_REPORT_EXCLUDED_FORMS)
    dates = {'screening': {'dates': [
        {
            'date': '2025-02-01',
            'form': 'allowed_prior',
            'variable': f'excluded_date_{index}',
        }
        for index in range(len(excluded))
    ]}}
    current = [
        {
            'date': '2025-01-01',
            'form': form,
            'variable': f'excluded_date_{index}',
        }
        for index, form in enumerate(excluded)
    ]

    assert find_backward_visit_dates(
        dates, TP_ORDER, 'baseline', MISSING,
        current_observations=current) == []


def test_configured_forms_cannot_be_prior_from_stale_compact_extrema():
    current = [{
        'date': '2025-01-01',
        'form': 'allowed_current',
        'variable': 'allowed_current_date',
        'network': 'PRONET',
    }]
    for excluded_form in DATE_REPORT_EXCLUDED_FORMS:
        dates = {
            'screening': {
                'network_extrema': {
                    'PRONET': {
                        'latest': '2025-02-01',
                        'latest_form': excluded_form,
                        'latest_variable': 'allowed_current_date',
                    },
                },
            },
        }
        assert find_backward_visit_dates(
            dates, TP_ORDER, 'baseline', MISSING, network='PRONET',
            current_observations=current) == []


def test_network_specific_prior_extrema_do_not_cross_contaminate():
    dates = {
        'screening': {
            'earliest': '2025-01-01',
            'latest': '2025-03-01',
            'earliest_form': 'merged_a',
            'latest_form': 'merged_b',
            'network_extrema': {
                'PRONET': {
                    'earliest': '2025-01-01', 'latest': '2025-01-01',
                    'earliest_form': 'pronet', 'latest_form': 'pronet',
                    'earliest_variable': 'current_date',
                    'latest_variable': 'current_date'},
                'PRESCIENT': {
                    'earliest': '2025-03-01', 'latest': '2025-03-01',
                    'earliest_form': 'prescient', 'latest_form': 'prescient',
                    'earliest_variable': 'current_date',
                    'latest_variable': 'current_date'},
            },
        },
    }
    current = [{'date': '2025-02-01', 'form': 'current',
                'variable': 'current_date', 'network': 'PRONET'}]
    assert find_backward_visit_dates(
        dates, TP_ORDER, 'baseline', MISSING,
        network='PRONET', current_observations=current) == []


def test_iso_timestamp_current_date_is_accepted():
    dates = {'screening': _tp(
        '2025-01-20', variable='current_date')}
    current = [{'date': ' 2025-01-15 00:00:00 ', 'form': 'current',
                'variable': 'current_date'}]
    results = find_backward_visit_dates(
        dates, TP_ORDER, 'baseline', MISSING,
        current_observations=current)
    assert len(results) == 1
    assert results[0]['current_date'] == '2025-01-15'


def test_random_timelines_match_brute_force_oracle():
    rng = random.Random(1701)
    order = ['screening', 'baseline', 'month1', 'month2', 'month3']
    origin = date(2025, 1, 1)

    for _ in range(300):
        timeline = {}
        raw = {}
        for tp_index, tp in enumerate(order):
            observations = []
            raw[tp] = []
            for form_index in range(rng.randint(0, 4)):
                date_str = str(origin + timedelta(days=rng.randint(0, 50)))
                record = {
                    'date': date_str,
                    'form': f'form_{form_index}',
                    'variable': f'date_{form_index}',
                    'network': 'PRONET',
                }
                observations.append(record)
                raw[tp].append(record)
            if observations:
                timeline[tp] = {'dates': observations}

        for current_index, current_tp in enumerate(order):
            earlier = [
                observation
                for tp in order[:current_index]
                for observation in raw[tp]
            ]
            prior_max_by_variable = {}
            for observation in earlier:
                variable = observation['variable']
                prior_max_by_variable[variable] = max(
                    observation['date'],
                    prior_max_by_variable.get(variable, observation['date']))
            expected = {
                (observation['date'], observation['form'],
                 observation['variable'])
                for observation in raw[current_tp]
                if observation['variable'] in prior_max_by_variable
                and observation['date'] < prior_max_by_variable[
                    observation['variable']]
            }
            actual = {
                (finding['current_date'], finding['current_form'],
                 finding['current_variable'])
                for finding in find_backward_visit_dates(
                    timeline, order, current_tp, MISSING, network='PRONET')
            }
            assert actual == expected


def test_every_canonical_later_timepoint_participates():
    dates = {
        tp: _tp(str(date(2025, 1, 1) + timedelta(days=index + 10)))
        for index, tp in enumerate(FULL_TP_ORDER)
    }
    current = [{
        'date': '2025-01-01',
        'form': 'current',
        'variable': 'visit_date',
    }]

    for current_tp in FULL_TP_ORDER[1:]:
        results = find_backward_visit_dates(
            dates, FULL_TP_ORDER, current_tp, MISSING,
            current_observations=current)
        assert len(results) == 1, current_tp
        assert results[0]['prev_timepoint'] in FULL_TP_ORDER[
            :FULL_TP_ORDER.index(current_tp)]


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([os.path.realpath(__file__), '-v']))
