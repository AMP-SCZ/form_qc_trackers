"""
Regression tests for the LAST link of the cohort feature: the Cohort column
actually reaching the Excel tracker via
generate_reports/create_trackers.CreateTrackers.convert_to_shared_format.

The reconciliation tests in test_resolved_state.py stop at current_output.
This file covers what the user actually asked for — cohort surfacing in the
report — plus the REQUIRED defensive guard that keeps a current_output lacking
a `cohort` column (pre-cohort / zero-flag / stale run) from crashing the whole
reports stage with KeyError('Cohort').

convert_to_shared_format touches neither self.utils nor Dropbox, so we build
the instance with __new__ (bypassing __init__) and set only the two attributes
it reads: formatted_column_names. No config.json, no Utils stub, no Dropbox.
"""

import os
import sys

import pandas as pd

_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from generate_reports.create_trackers import CreateTrackers


# A minimal formatted_column_names mirroring the production shape (Cohort
# right after Participant). Every KEY must be a column present in the frame
# after convert_to_shared_format's groupby/agg + the derived date_resolved /
# flag_count columns, because the method hard-selects list(...values()).
_COMBINED = {
    'subject': 'Participant',
    'cohort': 'Cohort',
    'displayed_timepoint': 'Timepoint',
    'displayed_form': 'Form',
    'flag_count': 'Flag Count',
    'error_message': 'Flags',
    'time_since_last_detection': 'Days Since Detected',
    'date_resolved': 'Date Resolved',
    'manually_resolved': 'Manually Resolved',
    'priority_item': 'Priority Item',
}


def _tracker():
    ct = CreateTrackers.__new__(CreateTrackers)  # bypass __init__ (Utils/Dropbox)
    ct.formatted_column_names = {'PRONET': {'combined': dict(_COMBINED)}}
    return ct


def _raw_row(**overrides):
    # The columns convert_to_shared_format reads/aggregates. var_translations,
    # time_since_last_detection, priority_item and dates_resolved must be
    # present because agg_args references them unconditionally.
    row = {
        'subject': 'AB001',
        'cohort': 'CHR',
        'displayed_timepoint': 'tp_1',
        'displayed_form': 'form_a',
        'currently_resolved': False,
        'manually_resolved': '',
        'error_message': 'var_a : something is wrong',
        'var_translations': '',
        'time_since_last_detection': '',
        'priority_item': False,
        'dates_resolved': '',
    }
    row.update(overrides)
    return row


def test_cohort_column_present_and_positioned_in_tracker():
    """Cohort surfaces in the tracker, right after Participant, with its value."""
    ct = _tracker()
    raw = pd.DataFrame([_raw_row(subject='AB001', cohort='CHR'),
                        _raw_row(subject='AB002', cohort='HC')])
    out = ct.convert_to_shared_format(raw, 'PRONET')

    assert list(out.columns)[:2] == ['Participant', 'Cohort'], \
        f"Cohort must render right after Participant, got {list(out.columns)}"
    by_part = out.set_index('Participant')['Cohort'].to_dict()
    assert by_part == {'AB001': 'CHR', 'AB002': 'HC'}, \
        f"cohort values did not survive into the tracker: {by_part}"


def test_guard_prevents_keyerror_when_cohort_absent():
    """
    A current_output lacking `cohort` (pre-cohort / zero-flag / stale run)
    must NOT crash the reports stage. The guard adds an empty cohort column so
    the hard column-selection succeeds and Cohort renders blank.
    """
    ct = _tracker()
    raw = pd.DataFrame([_raw_row(subject='AB001')])
    raw = raw.drop(columns=['cohort'])  # simulate cohort-less current_output

    out = ct.convert_to_shared_format(raw, 'PRONET')  # must not KeyError

    assert 'Cohort' in out.columns
    assert list(out['Cohort']) == [''], \
        f"guard should yield a blank Cohort column, got {list(out['Cohort'])}"


def test_current_tracker_read_preserves_nullable_integer_columns(tmp_path):
    """The report stage must not put '' into nullable integer evidence."""
    path = tmp_path / 'combined_qc_flags.parquet'
    source = pd.DataFrame({
        'subject': ['AB001', 'AB002'],
        'reports': ['Main Report', None],
        'date_gap_days': pd.Series([10, pd.NA], dtype='Int64'),
        'time_since_last_detection': pd.Series([5, pd.NA], dtype='Int64'),
    })
    source.to_parquet(path, index=False)

    result = CreateTrackers._read_combined_tracker(path)

    assert str(result['date_gap_days'].dtype) == 'Int64'
    assert str(result['time_since_last_detection'].dtype) == 'Int64'
    assert result.loc[0, 'date_gap_days'] == 10
    assert pd.isna(result.loc[1, 'date_gap_days'])
    assert result.loc[1, 'reports'] == ''
