"""
Tests for generate_reports/date_report_sort.py (the pure helper that
orders the Date Report tab greatest-to-least by days apart).

Dependency-light: targets sort_date_report_rows directly (re + pandas),
without constructing CreateTrackers (which imports dropbox + Utils).
"""

import os
import sys

import pandas as pd

_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.realpath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from generate_reports.date_report_sort import sort_date_report_rows  # noqa: E402


def _flag(days):
    # Mirrors the real DateChecks message (which now names the earlier
    # date's variable) so the sort regex is exercised against production
    # wording, incl. the "chrx : " displayed_variable prefix.
    return (f"chrx_interview_date : Visit date at month1 (2025-01-15) is "
            f"{days} day(s) before chrpsychs_scr_interview_date "
            f"(2025-02-01) at the earlier timepoint screening.")


def _df(rows):
    cols = ['Participant', 'Flags', 'Date Resolved', 'Manually Resolved']
    return pd.DataFrame(rows, columns=cols)


def test_sorts_greatest_to_least_and_adds_column():
    df = _df([
        {'Participant': 'A', 'Flags': _flag(5),
         'Date Resolved': '', 'Manually Resolved': ''},
        {'Participant': 'B', 'Flags': _flag(40),
         'Date Resolved': '', 'Manually Resolved': ''},
        {'Participant': 'C', 'Flags': _flag(17),
         'Date Resolved': '', 'Manually Resolved': ''},
    ])
    out = sort_date_report_rows(df)
    assert list(out['Participant']) == ['B', 'C', 'A']
    assert list(out['Days Apart']) == [40, 17, 5]


def test_resolved_rows_float_to_bottom_then_by_days():
    df = _df([
        {'Participant': 'A', 'Flags': _flag(5),
         'Date Resolved': '', 'Manually Resolved': ''},
        {'Participant': 'B', 'Flags': _flag(100),
         'Date Resolved': '2026-06-01', 'Manually Resolved': ''},  # resolved
        {'Participant': 'C', 'Flags': _flag(17),
         'Date Resolved': '', 'Manually Resolved': ''},
        {'Participant': 'D', 'Flags': _flag(80),
         'Date Resolved': '', 'Manually Resolved': 'yes'},        # resolved
    ])
    out = sort_date_report_rows(df)
    # Unresolved (C=17, A=5) first by days desc, then resolved (B=100, D=80).
    assert list(out['Participant']) == ['C', 'A', 'B', 'D']


def test_unparseable_message_sorts_last_blank_column():
    df = _df([
        {'Participant': 'A', 'Flags': 'no day count here',
         'Date Resolved': '', 'Manually Resolved': ''},
        {'Participant': 'B', 'Flags': _flag(10),
         'Date Resolved': '', 'Manually Resolved': ''},
    ])
    out = sort_date_report_rows(df)
    assert list(out['Participant']) == ['B', 'A']
    assert out.iloc[0]['Days Apart'] == 10
    assert out.iloc[1]['Days Apart'] == ''   # unparseable -> blank


def test_empty_or_missing_flags_column_passthrough():
    empty = _df([])
    assert sort_date_report_rows(empty).empty
    no_flags = pd.DataFrame([{'Participant': 'A'}])
    out = sort_date_report_rows(no_flags)
    assert list(out['Participant']) == ['A']
    assert 'Days Apart' not in out.columns


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([os.path.realpath(__file__), '-v']))
