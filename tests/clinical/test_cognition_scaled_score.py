"""
Tests for the cognition standardized-score / range comparison fix in
qc_types/cognition_checks.standardized_score_check.

Background
----------
WAIS-IV (and some WASI-II) raw→scaled conversion tables collapse
multiple raw scores into one scaled-score cell, encoded as a range
string like "11-12" or "5-7". The previous comparison was
`redcap_score == convert_range_to_list(cell)`; that compares a
scalar to a list whenever the cell was a range, and was dtype-
sensitive even on scalar cells (int 11 vs string "11" never
matched). Net effect: the scaled-score-mismatch check essentially
never fired on real production data.

These tests exercise the new normalization at the call site without
needing the full FormCheck/Utils harness.
"""

import os
import sys
from collections import namedtuple

import pandas as pd
import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Stub a minimal qc_forms.form_check module so cognition_checks can
# import FormCheck without triggering the full Utils() disk reads.
# (cognition_checks does `from qc_forms.form_check import FormCheck`.)
import types as _types

if 'qc_forms' not in sys.modules:
    sys.modules['qc_forms'] = _types.ModuleType('qc_forms')
if 'qc_forms.form_check' not in sys.modules:
    fc_mod = _types.ModuleType('qc_forms.form_check')

    class _StubFormCheck:
        def __init__(self, *_, **__):
            pass

        @classmethod
        def standard_qc_check_filter(cls, fn):
            return fn
    fc_mod.FormCheck = _StubFormCheck
    sys.modules['qc_forms.form_check'] = fc_mod
    sys.modules['qc_forms'].form_check = fc_mod

# The real Utils.convert_range_to_list is the function under test
# upstream; import the real Utils class but use it only via the
# attributes we set ourselves.
from utils.utils import Utils  # noqa: E402
from qc_types.cognition_checks import CognitionChecks  # noqa: E402


_RAW_ROW_FIELDS = (
    'subjectid',
    'chriq_vocab_raw',
    'chriq_scaled_vocab',
    'chriq_matrix_raw',
    'chriq_scaled_matrix',
)
RawRow = namedtuple('RawRow', _RAW_ROW_FIELDS)


def _make_real_utils():
    """
    Minimal Utils for the comparison test. Bypass __init__ so we
    don't read config.json. We only need convert_range_to_list,
    missing_code_set, and missing_code_list.
    """
    u = object.__new__(Utils)
    u.missing_code_list = ['-3', '-9', -3, -9, 999, '999', '']
    u.missing_code_set = frozenset(u.missing_code_list)
    return u


def _make_iq_row(vc='', mr='', t_score='', scaled_score=''):
    """
    Build a namedtuple matching the columns the function reads
    (`vc`, `mr`, plus the assessment-specific scaled-score column).
    Both WAIS (`scaled_score`) and WASI (`t_score`) are supported
    so the same fixture covers both branches.
    """
    Row = namedtuple(
        'IQRow', ['vc', 'mr', 't_score', 'scaled_score'])
    return Row(vc=vc, mr=mr, t_score=t_score, scaled_score=scaled_score)


def _make_filtered_table(rows):
    """
    Build a single-column-set `filtered_table` whose itertuples()
    will yield the given rows. The function only iterates rows; it
    doesn't index the table by column name.
    """
    return pd.DataFrame(rows)


def _make_cognition_checks(assessment_table):
    """
    Construct a CognitionChecks instance without running its
    __init__ (which calls self.utils.load_dependency_json). Set
    only the attributes that standardized_score_check reads, plus
    stub create_row_output / final_output_list.
    """
    cc = object.__new__(CognitionChecks)
    cc.utils = _make_real_utils()
    cc.timepoint = 'baseline'
    cc.network = 'TEST'
    cc.final_output_list = []
    # The assessment table is fetched as
    #   self.cognition_csvs[f'iq_raw_conversion_{assessment}']
    cc.cognition_csvs = {
        'iq_raw_conversion_wais': assessment_table,
        'iq_raw_conversion_wasi': assessment_table,
    }

    # Stub create_row_output. The real implementation reads a lot of
    # FormCheck state; we only need to verify the flag fires, so
    # capture the minimal payload.
    def _stub_create_row_output(row, forms, variables,
                                error_message, output_changes):
        return {
            'subject': row.subjectid,
            'affected_forms': forms,
            'affected_variables': variables,
            'error_message': error_message,
            'output_changes': output_changes,
        }
    cc.create_row_output = _stub_create_row_output
    return cc


# ----------------------------------------------------------------
# The standardized_score_check function builds `filtered_table`
# from `standardized_score_table` filtered by an age-range header
# row. To exercise the new compare logic in isolation, we manually
# construct `filtered_table`-shaped data and invoke the function's
# inner loop via a thin re-entry: call standardized_score_check
# with a pre-built `standardized_score_table` shaped so that the
# age-bracket selection picks the only column-block. The function
# casts iq_age*12 to int and compares against the range it parses
# from the first row.
# ----------------------------------------------------------------

def _build_full_assessment_table(rows, scaled_col='scaled_score'):
    """
    Build a table whose row layout matches how `standardized_score_check`
    consumes the real CSV after `pd.read_csv(... keep_default_na=False)`.

    Real CSV file layout (line numbers in the actual file):
        line 1: range,range,range,...           ← becomes pd column headers
        line 2: 192-215,192-215,216-239,...     ← age-range row (iloc[0])
        line 3: vc,mr,scaled_score,...          ← column-name row (iloc[1])
        line 4+: data rows

    The function reads `iloc[0]` as the age-range vector and later does
    `filtered_table.columns = filtered_table.iloc[0]` after dropping
    `iloc[0]`, so what was originally `iloc[1]` (column-name row)
    becomes the new column index. The first iter row inside the
    `for iq_row in filtered_table.itertuples()` is therefore the
    column-name row itself, then the data rows.

    `rows` is a list of dicts with optional keys vc, mr, <scaled_col>,
    t_score. Only one age-bracket column-block is built; the function
    selects it as `range_to_use`.
    """
    columns = ['col_t', 'col_vc', 'col_mr', 'col_scaled']
    base = [
        # iloc[0] — age range string per cell. The function reads
        # this and picks the column block whose value matches the
        # subject's iq_age_mos.
        ['200-359', '200-359', '200-359', '200-359'],
        # iloc[1] — column-name labels. After the function does
        # `filtered_table.columns = filtered_table.iloc[0]` (where
        # iloc[0] now means iloc[1] of THIS fixture because it
        # already sliced iloc[1:]), these become the new columns.
        ['t_score', 'vc', 'mr', scaled_col],
    ]
    for r in rows:
        base.append([
            r.get('t_score', ''),
            r.get('vc', ''),
            r.get('mr', ''),
            r.get(scaled_col, ''),
        ])
    return pd.DataFrame(base, columns=columns)


# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------

def test_scalar_string_raw_score_matches():
    """
    vc='11' (scalar) and chriq_vocab_raw='11' (string). Old code
    sometimes failed on dtype mismatch; new code must match.
    """
    table = _build_full_assessment_table([
        {'vc': '11', 'mr': '8', 'scaled_score': '6'},
    ])
    cc = _make_cognition_checks(table)
    row = RawRow(
        subjectid='S001',
        chriq_vocab_raw='11',
        chriq_scaled_vocab='6',  # correct → no flag
        chriq_matrix_raw='99',   # no match in table
        chriq_scaled_matrix='',
    )
    cc.standardized_score_check(
        row, ['iq_assessment_wasiii_wiscv_waisiv'],
        ['chriq_fsiq'], {"reports": ['Main Report']},
        assessment='wais', iq_age=18.0)
    assert len(cc.final_output_list) == 0


def test_int_raw_score_matches_string_cell():
    """
    redcap raw score is int 11; conversion cell is string '11'.
    The new compare normalizes both to strings.
    """
    table = _build_full_assessment_table([
        {'vc': '11', 'mr': '8', 'scaled_score': '6'},
    ])
    cc = _make_cognition_checks(table)
    row = RawRow(
        subjectid='S001',
        chriq_vocab_raw=11,        # int — old `==` compare often missed
        chriq_scaled_vocab='99',   # mismatch → flag MUST fire
        chriq_matrix_raw='99',
        chriq_scaled_matrix='',
    )
    cc.standardized_score_check(
        row, ['iq_assessment_wasiii_wiscv_waisiv'],
        ['chriq_fsiq'], {"reports": ['Main Report']},
        assessment='wais', iq_age=18.0)
    flags = [
        f for f in cc.final_output_list
        if 'chriq_scaled_vocab' in f['affected_variables']]
    assert len(flags) == 1


def test_range_cell_matches_member():
    """
    Conversion cell is '11-12' (range); raw score is 11.
    Old code never matched (scalar vs list); new code must match.
    Pair with a SCALED mismatch so the flag fires and the path is
    observable.
    """
    table = _build_full_assessment_table([
        {'vc': '11-12', 'mr': '8', 'scaled_score': '6'},
    ])
    cc = _make_cognition_checks(table)
    row = RawRow(
        subjectid='S001',
        chriq_vocab_raw='11',
        chriq_scaled_vocab='99',   # mismatch → flag fires
        chriq_matrix_raw='99',
        chriq_scaled_matrix='',
    )
    cc.standardized_score_check(
        row, ['iq_assessment_wasiii_wiscv_waisiv'],
        ['chriq_fsiq'], {"reports": ['Main Report']},
        assessment='wais', iq_age=18.0)
    flags = [
        f for f in cc.final_output_list
        if 'chriq_scaled_vocab' in f['affected_variables']]
    assert len(flags) == 1


def test_range_cell_no_match_outside():
    """
    Conversion cell '11-12'; raw score '13' (outside range).
    No match → no flag.
    """
    table = _build_full_assessment_table([
        {'vc': '11-12', 'mr': '8', 'scaled_score': '6'},
    ])
    cc = _make_cognition_checks(table)
    row = RawRow(
        subjectid='S001',
        chriq_vocab_raw='13',
        chriq_scaled_vocab='99',   # would mismatch if compared, but
                                   # raw doesn't match the cell at all
        chriq_matrix_raw='99',
        chriq_scaled_matrix='',
    )
    cc.standardized_score_check(
        row, ['iq_assessment_wasiii_wiscv_waisiv'],
        ['chriq_fsiq'], {"reports": ['Main Report']},
        assessment='wais', iq_age=18.0)
    assert cc.final_output_list == []


def test_scaled_mismatch_after_raw_match_fires():
    """
    Explicit double-check: raw matches via range, scaled is wrong
    → exactly one flag.
    """
    table = _build_full_assessment_table([
        {'vc': '5-7', 'mr': '8', 'scaled_score': '3'},
    ])
    cc = _make_cognition_checks(table)
    row = RawRow(
        subjectid='S001',
        chriq_vocab_raw='6',
        chriq_scaled_vocab='5',    # should be 3
        chriq_matrix_raw='99',
        chriq_scaled_matrix='',
    )
    cc.standardized_score_check(
        row, ['iq_assessment_wasiii_wiscv_waisiv'],
        ['chriq_fsiq'], {"reports": ['Main Report']},
        assessment='wais', iq_age=18.0)
    assert len(cc.final_output_list) == 1
    assert (
        'chriq_scaled_vocab'
        in cc.final_output_list[0]['affected_variables'])


def test_correct_scaled_after_raw_match_no_flag():
    """
    Raw matches via range, scaled is correct → no flag.
    """
    table = _build_full_assessment_table([
        {'vc': '5-7', 'mr': '8', 'scaled_score': '3'},
    ])
    cc = _make_cognition_checks(table)
    row = RawRow(
        subjectid='S001',
        chriq_vocab_raw='6',
        chriq_scaled_vocab='3',    # matches the QC value → no flag
        chriq_matrix_raw='99',
        chriq_scaled_matrix='',
    )
    cc.standardized_score_check(
        row, ['iq_assessment_wasiii_wiscv_waisiv'],
        ['chriq_fsiq'], {"reports": ['Main Report']},
        assessment='wais', iq_age=18.0)
    assert cc.final_output_list == []


def test_blank_raw_score_no_match():
    """
    A blank chriq_vocab_raw must never coincidentally match an
    empty cell. The cell column on real WASI sheets often has
    empty entries on some rows (see iq_raw_conversion_wasi.csv
    line 5+ where many cells are blank).
    """
    table = _build_full_assessment_table([
        {'vc': '', 'mr': '8', 'scaled_score': '3'},
    ])
    cc = _make_cognition_checks(table)
    row = RawRow(
        subjectid='S001',
        chriq_vocab_raw='',
        chriq_scaled_vocab='99',   # would be a false positive if blank
                                   # matched blank
        chriq_matrix_raw='99',
        chriq_scaled_matrix='',
    )
    cc.standardized_score_check(
        row, ['iq_assessment_wasiii_wiscv_waisiv'],
        ['chriq_fsiq'], {"reports": ['Main Report']},
        assessment='wais', iq_age=18.0)
    assert cc.final_output_list == []
