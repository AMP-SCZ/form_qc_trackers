"""Differential parity test for roadmap item #8 (vectorize/restructure the
per-cell O(rows x cols) loops in MultiTPDataCollector).

Strategy: the `_ref_*` functions below are VERBATIM copies of the original
(pre-#8) implementations, frozen here as an independent oracle. The test
asserts the production methods produce byte-identical accumulator dicts to
the oracle on synthetic frames covering the edge cases that matter:
missing codes (incl. 999/-9), the 888 non-missing code, floats, valid dates
straddling the 2022-01-01 cutoff, invalid date strings, numeric-looking
strings, blanks, multi-column-per-form date sharing, and accumulation across
multiple frame calls. A duplicate-column frame is checked separately.

This is differential testing (old impl vs new impl), not code-plus-its-own-
test: the oracle is the original algorithm, independent of the new one.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from _qc_harness import seed_config  # noqa: E402

seed_config()
from datetime import datetime  # noqa: E402
from process_variables.collect_multi_timepoint_data import MultiTPDataCollector  # noqa: E402
from utils.utils import Utils  # noqa: E402

UTILS = Utils()


# --------------------------------------------------------------------------
# Frozen oracles: VERBATIM original (pre-#8) algorithms.
# --------------------------------------------------------------------------
def _ref_collect_earliest_date(combined_df, forms_per_var, important_form_vars, utils, acc):
    all_cols = combined_df.columns
    for row in combined_df.itertuples():
        for col in all_cols:
            if col in forms_per_var.keys():
                form = forms_per_var[col]
                if form in important_form_vars.keys():
                    interview_date_var = important_form_vars[form]['interview_date_var']
                    missing_var = important_form_vars[form]['missing_var']
                    missing_spec_var = important_form_vars[form]['missing_spec_var']
                    if interview_date_var != '' and interview_date_var in all_cols:
                        interview_date_val = getattr(row, interview_date_var)
                        if (interview_date_val not in (utils.missing_code_list + ['']) and
                                utils.check_if_val_date_format(str(interview_date_val))):
                            datetime_format_val = datetime.strptime(str(interview_date_val), "%Y-%m-%d")
                            if datetime_format_val < datetime.strptime("2022-01-01", "%Y-%m-%d"):
                                continue
                            acc.setdefault(col, str(interview_date_val))
                            if (datetime_format_val < datetime.strptime(acc[col], "%Y-%m-%d")):
                                acc[col] = str(interview_date_val)
    return acc


def _ref_collect_variable_type_distributions(combined_df, utils, acc):
    for row in combined_df.itertuples():
        for var in combined_df.columns:
            var_val = getattr(row, var)
            acc.setdefault(var, {'missing_code': 0, 'num': 0, 'date': 0,
                                 'blank': 0, 'string': 0, 'other': 0})
            if var_val in utils.missing_code_set:
                acc[var]['missing_code'] += 1
            elif utils.can_be_float(var_val):
                acc[var]['num'] += 1
            elif utils.check_if_val_date_format(str(var_val).split(' ')[0]):
                acc[var]['date'] += 1
            elif var_val == '':
                acc[var]['blank'] += 1
            elif isinstance(var_val, str):
                acc[var]['string'] += 1
            else:
                acc[var]['other'] += 1
    return acc


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------
FORMS_PER_VAR = {
    'chrA_var1': 'formA', 'chrA_var2': 'formA', 'chrA_date': 'formA',
    'chrB_var1': 'formB', 'chrB_date': 'formB',
    'chrC_var1': 'formC',            # formC has empty interview_date_var
    'chrD_var1': 'formD',            # formD not in important_form_vars
    'chrE_var1': 'formE',            # formE date var not present in df
}
IMPORTANT_FORM_VARS = {
    'formA': {'interview_date_var': 'chrA_date', 'missing_var': '', 'missing_spec_var': ''},
    'formB': {'interview_date_var': 'chrB_date', 'missing_var': 'chrB_miss', 'missing_spec_var': ''},
    'formC': {'interview_date_var': '', 'missing_var': '', 'missing_spec_var': ''},
    'formE': {'interview_date_var': 'chrE_date_absent', 'missing_var': '', 'missing_spec_var': ''},
}


def _frame_1():
    return pd.DataFrame({
        'subjectid': ['S1', 'S2', 'S3'],
        'chrA_var1': ['x', '5', ''],
        'chrA_var2': ['888', '999', '3.14'],     # 888 is NOT a missing code -> num
        'chrA_date': ['2023-05-01', '2021-01-01', '2024-12-31'],  # 2021 < cutoff
        'chrB_var1': ['hello', '', '-9'],
        'chrB_date': ['2022-01-01', 'not-a-date', '2022-06-15'],
        'chrB_miss': ['1', '', '0'],
        'chrC_var1': ['2020-03-03', '2025-07-07', ''],
        'chrD_var1': ['1', '2', '3'],
        'chrE_var1': ['a', 'b', 'c'],
        'chrnum': [5, 999, 3],   # int dtype -> exercises numpy-scalar classification path
    })


def _frame_2():
    # Second frame to exercise cross-call accumulation (min should drop).
    return pd.DataFrame({
        'subjectid': ['S4', 'S5'],
        'chrA_var1': ['z', '0'],
        'chrA_var2': ['7', '-3'],
        'chrA_date': ['2022-02-02', '2030-01-01'],  # 2022-02-02 < frame1 min 2023-05-01
        'chrB_var1': ['q', 'r'],
        'chrB_date': ['2023-09-09', '2022-12-31'],
        'chrB_miss': ['', '1'],
        'chrC_var1': ['', ''],
        'chrD_var1': ['4', '5'],
        'chrE_var1': ['d', 'e'],
        'chrnum': [7, -3],   # int dtype; -3 is a missing code
    })


def _bare_collector():
    c = object.__new__(MultiTPDataCollector)
    c.utils = UTILS
    c.forms_per_var = FORMS_PER_VAR
    c.important_form_vars = IMPORTANT_FORM_VARS
    c.earliest_date_per_var = {}
    c.variable_type_distributions = {}
    return c


def test_collect_earliest_date_parity_single_and_accumulated():
    frames = [_frame_1(), _frame_2()]
    # oracle
    ref = {}
    for f in frames:
        _ref_collect_earliest_date(f, FORMS_PER_VAR, IMPORTANT_FORM_VARS, UTILS, ref)
    # production
    c = _bare_collector()
    for f in frames:
        c.collect_earliest_date(f)
    assert c.earliest_date_per_var == ref, (
        f"earliest_date_per_var diverged.\n new={c.earliest_date_per_var}\n ref={ref}")
    # sanity: non-trivial output (the test must actually exercise the logic)
    assert ref, "fixture produced no earliest dates — degenerate test"
    assert ref.get('chrA_date') == '2022-02-02'  # min across both frames, >= cutoff


def test_collect_variable_type_distributions_parity():
    frames = [_frame_1(), _frame_2()]
    ref = {}
    for f in frames:
        _ref_collect_variable_type_distributions(f, UTILS, ref)
    c = _bare_collector()
    for f in frames:
        c.collect_variable_type_distributions(f)
    assert c.variable_type_distributions == ref, (
        f"variable_type_distributions diverged.\n new={c.variable_type_distributions}\n ref={ref}")
    # sanity: each category actually got exercised somewhere
    totals = {k: 0 for k in ('missing_code', 'num', 'date', 'blank', 'string')}
    for d in ref.values():
        for k in totals:
            totals[k] += d[k]
    for k, v in totals.items():
        assert v > 0, f"category {k} never exercised by fixtures — strengthen them"
    # key insertion order (affects JSON serialization) must match
    assert list(c.variable_type_distributions.keys()) == list(ref.keys())


def test_collect_variable_type_distributions_empty_frame():
    # 0-row frame: original setdefaults nothing (loop never runs). The
    # restructured version must NOT pre-create zero-count entries.
    empty = _frame_1().iloc[0:0]
    ref = {}
    _ref_collect_variable_type_distributions(empty, UTILS, ref)
    c = _bare_collector()
    c.collect_variable_type_distributions(empty)
    assert c.variable_type_distributions == ref == {}


def test_duplicate_column_fallback_matches_itertuples_semantics():
    # Duplicate column names: itertuples mangles the 2nd occurrence, so the
    # original counts the FIRST column's value twice. The production method
    # must match that exactly (falls back to the itertuples path on dups).
    dup = pd.DataFrame([['5', '7', '2023-01-01']], columns=['chrA_var1', 'chrA_var1', 'chrA_date'])
    ref = {}
    _ref_collect_variable_type_distributions(dup, UTILS, ref)
    c = _bare_collector()
    c.collect_variable_type_distributions(dup)
    assert c.variable_type_distributions == ref, (
        f"dup-column behavior diverged.\n new={c.variable_type_distributions}\n ref={ref}")
