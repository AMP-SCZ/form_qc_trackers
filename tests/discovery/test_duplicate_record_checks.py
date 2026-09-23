"""
Tests for qc_types/discovery/duplicate_record_checks.py (rewrite).

The detector now reads long-format parquets (not per-tp CSVs) and has
two passes:
  1. cross_subject_fingerprint — same site/form/tp across subjects
  2. within_subject_form_copy — same subject across consecutive tps

Test cases:
  DUP-1: two subjects with identical form rows at same site/tp → 2
         cross-subject records (one per subject)
  DUP-2: all distinct → 0 records
  DUP-3: three subjects identical at same site → 3 records, group_size=3
  DUP-4: within-subject form copy at consecutive tps → 1 within_subject
         record on the later tp
  DUP-5: <min_cols form → cross-subject pass skipped (counter increments)
  DUP-6: missing-code rows excluded from fingerprint (do not falsely
         match identical missing-code patterns)
  DUP-7: excluded source_form (digital_biomarkers_*) → 0 records
  DUP-8: cross-site identical fingerprints do NOT match (site_prefix
         scopes the cross-subject pass)
  Schema: missing required column → RuntimeError
  Empty:  no candidates → DataFrame with declared schema/dtypes
  File:   end-to-end run() — parquet readable, no .tmp residue
"""

import os
import sys
import pytest
import pandas as pd

_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc_types.discovery.duplicate_record_checks import (
    DuplicateRecordChecks)


LONG_FORMAT_COLUMNS = [
    'subjectid', 'network', 'timepoint', 'cohort', 'event_date',
    'source_form', 'variable', 'value', 'value_numeric',
    'is_missing_code',
]

TEST_TP_ORDER = ['screening', 'baseline', 'month1', 'month2']


def _make_long_df(rows):
    full = []
    for r in rows:
        full.append({
            'subjectid': r['subjectid'],
            'network': r.get('network', 'TEST'),
            'timepoint': r.get('timepoint', 'baseline'),
            'cohort': r.get('cohort', ''),
            'event_date': r.get('event_date', ''),
            'source_form': r.get('source_form', 'form_a'),
            'variable': r['variable'],
            'value': r['value'],
            'value_numeric': r['value_numeric'],
            'is_missing_code': r['is_missing_code'],
        })
    return pd.DataFrame(full, columns=LONG_FORMAT_COLUMNS)


def _subject_form_rows(
    subjectid, values_by_var,
    timepoint='baseline', source_form='form_a',
):
    """Emit one row per (variable, value) for one subject at one tp."""
    rows = []
    for var, val in values_by_var.items():
        rows.append({
            'subjectid': subjectid, 'timepoint': timepoint,
            'variable': var, 'source_form': source_form,
            'value': str(val), 'value_numeric': float(val),
            'is_missing_code': False,
        })
    return rows


@pytest.fixture
def detector_factory(tmp_path, monkeypatch):
    deps = tmp_path / "dependencies"
    out = tmp_path / "output"
    deps.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)

    class FakeUtils:
        absolute_path = str(tmp_path)

        def create_timepoint_list(self):
            return list(TEST_TP_ORDER)

    import qc_types.discovery.duplicate_record_checks \
        as detector_module
    monkeypatch.setattr(detector_module, 'Utils', FakeUtils)

    def _make(**overrides):
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "duplicate_record": {
                    "networks": ["TEST"],
                    "min_cols_for_fingerprint": 8,
                    "dup_cols_cap": 30,
                    "cross_subject_min_group_size": 2,
                    "cross_subject_min_match_fraction": 0.85,
                    "within_subject_min_match_fraction": 0.85,
                    "within_subject_min_form_vars": 5,
                    **overrides,
                }
            },
        }
        return DuplicateRecordChecks(config_dict=cfg)

    return _make


def _form_values_for_test(seed_offset=0):
    """8 fields with distinct values (distinguishes 'identical' from
    'all-zeros looks identical to all-zeros')."""
    return {
        f"f{i}": float(i + 1 + seed_offset)
        for i in range(8)
    }


# ---------------------------------------------------------------- DUP-1

def test_dup1_two_subjects_identical(detector_factory):
    detector = detector_factory()
    vals = _form_values_for_test()
    # Same site (TS), same tp, same form, identical values.
    rows = []
    rows += _subject_form_rows('TS001', vals)
    rows += _subject_form_rows('TS002', vals)
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    cross = result[
        result['algorithm'] == 'cross_subject_fingerprint']
    assert len(cross) == 2
    assert set(cross['subjectid']) == {'TS001', 'TS002'}
    assert float(cross.iloc[0]['group_size']) == 2.0


# ---------------------------------------------------------------- DUP-2

def test_dup2_all_distinct_zero_records(detector_factory):
    detector = detector_factory()
    rows = []
    for k in range(5):
        rows += _subject_form_rows(
            f'TS{k:03d}', _form_values_for_test(seed_offset=k * 7))
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (
        result['algorithm'] == 'cross_subject_fingerprint'
    ).sum() == 0


# ---------------------------------------------------------------- DUP-3

def test_dup3_three_subjects_identical(detector_factory):
    detector = detector_factory()
    vals = _form_values_for_test()
    rows = []
    rows += _subject_form_rows('TS001', vals)
    rows += _subject_form_rows('TS002', vals)
    rows += _subject_form_rows('TS003', vals)
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    cross = result[
        result['algorithm'] == 'cross_subject_fingerprint']
    assert len(cross) == 3
    assert all(float(g) == 3.0 for g in cross['group_size'])


# ---------------------------------------------------------------- DUP-4

def test_dup4_within_subject_form_copy(detector_factory):
    detector = detector_factory()
    vals = _form_values_for_test()
    # Subject S1: identical values at baseline AND month1 on form_a.
    rows = []
    rows += _subject_form_rows(
        'TS001', vals, timepoint='baseline')
    rows += _subject_form_rows(
        'TS001', vals, timepoint='month1')
    # Add another subject so the cross-subject pass isn't tempted to
    # match TS001 against itself (it won't, but covers the path).
    rows += _subject_form_rows(
        'TS002', _form_values_for_test(seed_offset=50),
        timepoint='baseline')
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    within = result[result['algorithm'] == 'within_subject_form_copy']
    assert len(within) == 1
    r = within.iloc[0]
    assert r['subjectid'] == 'TS001'
    assert float(r['match_fraction']) == 1.0
    # The pair was baseline → month1; the flag is anchored at the
    # later visit (the visit that was copied INTO).
    assert r['timepoint'] == 'month1'


# ---------------------------------------------------------------- DUP-5

def test_dup5_too_few_cols_skipped(detector_factory):
    detector = detector_factory(min_cols_for_fingerprint=20)
    vals = _form_values_for_test()  # only 8 vars
    rows = []
    rows += _subject_form_rows('TS001', vals)
    rows += _subject_form_rows('TS002', vals)
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (
        result['algorithm'] == 'cross_subject_fingerprint'
    ).sum() == 0
    assert detector._counters[
        'cross_subject_groups_skipped_too_few_cols'] >= 1


# ---------------------------------------------------------------- DUP-6

def test_dup6_missing_code_rows_excluded(detector_factory):
    detector = detector_factory()
    # Two subjects whose only "match" would be on missing-code rows
    # (which the producer flags as is_missing_code=True). They should
    # not match.
    rows = []
    for sid in ('TS001', 'TS002'):
        for i in range(8):
            rows.append({
                'subjectid': sid, 'timepoint': 'baseline',
                'variable': f'f{i}',
                'source_form': 'form_a',
                'value': '999', 'value_numeric': 999.0,
                'is_missing_code': True,
            })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (
        result['algorithm'] == 'cross_subject_fingerprint'
    ).sum() == 0


# ---------------------------------------------------------------- DUP-7

def test_dup7_excluded_source_form(detector_factory):
    detector = detector_factory()
    vals = _form_values_for_test()
    rows = []
    rows += _subject_form_rows(
        'TS001', vals,
        source_form='digital_biomarkers_axivity_checkin')
    rows += _subject_form_rows(
        'TS002', vals,
        source_form='digital_biomarkers_axivity_checkin')
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['rows_excluded_by_form_pattern'] > 0


# ---------------------------------------------------------------- DUP-8

def test_dup8_different_sites_no_cross_match(detector_factory):
    detector = detector_factory()
    vals = _form_values_for_test()
    # Different first-two-char prefixes → different sites → cross-
    # subject pass scopes to within site, so these don't match.
    rows = []
    rows += _subject_form_rows('AA001', vals)
    rows += _subject_form_rows('BB001', vals)
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (
        result['algorithm'] == 'cross_subject_fingerprint'
    ).sum() == 0


# ---------------------------------------------------------------- Schema

def test_schema_validation_missing_column_raises(
    detector_factory, tmp_path
):
    detector = detector_factory()
    bogus = pd.DataFrame({
        'subjectid': ['S1'], 'network': ['TEST'],
        'timepoint': ['baseline'], 'variable': ['v'],
        'source_form': [''], 'value': ['50'],
        'value_numeric': [50.0],
    })
    bogus_path = (
        tmp_path / "dependencies"
        / "multi_tp_TEST_numeric_long.parquet")
    bogus.to_parquet(bogus_path, index=False)
    with pytest.raises(RuntimeError) as excinfo:
        detector._load_inputs()
    assert 'is_missing_code' in str(excinfo.value)


# ---------------------------------------------------------------- Empty

def test_empty_output_schema(detector_factory):
    detector = detector_factory()
    long_df = _make_long_df([])
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    expected_cols = list(DuplicateRecordChecks.OUTPUT_COLUMNS)
    assert list(result.columns) == expected_cols


# ---------------------------------------------------------------- File

def test_file_output_atomic(detector_factory, tmp_path):
    detector = detector_factory()
    vals = _form_values_for_test()
    rows = []
    rows += _subject_form_rows('TS001', vals)
    rows += _subject_form_rows('TS002', vals)
    long_path = (
        tmp_path / "dependencies"
        / "multi_tp_TEST_numeric_long.parquet")
    _make_long_df(rows).to_parquet(long_path, index=False)
    detector.run()
    out_path = (
        tmp_path / "output" / "combined_outputs" / "discovery"
        / "duplicate_record_candidates.parquet")
    assert out_path.exists()
    out_df = pd.read_parquet(out_path)
    assert len(out_df) >= 2
    tmps = list(out_path.parent.glob("*.tmp"))
    assert len(tmps) == 0
