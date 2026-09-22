"""
Tests for the rewritten qc_types/discovery/copy_forward_checks.py.

The detector now operates at FORM level: flag a (subject, source_form,
tp_pair) where a large fraction of the form's numeric variables are
bit-identical across adjacent visits, weighted by cohort variability
(forms where the cohort is uniformly stable don't fire).

Test cases:
  CF-1: subject copies entire form baseline → month1 → 1 flag at month1
  CF-2: subject has small fraction of identical variables → 0 flags
  CF-3: form where cohort is uniformly stable (variability factor = 0)
        → 0 flags even if subject's form is identical (it's the
        cohort's baseline, not evidence of copy)
  CF-4: fewer than min_form_vars compared → tp-pair skipped
  CF-5: excluded source_form (digital_biomarkers_*) → 0 flags
  Schema: missing required col → RuntimeError
  Empty:  no flags → DataFrame with declared schema/dtypes
  File:   end-to-end run() — parquet readable, no .tmp residue
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc_types.discovery.copy_forward_checks import CopyForwardChecks


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
    subjectid, values_by_var, timepoint='baseline',
    source_form='form_a',
):
    rows = []
    for var, val in values_by_var.items():
        rows.append({
            'subjectid': subjectid, 'timepoint': timepoint,
            'variable': var, 'source_form': source_form,
            'value': str(val), 'value_numeric': float(val),
            'is_missing_code': False,
        })
    return rows


def _variable_cohort_rows(
    n_subjects=22, vars_per_subject=10,
    source_form='form_a', timepoint='baseline', seed=11,
):
    """
    Cohort with `vars_per_subject` numeric variables and non-trivial
    subject-to-subject variability so the cohort MAD per variable is
    > 0 (necessary for copy-forward severity to be non-zero).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        sid = f"S{i+1:03d}"
        vals_by_var = {
            f"v{j}": float(50 + rng.integers(0, 30) + j)
            for j in range(vars_per_subject)
        }
        rows += _subject_form_rows(
            sid, vals_by_var, timepoint=timepoint,
            source_form=source_form)
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

    import qc_types.discovery.copy_forward_checks as detector_module
    monkeypatch.setattr(detector_module, 'Utils', FakeUtils)

    def _make(**overrides):
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "copy_forward": {
                    "networks": ["TEST"],
                    "min_match_fraction": 0.85,
                    "min_form_vars": 8,
                    "min_cohort_obs_per_variable": 20,
                    **overrides,
                }
            },
        }
        return CopyForwardChecks(config_dict=cfg)

    return _make


# ---------------------------------------------------------------- CF-1

def test_cf1_full_form_copy_fires(detector_factory):
    detector = detector_factory()
    rows = _variable_cohort_rows(
        n_subjects=22, vars_per_subject=10,
        timepoint='baseline')
    rows += _variable_cohort_rows(
        n_subjects=22, vars_per_subject=10,
        timepoint='month1', seed=17)
    # COPIER: identical full form at baseline and month1.
    copier_vals = {f"v{j}": float(100 + j) for j in range(10)}
    rows += _subject_form_rows(
        'COPIER', copier_vals, timepoint='baseline')
    rows += _subject_form_rows(
        'COPIER', copier_vals, timepoint='month1')
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    cop = result[result['subjectid'] == 'COPIER']
    assert len(cop) == 1
    assert cop.iloc[0]['timepoint'] == 'month1'
    assert float(cop.iloc[0]['match_fraction']) >= 0.85


# ---------------------------------------------------------------- CF-2

def test_cf2_partial_change_no_flag(detector_factory):
    detector = detector_factory()
    rows = _variable_cohort_rows(
        n_subjects=22, vars_per_subject=10,
        timepoint='baseline')
    rows += _variable_cohort_rows(
        n_subjects=22, vars_per_subject=10,
        timepoint='month1', seed=17)
    base = {f"v{j}": float(100 + j) for j in range(10)}
    next_tp = dict(base)
    for j in (0, 1, 2, 3, 4, 5):
        next_tp[f"v{j}"] = float(200 + j)  # change 6 vars
    rows += _subject_form_rows(
        'PARTIAL', base, timepoint='baseline')
    rows += _subject_form_rows(
        'PARTIAL', next_tp, timepoint='month1')
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (result['subjectid'] == 'PARTIAL').sum() == 0


# ---------------------------------------------------------------- CF-3

def test_cf3_uniformly_stable_form_zero_weight(detector_factory):
    """
    Cohort uniformly identical at every variable at every visit →
    cohort MAD = 0 → variability factor = 0 → severity 0 → no flag,
    even when the subject's form is 100% identical across visits.
    """
    detector = detector_factory()
    rows = []
    fixed_vals = {f"v{j}": float(50) for j in range(10)}
    for i in range(22):
        sid = f"S{i+1:03d}"
        for tp in ('baseline', 'month1'):
            rows += _subject_form_rows(
                sid, fixed_vals, timepoint=tp)
    rows += _subject_form_rows(
        'COPIER', fixed_vals, timepoint='baseline')
    rows += _subject_form_rows(
        'COPIER', fixed_vals, timepoint='month1')
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (result['subjectid'] == 'COPIER').sum() == 0
    assert (
        detector._counters['flags_zero_after_variability_weighting']
        >= 1)


# ---------------------------------------------------------------- CF-4

def test_cf4_too_few_form_vars_skipped(detector_factory):
    detector = detector_factory(min_form_vars=12)
    rows = _variable_cohort_rows(
        n_subjects=22, vars_per_subject=10,
        timepoint='baseline')
    rows += _variable_cohort_rows(
        n_subjects=22, vars_per_subject=10,
        timepoint='month1', seed=17)
    copier_vals = {f"v{j}": float(100 + j) for j in range(10)}
    rows += _subject_form_rows(
        'COPIER', copier_vals, timepoint='baseline')
    rows += _subject_form_rows(
        'COPIER', copier_vals, timepoint='month1')
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (result['subjectid'] == 'COPIER').sum() == 0
    assert detector._counters[
        'tp_pairs_skipped_too_few_joint_vars'] >= 1


# ---------------------------------------------------------------- CF-5

def test_cf5_excluded_source_form(detector_factory):
    detector = detector_factory()
    form = 'digital_biomarkers_axivity_checkin'
    rows = _variable_cohort_rows(
        n_subjects=22, vars_per_subject=10,
        timepoint='baseline', source_form=form)
    rows += _variable_cohort_rows(
        n_subjects=22, vars_per_subject=10,
        timepoint='month1', seed=17, source_form=form)
    copier_vals = {f"v{j}": float(100 + j) for j in range(10)}
    rows += _subject_form_rows(
        'COPIER', copier_vals, timepoint='baseline',
        source_form=form)
    rows += _subject_form_rows(
        'COPIER', copier_vals, timepoint='month1',
        source_form=form)
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['rows_excluded_by_form_pattern'] > 0


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
    expected_cols = list(CopyForwardChecks.OUTPUT_COLUMNS)
    assert list(result.columns) == expected_cols


# ---------------------------------------------------------------- File

def test_file_output_atomic(detector_factory, tmp_path):
    detector = detector_factory()
    rows = _variable_cohort_rows(
        n_subjects=22, vars_per_subject=10,
        timepoint='baseline')
    rows += _variable_cohort_rows(
        n_subjects=22, vars_per_subject=10,
        timepoint='month1', seed=17)
    copier_vals = {f"v{j}": float(100 + j) for j in range(10)}
    rows += _subject_form_rows(
        'COPIER', copier_vals, timepoint='baseline')
    rows += _subject_form_rows(
        'COPIER', copier_vals, timepoint='month1')
    long_path = (
        tmp_path / "dependencies"
        / "multi_tp_TEST_numeric_long.parquet")
    _make_long_df(rows).to_parquet(long_path, index=False)
    detector.run()
    out_path = (
        tmp_path / "output" / "combined_outputs" / "discovery"
        / "copy_forward_candidates.parquet")
    assert out_path.exists()
    out_df = pd.read_parquet(out_path)
    assert (out_df['subjectid'] == 'COPIER').sum() == 1
    tmps = list(out_path.parent.glob("*.tmp"))
    assert len(tmps) == 0
