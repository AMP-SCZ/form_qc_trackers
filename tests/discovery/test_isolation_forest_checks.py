"""
Tests for qc_types/discovery/isolation_forest_checks.py.

Cases:
  IF-1: clean cohort + clear joint-distribution outlier → 1+ flag
  IF-2: clean cohort, no outliers → 0 flags
  IF-3: too few joint observations → skipped
  IF-4: excluded source_form → 0 flags
  IF-5: severity_normalized in [0, 100] and top_contributing_variables
        populated
"""

import os
import sys
import importlib
import pytest
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Skip the whole module if sklearn isn't installed — the detector
# itself degrades gracefully in production but the tests are about
# behavior when sklearn IS available.
sklearn = pytest.importorskip("sklearn")

from qc_types.discovery.isolation_forest_checks import (
    IsolationForestChecks)


LONG_FORMAT_COLUMNS = [
    'subjectid', 'network', 'timepoint', 'cohort', 'event_date',
    'source_form', 'variable', 'value', 'value_numeric',
    'is_missing_code',
]


def _make_long_df(rows):
    full = []
    for r in rows:
        full.append({
            'subjectid': r['subjectid'],
            'network': r.get('network', 'TEST'),
            'timepoint': r.get('timepoint', 'baseline'),
            'cohort': '',
            'event_date': '',
            'source_form': r.get('source_form', 'form_a'),
            'variable': r['variable'],
            'value': r['value'],
            'value_numeric': r['value_numeric'],
            'is_missing_code': r['is_missing_code'],
        })
    return pd.DataFrame(full, columns=LONG_FORMAT_COLUMNS)


@pytest.fixture
def detector_factory(tmp_path, monkeypatch):
    deps = tmp_path / "dependencies"
    out = tmp_path / "output"
    deps.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)

    class FakeUtils:
        absolute_path = str(tmp_path)

        def create_timepoint_list(self):
            return ['screening', 'baseline', 'month1', 'month2']

    import qc_types.discovery.isolation_forest_checks as m
    monkeypatch.setattr(m, 'Utils', FakeUtils)

    def _make(**overrides):
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "isolation_forest": {
                    "networks": ["TEST"],
                    "min_vars": 3,
                    "max_vars": 40,
                    "min_obs_per_variable": 30,
                    "min_joint_obs": 30,
                    "n_estimators": 50,
                    "contamination": "auto",
                    "residual_z_threshold": 4.0,
                    "random_state": 42,
                    **overrides,
                }
            },
        }
        return IsolationForestChecks(config_dict=cfg)

    return _make


def _gaussian_cluster_rows(
    n_subjects=40, vars_=('v1', 'v2', 'v3', 'v4'),
    center=0.0, sigma=1.0, source_form='form_a', seed=42,
):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        sid = f"S{i+1:03d}"
        for var in vars_:
            val = float(rng.normal(center, sigma))
            rows.append({
                'subjectid': sid, 'timepoint': 'baseline',
                'variable': var, 'source_form': source_form,
                'value': str(val), 'value_numeric': val,
                'is_missing_code': False,
            })
    return rows


# ---------------------------------------------------------------- IF-1

def test_if1_extreme_outlier_flagged(detector_factory):
    detector = detector_factory()
    rows = _gaussian_cluster_rows(n_subjects=40)
    # OUT: extreme on multiple dimensions simultaneously — IF should
    # isolate it in very few splits.
    for var, val in zip(
        ('v1', 'v2', 'v3', 'v4'), (50.0, 50.0, -50.0, 50.0)):
        rows.append({
            'subjectid': 'OUT', 'timepoint': 'baseline',
            'variable': var, 'source_form': 'form_a',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (result['subjectid'] == 'OUT').sum() >= 1


# ---------------------------------------------------------------- IF-2

def test_if2_clean_cohort_no_flags(detector_factory):
    detector = detector_factory()
    rows = _gaussian_cluster_rows(n_subjects=40)
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    # A clean Gaussian cohort shouldn't produce robust-z |z| ≥ 4
    # against a properly-fit IF. Some noise rows may still flag —
    # ensure we don't get an avalanche.
    assert len(result) <= 2


# ---------------------------------------------------------------- IF-3

def test_if3_too_few_joint_obs(detector_factory):
    detector = detector_factory()
    rows = _gaussian_cluster_rows(n_subjects=20)  # below min_joint
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0


# ---------------------------------------------------------------- IF-4

def test_if4_excluded_source_form(detector_factory):
    detector = detector_factory()
    rows = _gaussian_cluster_rows(
        n_subjects=40, source_form='digital_biomarkers_axivity_checkin')
    for var, val in zip(
        ('v1', 'v2', 'v3', 'v4'), (50.0, 50.0, -50.0, 50.0)):
        rows.append({
            'subjectid': 'OUT', 'timepoint': 'baseline',
            'variable': var,
            'source_form': 'digital_biomarkers_axivity_checkin',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['rows_excluded_by_form_pattern'] > 0


# ---------------------------------------------------------------- IF-5

def test_if5_severity_normalized_and_top_features(detector_factory):
    detector = detector_factory()
    rows = _gaussian_cluster_rows(n_subjects=40)
    for var, val in zip(
        ('v1', 'v2', 'v3', 'v4'), (50.0, 50.0, -50.0, 50.0)):
        rows.append({
            'subjectid': 'OUT', 'timepoint': 'baseline',
            'variable': var, 'source_form': 'form_a',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) >= 1
    assert 'severity_normalized' in result.columns
    assert (result['severity_normalized'] >= 0).all()
    assert (result['severity_normalized'] <= 100).all()
    # Every flag carries a contributing-variables string that names
    # at least one of the variables actually present in the cohort.
    contributing = result[
        result['subjectid'] == 'OUT'
    ].iloc[0]['top_contributing_variables']
    assert any(
        v in contributing for v in ('v1', 'v2', 'v3', 'v4'))
