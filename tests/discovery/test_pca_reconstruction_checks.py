"""
Tests for qc_types/discovery/pca_reconstruction_checks.py.

Cases:
  PC-1: cohort with strong linear structure + outlier whose pattern
        is orthogonal to the leading subspace → 1+ flag
  PC-2: too few variables → skipped
  PC-3: too few joint observations → skipped
  PC-4: excluded source_form → 0 flags
  PC-5: severity_normalized in [0, 100]
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

from qc_types.discovery.pca_reconstruction_checks import (
    PCAReconstructionChecks)


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

    import qc_types.discovery.pca_reconstruction_checks as m
    monkeypatch.setattr(m, 'Utils', FakeUtils)

    def _make(**overrides):
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "pca_reconstruction": {
                    "networks": ["TEST"],
                    "min_vars": 3,
                    "max_vars": 40,
                    "min_obs_per_variable": 30,
                    "min_joint_obs": 30,
                    "variance_explained_target": 0.90,
                    "max_components": 10,
                    "residual_z_threshold": 4.0,
                    **overrides,
                }
            },
        }
        return PCAReconstructionChecks(config_dict=cfg)

    return _make


def _linear_subspace_rows(n_subjects=40, seed=42):
    """
    Cohort lives on a 1-d line in 4-d space:
    v1 = t + noise, v2 = 0.9 t + noise, v3 = -0.8 t + noise,
    v4 = 1.1 t + noise.
    A single low-rank cluster — perfect for PCA reconstruction.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        sid = f"S{i+1:03d}"
        t = float((i * 0.7) % 20.0)
        for var, mul in (('v1', 1.0), ('v2', 0.9),
                         ('v3', -0.8), ('v4', 1.1)):
            val = mul * t + float(rng.normal(0, 0.2))
            rows.append({
                'subjectid': sid, 'timepoint': 'baseline',
                'variable': var, 'source_form': 'form_a',
                'value': str(val), 'value_numeric': val,
                'is_missing_code': False,
            })
    return rows


# ---------------------------------------------------------------- PC-1

def test_pc1_off_subspace_outlier(detector_factory):
    detector = detector_factory()
    rows = _linear_subspace_rows(n_subjects=40)
    # OUT lies far OFF the 1-d line: v1=10, v2=10, v3=10, v4=10.
    # The cohort's leading PC is the (1, 0.9, -0.8, 1.1) direction;
    # all-equal entries are orthogonal to that and leave a large
    # reconstruction residual.
    rows += [
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': v, 'source_form': 'form_a',
         'value': '10', 'value_numeric': 10.0,
         'is_missing_code': False}
        for v in ('v1', 'v2', 'v3', 'v4')
    ]
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (result['subjectid'] == 'OUT').sum() >= 1


# ---------------------------------------------------------------- PC-2

def test_pc2_too_few_vars(detector_factory):
    detector = detector_factory(min_vars=10)
    rows = _linear_subspace_rows(n_subjects=40)  # only 4 vars
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['form_groups_skipped_too_few_vars'] >= 1


# ---------------------------------------------------------------- PC-3

def test_pc3_too_few_joint_obs(detector_factory):
    detector = detector_factory()
    rows = _linear_subspace_rows(n_subjects=20)  # below min_joint
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0


# ---------------------------------------------------------------- PC-4

def test_pc4_excluded_source_form(detector_factory):
    detector = detector_factory()
    rows = _linear_subspace_rows(n_subjects=40)
    for r in rows:
        r['source_form'] = 'digital_biomarkers_axivity_checkin'
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['rows_excluded_by_form_pattern'] > 0


# ---------------------------------------------------------------- PC-5

def test_pc5_severity_normalized_present(detector_factory):
    detector = detector_factory()
    rows = _linear_subspace_rows(n_subjects=40)
    rows += [
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': v, 'source_form': 'form_a',
         'value': '10', 'value_numeric': 10.0,
         'is_missing_code': False}
        for v in ('v1', 'v2', 'v3', 'v4')
    ]
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) >= 1
    assert 'severity_normalized' in result.columns
    assert (result['severity_normalized'] >= 0).all()
    assert (result['severity_normalized'] <= 100).all()
