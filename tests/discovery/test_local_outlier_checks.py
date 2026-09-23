"""
Tests for qc_types/discovery/local_outlier_checks.py.

Cases:
  LO-1: cohort forms two dense pockets; subject sits between them →
        large k-NN distance → flagged
  LO-2: subject in dense pocket → not flagged
  LO-3: too few joint observations → skipped
  LO-4: excluded source_form → 0 flags
  LO-5: severity_normalized in [0, 100]
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

from qc_types.discovery.local_outlier_checks import LocalOutlierChecks


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

    import qc_types.discovery.local_outlier_checks as m
    monkeypatch.setattr(m, 'Utils', FakeUtils)

    def _make(**overrides):
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "local_outlier": {
                    "networks": ["TEST"],
                    "n_neighbors": 5,
                    "min_vars": 3,
                    "max_vars": 30,
                    "min_obs_per_variable": 30,
                    "min_joint_obs": 30,
                    "max_form_rows": 10000,
                    "residual_z_threshold": 4.0,
                    **overrides,
                }
            },
        }
        return LocalOutlierChecks(config_dict=cfg)

    return _make


def _bimodal_cohort(n_per_cluster=20, seed=42):
    """
    Two dense Gaussian clusters in 4-d:
      cluster A at (0, 0, 0, 0), σ=0.5
      cluster B at (10, 10, 10, 10), σ=0.5
    """
    rng = np.random.default_rng(seed)
    rows = []
    sid_n = 0
    for center in (np.zeros(4), np.full(4, 10.0)):
        for _ in range(n_per_cluster):
            sid_n += 1
            sid = f"S{sid_n:03d}"
            x = center + rng.normal(0, 0.5, size=4)
            for v, val in zip(('v1', 'v2', 'v3', 'v4'), x):
                rows.append({
                    'subjectid': sid, 'timepoint': 'baseline',
                    'variable': v, 'source_form': 'form_a',
                    'value': str(val), 'value_numeric': float(val),
                    'is_missing_code': False,
                })
    return rows


# ---------------------------------------------------------------- LO-1

def test_lo1_between_clusters_flagged(detector_factory):
    detector = detector_factory()
    rows = _bimodal_cohort(n_per_cluster=20)
    # MID subject sits halfway between the clusters — far from all
    # neighbors in both, so its 5th-NN distance is large.
    for v, val in zip(('v1', 'v2', 'v3', 'v4'),
                      (5.0, 5.0, 5.0, 5.0)):
        rows.append({
            'subjectid': 'MID', 'timepoint': 'baseline',
            'variable': v, 'source_form': 'form_a',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (result['subjectid'] == 'MID').sum() >= 1


# ---------------------------------------------------------------- LO-2

def test_lo2_dense_pocket_not_flagged(detector_factory):
    detector = detector_factory()
    rows = _bimodal_cohort(n_per_cluster=20)
    # NORMAL subject sits at cluster A centroid → tight neighborhood.
    for v, val in zip(('v1', 'v2', 'v3', 'v4'),
                      (0.0, 0.0, 0.0, 0.0)):
        rows.append({
            'subjectid': 'NORMAL', 'timepoint': 'baseline',
            'variable': v, 'source_form': 'form_a',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (result['subjectid'] == 'NORMAL').sum() == 0


# ---------------------------------------------------------------- LO-3

def test_lo3_too_few_joint(detector_factory):
    detector = detector_factory()
    rows = _bimodal_cohort(n_per_cluster=10)  # 20 joint < 30
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0


# ---------------------------------------------------------------- LO-4

def test_lo4_excluded_source_form(detector_factory):
    detector = detector_factory()
    rows = _bimodal_cohort(n_per_cluster=20)
    for r in rows:
        r['source_form'] = 'digital_biomarkers_axivity_checkin'
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['rows_excluded_by_form_pattern'] > 0


# ---------------------------------------------------------------- LO-5

def test_lo5_severity_normalized_present(detector_factory):
    detector = detector_factory()
    rows = _bimodal_cohort(n_per_cluster=20)
    for v, val in zip(('v1', 'v2', 'v3', 'v4'),
                      (5.0, 5.0, 5.0, 5.0)):
        rows.append({
            'subjectid': 'MID', 'timepoint': 'baseline',
            'variable': v, 'source_form': 'form_a',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) >= 1
    assert 'severity_normalized' in result.columns
    assert (result['severity_normalized'] >= 0).all()
    assert (result['severity_normalized'] <= 100).all()
