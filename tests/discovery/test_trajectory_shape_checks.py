"""
Tests for qc_types/discovery/trajectory_shape_checks.py.

Cases:
  TS-1: cohort has flat-ish trajectories; one subject has a sharp V
        shape → flagged
  TS-2: cohort trajectories are linearly increasing; subject's
        trajectory matches → not flagged
  TS-3: too few tps with enough coverage → skipped
  TS-4: too few subjects → skipped
  TS-5: excluded source_form → 0 flags
  TS-6: severity_normalized in [0, 100]
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

from qc_types.discovery.trajectory_shape_checks import (
    TrajectoryShapeChecks)


LONG_FORMAT_COLUMNS = [
    'subjectid', 'network', 'timepoint', 'cohort', 'event_date',
    'source_form', 'variable', 'value', 'value_numeric',
    'is_missing_code',
]

TEST_TP_ORDER = ['screening', 'baseline', 'month1', 'month2',
                 'month3', 'month4', 'month6', 'month12']


def _make_long_df(rows):
    full = []
    for r in rows:
        full.append({
            'subjectid': r['subjectid'],
            'network': r.get('network', 'TEST'),
            'timepoint': r['timepoint'],
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
            return list(TEST_TP_ORDER)

    import qc_types.discovery.trajectory_shape_checks as m
    monkeypatch.setattr(m, 'Utils', FakeUtils)

    def _make(**overrides):
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "trajectory_shape": {
                    "networks": ["TEST"],
                    "min_tps": 3,
                    "min_obs_per_tp": 20,
                    "min_valid_tps_per_subject": 3,
                    "min_subjects": 30,
                    "variance_explained_target": 0.85,
                    "max_components": 5,
                    "residual_z_threshold": 4.0,
                    **overrides,
                }
            },
        }
        return TrajectoryShapeChecks(config_dict=cfg)

    return _make


def _flat_trajectory_cohort(n_subjects=40, variable='test_var',
                            source_form='form_a', seed=42):
    """
    n subjects with near-flat trajectories around 50 across 4 tps.
    """
    rng = np.random.default_rng(seed)
    rows = []
    tps = ['baseline', 'month1', 'month2', 'month3']
    for i in range(n_subjects):
        sid = f"S{i+1:03d}"
        baseline = 50.0 + float(rng.normal(0, 1.0))
        for tp in tps:
            val = baseline + float(rng.normal(0, 0.5))
            rows.append({
                'subjectid': sid, 'timepoint': tp,
                'variable': variable, 'source_form': source_form,
                'value': str(val), 'value_numeric': val,
                'is_missing_code': False,
            })
    return rows


def _linear_trajectory_cohort(
    n_subjects=40, variable='test_var',
    source_form='form_a', seed=42,
):
    """n subjects with linearly increasing trajectories."""
    rng = np.random.default_rng(seed)
    rows = []
    tps = ['baseline', 'month1', 'month2', 'month3']
    for i in range(n_subjects):
        sid = f"S{i+1:03d}"
        baseline = 50.0 + float(rng.normal(0, 1.0))
        slope = 2.0 + float(rng.normal(0, 0.2))
        for j, tp in enumerate(tps):
            val = baseline + slope * j + float(rng.normal(0, 0.3))
            rows.append({
                'subjectid': sid, 'timepoint': tp,
                'variable': variable, 'source_form': source_form,
                'value': str(val), 'value_numeric': val,
                'is_missing_code': False,
            })
    return rows


# ---------------------------------------------------------------- TS-1

def test_ts1_sharp_V_shape_flagged(detector_factory):
    detector = detector_factory()
    rows = _flat_trajectory_cohort(n_subjects=40)
    # V subject: large dip at month1, recovery at month2/3.
    for tp, val in zip(
        ['baseline', 'month1', 'month2', 'month3'],
        [50.0, 10.0, 50.0, 50.0],
    ):
        rows.append({
            'subjectid': 'V', 'timepoint': tp,
            'variable': 'test_var', 'source_form': 'form_a',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (result['subjectid'] == 'V').sum() >= 1


# ---------------------------------------------------------------- TS-2

def test_ts2_normal_trajectory_not_flagged(detector_factory):
    detector = detector_factory()
    rows = _linear_trajectory_cohort(n_subjects=40)
    # Plant a subject who follows the cohort pattern exactly.
    for j, tp in enumerate(
            ['baseline', 'month1', 'month2', 'month3']):
        val = 50.0 + 2.0 * j  # matches the median cohort slope
        rows.append({
            'subjectid': 'CONFORM', 'timepoint': tp,
            'variable': 'test_var', 'source_form': 'form_a',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (result['subjectid'] == 'CONFORM').sum() == 0


# ---------------------------------------------------------------- TS-3

def test_ts3_too_few_tps(detector_factory):
    detector = detector_factory(min_tps=10)
    rows = _flat_trajectory_cohort(n_subjects=40)
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters[
        'variable_groups_skipped_too_few_tps'] >= 1


# ---------------------------------------------------------------- TS-4

def test_ts4_too_few_subjects(detector_factory):
    detector = detector_factory(min_subjects=100)
    rows = _flat_trajectory_cohort(n_subjects=40)
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters[
        'variable_groups_skipped_too_few_subjects'] >= 1


# ---------------------------------------------------------------- TS-5

def test_ts5_excluded_source_form(detector_factory):
    detector = detector_factory()
    rows = _flat_trajectory_cohort(
        n_subjects=40, source_form='digital_biomarkers_axivity_checkin')
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['rows_excluded_by_form_pattern'] > 0


# ---------------------------------------------------------------- TS-6

def test_ts6_severity_normalized_present(detector_factory):
    detector = detector_factory()
    rows = _flat_trajectory_cohort(n_subjects=40)
    for tp, val in zip(
        ['baseline', 'month1', 'month2', 'month3'],
        [50.0, 10.0, 50.0, 50.0],
    ):
        rows.append({
            'subjectid': 'V', 'timepoint': tp,
            'variable': 'test_var', 'source_form': 'form_a',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) >= 1
    assert 'severity_normalized' in result.columns
    assert (result['severity_normalized'] >= 0).all()
    assert (result['severity_normalized'] <= 100).all()


# --------------------------------------------------- TS-7 evidence_timepoints

def test_ts7_evidence_timepoints_metadata(detector_factory):
    """
    Patch 6: trajectory_shape must emit evidence_timepoints and
    evidence_max_timepoint columns. For trajectory_shape the
    evidence pool is every observed scheduled tp the subject
    contributed; evidence_max_timepoint is the canonically-latest.
    """
    detector = detector_factory()
    rows = _flat_trajectory_cohort(n_subjects=40)
    for tp, val in zip(
        ['baseline', 'month1', 'month2', 'month3'],
        [50.0, 10.0, 50.0, 50.0],
    ):
        rows.append({
            'subjectid': 'V', 'timepoint': tp,
            'variable': 'test_var', 'source_form': 'form_a',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert 'evidence_timepoints' in result.columns
    assert 'evidence_max_timepoint' in result.columns
    # tps_used already contains canonical-sorted tps; the new
    # column should be a 1:1 copy.
    flag = result[result['subjectid'] == 'V'].iloc[0]
    assert flag['evidence_timepoints'] == flag['tps_used']
    # evidence_max_timepoint is the canonically-latest of the
    # comma-joined list. In our cohort that's 'month3'.
    tps = flag['evidence_timepoints'].split(',')
    assert flag['evidence_max_timepoint'] == tps[-1]
    assert flag['evidence_max_timepoint'] == 'month3'
