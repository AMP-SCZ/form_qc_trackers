"""
Tests for qc_types/discovery/mahalanobis_checks.py.

Cases:
  MH-1: 3 correlated cohort vars + 1 outlier triple → 1+ flag
  MH-2: cohort below min_joint_obs → skipped
  MH-3: form with fewer eligible vars than min_tuple_size → skipped
  MH-4: excluded source_form (digital_biomarkers_*) → 0 flags
  MH-5: severity_normalized populated in [0, 100]
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

from qc_types.discovery.mahalanobis_checks import MahalanobisChecks


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


def _correlated_triplet_rows(
    n_subjects=40, vars_=('a', 'b', 'c'), source_form='form_a',
    seed=42,
):
    """
    n_subjects each contributing values for (a, b, c) where
    b = 0.9*a + small noise and c = 0.85*a + 1.2 + small noise.
    All three are strongly intercorrelated → ideal joint cluster
    for Mahalanobis detection.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        sid = f"S{i+1:03d}"
        a = float(50.0 + (i * 0.7) % 30.0)
        b = 0.9 * a + float(rng.normal(0, 0.3))
        c = 0.85 * a + 1.2 + float(rng.normal(0, 0.4))
        for var, val in zip(vars_, (a, b, c)):
            rows.append({
                'subjectid': sid, 'timepoint': 'baseline',
                'variable': var, 'source_form': source_form,
                'value': str(val), 'value_numeric': val,
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

    import qc_types.discovery.mahalanobis_checks as detector_module
    monkeypatch.setattr(detector_module, 'Utils', FakeUtils)

    def _make(**overrides):
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "mahalanobis": {
                    "networks": ["TEST"],
                    "max_tuple_size": 5,
                    "min_tuple_size": 3,
                    "min_obs_per_variable": 30,
                    "min_joint_obs": 30,
                    "shrinkage_lambda": 0.01,
                    "residual_z_threshold": 4.0,
                    **overrides,
                }
            },
        }
        return MahalanobisChecks(config_dict=cfg)

    return _make


# ---------------------------------------------------------------- MH-1

def test_mh1_multivariate_outlier_fires(detector_factory):
    detector = detector_factory()
    rows = _correlated_triplet_rows(n_subjects=40)
    # OUT: each variable plausible alone (a=60, b=60, c=60) but the
    # joint pattern is impossible because in cohort b ≈ 0.9*a and
    # c ≈ 0.85*a + 1.2 — so for a=60 the joint expected is roughly
    # (60, 54, 52); OUT's (60, 60, 60) sits far from the cluster.
    rows += [
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': 'a', 'source_form': 'form_a',
         'value': '60', 'value_numeric': 60.0,
         'is_missing_code': False},
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': 'b', 'source_form': 'form_a',
         'value': '60', 'value_numeric': 60.0,
         'is_missing_code': False},
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': 'c', 'source_form': 'form_a',
         'value': '60', 'value_numeric': 60.0,
         'is_missing_code': False},
    ]
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    flagged = result[result['subjectid'] == 'OUT']
    assert len(flagged) >= 1
    r = flagged.iloc[0]
    assert 'a' in r['tuple_variables']
    assert 'b' in r['tuple_variables']
    assert 'c' in r['tuple_variables']


# ---------------------------------------------------------------- MH-2

def test_mh2_insufficient_joint_obs(detector_factory):
    detector = detector_factory()
    rows = _correlated_triplet_rows(n_subjects=10)  # below threshold
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert (
        detector._counters['form_groups_skipped_too_few_vars'] >= 1
        or detector._counters['form_groups_skipped_too_few_joint']
        >= 1)


# ---------------------------------------------------------------- MH-3

def test_mh3_too_few_eligible_vars(detector_factory):
    detector = detector_factory(min_tuple_size=4)
    rows = _correlated_triplet_rows(n_subjects=40)  # only 3 vars
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    # Greedy picks 3 vars, fewer than min_tuple_size=4 → skipped.
    assert detector._counters['form_groups_skipped_too_few_vars'] >= 1


# ---------------------------------------------------------------- MH-4

def test_mh4_excluded_source_form(detector_factory):
    detector = detector_factory()
    rows = _correlated_triplet_rows(
        n_subjects=40, source_form='digital_biomarkers_axivity_checkin')
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['rows_excluded_by_form_pattern'] > 0


# ---------------------------------------------------------------- MH-5

def test_mh5_severity_normalized_present(detector_factory):
    detector = detector_factory()
    rows = _correlated_triplet_rows(n_subjects=40)
    rows += [
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': v, 'source_form': 'form_a',
         'value': '60', 'value_numeric': 60.0,
         'is_missing_code': False}
        for v in ('a', 'b', 'c')
    ]
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) >= 1
    assert 'severity_normalized' in result.columns
    assert (result['severity_normalized'] >= 0).all()
    assert (result['severity_normalized'] <= 100).all()


# ---------------------------------------------------------------- MH-6

def test_mh6_mcd_robust_when_cohort_contaminated(detector_factory):
    """
    Plant a 'clean' cohort of 40 subjects + 12 'fake outlier'
    subjects whose joint pattern is off the line, and one extra
    target outlier OUT. Sample-mean Mahalanobis pulls the centroid
    toward the contamination cluster and risks missing OUT; MCD
    discards the contaminated subset, so its centroid stays on
    the clean cohort and OUT gets clearly flagged.

    Concrete check: with covariance_estimator='mcd' the OUT subject
    is among the flagged subjects.
    """
    import numpy as np
    detector = detector_factory(
        covariance_estimator='mcd',
        mcd_subset_fraction=0.75,
    )
    rng = np.random.default_rng(7)
    rows = []
    # Clean cohort: b ≈ 0.9*a, c ≈ 0.85*a + 1.2.
    for i in range(40):
        sid = f"C{i+1:03d}"
        a = float(50 + (i * 0.7) % 30)
        b = 0.9 * a + float(rng.normal(0, 0.3))
        c = 0.85 * a + 1.2 + float(rng.normal(0, 0.4))
        for v, val in zip(('a', 'b', 'c'), (a, b, c)):
            rows.append({
                'subjectid': sid, 'timepoint': 'baseline',
                'variable': v, 'source_form': 'form_a',
                'value': str(val), 'value_numeric': val,
                'is_missing_code': False,
            })
    # Contaminated 'shadow cluster' offset from the line.
    for i in range(12):
        sid = f"X{i+1:03d}"
        a = float(60 + i * 0.3)
        b = 0.9 * a + 8.0  # offset b by ~+8
        c = 0.85 * a + 1.2 - 6.0  # offset c by ~-6
        for v, val in zip(('a', 'b', 'c'), (a, b, c)):
            rows.append({
                'subjectid': sid, 'timepoint': 'baseline',
                'variable': v, 'source_form': 'form_a',
                'value': str(val), 'value_numeric': val,
                'is_missing_code': False,
            })
    # The target outlier — far from BOTH clusters.
    rows += [
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': v, 'source_form': 'form_a',
         'value': '60', 'value_numeric': 60.0,
         'is_missing_code': False}
        for v in ('a', 'b', 'c')
    ]
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    flagged = set(result['subjectid'])
    assert 'OUT' in flagged
    # MCD should have run multiple concentration steps.
    assert detector._counters[
        'mcd_concentration_iterations_total'] >= 1


# ---------------------------------------------------------------- MH-7

def test_mh7_invalid_covariance_estimator_raises(detector_factory):
    with pytest.raises(RuntimeError) as excinfo:
        detector_factory(covariance_estimator='unknown')
    assert 'covariance_estimator' in str(excinfo.value)
