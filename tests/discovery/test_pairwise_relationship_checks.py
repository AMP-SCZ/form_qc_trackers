"""
Tests for qc_types/discovery/pairwise_relationship_checks.py.

Cases:
  PR-1: cohort with strong correlation between (height, weight); a
        single subject with anomalous pair (tall, very-low-weight)
        → 1+ flag
  PR-2: weakly correlated pair (|r| below threshold) → 0 flags even
        with a synthetic outlier
  PR-3: pair with insufficient joint observations → skipped
  PR-4: pair with one constant variable → skipped
  PR-5: excluded source_form (digital_biomarkers_*) → 0 flags
  PR-6: excluded variable pattern (*_complete) → 0 flags
  PR-7: per-pair cap caps emission from one over-firing pair
  Schema: missing required col → RuntimeError
  Empty:  no flags → DataFrame with declared schema/dtypes
  File:   end-to-end run() produces a readable parquet; no .tmp residue
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

from qc_types.discovery.pairwise_relationship_checks import (
    PairwiseRelationshipChecks)


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


def _correlated_pair_rows(
    n=40, var_a='height_cm', var_b='weight_kg',
    source_form='form_a', seed=42,
):
    """
    n subjects with var_a uniformly in [150, 195] and
    var_b = 0.7 * (var_a - 150) + 50 + small_noise → strong linear
    correlation, no NaNs. Each subject's row appears at 'baseline'.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sid = f"S{i+1:03d}"
        a = float(150 + (i * 1.1) % 45)
        # Small deterministic-ish noise so the cohort MAD of residuals
        # isn't zero.
        b = 0.7 * (a - 150) + 50 + float(rng.normal(0, 0.4))
        rows.append({
            'subjectid': sid, 'timepoint': 'baseline',
            'variable': var_a, 'source_form': source_form,
            'value': str(a), 'value_numeric': a,
            'is_missing_code': False,
        })
        rows.append({
            'subjectid': sid, 'timepoint': 'baseline',
            'variable': var_b, 'source_form': source_form,
            'value': str(b), 'value_numeric': b,
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

    import qc_types.discovery.pairwise_relationship_checks \
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
                "pairwise_relationship": {
                    "networks": ["TEST"],
                    "min_obs_per_pair": 30,
                    "min_correlation": 0.6,
                    "residual_z_threshold": 4.0,
                    "max_flags_per_variable": None,
                    **overrides,
                }
            },
        }
        return PairwiseRelationshipChecks(config_dict=cfg)

    return _make


# ---------------------------------------------------------------- PR-1

def test_pr1_correlated_pair_with_outlier(detector_factory):
    detector = detector_factory()
    rows = _correlated_pair_rows(n=40)
    # Outlier: tall (190 cm) but very low weight (10 kg) — way off
    # the line for the cohort.
    rows += [
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': 'height_cm', 'source_form': 'form_a',
         'value': '190', 'value_numeric': 190.0,
         'is_missing_code': False},
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': 'weight_kg', 'source_form': 'form_a',
         'value': '10', 'value_numeric': 10.0,
         'is_missing_code': False},
    ]
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    out_rows = result[result['subjectid'] == 'OUT']
    assert len(out_rows) >= 1
    r = out_rows.iloc[0]
    # Both vars must appear in related/affected.
    assert 'height_cm' in r['affected_variables']
    assert 'weight_kg' in r['affected_variables']
    assert r['correlation_r'] > 0.6


# ---------------------------------------------------------------- PR-2

def test_pr2_weak_correlation_skipped(detector_factory):
    detector = detector_factory()
    rng = np.random.default_rng(7)
    rows = []
    for i in range(40):
        sid = f"S{i+1:03d}"
        rows.append({
            'subjectid': sid, 'timepoint': 'baseline',
            'variable': 'noisy_a', 'source_form': 'form_a',
            'value': str(rng.normal(50, 5)),
            'value_numeric': float(rng.normal(50, 5)),
            'is_missing_code': False,
        })
        rows.append({
            'subjectid': sid, 'timepoint': 'baseline',
            'variable': 'noisy_b', 'source_form': 'form_a',
            'value': str(rng.normal(100, 5)),
            'value_numeric': float(rng.normal(100, 5)),
            'is_missing_code': False,
        })
    # Outlier with extreme values — should NOT fire because the
    # pair isn't correlated above threshold.
    rows += [
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': 'noisy_a', 'source_form': 'form_a',
         'value': '999', 'value_numeric': 999.0,
         'is_missing_code': False},
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': 'noisy_b', 'source_form': 'form_a',
         'value': '-999', 'value_numeric': -999.0,
         'is_missing_code': False},
    ]
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['pairs_skipped_low_correlation'] >= 1


# ---------------------------------------------------------------- PR-3

def test_pr3_insufficient_joint_skipped(detector_factory):
    detector = detector_factory(min_obs_per_pair=30)
    rows = _correlated_pair_rows(n=20)  # below threshold
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['pairs_skipped_insufficient_joint'] >= 1


# ---------------------------------------------------------------- PR-4

def test_pr4_constant_variable_skipped(detector_factory):
    detector = detector_factory()
    rows = []
    for i in range(40):
        sid = f"S{i+1:03d}"
        rows.append({
            'subjectid': sid, 'timepoint': 'baseline',
            'variable': 'flat', 'source_form': 'form_a',
            'value': '5', 'value_numeric': 5.0,
            'is_missing_code': False,
        })
        rows.append({
            'subjectid': sid, 'timepoint': 'baseline',
            'variable': 'varied', 'source_form': 'form_a',
            'value': str(i), 'value_numeric': float(i),
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['pairs_skipped_constant_variable'] >= 1


# ---------------------------------------------------------------- PR-5

def test_pr5_excluded_source_form(detector_factory):
    detector = detector_factory()
    rows = _correlated_pair_rows(
        n=40, source_form='digital_biomarkers_axivity_checkin')
    rows += [
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': 'height_cm',
         'source_form': 'digital_biomarkers_axivity_checkin',
         'value': '190', 'value_numeric': 190.0,
         'is_missing_code': False},
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': 'weight_kg',
         'source_form': 'digital_biomarkers_axivity_checkin',
         'value': '10', 'value_numeric': 10.0,
         'is_missing_code': False},
    ]
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['rows_excluded_by_form_pattern'] > 0


# ---------------------------------------------------------------- PR-6

def test_pr6_excluded_variable_pattern(detector_factory):
    detector = detector_factory()
    rows = _correlated_pair_rows(
        n=40, var_a='thing_complete', var_b='weight_kg')
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['rows_excluded_by_variable_pattern'] > 0


# ---------------------------------------------------------------- PR-7

def test_pr7_per_pair_cap(detector_factory):
    """
    Multiple subjects are extreme outliers from the cohort line; with
    max_flags_per_pair=2, only the top 2 are kept.
    """
    detector = detector_factory(max_flags_per_pair=2)
    rows = _correlated_pair_rows(n=40)
    # Five outliers off the line.
    for k, sid in enumerate(['O1', 'O2', 'O3', 'O4', 'O5']):
        a = 170.0
        b = 5.0 + 5.0 * k  # very low weights, well below the line
        rows += [
            {'subjectid': sid, 'timepoint': 'baseline',
             'variable': 'height_cm', 'source_form': 'form_a',
             'value': str(a), 'value_numeric': a,
             'is_missing_code': False},
            {'subjectid': sid, 'timepoint': 'baseline',
             'variable': 'weight_kg', 'source_form': 'form_a',
             'value': str(b), 'value_numeric': b,
             'is_missing_code': False},
        ]
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    # Capped at 2 from that one pair.
    assert len(result) == 2
    assert detector._counters['flags_dropped_by_per_pair_cap'] >= 3


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
    expected_cols = list(PairwiseRelationshipChecks.OUTPUT_COLUMNS)
    assert list(result.columns) == expected_cols


# ---------------------------------------------------------------- File

def test_file_output_atomic(detector_factory, tmp_path):
    detector = detector_factory()
    rows = _correlated_pair_rows(n=40)
    rows += [
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': 'height_cm', 'source_form': 'form_a',
         'value': '190', 'value_numeric': 190.0,
         'is_missing_code': False},
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': 'weight_kg', 'source_form': 'form_a',
         'value': '10', 'value_numeric': 10.0,
         'is_missing_code': False},
    ]
    long_path = (
        tmp_path / "dependencies"
        / "multi_tp_TEST_numeric_long.parquet")
    _make_long_df(rows).to_parquet(long_path, index=False)
    detector.run()
    out_path = (
        tmp_path / "output" / "combined_outputs"
        / "discovery" / "pairwise_relationship_candidates.parquet")
    assert out_path.exists()
    out_df = pd.read_parquet(out_path)
    assert (out_df['subjectid'] == 'OUT').sum() >= 1
    tmps = list(out_path.parent.glob("*.tmp"))
    assert len(tmps) == 0
