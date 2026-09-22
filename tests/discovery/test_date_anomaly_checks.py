"""
Tests for qc_types/discovery/date_anomaly_checks.py.

Covers:
  DA-1: monotonicity violation (date moves backward) → 1 flag
  DA-2: clean schedule, all intervals identical → 0 interval_z flags
  DA-3: per-tp-pair pool catches an interval anomaly inside one
        schedule window that would be hidden by variable-wide MAD
  DA-4: excluded source_form (digital_biomarkers_*) → 0 flags
  DA-5: excluded variable pattern → 0 flags
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

from qc_types.discovery.date_anomaly_checks import DateAnomalyChecks


LONG_FORMAT_COLUMNS = [
    'subjectid', 'network', 'timepoint', 'cohort', 'event_date',
    'source_form', 'variable', 'value', 'value_date',
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
            'cohort': r.get('cohort', ''),
            'event_date': r.get('event_date', ''),
            'source_form': r.get('source_form', 'form_a'),
            'variable': r['variable'],
            'value': r['value'],
            'value_date': pd.to_datetime(r['value_date']),
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

    import qc_types.discovery.date_anomaly_checks as detector_module
    monkeypatch.setattr(detector_module, 'Utils', FakeUtils)

    def _make(**overrides):
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "date_anomaly": {
                    "networks": ["TEST"],
                    "robust_z_threshold": 4.0,
                    "min_observations_per_variable": 30,
                    "min_intervals_per_tp_pair": 10,
                    "min_monotonicity_days_back": 1,
                    "max_flags_to_write": None,
                    **overrides,
                }
            },
        }
        return DateAnomalyChecks(config_dict=cfg)

    return _make


def _cohort_visit_dates(n_subjects=15):
    """
    n subjects each with screening / baseline / month1 / month2 dates
    on a ~30-day schedule with deterministic per-subject jitter so the
    per-(tp-pair) interval MAD is non-zero (the detector skips
    degenerate-MAD pools, and a synthetic cohort with all-identical
    intervals trips that path).
    """
    rows = []
    for i in range(n_subjects):
        sid = f"S{i+1:03d}"
        base = pd.Timestamp('2025-01-01') + pd.Timedelta(days=i)
        # Deterministic per-pair jitter pattern in {-2..+2 days}.
        jitter_by_pair = (
            (i % 5) - 2,
            ((i + 2) % 5) - 2,
            ((i + 4) % 5) - 2,
        )
        offsets = [0]
        cum = 0
        for k, base_step in enumerate((30, 30, 30)):
            cum += base_step + jitter_by_pair[k]
            offsets.append(cum)
        for j, tp in enumerate(
                ['screening', 'baseline', 'month1', 'month2']):
            d = base + pd.Timedelta(days=offsets[j])
            rows.append({
                'subjectid': sid, 'timepoint': tp,
                'variable': 'visit_date',
                'value': str(d.date()),
                'value_date': d,
                'is_missing_code': False,
            })
    return rows


# ---------------------------------------------------------------- DA-1

def test_da1_monotonicity_violation(detector_factory):
    detector = detector_factory()
    rows = _cohort_visit_dates(n_subjects=15)
    # BACKWARDS: month1 date is BEFORE baseline.
    rows += [
        {'subjectid': 'BACK', 'timepoint': 'screening',
         'variable': 'visit_date',
         'value': '2025-01-01', 'value_date': pd.Timestamp('2025-01-01'),
         'is_missing_code': False},
        {'subjectid': 'BACK', 'timepoint': 'baseline',
         'variable': 'visit_date',
         'value': '2025-02-01', 'value_date': pd.Timestamp('2025-02-01'),
         'is_missing_code': False},
        {'subjectid': 'BACK', 'timepoint': 'month1',
         'variable': 'visit_date',
         'value': '2024-12-01', 'value_date': pd.Timestamp('2024-12-01'),
         'is_missing_code': False},
        {'subjectid': 'BACK', 'timepoint': 'month2',
         'variable': 'visit_date',
         'value': '2025-04-01', 'value_date': pd.Timestamp('2025-04-01'),
         'is_missing_code': False},
    ]
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    back = result[result['subjectid'] == 'BACK']
    mono_flags = back[back['algorithm'] == 'date_monotonicity']
    assert len(mono_flags) >= 1


# ---------------------------------------------------------------- DA-2

def test_da2_clean_schedule_zero_interval_flags(detector_factory):
    detector = detector_factory()
    rows = _cohort_visit_dates(n_subjects=15)
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    iv_flags = result[result['algorithm'] == 'date_interval_z']
    assert len(iv_flags) == 0


# ---------------------------------------------------------------- DA-3

def test_da3_per_tp_pair_pool_catches_window_anomaly(
    detector_factory,
):
    """
    Each subject has 3 adjacent tp-pair intervals (~30 days each).
    OUTLIER subject has a wildly large screening→baseline interval
    (300 days). Under the per-tp-pair pool this fires; under a
    variable-wide pool it would be hidden by the mixed scale.
    """
    detector = detector_factory(min_intervals_per_tp_pair=10)
    rows = _cohort_visit_dates(n_subjects=15)
    # OUTLIER subject — huge first interval.
    rows += [
        {'subjectid': 'OUT', 'timepoint': 'screening',
         'variable': 'visit_date',
         'value': '2025-01-01',
         'value_date': pd.Timestamp('2025-01-01'),
         'is_missing_code': False},
        {'subjectid': 'OUT', 'timepoint': 'baseline',
         'variable': 'visit_date',
         'value': '2025-11-01',
         'value_date': pd.Timestamp('2025-11-01'),
         'is_missing_code': False},
        {'subjectid': 'OUT', 'timepoint': 'month1',
         'variable': 'visit_date',
         'value': '2025-12-01',
         'value_date': pd.Timestamp('2025-12-01'),
         'is_missing_code': False},
        {'subjectid': 'OUT', 'timepoint': 'month2',
         'variable': 'visit_date',
         'value': '2026-01-01',
         'value_date': pd.Timestamp('2026-01-01'),
         'is_missing_code': False},
    ]
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    out_rows = result[
        (result['subjectid'] == 'OUT')
        & (result['algorithm'] == 'date_interval_z')]
    assert len(out_rows) >= 1


# ---------------------------------------------------------------- DA-4

def test_da4_excluded_source_form(detector_factory):
    detector = detector_factory()
    rows = []
    base = pd.Timestamp('2025-01-01')
    for i in range(15):
        sid = f"S{i+1:03d}"
        for j, tp in enumerate(
                ['screening', 'baseline', 'month1', 'month2']):
            rows.append({
                'subjectid': sid, 'timepoint': tp,
                'variable': 'device_date',
                'source_form': 'digital_biomarkers_axivity_checkin',
                'value': '2025-01-01',
                'value_date': base + pd.Timedelta(days=j * 30 + i),
                'is_missing_code': False,
            })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['rows_excluded_by_form_pattern'] > 0


# ---------------------------------------------------------------- DA-5

def test_da5_excluded_variable_pattern(detector_factory):
    # Opt the default `_date` exclusion back on for this test.
    detector = detector_factory(
        excluded_variable_patterns=['_date'])
    rows = _cohort_visit_dates(n_subjects=15)
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['rows_excluded_by_variable_pattern'] > 0
