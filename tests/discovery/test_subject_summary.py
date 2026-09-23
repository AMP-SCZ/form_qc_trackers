"""
Tests for qc_types/discovery/subject_summary.py.

Cases:
  SS-1: two detector parquets exist; aggregator combines per-subject
        severities; top row matches the highest-total subject
  SS-2: no detector parquets → empty summary with declared schema
  SS-3: corrupt detector parquet → skipped with warning, rest still
        aggregate
  SS-4: subject column blank / 'aggregate'-style rows are dropped
  SS-5: max_subjects cap honored
"""

import os
import sys
import json
import pytest
import pandas as pd

_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc_types.discovery.subject_summary import SubjectSummary


def _make_detector_row(
    subjectid='S001', network='TEST', sev=10.0, msg='test',
    algorithm='test_detector',
):
    return {
        'subject': subjectid, 'subjectid': subjectid,
        'network': network, 'timepoint': 'baseline',
        'variable': 'v', 'source_form': 'f',
        'algorithm': algorithm,
        'severity_score': sev,
        'severity_normalized': sev,
        'error_message': msg,
    }


@pytest.fixture
def summary_factory(tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir(exist_ok=True)
    disc = out / "combined_outputs" / "discovery"
    disc.mkdir(parents=True, exist_ok=True)

    class FakeUtils:
        absolute_path = str(tmp_path)

        def create_timepoint_list(self):
            return ['screening', 'baseline', 'month1']

    import qc_types.discovery.subject_summary as detector_module
    monkeypatch.setattr(detector_module, 'Utils', FakeUtils)

    def _make(**overrides):
        cfg = {
            "paths": {
                "dependencies_path": str(tmp_path) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "subject_summary": {
                    "max_subjects": None,
                    **overrides,
                }
            },
        }
        return SubjectSummary(config_dict=cfg), disc

    return _make


def _write_parquet(disc_dir, fname, rows):
    path = disc_dir / fname
    if not rows:
        df = pd.DataFrame({
            'subject': pd.Series(dtype='object'),
            'subjectid': pd.Series(dtype='object'),
            'network': pd.Series(dtype='object'),
            'severity_normalized': pd.Series(dtype='float64'),
            'algorithm': pd.Series(dtype='object'),
            'error_message': pd.Series(dtype='object'),
        })
    else:
        df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------- SS-1

def test_ss1_aggregates_across_detectors(summary_factory):
    ss, disc = summary_factory()
    _write_parquet(disc, 'point_spike_candidates.parquet', [
        _make_detector_row('S001', sev=80.0,
                           algorithm='point_spike',
                           msg='spike at month1'),
        _make_detector_row('S002', sev=30.0,
                           algorithm='point_spike',
                           msg='spike at baseline'),
    ])
    _write_parquet(disc, 'pairwise_relationship_candidates.parquet', [
        _make_detector_row('S001', sev=60.0,
                           algorithm='pairwise_relationship',
                           msg='off line'),
        _make_detector_row('S003', sev=50.0,
                           algorithm='pairwise_relationship',
                           msg='off line'),
    ])
    ss.run()
    result_path = disc / "subject_anomaly_summary.parquet"
    out = pd.read_parquet(result_path)
    # S001 has total 80+60=140; should rank first.
    assert out.iloc[0]['subject'] == 'S001'
    assert out.iloc[0]['total_flags'] == 2
    assert out.iloc[0]['total_severity_normalized'] == 140.0
    assert out.iloc[0]['detectors_firing'] == 2
    # Top detector for S001 is the spike (higher single sev=80).
    assert out.iloc[0]['top_detector'] == 'point_spike'


# ---------------------------------------------------------------- SS-2

def test_ss2_empty_when_no_detectors(summary_factory):
    ss, disc = summary_factory()
    ss.run()
    result_path = disc / "subject_anomaly_summary.parquet"
    out = pd.read_parquet(result_path)
    assert len(out) == 0
    # All declared columns present, dtypes intact.
    expected_cols = list(SubjectSummary.OUTPUT_COLUMNS)
    assert list(out.columns) == expected_cols


# ---------------------------------------------------------------- SS-3

def test_ss3_corrupt_parquet_skipped(summary_factory, tmp_path):
    ss, disc = summary_factory()
    # Write a non-parquet file with the expected name.
    bogus = disc / 'point_spike_candidates.parquet'
    bogus.write_bytes(b'not a parquet file at all')
    _write_parquet(disc, 'pairwise_relationship_candidates.parquet', [
        _make_detector_row('S001', sev=50.0,
                           algorithm='pairwise_relationship'),
    ])
    ss.run()
    result_path = disc / "subject_anomaly_summary.parquet"
    out = pd.read_parquet(result_path)
    # The pairwise flag for S001 must still be in the summary.
    assert (out['subject'] == 'S001').any()


# ---------------------------------------------------------------- SS-4

def test_ss4_aggregate_style_rows_dropped(summary_factory):
    """Missingness site-aggregate rows have a blank `subject` and
    a `(site_aggregate)` variable; they should not roll up into a
    subject row."""
    ss, disc = summary_factory()
    _write_parquet(disc, 'missingness_anomaly_summary.parquet', [
        {
            'subject': '',
            'subjectid': '',
            'network': 'TEST',
            'variable': '(site_aggregate)',
            'severity_normalized': 90.0,
            'algorithm': 'missingness_summary_v1',
            'error_message': 'site outlier',
        },
        _make_detector_row('S001', sev=40.0,
                           algorithm='point_spike'),
    ])
    _write_parquet(disc, 'point_spike_candidates.parquet', [
        _make_detector_row('S001', sev=20.0,
                           algorithm='point_spike'),
    ])
    ss.run()
    result_path = disc / "subject_anomaly_summary.parquet"
    out = pd.read_parquet(result_path)
    # The site-aggregate row had blank subject → dropped. S001 must
    # appear; no empty / aggregate row should.
    assert (out['subject'] == 'S001').any()
    assert (out['subject'] == '').sum() == 0
    assert (out['subject'].str.contains('aggregate', case=False)
            ).sum() == 0


# ---------------------------------------------------------------- SS-5

def test_ss5_max_subjects_cap(summary_factory):
    ss, disc = summary_factory(max_subjects=2)
    _write_parquet(disc, 'point_spike_candidates.parquet', [
        _make_detector_row(f'S{i:03d}', sev=float(50 + i))
        for i in range(5)
    ])
    ss.run()
    result_path = disc / "subject_anomaly_summary.parquet"
    out = pd.read_parquet(result_path)
    assert len(out) == 2
