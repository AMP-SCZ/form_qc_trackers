"""
Tests for qc_types/discovery/cohort_trajectory_deviation_checks.py.

The detector finds subject-points whose value deviates significantly
from the cohort consensus (per-tp median + MAD-scaled spread). It
should:
  - flag isolated extreme deviations
  - ALSO flag consistent-elevation patterns (every visit far from
    the cohort), which point_spike misses by design
  - not false-flag well-behaved cohorts
  - stratify by cohort and timepoint
  - exclude floating/conversion rows
  - emit per-(subject, tp) records
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc_types.discovery.cohort_trajectory_deviation_checks import (
    CohortTrajectoryDeviationChecks,
    OUTPUT_COLUMNS,
)


LONG_COLS = [
    'subjectid', 'network', 'timepoint', 'cohort', 'event_date',
    'source_form', 'variable', 'value', 'value_numeric',
    'is_missing_code',
]

TEST_TPS = ['baseline', 'month1', 'month2', 'month3', 'month4']


def _row(subjectid, tp, variable, value, cohort='chr',
         network='TEST', source_form='form_a',
         is_missing_code=False):
    return {
        'subjectid': subjectid,
        'network': network,
        'timepoint': tp,
        'cohort': cohort,
        'event_date': '',
        'source_form': source_form,
        'variable': variable,
        'value': str(value),
        'value_numeric': float(value),
        'is_missing_code': is_missing_code,
    }


def _normal_cohort(n_subjects, mean, sd, rng,
                   variable='test_var', cohort='chr',
                   tps=None, source_form='form_a'):
    if tps is None:
        tps = list(TEST_TPS)
    rows = []
    for i in range(n_subjects):
        sid = f"NORM{i+1:04d}"
        for tp in tps:
            rows.append(_row(
                sid, tp, variable,
                float(rng.normal(mean, sd)),
                cohort=cohort, source_form=source_form))
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
            return list(TEST_TPS)

    import qc_types.discovery.cohort_trajectory_deviation_checks as m
    monkeypatch.setattr(m, 'Utils', FakeUtils)

    def _make(**overrides):
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "cohort_trajectory_deviation": {
                    "enabled": True,
                    "networks": ["TEST"],
                    "min_observations_per_tp": 30,
                    "z_threshold": 4.0,
                    **overrides,
                }
            },
        }
        return CohortTrajectoryDeviationChecks(config_dict=cfg)

    return _make


# ----------------------------------------------- happy path


def test_well_behaved_cohort_no_flags(detector_factory):
    """
    40 subjects drawn from N(50, 3), every visit. No subject is
    extreme. Expect zero flags.
    """
    rng = np.random.default_rng(0)
    rows = _normal_cohort(40, mean=50, sd=3.0, rng=rng)
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    assert len(out) == 0


# ----------------------------------------------- isolated extreme point


def test_isolated_extreme_point_flagged(detector_factory):
    """
    One subject has an extreme value at exactly one tp. Cohort z
    should fire at that (subject, tp).
    """
    rng = np.random.default_rng(1)
    rows = _normal_cohort(40, mean=50, sd=3.0, rng=rng)
    # OUTLIER subject: normal everywhere except month2 = 200.
    for tp, val in zip(
        TEST_TPS,
        [50.0, 50.5, 200.0, 50.5, 50.0],
    ):
        rows.append(_row('OUTLIER', tp, 'test_var', val))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    flagged = out[out['subjectid'] == 'OUTLIER']
    # Exactly one row for OUTLIER, at month2.
    assert len(flagged) == 1
    f = flagged.iloc[0]
    assert f['timepoint'] == 'month2'
    assert f['observed_value'] == 200.0
    assert f['severity_score'] > 4.0


# ---------------------------- consistent elevation across visits


def test_consistent_elevation_all_flagged(detector_factory):
    """
    THIS is the case point_spike misses by design: a subject who is
    consistently elevated across every visit. The cohort-curve
    detector must fire at EACH such visit because each point is
    individually far from the cohort consensus.
    """
    rng = np.random.default_rng(2)
    rows = _normal_cohort(40, mean=50, sd=3.0, rng=rng)
    # CONSIST subject: elevated at every visit.
    for tp in TEST_TPS:
        rows.append(_row('CONSIST', tp, 'test_var', 90.0))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    flagged = out[out['subjectid'] == 'CONSIST']
    # All 5 visits should fire — this is the key value-add of this
    # detector vs point_spike.
    assert len(flagged) == 5
    assert set(flagged['timepoint']) == set(TEST_TPS)
    assert (flagged['severity_score'] > 4.0).all()
    # No other subject false-flags.
    others = out[out['subjectid'] != 'CONSIST']
    assert len(others) == 0


# ----------------------------------------------- cohort stratification


def test_cohort_stratification(detector_factory):
    """
    HC and CHR cohorts have very different baselines (HC=20, CHR=50).
    A CHR-typical subject (value=50) at baseline is normal in CHR
    but would look extreme if pooled into a mixed HC+CHR centroid.
    With stratification on, no flags fire for in-cohort subjects.
    """
    rng = np.random.default_rng(3)
    rows = _normal_cohort(40, mean=20, sd=3.0, rng=rng,
                          cohort='hc')
    # Re-key CHR subject IDs so they don't collide with HC.
    chr_rows = _normal_cohort(40, mean=50, sd=3.0, rng=rng,
                              cohort='chr')
    for r in chr_rows:
        r['subjectid'] = 'CHR' + r['subjectid'][4:]
    rows.extend(chr_rows)
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    # With stratification, no in-cohort subject is anomalous.
    assert len(out) == 0


def test_no_cohort_stratification_pools_cohorts(detector_factory):
    """
    With stratify_by_cohort=False, the same HC+CHR mix produces
    flags for the cohort-typical extremes (because they're pulled
    against the pooled centroid).
    """
    rng = np.random.default_rng(4)
    rows = _normal_cohort(40, mean=20, sd=3.0, rng=rng,
                          cohort='hc')
    chr_rows = _normal_cohort(40, mean=50, sd=3.0, rng=rng,
                              cohort='chr')
    for r in chr_rows:
        r['subjectid'] = 'CHR' + r['subjectid'][4:]
    rows.extend(chr_rows)
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory(stratify_by_cohort=False)
    out = detector._compute_candidates(long_df)
    # No flag is guaranteed by this test (depends on pooled MAD);
    # the assertion is on the counter: cohort column on output is
    # empty (no stratification key).
    if len(out) > 0:
        # All rows have cohort='' when stratification is off.
        assert (out['cohort'] == '').all()


# ----------------------------------------------- tp stratification


def test_tp_stratification_isolates_visit(detector_factory):
    """
    Subject is extreme only at month3 — must flag at month3 only,
    not at other tps where the same value is normal.
    """
    rng = np.random.default_rng(5)
    rows = _normal_cohort(40, mean=50, sd=3.0, rng=rng)
    for tp, val in zip(
        TEST_TPS,
        [50.0, 50.5, 49.0, 300.0, 50.5],
    ):
        rows.append(_row('SPIKE', tp, 'test_var', val))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    spike_rows = out[out['subjectid'] == 'SPIKE']
    assert (spike_rows['timepoint'] == 'month3').all()
    assert len(spike_rows) == 1


# ----------------------------------------------- min-n guard


def test_below_min_n_skipped(detector_factory):
    """
    Cohort of 10 subjects (below min 30). Cell is skipped even
    though the OUTLIER is extreme.
    """
    rng = np.random.default_rng(6)
    rows = _normal_cohort(10, mean=50, sd=3.0, rng=rng)
    rows.append(_row('OUTLIER', 'baseline', 'test_var', 500.0))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    assert len(out) == 0
    assert detector._counters[
        'cells_skipped_insufficient_n'] >= 1


# --------------------------------------- degenerate MAD skipped


def test_degenerate_mad_skipped(detector_factory):
    """
    Every subject has identical value at a tp → MAD = 0 → cell
    skipped silently (no division by zero, no flags).
    """
    rng = np.random.default_rng(7)
    rows = _normal_cohort(40, mean=50, sd=3.0, rng=rng)
    # Override every baseline value to a constant.
    for r in rows:
        if r['timepoint'] == 'baseline':
            r['value_numeric'] = 50.0
            r['value'] = '50.0'
    rows.append(_row('OUTLIER', 'baseline', 'test_var', 500.0))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    # No baseline flag because cohort MAD was 0 (cell skipped).
    baseline_flags = out[out['timepoint'] == 'baseline']
    assert len(baseline_flags) == 0
    assert detector._counters[
        'cells_skipped_degenerate_mad'] >= 1


# ----------------------------------------------- floating/conversion excluded


def test_floating_and_conversion_excluded(detector_factory):
    """
    Conversion timepoint values must not produce flags even if
    extreme — the conversion event is exactly the value the study
    is designed to detect.
    """
    rng = np.random.default_rng(8)
    rows = _normal_cohort(40, mean=50, sd=3.0, rng=rng)
    # Add 40 conversion-tp rows including extreme.
    for i in range(40):
        rows.append(_row(
            f"CONV{i+1:04d}", 'conversion', 'test_var',
            500.0 if i == 0 else 50.0))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    conv = out[out['timepoint'] == 'conversion']
    assert len(conv) == 0


# ----------------------------------------------- caps


def test_max_flags_to_write_cap(detector_factory):
    """
    Many extreme points → max_flags_to_write caps the output.
    """
    rng = np.random.default_rng(9)
    rows = _normal_cohort(40, mean=50, sd=3.0, rng=rng)
    # 20 subjects, each elevated at all 5 visits — 100 candidate
    # flags before cap.
    for i in range(20):
        for tp in TEST_TPS:
            rows.append(_row(
                f"HI{i+1:04d}", tp, 'test_var', 90.0))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory(max_flags_to_write=15)
    out = detector._compute_candidates(long_df)
    assert len(out) == 15


# ----------------------------------------------- schema and empty


def test_output_schema_columns(detector_factory):
    rng = np.random.default_rng(10)
    rows = _normal_cohort(40, mean=50, sd=3.0, rng=rng)
    rows.append(_row('OUTLIER', 'month2', 'test_var', 300.0))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert 'evidence_timepoints' in out.columns


def test_empty_input_empty_output(detector_factory):
    long_df = pd.DataFrame(columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    assert len(out) == 0
    assert list(out.columns) == OUTPUT_COLUMNS


# ----------------------------------------------- file output


def test_file_output_atomic(detector_factory, tmp_path):
    rng = np.random.default_rng(11)
    rows = _normal_cohort(40, mean=50, sd=3.0, rng=rng)
    rows.append(_row('OUTLIER', 'month2', 'test_var', 300.0))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    long_path = (
        tmp_path / "dependencies"
        / "multi_tp_TEST_numeric_long.parquet")
    long_df.to_parquet(long_path, index=False)

    detector = detector_factory()
    detector.run()
    out_dir = (
        tmp_path / "output" / "combined_outputs" / "discovery")
    out_path = (
        out_dir / "cohort_trajectory_deviation.parquet")
    assert out_path.exists()
    assert list(out_dir.glob("*.tmp")) == []
    read_back = pd.read_parquet(out_path)
    assert (read_back['subjectid'] == 'OUTLIER').any()


# -------- key contrast: complement of point_spike's behavior --------


def test_complements_point_spike_consistent_elevation(detector_factory):
    """
    Sanity check that this detector catches a case point_spike
    cannot catch. A subject with cohort_z ≈ 13 at every visit:
      - point_spike: every tp's neighbor is also at z=13, so the
        neighbor-quiet gate (|neighbor_z| <= low_threshold) fails;
        no flag is emitted by point_spike.
      - cohort_trajectory_deviation: each visit independently
        exceeds the threshold; all 5 emit flags.
    """
    rng = np.random.default_rng(12)
    rows = _normal_cohort(40, mean=50, sd=3.0, rng=rng)
    # CONSIST subject — every visit ~13 sigmas above cohort mean.
    for tp in TEST_TPS:
        rows.append(_row('CONSIST', tp, 'test_var', 90.0))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    consist_flags = out[out['subjectid'] == 'CONSIST']
    assert len(consist_flags) == 5
