"""
Tests for qc_types/discovery/longitudinal_delta_checks.py.

Mirrors the LA test isolation pattern.

Test cases:
  LD-1: stable subjects + 1 abrupt-step subject → 1 flag at the
        step
  LD-2: all-missing-code → 0 flags
  LD-3: variable below min_observations_per_variable → 0 flags
  LD-4: missing-code regression — value_numeric=999 with
        is_missing_code=True must NOT contribute to deltas
  LD-5: floating timepoint excluded — adjacency baseline →
        floating not produced, so a value swing baseline →
        floating produces no flag
  LD-6: degenerate delta MAD (every subject has same delta) → 0
        flags
  Schema: missing required col → RuntimeError
  Empty:  no flags → DataFrame with 26 columns and correct dtypes
  File:   end-to-end run() — parquet exists, is readable, no
          .tmp residue
"""

import os
import sys
import pytest
import pandas as pd

_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc_types.discovery.longitudinal_delta_checks import (
    LongitudinalDeltaChecks)


LONG_FORMAT_COLUMNS = [
    'subjectid', 'network', 'timepoint', 'cohort', 'event_date',
    'source_form', 'variable', 'value', 'value_numeric',
    'is_missing_code',
]

TEST_TP_ORDER = ['screening', 'baseline',
                 'month1', 'month2', 'month3', 'month4',
                 'month6', 'month12']


def _make_long_df(rows):
    full = []
    for r in rows:
        full.append({
            'subjectid': r['subjectid'],
            'network': r.get('network', 'TEST'),
            'timepoint': r['timepoint'],
            'cohort': r.get('cohort', ''),
            'event_date': r.get('event_date', ''),
            'source_form': r.get('source_form', ''),
            'variable': r['variable'],
            'value': r['value'],
            'value_numeric': r['value_numeric'],
            'is_missing_code': r['is_missing_code'],
        })
    return pd.DataFrame(full, columns=LONG_FORMAT_COLUMNS)


def _make_stable_subjects(n_subjects=11, variable='test_var'):
    """
    11 subjects with 4 tps each. Each subject's deltas are
    drawn from {-1, 0, +1, +0.5} so the cohort delta MAD is
    well-defined and small.
    """
    rows = []
    paths = [
        [50.0, 51.0, 50.0, 51.0],
        [50.0, 50.0, 51.0, 50.5],
        [50.0, 49.0, 50.0, 50.5],
        [50.0, 51.0, 51.5, 51.0],
        [50.0, 50.5, 50.0, 50.5],
        [50.0, 50.0, 50.5, 51.0],
        [50.0, 49.5, 50.0, 49.5],
        [50.0, 51.0, 50.5, 51.0],
        [50.0, 50.5, 51.0, 50.5],
        [50.0, 50.0, 51.0, 50.5],
        [50.0, 50.5, 50.0, 50.5],
    ]
    tps = ['baseline', 'month1', 'month2', 'month3']
    for i in range(n_subjects):
        sid = f"S{i+1:03d}"
        for tp, val in zip(tps, paths[i % len(paths)]):
            rows.append({
                'subjectid': sid, 'timepoint': tp,
                'variable': variable,
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

    import qc_types.discovery.longitudinal_delta_checks \
        as detector_module
    monkeypatch.setattr(detector_module, 'Utils', FakeUtils)

    def _make(**ld_overrides):
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "longitudinal_delta": {
                    "networks": ["TEST"],
                    "min_observations_per_variable": 30,
                    "robust_z_threshold": 4.0,
                    "max_flags_to_write": None,
                    **ld_overrides,
                }
            },
        }
        return LongitudinalDeltaChecks(config_dict=cfg)

    return _make


# ---------------------------------------------------------------- LD-1

def test_ld1_single_abrupt_step(detector_factory):
    detector = detector_factory()
    rows = _make_stable_subjects(n_subjects=11)
    # JUMP subject: smooth at first then a 200-point jump.
    for tp, val in zip(
        ['baseline', 'month1', 'month2', 'month3'],
        [50.0, 51.0, 50.0, 250.0],
    ):
        rows.append({
            'subjectid': 'JUMP', 'timepoint': tp,
            'variable': 'test_var',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)

    result = detector._compute_candidates(long_df)
    jump_flags = result[result['subjectid'] == 'JUMP']
    assert len(jump_flags) == 1
    flag = jump_flags.iloc[0]
    assert flag['timepoint'] == 'month3'
    assert flag['prev_timepoint'] == 'month2'
    assert flag['observed_value'] == 250.0
    assert flag['expected_value'] == 50.0
    assert flag['delta_value'] == 200.0
    assert flag['severity_score'] >= 4.0
    assert flag['affected_timepoints'] == 'month2,month3'


# ---------------------------------------------------------------- LD-2

def test_ld2_all_missing_codes_zero_flags(detector_factory):
    detector = detector_factory()
    rows = []
    for i in range(12):
        sid = f"S{i+1:03d}"
        for j, tp in enumerate(
                ['baseline', 'month1', 'month2', 'month3']):
            rows.append({
                'subjectid': sid, 'timepoint': tp,
                'variable': 'test_var',
                'value': '999',
                'value_numeric': float('nan'),
                'is_missing_code': True,
            })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0


# ---------------------------------------------------------------- LD-3

def test_ld3_below_min_obs_zero_flags(detector_factory):
    detector = detector_factory(min_observations_per_variable=30)
    rows = []
    for i in range(3):
        sid = f"S{i+1:03d}"
        for j, tp in enumerate(
                ['baseline', 'month1', 'month2', 'month3']):
            val = 50.0 if not (i == 2 and tp == 'month3') else 5000.0
            rows.append({
                'subjectid': sid, 'timepoint': tp,
                'variable': 'test_var',
                'value': str(val), 'value_numeric': val,
                'is_missing_code': False,
            })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters[
        'variables_skipped_insufficient_obs'] == 1


# ---------------------------------------------------------------- LD-4

def test_ld4_missing_code_with_numeric_value_not_in_delta(
    detector_factory
):
    """
    Adversarial: value_numeric=999 with is_missing_code=True at
    month1 must not be used as the value in the baseline →
    month1 delta. After filtering the 999 row out, the subject's
    surviving observations are baseline=50, month2=51, month3=50;
    deltas are 51-50=+1 and 50-51=-1, both well within scale.
    No flag from this subject.
    """
    detector = detector_factory()
    rows = _make_stable_subjects(n_subjects=11)
    rows.append({
        'subjectid': 'ADV', 'timepoint': 'baseline',
        'variable': 'test_var',
        'value': '50.0', 'value_numeric': 50.0,
        'is_missing_code': False,
    })
    rows.append({
        'subjectid': 'ADV', 'timepoint': 'month1',
        'variable': 'test_var',
        'value': '999', 'value_numeric': 999.0,
        'is_missing_code': True,
    })
    rows.append({
        'subjectid': 'ADV', 'timepoint': 'month2',
        'variable': 'test_var',
        'value': '51.0', 'value_numeric': 51.0,
        'is_missing_code': False,
    })
    rows.append({
        'subjectid': 'ADV', 'timepoint': 'month3',
        'variable': 'test_var',
        'value': '50.0', 'value_numeric': 50.0,
        'is_missing_code': False,
    })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    adv_flags = result[result['subjectid'] == 'ADV']
    # 999.0 must not appear as expected_value or observed_value.
    assert (adv_flags['expected_value'] == 999.0).sum() == 0
    assert (adv_flags['observed_value'] == 999.0).sum() == 0


# ---------------------------------------------------------------- LD-5

def test_ld5_floating_timepoint_excluded(detector_factory):
    """
    Subject FLOATER has stable values at baseline / month1 /
    month2, then a 200-point swing at the 'floating' tp. Because
    'floating' is excluded from the canonical sequence, no
    adjacency from month2 → floating is built, and no flag is
    produced.
    """
    detector = detector_factory()
    rows = _make_stable_subjects(n_subjects=11)
    rows.append({
        'subjectid': 'FLOATER', 'timepoint': 'baseline',
        'variable': 'test_var',
        'value': '50.0', 'value_numeric': 50.0,
        'is_missing_code': False,
    })
    rows.append({
        'subjectid': 'FLOATER', 'timepoint': 'month1',
        'variable': 'test_var',
        'value': '51.0', 'value_numeric': 51.0,
        'is_missing_code': False,
    })
    rows.append({
        'subjectid': 'FLOATER', 'timepoint': 'month2',
        'variable': 'test_var',
        'value': '50.0', 'value_numeric': 50.0,
        'is_missing_code': False,
    })
    rows.append({
        'subjectid': 'FLOATER', 'timepoint': 'floating',
        'variable': 'test_var',
        'value': '250.0', 'value_numeric': 250.0,
        'is_missing_code': False,
    })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    floater_flags = result[result['subjectid'] == 'FLOATER']
    assert len(floater_flags) == 0


# ---------------------------------------------------------------- LD-6

def test_ld6_degenerate_delta_mad_zero_flags(detector_factory):
    """
    Variable where every subject has exactly the same step
    (e.g., monotonic +1 each tp): pooled-delta-MAD = 0 →
    detector skips the variable entirely (no fallback).
    """
    detector = detector_factory()
    rows = []
    for i in range(11):
        sid = f"S{i+1:03d}"
        for j, tp in enumerate(
                ['baseline', 'month1', 'month2', 'month3']):
            val = 50.0 + j  # delta is exactly +1 for everyone
            rows.append({
                'subjectid': sid, 'timepoint': tp,
                'variable': 'test_var',
                'value': str(val), 'value_numeric': val,
                'is_missing_code': False,
            })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    # 4 tps → 3 adjacent tp-pairs; each pool has the same +1 delta for
    # every subject, so MAD is 0 in every pair and all three are skipped.
    assert detector._counters[
        'tp_pair_groups_skipped_degenerate_mad'] == 3


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
        # is_missing_code intentionally omitted.
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
    rows = []
    for i in range(5):
        rows.append({
            'subjectid': f'S{i+1}', 'timepoint': 'baseline',
            'variable': 'test_var', 'value': '999',
            'value_numeric': float('nan'),
            'is_missing_code': True,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0

    expected_cols = list(LongitudinalDeltaChecks.OUTPUT_COLUMNS)
    assert list(result.columns) == expected_cols
    for col in (
        'value_numeric', 'expected_value', 'observed_value',
        'metric_value', 'severity_score', 'threshold',
        'prev_value_numeric', 'delta_value',
    ):
        assert result[col].dtype == 'float64', (
            f"{col} dtype is {result[col].dtype}, "
            f"expected float64"
        )


# ---------------------------------------------------------------- File

def test_file_output_atomic_and_readable(
    detector_factory, tmp_path
):
    detector = detector_factory()
    rows = _make_stable_subjects(n_subjects=11)
    for tp, val in zip(
        ['baseline', 'month1', 'month2', 'month3'],
        [50.0, 51.0, 50.0, 250.0],
    ):
        rows.append({
            'subjectid': 'JUMP', 'timepoint': tp,
            'variable': 'test_var',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    long_path = (
        tmp_path / "dependencies"
        / "multi_tp_TEST_numeric_long.parquet")
    long_df.to_parquet(long_path, index=False)

    detector.run()

    out_dir = (
        tmp_path / "output" / "combined_outputs" / "discovery")
    out_path = (
        out_dir / "longitudinal_delta_candidates.parquet")
    assert out_path.exists()
    out_df = pd.read_parquet(out_path)
    assert (out_df['subjectid'] == 'JUMP').sum() >= 1
    tmps = list(out_dir.glob("*.tmp"))
    assert len(tmps) == 0
