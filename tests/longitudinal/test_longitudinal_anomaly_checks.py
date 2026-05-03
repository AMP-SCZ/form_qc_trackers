"""
Tests for qc_types/longitudinal_anomaly_checks.py (Step 4 of N1).

Test cases:
  LA-1: single outlier with stable scale (multiple normal subjects)
  LA-2: all missing-code rows → zero flags
  LA-3: subject below min_points_per_subject → zero flags
  LA-4: missing-code coercion regression — value '999' with
        value_numeric=999.0 but is_missing_code=True must not be
        scored
  LA-5: variable below min_observations_per_variable → zero flags
  Schema:  missing required column → RuntimeError naming the column
  Empty:   no anomalies → DataFrame with all 23 expected columns
           and correct numeric dtypes
  File:    end-to-end run() with synthetic input parquet in tmp_path,
           verifying parquet exists, is readable, and no .tmp residue

Isolation:
  Tests are FULLY isolated from the real project config.json. The
  detector_factory fixture monkeypatches the `Utils` symbol in the
  detector module's namespace with a minimal FakeUtils stub, so the
  detector's __init__ never instantiates the real Utils (which
  would read the project config.json for path resolution). Every
  setting comes from the per-test synthetic config_dict; every read
  and write happens under tmp_path.

  No real CSV files. No Dropbox. No production paths. The runner
  (run_longitudinal_anomaly.py) is not imported or invoked.
"""

import os
import sys
import pytest
import pandas as pd

# Add project root to sys.path so the detector import resolves
# regardless of pytest's cwd / discovery rootdir. Three dirname()
# hops: this file → tests/longitudinal/ → tests/ → project root.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc_types.longitudinal_anomaly_checks import (
    LongitudinalAnomalyChecks)


# Mirror of the producer's long-format schema. Used by _make_long_df
# so test inputs have the exact columns the detector expects.
LONG_FORMAT_COLUMNS = [
    'subjectid', 'network', 'timepoint', 'cohort', 'event_date',
    'source_form', 'variable', 'value', 'value_numeric',
    'is_missing_code',
]


def _make_long_df(rows):
    """
    Build a long-format DataFrame from a list of dicts. Required
    keys per row: subjectid, timepoint, variable, value,
    value_numeric, is_missing_code. Optional: network (default
    'TEST'), source_form, cohort, event_date.
    """
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


def _make_normal_subjects(
    n_subjects=11, vals=(50.0, 51.0, 49.0, 52.0),
    variable='test_var', sid_prefix='S',
):
    """
    n_subjects subjects, each with len(vals) timepoints, all
    is_missing_code=False, real value_numeric. Default vals are
    centered on ~50 with low spread so they don't accidentally
    trigger flags in tests that add a separate outlier.
    """
    rows = []
    for i in range(n_subjects):
        sid = f"{sid_prefix}{i+1:03d}"
        for j, val in enumerate(vals):
            rows.append({
                'subjectid': sid,
                'timepoint': f'tp_{j+1}',
                'variable': variable,
                'value': str(val),
                'value_numeric': val,
                'is_missing_code': False,
            })
    return rows


@pytest.fixture
def detector_factory(tmp_path, monkeypatch):
    """
    Builds a LongitudinalAnomalyChecks instance with paths under
    tmp_path. Real Utils is replaced with FakeUtils so the
    detector's __init__ never touches the real project config.json.
    Per-test threshold overrides go through **lon_overrides on the
    returned callable.
    """
    deps = tmp_path / "dependencies"
    out = tmp_path / "output"
    deps.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)

    class FakeUtils:
        # The detector currently only references self.utils to
        # access self.utils.absolute_path inside the
        # `if config_dict is None` branch — and tests always
        # pass config_dict, so that branch never runs. The
        # attribute is kept for safety in case a future detector
        # code path accesses it. No methods needed: missing_code_set,
        # can_be_float, etc. are not used by the detector (they're
        # the producer's responsibility).
        absolute_path = str(tmp_path)

    # Replace the Utils symbol in the detector module's namespace.
    # Python late-binds names inside __init__ via the module
    # globals, so this patch takes effect on every subsequent
    # LongitudinalAnomalyChecks() call inside this test. monkeypatch
    # auto-restores the original Utils at test teardown.
    import qc_types.longitudinal_anomaly_checks as detector_module
    monkeypatch.setattr(detector_module, 'Utils', FakeUtils)

    def _make(**lon_overrides):
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "longitudinal_anomaly": {
                "networks": ["TEST"],
                "min_points_per_subject": 3,
                "min_observations_per_variable": 30,
                "robust_z_threshold": 3.5,
                "max_flags_to_write": None,
                **lon_overrides,
            },
        }
        return LongitudinalAnomalyChecks(config_dict=cfg)

    return _make


# ---------------------------------------------------------------- LA-1

def test_la1_single_outlier_with_stable_scale(detector_factory):
    """
    11 normal subjects (residuals tightly clustered) + 1 outlier
    subject whose final timepoint observation is far above the
    subject's median. The pooled MAD is dominated by the 44 normal
    residuals; the outlier residual sits ~100σ out and is the only
    flag produced.
    """
    detector = detector_factory()
    rows = _make_normal_subjects(n_subjects=11)
    # OUTLIER subject: median = 50.5, residuals = [-0.5, 0.5, -1.5,
    # 149.5]. The 149.5 residual is the only flag we expect.
    for j, val in enumerate([50.0, 51.0, 49.0, 200.0]):
        rows.append({
            'subjectid': 'OUTLIER',
            'timepoint': f'tp_{j+1}',
            'variable': 'test_var',
            'value': str(val),
            'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)

    result = detector._compute_anomalies(long_df)

    assert len(result) == 1, f"expected 1 flag, got {len(result)}"
    flag = result.iloc[0]
    assert flag['subjectid'] == 'OUTLIER'
    assert flag['timepoint'] == 'tp_4'
    assert flag['observed_value'] == 200.0
    assert flag['severity_score'] > 3.5
    assert flag['algorithm'] == 'within_subject_robust_z'
    assert flag['metric_name'] == 'within_subject_robust_z'

    # Tracker-compatible aliases populated correctly.
    assert flag['subject'] == flag['subjectid']
    assert flag['displayed_variable'] == flag['variable']
    assert flag['displayed_timepoint'] == flag['timepoint']

    # No normal subject is flagged.
    normal_in_result = result[
        result['subjectid'].str.startswith('S')]
    assert len(normal_in_result) == 0


# ---------------------------------------------------------------- LA-2

def test_la2_all_missing_codes_zero_flags(detector_factory):
    """
    Every input row has is_missing_code=True. The scoring filter
    excludes all of them; result is the empty schema-valid DF.
    """
    detector = detector_factory()
    rows = []
    codes = ['-3', '999', '1909-09-09']
    # 12 subjects × 3 obs = 36 rows total (above min_observations_
    # _per_variable=30, but all filtered out by is_missing_code).
    for i in range(12):
        sid = f"S{i+1:03d}"
        for j, code in enumerate(codes):
            rows.append({
                'subjectid': sid,
                'timepoint': f'tp_{j+1}',
                'variable': 'test_var',
                'value': code,
                'value_numeric': float('nan'),
                'is_missing_code': True,
            })
    long_df = _make_long_df(rows)

    result = detector._compute_anomalies(long_df)
    assert len(result) == 0


# ---------------------------------------------------------------- LA-3

def test_la3_below_min_points_zero_flags(detector_factory):
    """
    A subject with only 2 valid observations is filtered before
    scoring. Even an extreme value at one of those two timepoints
    is not flagged because the subject is dropped before residual
    computation.
    """
    detector = detector_factory()
    rows = _make_normal_subjects(n_subjects=10)  # 40 normal obs
    # Subject TOO_FEW has only 2 obs; the 50000 would otherwise be
    # an obvious outlier.
    rows.append({
        'subjectid': 'TOO_FEW', 'timepoint': 'tp_1',
        'variable': 'test_var', 'value': '50.0',
        'value_numeric': 50.0, 'is_missing_code': False,
    })
    rows.append({
        'subjectid': 'TOO_FEW', 'timepoint': 'tp_2',
        'variable': 'test_var', 'value': '50000.0',
        'value_numeric': 50000.0, 'is_missing_code': False,
    })
    long_df = _make_long_df(rows)

    result = detector._compute_anomalies(long_df)

    # TOO_FEW subject not in result.
    assert (result['subjectid'] == 'TOO_FEW').sum() == 0
    # Normal subjects also not flagged.
    assert len(result) == 0


# ---------------------------------------------------------------- LA-4

def test_la4_missing_code_999_not_scored(detector_factory):
    """
    Regression test: even if value_numeric=999.0 is left populated
    alongside is_missing_code=True (a hypothetical producer bug),
    the detector must NOT score that observation. Filter is
    `value_numeric.notna() & ~is_missing_code` — both conditions
    are required for scoring eligibility.
    """
    detector = detector_factory()
    rows = _make_normal_subjects(n_subjects=10)  # 40 normal obs
    # ADV subject's first obs is the adversarial row.
    rows.append({
        'subjectid': 'ADV', 'timepoint': 'tp_1',
        'variable': 'test_var', 'value': '999',
        'value_numeric': 999.0,   # adversarial: not NaN
        'is_missing_code': True,  # but flagged as missing-code
    })
    # ADV's other 3 obs are within normal range so the subject
    # itself doesn't get filtered for too-few-points.
    for j, val in enumerate([50.0, 51.0, 49.0]):
        rows.append({
            'subjectid': 'ADV', 'timepoint': f'tp_{j+2}',
            'variable': 'test_var', 'value': str(val),
            'value_numeric': val, 'is_missing_code': False,
        })
    long_df = _make_long_df(rows)

    result = detector._compute_anomalies(long_df)

    # The 999.0 row must not be scored as an anomaly.
    flagged_999 = result[
        (result['subjectid'] == 'ADV')
        & (result['observed_value'] == 999.0)
    ]
    assert len(flagged_999) == 0
    # No flags overall (ADV's valid obs are within normal range).
    assert len(result) == 0


# ---------------------------------------------------------------- LA-5

def test_la5_below_min_obs_per_variable_zero_flags(detector_factory):
    """
    Variable has 20 valid observations (below min 30). Detector
    skips the variable entirely; even a clear outlier observation
    produces no flag.
    """
    detector = detector_factory(min_observations_per_variable=30)
    rows = []
    # 5 subjects × 4 obs = 20 obs total. Subject 5 contains a
    # 5000 outlier that would otherwise fire if the variable were
    # scored.
    for i in range(5):
        sid = f"S{i+1:03d}"
        if i == 4:
            vals = [50.0, 51.0, 49.0, 5000.0]
        else:
            vals = [50.0, 51.0, 49.0, 52.0]
        for j, val in enumerate(vals):
            rows.append({
                'subjectid': sid,
                'timepoint': f'tp_{j+1}',
                'variable': 'test_var',
                'value': str(val),
                'value_numeric': val,
                'is_missing_code': False,
            })
    long_df = _make_long_df(rows)

    result = detector._compute_anomalies(long_df)
    assert len(result) == 0


# ---------------------------------------------------------------- Schema

def test_schema_validation_missing_column_raises(
    detector_factory, tmp_path
):
    """
    A long-format parquet missing a required column must cause
    _load_inputs to raise RuntimeError naming the missing column.
    """
    detector = detector_factory()
    bogus = pd.DataFrame({
        'subjectid': ['S1', 'S2'],
        'network': ['TEST', 'TEST'],
        'timepoint': ['tp_1', 'tp_2'],
        'variable': ['var', 'var'],
        'source_form': ['', ''],
        'value': ['50', '51'],
        'value_numeric': [50.0, 51.0],
        # is_missing_code intentionally omitted.
    })
    bogus_path = (
        tmp_path / "dependencies"
        / "multi_tp_TEST_numeric_long.parquet")
    bogus.to_parquet(bogus_path, index=False)

    with pytest.raises(RuntimeError) as excinfo:
        detector._load_inputs()
    msg = str(excinfo.value)
    assert 'is_missing_code' in msg


# ---------------------------------------------------------------- Empty schema

def test_empty_output_schema(detector_factory):
    """
    No anomalies produced → empty DataFrame with all 23 expected
    columns in OUTPUT_COLUMNS order and correct numeric dtypes.
    """
    detector = detector_factory()
    # All-missing-code input → filter excludes everything.
    rows = []
    for i in range(5):
        rows.append({
            'subjectid': f'S{i+1}', 'timepoint': 'tp_1',
            'variable': 'test_var', 'value': '999',
            'value_numeric': float('nan'), 'is_missing_code': True,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_anomalies(long_df)

    assert len(result) == 0

    expected_cols = [
        # Audit columns.
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric',
        'expected_value', 'observed_value',
        'algorithm', 'metric_name', 'metric_value',
        'severity_score', 'threshold', 'error_message',
        'dates_detected',
        # Tracker-compatible aliases.
        'subject', 'displayed_variable', 'displayed_timepoint',
        'affected_variables', 'affected_timepoints',
        'affected_forms', 'displayed_form',
    ]
    assert list(result.columns) == expected_cols

    for col in (
        'value_numeric', 'expected_value', 'observed_value',
        'metric_value', 'severity_score', 'threshold',
    ):
        assert result[col].dtype == 'float64', (
            f"{col}: dtype is {result[col].dtype}, "
            f"expected float64")


# ---------------------------------------------------------------- File output

def test_file_output_atomic_and_readable(
    detector_factory, tmp_path
):
    """
    End-to-end run() with a synthetic long-format input parquet
    in tmp_path. Verify the output parquet exists at the expected
    path, is readable via pd.read_parquet, contains the expected
    anomaly, and no .tmp file is left behind.
    """
    detector = detector_factory()

    # Reuse LA-1's data shape: 11 normal + 1 outlier.
    rows = _make_normal_subjects(n_subjects=11)
    for j, val in enumerate([50.0, 51.0, 49.0, 200.0]):
        rows.append({
            'subjectid': 'OUTLIER',
            'timepoint': f'tp_{j+1}',
            'variable': 'test_var',
            'value': str(val),
            'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    long_path = (
        tmp_path / "dependencies"
        / "multi_tp_TEST_numeric_long.parquet")
    long_df.to_parquet(long_path, index=False)

    detector.run()

    out_dir = (
        tmp_path / "output" / "combined_outputs"
        / "longitudinal_anomalies")
    out_path = out_dir / "longitudinal_anomaly_flags.parquet"

    # 1. parquet exists.
    assert out_path.exists()
    # 2. readable, expected content.
    out_df = pd.read_parquet(out_path)
    assert len(out_df) == 1
    assert out_df.iloc[0]['observed_value'] == 200.0
    assert out_df.iloc[0]['severity_score'] > 3.5
    # 3. atomic write left no .tmp residue.
    tmps = list(out_dir.glob("*.tmp"))
    assert len(tmps) == 0, f"unexpected .tmp files: {tmps}"
