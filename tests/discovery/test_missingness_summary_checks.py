"""
Tests for qc_types/discovery/missingness_summary_checks.py.

Isolation: FakeUtils monkeypatch, tmp_path, synthetic long parquet.
No real config.json, no Dropbox.
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

import qc_types.discovery.missingness_summary_checks as msm
from qc_types.discovery.missingness_summary_checks import (
    MissingnessSummaryChecks,
)

LONG_COLS = [
    'subjectid', 'network', 'timepoint', 'cohort', 'event_date',
    'source_form', 'variable', 'value', 'value_numeric',
    'is_missing_code',
]


def _row(sid, net, tp, var, miss, val='1', vnum=1.0, sform='f'):
    return {
        'subjectid': sid,
        'network': net,
        'timepoint': tp,
        'cohort': '',
        'event_date': '',
        'source_form': sform,
        'variable': var,
        'value': val,
        'value_numeric': vnum,
        'is_missing_code': miss,
    }


@pytest.fixture
def ms_factory(tmp_path, monkeypatch):
    deps = tmp_path / "dependencies"
    out = tmp_path / "output"
    deps.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)

    class FakeUtils:
        def __init__(self):
            self.absolute_path = str(tmp_path)

    monkeypatch.setattr(msm, 'Utils', FakeUtils)

    def _make(**ms_over):
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + os.sep,
                "output_path": str(out) + os.sep,
            },
            "testing_enabled": "False",
            "discovery": {
                "missingness_summary": {
                    "enabled": True,
                    "networks": ["TEST"],
                    "min_rows_per_slice": 20,
                    "robust_z_threshold": 3.0,
                    "max_rows_to_write": 1000,
                    **ms_over,
                }
            },
        }
        return MissingnessSummaryChecks(config_dict=cfg)

    return _make, deps, out


def test_ms1_variable_high_missingness_flagged(ms_factory):
    """One variable with ~90% missing vs peers ~3% at same (net,tp)."""
    make, deps, out = ms_factory
    rows = []
    tps = 'baseline'
    # Five "normal" variables: 30 rows, 1 missing each
    for vi in range(5):
        var = f"v_norm_{vi}"
        for i in range(30):
            rows.append(_row(
                f"AA{i:03d}", "TEST", tps, var,
                miss=(i == 0),
            ))
    # Outlier variable: 30 rows, 27 missing
    for i in range(30):
        rows.append(_row(
            f"BB{i:03d}", "TEST", tps, "v_outlier",
            miss=(i < 27),
        ))
    df = pd.DataFrame(rows, columns=LONG_COLS)
    df.to_parquet(deps / "multi_tp_TEST_numeric_long.parquet", index=False)

    det = make(robust_z_threshold=2.0)
    det.run()
    outp = (
        out / "combined_outputs" / "discovery"
        / "missingness_anomaly_summary.parquet"
    )
    assert outp.is_file()
    res = pd.read_parquet(outp)
    var_rows = res[res['anomaly_type'] == 'variable_rate']
    assert len(var_rows) >= 1
    assert (var_rows['variable'] == 'v_outlier').any()


def test_ms2_site_high_missingness_flagged(ms_factory):
    """One site prefix with far more missing rows than peer sites."""
    make, deps, out = ms_factory
    rows = []
    tp = 'month1'
    # Three sites: low missing rate (~10/100 = 0.1)
    for site, prefix in [('AA', 'AA'), ('AB', 'AB'), ('AC', 'AC')]:
        for i in range(100):
            miss = i < 10
            rows.append(_row(
                f"{prefix}{i:03d}", "TEST", tp, 'shared_var',
                miss=miss,
            ))
    # Fourth site: very high missing
    for i in range(100):
        rows.append(_row(
            f"ZZ{i:03d}", "TEST", tp, 'shared_var',
            miss=(i < 88),
        ))
    df = pd.DataFrame(rows, columns=LONG_COLS)
    df.to_parquet(deps / "multi_tp_TEST_numeric_long.parquet", index=False)

    det = make(robust_z_threshold=2.0)
    det.run()
    res = pd.read_parquet(
        out / "combined_outputs" / "discovery"
        / "missingness_anomaly_summary.parquet"
    )
    site_rows = res[res['anomaly_type'] == 'site_rate']
    assert len(site_rows) >= 1
    assert (site_rows['site_prefix'] == 'ZZ').any()


def test_ms3_empty_input_writes_typed_empty(ms_factory):
    make, deps, out = ms_factory
    # Minimal valid empty-like: file exists with 0 rows — actually
    # producer would not emit this; simulate by empty concat in code
    # path by writing 0-row parquet with full schema
    empty = pd.DataFrame(columns=LONG_COLS)
    empty.to_parquet(
        deps / "multi_tp_TEST_numeric_long.parquet", index=False)

    det = make()
    det.run()
    outp = (
        out / "combined_outputs" / "discovery"
        / "missingness_anomaly_summary.parquet"
    )
    res = pd.read_parquet(outp)
    assert len(res) == 0
    assert set(res.columns) == set(
        MissingnessSummaryChecks.OUTPUT_COLUMNS
    )


def test_ms4_missing_required_column_raises(ms_factory):
    make, deps, _out = ms_factory
    bad = pd.DataFrame({'subjectid': ['a']})
    bad.to_parquet(
        deps / "multi_tp_TEST_numeric_long.parquet", index=False)
    det = make()
    with pytest.raises(RuntimeError, match="missing columns"):
        det.run()


def test_ms5_excluded_source_form(ms_factory):
    """Rows on digital_biomarkers_* forms must be excluded before
    the missing-rate roll-up, regardless of how many missing-code
    rows they have."""
    make, deps, _out = ms_factory
    rows = []
    for vi in range(5):
        for i in range(25):
            rows.append(_row(
                f"AA{i:03d}", "TEST", "baseline", f"vd_{vi}",
                miss=True,
                sform='digital_biomarkers_axivity_checkin'))
    df = pd.DataFrame(rows, columns=LONG_COLS)
    df.to_parquet(
        deps / "multi_tp_TEST_numeric_long.parquet", index=False)
    det = make()
    det.run()
    result_path = (
        _out / "combined_outputs" / "discovery"
        / "missingness_anomaly_summary.parquet")
    result = pd.read_parquet(result_path)
    assert len(result) == 0
    assert det._counters['rows_excluded_by_form_pattern'] > 0


def test_ms6_excluded_variable_pattern(ms_factory):
    """Variables matching default `_complete` pattern must be
    excluded before the roll-up."""
    make, deps, _out = ms_factory
    rows = []
    for i in range(30):
        rows.append(_row(
            f"AA{i:03d}", "TEST", "baseline", "form_complete",
            miss=(i < 27)))
    df = pd.DataFrame(rows, columns=LONG_COLS)
    df.to_parquet(
        deps / "multi_tp_TEST_numeric_long.parquet", index=False)
    det = make()
    det.run()
    result_path = (
        _out / "combined_outputs" / "discovery"
        / "missingness_anomaly_summary.parquet")
    result = pd.read_parquet(result_path)
    assert len(result) == 0
    assert det._counters['rows_excluded_by_variable_pattern'] > 0


def test_ms7_severity_normalized_present(ms_factory):
    """Each flag row carries severity_normalized in [0, 100]."""
    make, deps, _out = ms_factory
    rows = []
    # Five "normal" vars + one outlier var, mirroring ms1.
    for vi in range(5):
        for i in range(30):
            rows.append(_row(
                f"AA{i:03d}", "TEST", "baseline", f"v_norm_{vi}",
                miss=(i == 0)))
    for i in range(30):
        rows.append(_row(
            f"BB{i:03d}", "TEST", "baseline", "v_outlier",
            miss=(i < 27)))
    df = pd.DataFrame(rows, columns=LONG_COLS)
    df.to_parquet(
        deps / "multi_tp_TEST_numeric_long.parquet", index=False)
    det = make()
    det.run()
    result_path = (
        _out / "combined_outputs" / "discovery"
        / "missingness_anomaly_summary.parquet")
    result = pd.read_parquet(result_path)
    assert 'severity_normalized' in result.columns
    assert (result['severity_normalized'] >= 0).all()
    assert (result['severity_normalized'] <= 100).all()
