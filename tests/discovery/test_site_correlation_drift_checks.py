"""
Tests for qc_types/discovery/site_correlation_drift_checks.py.

Verifies the detector flags sites whose scale-relationship Spearman
ρ diverges from peers (Fisher-z Δ > threshold), while NOT flagging
sites where the peer reference is weak or the cohort genuinely
differs.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc_types.discovery.site_correlation_drift_checks import (
    SiteCorrelationDriftChecks,
    OUTPUT_COLUMNS,
)


LONG_COLS = [
    'subjectid', 'network', 'timepoint', 'cohort', 'event_date',
    'source_form', 'variable', 'value', 'value_numeric',
    'is_missing_code',
]

TEST_TPS = ['baseline', 'month1', 'month2', 'month3']


def _row(subjectid, tp, variable, value_numeric, cohort='chr',
         network='TEST', source_form='form_a'):
    return {
        'subjectid': subjectid,
        'network': network,
        'timepoint': tp,
        'cohort': cohort,
        'event_date': '',
        'source_form': source_form,
        'variable': variable,
        'value': str(value_numeric),
        'value_numeric': float(value_numeric),
        'is_missing_code': False,
    }


def _correlated_site(site, n, slope, noise, rng,
                     tp='baseline', cohort='chr',
                     var_a='bprs_pos', var_b='bprs_neg'):
    """
    Build n subjects at one site where val_b = slope*val_a + noise.
    slope=+1 → positive correlation; slope=0 → uncorrelated;
    slope=-1 → negative correlation.
    """
    rows = []
    for i in range(n):
        subjectid = f"{site}{i+1:04d}"
        a = float(rng.normal(20, 4))
        b = slope * a + float(rng.normal(0, noise))
        rows.append(_row(subjectid, tp, var_a, a,
                         cohort=cohort))
        rows.append(_row(subjectid, tp, var_b, b,
                         cohort=cohort))
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

    import qc_types.discovery.site_correlation_drift_checks as mod
    monkeypatch.setattr(mod, 'Utils', FakeUtils)

    def _make(pairs, **overrides):
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "site_correlation_drift": {
                    "enabled": True,
                    "networks": ["TEST"],
                    "min_n_per_cell": 30,
                    "delta_fisher_z_threshold": 0.4,
                    "default_min_ref_rho": 0.2,
                    "pairs": pairs,
                    **overrides,
                }
            },
        }
        return SiteCorrelationDriftChecks(config_dict=cfg)

    return _make


_DEFAULT_PAIR = [{
    "var_a": "bprs_pos",
    "var_b": "bprs_neg",
    "label": "BPRS positive vs negative subscale",
    "direction": "positive",
}]


# ----------------------------------------------- happy path


def test_all_sites_same_correlation_no_flags(detector_factory):
    """
    Five sites × 40 subjects, all with slope=+1 (strong positive ρ).
    No site is divergent. Expect zero flags.
    """
    rng = np.random.default_rng(0)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL', 'OR'):
        rows.extend(_correlated_site(
            site, 40, slope=1.0, noise=2.0, rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory(pairs=_DEFAULT_PAIR)
    out = detector._compute_candidates(long_df)
    flags = out[out['is_audit_only'] == False]
    assert len(flags) == 0


# ----------------------------------------------- one site decoupled


def test_one_site_decoupled_flagged(detector_factory):
    """
    Four sites have positive ρ; OR has slope=0 (decoupled). OR
    should flag.
    """
    rng = np.random.default_rng(1)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_correlated_site(
            site, 40, slope=1.0, noise=2.0, rng=rng))
    rows.extend(_correlated_site(
        'OR', 40, slope=0.0, noise=8.0, rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory(pairs=_DEFAULT_PAIR)
    out = detector._compute_candidates(long_df)
    flags = out[out['is_audit_only'] == False]
    assert (flags['site'] == 'OR').any()
    flag = flags[flags['site'] == 'OR'].iloc[0]
    assert flag['site_rho'] < 0.3
    assert flag['ref_rho'] > 0.5
    assert flag['delta_fisher_z'] < -0.4
    # No other site should be flagged on this pair.
    other = flags[flags['site'] != 'OR']
    assert len(other) == 0


# ----------------------------------------------- sign flip is the strongest signal


def test_one_site_sign_flipped_flagged(detector_factory):
    rng = np.random.default_rng(2)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_correlated_site(
            site, 40, slope=1.0, noise=2.0, rng=rng))
    rows.extend(_correlated_site(
        'OR', 40, slope=-1.0, noise=2.0, rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory(pairs=_DEFAULT_PAIR)
    out = detector._compute_candidates(long_df)
    flags = out[out['is_audit_only'] == False]
    assert (flags['site'] == 'OR').any()
    flag = flags[flags['site'] == 'OR'].iloc[0]
    assert flag['site_rho'] < -0.5
    assert flag['delta_fisher_z'] < -1.0


# ----------------------------------------------- weak peer reference → no flag


def test_weak_peer_reference_skips_cell(detector_factory):
    """
    All peer sites have slope=0 (weak ρ); OR also has weak ρ. The
    cell should be skipped because we can't claim the peer
    relationship is strong enough to compare against.
    """
    rng = np.random.default_rng(3)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_correlated_site(
            site, 40, slope=0.0, noise=8.0, rng=rng))
    # OR also low but in opposite direction — small delta, weak ref.
    rows.extend(_correlated_site(
        'OR', 40, slope=-0.1, noise=8.0, rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory(pairs=_DEFAULT_PAIR)
    out = detector._compute_candidates(long_df)
    flags = out[out['is_audit_only'] == False]
    assert len(flags) == 0
    assert (
        detector._counters['cells_skipped_weak_reference_rho'] >= 1)


# ----------------------------------------------- cohort stratification


def test_cohort_stratification(detector_factory):
    """
    OR has slope=0 on CHR only; HC at OR matches peers. The CHR
    cell at OR should flag, the HC cell should not.
    """
    rng = np.random.default_rng(4)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL', 'OR'):
        rows.extend(_correlated_site(
            site, 40, slope=1.0, noise=2.0, rng=rng,
            cohort='hc'))
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_correlated_site(
            site, 40, slope=1.0, noise=2.0, rng=rng,
            cohort='chr'))
    rows.extend(_correlated_site(
        'OR', 40, slope=0.0, noise=8.0, rng=rng,
        cohort='chr'))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory(pairs=_DEFAULT_PAIR)
    out = detector._compute_candidates(long_df)
    flags = out[out['is_audit_only'] == False]
    or_chr = flags[(flags['site'] == 'OR')
                   & (flags['cohort'] == 'chr')]
    or_hc = flags[(flags['site'] == 'OR')
                  & (flags['cohort'] == 'hc')]
    assert len(or_chr) >= 1
    assert len(or_hc) == 0


# ----------------------------------------------- sample-size guard


def test_small_n_audit_only(detector_factory):
    """
    Site BB has n=10. Even with slope=0 (broken correlation), BB
    must emit an AUDIT row, not a flag.
    """
    rng = np.random.default_rng(5)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_correlated_site(
            site, 40, slope=1.0, noise=2.0, rng=rng))
    rows.extend(_correlated_site(
        'BB', 10, slope=0.0, noise=8.0, rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory(pairs=_DEFAULT_PAIR)
    out = detector._compute_candidates(long_df)
    bb_rows = out[out['site'] == 'BB']
    assert len(bb_rows) >= 1
    assert (bb_rows['is_audit_only']).all()
    # No other site should false-flag.
    others = out[(out['site'] != 'BB')
                 & (out['is_audit_only'] == False)]
    assert len(others) == 0


# ----------------------------------------------- schema and empty input


def test_output_schema_columns(detector_factory):
    rng = np.random.default_rng(6)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_correlated_site(
            site, 40, slope=1.0, noise=2.0, rng=rng))
    rows.extend(_correlated_site(
        'OR', 40, slope=-1.0, noise=2.0, rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory(pairs=_DEFAULT_PAIR)
    out = detector._compute_candidates(long_df)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert 'evidence_timepoints' in out.columns


def test_empty_input_empty_output(detector_factory):
    long_df = pd.DataFrame(columns=LONG_COLS)
    detector = detector_factory(pairs=_DEFAULT_PAIR)
    out = detector._compute_candidates(long_df)
    assert len(out) == 0
    assert list(out.columns) == OUTPUT_COLUMNS


def test_empty_pair_list_empty_output(detector_factory):
    """No pairs to check → no work to do."""
    rng = np.random.default_rng(7)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_correlated_site(
            site, 40, slope=1.0, noise=2.0, rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory(pairs=[])
    out = detector._compute_candidates(long_df)
    assert len(out) == 0


# ----------------------------------------------- file output


def test_file_output_atomic(detector_factory, tmp_path):
    rng = np.random.default_rng(8)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_correlated_site(
            site, 40, slope=1.0, noise=2.0, rng=rng))
    rows.extend(_correlated_site(
        'OR', 40, slope=-1.0, noise=2.0, rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    long_path = (
        tmp_path / "dependencies"
        / "multi_tp_TEST_numeric_long.parquet")
    long_df.to_parquet(long_path, index=False)

    detector = detector_factory(pairs=_DEFAULT_PAIR)
    detector.run()
    out_dir = (
        tmp_path / "output" / "combined_outputs" / "discovery")
    out_path = out_dir / "site_correlation_drift.parquet"
    assert out_path.exists()
    assert list(out_dir.glob("*.tmp")) == []
    read_back = pd.read_parquet(out_path)
    flags = read_back[read_back['is_audit_only'] == False]
    assert (flags['site'] == 'OR').any()
