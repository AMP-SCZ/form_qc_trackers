"""
Tests for qc_types/discovery/site_distribution_drift_checks.py.

Goal: confirm the detector correctly identifies systematic site-
level distribution shifts while NOT confusing them with true
cohort/timepoint differences.
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

from qc_types.discovery.site_distribution_drift_checks import (
    SiteDistributionDriftChecks,
    OUTPUT_COLUMNS,
)


LONG_COLS = [
    'subjectid', 'network', 'timepoint', 'cohort', 'event_date',
    'source_form', 'variable', 'value', 'value_numeric',
    'is_missing_code',
]

TEST_TPS = ['baseline', 'month1', 'month2', 'month3']


def _row(subjectid, tp, variable, value_numeric, cohort='chr',
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
        'value': str(value_numeric),
        'value_numeric': float(value_numeric),
        'is_missing_code': is_missing_code,
    }


def _site_cohort(site, n_subjects, vals_dist, tp='baseline',
                 variable='test_var', cohort='chr',
                 source_form='form_a', rng=None):
    """
    Build n_subjects subjects at one site, sampling values from
    vals_dist (callable taking rng → float).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    rows = []
    for i in range(n_subjects):
        subjectid = f"{site}{i+1:04d}"
        rows.append(_row(
            subjectid, tp, variable,
            float(vals_dist(rng)),
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
        all_sites = {
            'TEST': ['OR', 'WU', 'YA', 'KC', 'NL', 'SD', 'IR'],
            'PRONET': [],
            'PRESCIENT': [],
        }

        def create_timepoint_list(self):
            return list(TEST_TPS)

    import qc_types.discovery.site_distribution_drift_checks as mod
    monkeypatch.setattr(mod, 'Utils', FakeUtils)

    def _make(variable_ranges=None, **overrides):
        if variable_ranges is not None:
            (deps / "variable_ranges.json").write_text(
                json.dumps(variable_ranges), encoding="utf-8")
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "site_distribution_drift": {
                    "enabled": True,
                    "networks": ["TEST"],
                    "min_n_per_cell": 30,
                    "z_threshold_median": 3.0,
                    "z_threshold_iqr": 3.0,
                    "delta_floor_ceiling_threshold": 0.15,
                    **overrides,
                }
            },
        }
        return SiteDistributionDriftChecks(config_dict=cfg)

    return _make


# ----------------------------------------------- happy-path baseline


def test_all_sites_equal_no_flags(detector_factory):
    """
    Five sites × 40 subjects, all drawing from N(50, 4). No site is
    systematically shifted. Expect zero flags.
    """
    rng = np.random.default_rng(0)
    rows = []
    for site in ('OR', 'WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_cohort(
            site, 40, lambda r: r.normal(50, 4), rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)

    detector = detector_factory()
    formatted, _ = detector._compute_candidates(long_df)
    flags = formatted[formatted['is_audit_only'] == False]
    assert len(flags) == 0


# ----------------------------------------------- median shift


def test_one_site_median_shift_flagged(detector_factory):
    """
    Five sites at N(50, 3); one site SHIFTED at N(60, 3). Median
    z should fire.
    """
    rng = np.random.default_rng(1)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_cohort(
            site, 40, lambda r: r.normal(50, 3), rng=rng))
    rows.extend(_site_cohort(
        'OR', 40, lambda r: r.normal(60, 3), rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)

    detector = detector_factory()
    formatted, _ = detector._compute_candidates(long_df)
    flags = formatted[formatted['is_audit_only'] == False]
    assert (flags['site'] == 'OR').any(), (
        "expected a flag for the shifted site OR")
    # No other site flagged.
    assert flags['site'].nunique() == 1
    flag = flags[flags['site'] == 'OR'].iloc[0]
    assert flag['z_median'] > 3.0


# ----------------------------------------------- IQR compression


def test_one_site_iqr_compression_flagged(detector_factory):
    """
    Four sites with normal spread; one site with tightly clustered
    values (low IQR). z_iqr should fire negative.
    """
    rng = np.random.default_rng(2)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_cohort(
            site, 40, lambda r: r.normal(50, 5), rng=rng))
    # Compressed site — values clustered very near 50.
    rows.extend(_site_cohort(
        'OR', 40, lambda r: r.normal(50, 0.4), rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)

    detector = detector_factory()
    formatted, _ = detector._compute_candidates(long_df)
    flags = formatted[formatted['is_audit_only'] == False]
    assert (flags['site'] == 'OR').any()
    flag = flags[flags['site'] == 'OR'].iloc[0]
    assert flag['z_iqr'] < -3.0


# ----------------------------------------------- ceiling clustering


def test_one_site_ceiling_clustered_flagged(detector_factory):
    """
    Variable range 0-7 (e.g. BPRS item). Four sites have realistic
    spread; one site has 50% of values at the ceiling (7).
    """
    variable_ranges = {'bprs_item': {'min': 0, 'max': 7}}
    rng = np.random.default_rng(3)
    rows = []

    def normal_clipped(r):
        v = r.normal(3.0, 1.2)
        return float(np.clip(round(v), 0, 7))

    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_cohort(
            site, 40, normal_clipped, variable='bprs_item',
            rng=rng))
    # Ceiling site — half the subjects sit at 7.
    def ceiling_dist(r):
        if r.random() < 0.5:
            return 7.0
        return float(np.clip(round(r.normal(3.0, 1.2)), 0, 7))
    rows.extend(_site_cohort(
        'OR', 40, ceiling_dist, variable='bprs_item', rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)

    detector = detector_factory(variable_ranges=variable_ranges)
    formatted, _ = detector._compute_candidates(long_df)
    flags = formatted[formatted['is_audit_only'] == False]
    assert (flags['site'] == 'OR').any()
    flag = flags[flags['site'] == 'OR'].iloc[0]
    assert flag['delta_p_at_ceiling'] > 0.15


# ----------------------------------------------- cohort stratification


def test_hc_shift_does_not_flag_chr_stratum(detector_factory):
    """
    The HC cohort at site OR is shifted by +10 vs other-site HC, but
    CHR is uniform across sites. The CHR stratum at OR must not
    flag (we should not see a CHR row for OR); we DO expect HC
    rows for OR to flag.
    """
    rng = np.random.default_rng(4)
    rows = []
    # CHR cohort — uniform across sites.
    for site in ('WU', 'YA', 'KC', 'NL', 'OR'):
        rows.extend(_site_cohort(
            site, 40, lambda r: r.normal(50, 4),
            cohort='chr', rng=rng))
    # HC cohort — only OR is shifted.
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_cohort(
            site, 40, lambda r: r.normal(30, 4),
            cohort='hc', rng=rng))
    rows.extend(_site_cohort(
        'OR', 40, lambda r: r.normal(40, 4),
        cohort='hc', rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)

    detector = detector_factory()
    formatted, _ = detector._compute_candidates(long_df)
    flags = formatted[formatted['is_audit_only'] == False]

    chr_or = flags[(flags['site'] == 'OR')
                   & (flags['cohort'] == 'chr')]
    hc_or = flags[(flags['site'] == 'OR')
                  & (flags['cohort'] == 'hc')]
    assert len(chr_or) == 0, (
        "CHR stratum at OR should not flag; HC shift must not bleed.")
    assert len(hc_or) >= 1, (
        "HC stratum at OR is shifted +10 — should flag.")


# ----------------------------------------------- tp stratification


def test_tp_stratification_isolates_shift(detector_factory):
    """
    OR is shifted only at month1; baseline is uniform. The flag
    should be at month1, not baseline.
    """
    rng = np.random.default_rng(5)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL', 'OR'):
        rows.extend(_site_cohort(
            site, 40, lambda r: r.normal(50, 4),
            tp='baseline', rng=rng))
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_cohort(
            site, 40, lambda r: r.normal(50, 4),
            tp='month1', rng=rng))
    rows.extend(_site_cohort(
        'OR', 40, lambda r: r.normal(62, 4),
        tp='month1', rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)

    detector = detector_factory()
    formatted, _ = detector._compute_candidates(long_df)
    flags = formatted[formatted['is_audit_only'] == False]
    or_flags = flags[flags['site'] == 'OR']
    # All OR flags should be at month1 only.
    assert (or_flags['timepoint'] == 'month1').all()
    assert len(or_flags) >= 1


# ----------------------------------------------- sample-size guard


def test_small_n_emits_audit_not_flag(detector_factory):
    """
    Site BB has only 10 subjects (below min_n_per_cell=30); even
    if shifted, BB should appear as an AUDIT row, not a flag.
    Other sites have n=40 so the reference is valid.
    """
    rng = np.random.default_rng(6)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_cohort(
            site, 40, lambda r: r.normal(50, 4), rng=rng))
    rows.extend(_site_cohort(
        'BB', 10, lambda r: r.normal(80, 4), rng=rng))  # shifted but small
    long_df = pd.DataFrame(rows, columns=LONG_COLS)

    detector = detector_factory()
    formatted, _ = detector._compute_candidates(long_df)
    bb_rows = formatted[formatted['site'] == 'BB']
    assert len(bb_rows) >= 1
    assert (bb_rows['is_audit_only']).all(), (
        "BB has n=10; must be audit-only (not flagged).")
    # Other sites should not produce false flags either.
    flagged = formatted[formatted['is_audit_only'] == False]
    assert len(flagged) == 0


# ----------------------------------------------- floating/conversion excluded


def test_floating_and_conversion_excluded(detector_factory):
    """
    Patch 4 convention: floating/conversion rows must be excluded
    from scoring even if heavily shifted at one site.
    """
    rng = np.random.default_rng(7)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL', 'OR'):
        rows.extend(_site_cohort(
            site, 40, lambda r: r.normal(50, 4),
            tp='baseline', rng=rng))
    # At conversion, OR has wildly different values — should NOT flag.
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_cohort(
            site, 40, lambda r: r.normal(50, 4),
            tp='conversion', rng=rng))
    rows.extend(_site_cohort(
        'OR', 40, lambda r: r.normal(200, 4),
        tp='conversion', rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)

    detector = detector_factory()
    formatted, _ = detector._compute_candidates(long_df)
    conversion_rows = formatted[
        formatted['timepoint'] == 'conversion']
    assert len(conversion_rows) == 0


# ----------------------------------------------- schema and empty cases


def test_output_schema_columns(detector_factory):
    rng = np.random.default_rng(8)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_cohort(
            site, 40, lambda r: r.normal(50, 4), rng=rng))
    rows.extend(_site_cohort(
        'OR', 40, lambda r: r.normal(60, 3), rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)

    detector = detector_factory()
    formatted, _ = detector._compute_candidates(long_df)
    assert list(formatted.columns) == OUTPUT_COLUMNS
    assert 'evidence_timepoints' in formatted.columns
    assert 'evidence_max_timepoint' in formatted.columns


def test_empty_input_empty_output(detector_factory):
    long_df = pd.DataFrame(columns=LONG_COLS)
    detector = detector_factory()
    formatted, _ = detector._compute_candidates(long_df)
    assert len(formatted) == 0
    assert list(formatted.columns) == OUTPUT_COLUMNS


# ----------------------------------------------- file output / atomic write


def test_file_output_atomic_and_readable(detector_factory, tmp_path):
    rng = np.random.default_rng(9)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_cohort(
            site, 40, lambda r: r.normal(50, 4), rng=rng))
    rows.extend(_site_cohort(
        'OR', 40, lambda r: r.normal(60, 3), rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    long_path = (
        tmp_path / "dependencies"
        / "multi_tp_TEST_numeric_long.parquet")
    long_df.to_parquet(long_path, index=False)

    detector = detector_factory()
    detector.run()
    out_dir = (
        tmp_path / "output" / "combined_outputs" / "discovery")
    out_path = out_dir / "site_distribution_drift.parquet"
    assert out_path.exists()
    # Atomic write left no .tmp residue.
    assert list(out_dir.glob("*.tmp")) == []
    read_back = pd.read_parquet(out_path)
    flags = read_back[read_back['is_audit_only'] == False]
    assert (flags['site'] == 'OR').any()
