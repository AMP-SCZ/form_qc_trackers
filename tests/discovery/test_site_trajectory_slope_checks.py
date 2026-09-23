"""
Tests for qc_types/discovery/site_trajectory_slope_checks.py.

The detector fits a MixedLM with subject random intercept and site
random slope on canonical_tp_index, then flags sites whose slope
BLUP is an outlier among peer sites. Tests verify it catches a
site whose trajectory slope is shifted, doesn't false-flag a clean
cohort, and handles small sites via audit-only rows.
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

pytest.importorskip("statsmodels")

from qc_types.discovery.site_trajectory_slope_checks import (
    SiteTrajectorySlopeChecks,
    OUTPUT_COLUMNS,
)


LONG_COLS = [
    'subjectid', 'network', 'timepoint', 'cohort', 'event_date',
    'source_form', 'variable', 'value', 'value_numeric',
    'is_missing_code',
]

TEST_TPS = ['baseline', 'month1', 'month2', 'month3', 'month4']


def _row(subjectid, tp, variable, value, cohort='chr',
         network='TEST', source_form='form_a'):
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
        'is_missing_code': False,
    }


def _site_trajectory_rows(site, n_subjects, slope_per_tp,
                          baseline_mean, noise_sd, rng,
                          variable='test_var', cohort='chr',
                          tps=None):
    """
    n_subjects subjects, each observed at each tp in `tps`. Each
    subject's value at tp_i = baseline_mean + subject_offset +
    slope_per_tp * i + N(0, noise_sd).
    """
    if tps is None:
        tps = list(TEST_TPS)
    rows = []
    for i in range(n_subjects):
        subjectid = f"{site}{i+1:04d}"
        subj_offset = float(rng.normal(0, 1.5))
        for j, tp in enumerate(tps):
            val = (baseline_mean + subj_offset
                   + slope_per_tp * j
                   + float(rng.normal(0, noise_sd)))
            rows.append(_row(
                subjectid, tp, variable, val,
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

    import qc_types.discovery.site_trajectory_slope_checks as mod
    monkeypatch.setattr(mod, 'Utils', FakeUtils)

    def _make(**overrides):
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "site_trajectory_slope": {
                    "enabled": True,
                    # Sprint 1 P0-5: tests use synthetic single-
                    # variable data; explicitly allow unbounded so
                    # the safety-rail at __init__ doesn't reject
                    # the empty allowlist.
                    "allow_unbounded": True,
                    "networks": ["TEST"],
                    "min_obs_per_subject": 3,
                    "min_subjects_per_site": 20,
                    "min_eligible_sites": 3,
                    "z_threshold": 2.5,
                    **overrides,
                }
            },
        }
        return SiteTrajectorySlopeChecks(config_dict=cfg)

    return _make


# ----------------------------------------------- happy path


def test_all_sites_same_slope_no_flags(detector_factory):
    """
    Four sites, all with slope=+1.0/tp. No site is an outlier.
    Expect zero flags.
    """
    rng = np.random.default_rng(0)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_trajectory_rows(
            site, 30, slope_per_tp=1.0, baseline_mean=50,
            noise_sd=1.0, rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    flags = out[out['is_audit_only'] == False]
    assert len(flags) == 0


# ----------------------------------------------- one site shifted slope


def test_one_site_shifted_slope_flagged(detector_factory):
    """
    Four sites with slope=+1.0/tp, OR with slope=+5.0/tp (large
    positive deviation). OR's BLUP should be a strong outlier.
    """
    rng = np.random.default_rng(1)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_trajectory_rows(
            site, 30, slope_per_tp=1.0, baseline_mean=50,
            noise_sd=1.0, rng=rng))
    rows.extend(_site_trajectory_rows(
        'OR', 30, slope_per_tp=5.0, baseline_mean=50,
        noise_sd=1.0, rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    flags = out[out['is_audit_only'] == False]
    assert (flags['site'] == 'OR').any(), (
        "expected OR to flag with its +5.0/tp slope shift")
    # Other peer sites should not flag.
    other = flags[flags['site'] != 'OR']
    assert len(other) == 0
    flag = flags[flags['site'] == 'OR'].iloc[0]
    assert flag['slope_blup'] > 1.0  # site BLUP captures the shift
    assert abs(flag['z_slope']) > 2.5


# ----------------------------------------------- cohort fixed effect


def test_cohort_fixed_effect_does_not_flag(detector_factory):
    """
    Different cohorts have legitimately different slopes, but no
    site has a within-cohort outlier slope. Should not flag.
    """
    rng = np.random.default_rng(2)
    rows = []
    # CHR cohort: slope=+1.0
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_trajectory_rows(
            site, 30, slope_per_tp=1.0, baseline_mean=60,
            noise_sd=1.0, rng=rng, cohort='chr'))
    # HC cohort: slope=0.0 (flat)
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_trajectory_rows(
            site, 30, slope_per_tp=0.0, baseline_mean=20,
            noise_sd=1.0, rng=rng, cohort='hc'))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    flags = out[out['is_audit_only'] == False]
    assert len(flags) == 0


# ----------------------------------------------- sample-size guard


def test_small_site_emits_audit_only(detector_factory):
    """
    Site BB has only 5 subjects. It must not enter the model; it
    should emit an audit row instead. Other sites with normal
    slopes are not flagged.
    """
    rng = np.random.default_rng(3)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_trajectory_rows(
            site, 30, slope_per_tp=1.0, baseline_mean=50,
            noise_sd=1.0, rng=rng))
    rows.extend(_site_trajectory_rows(
        'BB', 5, slope_per_tp=10.0, baseline_mean=50,
        noise_sd=1.0, rng=rng))  # tiny but extreme
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    bb_rows = out[out['site'] == 'BB']
    assert (bb_rows['is_audit_only']).all()
    flags = out[out['is_audit_only'] == False]
    assert len(flags) == 0


# ----------------------------------------------- too few eligible sites


def test_too_few_eligible_sites_skipped(detector_factory):
    """
    Only two sites have n_subjects > min — below min_eligible_sites=3.
    Variable should be silently skipped.
    """
    rng = np.random.default_rng(4)
    rows = []
    for site in ('WU', 'YA'):
        rows.extend(_site_trajectory_rows(
            site, 30, slope_per_tp=1.0, baseline_mean=50,
            noise_sd=1.0, rng=rng))
    # KC has too few.
    rows.extend(_site_trajectory_rows(
        'KC', 5, slope_per_tp=1.0, baseline_mean=50,
        noise_sd=1.0, rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    flags = out[out['is_audit_only'] == False]
    assert len(flags) == 0
    assert (detector._counters[
        'variables_skipped_too_few_sites'] >= 1)


# ----------------------------------------------- floating/conversion excluded


def test_floating_and_conversion_excluded(detector_factory):
    """
    Conversion rows must not enter the mixed model. If they did,
    the slope estimates would be polluted by extreme conversion
    values.
    """
    rng = np.random.default_rng(5)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_trajectory_rows(
            site, 30, slope_per_tp=1.0, baseline_mean=50,
            noise_sd=1.0, rng=rng))
    # Add conversion rows at OR with wildly high values.
    for i in range(30):
        rows.append(_row(
            f"OR{i+1:04d}", 'conversion', 'test_var', 500.0))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    detector = detector_factory()
    out = detector._compute_candidates(long_df)
    # OR has zero scheduled-tp observations → would be filtered out
    # by min_obs_per_subject and never enter the model.
    conv_rows = out[out['timepoint'] == 'conversion']
    assert len(conv_rows) == 0


# ----------------------------------------------- schema and empty


def test_output_schema_columns(detector_factory):
    rng = np.random.default_rng(6)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_trajectory_rows(
            site, 30, slope_per_tp=1.0, baseline_mean=50,
            noise_sd=1.0, rng=rng))
    rows.extend(_site_trajectory_rows(
        'OR', 30, slope_per_tp=5.0, baseline_mean=50,
        noise_sd=1.0, rng=rng))
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


def test_empty_allowlist_without_allow_unbounded_raises(
    tmp_path, monkeypatch
):
    """
    Sprint 1 P0-5: an empty variables_allowlist without explicit
    allow_unbounded=True must FAIL at __init__ — running MixedLM
    against every variable at AMPSCZ scale is hours-to-days, so
    silent "run on everything" is the wrong default.
    """
    class FakeUtils:
        absolute_path = str(tmp_path)

        def create_timepoint_list(self):
            return list(TEST_TPS)

    import qc_types.discovery.site_trajectory_slope_checks as mod
    monkeypatch.setattr(mod, 'Utils', FakeUtils)
    cfg = {
        "paths": {
            "dependencies_path": str(tmp_path) + "/",
            "output_path": str(tmp_path) + "/",
        },
        "testing_enabled": "False",
        "discovery": {
            "site_trajectory_slope": {
                "enabled": True,
                "networks": ["TEST"],
                # variables_allowlist intentionally omitted (==[]).
                # allow_unbounded intentionally omitted (==False).
            }
        },
    }
    with pytest.raises(RuntimeError) as ei:
        SiteTrajectorySlopeChecks(config_dict=cfg)
    msg = str(ei.value)
    assert "variables_allowlist" in msg
    assert "allow_unbounded" in msg


def test_allow_unbounded_lets_empty_allowlist_run(
    tmp_path, monkeypatch
):
    """
    Same configuration with allow_unbounded=True must instantiate
    cleanly (the operator has acknowledged the cost).
    """
    class FakeUtils:
        absolute_path = str(tmp_path)

        def create_timepoint_list(self):
            return list(TEST_TPS)

    import qc_types.discovery.site_trajectory_slope_checks as mod
    monkeypatch.setattr(mod, 'Utils', FakeUtils)
    cfg = {
        "paths": {
            "dependencies_path": str(tmp_path) + "/",
            "output_path": str(tmp_path) + "/",
        },
        "testing_enabled": "False",
        "discovery": {
            "site_trajectory_slope": {
                "enabled": True,
                "networks": ["TEST"],
                "allow_unbounded": True,
            }
        },
    }
    detector = SiteTrajectorySlopeChecks(config_dict=cfg)
    assert detector.allow_unbounded is True


def test_file_output_atomic(detector_factory, tmp_path):
    rng = np.random.default_rng(7)
    rows = []
    for site in ('WU', 'YA', 'KC', 'NL'):
        rows.extend(_site_trajectory_rows(
            site, 30, slope_per_tp=1.0, baseline_mean=50,
            noise_sd=1.0, rng=rng))
    rows.extend(_site_trajectory_rows(
        'OR', 30, slope_per_tp=5.0, baseline_mean=50,
        noise_sd=1.0, rng=rng))
    long_df = pd.DataFrame(rows, columns=LONG_COLS)
    long_path = (
        tmp_path / "dependencies"
        / "multi_tp_TEST_numeric_long.parquet")
    long_df.to_parquet(long_path, index=False)

    detector = detector_factory()
    detector.run()
    out_dir = (
        tmp_path / "output" / "combined_outputs" / "discovery")
    out_path = out_dir / "site_trajectory_slope.parquet"
    assert out_path.exists()
    assert list(out_dir.glob("*.tmp")) == []
    read_back = pd.read_parquet(out_path)
    flags = read_back[read_back['is_audit_only'] == False]
    assert (flags['site'] == 'OR').any()
