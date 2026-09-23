"""
Tests for qc_types/discovery/point_spike_checks.py.

Test cases:
  PS-1: stable cohort at one tp + 1 spike subject whose neighbor tps
        are quiet → 1 flag at the spike tp
  PS-2: high-z point but adjacent visit is also high-z → not a spike →
        0 flags (the subject is broadly off, not a single-tp anomaly)
  PS-3: excluded source_form (digital_biomarkers_*) → 0 flags
  PS-4: missing-code regression — value_numeric=999 with
        is_missing_code=True does not feed the cohort distribution
  PS-5: insufficient cohort obs at the tp → 0 flags
  PS-6: degenerate MAD (every subject identical at the tp) → 0 flags
  Schema: missing required col → RuntimeError
  Empty:  no flags → DataFrame with expected columns and dtypes
  File:   end-to-end run() — parquet exists, is readable, no .tmp
          residue
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

from qc_types.discovery.point_spike_checks import PointSpikeChecks


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


def _stable_cohort_rows(n_subjects=22, variable='test_var',
                       source_form='form_a'):
    """
    n subjects with values centered near 50 at each tp, with enough
    spread so MAD > 0 (the spike detector skips degenerate-MAD
    timepoints). Each subject is assigned a deterministic offset so
    tests stay reproducible.
    """
    rows = []
    tps = ['baseline', 'month1', 'month2', 'month3']
    for i in range(n_subjects):
        sid = f"S{i+1:03d}"
        # Per-subject deterministic offset in {-2.0, -1.5, ..., +2.0}
        offset = -2.0 + (i % 9) * 0.5
        for j, tp in enumerate(tps):
            # Mild per-tp jitter so MAD is non-zero at every tp.
            val = 50.0 + offset + 0.1 * j
            rows.append({
                'subjectid': sid, 'timepoint': tp,
                'variable': variable, 'source_form': source_form,
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

    import qc_types.discovery.point_spike_checks as detector_module
    monkeypatch.setattr(detector_module, 'Utils', FakeUtils)

    def _make(**ps_overrides):
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "point_spike": {
                    "networks": ["TEST"],
                    "min_observations_per_tp": 20,
                    "high_threshold": 4.0,
                    "low_threshold": 2.0,
                    "require_at_least_one_neighbor": True,
                    "max_flags_to_write": None,
                    **ps_overrides,
                }
            },
        }
        return PointSpikeChecks(config_dict=cfg)

    return _make


# ---------------------------------------------------------------- PS-1

def test_ps1_single_spike_with_quiet_neighbors(detector_factory):
    detector = detector_factory()
    rows = _stable_cohort_rows(n_subjects=22)
    # SPIKE subject: stable at baseline/month1, big spike at month2,
    # back to baseline at month3.
    for tp, val in zip(
        ['baseline', 'month1', 'month2', 'month3'],
        [50.0, 50.5, 250.0, 50.5],
    ):
        rows.append({
            'subjectid': 'SPIKE', 'timepoint': tp,
            'variable': 'test_var', 'source_form': 'form_a',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)

    result = detector._compute_candidates(long_df)
    spike_flags = result[result['subjectid'] == 'SPIKE']
    assert len(spike_flags) == 1
    flag = spike_flags.iloc[0]
    assert flag['timepoint'] == 'month2'
    assert flag['observed_value'] == 250.0
    assert flag['num_neighbors_considered'] == 2
    assert abs(flag['prev_neighbor_z']) <= 2.0
    assert abs(flag['next_neighbor_z']) <= 2.0
    # Patch 6 — evidence_timepoints metadata. Spike at month2 with
    # observed prev=month1 and next=month3. Canonical order from
    # TEST_TP_ORDER puts the three tps in order month1,month2,month3.
    assert 'evidence_timepoints' in flag.index
    assert 'evidence_max_timepoint' in flag.index
    assert flag['evidence_timepoints'] == 'month1,month2,month3'
    assert flag['evidence_max_timepoint'] == 'month3'


def test_ps_evidence_timepoints_baseline_spike(detector_factory):
    """
    Earliest-tp spike: no prev neighbor, only a next neighbor.
    evidence_max_timepoint = next-neighbor tp, NOT the spike tp,
    which is exactly the leakage signal a downstream ML consumer
    needs to see.
    """
    detector = detector_factory()
    rows = _stable_cohort_rows(n_subjects=22)
    # Subject only has baseline + month1 + month2. Spike at baseline.
    for tp, val in zip(
        ['baseline', 'month1', 'month2'],
        [250.0, 50.5, 50.5],
    ):
        rows.append({
            'subjectid': 'BL_SPIKE', 'timepoint': tp,
            'variable': 'test_var', 'source_form': 'form_a',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    flag_rows = result[result['subjectid'] == 'BL_SPIKE']
    assert len(flag_rows) == 1
    flag = flag_rows.iloc[0]
    assert flag['timepoint'] == 'baseline'
    # Only the next neighbor was used (no prev).
    assert flag['evidence_timepoints'] == 'baseline,month1'
    assert flag['evidence_max_timepoint'] == 'month1'


# ---------------------------------------------------------------- PS-2

def test_ps2_broad_outlier_not_a_spike(detector_factory):
    """
    Subject is high at month1 AND month2 — broadly off, not a single-tp
    spike. The month1 candidate fails because next_neighbor_z (month2)
    is loud; the month2 candidate fails because prev_neighbor_z is loud.
    """
    detector = detector_factory()
    rows = _stable_cohort_rows(n_subjects=22)
    for tp, val in zip(
        ['baseline', 'month1', 'month2', 'month3'],
        [50.0, 250.0, 250.0, 50.5],
    ):
        rows.append({
            'subjectid': 'BROAD', 'timepoint': tp,
            'variable': 'test_var', 'source_form': 'form_a',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (result['subjectid'] == 'BROAD').sum() == 0


# ---------------------------------------------------------------- PS-3

def test_ps3_excluded_source_form(detector_factory):
    """
    The same spike pattern on a digital_biomarkers_ form produces zero
    flags because those rows are filtered before scoring.
    """
    detector = detector_factory()
    rows = _stable_cohort_rows(
        n_subjects=22, source_form='digital_biomarkers_axivity_checkin')
    for tp, val in zip(
        ['baseline', 'month1', 'month2', 'month3'],
        [50.0, 50.5, 250.0, 50.5],
    ):
        rows.append({
            'subjectid': 'SPIKE', 'timepoint': tp,
            'variable': 'test_var',
            'source_form': 'digital_biomarkers_axivity_checkin',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['rows_excluded_by_form_pattern'] > 0


# ---------------------------------------------------------------- PS-4

def test_ps4_missing_code_excluded_from_cohort(detector_factory):
    """
    A row with value_numeric=999 and is_missing_code=True must not
    enter the cohort median/MAD computation; otherwise the spike's
    cohort_z is suppressed.
    """
    detector = detector_factory()
    rows = _stable_cohort_rows(n_subjects=22)
    # An adversarial sibling at month2 carrying 999 with missing-code
    # true — if leaked into the cohort, it would inflate the MAD.
    rows.append({
        'subjectid': 'ADV', 'timepoint': 'month2',
        'variable': 'test_var', 'source_form': 'form_a',
        'value': '999', 'value_numeric': 999.0,
        'is_missing_code': True,
    })
    for tp, val in zip(
        ['baseline', 'month1', 'month2', 'month3'],
        [50.0, 50.5, 250.0, 50.5],
    ):
        rows.append({
            'subjectid': 'SPIKE', 'timepoint': tp,
            'variable': 'test_var', 'source_form': 'form_a',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (result['subjectid'] == 'SPIKE').sum() == 1


# ---------------------------------------------------------------- PS-5

def test_ps5_insufficient_cohort_at_tp_zero_flags(detector_factory):
    """
    Only 5 subjects at each tp; the per-(network, variable, tp) pool
    is below min_observations_per_tp (20) — variable is skipped at
    every tp and no spike fires.
    """
    detector = detector_factory(min_observations_per_tp=20)
    rows = []
    for i in range(5):
        sid = f"S{i+1}"
        for tp, val in zip(
            ['baseline', 'month1', 'month2', 'month3'],
            [50.0, 50.5, 50.0, 50.5],
        ):
            rows.append({
                'subjectid': sid, 'timepoint': tp,
                'variable': 'test_var', 'source_form': 'form_a',
                'value': str(val), 'value_numeric': val,
                'is_missing_code': False,
            })
    rows.append({
        'subjectid': 'SPIKE', 'timepoint': 'month2',
        'variable': 'test_var', 'source_form': 'form_a',
        'value': '250.0', 'value_numeric': 250.0,
        'is_missing_code': False,
    })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters[
        'tp_groups_skipped_insufficient_obs'] > 0


# ---------------------------------------------------------------- PS-6

def test_ps6_degenerate_mad_zero_flags(detector_factory):
    """
    Every subject at month2 has identical value → cohort MAD is 0 →
    the tp is skipped. No spike can be scored at that tp.
    """
    detector = detector_factory()
    rows = []
    for i in range(22):
        sid = f"S{i+1:03d}"
        for tp, val in zip(
            ['baseline', 'month1', 'month2', 'month3'],
            [50.0 + (i % 3) * 0.1, 50.0 + (i % 3) * 0.1,
             50.0,
             50.0 + (i % 3) * 0.1],
        ):
            rows.append({
                'subjectid': sid, 'timepoint': tp,
                'variable': 'test_var', 'source_form': 'form_a',
                'value': str(val), 'value_numeric': val,
                'is_missing_code': False,
            })
    rows.append({
        'subjectid': 'SPIKE', 'timepoint': 'month2',
        'variable': 'test_var', 'source_form': 'form_a',
        'value': '250.0', 'value_numeric': 250.0,
        'is_missing_code': False,
    })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    # month2 cohort is degenerate; SPIKE's month2 obs cannot be scored.
    assert (
        (result['subjectid'] == 'SPIKE')
        & (result['timepoint'] == 'month2')
    ).sum() == 0


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
            'variable': 'test_var', 'source_form': 'form_a',
            'value': '999', 'value_numeric': float('nan'),
            'is_missing_code': True,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0

    from qc_types.discovery.point_spike_checks import (
        PointSpikeChecks as _PSC)
    expected_cols = list(_PSC.OUTPUT_COLUMNS)
    assert list(result.columns) == expected_cols
    for col in (
        'value_numeric', 'expected_value', 'observed_value',
        'metric_value', 'severity_score', 'threshold',
        'cohort_median', 'cohort_mad_scale',
        'prev_neighbor_z', 'next_neighbor_z',
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
    rows = _stable_cohort_rows(n_subjects=22)
    for tp, val in zip(
        ['baseline', 'month1', 'month2', 'month3'],
        [50.0, 50.5, 250.0, 50.5],
    ):
        rows.append({
            'subjectid': 'SPIKE', 'timepoint': tp,
            'variable': 'test_var', 'source_form': 'form_a',
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
        out_dir / "point_spike_candidates.parquet")
    assert out_path.exists()
    out_df = pd.read_parquet(out_path)
    assert (out_df['subjectid'] == 'SPIKE').sum() >= 1
    tmps = list(out_dir.glob("*.tmp"))
    assert len(tmps) == 0


# =========================================================================
# Tier-1 review-driven tests
# =========================================================================

# ---------------------------------------------------------------- PS-7

def test_ps7_excluded_variable_pattern(detector_factory):
    """
    A spike pattern on a variable matching `*_complete` (default
    excluded pattern) produces zero flags.
    """
    detector = detector_factory()
    rows = _stable_cohort_rows(n_subjects=22, variable='test_complete')
    for tp, val in zip(
        ['baseline', 'month1', 'month2', 'month3'],
        [50.0, 50.5, 250.0, 50.5],
    ):
        rows.append({
            'subjectid': 'SPIKE', 'timepoint': tp,
            'variable': 'test_complete', 'source_form': 'form_a',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert len(result) == 0
    assert detector._counters['rows_excluded_by_variable_pattern'] > 0


# ---------------------------------------------------------------- PS-8

def test_ps8_nan_cohort_z_skipped(detector_factory):
    """
    A row whose value_numeric is NaN must not reach the spike branch
    even if it passes the producer's is_missing_code filter (defense
    against producer drift). The pre-filter already drops NaN values,
    but the NaN guard inside _select_spikes is the second line of
    defense; we exercise it here by directly invoking _select_spikes
    on a scored frame that includes a NaN cohort_z.
    """
    import numpy as np
    detector = detector_factory()
    # Build the smallest possible scored frame: one subject, two tps,
    # cohort_z NaN at the first tp, very high z at the second.
    scored = pd.DataFrame({
        'subjectid': ['S1', 'S1'],
        'network': ['TEST', 'TEST'],
        'timepoint': ['baseline', 'month1'],
        'variable': ['v', 'v'],
        'source_form': ['form_a', 'form_a'],
        'value': ['x', 'y'],
        'value_numeric': [10.0, 250.0],
        'cohort_median': [10.0, 10.0],
        'cohort_mad_scale': [1.0, 1.0],
        'cohort_z': [np.nan, 240.0],
    })
    scoring = scored.drop(columns=[
        'cohort_median', 'cohort_mad_scale', 'cohort_z'
    ]).copy()
    scoring['is_missing_code'] = False
    result = detector._select_spikes(scored, scoring)
    # The NaN row never enters the spike branch; the lone candidate
    # at month1 has no neighbors (baseline filtered as NaN), so under
    # default require_at_least_one_neighbor it's also dropped.
    assert len(result) == 0
    assert detector._counters['candidates_dropped_nan_z'] >= 1


# ---------------------------------------------------------------- PS-9

def test_ps9_duplicate_subj_var_tp_deduped(detector_factory):
    """
    Duplicate (network, variable, subjectid, timepoint) rows in the
    scored frame are deduped (keep='first') with a warning + counter,
    matching the producer-resilient pattern every sibling detector
    uses. Crashing the whole discovery chain on a producer-side
    re-export quirk is worse than picking the first row.
    """
    detector = detector_factory()
    scored = pd.DataFrame({
        'subjectid': ['S1', 'S1'],
        'network': ['TEST', 'TEST'],
        'timepoint': ['baseline', 'baseline'],
        'variable': ['v', 'v'],
        'source_form': ['form_a', 'form_a'],
        'value': ['a', 'b'],
        'value_numeric': [10.0, 11.0],
        'cohort_median': [10.0, 10.0],
        'cohort_mad_scale': [1.0, 1.0],
        'cohort_z': [0.0, 1.0],
    })
    scoring = scored.drop(columns=[
        'cohort_median', 'cohort_mad_scale', 'cohort_z'
    ]).copy()
    scoring['is_missing_code'] = False
    # Should NOT raise.
    result = detector._select_spikes(scored, scoring)
    # 2 duplicate rows fed in, 1 retained after dedupe → no flags
    # (neither value clears the threshold), but the counter records
    # the dedupe so the operator can see it.
    assert detector._counters['scored_rows_deduped'] >= 1
    assert len(result) == 0


# ---------------------------------------------------------------- PS-10

def test_ps10_raw_neighbor_filtered_out_skips_candidate(detector_factory):
    """
    Subject SPIKE has 4 raw obs (baseline/month1/month2/month3). The
    cohort at month1 is entirely identical (MAD=0) so it's dropped
    from scored. The month2 candidate has month3 as its "next"
    neighbor (quiet) but month1 exists in raw and is unscored. Under
    the new semantics, that candidate is dropped — we can't verify
    the prev neighbor is quiet.
    """
    detector = detector_factory()
    rows = []
    for i in range(22):
        sid = f"S{i+1:03d}"
        # baseline has variance; month1 is identical across cohort
        # (degenerate); month2 and month3 have variance.
        rows.append({
            'subjectid': sid, 'timepoint': 'baseline',
            'variable': 'test_var', 'source_form': 'form_a',
            'value': str(50.0 + (i % 7) * 0.3),
            'value_numeric': 50.0 + (i % 7) * 0.3,
            'is_missing_code': False,
        })
        rows.append({
            'subjectid': sid, 'timepoint': 'month1',
            'variable': 'test_var', 'source_form': 'form_a',
            'value': '50.0', 'value_numeric': 50.0,
            'is_missing_code': False,
        })
        for tp, val in (
            ('month2', 50.0 + (i % 7) * 0.3),
            ('month3', 50.0 + (i % 7) * 0.3),
        ):
            rows.append({
                'subjectid': sid, 'timepoint': tp,
                'variable': 'test_var', 'source_form': 'form_a',
                'value': str(val), 'value_numeric': val,
                'is_missing_code': False,
            })
    for tp, val in zip(
        ['baseline', 'month1', 'month2', 'month3'],
        [50.5, 50.0, 250.0, 50.5],
    ):
        rows.append({
            'subjectid': 'SPIKE', 'timepoint': tp,
            'variable': 'test_var', 'source_form': 'form_a',
            'value': str(val), 'value_numeric': val,
            'is_missing_code': False,
        })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    # SPIKE's month2 z is huge, but its month1 raw obs exists yet was
    # filtered (degenerate cohort) → candidate dropped under the new
    # "unknown is loud" rule.
    assert (
        (result['subjectid'] == 'SPIKE')
        & (result['timepoint'] == 'month2')
    ).sum() == 0
    assert detector._counters[
        'candidates_dropped_unscored_neighbor'] >= 1


# ---------------------------------------------------------------- PS-11

def test_ps11_severity_is_abs_z(detector_factory):
    """
    Severity ranking is now |cohort_z|, not |z| − max(neighbor |z|).
    Confirm: a subject with z=12 and a neighbor z=1.9 outranks a
    subject with z=8 and a neighbor z=0.0. Under the OLD formula,
    severities would have been 10.1 vs 8.0 (close); under the NEW
    formula, 12 vs 8 (the obviously-bigger outlier wins).
    """
    detector = detector_factory()
    rows = _stable_cohort_rows(n_subjects=22)
    # SUBJ_BIG: huge spike at month2, mild neighbor noise at month1.
    big_path = {
        'baseline': 50.0, 'month1': 51.9, 'month2': 1000.0,
        'month3': 50.5,
    }
    # SUBJ_SMALL: smaller spike at month2, perfectly quiet neighbors.
    small_path = {
        'baseline': 50.0, 'month1': 50.0, 'month2': 200.0,
        'month3': 50.0,
    }
    for sid, path in (('BIG', big_path), ('SMALL', small_path)):
        for tp, val in path.items():
            rows.append({
                'subjectid': sid, 'timepoint': tp,
                'variable': 'test_var', 'source_form': 'form_a',
                'value': str(val), 'value_numeric': val,
                'is_missing_code': False,
            })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    flagged = result[result['timepoint'] == 'month2'].copy()
    big = flagged[flagged['subjectid'] == 'BIG']
    small = flagged[flagged['subjectid'] == 'SMALL']
    assert len(big) == 1
    assert len(small) == 1
    assert (
        float(big.iloc[0]['severity_score'])
        > float(small.iloc[0]['severity_score'])
    ), (
        f"Expected BIG severity > SMALL severity; got "
        f"BIG={big.iloc[0]['severity_score']}, "
        f"SMALL={small.iloc[0]['severity_score']}"
    )


# ---------------------------------------------------------------- PS-12

def test_ps12_per_variable_cap(detector_factory):
    """
    With max_flags_per_variable=2, one variable that would otherwise
    produce 5 spikes is capped to 2 rows.
    """
    detector = detector_factory(max_flags_per_variable=2)
    rows = _stable_cohort_rows(n_subjects=22)
    # Five subjects each get a spike at month2 on the same variable.
    for k, sid in enumerate(['A', 'B', 'C', 'D', 'E']):
        for tp, val in zip(
            ['baseline', 'month1', 'month2', 'month3'],
            [50.0, 50.5, 100.0 + 30.0 * k, 50.5],
        ):
            rows.append({
                'subjectid': sid, 'timepoint': tp,
                'variable': 'test_var', 'source_form': 'form_a',
                'value': str(val), 'value_numeric': val,
                'is_missing_code': False,
            })
    long_df = _make_long_df(rows)
    result = detector._compute_candidates(long_df)
    assert (result['variable'] == 'test_var').sum() == 2
    assert detector._counters[
        'spikes_dropped_by_per_variable_cap'] >= 3
