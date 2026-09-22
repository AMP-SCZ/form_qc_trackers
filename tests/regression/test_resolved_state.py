"""
Regression tests for generate_reports/calculate_resolved_errors.py.

Covers:
  - first-run smoke (no old/new/current present → no crash, no false flags)
  - Patch #3: missing-new with prior history → FileNotFoundError
  - Patch #3: missing-new with no history → benign empty new_df
  - resolved/reopened state machine: was-resolved row in old that
    re-appears in new gets `currently_resolved=False` and a fresh
    `dates_detected` entry (the reopen path)
  - resolved-only-in-old: open flag in old that disappears from new
    gets marked `currently_resolved=True` with today appended to
    `dates_resolved`

Tests are fully isolated from the real project config.json. The
calc_factory fixture monkeypatches `Utils` in the module namespace
with a minimal FakeUtils stub. All paths and config values come
from per-test synthetic config_dicts written to tmp_path. No real
CSV files. No Dropbox. No production paths.
"""

import os
import sys
import json
import pytest
import pandas as pd
from datetime import datetime

# Add project root to sys.path so the import resolves regardless
# of pytest cwd. Three dirname() hops: this file → tests/regression
# → tests → project root.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import generate_reports.calculate_resolved_errors as calc_mod


# Mirror of the relevant subset of form_check.create_row_output's
# schema. Tests build rows from this template.
_DEFAULT_ROW = {
    'network': 'TEST',
    'subject': 'S001',
    'cohort': 'CHR',
    'affected_timepoints': 'tp_1',
    'subject_current_timepoint': 'tp_1',
    'affected_forms': 'form_a',
    'affected_variables': 'var_a',
    'displayed_form': 'form_a',
    'displayed_timepoint': 'tp_1',
    'displayed_variable': 'var_a',
    'var_translations': '',
    'error_message': 'var_a : something is wrong',
    'error_removed': False,
    'reports': 'Main Report',
    'withdrawn_status': False,
    'inclusion_status': 'included',
    'excluded_enabled': False,
    'withdrawn_enabled': False,
    'nda_excluder': True,
    'priority': False,
    'priority_item': False,
    'dates_detected': '2026-04-01',
    'time_since_last_detection': '',
    'dates_resolved': '',
    'currently_resolved': False,
    'manually_resolved': '',
    'comments': '',
    'site_comments': '',
}


def _make_row(**overrides):
    row = dict(_DEFAULT_ROW)
    row.update(overrides)
    return row


def _write_parquet(path, rows):
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(_DEFAULT_ROW.keys()))
    df.to_parquet(path, index=False)


@pytest.fixture
def calc_factory(tmp_path, monkeypatch):
    """
    Build a CalculateResolvedErrors instance with paths under
    tmp_path. Real Utils is replaced with FakeUtils so the
    constructor never reads the real project config.json.
    """
    deps = tmp_path / "dependencies"
    out_root = tmp_path / "output"
    combined_outputs = out_root / "combined_outputs"
    deps.mkdir(exist_ok=True)
    (combined_outputs / "old_output").mkdir(parents=True, exist_ok=True)
    (combined_outputs / "new_output").mkdir(parents=True, exist_ok=True)
    (combined_outputs / "current_output").mkdir(parents=True, exist_ok=True)

    cfg = {
        "paths": {
            "dependencies_path": str(deps) + "/",
            "combined_csv_path": str(tmp_path / "csvs") + "/",
            "output_path": str(out_root) + "/",
        },
        "testing_enabled": "False",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))

    class FakeUtils:
        # Includes the synthetic 'TEST' network used by _make_row so the
        # scoped-network guard keeps every fixture row in scope, matching
        # the pre-scoping behavior these tests were written against.
        pipeline_networks = ("TEST", "PRONET", "PRESCIENT")
        absolute_path = str(tmp_path)
        config_info = cfg

        def load_dependency_json(self, name):
            # melbourne_ra_subs.json lookup; return empty mapping so
            # CalculateResolvedErrors __init__ doesn't fail.
            return {}

        def check_if_val_date_format(self, val):
            try:
                datetime.strptime(str(val), "%Y-%m-%d")
                return True
            except (ValueError, TypeError):
                return False

        def days_since_today(self, val):
            d = datetime.strptime(str(val), "%Y-%m-%d")
            return (datetime.today() - d).days

        def reverse_dictionary(self, inp_dict):
            return {value: key for key, value in inp_dict.items()}

    monkeypatch.setattr(calc_mod, 'Utils', FakeUtils)

    def _make():
        # formatted_col_names is only used inside loop_dropbox_files,
        # which the tests don't exercise. Safe to pass {}.
        return calc_mod.CalculateResolvedErrors(formatted_col_names={})

    return _make, tmp_path, combined_outputs


# ---------------------------------------------------- First-run smoke

def test_first_run_no_history(calc_factory):
    make, _tmp, combined = calc_factory
    calc = make()
    # No old, no new, no current — fresh install. determine_resolved_
    # rows should run without crashing and without writing anything
    # malformed.
    calc.determine_resolved_rows()
    current = combined / "current_output" / "combined_qc_flags.parquet"
    # Either no current was produced (legitimate first run) or an
    # empty parquet was written. Both are acceptable; what's NOT
    # acceptable is a crash or a non-empty current with phantom rows.
    if current.exists():
        df = pd.read_parquet(current)
        assert len(df) == 0, "first-run current_output must be empty"


def test_optional_numeric_evidence_survives_current_parquet_write(calc_factory):
    """Regression for ArrowTypeError on date_gap_days.

    A date flag supplies an integer gap while every non-date flag has null in
    that optional column. The old global ``fillna('')`` changed the numeric
    column to object containing floats + strings, which PyArrow refused to
    serialize during new -> current promotion.
    """
    make, _tmp, combined = calc_factory
    calc = make()
    new_path = combined / "new_output" / "combined_qc_flags.parquet"
    rows = [
        _make_row(subject="S001", displayed_variable="visit_date",
                  error_message="visit date is before baseline",
                  date_gap_days=10),
        _make_row(subject="S002", displayed_variable="var_b",
                  error_message="var_b is invalid"),
    ]
    pd.DataFrame(rows).to_parquet(new_path, index=False)

    calc.determine_resolved_rows()

    current_path = combined / "current_output" / "combined_qc_flags.parquet"
    current = pd.read_parquet(current_path).sort_values("subject")
    assert current["date_gap_days"].iloc[0] == 10
    assert pd.isna(current["date_gap_days"].iloc[1])
    assert pd.api.types.is_integer_dtype(current["date_gap_days"].dtype)
    assert str(current["time_since_last_detection"].dtype) == "Int64"
    assert current["time_since_last_detection"].isna().all()


def test_numeric_output_normalization_is_idempotent_on_nullable_integer():
    """The read and write boundaries both normalize the frame.

    pandas 1.x crashes if ``replace('', pd.NA)`` is applied to an already-Int64
    series, so the second pass must recognize and preserve numeric dtype.
    """
    legacy = pd.DataFrame({"date_gap_days": pd.Series([10.0, "", None],
                                                        dtype="object")})
    once = calc_mod.CalculateResolvedErrors._normalize_output_frame(legacy)
    twice = calc_mod.CalculateResolvedErrors._normalize_output_frame(once)
    assert str(once["date_gap_days"].dtype) == "Int64"
    pd.testing.assert_series_equal(once["date_gap_days"],
                                   twice["date_gap_days"])


def test_dropbox_merge_preserves_nullable_numeric_evidence(calc_factory):
    """Merging reviewer edits must not fill Int64 nulls with empty strings."""
    make, _tmp, _combined = calc_factory
    calc = make()
    before = _current_flags()
    before['date_gap_days'] = pd.Series(
        [10, pd.NA, 3, pd.NA, 1], dtype='Int64')

    main = pd.DataFrame([{
        'Participant': 'AB001',
        'Timepoint': 'tp_1',
        'Form': 'form_a',
        'Flags': 'var_a : msg A',
        'Manually Resolved': 'yes',
    }])
    data = _make_workbook_bytes({'Main Report': main})

    result = calc.read_dropbox_data(
        before, _COMBINED_COLS, ['manually_resolved'],
        'unused/path.xlsx', None, 'PRONET', ['Main Report'],
        preloaded_data=data)

    assert str(result['date_gap_days'].dtype) == 'Int64'
    assert result.loc[result['subject'] == 'AB001',
                      'date_gap_days'].iloc[0] == 10
    assert pd.isna(result.loc[result['subject'] == 'AB002',
                             'date_gap_days'].iloc[0])
    assert result.loc[result['subject'] == 'AB001',
                      'manually_resolved'].iloc[0] == 'yes'


# ---------------------------------------------------- Patch #3

def test_missing_new_with_prior_history_raises(calc_factory):
    """
    Patch #3 contract: if new_output is missing but old_output or
    current_output has non-zero size, halt with FileNotFoundError
    rather than treating new as empty (which would mark every
    previously-open flag as resolved).
    """
    make, _tmp, combined = calc_factory
    calc = make()
    # Create an old_output with one open flag.
    old_path = combined / "old_output" / "combined_qc_flags.parquet"
    _write_parquet(old_path, [_make_row(currently_resolved=False)])
    # new_output deliberately absent.
    with pytest.raises(FileNotFoundError) as excinfo:
        calc.determine_resolved_rows()
    msg = str(excinfo.value)
    assert 'new_output missing' in msg
    assert 'destroying QC history' in msg


def test_missing_new_no_history_is_benign(calc_factory):
    """
    Patch #3 contract: if new_output AND old/current are all absent
    (true first run), treating new as empty is benign — no crash,
    no phantom flags.
    """
    make, _tmp, combined = calc_factory
    calc = make()
    # No files anywhere. Should run cleanly.
    calc.determine_resolved_rows()


# ---------------------------------------------------- Reopen path

def test_reopen_when_old_resolved_appears_in_new(calc_factory):
    """
    State machine: row exists in BOTH old and new, currently_resolved_
    old is True (was resolved last run). The flag has reappeared, so
    it must be marked currently_resolved=False and today's date
    appended to dates_detected.
    """
    make, _tmp, combined = calc_factory
    calc = make()
    today = str(datetime.today().date())

    old_row = _make_row(
        currently_resolved=True,
        dates_detected='2026-03-01',
        dates_resolved='2026-03-15',
        manually_resolved='yes',
        comments='historical reviewer context',
    )
    new_row = _make_row(
        currently_resolved=False,
        dates_detected='2026-04-01',
        dates_resolved='',
    )
    _write_parquet(
        combined / "old_output" / "combined_qc_flags.parquet",
        [old_row])
    _write_parquet(
        combined / "new_output" / "combined_qc_flags.parquet",
        [new_row])

    calc.determine_resolved_rows()

    current = combined / "current_output" / "combined_qc_flags.parquet"
    assert current.exists()
    df = pd.read_parquet(current)
    # astype(object) first: the parquet handoff preserves nullable
    # Int64 dtypes, and fillna('') on those raises in modern pandas.
    df = df.astype(object).fillna('')
    assert len(df) == 1
    row = df.iloc[0]
    assert row['currently_resolved'] is False or row['currently_resolved'] == False, \
        f"reopened row must have currently_resolved=False, got {row['currently_resolved']!r}"
    # dates_detected should now end with today's date (newly appended).
    last_detected = str(row['dates_detected']).split(' | ')[-1]
    assert last_detected == today, \
        f"expected dates_detected to end with {today}, got {row['dates_detected']!r}"
    assert row['manually_resolved'] == '', \
        "a genuinely reopened flag must require a fresh manual resolution"
    assert row['comments'] == 'historical reviewer context'


def test_cross_check_id_preserves_history_across_display_changes(calc_factory):
    """Stable Cross Check identity must outlive form/variable/label changes."""
    make, _tmp, combined = calc_factory
    calc = make()

    old_row = _make_row(
        reports='Cross Checks',
        check_id='CROSS-QC-005',
        displayed_form='old_form',
        displayed_variable='old_variable',
        error_message='[CROSS-QC-005] Cross-form review: old label.',
        dates_detected='2026-03-01',
        manually_resolved='yes',
        comments='network review retained',
        site_comments='site review retained',
    )
    new_row = _make_row(
        reports='Cross Checks',
        check_id='CROSS-QC-005',
        displayed_form='new_form',
        displayed_variable='new_variable',
        error_message='[CROSS-QC-005] Cross-form review: improved label.',
        dates_detected='2026-04-01',
        manually_resolved='',
        comments='',
        site_comments='',
    )
    _write_parquet(
        combined / 'old_output' / 'combined_qc_flags.parquet', [old_row])
    _write_parquet(
        combined / 'new_output' / 'combined_qc_flags.parquet', [new_row])

    calc.determine_resolved_rows()

    current = pd.read_parquet(
        combined / 'current_output' / 'combined_qc_flags.parquet')
    assert len(current) == 1
    result = current.iloc[0]
    assert result['displayed_form'] == 'new_form'
    assert result['displayed_variable'] == 'new_variable'
    assert result['error_message'].endswith('improved label.')
    assert result['currently_resolved'] in (False, 'False')
    assert result['dates_detected'] == '2026-03-01'
    assert result['manually_resolved'] == 'yes'
    assert result['comments'] == 'network review retained'
    assert result['site_comments'] == 'site review retained'


def test_cross_check_id_is_backfilled_from_legacy_message(calc_factory):
    """First release with check_id must match older history via message prefix."""
    make, _tmp, combined = calc_factory
    calc = make()

    old_row = _make_row(
        reports='Cross Checks',
        displayed_form='old_form',
        displayed_variable='old_variable',
        error_message='[CROSS-QC-006] Cross-form review: legacy wording.',
        comments='legacy comment',
    )
    new_row = _make_row(
        reports='Cross Checks',
        check_id='CROSS-QC-006',
        displayed_form='new_form',
        displayed_variable='new_variable',
        error_message='[CROSS-QC-006] Cross-form review: new wording.',
    )
    _write_parquet(
        combined / 'old_output' / 'combined_qc_flags.parquet', [old_row])
    _write_parquet(
        combined / 'new_output' / 'combined_qc_flags.parquet', [new_row])

    calc.determine_resolved_rows()

    current = pd.read_parquet(
        combined / 'current_output' / 'combined_qc_flags.parquet')
    assert len(current) == 1
    assert current.iloc[0]['check_id'] == 'CROSS-QC-006'
    assert current.iloc[0]['comments'] == 'legacy comment'


def test_cross_check_history_identity_is_network_scoped(calc_factory):
    """The same subject/check ID in two networks must remain two histories."""
    make, _tmp, combined = calc_factory
    calc = make()

    shared = dict(
        subject='SHARED001',
        displayed_timepoint='baseline',
        reports='Cross Checks',
        check_id='CROSS-QC-005',
        displayed_form='form_a',
        displayed_variable='variable_a',
        error_message='[CROSS-QC-005] Cross-form review: example.',
    )
    old_row = _make_row(
        **shared, network='PRONET', comments='PRONET history')
    new_row = _make_row(**shared, network='PRESCIENT', comments='')
    _write_parquet(
        combined / 'old_output' / 'combined_qc_flags.parquet', [old_row])
    _write_parquet(
        combined / 'new_output' / 'combined_qc_flags.parquet', [new_row])

    calc.determine_resolved_rows()

    current = pd.read_parquet(
        combined / 'current_output' / 'combined_qc_flags.parquet')
    assert len(current) == 2
    by_network = current.set_index('network')
    assert bool(by_network.loc['PRONET', 'currently_resolved']) is True
    assert by_network.loc['PRONET', 'comments'] == 'PRONET history'
    assert bool(by_network.loc['PRESCIENT', 'currently_resolved']) is False
    assert by_network.loc['PRESCIENT', 'comments'] == ''


# ---------------------------------------------------- Resolve path

def test_open_old_disappears_from_new_marks_resolved(calc_factory):
    """
    State machine: open row in old (currently_resolved=False) is
    absent from new. The flag has been fixed; mark currently_
    resolved=True and append today to dates_resolved.
    """
    make, _tmp, combined = calc_factory
    calc = make()
    today = str(datetime.today().date())

    old_row = _make_row(
        currently_resolved=False,
        dates_detected='2026-03-01',
        dates_resolved='',
    )
    _write_parquet(
        combined / "old_output" / "combined_qc_flags.parquet",
        [old_row])
    # new_output exists but is empty (no flags this run).
    _write_parquet(
        combined / "new_output" / "combined_qc_flags.parquet",
        [])

    calc.determine_resolved_rows()

    current = combined / "current_output" / "combined_qc_flags.parquet"
    assert current.exists()
    df = pd.read_parquet(current)
    # astype(object) first: the parquet handoff preserves nullable
    # Int64 dtypes, and fillna('') on those raises in modern pandas.
    df = df.astype(object).fillna('')
    assert len(df) == 1
    row = df.iloc[0]
    assert row['currently_resolved'] is True or row['currently_resolved'] == True, \
        f"resolved row must have currently_resolved=True, got {row['currently_resolved']!r}"
    last_resolved = str(row['dates_resolved']).split(' | ')[-1]
    assert last_resolved == today, \
        f"expected dates_resolved to end with {today}, got {row['dates_resolved']!r}"


# ------------------------------------- Newly-added column survives reconciliation
#
# When a column is added to combined_qc_flags (here: `cohort`), the first run
# after the change has it in new_output but NOT in old_output. The old/new
# outer merge leaves such a single-side column UN-suffixed (bare `cohort`),
# and append_all_cols must still copy it into current_output. Without the
# `elif col in present_cols` fallback the column is silently dropped and never
# reaches current_output (and, since current is promoted to old each run, it
# would stay dropped forever). These tests fail on the pre-fix code and pass
# after it.


def test_cohort_survives_old_without_new_with(calc_factory):
    """
    Transition run: old_output predates the cohort column (key removed),
    new_output carries it on the SAME merge keys. current_output must end up
    with cohort populated from the new side.
    """
    make, _tmp, combined = calc_factory
    calc = make()

    old_row = _make_row()
    old_row.pop('cohort')  # old_output written before cohort existed
    new_row = _make_row(cohort='HC')  # same merge keys as _DEFAULT_ROW

    _write_parquet(
        combined / "old_output" / "combined_qc_flags.parquet",
        [old_row])
    _write_parquet(
        combined / "new_output" / "combined_qc_flags.parquet",
        [new_row])

    calc.determine_resolved_rows()

    current = combined / "current_output" / "combined_qc_flags.parquet"
    assert current.exists()
    df = pd.read_parquet(current)
    # astype(object) first: the parquet handoff preserves nullable
    # Int64 dtypes, and fillna('') on those raises in modern pandas.
    df = df.astype(object).fillna('')
    assert len(df) == 1
    assert 'cohort' in df.columns, \
        "cohort column was dropped by reconciliation (append_all_cols fallback missing)"
    assert df.iloc[0]['cohort'] == 'HC', \
        f"expected cohort='HC' to survive into current_output, got {df.iloc[0]['cohort']!r}"


def test_cohort_survives_steady_state_both_sides(calc_factory):
    """
    Steady state: cohort present in BOTH old and new (the suffixed
    `cohort_new` branch — a different code path than the transition-run
    fallback). It must pass through to current_output unchanged.
    """
    make, _tmp, combined = calc_factory
    calc = make()

    old_row = _make_row(cohort='CHR')
    new_row = _make_row(cohort='CHR')

    _write_parquet(
        combined / "old_output" / "combined_qc_flags.parquet",
        [old_row])
    _write_parquet(
        combined / "new_output" / "combined_qc_flags.parquet",
        [new_row])

    calc.determine_resolved_rows()

    current = combined / "current_output" / "combined_qc_flags.parquet"
    df = pd.read_parquet(current)
    # astype(object) first: the parquet handoff preserves nullable
    # Int64 dtypes, and fillna('') on those raises in modern pandas.
    df = df.astype(object).fillna('')
    assert len(df) == 1
    assert df.iloc[0]['cohort'] == 'CHR', \
        f"expected cohort='CHR' to pass through, got {df.iloc[0]['cohort']!r}"


# ------------------------------------- Per-report manual-resolution authority
#
# These exercise read_dropbox_data directly (via preloaded_data, so no
# Dropbox is touched). They pin the rule that:
#   - individual/team report sheets may set `manually_resolved` only for
#     flags that do NOT also belong to an authoritative report
#     (Main Report / Non Team Forms);
#   - those two authoritative reports keep full control over their own flags
#     and can only be changed from their own sheets.

# Minimal raw->formatted column map (the subset read_dropbox_data needs).
_COMBINED_COLS = {
    'subject': 'Participant',
    'displayed_timepoint': 'Timepoint',
    'displayed_form': 'Form',
    'error_message': 'Flags',
    'manually_resolved': 'Manually Resolved',
}


def _make_workbook_bytes(sheets):
    """sheets: {sheet_name: DataFrame with FORMATTED column names}."""
    from io import BytesIO
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


def _current_flags():
    # One flag per report-membership scenario.
    return pd.DataFrame([
        # A: Main Report only.
        {'network': 'PRONET', 'subject': 'AB001', 'displayed_timepoint': 'tp_1',
         'displayed_form': 'form_a', 'error_message': 'var_a : msg A',
         'reports': 'Main Report', 'manually_resolved': ''},
        # B: Cognition Report only (a single individual report).
        {'network': 'PRONET', 'subject': 'AB002', 'displayed_timepoint': 'tp_1',
         'displayed_form': 'form_b', 'error_message': 'var_b : msg B',
         'reports': 'Cognition Report', 'manually_resolved': ''},
        # C: Secondary + Cognition (two individual reports, no authority).
        {'network': 'PRONET', 'subject': 'AB003', 'displayed_timepoint': 'tp_1',
         'displayed_form': 'form_c', 'error_message': 'var_c : msg C',
         'reports': 'Secondary Report | Cognition Report',
         'manually_resolved': ''},
        # D: Main Report + Cognition (Main present -> protected flag).
        {'network': 'PRONET', 'subject': 'AB004', 'displayed_timepoint': 'tp_1',
         'displayed_form': 'form_d', 'error_message': 'var_d : msg D',
         'reports': 'Main Report | Cognition Report', 'manually_resolved': ''},
        # E: Non Team Forms only (authoritative report).
        {'network': 'PRONET', 'subject': 'AB005', 'displayed_timepoint': 'tp_1',
         'displayed_form': 'form_e', 'error_message': 'var_e : msg E',
         'reports': 'Non Team Forms', 'manually_resolved': ''},
    ])


def test_individual_report_resolution_respects_authority(calc_factory):
    """
    The guarded pass (protected_reports = Main Report / Non Team Forms)
    reading a team sheet may resolve only flags that are NOT in an
    authoritative report.
    """
    make, _tmp, _combined = calc_factory
    calc = make()

    # Cognition Report sheet marks B, C, D, E resolved.
    cog = pd.DataFrame([
        {'Participant': s, 'Timepoint': 'tp_1', 'Form': f,
         'Flags': msg, 'Manually Resolved': 'yes'}
        for s, f, msg in [
            ('AB002', 'form_b', 'var_b : msg B'),
            ('AB003', 'form_c', 'var_c : msg C'),
            ('AB004', 'form_d', 'var_d : msg D'),
            ('AB005', 'form_e', 'var_e : msg E'),
        ]
    ])
    data = _make_workbook_bytes({'Cognition Report': cog})

    result = calc.read_dropbox_data(
        _current_flags(), _COMBINED_COLS, ['manually_resolved'],
        'unused/path.xlsx', None, 'PRONET', [],
        excl_report=False,
        protected_reports=['Main Report', 'Non Team Forms'],
        preloaded_data=data,
    ).set_index('subject')

    # Individual-only flags: the team sheet's mark sticks.
    assert result.loc['AB002', 'manually_resolved'] == 'yes'
    assert result.loc['AB003', 'manually_resolved'] == 'yes'
    # Not present in the team sheet at all -> untouched.
    assert result.loc['AB001', 'manually_resolved'] == ''
    # Belongs to Main Report -> a team sheet cannot resolve it.
    assert result.loc['AB004', 'manually_resolved'] == ''
    # Belongs to Non Team Forms -> a team sheet cannot resolve it.
    assert result.loc['AB005', 'manually_resolved'] == ''


def test_authoritative_reports_resolve_their_own_flags(calc_factory):
    """
    Unguarded passes for Main Report and Non Team Forms resolve their own
    flags (including ones that also live in an individual report).
    """
    make, _tmp, _combined = calc_factory
    calc = make()

    main = pd.DataFrame([
        {'Participant': 'AB004', 'Timepoint': 'tp_1', 'Form': 'form_d',
         'Flags': 'var_d : msg D', 'Manually Resolved': 'yes'},
    ])
    ntf = pd.DataFrame([
        {'Participant': 'AB005', 'Timepoint': 'tp_1', 'Form': 'form_e',
         'Flags': 'var_e : msg E', 'Manually Resolved': 'yes'},
    ])
    data = _make_workbook_bytes({'Main Report': main, 'Non Team Forms': ntf})

    df = _current_flags()
    # Mirror the production order: Main Report (authoritative) then
    # Non Team Forms (authoritative), both unguarded.
    df = calc.read_dropbox_data(
        df, _COMBINED_COLS, ['manually_resolved'],
        'unused/path.xlsx', None, 'PRONET', ['Main Report'],
        preloaded_data=data)
    df = calc.read_dropbox_data(
        df, _COMBINED_COLS, ['manually_resolved'],
        'unused/path.xlsx', None, 'PRONET', ['Non Team Forms'],
        preloaded_data=data).set_index('subject')

    assert df.loc['AB004', 'manually_resolved'] == 'yes'
    assert df.loc['AB005', 'manually_resolved'] == 'yes'


def test_team_sheet_cannot_clear_authoritative_resolution(calc_factory):
    """
    End-to-end of the three-pass order for a flag in both Main Report and a
    team report: Main Report resolves it; the later guarded team pass (where
    the team left it blank) must not clear it.
    """
    make, _tmp, _combined = calc_factory
    calc = make()

    main = pd.DataFrame([
        {'Participant': 'AB004', 'Timepoint': 'tp_1', 'Form': 'form_d',
         'Flags': 'var_d : msg D', 'Manually Resolved': 'yes'},
    ])
    # Cognition sheet lists the same flag but leaves it blank.
    cog = pd.DataFrame([
        {'Participant': 'AB004', 'Timepoint': 'tp_1', 'Form': 'form_d',
         'Flags': 'var_d : msg D', 'Manually Resolved': ''},
    ])
    data = _make_workbook_bytes({'Main Report': main, 'Cognition Report': cog})

    df = _current_flags()
    df = calc.read_dropbox_data(
        df, _COMBINED_COLS, ['manually_resolved'],
        'unused/path.xlsx', None, 'PRONET', ['Main Report'],
        preloaded_data=data)
    df = calc.read_dropbox_data(
        df, _COMBINED_COLS, ['manually_resolved'],
        'unused/path.xlsx', None, 'PRONET', [],
        excl_report=False,
        protected_reports=['Main Report', 'Non Team Forms'],
        preloaded_data=data).set_index('subject')

    assert df.loc['AB004', 'manually_resolved'] == 'yes'


def test_all_sheets_pass_quarantines_malformed_sheets(calc_factory):
    """
    The all-sheets pass (excl_report=False) visits every tab, including a
    reviewer-added scratch tab with no pipeline schema. It must skip such a
    sheet rather than KeyError, and still apply the valid report sheet. A
    non-string value in the Flags column must not crash the split either.
    """
    make, _tmp, _combined = calc_factory
    calc = make()

    # Valid individual-report sheet that should still be applied.
    cog = pd.DataFrame([
        {'Participant': 'AB002', 'Timepoint': 'tp_1', 'Form': 'form_b',
         'Flags': 'var_b : msg B', 'Manually Resolved': 'yes'},
        # A bare number typed into Flags must not AttributeError on split.
        {'Participant': 'AB999', 'Timepoint': 'tp_1', 'Form': 'junk',
         'Flags': 12345, 'Manually Resolved': 'yes'},
    ])
    # Reviewer-added scratch tab with none of the merge keys.
    scratch = pd.DataFrame([{'Notes': 'remember to follow up', 'Owner': 'RA1'}])
    data = _make_workbook_bytes({'Cognition Report': cog, 'Scratch': scratch})

    result = calc.read_dropbox_data(
        _current_flags(), _COMBINED_COLS, ['manually_resolved'],
        'unused/path.xlsx', None, 'PRONET', [],
        excl_report=False,
        protected_reports=['Main Report', 'Non Team Forms'],
        preloaded_data=data,
    ).set_index('subject')

    # Did not crash; the valid sheet's mark still applied.
    assert result.loc['AB002', 'manually_resolved'] == 'yes'
    # The malformed Flags row simply found no merge match — no phantom rows.
    assert 'AB999' not in result.index


def test_duplicate_rows_in_sheet_do_not_multiply_current_output(calc_factory):
    """
    A reviewer copy-pasting a flag row inside a report sheet must not multiply
    that flag's row in the canonical current_output via the left-merge.
    """
    make, _tmp, _combined = calc_factory
    calc = make()

    # Same flag duplicated three times. The original is blank and the edited
    # copies agree, so the one unambiguous action must be retained.
    identity = {
        'Participant': 'AB002', 'Timepoint': 'tp_1', 'Form': 'form_b',
        'Flags': 'var_b : msg B'}
    cog = pd.DataFrame([
        {**identity, 'Manually Resolved': ''},
        {**identity, 'Manually Resolved': 'yes'},
        {**identity, 'Manually Resolved': 'yes'},
    ])
    data = _make_workbook_bytes({'Cognition Report': cog})

    before = _current_flags()
    result = calc.read_dropbox_data(
        before, _COMBINED_COLS, ['manually_resolved'],
        'unused/path.xlsx', None, 'PRONET', [],
        excl_report=False,
        protected_reports=['Main Report', 'Non Team Forms'],
        preloaded_data=data,
    )

    # Row count is unchanged (no phantom duplicates), and the mark applied once.
    assert len(result) == len(before)
    assert result.set_index('subject').loc['AB002', 'manually_resolved'] == 'yes'


def test_conflicting_duplicate_workbook_rows_fail_closed(
        calc_factory, capsys):
    """Contradictory duplicate edits must not be resolved by row ordering."""
    make, _tmp, _combined = calc_factory
    calc = make()
    identity = {
        'Participant': 'AB002', 'Timepoint': 'tp_1', 'Form': 'form_b',
        'Flags': 'var_b : msg B'}
    cog = pd.DataFrame([
        {**identity, 'Manually Resolved': 'yes'},
        {**identity, 'Manually Resolved': 'no'},
    ])

    result = calc.read_dropbox_data(
        _current_flags(), _COMBINED_COLS, ['manually_resolved'],
        'unused/path.xlsx', None, 'PRONET', ['Cognition Report'],
        preloaded_data=_make_workbook_bytes({'Cognition Report': cog}),
    ).set_index('subject')

    assert result.loc['AB002', 'manually_resolved'] == ''
    assert 'conflicting duplicate flag identity' in capsys.readouterr().out


def test_guard_fails_closed_when_reports_column_missing(calc_factory):
    """
    If protection is requested but current_output has no `reports` column, the
    guard must block ALL overrides (fail closed) rather than apply them
    unconditionally — otherwise a team sheet could change a Main/NTF flag.
    """
    make, _tmp, _combined = calc_factory
    calc = make()

    # current_output WITHOUT a `reports` column.
    no_reports = pd.DataFrame([
        {'network': 'PRONET', 'subject': 'AB002', 'displayed_timepoint': 'tp_1',
         'displayed_form': 'form_b', 'error_message': 'var_b : msg B',
         'manually_resolved': ''},
    ])
    cog = pd.DataFrame([
        {'Participant': 'AB002', 'Timepoint': 'tp_1', 'Form': 'form_b',
         'Flags': 'var_b : msg B', 'Manually Resolved': 'yes'},
    ])
    data = _make_workbook_bytes({'Cognition Report': cog})

    result = calc.read_dropbox_data(
        no_reports, _COMBINED_COLS, ['manually_resolved'],
        'unused/path.xlsx', None, 'PRONET', [],
        excl_report=False,
        protected_reports=['Main Report', 'Non Team Forms'],
        preloaded_data=data,
    ).set_index('subject')

    # Fail-closed: no override applied because membership can't be verified.
    assert result.loc['AB002', 'manually_resolved'] == ''


# ------------------------------------- Corrupt / unreadable tracker workbooks
#
# A single corrupt Dropbox tracker (torn/truncated zip, 0-byte, non-zip, or a
# save interrupted mid-write) must NOT abort the whole reports stage. The
# observed production crash was `zipfile.BadZipFile: File is not a zip file`
# at `pd.ExcelFile(BytesIO(data))`. read_dropbox_data should skip that file,
# record it, and return current_output unchanged.


def _truncated_xlsx_bytes():
    """A real torn zip: valid xlsx with its tail (central directory) cut off.
    Starts with the PK magic, so pandas calls zipfile.ZipFile and raises
    BadZipFile — the exact production traceback."""
    valid = _make_workbook_bytes({'Main Report': pd.DataFrame([
        {'Participant': 'AB001', 'Timepoint': 'tp_1', 'Form': 'form_a',
         'Flags': 'var_a : msg A', 'Manually Resolved': 'yes'},
    ])})
    return valid[:len(valid) // 2]


def test_malformed_worksheet_is_quarantined_not_fatal(
        calc_factory, monkeypatch, capsys):
    """Worksheet XML can fail lazily after ExcelFile metadata opens."""
    make, _tmp, _combined = calc_factory
    calc = make()
    before = _current_flags()
    data = _make_workbook_bytes({'Main Report': pd.DataFrame([{
        'Participant': 'AB001', 'Timepoint': 'tp_1', 'Form': 'form_a',
        'Flags': 'var_a : msg A', 'Manually Resolved': 'yes'}])})
    original_excel_file = calc_mod.pd.ExcelFile

    # Preserve the metadata-open call but simulate the separate lazy sheet-read
    # failure seen with a damaged xl/worksheets/sheet*.xml part.
    monkeypatch.setattr(calc_mod.pd, 'ExcelFile', original_excel_file)
    monkeypatch.setattr(
        calc_mod.pd, 'read_excel',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError('malformed worksheet XML')))
    path = 'PRONET/combined/PRONET_Output_V2.xlsx'

    result = calc.read_dropbox_data(
        before, _COMBINED_COLS, ['manually_resolved'],
        path, None, 'PRONET', ['Main Report'], preloaded_data=data)

    pd.testing.assert_frame_equal(result, before)
    assert path in calc._corrupt_workbooks
    assert 'CORRUPT WORKSHEET' in capsys.readouterr().out


@pytest.mark.parametrize('bad_bytes,label', [
    (_truncated_xlsx_bytes(), 'torn-zip (BadZipFile)'),
    (b'', 'zero-byte (ValueError)'),
    (b'PK\x03\x04' + b'\x00' * 32, 'zip-magic-then-garbage (BadZipFile)'),
    (b'this is plainly not a workbook\n', 'non-zip text (ValueError)'),
])
def test_corrupt_workbook_is_skipped_not_fatal(calc_factory, bad_bytes, label):
    make, _tmp, _combined = calc_factory
    calc = make()

    before = _current_flags()
    path = 'PRONET/combined/PRONET_Output_V2.xlsx'

    # Must not raise (production crash was an unhandled BadZipFile here).
    result = calc.read_dropbox_data(
        before, _COMBINED_COLS, ['manually_resolved'],
        path, None, 'PRONET', ['Main Report'],
        preloaded_data=bad_bytes,
    )

    # current_output returned untouched: same rows, nothing resolved.
    assert len(result) == len(before), label
    assert list(result['manually_resolved']) == [''] * len(before), label
    # The bad file is recorded so the end-of-run summary can name it.
    assert path in calc._corrupt_workbooks, label


# ------------------------------------------- Auto-resolution volume

@pytest.mark.parametrize('resolved_count, still_open_count', [
    (10, 0),
    (500, 500),
    (600, 0),
])
def test_auto_resolution_writes_state_without_volume_limit(
        calc_factory, monkeypatch, resolved_count, still_open_count):
    """Persist resolutions regardless of their count or share of open flags."""
    make, _tmp, combined = calc_factory
    monkeypatch.delenv('QC_ALLOW_MASS_RESOLUTION', raising=False)
    calc = make()

    old_rows = [
        _make_row(subject=f'S{i:04d}', currently_resolved=False)
        for i in range(resolved_count + still_open_count)
    ]
    _write_parquet(
        combined / "old_output" / "combined_qc_flags.parquet", old_rows)
    _write_parquet(
        combined / "new_output" / "combined_qc_flags.parquet",
        old_rows[resolved_count:])

    calc.determine_resolved_rows()

    current = combined / "current_output" / "combined_qc_flags.parquet"
    assert current.exists()
    df = pd.read_parquet(current).astype(object).fillna('')
    assert len(df) == resolved_count + still_open_count
    truthy = (True, 'True', 'true', 'TRUE', 1, '1')
    resolved = df['currently_resolved'].isin(truthy)
    expected_subjects = {f'S{i:04d}' for i in range(resolved_count)}
    assert set(df.loc[resolved, 'subject']) == expected_subjects
    assert (df.loc[resolved, 'dates_resolved']
            == str(datetime.today().date())).all()
    assert len(df.loc[~resolved]) == still_open_count
    assert (df.loc[~resolved, 'dates_resolved'] == '').all()
    assert (df['dates_detected'] == '2026-04-01').all()


# ------------------------------------------- Conversion Report authority
#
# Conversion flags are routed to 'Conversion Report' and nothing else
# (qc_types/clinical_checks/conversion_checks.py emits
# {"reports": ["Conversion Report"]} at every emit site), so that sheet is
# the only surface where a reviewer can resolve them. Two rules are pinned
# here: the sheet owns its flags' manual-resolution state, and that state
# survives a resolve/reopen round trip.


def test_conversion_report_sheet_owns_its_manual_resolution(calc_factory):
    """The dedicated Conversion Report pass sets manual resolution, and the
    later guarded all-sheets pass cannot change it from another tab."""
    make, _tmp, _combined = calc_factory
    calc = make()

    current = pd.DataFrame([
        {'network': 'PRESCIENT', 'subject': 'AB001',
         'displayed_timepoint': 'floating', 'displayed_form': 'conversion_form',
         'error_message': 'chrconv_method : marked converted, no criteria',
         'reports': calc_mod.CalculateResolvedErrors.CONVERSION_REPORT_NAME,
         'manually_resolved': ''},
    ])
    conversion = pd.DataFrame([
        {'Participant': 'AB001', 'Timepoint': 'floating',
         'Form': 'conversion_form',
         'Flags': 'chrconv_method : marked converted, no criteria',
         'Manually Resolved': 'yes'},
    ])
    # A team sheet lists the same flag but leaves it blank, and would be
    # visited by the generic all-sheets pass.
    cog = pd.DataFrame([
        {'Participant': 'AB001', 'Timepoint': 'floating',
         'Form': 'conversion_form',
         'Flags': 'chrconv_method : marked converted, no criteria',
         'Manually Resolved': ''},
    ])
    data = _make_workbook_bytes({
        'Conversion Report': conversion, 'Cognition Report': cog})

    df = calc.read_dropbox_data(
        current, _COMBINED_COLS, ['manually_resolved'],
        'unused/path.xlsx', None, 'PRESCIENT',
        [calc_mod.CalculateResolvedErrors.CONVERSION_REPORT_NAME],
        preloaded_data=data)
    df = calc.read_dropbox_data(
        df, _COMBINED_COLS, ['manually_resolved'],
        'unused/path.xlsx', None, 'PRESCIENT', [],
        excl_report=False,
        protected_reports=[
            'Main Report', 'Non Team Forms',
            calc_mod.CalculateResolvedErrors.CONVERSION_REPORT_NAME],
        preloaded_data=data).set_index('subject')

    assert df.loc['AB001', 'manually_resolved'] == 'yes'


def test_conversion_flag_keeps_manual_resolution_across_reopen(calc_factory):
    """A conversion flag that drops out for a run and returns keeps its mark.

    Conversion checks skip (rather than fire) when the upstream subject_info
    conversion-criteria booleans are absent, so this round trip happens with
    no change to the underlying data. A Main Report flag in the same state is
    the control: it still requires a fresh manual resolution.
    """
    make, _tmp, combined = calc_factory
    calc = make()

    conv_kwargs = dict(
        subject='S001',
        displayed_timepoint='floating',
        displayed_form='conversion_form',
        displayed_variable='chrconv_method',
        error_message='chrconv_method : marked converted, no criteria',
        reports=calc_mod.CalculateResolvedErrors.CONVERSION_REPORT_NAME,
    )
    main_kwargs = dict(
        subject='S002',
        error_message='var_a : something is wrong',
        reports='Main Report',
    )
    old_rows = [
        _make_row(currently_resolved=True, dates_detected='2026-03-01',
                  dates_resolved='2026-03-15', manually_resolved='yes',
                  **conv_kwargs),
        _make_row(currently_resolved=True, dates_detected='2026-03-01',
                  dates_resolved='2026-03-15', manually_resolved='yes',
                  **main_kwargs),
    ]
    new_rows = [
        _make_row(currently_resolved=False, dates_detected='2026-04-01',
                  **conv_kwargs),
        _make_row(currently_resolved=False, dates_detected='2026-04-01',
                  **main_kwargs),
    ]
    _write_parquet(
        combined / "old_output" / "combined_qc_flags.parquet", old_rows)
    _write_parquet(
        combined / "new_output" / "combined_qc_flags.parquet", new_rows)

    calc.determine_resolved_rows()

    df = pd.read_parquet(
        combined / "current_output" / "combined_qc_flags.parquet")
    df = df.astype(object).fillna('').set_index('subject')
    assert df.loc['S001', 'manually_resolved'] == 'yes', \
        "a reopened conversion flag must keep its manual resolution"
    assert df.loc['S002', 'manually_resolved'] == '', \
        "non-conversion reopen behaviour must be unchanged"


@pytest.mark.parametrize('new_reports', ['', float('nan')])
def test_conversion_reopen_falls_back_to_old_routing(calc_factory, new_reports):
    """The mark survives even when the new row carries no routing.

    FormCheck clears `reports` for withdrawn / excluded / non-recruited
    subjects, so the reopened row's own routing can be blank (or, on a
    degraded frame, null). The old side then decides whether this was a
    conversion flag.
    """
    make, _tmp, combined = calc_factory
    calc = make()

    keys = dict(
        subject='S001',
        displayed_timepoint='floating',
        displayed_form='conversion_form',
        displayed_variable='chrconv_method',
        error_message='chrconv_method : marked converted, no criteria',
    )
    old_row = _make_row(
        currently_resolved=True, dates_detected='2026-03-01',
        dates_resolved='2026-03-15', manually_resolved='yes',
        reports=calc_mod.CalculateResolvedErrors.CONVERSION_REPORT_NAME,
        **keys)
    new_row = _make_row(
        currently_resolved=False, dates_detected='2026-04-01',
        reports=new_reports, **keys)
    _write_parquet(
        combined / "old_output" / "combined_qc_flags.parquet", [old_row])
    _write_parquet(
        combined / "new_output" / "combined_qc_flags.parquet", [new_row])

    calc.determine_resolved_rows()

    df = pd.read_parquet(
        combined / "current_output" / "combined_qc_flags.parquet")
    df = df.astype(object).fillna('').set_index('subject')
    assert df.loc['S001', 'manually_resolved'] == 'yes'
