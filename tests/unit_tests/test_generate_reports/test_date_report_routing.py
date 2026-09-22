from io import BytesIO
from types import SimpleNamespace

import pandas as pd
import pytest
from openpyxl import load_workbook
from openpyxl.styles import Border, PatternFill, Side

from generate_reports.calculate_resolved_errors import CalculateResolvedErrors
from generate_reports.create_trackers import CreateTrackers
from qc_types.date_check_logic import DATE_REPORT_EXCLUDED_FORMS


def _column_mapping():
    return {
        'subject': 'Participant',
        'error_message': 'Flags',
        'date_resolved': 'Date Resolved',
        'manually_resolved': 'Manually Resolved',
    }


def _date_flag_row(row_id, reports='Date Report', **overrides):
    row = {
        'row_id': row_id,
        'subject': 'PI001',
        'reports': reports,
        'displayed_form': 'visit_form',
        'affected_forms': 'visit_form',
        'displayed_variable': 'visit_date',
        'affected_variables': 'visit_date',
        'error_message': (
            'visit_date : Visit date at month1 (2025-01-01) is 10 '
            'day(s) before visit_date (2025-01-11) at the earlier '
            'timepoint baseline.'),
    }
    row.update(overrides)
    return row


def _workbook_bytes(sheets):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return buffer.getvalue()








def test_excluded_forms_are_removed_from_main_date_report_rows():
    excluded = sorted(DATE_REPORT_EXCLUDED_FORMS)
    frame = pd.DataFrame({
        'row_id': ['allowed', 'displayed']
        + [f'affected_{index}' for index in range(len(excluded))]
        + ['serialized'],
        'displayed_form': [
            'current_health_status', excluded[0],
            *(['allowed_form'] * len(excluded)), 'allowed_form',
        ],
        'affected_forms': [
            'current_health_status | allowed_prior',
            'allowed_form',
            *[f'allowed_form | {form}' for form in excluded],
            repr(['allowed_form', excluded[-1]]),
        ],
    })

    result = CreateTrackers._exclude_date_report_form_rows(frame)

    assert result['row_id'].tolist() == ['allowed']


def test_cross_variable_history_is_removed_from_date_report_rows():
    frame = pd.DataFrame([
        {
            'row_id': 'same_current',
            'displayed_variable': 'visit_date',
            'affected_variables': 'visit_date',
            'error_message': (
                'visit_date : Visit date at month1 (2025-01-01) is 10 '
                'day(s) before visit_date (2025-01-11) at the earlier '
                'timepoint baseline.'),
            'currently_resolved': False,
        },
        {
            'row_id': 'cross_resolved',
            'displayed_variable': 'later_date',
            'affected_variables': 'later_date | earlier_date',
            'error_message': (
                'later_date : Visit date at month1 (2025-01-01) is 10 '
                'day(s) before earlier_date (2025-01-11) at the earlier '
                'timepoint baseline.'),
            'currently_resolved': True,
        },
        {
            'row_id': 'same_legacy_metadata',
            'displayed_variable': 'visit_date',
            'affected_variables': 'visit_date',
            'error_message': 'legacy message without a parseable prior field',
            'currently_resolved': True,
        },
        {
            'row_id': 'contradictory',
            'displayed_variable': 'visit_date',
            'affected_variables': 'visit_date | other_date',
            'error_message': (
                'visit_date : Visit date at month1 (2025-01-01) is 10 '
                'day(s) before visit_date (2025-01-11) at the earlier '
                'timepoint baseline.'),
            'currently_resolved': True,
        },
        {
            'row_id': 'unknown',
            'displayed_variable': '',
            'affected_variables': '',
            'error_message': 'legacy message without field metadata',
            'currently_resolved': True,
        },
    ])

    result = CreateTrackers._exclude_cross_variable_date_report_rows(frame)

    assert result['row_id'].tolist() == ['same_current']












def test_unapproved_nonmain_report_remains_out_of_pronet_site_workbook(
        tmp_path):
    trackers = object.__new__(CreateTrackers)
    trackers.all_sites = {'PRONET': ['KC']}
    trackers.site_reports = ['Main Report', trackers.DATE_REPORT_NAME]
    trackers.utils = SimpleNamespace(pipeline_networks=("PRONET", "PRESCIENT"), 
        site_full_name_translations={'KC': "King's College, UK (KC)"})
    trackers.dropbox_output_path = f'{tmp_path}/'
    trackers.loop_ras = lambda *args: None

    writes = []
    trackers.format_excl_sheet = lambda *args: writes.append(args)

    trackers.loop_sites(
        'PRONET', 'Blood Report',
        pd.DataFrame({'Participant': ['KC001']}))

    assert writes == []


def test_failed_upload_makes_report_stage_fail_instead_of_looking_successful(
        tmp_path):
    upload_root = tmp_path / 'formatted_outputs' / 'dropbox_files'
    combined = upload_root / 'PRONET' / 'combined'
    combined.mkdir(parents=True)
    (combined / 'PRONET_Output_V2.xlsx').write_bytes(b'workbook bytes')

    trackers = object.__new__(CreateTrackers)
    trackers.output_path = f'{tmp_path}/'
    trackers.utils = SimpleNamespace(pipeline_networks=("PRONET", "PRESCIENT"), 
        collect_dropbox_credentials=lambda: object())
    attempts = []

    def fail_upload(full_path, local_path, dbx=None):
        attempts.append((full_path, local_path, dbx))
        raise OSError('simulated upload failure')

    trackers.save_to_dropbox = fail_upload

    with pytest.raises(RuntimeError, match='partially updated'):
        trackers.upload_trackers()

    assert len(attempts) == 1
