"""Regression coverage for blank_stale_sheets.

A pipeline-owned tab that exists in a workbook on disk but was not
rewritten this run must be blanked (header-only), never shipped stale —
the Aug-2026 production incident shipped month-old Date/Scid/Blood rows
verbatim for weeks because dead check families never touched their old
sheets again.
"""

import os

import pandas as pd

from generate_reports.create_trackers import CreateTrackers


def _column_mapping():
    return {
        "subject": "Participant",
        "cohort": "Cohort",
        "displayed_timepoint": "Timepoint",
        "displayed_form": "Form",
        "flag_count": "Flag Count",
        "error_message": "Flags",
        "var_translations": "Translations",
        "time_since_last_detection": "Days Since Detected",
        "date_resolved": "Date Resolved",
        "manually_resolved": "Manually Resolved",
        "comments": "Network Comments",
        "site_comments": "Site Comments",
        "priority_item": "Priority Item",
    }


def _write_workbook(full_path, sheet_names):
    with pd.ExcelWriter(full_path, engine="openpyxl") as writer:
        for sheet_name in sheet_names:
            pd.DataFrame({"Participant": ["XX001"]}).to_excel(
                writer, sheet_name=sheet_name, index=False)


def _sweeper(tmp_path):
    trackers = object.__new__(CreateTrackers)
    trackers.formatted_column_names = {
        "PRESCIENT": {"combined": _column_mapping()}}
    blank_calls = []
    trackers.format_excl_sheet = (
        lambda frame, report, folder, filename:
        blank_calls.append((report, folder, filename, frame.copy())))
    return trackers, blank_calls


def test_blanks_only_unwritten_pipeline_tabs(tmp_path):
    trackers, blank_calls = _sweeper(tmp_path)
    folder = f"{tmp_path}/PRESCIENT/combined/"
    os.makedirs(folder)
    filename = "PRESCIENT_Output_V2.xlsx"
    full_path = folder + filename
    # On disk: one tab rewritten this run, one stale pipeline tab, and
    # one foreign tab the pipeline does not own.
    _write_workbook(
        full_path, ["Main Report", "Blood Report", "Operator Notes"])
    trackers._sheets_written_by_path = {full_path: {"Main Report"}}

    trackers.blank_stale_sheets()

    assert [(call[0], call[1], call[2]) for call in blank_calls] == [
        ("Blood Report", folder, filename)]
    stale_frame = blank_calls[0][3]
    assert stale_frame.empty
    assert stale_frame.columns.tolist() == list(_column_mapping().values())




def test_missing_workbook_is_ignored(tmp_path):
    trackers, blank_calls = _sweeper(tmp_path)
    folder = f"{tmp_path}/PRESCIENT/combined/"
    full_path = folder + "PRESCIENT_Output_V2.xlsx"
    trackers._sheets_written_by_path = {full_path: {"Main Report"}}

    trackers.blank_stale_sheets()

    assert blank_calls == []
