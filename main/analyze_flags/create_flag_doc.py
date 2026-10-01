"""Generate a Word document summarizing active Pharmaceutical QC flags."""

import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from parse_pharm_flags import normalize_flag, parse_flag_cell, read_input

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "Pharmaceutical_QC_Flag_Guide.docx"

NETWORK_FILES = [
    ("Pronet", HERE / "pronet_pharm_flags.xlsx"),
    ("Prescient", HERE / "prescient_pharm_flags.xlsx"),
]

FLAG_COLUMN_NAMES = ("Flags", "error_message")
FLAG_COUNT_COLUMN_NAMES = ("Flag Count", "flag_count")
SUBJECT_COLUMN_NAMES = ("Participant", "subjectid", "Subject", "subject", "ID")
TIMEPOINT_COLUMN_NAMES = ("Timepoint", "displayed_timepoint")
FORM_COLUMN_NAMES = ("Form", "displayed_form")
CURRENTLY_RESOLVED_COLUMN_NAMES = ("currently_resolved", "Currently Resolved")
DATE_RESOLVED_COLUMN_NAMES = ("Date Resolved", "date_resolved")
MANUALLY_RESOLVED_COLUMN_NAMES = ("Manually Resolved", "manually_resolved")

_FALSE_RESOLUTION_MARKERS = {
    "",
    "false",
    "0",
    "0.0",
    "no",
    "none",
    "nan",
    "<na>",
}

FLAGS = [
    {
        "id": "MED-QC-01",
        "title": "Modification Date Before Latest Assessment",
        "what": (
            "The form's latest modification date (the later of chrpharm_date_mod "
            "and chrpharm_date_mod_2) is earlier than the subject's most recent "
            "assessment date across all timepoints, suggesting the medication "
            "record has not been reviewed since that assessment. Current form "
            "only. PRESCIENT flags appear on the Secondary Report."
        ),
        "action": (
            "Update the modification date to reflect the most recent assessment."
        ),
        "example": (
            "Latest mod date is 2024-04-01, but the latest assessment date is "
            "2024-05-15."
        ),
        "variables": (
            "chrpharm_date_mod, chrpharm_date_mod_2, plus the latest assessment "
            "date for the subject across all timepoints."
        ),
    },
    {
        "id": "MED-QC-02",
        "title": "Medication Start Date After End Date",
        "what": (
            "A medication course's start (onset) date falls after its end "
            "(offset) date, which is impossible. Courses marked as ongoing "
            "(offset 1901-01-01) are not checked. Applies to both the current "
            "and past forms."
        ),
        "action": (
            "Correct the dates or swap them if they were entered in the wrong "
            "order."
        ),
        "example": (
            "Onset is 2024-06-15 but offset is 2024-04-01."
        ),
        "variables": (
            "chrpharm_medN_onset, chrpharm_medN_offset (and *_past variants)."
        ),
    },
    {
        "id": "MED-QC-03",
        "title": "Medication Date After Form Modification Date",
        "what": (
            "A medication onset or offset date is more recent than the form's "
            "latest modification date (the later of chrpharm_date_mod and "
            "chrpharm_date_mod_2). Medication dates cannot postdate the last "
            "time the form was edited. Ongoing offsets are not checked. Current "
            "form only. PRESCIENT flags appear on the Secondary Report."
        ),
        "action": (
            "Correct whichever date is wrong."
        ),
        "example": (
            "Medication onset is 2024-09-01, but the latest mod date is "
            "2024-07-15."
        ),
        "variables": (
            "chrpharm_medN_onset, chrpharm_medN_offset, chrpharm_date_mod, "
            "chrpharm_date_mod_2."
        ),
    },
    {
        "id": "MED-QC-04",
        "title": "First-Dose Date Does Not Match Onset Date",
        "what": (
            "The calendar date of a course's first-dose datetime "
            "(chrpharm_firstdose_medN) differs from that course's onset date. "
            "The first dose is given on the day the course starts, so the two "
            "dates should match. Checked for current medication courses 1-60."
        ),
        "action": (
            "Update whichever date is incorrect so both match."
        ),
        "example": (
            "Onset is 2024-03-10, but first-dose shows 2024-03-12."
        ),
        "variables": (
            "chrpharm_medN_onset, chrpharm_firstdose_medN."
        ),
    },
    {
        "id": "MED-QC-05",
        "title": "Possible Duplicate Medication Course",
        "what": (
            "Two courses record the same medication name, their date ranges "
            "genuinely overlap (more than touching at a shared boundary date), "
            "and both are marked as simultaneously used (use flag = 1) - a "
            "likely double entry. Both courses need real start and end dates, "
            "so ongoing courses are not compared. Applies to both forms. "
            "Courses named with a special code (777 or 888) are reported on "
            "the Secondary Report rather than the Main Report."
        ),
        "action": (
            "Remove the duplicate, or clarify the distinction if they are "
            "legitimately separate courses."
        ),
        "example": (
            "Courses 3 and 7 are both \"Sertraline\" with overlapping ranges "
            "and simultaneous use = 1."
        ),
        "variables": (
            "chrpharm_medN_name, chrpharm_medN_onset, chrpharm_medN_offset, "
            "chrpharm_medN_use (and *_past variants)."
        ),
    },
    {
        "id": "MED-QC-06",
        "title": "No-Medication Period Overlaps With Another Course",
        "what": (
            "A no-medication period (name 999) genuinely overlaps another "
            "medication course. The two may share a single boundary date (one "
            "ends the day the other starts) but must not overlap beyond that. "
            "Ongoing offsets are not checked. Applies to both forms."
        ),
        "action": (
            "Adjust the date ranges so they do not overlap, or remove the "
            "no-medication entry."
        ),
        "example": (
            "No-medication runs 2024-01-01 to 2024-03-31, but another course "
            "runs 2024-02-15 to 2024-06-01."
        ),
        "variables": (
            "chrpharm_medN_name, chrpharm_medN_onset, chrpharm_medN_offset "
            "(and *_past variants)."
        ),
    },
    {
        "id": "MED-QC-07",
        "title": "No-Medication Period Missing End Date",
        "what": (
            "A no-medication period (name 999) has no end date - its offset is "
            "missing or coded as ongoing - even though another medication "
            "course starts after it. Once a medication begins, the "
            "no-medication period must have ended. Applies to both forms."
        ),
        "action": (
            "Add an end date to the no-medication period, or review the later "
            "course."
        ),
        "example": (
            "No-medication starts 2024-01-01 with no end date, but another "
            "medication starts 2024-04-01."
        ),
        "variables": (
            "chrpharm_medN_name, chrpharm_medN_onset, chrpharm_medN_offset "
            "(and *_past variants)."
        ),
    },
    {
        "id": "MED-QC-08",
        "title": "Ongoing Medication in Past Treatment Form",
        "what": (
            "A medication in the past treatment form has its offset coded as "
            "ongoing (1901-01-01). Past-form courses are finished by "
            "definition, so an ongoing course belongs in the current form "
            "instead. Past form only."
        ),
        "action": (
            "Move the medication to the current form, or add a proper end date."
        ),
        "example": (
            "A past-form medication's offset is 1901-01-01. It should be in "
            "the current form instead."
        ),
        "variables": (
            "chrpharm_medN_offset_past."
        ),
    },
    {
        "id": "MED-QC-09",
        "title": "Ongoing Flag Conflicts With Offset Date",
        "what": (
            "A course's ongoing/intermittent flag is set to 2 (ongoing), yet "
            "the same course's offset holds a real end date rather than the "
            "ongoing code (1901-01-01). One of the two must be wrong. Current "
            "form only. In the current implementation this flag is routed to "
            "the Secondary Report; older builds report it on the Main Report."
        ),
        "action": (
            "If the course is ongoing, change the offset to 1901-01-01. If it "
            "has ended, update the ongoing flag."
        ),
        "example": (
            "The ongoing flag for course 3 is set to 2, but "
            "chrpharm_med3_offset = 2024-08-01."
        ),
        "variables": (
            "chrpharm_interm_meds (course 1) or chrpharm_interm_meds_{N-1} "
            "(course N for N > 1), chrpharm_medN_offset."
        ),
    },
    {
        "id": "MED-QC-10",
        "title": "Current Medication Ends Before Past Form Date",
        "what": (
            "A current-form medication course ended before the past treatment "
            "form's assessment date (chrpharm_interview_date). A course that "
            "was already over by that assessment belongs in the past form, not "
            "the current one. Ongoing offsets are not checked. Current form "
            "only."
        ),
        "action": (
            "Move the course to the past form, or correct the dates."
        ),
        "example": (
            "chrpharm_interview_date is 2024-03-01, but a current medication's "
            "offset is 2024-02-15."
        ),
        "variables": (
            "chrpharm_medN_offset, chrpharm_interview_date."
        ),
    },
    {
        "id": "MED-QC-11",
        "title": "No Information Code (888) With Other Data Present",
        "what": (
            "A medication name is coded 888 (no information), yet other fields "
            "for the same course (use, dosage, dosage_2, frequency, "
            "compliance, or indication) contain data. If some information is "
            "known, 777 may be the better code; otherwise the extra fields "
            "should be cleared. Applies to both forms."
        ),
        "action": (
            "Change 888 to 777 if some information is known, or clear the "
            "other fields."
        ),
        "example": (
            "Name is 888, but dosage shows 50mg and frequency shows 1."
        ),
        "variables": (
            "chrpharm_medN_name, plus chrpharm_medN_use, chrpharm_medN_dosage, "
            "chrpharm_medN_dosage_2, chrpharm_medN_frequency, chrpharm_medN_comp, "
            "chrpharm_medN_indication (and *_past variants)."
        ),
    },
    {
        "id": "MED-QC-12",
        "title": "No Medication Code (999) With Other Data Present",
        "what": (
            "A medication name is coded 999 (no medication), yet other fields "
            "for the same course (indication, dosage, dosage_2, frequency, or "
            "compliance) contain data - contradicting the claim that no "
            "medication was taken. Applies to both forms. Fields holding "
            "zero-like values (0, 0.0) are reported as a separate flag on the "
            "Secondary Report, so a course with both real and zero values "
            "raises two flags."
        ),
        "action": (
            "Clear the extra fields, or update the medication name if a "
            "medication was actually taken."
        ),
        "example": (
            "Name is 999, but compliance shows 85%."
        ),
        "variables": (
            "chrpharm_medN_name, plus chrpharm_medN_indication, "
            "chrpharm_medN_dosage, chrpharm_medN_dosage_2, chrpharm_medN_frequency, "
            "chrpharm_medN_comp (and *_past variants)."
        ),
    },
    {
        "id": "MED-QC-13",
        "title": "Blinded Medication Code Still Present",
        "what": (
            "A medication name still uses a blinded-study code (573, 542, 538, "
            "or 539). Once the study is unblinded, the actual medication name "
            "should be recorded; no action is needed while blinding is still "
            "in place. Applies to both forms."
        ),
        "action": (
            "Update the name if the medication is now known. No action needed "
            "if the study is still blinded."
        ),
        "example": (
            "Name is 573 (blinded). If unblinded, replace with the actual "
            "medication name."
        ),
        "variables": (
            "chrpharm_medN_name (and *_past variants)."
        ),
    },
    {
        "id": "MED-QC-14",
        "title": "Compliance Value Out of Range",
        "what": (
            "A compliance value falls outside the valid range of 0 to 100. "
            "Study missing codes (-3, -9, -99, 999) are treated as missing "
            "data and never flagged here. Applies to both forms."
        ),
        "action": "Correct the value to a number between 0 and 100.",
        "example": "Compliance is 150.",
        "variables": "chrpharm_medN_comp (and *_past variants).",
    },
    {
        "id": "MED-QC-15",
        "title": "Invalid Negative Frequency Code",
        "what": (
            "A frequency value is negative and is not one of the recognized "
            "codes. -3 is the accepted missingness code, and the other study "
            "missing codes (-9, -99) are treated as missing data, so this flag "
            "fires only on other negative values such as -1 or -5. Applies to "
            "both forms."
        ),
        "action": (
            "Code as -3 if missing, or enter the correct positive value."
        ),
        "example": "Frequency is -1.",
        "variables": "chrpharm_medN_frequency (and *_past variants).",
    },
    {
        "id": "MED-QC-16",
        "title": "Acute Injection Course Longer Than Two Weeks",
        "what": (
            "A medication coded as an acute injection (codes 724, 725, 257, "
            "727, 123, 151, 11, 726, 271, 76, 245, or 13) has a course lasting "
            "longer than 14 days. Acute injections are short-term by nature, "
            "so a longer course is suspect. Ongoing offsets are not checked. "
            "Applies to both forms."
        ),
        "action": (
            "Verify the dates and confirm whether this is truly an acute "
            "injection."
        ),
        "example": (
            "Medication name is coded 724 (acute injection), and the course "
            "spans 60 days (2024-01-01 to 2024-03-01)."
        ),
        "variables": (
            "chrpharm_medN_name, chrpharm_medN_onset, chrpharm_medN_offset "
            "(and *_past variants)."
        ),
    },
    {
        "id": "MED-QC-17",
        "title": "Combination Medication Dose Missing '+' Separator",
        "what": (
            "A combination medication (name code 110, 146, or 280, e.g. "
            "Amitriptyline + Perphenazine) has a dosage that does not record "
            "both component doses as two numbers joined by '+'. Entering only "
            "one number leaves the other component's dose uncaptured. Applies "
            "to both forms."
        ),
        "action": (
            "Record both component doses separated by a '+' (e.g. '25+4')."
        ),
        "example": (
            "Name is 146 (Amitriptyline + Perphenazine), but dosage shows '25' "
            "instead of '25+4'."
        ),
        "variables": (
            "chrpharm_medN_name, chrpharm_medN_dosage (and *_past variants)."
        ),
    },
    {
        "id": "MED-QC-18",
        "title": "Medication Dose Exceeds Clinical-Use Cutoff",
        "what": (
            "A medication's recorded dose exceeds the accepted clinical-use "
            "cutoff for that medication (e.g. Risperidone > 8 mg, Olanzapine "
            "> 20 mg, Clozapine > 900 mg). Codes 452 and 547 are long-acting "
            "injectables checked against per-injection cutoffs. Combination "
            "dose strings (e.g. '10+5') are skipped. Applies to both forms."
        ),
        "action": (
            "Verify the dose is recorded correctly."
        ),
        "example": (
            "chrpharm_med1_dosage is 50 mg for Risperidone (code 300), which "
            "exceeds the 8 mg cutoff."
        ),
        "variables": (
            "chrpharm_medN_name, chrpharm_medN_dosage (and *_past variants)."
        ),
    },
    {
        "id": "MED-QC-19",
        "title": "Lifetime Antipsychotic Exposure Inconsistent With Past Treatment",
        "what": (
            "The lifetime antipsychotic exposure form (chrap_*) and the past "
            "pharmaceutical treatment form disagree: either lifetime use of an "
            "antipsychotic is reported (chrap_X = 1) but no matching course "
            "appears in the past form, or a matching course is recorded but "
            "chrap_X is coded as something other than 1 (e.g. 0). Blank chrap "
            "values are skipped. Requires the lifetime exposure form to be "
            "marked complete, the past form to be complete or a later "
            "timepoint to exist, and chrap_missing not to be 1. Screening only."
        ),
        "action": (
            "Reconcile the lifetime antipsychotic exposure form with the past "
            "pharmaceutical treatment form."
        ),
        "example": (
            "chrap_olz = 1 (lifetime olanzapine use), but no course with the "
            "matching pharm code appears in past_pharmaceutical_treatment."
        ),
        "variables": (
            "chrap_* (lifetime AP exposure variables), chrpharm_medN_name_past, "
            "chrap_missing, plus the form completion variables."
        ),
    },
    {
        "id": "SCREENING-CHRONOLOGY",
        "title": "Past Treatment Date More Than 20 Days After Current Treatment Date",
        "what": (
            "The past pharmaceutical treatment interview date is more than 20 "
            "days after the current pharmaceutical treatment date. The two "
            "interviews should take place close together, so a larger gap "
            "suggests a data entry error in one of the dates. Screening only."
        ),
        "action": (
            "Correct the dates so the past date is not more than 20 days after "
            "the current date."
        ),
        "example": (
            "Past date is 2024-06-01, current date is 2024-03-15 (77 days "
            "later)."
        ),
        "variables": (
            "chrpharm_interview_date, chrpharm_date_first."
        ),
    },
]


def _normalized_column_name(value):
    """Normalize tracker/raw column names for tolerant matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def _find_column(df, candidates):
    by_normalized_name = {
        _normalized_column_name(column): column for column in df.columns
    }
    for candidate in candidates:
        match = by_normalized_name.get(_normalized_column_name(candidate))
        if match is not None:
            return match
    return None


def _text(value):
    if value is None:
        return ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return ""
    return str(value).strip()


def _clean_flag_text(value):
    return re.sub(r"\s+", " ", _text(value)).strip()


def _currently_resolved(value):
    """Match the pipeline's truthy values for automatic resolution."""
    return _text(value).casefold() in {"true", "1", "1.0"}


def _has_resolution_marker(value):
    """Return whether a formatted tracker cell positively marks resolution."""
    return _text(value).casefold() not in _FALSE_RESOLUTION_MARKERS


def _row_is_resolved(row, current_column, date_column, manual_column):
    # The canonical raw boolean is authoritative.  Raw resolution-date history
    # can remain populated after a flag reopens, so only use the formatted
    # tracker's Date Resolved field when that boolean is unavailable.
    if current_column is not None:
        resolved = _currently_resolved(row[current_column])
    elif date_column is not None:
        resolved = _has_resolution_marker(row[date_column])
    else:
        resolved = False

    if manual_column is not None:
        resolved = resolved or _has_resolution_marker(row[manual_column])
    return resolved


def _expected_flag_count(value):
    """Return an integer tracker Flag Count, or None for a blank cell."""
    text = _text(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        raise ValueError(f"not a number: {text!r}") from None
    if number < 0 or not number.is_integer():
        raise ValueError(f"not a non-negative whole number: {text!r}")
    return int(number)


def _split_output_flags(value, is_tracker_row):
    """Split a tracker row without treating message punctuation as separators."""
    if not is_tracker_row:
        return parse_flag_cell(value)

    text = _text(value)
    if not text:
        return []
    return [part.strip() for part in text.split(" | ") if part.strip()]


def _as_sheet_mapping(output):
    if isinstance(output, Mapping):
        return dict(output)
    if isinstance(output, pd.DataFrame):
        return {"Input": output}
    raise TypeError(
        "Expected a DataFrame or a {sheet_name: DataFrame} mapping, "
        f"got {type(output).__name__}."
    )


def count_output_flags(output, source_name="output"):
    """Count active individual flag instances in one network output.

    Full tracker workbooks repeat the same logical QC flag on each report tab
    to which it is routed.  This function therefore explodes merged ``Flags``
    cells first, then de-duplicates exact flag identities across tabs.  The
    identity retains participant, timepoint, form, and the full raw message so
    separate medication courses and onset/offset violations belonging to the
    same QC rule remain separate instances.
    """
    sheets = _as_sheet_mapping(output)
    usable_sheets = []

    for sheet_name, df in sheets.items():
        if not isinstance(df, pd.DataFrame):
            continue
        flag_column = _find_column(df, FLAG_COLUMN_NAMES)
        if flag_column is not None:
            usable_sheets.append((str(sheet_name), df, flag_column))

    if not usable_sheets:
        available = ", ".join(map(str, sheets)) or "<none>"
        raise ValueError(
            f"No worksheet in {source_name} has a Flags column. "
            f"Available worksheets: {available}."
        )

    multiple_sheets = len(usable_sheets) > 1
    exploded_rows = []
    flag_count_errors = []
    sheets_without_resolution_status = []

    for sheet_name, df, flag_column in usable_sheets:
        subject_column = _find_column(df, SUBJECT_COLUMN_NAMES)
        timepoint_column = _find_column(df, TIMEPOINT_COLUMN_NAMES)
        form_column = _find_column(df, FORM_COLUMN_NAMES)
        flag_count_column = _find_column(df, FLAG_COUNT_COLUMN_NAMES)
        current_column = _find_column(df, CURRENTLY_RESOLVED_COLUMN_NAMES)
        date_column = _find_column(df, DATE_RESOLVED_COLUMN_NAMES)
        manual_column = _find_column(df, MANUALLY_RESOLVED_COLUMN_NAMES)

        if not (current_column or date_column or manual_column):
            sheets_without_resolution_status.append(sheet_name)

        # Cross-tab de-duplication is unsafe without the workbook identity used
        # by the report generator.  Standard tracker tabs always contain these
        # fields, so fail loudly rather than silently merging different people.
        missing_identity = [
            label
            for label, column in (
                ("participant", subject_column),
                ("timepoint", timepoint_column),
                ("form", form_column),
            )
            if column is None
        ]
        if multiple_sheets and missing_identity and not df.empty:
            raise ValueError(
                f"Cannot safely de-duplicate report tabs in {source_name}: "
                f"worksheet '{sheet_name}' is missing "
                f"{', '.join(missing_identity)} column(s)."
            )

        for source_position, (_, row) in enumerate(df.iterrows(), start=2):
            is_tracker_row = flag_count_column is not None
            raw_flags = _split_output_flags(row[flag_column], is_tracker_row)

            if flag_count_column is not None:
                try:
                    expected = _expected_flag_count(row[flag_count_column])
                except ValueError as exc:
                    flag_count_errors.append(
                        f"{sheet_name} row {source_position}: Flag Count is {exc}"
                    )
                else:
                    if expected is None and raw_flags:
                        flag_count_errors.append(
                            f"{sheet_name} row {source_position}: Flag Count is "
                            f"blank, parsed={len(raw_flags)}"
                        )
                    elif expected is not None and expected != len(raw_flags):
                        flag_count_errors.append(
                            f"{sheet_name} row {source_position}: "
                            f"Flag Count={expected}, parsed={len(raw_flags)}"
                        )

            resolved = _row_is_resolved(
                row, current_column, date_column, manual_column
            )

            if subject_column is None:
                # Single-table non-tracker inputs retain their source row so
                # identical messages belonging to different records are not
                # collapsed.
                subject = f"__{sheet_name}_ROW_{source_position}"
            else:
                subject = _text(row[subject_column])

            timepoint = (
                _text(row[timepoint_column]) if timepoint_column is not None else ""
            )
            form = _text(row[form_column]) if form_column is not None else ""

            if multiple_sheets and raw_flags:
                blank_identity = [
                    label
                    for label, value in (
                        ("participant", subject),
                        ("timepoint", timepoint),
                        ("form", form),
                    )
                    if not value
                ]
                if blank_identity:
                    raise ValueError(
                        f"Cannot safely de-duplicate report tabs in {source_name}: "
                        f"worksheet '{sheet_name}' row {source_position} has blank "
                        f"{', '.join(blank_identity)} value(s)."
                    )

            for flag_position, raw_flag in enumerate(raw_flags, start=1):
                cleaned_flag = _clean_flag_text(raw_flag)
                if not cleaned_flag:
                    continue
                qc_id, normalized, short_label = normalize_flag(cleaned_flag)
                exploded_rows.append(
                    {
                        "source_sheet": sheet_name,
                        "source_position": source_position,
                        "flag_position": flag_position,
                        "subject": subject,
                        "timepoint": timepoint,
                        "form": form,
                        "raw_flag": cleaned_flag,
                        "qc_id": qc_id,
                        "normalized_flag": normalized,
                        "short_label": short_label,
                        "resolved": resolved,
                    }
                )

    if flag_count_errors:
        examples = "; ".join(flag_count_errors[:3])
        raise ValueError(
            f"Cannot produce reliable counts for {source_name}: "
            f"{len(flag_count_errors)} Flag Count error(s). {examples}"
        )

    if sheets_without_resolution_status:
        print(
            f"Warning: {source_name} worksheet(s) "
            f"{', '.join(sheets_without_resolution_status)} have no "
            "resolution-status columns; their parsed flags will be treated "
            "as active."
        )

    if not exploded_rows:
        return {}

    exploded = pd.DataFrame(exploded_rows)
    if multiple_sheets:
        base_identity = ["subject", "timepoint", "form", "raw_flag"]
        # Preserve repeated identical instances within a tab while pairing the
        # corresponding first/second/etc. copies on other routed report tabs.
        # Without this ordinal, set-like cross-tab de-duplication would collapse
        # two real identical instances into one.
        exploded["occurrence_rank"] = exploded.groupby(
            ["source_sheet", *base_identity],
            dropna=False,
            sort=False,
        ).cumcount()
        identity_columns = [*base_identity, "occurrence_rank"]
    else:
        # No report-tab replication is possible in a single table.  Preserve
        # every source occurrence, even when an extract omits timepoint/form
        # identity or repeats the same message on separate rows.
        identity_columns = [
            "source_sheet",
            "source_position",
            "flag_position",
        ]

    # A logical flag may be copied to Main, Secondary, and/or Non Team tabs.
    # A positive resolution marker on any copy resolves that logical flag.
    logical_flags = (
        exploded.groupby(identity_columns, dropna=False, sort=False)
        .agg(
            qc_id=("qc_id", "first"),
            resolved=("resolved", "any"),
        )
        .reset_index()
    )
    active_flags = logical_flags.loc[~logical_flags["resolved"]]

    unknown_count = int(
        active_flags["qc_id"].isin([None, "RAW-NUMBERS-REMOVED"]).sum()
    )
    if unknown_count:
        print(
            f"Warning: {source_name} contains {unknown_count} active flag(s) "
            "that do not match a known normalization rule and are not included "
            "in the guide counts."
        )

    known_ids = {flag["id"] for flag in FLAGS}
    known = active_flags.loc[active_flags["qc_id"].isin(known_ids)]
    if known.empty:
        return {}
    return known.groupby("qc_id").size().astype(int).to_dict()


def get_network_counts():
    """Return active flag-instance counts from each network output."""
    counts = {}
    for network, path in NETWORK_FILES:
        if not path.exists():
            print(f"Skipping {network}: file not found at {path}")
            counts[network] = {}
            continue
        output = read_input(path, sheet_name=None)
        counts[network] = count_output_flags(output, source_name=path.name)
    return counts


def build_doc():
    network_counts = get_network_counts()
    network_names = [name for name, _ in NETWORK_FILES]

    doc = Document()

    # -- Title --
    title = doc.add_heading("Pharmaceutical Treatment QC Flags", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # -- Intro --
    doc.add_paragraph(
        "Reference for QC flags raised during pharmaceutical treatment data review. "
        f"Counts reflect currently unresolved individual flag instances from "
        f"{' and '.join(network_names)}, de-duplicated across report tabs. "
        "Descriptions document the current version of the checks; the flag "
        "exports the counts come from were produced by an earlier build, so "
        "report routing and the exact medication code lists may differ slightly."
    )

    # -- Single consolidated table --
    headers = ["Short Name", "What It Checks", "Variables", "Action"] + [
        f"{name} Active Flag Count" for name in network_names
    ]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr = table.rows[0].cells
    for i, text in enumerate(headers):
        hdr[i].text = text
        for p in hdr[i].paragraphs:
            for run in p.runs:
                run.bold = True

    for f in FLAGS:
        row = table.add_row().cells
        row[0].text = f["title"]
        row[1].text = f["what"]
        row[2].text = f.get("variables", "")
        row[3].text = f["action"]
        for j, name in enumerate(network_names):
            count = network_counts.get(name, {}).get(f["id"], 0)
            row[4 + j].text = str(count)

    doc.save(str(OUTPUT))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    build_doc()
