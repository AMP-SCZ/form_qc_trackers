import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_FILE = Path("pronet_pharm_flags.xlsx")
OUTPUT_FILE = Path("pronet_pharm_summary.xlsx")
CHART_DIR = Path("flag_charts")
FLAG_COLUMN = "Flags"
PREFERRED_SUBJECT_COLUMNS = ["Participant", "subjectid", "Subject", "subject", "ID"]


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


RULES = [
    (
        "MED-QC-01",
        re.compile(
            r"^Latest modification date \([^)]+ = \d{4}-\d{2}-\d{2}\) is before the latest assessment date \([^)]+ = \d{4}-\d{2}-\d{2}\)\. Please update the pharmaceutical treatment modification date to reflect the most recent assessment\.$"
        ),
        "Latest modification date is before the latest assessment date. Please update the pharmaceutical treatment modification date to reflect the most recent assessment.",
        "Mod date < assessment",
    ),
    (
        "MED-QC-02",
        re.compile(
            r"^[A-Za-z0-9_]+ \(\d{4}-\d{2}-\d{2}\) is later than [A-Za-z0-9_]+ \(\d{4}-\d{2}-\d{2}\)\. A medication course cannot end before it starts\.$"
        ),
        "<onset_var> is later than <offset_var>. A medication course cannot end before it starts.",
        "Onset after offset",
    ),
    (
        "MED-QC-03",
        re.compile(
            r"^[A-Za-z0-9_]+ \(\d{4}-\d{2}-\d{2}\) is later than [A-Za-z0-9_]+ \(\d{4}-\d{2}-\d{2}\)\. Medication onset/offset dates cannot occur after the form modification date\.$"
        ),
        "<med_date_var> is later than <mod_var>. Medication onset/offset dates cannot occur after the form modification date.",
        "Date after mod date",
    ),
    (
        "MED-QC-04",
        re.compile(
            r"^[A-Za-z0-9_]+ \(\d{4}-\d{2}-\d{2}\) does not match the date portion of [A-Za-z0-9_]+ \(\d{4}-\d{2}-\d{2}\)\. The first-dose datetime should have the same calendar date as the medication onset date\.$"
        ),
        "<onset_var> does not match the date portion of <firstdose_var>. The first-dose datetime should have the same calendar date as the medication onset date.",
        "1st dose mismatch",
    ),
    (
        "MED-QC-05",
        re.compile(
            # The medication name is interpolated raw and can itself contain
            # parentheses (e.g. "Olanzapine (Zyprexa)"), so the name group is
            # .+ anchored on the surrounding literal text, not [^)]+.
            r"^[A-Za-z0-9_]+ \(\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}\) and [A-Za-z0-9_]+ \(\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}\) have the same medication name \(.+\), overlapping date ranges, and both courses are marked as simultaneously used \([A-Za-z0-9_]+ = 1 and [A-Za-z0-9_]+ = 1\)\. Please check for a duplicate medication course\.$"
        ),
        "<name_var_1> and <name_var_2> have the same medication name, overlapping date ranges, and both courses are marked as simultaneously used. Please check for a duplicate medication course.",
        "Duplicate med",
    ),
    (
        "MED-QC-06",
        re.compile(
            r"^[A-Za-z0-9_]+ contains 999 \(no medication\) from \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}, but its date range strictly overlaps with another medication course \([A-Za-z0-9_]+: \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}\)\. A no-medication period can touch another course on the same start/end date, but it should not overlap beyond that boundary\.$"
        ),
        "<name_var> contains 999 (no medication), but its date range strictly overlaps with another medication course. A no-medication period can touch another course on the same start/end date, but it should not overlap beyond that boundary.",
        "No-med overlap",
    ),
    (
        "MED-QC-07",
        re.compile(
            r"^[A-Za-z0-9_]+ contains 999 \(no medication\) starting \d{4}-\d{2}-\d{2}, but [A-Za-z0-9_]+ is missing or coded as ongoing while another medication course \([A-Za-z0-9_]+\) starts later on \d{4}-\d{2}-\d{2}\. Please end the no-medication period or review the later medication course\.$"
        ),
        "<name_var> contains 999 (no medication), but <offset_var> is missing or coded as ongoing while another medication course starts later. Please end the no-medication period or review the later medication course.",
        "No-med no offset",
    ),
    (
        "MED-QC-08",
        re.compile(
            r"^[A-Za-z0-9_]+ is coded as ongoing \(1901-01-01\), but medication courses in the past pharmaceutical treatment form cannot be ongoing\. Please add this medication in the current pharmaceutical treatment form\.$"
        ),
        "<offset_var> is coded as ongoing (1901-01-01), but medication courses in the past pharmaceutical treatment form cannot be ongoing. Please add this medication in the current pharmaceutical treatment form.",
        "Ongoing in past",
    ),
    (
        "MED-QC-09",
        re.compile(
            r"^[A-Za-z0-9_]+ is coded as 2, but [A-Za-z0-9_]+ is also populated with \d{4}-\d{2}-\d{2} instead of the ongoing code\. Please confirm whether this course is ongoing and reconcile the ongoing/intermittent flag with the offset date\.$"
        ),
        "<ongoing_var> is coded as 2, but <offset_var> is also populated with a date instead of the ongoing code. Please confirm whether this course is ongoing and reconcile the ongoing/intermittent flag with the offset date.",
        "Ongoing conflict",
    ),
    (
        "MED-QC-10",
        re.compile(
            r"^[A-Za-z0-9_]+ \(\d{4}-\d{2}-\d{2}\) is before [A-Za-z0-9_]+ \(\d{4}-\d{2}-\d{2}\)\. A current-form medication course should not end before the past pharmaceutical treatment assessment date\.$"
        ),
        "<offset_var> is before <past_form_date_var>. A current-form medication course should not end before the past pharmaceutical treatment assessment date.",
        "Ends before past",
    ),
    (
        "MED-QC-11",
        re.compile(
            r"^[A-Za-z0-9_]+ is coded as 888 \(no information\), but other course-level fields contain data \([^)]+\)\. Please check whether 777 would be more appropriate\.$"
        ),
        "<name_var> is coded as 888 (no information), but other course-level fields contain data. Please check whether 777 would be more appropriate.",
        "888 with data",
    ),
    (
        "MED-QC-12",
        re.compile(
            r"^[A-Za-z0-9_]+ is coded as 999 \(no medication\), but other course-level fields contain data \([^)]+\)\. Please check whether this period was really no medication\.$"
        ),
        "<name_var> is coded as 999 (no medication), but other course-level fields contain data. Please check whether this period was really no medication.",
        "999 with data",
    ),
    (
        # MED-QC-12 zero-values variant (routes to Secondary Report): the
        # contradicting fields hold zero-like numeric values rather than real data.
        "MED-QC-12",
        re.compile(
            r"^[A-Za-z0-9_]+ is coded as 999 \(no medication\), but other course-level fields contain zero values \([^)]+\)\. Please check whether this period was really no medication\.$"
        ),
        "<name_var> is coded as 999 (no medication), but other course-level fields contain zero values. Please check whether this period was really no medication.",
        "999 with zeros",
    ),
    (
        "MED-QC-13",
        re.compile(
            r"^[A-Za-z0-9_]+ is coded as blinded medication \([^)]+\)\. Please check the medication name and update it if the medication is known now\.$"
        ),
        "<name_var> is coded as blinded medication (<code>). Please check the medication name and update it if the medication is known now.",
        "Blinded med",
    ),
    (
        "SCREENING-CHRONOLOGY",
        re.compile(
            r"^Past pharmaceutical treatment date \(\d{4}-\d{2}-\d{2}\) is more than 20 days later than current pharmaceutical treatment date \(\d{4}-\d{2}-\d{2}\)\.$"
        ),
        "Past pharmaceutical treatment date is more than 20 days later than current pharmaceutical treatment date.",
        "Past > current date by >20 days",
    ),
    (
        "MED-QC-14",
        re.compile(
            # Values are interpolated raw from the record, so they may carry
            # stray whitespace or unusual float reprs ("-.5", "1e+03"); match
            # on the distinctive trailing literal instead of a number pattern.
            r"^[A-Za-z0-9_]+ is .+, but compliance must be between 0 and 100 inclusive\.$"
        ),
        "<comp_var> is outside 0-100 inclusive.",
        "Compliance range",
    ),
    (
        "MED-QC-15",
        re.compile(
            r"^[A-Za-z0-9_]+ is .+\. Values below 0 are only allowed when coded as -3 for missingness\.$"
        ),
        "<freq_var> is below 0 but not -3. Values below 0 are only allowed when coded as -3 for missingness.",
        "Bad freq code",
    ),
    (
        "MED-QC-16",
        re.compile(
            r"^[A-Za-z0-9_]+ \([^)]+\) is coded as an acute injection medication, but the course lasts \d+ days \([A-Za-z0-9_]+ = \d{4}-\d{2}-\d{2}, [A-Za-z0-9_]+ = \d{4}-\d{2}-\d{2}\), which is longer than two weeks\. Please review whether this course is truly an acute injection\.$"
        ),
        "<name_var> is coded as an acute injection medication, but the course lasts >14 days. Please review whether this course is truly an acute injection.",
        "Injection >14 days",
    ),
    (
        "MED-QC-17",
        re.compile(
            r"^[A-Za-z0-9_]+ \(\d+\) is a combination medication , but [A-Za-z0-9_]+ \(.+\) does not contain a '\+' separating two numerical values\. Please record the Amitriptyline and Perphenazine dose separated by a '\+' \(e\.g\. '25\+4'\)\.$"
        ),
        "<name_var> is a combination medication, but <dosage_var> does not contain a '+' separating two numerical values. Please record both component doses separated by a '+'.",
        "Combo dose format",
    ),
    (
        "MED-QC-18",
        re.compile(
            r"^[A-Za-z0-9_]+ is .+ mg for medication code \S+, which exceeds the clinical-use cutoff of \d+ mg\. Please verify the dose is recorded correctly\.$"
        ),
        "<dosage_var> exceeds the clinical-use dose cutoff for its medication code. Please verify the dose is recorded correctly.",
        "Dose over cutoff",
    ),
    (
        # MED-QC-19 variant 1: lifetime AP exposure says yes, past form has no match.
        "MED-QC-19",
        re.compile(
            # The medication name is interpolated into the message and can itself
            # contain parentheses (e.g. "Risperidone (Risperdal)"), so the name
            # groups use .+ anchored on the surrounding literal text rather than
            # a [^)]+ class.
            r"^[A-Za-z0-9_]+ is coded as 1 \(lifetime use of .+\) on lifetime_ap_exposure_screen, but no matching .+ course is recorded in past_pharmaceutical_treatment \(expected one of chrpharm_med\{N\}_name_past in \{[^}]*\}\)\. Please reconcile the two forms\.$"
        ),
        "<chrap_var> is coded as 1 (lifetime antipsychotic use) on lifetime_ap_exposure_screen, but no matching course is recorded in past_pharmaceutical_treatment. Please reconcile the two forms.",
        "AP lifetime, no past course",
    ),
    (
        # MED-QC-19 variant 2: past form records an AP course, lifetime exposure says no.
        "MED-QC-19",
        re.compile(
            r"^[A-Za-z0-9_]+ is coded as .+ \(not 1 / no lifetime use of .+\), but past_pharmaceutical_treatment records a matching .+ course \(.+ = .+\)\. Please reconcile the two forms\.$"
        ),
        "<chrap_var> is not coded as lifetime use, but past_pharmaceutical_treatment records a matching antipsychotic course. Please reconcile the two forms.",
        "AP past course, not lifetime",
    ),
    # ------------------------------------------------------------------
    # Deployed-wording variants. The tracker exports are generated by the
    # deployed pharm checker (see formqc_local_tests/qc_types/clinical_checks/
    # pharm_checks.py), whose messages are worded differently from
    # pharm_checks_new.py. Only MED-QC-01 shares its wording across both
    # versions. Each rule below reuses the normalized text and short label of
    # its new-wording counterpart so counts aggregate by qc_id either way.
    # ------------------------------------------------------------------
    (
        "MED-QC-02",
        re.compile(
            r"^This medication course ends before it starts \([A-Za-z0-9_]+ = \d{4}-\d{2}-\d{2}, [A-Za-z0-9_]+ = \d{4}-\d{2}-\d{2}\)\. Please correct the onset or offset date\.$"
        ),
        "<onset_var> is later than <offset_var>. A medication course cannot end before it starts.",
        "Onset after offset",
    ),
    (
        "MED-QC-03",
        re.compile(
            r"^A medication date \([A-Za-z0-9_]+ = \d{4}-\d{2}-\d{2}\) is later than the form's modification date \([A-Za-z0-9_]+ = \d{4}-\d{2}-\d{2}\)\. Medication onset/offset dates cannot occur after the form modification date\.$"
        ),
        "<med_date_var> is later than <mod_var>. Medication onset/offset dates cannot occur after the form modification date.",
        "Date after mod date",
    ),
    (
        "MED-QC-04",
        re.compile(
            r"^The medication onset date \([A-Za-z0-9_]+ = \d{4}-\d{2}-\d{2}\) does not match the date portion of the first-dose datetime \([A-Za-z0-9_]+ = \d{4}-\d{2}-\d{2}\)\. These should be the same calendar date\.$"
        ),
        "<onset_var> does not match the date portion of <firstdose_var>. The first-dose datetime should have the same calendar date as the medication onset date.",
        "1st dose mismatch",
    ),
    (
        "MED-QC-05",
        re.compile(
            r"^Possible duplicate medication course: two courses record the same medication \([A-Za-z0-9_]+ = [A-Za-z0-9_]+ = .+\) with overlapping dates \(\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2} vs \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}\), and both are marked as simultaneously used \([A-Za-z0-9_]+ = 1, [A-Za-z0-9_]+ = 1\)\. Please check for a duplicate medication course\.$"
        ),
        "<name_var_1> and <name_var_2> have the same medication name, overlapping date ranges, and both courses are marked as simultaneously used. Please check for a duplicate medication course.",
        "Duplicate med",
    ),
    (
        "MED-QC-06",
        re.compile(
            r"^A no-medication period \([A-Za-z0-9_]+ = 999, \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}\) overlaps another medication course \([A-Za-z0-9_]+, \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}\)\. A no-medication period can touch another course on the same start/end date, but it should not overlap beyond that boundary\.$"
        ),
        "<name_var> contains 999 (no medication), but its date range strictly overlaps with another medication course. A no-medication period can touch another course on the same start/end date, but it should not overlap beyond that boundary.",
        "No-med overlap",
    ),
    (
        "MED-QC-07",
        re.compile(
            r"^A no-medication period has no end date \([A-Za-z0-9_]+ = 999, [A-Za-z0-9_]+ = \d{4}-\d{2}-\d{2}, [A-Za-z0-9_]+ missing or coded as ongoing\), but a later medication course starts after it \([A-Za-z0-9_]+, onset \d{4}-\d{2}-\d{2}\)\. Please end the no-medication period or review the later medication course\.$"
        ),
        "<name_var> contains 999 (no medication), but <offset_var> is missing or coded as ongoing while another medication course starts later. Please end the no-medication period or review the later medication course.",
        "No-med no offset",
    ),
    (
        "MED-QC-08",
        re.compile(
            r"^A medication course is coded as ongoing \([A-Za-z0-9_]+ = 1901-01-01\), but courses in the past pharmaceutical treatment form cannot be ongoing\. Please add this medication in the current pharmaceutical treatment form\.$"
        ),
        "<offset_var> is coded as ongoing (1901-01-01), but medication courses in the past pharmaceutical treatment form cannot be ongoing. Please add this medication in the current pharmaceutical treatment form.",
        "Ongoing in past",
    ),
    (
        "MED-QC-09",
        re.compile(
            r"^This course is flagged as ongoing/intermittent \([A-Za-z0-9_]+ = 2\), but a real end date is recorded \([A-Za-z0-9_]+ = \d{4}-\d{2}-\d{2}\) rather than the ongoing code \(1901-01-01\)\. Please reconcile the ongoing/intermittent flag with the offset date\.$"
        ),
        "<ongoing_var> is coded as 2, but <offset_var> is also populated with a date instead of the ongoing code. Please confirm whether this course is ongoing and reconcile the ongoing/intermittent flag with the offset date.",
        "Ongoing conflict",
    ),
    (
        "MED-QC-10",
        re.compile(
            r"^A current-form medication course ends \([A-Za-z0-9_]+ = \d{4}-\d{2}-\d{2}\) before the past pharmaceutical treatment assessment date \([A-Za-z0-9_]+ = \d{4}-\d{2}-\d{2}\)\. A current-form medication course should not end before that assessment date\.$"
        ),
        "<offset_var> is before <past_form_date_var>. A current-form medication course should not end before the past pharmaceutical treatment assessment date.",
        "Ends before past",
    ),
    (
        "MED-QC-11",
        re.compile(
            r"^The medication name is coded as no-information \([A-Za-z0-9_]+ = 888\), but other course-level fields contain data \([^)]+\)\. Please check whether 777 would be more appropriate\.$"
        ),
        "<name_var> is coded as 888 (no information), but other course-level fields contain data. Please check whether 777 would be more appropriate.",
        "888 with data",
    ),
    (
        "MED-QC-12",
        re.compile(
            r"^This course is coded as no medication \([A-Za-z0-9_]+ = 999\), but other course-level fields contain data \([^)]+\)\. Please check whether this period was really no medication\.$"
        ),
        "<name_var> is coded as 999 (no medication), but other course-level fields contain data. Please check whether this period was really no medication.",
        "999 with data",
    ),
    (
        "MED-QC-12",
        re.compile(
            r"^This course is coded as no medication \([A-Za-z0-9_]+ = 999\), but other course-level fields contain zero values \([^)]+\)\. Please check whether this period was really no medication\.$"
        ),
        "<name_var> is coded as 999 (no medication), but other course-level fields contain zero values. Please check whether this period was really no medication.",
        "999 with zeros",
    ),
    (
        "MED-QC-13",
        re.compile(
            r"^The medication name is a blinded-medication code \([A-Za-z0-9_]+ = [^)]+\)\. Please check the medication name and update it if the medication is known now\.$"
        ),
        "<name_var> is coded as blinded medication (<code>). Please check the medication name and update it if the medication is known now.",
        "Blinded med",
    ),
    (
        "SCREENING-CHRONOLOGY",
        re.compile(
            r"^Past pharmaceutical treatment date \(\d{4}-\d{2}-\d{2}\) is more than 20 days after the current pharmaceutical treatment date \(\d{4}-\d{2}-\d{2}\)\. Please verify both interview dates\.$"
        ),
        "Past pharmaceutical treatment date is more than 20 days later than current pharmaceutical treatment date.",
        "Past > current date by >20 days",
    ),
    (
        "MED-QC-14",
        re.compile(
            r"^Medication compliance is out of range \([A-Za-z0-9_]+ = [^)]+\)\. Compliance must be between 0 and 100 inclusive\.$"
        ),
        "<comp_var> is outside 0-100 inclusive.",
        "Compliance range",
    ),
    (
        "MED-QC-15",
        re.compile(
            r"^Medication frequency is negative \([A-Za-z0-9_]+ = [^)]+\)\. Values below 0 are only allowed when coded as -3 for missingness\.$"
        ),
        "<freq_var> is below 0 but not -3. Values below 0 are only allowed when coded as -3 for missingness.",
        "Bad freq code",
    ),
    (
        "MED-QC-16",
        re.compile(
            r"^An acute-injection medication course \([A-Za-z0-9_]+ = [^)]+\) lasts \d+ days \([A-Za-z0-9_]+ = \d{4}-\d{2}-\d{2}, [A-Za-z0-9_]+ = \d{4}-\d{2}-\d{2}\), which is longer than two weeks\. Please review whether this course is truly an acute injection\.$"
        ),
        "<name_var> is coded as an acute injection medication, but the course lasts >14 days. Please review whether this course is truly an acute injection.",
        "Injection >14 days",
    ),
    (
        "MED-QC-17",
        re.compile(
            r"^A combination medication \([A-Za-z0-9_]+ = [^)]+\) does not record both component doses \([A-Za-z0-9_]+ = '.*'\)\. Please enter the two doses separated by '\+' \(e\.g\. '25\+4'\)\.$"
        ),
        "<name_var> is a combination medication, but <dosage_var> does not contain a '+' separating two numerical values. Please record both component doses separated by a '+'.",
        "Combo dose format",
    ),
    (
        "MED-QC-18",
        re.compile(
            r"^The recorded dose for medication code \S+ \([A-Za-z0-9_]+ = -?\d+(?:\.\d+)?(?:[eE][+-]?\d+)? mg\) exceeds the clinical-use cutoff of \d+ mg\. Please verify the dose is recorded correctly\.$"
        ),
        "<dosage_var> exceeds the clinical-use dose cutoff for its medication code. Please verify the dose is recorded correctly.",
        "Dose over cutoff",
    ),
    (
        "MED-QC-19",
        re.compile(
            r"^Lifetime use of .+ is reported on lifetime_ap_exposure_screen \([A-Za-z0-9_]+ = 1\), but no matching .+ course is recorded in past_pharmaceutical_treatment \(expected one of chrpharm_med\{N\}_name_past in \{[^}]*\}\)\. Please reconcile the two forms\.$"
        ),
        "<chrap_var> is coded as 1 (lifetime antipsychotic use) on lifetime_ap_exposure_screen, but no matching course is recorded in past_pharmaceutical_treatment. Please reconcile the two forms.",
        "AP lifetime, no past course",
    ),
    (
        "MED-QC-19",
        re.compile(
            r"^No lifetime use of .+ is reported on lifetime_ap_exposure_screen \([A-Za-z0-9_]+ = .+, not 1\), but past_pharmaceutical_treatment records a matching .+ course \(.+ = .+\)\. Please reconcile the two forms\.$"
        ),
        "<chrap_var> is not coded as lifetime use, but past_pharmaceutical_treatment records a matching antipsychotic course. Please reconcile the two forms.",
        "AP past course, not lifetime",
    ),
    (
        # Deployed variant of MED-QC-19 v2 before med names were added to the
        # message ("not lifetime use" instead of "not 1 / no lifetime use of X").
        "MED-QC-19",
        re.compile(
            r"^[A-Za-z0-9_]+ is coded as .+ \(not 1 / not lifetime use\), but past_pharmaceutical_treatment records a matching antipsychotic course \(.+ = .+\)\. Please reconcile the two forms\.$"
        ),
        "<chrap_var> is not coded as lifetime use, but past_pharmaceutical_treatment records a matching antipsychotic course. Please reconcile the two forms.",
        "AP past course, not lifetime",
    ),
    (
        # Deployed variant of MED-QC-19 v1 before med names were added.
        "MED-QC-19",
        re.compile(
            r"^[A-Za-z0-9_]+ is coded as 1 \(lifetime antipsychotic use\) on lifetime_ap_exposure_screen, but no matching course is recorded in past_pharmaceutical_treatment \(expected one of chrpharm_med\{N\}_name_past in \{[^}]*\}\)\. Please reconcile the two forms\.$"
        ),
        "<chrap_var> is coded as 1 (lifetime antipsychotic use) on lifetime_ap_exposure_screen, but no matching course is recorded in past_pharmaceutical_treatment. Please reconcile the two forms.",
        "AP lifetime, no past course",
    ),
    (
        # General (non-pharm-specific) blank-variable flag that shares tracker
        # rows with the pharm form. Not a MED-QC id; kept out of the doc but
        # named so it doesn't pollute summaries as RAW fallback.
        "VARIABLE-BLANK",
        re.compile(r"^Variable is blank\.$"),
        "Variable is blank.",
        "Blank variable",
    ),
    (
        # Sibling of the blank-variable flag (general_checks.check_if_missing_code).
        "VARIABLE-MISSING-CODE",
        re.compile(r"^Variable is a missing code\.$"),
        "Variable is a missing code.",
        "Missing-code variable",
    ),
]


def parse_flag_cell(cell):
    if pd.isna(cell):
        return []

    text = str(cell).strip()
    if not text:
        return []

    parts = re.split(r"\s*(?:\||\n|\r|;|•|\u2022)\s*", text)
    return [p.strip() for p in parts if p and p.strip()]


_VAR_PREFIX_RE = re.compile(r"^[A-Za-z0-9_]+\s*:\s*")


def normalize_flag(flag: str):
    flag = _clean(flag)
    if not flag:
        return None, None, None

    body = _VAR_PREFIX_RE.sub("", flag, count=1)

    for qc_id, regex, normalized_text, short_label in RULES:
        if regex.match(body):
            return qc_id, normalized_text, short_label

    normalized = re.sub(r"\d", "", body)
    normalized = re.sub(r"\(\s*[-./]+\s*\)", "", normalized)
    normalized = re.sub(r"[-./]{2,}", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    short = normalized[:18] + "..." if len(normalized) > 18 else normalized
    return "RAW-NUMBERS-REMOVED", normalized, short


def summarize_flags(df, flag_column="Flags", subject_column=None):
    if flag_column not in df.columns:
        raise ValueError(
            f"Column '{flag_column}' not found. Available columns: {list(df.columns)}"
        )

    rows = []
    for idx, record in df.iterrows():
        if subject_column and subject_column in df.columns:
            subject = record[subject_column]
        else:
            subject = f"ROW_{idx + 1}"

        for raw_flag in parse_flag_cell(record[flag_column]):
            qc_id, normalized, short_label = normalize_flag(raw_flag)
            rows.append(
                {
                    "source_row": idx + 1,
                    "subject": subject,
                    "raw_flag": raw_flag,
                    "qc_id": qc_id,
                    "normalized_flag": normalized,
                    "short_label": short_label,
                }
            )

    exploded = pd.DataFrame(rows)

    if exploded.empty:
        summary = pd.DataFrame(
            columns=[
                "qc_id",
                "normalized_flag",
                "short_label",
                "count",
                "subjects_with_flag",
                "example_raw_flag",
            ]
        )
        return summary, exploded

    summary = (
        exploded.groupby(["qc_id", "normalized_flag", "short_label"], dropna=False)
        .agg(
            count=("raw_flag", "size"),
            subjects_with_flag=("subject", pd.Series.nunique),
            example_raw_flag=("raw_flag", "first"),
        )
        .reset_index()
        .sort_values(
            ["count", "subjects_with_flag", "qc_id", "normalized_flag"],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )

    return summary, exploded


def read_input(path: Path, sheet_name=0):
    """Read a CSV or Excel input.

    ``sheet_name`` follows :func:`pandas.read_excel`: the default reads the
    first worksheet for backwards compatibility, while ``None`` returns every
    worksheet as a ``{sheet_name: DataFrame}`` mapping.  CSV inputs always
    return a single DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    return pd.read_excel(path, sheet_name=sheet_name)


def detect_subject_column(df):
    for col in PREFERRED_SUBJECT_COLUMNS:
        if col in df.columns:
            return col
    return None


def shorten_text(text: str, max_len: int = 65) -> str:
    text = _clean(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def create_top_5_flag_graphs(summary: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary.empty:
        return pd.DataFrame(
            columns=["graph_type", "rank", "qc_id", "normalized_flag", "count", "chart_file"]
        )

    top5 = summary.head(5).copy()
    top5["rank"] = range(1, len(top5) + 1)

    raw_labels = top5["short_label"].tolist()
    seen = {}
    labels = []
    for lbl in raw_labels:
        seen[lbl] = seen.get(lbl, 0) + 1
        labels.append(lbl if seen[lbl] == 1 else f"{lbl} ({seen[lbl]})")
    counts = top5["count"].astype(int).tolist()
    ranks = top5["rank"].tolist()

    chart_rows = []

    # Vertical bar chart
    bar_file = output_dir / "top_5_flags_bar_chart.png"
    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, counts)
    plt.xlabel("Top 5 Flags")
    plt.ylabel("Count")
    plt.title("Top 5 Normalized Flags - Bar Chart")
    plt.xticks(rotation=45, ha="right")

    y_offset = max(counts) * 0.01 if counts else 1
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(bar_file, dpi=150, bbox_inches="tight")
    plt.close()

    # Horizontal bar chart (reverse so highest count is at the top)
    hbar_file = output_dir / "top_5_flags_horizontal_bar_chart.png"
    plt.figure(figsize=(12, 7))
    h_labels = labels[::-1]
    h_counts = counts[::-1]
    bars = plt.barh(h_labels, h_counts)
    plt.xlabel("Count")
    plt.ylabel("Top 5 Flags")
    plt.title("Top 5 Normalized Flags - Horizontal Bar Chart")

    x_offset = max(counts) * 0.01 if counts else 1
    for bar, count in zip(bars, h_counts):
        plt.text(
            bar.get_width() + x_offset,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
        )

    plt.tight_layout()
    plt.savefig(hbar_file, dpi=150, bbox_inches="tight")
    plt.close()

    # Line chart by rank
    line_file = output_dir / "top_5_flags_line_chart.png"
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, counts, marker="o")
    plt.xticks(ranks, [f"#{r}" for r in ranks])
    plt.xlabel("Flag Rank")
    plt.ylabel("Count")
    plt.title("Top 5 Flags - Line Chart")
    plt.grid(True, alpha=0.3)

    for x, y in zip(ranks, counts):
        plt.text(x, y + y_offset, str(y), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(line_file, dpi=150, bbox_inches="tight")
    plt.close()

    for graph_type, chart_file in [
        ("Bar Chart", bar_file),
        ("Horizontal Bar Chart", hbar_file),
        ("Line Chart", line_file),
    ]:
        temp = top5[["rank", "qc_id", "normalized_flag", "count"]].copy()
        temp.insert(0, "graph_type", graph_type)
        temp["chart_file"] = str(chart_file)
        chart_rows.append(temp)

    return pd.concat(chart_rows, ignore_index=True)


def get_writable_output_path(path: Path) -> Path:
    if not path.exists():
        return path

    try:
        with open(path, "ab"):
            return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def main():
    print(f"Reading: {INPUT_FILE}")
    df = read_input(INPUT_FILE)

    subject_col = detect_subject_column(df)
    print(f"Using subject column: {subject_col}")
    print(f"Using flag column: {FLAG_COLUMN}")

    summary, exploded = summarize_flags(
        df,
        flag_column=FLAG_COLUMN,
        subject_column=subject_col,
    )

    rules_df = pd.DataFrame(
        [(qc_id, regex.pattern, normalized, short) for qc_id, regex, normalized, short in RULES],
        columns=["qc_id", "regex", "normalized_flag", "short_label"],
    )

    graph_index_df = create_top_5_flag_graphs(summary, CHART_DIR)
    output_path = get_writable_output_path(OUTPUT_FILE)

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            summary.to_excel(writer, index=False, sheet_name="Flag Summary")
            exploded.to_excel(writer, index=False, sheet_name="Exploded Flags")
            rules_df.to_excel(writer, index=False, sheet_name="Normalization Rules")
            graph_index_df.to_excel(writer, index=False, sheet_name="Top 5 Graphs")
    except PermissionError as e:
        raise PermissionError(
            f"Could not write output file '{output_path}'. "
            "It may already be open in Excel. Close it and try again."
        ) from e

    print(f"Wrote: {output_path.resolve()}")
    print(f"Saved graphs to: {CHART_DIR.resolve()}")


if __name__ == "__main__":
    main()
