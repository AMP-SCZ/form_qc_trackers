"""Focused contract tests for the dictionary-driven Proposed Checks engine.

The production dictionary is intentionally not read for individual rule tests.
ProposedChecks accepts a small dataframe through
``form_check_info['_proposed_checks_dictionary']`` so each test exercises one
rule family without depending on the canonical Utils dictionary loader.
"""

from __future__ import annotations

import ast
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from qc_forms import form_check as form_check_module
import qc_types.proposed_checks as proposed
from generate_reports.calculate_resolved_errors import CalculateResolvedErrors
from generate_reports.create_trackers import CreateTrackers


ROOT = Path(__file__).resolve().parents[3]
REPORT_NAME = "Proposed Checks"

DD_COLUMNS = (
    "Variable / Field Name",
    "Form Name",
    "Section Header",
    "Field Type",
    "Field Label",
    "Choices, Calculations, OR Slider Labels",
    "Field Note",
    "Text Validation Type OR Show Slider Number",
    "Text Validation Min",
    "Text Validation Max",
    "Identifier?",
    "Branching Logic (Show field only if...)",
    "Required Field?",
    "Custom Alignment",
    "Question Number (surveys only)",
    "Matrix Group Name",
    "Matrix Ranking?",
    "Field Annotation",
)


def _dd_row(
        variable, *, field_type="text", choices="", validation="",
        minimum="", maximum="", branch="", required="", annotation="",
        identifier=""):
    row = {column: "" for column in DD_COLUMNS}
    row.update({
        "Variable / Field Name": variable,
        "Form Name": "test_form",
        "Field Type": field_type,
        "Field Label": variable.replace("_", " ").title(),
        "Choices, Calculations, OR Slider Labels": choices,
        "Text Validation Type OR Show Slider Number": validation,
        "Text Validation Min": minimum,
        "Text Validation Max": maximum,
        "Identifier?": identifier,
        "Branching Logic (Show field only if...)": branch,
        "Required Field?": required,
        "Field Annotation": annotation,
    })
    return row


def _dictionary(*rows):
    return pd.DataFrame(rows, columns=DD_COLUMNS)


def _form_rows(form, *rows):
    """Assign the small synthetic dictionary rows to one real form."""
    for row in rows:
        row["Form Name"] = form
    return rows


def _form_check_info(dictionary, curated_rules=None):
    form_info = {
        "completion_var": "test_form_complete",
        "missing_var": "test_form_missing",
        "non_branch_logic_vars": [],
        "interview_date_var": "",
    }
    schedules = {"baseline": ["test_form"]}
    converted_branching_logic = {}
    for _, row in dictionary.iterrows():
        variable = row["Variable / Field Name"]
        branch = row["Branching Logic (Show field only if...)"]
        if branch == "[test_gate] = '1'":
            converted_branching_logic[variable] = {
                "converted_branching_logic": (
                    "hasattr(curr_row, 'test_gate') and "
                    "str(curr_row.test_gate) in ['1', '1.0']"),
            }
    return {
        "subject_info": {
            "S1": {
                "cohort": "CHR",
                "inclusion_status": "included",
                "visit_status": "baseline",
            },
        },
        "general_check_vars": {
            "excluded_vars": {"PRONET": [], "PRESCIENT": []},
            "excluded_forms": {"PRONET": [], "PRESCIENT": []},
        },
        "important_form_vars": {"test_form": form_info},
        # Include both spellings because legacy FormCheck consumers use the
        # source cohort spelling while newer dataframe-wide adapters normalize.
        "forms_per_timepoint": {"CHR": schedules, "chr": schedules},
        "converted_branching_logic": converted_branching_logic,
        "excluded_branching_logic_vars": {},
        "team_report_forms": {},
        "grouped_variables": {
            "var_forms": {
                row["Variable / Field Name"]: row["Form Name"]
                for _, row in dictionary.iterrows()
            },
            "var_translations": {},
            "scid_vars": {"module_b_vars": [], "module_c_vars": []},
        },
        "variables_added_later": {},
        "raw_csv_conversions": {},
        "variable_ranges": {"PRONET": {}, "PRESCIENT": {}},
        "earliest_latest_dates_per_tp": {},
        "missingness_domain_forms": {},
        "cognition_csvs": {},
        "config_info": {
            "withdrawn_enabled": False,
            "recruited_only": False,
        },
        "_proposed_checks_dictionary": dictionary,
        "_proposed_checks_curated_rules": list(curated_rules or []),
    }


def _combined_row(**values):
    row = {
        "subjectid": "S1",
        "visit_status": "baseline",
        "visit_status_string": "baseline",
        "recruitment_status_v2": "recruited",
        "test_form_complete": 2,
        "test_form_missing": 0,
    }
    row.update(values)
    return pd.DataFrame([row])


def _run(dictionary, values, curated_rules=None):
    checker = proposed.ProposedChecks(
        _combined_row(**values),
        "baseline",
        "PRONET",
        _form_check_info(dictionary, curated_rules),
    )
    return checker()


def _run_on_form(dictionary, values, form):
    """Run a compact dictionary whose fields belong to one clinical form."""
    dictionary = dictionary.copy()
    dictionary["Form Name"] = form
    info = _form_check_info(dictionary)
    form_info = info["important_form_vars"].pop("test_form")
    info["important_form_vars"][form] = form_info
    for cohort in ("CHR", "chr"):
        info["forms_per_timepoint"][cohort]["baseline"] = [form]
    return proposed.ProposedChecks(
        _combined_row(**values), "baseline", "PRONET", info)()


def _findings_with_id(output, check_id):
    return [finding for finding in output
            if finding.get("check_id") == check_id]


def _assert_proposed_route(finding):
    reports = finding.get("reports", "")
    tokens = reports if isinstance(reports, list) else str(reports).split(" | ")
    assert tokens == [REPORT_NAME]
    assert finding.get("nda_excluder") is False


def test_production_catalog_uses_utils_dictionary_loader_once_per_run():
    dictionary = _dictionary(_dd_row("test_choice"))
    calls = []
    checker = object.__new__(proposed.ProposedChecks)
    checker.utils = SimpleNamespace(
        config_info={"paths": {"dependencies_path": "/canonical/dependencies/"}},
        read_data_dictionary=lambda: calls.append("read") or dictionary,
    )

    first = checker._resolve_catalog({})
    second = checker._resolve_catalog({})

    assert calls == ["read"]
    assert first is second
    assert "test_choice" in first.fields


def test_proposed_imports_use_production_top_level_utility_packages():
    tree = ast.parse(
        (ROOT / "main" / "qc_forms" / "qc_types" / "proposed_checks.py").read_text(encoding="utf-8"))
    modules = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "qc_forms.form_check" in modules
    assert "utils.branching_logic_eval" in modules
    assert "process_variables.transform_calculated_fields" in modules
    assert "process_variables.transform_branching_logic" in modules
    assert "qc_forms.utils.branching_logic_eval" not in modules
    assert "qc_forms.process_variables.transform_calculated_fields" not in modules


def test_guid_birth_components_must_form_and_match_the_governed_dob():
    dictionary = _dictionary(*_form_rows(
        "guid_form",
        _dd_row("chrguid_mob", validation="number"),
        _dd_row("chrguid_dob", validation="number"),
        _dd_row("chrguid_yob", validation="number"),
        _dd_row("chrdemo_dob", validation="date_ymd"),
    ))

    output = _run(dictionary, {
        "chrguid_mob": 2,
        "chrguid_dob": 29,
        "chrguid_yob": 2000,
        "chrdemo_dob": "2000-02-28",
    })

    assert _findings_with_id(output, "IDENT-003")


def test_ae_blank_clinician_detail_is_silent_but_chronology_remains():
    dictionary = _dictionary(*_form_rows(
        "adverse_events",
        _dd_row("chrae_aescreen", field_type="radio", choices="0, No | 1, Yes"),
        _dd_row("chrae_tp1", field_type="radio", choices="0, Baseline"),
        _dd_row("chrae_ae1"),
        _dd_row("chrae_aes1date", validation="date_ymd"),
        _dd_row("chrae_d1", field_type="radio", choices="1, Mild"),
        _dd_row("chrae_dr1", field_type="radio", choices="1, Display"),
        _dd_row("chr_ae1date_dr", validation="date_ymd"),
        _dd_row("chrae_sig1_dr"),
    ))

    output = _run(dictionary, {
        "chrae_aescreen": 1,
        "chrae_tp1": 0,
        "chrae_ae1": "Headache",
        "chrae_aes1date": "2025-02-03",
        "chrae_d1": 1,
        "chrae_dr1": 1,
        "chr_ae1date_dr": "2025-02-02",
        "chrae_sig1_dr": "",
    })

    assert not _findings_with_id(output, "EP-AE-006")
    assert _findings_with_id(output, "EP-AE-007")


def test_cbc_and_blood_extensions_cover_sum_position_and_review_date():
    cbc_rows = _form_rows(
        "cbc_with_differential",
        *[_dd_row(name, validation="number") for name in (
            "chrcbc_neut", "chrcbc_lymph", "chrcbc_monos", "chrcbc_eos",
            "chrcbc_baso", "chrcbc_wbcsum")],
    )
    blood_rows = _form_rows(
        "blood_sample_preanalytic_quality_assurance",
        _dd_row("chrblood_drawdate", validation="datetime_ymd"),
        _dd_row("chrblood_rack_barcode"),
        _dd_row("chrblood_wb1pos"),
        _dd_row("chrblood_wb2pos"),
    )
    gcp_rows = _form_rows(
        "gcp_cbc_with_differential",
        _dd_row("chrcbccs_review_date", validation="datetime_ymd"),
    )
    dictionary = _dictionary(*cbc_rows, *blood_rows, *gcp_rows)

    output = _run(dictionary, {
        "chrcbc_neut": 1,
        "chrcbc_lymph": 1,
        "chrcbc_monos": 1,
        "chrcbc_eos": 1,
        "chrcbc_baso": 1,
        "chrcbc_wbcsum": 1,
        "chrblood_drawdate": "2025-02-03 10:00",
        "chrblood_rack_barcode": "rack-1",
        "chrblood_wb1pos": "A1",
        "chrblood_wb2pos": "A1",
        "chrcbccs_review_date": "2025-02-02 10:00",
    })

    assert _findings_with_id(output, "CBC-007")
    assert _findings_with_id(output, "BLOOD-005")
    assert _findings_with_id(output, "GCP-CBC-005")


def test_saliva_bed_wake_shared_freeze_and_rack_extensions():
    dictionary = _dictionary(*_form_rows(
        "daily_activity_and_saliva_sample_collection",
        _dd_row("chrsaliva_beddate", validation="date_ymd"),
        _dd_row("chrsaliva_bedtime", validation="time"),
        _dd_row("chrsaliva_wakedate", validation="date_ymd"),
        _dd_row("chrsaliva_waketime", validation="time"),
        _dd_row("chrsaliva_coldate", validation="date_ymd"),
        _dd_row("chrsaliva_time1", validation="time"),
        _dd_row("chrsaliva_time2", validation="time"),
        _dd_row("chrsaliva_time3", validation="time"),
        _dd_row("chrsaliva_frztime", validation="datetime_ymd"),
        _dd_row("chrsaliva_box1a"),
        _dd_row("chrsaliva_box2a"),
    ))

    output = _run(dictionary, {
        "chrsaliva_beddate": "2025-02-03",
        "chrsaliva_bedtime": "23:00",
        "chrsaliva_wakedate": "2025-02-03",
        "chrsaliva_waketime": "06:00",
        "chrsaliva_coldate": "2025-02-03",
        "chrsaliva_time1": "05:00",
        "chrsaliva_time2": "06:00",
        "chrsaliva_time3": "07:00",
        "chrsaliva_frztime": "2025-02-03 06:00",
        "chrsaliva_box1a": "rack-a",
        "chrsaliva_box2a": "rack-b",
    })

    assert _findings_with_id(output, "SALIVA-006")
    assert _findings_with_id(output, "SALIVA-007")
    assert _findings_with_id(output, "SALIVA-008")


def test_administrative_missing_reasons_and_responses_may_be_blank():
    missing_rows = _form_rows(
        "missing_data",
        _dd_row("chrmiss_domain", field_type="radio", choices="0, No | 1, Yes"),
        _dd_row("chrmiss_domain_spec", field_type="radio", choices="M1, Refusal"),
        _dd_row("chrmiss_domain_type", field_type="checkbox", choices="1, Clinical | 2, EEG"),
    )
    scale_rows = _form_rows(
        "cdss",
        _dd_row("chrcdss_missing", field_type="radio", choices="0, No | 1, Yes"),
        _dd_row("chrcdss_missing_spec", field_type="radio", choices="M1, Reason"),
    )
    pgis_rows = _form_rows(
        "pgis",
        _dd_row("chrpgi_missing", field_type="radio", choices="0, No | 1, Yes"),
        _dd_row("chrpgi_missing_spec", field_type="radio", choices="M1, Reason"),
        _dd_row("chrpgi_2", field_type="radio", choices="1, Normal"),
    )
    dictionary = _dictionary(*missing_rows, *scale_rows, *pgis_rows)

    output = _run(dictionary, {
        "chrmiss_domain": 1,
        "chrmiss_domain_spec": "",
        "chrmiss_domain_type___1": 0,
        "chrmiss_domain_type___2": 0,
        "chrcdss_missing": 1,
        "chrcdss_missing_spec": "",
        "chrpgi_missing": 0,
        "chrpgi_missing_spec": "",
        "chrpgi_2": "",
    })

    for check_id in (
            "MISS-ADMIN-001", "MISS-ADMIN-002",
            "SCALE-MISS-001", "SCALE-MISS-002"):
        assert not _findings_with_id(output, check_id)


def test_administrative_coenrollment_tbi_and_sibling_extensions():
    coen_rows = _form_rows(
        "coenrollment_form",
        _dd_row("chrcoen_study1_name"), _dd_row("chrcoen_study1_tp"),
        _dd_row("chrcoen_study1_start", validation="date_ymd"),
        _dd_row("chrcoen_study1_id"),
        _dd_row("chrcoen_study1_amp_status", field_type="radio", choices="15, Completed Time-Point"),
        _dd_row("chrcoen_study1_timepoint", field_type="radio", choices="1, Month 1"),
        _dd_row("chrcoen_other_studies2", field_type="radio", choices="0, No | 1, Yes"),
        _dd_row("chrcoen_study2_name"), _dd_row("chrcoen_study2_tp"),
        _dd_row("chrcoen_study2_start", validation="date_ymd"),
        _dd_row("chrcoen_study2_id"),
        _dd_row("chrcoen_study2_amp_status", field_type="radio", choices="1, Allocated"),
    )
    tbi_rows = _form_rows(
        "traumatic_brain_injury_screen",
        _dd_row("chrtbi_subject_head_injury", field_type="radio", choices="0, No | 1, Yes"),
        _dd_row("chrtbi_subject_times", field_type="radio", choices="1, 1 | 2, 2 | 3, 3+"),
        _dd_row("chrtbi_subject_symptoms"),
        _dd_row("chrtbi_number_injs", validation="integer"),
    )
    demo_rows = _form_rows(
        "sociodemographics",
        _dd_row("chrdemo_sibling", field_type="radio", choices="1, No | 2, Twin"),
        _dd_row("chrdemo_sibling_id"),
    )
    dictionary = _dictionary(*coen_rows, *tbi_rows, *demo_rows)

    output = _run(dictionary, {
        "chrcoen_study1_name": "Study A", "chrcoen_study1_tp": 1,
        "chrcoen_study1_start": "2025-01-01", "chrcoen_study1_id": "X1",
        "chrcoen_study1_amp_status": 15, "chrcoen_study1_timepoint": "",
        "chrcoen_other_studies2": 0,
        "chrcoen_study2_name": "Study A", "chrcoen_study2_tp": 1,
        "chrcoen_study2_start": "2025-01-01", "chrcoen_study2_id": "X1",
        "chrcoen_study2_amp_status": 1,
        "chrtbi_subject_head_injury": 0,
        "chrtbi_subject_times": 2,
        "chrtbi_subject_symptoms": "stale",
        "chrtbi_number_injs": 1,
        "chrdemo_sibling": 2,
        "chrdemo_sibling_id": "",
    })

    assert _findings_with_id(output, "COENROLL-005")
    assert not _findings_with_id(output, "COENROLL-006")
    assert _findings_with_id(output, "COENROLL-007")
    assert _findings_with_id(output, "TBI-006")
    assert _findings_with_id(output, "TBI-007")
    assert not _findings_with_id(output, "IDENT-009")


def test_current_health_extended_chronology_and_state_checks():
    output = _run_on_form(
        _dictionary(
            _dd_row("chrchs_interview_date", validation="date_ymd"),
            _dd_row("chrchs_sick", field_type="radio",
                    choices="0, Not sick | 1, Mild | 2, Moderate | 3, Severe"),
            _dd_row("chrchs_sickdate", validation="date_ymd"),
            _dd_row("chrchs_covidinf", field_type="radio",
                    choices="0, No | 1, Maybe | 2, Diagnosed | 3, Positive"),
            _dd_row("chrchs_covidinfdate", validation="date_ymd"),
            _dd_row("chrchs_covidinfdays", validation="integer"),
            _dd_row("chrchs_covidvac", field_type="checkbox",
                    choices="0, No | 1, Pfizer | 99, Other"),
            _dd_row("chrchs_covidvacnum", validation="integer"),
            _dd_row("chrchs_covidvac1date", validation="date_ymd"),
            _dd_row("chrchs_covidvac2date", validation="date_ymd"),
        ),
        {
            "chrchs_interview_date": "2025-01-10",
            "chrchs_sick": 0,
            "chrchs_sickdate": "2025-01-11",
            "chrchs_covidinf": 3,
            "chrchs_covidinfdate": "2025-01-08",
            "chrchs_covidinfdays": 5,
            "chrchs_covidvac___0": 0,
            "chrchs_covidvac___1": 1,
            "chrchs_covidvac___99": 0,
            "chrchs_covidvacnum": 2,
            "chrchs_covidvac1date": "2024-01-01",
            "chrchs_covidvac2date": "",
        },
        "current_health_status")

    assert _findings_with_id(output, "HEALTH-010")
    assert _findings_with_id(output, "HEALTH-011")
    assert _findings_with_id(output, "HEALTH-013")
    assert not _findings_with_id(output, "HEALTH-014")


def test_current_medication_introduction_and_terminal_state_checks():
    output = _run_on_form(
        _dictionary(
            _dd_row("chrpharm_med1_name", field_type="dropdown", choices="1, Medication"),
            _dd_row("chrpharm_med1_tp", field_type="dropdown", choices="0, Baseline"),
            _dd_row("chrpharm_med1_onset", validation="date_ymd"),
            _dd_row("chrpharm_med1_use", field_type="dropdown", choices="1, Regular"),
            _dd_row("chrpharm_med1_datasource", field_type="dropdown", choices="1, Pharmacy"),
            _dd_row("chrpharm_med1_indication", field_type="dropdown", choices="1, Sleep"),
            _dd_row("chrpharm_med1_mo0", field_type="radio",
                    choices="1, Continuing | 2, Stopped | 3, Missed"),
            _dd_row("chrpharm_med1_mo1", field_type="radio",
                    choices="1, Continuing | 2, Stopped | 3, Missed"),
            _dd_row("chrpharm_med1_offset", validation="date_ymd"),
        ),
        {
            "chrpharm_med1_name": 1,
            "chrpharm_med1_tp": 0,
            "chrpharm_med1_onset": "2024-01-01",
            "chrpharm_med1_use": 1,
            "chrpharm_med1_datasource": 1,
            "chrpharm_med1_indication": 1,
            "chrpharm_med1_mo0": 1,
            "chrpharm_med1_mo1": 2,
            "chrpharm_med1_offset": "",
        },
        "current_pharmaceutical_treatment_floating_med_125")

    assert _findings_with_id(output, "MED-STATE-002")
    assert not _findings_with_id(output, "MED-STATE-004")


def test_axivity_qaqc_requires_exactly_one_state():
    output = _run_on_form(
        _dictionary(
            _dd_row("chraxci_interview_date", validation="date_ymd"),
            _dd_row("chraxci_qaqc1", field_type="checkbox",
                    choices="1, Yes | 2, No | 3, N/A"),
        ),
        {
            "chraxci_interview_date": "2025-01-01",
            "chraxci_qaqc1___1": 1,
            "chraxci_qaqc1___2": 1,
            "chraxci_qaqc1___3": 0,
        },
        "digital_biomarkers_axivity_checkin")

    findings = _findings_with_id(output, "AXIVITY-005")
    assert len(findings) == 1
    assert findings[0]["proposed_variable_values"] == (
        "chraxci_qaqc1___1=1; chraxci_qaqc1___2=1; "
        "chraxci_qaqc1___3=0")


def test_figs_roster_ignores_unselected_checkbox_columns_but_flags_real_extra_row():
    dictionary = _dictionary(
        _dd_row("chrfigs_numsib", validation="integer"),
        _dd_row("chrfigs_sibling1_info", field_type="yesno"),
        _dd_row("chrfigs_sibling1_diag", field_type="checkbox",
                choices="1, Diagnosis A | 2, Diagnosis B"),
    )
    clean = _run_on_form(
        dictionary,
        {
            "chrfigs_numsib": 0,
            "chrfigs_sibling1_info": "",
            "chrfigs_sibling1_diag___1": 0,
            "chrfigs_sibling1_diag___2": 0,
        },
        "family_interview_for_genetic_studies_figs")
    assert not _findings_with_id(clean, "FIGS-003")

    extra = _run_on_form(
        dictionary,
        {
            "chrfigs_numsib": 0,
            "chrfigs_sibling1_info": 1,
            "chrfigs_sibling1_diag___1": 0,
            "chrfigs_sibling1_diag___2": 0,
        },
        "family_interview_for_genetic_studies_figs")
    assert _findings_with_id(extra, "FIGS-003")


@pytest.fixture(autouse=True)
def _isolated_runtime(monkeypatch):
    class _TestUtils:
        absolute_path = ""
        missing_code_list = [-3, -9, -99, 999]
        missing_code_set = frozenset(missing_code_list)

        @staticmethod
        def create_timepoint_list():
            return [
                "screening", "baseline", "month1", "month2", "month3",
                "month6", "month9", "month12", "month18", "month24",
            ]

        @staticmethod
        def can_be_float(value):
            try:
                float(value)
            except (TypeError, ValueError):
                return False
            return True

    # FormCheck resolves Utils from its own module globals at construction.
    # Replacing it keeps these unit tests independent of config.json and the
    # repository/deployment package-name layout.
    monkeypatch.setattr(form_check_module, "Utils", _TestUtils)
    clear = getattr(proposed, "clear_catalog_cache", None)
    if callable(clear):
        clear()
    yield
    if callable(clear):
        clear()


def test_declared_radio_domain_rejects_an_unknown_code_and_routes_once():
    dictionary = _dictionary(_dd_row(
        "test_choice", field_type="radio", choices="0, No | 1, Yes"))

    output = _run(dictionary, {"test_choice": 7})

    findings = _findings_with_id(output, "DD-DOMAIN-001")
    assert len(findings) == 1
    assert findings[0]["displayed_variable"] == "test_choice"
    assert findings[0]["proposed_variable_values"] == "test_choice=7"
    _assert_proposed_route(findings[0])


def test_legacy_missing_code_owner_suppresses_derivative_domain_noise():
    dictionary = _dictionary(_dd_row(
        "test_yesno", field_type="yesno", annotation="@NOMISSING"))
    info = _form_check_info(dictionary)
    info["general_check_vars"]["missing_code_vars"] = {
        "PRONET": {"Main Report": {"test_form": ["test_yesno"]}},
        "PRESCIENT": {"Main Report": {"test_form": ["test_yesno"]}},
    }

    output = proposed.ProposedChecks(
        _combined_row(test_yesno=999), "baseline", "PRONET", info)()

    assert not _findings_with_id(output, "DD-MISS-001")
    assert not _findings_with_id(output, "DD-DOMAIN-002")


@pytest.mark.parametrize(
    ("value", "expected_check_id"),
    [
        ("not-a-number", "DD-NUM-001"),
        ("1.5", "DD-NUM-002"),
        (11, "DD-NUM-003"),
    ],
)
def test_numeric_parser_integrality_and_inclusive_bounds(
        value, expected_check_id):
    dictionary = _dictionary(_dd_row(
        "test_integer", validation="integer", minimum="0", maximum="10"))

    output = _run(dictionary, {"test_integer": value})

    findings = _findings_with_id(output, expected_check_id)
    assert len(findings) == 1
    _assert_proposed_route(findings[0])

    # A primary parse failure must suppress derivative integrality/range
    # noise for the same value.  Parseable values may proceed to one later
    # numeric stage, but must still produce one primary finding per cause.
    numeric_findings = [
        finding for finding in output
        if finding.get("check_id", "").startswith("DD-NUM-")
        and finding.get("displayed_variable") == "test_integer"
    ]
    assert [finding["check_id"] for finding in numeric_findings] == [
        expected_check_id]


def test_legacy_range_owner_suppresses_duplicate_dictionary_range_finding():
    dictionary = _dictionary(_dd_row(
        "test_integer", validation="integer", minimum="0", maximum="10"))
    info = _form_check_info(dictionary)
    info["variable_ranges"] = {
        "PRONET": {
            "test_integer": {"min": 0, "max": 10, "form": "test_form"}},
        "PRESCIENT": {},
    }

    output = proposed.ProposedChecks(
        _combined_row(test_integer=11), "baseline", "PRONET", info)()

    assert not _findings_with_id(output, "DD-NUM-003")


def test_date_validator_rejects_a_canonical_shaped_impossible_date():
    dictionary = _dictionary(_dd_row(
        "test_date", validation="date_ymd"))

    output = _run(dictionary, {"test_date": "2025-02-30"})

    findings = _findings_with_id(output, "DD-FMT-001")
    assert len(findings) == 1
    _assert_proposed_route(findings[0])


def test_explicit_current_event_calculation_resolves_the_current_row():
    dictionary = _dictionary(
        _dd_row("test_source", validation="integer"),
        _dd_row(
            "test_calculation", field_type="calc",
            choices="[baseline_arm_1][test_source] + 1"),
    )

    output = _run(dictionary, {
        "test_source": 2,
        "test_calculation": 3,
    })

    assert not _findings_with_id(output, "DD-CALC-001")
    assert not [
        finding for finding in _findings_with_id(output, "DD-SCHEMA-005")
        if "baseline_arm_1" in finding.get("error_message", "")
    ]


def test_unconditionally_required_scalar_blank_is_not_a_proposed_finding():
    dictionary = _dictionary(_dd_row("test_required", required="y"))

    output = _run(dictionary, {"test_required": ""})

    assert not _findings_with_id(output, "DD-REQ-001")


@pytest.mark.parametrize(
    "missing_value",
    [
        "", -3, -9, -99, 999,
        "1901-01-01", "1903-03-03", "1909-03-03", "1909-09-09",
    ],
)
def test_blanks_and_governed_sentinels_emit_no_generic_participant_findings(
        missing_value):
    dictionary = _dictionary(
        _dd_row("test_required", required="y"),
        _dd_row(
            "test_required_checkbox", field_type="checkbox",
            choices="1, First | 2, Second", required="y"),
        _dd_row(
            "test_choice", field_type="radio", choices="0, No | 1, Yes"),
        _dd_row(
            "test_number", validation="integer", minimum="0", maximum="10"),
        _dd_row("test_date", validation="date_ymd"),
        _dd_row(
            "test_calculation", field_type="calc",
            choices="[test_number] + 1"),
    )

    output = _run(dictionary, {
        "test_required": missing_value,
        "test_required_checkbox___1": 0,
        "test_required_checkbox___2": "0",
        "test_choice": missing_value,
        "test_number": missing_value,
        "test_date": missing_value,
        "test_calculation": missing_value,
        "test_form_complete": 2,
    })

    forbidden_ids = {
        "DD-REQ-001", "DD-REQ-002",
        "FORM-002", "FORM-003", "FORM-006",
    }
    forbidden_prefixes = (
        "DD-MISS-", "DD-DOMAIN-", "DD-NUM-", "DD-FMT-", "DD-CALC-",
    )
    assert not [
        finding for finding in output
        if finding.get("check_id") in forbidden_ids
        or finding.get("check_id", "").startswith(forbidden_prefixes)
    ]


def test_removed_form_missing_message_is_absent_even_with_data_and_complete_status():
    dictionary = _dictionary(
        _dd_row("test_form_missing", field_type="yesno"),
        _dd_row(
            "test_form_missing_reason", field_type="radio",
            choices="M1, Refusal"),
        _dd_row("test_substantive"),
    )
    info = _form_check_info(dictionary)
    info["important_form_vars"]["test_form"][
        "missing_spec_var"] = "test_form_missing_reason"

    output = proposed.ProposedChecks(
        _combined_row(
            test_form_missing=1,
            test_form_missing_reason="",
            test_substantive="present",
            test_form_complete=2,
        ),
        "baseline", "PRONET", info,
    )()

    removed_message = (
        "Form is marked missing while substantive data or Complete status are present.")
    assert all(
        removed_message not in str(finding.get("error_message", ""))
        for finding in output)
    assert not [
        finding for finding in output
        if finding.get("check_id", "").startswith("DD-MISS-")
        or finding.get("check_id") in {"FORM-002", "FORM-003", "FORM-006"}
    ]


@pytest.mark.parametrize(
    ("completion", "substantive"),
    [(2, ""), (0, "present"), (1, "present"), ("", "present")],
)
def test_completion_content_and_repeat_blank_conflicts_are_not_proposed_findings(
        completion, substantive):
    dictionary = _dictionary(_dd_row("test_substantive"))
    output = _run(dictionary, {
        "test_substantive": substantive,
        "test_form_complete": completion,
        "redcap_repeat_instrument": "test_form",
        "redcap_repeat_instance": 1,
    })

    for check_id in ("FORM-002", "FORM-003", "FORM-006"):
        assert not _findings_with_id(output, check_id)


def test_nonblank_invalid_values_still_emit_findings():
    dictionary = _dictionary(
        _dd_row(
            "test_choice", field_type="radio", choices="0, No | 1, Yes"),
        _dd_row(
            "test_integer", validation="integer", minimum="0", maximum="10"),
        _dd_row("test_date", validation="date_ymd"),
        _dd_row("test_source", validation="integer"),
        _dd_row(
            "test_calculation", field_type="calc",
            choices="[test_source] + 1"),
    )

    output = _run(dictionary, {
        "test_choice": 7,
        "test_integer": 11,
        "test_date": "2025-02-30",
        "test_source": 2,
        "test_calculation": 99,
        "test_form_complete": 7,
    })

    for check_id in (
            "DD-DOMAIN-001", "DD-NUM-003", "DD-FMT-001", "DD-CALC-001"):
        assert _findings_with_id(output, check_id), check_id

    calculation = _findings_with_id(output, "DD-CALC-001")
    assert str(calculation[0]["affected_variables"]).split(" | ") == [
        "test_calculation", "test_source"]
    assert calculation[0]["proposed_variable_values"] == (
        "test_calculation=99; test_source=2")

    # An out-of-domain completion status is no longer a proposed finding.
    assert not _findings_with_id(output, "FORM-001")


@pytest.mark.parametrize("sentinel", ["NA", "N/A", "na", "n/a"])
def test_free_text_no_data_sentinels_emit_no_participant_findings(sentinel):
    """A no-data sentinel is not a value to validate.

    ``NA`` is absent from utils.py _MISSING_CODE_LIST, so it reaches the
    generic checks as substantive text unless MISSING_TEXT covers it.
    """

    dictionary = _dictionary(
        _dd_row("test_date", validation="date_ymd"),
        _dd_row("test_choice", field_type="radio", choices="0, No | 1, Yes"),
        _dd_row("test_integer", validation="integer", minimum="0", maximum="10"),
    )

    output = _run(dictionary, {
        "test_date": sentinel,
        "test_choice": sentinel,
        "test_integer": sentinel,
    })

    for check_id in ("DD-FMT-001", "DD-DOMAIN-001", "DD-NUM-001", "DD-NUM-003"):
        assert not _findings_with_id(output, check_id), check_id


@pytest.mark.parametrize(
    "stored",
    [
        "", -3, -9, -99, 999, "NA",
        "1901-01-01", "1903-03-03", "1909-03-03", "1909-09-09",
    ],
)
def test_missing_calculation_target_does_not_emit_mismatch(stored):
    dictionary = _dictionary(
        _dd_row("test_source", validation="integer"),
        _dd_row(
            "test_calculation", field_type="calc",
            choices="[test_source] + 1"),
    )

    output = _run(dictionary, {
        "test_source": 2,
        "test_calculation": stored,
    })

    assert not _findings_with_id(output, "DD-CALC-001")


@pytest.mark.parametrize(
    "missing_date",
    [
        "", -9, -99, 999,
        "1901-01-01", "1903-03-03", "1909-03-03", "1909-09-09",
    ],
)
def test_curated_stale_detail_rule_ignores_blank_and_sentinel_values(
        missing_date):
    output = _run_on_form(
        _dictionary(
            _dd_row(
                "chrchs_sick", field_type="radio",
                choices="0, Not sick | 1, Sick"),
            _dd_row("chrchs_sickdate", validation="date_ymd"),
        ),
        {"chrchs_sick": 0, "chrchs_sickdate": missing_date},
        "current_health_status",
    )

    assert not _findings_with_id(output, "HEALTH-011")


def test_false_branch_with_stale_participant_value_is_a_branch_warning():
    dictionary = _dictionary(
        _dd_row(
            "test_gate", field_type="radio", choices="0, No | 1, Yes"),
        _dd_row("test_detail", branch="[test_gate] = '1'"),
    )

    output = _run(dictionary, {"test_gate": 0, "test_detail": "stale"})

    findings = _findings_with_id(output, "DD-BRANCH-001")
    assert len(findings) == 1
    assert findings[0]["displayed_variable"] == "test_detail"
    assert str(findings[0]["affected_variables"]).split(" | ") == [
        "test_detail", "test_gate"]
    assert findings[0]["proposed_variable_values"] == (
        "test_detail=[REDACTED]; test_gate=0")
    _assert_proposed_route(findings[0])


@pytest.mark.parametrize(
    "controller",
    ["", -9, 999, "1909-09-09"],
)
def test_missing_branch_controller_cannot_create_branch_or_scid_hidden_findings(
        controller):
    dictionary = _dictionary(
        _dd_row(
            "test_gate", field_type="radio", choices="0, No | 1, Yes"),
        _dd_row("test_detail", branch="[test_gate] = '1'"),
        _dd_row(
            "chrscid_c10", field_type="calc",
            choices="if([test_gate]=1,3,0)",
            branch="[test_gate] = '1'"),
    )

    output = _run_on_form(
        dictionary,
        {"test_gate": controller, "test_detail": "stale", "chrscid_c10": 3},
        "scid5_psychosis_mood_substance_abuse",
    )

    assert not _findings_with_id(output, "DD-BRANCH-001")
    assert not _findings_with_id(output, "SCID-DX-004")


def test_definitive_false_branch_retains_branch_and_scid_hidden_findings():
    dictionary = _dictionary(
        _dd_row(
            "test_gate", field_type="radio", choices="0, No | 1, Yes"),
        _dd_row("test_detail", branch="[test_gate] = '1'"),
        _dd_row(
            "chrscid_c10", field_type="calc",
            choices="if([test_gate]=1,3,0)",
            branch="[test_gate] = '1'"),
    )

    output = _run_on_form(
        dictionary,
        {"test_gate": 0, "test_detail": "stale", "chrscid_c10": 3},
        "scid5_psychosis_mood_substance_abuse",
    )

    assert _findings_with_id(output, "DD-BRANCH-001")
    assert _findings_with_id(output, "SCID-DX-004")


@pytest.mark.parametrize(
    "controller",
    ["", -9, 999, "1909-09-09"],
)
def test_guid_indirect_rules_ignore_unknown_controllers_and_partial_dob(
        controller):
    dictionary = _dictionary(
        _dd_row("chrguid_sex_chr", field_type="radio",
                choices="1, Female | 2, Male"),
        _dd_row("chrguid_sex_hc", field_type="radio",
                choices="1, Female | 2, Male"),
        _dd_row("chrdemo_sexassigned", field_type="radio",
                choices="1, Female | 2, Male"),
        _dd_row("chrguid_guid_pseudo", field_type="radio",
                choices="1, Real | 2, Pseudo"),
        _dd_row("chrguid_guid"),
        _dd_row("chrguid_pseudoguid"),
        _dd_row("chrguid_middlename1", field_type="yesno"),
        _dd_row("chrguid_middlename"),
        _dd_row("chrguid_mob", validation="integer"),
        _dd_row("chrguid_dob", validation="integer"),
        _dd_row("chrguid_yob", validation="integer"),
        _dd_row("chrdemo_dob", validation="date_ymd"),
    )

    output = _run_on_form(
        dictionary,
        {
            "chrguid_sex_chr": controller,
            "chrguid_sex_hc": controller,
            "chrdemo_sexassigned": 1,
            "chrguid_guid_pseudo": controller,
            "chrguid_guid": "real-guid",
            "chrguid_pseudoguid": "pseudo-guid",
            "chrguid_middlename1": controller,
            "chrguid_middlename": "Middle",
            "chrguid_mob": 2,
            "chrguid_dob": controller,
            "chrguid_yob": 2000,
            "chrdemo_dob": "2000-02-29",
        },
        "guid_form",
    )

    for check_id in ("IDENT-001", "IDENT-002", "IDENT-003", "IDENT-004", "IDENT-008"):
        assert not _findings_with_id(output, check_id)


def test_guid_indirect_rules_retain_definitive_stale_inverse_findings():
    dictionary = _dictionary(
        _dd_row("chrguid_sex_chr", field_type="radio",
                choices="1, Female | 2, Male"),
        _dd_row("chrguid_sex_hc", field_type="radio",
                choices="1, Female | 2, Male"),
        _dd_row("chrdemo_sexassigned", field_type="radio",
                choices="1, Female | 2, Male"),
        _dd_row("chrguid_guid_pseudo", field_type="radio",
                choices="1, Real | 2, Pseudo"),
        _dd_row("chrguid_guid"),
        _dd_row("chrguid_pseudoguid"),
        _dd_row("chrguid_middlename1", field_type="yesno"),
        _dd_row("chrguid_middlename"),
        _dd_row("chrguid_mob", validation="integer"),
        _dd_row("chrguid_dob", validation="integer"),
        _dd_row("chrguid_yob", validation="integer"),
        _dd_row("chrdemo_dob", validation="date_ymd"),
    )

    output = _run_on_form(
        dictionary,
        {
            "chrguid_sex_chr": 1,
            "chrguid_sex_hc": 2,
            "chrdemo_sexassigned": 2,
            "chrguid_guid_pseudo": 1,
            "chrguid_guid": "real-guid",
            "chrguid_pseudoguid": "stale-pseudo-guid",
            "chrguid_middlename1": 0,
            "chrguid_middlename": "Stale middle name",
            "chrguid_mob": 2,
            "chrguid_dob": 29,
            "chrguid_yob": 2000,
            "chrdemo_dob": "2000-02-28",
        },
        "guid_form",
    )

    for check_id in ("IDENT-001", "IDENT-002", "IDENT-003", "IDENT-004", "IDENT-008"):
        assert _findings_with_id(output, check_id)


@pytest.mark.parametrize("course_count", ["", -9, 999, "1909-09-09"])
def test_ap_course_data_does_not_contradict_an_unknown_course_count(course_count):
    dictionary = _dictionary(
        _dd_row("chrap_risp", field_type="yesno"),
        _dd_row("chrap_rispcourse", validation="integer"),
        _dd_row("chrap_rispdose1", validation="number"),
        _dd_row("chrap_rispdays1", validation="number"),
    )

    output = _run_on_form(
        dictionary,
        {
            "chrap_risp": 1,
            "chrap_rispcourse": course_count,
            "chrap_rispdose1": 2,
            "chrap_rispdays1": 10,
        },
        "lifetime_ap_exposure_screen",
    )

    assert not _findings_with_id(output, "AP-001")
    assert not _findings_with_id(output, "AP-002")


def test_ap_course_data_beyond_a_definitive_zero_count_still_flags():
    output = _run_on_form(
        _dictionary(
            _dd_row("chrap_risp", field_type="yesno"),
            _dd_row("chrap_rispcourse", validation="integer"),
            _dd_row("chrap_rispdose1", validation="number"),
            _dd_row("chrap_rispdays1", validation="number"),
        ),
        {
            "chrap_risp": 1,
            "chrap_rispcourse": 0,
            "chrap_rispdose1": 2,
            "chrap_rispdays1": 10,
        },
        "lifetime_ap_exposure_screen",
    )

    assert _findings_with_id(output, "AP-001")
    assert _findings_with_id(output, "AP-002")


@pytest.mark.parametrize("gate", ["", -9, 999, "1909-09-09"])
def test_ae_clinician_details_do_not_contradict_an_unknown_display_gate(gate):
    dictionary = _dictionary(
        _dd_row("chrae_aescreen", field_type="radio", choices="0, No | 1, Yes"),
        _dd_row("chrae_tp1", field_type="radio", choices="0, Baseline"),
        _dd_row("chrae_ae1"),
        _dd_row("chrae_aes1date", validation="date_ymd"),
        _dd_row("chrae_d1", field_type="radio", choices="1, Mild"),
        _dd_row("chrae_dr1", field_type="radio", choices="0, Hidden | 1, Display"),
        _dd_row("chr_ae1date_dr", validation="date_ymd"),
        _dd_row("chrae_sig1_dr"),
    )

    output = _run_on_form(
        dictionary,
        {
            "chrae_aescreen": 1,
            "chrae_tp1": 0,
            "chrae_ae1": "Headache",
            "chrae_aes1date": "2025-02-02",
            "chrae_d1": 1,
            "chrae_dr1": gate,
            "chr_ae1date_dr": "2025-02-03",
            "chrae_sig1_dr": "AB",
        },
        "adverse_events",
    )

    assert not _findings_with_id(output, "EP-AE-006")


def test_ae_clinician_details_with_a_definitive_hidden_gate_still_flag():
    output = _run_on_form(
        _dictionary(
            _dd_row("chrae_aescreen", field_type="radio", choices="0, No | 1, Yes"),
            _dd_row("chrae_tp1", field_type="radio", choices="0, Baseline"),
            _dd_row("chrae_ae1"),
            _dd_row("chrae_aes1date", validation="date_ymd"),
            _dd_row("chrae_d1", field_type="radio", choices="1, Mild"),
            _dd_row("chrae_dr1", field_type="radio", choices="0, Hidden | 1, Display"),
            _dd_row("chr_ae1date_dr", validation="date_ymd"),
            _dd_row("chrae_sig1_dr"),
        ),
        {
            "chrae_aescreen": 1,
            "chrae_tp1": 0,
            "chrae_ae1": "Headache",
            "chrae_aes1date": "2025-02-02",
            "chrae_d1": 1,
            "chrae_dr1": 0,
            "chr_ae1date_dr": "2025-02-03",
            "chrae_sig1_dr": "AB",
        },
        "adverse_events",
    )

    assert _findings_with_id(output, "EP-AE-006")


@pytest.mark.parametrize("source", ["", -9, 999, "1909-09-09"])
def test_gcp_significance_does_not_contradict_an_unknown_cbc_source(source):
    output = _run_on_form(
        _dictionary(
            _dd_row("chrcbc_rbcinrange", field_type="radio",
                    choices="0, In range | 1, Outside range"),
            _dd_row("chrgpc_rbcsig", field_type="radio",
                    choices="0, Not significant | 1, Significant"),
        ),
        {"chrcbc_rbcinrange": source, "chrgpc_rbcsig": 1},
        "gcp_cbc_with_differential",
    )

    assert not _findings_with_id(output, "GCP-CBC-001")


def test_gcp_significance_with_definitive_in_range_source_still_flags():
    output = _run_on_form(
        _dictionary(
            _dd_row("chrcbc_rbcinrange", field_type="radio",
                    choices="0, In range | 1, Outside range"),
            _dd_row("chrgpc_rbcsig", field_type="radio",
                    choices="0, Not significant | 1, Significant"),
        ),
        {"chrcbc_rbcinrange": 0, "chrgpc_rbcsig": 1},
        "gcp_cbc_with_differential",
    )

    assert _findings_with_id(output, "GCP-CBC-001")


@pytest.mark.parametrize("routing", ["", -9, 999, "1909-09-09"])
def test_eeg_zip_count_does_not_contradict_unknown_routing_controls(routing):
    output = _run_on_form(
        _dictionary(
            _dd_row("chreeg_zip", validation="integer"),
            _dd_row("chreeg_hub", field_type="yesno"),
            _dd_row("chreeg_drive", field_type="yesno"),
        ),
        {"chreeg_zip": 3, "chreeg_hub": routing, "chreeg_drive": routing},
        "eeg_run_sheet",
    )

    assert not _findings_with_id(output, "EEG-005")


def test_eeg_zip_count_with_definitive_negative_routing_still_flags():
    output = _run_on_form(
        _dictionary(
            _dd_row("chreeg_zip", validation="integer"),
            _dd_row("chreeg_hub", field_type="yesno"),
            _dd_row("chreeg_drive", field_type="yesno"),
        ),
        {"chreeg_zip": 3, "chreeg_hub": 0, "chreeg_drive": 0},
        "eeg_run_sheet",
    )

    assert _findings_with_id(output, "EEG-005")


@pytest.mark.parametrize("target", ["", -9, 999, "1909-09-09"])
def test_cdss_source_does_not_contradict_an_unknown_target(target):
    output = _run_on_form(
        _dictionary(
            _dd_row("chrcdss_calg1", field_type="radio",
                    choices="0, Absent | 1, Present"),
            _dd_row("chrcdss_calg6", field_type="radio",
                    choices="0, Absent | 1, Present"),
        ),
        {"chrcdss_calg1": 0, "chrcdss_calg6": target},
        "cdss",
    )

    assert not _findings_with_id(output, "CDSS-001")


def test_cdss_source_with_definitive_positive_target_still_flags():
    output = _run_on_form(
        _dictionary(
            _dd_row("chrcdss_calg1", field_type="radio",
                    choices="0, Absent | 1, Present"),
            _dd_row("chrcdss_calg6", field_type="radio",
                    choices="0, Absent | 1, Present"),
        ),
        {"chrcdss_calg1": 0, "chrcdss_calg6": 1},
        "cdss",
    )

    assert _findings_with_id(output, "CDSS-001")


def test_undated_lifetime_history_does_not_support_a_later_reversal():
    assist_form = "assist"
    scid_form = "scid5_psychosis_mood_substance_abuse"
    dictionary = _dictionary(
        *_form_rows(
            assist_form,
            _dd_row("chrassist_interview_date", validation="date_ymd"),
            _dd_row("chrassist_whoassist_use1", field_type="yesno"),
        ),
        *_form_rows(
            scid_form,
            _dd_row("chrscid_interview_date", validation="date_ymd"),
            _dd_row("chrscid_c10", field_type="radio", choices="0, No | 3, Yes"),
        ),
    )
    dictionary.loc[
        dictionary["Variable / Field Name"].eq("chrscid_c10"), "Field Label"
    ] = "Lifetime diagnosis"
    info = _form_check_info(dictionary)
    form_info = info["important_form_vars"].pop("test_form")
    info["important_form_vars"][assist_form] = dict(form_info)
    info["important_form_vars"][scid_form] = dict(form_info)
    for cohort in ("CHR", "chr"):
        info["forms_per_timepoint"][cohort] = {
            "baseline": [assist_form, scid_form],
            "month1": [assist_form, scid_form],
        }

    proposed.ProposedChecks(
        _combined_row(
            chrassist_interview_date="",
            chrassist_whoassist_use1=1,
            chrscid_interview_date="",
            chrscid_c10=3,
        ),
        "baseline", "PRONET", info,
    )()
    later = proposed.ProposedChecks(
        _combined_row(
            chrassist_interview_date="2025-01-01",
            chrassist_whoassist_use1=0,
            chrscid_interview_date="2025-01-01",
            chrscid_c10=0,
        ),
        "month1", "PRONET", info,
    )()

    assert not _findings_with_id(later, "ASSIST-LONG-001")
    assert not _findings_with_id(later, "SCID-LONG-001")

    proposed.ProposedChecks.reset_run_state()
    proposed.ProposedChecks(
        _combined_row(
            chrassist_interview_date="2024-01-01",
            chrassist_whoassist_use1=1,
            chrscid_interview_date="2024-01-01",
            chrscid_c10=3,
        ),
        "baseline", "PRONET", info,
    )()
    dated_later = proposed.ProposedChecks(
        _combined_row(
            chrassist_interview_date="2025-01-01",
            chrassist_whoassist_use1=0,
            chrscid_interview_date="2025-01-01",
            chrscid_c10=0,
        ),
        "month1", "PRONET", info,
    )()

    assert _findings_with_id(dated_later, "ASSIST-LONG-001")
    assert _findings_with_id(dated_later, "SCID-LONG-001")


def test_expanded_checkbox_missing_choice_is_not_substantive_form_data():
    form = "digital_biomarkers_axivity_checkin"
    dictionary = _dictionary(*_form_rows(
        form,
        _dd_row(
            "chraxci_qaqc1", field_type="checkbox",
            choices="-99, Missing | 1, Yes | 2, No | 3, N/A"),
    ))
    info = _form_check_info(dictionary)
    form_info = info["important_form_vars"].pop("test_form")
    info["important_form_vars"][form] = form_info
    for cohort in ("CHR", "chr"):
        info["forms_per_timepoint"][cohort]["baseline"] = []

    missing_checker = proposed.ProposedChecks(
        _combined_row(**{
            "chraxci_qaqc1___-99": 1,
            "chraxci_qaqc1___1": 0,
            "chraxci_qaqc1___2": 0,
            "chraxci_qaqc1___3": 0,
        }),
        "baseline", "PRONET", info,
    )
    missing_output = missing_checker()

    assert form not in missing_checker._active_forms
    assert not any(
        finding.get("check_id", "").startswith(("DD-DOMAIN", "DD-BRANCH"))
        for finding in missing_output
    )
    assert not _findings_with_id(missing_output, "AXIVITY-005")

    substantive_checker = proposed.ProposedChecks(
        _combined_row(**{
            "chraxci_qaqc1___-99": 0,
            "chraxci_qaqc1___1": 1,
            "chraxci_qaqc1___2": 1,
            "chraxci_qaqc1___3": 0,
        }),
        "baseline", "PRONET", info,
    )

    assert form in substantive_checker._active_forms
    assert _findings_with_id(substantive_checker(), "AXIVITY-005")


@pytest.mark.parametrize(
    ("source", "stored", "expected_check_id"),
    [
        (3, 0, "SCID-DX-001"),  # criteria met, endpoint not positive
        (0, 3, "SCID-DX-002"),  # endpoint positive, criteria not met
        ("", 3, None),
        (-9, 3, None),
        (3, "", None),
        (3, -9, None),
        (3, 999, None),
    ],
)
def test_scid_formula_engine_checks_only_definitive_diagnosis_directions(
        source, stored, expected_check_id):
    scid_form = "scid5_psychosis_mood_substance_abuse"
    dictionary = _dictionary(
        _dd_row("chrscid_source", field_type="radio", choices="0, No | 3, Yes"),
        _dd_row(
            "chrscid_c10", field_type="calc",
            choices="if([chrscid_source]=3,3,0)"),
    )
    dictionary["Form Name"] = scid_form
    info = _form_check_info(dictionary)
    form_info = info["important_form_vars"].pop("test_form")
    info["important_form_vars"][scid_form] = form_info
    for cohort in ("CHR", "chr"):
        info["forms_per_timepoint"][cohort]["baseline"] = [scid_form]

    output = proposed.ProposedChecks(
        _combined_row(chrscid_source=source, chrscid_c10=stored),
        "baseline", "PRONET", info)()

    if expected_check_id is None:
        assert not any(
            _findings_with_id(output, check_id)
            for check_id in ("SCID-DX-001", "SCID-DX-002", "SCID-DX-003", "SCID-DX-005")
        )
    else:
        findings = _findings_with_id(output, expected_check_id)
        assert len(findings) == 1
        _assert_proposed_route(findings[0])


def test_structured_manual_scid_chain_skips_a_blank_target():
    output = _run_on_form(
        _dictionary(
            _dd_row("chrscid_d2", field_type="radio", choices="1, No | 3, Yes"),
            _dd_row("chrscid_d3", field_type="radio", choices="1, No | 3, Yes"),
            _dd_row("chrscid_d4", field_type="radio", choices="1, Manic | 2, Hypomanic"),
        ),
        {"chrscid_d2": 3, "chrscid_d3": 3, "chrscid_d4": ""},
        "scid5_psychosis_mood_substance_abuse")

    assert not any(
        _findings_with_id(output, check_id)
        for check_id in ("SCID-DX-001", "SCID-DX-002", "SCID-DX-003")
    )


def test_structured_manual_scid_chain_retains_a_definitive_contradiction():
    output = _run_on_form(
        _dictionary(
            _dd_row("chrscid_d2", field_type="radio", choices="1, No | 3, Yes"),
            _dd_row("chrscid_d3", field_type="radio", choices="1, No | 3, Yes"),
            _dd_row("chrscid_d4", field_type="radio", choices="1, Manic | 2, Hypomanic"),
        ),
        {"chrscid_d2": 1, "chrscid_d3": 3, "chrscid_d4": 1},
        "scid5_psychosis_mood_substance_abuse")

    findings = _findings_with_id(output, "SCID-DX-002")
    assert len(findings) == 1
    _assert_proposed_route(findings[0])


def test_scid_onset_and_year_last_met_use_scid_date_and_governed_dob():
    output = _run_on_form(
        _dictionary(
            _dd_row("chrscid_interview_date", validation="date_ymd"),
            _dd_row("chrdemo_dob", validation="date_ymd"),
            _dd_row("chrscid_a52", validation="integer"),
            _dd_row("chrscid_e164_302", validation="integer"),
            _dd_row("chrscid_e300", validation="integer"),
        ),
        {
            "chrscid_interview_date": "2020-06-01",
            "chrdemo_dob": "2000-06-02",  # age 19 at the SCID interview
            "chrscid_a52": 20,
            "chrscid_e164_302": 15,
            "chrscid_e300": 2010,
        },
        "scid5_psychosis_mood_substance_abuse")

    assert _findings_with_id(output, "SCID-AGE-001")
    assert _findings_with_id(output, "SCID-YEAR-002")


def test_schizotypal_calculation_mismatch_has_a_dedicated_recomputation_id():
    output = _run_on_form(
        _dictionary(
            _dd_row("chrschizotypal_par00a1", field_type="radio",
                    choices="0, Absent | 1, Subthreshold | 2, Threshold"),
            _dd_row(
                "chrschizotypal_szt22a5", field_type="calc",
                choices="if([chrschizotypal_par00a1]=2,2,0)"),
            _dd_row(
                "chrschizotypal_threshold_2", field_type="calc",
                choices="if([chrschizotypal_szt22a5]=2,1,0)"),
            _dd_row(
                "chrschizotypal_summ", field_type="calc",
                choices="if([chrschizotypal_threshold_2]>4,1,0)"),
        ),
        {
            "chrschizotypal_par00a1": 2,
            "chrschizotypal_szt22a5": 0,
            "chrschizotypal_threshold_2": 0,
            "chrschizotypal_summ": 0,
        },
        "scid5_schizotypal_personality_sciddpq")

    assert _findings_with_id(output, "SCID-SZT-001")


def test_cssrs_unknown_item_or_count_is_not_coerced_to_no():
    baseline_items = [
        _dd_row(f"chrcssrsb_si{number}", field_type="radio",
                choices="1, Yes | 2, No")
        for number in range(1, 6)
    ]
    output = _run_on_form(
        _dictionary(
            *baseline_items,
            _dd_row("chrcssrsb_idintsvl", field_type="radio",
                    choices="1, One | 2, Two | 3, Three | 4, Four | 5, Five"),
            _dd_row("chrcssrsb_cssrs_actual", field_type="radio",
                    choices="0, No | 1, Yes"),
            _dd_row("chrcssrsb_cssrs_num_attempt", validation="integer"),
        ),
        {
            "chrcssrsb_si1": 1,
            "chrcssrsb_si2": 2,
            "chrcssrsb_si3": 2,
            "chrcssrsb_si4": 2,
            "chrcssrsb_si5": -9,
            "chrcssrsb_idintsvl": 1,
            "chrcssrsb_cssrs_actual": 1,
            "chrcssrsb_cssrs_num_attempt": -9,
        },
        "cssrs_baseline")

    assert not _findings_with_id(output, "CSSRS-001")
    assert not _findings_with_id(output, "CSSRS-003")


def test_cssrs_blank_target_or_count_does_not_create_a_finding():
    baseline_items = [
        _dd_row(f"chrcssrsb_si{number}", field_type="radio",
                choices="1, Yes | 2, No")
        for number in range(1, 6)
    ]
    output = _run_on_form(
        _dictionary(
            *baseline_items,
            _dd_row("chrcssrsb_idintsvl", field_type="radio",
                    choices="1, One | 2, Two | 3, Three | 4, Four | 5, Five"),
            _dd_row("chrcssrsb_cssrs_actual", field_type="radio",
                    choices="0, No | 1, Yes"),
            _dd_row("chrcssrsb_cssrs_num_attempt", validation="integer"),
        ),
        {
            "chrcssrsb_si1": 1,
            "chrcssrsb_si2": 2,
            "chrcssrsb_si3": 2,
            "chrcssrsb_si4": 2,
            "chrcssrsb_si5": 2,
            "chrcssrsb_idintsvl": "",
            "chrcssrsb_cssrs_actual": 1,
            "chrcssrsb_cssrs_num_attempt": "",
        },
        "cssrs_baseline")

    assert not _findings_with_id(output, "CSSRS-001")
    assert not _findings_with_id(output, "CSSRS-003")


def test_cssrs_definitive_maximum_and_count_contradictions_are_retained():
    baseline_items = [
        _dd_row(f"chrcssrsb_si{number}", field_type="radio",
                choices="1, Yes | 2, No")
        for number in range(1, 6)
    ]
    output = _run_on_form(
        _dictionary(
            *baseline_items,
            _dd_row("chrcssrsb_idintsvl", field_type="radio",
                    choices="1, One | 2, Two | 3, Three | 4, Four | 5, Five"),
            _dd_row("chrcssrsb_cssrs_actual", field_type="radio",
                    choices="0, No | 1, Yes"),
            _dd_row("chrcssrsb_cssrs_num_attempt", validation="integer"),
        ),
        {
            "chrcssrsb_si1": 1,
            "chrcssrsb_si2": 2,
            "chrcssrsb_si3": 2,
            "chrcssrsb_si4": 2,
            "chrcssrsb_si5": 2,
            "chrcssrsb_idintsvl": 2,
            "chrcssrsb_cssrs_actual": 1,
            "chrcssrsb_cssrs_num_attempt": 0,
        },
        "cssrs_baseline")

    assert _findings_with_id(output, "CSSRS-001")
    assert _findings_with_id(output, "CSSRS-003")


def test_psychs_missing_sentinel_and_followup_interview_mapping_are_safe():
    output = _run_on_form(
        _dictionary(
            _dd_row("chrpsychs_p1a0", validation="integer"),
            _dd_row("chrpsychs_p1b0", validation="integer"),
            _dd_row("chrpsychs_p1d0", validation="integer"),
            _dd_row("chrpsychs_fu_p1_on", validation="date_ymd"),
            _dd_row("chrpsychs_scr_interview_date", validation="date_ymd"),
            _dd_row("chrpsychs_fu_interview_date", validation="date_ymd"),
        ),
        {
            "chrpsychs_p1a0": -9,
            "chrpsychs_p1b0": 1,
            "chrpsychs_p1d0": 2,
            "chrpsychs_fu_p1_on": "2024-02-01",
            "chrpsychs_scr_interview_date": "2025-01-01",
            "chrpsychs_fu_interview_date": "2024-01-01",
        },
        "psychs_p1p8_fu")

    assert not _findings_with_id(output, "PSYCHS-001")
    assert _findings_with_id(output, "PSYCHS-003")


def test_reviewed_psychs_alert_requires_valid_visible_operands():
    a1 = _dd_row(
        "chrpsychs_scr_1a1", field_type="radio",
        choices="0, Absent | 1, One | 2, Two | 3, Three | 4, Four | 5, Five | 6, Six")
    a2 = _dd_row(
        "chrpsychs_scr_1a2", field_type="radio",
        choices="0, Zero | 1, One | 2, Two | 3, Three | 4, Four | 5, Five | 6, Six")
    a3 = _dd_row(
        "chrpsychs_scr_1a3", field_type="yesno",
        branch="[chrpsychs_scr_1a1]=6")
    error = _dd_row(
        "chrpsychs_scr_1a3_err", field_type="descriptive",
        branch="[chrpsychs_scr_1a2]<4 and [chrpsychs_scr_1a3]=1")
    error["Field Label"] = "<div>This rating conflicts with A.2.</div>"
    dictionary = _dictionary(a1, a2, a3, error)

    definitive = _run_on_form(
        dictionary,
        {"chrpsychs_scr_1a1": 6, "chrpsychs_scr_1a2": 3,
         "chrpsychs_scr_1a3": 1},
        "psychs_p1p8")
    hidden_source = _run_on_form(
        dictionary,
        {"chrpsychs_scr_1a1": 5, "chrpsychs_scr_1a2": 3,
         "chrpsychs_scr_1a3": 1},
        "psychs_p1p8")
    sentinel_source = _run_on_form(
        dictionary,
        {"chrpsychs_scr_1a1": 6, "chrpsychs_scr_1a2": -9,
         "chrpsychs_scr_1a3": 1},
        "psychs_p1p8")

    assert len(_findings_with_id(definitive, "PSYCHS-005")) == 1
    assert not _findings_with_id(hidden_source, "PSYCHS-005")
    assert not _findings_with_id(sentinel_source, "PSYCHS-005")


def test_reviewed_psychs_datediff_alert_replays_exact_dictionary_logic():
    date_field = _dd_row(
        "chrpsychs_scr_1d8_date", validation="date_ymd",
        branch="[chrpsychs_scr_1d8]=1")
    error = _dd_row(
        "chrpsychs_scr_1a8_err3", field_type="descriptive",
        branch=("datediff([chrpsychs_scr_interview_date], "
                "[chrpsychs_scr_1d8_date], 'd', 'ymd')>90"))
    error["Field Label"] = "This date conflicts with the 90-day rating."
    dictionary = _dictionary(
        _dd_row("chrpsychs_scr_interview_date", validation="date_ymd"),
        _dd_row("chrpsychs_scr_1d8", field_type="yesno"),
        date_field, error)

    output = _run_on_form(
        dictionary,
        {"chrpsychs_scr_interview_date": "2025-05-01",
         "chrpsychs_scr_1d8": 1,
         "chrpsychs_scr_1d8_date": "2025-01-01"},
        "psychs_p1p8")

    assert _findings_with_id(output, "PSYCHS-006")


def test_scid_binary_subtype_rules_require_complete_substantive_family():
    dictionary = _dictionary(
        _dd_row("chrscid_a54", field_type="radio",
                choices="1, Absent | 2, Subthreshold | 3, Threshold | 4, Unknown"),
        _dd_row("chrscid_a55", field_type="radio", choices="0, Absent | 1, Present",
                branch="[chrscid_a54]>1"),
        _dd_row("chrscid_a56", field_type="radio", choices="0, Absent | 1, Present",
                branch="[chrscid_a54]>1"),
    )
    both_absent = _run_on_form(
        dictionary, {"chrscid_a54": 3, "chrscid_a55": 0, "chrscid_a56": 0},
        "scid5_psychosis_mood_substance_abuse")
    both_present = _run_on_form(
        dictionary, {"chrscid_a54": 3, "chrscid_a55": 1, "chrscid_a56": 1},
        "scid5_psychosis_mood_substance_abuse")
    incomplete = _run_on_form(
        dictionary, {"chrscid_a54": 3, "chrscid_a55": 0, "chrscid_a56": -9},
        "scid5_psychosis_mood_substance_abuse")

    assert _findings_with_id(both_absent, "SCID-SYM-001")
    assert _findings_with_id(both_present, "SCID-SYM-002")
    assert not _findings_with_id(incomplete, "SCID-SYM-001")
    assert not _findings_with_id(incomplete, "SCID-SYM-002")


def test_bprs_new_threshold_gap_is_exact_and_sentinel_safe():
    dictionary = _dictionary(
        _dd_row("chrbprs_bprs_susp", field_type="radio",
                choices="-9, Not assessed | 1, One | 2, Two | 3, Three | 4, Four | 5, Five | 6, Six | 7, Seven"),
        _dd_row("chrbprs_bprs_unus", field_type="radio",
                choices="-9, Not assessed | 1, One | 2, Two | 3, Three | 4, Four | 5, Five | 6, Six | 7, Seven"),
    )

    new_gap = _run_on_form(
        dictionary, {"chrbprs_bprs_susp": 4, "chrbprs_bprs_unus": 1}, "bprs")
    known_consistent = _run_on_form(
        dictionary, {"chrbprs_bprs_susp": 4, "chrbprs_bprs_unus": 2}, "bprs")
    unknown = _run_on_form(
        dictionary, {"chrbprs_bprs_susp": 6, "chrbprs_bprs_unus": -9}, "bprs")

    assert _findings_with_id(new_gap, "BPRS-002")
    assert not _findings_with_id(known_consistent, "BPRS-002")
    assert not _findings_with_id(unknown, "BPRS-002")


def test_cbc_and_blood_dates_compare_calendar_day_not_raw_datetime_text():
    dictionary = _dictionary(
        _dd_row("chrcbc_interview_date", validation="date_ymd"),
        _dd_row("chrblood_interview_date", validation="date_ymd"),
        _dd_row("chrblood_drawdate", validation="datetime_ymd"),
    )
    same_day = _run_on_form(
        dictionary,
        {"chrcbc_interview_date": "2025-03-01",
         "chrblood_interview_date": "2025-03-01",
         "chrblood_drawdate": "2025-03-01 14:30"},
        "blood_sample_preanalytic_quality_assurance")
    next_day = _run_on_form(
        dictionary,
        {"chrcbc_interview_date": "2025-03-02",
         "chrblood_interview_date": "2025-03-02",
         "chrblood_drawdate": "2025-03-01 14:30"},
        "blood_sample_preanalytic_quality_assurance")

    assert not _findings_with_id(same_day, "CBC-006")
    assert not _findings_with_id(same_day, "BLOOD-012")
    assert _findings_with_id(next_day, "CBC-006")
    assert _findings_with_id(next_day, "BLOOD-012")


def test_psychs_past_reference_uses_only_earlier_verified_calculation():
    screen_form = "psychs_p1p8"
    follow_form = "psychs_p1p8_fu"
    dictionary = _dictionary(
        *_form_rows(
            screen_form,
            _dd_row("chrpsychs_scr_interview_date", validation="date_ymd"),
            _dd_row("chrpsychs_scr_1source", field_type="yesno"),
            _dd_row("chrpsychs_scr_1a6", field_type="calc",
                    choices="if([chrpsychs_scr_1source]=1,1,0)"),
        ),
        *_form_rows(
            follow_form,
            _dd_row("chrpsychs_fu_interview_date", validation="date_ymd"),
            _dd_row("chrpsychs_fu_1c6_prev", field_type="radio",
                    choices="0, No | 1, Yes"),
        ),
    )
    info = _form_check_info(dictionary)
    form_info = info["important_form_vars"].pop("test_form")
    info["important_form_vars"][screen_form] = dict(form_info)
    info["important_form_vars"][follow_form] = dict(form_info)
    for cohort in ("CHR", "chr"):
        info["forms_per_timepoint"][cohort] = {
            "baseline": [screen_form], "month1": [follow_form]}

    proposed.ProposedChecks.reset_run_state()
    proposed.ProposedChecks(
        _combined_row(
            chrpsychs_scr_interview_date="2025-01-01",
            chrpsychs_scr_1source=1, chrpsychs_scr_1a6=1,
            chrpsychs_fu_interview_date="", chrpsychs_fu_1c6_prev=""),
        "baseline", "PRONET", info)()
    later = proposed.ProposedChecks(
        _combined_row(
            chrpsychs_scr_interview_date="", chrpsychs_scr_1source="",
            chrpsychs_scr_1a6="", chrpsychs_fu_interview_date="2025-02-01",
            chrpsychs_fu_1c6_prev=0),
        "month1", "PRONET", info)()

    assert _findings_with_id(later, "PSYCHS-LONG-001")

    proposed.ProposedChecks.reset_run_state()
    proposed.ProposedChecks(
        _combined_row(
            chrpsychs_scr_interview_date="2025-01-01",
            chrpsychs_scr_1source=1, chrpsychs_scr_1a6=0,
            chrpsychs_fu_interview_date="", chrpsychs_fu_1c6_prev=""),
        "baseline", "PRONET", info)()
    unverified_history = proposed.ProposedChecks(
        _combined_row(
            chrpsychs_scr_interview_date="", chrpsychs_scr_1source="",
            chrpsychs_scr_1a6="", chrpsychs_fu_interview_date="2025-02-01",
            chrpsychs_fu_1c6_prev=0),
        "month1", "PRONET", info)()

    assert not _findings_with_id(unverified_history, "PSYCHS-LONG-001")
    proposed.ProposedChecks.reset_run_state()


def test_curated_predicate_uses_the_same_output_and_routing_contract():
    dictionary = _dictionary(
        _dd_row("left_answer", field_type="yesno"),
        _dd_row("right_answer", field_type="yesno"),
    )
    curated_rules = [{
        "check_id": "CURATED-TEST-001",
        "form": "test_form",
        "variables": ["left_answer", "right_answer"],
        "predicate": (
            lambda row: row.left_answer == 1 and row.right_answer == 0),
        "message": "Left Yes requires Right Yes.",
    }]

    output = _run(
        dictionary,
        {"left_answer": 1, "right_answer": 0},
        curated_rules,
    )

    findings = _findings_with_id(output, "CURATED-TEST-001")
    assert len(findings) == 1
    assert str(findings[0]["affected_variables"]).split(" | ") == [
        "left_answer", "right_answer"]
    assert findings[0]["proposed_variable_values"] == (
        "left_answer=1; right_answer=0")
    _assert_proposed_route(findings[0])


def test_evidence_redacts_identifiers_and_unrestricted_text():
    dictionary = _dictionary(
        _dd_row(
            "secret_identifier", field_type="yesno", identifier="y"),
        _dd_row("clinical_note"),
        _dd_row(
            "safe_code", field_type="radio", choices="0, No | 1, Yes"),
    )
    curated_rules = [{
        "check_id": "CURATED-PRIVACY-001",
        "form": "test_form",
        "variables": ["secret_identifier", "clinical_note", "safe_code"],
        "predicate": lambda _row: True,
        "message": "Structured fields require review.",
    }]

    output = _run(
        dictionary,
        {
            "secret_identifier": 1,
            "clinical_note": "Participant name | private narrative",
            "safe_code": 1,
        },
        curated_rules,
    )

    findings = _findings_with_id(output, "CURATED-PRIVACY-001")
    assert len(findings) == 1
    assert findings[0]["proposed_variable_values"] == (
        "secret_identifier=[REDACTED]; clinical_note=[REDACTED]; "
        "safe_code=1")
    assert "Participant name" not in findings[0]["proposed_variable_values"]


def test_evidence_quotes_literal_reserved_tokens():
    dictionary = _dictionary(
        _dd_row("safe_value", validation="integer"),
    )
    curated_rules = [{
        "check_id": "CURATED-RESERVED-001",
        "form": "test_form",
        "variables": ["safe_value"],
        "predicate": lambda _row: True,
        "message": "Structured field requires review.",
    }]

    output = _run(
        dictionary,
        {"safe_value": "<blank>"},
        curated_rules,
    )

    findings = _findings_with_id(output, "CURATED-RESERVED-001")
    assert len(findings) == 1
    assert findings[0]["proposed_variable_values"] == (
        'safe_value=@json:"<blank>"')


def test_configuration_evidence_marks_an_unavailable_variable():
    output = _run(
        _dictionary(_dd_row("BadName", field_type="yesno")),
        {},
    )

    findings = [
        finding for finding in _findings_with_id(output, "DD-SCHEMA-004")
        if finding.get("displayed_variable") == "BadName"
    ]
    assert len(findings) == 1
    assert findings[0]["proposed_variable_values"] == "BadName=<unavailable>"


def test_clean_values_do_not_emit_representative_generic_or_curated_rules():
    dictionary = _dictionary(
        _dd_row(
            "test_choice", field_type="radio", choices="0, No | 1, Yes"),
        _dd_row(
            "test_integer", validation="integer", minimum="0", maximum="10"),
        _dd_row("test_date", validation="date_ymd"),
        _dd_row("test_required", required="y"),
        _dd_row("test_gate", field_type="yesno"),
        _dd_row("test_detail", branch="[test_gate] = '1'"),
    )

    output = _run(dictionary, {
        "test_choice": "1.0",
        "test_integer": "10.0",
        "test_date": "2025-02-28",
        "test_required": "present",
        "test_gate": 0,
        "test_detail": "",
    })

    assert not any(finding.get("check_id", "").startswith(
        ("DD-DOMAIN-", "DD-NUM-", "DD-FMT-", "DD-REQ-", "DD-BRANCH-"))
        for finding in output)


def test_unselected_checkbox_zeroes_do_not_count_as_substantive_form_data():
    """REDCap exports every unchecked checkbox as 0, not as a blank cell."""
    dictionary = _dictionary(_dd_row(
        "test_checkbox", field_type="checkbox",
        choices="1, First | 2, Second"))

    output = _run(dictionary, {
        "test_checkbox___1": 0,
        "test_checkbox___2": "0",
        "test_form_complete": 0,
    })

    assert not _findings_with_id(output, "FORM-003")
    assert not _findings_with_id(output, "DD-MISS-005")


def test_exported_identifier_marked_response_counts_as_substantive_data():
    dictionary = _dictionary(_dd_row(
        "test_clinical_response", field_type="yesno", identifier="y"))

    output = _run(dictionary, {
        "test_clinical_response": 1,
        "test_form_complete": 2,
    })

    assert not _findings_with_id(output, "FORM-002")


def test_participant_specific_subject_info_age_drives_age_bound_checks():
    dictionary = _dictionary(
        _dd_row("chrpds_pds_f5b_p", field_type="radio",
                choices="1, No | 4, Yes | 999, Unknown"),
        _dd_row("chrpds_pds_f6_p", validation="number"),
    )
    info = _form_check_info(dictionary)
    info["subject_info"]["S1"]["age"] = 15

    output = proposed.ProposedChecks(
        _combined_row(chrpds_pds_f5b_p=4, chrpds_pds_f6_p=20),
        "baseline", "PRONET", info)()

    assert len(_findings_with_id(output, "PDS-002")) == 1


def test_participant_specific_subject_info_age_drives_scid_onset_bound():
    dictionary = _dictionary(_dd_row(
        "chrscid_a52", validation="number", annotation="99 = Unknown"))
    info = _form_check_info(dictionary)
    info["subject_info"]["S1"]["age"] = 15

    output = proposed.ProposedChecks(
        _combined_row(chrscid_a52=20), "baseline", "PRONET", info)()

    assert len(_findings_with_id(output, "SCID-AGE-001")) == 1


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


def test_proposed_tracker_flags_include_aligned_variable_values():
    trackers = object.__new__(CreateTrackers)
    trackers.formatted_column_names = {
        "PRESCIENT": {"combined": _column_mapping()},
    }
    common = {
        "subject": "ME001",
        "cohort": "CHR",
        "displayed_timepoint": "baseline",
        "displayed_form": "test_form",
        "currently_resolved": False,
        "manually_resolved": "",
        "var_translations": "",
        "time_since_last_detection": 0,
        "dates_resolved": "",
        "comments": "",
        "site_comments": "",
        "priority_item": False,
        "reports": REPORT_NAME,
    }
    canonical_messages = [
        "field_a : [RULE-A] First relationship failed.",
        "field_b : [RULE-B] Second relationship failed.",
    ]
    raw = pd.DataFrame([
        {
            **common,
            "error_message": canonical_messages[0],
            "proposed_variable_values": "field_a=7; controller=0",
        },
        {
            **common,
            "error_message": canonical_messages[1],
            "proposed_variable_values": "field_b=3; source=9",
        },
    ])

    result = trackers.convert_to_shared_format(raw, "PRESCIENT")

    assert len(result) == 1
    assert result.loc[0, "Flag Count"] == 2
    assert result.loc[0, "Flags"] == (
        canonical_messages[0]
        + "\nVariables & values v3: field_a=7; controller=0 | "
        + canonical_messages[1]
        + "\nVariables & values v3: field_b=3; source=9")
    assert raw["error_message"].tolist() == canonical_messages


def _workbook_bytes(sheets):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return buffer.getvalue()








def test_proposed_check_reviewer_state_uses_safe_form_and_message_identity():
    calculator = object.__new__(CalculateResolvedErrors)
    calculator.utils = SimpleNamespace(
        reverse_dictionary=lambda mapping: {
            value: key for key, value in mapping.items()})
    calculator._corrupt_workbooks = []

    common = {
        "network": "PRONET",
        "subject": "KC001",
        "displayed_timepoint": "baseline",
        "displayed_form": "test_form",
        # Deliberately identical family ID.  Proposed dictionary rules can
        # instantiate once for each of thousands of fields, so this ID alone
        # must never be used as the reviewer-state merge key.
        "check_id": "DD-NUM-001",
        "reports": REPORT_NAME,
        "manually_resolved": "",
        "comments": "",
    }
    current = pd.DataFrame([
        {
            **common,
            "displayed_variable": "field_a",
            "error_message": (
                "field_a : [DD-NUM-001] Numeric field contains malformed value."),
        },
        {
            **common,
            "displayed_variable": "field_b",
            "error_message": (
                "field_b : [DD-NUM-001] Numeric field contains malformed value."),
        },
    ])
    sheet = pd.DataFrame([{
        "Participant": "KC001",
        "Timepoint": "baseline",
        "Form": "test_form",
        "Flags": (
            "field_a : [DD-NUM-001] Numeric field contains malformed value."
            "\nVariables & values: field_a=not-a-number"),
        "Manually Resolved": "yes",
        "Network Comments": "Verified field A against source",
    }])

    result = calculator.read_dropbox_data(
        current,
        _column_mapping(),
        ["manually_resolved", "comments"],
        "unused/PRONET/combined/PRONET_Output.xlsx",
        None,
        "PRONET",
        [REPORT_NAME],
        preloaded_data=_workbook_bytes({REPORT_NAME: sheet}),
    ).set_index("displayed_variable")

    assert result.loc["field_a", "manually_resolved"] == "yes"
    assert result.loc["field_a", "comments"] == (
        "Verified field A against source")
    assert result.loc["field_b", "manually_resolved"] == ""
    assert result.loc["field_b", "comments"] == ""


def _main_tree():
    return ast.parse((ROOT / "main" / "qc_forms" / "qc_forms_main.py").read_text(encoding="utf-8"))


def _iterate_combined_dfs():
    for node in ast.walk(_main_tree()):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "iterate_combined_dfs"):
            return node
    raise AssertionError("QCFormsMain.iterate_combined_dfs was not found")


def test_proposed_checks_is_imported_from_qc_types():
    imports = [node for node in ast.walk(_main_tree())
               if isinstance(node, ast.ImportFrom)]
    assert any(
        node.module == "qc_forms.qc_types.proposed_checks"
        and any(alias.name == "ProposedChecks" for alias in node.names)
        for node in imports
    )


def test_proposed_checks_defines_every_private_method_it_invokes():
    """Keep a very large check catalog free of late AttributeError failures."""
    tree = ast.parse(
        (ROOT / "main" / "qc_forms" / "qc_types" / "proposed_checks.py").read_text(
            encoding="utf-8"))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ProposedChecks")
    defined = {
        node.name for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    invoked = {
        node.func.attr
        for node in ast.walk(class_node)
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr.startswith("_"))
    }

    assert invoked <= defined, (
        "Undefined ProposedChecks methods: "
        + ", ".join(sorted(invoked - defined)))


def test_proposed_checks_runs_once_per_network_timepoint_before_row_loop():
    function = _iterate_combined_dfs()
    resets = [
        node for node in ast.walk(function)
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "reset_run_state"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ProposedChecks")
    ]
    assert len(resets) == 1
    constructor_calls = [
        node for node in ast.walk(function)
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ProposedChecks")
    ]
    assert len(constructor_calls) == 1
    constructor = constructor_calls[0]
    assert len(constructor.args) == 4
    assert (isinstance(constructor.args[0], ast.Name)
            and constructor.args[0].id == "combined_df")
    assert resets[0].lineno < constructor.lineno

    row_loops = [
        node for node in ast.walk(function)
        if (isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "row")
    ]
    assert len(row_loops) == 1
    row_loop = row_loops[0]
    assert constructor.lineno < row_loop.lineno
    assert not row_loop.lineno <= constructor.lineno <= row_loop.end_lineno

    extensions = [
        node for node in ast.walk(function)
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "extend"
            and node.args
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Name)
            and node.args[0].func.id == "proposed_checks")
    ]
    assert len(extensions) == 1
    assert constructor.lineno < extensions[0].lineno < row_loop.lineno


@pytest.mark.parametrize("prefix, form", [
    ("chrpsychs_scr_", "psychs_p9ac32"),
    ("chrpsychs_fu_", "psychs_p9ac32_fu"),
    ("hcpsychs_fu_", "psychs_p9ac32_fu_hc"),
])
@pytest.mark.parametrize("stop, flagged", [
    ("2025-01-01T00:30:00+14:00", False),
    ("2025-01-01T23:30:00-12:00", False),
    ("2024-12-31T12:00:00Z", True),
    ("2024-01-01T12:00:00Z", True),
    ("1909-09-09T12:00:00Z", False),
    ("bad date", False),
])
def test_psychs_global_reduction_handles_timezone_offsets(prefix, form, stop, flagged):
    # Match the deployed dictionary: start has date_ymd validation, stop has none.
    controller, start_date, stop_date = (
        prefix + "e3", prefix + "e3_start_date", prefix + "e3_stop_date")
    dictionary = _dictionary(
        _dd_row(controller, field_type="yesno"),
        _dd_row(start_date, validation="date_ymd"),
        _dd_row(stop_date),
    )
    output = _run_on_form(dictionary, {
        controller: 1, start_date: "2024-01-02", stop_date: stop,
    }, form)

    assert bool(_findings_with_id(output, "PSYCHS-007")) is flagged
