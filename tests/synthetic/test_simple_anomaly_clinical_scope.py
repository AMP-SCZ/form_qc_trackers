"""Regression tests for the anomaly report's clinical-only contract."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from simple_anomaly_detection.clinical_variables import (
    ClinicalVariableContractError,
    filter_findings_to_clinical,
    load_clinical_variable_catalog,
    scope_slices_to_clinical,
)


CLINICAL = "chrbprs_bprs_depr"
CLINICAL_2 = "chrbprs_bprs_anxi"
CLINICAL_CHECKBOX = f"{CLINICAL}___1"
INTERVIEW_DATE = "chrbprs_interview_date"
RATER_ID = "chrbprs_redcap_user"
NONCLINICAL = "chreeg_run1"


@pytest.fixture(scope="module")
def catalog():
    return load_clinical_variable_catalog()


def _finding(variable: str, anomaly_type: str = "test", **extra) -> dict:
    row = {
        "anomaly_type": anomaly_type,
        "severity_score": 80,
        "raw_score": 8,
        "network": "PRONET",
        "timepoint": "baseline",
        "site_id": "AA",
        "subjectid": "AA001",
        "variable": variable,
        "method": "test",
    }
    row.update(extra)
    return row


def test_catalog_uses_clinical_domain_and_handles_checkbox_exports(catalog):
    assert len(catalog.clinical_forms) == 34
    assert catalog.is_clinical(CLINICAL)
    assert catalog.is_clinical(CLINICAL_CHECKBOX)
    # These are genuine clinical responses/derived ratings even though their
    # labels contain routing or automatic-calculation language.
    for variable in ("chrschizotypal_szt22a5", "chrscid_a172",
                     "chrscid_b23", "chrscid_c1",
                     "chrdemo_parent_edu_other",
                     "chrschizotypal_other_dx"):
        assert catalog.is_clinical(variable), variable
    assert catalog.is_auxiliary(INTERVIEW_DATE)
    assert catalog.is_auxiliary(RATER_ID)
    assert not catalog.is_clinical(INTERVIEW_DATE)
    assert not catalog.is_clinical("chrbprs_dateerror1")
    for variable in ("chrpsychs_scr_1a0_error", "chrscid_lifechart_check",
                     "chrpsychs_scr_overview", "chrgfs_prompt1",
                     "chrpsychs_scr_pro_1b19", "chrgfr_scoreerror",
                     "chrgfs_currscorehigh", "chrsofas_currscorelow",
                     "chrscid_no_drug_12month", "chrscid_c29",
                     "chrscid_substance_overview_2", "chrgfr_prompt2a",
                     "chrbprs_bprs_anxi_p", "chrcdss_cdssinterviewer",
                     "chrnsipr_encourage", "chrnsipr_item7_desc",
                     "chrnsipr_anhed_aff_p1", "chrnsipr_nsipr1_pr1",
                     "chrscid_psychotic_desc", "chrscid_c58_descrip"):
        assert not catalog.is_clinical(variable), variable
    # Pubertal Development Scale uses *_p for real response fields, so prompt
    # exclusions stay instrument-specific rather than blanket suffix rules.
    assert catalog.is_clinical("chrpds_pds_1_p")
    assert not catalog.is_clinical(NONCLINICAL)
    assert not catalog.is_clinical("unmapped_new_field")


def test_slice_scope_keeps_only_clinical_data_and_required_context(catalog):
    frame = pd.DataFrame({
        "subjectid": ["AA001", "AA002"],
        CLINICAL: [1, 2],
        CLINICAL_CHECKBOX: [0, 1],
        INTERVIEW_DATE: ["2026-01-01", "2026-01-02"],
        RATER_ID: ["R1", "R2"],
        NONCLINICAL: [10, 20],
        "chrbprs_dateerror1": ["", "Please fix"],
        "unmapped_new_field": [3, 4],
    })
    scoped, audit = scope_slices_to_clinical(
        {("PRONET", "baseline"): frame}, catalog)

    assert scoped[("PRONET", "baseline")].columns.tolist() == [
        "subjectid", CLINICAL, CLINICAL_CHECKBOX, INTERVIEW_DATE, RATER_ID]
    by_var = audit.set_index("variable")
    assert by_var.loc[CLINICAL, "disposition"] == "retained_clinical"
    assert by_var.loc[INTERVIEW_DATE, "disposition"] == "retained_context"
    assert by_var.loc[NONCLINICAL, "disposition"] == "excluded"
    assert "nonclinical form" in by_var.loc[NONCLINICAL, "reason"]
    assert "fail-closed" in by_var.loc["unmapped_new_field", "reason"]


def test_finding_guard_requires_every_real_variable_leg_to_be_clinical(catalog):
    frame = pd.DataFrame([
        _finding(CLINICAL),
        _finding(CLINICAL_CHECKBOX),
        _finding(f"{CLINICAL} | {NONCLINICAL}", primary_variable=CLINICAL),
        _finding(NONCLINICAL, primary_variable=CLINICAL),
        _finding(
            f"{CLINICAL}=Married, living together & {CLINICAL_2}=A & B",
            anomaly_type="correlative_outlier",
            variables_involved=f"{CLINICAL}, {CLINICAL_2}",
        ),
        _finding(
            CLINICAL,
            anomaly_type="correlative_outlier",
            variables_involved=f"{CLINICAL}, {NONCLINICAL}",
        ),
        _finding(
            CLINICAL,
            anomaly_type="correlative_outlier",
            residual_target=CLINICAL,
            variables_involved=(
                f"{CLINICAL} (residual target) + {CLINICAL_2}, "
                "chrbprs_bprs_suic"),
        ),
        _finding(INTERVIEW_DATE, anomaly_type="date_anomaly_simple"),
        _finding("unmapped_new_field"),
        _finding(CLINICAL, anomaly_type="rater_effect",
                 variables_involved=f"{RATER_ID} -> {CLINICAL}"),
        _finding("(subject composite)", anomaly_type="whole_subject_outlier"),
        _finding("(site composite)", anomaly_type="whole_site_outlier",
                 subjectid="(site-level)"),
    ])

    kept, dropped = filter_findings_to_clinical(frame, catalog)
    assert dropped == 5
    assert kept["variable"].tolist() == [
        CLINICAL,
        CLINICAL_CHECKBOX,
        f"{CLINICAL}=Married, living together & {CLINICAL_2}=A & B",
        CLINICAL,
        CLINICAL,
        "(subject composite)",
        "(site composite)",
    ]


def test_data_dictionary_narrows_the_default_allowlist(catalog):
    """The dictionary disqualifies fields no name rule could recognize."""
    # PHI the name rules cannot see: a postal code and free-text head-injury
    # narrative were both reportable before the dictionary was consulted.
    for variable in ("chrpps_postal", "chrtbi_parent_anterograde",
                     "chrscid_a48_other_episodes"):
        assert not catalog.is_clinical(variable), variable
        assert "Identifier?" in catalog.disposition(variable)[1], variable
    # A date field whose name never says "date", so looks_like_date_column
    # cannot catch it. The dictionary's validation column can.
    assert not catalog.is_clinical("chrmed_cond10_onset")
    assert "date/time" in catalog.disposition("chrmed_cond10_onset")[1]
    # Display-only rendered HTML that captures nothing.
    assert not catalog.is_clinical("chrgfr_gf_scale")
    assert "captures no data" in catalog.disposition("chrgfr_gf_scale")[1]

    # Free text that is not identifier-marked is still not a measurement.
    for variable in ("chrpps_othered", "chrpps_othstress"):
        assert not catalog.is_clinical(variable), variable
        assert "free text" in catalog.disposition(variable)[1], variable

    # Scored items the dictionary mismarks as identifiers stay reportable.
    assert catalog.is_clinical("chrschizotypal_par00a7")
    assert catalog.is_clinical("chrtbi_parent_times")

    # The dictionary only narrows: fields it calls measurements but the
    # name/label rules reject (interviewer attestations, SCID routing) stay out.
    for variable in ("chrpsychs_scr_1b19_app", "chrscid_c29",
                     "chrscid_lifechart_check"):
        assert not catalog.is_clinical(variable), variable

    assert catalog.dictionary_excluded_count > 0
    assert catalog.data_dictionary_path.lower().endswith(".csv")


def test_data_dictionary_is_required_and_must_cover_every_clinical_form(
        tmp_path: Path):
    with pytest.raises(ClinicalVariableContractError, match="not found"):
        load_clinical_variable_catalog(
            data_dictionary_path=str(tmp_path / "absent_dictionary.csv"))

    source = pd.read_csv(load_clinical_variable_catalog().data_dictionary_path,
                         keep_default_na=False, low_memory=False)
    source.columns = [str(column).strip() for column in source.columns]
    forms = source["Form Name"].astype(str).str.strip().str.lower()
    # A dictionary that silently loses an instrument must fail, not quietly
    # leave every variable on that form unexamined and reportable.
    truncated = tmp_path / "truncated_dictionary.csv"
    source.loc[~forms.isin({"bprs", "cdss"})].to_csv(truncated, index=False)
    with pytest.raises(ClinicalVariableContractError,
                       match="does not describe clinical form"):
        load_clinical_variable_catalog(data_dictionary_path=str(truncated))

    incomplete = tmp_path / "no_identifier_column.csv"
    source.drop(columns=["Identifier?"]).to_csv(incomplete, index=False)
    with pytest.raises(ClinicalVariableContractError,
                       match="missing required column"):
        load_clinical_variable_catalog(data_dictionary_path=str(incomplete))

    # Form names alone are not enough. A dictionary whose VARIABLE names have
    # drifted keeps every form, so a form-level guard passes while the whole
    # instrument goes unscreened and its PHI/display fields become reportable.
    drifted = source.copy()
    on_form = drifted["Form Name"].astype(str).str.strip().str.lower().eq(
        "scid5_psychosis_mood_substance_abuse")
    drifted.loc[on_form, "Variable / Field Name"] = (
        "zz_" + drifted.loc[on_form, "Variable / Field Name"].astype(str))
    drift_path = tmp_path / "drifted_variable_names.csv"
    drifted.to_csv(drift_path, index=False)
    with pytest.raises(ClinicalVariableContractError,
                       match="does not describe"):
        load_clinical_variable_catalog(data_dictionary_path=str(drift_path))


def test_dependencies_path_relocates_every_metadata_contract(tmp_path: Path):
    """One argument moves the whole metadata folder, contents unchanged."""
    import shutil

    from simple_anomaly_detection.runner import SimpleAnomalyDetector

    baseline = load_clinical_variable_catalog()
    # Deliberately NOT named "dependencies": resolving the dictionary through
    # the folder's parent would search a sibling the caller never named.
    relocated = tmp_path / "metadata"
    (relocated / "data_dictionary").mkdir(parents=True)
    source_dependencies = Path(baseline.domain_map_path).parent
    for name in ("missingness_domain_forms.json", "grouped_variables.json",
                 "important_form_vars.json"):
        shutil.copy(source_dependencies / name, relocated / name)
    shutil.copy(baseline.data_dictionary_path,
                relocated / "data_dictionary" / "current_data_dictionary.csv")

    moved = load_clinical_variable_catalog(dependencies_path=str(relocated))
    assert moved.clinical_variables == baseline.clinical_variables
    assert moved.dictionary_excluded_count == baseline.dictionary_excluded_count
    # Every contract, including the dictionary, must come from the new folder.
    for path in (moved.domain_map_path, moved.grouped_variables_path,
                 moved.important_form_vars_path, moved.data_dictionary_path):
        assert Path(path).is_relative_to(tmp_path), path

    with pytest.raises(ClinicalVariableContractError, match="does not exist"):
        load_clinical_variable_catalog(
            dependencies_path=str(tmp_path / "absent"))

    # The runner accepts it, resolves the form contract from it, and rejects a
    # path that is not a directory.
    workspace = tmp_path / "io"
    workspace.mkdir()
    detector = SimpleAnomalyDetector(
        input_path=str(workspace), output_path=str(workspace),
        dependencies_path=str(relocated))
    assert detector.dependencies_path == str(relocated.resolve())
    assert Path(detector._form_contract_path()) == (
        relocated / "important_form_vars.json")
    with pytest.raises(ValueError, match="not a directory"):
        SimpleAnomalyDetector(
            input_path=str(workspace), output_path=str(workspace),
            dependencies_path=str(tmp_path / "absent"))


def test_missing_contract_fails_loudly(tmp_path: Path):
    with pytest.raises(ClinicalVariableContractError, match="Cannot load"):
        load_clinical_variable_catalog(
            domain_map_path=str(tmp_path / "missing_domains.json"))

    empty = tmp_path / "empty_domains.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(ClinicalVariableContractError, match="non-empty"):
        load_clinical_variable_catalog(domain_map_path=str(empty))

    drifted = json.loads(Path(
        load_clinical_variable_catalog().domain_map_path
    ).read_text(encoding="utf-8"))
    clinical_entry = next(
        entry for entry in drifted.values()
        if entry.get("label") == "Clinical measures")
    clinical_entry["forms"].append("renamed_form_missing_from_var_map")
    drift_path = tmp_path / "drifted_domains.json"
    drift_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ClinicalVariableContractError,
                       match="have no variable mapping"):
        load_clinical_variable_catalog(domain_map_path=str(drift_path))


def test_zero_observed_clinical_variables_fails_and_removes_stale_reports(
        tmp_path: Path):
    from simple_anomaly_detection.runner import SimpleAnomalyDetector

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (output_dir / "by_type").mkdir(parents=True)
    (output_dir / "audit_candidates").mkdir()
    pd.DataFrame({
        "subjectid": ["AA001", "AA002"],
        NONCLINICAL: [1, 2],
        "unmapped_new_field": [3, 4],
    }).to_csv(
        input_dir / "AMPSCZ-combined-redcap_baseline_PRONET.csv",
        index=False,
    )
    stale_files = [
        output_dir / "anomaly_report.xlsx",
        output_dir / "summary.csv",
        output_dir / "run_status.csv",
        output_dir / "clinical_variable_scope.csv",
        output_dir / "by_type" / "old.csv",
        output_dir / "audit_candidates" / "old_candidates.csv",
    ]
    for path in stale_files:
        path.write_text("stale unrestricted output", encoding="utf-8")

    detector = SimpleAnomalyDetector(
        input_path=str(input_dir), output_path=str(output_dir),
        networks=("PRONET",), timepoints=("baseline",))
    with pytest.raises(ClinicalVariableContractError,
                       match="zero observed"):
        detector.run()

    assert not any(path.exists() for path in stale_files)
    marker = (output_dir / "RUN_STATUS.txt").read_text(encoding="utf-8")
    assert "status=failed_no_observed_clinical_variables" in marker


def test_runner_scopes_calculations_and_every_written_finding(
        tmp_path: Path, monkeypatch):
    from simple_anomaly_detection import runner

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source = pd.DataFrame({
        "subjectid": [f"AA{i:03d}" for i in range(40)],
        CLINICAL: list(range(40)),
        CLINICAL_CHECKBOX: [i % 2 for i in range(40)],
        INTERVIEW_DATE: ["2026-01-01"] * 40,
        RATER_ID: ["R1"] * 20 + ["R2"] * 20,
        NONCLINICAL: list(range(40)),
        "unmapped_new_field": list(range(40)),
    })
    source.to_csv(
        input_dir / "AMPSCZ-combined-redcap_baseline_PRONET.csv",
        index=False,
    )

    def _detect_all(per_slice, **_):
        received = per_slice[("PRONET", "baseline")]
        assert received.columns.tolist() == [
            "subjectid", CLINICAL, CLINICAL_CHECKBOX,
            INTERVIEW_DATE, RATER_ID]
        return [
            _finding(CLINICAL),
            _finding(CLINICAL_CHECKBOX),
            _finding(f"{CLINICAL} | {NONCLINICAL}"),
            _finding(INTERVIEW_DATE, anomaly_type="date_anomaly_simple"),
            _finding(NONCLINICAL),
            _finding(CLINICAL, anomaly_type="rater_effect",
                     variables_involved=f"{RATER_ID} -> {CLINICAL}"),
            _finding("(subject composite)",
                     anomaly_type="whole_subject_outlier"),
        ]

    monkeypatch.setattr(
        runner, "_DETECTORS", [("clinical_scope_probe",
                               SimpleNamespace(detect_all=_detect_all))])
    detector = runner.SimpleAnomalyDetector(
        input_path=str(input_dir),
        output_path=str(output_dir),
        networks=("PRONET",),
        timepoints=("baseline",),
    )
    results = detector.run()

    expected = [CLINICAL, CLINICAL_CHECKBOX, CLINICAL,
                "(subject composite)"]
    assert Counter(results["clinical_scope_probe"]["variable"]) == Counter(expected)
    assert Counter(results["combined_ranked"]["variable"]) == Counter(expected)
    audit_candidates = pd.read_csv(
        output_dir / "audit_candidates" /
        "clinical_scope_probe_candidates.csv")
    assert Counter(audit_candidates["variable"]) == Counter(expected)
    workbook_detail = pd.read_excel(
        output_dir / "anomaly_report.xlsx",
        sheet_name="clinical_scope_probe")
    assert Counter(workbook_detail["variable"]) == Counter(expected)
    scope = pd.read_csv(output_dir / "clinical_variable_scope.csv")
    assert set(scope.loc[scope["disposition"] == "excluded", "variable"]) == {
        NONCLINICAL, "unmapped_new_field"}
