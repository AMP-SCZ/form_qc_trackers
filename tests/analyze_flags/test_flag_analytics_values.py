"""Regression coverage for resolved-flag before/after value reporting."""

import os
from types import SimpleNamespace

import pandas as pd
import pytest

import analyze_flags.estimate_resolved as estimate_resolved_module
import analyze_flags.flag_analytics as flag_analytics_module
from analyze_flags.estimate_resolved import ResolvedEstimator
from analyze_flags.flag_analytics import FlagAnalytics
from analyze_flags.value_extraction import (
    BeforeValue,
    infer_before_value,
    strip_proposed_evidence,
)
from qc_types.proposed_checks import ProposedChecks


def _assert_before(variable, message, value, source):
    result = infer_before_value(variable, message)

    assert result == BeforeValue(value=value, source=source)
    assert result.value == value
    assert result.source == source


def _bare_estimator(variable_map=None):
    estimator = object.__new__(ResolvedEstimator)
    estimator.template_map = {}
    estimator.variable_map = dict(variable_map or {})
    estimator._seen_templates = {}
    estimator._seen_variables = {}
    estimator._current_network = "PRESCIENT"
    return estimator


def _bare_analytics(output_dir):
    analytics = object.__new__(FlagAnalytics)
    analytics.analyze_flags_dir = str(output_dir)
    analytics.comb_csv_path = str(output_dir) + os.sep
    analytics.depend_path = str(output_dir) + os.sep
    analytics.utils = SimpleNamespace(
        missing_code_set=set(),
        missing_code_list=[],
    )
    return analytics


def _tracker_frame(specific_flags):
    return pd.DataFrame(
        {
            "Subject": ["P001"],
            "Timepoint": ["baseline"],
            "General Flag": ["example_form"],
            "Specific Flags": [specific_flags],
        }
    )


def _resolved_row(
    subject,
    *,
    variable="field_a",
    message="Variable is blank.",
    message_variable=None,
    source="V1",
    timepoint="baseline",
):
    return {
        "Subject": subject,
        "Timepoint": timepoint,
        "General_Flag": "example_form",
        "variable": variable,
        "message_variable": message_variable or variable,
        "message": message,
        "canonical_template": "example canonical template",
        "Latest_seen": "2026-01-01",
        "source": source,
        "is_currently_open": False,
        "resolution_excluded": False,
    }


def test_blank_flag_has_a_known_empty_before_value():
    _assert_before(
        "field_a",
        "Variable is blank.",
        "",
        "blank_flag",
    )


@pytest.mark.parametrize(
    "message",
    [
        "Value is empty",
        "value is empty",
        "Value is empty.",
        "[GEN-001] Value is empty.",
    ],
)
def test_legacy_blank_wording_has_a_known_empty_before_value(message):
    _assert_before("field_a", message, "", "blank_flag")


@pytest.mark.parametrize(
    "message",
    [
        "Value is empty for field_a.",
        "The Value is empty.",
        "Variable is a missing code.",
    ],
)
def test_near_blank_wordings_still_fail_closed(message):
    assert infer_before_value("field_a", message) is None


def test_blank_proof_survives_the_pharm_date_variable_bar():
    # The pharm-date suppression blocks interpolation trust, not the
    # semantic blank assertion.
    _assert_before("chrpharm_date_mod", "Variable is blank.", "", "blank_flag")


def test_proposed_evidence_uses_the_exact_variable_not_a_prefix_match():
    _assert_before(
        "field_a",
        (
            "[RULE-001] Related fields disagree.\n"
            "Variables & values: field_a_extra=wrong; field_a=7; controller=0"
        ),
        "7",
        "proposed_evidence",
    )


@pytest.mark.parametrize(
    "rendered_value",
    ["[REDACTED]", "<unavailable>", "a value cut off..."],
)
def test_proposed_evidence_that_does_not_contain_the_whole_value_is_unknown(
    rendered_value,
):
    message = (
        "[RULE-001] Related fields disagree.\n"
        f"Variables & values: field_a={rendered_value}; controller=0"
    )

    assert infer_before_value("field_a", message) is None


@pytest.mark.parametrize(
    ("variable", "message", "value"),
    [
        (
            "chrblood_wbvol1",
            "Volume (chrblood_wbvol1 = -0.25) is less than or equal to 0",
            "-0.25",
        ),
        (
            "chrpas_pmod_adult3v3",
            (
                "chrpas_pmod_adult3v1 is equal to 2, but "
                "chrpas_pmod_adult3v3 is equal to 7."
            ),
            "7",
        ),
        (
            "chrblood_wbid1",
            "Barcode (12A4) contains non-numeric characters.",
            "12A4",
        ),
        (
            "chrguid_guid",
            "GUID in incorrect format. GUID was reported to be NDAR-bad.",
            "NDAR-bad",
        ),
        (
            "test_number",
            "[DD-NUM-001] Numeric field contains malformed value 'not-a-number'.",
            "not-a-number",
        ),
    ],
)
def test_unambiguous_value_bearing_messages_supply_the_before_value(
    variable,
    message,
    value,
):
    _assert_before(variable, message, value, "message_pattern")


@pytest.mark.parametrize(
    ("variable", "message", "value"),
    [
        (
            "chrcssrsb_recent",
            "Recent variable (chrcssrsb_recent) was answered as yes, but "
            "lifetime variable (chrcssrsb_lifetime) was answered as no.",
            "2",
        ),
        (
            "chrcssrsb_lifetime",
            "Recent variable (chrcssrsb_recent) was answered as yes, but "
            "lifetime variable (chrcssrsb_lifetime) was answered as no.",
            "1",
        ),
        (
            "chrbprs_bprs_somc",
            "chrbprs_bprs_somc is 6.0, but chrbprs_bprs_unus is 2.",
            "6.0",
        ),
        (
            "chrbprs_bprs_unus",
            "chrbprs_bprs_somc is 6, but chrbprs_bprs_unus is 2.0.",
            "2.0",
        ),
        (
            "chroasis_oasis_1",
            "Marked as having no anxiety in chroasis_oasis_1, but "
            "chroasis_oasis_4 is equal to 3.",
            "0",
        ),
        (
            "chrblood_drawdate",
            "Blood draw date (2026-01-02 09:15) is later than date sent to "
            "lab (2026-01-01 17:00).",
            "2026-01-02 09:15",
        ),
        (
            "chrblood_interview_date",
            "Month 2 blood interview date (2026-01-01) is before the "
            "baseline blood interview date (2026-02-01).",
            "2026-02-01",
        ),
        (
            "chrblood_interview_date",
            "Baseline (2026-01-01) and month 2 (2026-05-01) blood interview "
            "dates are 120 days apart (should be within 90 days).",
            "2026-01-01",
        ),
        (
            "chrbprs_bprs_hall",
            "chrbprs_bprs_hall is 6, but participant is not marked as "
            "converted.",
            "6",
        ),
    ],
)
def test_active_message_contracts_recover_exact_before_values(
    variable,
    message,
    value,
):
    _assert_before(variable, message, value, "message_pattern")


def test_untrusted_legacy_message_cannot_spoof_proposed_evidence():
    message = (
        "GUID in incorrect format. GUID was reported to be BAD\n"
        "Variables & values v3: chrguid_guid=FORGED."
    )

    assert infer_before_value("chrguid_guid", message) is None
    assert strip_proposed_evidence(message) == message


@pytest.mark.parametrize(
    ("variable", "message"),
    [
        ("field_a", "[RULE-001] Related fields disagree."),
        ("field_a", "field_a is greater than 10."),
        ("field_a", "other_field is equal to 7."),
        (
            "field_a",
            "field_a is equal to 1, but field_a is equal to 2.",
        ),
        (
            "chrbprs_bprs_somc",
            "T-Score baseline not done properly.Entered value was 47, "
            "but should be 52",
        ),
        (
            # Two dates in one paren cannot prove a single exact value.
            "chrpharm_date_mod",
            "pharm modification dates (2024-05-20 and 2024-12-23) are "
            "out of range of current and previous visits",
        ),
        (
            # "not marked as complete" is any of {0, 1, blank}.
            "bprs_complete",
            "bprs not marked as complete, but subject has moved onto "
            "next timepoint",
        ),
        (
            # The displayed operand's condition is a range (< 2), and the
            # other operand's is "greater than" without its exact value.
            "chroasis_oasis_4",
            "chroasis_oasis_3 states that lifestyle was not affected, "
            "but chroasis_oasis_4 is greater than 1.",
        ),
        (
            # The mod date "is missing" names no exact stored value for
            # the displayed modification-date variable.
            "chrpharm_date_mod",
            "Pharmaceutical treatment form modification date (2023-01-01) "
            "is missing, but there is a later assessment date in "
            "chrbprs_interview_date (2024-02-02). The modification date "
            "needs to be updated.",
        ),
        (
            # V1 wording of the same family; no emitter survives to prove
            # what the interpolation contained.
            "chrbprs_interview_date",
            "Pharmaceutical treatment form modification date (2023-01-01) "
            "is missing, but there is a later assessment date in "
            "chrbprs_interview_date (2024-02-02). The modification date "
            "needs to be updated.",
        ),
        (
            # Lab values are exposed to dtype-inference re-rendering, and
            # reviewers refuted recoveries from this family.
            "chrcbc_hct",
            "Incorrect units used (chrcbc_hct = 100)",
        ),
        (
            # The duplicate-fluid producer normalizes the value
            # ("12345.0" -> "12345") before interpolating it.
            "chrblood_rack_barcode",
            "Duplicate blood ID/barcode value (chrblood_rack_barcode = "
            "12345) also found on other subject(s): AB12345 (12345 at "
            "baseline).",
        ),
        (
            "chrblood_wb1pos",
            "Duplicate positions found in two different subjects (A5).",
        ),
        (
            # MED-QC-03 interpolates parse-and-reformatted dates, not the
            # raw stored strings.
            "chrpharm_med3_offset",
            "chrpharm_med3_offset (2024-05-01) is later than "
            "chrpharm_date_mod (2024-04-01). Medication onset/offset "
            "dates cannot occur after the form modification date.",
        ),
        (
            # Even a var = value rendering in the ongoing family carries a
            # reformatted offset date; the whole family fails closed.
            "chrpharm_med2_offset",
            "This course is flagged as ongoing/intermittent "
            "(chrpharm_interm_meds = 2), but a real end date is recorded "
            "(chrpharm_med2_offset = 2024-05-06) rather than the ongoing "
            "code (1901-01-01). Please reconcile the ongoing/intermittent "
            "flag with the offset date.",
        ),
        (
            # Pharm date variables are barred from message-pattern proof
            # even for otherwise-generic equality prose.
            "chrpharm_date_mod",
            "chrpharm_date_mod is equal to 2024-01-01.",
        ),
        (
            # Reviewer-refuted family.
            "chrscid_c33_c1",
            "Difference between demographics age (22.5) and "
            "chrscid_c33_c1 (25) is 2.5.",
        ),
        (
            # V1-only wordings with no surviving emitter.
            "chriq_tscore_sum",
            "T-Score baseline not done properly.Entered value was 47, "
            "but should be 52",
        ),
        (
            "chrsofas_interview_date",
            "Date (2030-01-01) is in the future",
        ),
        (
            # PRESCIENT missingness can come from the completion status or
            # the sparse-form heuristic, so the button need not hold 1.
            "chrcdss_missing",
            "cdss marked as missing on the form, but the missing_data form "
            "has no missing-data triggers set for this timepoint.",
        ),
    ],
)
def test_messages_without_one_reliable_target_value_are_unknown(variable, message):
    assert infer_before_value(variable, message) is None


def test_cross_check_contract_recovers_only_exact_displayed_value():
    assert infer_before_value(
        "chrchs_mar",
        "[CROSS-QC-005] Cross-form review: disagreement.",
    ) == BeforeValue("0", "check_contract")
    assert infer_before_value(
        "chrcdss_calg8",
        "[CROSS-QC-010] Cross-form review: disagreement.",
    ) is None
    # The fixed contract is guarded by the expected tracker-prefix variable.
    assert infer_before_value(
        "chrsaliva_mar",
        "[CROSS-QC-005] Cross-form review: disagreement.",
    ) is None


def test_sentence_terminator_does_not_strip_a_real_trailing_period():
    assert infer_before_value(
        "field_a", "field_a is equal to BAD.."
    ) == BeforeValue("BAD.", "message_pattern")
    assert infer_before_value(
        "chrguid_guid",
        "GUID in incorrect format. GUID was reported to be BAD..",
    ) == BeforeValue("BAD.", "message_pattern")


def test_ambiguous_historical_blank_evidence_fails_closed():
    base = "[RULE-001] Numeric field contains malformed value '<blank>'."
    assert infer_before_value(
        "field_a",
        base + "\nVariables & values: field_a=<blank>",
    ) is None
    assert infer_before_value(
        "field_a",
        base + "\nVariables & values: field_a='<blank>'",
    ) == BeforeValue("<blank>", "proposed_evidence")
    assert infer_before_value(
        "field_a",
        base + "\nVariables & values v2: field_a=<blank>",
    ) is None
    assert infer_before_value(
        "field_a",
        base + "\nVariables & values v3: field_a=<blank>",
    ) == BeforeValue("", "proposed_evidence")


def test_resolved_stats_retains_all_entries_and_writes_only_known_before_after(
    tmp_path,
    monkeypatch,
):
    analytics = object.__new__(FlagAnalytics)
    analytics.analyze_flags_dir = str(tmp_path)
    analytics.comb_csv_path = str(tmp_path) + "/"
    analytics.utils = SimpleNamespace(
        missing_code_set=set(),
        missing_code_list=[],
    )

    combined = pd.DataFrame(
        {
            "subjectid": ["P001"],
            "field_a": ["  corrected  "],
            "field_b": ["42"],
        }
    )
    monkeypatch.setattr(
        analytics,
        "_load_combined_csv",
        lambda _path: (combined, None),
    )

    resolved = pd.DataFrame(
        [
            {
                "Subject": "P001",
                "Timepoint": "baseline",
                "General_Flag": "example_form",
                "variable": "field_a",
                "message": "Variable is blank.",
                "canonical_template": "Variable is blank.",
                "Latest_seen": "2026-01-01",
                "is_currently_open": False,
                "resolution_excluded": False,
            },
            {
                "Subject": "P001",
                "Timepoint": "baseline",
                "General_Flag": "example_form",
                "variable": "field_b",
                "message": "[RULE-002] Related fields disagree.",
                "canonical_template": "[RULE-002] Related fields disagree.",
                "Latest_seen": "2026-01-02",
                "is_currently_open": False,
                "resolution_excluded": False,
            },
        ]
    )

    analytics._write_resolved_value_stats(resolved, "PRESCIENT")

    detail = pd.read_csv(
        tmp_path / "resolved_values_per_entry_PRESCIENT.csv",
        dtype=str,
        keep_default_na=False,
    ).set_index("variable")
    assert set(detail.index) == {"field_a", "field_b"}
    assert detail.loc["field_a", "before_value_available"].casefold() == "true"
    assert detail.loc["field_a", "before_value"] == ""
    assert detail.loc["field_a", "before_value_source"] == "blank_flag"
    assert detail.loc["field_a", "current_value"] == "  corrected  "
    assert detail.loc["field_b", "before_value_available"].casefold() == "false"
    assert detail.loc["field_b", "before_value"] == ""
    assert detail.loc["field_b", "before_value_source"] == ""
    assert detail.loc["field_b", "current_value"] == "42"

    before_after = pd.read_csv(
        tmp_path / "resolved_before_after_values_PRESCIENT.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert list(before_after["variable"]) == ["field_a"]
    assert before_after.loc[0, "before_value"] == ""
    assert before_after.loc[0, "before_value_source"] == "blank_flag"
    assert before_after.loc[0, "current_value"] == "  corrected  "


def test_duplicate_subject_rows_fail_closed_for_current_value(
    tmp_path,
    monkeypatch,
):
    analytics = _bare_analytics(tmp_path)
    duplicate_current = pd.DataFrame(
        {
            "subjectid": ["P001", "P001"],
            "field_a": ["first", "second"],
        }
    )
    monkeypatch.setattr(
        analytics,
        "_load_combined_csv",
        lambda _path: (duplicate_current, None),
    )

    analytics._write_resolved_value_stats(
        pd.DataFrame([_resolved_row("P001")]),
        "PRESCIENT",
    )

    detail = pd.read_csv(
        tmp_path / "resolved_values_per_entry_PRESCIENT.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert list(detail["status"]) == [analytics.STATUS_SUBJECT_DUPLICATE]
    assert list(detail["current_value"]) == [""]
    paired = pd.read_csv(
        tmp_path / "resolved_before_after_values_PRESCIENT.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert paired.empty


def test_resolved_outputs_preserve_long_exact_values(
    tmp_path,
    monkeypatch,
):
    analytics = _bare_analytics(tmp_path)
    before = "B" * 401
    current = "C" * 407
    monkeypatch.setattr(
        analytics,
        "_load_combined_csv",
        lambda _path: (
            pd.DataFrame(
                {"subjectid": ["P001"], "chrguid_guid": [current]}
            ),
            None,
        ),
    )
    frame = pd.DataFrame(
        [
            _resolved_row(
                "P001",
                variable="chrguid_guid",
                message=(
                    "GUID in incorrect format. GUID was reported to be "
                    f"{before}."
                ),
            )
        ]
    )

    analytics._write_resolved_value_stats(frame, "PRESCIENT")

    detail = pd.read_csv(
        tmp_path / "resolved_values_per_entry_PRESCIENT.csv",
        dtype=str,
        keep_default_na=False,
    )
    paired = pd.read_csv(
        tmp_path / "resolved_before_after_values_PRESCIENT.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert list(detail["before_value"]) == [before]
    assert list(detail["current_value"]) == [current]
    assert list(paired["before_value"]) == [before]
    assert list(paired["current_value"]) == [current]


def test_variable_mapping_preserves_raw_message_variable_for_before_lookup(
    tmp_path,
    monkeypatch,
):
    estimator = _bare_estimator({"legacy_field": "current_field"})
    unique_rows = {}
    seen_at = pd.Timestamp("2026-01-01", tz="UTC")
    estimator._merge_revision_rows(
        _tracker_frame(
            "legacy_field : legacy_field is equal to 7."
        ),
        seen_at,
        estimator.SOURCE_V1,
        unique_rows,
    )
    estimator.analyze_flags_dir = str(tmp_path)
    estimator._write_history_csv(
        "PRESCIENT",
        unique_rows,
        {
            estimator.SOURCE_V1: seen_at,
            estimator.SOURCE_V2: seen_at + pd.Timedelta(days=1),
        },
    )

    history = pd.read_csv(
        tmp_path / "open_tracker_row_history_PRESCIENT.csv",
        keep_default_na=False,
    )
    assert list(history["variable"]) == ["current_field"]
    assert list(history["message_variable"]) == ["legacy_field"]
    assert list(history["message"]) == ["legacy_field is equal to 7."]

    history["is_currently_open"] = False
    history["resolution_excluded"] = False
    analytics = _bare_analytics(tmp_path)
    monkeypatch.setattr(
        analytics,
        "_load_combined_csv",
        lambda _path: (
            pd.DataFrame(
                {"subjectid": ["P001"], "current_field": ["9"]}
            ),
            None,
        ),
    )
    analytics._write_resolved_value_stats(history, "PRESCIENT")

    paired = pd.read_csv(
        tmp_path / "resolved_before_after_values_PRESCIENT.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert list(paired["variable"]) == ["current_field"]
    assert list(paired["before_value"]) == ["7"]
    assert list(paired["current_value"]) == ["9"]


def test_history_message_is_anchored_to_pronet_live_source(tmp_path):
    estimator = _bare_estimator()
    estimator._current_network = "PRONET"
    unique_rows = {}
    v1_seen = pd.Timestamp("2026-01-01", tz="UTC")
    v2_seen = pd.Timestamp("2026-01-03", tz="UTC")

    estimator._merge_revision_rows(
        _tracker_frame("field_a : field_a is equal to 7."),
        v1_seen,
        estimator.SOURCE_V1,
        unique_rows,
    )
    estimator._merge_revision_rows(
        _tracker_frame("field_a : field_a is equal to 8."),
        v2_seen,
        estimator.SOURCE_V2,
        unique_rows,
    )
    estimator.analyze_flags_dir = str(tmp_path)
    estimator._write_history_csv(
        "PRONET",
        unique_rows,
        {
            estimator.SOURCE_V1: v1_seen + pd.Timedelta(days=1),
            estimator.SOURCE_V2: v2_seen,
        },
    )

    history = pd.read_csv(
        tmp_path / "open_tracker_row_history_PRONET.csv",
        keep_default_na=False,
    )
    assert len(history) == 1
    assert history.loc[0, "message"] == "field_a is equal to 7."
    assert history.loc[0, "message_source"] == "V1"
    assert history.loc[0, "message_variable"] == "field_a"
    assert str(history.loc[0, "Latest_seen"]).startswith("2026-01-01")


@pytest.mark.parametrize("evidence_first", [False, True])
def test_same_timestamp_duplicate_tabs_prefer_inferable_evidence(
    evidence_first,
):
    plain = "field_a : [RULE-001] Related fields disagree."
    with_evidence = (
        plain + "\nVariables & values: field_a=7; controller=0"
    )
    messages = (
        [with_evidence, plain] if evidence_first else [plain, with_evidence]
    )
    frame = pd.concat(
        [_tracker_frame(message) for message in messages],
        ignore_index=True,
    )
    estimator = _bare_estimator()
    unique_rows = {}

    estimator._merge_revision_rows(
        frame,
        pd.Timestamp("2026-01-01", tz="UTC"),
        estimator.SOURCE_V2,
        unique_rows,
    )

    assert len(unique_rows) == 1
    record = next(iter(unique_rows.values()))
    assert record["message_variable"] == "field_a"
    before = infer_before_value(
        record["message_variable"],
        record["message"],
    )
    assert before == BeforeValue("7", "proposed_evidence")


def test_same_timestamp_conflicting_values_fail_closed():
    base = "field_a : [RULE-001] Related fields disagree."
    frame = pd.concat(
        [
            _tracker_frame(
                base + "\nVariables & values: field_a=7; controller=0"
            ),
            _tracker_frame(
                base + "\nVariables & values: field_a=8; controller=0"
            ),
        ],
        ignore_index=True,
    )
    estimator = _bare_estimator()
    unique_rows = {}

    estimator._merge_revision_rows(
        frame,
        pd.Timestamp("2026-01-01", tz="UTC"),
        estimator.SOURCE_V2,
        unique_rows,
    )

    record = next(iter(unique_rows.values()))
    assert record["message_value_conflict"] is True
    observation = record["messages_per_source"][estimator.SOURCE_V2]
    assert estimator._observation_before(observation) is None


def test_resolved_pairs_exclude_pronet_v2_only_but_keep_prescient_v1_only(
    tmp_path,
    monkeypatch,
):
    analytics = _bare_analytics(tmp_path)
    combined = pd.DataFrame(
        {
            "subjectid": ["PR-TEST", "PR-LIVE", "PS-HIST"],
            "field_a": ["test-current", "live-current", "historic-current"],
        }
    )
    monkeypatch.setattr(
        analytics,
        "_load_combined_csv",
        lambda _path: (combined, None),
    )

    analytics._write_resolved_value_stats(
        pd.DataFrame(
            [
                _resolved_row("PR-TEST", source="V2"),
                _resolved_row("PR-LIVE", source="V1"),
            ]
        ),
        "PRONET",
    )
    pronet_pairs = pd.read_csv(
        tmp_path / "resolved_before_after_values_PRONET.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert set(pronet_pairs["Subject"]) == {"PR-LIVE"}

    analytics._write_resolved_value_stats(
        pd.DataFrame([_resolved_row("PS-HIST", source="V1")]),
        "PRESCIENT",
    )
    prescient_pairs = pd.read_csv(
        tmp_path / "resolved_before_after_values_PRESCIENT.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert set(prescient_pairs["Subject"]) == {"PS-HIST"}
    assert list(prescient_pairs["current_value"]) == ["historic-current"]


def test_multiple_timepoints_loads_current_value_from_dependency_csv(tmp_path):
    output_dir = tmp_path / "output"
    combined_dir = tmp_path / "combined"
    dependency_dir = tmp_path / "dependencies"
    output_dir.mkdir()
    combined_dir.mkdir()
    dependency_dir.mkdir()

    dependency_csv = (
        dependency_dir / "multi_tp_PRESCIENT_combined.csv"
    )
    pd.DataFrame(
        {"subjectid": ["P001"], "field_a": ["dependency-current"]}
    ).to_csv(dependency_csv, index=False)
    pd.DataFrame(
        {"subjectid": ["P001"], "field_a": ["wrong-standard-current"]}
    ).to_csv(
        combined_dir
        / (
            "AMPSCZ-combined-redcap_multiple_timepoints_"
            "PRESCIENT-day1to1.csv"
        ),
        index=False,
    )

    analytics = _bare_analytics(output_dir)
    analytics.comb_csv_path = str(combined_dir) + os.sep
    analytics.depend_path = str(dependency_dir) + os.sep

    analytics._write_resolved_value_stats(
        pd.DataFrame(
            [
                _resolved_row(
                    "P001",
                    source="V2",
                    timepoint="multiple_timepoints",
                )
            ]
        ),
        "PRESCIENT",
    )

    paired = pd.read_csv(
        output_dir / "resolved_before_after_values_PRESCIENT.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert list(paired["current_value"]) == ["dependency-current"]


@pytest.mark.parametrize(
    ("network", "live_source", "non_live_source"),
    [
        ("PRESCIENT", "V2", "V1"),
        ("PRONET", "V1", "V2"),
    ],
)
def test_estimator_preserves_history_when_only_non_live_source_processed(
    tmp_path,
    monkeypatch,
    network,
    live_source,
    non_live_source,
):
    estimator = _bare_estimator()
    estimator.NETWORKS = [network]
    estimator.analyze_flags_dir = str(tmp_path)
    estimator._tracker_paths = lambda _network: [
        ("/trackers/", "non-live.xlsx", non_live_source),
        ("/trackers/", "live.xlsx", live_source),
    ]

    processed_at = pd.Timestamp("2026-01-01", tz="UTC")

    def walk(
        _path,
        source_label,
        _unique_rows,
        _jump_rows,
        _affected_rows,
        _revision_rows,
        days_back=None,
    ):
        assert days_back is None
        if source_label == non_live_source:
            return 0, processed_at, processed_at
        return 0, None, None

    estimator._walk_revisions = walk

    history_path = tmp_path / f"open_tracker_row_history_{network}.csv"
    metadata_path = tmp_path / f"tracker_metadata_{network}.json"
    history_path.write_text("last-good-history\n", encoding="utf-8")
    metadata_path.write_text('{"last": "good"}\n', encoding="utf-8")
    network_writes = []

    def replace_history(*_args):
        network_writes.append("history")
        history_path.write_text("synthetic-resolution\n", encoding="utf-8")

    def replace_metadata(*_args):
        network_writes.append("metadata")
        metadata_path.write_text('{"synthetic": true}\n', encoding="utf-8")

    estimator._write_history_csv = replace_history
    estimator._write_metadata_json = replace_metadata
    estimator._write_revision_counts = lambda *_args: None
    monkeypatch.setattr(
        estimate_resolved_module,
        "write_jumps_workbook",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        estimate_resolved_module,
        "update_template_mapping",
        lambda *_args: None,
    )

    estimator.run_script()

    assert network_writes == []
    assert history_path.read_text(encoding="utf-8") == "last-good-history\n"
    assert metadata_path.read_text(encoding="utf-8") == '{"last": "good"}\n'


@pytest.mark.parametrize(
    "rendered",
    ["alpha\u00a6beta", "alpha\uff1bbeta", "alpha beta"],
)
def test_proposed_evidence_rejects_unquoted_lossy_renderings(rendered):
    message = (
        "[RULE-001] Related fields disagree.\n"
        f"Variables & values v2: field_a={rendered}; controller=0"
    )

    assert infer_before_value("field_a", message) is None


def test_tagged_proposed_evidence_round_trips_the_exact_raw_string():
    raw = "  alpha|beta; gamma\nsecond\tline\\'\"  "
    rendered = ProposedChecks._render_evidence_value(raw)
    assert rendered.startswith("@json:")
    assert "|" not in rendered
    assert ";" not in rendered
    message = (
        "[RULE-001] Related fields disagree.\n"
        f"Variables & values v3: field_a={rendered}; controller=0"
    )

    assert infer_before_value("field_a", message) == BeforeValue(
        raw,
        "proposed_evidence",
    )


@pytest.mark.parametrize("case", ["empty", "no_eligible_resolved"])
def test_successful_empty_resolved_run_overwrites_all_value_outputs(
    tmp_path,
    monkeypatch,
    case,
):
    analytics = _bare_analytics(tmp_path)
    network = "PRESCIENT" if case == "empty" else "PRONET"
    frame = pd.DataFrame([_resolved_row("P001", source="V2")])
    if case == "empty":
        frame = frame.iloc[0:0].copy()

    basenames = [
        f"resolved_variable_values_{network}.csv",
        f"resolved_values_per_entry_{network}.csv",
        f"resolved_before_after_values_{network}.csv",
    ]
    for basename in basenames:
        (tmp_path / basename).write_text("stale\nvalue\n", encoding="utf-8")

    monkeypatch.setattr(
        analytics,
        "_load_combined_csv",
        lambda _path: pytest.fail("empty runs must not load current data"),
    )

    analytics._write_resolved_value_stats(frame, network)

    summary_columns = [
        "canonical_template",
        *(f"count_{status}" for status in analytics.STATUS_COLUMNS),
        "total_resolved",
    ]
    detail_columns = [
        "Subject",
        "Timepoint",
        "General_Flag",
        "variable",
        "message_variable",
        "canonical_template",
        "before_value_available",
        "before_value",
        "before_value_source",
        "Latest_seen",
        "status",
        "current_value",
    ]
    summary = pd.read_csv(tmp_path / basenames[0], keep_default_na=False)
    detail = pd.read_csv(tmp_path / basenames[1], keep_default_na=False)
    paired = pd.read_csv(tmp_path / basenames[2], keep_default_na=False)

    assert summary.empty
    assert list(summary.columns) == summary_columns
    assert detail.empty
    assert list(detail.columns) == detail_columns
    assert paired.empty
    assert list(paired.columns) == detail_columns


def test_estimator_preserves_history_when_newest_live_revision_was_not_processed(
    tmp_path,
    monkeypatch,
):
    estimator = _bare_estimator()
    estimator.NETWORKS = ["PRESCIENT"]
    estimator.analyze_flags_dir = str(tmp_path)
    estimator._tracker_paths = lambda _network: [
        ("/trackers/", "live.xlsx", estimator.SOURCE_V2),
    ]
    newest_processed = pd.Timestamp("2026-01-01", tz="UTC")
    newest_listed = pd.Timestamp("2026-01-02", tz="UTC")
    estimator._walk_revisions = lambda *_args, **_kwargs: (
        0,
        newest_processed,
        newest_listed,
    )

    history_path = tmp_path / "open_tracker_row_history_PRESCIENT.csv"
    metadata_path = tmp_path / "tracker_metadata_PRESCIENT.json"
    history_path.write_text("last-good-history\n", encoding="utf-8")
    metadata_path.write_text('{"last": "good"}\n', encoding="utf-8")
    network_writes = []
    estimator._write_history_csv = (
        lambda *_args: network_writes.append("history")
    )
    estimator._write_metadata_json = (
        lambda *_args: network_writes.append("metadata")
    )
    estimator._write_revision_counts = lambda *_args: None
    monkeypatch.setattr(
        estimate_resolved_module,
        "write_jumps_workbook",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        estimate_resolved_module,
        "update_template_mapping",
        lambda *_args: None,
    )

    estimator.run_script()

    assert network_writes == []
    assert history_path.read_text(encoding="utf-8") == "last-good-history\n"
    assert metadata_path.read_text(encoding="utf-8") == '{"last": "good"}\n'


def test_history_missing_subject_preserves_prior_value_outputs(tmp_path):
    analytics = _bare_analytics(tmp_path)
    analytics.apply_jump_exclusions = False
    analytics.as_of_date = None
    network = "PRESCIENT"
    history_path = tmp_path / "open_tracker_row_history_PRESCIENT.csv"
    pd.DataFrame(
        [
            {
                "Timepoint": "baseline",
                "General_Flag": "example_form",
                "variable": "field_a",
                "message": "Variable is blank.",
                "canonical_template": "blank template",
                "Earliest_seen": "2026-01-01",
                "Latest_seen": "2026-01-02",
                "source": "V2",
                "is_currently_open": True,
            }
        ]
    ).to_csv(history_path, index=False)

    basenames = [
        f"resolved_variable_values_{network}.csv",
        f"resolved_values_per_entry_{network}.csv",
        f"resolved_before_after_values_{network}.csv",
    ]
    prior_contents = {}
    for basename in basenames:
        path = tmp_path / basename
        content = f"stale-{basename}\nvalue\n"
        path.write_text(content, encoding="utf-8")
        prior_contents[path] = content

    analytics._run_for_network(network)

    assert {
        path: path.read_text(encoding="utf-8") for path in prior_contents
    } == prior_contents


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("is_currently_open", "unknown"),
        ("message_value_conflict", "unknown"),
        ("resolution_eligible", "unknown"),
    ],
)
def test_invalid_history_booleans_preserve_prior_value_outputs(
    tmp_path,
    column,
    invalid_value,
):
    analytics = _bare_analytics(tmp_path)
    analytics.apply_jump_exclusions = False
    analytics.as_of_date = None
    network = "PRESCIENT"
    row = {
        "Subject": "P001",
        "Timepoint": "baseline",
        "General_Flag": "example_form",
        "variable": "field_a",
        "message": "Variable is blank.",
        "canonical_template": "blank template",
        "Earliest_seen": "2026-01-01",
        "Latest_seen": "2026-01-02",
        "source": "V2",
        "is_currently_open": False,
        "message_value_conflict": False,
        "resolution_eligible": True,
    }
    row[column] = invalid_value
    pd.DataFrame([row]).to_csv(
        tmp_path / "open_tracker_row_history_PRESCIENT.csv",
        index=False,
    )

    prior_contents = {}
    for basename in (
        f"resolved_variable_values_{network}.csv",
        f"resolved_values_per_entry_{network}.csv",
        f"resolved_before_after_values_{network}.csv",
    ):
        path = tmp_path / basename
        content = f"last-good-{basename}\n"
        path.write_text(content, encoding="utf-8")
        prior_contents[path] = content

    analytics._run_for_network(network)

    assert {
        path: path.read_text(encoding="utf-8") for path in prior_contents
    } == prior_contents


def test_valid_header_only_history_clears_prior_value_outputs(tmp_path):
    analytics = _bare_analytics(tmp_path)
    analytics.apply_jump_exclusions = False
    analytics.as_of_date = None
    network = "PRESCIENT"
    history_columns = [
        "Subject",
        "Timepoint",
        "General_Flag",
        "variable",
        "canonical_template",
        "Earliest_seen",
        "Latest_seen",
        "is_currently_open",
    ]
    pd.DataFrame(columns=history_columns).to_csv(
        tmp_path / "open_tracker_row_history_PRESCIENT.csv",
        index=False,
    )

    basenames = [
        f"resolved_variable_values_{network}.csv",
        f"resolved_values_per_entry_{network}.csv",
        f"resolved_before_after_values_{network}.csv",
    ]
    for basename in basenames:
        (tmp_path / basename).write_text("stale\nvalue\n", encoding="utf-8")
    for group_label in ("form", "timepoint"):
        (tmp_path / f"outstanding_by_{group_label}_{network}.csv").write_text(
            "stale\nvalue\n", encoding="utf-8")
        (
            tmp_path
            / f"outstanding_stacked_by_{group_label}_{network}.png"
        ).write_bytes(b"stale plot")
    (tmp_path / f"flag_template_counts_{network}.csv").write_text(
        "stale\nvalue\n", encoding="utf-8")

    analytics._run_for_network(network)

    summary_columns = [
        "canonical_template",
        *(f"count_{status}" for status in analytics.STATUS_COLUMNS),
        "total_resolved",
    ]
    summary = pd.read_csv(tmp_path / basenames[0], keep_default_na=False)
    detail = pd.read_csv(tmp_path / basenames[1], keep_default_na=False)
    paired = pd.read_csv(tmp_path / basenames[2], keep_default_na=False)
    assert summary.empty
    assert list(summary.columns) == summary_columns
    assert detail.empty
    assert list(detail.columns) == analytics.RESOLVED_VALUE_DETAIL_COLUMNS
    assert paired.empty
    assert list(paired.columns) == analytics.RESOLVED_VALUE_DETAIL_COLUMNS
    for group_label in ("form", "timepoint"):
        stacked = pd.read_csv(
            tmp_path / f"outstanding_by_{group_label}_{network}.csv",
            keep_default_na=False,
        )
        assert stacked.empty
        assert list(stacked.columns) == ["date", "total"]
        assert not (
            tmp_path
            / f"outstanding_stacked_by_{group_label}_{network}.png"
        ).exists()
    counts = pd.read_csv(
        tmp_path / f"flag_template_counts_{network}.csv",
        keep_default_na=False,
    )
    assert counts.empty
    assert list(counts.columns) == [
        "canonical_template",
        "ever_raised",
        "currently_outstanding",
        "example_form",
        "example_variable",
    ]


def test_successful_no_clinical_history_clears_non_value_outputs(tmp_path):
    analytics = _bare_analytics(tmp_path)
    analytics.apply_jump_exclusions = False
    analytics.as_of_date = None
    network = "PRESCIENT"
    pd.DataFrame(
        [
            {
                "Subject": "P001",
                "Timepoint": "baseline",
                "General_Flag": "nonclinical_form",
                "variable": "field_a",
                "canonical_template": "nonclinical template",
                "Earliest_seen": "2026-01-01",
                "Latest_seen": "2026-01-02",
                "is_currently_open": True,
            }
        ]
    ).to_csv(
        tmp_path / "open_tracker_row_history_PRESCIENT.csv",
        index=False,
    )
    for group_label in ("form", "timepoint"):
        (tmp_path / f"outstanding_by_{group_label}_{network}.csv").write_text(
            "stale\nvalue\n", encoding="utf-8")
        (
            tmp_path
            / f"outstanding_stacked_by_{group_label}_{network}.png"
        ).write_bytes(b"stale plot")
    (tmp_path / f"flag_template_counts_{network}.csv").write_text(
        "stale\nvalue\n", encoding="utf-8")

    analytics._run_for_network(network)

    for group_label in ("form", "timepoint"):
        stacked = pd.read_csv(
            tmp_path / f"outstanding_by_{group_label}_{network}.csv",
            keep_default_na=False,
        )
        assert stacked.empty
        assert list(stacked.columns) == ["date", "total"]
        assert not (
            tmp_path
            / f"outstanding_stacked_by_{group_label}_{network}.png"
        ).exists()
    counts = pd.read_csv(
        tmp_path / f"flag_template_counts_{network}.csv",
        keep_default_na=False,
    )
    assert counts.empty


def test_flag_analytics_repo_root_is_platform_native():
    expected = os.path.dirname(
        os.path.dirname(os.path.realpath(flag_analytics_module.__file__))
    )

    assert flag_analytics_module.parent_dir == expected
    assert os.path.isabs(flag_analytics_module.parent_dir)


@pytest.mark.parametrize("raw", ["@json", "@jsonfoo", "@json:literal"])
def test_v3_tag_namespace_literals_round_trip_exactly(raw):
    rendered = ProposedChecks._render_evidence_value(raw)
    assert rendered.startswith("@json:")
    assert rendered != raw
    message = (
        "[RULE-001] Related fields disagree.\n"
        f"Variables & values v3: field_a={rendered}; controller=0"
    )

    assert infer_before_value("field_a", message) == BeforeValue(
        raw,
        "proposed_evidence",
    )

@pytest.mark.parametrize("v1_outcome", ["raises", "rejected"])
def test_prescient_fresh_v2_preserves_prior_v1_history_when_v1_unavailable(
    tmp_path,
    monkeypatch,
    v1_outcome,
):
    estimator = _bare_estimator()
    estimator.NETWORKS = ["PRESCIENT"]
    estimator.analyze_flags_dir = str(tmp_path)
    estimator._tracker_paths = lambda _network: [
        ("/trackers/", "current-v2.xlsx", estimator.SOURCE_V2),
        ("/trackers/", "historic-v1.xlsx", estimator.SOURCE_V1),
    ]

    history_path = tmp_path / "open_tracker_row_history_PRESCIENT.csv"
    pd.DataFrame(
        [
            {
                "Subject": "P001",
                "Timepoint": "baseline",
                "General_Flag": "legacy_form",
                "variable": "legacy_field",
                "message_variable": "legacy_field",
                "message": "Variable is blank.",
                "canonical_template": "<var> : Variable is blank.",
                "Earliest_seen": "2025-01-01",
                "Latest_seen": "2025-12-31",
                "source": "V1",
                "is_currently_open": False,
                "message_value_conflict": False,
                "resolution_eligible": True,
            }
        ]
    ).to_csv(history_path, index=False)
    prior_history = history_path.read_text(encoding="utf-8")

    metadata_path = tmp_path / "tracker_metadata_PRESCIENT.json"
    prior_metadata = (
        "{\n"
        '  "network": "PRESCIENT",\n'
        '  "live_source": "V1",\n'
        '  "newest_revision_V1": "2025-12-31T00:00:00+00:00",\n'
        '  "newest_revision_V2": null,\n'
        '  "newest_revision_overall": "2025-12-31T00:00:00+00:00"\n'
        "}\n"
    )
    metadata_path.write_text(prior_metadata, encoding="utf-8")

    v1_listed = pd.Timestamp("2026-01-02", tz="UTC")
    v2_fresh = pd.Timestamp("2026-01-03", tz="UTC")
    walked_sources = []

    def walk(_path, source_label, *_args, **_kwargs):
        walked_sources.append(source_label)
        if source_label == estimator.SOURCE_V2:
            return 0, v2_fresh, v2_fresh
        if v1_outcome == "raises":
            raise RuntimeError("historic V1 tracker unavailable")
        return 0, None, v1_listed

    estimator._walk_revisions = walk
    network_writes = []
    estimator._write_history_csv = (
        lambda *_args: network_writes.append("history")
    )
    estimator._write_metadata_json = (
        lambda *_args: network_writes.append("metadata")
    )
    estimator._write_revision_counts = lambda *_args: None
    monkeypatch.setattr(
        estimate_resolved_module,
        "write_jumps_workbook",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        estimate_resolved_module,
        "update_template_mapping",
        lambda *_args: None,
    )

    estimator.run_script()

    assert walked_sources == [estimator.SOURCE_V2, estimator.SOURCE_V1]
    assert network_writes == []
    assert history_path.read_text(encoding="utf-8") == prior_history
    assert metadata_path.read_text(encoding="utf-8") == prior_metadata


def test_zero_byte_history_preserves_all_value_outputs_without_raising(tmp_path):
    analytics = _bare_analytics(tmp_path)
    analytics.apply_jump_exclusions = False
    analytics.as_of_date = None
    network = "PRESCIENT"
    (tmp_path / "open_tracker_row_history_PRESCIENT.csv").write_bytes(b"")

    basenames = [
        f"resolved_variable_values_{network}.csv",
        f"resolved_values_per_entry_{network}.csv",
        f"resolved_before_after_values_{network}.csv",
    ]
    prior_contents = {}
    for basename in basenames:
        path = tmp_path / basename
        content = f"prior-{basename}\nkeep-me\n"
        path.write_text(content, encoding="utf-8")
        prior_contents[path] = content

    analytics._run_for_network(network)

    assert {
        path: path.read_text(encoding="utf-8") for path in prior_contents
    } == prior_contents
