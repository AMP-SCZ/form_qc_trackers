"""Regression tests for the audited cross-form check catalog and adapter."""

from pathlib import Path
import os
import subprocess
import sys

import pandas as pd
import pytest

import test_cross_checks as source
import qc_types.cross_checks as adapter


ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROW_FIELDS = {
    "visit_status": "baseline",
    "visit_status_string": "baseln",
    "recruitment_status_v2": "recruited",
}


def _catalog_rule(number):
    return next(rule for rule in adapter.RULE_CATALOG
                if rule.number == number)


def test_source_catalog_import_is_side_effect_free(tmp_path):
    """The former standalone script must be safe for the pipeline to import."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env.get("PYTHONPATH", "")])

    result = subprocess.run(
        [sys.executable, "-c", "import test_cross_checks"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_catalog_has_a_complete_exclusive_audit_disposition():
    all_numbers = set(range(1, 147))

    assert len(source.RULES) == 146
    assert set(source.RULE_LABELS) == all_numbers
    assert source.ENABLED_RULE_NUMBERS.isdisjoint(
        source.DISABLED_RULE_REASONS)
    assert (source.ENABLED_RULE_NUMBERS
            | set(source.DISABLED_RULE_REASONS)) == all_numbers
    assert tuple(rule.number for rule in adapter.RULE_CATALOG) == tuple(
        sorted(source.ENABLED_RULE_NUMBERS))
    assert len({rule.check_id for rule in adapter.RULE_CATALOG}) == len(
        adapter.RULE_CATALOG)
    assert all(rule.check_id == f"CROSS-QC-{rule.number:03d}"
               for rule in adapter.RULE_CATALOG)
    assert all("df.filter" not in rule.expression
               for rule in adapter.RULE_CATALOG)


def test_false_positive_and_duplicate_rule_families_stay_disabled():
    invalid_current_vs_lifetime = {2, 123, 125, 140}
    duplicate_pharmaceutical = set(range(40, 86))
    duplicate_existing_check = {26}
    medical_scope_mismatch = {87, 89, 95, 97, 99, 103, 105, 107, 124}
    ambiguous_same_day_ordering = {126, 127, 134, 135}
    missingness = {
        number for number, reason in source.DISABLED_RULE_REASONS.items()
        if "missingness" in reason.casefold()
    }

    assert invalid_current_vs_lifetime.isdisjoint(
        source.ENABLED_RULE_NUMBERS)
    assert duplicate_pharmaceutical.isdisjoint(
        source.ENABLED_RULE_NUMBERS)
    assert duplicate_existing_check.isdisjoint(
        source.ENABLED_RULE_NUMBERS)
    assert medical_scope_mismatch.isdisjoint(
        source.ENABLED_RULE_NUMBERS)
    assert ambiguous_same_day_ordering.isdisjoint(
        source.ENABLED_RULE_NUMBERS)
    assert all(
        "blood draw" in source.DISABLED_RULE_REASONS[number].casefold()
        and "ordering" in source.DISABLED_RULE_REASONS[number].casefold()
        for number in ambiguous_same_day_ordering
    )
    assert all(
        "medical" in source.DISABLED_RULE_REASONS[number].casefold()
        or "medicine" in source.DISABLED_RULE_REASONS[number].casefold()
        or "prescribed" in source.DISABLED_RULE_REASONS[number].casefold()
        for number in medical_scope_mismatch
    )
    assert missingness
    assert missingness.isdisjoint(source.ENABLED_RULE_NUMBERS)


def test_pronet_path_uses_the_export_filename_case():
    assert source.build_file_path(
        "month12", "PRONET", "C:/combined/"
    ) == (
        "C:/combined/AMPSCZ-combined-redcap_month_12_"
        "ProNET-day1to1.csv"
    )
    assert source.build_file_path(
        "floating", "PRESCIENT", "C:/combined/"
    ).endswith(
        "AMPSCZ-combined-redcap_floating_forms_PRESCIENT-day1to1.csv"
    )


def test_legacy_pharmaceutical_matrix_rules_keep_repeating_columns():
    """Standalone diagnostics must retain and normalize regex-selected slots."""
    raw = pd.DataFrame({
        "chrap_date": ["2025-02-01", "2025-02-01"],
        "chrpharm_interview_date": ["2025-03-01", "2025-01-01"],
        "chrpharm_date_first": ["2025-03-01", "2025-01-01"],
        "chrap_ams": [1, 0],
        # Float-shaped codes reproduce pandas' inference for columns with
        # blanks while still representing REDCap integer choice codes.
        "chrpharm_med1_tp": [-1.0, -1.0],
        "chrpharm_med1_name": [999.0, 238.0],
        "chrpharm_med1_name_past": [999.0, 999.0],
    })

    forward = source.prep_df_for_rule(raw, source.RULES[39])
    reverse = source.prep_df_for_rule(raw, source.RULES[40])

    expected_slots = {
        "chrpharm_med1_tp",
        "chrpharm_med1_name",
        "chrpharm_med1_name_past",
    }
    assert expected_slots <= set(forward.columns)
    assert expected_slots <= set(reverse.columns)
    assert source.evaluate_rule_mask(forward, 40).tolist() == [True, False]
    assert source.evaluate_rule_mask(reverse, 41).tolist() == [False, True]


def test_matrix_rule_preparation_fails_when_regex_matches_zero_columns():
    """An absent repeating-form matrix must not look like all False values."""
    raw = pd.DataFrame({
        "chrap_date": ["2025-02-01"],
        "chrpharm_interview_date": ["2025-03-01"],
        "chrpharm_date_first": ["2025-03-01"],
        "chrap_ams": [1],
    })

    with pytest.raises(KeyError, match="matched zero columns"):
        source.prep_df_for_rule(raw, source.RULES[39])


def test_matrix_rule_evaluation_revalidates_manually_prepared_columns():
    """Direct evaluate callers receive the same fail-loud matrix guarantee."""
    direct_only = pd.DataFrame({
        "chrap_date": [pd.Timestamp("2025-02-01")],
        "chrpharm_interview_date": [pd.Timestamp("2025-03-01")],
        "chrpharm_date_first": [pd.Timestamp("2025-03-01")],
        "chrap_ams": [1],
    })

    with pytest.raises(
        KeyError, match="Rule 40.*matched zero columns"
    ):
        source.evaluate_rule_mask(direct_only, 40)


def test_dates_are_calendar_normalized_and_sentinels_become_nat():
    raw = pd.DataFrame({
        "chrsaliva_interview_date": [
            "2025-04-03 00:01:00",
            "2025-04-04 00:01:00",
            "1901-01-01",
        ],
        "chrchs_interview_date": [
            "2025-04-03 23:59:00",
            "2025-04-03 23:59:00",
            "2025-04-03",
        ],
        # This is an onset date despite its nonstandard variable name.
        "chrchs_tobuse": ["2020-01-01", "-9", ""],
    })

    prepared = source.prepare_dataframe(raw, raw.columns)
    same_calendar_day = (
        prepared["chrsaliva_interview_date"]
        - prepared["chrchs_interview_date"]
    ).dt.days.abs() < 1

    assert same_calendar_day.tolist() == [True, False, False]
    assert prepared.loc[0, "chrchs_tobuse"] == pd.Timestamp("2020-01-01")
    assert pd.isna(prepared.loc[1, "chrchs_tobuse"])
    assert pd.isna(prepared.loc[2, "chrchs_tobuse"])
    assert pd.isna(prepared.loc[2, "chrsaliva_interview_date"])

    sentinel_dates = ("1901-01-01", "1903-03-03", "1909-09-09")
    serialized_sentinels = pd.DataFrame({
        "chrchs_tobuse": [
            serialized
            for date in sentinel_dates
            for serialized in (
                f"{date} 00:00:00",
                f"{date}T12:34:56",
            )
        ],
    })
    normalized_sentinels = source.prepare_dataframe(
        serialized_sentinels, ["chrchs_tobuse"])

    assert normalized_sentinels["chrchs_tobuse"].isna().all()


def test_mixed_iso_date_and_datetime_rows_parse_independently():
    """A neighboring ISO representation must not change a row's result."""
    representations = [
        "2025-04-03",
        "2025-04-03 12:34:56",
        "2025-04-03T23:59:59",
    ]
    raw = pd.DataFrame({
        "chrsaliva_interview_date": representations,
        "chrchs_interview_date": list(reversed(representations)),
    })

    forward = source.prepare_dataframe(raw, raw.columns)
    reverse = source.prepare_dataframe(
        raw.iloc[::-1].reset_index(drop=True), raw.columns)

    assert forward.notna().all().all()
    assert reverse.notna().all().all()
    assert forward.nunique().eq(1).all()
    assert reverse.nunique().eq(1).all()
    assert forward.iloc[0].eq(pd.Timestamp("2025-04-03")).all()
    assert reverse.iloc[0].eq(pd.Timestamp("2025-04-03")).all()


def test_enabled_bprs_cdss_suicidality_rules_require_same_calendar_day():
    raw = pd.DataFrame({
        "chrbprs_interview_date": [
            "2025-01-02 23:59:00",
            "2025-01-03",
            "2025-01-01",
        ],
        "chrcdss_interview_date": [
            "2025-01-02",
            "2025-01-02",
            "2025-01-02",
        ],
        "chrcdss_calg8": [0, 0, 0],
        "chrbprs_bprs_suic": [4, 4, 4],
    })
    prepared = source.prep_df_for_rule(raw, source.RULES[8])

    # Same calendar day is included. Both later and earlier adjacent dates
    # are excluded, making the temporal gate symmetric.
    assert source.evaluate_rule_mask(prepared, 9).tolist() == [
        True, False, False]

    reverse_values = raw.assign(
        chrcdss_calg8=[2, 2, 2],
        chrbprs_bprs_suic=[2, 2, 2],
    )
    prepared_reverse = source.prep_df_for_rule(
        reverse_values, source.RULES[9])
    assert source.evaluate_rule_mask(prepared_reverse, 10).tolist() == [
        True, False, False]


def test_corrected_bprs_not_present_code_is_one():
    raw = pd.DataFrame({
        "chrbprs_interview_date": ["2025-01-01", "2025-01-01"],
        "chrcdss_interview_date": ["2025-01-01", "2025-01-01"],
        "chrcdss_calg1": [2, 2],
        "chrbprs_bprs_depr": [1, 0],
    })
    prepared = source.prep_df_for_rule(raw, source.RULES[7])

    assert source.evaluate_rule_mask(prepared, 8).tolist() == [True, False]
    assert '(df["chrbprs_bprs_depr"] == 1)' in source.RULES[10]


def test_cssrs_lifetime_and_past_month_no_codes_are_distinct():
    raw = pd.DataFrame({
        "chrcssrsb_interview_date": ["2025-01-01"] * 3,
        "chrbprs_interview_date": ["2025-01-01"] * 3,
        "chrbprs_bprs_suic": [4, 4, 4],
        # Lifetime: 2 means No. Past month: 0 means No.
        "chrcssrsb_si2l": [2, 1, 1],
        "chrcssrsb_css_sim2": [1, 0, 2],
    })
    prepared = source.prep_df_for_rule(raw, source.RULES[12])

    assert source.evaluate_rule_mask(prepared, 13).tolist() == [
        True, True, False]
    for number in (13, 14, 15, 24, 25, 28):
        expression = source.RULES[number - 1]
        for variable in source.extract_columns(expression):
            if "css_sim" in variable:
                assert f'df["{variable}"] == 0' in expression


def _install_minimal_form_check(monkeypatch, *, catalog, subjects,
                                schedules, variable_forms,
                                converted_logic=None,
                                excluded_logic=None,
                                missing_converted=(),
                                missing_important_forms=(),
                                blank_completion_forms=(),
                                missing_subject_metadata=(),
                                forced_create_row_reports=None):
    """Replace only FormCheck infrastructure, leaving adapter logic intact."""
    def fake_base_init(instance, timepoint, network, form_check_info):
        instance.timepoint = timepoint
        instance.network = network
        instance.final_output_list = []
        instance.rule_status = {}
        instance.subject_info = {
            subject: {
                "inclusion_status": "included",
                "visit_status": "baseline",
                "age": "unknown",
                **metadata,
            }
            for subject, metadata in subjects.items()
        }
        for metadata in instance.subject_info.values():
            for key in missing_subject_metadata:
                metadata.pop(key, None)
        instance.forms_per_tp = schedules
        instance.grouped_vars = {
            "var_forms": variable_forms,
            "var_translations": {},
        }
        instance.conv_bl = {
            variable: {
                "variable": variable,
                "original_branching_logic": "",
                "converted_branching_logic": "",
            }
            for variable in variable_forms
        }
        instance.conv_bl.update(converted_logic or {})
        for variable in missing_converted:
            instance.conv_bl.pop(variable, None)
        instance.excl_bl = excluded_logic or {}
        instance.important_form_vars = {
            form: {
                "completion_var": f"{form}_complete",
                "missing_var": f"{form}_missing",
                "interview_date_var": "",
                "non_branch_logic_vars": [],
            }
            for form in set(variable_forms.values())
        }
        for form in missing_important_forms:
            instance.important_form_vars.pop(form, None)
        for form in blank_completion_forms:
            instance.important_form_vars[form]["completion_var"] = ""

    def fake_standard_filter(instance, row, form):
        return (
            getattr(row, f"{form}_complete") == 2
            and getattr(row, f"{form}_missing") == 0
        )

    def fake_create_row(instance, row, forms, variables, message, changes):
        output = {
            "network": instance.network,
            "subject": row.subjectid,
            "affected_timepoints": instance.timepoint,
            "affected_forms": tuple(forms),
            "affected_variables": tuple(variables),
            "displayed_form": forms[0],
            "displayed_variable": variables[0],
            "error_message": message,
            # Model FormCheck's inherited screening/form highlighting so the
            # adapter test proves that CrossChecks deliberately clears it.
            "priority": True,
            "priority_item": True,
            **changes,
        }
        if forced_create_row_reports is not None:
            # Simulate FormCheck's global withdrawn/recruited/inclusion routing
            # clearing the requested tracker route after output overrides.
            output["reports"] = forced_create_row_reports
        return output

    monkeypatch.setattr(adapter.FormCheck, "__init__", fake_base_init)
    monkeypatch.setattr(adapter.CrossChecks, "standard_form_filter",
                        fake_standard_filter)
    monkeypatch.setattr(adapter.CrossChecks, "create_row_output",
                        fake_create_row)
    monkeypatch.setattr(adapter.CrossChecks, "RULE_CATALOG", catalog)


def test_checker_prepares_once_and_emits_stable_rows(monkeypatch):
    catalog = (_catalog_rule(5), _catalog_rule(6))
    variables = {
        "chrchs_interview_date": "chs",
        "chrchs_mar": "chs",
        "chrchs_tob": "chs",
        "chrsaliva_interview_date": "saliva",
        "chrsaliva_mar": "saliva",
        "chrsaliva_tob": "saliva",
    }
    subjects = {
        "S1": {"cohort": "CHR"},
        "S2": {"cohort": "CHR"},
        "S3": {"cohort": "HC"},
    }
    schedules = {
        # Deliberately mixed/incomplete: schedules are ignored by this tab.
        "chr": {"BASELINE": ["chs", "saliva"]},
        "hc": {"baseline": ["chs"]},
    }
    _install_minimal_form_check(
        monkeypatch,
        catalog=catalog,
        subjects=subjects,
        schedules=schedules,
        variable_forms=variables,
    )

    real_prepare = adapter.prepare_dataframe
    prepare_calls = []

    def counted_prepare(*args, **kwargs):
        prepared = real_prepare(*args, **kwargs)
        prepare_calls.append(tuple(prepared.columns))
        return prepared

    monkeypatch.setattr(adapter, "prepare_dataframe", counted_prepare)
    rows = pd.DataFrame([
        dict(subjectid="S1", chrchs_interview_date="2025-01-01 23:00",
             chrsaliva_interview_date="2025-01-01 01:00", chrchs_mar=0,
             chrsaliva_mar=1, chrchs_tob=0, chrsaliva_tob=1,
             chs_complete=2, chs_missing=0,
             saliva_complete=2, saliva_missing=0,
             **RUNTIME_ROW_FIELDS),
        # An explicit missing-form button must not suppress either rule hit.
        dict(subjectid="S2", chrchs_interview_date="2025-01-01",
             chrsaliva_interview_date="2025-01-01", chrchs_mar=0,
             chrsaliva_mar=1, chrchs_tob=0, chrsaliva_tob=1,
             chs_complete=2, chs_missing=0,
             saliva_complete=0, saliva_missing=1,
             **RUNTIME_ROW_FIELDS),
        # An unscheduled source form must not suppress either rule hit.
        dict(subjectid="S3", chrchs_interview_date="2025-01-01",
             chrsaliva_interview_date="2025-01-01", chrchs_mar=0,
             chrsaliva_mar=1, chrchs_tob=0, chrsaliva_tob=1,
             chs_complete=2, chs_missing=0,
             saliva_complete=2, saliva_missing=0,
             **RUNTIME_ROW_FIELDS),
    ])
    # Simulate the many unrelated columns in a real combined export.  They
    # remain available on the raw candidate row, but must not be copied into
    # the normalized operand frame.
    rows["unrelated_export_payload"] = "large-unused-value"

    checker = adapter.CrossChecks(rows, "baseline", "PRONET", {})
    output = checker()

    assert len(prepare_calls) == 1
    assert set(prepare_calls[0]) == set(variables)
    assert "unrelated_export_payload" not in prepare_calls[0]
    assert [row["check_id"] for row in output] == [
        "CROSS-QC-005", "CROSS-QC-005", "CROSS-QC-005",
        "CROSS-QC-006", "CROSS-QC-006", "CROSS-QC-006",
    ]
    assert [row["subject"] for row in output] == [
        "S1", "S2", "S3", "S1", "S2", "S3"]
    assert all(row["reports"] == "Cross Checks" for row in output)
    assert all(row["nda_excluder"] is False for row in output)
    assert all(row["priority"] is False for row in output)
    assert all(row["priority_item"] is False for row in output)
    assert all(row["affected_forms"] == ("chs", "saliva")
               for row in output)
    assert output[0]["displayed_variable"] == "chrchs_mar"
    assert output[3]["displayed_variable"] == "chrchs_tob"
    assert all(row["affected_variables"][-2:] == (
        "chrsaliva_interview_date", "chrchs_interview_date")
        for row in output)
    assert all(row["error_message"].startswith(
        f"[{row['check_id']}] Cross-form review:") for row in output)
    for check_id in ("CROSS-QC-005", "CROSS-QC-006"):
        assert checker.rule_metrics[check_id] == {
            "status": "evaluated",
            "raw_candidates": 3,
            "emitted": 3,
            "routed": 3,
            "suppressed_form_or_schedule": 0,
            "suppressed_branching": 0,
            "suppressed_duplicate_identity": 0,
            "suppressed_by_global_routing": 0,
            "filters_bypassed": True,
        }


def test_checker_restores_route_after_global_participant_filters(monkeypatch):
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={
            "S1": {
                "cohort": "unknown",
                "inclusion_status": "excluded",
            },
        },
        schedules={},
        variable_forms=_rule_5_variables(),
        # Model create_row_output clearing the route for a removed,
        # non-recruited, excluded participant.
        forced_create_row_reports="",
    )
    row = _rule_5_row()
    row["visit_status_string"] = "removed"
    row["recruitment_status_v2"] = "not_recruited"

    checker = adapter.CrossChecks(
        pd.DataFrame([row]), "baseline", "PRONET", {})

    assert checker()[0]["reports"] == "Cross Checks"
    assert checker()[0]["excluded_enabled"] is True
    assert checker()[0]["withdrawn_enabled"] is True
    assert checker.rule_metrics["CROSS-QC-005"]["routed"] == 1


def _rule_5_variables():
    return {
        "chrchs_interview_date": "chs",
        "chrchs_mar": "chs",
        "chrsaliva_interview_date": "saliva",
        "chrsaliva_mar": "saliva",
    }


def _rule_5_row(subject="S1"):
    return {
        "subjectid": subject,
        "chrchs_interview_date": "2025-01-01",
        "chrsaliva_interview_date": "2025-01-01",
        "chrchs_mar": 0,
        "chrsaliva_mar": 1,
        "chs_complete": 2,
        "chs_missing": 0,
        "saliva_complete": 2,
        "saliva_missing": 0,
        **RUNTIME_ROW_FIELDS,
    }


def test_checker_rejects_duplicate_subject_rows_before_matrix_evaluation(
        monkeypatch):
    variables = _rule_5_variables()
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHR"}},
        schedules={"chr": {"baseline": ["chs", "saliva"]}},
        variable_forms=variables,
    )

    with pytest.raises(adapter.CrossCheckInputError, match=(
            "exactly one row per subject.*S1")):
        adapter.CrossChecks(
            pd.DataFrame([_rule_5_row(), _rule_5_row()]),
            "baseline", "PRONET", {})


def test_checker_marks_rule_unavailable_when_operand_column_is_missing(
        monkeypatch):
    variables = _rule_5_variables()
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHR"}},
        schedules={"chr": {"baseline": ["chs", "saliva"]}},
        variable_forms=variables,
    )
    row = _rule_5_row()
    del row["chrsaliva_mar"]

    checker = adapter.CrossChecks(
        pd.DataFrame([row]), "baseline", "PRONET", {})

    assert checker() == []
    assert checker.rule_status["CROSS-QC-005"] == (
        "unavailable_missing_columns", ("chrsaliva_mar",))
    assert checker.rule_metrics["CROSS-QC-005"] == {
        "status": "unavailable_missing_columns",
        "missing_columns": ("chrsaliva_mar",),
        "raw_candidates": 0,
        "emitted": 0,
        "routed": 0,
        "filters_bypassed": True,
    }


def test_checker_reports_all_missing_operands_without_consulting_schedule(
        monkeypatch):
    variables = _rule_5_variables()
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHR"}},
        schedules={"chr": {"baseline": ["chs"]}},
        variable_forms=variables,
    )

    checker = adapter.CrossChecks(
        pd.DataFrame([{"subjectid": "S1", **RUNTIME_ROW_FIELDS}]),
        "baseline", "PRONET", {})

    assert checker() == []
    assert checker.rule_status["CROSS-QC-005"] == (
        "unavailable_missing_columns", _catalog_rule(5).variables)
    assert checker.rule_metrics["CROSS-QC-005"]["status"] == (
        "unavailable_missing_columns")
    assert checker.rule_metrics["CROSS-QC-005"]["filters_bypassed"] is True


@pytest.mark.parametrize(
    "schedules",
    [
        {},
        {"chr": {}},
    ],
)
def test_checker_bypasses_missing_cohort_and_timepoint_schedules(
        monkeypatch, schedules):
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHR"}},
        schedules=schedules,
        variable_forms=_rule_5_variables(),
    )

    checker = adapter.CrossChecks(
        pd.DataFrame([_rule_5_row()]),
        "baseline", "PRONET", {})

    assert [row["subject"] for row in checker()] == ["S1"]


def test_checker_bypasses_explicit_unknown_cohort(monkeypatch):
    variables = _rule_5_variables()
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={
            "ELIGIBLE": {"cohort": "CHR"},
            "UNCLASSIFIED": {"cohort": "unknown"},
        },
        schedules={"chr": {"baseline": ["chs", "saliva"]}},
        variable_forms=variables,
    )

    checker = adapter.CrossChecks(
        pd.DataFrame([
            _rule_5_row("ELIGIBLE"),
            _rule_5_row("UNCLASSIFIED"),
        ]),
        "baseline", "PRONET", {})

    assert [row["subject"] for row in checker()] == [
        "ELIGIBLE", "UNCLASSIFIED"]
    assert checker.rule_metrics["CROSS-QC-005"]["emitted"] == 2


def test_checker_bypasses_unsupported_non_unknown_cohort(monkeypatch):
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHRR"}},
        schedules={},
        variable_forms=_rule_5_variables(),
    )

    checker = adapter.CrossChecks(
        pd.DataFrame([_rule_5_row()]),
        "baseline", "PRONET", {})

    assert [row["subject"] for row in checker()] == ["S1"]


def test_checker_keeps_output_form_mapping_as_a_structural_contract(
        monkeypatch):
    variables = _rule_5_variables()
    variables.pop("chrsaliva_mar")
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHR"}},
        schedules={"chr": {"baseline": ["chs", "saliva"]}},
        variable_forms=variables,
    )

    with pytest.raises(adapter.CrossCheckConfigurationError, match=(
            "operand.*no source-form mapping.*chrsaliva_mar")):
        adapter.CrossChecks(
            pd.DataFrame([_rule_5_row()]),
            "baseline", "PRONET", {})


@pytest.mark.parametrize(
    "dependency_kwargs",
    [
        {"missing_important_forms": ("saliva",)},
        {"missing_converted": ("chrsaliva_mar",)},
    ],
    ids=["important-form-metadata", "converted-branching-metadata"],
)
def test_checker_bypasses_filter_only_dependency_metadata(
        monkeypatch, dependency_kwargs):
    variables = _rule_5_variables()
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHR"}},
        schedules={"chr": {"baseline": ["chs", "saliva"]}},
        variable_forms=variables,
        **dependency_kwargs,
    )

    checker = adapter.CrossChecks(
        pd.DataFrame([_rule_5_row()]),
        "baseline", "PRONET", {})

    assert [row["subject"] for row in checker()] == ["S1"]


def test_checker_bypasses_explicitly_excluded_operand_branching(
        monkeypatch):
    variables = _rule_5_variables()
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHR"}},
        schedules={"chr": {"baseline": ["chs", "saliva"]}},
        variable_forms=variables,
        excluded_logic={"chrsaliva_mar": "unsupported"},
    )

    checker = adapter.CrossChecks(
        pd.DataFrame([_rule_5_row()]),
        "baseline", "PRONET", {})

    assert [row["subject"] for row in checker()] == ["S1"]


def test_checker_bypasses_branching_expression_and_its_missing_input(
        monkeypatch):
    variables = _rule_5_variables()
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHR"}},
        schedules={"chr": {"baseline": ["chs", "saliva"]}},
        variable_forms=variables,
        converted_logic={
            "chrsaliva_mar": {
                "converted_branching_logic": (
                    "hasattr(curr_row, 'saliva_gate') and "
                    "curr_row.saliva_gate == 1"),
            },
        },
    )
    # saliva_gate is intentionally absent, but the raw catalog predicate hits.
    checker = adapter.CrossChecks(
        pd.DataFrame([_rule_5_row()]), "baseline", "PRONET", {})

    assert [row["subject"] for row in checker()] == ["S1"]


@pytest.mark.parametrize(
    "column",
    ["visit_status_string", "recruitment_status_v2"],
)
def test_checker_keeps_required_output_column_contract(
        monkeypatch, column):
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHR"}},
        schedules={"chr": {"baseline": ["chs", "saliva"]}},
        variable_forms=_rule_5_variables(),
    )
    row = _rule_5_row()
    row.pop(column)

    with pytest.raises(adapter.CrossCheckInputError, match=(
            f"required output column.*{column}")):
        adapter.CrossChecks(
            pd.DataFrame([row]), "baseline", "PRONET", {})


def test_checker_does_not_require_row_visit_status_for_applicability(
        monkeypatch):
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHR"}},
        schedules={},
        variable_forms=_rule_5_variables(),
    )
    row = _rule_5_row()
    row.pop("visit_status")

    checker = adapter.CrossChecks(
        pd.DataFrame([row]), "baseline", "PRONET", {})

    assert [output["subject"] for output in checker()] == ["S1"]


@pytest.mark.parametrize("column", ["saliva_complete", "saliva_missing"])
def test_checker_bypasses_missing_form_gate_columns(
        monkeypatch, column):
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHR"}},
        schedules={"chr": {"baseline": ["chs", "saliva"]}},
        variable_forms=_rule_5_variables(),
    )
    row = _rule_5_row()
    row.pop(column)

    checker = adapter.CrossChecks(
        pd.DataFrame([row]), "baseline", "PRONET", {})

    assert [output["subject"] for output in checker()] == ["S1"]


@pytest.mark.parametrize(
    ("column", "value"),
    [("saliva_complete", 0), ("saliva_missing", 1)],
    ids=["incomplete-form", "explicitly-missing-form"],
)
def test_checker_bypasses_form_completion_and_missingness_state(
        monkeypatch, column, value):
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHR"}},
        schedules={"chr": {"baseline": ["chs", "saliva"]}},
        variable_forms=_rule_5_variables(),
    )
    row = _rule_5_row()
    row[column] = value

    checker = adapter.CrossChecks(
        pd.DataFrame([row]), "baseline", "PRONET", {})

    assert [output["subject"] for output in checker()] == ["S1"]


def test_checker_bypasses_blank_completion_mapping(monkeypatch):
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHR"}},
        schedules={"chr": {"baseline": ["chs", "saliva"]}},
        variable_forms=_rule_5_variables(),
        blank_completion_forms=("saliva",),
    )

    checker = adapter.CrossChecks(
        pd.DataFrame([_rule_5_row()]),
        "baseline", "PRONET", {})

    assert [row["subject"] for row in checker()] == ["S1"]


def test_checker_bypasses_prescient_adjusted_completion_column(monkeypatch):
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHR"}},
        schedules={"chr": {"baseline": ["chs", "saliva"]}},
        variable_forms=_rule_5_variables(),
    )
    row = _rule_5_row()
    row["chs_complete_rpms"] = 2
    # Deliberately omit the network-adjusted saliva completion field while
    # retaining the base field used by ProNET.

    checker = adapter.CrossChecks(
        pd.DataFrame([row]), "baseline", "PRESCIENT", {})

    assert [output["subject"] for output in checker()] == ["S1"]


@pytest.mark.parametrize("field", ["inclusion_status", "visit_status"])
def test_checker_fails_early_for_missing_subject_routing_metadata(
        monkeypatch, field):
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects={"S1": {"cohort": "CHR", field: ""}},
        schedules={"chr": {"baseline": ["chs", "saliva"]}},
        variable_forms=_rule_5_variables(),
    )

    with pytest.raises(adapter.CrossCheckConfigurationError, match=(
            f"subject_info.*output field.*{field}")):
        adapter.CrossChecks(
            pd.DataFrame([_rule_5_row()]),
            "baseline", "PRONET", {})


@pytest.mark.parametrize("age_value", [None, "unknown", 19])
def test_pubertal_rule_bypasses_special_form_age_condition(
        monkeypatch, age_value):
    variables = {
        "chrpds_interview_date": "pubertal_developmental_scale",
        "chrpds_pds_f5b_p": "pubertal_developmental_scale",
        "chrchs_interview_date": "current_health_status",
        "chrchs_mens": "current_health_status",
    }
    forms = set(variables.values())
    subject = {"cohort": "CHR"}
    missing_metadata = ()
    if age_value is None:
        missing_metadata = ("age",)
    else:
        subject["age"] = age_value
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(31),),
        subjects={"S1": subject},
        schedules={"chr": {"baseline": sorted(forms)}},
        variable_forms=variables,
        missing_subject_metadata=missing_metadata,
    )
    row = {
        "subjectid": "S1",
        "chrpds_interview_date": "2025-01-01",
        "chrpds_pds_f5b_p": 1,
        "chrchs_interview_date": "2025-01-01",
        "chrchs_mens": 1,
        "chrdemo_sexassigned": 2,
        **RUNTIME_ROW_FIELDS,
    }
    for form in forms:
        row[f"{form}_complete"] = 2
        row[f"{form}_missing"] = 0

    checker = adapter.CrossChecks(
        pd.DataFrame([row]), "baseline", "PRONET", {})
    assert checker.rule_status["CROSS-QC-031"] == ("evaluated", 1)
    assert [output["subject"] for output in checker()] == ["S1"]


def test_checker_bypasses_numeric_sentinel_mask_but_keeps_date_normalization(
        monkeypatch):
    catalog = (_catalog_rule(9),)
    variables = {
        "chrbprs_interview_date": "bprs",
        "chrbprs_bprs_suic": "bprs",
        "chrcdss_interview_date": "cdss",
        "chrcdss_calg8": "cdss",
    }
    subjects = {
        subject: {"cohort": "CHR"} for subject in ("VALID", "NUM", "DATE")
    }
    schedules = {"chr": {"baseline": ["bprs", "cdss"]}}
    _install_minimal_form_check(
        monkeypatch,
        catalog=catalog,
        subjects=subjects,
        schedules=schedules,
        variable_forms=variables,
    )
    rows = pd.DataFrame([
        dict(subjectid="VALID", chrbprs_interview_date="2025-01-01",
             chrcdss_interview_date="2025-01-01", chrcdss_calg8=0,
             chrbprs_bprs_suic=4, bprs_complete=2, bprs_missing=0,
             cdss_complete=2, cdss_missing=0, **RUNTIME_ROW_FIELDS),
        # The expression itself treats 999 as > 3; no observation mask may
        # suppress it on the intentionally unfiltered review surface.
        dict(subjectid="NUM", chrbprs_interview_date="2025-01-01",
             chrcdss_interview_date="2025-01-01", chrcdss_calg8=0,
             chrbprs_bprs_suic=999, bprs_complete=2, bprs_missing=0,
             cdss_complete=2, cdss_missing=0, **RUNTIME_ROW_FIELDS),
        dict(subjectid="DATE", chrbprs_interview_date="1901-01-01",
             chrcdss_interview_date="1901-01-01", chrcdss_calg8=0,
             chrbprs_bprs_suic=4, bprs_complete=2, bprs_missing=0,
             cdss_complete=2, cdss_missing=0, **RUNTIME_ROW_FIELDS),
    ])

    checker = adapter.CrossChecks(rows, "baseline", "PRONET", {})

    # Date parsing remains normalization, not an applicability filter, so the
    # 1901 sentinel becomes NaT and the date comparison itself is false.
    assert [row["subject"] for row in checker()] == ["VALID", "NUM"]
    assert checker.rule_status["CROSS-QC-009"] == ("evaluated", 2)
    assert checker.rule_metrics["CROSS-QC-009"]["filters_bypassed"] is True


def test_checker_bypasses_inactive_operand_branch(
        monkeypatch):
    """The catalog predicate, not current REDCap visibility, controls output."""
    variables = {
        "chrchs_interview_date": "chs",
        "chrchs_mar": "chs",
        "chrsaliva_interview_date": "saliva",
        "chrsaliva_mar": "saliva",
    }
    subjects = {
        "INACTIVE": {"cohort": "CHR"},
        "ACTIVE": {"cohort": "CHR"},
    }
    _install_minimal_form_check(
        monkeypatch,
        catalog=(_catalog_rule(5),),
        subjects=subjects,
        schedules={"chr": {"baseline": ["chs", "saliva"]}},
        variable_forms=variables,
        converted_logic={
            "chrsaliva_mar": {
                "converted_branching_logic": "curr_row.saliva_gate == 1",
            },
        },
    )
    common = {
        "chrchs_interview_date": "2025-06-01",
        "chrsaliva_interview_date": "2025-06-01",
        "chrchs_mar": 0,
        "chrsaliva_mar": 1,
        "chs_complete": 2,
        "chs_missing": 0,
        "saliva_complete": 2,
        "saliva_missing": 0,
        **RUNTIME_ROW_FIELDS,
    }
    rows = pd.DataFrame([
        {"subjectid": "INACTIVE", "saliva_gate": 0, **common},
        {"subjectid": "ACTIVE", "saliva_gate": 1, **common},
    ])

    checker = adapter.CrossChecks(rows, "baseline", "PRONET", {})

    assert checker.rule_status["CROSS-QC-005"] == ("evaluated", 2)
    assert [row["subject"] for row in checker()] == ["INACTIVE", "ACTIVE"]
    assert checker.rule_metrics["CROSS-QC-005"]["suppressed_branching"] == 0


def test_nitrous_category_mismatch_uses_composite_assist_scid_evidence():
    """ASSIST inhalant can map to SCID inhalant or SCID Other (nitrous)."""
    dates = {
        "chrassist_interview_date": ["2025-03-01"] * 3,
        "chrscid_interview_date": ["2025-03-01"] * 3,
    }

    assist_inhalant_yes = pd.DataFrame({
        **dates,
        "chrassist_whoassist_use6": [1, 1, 1],
        "chrscid_inhalant_yn": [0, 0, 1],
        "chrscid_othersub_yn": [0, 1, 0],
    })
    prepared_100 = source.prep_df_for_rule(
        assist_inhalant_yes, source.RULES[99])
    # Only a No on both SCID destinations is contradictory.  SCID Other=Yes
    # is consistent with ASSIST inhalant=Yes when the substance is nitrous.
    assert source.evaluate_rule_mask(prepared_100, 100).tolist() == [
        True, False, False]

    scid_other_yes = pd.DataFrame({
        **dates,
        "chrassist_whoassist_use10": [0, 0, 1],
        "chrassist_whoassist_use6": [0, 1, 0],
        "chrscid_othersub_yn": [1, 1, 1],
    })
    prepared_107 = source.prep_df_for_rule(
        scid_other_yes, source.RULES[106])
    # SCID Other=Yes conflicts only when both possible ASSIST destinations
    # (Other and inhalant/nitrous) are No.
    assert source.evaluate_rule_mask(prepared_107, 107).tolist() == [
        True, False, False]

    assert set(_catalog_rule(100).variables) >= {
        "chrassist_whoassist_use6",
        "chrscid_inhalant_yn",
        "chrscid_othersub_yn",
    }
    # Keep the corrected source expression for traceability, but do not place
    # this reverse comparison in the operational catalog: SCID Other includes
    # medicines/OTC exposures whereas ASSIST asks non-medical use only.
    assert set(source.extract_columns(source.RULES[106])) >= {
        "chrassist_whoassist_use10",
        "chrassist_whoassist_use6",
        "chrscid_othersub_yn",
    }
    assert 107 in source.DISABLED_RULE_REASONS
    assert 107 not in source.ENABLED_RULE_NUMBERS
    assert all(rule.number != 107 for rule in adapter.RULE_CATALOG)


def test_rule_98_uses_only_the_unambiguous_cocaine_arm(monkeypatch):
    """Ecstasy/diet pills make the ASSIST amphetamine arm ambiguous."""
    rule = _catalog_rule(98)
    assert rule.alternative_evidence_groups == ()
    assert 98 not in source.ALTERNATIVE_EVIDENCE_GROUPS
    assert "chrassist_whoassist_use5" not in rule.variables

    variables = {
        "chrassist_interview_date": "assist",
        "chrassist_whoassist_use4": "assist",
        "chrscid_interview_date": "scid",
        "chrscid_stimulant_yn": "scid",
    }
    subject_ids = (
        "COCAINE_ONLY",
        "COCAINE_NO",
        "COCAINE_MISSING",
        "MANDATORY_MISSING",
    )
    _install_minimal_form_check(
        monkeypatch,
        catalog=(rule,),
        subjects={subject: {"cohort": "CHR"} for subject in subject_ids},
        schedules={"chr": {"baseline": ["assist", "scid"]}},
        variable_forms=variables,
    )
    common = {
        "chrassist_interview_date": "2025-02-01",
        "chrscid_interview_date": "2025-02-01",
        "chrscid_stimulant_yn": 0,
        "assist_complete": 2,
        "assist_missing": 0,
        "scid_complete": 2,
        "scid_missing": 0,
        **RUNTIME_ROW_FIELDS,
    }
    rows = pd.DataFrame([
        {"subjectid": "COCAINE_ONLY", "chrassist_whoassist_use4": 1,
         **common},
        {"subjectid": "COCAINE_NO", "chrassist_whoassist_use4": 0,
         **common},
        {"subjectid": "COCAINE_MISSING", "chrassist_whoassist_use4": -9,
         **common},
        {"subjectid": "MANDATORY_MISSING", "chrassist_whoassist_use4": 1,
         **{**common, "chrscid_stimulant_yn": None}},
    ])

    checker = adapter.CrossChecks(rows, "baseline", "PRONET", {})

    assert checker.rule_status["CROSS-QC-098"] == ("evaluated", 1)
    assert [row["subject"] for row in checker()] == ["COCAINE_ONLY"]
