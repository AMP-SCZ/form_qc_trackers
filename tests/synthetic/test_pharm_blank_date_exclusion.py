"""Blank-only medication interview-date exclusion, using synthetic inputs."""

import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.realpath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _qc_harness import install_shim  # noqa: E402

install_shim()

from qc_forms.qc_types.general_checks import GeneralChecks  # noqa: E402
from process_variables.organize_reports import OrganizeReports  # noqa: E402


@pytest.mark.parametrize("network", ["PRONET", "PRESCIENT"])
@pytest.mark.parametrize("report", ["Main Report", "Secondary Report"])
@pytest.mark.parametrize("form_passes_filter", [True, False])
def test_stale_blank_dependencies_skip_interview_date_only(
        network, report, form_passes_filter):
    form = "past_pharmaceutical_treatment"
    date_var = "chrpharm_interview_date"
    medication_var = "chrpharm_med1_name_past"
    checker = object.__new__(GeneralChecks)
    checker.network = network
    checker.general_check_vars = {
        "blank_check_vars": {
            network: {"Main Report": {}, "Secondary Report": {}}},
        "excluded_forms": {network: []},
    }
    checker.general_check_vars["blank_check_vars"][network][report] = {
        form: [date_var, medication_var]}
    checker.forms_per_report = {}
    checker.standard_form_filter = lambda row, form: form_passes_filter
    checker.prescient_scid_filter = lambda var, row: False
    checker.excl_bl = {}
    # An excluded date must never reach branching-logic evaluation, even with
    # old dependency lists. The medication field still follows its real path.
    checker.conv_bl = {medication_var: {"converted_branching_logic": ""}}
    checker.final_output_list = []
    checker.create_row_output = lambda row, forms, variables, message, changes: {
        "affected_variables": variables,
        "error_message": message,
        **changes,
    }
    row = SimpleNamespace(**{date_var: "", medication_var: ""})

    checker.check_blank_values(row)

    assert checker.final_output_list == [{
        "affected_variables": [medication_var],
        "error_message": "Variable is blank.",
        "reports": [report],
        "priority_item": True,
    }]


@pytest.mark.parametrize("field_type", ["text", "notes"])
@pytest.mark.parametrize("explicit_addition", [False, True])
def test_generated_blank_lists_exclude_interview_date_only(
        field_type, explicit_addition):
    organizer = object.__new__(OrganizeReports)
    form = "past_pharmaceutical_treatment"
    organizer.all_psychs_forms = []
    organizer.self_report_forms = []
    organizer.data_dict_df = pd.DataFrame([
        {"Variable / Field Name": "chrpharm_interview_date",
         "Form Name": form, "Field Type": field_type,
         "Identifier?": "", "Required Field?": "y"},
        {"Variable / Field Name": "chrpharm_med1_name_past",
         "Form Name": form, "Field Type": "text",
         "Identifier?": "", "Required Field?": "y"},
        {"Variable / Field Name": "other_interview_date",
         "Form Name": "other_form", "Field Type": "text",
         "Identifier?": "", "Required Field?": "y"},
    ])
    if explicit_addition:
        organizer.define_additional_blank_check_vars = (
            lambda: ["chrpharm_interview_date", "chrpharm_med1_name_past"])

    generated = organizer.organize_blank_check_vars()

    for network in ("PRONET", "PRESCIENT"):
        assert generated[network]["Main Report"][form] == [
            "chrpharm_med1_name_past"]
        assert generated[network]["Main Report"]["other_form"] == [
            "other_interview_date"]
        for report in generated[network].values():
            assert all("chrpharm_interview_date" not in variables
                       for variables in report.values())
