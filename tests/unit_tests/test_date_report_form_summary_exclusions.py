import pandas as pd
import pytest

from date_report_form_summary import filter_excluded_date_flags


EXCLUDED_FORMS = (
    (
        "Past pharmaceutical",
        "past_pharmaceutical_treatment",
        "chrpharm_interview_date",
    ),
    (
        "PSYCHS AV recording run sheet",
        "psychs_av_recording_run_sheet",
        "chrpsychs_av_interview_date",
    ),
    (
        "MRI-Incidental",
        "mri_incidental_findings_run_sheet",
        "chrif_interview_date",
    ),
    (
        "CurrentHealthStatus_GCP",
        "gcp_current_health_status",
        "chrgpc_date",
    ),
    (
        "CBC with differential",
        "cbc_with_differential",
        "chrcbc_interview_date",
    ),
    (
        "GCP-CBC with differential",
        "gcp_cbc_with_differential",
        "chrcbccs_review_date",
    ),
)

EXCLUDED_NAMES = tuple(
    name
    for display_name, canonical_name, _ in EXCLUDED_FORMS
    for name in (display_name, canonical_name)
)


@pytest.mark.parametrize("excluded_name", EXCLUDED_NAMES)
@pytest.mark.parametrize("excluded_side", ("current", "previous"))
def test_raw_flag_is_discarded_when_either_form_is_excluded(
        excluded_name, excluded_side):
    current_form = (
        excluded_name if excluded_side == "current" else "allowed_current"
    )
    previous_form = (
        excluded_name if excluded_side == "previous" else "allowed_previous"
    )
    frame = pd.DataFrame({
        "row_id": ["excluded", "allowed"],
        "displayed_form": [current_form, "control_current"],
        "affected_forms": [
            repr([current_form, previous_form]),
            repr(["control_current", "control_previous"]),
        ],
    })

    result = filter_excluded_date_flags(frame)

    assert result["row_id"].tolist() == ["allowed"]
    assert result["affected_forms"].tolist() == [
        repr(["control_current", "control_previous"])
    ]


@pytest.mark.parametrize("excluded_name", EXCLUDED_NAMES)
def test_formatted_flag_is_discarded_when_current_form_is_excluded(
        excluded_name):
    variable_to_form = {"allowed_prior_date": "allowed_previous"}
    frame = pd.DataFrame({
        "row_id": ["excluded", "allowed"],
        "Form": [excluded_name, "control_current"],
        "Flags": [
            _formatted_message("allowed_prior_date"),
            _formatted_message("allowed_prior_date"),
        ],
        "Flag Count": [1, 1],
    })

    result = filter_excluded_date_flags(frame, variable_to_form)

    assert result["row_id"].tolist() == ["allowed"]


@pytest.mark.parametrize(
    ("excluded_form", "previous_date_variable"),
    [(canonical, variable) for _, canonical, variable in EXCLUDED_FORMS],
)
def test_formatted_flag_is_discarded_when_mapped_previous_form_is_excluded(
        excluded_form, previous_date_variable):
    variable_to_form = {
        previous_date_variable: excluded_form,
        "allowed_prior_date": "allowed_previous",
    }
    frame = pd.DataFrame({
        "row_id": ["excluded", "allowed"],
        "Form": ["allowed_current", "control_current"],
        "Flags": [
            _formatted_message(previous_date_variable),
            _formatted_message("allowed_prior_date"),
        ],
        "Flag Count": [1, 1],
    })

    result = filter_excluded_date_flags(frame, variable_to_form)

    assert result["row_id"].tolist() == ["allowed"]


def test_formatted_merged_row_keeps_only_allowed_individual_messages():
    excluded_message = _formatted_message("chrgpc_date")
    allowed_message = _formatted_message("allowed_prior_date")
    frame = pd.DataFrame({
        "row_id": ["partially_excluded"],
        "Form": ["allowed_current"],
        "Flags": [f"{excluded_message} | {allowed_message}"],
        "Flag Count": [2],
    })

    result = filter_excluded_date_flags(
        frame,
        {
            "chrgpc_date": "gcp_current_health_status",
            "allowed_prior_date": "allowed_previous",
        },
    )

    assert result["row_id"].tolist() == ["partially_excluded"]
    assert result["Flags"].tolist() == [allowed_message]
    assert result["Flag Count"].tolist() == [1]
    assert result.attrs["excluded_date_flags"] == 1


def _formatted_message(previous_date_variable):
    return (
        "current_date : Visit date at month2 (2024-01-01) is 5 day(s) "
        f"before {previous_date_variable} (2024-01-06) at the earlier "
        "timepoint month1."
    )
