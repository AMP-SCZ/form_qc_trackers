"""Regression coverage for Date Report form and history exclusions.

These tests intentionally exercise the pure comparison helper and the final
tracker-row presentation guard.  Together those are the two boundaries that
must reject excluded forms: new flags should not be generated, and historical
flags created before the exclusion was introduced should not reappear.
"""

import numpy as np
import pandas as pd
import pytest

from generate_reports.create_trackers import CreateTrackers
from qc_types.date_check_logic import find_backward_visit_dates


TP_ORDER = ["screening", "baseline"]
MISSING_VALUES = ("", "-9", "-99", "999")

EXCLUDED_FORM_PAIRS = (
    ("Past pharmaceutical", "past_pharmaceutical_treatment"),
    ("PSYCHS AV recording run sheet", "psychs_av_recording_run_sheet"),
    ("MRI-Incidental", "mri_incidental_findings_run_sheet"),
    ("CurrentHealthStatus_GCP", "gcp_current_health_status"),
    ("CBC with differential", "cbc_with_differential"),
    ("GCP-CBC with differential", "gcp_cbc_with_differential"),
)
EXCLUDED_FORM_TOKENS = tuple(
    token for pair in EXCLUDED_FORM_PAIRS for token in pair
)
EXCLUDED_DATE_VARIABLES = (
    "chrpharm_interview_date",
    "chrpsychs_av_interview_date",
    "chrif_interview_date",
    "chrgpc_date",
    "chrcbc_interview_date",
    "chrcbccs_review_date",
)


def _observation(date, form, variable, network="PRONET"):
    return {
        "date": date,
        "form": form,
        "variable": variable,
        "network": network,
    }


@pytest.mark.parametrize("excluded_form", EXCLUDED_FORM_TOKENS)
def test_excluded_alias_or_canonical_form_cannot_be_current_observation(
        excluded_form):
    dates = {
        "screening": {
            "dates": [
                _observation(
                    "2025-02-01", "allowed_prior", "neutral_date")
            ]
        }
    }
    current = [
        _observation(
            "2025-01-01", excluded_form, "neutral_date")
    ]

    assert find_backward_visit_dates(
        dates,
        TP_ORDER,
        "baseline",
        missing_values=MISSING_VALUES,
        network="PRONET",
        current_observations=current,
    ) == []


@pytest.mark.parametrize("excluded_form", EXCLUDED_FORM_TOKENS)
def test_excluded_alias_or_canonical_form_cannot_be_prior_observation(
        excluded_form):
    dates = {
        "screening": {
            "dates": [
                _observation(
                    "2025-02-01", excluded_form, "neutral_date")
            ]
        }
    }
    current = [
        _observation(
            "2025-01-01", "allowed_current", "neutral_date")
    ]

    assert find_backward_visit_dates(
        dates,
        TP_ORDER,
        "baseline",
        missing_values=MISSING_VALUES,
        network="PRONET",
        current_observations=current,
    ) == []


@pytest.mark.parametrize("column", ("displayed_form", "affected_forms"))
@pytest.mark.parametrize(
    "display_alias", [pair[0] for pair in EXCLUDED_FORM_PAIRS]
)
def test_tracker_drops_every_display_alias_from_either_form_column(
        display_alias, column):
    excluded_row = {
        "row_id": "excluded",
        "displayed_form": "allowed_current",
        "affected_forms": "allowed_current | allowed_prior",
    }
    excluded_row[column] = (
        display_alias
        if column == "displayed_form"
        else f"allowed_current | {display_alias}"
    )
    frame = pd.DataFrame([
        excluded_row,
        {
            "row_id": "allowed",
            "displayed_form": "allowed_current",
            "affected_forms": "allowed_current | allowed_prior",
        },
    ])

    result = CreateTrackers._exclude_date_report_form_rows(frame)

    assert result["row_id"].tolist() == ["allowed"]


@pytest.mark.parametrize("display_alias", [pair[0] for pair in EXCLUDED_FORM_PAIRS])
def test_tracker_drops_numpy_style_serialized_form_strings(display_alias):
    serialized = str(np.array(["allowed_current", display_alias]))
    assert "," not in serialized
    frame = pd.DataFrame([
        {
            "row_id": "excluded",
            "displayed_form": "allowed_current",
            "affected_forms": serialized,
        },
        {
            "row_id": "allowed",
            "displayed_form": "allowed_current",
            "affected_forms": str(np.array(["allowed_current", "allowed_prior"])),
        },
    ])

    result = CreateTrackers._exclude_date_report_form_rows(frame)

    assert result["row_id"].tolist() == ["allowed"]


@pytest.mark.parametrize(
    "column", ("affected_variables", "displayed_variable", "error_message")
)
@pytest.mark.parametrize("excluded_variable", EXCLUDED_DATE_VARIABLES)
def test_tracker_drops_historical_rows_identified_only_by_date_variable(
        excluded_variable, column):
    excluded_row = {
        "row_id": "excluded",
        "displayed_form": "allowed_current",
        "affected_forms": "allowed_current | allowed_prior",
        "affected_variables": "allowed_current_date | allowed_prior_date",
        "displayed_variable": "allowed_current_date",
        "error_message": "ordinary allowed date flag",
    }
    if column == "affected_variables":
        excluded_row[column] = (
            f"allowed_current_date | {excluded_variable}"
        )
    elif column == "displayed_variable":
        excluded_row[column] = excluded_variable
    else:
        excluded_row[column] = (
            "allowed_current_date : Visit date at baseline (2025-01-01) "
            f"is 31 day(s) before {excluded_variable} (2025-02-01) at the "
            "earlier timepoint screening."
        )
    frame = pd.DataFrame([
        excluded_row,
        {
            "row_id": "allowed",
            "displayed_form": "allowed_current",
            "affected_forms": "allowed_current | allowed_prior",
            "affected_variables": (
                "allowed_current_date | allowed_prior_date"
            ),
            "displayed_variable": "allowed_current_date",
            "error_message": "ordinary allowed date flag",
        },
    ])

    result = CreateTrackers._exclude_date_report_form_rows(frame)

    assert result["row_id"].tolist() == ["allowed"]


def test_regular_current_health_status_remains_in_date_report():
    frame = pd.DataFrame([{
        "row_id": "allowed",
        "displayed_form": "current_health_status",
        "affected_forms": "current_health_status | allowed_prior",
        "affected_variables": (
            "chrchs_interview_date | allowed_prior_date"
        ),
        "displayed_variable": "chrchs_interview_date",
        "error_message": (
            "chrchs_interview_date : Visit date at baseline (2025-01-01) "
            "is 31 day(s) before allowed_prior_date (2025-02-01) at the "
            "earlier timepoint screening."
        ),
    }])

    result = CreateTrackers._exclude_date_report_form_rows(frame)

    assert result["row_id"].tolist() == ["allowed"]


def test_regular_current_health_status_can_generate_same_variable_flag():
    dates = {
        "screening": {
            "dates": [_observation(
                "2025-02-01", "current_health_status",
                "chrchs_interview_date")]
        }
    }
    current = [_observation(
        "2025-01-01", "current_health_status",
        "chrchs_interview_date")]

    results = find_backward_visit_dates(
        dates,
        TP_ORDER,
        "baseline",
        missing_values=MISSING_VALUES,
        network="PRONET",
        current_observations=current,
    )

    assert len(results) == 1
    assert results[0]["current_variable"] == "chrchs_interview_date"
    assert results[0]["prev_variable"] == "chrchs_interview_date"
