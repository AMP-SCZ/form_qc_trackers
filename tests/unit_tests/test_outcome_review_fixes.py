"""Regression coverage for the 2026-08-22 outcome-calculation bug fixes.

Each test pins one confirmed finding from OUTCOME_CALCULATIONS_REVIEW.md
(C1-C8) using synthetic frames only.
"""

from datetime import date

import pandas as pd
import pytest

from outcome_aggregation import coerce_outcome_numeric, reverse_outcome_numeric
from outcome_calculations import (
    all_visits_list,
    compute_outcomes,
    create_condition_value,
    create_max_scid,
    create_min_date,
    create_total_division,
    create_use_value,
)
from outcome_inputs import OutcomeInputError
from outcome_scoring import pps_age_sex_score


def _events_frame(events, **columns):
    frame = pd.DataFrame({"redcap_event_name": list(events)})
    for name, values in columns.items():
        frame[name] = values
    return frame


def test_c1_helpers_do_not_mutate_the_caller_frame():
    events = ["baseline_arm_1", "month_1_arm_1"]
    frame = _events_frame(
        events, field_a=["2", "3"], date_a=["2024-01-01", "1909-09-09"]
    )
    original = frame.copy(deep=True)

    create_use_value("out_a", frame, frame, ["field_a"], "basel",
                     all_visits_list, "int")
    create_condition_value("out_b", frame, frame, "basel",
                           all_visits_list, "int", -300)
    create_min_date("out_c", frame, frame, ["date_a"], "basel",
                    all_visits_list, "str")

    assert "value" not in frame.columns
    pd.testing.assert_frame_equal(frame, original)


def test_c1_scid_rollup_value_survives_prior_helper_calls():
    events = ["baseline_arm_1", "month_1_arm_1"]
    df_all = _events_frame(events, field_a=["", ""])
    # The pre-fix failure mode: this call leaked df_all['value'] = -900.
    create_use_value("earlier_outcome", df_all, df_all, ["field_a"], "basel",
                     all_visits_list, "int")

    condition = _events_frame(events, value=[1, 0])
    rollup = create_use_value("chrscid_any_psychosis", condition, df_all,
                              ["value"], "basel", all_visits_list, "int")

    baseline_value = rollup.loc[
        rollup["redcap_event_name"] == "baseline_arm_1", "value"
    ].iloc[0]
    assert baseline_value == 1


def test_c2_max_scid_missing_codes_never_become_severity():
    events = ["baseline_arm_1", "month_1_arm_1"]
    frame = _events_frame(events, e1=["999", "-9"], e2=["2", "999"])

    result = create_max_scid("max_substance", frame, frame, ["e1", "e2"],
                             "basel", all_visits_list, "int")
    by_event = result.set_index("redcap_event_name")["value"]

    # A valid criteria count wins over a co-occurring missing code...
    assert by_event["baseline_arm_1"] == 2
    # ...and an all-missing-code row is missing, never max(999, -9).
    assert by_event["month_1_arm_1"] == -300  # month_1 outside 'basel' voi
    assert (result["value"] != 999).all()


def test_c3_all_junk_date_row_is_missing_not_literal_nat():
    events = ["baseline_arm_1"]
    frame = _events_frame(events, d1=["-900"], d2=["-3"])

    result = create_min_date("onset", frame, frame, ["d1", "d2"], "basel",
                             all_visits_list, "str")
    baseline_value = result.loc[
        result["redcap_event_name"] == "baseline_arm_1", "value"
    ].iloc[0]

    assert baseline_value == date(1909, 9, 9)


def test_c4_arm_fix_applies_without_any_voi_matching_visit():
    # An HC (arm_2) subject with only screening/baseline visits and a voi
    # that matches none of them: the other arm's rows must read -300, not
    # -900.
    events = ["screening_arm_2", "baseline_arm_2"]
    frame = _events_frame(events, item=["1", "2"])
    voi = "month_2_|month_6_arm_1|month_12_arm_1"

    result = create_total_division("outcome_x", frame, frame, ["item"], 1,
                                   voi, all_visits_list, "int")
    by_event = result.set_index("redcap_event_name")["value"]

    assert by_event["month_6_arm_1"] == -300
    assert by_event["month_12_arm_1"] == -300


def test_c7_reversal_leaves_out_of_range_values_alone():
    frame = pd.DataFrame({"item": ["3", "-9", "999", "0"]})
    reversed_frame = reverse_outcome_numeric(frame, 4)

    assert reversed_frame["item"].tolist() == [1.0, -9.0, 999.0, 4.0]


def test_c7_coerce_int_treats_non_finite_as_missing():
    frame = pd.DataFrame({"item": ["inf", "-inf", "2"]})
    coerced = coerce_outcome_numeric(frame, "int")

    assert coerced["item"].tolist() == [-900, -900, 2]


def test_c8_pps_age_sex_boundary_at_exactly_36():
    assert pps_age_sex_score(36, "male") == 0
    assert pps_age_sex_score(35.99, "male") == 2
    assert pps_age_sex_score(37, "male") == 0


def test_c6_missing_age_subject_still_computes_and_bad_arm_raises(tmp_path):
    rows = [
        {"redcap_event_name": "screening_arm_1", "chrcrit_part": 1},
        {
            "redcap_event_name": "baseline_arm_1",
            "chrcrit_part": 1,
            "chrdemo_sexassigned": 2,
        },
        {"redcap_event_name": "month_1_arm_1", "chrcrit_part": 1},
    ]
    result = compute_outcomes(
        "AA00003",
        subject_data=pd.DataFrame(rows),
        network_name="PRONET",
        run_version="run_outcome",
        output_dir=tmp_path,
    )
    assert result is not None and not result.empty

    with pytest.raises(OutcomeInputError, match="neither arm_1 nor arm_2"):
        compute_outcomes(
            "AA00004",
            subject_data=pd.DataFrame(
                [{"redcap_event_name": "baseline", "chrcrit_part": 1}]
            ),
            network_name="PRONET",
            run_version="run_outcome",
            output_dir=tmp_path,
        )
