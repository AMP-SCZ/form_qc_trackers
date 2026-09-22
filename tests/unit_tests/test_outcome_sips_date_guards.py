import pandas as pd
import pytest

from outcome_calculations import create_sips_groups, create_sips_groups_scr


@pytest.mark.parametrize(
    ("group_date", "conversion_date", "psychosis", "expected"),
    [
        ("1903-03-03", "1909-09-09", "1", -900),
        ("1909-09-09", "1903-03-03", "1", -900),
        ("2020-01-01", "2021-01-01", "1", 1),
        ("2022-01-01", "2021-01-01", "1", 0),
        ("1903-03-03", "1909-09-09", "0", 1),
    ],
)
def test_screening_sips_date_comparison_is_rowwise_and_sentinel_safe(
    group_date, conversion_date, psychosis, expected
) -> None:
    event = "screening_arm_1"
    frame = pd.DataFrame(
        {
            "redcap_event_name": [event],
            "group_onset": [group_date],
            "group_present": ["1"],
            "psychosis": [psychosis],
        }
    )
    onset = pd.DataFrame(
        {"redcap_event_name": [event], "value": [conversion_date]}
    )

    result = create_sips_groups_scr(
        "result",
        frame,
        ["group_onset"],
        "screening",
        [event],
        onset,
        ["group_present"],
        "psychosis",
    )

    assert result.loc[0, "value"] == expected


def test_followup_sips_rejects_sentinel_date_ordering() -> None:
    events = ["screening_arm_1", "baseline_arm_1"]
    frame = pd.DataFrame(
        {
            "redcap_event_name": events,
            "group_onset": [None, "1903-03-03"],
            "group_present": [None, "1"],
            "conversion": [None, "1"],
        }
    )
    screening = pd.DataFrame(
        {"redcap_event_name": events, "value": [-900, -300]}
    )
    conversion = pd.DataFrame(
        {
            "redcap_event_name": events,
            "value": ["1903-03-03", "1909-09-09"],
        }
    )

    current, _lifetime = create_sips_groups(
        "current",
        "lifetime",
        frame,
        screening,
        ["group_onset"],
        "baseline",
        events,
        conversion,
        ["group_present"],
        "conversion",
        "screening|baseline",
    )

    baseline_value = current.loc[
        current["redcap_event_name"].eq("baseline_arm_1"), "value"
    ].item()
    assert baseline_value == -900
