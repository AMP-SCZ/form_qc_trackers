import pandas as pd
import pytest

from outcome_calculations import create_total_division


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("1903-03-03T00:00:00", -300),
        ("not a numeric outcome", -900),
    ],
)
def test_total_division_coerces_combined_csv_values_before_aggregation(
    raw_value: str,
    expected: int,
) -> None:
    frame = pd.DataFrame(
        {
            "redcap_event_name": ["baseline_arm_1"],
            "test_item": [raw_value],
        }
    )

    result = create_total_division(
        "test_total",
        frame.copy(deep=True),
        frame.copy(deep=True),
        ["test_item"],
        1,
        "baseline",
        ["baseline_arm_1"],
        "int",
    )

    assert result.to_dict("records") == [
        {
            "variable": "test_total",
            "redcap_event_name": "baseline_arm_1",
            "value": expected,
        }
    ]
