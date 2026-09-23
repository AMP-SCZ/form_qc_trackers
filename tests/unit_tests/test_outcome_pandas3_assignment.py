import pandas as pd
from pandas.api.types import is_integer_dtype

from outcome_calculations import create_condition_value


def test_condition_value_replaces_string_column_with_numeric_dtype() -> None:
    event_names = pd.Series(["baseline_arm_1"], dtype="string")
    calculated = pd.DataFrame({"redcap_event_name": event_names})
    visits = pd.DataFrame({"redcap_event_name": event_names.copy()})

    result = create_condition_value(
        "condition_outcome",
        calculated,
        visits,
        "baseline",
        ["baseline_arm_1", "month_1_arm_1"],
        "int",
        1,
    )

    assert result["value"].tolist() == [1, -300]
    assert is_integer_dtype(result["value"].dtype)
