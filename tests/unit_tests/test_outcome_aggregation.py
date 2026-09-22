import pandas as pd
import pytest

from outcome_aggregation import combine_scid_conditions
from outcome_inputs import OutcomeInputError


def _condition(values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variable": ["condition"] * len(values),
            "redcap_event_name": [f"event_{index}" for index in range(len(values))],
            "value": values,
        }
    )


def test_binary_scid_conditions_combine_without_column_collisions() -> None:
    combined = combine_scid_conditions(
        [
            _condition([0, -900, 0, 1]),
            _condition([0, 0, -900, -900]),
            _condition([0, -900, 1, 0]),
        ]
    )

    assert combined["value"].tolist() == [0, -900, 1, 1]
    assert combined.columns.tolist() == ["redcap_event_name", "value"]


def test_substance_scid_conditions_use_maximum_positive_value() -> None:
    combined = combine_scid_conditions(
        [_condition([0, -900, 2]), _condition([0, 0, 3])],
        maximum=True,
    )

    assert combined["value"].tolist() == [0, -900, 3]


def test_duplicate_scid_events_are_rejected() -> None:
    frame = _condition([0, 1])
    frame["redcap_event_name"] = "same_event"

    with pytest.raises(OutcomeInputError, match="duplicate REDCap events"):
        combine_scid_conditions([frame])


def test_scid_conditions_reject_missing_event_without_naming_events() -> None:
    first = _condition([0, 1])
    second = _condition([0])

    with pytest.raises(OutcomeInputError) as error:
        combine_scid_conditions([first, second])

    message = str(error.value)
    assert "must have identical REDCap event indexes" in message
    assert "frame 0 count=2, frame 1 count=1" in message
    assert "event_0" not in message
    assert "event_1" not in message


def test_scid_conditions_align_reordered_events() -> None:
    first = _condition([0, 0])
    second = _condition([1, 0]).iloc[::-1].reset_index(drop=True)

    combined = combine_scid_conditions([first, second])

    assert combined["redcap_event_name"].tolist() == ["event_0", "event_1"]
    assert combined["value"].tolist() == [1, 0]


def test_scid_conditions_reject_missing_event_keys() -> None:
    frame = _condition([0, 1])
    frame.loc[1, "redcap_event_name"] = None

    with pytest.raises(OutcomeInputError) as error:
        combine_scid_conditions([frame])

    message = str(error.value)
    assert "frame 0 missing count=1" in message
    assert "event_0" not in message
