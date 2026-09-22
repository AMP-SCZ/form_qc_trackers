import pandas as pd
import pytest

from outcome_aggregation import merge_named_outcome_columns
from outcome_inputs import OutcomeInputError


def test_named_outcome_columns_merge_without_boilerplate_collisions() -> None:
    first = pd.DataFrame(
        {
            "variable": ["one", "one"],
            "redcap_event_name": ["event_1", "event_2"],
            "value": [10, 20],
            "sips_p1": [1, 2],
            "data_type": ["Integer", "Integer"],
        }
    )
    second = pd.DataFrame(
        {
            "variable": ["two", "two"],
            "redcap_event_name": ["event_1", "event_2"],
            "value": [30, 40],
            "sips_p2": [3, 4],
        }
    )

    merged = merge_named_outcome_columns([first, second])

    assert merged.columns.tolist() == [
        "redcap_event_name",
        "sips_p1",
        "sips_p2",
    ]
    assert merged[["sips_p1", "sips_p2"]].values.tolist() == [[1, 3], [2, 4]]


def test_named_outcome_columns_reject_duplicate_payload_names() -> None:
    frame = pd.DataFrame(
        {"redcap_event_name": ["event_1"], "named_value": [1]}
    )

    with pytest.raises(OutcomeInputError, match="duplicate named columns"):
        merge_named_outcome_columns([frame, frame.copy()])


@pytest.mark.parametrize(
    ("second_events", "expected_counts"),
    [
        (
            ["event_1"],
            "frame 0 count=2, frame 1 count=1, "
            "frame 0 only count=1, frame 1 only count=0",
        ),
        (
            ["event_1", "event_3"],
            "frame 0 count=2, frame 1 count=2, "
            "frame 0 only count=1, frame 1 only count=1",
        ),
    ],
    ids=["missing-event", "replacement-event"],
)
def test_named_outcome_columns_reject_nonidentical_event_indexes(
    second_events, expected_counts
) -> None:
    first = pd.DataFrame(
        {
            "redcap_event_name": ["event_1", "event_2"],
            "first_value": [1, 2],
        }
    )
    second = pd.DataFrame(
        {
            "redcap_event_name": second_events,
            "second_value": list(range(len(second_events))),
        }
    )

    with pytest.raises(OutcomeInputError) as error:
        merge_named_outcome_columns([first, second])

    message = str(error.value)
    assert "must have identical REDCap event indexes" in message
    assert expected_counts in message
    assert not any(event in message for event in {"event_1", "event_2", "event_3"})


def test_named_outcome_columns_align_reordered_events() -> None:
    first = pd.DataFrame(
        {
            "redcap_event_name": ["event_1", "event_2"],
            "first_value": [1, 2],
        }
    )
    second = pd.DataFrame(
        {
            "redcap_event_name": ["event_2", "event_1"],
            "second_value": [4, 3],
        }
    )

    merged = merge_named_outcome_columns([first, second])

    assert merged["redcap_event_name"].tolist() == ["event_1", "event_2"]
    assert merged[["first_value", "second_value"]].values.tolist() == [
        [1, 3],
        [2, 4],
    ]


def test_named_outcome_columns_reject_missing_event_keys() -> None:
    frame = pd.DataFrame(
        {
            "redcap_event_name": ["event_1", None],
            "named_value": [1, 2],
        }
    )

    with pytest.raises(OutcomeInputError) as error:
        merge_named_outcome_columns([frame])

    message = str(error.value)
    assert "frame 0 missing count=1" in message
    assert "event_1" not in message


def test_named_outcome_columns_reject_duplicate_events() -> None:
    frame = pd.DataFrame(
        {
            "redcap_event_name": ["event_1", "event_1"],
            "named_value": [1, 2],
        }
    )

    with pytest.raises(OutcomeInputError, match="duplicate REDCap events"):
        merge_named_outcome_columns([frame])
