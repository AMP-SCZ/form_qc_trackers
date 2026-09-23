import pandas as pd
import pytest

from outcome_calculations import _merge_pas_adult_values


def test_pas_adult_merge_is_explicitly_optional() -> None:
    earlier_stages = pd.DataFrame(
        {
            'redcap_event_name': ['event_1'],
            'value_child': [0.1],
            'value_early': [0.2],
            'value_late': [0.3],
        }
    )

    assert _merge_pas_adult_values(earlier_stages, None) is None


def test_pas_adult_merge_propagates_structural_errors() -> None:
    earlier_stages = pd.DataFrame(
        {
            'redcap_event_name': ['event_1'],
            'value_child': [0.1],
        }
    )
    malformed_adult = pd.DataFrame({'value_adult': [0.4]})

    with pytest.raises(KeyError, match='redcap_event_name'):
        _merge_pas_adult_values(earlier_stages, malformed_adult)


def test_pas_adult_merge_includes_adulthood_values() -> None:
    earlier_stages = pd.DataFrame(
        {
            'redcap_event_name': ['event_1'],
            'value_child': [0.1],
        }
    )
    adulthood = pd.DataFrame(
        {
            'redcap_event_name': ['event_1'],
            'value_adult': [0.4],
        }
    )

    merged = _merge_pas_adult_values(earlier_stages, adulthood)

    assert merged.to_dict('records') == [
        {
            'redcap_event_name': 'event_1',
            'value_child': 0.1,
            'value_adult': 0.4,
        }
    ]
