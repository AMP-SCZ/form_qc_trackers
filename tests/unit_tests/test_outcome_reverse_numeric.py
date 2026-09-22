import pandas as pd

from outcome_aggregation import reverse_outcome_numeric


def test_reverse_outcome_numeric_preserves_missing_values() -> None:
    source = pd.DataFrame({
        "item": ["1", "1903-03-03T00:00:00", "1909-09-09", "bad"],
    })

    result = reverse_outcome_numeric(source, 4)

    assert result["item"].tolist() == [3.0, -300.0, -900.0, -900.0]
    assert source["item"].tolist() == [
        "1", "1903-03-03T00:00:00", "1909-09-09", "bad"
    ]
