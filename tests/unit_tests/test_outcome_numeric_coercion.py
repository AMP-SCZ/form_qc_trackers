import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_float_dtype, is_integer_dtype

from outcome_aggregation import coerce_outcome_numeric


def test_integer_coercion_preserves_shape_and_normalizes_invalid_cells() -> None:
    source = pd.DataFrame(
        {
            "left": [
                " 42 ",
                "",
                "   ",
                np.nan,
                pd.NA,
                "not a number",
                "1903-03-03",
                "1909-09-09",
                "-300",
            ],
            "right": [
                "7",
                None,
                ".",
                "N/A",
                "arbitrary text",
                "999",
                "1903-03-03T00:00:00",
                "1909-09-09T00:00:00",
                "-900",
            ],
        },
        index=[f"row-{index}" for index in range(9)],
    )

    result = coerce_outcome_numeric(source, "int")

    assert result.columns.tolist() == source.columns.tolist()
    assert result.index.tolist() == source.index.tolist()
    assert result.values.tolist() == [
        [42, 7],
        [-900, -900],
        [-900, -900],
        [-900, -900],
        [-900, -900],
        [-900, 999],
        [-300, -300],
        [-900, -900],
        [-300, -900],
    ]
    assert all(is_integer_dtype(dtype) for dtype in result.dtypes)
    # The helper is an aggregation boundary and must not rewrite its caller's
    # raw outcome frame in place.
    assert source.loc["row-0", "left"] == " 42 "
    assert source.loc["row-6", "left"] == "1903-03-03"


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("1903-03-03", -300),
        ("1903-03-03T00:00:00", -300),
        ("1903-03-03 00:00:00", -300),
        ("1909-09-09", -900),
        ("1909-09-09T00:00:00", -900),
        ("1909-09-09 00:00:00", -900),
    ],
)
def test_date_sentinel_spellings_survive_numeric_coercion(
    raw_value: str,
    expected: int,
) -> None:
    result = coerce_outcome_numeric(pd.DataFrame({"value": [raw_value]}), "int")

    assert result.loc[0, "value"] == expected


def test_float_coercion_casts_numeric_strings_and_preserves_numeric_sentinels(
) -> None:
    source = pd.DataFrame(
        {
            "value": [
                "3.25",
                " 1e2 ",
                8,
                8.5,
                "-300",
                -900,
                "999",
                "bad",
            ]
        }
    )

    result = coerce_outcome_numeric(source, "float")

    assert result["value"].tolist() == [
        3.25,
        100.0,
        8.0,
        8.5,
        -300.0,
        -900.0,
        999.0,
        -900.0,
    ]
    assert is_float_dtype(result["value"].dtype)
