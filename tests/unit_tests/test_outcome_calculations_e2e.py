from pathlib import Path

import pandas as pd
import pytest

from outcome_calculations import compute_outcomes
from outcome_checkpoints import CHECKPOINT_COLUMNS


def _minimal_events(
    arm: int,
    participant_value: int,
    age_years_column: str,
    age_months_column: str,
    sex: int,
) -> pd.DataFrame:
    """A synthetic subject with every source event required by calculations."""
    rows = [
        {
            "redcap_event_name": f"screening_arm_{arm}",
            "chrcrit_part": participant_value,
        },
        {
            "redcap_event_name": f"baseline_arm_{arm}",
            "chrcrit_part": participant_value,
            age_years_column: 20,
            age_months_column: 0,
            "chrdemo_sexassigned": sex,
        },
        {
            "redcap_event_name": f"month_1_arm_{arm}",
            "chrcrit_part": participant_value,
        },
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    (
        "subject_id",
        "arm",
        "participant_value",
        "age_years_column",
        "age_months_column",
        "sex",
    ),
    [
        ("AA00001", 1, 1, "chrdemo_age_yrs_chr", "chrdemo_age_mos_chr", 1),
        ("BB00002", 2, 2, "chrdemo_age_yrs_hc", "chrdemo_age_mos_hc", 2),
    ],
    ids=("chr-arm-1", "hc-arm-2"),
)
def test_compute_outcomes_writes_complete_checkpoint_ready_subject(
    tmp_path: Path,
    subject_id: str,
    arm: int,
    participant_value: int,
    age_years_column: str,
    age_months_column: str,
    sex: int,
) -> None:
    result = compute_outcomes(
        subject_id,
        subject_data=_minimal_events(
            arm,
            participant_value,
            age_years_column,
            age_months_column,
            sex,
        ),
        network_name="PRONET",
        run_version="run_outcome",
        output_dir=tmp_path,
    )

    assert result is not None
    assert tuple(result.columns) == CHECKPOINT_COLUMNS
    assert not result.empty
    assert set(result["ID"]) == {subject_id}

    # Since 2026-08-22 the combined per-network aggregate is the only outcome
    # deliverable: compute_outcomes must not write a per-subject measure tree.
    assert not (tmp_path / "subject_outcomes").exists()
