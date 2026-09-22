from pathlib import Path

import pandas as pd
import pytest

from outcome_inputs import (
    OutcomeInputError,
    combined_csv_file,
    load_combined_subjects,
)


def _write_event_slice(
    root: Path, timepoint: str, event_name: str
) -> None:
    pd.DataFrame(
        [{"subjectid": "S1", "redcap_event_name": event_name}]
    ).to_csv(combined_csv_file(root, "PRONET", timepoint), index=False)


def test_explicit_event_aliases_are_canonicalized(tmp_path: Path) -> None:
    _write_event_slice(tmp_path, "baseline", "BASELINE_ARM_1")
    _write_event_slice(tmp_path, "month1", "month1_arm_1")
    _write_event_slice(tmp_path, "floating", "floating_arm_1")

    subjects = load_combined_subjects(
        tmp_path,
        "PRONET",
        timepoints=("baseline", "month1", "floating"),
    )

    assert subjects["S1"]["redcap_event_name"].tolist() == [
        "baseline_arm_1",
        "month_1_arm_1",
        "floating_forms_arm_1",
    ]


def test_explicit_unsupported_arm_is_rejected(tmp_path: Path) -> None:
    _write_event_slice(tmp_path, "baseline", "baseline_arm_3")

    with pytest.raises(OutcomeInputError, match="unsupported REDCap arm"):
        load_combined_subjects(
            tmp_path,
            "PRONET",
            timepoints=("baseline",),
        )
