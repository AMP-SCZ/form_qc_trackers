from pathlib import Path

import pandas as pd
import pytest

from outcome_inputs import (
    OutcomeInputError,
    combined_csv_file,
    load_combined_subjects,
)


def test_cohort_derived_mixed_arm_history_is_rejected(tmp_path: Path) -> None:
    pd.DataFrame(
        [{"subjectid": "S1", "chrcrit_part": "1"}]
    ).to_csv(combined_csv_file(tmp_path, "PRESCIENT", "baseline"), index=False)
    pd.DataFrame(
        [{"subjectid": "S1", "chrcrit_part": "2"}]
    ).to_csv(combined_csv_file(tmp_path, "PRESCIENT", "month1"), index=False)

    with pytest.raises(OutcomeInputError, match="mixed REDCap arm histories"):
        load_combined_subjects(
            tmp_path,
            "PRESCIENT",
            timepoints=("baseline", "month1"),
        )


def test_explicit_and_cohort_arm_conflict_is_rejected(tmp_path: Path) -> None:
    pd.DataFrame(
        [{"subjectid": "S1", "redcap_event_name": "baseline_arm_1"}]
    ).to_csv(combined_csv_file(tmp_path, "PRESCIENT", "baseline"), index=False)
    pd.DataFrame(
        [{"subjectid": "S1", "chrcrit_part": "2"}]
    ).to_csv(combined_csv_file(tmp_path, "PRESCIENT", "month1"), index=False)

    with pytest.raises(OutcomeInputError, match="mixed REDCap arm histories"):
        load_combined_subjects(
            tmp_path,
            "PRESCIENT",
            timepoints=("baseline", "month1"),
        )
