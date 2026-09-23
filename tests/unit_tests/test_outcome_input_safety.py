from pathlib import Path

import pandas as pd
import pytest

from outcome_calculations import _safe_output_component, compute_outcomes
from outcome_inputs import (
    OutcomeInputError,
    combined_csv_file,
    load_combined_subjects,
)


def test_duplicate_combined_csv_headers_are_rejected(tmp_path: Path) -> None:
    path = combined_csv_file(tmp_path, "PRONET", "baseline")
    path.write_text(
        "subjectid,subjectid,chrcrit_part\nS1,S1,1\n",
        encoding="utf-8",
    )

    with pytest.raises(OutcomeInputError, match="duplicate header"):
        load_combined_subjects(
            tmp_path, "PRONET", timepoints=("baseline",)
        )

@pytest.mark.parametrize(
    ("data_row", "actual_fields"),
    [
        ("S1,1\n", 2),
        ("S1,1,note,extra\n", 4),
    ],
)
def test_csv_rows_must_match_header_field_count(
    tmp_path: Path, data_row: str, actual_fields: int
) -> None:
    path = combined_csv_file(tmp_path, "PRONET", "baseline")
    path.write_text(
        f"subjectid,chrcrit_part,note\n{data_row}",
        encoding="utf-8",
    )

    with pytest.raises(
        OutcomeInputError,
        match=rf"{actual_fields} field.*expected 3",
    ) as exc_info:
        load_combined_subjects(
            tmp_path, "PRONET", timepoints=("baseline",)
        )
    assert "S1" not in str(exc_info.value)


def test_quoted_newline_is_one_logical_csv_field(tmp_path: Path) -> None:
    path = combined_csv_file(tmp_path, "PRONET", "baseline")
    path.write_text(
        'subjectid,chrcrit_part,note\nS1,1,"line one\nline two"\n',
        encoding="utf-8",
    )

    subjects = load_combined_subjects(
        tmp_path, "PRONET", timepoints=("baseline",)
    )

    assert subjects["S1"].loc[0, "note"].splitlines() == ["line one", "line two"]



@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"network": "PRESCIENT"}, "row-level network metadata"),
        ({"timepoint": "month1"}, "row-level timepoint metadata"),
    ],
)
def test_row_metadata_must_match_selected_combined_slice(
    tmp_path: Path, metadata: dict[str, str], message: str
) -> None:
    row = {"subjectid": "S1", "chrcrit_part": "1", **metadata}
    pd.DataFrame([row]).to_csv(
        combined_csv_file(tmp_path, "PRONET", "baseline"), index=False
    )

    sensitive_value = next(iter(metadata.values()))
    with pytest.raises(OutcomeInputError, match=message) as exc_info:
        load_combined_subjects(
            tmp_path, "PRONET", timepoints=("baseline",)
        )
    diagnostic = str(exc_info.value)
    assert sensitive_value not in diagnostic
    assert "S1" not in diagnostic


def test_blank_row_metadata_is_allowed(tmp_path: Path) -> None:
    pd.DataFrame(
        [{
            "subjectid": "S1",
            "chrcrit_part": "1",
            "network": "",
            "timepoint": "",
            "redcap_repeat_instrument": "",
            "redcap_repeat_instance": "",
        }]
    ).to_csv(combined_csv_file(tmp_path, "PRONET", "baseline"), index=False)

    subjects = load_combined_subjects(
        tmp_path, "PRONET", timepoints=("baseline",)
    )

    assert subjects["S1"].loc[0, "redcap_event_name"] == "baseline_arm_1"


def test_mixed_arm_history_is_rejected_before_calculation(tmp_path: Path) -> None:
    pd.DataFrame(
        [{"subjectid": "S1", "redcap_event_name": "baseline_arm_1"}]
    ).to_csv(combined_csv_file(tmp_path, "PRESCIENT", "baseline"), index=False)
    pd.DataFrame(
        [{"subjectid": "S1", "redcap_event_name": "month_1_arm_2"}]
    ).to_csv(combined_csv_file(tmp_path, "PRESCIENT", "month1"), index=False)

    with pytest.raises(OutcomeInputError, match="mixed REDCap arm histories"):
        load_combined_subjects(
            tmp_path,
            "PRESCIENT",
            timepoints=("baseline", "month1"),
        )


def test_explicit_event_arm_must_match_same_row_cohort(tmp_path: Path) -> None:
    pd.DataFrame(
        [{
            "subjectid": "S1",
            "redcap_event_name": "baseline_arm_1",
            "chrcrit_part": "2",
        }]
    ).to_csv(combined_csv_file(tmp_path, "PRESCIENT", "baseline"), index=False)

    with pytest.raises(
        OutcomeInputError,
        match="explicit event arm conflicts with chrcrit_part for 1 row",
    ):
        load_combined_subjects(
            tmp_path, "PRESCIENT", timepoints=("baseline",)
        )


def test_dual_event_columns_are_canonicalized_before_comparison(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        [{
            "subjectid": "S1",
            "redcap_event_name": "month1_arm_1",
            "event_name": "MONTH_1_ARM_1",
        }]
    ).to_csv(combined_csv_file(tmp_path, "PRONET", "month1"), index=False)

    subjects = load_combined_subjects(
        tmp_path, "PRONET", timepoints=("month1",)
    )

    assert subjects["S1"].loc[0, "redcap_event_name"] == "month_1_arm_1"


def test_dual_event_columns_must_agree_on_each_row(tmp_path: Path) -> None:
    subject_id = "SENSITIVE001"
    pd.DataFrame(
        [{
            "subjectid": subject_id,
            "redcap_event_name": "baseline_arm_1",
            "event_name": "baseline_arm_2",
        }]
    ).to_csv(combined_csv_file(tmp_path, "PRONET", "baseline"), index=False)

    with pytest.raises(
        OutcomeInputError,
        match="redcap_event_name and event_name disagree for 1 row",
    ) as exc_info:
        load_combined_subjects(
            tmp_path, "PRONET", timepoints=("baseline",)
        )
    assert subject_id not in str(exc_info.value)


def test_each_dual_event_column_must_match_its_slice(tmp_path: Path) -> None:
    subject_id = "SENSITIVE001"
    pd.DataFrame(
        [{
            "subjectid": subject_id,
            "redcap_event_name": "baseline_arm_1",
            "event_name": "month_1_arm_1",
        }]
    ).to_csv(combined_csv_file(tmp_path, "PRONET", "baseline"), index=False)

    with pytest.raises(
        OutcomeInputError,
        match="event_name conflicts with timepoint 'baseline' for 1 row",
    ) as exc_info:
        load_combined_subjects(
            tmp_path, "PRONET", timepoints=("baseline",)
        )
    assert subject_id not in str(exc_info.value)


@pytest.mark.parametrize(
    "repeat_metadata",
    [
        {"redcap_repeat_instrument": "repeat_form"},
        {"redcap_repeat_instance": "1"},
        {
            "redcap_repeat_instrument": "repeat_form",
            "redcap_repeat_instance": "1",
        },
    ],
)
def test_repeating_rows_are_rejected_before_subject_filtering(
    tmp_path: Path, repeat_metadata: dict[str, str]
) -> None:
    subject_id = "SENSITIVE001"
    pd.DataFrame(
        [{
            "subjectid": subject_id,
            "chrcrit_part": "1",
            **repeat_metadata,
        }]
    ).to_csv(combined_csv_file(tmp_path, "PRONET", "baseline"), index=False)

    with pytest.raises(
        OutcomeInputError,
        match="do not support repeating REDCap rows; found 1 row",
    ) as exc_info:
        load_combined_subjects(
            tmp_path,
            "PRONET",
            timepoints=("baseline",),
            subject_ids=("OTHER",),
        )
    assert subject_id not in str(exc_info.value)


def test_output_path_components_cannot_escape_output_root(tmp_path: Path) -> None:
    component = _safe_output_component(
        "../../outside\\nested:CON", "subject"
    )

    assert component.startswith("subject-")
    assert not any(character in component for character in ("/", "\\", ":", "."))
    (tmp_path / component).resolve().relative_to(tmp_path.resolve())

def test_invalid_row_metadata_diagnostic_omits_row_values(
    tmp_path: Path,
) -> None:
    sensitive_subject = "PRIVATE_SUBJECT_ROW"
    sensitive_metadata = "PRIVATE_NETWORK_ROW_VALUE"
    pd.DataFrame(
        [{
            "subjectid": sensitive_subject,
            "chrcrit_part": "1",
            "network": sensitive_metadata,
        }]
    ).to_csv(
        combined_csv_file(tmp_path, "PRONET", "baseline"),
        index=False,
    )

    with pytest.raises(
        OutcomeInputError, match="invalid row-level network metadata"
    ) as exc_info:
        load_combined_subjects(
            tmp_path, "PRONET", timepoints=("baseline",)
        )

    diagnostic = str(exc_info.value)
    assert sensitive_metadata not in diagnostic
    assert sensitive_subject not in diagnostic


def test_empty_subject_diagnostic_omits_subject_id(tmp_path: Path) -> None:
    sensitive_subject = "PRIVATE_SUBJECT_EMPTY"

    with pytest.raises(OutcomeInputError, match="has no rows") as exc_info:
        compute_outcomes(
            sensitive_subject,
            subject_data=pd.DataFrame(),
            network_name="PRONET",
            run_version="run_outcome",
            output_dir=tmp_path,
        )

    assert sensitive_subject not in str(exc_info.value)
