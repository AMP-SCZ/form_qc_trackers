import importlib
import multiprocessing
import sys
from pathlib import Path

import pandas as pd
import pytest

from outcome_inputs import (
    OutcomeInputError,
    combined_csv_file,
    load_combined_subjects,
    pull_data,
)
from outcome_runner import requested_subject_ids


def _write_slice(
    directory: Path,
    network: str,
    timepoint: str,
    rows: list[dict[str, object]],
) -> Path:
    path = combined_csv_file(directory, network, timepoint)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_canonical_slices_reconstruct_chr_and_hc_events_and_json_shape(
    tmp_path: Path,
) -> None:
    screening_path = _write_slice(
        tmp_path,
        "pronet",
        "screening",
        [
            {
                "subjectid": " CHR001 ",
                "chrcrit_part": " 1 ",
                "screen_only": " alpha ",
                "ordinary_blank": "   ",
                "raw_na": " N/A ",
                "raw_dot": " . ",
                "raw_999": " 999 ",
                "raw_date_code": " 1909-09-09 ",
            },
            {
                "subjectid": " HC001 ",
                "chrcrit_part": " 2 ",
                "screen_only": " gamma ",
                "ordinary_blank": "",
                "raw_na": "NA",
                "raw_dot": ".",
                "raw_999": "999",
                "raw_date_code": "1903-03-03",
            },
        ],
    )
    baseline_path = _write_slice(
        tmp_path,
        "PRONET",
        "baseline",
        [
            {
                "subjectid": "CHR001",
                "chrcrit_part": "",
                "baseline_only": " beta ",
            },
            {
                "subjectid": "HC001",
                "chrcrit_part": "",
                "baseline_only": " delta ",
            },
        ],
    )
    month1_path = _write_slice(
        tmp_path,
        "ProNET",
        "month_1",
        [
            {"subjectid": "CHR001", "month_only": " one "},
            {"subjectid": "HC001", "month_only": " two "},
        ],
    )

    assert screening_path.name == (
        "AMPSCZ-combined-redcap_screening_ProNET-day1to1.csv"
    )
    assert baseline_path.name == (
        "AMPSCZ-combined-redcap_baseline_ProNET-day1to1.csv"
    )
    assert month1_path.name == (
        "AMPSCZ-combined-redcap_month_1_ProNET-day1to1.csv"
    )

    subjects = load_combined_subjects(
        tmp_path,
        "PRONET",
        timepoints=("screening", "baseline", "month1"),
    )

    assert list(subjects) == ["CHR001", "HC001"]
    chr_frame = subjects["CHR001"]
    hc_frame = subjects["HC001"]
    assert chr_frame["redcap_event_name"].tolist() == [
        "screening_arm_1",
        "baseline_arm_1",
        "month_1_arm_1",
    ]
    assert hc_frame["redcap_event_name"].tolist() == [
        "screening_arm_2",
        "baseline_arm_2",
        "month_1_arm_2",
    ]

    assert chr_frame["subjectid"].tolist() == ["CHR001"] * 3
    assert chr_frame.loc[0, "screen_only"] == "alpha"
    assert chr_frame.loc[1, "baseline_only"] == "beta"
    assert chr_frame.loc[2, "month_only"] == "one"
    assert pd.isna(chr_frame.loc[0, "ordinary_blank"])
    assert pd.isna(chr_frame.loc[1, "screen_only"])
    assert pd.isna(chr_frame.loc[0, "baseline_only"])
    assert chr_frame.loc[0, "raw_na"] == "N/A"
    assert chr_frame.loc[0, "raw_dot"] == "."
    assert chr_frame.loc[0, "raw_999"] == "999"
    assert chr_frame.loc[0, "raw_date_code"] == "1909-09-09"


def test_explicit_event_is_preserved_and_supplies_arm_to_other_slices(
    tmp_path: Path,
) -> None:
    _write_slice(
        tmp_path,
        "PRESCIENT",
        "baseline",
        [{"subjectid": "EXPLICIT", "redcap_event_name": " baseline_arm_2 "}],
    )
    _write_slice(
        tmp_path,
        "PRESCIENT",
        "month1",
        [{"subjectid": "EXPLICIT", "value": " 7 "}],
    )

    subjects = load_combined_subjects(
        tmp_path,
        "prescient",
        timepoints=("baseline", "month_1"),
    )

    frame = subjects["EXPLICIT"]
    assert frame["redcap_event_name"].tolist() == [
        "baseline_arm_2",
        "month_1_arm_2",
    ]
    assert frame.loc[1, "value"] == "7"


def test_pull_data_returns_deeply_isolated_subject_copies() -> None:
    source = pd.DataFrame(
        {
            "subjectid": ["S1"],
            "redcap_event_name": ["baseline_arm_1"],
            "value": ["original"],
        }
    )
    subjects = {"S1": source}

    first = pull_data(subjects, " S1 ")
    first.loc[0, "value"] = "mutated"
    first["new_column"] = "added"
    second = pull_data(subjects, "S1")

    assert second.loc[0, "value"] == "original"
    assert source.loc[0, "value"] == "original"
    assert "new_column" not in second.columns
    assert "new_column" not in source.columns
    with pytest.raises(OutcomeInputError, match="absent from the combined CSVs"):
        pull_data(subjects, "missing")


def test_missing_combined_csv_fails_with_the_resolved_path(tmp_path: Path) -> None:
    expected_path = combined_csv_file(tmp_path, "PRONET", "baseline")

    with pytest.raises(OutcomeInputError, match="Missing combined CSV") as exc_info:
        load_combined_subjects(
            tmp_path,
            "PRONET",
            timepoints=("baseline",),
        )

    assert str(expected_path) in str(exc_info.value)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"value": "1"}], "missing required 'subjectid' column"),
        ([{"subjectid": "   ", "value": "1"}], "blank subjectid"),
    ],
)
def test_bad_subject_identifier_columns_are_rejected(
    tmp_path: Path,
    rows: list[dict[str, object]],
    message: str,
) -> None:
    _write_slice(tmp_path, "PRONET", "baseline", rows)

    with pytest.raises(OutcomeInputError, match=message):
        load_combined_subjects(
            tmp_path,
            "PRONET",
            timepoints=("baseline",),
        )


def test_header_only_combined_csv_is_rejected(tmp_path: Path) -> None:
    path = combined_csv_file(tmp_path, "PRONET", "baseline")
    pd.DataFrame(columns=["subjectid", "value"]).to_csv(path, index=False)

    with pytest.raises(OutcomeInputError, match="zero participant rows"):
        load_combined_subjects(
            tmp_path,
            "PRONET",
            timepoints=("baseline",),
        )


def test_unreadable_combined_csv_is_wrapped_as_input_error(tmp_path: Path) -> None:
    path = combined_csv_file(tmp_path, "PRONET", "baseline")
    path.write_bytes(b"subjectid,value\nS1,\xff\n")

    with pytest.raises(OutcomeInputError, match="Could not parse combined CSV"):
        load_combined_subjects(
            tmp_path,
            "PRONET",
            timepoints=("baseline",),
        )


def test_explicit_event_must_match_its_combined_slice(tmp_path: Path) -> None:
    _write_slice(
        tmp_path,
        "PRONET",
        "baseline",
        [{"subjectid": "S1", "redcap_event_name": "month_1_arm_1"}],
    )

    with pytest.raises(OutcomeInputError, match="conflicts with timepoint"):
        load_combined_subjects(
            tmp_path,
            "PRONET",
            timepoints=("baseline",),
        )


def test_subject_without_event_or_cohort_cannot_be_assigned_an_arm(
    tmp_path: Path,
) -> None:
    _write_slice(
        tmp_path,
        "PRONET",
        "baseline",
        [{"subjectid": "UNKNOWN", "value": "1"}],
    )

    with pytest.raises(OutcomeInputError, match="Could not determine a REDCap arm"):
        load_combined_subjects(
            tmp_path,
            "PRONET",
            timepoints=("baseline",),
        )


def test_duplicate_subject_event_rows_are_rejected(tmp_path: Path) -> None:
    _write_slice(
        tmp_path,
        "PRONET",
        "baseline",
        [
            {"subjectid": "DUP", "chrcrit_part": "1", "value": "first"},
            {"subjectid": "DUP", "chrcrit_part": "1", "value": "second"},
        ],
    )

    with pytest.raises(OutcomeInputError, match="duplicate subject/event pair"):
        load_combined_subjects(
            tmp_path,
            "PRONET",
            timepoints=("baseline",),
        )


def test_requested_subject_ids_uses_every_combined_subject_for_full_run() -> None:
    assert requested_subject_ids("ProNET", "run_outcome") is None
    assert requested_subject_ids(
        "PRONET",
        "run_outcome",
        explicit_ids=(" S2 ", "S1", "S2"),
    ) == ["S2", "S1"]


def test_outcome_calculations_import_has_no_cli_or_input_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("outcome_calculations performed I/O during import")

    monkeypatch.setattr(pd, "read_csv", unexpected_call)
    monkeypatch.setattr(multiprocessing, "Pool", unexpected_call)
    monkeypatch.setattr(
        sys,
        "argv",
        ["outcome_calculations.py", "PRONET", "run_outcome"],
    )
    sys.modules.pop("outcome_calculations", None)

    module = importlib.import_module("outcome_calculations")

    assert callable(module.compute_outcomes)
    assert len(module.all_visits_list) == 36

def test_missing_subject_diagnostic_omits_requested_identifier() -> None:
    sensitive_subject = "PRIVATE_SUBJECT_NOT_FOUND"

    with pytest.raises(
        OutcomeInputError, match="absent from the combined CSVs"
    ) as exc_info:
        pull_data({}, sensitive_subject)

    assert sensitive_subject not in str(exc_info.value)
