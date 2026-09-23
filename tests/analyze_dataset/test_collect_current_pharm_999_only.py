"""Tests for collecting participants coded only as 999 on current pharm forms."""

import numpy as np
import pandas as pd
import pytest

from analyze_dataset.collect_current_pharm_999_only import (
    CurrentPharm999OnlyCollector,
    collect_999_only_participants,
    normalize_med_code,
)


def _records(result):
    """Allow the public helper to return records or a tabular DataFrame."""
    if isinstance(result, pd.DataFrame):
        return result.to_dict("records")
    return list(result)


def _subject_ids(result):
    return [record["subjectid"] for record in _records(result)]


def _complete_current_pharm_row(subjectid, **values):
    row = {"subjectid": subjectid}
    row.update(
        {
            f"chrpharm_med{med_number}_name": ""
            for med_number in range(1, 51)
        }
    )
    row.update(values)
    return row


@pytest.mark.parametrize("value", [999, 999.0, "999", " 999.0 "])
def test_normalize_med_code_accepts_999_dtype_variants(value):
    assert normalize_med_code(value) == "999"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (777, "777"),
        (888.0, "888"),
        (-9.0, "-9"),
        ("1999", "1999"),
        ("999mg", "999mg"),
    ],
)
def test_normalize_med_code_does_not_substring_match_999(value, expected):
    assert normalize_med_code(value) == expected


def test_collect_qualifies_only_when_observed_current_codes_are_exactly_999():
    frame = pd.DataFrame(
        [
            {
                "subjectid": "ONLY_ONE",
                "chrpharm_med1_name": "999",
            },
            {
                "subjectid": "ONLY_MANY",
                "chrpharm_med1_name": 999.0,
                "chrpharm_med50_name": " 999.0 ",
            },
            {
                "subjectid": "BLANKS_IGNORED",
                "chrpharm_med1_name": 999,
                "chrpharm_med2_name": "",
                "chrpharm_med3_name": "   ",
                "chrpharm_med4_name": None,
                "chrpharm_med5_name": np.nan,
                "chrpharm_med6_name": pd.NA,
            },
            {
                "subjectid": "WITH_REAL_MED",
                "chrpharm_med1_name": 999,
                "chrpharm_med26_name": 300,
            },
            {
                "subjectid": "REAL_MED_ONLY",
                "chrpharm_med1_name": 300,
            },
            {
                "subjectid": "NO_CODES",
                "chrpharm_med1_name": "",
            },
            {
                "subjectid": "SUBSTRING_ONLY",
                "chrpharm_med1_name": "1999",
            },
        ]
    )

    result = collect_999_only_participants(
        frame,
        network="PRONET",
        source_file="floating_forms.csv",
    )

    records = _records(result)
    assert [record["subjectid"] for record in records] == [
        "BLANKS_IGNORED",
        "ONLY_MANY",
        "ONLY_ONE",
    ]
    assert all(record["network"] == "PRONET" for record in records)
    assert all(
        record["source_file"] == "floating_forms.csv" for record in records
    )


@pytest.mark.parametrize("other_code", [777, 888, -9, 300, "999mg"])
def test_any_other_observed_current_code_disqualifies(other_code):
    frame = pd.DataFrame(
        {
            "subjectid": ["SUB001"],
            "chrpharm_med1_name": [999],
            "chrpharm_med2_name": [other_code],
        }
    )

    result = collect_999_only_participants(frame, network="PRESCIENT")

    assert _records(result) == []


def test_only_exact_current_medication_name_columns_are_considered():
    frame = pd.DataFrame(
        [
            {
                "subjectid": "PAST_REAL_IGNORED",
                "chrpharm_med1_name": 999,
                "chrpharm_med1_name_past": 300,
            },
            {
                "subjectid": "SIMILAR_NOISE_IGNORED",
                "chrpharm_med26_name": 999,
                "chrpharm_med26_name_extra": 300,
                "prefix_chrpharm_med27_name": 300,
                "chrpharm_med27_dosage": 300,
            },
            {
                "subjectid": "PAST_999_ONLY_DOES_NOT_QUALIFY",
                "chrpharm_med1_name": "",
                "chrpharm_med1_name_past": 999,
            },
            {
                "subjectid": "SECOND_FORM_REAL_MED_COUNTS",
                "chrpharm_med1_name": 999,
                "chrpharm_med50_name": 300,
            },
        ]
    )

    result = collect_999_only_participants(frame, network="PRONET")

    assert _subject_ids(result) == [
        "PAST_REAL_IGNORED",
        "SIMILAR_NOISE_IGNORED",
    ]


def test_rows_are_aggregated_per_participant_before_qualification():
    frame = pd.DataFrame(
        [
            {"subjectid": "ONLY_999", "chrpharm_med1_name": 999},
            {"subjectid": "MIXED_ACROSS_ROWS", "chrpharm_med1_name": 999},
            {"subjectid": "ONLY_999", "chrpharm_med1_name": "999.0"},
            {"subjectid": "MIXED_ACROSS_ROWS", "chrpharm_med1_name": 300},
            {"subjectid": "ONLY_999", "chrpharm_med1_name": ""},
        ]
    )

    result = collect_999_only_participants(frame, network="PRONET")

    assert _subject_ids(result) == ["ONLY_999"]


def test_output_is_deduplicated_and_sorted_by_subjectid():
    frame = pd.DataFrame(
        [
            {"subjectid": "ZZ00003", "chrpharm_med1_name": 999},
            {"subjectid": "AA00001", "chrpharm_med1_name": 999},
            {"subjectid": "MM00002", "chrpharm_med1_name": 999},
            {"subjectid": "AA00001", "chrpharm_med1_name": 999.0},
        ]
    )

    result = collect_999_only_participants(frame, network="PRESCIENT")

    assert _subject_ids(result) == ["AA00001", "MM00002", "ZZ00003"]


def test_collector_reads_both_canonical_floating_files_and_writes_csv(tmp_path):
    pd.DataFrame(
        [
            _complete_current_pharm_row(
                "P_ONLY", chrpharm_med1_name=999
            ),
            _complete_current_pharm_row(
                "P_MIXED",
                chrpharm_med1_name=999,
                chrpharm_med26_name=300,
            ),
        ]
    ).to_csv(
        tmp_path / "AMPSCZ-combined-redcap_floating_forms_ProNET-day1to1.csv",
        index=False,
    )
    pd.DataFrame(
        [
            _complete_current_pharm_row(
                "S_ONLY", chrpharm_med50_name="999.0"
            )
        ]
    ).to_csv(
        tmp_path
        / "AMPSCZ-combined-redcap_floating_forms_PRESCIENT-day1to1.csv",
        index=False,
    )

    output_path = tmp_path / "output" / "participants.csv"
    collector = CurrentPharm999OnlyCollector(tmp_path)
    result = collector.run_script(output_path)

    assert result[["network", "subjectid"]].to_dict("records") == [
        {"network": "PRESCIENT", "subjectid": "S_ONLY"},
        {"network": "PRONET", "subjectid": "P_ONLY"},
    ]
    assert collector.n_files_read == 2
    assert collector.n_rows_scanned == 3
    assert collector.n_participants_collected == 2
    assert pd.read_csv(output_path, dtype=str)["subjectid"].tolist() == [
        "S_ONLY",
        "P_ONLY",
    ]


def test_run_script_does_not_write_when_no_inputs_exist(tmp_path):
    output_path = tmp_path / "participants.csv"
    collector = CurrentPharm999OnlyCollector(tmp_path)

    with pytest.raises(ValueError, match="no usable current-pharm"):
        collector.run_script(output_path)

    assert not output_path.exists()


def test_collector_runs_when_26_of_50_medication_columns_are_missing(
    tmp_path, capsys
):
    partial_row = {"subjectid": "PARTIAL"}
    partial_row.update(
        {f"chrpharm_med{med_number}_name": "" for med_number in range(1, 25)}
    )
    partial_row["chrpharm_med1_name"] = 999
    pd.DataFrame([partial_row]).to_csv(
        tmp_path / "AMPSCZ-combined-redcap_floating_forms_ProNET-day1to1.csv",
        index=False,
    )
    output_path = tmp_path / "participants.csv"
    collector = CurrentPharm999OnlyCollector(tmp_path, networks=["PRONET"])

    result = collector.run_script(output_path)

    assert result["subjectid"].tolist() == ["PARTIAL"]
    assert output_path.exists()
    assert "missing 26 of 50" in capsys.readouterr().err


def test_strict_schema_check_can_still_require_all_50_columns(tmp_path):
    pd.DataFrame(
        [{"subjectid": "PARTIAL", "chrpharm_med1_name": 999}]
    ).to_csv(
        tmp_path / "AMPSCZ-combined-redcap_floating_forms_ProNET-day1to1.csv",
        index=False,
    )
    output_path = tmp_path / "participants.csv"
    collector = CurrentPharm999OnlyCollector(
        tmp_path,
        networks=["PRONET"],
        require_all_medication_columns=True,
    )

    with pytest.raises(ValueError, match="missing 49 of 50"):
        collector.run_script(output_path)

    assert not output_path.exists()


def test_collector_requires_every_selected_network_by_default(tmp_path):
    pd.DataFrame(
        [_complete_current_pharm_row("P_ONLY", chrpharm_med1_name=999)]
    ).to_csv(
        tmp_path / "AMPSCZ-combined-redcap_floating_forms_ProNET-day1to1.csv",
        index=False,
    )
    output_path = tmp_path / "participants.csv"
    collector = CurrentPharm999OnlyCollector(tmp_path)

    with pytest.raises(ValueError, match="PRESCIENT"):
        collector.run_script(output_path)

    assert not output_path.exists()


def test_header_only_input_does_not_replace_prior_output(tmp_path):
    columns = ["subjectid"] + [
        f"chrpharm_med{med_number}_name" for med_number in range(1, 51)
    ]
    pd.DataFrame(columns=columns).to_csv(
        tmp_path / "AMPSCZ-combined-redcap_floating_forms_ProNET-day1to1.csv",
        index=False,
    )
    output_path = tmp_path / "participants.csv"
    output_path.write_text("prior valid output\n", encoding="utf-8")
    collector = CurrentPharm999OnlyCollector(tmp_path, networks=["PRONET"])

    with pytest.raises(ValueError, match="no usable current-pharm"):
        collector.run_script(output_path)

    assert output_path.read_text(encoding="utf-8") == "prior valid output\n"


def test_atomic_write_preserves_prior_output_on_serialization_error(
    tmp_path, monkeypatch
):
    pd.DataFrame(
        [_complete_current_pharm_row("P_ONLY", chrpharm_med1_name=999)]
    ).to_csv(
        tmp_path / "AMPSCZ-combined-redcap_floating_forms_ProNET-day1to1.csv",
        index=False,
    )
    output_path = tmp_path / "participants.csv"
    output_path.write_text("prior valid output\n", encoding="utf-8")
    collector = CurrentPharm999OnlyCollector(tmp_path, networks=["PRONET"])

    def fail_after_partial_write(self, path, *args, **kwargs):
        path.write_text("partial", encoding="utf-8")
        raise OSError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_after_partial_write)

    with pytest.raises(OSError, match="simulated write failure"):
        collector.run_script(output_path)

    assert output_path.read_text(encoding="utf-8") == "prior valid output\n"
    assert list(tmp_path.glob(".*.tmp")) == []
