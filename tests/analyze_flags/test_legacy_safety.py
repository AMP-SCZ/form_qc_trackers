"""Regression tests for legacy analyze_flags artifact safety."""

import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from analyze_flags.extract_blank_flags import BlankFlagExtractor
from analyze_flags.export_scid5_flags import ScidFlagExporter
from analyze_flags.graph_recovered_flags import ResolvedGrapher
from analyze_flags import manual_review
from analyze_flags.manual_review import JumpDecisionIntegrityError
from analyze_flags.paths import open_tracker_row_history_csv_basename


def _blank_extractor():
    extractor = object.__new__(BlankFlagExtractor)
    extractor.blank_records = set()
    extractor._excel_engine = "openpyxl"
    extractor.utils = SimpleNamespace(
        missing_code_set=set(), missing_code_list=[],
    )
    return extractor


def _tracker_frame(subject, variable):
    return pd.DataFrame({
        "Subject": [subject],
        "Timepoint": ["baseline"],
        "Specific Flags": [f"{variable} : Variable is blank."],
    })


class _RevisionDropbox:
    def request_json_object(self, *args, **kwargs):
        return {
            "entries": [{
                "rev": "rev-1",
                "server_modified": "2026-01-02T12:00:00Z",
                "content_hash": "hash-1",
            }],
            "has_more": False,
        }

    def files_download(self, path, rev):
        return None, SimpleNamespace(content=b"workbook")


def test_blank_excel_reader_requests_all_sheets(monkeypatch):
    extractor = _blank_extractor()
    captured = {}

    def fake_read_excel(*args, **kwargs):
        captured.update(kwargs)
        return {"Main Report": _tracker_frame("S1", "a")}

    monkeypatch.setattr("analyze_flags.extract_blank_flags.pd.read_excel",
                        fake_read_excel)
    sheets = extractor._read_tracker_excel(b"not-used")
    assert list(sheets) == ["Main Report"]
    assert captured["sheet_name"] is None


def test_blank_walk_collects_flags_from_every_authoritative_sheet(monkeypatch):
    extractor = _blank_extractor()
    dbx = _RevisionDropbox()
    extractor.utils = SimpleNamespace(collect_dropbox_credentials=lambda: dbx)
    monkeypatch.setattr(
        extractor,
        "_read_tracker_excel",
        lambda content: {
            "Main Report": _tracker_frame("S1", "field_a"),
            "Proposed Checks": _tracker_frame("S2", "field_b"),
            "Operator backup": _tracker_frame("S3", "stale_field"),
        },
    )
    records = set()
    assert extractor.extract_blank_flags(
        "/tracker.xlsx", "PRESCIENT", target_records=records,
    )
    assert records == {
        ("PRESCIENT", "S1", "field_a", "baseline"),
        ("PRESCIENT", "S2", "field_b", "baseline"),
    }


def test_blank_walk_rejects_partly_malformed_authoritative_revision(monkeypatch):
    extractor = _blank_extractor()
    dbx = _RevisionDropbox()
    extractor.utils = SimpleNamespace(collect_dropbox_credentials=lambda: dbx)
    monkeypatch.setattr(
        extractor,
        "_read_tracker_excel",
        lambda content: {
            "Main Report": _tracker_frame("S1", "field_a"),
            "Proposed Checks": pd.DataFrame({"unexpected": ["value"]}),
        },
    )
    records = set()
    assert not extractor.extract_blank_flags(
        "/tracker.xlsx", "PRESCIENT", target_records=records,
    )


def test_blank_parser_does_not_treat_a_raw_value_pipe_as_another_flag():
    extractor = _blank_extractor()
    frame = pd.DataFrame({
        "Subject": ["S1"],
        "Timepoint": ["baseline"],
        "Specific Flags": [
            "chrguid_guid : GUID in incorrect format. GUID was reported to be "
            "bad|field_a:Variable is blank."
        ],
    })

    extractor._collect_blank_records(
        frame, "Subject", "Timepoint", "Specific Flags", "PRESCIENT")

    assert extractor.blank_records == set()


def test_blank_parser_accepts_the_check_id_prefixed_blank_contract():
    extractor = _blank_extractor()
    frame = pd.DataFrame({
        "Subject": ["S1"],
        "Timepoint": ["baseline"],
        "Specific Flags": [
            "field_a : [REQUIRED-001] Variable is blank."
        ],
    })

    extractor._collect_blank_records(
        frame, "Subject", "Timepoint", "Specific Flags", "PRESCIENT")

    assert extractor.blank_records == {
        ("PRESCIENT", "S1", "field_a", "baseline"),
    }


def test_blank_parser_accepts_the_legacy_value_is_empty_contract():
    extractor = _blank_extractor()
    frame = pd.DataFrame({
        "Subject": ["S1", "S2", "S3"],
        "Timepoint": ["screen", "baseln", "month_2"],
        "Specific Flags": [
            "field_a : Value is empty",
            "field_b : value is empty.",
            # Free text containing the phrase must not match the contract.
            "field_c : Notes say Value is empty somewhere in the middle.",
        ],
    })

    extractor._collect_blank_records(
        frame, "Subject", "Timepoint", "Specific Flags", "PRESCIENT")

    assert extractor.blank_records == {
        ("PRESCIENT", "S1", "field_a", "screening"),
        ("PRESCIENT", "S2", "field_b", "baseline"),
    }


def test_blank_loop_preserves_records_when_any_source_is_partial(monkeypatch):
    extractor = _blank_extractor()
    extractor.NETWORKS = ["PRESCIENT"]
    extractor.dropbox_path = "/"
    extractor.config_info = {"testing_enabled": "True"}
    extractor.blank_records = {("OLD", "S0", "v", "baseline")}
    monkeypatch.setattr(
        extractor, "_tracker_paths",
        lambda network: [("/", "good.xlsx"), ("/", "partial.xlsx")],
    )

    def fake_extract(path, network, days_back=None, page_limit=100,
                     target_records=None):
        target_records.add((network, "S1", "new", "baseline"))
        return not path.endswith("partial.xlsx")

    monkeypatch.setattr(extractor, "extract_blank_flags", fake_extract)
    assert not extractor.loop_dropbox()
    assert extractor.blank_records == {("OLD", "S0", "v", "baseline")}


def test_duplicate_subject_values_must_agree(monkeypatch):
    extractor = _blank_extractor()
    extractor.blank_records = {
        ("PRESCIENT", "S1", "field_a", "baseline"),
    }
    cdf = pd.DataFrame({
        "subjectid": ["S1", "S1"],
        "field_a": ["before", "after"],
    })
    monkeypatch.setattr(extractor, "_combined_csv_path", lambda *args: "x.csv")
    monkeypatch.setattr(extractor, "_load_combined_csv",
                        lambda path: (cdf, None))
    rows = extractor.compare_against_combined_csvs()
    assert rows[0][-2:] == ("", "subject_ambiguous")


def test_duplicate_subject_identical_values_remain_determinable(monkeypatch):
    extractor = _blank_extractor()
    extractor.blank_records = {
        ("PRESCIENT", "S1", "field_a", "baseline"),
    }
    cdf = pd.DataFrame({
        "subjectid": [" S1", "S1 "],
        "field_a": ["same", "same"],
    })
    monkeypatch.setattr(extractor, "_combined_csv_path", lambda *args: "x.csv")
    monkeypatch.setattr(extractor, "_load_combined_csv",
                        lambda path: (cdf, None))
    rows = extractor.compare_against_combined_csvs()
    assert rows[0][-2:] == ("same", "filled")


def test_jump_exclusions_fail_closed_without_affected_entries(monkeypatch,
                                                               tmp_path):
    (tmp_path / "jumps_PRESCIENT.xlsx").write_bytes(b"placeholder")
    jumps = pd.DataFrame([{
        "jump_id": "j1", "include": "", "direction": "removed",
        "n_entries": "1",
    }])

    def fake_read(path, sheet_name):
        if sheet_name == "jumps":
            return jumps
        return pd.DataFrame()

    monkeypatch.setattr(manual_review, "_read_sheet", fake_read)
    with pytest.raises(JumpDecisionIntegrityError):
        manual_review.load_jump_decisions(str(tmp_path), "PRESCIENT")


def _grapher(tmp_path):
    grapher = object.__new__(ResolvedGrapher)
    grapher.analyze_flags_dir = str(tmp_path)
    grapher.utils = SimpleNamespace()
    grapher.total_flags = {
        "PRESCIENT": {"all": 0, "filtered": 0},
        "PRONET": {"all": 0, "filtered": 0},
    }
    grapher.jump_excluded = {
        "PRESCIENT": {"added": 0, "removed": 0},
        "PRONET": {"added": 0, "removed": 0},
    }
    return grapher


def _history_row(**updates):
    row = {
        "Subject": "S1",
        "Timepoint": "baseline",
        "General_Flag": "mood",
        "variable": "field_a",
        "canonical_template": "flag {variable}",
        "Earliest_seen": "malformed-but-irrelevant",
        "Latest_seen": "2026-01-01T00:00:00Z",
        "Resolution_observed": "2026-01-04T17:30:00Z",
        "is_currently_open": "False",
    }
    row.update(updates)
    return row


def test_graph_buckets_by_resolution_observed_and_ignores_bad_earliest(tmp_path):
    grapher = _grapher(tmp_path)
    path = tmp_path / "history.csv"
    pd.DataFrame([_history_row()]).to_csv(path, index=False)
    history = grapher._load_history(str(path))
    buckets = grapher._bucket_by_day(history, set(), "PRESCIENT", "all")
    assert list(buckets) == [pd.Timestamp("2026-01-04", tz="UTC")]


def test_graph_old_history_falls_back_to_latest_seen(tmp_path):
    grapher = _grapher(tmp_path)
    path = tmp_path / "old_history.csv"
    row = _history_row()
    del row["Resolution_observed"]
    pd.DataFrame([row]).to_csv(path, index=False)
    history = grapher._load_history(str(path))
    assert history.loc[0, "Resolution_observed"] == pd.Timestamp(
        "2026-01-01", tz="UTC"
    )


def test_graph_excludes_testing_only_resolutions(tmp_path):
    grapher = _grapher(tmp_path)
    path = tmp_path / "history.csv"
    pd.DataFrame([
        _history_row(Subject="LIVE", resolution_eligible="True"),
        _history_row(Subject="TEST", resolution_eligible="False"),
    ]).to_csv(path, index=False)

    history = grapher._load_history(str(path), "PRONET")

    assert list(history["Subject"]) == ["LIVE"]


def test_graph_daily_series_includes_zero_days(tmp_path):
    grapher = _grapher(tmp_path)
    day1 = pd.Timestamp("2026-01-01", tz="UTC")
    day3 = pd.Timestamp("2026-01-03", tz="UTC")
    series = grapher._build_per_day_series(
        {day1: [{}], day3: [{}, {}]}, "PRESCIENT", "all",
    )
    assert list(series.values()) == [1, 0, 2]
    assert grapher.total_flags["PRESCIENT"]["all"] == 3


def test_graph_removed_counter_is_not_doubled_for_filtered_view(tmp_path):
    grapher = _grapher(tmp_path)
    frame = pd.DataFrame([_history_row()])
    frame["Resolution_observed"] = pd.to_datetime(
        frame["Resolution_observed"], utc=True
    ).dt.normalize()
    frame["is_currently_open"] = False
    key = ("S1", "baseline", "mood", "field_a", "flag {variable}")
    grapher._bucket_by_day(frame, {key}, "PRESCIENT", "all")
    grapher._bucket_by_day(frame, {key}, "PRESCIENT", "filtered")
    assert grapher.jump_excluded["PRESCIENT"]["removed"] == 1


def test_successful_empty_graph_clears_stale_plot_and_subset(tmp_path):
    grapher = _grapher(tmp_path)
    png = tmp_path / "resolved_over_time_PRESCIENT_filtered.png"
    png.write_bytes(b"stale")
    subset = tmp_path / "PRESCIENT_filtered_history.csv"
    subset.write_text("stale\n", encoding="utf-8")
    empty = pd.DataFrame(columns=list(_history_row()))
    grapher._produce_graph(empty, "PRESCIENT", "filtered", set())
    assert not png.exists()
    assert pd.read_csv(subset).empty


def _empty_history_frame():
    return pd.DataFrame(columns=[
        "General_Flag", "canonical_template", "is_currently_open", "variable",
    ])


def test_scid_export_preserves_last_good_when_one_network_missing(tmp_path):
    exporter = object.__new__(ScidFlagExporter)
    exporter.analyze_flags_dir = str(tmp_path)
    prescient = tmp_path / open_tracker_row_history_csv_basename("PRESCIENT")
    _empty_history_frame().to_csv(prescient, index=False)
    output = tmp_path / exporter.OUTPUT_BASENAME
    output.write_bytes(b"last-good")
    assert not exporter.run_script()
    assert output.read_bytes() == b"last-good"


def test_scid_export_valid_empty_inputs_clear_stale_rows(tmp_path):
    exporter = object.__new__(ScidFlagExporter)
    exporter.analyze_flags_dir = str(tmp_path)
    for network in exporter.NETWORKS:
        path = tmp_path / open_tracker_row_history_csv_basename(network)
        _empty_history_frame().to_csv(path, index=False)
    assert exporter.run_script()
    result = pd.read_excel(
        tmp_path / exporter.OUTPUT_BASENAME,
        sheet_name="Templates",
        keep_default_na=False,
    )
    assert result.empty
    assert list(result.columns) == exporter.SUMMARY_COLUMNS


def test_scid_export_excludes_testing_only_history(tmp_path):
    exporter = object.__new__(ScidFlagExporter)
    exporter.analyze_flags_dir = str(tmp_path)
    empty = _empty_history_frame()
    empty["resolution_eligible"] = pd.Series(dtype=bool)
    empty.to_csv(
        tmp_path / open_tracker_row_history_csv_basename("PRESCIENT"),
        index=False,
    )
    pd.DataFrame([
        {
            "General_Flag": "scid5_psychosis_mood_substance_abuse",
            "canonical_template": "live template",
            "is_currently_open": False,
            "variable": "live_field",
            "resolution_eligible": True,
        },
        {
            "General_Flag": "scid5_psychosis_mood_substance_abuse",
            "canonical_template": "testing template",
            "is_currently_open": False,
            "variable": "testing_field",
            "resolution_eligible": False,
        },
    ]).to_csv(
        tmp_path / open_tracker_row_history_csv_basename("PRONET"),
        index=False,
    )

    assert exporter.run_script()
    result = pd.read_excel(
        tmp_path / exporter.OUTPUT_BASENAME,
        sheet_name="Templates",
        keep_default_na=False,
    )
    assert list(result["canonical_template"]) == ["live template"]


def test_scid_export_preserves_last_good_on_invalid_open_state(tmp_path):
    exporter = object.__new__(ScidFlagExporter)
    exporter.analyze_flags_dir = str(tmp_path)
    for network in exporter.NETWORKS:
        frame = _empty_history_frame()
        if network == "PRONET":
            frame = pd.DataFrame([{
                "General_Flag": "scid5_psychosis_mood_substance_abuse",
                "canonical_template": "template",
                "is_currently_open": "unknown",
                "variable": "field_a",
            }])
        frame.to_csv(
            tmp_path / open_tracker_row_history_csv_basename(network),
            index=False,
        )
    output = tmp_path / exporter.OUTPUT_BASENAME
    output.write_bytes(b"last-good")

    assert not exporter.run_script()
    assert output.read_bytes() == b"last-good"
