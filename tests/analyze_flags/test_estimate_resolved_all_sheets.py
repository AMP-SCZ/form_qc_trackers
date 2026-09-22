"""Regression coverage for collecting flags from every tracker report tab."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import analyze_flags.estimate_resolved as estimate_resolved_module
from analyze_flags.estimate_resolved import ResolvedEstimator


class _FakeDropbox:
    def check_and_refresh_access_token(self):
        return None

    def request_json_object(self, *_args):
        return {
            "entries": [{
                "rev": "rev-1",
                "server_modified": "2026-01-02T00:00:00Z",
            }],
            "has_more": False,
        }

    def files_download(self, **_kwargs):
        return None, SimpleNamespace(content=b"workbook bytes")


class _RevisionDropbox:
    def __init__(self, revisions):
        self.revisions = revisions

    def check_and_refresh_access_token(self):
        return None

    def request_json_object(self, *_args):
        return {
            "entries": [
                {"rev": rev, "server_modified": when}
                for rev, when in self.revisions
            ],
            "has_more": False,
        }

    def files_download(self, **kwargs):
        return None, SimpleNamespace(
            content=kwargs["rev"].encode("utf-8"))


def _tracker_sheet(specific_flags):
    return pd.DataFrame([{
        "Subject": "P001",
        "Timepoint": "baseline",
        "Additional Comments": "",
        "Manually Marked as Resolved": "",
        "Date Resolved": "",
        "General Flag": "example_form",
        "Specific Flags": specific_flags,
    }])


def _estimator_for_dropbox(dbx):
    estimator = object.__new__(ResolvedEstimator)
    estimator.utils = SimpleNamespace(
        collect_dropbox_credentials=lambda: dbx,
    )
    estimator.sample_every_days = 0
    estimator.jump_threshold = 50
    estimator.template_map = {}
    estimator.variable_map = {}
    estimator._seen_templates = {}
    estimator._seen_variables = {}
    estimator._current_network = "PRESCIENT"
    return estimator


def test_repo_root_discovery_uses_platform_path_semantics():
    assert Path(estimate_resolved_module.parent_dir) == (
        Path(estimate_resolved_module.__file__).resolve().parents[1]
    )


def test_revision_walk_collects_and_deduplicates_every_recognized_sheet(
    monkeypatch,
):
    dbx = _FakeDropbox()
    estimator = object.__new__(ResolvedEstimator)
    estimator.utils = SimpleNamespace(
        collect_dropbox_credentials=lambda: dbx,
    )
    estimator.sample_every_days = 0
    estimator.jump_threshold = 50
    estimator.template_map = {}
    estimator.variable_map = {}
    estimator._seen_templates = {}
    estimator._seen_variables = {}
    estimator._current_network = "PRESCIENT"

    monkeypatch.setattr(
        pd,
        "read_excel",
        lambda *_args, **_kwargs: {
            "Main Report": _tracker_sheet("field_a : Variable is blank."),
            # field_a is deliberately repeated on another tab; its episode
            # key must dedupe while field_b is still collected.
            "Proposed Checks": _tracker_sheet(
                "field_a : Variable is blank. | "
                "field_b : field_b value (99) is out of range"
            ),
            "Instructions": pd.DataFrame({"note": ["not a report"]}),
        },
    )

    unique_rows = {}
    jump_rows = []
    affected_rows = []
    revision_rows = []
    rejected, newest_processed, newest_listed = estimator._walk_revisions(
        "/tracker.xlsx",
        estimator.SOURCE_V2,
        unique_rows,
        jump_rows,
        affected_rows,
        revision_rows,
        page_limit=100,
    )

    assert rejected == 0
    assert newest_processed.isoformat() == "2026-01-02T00:00:00+00:00"
    assert newest_listed.isoformat() == "2026-01-02T00:00:00+00:00"
    assert {key[3] for key in unique_rows} == {"field_a", "field_b"}
    assert len(unique_rows) == 2
    assert revision_rows == [{
        "source": estimator.SOURCE_V2,
        "revision_when": "2026-01-02T00:00:00+00:00",
        "open_entries": 2,
        "added_since_prev": "",
        "removed_since_prev": "",
    }]


def test_revision_walk_ignores_tracker_shaped_backup_sheet(monkeypatch):
    authoritative_reports = [
        "Main Report",
        "Secondary Report",
        "Date Report",
        "Cross Checks",
        "Proposed Checks",
        "Medication Flags",
        "Non Team Forms",
        "Cognition Report",
        "Blood Report",
        "Fluids Report",
        "Scid Report",
        "MRI Report",
        "EEG Report",
        "Digital Report",
        "Incomplete Forms",
        "Missingness Report",
        "Conversion Report",
    ]
    workbook = {
        report: _tracker_sheet(
            f"field_{index} : Variable is blank."
        )
        for index, report in enumerate(authoritative_reports)
    }
    workbook["Old Backup"] = _tracker_sheet(
        "backup_field : Variable is blank."
    )

    dbx = _FakeDropbox()
    estimator = object.__new__(ResolvedEstimator)
    estimator.utils = SimpleNamespace(
        collect_dropbox_credentials=lambda: dbx,
    )
    estimator.sample_every_days = 0
    estimator.jump_threshold = 50
    estimator.template_map = {}
    estimator.variable_map = {}
    estimator._seen_templates = {}
    estimator._seen_variables = {}
    estimator._current_network = "PRESCIENT"
    monkeypatch.setattr(pd, "read_excel", lambda *_args, **_kwargs: workbook)

    unique_rows = {}
    rejected, newest_processed, newest_listed = estimator._walk_revisions(
        "/tracker.xlsx",
        estimator.SOURCE_V2,
        unique_rows,
        [],
        [],
        [],
        page_limit=100,
    )

    assert rejected == 0
    assert newest_processed.isoformat() == "2026-01-02T00:00:00+00:00"
    assert newest_listed.isoformat() == "2026-01-02T00:00:00+00:00"
    assert {key[3] for key in unique_rows} == {
        f"field_{index}" for index in range(len(authoritative_reports))
    }
    assert "backup_field" not in {key[3] for key in unique_rows}


def test_transitional_sheet_coalesces_v2_and_v1_aliases_per_row(
    monkeypatch,
):
    transitional = pd.DataFrame(
        [
            {
                "Subject": "",
                "Participant": "P001",
                "Timepoint": "baseline",
                "Additional Comments": "",
                "Site Comments": "legacy comment",
                "Manually Marked as Resolved": "",
                "Manually Resolved": "",
                "Date Resolved": "",
                "General Flag": "",
                "Form": "legacy_form",
                "Specific Flags": "",
                "Flags": (
                    "legacy_field : legacy_field is equal to 7."
                ),
            },
            {
                "Subject": "P002",
                "Participant": "LEGACY-WRONG-SUBJECT",
                "Timepoint": "baseline",
                "Additional Comments": "v2 comment",
                "Site Comments": "legacy wrong comment",
                "Manually Marked as Resolved": "",
                "Manually Resolved": "",
                "Date Resolved": "",
                "General Flag": "v2_form",
                "Form": "legacy_wrong_form",
                "Specific Flags": (
                    "v2_field : v2_field is equal to 8."
                ),
                "Flags": "wrong_field : Variable is blank.",
            },
            {
                "Subject": "",
                "Participant": "P003",
                "Timepoint": "baseline",
                "Additional Comments": "",
                "Site Comments": "resolved legacy row",
                "Manually Marked as Resolved": "",
                "Manually Resolved": "yes",
                "Date Resolved": "",
                "General Flag": "",
                "Form": "resolved_form",
                "Specific Flags": "",
                "Flags": "resolved_field : Variable is blank.",
            },
        ]
    )

    dbx = _FakeDropbox()
    estimator = object.__new__(ResolvedEstimator)
    estimator.utils = SimpleNamespace(
        collect_dropbox_credentials=lambda: dbx,
    )
    estimator.sample_every_days = 0
    estimator.jump_threshold = 50
    estimator.template_map = {}
    estimator.variable_map = {}
    estimator._seen_templates = {}
    estimator._seen_variables = {}
    estimator._current_network = "PRESCIENT"
    monkeypatch.setattr(
        pd,
        "read_excel",
        lambda *_args, **_kwargs: {"Main Report": transitional},
    )

    unique_rows = {}
    rejected, newest_processed, newest_listed = estimator._walk_revisions(
        "/tracker.xlsx",
        estimator.SOURCE_V2,
        unique_rows,
        [],
        [],
        [],
        page_limit=100,
    )

    assert rejected == 0
    assert newest_processed.isoformat() == "2026-01-02T00:00:00+00:00"
    assert newest_listed.isoformat() == "2026-01-02T00:00:00+00:00"
    assert {(key[0], key[2], key[3]) for key in unique_rows} == {
        ("P001", "legacyform", "legacy_field"),
        ("P002", "v2form", "v2_field"),
    }
    assert all("wrong_field" not in key for key in unique_rows)
    assert all("resolved_field" not in key for key in unique_rows)


def test_newest_revision_with_any_malformed_authoritative_sheet_is_rejected(
    monkeypatch,
):
    dbx = _RevisionDropbox([
        ("rev-new", "2026-01-02T00:00:00Z"),
        ("rev-old", "2026-01-01T00:00:00Z"),
    ])
    estimator = _estimator_for_dropbox(dbx)
    workbooks = {
        "rev-new": {
            "Main Report": _tracker_sheet(
                "partial_field : Variable is blank."),
            "Proposed Checks": pd.DataFrame({"broken": ["schema"]}),
        },
        "rev-old": {
            "Main Report": _tracker_sheet(
                "historical_field : Variable is blank."),
        },
    }
    monkeypatch.setattr(
        pd,
        "read_excel",
        lambda content, **_kwargs: workbooks[
            content.getvalue().decode("utf-8")],
    )

    unique_rows = {}
    rejected, newest_processed, newest_listed = estimator._walk_revisions(
        "/tracker.xlsx",
        estimator.SOURCE_V2,
        unique_rows,
        [],
        [],
        [],
        page_limit=100,
    )

    assert rejected == 0
    assert newest_processed is None
    assert newest_listed.isoformat() == "2026-01-02T00:00:00+00:00"
    assert {key[3] for key in unique_rows} == {"historical_field"}
    assert "partial_field" not in {key[3] for key in unique_rows}


def test_older_malformed_revision_rejects_the_partial_history_walk(monkeypatch):
    dbx = _RevisionDropbox([
        ("rev-new", "2026-01-02T00:00:00Z"),
        ("rev-old", "2026-01-01T00:00:00Z"),
    ])
    estimator = _estimator_for_dropbox(dbx)
    workbooks = {
        "rev-new": {
            "Main Report": _tracker_sheet(
                "current_field : Variable is blank."),
        },
        "rev-old": {
            "Main Report": _tracker_sheet(
                "historical_field : Variable is blank."),
            "Proposed Checks": pd.DataFrame({"broken": ["schema"]}),
        },
    }
    monkeypatch.setattr(
        pd,
        "read_excel",
        lambda content, **_kwargs: workbooks[
            content.getvalue().decode("utf-8")],
    )

    unique_rows = {}
    rejected, newest_processed, newest_listed = estimator._walk_revisions(
        "/tracker.xlsx",
        estimator.SOURCE_V2,
        unique_rows,
        [],
        [],
        [],
        page_limit=100,
    )

    assert rejected == 0
    assert newest_processed is None
    assert newest_listed.isoformat() == "2026-01-02T00:00:00+00:00"
    # The caller stages this dictionary and discards it because the returned
    # completeness signal is None.
    assert {key[3] for key in unique_rows} == {"current_field"}


def test_revision_gap_creates_distinct_episodes_and_resolution_timestamp(
    tmp_path,
    monkeypatch,
):
    dbx = _RevisionDropbox([
        ("rev-3", "2026-01-03T00:00:00Z"),
        ("rev-2", "2026-01-02T00:00:00Z"),
        ("rev-1", "2026-01-01T00:00:00Z"),
    ])
    estimator = _estimator_for_dropbox(dbx)
    workbooks = {
        "rev-3": {
            "Main Report": _tracker_sheet(
                "field_a : field_a is equal to 2."),
        },
        "rev-2": {"Main Report": _tracker_sheet("")},
        "rev-1": {
            "Main Report": _tracker_sheet(
                "field_a : field_a is equal to 1."),
        },
    }
    monkeypatch.setattr(
        pd,
        "read_excel",
        lambda content, **_kwargs: workbooks[
            content.getvalue().decode("utf-8")],
    )

    unique_rows = {}
    rejected, newest_processed, newest_listed = estimator._walk_revisions(
        "/tracker.xlsx",
        estimator.SOURCE_V2,
        unique_rows,
        [],
        [],
        [],
        page_limit=100,
    )

    assert rejected == 0
    assert newest_processed == newest_listed
    assert len(unique_rows) == 2
    assert {record["message"] for record in unique_rows.values()} == {
        "field_a is equal to 1.",
        "field_a is equal to 2.",
    }

    estimator.analyze_flags_dir = str(tmp_path)
    estimator._write_history_csv(
        "PRESCIENT",
        unique_rows,
        {
            estimator.SOURCE_V1: None,
            estimator.SOURCE_V2: newest_processed,
        },
    )
    history = pd.read_csv(
        tmp_path / "open_tracker_row_history_PRESCIENT.csv",
        keep_default_na=False,
    ).sort_values("Earliest_seen").reset_index(drop=True)

    assert len(history) == 2
    assert history["episode_id"].nunique() == 2
    assert history.loc[0, "message"] == "field_a is equal to 1."
    assert str(history.loc[0, "Resolution_observed"]).startswith(
        "2026-01-02")
    assert not bool(history.loc[0, "is_currently_open"])
    assert history.loc[1, "message"] == "field_a is equal to 2."
    assert history.loc[1, "Resolution_observed"] == ""
    assert bool(history.loc[1, "is_currently_open"])


def test_same_occurrence_ordinals_merge_across_tracker_sources(monkeypatch):
    revisions_v1 = [
        ("v1-3", "2026-01-03T00:00:00Z"),
        ("v1-2", "2026-01-02T00:00:00Z"),
        ("v1-1", "2026-01-01T00:00:00Z"),
    ]
    revisions_v2 = [
        ("v2-3", "2026-01-03T01:00:00Z"),
        ("v2-2", "2026-01-02T01:00:00Z"),
        ("v2-1", "2026-01-01T01:00:00Z"),
    ]
    workbooks = {
        "v1-3": {"Main Report": _tracker_sheet(
            "field_a : field_a is equal to 2.")},
        "v1-2": {"Main Report": _tracker_sheet("")},
        "v1-1": {"Main Report": _tracker_sheet(
            "field_a : field_a is equal to 1.")},
        "v2-3": {"Main Report": _tracker_sheet(
            "field_a : field_a is equal to 2.")},
        "v2-2": {"Main Report": _tracker_sheet("")},
        "v2-1": {"Main Report": _tracker_sheet(
            "field_a : field_a is equal to 1.")},
    }
    monkeypatch.setattr(
        pd,
        "read_excel",
        lambda content, **_kwargs: workbooks[
            content.getvalue().decode("utf-8")],
    )
    estimator = _estimator_for_dropbox(_RevisionDropbox(revisions_v1))
    unique_rows = {}

    estimator._walk_revisions(
        "/v1.xlsx", estimator.SOURCE_V1, unique_rows, [], [], [])
    estimator.utils = SimpleNamespace(
        collect_dropbox_credentials=lambda: _RevisionDropbox(revisions_v2))
    estimator._walk_revisions(
        "/v2.xlsx", estimator.SOURCE_V2, unique_rows, [], [], [])

    assert len(unique_rows) == 2
    assert all(
        record["sources"] == {estimator.SOURCE_V1, estimator.SOURCE_V2}
        for record in unique_rows.values()
    )
