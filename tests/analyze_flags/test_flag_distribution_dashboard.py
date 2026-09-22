"""Regression coverage for the SCID outcome-input flag distribution."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from analyze_flags import flag_distribution_dashboard as dashboard
from analyze_flags.canonicalize import explode_specific_flags, normalize_form_name
from analyze_flags.flag_distribution_dashboard import (
    SCID_FORM_KEYS,
    load_scid_outcome_variables,
    scid_outcome_matches,
)


SCID_FORM = "scid5_psychosis_mood_substance_abuse"
SCID_PERSONALITY_FORM = "SCID5 Schizotypal Personality SCIDDPQ"
NETWORKS = ["PRONET", "PRESCIENT"]


@pytest.fixture(autouse=True)
def isolated_dashboard_config(tmp_path, monkeypatch):
    """Exercise configured lookup using shipped schema, never host config."""
    project = tmp_path / "dashboard_project"
    dependencies = project / "dependencies"
    dependencies.mkdir(parents=True)
    source = (
        Path(__file__).resolve().parents[3]
        / "dependencies" / "scid_outcome_variables.json"
    )
    (dependencies / source.name).write_bytes(source.read_bytes())
    (project / "config.json").write_text(
        json.dumps({"paths": {"dependencies_path": "dependencies"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "PROJECT_ROOT", str(project))


def test_missing_configured_dependency_does_not_fall_back(tmp_path):
    project = Path(dashboard.PROJECT_ROOT)
    (project / "config.json").write_text(
        json.dumps({"paths": {"dependencies_path": "missing_dependencies"}}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        load_scid_outcome_variables()


def _records(form, flags):
    return [
        (normalize_form_name(form), form, template, variable, message)
        for variable, message, template in explode_specific_flags(flags)
    ]


def _aggregate():
    agg = dashboard.Aggregate(NETWORKS)
    agg.add("PRONET", _records(
        SCID_FORM,
        "chrscid_c37 : Variable is blank. | "
        "chrscid_c11 : Conflicts with [chrscid_c14] and chrscid_b1. | "
        "chrscid_d63 : Variable is blank. | "
        "chrscid_d63___2 : Variable is blank. | "
        "chrscid_c11 : Variable is blank. | "
        "chrscid_sz : Variable is blank.",
    ))
    agg.add("PRONET", _records("other_form", "chrscid_c37 : Variable is blank."))
    agg.add("PRESCIENT", _records(
        SCID_PERSONALITY_FORM,
        "chrscid_c11 : Conflicts with chrscid_c370. | "
        "chrscid_c11 : Conflicts with chrscid_a1.",
    ))
    return agg


def _render(agg):
    return dashboard.render_html(
        agg, NETWORKS, {name: dashboard.LoadStats() for name in NETWORKS},
        top_forms=5, top_types=8, open_only=False, title="Flag distribution",
    )


def _scid_section(markup):
    sections = re.findall(r'<section\b[^>]*>.*?</section>', markup, re.S)
    assert len(sections) == 3
    return sections[-1]


def test_outcome_inputs_include_gates_and_checkbox_choices_but_not_outputs(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    variables = load_scid_outcome_variables()

    assert isinstance(variables, frozenset)
    assert len(variables) == 49
    assert {
        "chrscid_c37", "chrscid_c14", "chrscid_a1", "chrscid_b1",
        "chrscid_c27", "chrscid_d47_d52___1", "chrscid_d63___1",
        "chrscid_no_drug_lifetime___1", "chrscid_cannabis_yn",
        "chrscid_e138_139", "chrscid_e303_305",
    } <= variables
    assert variables.isdisjoint({
        "chrscid_sz", "chrscid_delusional", "chrscid_any_psychosis",
        "chrscid_any_mood", "chrscid_c11", "chrscid_d63___2",
    })


@pytest.mark.parametrize("absolute_path", [False, True])
def test_outcome_inputs_load_from_configured_dependencies_path(
        tmp_path, monkeypatch, absolute_path):
    project_dir = tmp_path / "project"
    dependencies_dir = project_dir / "custom_dependencies"
    dependencies_dir.mkdir(parents=True)
    configured_path = (
        str(dependencies_dir) if absolute_path else "custom_dependencies"
    )
    (project_dir / "config.json").write_text(
        json.dumps({"paths": {"dependencies_path": configured_path}}),
        encoding="utf-8",
    )
    (dependencies_dir / "scid_outcome_variables.json").write_text(
        json.dumps(["chrscid_configured_input", "chrscid_configured_gate"]),
        encoding="utf-8",
    )
    unrelated_dir = tmp_path / "unrelated"
    unrelated_dir.mkdir()
    monkeypatch.chdir(unrelated_dir)
    monkeypatch.setattr(dashboard, "PROJECT_ROOT", str(project_dir))

    assert load_scid_outcome_variables() == frozenset({
        "chrscid_configured_input", "chrscid_configured_gate",
    })


def test_outcome_inputs_load_from_json_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "PROJECT_ROOT", str(tmp_path / "no_config"))
    source = tmp_path / "scid_outcome_variables.json"
    source.write_text(
        json.dumps(["chrscid_new_input", "chrscid_gate"]),
        encoding="utf-8",
    )

    assert load_scid_outcome_variables(source) == frozenset({
        "chrscid_new_input", "chrscid_gate",
    })


@pytest.mark.parametrize("contents", [{}, [], [123], ["not_a_scid_variable"]])
def test_outcome_inputs_reject_invalid_dependency_lists(tmp_path, contents):
    source = tmp_path / "scid_outcome_variables.json"
    source.write_text(json.dumps(contents), encoding="utf-8")

    with pytest.raises(ValueError):
        load_scid_outcome_variables(source)


@pytest.mark.parametrize(("variable", "message", "expected"), [
    ("chrscid_c37", "Variable is blank.", {"chrscid_c37"}),
    ("chrscid_c11", "Conflicts with [chrscid_c37] and chrscid_b1.",
     {"chrscid_c37", "chrscid_b1"}),
    ("CHRSCID_C37", "CHRSCID_C37 is repeated.", {"chrscid_c37"}),
    ("chrscid_d63", "Variable is blank.", {"chrscid_d63___1"}),
    ("chrscid_d63___1", "Variable is blank.", {"chrscid_d63___1"}),
    ("chrscid_d63___2", "Variable is blank.", set()),
    ("chrscid_c370", "chrscid_c37_extra and prefix_chrscid_c37", set()),
    ("chrscid_sz", "Variable is blank.", set()),
])
def test_input_matching_uses_identifiers_and_exact_checkbox_choices(
        variable, message, expected):
    assert scid_outcome_matches(
        variable, message, load_scid_outcome_variables()) == frozenset(expected)


def test_aggregation_counts_each_scid_entry_once_and_preserves_old_totals():
    agg = _aggregate()

    assert SCID_FORM_KEYS == {
        normalize_form_name(SCID_FORM), normalize_form_name(SCID_PERSONALITY_FORM),
    }
    assert agg.scid_by_network == {"PRONET": 6, "PRESCIENT": 2}
    assert agg.scid_outcome_by_network == {"PRONET": 3, "PRESCIENT": 1}
    assert agg.total == 9
    assert agg.network_totals == {"PRONET": 7, "PRESCIENT": 2}
    assert sum(agg.form_totals.values()) == agg.total
    for form, total in agg.form_totals.items():
        assert sum(agg.type_by_form[form].values()) == total
        assert sum(row[2] for row in agg.types_for_form(form, top_types=1)) == total


def test_sheetless_main_report_and_open_only_filters_apply_to_new_counts(tmp_path):
    source = tmp_path / "flags.csv"
    pd.DataFrame({
        "displayed_form": [SCID_FORM] * 4 + ["other_form"],
        "error_message": [
            "chrscid_c37 : Variable is blank. | chrscid_c11 : Variable is blank.",
            "chrscid_c37 : Variable is blank.",
            "chrscid_c37 : Variable is blank.",
            "chrscid_c37 : Variable is blank.",
            "chrscid_c37 : Variable is blank.",
        ],
        "reports": [
            "Main Report | Date Report", "Date Report", "Not Main Report",
            "Main Report", "Main Report",
        ],
        "currently_resolved": [False, False, False, True, False],
        "Subject": ["PRIVATE_SUBJECT"] * 5,
    }).to_csv(source, index=False)
    stats = dashboard.LoadStats()

    records = dashboard.load_entries([str(source)], "PRONET", stats, open_only=True)
    agg = dashboard.Aggregate(["PRONET"])
    agg.add("PRONET", records)

    assert stats.rows_read == 5
    assert stats.rows_not_main == 2
    assert stats.rows_resolved == 1
    assert stats.rows_used == 2
    assert stats.entries == 3
    assert agg.scid_by_network["PRONET"] == 2
    assert agg.scid_outcome_by_network["PRONET"] == 1
    assert records[0][3:] == ("chrscid_c37", "Variable is blank.")
    assert "PRIVATE_SUBJECT" not in repr(records)


def test_workbook_scope_and_all_sheets_switch_apply_to_new_counts(tmp_path, monkeypatch):
    source = tmp_path / "flags.xlsx"
    source.touch()
    main = pd.DataFrame({"Form": [SCID_FORM], "Flags": [
        "chrscid_c11 : Conflicts with chrscid_c37.",
    ]})
    secondary = pd.DataFrame({"Form": [SCID_FORM], "Flags": [
        "chrscid_c37 : Variable is blank.",
    ]})
    monkeypatch.setattr(dashboard, "read_tables", lambda _path: iter([
        (" Main Report ", main), ("Secondary Report", secondary),
    ]))

    for main_only, expected in [(True, 1), (False, 2)]:
        stats = dashboard.LoadStats()
        records = dashboard.load_entries(
            [str(source)], "PRONET", stats, open_only=False,
            main_report_only=main_only,
        )
        agg = dashboard.Aggregate(["PRONET"])
        agg.add("PRONET", records)
        assert agg.scid_by_network["PRONET"] == expected
        assert agg.scid_outcome_by_network["PRONET"] == expected
        assert bool(stats.skipped_sheets) is main_only


def test_final_chart_and_backing_csv_use_scid_denominator(tmp_path):
    agg = _aggregate()
    section = _scid_section(_render(agg))

    assert "Involving outcome variables" in section
    assert "Other SCID flags" in section
    assert "50.0%" in section
    assert "44.4%" not in section
    assert "% of SCID flags" in section
    assert "outcome_calculations.py" in section
    assert "scid_outcome_variables.json" in section
    assert "PRONET" in section and "PRESCIENT" in section

    paths = dashboard.write_backing_csvs(
        agg, NETWORKS, top_forms=5, top_types=8,
        out_html=str(tmp_path / "distribution.html"),
    )
    scid_path = tmp_path / "distribution_scid_outcome_flags.csv"
    assert str(scid_path) in paths
    rows = pd.read_csv(scid_path).set_index("group")
    assert list(rows.columns) == ["PRONET", "PRESCIENT", "total", "pct_of_scid_flags"]
    assert rows.loc["Involving outcome variables"].to_dict() == {
        "PRONET": 3, "PRESCIENT": 1, "total": 4, "pct_of_scid_flags": 50.0,
    }
    assert rows.loc["Other SCID flags"].to_dict() == {
        "PRONET": 3, "PRESCIENT": 1, "total": 4, "pct_of_scid_flags": 50.0,
    }
    assert len(paths) == 3
    assert all(Path(path).exists() for path in paths)


def test_dashboard_with_no_scid_entries_still_exports_zero_counts(tmp_path):
    agg = dashboard.Aggregate(NETWORKS)
    agg.add("PRONET", _records("other_form", "chrscid_c37 : Variable is blank."))
    section = _scid_section(_render(agg))

    assert "Involving outcome variables" in section
    assert "Other SCID flags" in section
    assert "0.0%" in section
    assert "nan" not in section.lower()
    assert "inf%" not in section.lower()

    dashboard.write_backing_csvs(
        agg, NETWORKS, top_forms=5, top_types=8,
        out_html=str(tmp_path / "empty.html"),
    )
    rows = pd.read_csv(tmp_path / "empty_scid_outcome_flags.csv")
    assert len(rows) == 2
    assert rows[["PRONET", "PRESCIENT", "total", "pct_of_scid_flags"]].eq(0).all().all()

