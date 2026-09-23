"""Tests for the combined-CSV source of the site radar generator.

These cover the claims the combined source makes that the anomaly-report
source cannot: the roster is every site in the study, the plotted number is a
measured statistic, and an empty spoke is a data-availability fact.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from site_network_radar import (
    RadarInputError,
    build_combined_scores,
    discover_combined_files,
    generate_radar_charts_from_combined,
    read_combined_site_statistics,
    select_representative_site_scores,
)

VARIABLE = "chrbprs_bprs_total"


def _write_slice(directory: Path, timepoint: str, network: str,
                 values_by_site: dict[str, list], *,
                 include_variable: bool = True) -> Path:
    rows = []
    for site, values in values_by_site.items():
        for index, value in enumerate(values):
            rows.append({"subjectid": f"{site}{index:05d}",
                         VARIABLE: "" if value is None else str(value)})
    frame = pd.DataFrame(rows)
    if not include_variable:
        frame = frame.drop(columns=[VARIABLE])
    path = (directory
            / f"AMPSCZ-combined-redcap_{timepoint}_{network}.csv")
    frame.to_csv(path, index=False)
    return path


def _basic_input(tmp_path: Path) -> Path:
    directory = tmp_path / "combined"
    directory.mkdir()
    _write_slice(directory, "baseline", "PRONET", {
        "AA": [30] * 40,
        "BB": [40] * 40,
        "CC": [50] * 40,
        "DD": [35] * 40,
        "EE": [45] * 40,
        # Present in the study but with a single usable observation.
        "FF": [60] + [None] * 9,
    })
    return directory


def test_filenames_resolve_network_and_exact_timepoint(tmp_path: Path):
    directory = tmp_path / "combined"
    directory.mkdir()
    for timepoint in ("baseline", "month_1", "month_12"):
        _write_slice(directory, timepoint, "PRONET", {"AA": [30] * 5})
    entries, _ = discover_combined_files(directory, None, ["month1"])
    assert [entry["timepoint"] for entry in entries] == ["month1"]
    assert all(entry["network"] == "PRONET" for entry in entries)


def test_missing_codes_are_not_treated_as_values(tmp_path: Path):
    directory = tmp_path / "combined"
    directory.mkdir()
    _write_slice(directory, "baseline", "PRONET", {
        "AA": [40, 40, 40, -9, 999, -3],
    })
    statistics, _, _ = read_combined_site_statistics(directory, [VARIABLE])
    row = statistics.iloc[0]
    assert row["n_subjects"] == 6
    assert row["n_observed"] == 3
    assert row["site_median"] == 40
    assert row["site_missing_pct"] == pytest.approx(50.0)


def test_roster_includes_sites_from_uncharted_timepoints(tmp_path: Path):
    directory = _basic_input(tmp_path)
    # A site whose first data lands at month 1 is still a study site, so it
    # must appear on the baseline chart as a site with no data there.
    _write_slice(directory, "month_1", "PRONET", {"ZZ": [40] * 30})
    statistics, roster, _ = read_combined_site_statistics(
        directory, [VARIABLE], timepoints=["baseline"])
    assert "ZZ" in set(roster["site_id"])
    assert "ZZ" not in set(statistics["site_id"])

    scores, universe = build_combined_scores(
        statistics, roster, [VARIABLE], reference_min_n=30)
    assert "ZZ" in universe["PRONET | baseline"]
    late = scores.loc[scores["site_id"] == "ZZ"].iloc[0]
    assert not late["recorded_in_input"]
    assert pd.isna(late["plot_radius"])


def test_every_roster_site_gets_exactly_one_spoke_row(tmp_path: Path):
    directory = _basic_input(tmp_path)
    statistics, roster, _ = read_combined_site_statistics(
        directory, [VARIABLE], timepoints=["baseline"])
    scores, universe = build_combined_scores(
        statistics, roster, [VARIABLE], reference_min_n=30)
    group = scores.loc[scores["chart_group"] == "PRONET | baseline"]
    assert sorted(group["site_id"]) == sorted(universe["PRONET | baseline"])
    assert not group["site_id"].duplicated().any()


def test_radius_is_measured_distance_from_the_cross_site_median(
        tmp_path: Path):
    directory = _basic_input(tmp_path)
    statistics, roster, _ = read_combined_site_statistics(
        directory, [VARIABLE], timepoints=["baseline"])
    scores, _ = build_combined_scores(
        statistics, roster, [VARIABLE], reference_min_n=30)
    indexed = scores.set_index("site_id")
    # Reference is the median of the five sites with n >= 30: 30/35/40/45/50.
    assert indexed.loc["AA", "reference_value"] == pytest.approx(40.0)
    assert indexed.loc["AA", "raw_value"] == pytest.approx(30.0)
    assert indexed.loc["AA", "plot_radius"] == pytest.approx(10.0)
    assert indexed.loc["AA", "deviation_direction"] == pytest.approx(-1.0)
    assert indexed.loc["CC", "deviation_direction"] == pytest.approx(1.0)
    # The small-n site is charted but must not shape the reference.
    assert indexed.loc["FF", "recorded_in_input"]
    assert indexed.loc["FF", "site_n"] == 1


def test_min_n_leaves_a_thin_site_labelled_but_unplotted(tmp_path: Path):
    directory = _basic_input(tmp_path)
    statistics, roster, _ = read_combined_site_statistics(
        directory, [VARIABLE], timepoints=["baseline"])
    scores, universe = build_combined_scores(
        statistics, roster, [VARIABLE], min_n=5, reference_min_n=30)
    thin = scores.set_index("site_id").loc["FF"]
    assert not thin["recorded_in_input"]
    assert pd.isna(thin["plot_radius"])
    # Still a spoke: dropping it would hide a real site.
    assert "FF" in universe["PRONET | baseline"]


def test_representative_selection_keeps_normal_anchor_and_extremes(
        tmp_path: Path):
    directory = _basic_input(tmp_path)
    output = tmp_path / "selected"
    result = generate_radar_charts_from_combined(
        directory,
        VARIABLE,
        output_dir=output,
        timepoints=["baseline"],
        scope="network",
        reference_min_n=30,
        site_limit=3,
        normal_site_count=1,
        dpi=72,
    )

    # Six available sites become three spokes. BB is the one near-reference
    # anchor; FF is the strongest deviation, and AA wins the tied next-extreme
    # slot deterministically by site id.
    assert result["sites_by_network"] == {
        "PRONET | baseline": ["AA", "BB", "FF"]}
    assert result["spoke_site_count"] == 3
    assert result["available_spoke_site_count"] == 6

    scores = pd.read_csv(output / "site_deviation_scores.csv")
    assert len(scores) == 6
    assert scores["reference_value"].eq(40.0).all()
    selected = scores.loc[scores["selected_for_chart"]].set_index("site_id")
    assert set(selected.index) == {"AA", "BB", "FF"}
    assert selected.loc["BB", "site_selection_role"] == "near_reference"
    assert selected.loc["BB", "reference_pool_member"]
    assert set(selected.query(
        "site_selection_role == 'largest_deviation'").index) == {"AA", "FF"}
    omitted = scores.loc[~scores["selected_for_chart"]]
    assert omitted["site_selection_role"].eq("not_selected").all()
    assert omitted["site_rank_by_deviation"].notna().all()

    manifest = pd.read_csv(output / "chart_manifest.csv")
    assert set(manifest["site_id"]) == {"AA", "BB", "FF"}
    assert manifest["spokes_total"].eq(3).all()
    selection = pd.read_csv(output / "selected_variables.csv").iloc[0]
    assert selection["available_site_count"] == 6
    assert selection["spoke_site_count"] == 3

    readme = (output / "README.txt").read_text(encoding="utf-8")
    assert "up to 3 measured sites per chart" in readme
    assert "computed before display selection" in readme
    assert "not a clinical classification" in readme


def test_fifteen_site_limit_guarantees_the_largest_deviations():
    site_ids = [f"S{rank:02d}" for rank in range(1, 21)]
    scores = pd.DataFrame({
        "chart_group": ["GROUP"] * 20,
        "variable": [VARIABLE] * 20,
        "site_id": site_ids,
        "recorded_in_input": [True] * 20,
        "raw_deviation": list(range(20, 0, -1)),
        "site_rank_by_deviation": list(range(1, 21)),
        "site_n": [100] * 20,
        "reference_pool_member": [True] * 20,
    })

    marked = select_representative_site_scores(
        scores, site_limit=15, normal_site_count=2)
    selected = marked.loc[marked["selected_for_chart"]]
    largest = selected.loc[
        selected["site_selection_role"] == "largest_deviation"]

    assert len(selected) == 15
    assert set(largest["site_id"]) == {
        f"S{rank:02d}" for rank in range(1, 14)}
    assert selected.loc[selected["site_selection_role"] == "near_reference",
                        "site_id"].tolist() == ["S19", "S20"]
    assert "S01" in set(selected["site_id"])


def test_selection_uses_full_precision_rank_and_reference_pool():
    scores = pd.DataFrame({
        "chart_group": ["GROUP"] * 5,
        "variable": [VARIABLE] * 5,
        "site_id": list("ABCDE"),
        "recorded_in_input": [True] * 5,
        # Deliberately identical rounded values: rank retains the true ordering.
        "raw_deviation": [1.0] * 5,
        "site_rank_by_deviation": [1, 2, 3, 4, 5],
        "site_n": [1, 100, 100, 100, 100],
        # E is closest but did not qualify to shape the reference.
        "reference_pool_member": [True, True, True, True, False],
    })

    largest_only = select_representative_site_scores(
        scores, site_limit=3, normal_site_count=0)
    assert set(largest_only.loc[
        largest_only["selected_for_chart"], "site_id"]) == {"A", "B", "C"}

    with_anchor = select_representative_site_scores(
        scores, site_limit=3, normal_site_count=1)
    anchor = with_anchor.loc[
        with_anchor["site_selection_role"] == "near_reference"].iloc[0]
    assert anchor["site_id"] == "D"
    assert anchor["reference_pool_member"]

    limited_pool = scores.copy()
    limited_pool["reference_pool_member"] = [False, False, False, True, False]
    over_requested = select_representative_site_scores(
        limited_pool, site_limit=3, normal_site_count=10)
    assert over_requested.loc[
        over_requested["site_selection_role"] == "near_reference",
        "site_id"].tolist() == ["D"]
    assert (over_requested["site_selection_role"] == "largest_deviation").sum() == 2


@pytest.mark.parametrize(
    "site_limit", [0, -1, 1.5, float("nan"), float("inf"), True])
def test_representative_selection_rejects_invalid_limit(
        tmp_path: Path, site_limit: object):
    directory = _basic_input(tmp_path)
    statistics, roster, _ = read_combined_site_statistics(
        directory, [VARIABLE], timepoints=["baseline"])
    scores, _ = build_combined_scores(statistics, roster, [VARIABLE])
    with pytest.raises(RadarInputError, match="site_limit"):
        select_representative_site_scores(scores, site_limit=site_limit)

@pytest.mark.parametrize(
    "normal_count", [-1, 1.5, float("nan"), float("inf"), True])
def test_representative_selection_rejects_invalid_normal_count(
        tmp_path: Path, normal_count: object):
    directory = _basic_input(tmp_path)
    statistics, roster, _ = read_combined_site_statistics(
        directory, [VARIABLE], timepoints=["baseline"])
    scores, _ = build_combined_scores(statistics, roster, [VARIABLE])
    with pytest.raises(RadarInputError, match="normal_site_count"):
        select_representative_site_scores(
            scores, site_limit=15, normal_site_count=normal_count)


def test_representative_selection_rejects_multiple_variables(tmp_path: Path):
    directory = _basic_input(tmp_path)
    with pytest.raises(RadarInputError, match="exactly one variable"):
        generate_radar_charts_from_combined(
            directory, [VARIABLE, "second_variable"], site_limit=15)


def test_reference_falls_back_and_warns_when_too_few_sites_qualify(
        tmp_path: Path):
    directory = tmp_path / "combined"
    directory.mkdir()
    _write_slice(directory, "baseline", "PRONET", {
        "AA": [30] * 4, "BB": [40] * 4, "CC": [50] * 4,
    })
    statistics, roster, _ = read_combined_site_statistics(directory, [VARIABLE])
    warnings: list[str] = []
    scores, _ = build_combined_scores(
        statistics, roster, [VARIABLE], reference_min_n=30, warnings=warnings)
    assert any("falls back" in item for item in warnings)
    assert scores["reference_value"].dropna().iloc[0] == pytest.approx(40.0)
    assert scores["reference_basis"].str.startswith(
        "fallback cross-site median").all()
    assert scores["reference_pool_member"].all()


def test_study_scope_charts_both_networks_against_one_reference(
        tmp_path: Path):
    directory = tmp_path / "combined"
    directory.mkdir()
    _write_slice(directory, "baseline", "PRONET",
                 {"AA": [30] * 40, "BB": [40] * 40})
    _write_slice(directory, "baseline", "PRESCIENT",
                 {"CC": [50] * 40, "DD": [60] * 40})
    statistics, roster, _ = read_combined_site_statistics(directory, [VARIABLE])
    scores, universe = build_combined_scores(
        statistics, roster, [VARIABLE], scope="study", reference_min_n=30)
    assert sorted(universe["STUDY | baseline"]) == ["AA", "BB", "CC", "DD"]
    assert scores["reference_value"].nunique() == 1


def test_variable_absent_everywhere_raises_a_clear_error(tmp_path: Path):
    directory = _basic_input(tmp_path)
    with pytest.raises(RadarInputError, match="contain the requested"):
        read_combined_site_statistics(directory, ["chrxyz_not_a_variable"])


def test_end_to_end_writes_charts_and_audit_tables(tmp_path: Path):
    directory = _basic_input(tmp_path)
    _write_slice(directory, "month_1", "PRONET", {"ZZ": [40] * 30})
    output = tmp_path / "out"
    result = generate_radar_charts_from_combined(
        directory, VARIABLE, output_dir=output, timepoints=["baseline"],
        scope="both", output_format="both", reference_min_n=30)

    assert result["chart_count"] == 2          # PRONET and STUDY pages
    assert list(output.glob("*.png"))
    assert (output / "site_network_radar.pdf").exists()
    for name in ("site_statistics.csv", "site_deviation_scores.csv",
                 "selected_variables.csv", "chart_manifest.csv",
                 "README.txt"):
        assert (output / name).exists(), name

    manifest = pd.read_csv(output / "chart_manifest.csv")
    # Every site, including the month-1-only one, is on every page.
    pronet = manifest.loc[manifest["chart_group"] == "PRONET | baseline"]
    assert "ZZ" in set(pronet["site_id"])
    assert not pronet.loc[pronet["site_id"] == "ZZ",
                          "recorded_in_input"].any()

    readme = (output / "README.txt").read_text(encoding="utf-8")
    # The report source's caveat must not follow a measured chart.
    assert "thresholded" in readme       # only as an explicit disclaimer
    assert "no flagged evidence in this thresholded report" not in readme


def test_output_directory_must_be_empty(tmp_path: Path):
    directory = _basic_input(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "stale.png").write_bytes(b"")
    with pytest.raises(RadarInputError, match="not empty"):
        generate_radar_charts_from_combined(
            directory, VARIABLE, output_dir=output, timepoints=["baseline"])
