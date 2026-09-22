"""Focused tests for the standalone site-network radar generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from site_network_radar import (
    RadarInputError,
    calculate_site_scores,
    expand_site_rows,
    generate_radar_charts,
    read_site_network_table,
)


MEDIAN = "cross-site MAD on site medians"
SPREAD = "log2 IQR ratio vs cross-site median IQR"
MISSING = "cross-site MAD on site missingness"
CORRELATION = "standardized Fisher-z difference: site vs rest of network"


def _row(site: str, variable: str, severity: float, method: str,
         **extra) -> dict:
    try:
        raw = float(severity) / 10.0
    except (TypeError, ValueError):
        raw = 1.0
    row = {
        "network": "PRONET",
        "timepoint": "baseline",
        "site_id": site,
        "variable": variable,
        "severity_score": severity,
        "raw_score": raw,
        "observed_value": f"site median {raw + 10:.4g} (n=10)",
        "expected_value": "cross-site median 10, sigma 1",
        "method": method,
    }
    if "iqr" in method.lower() or "spread" in method.lower():
        row["observed_value"] = f"site IQR {raw + 2:.4g}"
        row["expected_value"] = "cross-site median IQR 2"
    elif "fisher" in method.lower() or "correlation" in method.lower():
        row["observed_value"] = f"site rho={min(0.99, raw / 10):+.2f} (n=10)"
        row["expected_value"] = "rest-of-network rho=+0.10 (n=90)"
    elif "missing" in method.lower():
        row["observed_value"] = f"site missing rate {raw:.2f}%"
        row["expected_value"] = "cross-site missing-rate median 0.00%"
    row.update(extra)
    return row


def test_collapsed_sites_use_parallel_severities_without_zero_imputation():
    frame = pd.DataFrame([
        # The representative scalar is deliberately 99: each expanded site
        # must use its own parallel severity (80 / 60), never this value.
        _row("(2 sites)", "height", 99, MEDIAN,
             sites="AA, BB", site_severities="80, 60",
             observed_value=(
                 "AA: site median 18 (n=10); "
                 "BB: site median 16 (n=10)"),
             expected_value="cross-site median 10, sigma 1"),
        _row("AA", "height", 90, MEDIAN),  # duplicate issue: keep max 90
        _row("AA", "weight", 70, MEDIAN),
        _row("AA", "age", 50, MEDIAN),
        _row("AA", "score", 10, MEDIAN),
        _row("AA", "height", 60, SPREAD),
        _row("AA", "height", 40, MISSING),
        _row("AA", "height | weight", 80, CORRELATION),
    ])

    expanded, warnings = expand_site_rows(frame)
    assert warnings == []
    collapsed_aa = expanded[
        (expanded["site_id"] == "AA")
        & (expanded["variable"] == "height")
        & (expanded["family"] == "median")
    ]["severity_score"].tolist()
    assert collapsed_aa == [80.0, 90.0]
    assert expanded[
        (expanded["site_id"] == "BB")
        & (expanded["variable"] == "height")
    ]["severity_score"].tolist() == [60.0]

    scores, evidence = calculate_site_scores(expanded, source_rows=8)
    axes = (scores[["variable", "variable_order"]].drop_duplicates()
            .sort_values("variable_order"))
    assert axes["variable"].tolist() == [
        "height", "weight", "age", "score", "height | weight"]

    aa = scores[scores["site_id"] == "AA"].set_index("variable")
    bb = scores[scores["site_id"] == "BB"].set_index("variable")
    # Repeated height evidence across method families becomes one recorded
    # site-variable value and uses its strongest calibrated severity.
    assert aa.loc["height", "severity_score"] == 90.0
    assert aa.loc["weight", "severity_score"] == 70.0
    assert aa.loc["height | weight", "severity_score"] == 80.0
    assert aa["recorded_in_input"].sum() == 5
    assert bb.loc["height", "severity_score"] == 60.0
    assert bb["recorded_in_input"].sum() == 1
    # BB has no recorded weight evidence, so there is no synthesized BB-weight
    # row at zero or at the reference center.
    assert "weight" not in bb.index
    assert len(scores) == 6
    assert scores["recorded_in_input"].all()
    assert (scores["plot_radius"] == scores["raw_deviation"]).all()
    assert (scores["reference_radius"] == 0.0).all()
    # A lone flagged site stays at its raw distance from the detector reference.
    # It is not re-centered on the median of the thresholded output.
    assert aa.loc["weight", "raw_deviation"] == 7.0
    assert aa.loc["weight", "raw_value"] == 17.0
    assert aa.loc["weight", "recorded_site_count"] == 1
    assert len(evidence[
        (evidence["site_id"] == "AA")
        & (evidence["variable"] == "height")
    ]) == 1


def test_first_twenty_source_rows_choose_axes_but_later_evidence_is_used():
    rows = [
        _row("(2 sites)", "var_00", 99, MEDIAN,
             sites="AA, BB", site_severities="70, 60",
             observed_value=(
                 "AA: site median 17 (n=10); "
                 "BB: site median 16 (n=10)"),
             expected_value="cross-site median 10, sigma 1"),
    ]
    rows.extend(
        _row("AA", f"var_{index:02d}", 50 + index, MEDIAN)
        for index in range(1, 25)
    )
    # These rows occur after the 20-row spoke-selection window. They must not
    # add axes, but they must contribute evidence for already-selected var_00.
    rows.extend([
        _row("AA", "var_00", 95, SPREAD, timepoint="month6"),
        _row("CC", "var_00", 88, MISSING, timepoint="month12"),
    ])

    expanded, warnings = expand_site_rows(pd.DataFrame(rows))
    assert warnings == []
    scores, evidence = calculate_site_scores(expanded, source_rows=20)

    axes = (scores[["variable", "variable_order"]].drop_duplicates()
            .sort_values("variable_order"))
    assert axes["variable"].tolist() == [
        f"var_{index:02d}" for index in range(20)]
    aa_var0 = scores[
        (scores["site_id"] == "AA") & (scores["variable"] == "var_00")
    ].iloc[0]
    cc_var0 = scores[
        (scores["site_id"] == "CC") & (scores["variable"] == "var_00")
    ].iloc[0]
    assert aa_var0["severity_score"] == 95.0
    assert aa_var0["strongest_timepoint"] == "month6"
    assert cc_var0["severity_score"] == 88.0
    assert set(evidence["variable"]) <= {
        f"var_{index:02d}" for index in range(20)}


def test_source_row_window_is_applied_independently_per_network():
    frame = pd.DataFrame([
        _row("AA", "pro_first", 60, MEDIAN, network="PRONET"),
        _row("PA", "pre_first", 61, MEDIAN, network="PRESCIENT"),
        _row("AB", "pro_second", 62, MEDIAN, network="PRONET"),
        _row("PB", "pre_second", 63, MEDIAN, network="PRESCIENT"),
        _row("AC", "pro_later", 64, MEDIAN, network="PRONET"),
        _row("PC", "pre_later", 65, MEDIAN, network="PRESCIENT"),
    ])

    expanded, warnings = expand_site_rows(frame)
    assert warnings == []
    scores, _ = calculate_site_scores(expanded, source_rows=2)

    axes = (scores[["network", "variable", "variable_order"]]
            .drop_duplicates()
            .sort_values(["network", "variable_order"]))
    by_network = axes.groupby("network", sort=True)["variable"].apply(list)
    assert by_network["PRESCIENT"] == ["pre_first", "pre_second"]
    assert by_network["PRONET"] == ["pro_first", "pro_second"]


def test_malformed_collapsed_row_is_warned_and_skipped():
    frame = pd.DataFrame([
        _row("(2 sites)", "height", 95, MEDIAN,
             sites="AA, BB", site_severities="88"),
        _row("CC", "weight", 65, MEDIAN),
        _row("DD", "age", 55, "unknown future site method"),
        _row("EE", "age", 55, MEDIAN, network=pd.NA),
        _row("FF", "age", 55, ""),
    ])

    expanded, warnings = expand_site_rows(frame)
    assert expanded["site_id"].tolist() == ["CC"]
    assert any("length mismatch" in warning for warning in warnings)
    assert any("unknown site_network method" in warning for warning in warnings)
    assert any("blank network or variable" in warning for warning in warnings)
    assert any("blank method" in warning for warning in warnings)


def test_end_to_end_writes_png_pdf_and_audit_tables(tmp_path: Path):
    report = tmp_path / "anomaly_report.xlsx"
    frame = pd.DataFrame([
        _row("AA", "height", 80, MEDIAN),
        _row("AA", "height", 65, SPREAD),
        _row("BB", "weight", 70, MISSING),
        _row("BB", "height | weight", 75, CORRELATION),
    ])
    with pd.ExcelWriter(report) as writer:
        frame.to_excel(writer, sheet_name="Site_Network", index=False)

    output = tmp_path / "radars"
    result = generate_radar_charts(
        report,
        output_dir=output,
        output_format="both",
        max_sites_per_chart=2,
        dpi=72,
    )

    # Exactly one variable per chart: the three distinct selected variables
    # produce three separate pages, even though only two sites are present.
    assert result["chart_count"] == 3
    assert result["site_count"] == 2
    assert len(result["warnings"]) == 3
    assert all("recorded site spoke" in warning
               for warning in result["warnings"])
    pngs = list(output.glob("PRONET__*__radar_01__*.png"))
    assert len(pngs) == 3
    assert (output / "site_network_radar.pdf").is_file()
    assert (output / "site_deviation_scores.csv").is_file()
    assert (output / "site_deviation_evidence.csv").is_file()
    assert (output / "selected_variables.csv").is_file()
    assert (output / "chart_manifest.csv").is_file()
    readme = (output / "README.txt").read_text(encoding="utf-8")
    assert readme.startswith("Site-network radar chart output\n")
    assert "never plotted as zero" in readme
    assert "exactly one variable" in readme

    manifest = pd.read_csv(output / "chart_manifest.csv")
    assert set(manifest["site_id"]) == {"AA", "BB"}
    assert manifest["png"].nunique() == 3
    assert manifest.groupby("png")["variable"].nunique().eq(1).all()
    assert set(manifest["png"]) == {png.name for png in pngs}
    assert sorted(manifest["pdf_page"].unique()) == [1, 2, 3]
    # Every chart carries both sites, and each unflagged site is marked
    # unrecorded rather than dropped from the page.
    assert manifest.groupby("png")["site_id"].apply(set).eq({"AA", "BB"}).all()
    assert manifest.groupby("png")["recorded_in_input"].sum().eq(1).all()
    assert manifest.loc[
        (manifest["variable"] == "weight") & (manifest["site_id"] == "AA"),
        "recorded_in_input"].tolist() == [False]
    assert manifest.loc[
        ~manifest["recorded_in_input"], "raw_value"].isna().all()
    assert (manifest["spokes_total"] == 2).all()
    assert (manifest["recorded_sites_total"] == 1).all()
    selected = pd.read_csv(output / "selected_variables.csv")
    assert selected["variable"].tolist() == [
        "height", "weight", "height | weight"]


def test_every_site_gets_a_spoke_on_every_variable_page(tmp_path: Path):
    report = tmp_path / "site_network.csv"
    rows = [
        _row(site, "height", severity, MEDIAN)
        for site, severity in zip(
            ["AA", "BB", "CC", "DD", "EE"], [55, 60, 70, 80, 90])
    ]
    rows.extend([
        _row("AA", "weight", 65, MEDIAN),
        _row("CC", "weight", 75, MEDIAN),
        _row("ZZ", "height", "not numeric", MEDIAN),
    ])
    pd.DataFrame(rows).to_csv(report, index=False)

    output = tmp_path / "variable_pages"
    result = generate_radar_charts(
        report, output_dir=output, max_sites_per_chart=3, dpi=72)

    # Six sites are named in the report, so both variables span two pages of
    # three spokes. Every generated page still holds exactly one variable.
    assert result["chart_count"] == 4
    assert result["spoke_site_count"] == 6
    assert result["sites_by_network"] == {
        "PRONET": ["AA", "BB", "CC", "DD", "EE", "ZZ"]}
    manifest = pd.read_csv(output / "chart_manifest.csv")
    assert manifest.groupby("png")["variable"].nunique().eq(1).all()
    assert manifest.groupby("png").size().le(3).all()
    assert manifest.loc[manifest["variable"] == "height", "png"].nunique() == 2
    assert manifest.loc[manifest["variable"] == "weight", "png"].nunique() == 2
    # Both variables show all six sites; weight flagged only two of them.
    for variable in ("height", "weight"):
        assert set(manifest.loc[manifest["variable"] == variable,
                                "site_id"]) == {
            "AA", "BB", "CC", "DD", "EE", "ZZ"}
    assert set(manifest.loc[
        manifest["variable"] == "weight"].query("recorded_in_input")[
            "site_id"]) == {"AA", "CC"}
    # ZZ's only row has a nonnumeric severity: it is charted as a site with no
    # recorded evidence rather than vanishing from the report entirely.
    zz = manifest[manifest["site_id"] == "ZZ"]
    assert len(zz) == 2
    assert not zz["recorded_in_input"].any()
    assert zz["raw_value"].isna().all()
    assert zz["plot_radius"].isna().all()
    assert set(manifest.loc[manifest["variable"] == "height",
                            "recorded_sites_total"]) == {5}
    assert set(manifest["spokes_total"]) == {6}

    # The score table stays an evidence-only table: no synthesized rows.
    scores = pd.read_csv(output / "site_deviation_scores.csv")
    assert len(scores) == 7
    assert scores["recorded_in_input"].all()
    assert (scores["plot_radius"] == scores["raw_deviation"]).all()
    assert not ((scores["site_id"] == "BB")
                & (scores["variable"] == "weight")).any()
    assert "ZZ" not in set(scores["site_id"])
    assert any("nonnumeric severity" in warning for warning in result["warnings"])


def test_truncated_collapsed_observed_list_does_not_borrow_another_site_value():
    """A site named in `sites` but missing from `observed_value` has no data.

    The producing detector caps the observed_value list (currently 15 segments
    plus a "(+N more)" tail) while keeping every site in the parallel `sites`
    column. Falling back to the whole string would hand the truncated sites the
    first listed site's statistic, which then plots as their own evidence.
    """
    sites = [f"S{index:02d}" for index in range(18)]
    shown = "; ".join(f"{site}: site median {50 - index} (n=20)"
                      for index, site in enumerate(sites[:15]))
    frame = pd.DataFrame([
        _row("(18 sites)", "height", 90, MEDIAN,
             sites=", ".join(sites),
             site_severities=", ".join(["90"] * 18),
             observed_value=f"{shown}; (+3 more)",
             expected_value="cross-site median 10, sigma 1"),
    ])

    expanded, warnings = expand_site_rows(frame)
    assert sorted(expanded["site_id"]) == sites[:15]
    for dropped in ("S15", "S16", "S17"):
        assert dropped not in set(expanded["site_id"])
        assert any(f"site {dropped!r} is named in the collapsed sites list"
                   in warning for warning in warnings)
    # The listed sites keep their own values; none is duplicated onto another.
    assert expanded.set_index("site_id").loc["S00", "raw_value"] == 50.0
    assert expanded.set_index("site_id").loc["S14", "raw_value"] == 36.0
    assert expanded["raw_value"].nunique() == 15

    # A concrete single-site row has no per-site list, so its unprefixed
    # observed_value must still be parsed rather than treated as truncated.
    solo, solo_warnings = expand_site_rows(
        pd.DataFrame([_row("AA", "height", 70, MEDIAN)]))
    assert solo_warnings == []
    assert solo["raw_value"].tolist() == [17.0]


def test_unrecorded_sites_are_drawn_as_absence_not_as_a_value(tmp_path: Path):
    """Inspect the drawn artists, not just the data frame handed to them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from site_network_radar import _draw_radar

    rows = pd.DataFrame([
        {"site_id": "AA", "variable": "height", "raw_value": 14.0,
         "plot_radius": 4.0, "reference_value": 10.0, "raw_units": "u",
         "recorded_in_input": True},
        {"site_id": "BB", "variable": "height", "raw_value": float("nan"),
         "plot_radius": float("nan"), "reference_value": float("nan"),
         "raw_units": "", "recorded_in_input": False},
        {"site_id": "CC", "variable": "height", "raw_value": 12.0,
         "plot_radius": 2.0, "reference_value": 10.0, "raw_units": "u",
         "recorded_in_input": True},
    ])
    fig, ax = plt.subplots(subplot_kw={"polar": True})
    try:
        _draw_radar(ax, rows, "t", reference=10.0, same_reference=True,
                    upper_radius=5.0)
        labels = [text.get_text() for text in ax.get_xticklabels()]
        assert "value" in labels[0] and "delta" in labels[0]
        # No number may appear on the unrecorded spoke: a printed 0 would read
        # as a measured zero deviation.
        assert labels[1] == "BB\n(no recorded\nevidence)"
        assert "value" not in labels[1] and "0" not in labels[1]
        # One marker per recorded site, and none at the unrecorded angle.
        offsets = ax.collections[0].get_offsets()
        assert len(offsets) == 2
        assert all(np.isfinite(radius) for _, radius in offsets)
        # No shaded polygon while any spoke lacks evidence.
        assert not [patch for patch in ax.patches
                    if type(patch).__name__ == "Polygon"]
    finally:
        plt.close(fig)

    # With every spoke recorded, the profile is shaded.
    rows.loc[1, ["raw_value", "plot_radius", "reference_value"]] = [
        11.0, 1.0, 10.0]
    rows.loc[1, "recorded_in_input"] = True
    fig, ax = plt.subplots(subplot_kw={"polar": True})
    try:
        _draw_radar(ax, rows, "t", reference=10.0, same_reference=True,
                    upper_radius=5.0)
        assert [patch for patch in ax.patches
                if type(patch).__name__ == "Polygon"]
    finally:
        plt.close(fig)


def test_site_list_rejects_pseudo_sites_and_non_utf8(tmp_path: Path):
    report = tmp_path / "site_network.csv"
    pd.DataFrame([_row("AA", "height", 60, MEDIAN)]).to_csv(report, index=False)

    roster = tmp_path / "sites.txt"
    roster.write_text("BB\n(3 sites)\n(network-level)\n", encoding="utf-8")
    result = generate_radar_charts(
        report, output_dir=tmp_path / "out", site_list=roster, dpi=72)
    assert result["sites_by_network"] == {"PRONET": ["AA", "BB"]}
    assert sum("pseudo site label" in warning
               for warning in result["warnings"]) == 2

    binary = tmp_path / "latin1.txt"
    binary.write_bytes(b"# Z\xfcrich site\nBB\n")
    with pytest.raises(RadarInputError, match="not valid UTF-8"):
        generate_radar_charts(
            report, output_dir=tmp_path / "out2", site_list=binary, dpi=72)


def test_default_draws_all_sites_on_one_page_and_site_list_widens_roster(
        tmp_path: Path):
    report = tmp_path / "site_network.csv"
    rows = [
        _row(site, "height", severity, MEDIAN)
        for site, severity in zip(["AA", "BB", "CC"], [55, 60, 70])
    ]
    rows.append(_row("DD", "weight", 65, MEDIAN))
    pd.DataFrame(rows).to_csv(report, index=False)

    roster = tmp_path / "sites.txt"
    roster.write_text(
        "# study roster\n"
        "site_id\n"
        "EE\n"
        "PRONET:FF\n"
        "PRESCIENT:GG\n",
        encoding="utf-8")

    output = tmp_path / "all_sites"
    result = generate_radar_charts(
        report, output_dir=output, site_list=roster, dpi=72)

    # No pagination by default: one page per variable, every site on it.
    assert result["chart_count"] == 2
    assert result["sites_by_network"] == {
        "PRONET": ["AA", "BB", "CC", "DD", "EE", "FF"]}
    manifest = pd.read_csv(output / "chart_manifest.csv")
    assert (manifest["variable_page_count"] == 1).all()
    assert manifest.groupby("png")["site_id"].apply(set).eq(
        {"AA", "BB", "CC", "DD", "EE", "FF"}).all()
    # A site that exists only in the roster can never be recorded evidence.
    assert not manifest.loc[manifest["site_id"].isin({"EE", "FF"}),
                            "recorded_in_input"].any()
    assert manifest.loc[
        (manifest["variable"] == "weight") & manifest["recorded_in_input"],
        "site_id"].tolist() == ["DD"]
    assert any("header row" in warning for warning in result["warnings"])
    assert any("PRESCIENT" in warning and "not in the charted input" in warning
               for warning in result["warnings"])
    readme = (output / "README.txt").read_text(encoding="utf-8")
    assert "no limit (every site on one page)" in readme
    assert "PRONET: 6" in readme


def test_site_pagination_balances_away_single_spoke_remainder(tmp_path: Path):
    report = tmp_path / "nine_sites.csv"
    pd.DataFrame([
        _row(f"S{index:02d}", "height", 50 + index, MEDIAN)
        for index in range(9)
    ]).to_csv(report, index=False)

    output = tmp_path / "balanced_pages"
    result = generate_radar_charts(
        report, output_dir=output, max_sites_per_chart=8, dpi=72)

    assert result["chart_count"] == 2
    manifest = pd.read_csv(output / "chart_manifest.csv")
    assert sorted(manifest.groupby("png").size().tolist()) == [4, 5]
    assert not any("geometrically degenerate" in warning
                   for warning in result["warnings"])


def test_long_variable_names_have_bounded_unique_filenames(tmp_path: Path):
    report = tmp_path / "long_names.csv"
    common = "very_long_redcap_variable_" + ("x" * 280)
    pd.DataFrame([
        _row("AA", common + "_one", 60, MEDIAN),
        _row("BB", common + "_two", 70, MEDIAN),
    ]).to_csv(report, index=False)

    output = tmp_path / "long_name_output"
    result = generate_radar_charts(report, output_dir=output, dpi=72)

    assert result["chart_count"] == 2
    filenames = pd.read_csv(output / "chart_manifest.csv")["png"].tolist()
    assert len(set(filenames)) == 2
    assert all(len(name) < 140 for name in filenames)


def test_csv_preserves_na_site_and_filters_other_anomaly_types(tmp_path: Path):
    report = tmp_path / "site_network.csv"
    # Mixed-case headers also exercise conservative header normalization.
    pd.DataFrame([
        {
            "Network": "PRONET", "Timepoint": "baseline", "Site_ID": "NA",
            "Variable": "height", "Severity_Score": 70, "Method": MEDIAN,
            "Raw_Score": 7,
            "Observed_Value": "site median 17 (n=10)",
            "Expected_Value": "cross-site median 10, sigma 1",
            "Anomaly_Type": "site_network_outlier",
        },
        {
            "Network": "PRONET", "Timepoint": "baseline",
            "Site_ID": "(2 sites)", "Variable": "weight",
            "Severity_Score": 99, "Method": MEDIAN,
            "Sites": "NA, BB", "Site_Severities": "80, 60",
            "Raw_Score": 8,
            "Observed_Value": (
                "NA: site median 18 (n=10); "
                "BB: site median 16 (n=10)"),
            "Expected_Value": "cross-site median 10, sigma 1",
            "Anomaly_Type": "site_network_outlier",
        },
        {
            "Network": "PRONET", "Timepoint": "baseline", "Site_ID": "ZZ",
            "Variable": "height | weight", "Severity_Score": 100,
            "Raw_Score": 10,
            "Observed_Value": "site median 20 (n=10)",
            "Expected_Value": "cross-site median 10, sigma 1",
            "Method": "correlation residual median", "Anomaly_Type": "correlative",
        },
    ]).to_csv(report, index=False)

    frame = read_site_network_table(report)
    assert len(frame) == 2
    assert frame.attrs["filtered_non_site_rows"] == 1
    expanded, warnings = expand_site_rows(frame)
    assert warnings == []
    assert expanded[expanded["site_id"] == "NA"][
        "severity_score"].tolist() == [70.0, 80.0]
    assert expanded[expanded["site_id"] == "BB"][
        "severity_score"].tolist() == [60.0]

    result = generate_radar_charts(
        report, output_dir=tmp_path / "na_output", dpi=72)
    assert any("filtered 1 non-site_network_outlier" in warning
               for warning in result["warnings"])


def test_network_only_filenames_do_not_collide(tmp_path: Path):
    report = tmp_path / "collision.xlsx"
    frame = pd.DataFrame([
        _row("AA", "height", 60, MEDIAN, network="A/B"),
        _row("BB", "height", 65, MEDIAN, network="A B"),
        _row("CC", "height", 70, MEDIAN,
             network="PRONET", timepoint="month10"),
        _row("DD", "height", 75, MEDIAN,
             network="PRONET", timepoint="month2"),
    ])
    with pd.ExcelWriter(report) as writer:
        frame.to_excel(writer, sheet_name="site_network", index=False)

    output = tmp_path / "collision_output"
    result = generate_radar_charts(report, output_dir=output, dpi=72)
    assert result["chart_count"] == 3

    manifest = pd.read_csv(output / "chart_manifest.csv")
    assert manifest["png"].nunique() == 3
    assert len(list(output.glob("*.png"))) == 3
    assert "timepoint" not in manifest.columns
    assert manifest.loc[manifest["network"] == "PRONET", "png"].nunique() == 1

    # Reuse is refused so obsolete pages/PDFs cannot survive a later run.
    with pytest.raises(RadarInputError, match="not empty"):
        generate_radar_charts(report, output_dir=output, dpi=72)


def test_missing_required_columns_raise_clear_error(tmp_path: Path):
    report = tmp_path / "bad.csv"
    pd.DataFrame({"network": ["PRONET"]}).to_csv(report, index=False)

    with pytest.raises(RadarInputError, match="missing required column"):
        read_site_network_table(report)
