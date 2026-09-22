"""Redraw an existing site-radar output folder with numbered spoke labels.

The charts are rebuilt from that run's own ``chart_manifest.csv``, not from the
combined CSVs, so nothing but the spoke labels can change: the same site
statistics, the same reference, the same spoke order and the same page
filenames come back out. Spokes are labelled ``Site 1 .. Site N`` in the order
they are already drawn (alphabetical by site id), numbered per chart group.

The real site ids stay in every audit column, including a ``display_site_id``
crosswalk written into the redrawn manifest.

Usage
-----
    python relabel_site_radar.py bprs_site_radar_20260803_124012
    python relabel_site_radar.py <source> --output-dir <dir> --real-labels
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from site_network_radar import (
    COMBINED_STATISTICS,
    _POINT_COLOR_ABOVE,
    _POINT_COLOR_BELOW,
    _render_charts,
    _text,
)

_EXTRA_MANIFEST_COLUMNS = (
    "chart_group", "chart_scope_label", "scope", "timepoint", "statistic",
    "site_n", "n_observed", "n_subjects", "robust_z",
    "site_rank_by_deviation", "reference_pool_member", "selected_for_chart",
    "site_selection_role", "display_site_id",
)
# Files the redraw regenerates; everything else in the source folder is copied
# through untouched.
_REGENERATED = {"site_network_radar.pdf", "chart_manifest.csv"}


def _readme_number(text: str, pattern: str, default):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else default


def _run_settings(source: Path) -> dict:
    """Recover the run settings the footer text quotes, from README.txt."""
    try:
        text = (source / "README.txt").read_text(encoding="utf-8")
    except OSError:
        text = ""
    no_limit = bool(re.search(
        r"Maximum site spokes per chart:\s*no limit", text))
    return {
        "min_n": _readme_number(
            text, r"Minimum observations to plot a site:\s*(\d+)", 1),
        "reference_min_n": _readme_number(
            text, r"contribute to the reference:\s*(\d+)", 30),
        "site_limit": None if no_limit else _readme_number(
            text, r"Maximum site spokes per chart:\s*(\d+)", None),
        "normal_site_count": _readme_number(
            text, r"up to (\d+) actual reference-pool", 2),
        "readme_found": bool(text),
    }


def _figure_options(statistic: str, settings: dict) -> dict:
    """Rebuild the combined-source figure wording for this run."""
    _, _, stat_label = COMBINED_STATISTICS[statistic]
    options = {
        "source_mode": "combined",
        "radius_caption": f"Absolute distance from the cross-site {stat_label}",
        "radius_note": (f"radius = |site {stat_label} - cross-site "
                        f"{stat_label}|"),
        "recorded_phrase": "with a plotted statistic",
        "marker_legend": (
            (_POINT_COLOR_ABOVE, f"Site above the cross-site {stat_label}"),
            (_POINT_COLOR_BELOW, f"Site below the cross-site {stat_label}"),
        ),
        "absent_legend": ("Site with too few usable observations "
                          "(labelled, no marker)"),
        "footer_note": (
            f"Measured from the AMPSCZ combined CSVs: a site is plotted once "
            f"it has n >= {settings['min_n']} usable observation(s); the "
            f"cross-site reference is the median over sites with n >= "
            f"{settings['reference_min_n']}. An unmarked spoke means no "
            "usable data there, not a deviation of zero."),
    }
    if settings["site_limit"] is not None:
        options.update({
            "spoke_intro": "Spokes = selected sites in",
            "footer_note": (
                "Reference uses the complete eligible pool. Display: up to "
                f"{settings['site_limit']} measured sites: up to "
                f"{settings['normal_site_count']} nearest reference-pool "
                "site(s), then largest deviations. Near-reference is "
                "descriptive, not clinical."),
        })
    return options


def _load_manifest(source: Path) -> pd.DataFrame:
    manifest = pd.read_csv(source / "chart_manifest.csv")
    if "chart_group" not in manifest.columns:
        manifest["chart_group"] = manifest["network"]
    # A CSV round trip turns the flag into "True"/"False" text; be explicit so
    # an unrecorded spoke cannot come back truthy.
    manifest["recorded_in_input"] = manifest["recorded_in_input"].map(
        lambda value: value if isinstance(value, bool)
        else _text(value).casefold() in {"true", "1", "1.0"})
    if "deviation_direction" not in manifest.columns:
        # The manifest keeps only the absolute deviation, so the above/below
        # marker colour has to be recovered the way build_combined_scores
        # derived it: sign(site value - reference), NaN where nothing plots.
        value = pd.to_numeric(manifest["raw_value"], errors="coerce")
        reference = pd.to_numeric(manifest["reference_value"], errors="coerce")
        direction = np.sign(value - reference)
        manifest["deviation_direction"] = direction.where(
            manifest["recorded_in_input"] & value.notna())
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Redraw a site-radar output folder, labelling spokes "
                     "Site 1 .. Site N instead of the site abbreviations."))
    parser.add_argument("source", help="existing radar output directory")
    parser.add_argument(
        "--output-dir", default=None,
        help="destination (default: <source>_numbered)")
    parser.add_argument(
        "--real-labels", action="store_true",
        help=("redraw with the original site abbreviations; use this to "
              "verify the redraw reproduces the source charts"))
    parser.add_argument(
        "--format", choices=("png", "pdf", "both"), default="both",
        dest="output_format")
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    source = Path(args.source)
    if not (source / "chart_manifest.csv").is_file():
        print(f"ERROR: no chart_manifest.csv in {source}", file=sys.stderr)
        return 2
    output = Path(args.output_dir) if args.output_dir else source.with_name(
        source.name + "_numbered")
    output.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(source)
    settings = _run_settings(source)
    if not settings["readme_found"]:
        print("note: no README.txt found; footer text uses default min-n "
              "settings", file=sys.stderr)
    statistic = _text(manifest["statistic"].iloc[0]) or "median"
    site_universe = {
        _text(group): [_text(site) for site in rows["site_id"]]
        for group, rows in manifest.groupby("chart_group", sort=False)}

    options = _figure_options(statistic, settings)
    if not args.real_labels:
        options["numbered_site_labels"] = True

    warnings: list[str] = []
    chart_count, _ = _render_charts(
        manifest, site_universe, output, warnings,
        max_sites_per_chart=None,
        output_format=args.output_format, dpi=args.dpi,
        figure_options=options,
        extra_manifest_columns=_EXTRA_MANIFEST_COLUMNS)

    # Carry the untouched audit files across so the redrawn folder is a
    # complete replacement rather than charts with no provenance.
    for path in sorted(source.iterdir()):
        if (path.is_file() and path.name not in _REGENERATED
                and path.suffix.lower() != ".png"):
            shutil.copy2(path, output / path.name)

    if not args.real_labels:
        # The copied README describes charts labelled with site abbreviations,
        # which this folder's charts no longer are; leaving it unamended hands
        # someone a chart they cannot map back to a site.
        readme = output / "README.txt"
        if readme.is_file():
            with readme.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n\nSpoke labels in this folder\n"
                    "---------------------------\n"
                    "The charts here were redrawn from chart_manifest.csv with\n"
                    "the site abbreviations replaced by Site 1 .. Site N. The\n"
                    "numbering runs clockwise from the top of each chart, i.e.\n"
                    "alphabetically by site id, and restarts on every chart, so\n"
                    "Site 1 is a different site on the STUDY chart than on a\n"
                    "network chart. The display_site_id column of\n"
                    "chart_manifest.csv is the crosswalk back to the real site\n"
                    "ids. Every other statistic is unchanged: the redraw\n"
                    "reproduces the original charts pixel for pixel apart from\n"
                    "the labels.\n")

    print(f"Source  : {source.resolve()}")
    print(f"Settings: statistic={statistic}; min_n={settings['min_n']}; "
          f"reference_min_n={settings['reference_min_n']}; "
          f"site_limit={settings['site_limit']}")
    print("Labels  : "
          f"{'site abbreviations' if args.real_labels else 'Site 1 .. Site N'}")
    print(f"Redrew  : {chart_count} chart page(s)")
    for png in sorted(output.glob("*.png")):
        print(f"  chart : {png.name}")
    print(f"Output  : {output.resolve()}")
    for warning in warnings:
        print(f"  warn  : {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
