"""Demo: compact site radar charts for ``chrbprs_bprs_total``.

Reads the AMPSCZ combined CSVs directly rather than an anomaly report, so the
reference uses the complete eligible pool and every deviation is computed
before display selection. Each chart shows at most 15 measured sites, with up
to two reference-pool anchors and then the largest remaining absolute
deviations.
"Near-reference" describes this chart statistic, not clinical normality.

Usage
-----
    python demo_bprs_site_radar.py
    python demo_bprs_site_radar.py --input-dir D:/ampscz/combined
    python demo_bprs_site_radar.py --timepoint baseline --timepoint month12
    python demo_bprs_site_radar.py --all-timepoints --scope network

By default the input directory comes from ``config.json``
(``paths.combined_csv_path``), the timepoint is baseline, and both a
study-wide chart and one chart per network are produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from site_network_radar import (
    DEFAULT_REFERENCE_MIN_N,
    RadarInputError,
    discover_combined_files,
    generate_radar_charts_from_combined,
)

VARIABLE = "chrbprs_bprs_total"
DEFAULT_TIMEPOINT = "baseline"
DEFAULT_SITE_LIMIT = 15
DEFAULT_NORMAL_SITE_COUNT = 2
REPO_ROOT = Path(__file__).resolve().parent


def _configured_input_dir() -> Path | None:
    """Combined-CSV directory recorded in config.json, if it is readable."""
    config_path = REPO_ROOT / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    configured = (config.get("paths") or {}).get("combined_csv_path")
    return Path(configured) if configured else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Chart a compact, outlier-focused subset of sites for {VARIABLE}, "
            "measured against a full-population cross-site reference."))
    parser.add_argument(
        "--input-dir", default=None,
        help=("directory holding AMPSCZ-combined-redcap_*.csv "
              "(default: paths.combined_csv_path from config.json)"))
    parser.add_argument(
        "--output-dir", default=None,
        help="new or empty output directory (default: timestamped in the repo)")
    parser.add_argument(
        "--variable", action="append", dest="variables", default=None,
        help=f"override the charted variable (default: {VARIABLE})")
    parser.add_argument(
        "--timepoint", action="append", dest="timepoints", default=None,
        help=f"timepoint to chart; repeat for several (default: {DEFAULT_TIMEPOINT})")
    parser.add_argument(
        "--all-timepoints", action="store_true",
        help="chart every timepoint present in the input directory")
    parser.add_argument(
        "--network", action="append", dest="networks", default=None,
        help="restrict to a network; repeat for several (default: all found)")
    parser.add_argument(
        "--scope", choices=("network", "study", "both"), default="both",
        help=("compare within each network, against one study-wide reference, "
              "or both (default: both)"))
    parser.add_argument(
        "--statistic",
        choices=("median", "mean", "iqr", "missing"), default="median",
        help="per-site statistic to compare (default: median)")
    parser.add_argument(
        "--min-n", type=int, default=1,
        help="minimum observations before a site's value is plotted (default: 1)")
    parser.add_argument(
        "--reference-min-n", type=int, default=DEFAULT_REFERENCE_MIN_N,
        help=("minimum observations for a site to shape the cross-site "
              f"reference (default: {DEFAULT_REFERENCE_MIN_N})"))
    parser.add_argument(
        "--site-limit", type=int, default=DEFAULT_SITE_LIMIT,
        help=("maximum measured sites to display per chart; the most extreme "
              "deviations fill the non-anchor slots "
              f"(default: {DEFAULT_SITE_LIMIT})"))
    parser.add_argument(
        "--normal-sites", type=int, default=DEFAULT_NORMAL_SITE_COUNT,
        help=("maximum actual reference-pool sites nearest the reference "
              "before filling "
              "remaining slots with extreme absolute deviations "
              f"(default: {DEFAULT_NORMAL_SITE_COUNT})"))
    parser.add_argument(
        "--site-list", default=None,
        help="optional roster file adding sites absent from every CSV")
    parser.add_argument(
        "--format", choices=("png", "pdf", "both"), default="both",
        dest="output_format", help="chart output format (default: both)")
    parser.add_argument("--dpi", type=int, default=180,
                        help="PNG resolution (default: 180)")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)

    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        configured = _configured_input_dir()
        if configured is None:
            print("ERROR: no --input-dir given and config.json has no "
                  "paths.combined_csv_path", file=sys.stderr)
            return 2
        input_dir = configured
    if not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}",
              file=sys.stderr)
        return 2

    if args.all_timepoints:
        timepoints = None
    else:
        timepoints = args.timepoints or [DEFAULT_TIMEPOINT]

    # Surface what will actually be read before spending time on it: a wrong
    # --input-dir otherwise looks identical to a study with no BPRS data.
    try:
        found, discovery_warnings = discover_combined_files(
            input_dir, args.networks, timepoints)
    except RadarInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not found:
        print(f"ERROR: no combined CSVs matching the requested network(s)/"
              f"timepoint(s) in {input_dir}", file=sys.stderr)
        for warning in discovery_warnings[:10]:
            print(f"  note: {warning}", file=sys.stderr)
        return 2
    print(f"Input   : {input_dir}")
    print(f"Files   : {len(found)} "
          f"({', '.join(sorted({entry['network'] for entry in found}))}; "
          f"{', '.join(sorted({entry['timepoint'] for entry in found}))})")
    print(f"Variable: {', '.join(args.variables or [VARIABLE])}")
    print(f"Selection: up to {args.site_limit} sites; up to "
          f"{args.normal_sites} reference-pool anchor(s), then the most "
          "extreme deviations")

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / f"bprs_site_radar_{stamp}"

    try:
        result = generate_radar_charts_from_combined(
            input_dir,
            args.variables or [VARIABLE],
            output_dir=output_dir,
            networks=args.networks,
            timepoints=timepoints,
            statistic=args.statistic,
            scope=args.scope,
            min_n=args.min_n,
            reference_min_n=args.reference_min_n,
            site_list=args.site_list,
            site_limit=args.site_limit,
            normal_site_count=args.normal_sites,
            output_format=args.output_format,
            dpi=args.dpi,
        )
    except (RadarInputError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"Created {result['chart_count']} chart page(s) across "
          f"{result['variable_count']} chart group(s).")
    for group, sites in sorted(result["sites_by_network"].items()):
        print(f"  {group}: {len(sites)} site spoke(s)")
    print(f"Output  : {Path(result['output_dir']).resolve()}")
    for png in sorted(Path(result["output_dir"]).glob("*.png")):
        print(f"  chart : {png.name}")
    if result["warnings"]:
        print(f"Warnings: {len(result['warnings'])} (see README.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
