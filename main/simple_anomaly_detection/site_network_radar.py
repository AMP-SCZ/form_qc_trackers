"""Create variable-specific site-deviation radar charts.

Two sources are supported, and they make different claims.

``--from-combined`` (see ``generate_radar_charts_from_combined``) measures every
site directly from the AMPSCZ combined REDCap CSVs. It is the mode to use when
the question is "how does each site compare", because the site roster is every
site the study data contains -- not only the sites some detector flagged -- and
each spoke carries a statistic computed from that site's own rows. Radius is the
absolute distance from the cross-site reference (median of the per-site
statistic), so an empty spoke means "this site has no usable observations for
this variable at this timepoint", which is a statement about data availability.

The default anomaly-report source is described below and is unchanged.

The ``site_network`` tab is a thresholded, reviewer-selected anomaly table,
not a complete matrix of every measured site/variable combination. This tool
therefore compares the raw site statistic recorded for each variable; it never
treats an absent or nonnumeric site-variable value as zero. For each variable, a
site's chart radius is the raw absolute distance from that detector's raw
reference value (cross-site median/reference where applicable). The normalized
severity score is retained only for selecting the strongest duplicate evidence.

Usage
-----
    python site_network_radar.py anomaly_report.xlsx
    python site_network_radar.py anomaly_report.xlsx --network PRONET
    python site_network_radar.py anomaly_report.xlsx --format both --first-rows 20
    python site_network_radar.py anomaly_report.xlsx --site-list sites.txt

    python site_network_radar.py --from-combined combined_csvs/ \
        --variable chrbprs_bprs_total --timepoint baseline --scope both

Every chart represents exactly one variable, and every chart carries one spoke
per site in the network -- not only the sites this variable happened to flag.
The spoke set is the full roster of sites the report names anywhere (optionally
widened by ``--site-list``), collected before any severity/value parsing so a
site whose only rows are malformed still appears. Radius is raw distance from
the detector's reference, so the center means the cross-site median/reference.
A site with no recorded evidence for a variable is labelled on its spoke but
carries no marker: that asserts "not flagged in this thresholded report", which
is not the same claim as "zero deviation". Variables are selected in source
order from the first 20 usable rows per network by default; later rows may
still contribute site evidence for an already-selected variable.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEFAULT_SHEET = "site_network"
DEFAULT_SOURCE_ROWS = 20
REQUIRED_COLUMNS = frozenset({
    "network", "site_id", "variable", "severity_score", "raw_score",
    "observed_value", "expected_value", "method",
})

_PSEUDO_SITE_RE = re.compile(r"^\s*\(")
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_NUM_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?")
_POINT_COLOR = "#D55E00"
# Radius is an absolute distance, so a site far below the reference and one far
# above land on the same ring. On a measured chart that is a real loss of
# information, so the marker carries the sign. Both are Okabe-Ito colours and
# stay distinguishable under the common colour-vision deficiencies.
_POINT_COLOR_ABOVE = "#D55E00"
_POINT_COLOR_BELOW = "#0072B2"
_RECORDED_LABEL_COLOR = "#222222"
_UNRECORDED_LABEL_COLOR = "#8A8A8A"
# Charts now show every site by default, so this is a legibility warning
# threshold rather than a hard spoke cap.
_SPOKE_LEGIBILITY_WARN = 40

# --- combined-CSV source -------------------------------------------------
# Column names the AMPSCZ combined CSVs use for the participant identifier,
# in the same precedence order the detector runner applies.
COMBINED_ID_COLUMNS: tuple[str, ...] = (
    "subjectid", "subject_id", "src_subject_id", "study_id")
_BLANK_SUBJECT_IDS = frozenset({"", "nan", "none", "na", "n/a"})
# Longest-first so 'month_12' is matched before 'month_1'.
_COMBINED_TIMEPOINT_TOKENS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        [("screening", "screening"), ("baseline", "baseline"),
         ("conversion", "conversion"), ("floating_forms", "floating"),
         ("floating", "floating")]
        + [(f"month_{n}", f"month{n}") for n in
           (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24)]
        + [(f"month{n}", f"month{n}") for n in
           (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24)],
        key=lambda pair: len(pair[0]), reverse=True))
_COMBINED_NETWORK_TOKENS: tuple[str, ...] = ("PRONET", "PRESCIENT")
STUDY_SCOPE_LABEL = "STUDY"
# Statistic -> (per-site column, radial unit label, noun used in captions).
# The noun is bare ("median", not "site median") because the captions compose
# it as both "site {noun}" and "cross-site {noun}".
COMBINED_STATISTICS: dict[str, tuple[str, str, str]] = {
    "median": ("site_median", "raw variable units", "median"),
    "mean": ("site_mean", "raw variable units", "mean"),
    "iqr": ("site_iqr", "IQR units", "IQR"),
    "missing": ("site_missing_pct", "percentage points", "missing rate"),
}
# A site needs this many non-missing observations before its statistic is
# plotted at all; below it the spoke is labelled with its n and left unmarked.
DEFAULT_MIN_SITE_N = 1
# Sites contributing to the cross-site reference. Matches the site_network
# detector's MIN_N_PER_SITE so the reference means the same thing in both.
DEFAULT_REFERENCE_MIN_N = 30
# Fewer contributing sites than this and a cross-site median is not a
# meaningful reference; the code falls back and says so in the warnings.
_MIN_REFERENCE_SITES = 3


class RadarInputError(ValueError):
    """Raised when the supplied anomaly report cannot support the chart."""


def _text(value) -> str:
    if value is None:
        return ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return ""
    return str(value).strip()


def _split_csv(value) -> list[str]:
    return [part.strip() for part in _text(value).split(",") if part.strip()]


def _safe_name(value: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("-", _text(value)).strip("-._")
    return cleaned or "unspecified"


def _chart_filename(network: str, variable: str, page_number: int) -> str:
    """Human-readable filename with a stable collision-resistant suffix."""
    identity = (
        f"{_text(network)}\0{_text(variable)}\0{page_number}"
    ).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:10]
    # Keep generated paths comfortably below Windows' traditional path limit.
    # The digest, not the readable slug, supplies collision resistance.
    network_slug = _safe_name(network)[:40]
    variable_slug = _safe_name(variable)[:80]
    return (
        f"{network_slug}__{variable_slug}"
        f"__radar_{page_number:02d}__{digest}.png")


def _timepoint_label(value) -> str:
    text = _text(value)
    return text if text else "unspecified timepoint"


def _number_from_match(match: re.Match | None) -> float | None:
    if match is None:
        return None
    text = match.group(1)
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1]
    try:
        value = float(text)
    except ValueError:
        return None
    if not np.isfinite(value):
        return None
    return value


def _extract_first_number(pattern: str, text: str,
                          *, flags: int = re.IGNORECASE) -> float | None:
    return _number_from_match(re.search(pattern, text, flags))


def _observed_segment_for_site(observed_value, site_id: str, *,
                               collapsed: bool = False) -> str:
    """Return this site's phrase from a collapsed observed_value list.

    ``collapsed`` marks a multi-site row, whose ``observed_value`` is a
    per-site list. The producing detector truncates that list (currently to 15
    segments plus a "(+N more)" tail) while keeping every site in the parallel
    ``sites`` column, so a site can be named without its statistic being
    present. Falling back to the whole string there would hand that site the
    *first* listed site's numbers, which then plot as its own evidence. Return
    nothing instead: the caller drops the row and the site is charted as having
    no recorded evidence, which is the truth.
    """
    observed = _text(observed_value)
    site = _text(site_id)
    if not site:
        return "" if collapsed else observed
    if f"{site}:" not in observed:
        return "" if collapsed else observed
    for segment in observed.split(";"):
        label, sep, rest = segment.strip().partition(":")
        if sep and label.strip() == site:
            return rest.strip()
    return "" if collapsed else observed


def _parse_raw_values(record: pd.Series, family: str, site_id: str, *,
                      collapsed: bool = False
                      ) -> tuple[float, float, float, str] | None:
    """Extract raw observed/reference values from site_network text fields."""
    observed = _observed_segment_for_site(record.get("observed_value", ""),
                                          site_id, collapsed=collapsed)
    expected = _text(record.get("expected_value", ""))

    if family == "median":
        raw_value = _extract_first_number(r"site median\s+(" + _NUM_RE.pattern + r")", observed)
        reference = _extract_first_number(r"cross-site median\s+(" + _NUM_RE.pattern + r")", expected)
        units = "raw variable units"
    elif family == "spread":
        raw_value = _extract_first_number(r"site IQR\s+(" + _NUM_RE.pattern + r")", observed)
        reference = _extract_first_number(r"cross-site median IQR\s+(" + _NUM_RE.pattern + r")", expected)
        units = "IQR units"
    elif family == "missingness":
        raw_value = _extract_first_number(r"site missing rate\s+(" + _NUM_RE.pattern + r")", observed)
        reference = _extract_first_number(r"cross-site missing-rate median\s+(" + _NUM_RE.pattern + r")", expected)
        units = "percentage points"
    elif family == "correlation":
        raw_value = _extract_first_number(r"site rho\s*=\s*(" + _NUM_RE.pattern + r")", observed)
        reference = _extract_first_number(r"(?:rest-of-network|network)\s+rho\s*=\s*(" + _NUM_RE.pattern + r")", expected)
        units = "rho"
    else:
        return None

    if raw_value is None or reference is None:
        return None
    return raw_value, reference, abs(raw_value - reference), units


def _method_family(method: str) -> str | None:
    """Map current and legacy detector method labels to four stable axes."""
    name = _text(method).lower()
    if "cross-network" in name:
        # These rows are network-level and cannot be attributed to a site.
        return None
    if "missing" in name:
        return "missingness"
    if "iqr" in name or "spread" in name:
        return "spread"
    if "fisher" in name or "correlation" in name or "corr" in name:
        return "correlation"
    if "median" in name:
        return "median"
    return None


def _issue_key(variable: str, family: str) -> str:
    """Stable distinct-issue identity used before the top-K aggregation."""
    variable = _text(variable)
    if family != "correlation":
        return variable
    parts = [part.strip() for part in variable.split("|") if part.strip()]
    if len(parts) == 2 and "corr-drift" in parts[1].lower():
        return parts[0]
    if len(parts) == 2:
        return " | ".join(sorted(parts))
    return variable


def read_site_network_table(path: str | os.PathLike,
                            sheet: str = DEFAULT_SHEET) -> pd.DataFrame:
    """Read the site-network table from XLSX/XLSM or a direct CSV."""
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise RadarInputError(f"Input file does not exist: {source}")

    suffix = source.suffix.lower()
    try:
        if suffix == ".csv":
            # Site code "NA" is legitimate; pandas' default NA vocabulary
            # would otherwise silently convert that site to a missing value.
            frame = pd.read_csv(source, low_memory=False, keep_default_na=False)
        elif suffix in {".xlsx", ".xlsm"}:
            with pd.ExcelFile(source) as excel:
                actual = next(
                    (name for name in excel.sheet_names
                     if name.strip().lower() == sheet.strip().lower()), None)
                if actual is None:
                    raise RadarInputError(
                        f"Workbook has no {sheet!r} tab. Available tabs: "
                        + ", ".join(excel.sheet_names))
                frame = pd.read_excel(
                    excel, sheet_name=actual, keep_default_na=False)
        else:
            raise RadarInputError(
                f"Unsupported input type {suffix!r}; use .xlsx, .xlsm, or .csv")
    except RadarInputError:
        raise
    except Exception as exc:
        raise RadarInputError(f"Could not read {source}: {exc}") from exc

    normalized_columns = [_text(column).lower() for column in frame.columns]
    duplicates = sorted({column for column in normalized_columns
                         if normalized_columns.count(column) > 1})
    if duplicates:
        raise RadarInputError(
            "input has duplicate column name(s) after normalization: "
            + ", ".join(duplicates))
    frame.columns = normalized_columns

    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise RadarInputError(
            "site_network input is missing required column(s): "
            + ", ".join(missing))
    if "anomaly_type" in frame.columns:
        is_site = (frame["anomaly_type"].astype(str).str.strip().str.lower()
                   == "site_network_outlier")
        filtered_count = int((~is_site).sum())
        frame = frame.loc[is_site].copy()
        frame.attrs["filtered_non_site_rows"] = filtered_count
    if frame.empty:
        raise RadarInputError("The site_network input is empty; there is nothing to chart.")
    return frame


def expand_site_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Expand collapsed multi-site findings to one raw-statistic row per site.

    Current reports store parallel ``sites`` and ``site_severities`` lists. The
    charted values are parsed from ``observed_value``/``expected_value`` so
    collapsed rows can still preserve each listed site's raw observed statistic.
    """
    rows: list[dict] = []
    warnings: list[str] = []
    unknown_methods: set[str] = set()

    # Use a display row counter instead of the DataFrame index: callers of the
    # public API may provide string or otherwise non-numeric indexes.
    for source_row, (_, record) in enumerate(frame.iterrows(), start=2):
        network = _text(record.get("network"))
        variable = _text(record.get("variable"))
        method = _text(record.get("method"))
        if not method:
            warnings.append(f"row {source_row}: blank method; skipped")
            continue
        family = _method_family(method)
        if family is None:
            if method and "cross-network" not in method.lower():
                unknown_methods.add(method)
            continue
        if not network or not variable:
            warnings.append(
                f"row {source_row}: blank network or variable; skipped")
            continue

        site_id = _text(record.get("site_id"))
        concrete = bool(site_id) and not _PSEUDO_SITE_RE.match(site_id)
        if concrete:
            site_pairs = [(site_id, record.get("severity_score"))]
        else:
            sites = _split_csv(record.get("sites", ""))
            severities = _split_csv(record.get("site_severities", ""))
            if not sites:
                # Includes legacy '(N sites)' rows that did not retain the
                # concrete site list and '(network-level)' rows.
                if site_id and site_id != "(network-level)":
                    warnings.append(
                        f"row {source_row}: pseudo site {site_id!r} has no "
                        "parallel sites/site_severities columns; skipped")
                continue
            if len(sites) != len(severities):
                warnings.append(
                    f"row {source_row}: sites/site_severities length mismatch "
                    f"({len(sites)} vs {len(severities)}); skipped")
                continue
            site_pairs = list(zip(sites, severities))

        for site, severity_raw in site_pairs:
            site = _text(site)
            try:
                severity = float(severity_raw)
            except (TypeError, ValueError):
                warnings.append(
                    f"row {source_row}: nonnumeric severity for site "
                    f"{site!r}; skipped")
                continue
            if not np.isfinite(severity):
                warnings.append(
                    f"row {source_row}: non-finite severity for site "
                    f"{site!r}; skipped")
                continue
            if severity < 0 or severity > 100:
                warnings.append(
                    f"row {source_row}: severity {severity:g} outside "
                    "0..100; clipped")
                severity = float(np.clip(severity, 0.0, 100.0))

            raw_values = _parse_raw_values(record, family, site,
                                           collapsed=not concrete)
            if raw_values is None:
                if (not concrete
                        and f"{site}:" not in _text(
                            record.get("observed_value", ""))):
                    warnings.append(
                        f"row {source_row}: site {site!r} is named in the "
                        "collapsed sites list but the source report truncated "
                        "its observed_value segment; charted as no recorded "
                        "evidence rather than borrowing another site's value")
                else:
                    warnings.append(
                        f"row {source_row}: could not parse raw "
                        f"observed/reference values for site {site!r}; skipped")
                continue
            raw_value, reference_value, raw_deviation, raw_units = raw_values

            rows.append({
                "network": network,
                "timepoint": _timepoint_label(record.get("timepoint", "")),
                "site_id": site,
                "family": family,
                "variable": variable,
                "issue_key": _issue_key(variable, family),
                "severity_score": severity,
                "raw_score": record.get("raw_score"),
                "raw_value": raw_value,
                "reference_value": reference_value,
                "raw_deviation": raw_deviation,
                "raw_units": raw_units,
                "observed_value": _observed_segment_for_site(
                    record.get("observed_value", ""), site),
                "expected_value": _text(record.get("expected_value", "")),
                "method": method,
                "source_row": source_row,
            })

    if unknown_methods:
        warnings.append(
            "unknown site_network method(s) skipped: "
            + "; ".join(sorted(unknown_methods)))
    if not rows:
        raise RadarInputError(
            "No concrete site-level rows with usable raw values were found. "
            "Rows must include parseable observed_value and expected_value fields.")
    return pd.DataFrame(rows), warnings


def collect_site_universe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return every concrete site the report names, before any value parsing.

    The spoke set of a chart must not depend on whether a particular row's
    severity or raw statistic could be parsed: a site whose only rows are
    malformed is exactly the site a reviewer would notice missing.  This pass
    therefore reads site identities straight off the input, including rows that
    ``expand_site_rows`` later drops (unknown method, cross-network scope,
    nonnumeric severity, unparseable observed/expected text).

    Pseudo labels such as ``(3 sites)`` and ``(network-level)`` are not sites;
    the parallel ``sites`` list on those rows supplies the real identities.
    Returns one row per distinct ``network``/``timepoint``/``site_id`` so the
    caller can apply the same network and timepoint filters as the chart data.
    """
    rows: list[dict] = []
    for _, record in frame.iterrows():
        network = _text(record.get("network"))
        if not network:
            # An unattributable site cannot be placed on any network's chart.
            continue
        timepoint = _timepoint_label(record.get("timepoint", ""))
        site_id = _text(record.get("site_id"))
        candidates = [site_id] if site_id else []
        candidates.extend(_split_csv(record.get("sites", "")))
        for candidate in candidates:
            site = _text(candidate)
            if not site or _PSEUDO_SITE_RE.match(site):
                continue
            rows.append({"network": network, "timepoint": timepoint,
                         "site_id": site})
    universe = pd.DataFrame(
        rows, columns=["network", "timepoint", "site_id"])
    return universe.drop_duplicates().reset_index(drop=True)


def read_site_list(path: str | os.PathLike) -> tuple[list[tuple[str, str]],
                                                     list[str]]:
    """Read an explicit site roster used to widen the chart spoke set.

    One site per line.  ``SITE`` applies to every charted network; ``NETWORK:
    SITE`` applies to that network only.  Blank lines and ``#`` comments are
    ignored.  A roster is the only way to show a site the report never names,
    so parse failures are reported rather than silently dropped.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise RadarInputError(f"Could not read site list {source}: {exc}") from exc
    except UnicodeDecodeError as exc:
        # A single non-UTF-8 byte anywhere in the file (a comment with an
        # umlaut, a smart quote) would otherwise surface as a traceback.
        raise RadarInputError(
            f"Site list {source} is not valid UTF-8 ({exc}); save it as UTF-8"
        ) from exc

    entries: list[tuple[str, str]] = []
    warnings: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        network, separator, site = line.partition(":")
        if not separator:
            network, site = "", line
        network = _text(network)
        site = _text(site)
        if not site:
            warnings.append(
                f"site list line {line_number}: no site name; skipped")
            continue
        if site.casefold() in {"site", "site_id", "siteid", "network"}:
            # Tolerate a CSV-style header row without inventing a fake site.
            warnings.append(
                f"site list line {line_number}: treated {site!r} as a header "
                "row; skipped")
            continue
        if _PSEUDO_SITE_RE.match(site):
            # '(3 sites)' / '(network-level)' pasted straight out of a report's
            # site_id column are not sites; a spoke for one would inflate the
            # roster counts that downstream consumers read.
            warnings.append(
                f"site list line {line_number}: {site!r} is a report pseudo "
                "site label, not a site; skipped")
            continue
        entries.append((network, site))
    if not entries:
        raise RadarInputError(
            f"Site list {source} contains no usable site names.")
    return entries, warnings


def _network_site_universe(
    universe: pd.DataFrame,
    extra_sites: Sequence[tuple[str, str]] = (),
) -> dict[str, list[str]]:
    """Map each network to its full, stably ordered spoke roster."""
    by_network: dict[str, set[str]] = {}
    for record in universe.itertuples(index=False):
        by_network.setdefault(_text(record.network), set()).add(
            _text(record.site_id))
    for network, site in extra_sites:
        if network:
            targets = [name for name in by_network
                       if name.casefold() == network.casefold()]
        else:
            # An unqualified roster entry is a site every charted network
            # should account for.
            targets = list(by_network)
        for target in targets:
            by_network[target].add(site)
    return {
        network: sorted(sites, key=lambda value: (value.casefold(), value))
        for network, sites in by_network.items()
    }


def _select_variable_axes(expanded: pd.DataFrame,
                          source_rows: int) -> pd.DataFrame:
    """Select chart variables from the first usable source rows per network.

    One collapsed input row can expand to many site rows, so the limit is
    applied to distinct ``source_row`` values before variables are de-duplicated.
    Later report rows for a selected variable remain eligible as site evidence.
    """
    if source_rows < 1:
        raise RadarInputError("source_rows must be at least 1")

    axes: list[dict] = []
    for network, network_rows in expanded.groupby("network", sort=False):
        first_rows = (
            network_rows.sort_values(["source_row", "variable"])
            .drop_duplicates("source_row", keep="first")
            .head(source_rows)
        )
        variables = first_rows.drop_duplicates("variable", keep="first")
        for variable_order, row in enumerate(variables.itertuples(index=False)):
            axes.append({
                "network": network,
                "variable": row.variable,
                "variable_order": variable_order,
                "axis_source_row": int(row.source_row),
            })
    if not axes:
        raise RadarInputError("No variable axes could be selected.")
    return pd.DataFrame(axes)


def calculate_site_scores(
    expanded: pd.DataFrame,
    source_rows: int = DEFAULT_SOURCE_ROWS,
    *,
    top_k: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return recorded raw site deviations and strongest supporting evidence.

    Variables are selected from the first ``source_rows`` usable source rows
    in each network.  For each selected site-variable pair, all included
    timepoints and methods are considered and the maximum calibrated severity
    row is retained as the strongest evidence. The charted value from that row
    is the raw observed statistic's absolute distance from its raw reference.
    Missing site-variable pairs are not synthesized or imputed.
    ``top_k`` is accepted only as a backward-compatible alias for the former API
    and now controls the source-row limit.
    """
    if top_k is not None:
        source_rows = top_k
    axes = _select_variable_axes(expanded, source_rows)

    selected = expanded.merge(
        axes, on=["network", "variable"], how="inner", validate="many_to_one")
    keys = ["network", "site_id", "variable"]
    ordered = selected.sort_values(
        keys + ["severity_score", "source_row"],
        ascending=[True, True, True, False, True])
    evidence = ordered.drop_duplicates(keys, keep="first").copy()

    if evidence.empty:
        raise RadarInputError(
            "No sites have evidence for the selected variable axes.")

    site_summary = (
        evidence.groupby(["network", "site_id"], sort=False,
                         as_index=False)
        .agg(
            recorded_variables=("raw_deviation", "count"),
            mean_recorded_raw_deviation=("raw_deviation", "mean"),
            max_recorded_raw_deviation=("raw_deviation", "max"),
            max_recorded_severity=("severity_score", "max"),
        )
    )

    scores = evidence.merge(
        site_summary, on=["network", "site_id"], how="left",
        validate="many_to_one")
    scores["plot_radius"] = scores["raw_deviation"]
    scores["reference_radius"] = 0.0
    scores["reference_basis"] = "raw detector reference/cross-site median"
    scores["recorded_site_count"] = (
        scores.groupby(["network", "variable"], sort=False)["site_id"]
        .transform("size").astype(int))
    scores["site_rank_by_deviation"] = (
        scores.groupby(["network", "variable"], sort=False)[
            "raw_deviation"]
        .rank(method="min", ascending=False).astype(int))
    scores["recorded_in_input"] = True
    scores = scores.rename(columns={
        "timepoint": "strongest_timepoint",
        "method": "strongest_method",
        "source_row": "strongest_source_row",
    })

    numeric_to_round = [
        "severity_score", "raw_score", "raw_value", "reference_value",
        "raw_deviation", "plot_radius", "mean_recorded_raw_deviation",
        "max_recorded_raw_deviation", "max_recorded_severity",
    ]
    for column in numeric_to_round:
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    scores[numeric_to_round] = scores[numeric_to_round].round(2)
    scores = scores.sort_values(
        ["network", "variable_order", "site_id"],
        kind="stable").reset_index(drop=True)

    score_columns = [
        "network", "site_id", "variable", "variable_order",
        "axis_source_row", "plot_radius", "raw_deviation", "raw_value",
        "reference_value", "raw_units", "raw_score", "severity_score",
        "reference_radius", "reference_basis",
        "recorded_in_input", "recorded_site_count",
        "site_rank_by_deviation", "strongest_timepoint",
        "strongest_method", "strongest_source_row", "recorded_variables",
        "mean_recorded_raw_deviation", "max_recorded_raw_deviation",
        "max_recorded_severity", "observed_value", "expected_value",
    ]
    return scores[score_columns], evidence


def _radar_angles(site_count: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, site_count, endpoint=False)
    return np.concatenate([angles, angles[:1]])


def _balanced_page_sizes(item_count: int, page_capacity: int) -> list[int]:
    """Split items into the fewest pages while avoiding tiny remainders."""
    page_count = int(math.ceil(item_count / page_capacity))
    base, remainder = divmod(item_count, page_count)
    return [base + (index < remainder) for index in range(page_count)]


def _variable_spoke_frame(group: pd.DataFrame,
                          network_sites: Sequence[str]) -> pd.DataFrame:
    """Give one variable a spoke for every site, evidence or not.

    Recorded rows keep their scored values verbatim.  Sites with no evidence
    for this variable are appended with NaN measurements and
    ``recorded_in_input=False``; nothing is imputed, so the drawing layer can
    render them as an absence rather than as a value.
    """
    roster = list(dict.fromkeys(
        [_text(site) for site in network_sites]
        + [_text(site) for site in group["site_id"]]))
    spokes = pd.DataFrame({"site_id": roster}).merge(
        group.assign(site_id=group["site_id"].map(_text)),
        on="site_id", how="left", validate="one_to_one")
    # eq(True) rather than fillna(False): the left join leaves NaN, and
    # bool(NaN) is True.
    spokes["recorded_in_input"] = spokes["recorded_in_input"].eq(True)
    # Constant per-chart identity columns are lost by the left join for
    # unrecorded sites; restore them from the recorded evidence.
    for column, value in (("network", group["network"].iloc[0]),
                          ("variable", group["variable"].iloc[0]),
                          ("variable_order", group["variable_order"].iloc[0]),
                          ("raw_units", _text(group["raw_units"].iloc[0]))):
        if column in spokes.columns:
            spokes[column] = spokes[column].where(
                spokes["recorded_in_input"], value)
    return spokes.sort_values(
        by="site_id", key=lambda values: values.map(
            lambda value: _text(value).casefold()),
        kind="stable").reset_index(drop=True)


def _site_axis_label(site_id: str, raw_value: float, raw_deviation: float,
                     recorded: bool, *,
                     absent_text: str = "(no recorded\nevidence)",
                     detail: str = "") -> str:
    site = textwrap.fill(
        _text(site_id), width=14, break_long_words=False,
        break_on_hyphens=False)
    if not recorded or not np.isfinite(raw_value) or not np.isfinite(raw_deviation):
        # No number may be shown here: this site has no plotted statistic, and
        # a printed 0 would read as a measured zero deviation. ``absent_text``
        # states which kind of absence this is -- unflagged in a thresholded
        # report, or unmeasurable in the source data.
        return f"{site}\n{absent_text}"
    label = f"{site}\nvalue {raw_value:.2f}\ndelta {raw_deviation:.2f}"
    return f"{label}\n{detail}" if detail else label


def _spoke_label_size(spoke_count: int) -> float:
    """Ramp the axis-label size so a full site roster stays readable."""
    for threshold, size in ((12, 8.0), (20, 7.0), (30, 6.3), (44, 5.6),
                            (60, 5.0)):
        if spoke_count <= threshold:
            return size
    return 4.4


def _figure_size(spoke_count: int) -> tuple[float, float]:
    """Grow the canvas with the spoke count so labels do not collide."""
    for threshold, size in ((12, (10.0, 8.8)), (20, (12.0, 10.4)),
                            (32, (13.5, 11.6)), (48, (15.0, 13.0))):
        if spoke_count <= threshold:
            return size
    return (16.5, 14.2)


def _draw_radar(ax, variable_rows: pd.DataFrame, title: str, *,
                reference: float | None, same_reference: bool,
                upper_radius: float, source_mode: str = "report",
                radius_caption: str = "Raw distance from reference") -> None:
    """Draw one variable as a site-spoke raw-distance profile.

    ``variable_rows`` carries one row per site in the network, including sites
    with no recorded evidence for this variable (``plot_radius`` NaN).  Those
    spokes get a label and nothing else: any marker would have to sit at some
    radius, and radius 0 is the detector reference, i.e. a positive claim of no
    deviation that this thresholded report cannot support.  ``upper_radius``
    and ``reference`` are supplied per variable rather than per page so every
    page of a paginated variable shares one radial scale.
    """
    if variable_rows.empty:
        raise RadarInputError("Cannot draw a radar without site rows.")
    variables = variable_rows["variable"].drop_duplicates().tolist()
    if len(variables) != 1:
        raise RadarInputError(
            "Each radar figure must contain exactly one variable; received "
            f"{len(variables)}")

    site_ids = variable_rows["site_id"].map(_text).tolist()
    raw_values = pd.to_numeric(
        variable_rows["raw_value"], errors="coerce").to_numpy(dtype=float)
    radii = pd.to_numeric(
        variable_rows["plot_radius"], errors="coerce").to_numpy(dtype=float)
    recorded = (variable_rows["recorded_in_input"]
                .fillna(False).astype(bool).to_numpy())
    recorded = recorded & np.isfinite(radii)
    # Unrecorded spokes carry no units; take the first recorded one.
    unit_values = [value for value in variable_rows["raw_units"].map(_text)
                   if value]
    units = unit_values[0] if unit_values else ""
    angles = _radar_angles(len(site_ids))
    point_angles = angles[:-1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(point_angles)
    label_size = _spoke_label_size(len(site_ids))
    # ``site_n`` only exists on combined-CSV charts, where every roster site
    # has a known observation count even when it is too small to plot.
    site_ns = (
        pd.to_numeric(variable_rows["site_n"], errors="coerce")
        .to_numpy(dtype=float)
        if "site_n" in variable_rows.columns
        else np.full(len(site_ids), np.nan))
    spoke_labels = []
    for site, raw_value, raw_deviation, is_recorded, site_n in zip(
            site_ids, raw_values, radii, recorded, site_ns):
        if source_mode == "combined":
            # Measured source: an empty spoke is a data-availability fact, so
            # say which one rather than reusing the report's "not flagged".
            if not np.isfinite(site_n):
                absent_text = "(no usable\ndata)"
            elif site_n <= 0:
                absent_text = "(n=0)"
            else:
                absent_text = f"(n={int(site_n)};\nbelow minimum)"
            detail = f"n={int(site_n)}" if np.isfinite(site_n) else ""
        else:
            absent_text = "(no recorded\nevidence)"
            detail = ""
        spoke_labels.append(_site_axis_label(
            site, raw_value, raw_deviation, is_recorded,
            absent_text=absent_text, detail=detail))
    ax.set_xticklabels(spoke_labels, fontsize=label_size)
    for label, is_recorded in zip(ax.get_xticklabels(), recorded):
        label.set_color(_RECORDED_LABEL_COLOR if is_recorded
                        else _UNRECORDED_LABEL_COLOR)

    upper = float(upper_radius) if np.isfinite(upper_radius) else 1.0
    upper = max(1.0, upper)
    tick_values = np.linspace(0, upper, 5)[1:]
    ax.set_ylim(0, upper)
    ax.set_yticks(tick_values)
    ax.set_yticklabels([f"{value:.2g}" for value in tick_values], fontsize=7)
    # Park the radial scale midway between two spokes; on a full site roster
    # the old fixed 12 degrees lands on top of a neighbouring axis label.
    ax.set_rlabel_position(min(12.0, 180.0 / max(len(site_ids), 1)))
    for tick_label in ax.get_yticklabels():
        tick_label.set_bbox({"boxstyle": "square,pad=0.12",
                             "facecolor": "white", "edgecolor": "none",
                             "alpha": 0.85})
    ax.grid(color="#B8B8B8", linewidth=0.7, alpha=0.65)
    ax.spines["polar"].set_color("#8A8A8A")

    # NaN radii break the polyline at unrecorded sites instead of dragging it
    # through the center, so the drawn line only ever joins real evidence.
    closed_radii = np.concatenate([radii, radii[:1]])
    ax.plot(angles, closed_radii, color="#5E548E", linewidth=2.0, zorder=2)
    finite_angles = point_angles[recorded]
    finite_radii = radii[recorded]
    # Shade only when every spoke on the page has evidence. A polygon drawn
    # across an unrecorded spoke would give that spoke a nonzero radius, which
    # is the same false claim as planting a marker on it.
    if (recorded.all() and len(finite_radii) >= 3
            and np.any(finite_radii > 0)):
        ax.fill(angles, closed_radii, color="#9F86C0", alpha=0.18, zorder=1)
    if len(finite_radii):
        if "deviation_direction" in variable_rows.columns:
            # Signed marker colour recovers the direction the absolute radius
            # throws away: below-reference and above-reference sites otherwise
            # sit on the same ring.
            directions = pd.to_numeric(
                variable_rows["deviation_direction"],
                errors="coerce").to_numpy(dtype=float)[recorded]
            point_colors = [_POINT_COLOR_BELOW if value < 0
                            else _POINT_COLOR_ABOVE for value in directions]
        else:
            point_colors = _POINT_COLOR
        ax.scatter(
            finite_angles, finite_radii, color=point_colors,
            edgecolors="white", linewidths=0.8, s=48, zorder=4)

    reference_known = (reference is not None and np.isfinite(reference)
                       and same_reference)
    unit_suffix = f" ({units})" if units else ""
    if source_mode == "combined":
        # A complete site roster clusters most sites near the centre, so a
        # centred annotation either hides those markers or is hidden by them,
        # and a corner box collides with the spoke labels. The subtitle is the
        # only place on a full-roster chart that is guaranteed free.
        reference_text = (f"{float(reference):.4g}" if reference_known
                          else "unavailable")
        ax.set_title(
            f"{title}\n{radius_caption}{unit_suffix}"
            f"\ncentre = cross-site reference {reference_text}",
            fontsize=11, pad=18)
    else:
        center_text = (f"RAW REFERENCE\n{float(reference):.3g}"
                       if reference_known else "RAW REFERENCE\nvaries by row")
        # Below the evidence scatter (zorder 4): a site whose deviation is
        # small sits near the centre, and an opaque annotation drawn over it
        # would hide the very marker that proves the site was recorded.
        ax.text(
            0.0, 0.0, center_text,
            ha="center", va="center", fontsize=7.2, fontweight="bold",
            color="#333333", zorder=3,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                  "edgecolor": "#777777", "linewidth": 0.7, "alpha": 0.94})
        ax.set_title(f"{title}\n{radius_caption}{unit_suffix}",
                     fontsize=11, pad=18)


def _atomic_savefig(fig, output: Path, *, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.stem}.{os.getpid()}.tmp{output.suffix}")
    try:
        fig.savefig(tmp, dpi=dpi, bbox_inches="tight", facecolor="white")
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _build_figure(group: pd.DataFrame, network: str, variable: str,
                  page_number: int, page_count: int, source_rows: int,
                  total_sites: int, total_recorded_sites: int, *,
                  reference: float | None, same_reference: bool,
                  upper_radius: float, source_mode: str = "report",
                  radius_caption: str = "Raw distance from reference",
                  radius_note: str = ("radius = raw absolute distance from the "
                                      "detector reference"),
                  recorded_phrase: str = "with recorded evidence for this variable",
                  recorded_legend: str = "Site with recorded evidence",
                  absent_legend: str = ("Site with no recorded evidence "
                                        "(labelled, no marker)"),
                  spoke_scope_label: str | None = None,
                  footer_note: str | None = None,
                  marker_legend: Sequence[tuple[str, str]] | None = None):
    """Assemble one radar page.

    Every display string a combined-CSV chart has to reword is a keyword with
    the anomaly-report wording as its default, so the report path renders
    byte-for-byte as before while a measured chart never inherits a caveat
    ("no flagged evidence in this thresholded report") that is untrue of it.
    """
    site_count = len(group)
    page_recorded = int(group["recorded_in_input"].fillna(False).astype(bool).sum())
    fig, ax = plt.subplots(figsize=_figure_size(site_count),
                           subplot_kw={"polar": True})
    variable_title = textwrap.fill(_text(variable), width=56)
    title = (
        f"{network} | {variable_title} | site deviations"
        + (f" | page {page_number} of {page_count}" if page_count > 1 else "")
    )
    _draw_radar(ax, group, title, reference=reference,
                same_reference=same_reference, upper_radius=upper_radius,
                source_mode=source_mode, radius_caption=radius_caption)

    markers = (list(marker_legend) if marker_legend
               else [(_POINT_COLOR, recorded_legend)])
    handles = [
        Line2D([0], [0], color="#5E548E", linewidth=2,
               label=radius_caption),
        *[Line2D([0], [0], linestyle="none", marker="o", markersize=7,
                 markerfacecolor=color, markeredgecolor="white", label=text)
          for color, text in markers],
        # Deliberately handle-less: an unrecorded site is drawn as an absence,
        # so there is no mark for the key to reproduce.
        Line2D([], [], linestyle="none", marker="",
               label=absent_legend),
    ]
    legend = fig.legend(
        handles=handles, loc="lower center", ncol=min(4, len(handles)),
        frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, 0.082))
    legend.get_texts()[-1].set_color(_UNRECORDED_LABEL_COLOR)
    if page_count > 1:
        coverage = (f"page {page_number} of {page_count}: {site_count} of "
                    f"{total_sites} sites, {page_recorded} of "
                    f"{total_recorded_sites} {recorded_phrase}")
    else:
        coverage = (f"{total_sites} sites; {total_recorded_sites} "
                    f"{recorded_phrase}")
    scope_label = network if spoke_scope_label is None else spoke_scope_label
    fig.text(
        0.5, 0.040,
        f"Spokes = every site in {scope_label} ({coverage}); {radius_note}.",
        ha="center", va="bottom", fontsize=8, color="#444444",
    )
    if footer_note is None:
        footer_note = (
            f"Variables: first {source_rows} usable rows/network. An unmarked "
            "spoke means no flagged evidence in this thresholded report, not a "
            "measured deviation of zero.")
    fig.text(
        0.5, 0.007, footer_note,
        ha="center", va="bottom", fontsize=8, color="#444444",
    )
    fig.subplots_adjust(top=0.88, bottom=0.22, left=0.08, right=0.92)
    return fig


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def generate_radar_charts(
    report_path: str | os.PathLike,
    output_dir: str | os.PathLike | None = None,
    *,
    sheet: str = DEFAULT_SHEET,
    networks: Sequence[str] | None = None,
    timepoints: Sequence[str] | None = None,
    source_rows: int = DEFAULT_SOURCE_ROWS,
    top_k: int | None = None,
    max_sites_per_chart: int | None = None,
    site_list: str | os.PathLike | None = None,
    output_format: str = "png",
    dpi: int = 180,
) -> dict:
    """Generate radar pages plus auditable score/evidence tables.

    ``max_sites_per_chart`` defaults to ``None``: every site in the network is
    drawn on a single page per variable.  Pass an integer only to deliberately
    split a very large roster across pages.
    """
    if max_sites_per_chart is not None and max_sites_per_chart < 1:
        raise RadarInputError(
            "max_sites_per_chart must be at least 1, or None to draw every "
            "site on one page")
    if dpi < 72:
        raise RadarInputError("dpi must be at least 72")
    if output_format not in {"png", "pdf", "both"}:
        raise RadarInputError("output_format must be png, pdf, or both")
    if top_k is not None:
        # Backward-compatible API alias from the earlier family-axis version.
        source_rows = top_k
    if source_rows < 1:
        raise RadarInputError("source_rows must be at least 1")

    source = Path(report_path)
    frame = read_site_network_table(source, sheet=sheet)
    filtered_non_site = int(frame.attrs.get("filtered_non_site_rows", 0))
    expanded, warnings = expand_site_rows(frame)
    # Collected from the unparsed table so a site that only ever appears on
    # malformed rows still gets a spoke.
    universe = collect_site_universe(frame)
    if filtered_non_site:
        warnings.insert(
            0,
            f"filtered {filtered_non_site} non-site_network_outlier row(s) "
            "from the supplied table")

    extra_sites: list[tuple[str, str]] = []
    if site_list is not None:
        extra_sites, site_list_warnings = read_site_list(site_list)
        warnings.extend(site_list_warnings)

    if networks:
        wanted = {_text(item).upper() for item in networks}
        expanded = expanded[
            expanded["network"].astype(str).str.upper().isin(wanted)]
        universe = universe[
            universe["network"].astype(str).str.upper().isin(wanted)]
    if timepoints:
        wanted_tp = {_timepoint_label(item).lower() for item in timepoints}
        expanded = expanded[
            expanded["timepoint"].astype(str).str.lower().isin(wanted_tp)]
        universe = universe[
            universe["timepoint"].astype(str).str.lower().isin(wanted_tp)]
    if expanded.empty:
        raise RadarInputError("No site rows remain after the requested filters.")

    scores, evidence = calculate_site_scores(
        expanded, source_rows=source_rows)
    if scores.empty:
        raise RadarInputError("No site scores could be calculated.")

    site_universe = _network_site_universe(universe, extra_sites)
    for network, network_scores in scores.groupby("network", sort=False):
        # Union with the scored sites so the roster can never be narrower than
        # the evidence it has to display.
        name = _text(network)
        roster = set(site_universe.get(name, []))
        roster.update(_text(value) for value in network_scores["site_id"])
        site_universe[name] = sorted(
            roster, key=lambda value: (value.casefold(), value))
    # A network with no selected variable draws no chart, so counting its sites
    # would overstate what the run actually shows.
    charted = {_text(value) for value in scores["network"].unique()}
    site_universe = {network: sites for network, sites in site_universe.items()
                     if network in charted}
    unmatched = sorted({
        network for network, _ in extra_sites
        if network and not any(name.casefold() == network.casefold()
                               for name in site_universe)})
    for network in unmatched:
        warnings.append(
            f"site list names network {network!r}, which is not in the "
            "charted input; those roster entries were ignored")

    selected_variables = (
        scores[["network", "variable", "variable_order", "axis_source_row"]]
        .drop_duplicates()
        .sort_values(["network", "variable_order"])
        .reset_index(drop=True)
    )
    recorded_counts = (
        scores.groupby(["network", "variable"], sort=False)
        .size().rename("recorded_site_count").reset_index())
    selected_variables = selected_variables.merge(
        recorded_counts, on=["network", "variable"], how="left",
        validate="one_to_one")
    selected_variables["spoke_site_count"] = (
        selected_variables["network"].map(
            lambda value: len(site_universe.get(_text(value), [])))
        .astype(int))
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = source.with_name(f"{source.stem}_site_radar_{stamp}")
    else:
        output = Path(output_dir)
    if output.exists():
        if not output.is_dir():
            raise RadarInputError(f"Output path is not a directory: {output}")
        if any(output.iterdir()):
            raise RadarInputError(
                f"Output directory is not empty: {output}. Choose a new or "
                "empty directory so stale charts cannot be mistaken for this run.")
    else:
        output.mkdir(parents=True)

    _write_csv_atomic(scores, output / "site_deviation_scores.csv")
    _write_csv_atomic(evidence, output / "site_deviation_evidence.csv")
    _write_csv_atomic(selected_variables, output / "selected_variables.csv")

    chart_count, pdf_path = _render_charts(
        scores, site_universe, output, warnings,
        max_sites_per_chart=max_sites_per_chart,
        output_format=output_format, dpi=dpi, source_rows=source_rows)

    spoke_limit_text = ("no limit (every site on one page)"
                        if max_sites_per_chart is None
                        else str(max_sites_per_chart))
    site_list_text = (str(Path(site_list).resolve())
                      if site_list is not None else "none supplied")
    roster_text = "; ".join(
        f"{network}: {len(sites)}"
        for network, sites in sorted(site_universe.items())) or "none"

    readme = textwrap.dedent(f"""\
        Site-network radar chart output
        ===============================

        Source: {source.resolve()}
        Sheet: {sheet}
        Source rows considered for variable selection per network: {source_rows}
        Maximum site spokes per chart: {spoke_limit_text}
        Site roster file: {site_list_text}
        Sites charted per network: {roster_text}

        Chart structure
        ---------------
        Every chart represents exactly one variable, and every chart carries one
        spoke per site in that network -- not only the sites this variable
        flagged. The roster is every concrete site the report names anywhere,
        collected before severity/value parsing so a site whose only rows are
        malformed still gets a spoke, optionally widened by an explicit site
        list. A spoke with no recorded evidence for the variable is labelled
        "(no recorded evidence)" and carries no marker; the connecting line
        breaks there, and the profile is shaded only on charts where every
        spoke has evidence, because a polygon crossing an unrecorded spoke
        would imply a value for it.
        Variables are selected in source order from the first {source_rows}
        usable site_network rows per network. The row limit is applied before
        collapsed multi-site rows are expanded. Pagination is off by default;
        set max_sites_per_chart to split a large roster across pages, in which
        case all pages of one variable share a single radial scale.

        Raw values
        ----------
        For every selected variable, all usable rows in the filtered input are
        considered. Repeated site-variable evidence across detector methods and
        timepoints is collapsed to the row with the maximum calibrated severity,
        but the charted radius comes from the raw observed statistic on that
        row. Radius 0 is the raw detector reference: cross-site median for
        median/spread/missingness checks and network/rest-of-network rho for
        correlation checks. The plotted radius is abs(raw observed value -
        reference value), not severity_score. Method and timepoint provenance
        remain in site_deviation_evidence.csv.

        Important limitation
        --------------------
        The anomaly report is thresholded and may be presentation-capped. It is
        not a complete site-by-variable matrix. Unrecorded, missing, nonnumeric,
        or omitted site-variable pairs still get a labelled spoke so no site is
        invisible, but they stay out of the score tables and are
        never plotted as zero or as the median. An unmarked spoke means "no flagged
        evidence for this variable in this thresholded report"; it is not
        evidence that the site's deviation is zero. These charts therefore
        compare recorded flags and cannot reconstruct below-threshold site
        statistics. Without a --site-list roster, the spoke set is still bounded
        by the sites this report names: a site with no rows at all anywhere in
        the input cannot be inferred from the input. To chart every site the
        study actually contains, use the combined-CSV source instead
        (--from-combined / generate_radar_charts_from_combined).

        Audit files
        -----------
        site_deviation_scores.csv contains one row per recorded site-variable
        pair used by the charts, including raw_value, reference_value, and
        raw_deviation; it remains an evidence-only table and gains no rows for
        unrecorded spokes. site_deviation_evidence.csv preserves strongest
        source evidence. selected_variables.csv records variable-selection order
        and the spoke count per network. chart_manifest.csv maps every plotted
        spoke, recorded or not, to exactly one variable chart; its
        recorded_in_input column separates the two.

        Warnings
        --------
        """)
    warning_lines = [
        textwrap.fill(
            item, width=100, initial_indent="- ", subsequent_indent="  ")
        for item in warnings
    ] or ["- None"]
    readme = readme.rstrip() + "\n" + "\n".join(warning_lines) + "\n"
    _write_text_atomic(readme, output / "README.txt")

    return {
        "output_dir": output,
        "score_path": output / "site_deviation_scores.csv",
        "evidence_path": output / "site_deviation_evidence.csv",
        "selection_path": output / "selected_variables.csv",
        "axis_path": output / "selected_variables.csv",
        "manifest_path": output / "chart_manifest.csv",
        "pdf_path": pdf_path if output_format in {"pdf", "both"} else None,
        "chart_count": chart_count,
        "site_count": int(scores[["network", "site_id"]]
                          .drop_duplicates().shape[0]),
        # Every site drawn as a spoke, including those with no evidence.
        "spoke_site_count": int(sum(len(sites)
                                    for sites in site_universe.values())),
        "sites_by_network": {network: list(sites)
                             for network, sites in site_universe.items()},
        "variable_count": int(selected_variables[["network", "variable"]]
                              .drop_duplicates().shape[0]),
        "warnings": warnings,
    }


def _render_charts(scores: pd.DataFrame, site_universe: dict[str, list[str]],
                   output: Path, warnings: list[str], *,
                   max_sites_per_chart: int | None,
                   output_format: str, dpi: int,
                   source_rows: int = DEFAULT_SOURCE_ROWS,
                   figure_options: dict | None = None,
                   extra_manifest_columns: Sequence[str] = (),
                   ) -> tuple[int, Path]:
    """Draw every chart page, write the PDF and the spoke manifest.

    Shared by both sources. Charts are grouped by an optional ``chart_group``
    column, which lets the combined-CSV source put each (scope, timepoint) on
    its own page while the anomaly-report source keeps grouping by network --
    there, ``chart_group`` is simply the network, so the grouping, ordering,
    filenames and manifest are byte-for-byte what they were before.
    """
    figure_options = dict(figure_options or {})
    frame = scores.copy()
    if "chart_group" not in frame.columns:
        frame["chart_group"] = frame["network"]

    pdf_path = output / "site_network_radar.pdf"
    tmp_pdf = None
    pdf_writer = None
    chart_rows: list[dict] = []
    chart_count = 0
    if output_format in {"pdf", "both"}:
        tmp_pdf = pdf_path.with_name(
            f".{pdf_path.stem}.{os.getpid()}.tmp{pdf_path.suffix}")

    # Write PDF pages as they are drawn and close every figure immediately.
    # A large report can contain many network/site pages; retaining every
    # Matplotlib figure until the end would otherwise consume substantial RAM.
    try:
        if tmp_pdf is not None:
            pdf_writer = PdfPages(tmp_pdf)
        try:
            grouped = list(frame.groupby(
                ["chart_group", "variable"], sort=False, dropna=False))
            grouped.sort(key=lambda item: (
                _text(item[0][0]).casefold(),
                int(item[1]["variable_order"].iloc[0])))
            for (network, variable), group in grouped:
                # The manifest keeps the true network even when the page is
                # grouped by a wider label such as "STUDY | baseline".
                network_value = _text(group["network"].iloc[0])
                # Every site in the network becomes a spoke, in stable
                # alphabetical order, so page membership and angular positions
                # are reproducible and comparable across variables. Sites
                # without evidence carry NaN values, never a zero fill.
                ordered_sites = _variable_spoke_frame(
                    group, site_universe.get(_text(network), []))
                total_sites = len(ordered_sites)
                recorded_mask = (ordered_sites["recorded_in_input"]
                                 .fillna(False).astype(bool))
                total_recorded_sites = int(recorded_mask.sum())
                recorded_radii = pd.to_numeric(
                    ordered_sites.loc[recorded_mask, "plot_radius"],
                    errors="coerce").dropna()
                # One radial scale and one reference per variable so paginated
                # pages of the same variable stay directly comparable.
                max_radius = (float(recorded_radii.max())
                              if not recorded_radii.empty else 0.0)
                upper_radius = max(1.0, max_radius * 1.15)
                reference_values = pd.to_numeric(
                    ordered_sites.loc[recorded_mask, "reference_value"],
                    errors="coerce").dropna()
                reference = (float(reference_values.iloc[0])
                             if not reference_values.empty else None)
                same_reference = bool(
                    reference_values.empty
                    or np.allclose(reference_values.to_numpy(dtype=float),
                                   reference_values.iloc[0]))
                page_sizes = _balanced_page_sizes(
                    total_sites,
                    total_sites if max_sites_per_chart is None
                    else max_sites_per_chart)
                pages = len(page_sizes)
                if total_recorded_sites < 3:
                    warnings.append(
                        f"{network} / {variable}: only {total_recorded_sites} "
                        "recorded site spoke(s); the shaded profile is "
                        "geometrically degenerate, so interpret the score "
                        "table alongside the chart")
                crowded_pages = [
                    index + 1 for index, size in enumerate(page_sizes)
                    if size > _SPOKE_LEGIBILITY_WARN
                ]
                if crowded_pages:
                    page_label = ", ".join(str(value)
                                           for value in crowded_pages)
                    warnings.append(
                        f"{network} / {variable}: radar page(s) {page_label} "
                        f"draw more than {_SPOKE_LEGIBILITY_WARN} site spokes; "
                        "labels may crowd. Use max_sites_per_chart to "
                        "paginate if the page is unreadable")
                start = 0
                for page_index, page_size in enumerate(page_sizes):
                    chunk = ordered_sites.iloc[
                        start:start + page_size].copy()
                    start += page_size
                    filename = _chart_filename(
                        _text(network), _text(variable), page_index + 1)
                    png_path = output / filename
                    # The scope label is per chart group (a combined-CSV page
                    # is one scope at one timepoint), so it cannot live in the
                    # run-wide figure options.
                    options = dict(figure_options)
                    if "chart_scope_label" in group.columns:
                        scope_label = _text(
                            group["chart_scope_label"].iloc[0])
                        if scope_label:
                            options["spoke_scope_label"] = scope_label
                    fig = _build_figure(
                        chunk, _text(network), _text(variable),
                        page_index + 1, pages, source_rows,
                        total_sites, total_recorded_sites,
                        reference=reference, same_reference=same_reference,
                        upper_radius=upper_radius, **options)
                    try:
                        if output_format in {"png", "both"}:
                            _atomic_savefig(fig, png_path, dpi=dpi)
                        if pdf_writer is not None:
                            pdf_writer.savefig(
                                fig, bbox_inches="tight", facecolor="white")
                    finally:
                        plt.close(fig)
                    chart_count += 1
                    for position, row in enumerate(
                            chunk.itertuples(index=False)):
                        entry = {
                            "network": network_value,
                            "variable": variable,
                            "variable_order": int(row.variable_order),
                            "site_id": row.site_id,
                            "recorded_in_input": bool(row.recorded_in_input),
                            "raw_value": row.raw_value,
                            "reference_value": row.reference_value,
                            "raw_deviation": row.raw_deviation,
                            "plot_radius": row.plot_radius,
                            "raw_units": row.raw_units,
                            "severity_score": row.severity_score,
                            "spokes_total": total_sites,
                            "recorded_sites_total": total_recorded_sites,
                            "page": page_index + 1,
                            "variable_page_count": pages,
                            "pdf_page": chart_count,
                            "png": filename if output_format in {"png", "both"} else "",
                        }
                        if _text(network_value) != _text(network):
                            entry["chart_group"] = network
                        # itertuples renames non-identifier columns, so read
                        # the optional provenance columns off the frame.
                        for column in extra_manifest_columns:
                            if column in chunk.columns:
                                entry[column] = chunk[column].iloc[position]
                        chart_rows.append(entry)
        finally:
            if pdf_writer is not None:
                pdf_writer.close()
        if tmp_pdf is not None:
            os.replace(tmp_pdf, pdf_path)
    finally:
        if tmp_pdf is not None and tmp_pdf.exists():
            try:
                tmp_pdf.unlink()
            except OSError:
                pass

    manifest = pd.DataFrame(chart_rows)
    _write_csv_atomic(manifest, output / "chart_manifest.csv")
    return chart_count, pdf_path


# ---------------------------------------------------------------------------
# Combined-CSV source.
#
# The anomaly-report source above can only ever show the sites some detector
# flagged. This source reads the study data itself, so every site that has rows
# appears with its own measured statistic and the roster is complete by
# construction rather than by luck of what got flagged.
# ---------------------------------------------------------------------------


def _combined_value_helpers():
    """Import the detector's value-cleaning helpers, lazily.

    Deliberately not a module-level import: the anomaly-report path must keep
    working in a checkout that does not ship the detector package. Reusing
    ``to_numeric_clean`` rather than re-implementing it is what keeps this
    chart's missing-code handling identical to the site_network detector's --
    -3 / -9 / -99 / 999 are missing data, not values, and a chart that treated
    999 as a score would put the site's median in orbit.
    """
    try:
        from simple_anomaly_detection.common import (  # noqa: WPS433
            site_from_subjectid, to_numeric_clean)
    except ImportError as exc:  # pragma: no cover - depends on checkout layout
        raise RadarInputError(
            "The combined-CSV source needs the simple_anomaly_detection "
            f"package importable ({exc}); run from the repository root."
        ) from exc
    return site_from_subjectid, to_numeric_clean


def _token_present(text: str, token: str) -> bool:
    """True if ``token`` appears in ``text`` as a whole alphanumeric token.

    Mirrors the runner's ``_basename_tp_ok`` guard so ``month_1`` cannot match
    ``month_10`` / ``month_18`` and a chart is never labelled the wrong visit.
    """
    pattern = (r"(?<![a-z0-9])" + re.escape(token.lower())
               + r"(?![0-9])")
    return re.search(pattern, text.lower()) is not None


def _combined_timepoint(basename: str) -> str:
    """Normalized timepoint label parsed from a combined-CSV filename."""
    stem = Path(basename).stem
    for token, label in _COMBINED_TIMEPOINT_TOKENS:
        if _token_present(stem, token) or _token_present(
                stem, token.replace("_", "-")):
            return label
    return ""


def _combined_network(basename: str) -> str:
    stem = Path(basename).stem
    for token in _COMBINED_NETWORK_TOKENS:
        if _token_present(stem, token):
            return token
    return ""


def discover_combined_files(
    input_dir: str | os.PathLike,
    networks: Sequence[str] | None = None,
    timepoints: Sequence[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """List the combined REDCap CSVs in ``input_dir`` with parsed identities.

    Network and timepoint come from the filename, which is how the detector
    runner resolves them too. A file whose network cannot be identified is
    reported rather than silently skipped: dropping it would quietly shrink the
    site roster, which is the exact failure this source exists to fix.
    """
    directory = Path(input_dir)
    if not directory.is_dir():
        raise RadarInputError(
            f"Combined-CSV input directory does not exist: {directory}")

    wanted_networks = ({_text(item).upper() for item in networks}
                       if networks else None)
    wanted_timepoints = ({_text(item).lower() for item in timepoints}
                         if timepoints else None)

    entries: list[dict] = []
    warnings: list[str] = []
    for path in sorted(directory.glob("*.csv")):
        network = _combined_network(path.name)
        if not network:
            warnings.append(
                f"{path.name}: filename names no known network "
                f"({'/'.join(_COMBINED_NETWORK_TOKENS)}); skipped")
            continue
        timepoint = _combined_timepoint(path.name)
        if not timepoint:
            warnings.append(
                f"{path.name}: no recognizable timepoint token in the "
                "filename; skipped")
            continue
        if wanted_networks is not None and network not in wanted_networks:
            continue
        if (wanted_timepoints is not None
                and timepoint.lower() not in wanted_timepoints):
            continue
        entries.append({"path": str(path), "network": network,
                        "timepoint": timepoint})
    return entries, warnings


def read_combined_site_statistics(
    input_dir: str | os.PathLike,
    variables: Sequence[str],
    *,
    networks: Sequence[str] | None = None,
    timepoints: Sequence[str] | None = None,
    roster_all_timepoints: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Compute per-site statistics for ``variables`` straight from the CSVs.

    Returns ``(statistics, roster, warnings)``.

    ``roster`` is collected from every readable file, including files that do
    not carry the requested variable at all and -- when
    ``roster_all_timepoints`` is set -- files outside the charted timepoints.
    That is the point: a site that first enrolled at month 1, or whose form was
    not collected at the charted visit, still belongs on the chart as a site
    with no data there. Deriving the roster from the charted slice alone would
    re-create the very gap this source exists to close. Roster-only files cost
    one column of I/O: their identifier is read and nothing else.

    Only the identifier column and the requested variables are read from disk.
    """
    requested = [name for name in
                 dict.fromkeys(_text(item) for item in variables) if name]
    if not requested:
        raise RadarInputError("No variable names were supplied to chart.")

    entries, warnings = discover_combined_files(
        input_dir, networks, timepoints)
    if not entries:
        raise RadarInputError(
            f"No usable combined CSVs were found in {Path(input_dir)}. "
            "Expected files named like "
            "'AMPSCZ-combined-redcap_baseline_PRONET.csv'.")
    for entry in entries:
        entry["collect_stats"] = True
    if roster_all_timepoints and timepoints:
        charted = {entry["path"] for entry in entries}
        roster_only, _ = discover_combined_files(input_dir, networks, None)
        extra = [dict(entry, collect_stats=False) for entry in roster_only
                 if entry["path"] not in charted]
        if extra:
            other = sorted({entry["timepoint"] for entry in extra})
            warnings.append(
                f"site roster widened using {len(extra)} file(s) outside the "
                f"charted timepoint(s) ({', '.join(other)}) so sites that "
                "enrolled later still appear as spokes")
        entries = entries + extra

    site_from_subjectid, to_numeric_clean = _combined_value_helpers()
    stats_rows: list[dict] = []
    roster_rows: list[dict] = []
    files_with_variable: dict[str, int] = {name: 0 for name in requested}

    for entry in entries:
        path = Path(entry["path"])
        try:
            header = pd.read_csv(path, nrows=0, dtype=str,
                                 keep_default_na=False)
        except Exception as exc:
            warnings.append(f"{path.name}: header unreadable ({exc}); skipped")
            continue
        lookup: dict[str, str] = {}
        for column in header.columns:
            lookup.setdefault(_text(column).lower(), column)
        id_column = next((lookup[name] for name in COMBINED_ID_COLUMNS
                          if name in lookup), None)
        if id_column is None:
            warnings.append(
                f"{path.name}: no participant id column "
                f"({', '.join(COMBINED_ID_COLUMNS)}); skipped")
            continue

        present = ({name: lookup[name.lower()] for name in requested
                    if name.lower() in lookup}
                   if entry.get("collect_stats", True) else {})
        usecols = list(dict.fromkeys(
            [id_column] + [column for column in present.values()]))
        try:
            # keep_default_na=False mirrors the detector runner and keeps a
            # literal 'NA' site code from becoming a missing value.
            frame = pd.read_csv(path, usecols=usecols, dtype=str,
                                keep_default_na=False, low_memory=False)
        except Exception as exc:
            warnings.append(f"{path.name}: unreadable ({exc}); skipped")
            continue

        subject_ids = frame[id_column].astype(str).str.strip()
        usable = ~subject_ids.str.lower().isin(_BLANK_SUBJECT_IDS)
        blank_rows = int((~usable).sum())
        frame = frame.loc[usable].copy()
        frame[id_column] = subject_ids.loc[usable]
        before = len(frame)
        # One row per participant per visit, matching the detector: a repeated
        # id would otherwise weight that participant twice in its site median.
        frame = frame.drop_duplicates(subset=[id_column], keep="first")
        duplicate_rows = before - len(frame)
        if blank_rows or duplicate_rows:
            warnings.append(
                f"{path.name}: excluded {blank_rows} blank-id row(s) and "
                f"{duplicate_rows} repeated-id row(s)")
        if frame.empty:
            warnings.append(f"{path.name}: no usable participant rows")
            continue

        sites = frame[id_column].map(site_from_subjectid).map(_text)
        frame = frame.loc[sites.ne("")].copy()
        sites = sites.loc[sites.ne("")]
        for site in sorted(set(sites)):
            roster_rows.append({"network": entry["network"],
                                "timepoint": entry["timepoint"],
                                "site_id": site})

        for name in requested:
            column = present.get(name)
            if column is None:
                continue
            files_with_variable[name] += 1
            work = pd.DataFrame({
                "site_id": sites.to_numpy(),
                "value": to_numeric_clean(frame[column]).to_numpy(dtype=float),
            })
            grouped = work.groupby("site_id", sort=True)["value"]
            summary = grouped.agg(
                n_subjects="size", n_observed="count",
                site_median="median", site_mean="mean",
                q1=lambda values: values.quantile(0.25),
                q3=lambda values: values.quantile(0.75),
            ).reset_index()
            summary["site_iqr"] = summary["q3"] - summary["q1"]
            # Percentage points, not a fraction: the report source's
            # missingness axis is already in points, and mixing the two would
            # put the axes 100x apart.
            summary["site_missing_pct"] = 100.0 * (
                summary["n_subjects"] - summary["n_observed"]
            ) / summary["n_subjects"].where(summary["n_subjects"] > 0)
            summary["network"] = entry["network"]
            summary["timepoint"] = entry["timepoint"]
            summary["variable"] = name
            summary["source_file"] = path.name
            stats_rows.append(summary)

    for name, count in files_with_variable.items():
        if count == 0:
            warnings.append(
                f"variable {name!r} was not a column in any scanned combined "
                "CSV; check the spelling or the timepoint filter")
    roster = (pd.DataFrame(roster_rows,
                           columns=["network", "timepoint", "site_id"])
              .drop_duplicates().reset_index(drop=True))
    if not stats_rows:
        raise RadarInputError(
            "None of the scanned combined CSVs contain the requested "
            f"variable(s): {', '.join(requested)}.")
    statistics = pd.concat(stats_rows, ignore_index=True, sort=False)
    columns = ["network", "timepoint", "variable", "site_id", "n_subjects",
               "n_observed", "site_median", "site_mean", "site_iqr",
               "site_missing_pct", "q1", "q3", "source_file"]
    return statistics[columns], roster, warnings


def _mad_sigma(values: np.ndarray) -> float:
    """MAD-scaled sigma, or NaN when the values carry no robust spread."""
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return float("nan")
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    sigma = mad * 1.4826
    return sigma if sigma > 0 else float("nan")


def _scope_specs(statistics: pd.DataFrame, scope: str
                 ) -> list[tuple[str, str, tuple[str, ...]]]:
    """Return ``(scope_key, scope_label, networks)`` for each chart scope."""
    networks = [name for name in
                dict.fromkeys(_text(value)
                              for value in statistics["network"]) if name]
    specs: list[tuple[str, str, tuple[str, ...]]] = []
    if scope in {"network", "both"}:
        specs.extend((name, name, (name,)) for name in sorted(networks))
    if scope in {"study", "both"}:
        # One blended reference across networks. PRONET and PRESCIENT differ by
        # design, so this is a study-wide comparison, not a network one -- the
        # label says so on the chart.
        specs.append((STUDY_SCOPE_LABEL, "the study (all networks)",
                      tuple(sorted(networks))))
    return specs


def build_combined_scores(
    statistics: pd.DataFrame,
    roster: pd.DataFrame,
    variables: Sequence[str],
    *,
    statistic: str = "median",
    scope: str = "network",
    min_n: int = DEFAULT_MIN_SITE_N,
    reference_min_n: int = DEFAULT_REFERENCE_MIN_N,
    extra_sites: Sequence[tuple[str, str]] = (),
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Turn per-site statistics into the chart score frame plus site rosters.

    One row per (chart page, site) for every site in the scope's roster,
    recorded or not, so the drawing layer receives a complete spoke set and the
    audit CSV shows exactly why an empty spoke is empty.
    """
    if statistic not in COMBINED_STATISTICS:
        raise RadarInputError(
            f"statistic must be one of {', '.join(sorted(COMBINED_STATISTICS))}")
    if scope not in {"network", "study", "both"}:
        raise RadarInputError("scope must be network, study, or both")
    if min_n < 1:
        raise RadarInputError("min_n must be at least 1")
    if reference_min_n < 1:
        raise RadarInputError("reference_min_n must be at least 1")
    warnings = warnings if warnings is not None else []

    value_column, units, stat_label = COMBINED_STATISTICS[statistic]
    # A missing-rate needs a denominator of enrolled participants, not of
    # observed values; gating it on n_observed would drop exactly the sites
    # whose missingness is the finding.
    count_column = ("n_subjects" if statistic == "missing" else "n_observed")

    ordered_variables = [name for name in
                         dict.fromkeys(_text(item) for item in variables)
                         if name]
    variable_order = {name: index
                      for index, name in enumerate(ordered_variables)}

    site_networks: dict[str, str] = {}
    for record in roster.itertuples(index=False):
        site_networks.setdefault(_text(record.site_id), _text(record.network))

    rows: list[dict] = []
    site_universe: dict[str, list[str]] = {}

    for scope_key, scope_label, scope_networks in _scope_specs(
            statistics, scope):
        in_scope = statistics["network"].map(_text).isin(scope_networks)
        scope_stats = statistics.loc[in_scope]
        scope_roster = set(
            _text(value) for value in
            roster.loc[roster["network"].map(_text).isin(scope_networks),
                       "site_id"])
        for network_name, site in extra_sites:
            if not network_name or any(
                    network_name.casefold() == name.casefold()
                    for name in scope_networks) or scope_key == STUDY_SCOPE_LABEL:
                scope_roster.add(_text(site))
                site_networks.setdefault(_text(site), _text(network_name))
        roster_sites = sorted(scope_roster,
                              key=lambda value: (value.casefold(), value))

        for (timepoint, variable), part in scope_stats.groupby(
                ["timepoint", "variable"], sort=False):
            if _text(variable) not in variable_order:
                continue
            chart_group = f"{scope_key} | {_text(timepoint)}"
            site_universe[chart_group] = list(roster_sites)

            part = part.copy()
            part["site_id"] = part["site_id"].map(_text)
            duplicated = part["site_id"].duplicated(keep=False)
            if duplicated.any():
                # Only reachable when one 2-letter prefix is used by both
                # networks under --scope study. Keep the better-supported row
                # and say so rather than letting the spoke merge raise.
                collided = sorted(set(part.loc[duplicated, "site_id"]))
                warnings.append(
                    f"{chart_group} / {variable}: site prefix(es) "
                    f"{', '.join(collided)} appear in more than one network; "
                    "kept the row with the most observations")
                part = part.sort_values(
                    [count_column], ascending=False, kind="stable")
            part = part.drop_duplicates("site_id", keep="first")
            part = part.set_index("site_id")

            values = pd.to_numeric(part[value_column], errors="coerce")
            counts = pd.to_numeric(part[count_column],
                                   errors="coerce").fillna(0)
            plottable = values.notna() & (counts >= min_n)
            reference_pool = values[values.notna()
                                    & (counts >= reference_min_n)]
            if len(reference_pool) < _MIN_REFERENCE_SITES:
                fallback = values[plottable]
                warnings.append(
                    f"{chart_group} / {variable}: only {len(reference_pool)} "
                    f"site(s) reach n >= {reference_min_n}; the cross-site "
                    f"reference falls back to the median over all "
                    f"{len(fallback)} plotted site(s)")
                reference_pool = fallback
            if reference_pool.empty:
                warnings.append(
                    f"{chart_group} / {variable}: no site has a usable "
                    f"{stat_label}; no chart drawn")
                continue
            reference = float(np.nanmedian(
                reference_pool.to_numpy(dtype=float)))
            sigma = _mad_sigma(reference_pool.to_numpy(dtype=float))
            deviations = (values - reference).abs()
            ranks = deviations[plottable].rank(method="min", ascending=False)

            for site in roster_sites:
                has_row = site in part.index
                value = float(values.get(site, np.nan)) if has_row else np.nan
                count = float(counts.get(site, np.nan)) if has_row else np.nan
                recorded = bool(has_row and plottable.get(site, False))
                deviation = (abs(value - reference)
                             if recorded and np.isfinite(value) else np.nan)
                rows.append({
                    "chart_group": chart_group,
                    "chart_scope_label": scope_label,
                    "scope": scope_key,
                    "network": (_text(part.at[site, "network"]) if has_row
                                else site_networks.get(site, "")),
                    "timepoint": _text(timepoint),
                    "site_id": site,
                    "variable": _text(variable),
                    "variable_order": variable_order[_text(variable)],
                    "axis_source_row": 0,
                    "statistic": statistic,
                    "plot_radius": deviation,
                    "raw_deviation": deviation,
                    "deviation_direction": (
                        float(np.sign(value - reference))
                        if recorded and np.isfinite(value) else np.nan),
                    "raw_value": value if recorded else np.nan,
                    "reference_value": reference,
                    "raw_units": units,
                    "raw_score": np.nan,
                    "severity_score": np.nan,
                    "robust_z": (abs(value - reference) / sigma
                                 if recorded and np.isfinite(sigma)
                                 else np.nan),
                    "site_n": count,
                    "n_observed": (float(part.at[site, "n_observed"])
                                   if has_row else np.nan),
                    "n_subjects": (float(part.at[site, "n_subjects"])
                                   if has_row else np.nan),
                    "site_rank_by_deviation": (int(ranks.get(site))
                                               if recorded
                                               and site in ranks.index
                                               else np.nan),
                    "recorded_in_input": recorded,
                    "recorded_site_count": int(plottable.sum()),
                    "reference_site_count": int(len(reference_pool)),
                    "reference_basis": (
                        f"cross-site median of the site {stat_label} over "
                        f"sites with n >= {reference_min_n}"),
                    "source_file": (_text(part.at[site, "source_file"])
                                    if has_row else ""),
                })

    if not rows:
        raise RadarInputError(
            "No site statistics could be charted from the combined CSVs.")
    scores = pd.DataFrame(rows)
    numeric_to_round = ["plot_radius", "raw_deviation", "raw_value",
                        "reference_value", "robust_z"]
    for column in numeric_to_round:
        scores[column] = pd.to_numeric(scores[column],
                                       errors="coerce").round(4)
    scores = scores.sort_values(
        ["chart_group", "variable_order", "site_id"],
        kind="stable").reset_index(drop=True)
    return scores, site_universe


def generate_radar_charts_from_combined(
    input_dir: str | os.PathLike,
    variables: Sequence[str] | str,
    output_dir: str | os.PathLike | None = None,
    *,
    networks: Sequence[str] | None = None,
    timepoints: Sequence[str] | None = None,
    statistic: str = "median",
    scope: str = "network",
    min_n: int = DEFAULT_MIN_SITE_N,
    reference_min_n: int = DEFAULT_REFERENCE_MIN_N,
    site_list: str | os.PathLike | None = None,
    roster_all_timepoints: bool = True,
    max_sites_per_chart: int | None = None,
    output_format: str = "png",
    dpi: int = 180,
) -> dict:
    """Chart every study site for ``variables``, measured from combined CSVs.

    One chart per (scope, timepoint, variable). Every site the scanned CSVs
    name anywhere gets a spoke on every chart in its scope, whether or not it
    has data for that variable at that timepoint, so the roster is a property
    of the study rather than of what happened to be flagged.
    """
    if isinstance(variables, str):
        variables = [variables]
    variables = [name for name in
                 dict.fromkeys(_text(item) for item in variables) if name]
    if not variables:
        raise RadarInputError("Supply at least one variable to chart.")
    if max_sites_per_chart is not None and max_sites_per_chart < 1:
        raise RadarInputError(
            "max_sites_per_chart must be at least 1, or None to draw every "
            "site on one page")
    if dpi < 72:
        raise RadarInputError("dpi must be at least 72")
    if output_format not in {"png", "pdf", "both"}:
        raise RadarInputError("output_format must be png, pdf, or both")
    if statistic not in COMBINED_STATISTICS:
        raise RadarInputError(
            f"statistic must be one of {', '.join(sorted(COMBINED_STATISTICS))}")

    source = Path(input_dir)
    statistics, roster, warnings = read_combined_site_statistics(
        source, variables, networks=networks, timepoints=timepoints,
        roster_all_timepoints=roster_all_timepoints)

    extra_sites: list[tuple[str, str]] = []
    if site_list is not None:
        extra_sites, site_list_warnings = read_site_list(site_list)
        warnings.extend(site_list_warnings)

    scores, site_universe = build_combined_scores(
        statistics, roster, variables, statistic=statistic, scope=scope,
        min_n=min_n, reference_min_n=reference_min_n,
        extra_sites=extra_sites, warnings=warnings)

    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = source / f"site_radar_combined_{stamp}"
    else:
        output = Path(output_dir)
    if output.exists():
        if not output.is_dir():
            raise RadarInputError(f"Output path is not a directory: {output}")
        if any(output.iterdir()):
            raise RadarInputError(
                f"Output directory is not empty: {output}. Choose a new or "
                "empty directory so stale charts cannot be mistaken for this "
                "run.")
    else:
        output.mkdir(parents=True)

    _write_csv_atomic(statistics, output / "site_statistics.csv")
    _write_csv_atomic(scores, output / "site_deviation_scores.csv")
    selected_variables = (
        scores[["chart_group", "network", "timepoint", "variable",
                "variable_order", "recorded_site_count",
                "reference_site_count"]]
        .drop_duplicates(subset=["chart_group", "variable"])
        .sort_values(["chart_group", "variable_order"])
        .reset_index(drop=True))
    selected_variables["spoke_site_count"] = selected_variables[
        "chart_group"].map(lambda value: len(site_universe.get(value, [])))
    _write_csv_atomic(selected_variables, output / "selected_variables.csv")

    _, _, stat_label = COMBINED_STATISTICS[statistic]
    figure_options = {
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
            f"it has n >= {min_n} usable observation(s); the cross-site "
            f"reference is the median over sites with n >= {reference_min_n}. "
            "An unmarked spoke means no usable data there, not a deviation "
            "of zero."),
    }
    chart_count, pdf_path = _render_charts(
        scores, site_universe, output, warnings,
        max_sites_per_chart=max_sites_per_chart,
        output_format=output_format, dpi=dpi,
        figure_options=figure_options,
        extra_manifest_columns=("chart_group", "chart_scope_label", "scope",
                                "timepoint", "statistic", "site_n",
                                "n_observed", "n_subjects", "robust_z",
                                "site_rank_by_deviation"))

    roster_text = "; ".join(
        f"{group}: {len(sites)}"
        for group, sites in sorted(site_universe.items())) or "none"
    scanned = (statistics[["network", "timepoint", "source_file"]]
               .drop_duplicates())
    site_list_text = (str(Path(site_list).resolve())
                      if site_list is not None else "none supplied")
    readme = textwrap.dedent(f"""\
        Site radar charts measured from the AMPSCZ combined CSVs
        ========================================================

        Input directory: {source.resolve()}
        Variables: {', '.join(variables)}
        Statistic: {statistic} ({stat_label})
        Scope: {scope}
        Minimum observations to plot a site: {min_n}
        Minimum observations to contribute to the reference: {reference_min_n}
        Maximum site spokes per chart: {
            'no limit (every site on one page)'
            if max_sites_per_chart is None else max_sites_per_chart}
        Site roster file: {site_list_text}
        Files carrying the variable(s): {len(scanned)}
        Sites charted per page group: {roster_text}

        What a spoke means here
        -----------------------
        This output does NOT come from a thresholded anomaly report. Every site
        statistic is computed directly from the combined CSVs, so a spoke with
        a marker is a measured value and the roster is every site the scanned
        files name anywhere -- including sites that have no data for the
        charted variable or timepoint. Those appear as labelled spokes with no
        marker and their observation count in the label, which is a statement
        about data availability, not about deviation. Nothing is imputed and no
        absent site is plotted at zero.

        Radius
        ------
        radius = abs(site {stat_label} - cross-site {stat_label}). The centre is
        the cross-site reference: the median of the per-site {stat_label} taken
        over sites with at least {reference_min_n} observations. When fewer than
        {_MIN_REFERENCE_SITES} sites clear that bar the reference falls back to
        the median over every plotted site and the fallback is listed in the
        warnings below. Missing-rate charts are in percentage points.

        Missing codes
        -------------
        Values are cleaned with the site_network detector's own
        to_numeric_clean, so -3 / -9 / -99 / 999 are treated as missing rather
        than as scores. One row per participant per visit is used; blank and
        repeated participant ids are excluded and counted in the warnings.

        Scope
        -----
        scope=network compares each site against its own network. scope=study
        compares every site against one blended all-network reference; PRONET
        and PRESCIENT differ by design, so read a study-scope chart as a
        study-wide comparison rather than as evidence about one network.

        Audit files
        -----------
        site_statistics.csv is the per (network, timepoint, variable, site)
        statistics table straight from the CSVs, including n_subjects and
        n_observed. site_deviation_scores.csv has one row per plotted spoke,
        recorded or not, with the reference, the deviation, and a robust z.
        selected_variables.csv lists each chart page. chart_manifest.csv maps
        every spoke to its chart page and PNG.

        Warnings
        --------
        """)
    warning_lines = [
        textwrap.fill(item, width=100, initial_indent="- ",
                      subsequent_indent="  ")
        for item in warnings] or ["- None"]
    readme = readme.rstrip() + "\n" + "\n".join(warning_lines) + "\n"
    _write_text_atomic(readme, output / "README.txt")

    recorded = scores.loc[scores["recorded_in_input"].astype(bool)]
    return {
        "output_dir": output,
        "statistics_path": output / "site_statistics.csv",
        "score_path": output / "site_deviation_scores.csv",
        "selection_path": output / "selected_variables.csv",
        "manifest_path": output / "chart_manifest.csv",
        "pdf_path": pdf_path if output_format in {"pdf", "both"} else None,
        "chart_count": chart_count,
        "site_count": int(recorded[["chart_group", "site_id"]]
                          .drop_duplicates().shape[0]),
        "spoke_site_count": int(sum(len(sites)
                                    for sites in site_universe.values())),
        "sites_by_network": {group: list(sites)
                             for group, sites in site_universe.items()},
        "variable_count": int(scores[["chart_group", "variable"]]
                              .drop_duplicates().shape[0]),
        "warnings": warnings,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create site-level radar charts, either from the site_network tab "
            "of an anomaly report (default) or measured directly from the "
            "AMPSCZ combined CSVs (--from-combined), which charts every site "
            "in the study rather than only the flagged ones."))
    parser.add_argument(
        "report", nargs="?",
        help=("anomaly_report.xlsx or a CSV export of its site_network tab; "
              "omit when using --from-combined"))
    combined = parser.add_argument_group("combined-CSV source")
    combined.add_argument(
        "--from-combined", dest="combined_dir", default=None,
        help=("directory of AMPSCZ-combined-redcap_*.csv files; measures every "
              "site directly instead of reading an anomaly report"))
    combined.add_argument(
        "--variable", action="append", dest="variables", default=None,
        help="variable to chart; repeat for multiple (combined source only)")
    combined.add_argument(
        "--statistic", choices=tuple(sorted(COMBINED_STATISTICS)),
        default="median",
        help="per-site statistic to compare (default: median)")
    combined.add_argument(
        "--scope", choices=("network", "study", "both"), default="network",
        help=("compare sites within their network, against one study-wide "
              "reference, or both (default: network)"))
    combined.add_argument(
        "--min-n", type=int, default=DEFAULT_MIN_SITE_N,
        help=("minimum usable observations before a site's statistic is "
              f"plotted (default: {DEFAULT_MIN_SITE_N})"))
    combined.add_argument(
        "--reference-min-n", type=int, default=DEFAULT_REFERENCE_MIN_N,
        help=("minimum observations for a site to contribute to the cross-site "
              f"reference (default: {DEFAULT_REFERENCE_MIN_N})"))
    parser.add_argument(
        "--output-dir",
        help=("new or empty output directory "
              "(default: timestamped beside input)"))
    parser.add_argument("--sheet", default=DEFAULT_SHEET,
                        help=f"workbook tab name (default: {DEFAULT_SHEET})")
    parser.add_argument("--network", action="append", dest="networks",
                        help="network to include; repeat for multiple")
    parser.add_argument("--timepoint", action="append", dest="timepoints",
                        help="timepoint to include; repeat for multiple")
    parser.add_argument(
        "--first-rows", type=int, default=DEFAULT_SOURCE_ROWS,
        dest="source_rows",
        help=("select chart variables from the first N usable rows per "
              f"network (default: {DEFAULT_SOURCE_ROWS})"))
    parser.add_argument(
        "--top-k", type=int, dest="legacy_top_k", default=None,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-sites-per-chart", type=int, default=None,
        help=("split each variable's sites across pages of at most N spokes "
              "(default: no split, every site on one page)"))
    parser.add_argument(
        "--site-list", default=None,
        help=("optional roster file adding sites the report never names; one "
              "site per line, 'NETWORK:SITE' to scope one network, '#' "
              "comments ignored"))
    parser.add_argument("--format", choices=("png", "pdf", "both"),
                        default="png", dest="output_format",
                        help="chart output format (default: png)")
    parser.add_argument("--dpi", type=int, default=180,
                        help="PNG resolution (default: 180)")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.combined_dir and args.report:
        parser.error("give either a report or --from-combined, not both")
    if not args.combined_dir and not args.report:
        parser.error("supply a report path, or --from-combined DIR")
    if args.combined_dir and not args.variables:
        parser.error("--from-combined requires at least one --variable")
    if args.variables and not args.combined_dir:
        parser.error("--variable applies to --from-combined only")
    try:
        if args.combined_dir:
            result = generate_radar_charts_from_combined(
                args.combined_dir,
                args.variables,
                output_dir=args.output_dir,
                networks=args.networks,
                timepoints=args.timepoints,
                statistic=args.statistic,
                scope=args.scope,
                min_n=args.min_n,
                reference_min_n=args.reference_min_n,
                site_list=args.site_list,
                max_sites_per_chart=args.max_sites_per_chart,
                output_format=args.output_format,
                dpi=args.dpi,
            )
        else:
            result = generate_radar_charts(
                args.report,
                output_dir=args.output_dir,
                sheet=args.sheet,
                networks=args.networks,
                timepoints=args.timepoints,
                source_rows=(args.legacy_top_k
                             if args.legacy_top_k is not None
                             else args.source_rows),
                max_sites_per_chart=args.max_sites_per_chart,
                site_list=args.site_list,
                output_format=args.output_format,
                dpi=args.dpi,
            )
    except (RadarInputError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.combined_dir:
        print(f"Created {result['chart_count']} one-variable radar chart "
              f"page(s) across {result['variable_count']} chart group(s), "
              f"covering all {result['spoke_site_count']} site spoke(s) "
              f"({result['site_count']} carry a measured statistic).")
    else:
        print(f"Created {result['chart_count']} one-variable radar chart "
              f"page(s) across {result['variable_count']} network-variable "
              f"pair(s), covering all {result['spoke_site_count']} site "
              f"spoke(s) ({result['site_count']} site profile(s) carry "
              "recorded evidence).")
    print(f"Output: {result['output_dir'].resolve()}")
    if result["warnings"]:
        print(f"Warnings: {len(result['warnings'])} (see README.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
