"""Standalone flag-distribution dashboard for PRONET / PRESCIENT QC outputs.

Point it at the QC output file(s) for one or both networks and it writes a
self-contained HTML dashboard showing:

  1. the distribution of flags by form (stacked by network),
  2. within the top N forms (default 5), the distribution of the individual
     flag types raised on that form, and
  3. SCID flags referencing inputs to outcome_calculations.py versus other
     SCID flags, stacked by network and counted once per flag entry.

Usage
-----
    py -3.12 analyze_flags/flag_distribution_dashboard.py \
        --pronet  "<path to PRONET tracker.xlsx>" \
        --prescient "<path to PRESCIENT tracker.xlsx>" \
        --out flag_distribution.html

Either network may be omitted.  ``.xlsx``/``.xlsm``, ``.csv`` and ``.parquet``
inputs are all accepted.

Resolved and manually resolved rows are excluded by default, using the Date
Resolved, Manually Resolved and Currently Resolved columns (or their raw
pipeline spellings). Pass ``--include-resolved`` to count them as well.

Scope: Main Report only
-----------------------
Only the Main Report surface is counted.  In a workbook that means the sheet
named "Main Report" and nothing else -- Secondary Report, Date Report, Cross
Checks, Proposed Checks, Medication Flags and every other tab are skipped and
recorded in the load statistics.  In a sheetless input (the raw combined handoff as
``.csv``/``.parquet``) it means the rows whose ``reports`` cell carries the
'Main Report' token, split on ' | ' exactly as create_trackers._report_token_mask
does, so a report name that contains another cannot route a row into this
count.  A sheetless input carrying no ``reports`` column cannot be narrowed and
is taken as given -- the right answer for a Main Report sheet already extracted
to CSV -- and the load statistics record the files taken that way.

This is a *row* filter, not an entry filter.  create_trackers routes whole rows
to tabs and never narrows the ``error_message`` cell per tab (see
_select_prescient_main_report_rows), so a row routed to both Main Report and
Date Report carries its date entries onto Main with it.  Pass ``--all-sheets``
to restore the previous behaviour of counting every usable sheet.

Comparing the two networks' bars needs care under this scope: create_trackers
writes only the Medication Flags tab into the PRONET workbook (see
PRONET_REPORT_NAMES), so a PRONET Main Report sheet is legacy operator-owned
content, whereas PRESCIENT's Main Report is pipeline-written with eligible Date
Report rows folded into it.

Each network takes one output normally -- pass several paths only when they
hold *disjoint* flags.  Nothing here deduplicates, so a combined tracker plus
the per-site workbooks covering the same flags would double-count them; the
per-file entry counts printed to the console are there to make that visible.

Counting unit
-------------
The unit is one *flag entry*, not one tracker row.  A tracker row is a
(Participant, Timepoint, Form) group whose ``Flags`` cell holds many
``variable : message`` entries joined by " | ", so the per-form totals and the
per-type breakdowns are both built by exploding that cell with
``analyze_flags.canonicalize.explode_specific_flags``.  Counting the same unit
on both charts is what makes each form's type breakdown sum exactly to that
form's bar.

The workbook's own ``Flag Count`` column is *not* the source of truth here:
create_trackers.py computes it as a naive ``error_message.str.count('|') + 1``,
which over-counts messages containing raw pipes -- precisely what the shared
boundary regex in canonicalize.py exists to correct. Its sum is retained in
the load statistics as a cross-check only.

A "flag type" is the canonical message template (``canonicalize_message``):
the message with dates, numbers, parenthesised values, ranges, subject IDs and
timepoints collapsed to placeholders, so the same check against different
subjects and values counts as one type.

Column vocabularies
-------------------
Both spellings are accepted, so the tracker workbooks (display names ``Form`` /
``Flags``, per dependencies/current_col_names.json) and the raw combined
handoff (``displayed_form`` / ``error_message``) work interchangeably.  Sheets
carrying neither pair are skipped with a note rather than failing the run.

Output is aggregate-only by construction -- counts by form, type and network.
No participant identifier reaches the HTML or the backing CSVs.
"""

import argparse
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)

from analyze_flags.canonicalize import (  # noqa: E402
    explode_specific_flags,
    normalize_form_name,
)

# ---------------------------------------------------------------------------
# Input vocabulary
# ---------------------------------------------------------------------------

# Display spelling first, raw pipeline spelling second. Columns are always
# selected by name -- Cross Checks sheets carry an extra `Check ID` column, so
# positional selection would silently read the wrong field.
FORM_COLUMNS = ('Form', 'displayed_form')
FLAGS_COLUMNS = ('Flags', 'error_message')
DATE_RESOLVED_COLUMNS = ('Date Resolved', 'date_resolved')
MANUALLY_RESOLVED_COLUMNS = ('Manually Resolved', 'manually_resolved')
CURRENTLY_RESOLVED_COLUMNS = ('Currently Resolved', 'currently_resolved')
FLAG_COUNT_COLUMNS = ('Flag Count', 'flag_count')
# One spelling only: report routing lives in the raw handoff's `reports`
# column and is never carried into a workbook's display columns (it would be
# redundant there -- the tab *is* the routing), so there is no display
# vocabulary to pair it with.
REPORTS_COLUMNS = ('reports',)

# The one surface this dashboard counts. 'Main Report' is a shared literal
# across the pipeline (create_trackers.all_reports, KNOWN_REPORT_TABS), which
# is why it is matched by name here rather than by position or by prefix.
MAIN_REPORT_NAME = 'Main Report'
# Report membership in the raw handoff is a pipe-delimited token list, and
# create_trackers._report_token_mask splits on this exact separator. Substring
# matching would route a row whose report name merely contains another's.
REPORT_TOKEN_SEP = ' | '

EXCEL_SUFFIXES = ('.xlsx', '.xlsm', '.xltx', '.xltm')

# Resolution is read as "not falsy" rather than "not empty", because the two
# column vocabularies disagree about what an unresolved row looks like. In the
# display vocabulary an unresolved `Manually Resolved` cell is empty (see
# create_trackers.move_rows_to_bottom, which tests `!= ''`); in the raw
# vocabulary `manually_resolved` / `currently_resolved` are booleans, so under
# `dtype=str` an unresolved row reads as the non-empty string 'False'. Testing
# emptiness alone would mark every raw row resolved and empty out --open-only.
FALSY = frozenset(['', 'false', '0', 'no', 'f', 'nan', 'none', 'nat'])

BLANK_LABEL = '<blank form>'
OTHER_LABEL = 'Other flag types'

# Canonical templates all start with the constant '<var> : ' prefix (see
# canonicalize_message); it carries no information in a per-type label.
TEMPLATE_PREFIX = '<var> : '
SNAKE_RE = re.compile(r'[_\-]+')


# Maintained list of direct SCID inputs to outcome_calculations.py. Update this
# dependency when outcome inputs change; the dashboard does not parse that file.
SCID_OUTCOME_VARIABLES_FILENAME = 'scid_outcome_variables.json'
SCID_FORM_KEYS = frozenset(map(normalize_form_name, (
    'scid5_psychosis_mood_substance_abuse',
    'scid5_schizotypal_personality_sciddpq',
)))
_IDENTIFIER_RE = re.compile(r'\b[a-zA-Z][a-zA-Z0-9_]*\b')


def load_scid_outcome_variables(path=None):
    """Load SCID inputs from config.json's paths.dependencies_path directory."""
    if path is None:
        config_path = os.path.join(parent_dir, 'config.json')
        with open(config_path, encoding='utf-8') as handle:
            config = json.load(handle)
        try:
            dependencies_path = config['paths']['dependencies_path']
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f'{config_path}: paths.dependencies_path must be configured') from exc
        if not isinstance(dependencies_path, str) or not dependencies_path.strip():
            raise ValueError(
                f'{config_path}: paths.dependencies_path must be a non-empty string')
        # Absolute paths override the config directory; relative paths stay
        # anchored to that directory regardless of the invoking working directory.
        path = os.path.join(
            os.path.dirname(config_path), dependencies_path,
            SCID_OUTCOME_VARIABLES_FILENAME)
    with open(path, encoding='utf-8') as handle:
        variables = json.load(handle)
    if (not isinstance(variables, list) or not variables
            or any(not isinstance(field, str)
                   or not field.startswith('chrscid_')
                   or not field.isidentifier() for field in variables)):
        raise ValueError(f'{path}: expected a non-empty JSON list of SCID variable names')
    return frozenset(variables)


def scid_outcome_matches(variable, message, outcome_variables):
    """Match input identifiers in the flag field or its original message.

    A flag on a checkbox base field also involves the selected input choice.
    A different explicit choice does not: ``field___2`` must not match
    ``field___1``. A set avoids counting repeated mentions more than once.
    """
    tokens = set(_IDENTIFIER_RE.findall(f'{variable} {message}'.lower()))
    return frozenset(
        field for field in outcome_variables
        if field in tokens or ('___' in field and field.split('___', 1)[0] in tokens)
    )


def is_main_report_sheet(sheet_name):
    """Whether a workbook sheet is the Main Report tab.

    Compared case-insensitively after stripping, because Excel sheet names
    pick up trailing spaces that never show in the tab.
    """
    return str(sheet_name).strip().casefold() == MAIN_REPORT_NAME.casefold()


def main_report_row_mask(values):
    """Rows whose `reports` cell routes them to Main Report.

    Mirrors create_trackers._report_token_mask: exact token equality after
    splitting on REPORT_TOKEN_SEP, never a substring test.
    """
    return values.fillna('').astype(str).map(
        lambda value: MAIN_REPORT_NAME in value.split(REPORT_TOKEN_SEP))


def first_present(columns, candidates):
    """Return the first candidate name present in `columns`, else None."""
    for name in candidates:
        if name in columns:
            return name
    return None


def pretty_form_label(raw):
    """Human label for a form name, preserving the source spelling's words.

    Unspaced spellings (the V2 vocabulary -- 'sofas_screening', 'cdss')
    become Title Case; already-spaced spellings (the V1 vocabulary) are left
    as typed so an intentional acronym like 'SOFAS Screening' is not
    flattened to 'Sofas Screening'.
    """
    raw = str(raw).strip()
    if not raw:
        return BLANK_LABEL
    if ' ' not in raw:
        return SNAKE_RE.sub(' ', raw).title()
    return raw


def pretty_type_label(template):
    """Strip the constant '<var> : ' prefix from a canonical template."""
    text = str(template)
    if text.startswith(TEMPLATE_PREFIX):
        text = text[len(TEMPLATE_PREFIX):]
    return text.strip() or '(empty message)'


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_tables(path):
    """Yield (sheet_label, DataFrame) for one input file.

    Every sheet of a workbook is yielded -- deployed trackers carry Main
    Report, Medication Flags, Date Report, Cross Checks and friends. Selecting
    among them is left to `load_entries`, which needs to see the rejected tabs
    in order to record them in the load statistics. A sheetless input (.csv/.parquet)
    yields one frame under the empty label.
    """
    suffix = os.path.splitext(path)[1].lower()
    if suffix in EXCEL_SUFFIXES:
        sheets = pd.read_excel(
            path, sheet_name=None, dtype=str, keep_default_na=False)
        for sheet_name, frame in sheets.items():
            yield str(sheet_name), frame
    elif suffix == '.parquet':
        yield '', pd.read_parquet(path)
    else:
        yield '', pd.read_csv(
            path, dtype=str, keep_default_na=False, low_memory=False)


def resolved_mask(frame):
    """Boolean mask of rows the tracker already considers resolved.

    Resolution is spread across three optional columns; a file carrying none
    of them (a raw flag dump) yields an all-False mask, which is the honest
    answer -- nothing in it is marked resolved. Each column is read against
    FALSY, so an empty display cell and a raw boolean 'False' both count as
    unresolved while a real date or 'yes' counts as resolved.
    """
    mask = pd.Series(False, index=frame.index)
    for candidates in (DATE_RESOLVED_COLUMNS, MANUALLY_RESOLVED_COLUMNS,
                       CURRENTLY_RESOLVED_COLUMNS):
        column = first_present(frame.columns, candidates)
        if column is not None:
            values = frame[column].astype(str).str.strip().str.lower()
            mask |= ~values.isin(FALSY)
    return mask


class LoadStats():
    """Per-network tally of what was read, skipped and dropped.

    Kept beside the counts to record input coverage and parsing diagnostics.
    """

    def __init__(self):
        self.files = []
        self.skipped_sheets = []
        self.unfiltered_files = []
        self.rows_read = 0
        self.rows_not_main = 0
        self.rows_resolved = 0
        self.rows_used = 0
        self.rows_without_entries = 0
        self.entries = 0
        self.declared_flag_count = 0
        self.declared_flag_count_available = False


def load_entries(paths, network, stats, open_only=True, main_report_only=True):
    """Explode every usable Main Report surface of every path into records.

    Returns (form_key, form_label, canonical_template, variable, message)
    tuples. Original messages are retained only for outcome-input matching;
    aggregation never forwards them to the HTML or backing CSVs.

    Resolved and manually resolved rows are excluded unless `open_only=False`.

    Under `main_report_only` a workbook contributes only its 'Main Report'
    sheet, and a sheetless input only the rows its `reports` cell routes to
    Main Report. A sheetless input without that column cannot be narrowed, so
    it is used whole and recorded in `stats.unfiltered_files` -- the honest
    outcome for an already-extracted Main Report CSV, and recorded in the
    load statistics for anything else. Setting it False restores the old behaviour of
    counting every sheet that carries the column pair.
    """
    records = []
    for path in paths:
        if not os.path.exists(path):
            raise SystemExit(f"{network}: input not found: {path}")
        used_any = False
        before = len(records)
        for sheet_name, frame in read_tables(path):
            location = os.path.basename(path)
            if sheet_name:
                location += f" [{sheet_name}]"
            if main_report_only and sheet_name and not is_main_report_sheet(
                    sheet_name):
                stats.skipped_sheets.append(
                    f"{location} (not {MAIN_REPORT_NAME})")
                continue
            form_col = first_present(frame.columns, FORM_COLUMNS)
            flags_col = first_present(frame.columns, FLAGS_COLUMNS)
            if form_col is None or flags_col is None:
                stats.skipped_sheets.append(f"{location} (no Form/Flags pair)")
                continue
            used_any = True
            stats.rows_read += len(frame)

            # A sheetless input is the raw handoff, where the Main Report
            # surface is a row property rather than a tab. Narrow before the
            # resolution tally so every count below describes Main rows only.
            if main_report_only and not sheet_name:
                reports_col = first_present(frame.columns, REPORTS_COLUMNS)
                if reports_col is None:
                    if location not in stats.unfiltered_files:
                        stats.unfiltered_files.append(location)
                else:
                    on_main = main_report_row_mask(frame[reports_col])
                    stats.rows_not_main += int((~on_main).sum())
                    frame = frame[on_main]

            resolved = resolved_mask(frame)
            stats.rows_resolved += int(resolved.sum())
            if open_only:
                frame = frame[~resolved]

            count_col = first_present(frame.columns, FLAG_COUNT_COLUMNS)
            if count_col is not None:
                stats.declared_flag_count_available = True
                stats.declared_flag_count += int(
                    pd.to_numeric(frame[count_col], errors='coerce')
                    .fillna(0).sum())

            stats.rows_used += len(frame)
            for raw_form, raw_flags in zip(frame[form_col].astype(str),
                                           frame[flags_col].astype(str)):
                form_key = normalize_form_name(raw_form)
                form_label = pretty_form_label(raw_form)
                row_entries = 0
                for variable, message, template in explode_specific_flags(
                        raw_flags):
                    records.append((form_key, form_label, template,
                                    variable, message))
                    row_entries += 1
                if not row_entries:
                    stats.rows_without_entries += 1
        if used_any:
            stats.files.append(path)
            # Printed per file, not just per network: nothing deduplicates
            # across paths, so a doubled total is only visible here.
            print(f"  {network}: {os.path.basename(path)} -> "
                  f"{len(records) - before:,} flag entries")
        else:
            reason = (f"no usable {MAIN_REPORT_NAME} sheet"
                      if main_report_only else 'no usable sheet')
            stats.skipped_sheets.append(
                f"{os.path.basename(path)} ({reason})")
    stats.entries = len(records)
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class Aggregate():
    """Flag-entry counts keyed by normalized form, split by network.

    Forms are keyed on normalize_form_name so PRONET's 'sofas_screening' and
    a V1 'SOFAS Screening' land in one bar; the display label is the most
    frequent raw spelling seen for that key.
    """

    def __init__(self, networks):
        self.networks = list(networks)
        self.form_totals = Counter()
        self.form_by_network = defaultdict(Counter)
        self.form_labels = defaultdict(Counter)
        self.type_by_form = defaultdict(Counter)
        self.type_by_form_network = defaultdict(lambda: defaultdict(Counter))
        self.network_totals = Counter()
        self.outcome_variables = load_scid_outcome_variables()
        self.scid_by_network = Counter()
        self.scid_outcome_by_network = Counter()

    def add(self, network, records):
        for form_key, form_label, template, variable, message in records:
            self.form_totals[form_key] += 1
            self.form_by_network[form_key][network] += 1
            self.form_labels[form_key][form_label] += 1
            self.type_by_form[form_key][template] += 1
            self.type_by_form_network[form_key][template][network] += 1
            self.network_totals[network] += 1
            if form_key in SCID_FORM_KEYS:
                self.scid_by_network[network] += 1
                if scid_outcome_matches(variable, message, self.outcome_variables):
                    self.scid_outcome_by_network[network] += 1

    def scid_outcome_rows(self):
        """Two disjoint groups whose counts sum to all included SCID flags."""
        involved = dict(self.scid_outcome_by_network)
        other = {network: self.scid_by_network[network] - involved.get(network, 0)
                 for network in self.networks}
        return [
            ('Involving outcome variables', involved, sum(involved.values())),
            ('Other SCID flags', other, sum(other.values())),
        ]

    def label(self, form_key):
        labels = self.form_labels.get(form_key)
        if not labels:
            return BLANK_LABEL
        return labels.most_common(1)[0][0]

    @property
    def total(self):
        return sum(self.form_totals.values())

    def forms_ranked(self):
        """Form keys, most flags first; ties broken by label for stability."""
        return sorted(
            self.form_totals,
            key=lambda key: (-self.form_totals[key], self.label(key).lower()))

    def types_for_form(self, form_key, top_types):
        """(label, {network: count}, total, tooltip) rows for one form.

        Everything past `top_types` collapses into one explicit Other bucket,
        so the rows still sum exactly to the form's total.
        """
        counts = self.type_by_form[form_key]
        ranked = sorted(counts, key=lambda t: (-counts[t], t))
        rows = []
        for template in ranked[:top_types]:
            label = pretty_type_label(template)
            rows.append((label, dict(self.type_by_form_network[form_key][template]),
                         counts[template], label))
        tail = ranked[top_types:]
        if tail:
            per_network = Counter()
            tail_total = 0
            for template in tail:
                per_network.update(
                    self.type_by_form_network[form_key][template])
                tail_total += counts[template]
            label = f"{OTHER_LABEL} ({len(tail):,})"
            rows.append((label, dict(per_network), tail_total,
                         f"{len(tail):,} further flag types on this form"))
        return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# Categorical slots 1 and 2 of the reference palette, light and dark steps.
# Validated as a 2-slot categorical palette in both modes (all six checks
# PASS; worst adjacent CVD dE 24.7 light / 26.8 dark).
SERIES_COLORS = {
    'PRONET': ('#2a78d6', '#3987e5'),
    'PRESCIENT': ('#eb6834', '#d95926'),
}
FALLBACK_COLORS = [('#1baf7a', '#199e70'), ('#eda100', '#c98500')]

CSS = """
:root {
  color-scheme: light;
  --plane: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --baseline: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10);
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --series-4: #eda100;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --plane: #0d0d0d;
    --surface: #1a1a19;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --baseline: #383835;
    --border: rgba(255, 255, 255, 0.10);
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --series-4: #c98500;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --plane: #0d0d0d;
  --surface: #1a1a19;
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --muted: #898781;
  --grid: #2c2c2a;
  --baseline: #383835;
  --border: rgba(255, 255, 255, 0.10);
  --series-1: #3987e5;
  --series-2: #d95926;
  --series-3: #199e70;
  --series-4: #c98500;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 28px 64px;
  background: var(--plane);
  color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1180px; margin: 0 auto; }

header.page { display: flex; align-items: flex-start; gap: 16px; margin-bottom: 4px; }
header.page h1 { font-size: 22px; font-weight: 600; margin: 0; letter-spacing: -0.01em; }
.sub { color: var(--ink-2); margin: 6px 0 0; }
.sub code { font-size: 12px; color: var(--muted); }
.theme-toggle {
  margin-left: auto; flex: none; cursor: pointer;
  background: var(--surface); color: var(--ink-2);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 12px; font: inherit; font-size: 12px;
}
.theme-toggle:hover { color: var(--ink); }

.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin: 24px 0 8px; }
.tile {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 14px 18px; min-width: 150px; flex: 1 1 150px;
}
.tile .label { color: var(--ink-2); font-size: 12px; }
.tile .value { font-size: 26px; font-weight: 600; margin-top: 2px; }
.tile.hero { flex: 1 1 240px; }
.tile.hero .value { font-size: 48px; line-height: 1.05; font-weight: 600; }
.tile .value .unit { font-size: 13px; font-weight: 400; color: var(--muted); margin-left: 6px; }

section.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 22px 24px; margin-top: 20px;
}
section.card > h2 { font-size: 16px; font-weight: 600; margin: 0; }
section.card > .note { color: var(--ink-2); margin: 6px 0 0; font-size: 13px; }

.legend { display: flex; gap: 18px; flex-wrap: wrap; margin: 16px 0 4px; }
.legend .item { display: flex; align-items: center; gap: 7px; color: var(--ink-2); font-size: 13px; }
.swatch { width: 12px; height: 12px; border-radius: 3px; flex: none; }

.chart { margin-top: 14px; }
.row { display: flex; align-items: center; gap: 12px; padding: 3px 0; }
.row .name {
  width: 260px; flex: none; color: var(--ink-2);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.row.wide .name { width: 420px; }
.row .track {
  flex: 1 1 auto; min-width: 40px;
  border-left: 1px solid var(--baseline);
  padding-left: 1px;
}
.row .bar { display: flex; gap: 2px; height: 20px; }
.row .seg { min-width: 2px; height: 100%; }
.row .seg:last-child { border-radius: 0 4px 4px 0; }
.row .val {
  width: 108px; flex: none; text-align: right;
  color: var(--ink); font-variant-numeric: tabular-nums; font-size: 13px;
}
.row .val .pct { color: var(--muted); margin-left: 6px; font-size: 12px; }
.row:hover .name { color: var(--ink); }

.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 18px; margin-top: 16px; }
.panel { border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; }
.panel h3 { font-size: 14px; font-weight: 600; margin: 0 0 2px; }
.panel .rank { color: var(--muted); font-weight: 400; margin-right: 6px; }
.panel .meta { color: var(--ink-2); font-size: 12px; margin: 0 0 10px; }
.panel .row .name { width: 250px; font-size: 13px; }
.panel .row .val { width: 96px; }

details.table { margin-top: 18px; }
details.table summary { cursor: pointer; color: var(--ink-2); font-size: 13px; }
details.table summary:hover { color: var(--ink); }
table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); }
th { color: var(--ink-2); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.tablewrap { overflow-x: auto; }


#tip {
  position: fixed; z-index: 20; pointer-events: none; opacity: 0;
  transition: opacity 90ms ease; max-width: 460px;
  background: var(--surface); color: var(--ink);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 11px; font-size: 12px; line-height: 1.45;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.18);
}
#tip .tip-title { font-weight: 600; margin-bottom: 3px; }
#tip .tip-line { color: var(--ink-2); }
"""

JS = """
(function () {
  var root = document.documentElement;
  var btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.addEventListener('click', function () {
      var dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      var current = root.getAttribute('data-theme') || (dark ? 'dark' : 'light');
      root.setAttribute('data-theme', current === 'dark' ? 'light' : 'dark');
    });
  }
  var tip = document.getElementById('tip');
  function show(e) {
    var el = e.target.closest('[data-tip]');
    if (!el) { return; }
    tip.innerHTML = '<div class="tip-title"></div><div class="tip-line"></div>';
    tip.firstChild.textContent = el.getAttribute('data-tip');
    tip.lastChild.textContent = el.getAttribute('data-tip-line') || '';
    tip.style.opacity = '1';
    move(e);
  }
  function move(e) {
    var pad = 14;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var x = e.clientX + pad, y = e.clientY + pad;
    if (x + w > window.innerWidth - 8) { x = e.clientX - w - pad; }
    if (y + h > window.innerHeight - 8) { y = e.clientY - h - pad; }
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }
  document.addEventListener('mouseover', show);
  document.addEventListener('mousemove', function (e) {
    if (tip.style.opacity === '1') { move(e); }
  });
  document.addEventListener('mouseout', function (e) {
    if (!e.relatedTarget || !e.relatedTarget.closest('[data-tip]')) {
      tip.style.opacity = '0';
    }
  });
})();
"""


def esc(text):
    return html.escape(str(text), quote=True)


def fmt(number):
    return f"{number:,}"


def pct(part, whole):
    return f"{(100.0 * part / whole):.1f}%" if whole else "0.0%"


def series_var(index):
    return f"var(--series-{index + 1})"


def bar_row(label, per_network, total, max_total, networks, grand_total,
            tooltip=None, wide=False):
    """One stacked horizontal bar: label, bar, value at the tip (outside).

    The value sits outside the bar in its own column so it can never be
    clipped by a short bar, and interior segments carry the tooltip rather
    than an inline label they have no room for.
    """
    width = (100.0 * total / max_total) if max_total else 0.0
    segments = []
    for index, network in enumerate(networks):
        count = per_network.get(network, 0)
        if not count:
            continue
        line = (f"{network}: {fmt(count)} "
                f"({pct(count, total)} of this bar)")
        segments.append(
            f'<span class="seg" style="flex:{count} 1 0;'
            f'background:{series_var(index)}" '
            f'data-tip="{esc(tooltip or label)}" '
            f'data-tip-line="{esc(line)}"></span>')
    row_class = 'row wide' if wide else 'row'
    return (
        f'<div class="{row_class}">'
        f'<div class="name" data-tip="{esc(tooltip or label)}" '
        f'data-tip-line="{esc(fmt(total))} flags">{esc(label)}</div>'
        f'<div class="track"><div class="bar" style="width:{width:.4f}%">'
        f'{"".join(segments)}</div></div>'
        f'<div class="val">{fmt(total)}'
        f'<span class="pct">{pct(total, grand_total)}</span></div>'
        '</div>')


def legend_html(networks):
    """A legend is present whenever two or more series share a chart."""
    if len(networks) < 2:
        return ''
    items = ''.join(
        f'<div class="item"><span class="swatch" '
        f'style="background:{series_var(index)}"></span>{esc(network)}</div>'
        for index, network in enumerate(networks))
    return f'<div class="legend">{items}</div>'


def table_html(headers, rows):
    head = ''.join(
        f'<th class="num">{esc(h)}</th>' if i else f'<th>{esc(h)}</th>'
        for i, h in enumerate(headers))
    body = ''
    for row in rows:
        cells = ''.join(
            f'<td class="num">{esc(c)}</td>' if i else f'<td>{esc(c)}</td>'
            for i, c in enumerate(row))
        body += f'<tr>{cells}</tr>'
    return (f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def render_scid_outcome_section(agg, networks):
    """Show the share of SCID flag entries referencing outcome inputs."""
    rows = agg.scid_outcome_rows()
    scid_total = sum(agg.scid_by_network.values())
    outcome_total = sum(agg.scid_outcome_by_network.values())
    max_total = max(count for _label, _counts, count in rows)
    bars = ''.join(
        bar_row(label, counts, count, max_total, networks, scid_total, wide=True)
        for label, counts, count in rows)
    table = table_html(
        ['SCID flag group'] + list(networks) + ['Total', '% of SCID flags'],
        [[label] + [fmt(counts.get(network, 0)) for network in networks]
         + [fmt(count), pct(count, scid_total)]
         for label, counts, count in rows])
    variables = ', '.join(f'<code>{esc(field)}</code>'
                          for field in sorted(agg.outcome_variables))
    empty_note = ('<p class="note">No SCID flag entries in the selected '
                  'input scope.</p>' if not scid_total else '')
    return (
        '<section class="card" id="scid-outcome-flags">'
        '<h2>SCID flags involving outcome variables</h2>'
        f'<p class="note"><b>{fmt(outcome_total)} of {fmt(scid_total)} SCID '
        f'flag entries ({pct(outcome_total, scid_total)})</b> reference a '
        'variable used in the outcome calculations. Each flag is counted '
        'once, even when it references multiple outcome variables. '
        'Percentages are of SCID flags; segments split counts by network.</p>'
        f'{empty_note}{legend_html(networks)}'
        f'<div class="chart">{bars}</div>'
        '<p class="note">Matches the flagged field or an exact variable name '
        'in its message against the SCID inputs listed in '
        '<code>scid_outcome_variables.json</code> '
        '(from <code>outcome_calculations.py</code>), including screening gates and '
        'checkbox parent fields. Includes the SCID psychosis/mood/substance '
        'and schizotypal personality forms, using the same report and '
        'resolution filters as the charts above. This measures references '
        'to outcome inputs, not confirmed changes in calculated outcomes.</p>'
        '<details class="table"><summary>Table view &mdash; SCID outcome '
        f'flags</summary>{table}</details>'
        '<details class="table"><summary>Outcome input variables '
        f'({len(agg.outcome_variables)})</summary><p>{variables}</p></details>'
        '</section>')


def render_html(agg, networks, stats_by_network, top_forms, top_types,
                open_only, title, main_report_only=True):
    total = agg.total
    ranked = agg.forms_ranked()
    max_form_total = agg.form_totals[ranked[0]] if ranked else 0
    distinct_types = len({
        template
        for counts in agg.type_by_form.values()
        for template in counts})
    view_label = ('open (unresolved) rows only' if open_only
                  else 'every row in the input, resolved included')
    scope_label = (f'{MAIN_REPORT_NAME} only' if main_report_only
                   else 'every usable sheet')

    # --- tiles -------------------------------------------------------------
    tiles = [
        '<div class="tile hero"><div class="label">Flag entries</div>'
        f'<div class="value">{fmt(total)}'
        f'<span class="unit">across {fmt(len(ranked))} forms</span></div></div>']
    for index, network in enumerate(networks):
        count = agg.network_totals.get(network, 0)
        tiles.append(
            f'<div class="tile"><div class="label">'
            f'<span class="swatch" style="display:inline-block;vertical-align:-1px;'
            f'margin-right:6px;background:{series_var(index)}"></span>'
            f'{esc(network)}</div>'
            f'<div class="value">{fmt(count)}'
            f'<span class="unit">{pct(count, total)}</span></div></div>')
    tiles.append(
        '<div class="tile"><div class="label">Distinct flag types</div>'
        f'<div class="value">{fmt(distinct_types)}</div></div>')
    rows_used = sum(s.rows_used for s in stats_by_network.values())
    tiles.append(
        '<div class="tile"><div class="label">Tracker rows</div>'
        f'<div class="value">{fmt(rows_used)}</div></div>')

    # --- section 1: all forms ---------------------------------------------
    form_rows = ''.join(
        bar_row(agg.label(key), agg.form_by_network[key],
                agg.form_totals[key], max_form_total, networks, total,
                wide=True)
        for key in ranked)
    form_table = table_html(
        ['Form'] + list(networks) + ['Total', '% of all flags'],
        [[agg.label(key)]
         + [fmt(agg.form_by_network[key].get(n, 0)) for n in networks]
         + [fmt(agg.form_totals[key]), pct(agg.form_totals[key], total)]
         for key in ranked])

    section_forms = (
        '<section class="card">'
        '<h2>Flag entries by form</h2>'
        f'<p class="note">All {fmt(len(ranked))} forms, most-flagged first. '
        'Each bar is one form; segments split it by network.</p>'
        f'{legend_html(networks)}'
        f'<div class="chart">{form_rows}</div>'
        '<details class="table"><summary>Table view &mdash; flags by form</summary>'
        f'{form_table}</details>'
        '</section>')

    # --- section 2: types inside the top forms ----------------------------
    top_keys = ranked[:top_forms]
    panels = []
    type_table_rows = []
    for rank, key in enumerate(top_keys, start=1):
        form_total = agg.form_totals[key]
        rows = agg.types_for_form(key, top_types)
        max_type_total = max((r[2] for r in rows), default=0)
        bars = ''.join(
            bar_row(label, per_network, count, max_type_total, networks,
                    form_total, tooltip=tooltip)
            for label, per_network, count, tooltip in rows)
        shown = len(agg.type_by_form[key])
        panels.append(
            '<div class="panel">'
            f'<h3><span class="rank">{rank}.</span>{esc(agg.label(key))}</h3>'
            f'<p class="meta">{fmt(form_total)} flags &middot; '
            f'{fmt(shown)} distinct flag types &middot; '
            f'{pct(form_total, total)} of all flags</p>'
            f'{bars}</div>')
        for label, per_network, count, _tooltip in rows:
            type_table_rows.append(
                [agg.label(key), label]
                + [fmt(per_network.get(n, 0)) for n in networks]
                + [fmt(count), pct(count, form_total)])

    type_table = table_html(
        ['Form', 'Flag type'] + list(networks) + ['Count', '% of form'],
        type_table_rows)
    section_types = (
        '<section class="card">'
        f'<h2>Flag types within the top {len(top_keys)} forms</h2>'
        f'<p class="note">Up to {top_types} types per form; everything past '
        'that is pooled into one Other bucket, so each panel still sums to '
        'its form total. Percentages are of that form. Hover a bar for the '
        'full message template.</p>'
        f'{legend_html(networks)}'
        f'<div class="grid2">{"".join(panels)}</div>'
        '<details class="table"><summary>Table view &mdash; flag types by form'
        f'</summary>{type_table}</details>'
        '</section>')

    return (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{esc(title)}</title>\n'
        f'<style>{CSS}</style>\n</head>\n<body>\n<div class="wrap">\n'
        '<header class="page"><div>'
        f'<h1>{esc(title)}</h1>'
        f'<p class="sub">{esc(" and ".join(networks))} QC output &mdash; '
        f'{esc(scope_label)}, {esc(view_label)}</p>'
        '</div>'
        '<button class="theme-toggle" id="theme-toggle" type="button">'
        'Toggle theme</button></header>\n'
        f'<div class="tiles">{"".join(tiles)}</div>\n'
        f'{section_forms}\n{section_types}\n'
        f'{render_scid_outcome_section(agg, networks)}\n'
        '</div>\n<div id="tip" role="tooltip"></div>\n'
        f'<script>{JS}</script>\n</body>\n</html>\n')


# ---------------------------------------------------------------------------
# Backing CSVs
# ---------------------------------------------------------------------------

def write_backing_csvs(agg, networks, top_forms, top_types, out_html):
    """Write the three aggregate tables behind the charts, beside the HTML.

    Matches the repo convention of shipping a chart with the CSV it was drawn
    from, so the numbers can be checked without re-running the parse.
    """
    stem = os.path.splitext(out_html)[0]
    ranked = agg.forms_ranked()
    total = agg.total

    by_form = pd.DataFrame([
        dict([('form', agg.label(key))]
             + [(network, agg.form_by_network[key].get(network, 0))
                for network in networks]
             + [('total', agg.form_totals[key]),
                ('pct_of_all_flags', round(100.0 * agg.form_totals[key] / total, 4)
                 if total else 0.0)])
        for key in ranked])
    by_form_path = f"{stem}_by_form.csv"
    by_form.to_csv(by_form_path, index=False)

    type_rows = []
    for key in ranked[:top_forms]:
        form_total = agg.form_totals[key]
        for label, per_network, count, _tooltip in agg.types_for_form(
                key, top_types):
            type_rows.append(dict(
                [('form', agg.label(key)), ('flag_type', label)]
                + [(network, per_network.get(network, 0))
                   for network in networks]
                + [('count', count),
                   ('pct_of_form', round(100.0 * count / form_total, 4)
                    if form_total else 0.0)]))
    by_type_path = f"{stem}_top_form_flag_types.csv"
    pd.DataFrame(type_rows).to_csv(by_type_path, index=False)
    scid_total = sum(agg.scid_by_network.values())
    scid_rows = [
        dict([('group', label)]
             + [(network, counts.get(network, 0)) for network in networks]
             + [('total', count),
                ('pct_of_scid_flags', round(100.0 * count / scid_total, 4)
                 if scid_total else 0.0)])
        for label, counts, count in agg.scid_outcome_rows()]
    scid_path = f"{stem}_scid_outcome_flags.csv"
    pd.DataFrame(scid_rows).to_csv(scid_path, index=False)
    return by_form_path, by_type_path, scid_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=('Build an HTML dashboard of the flag distribution by '
                     'form, flag types within the top forms, and SCID flags '
                     'involving outcome variables, from '
                     'PRONET / PRESCIENT QC output files.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--pronet', nargs='+', default=[], metavar='PATH',
        help=('PRONET QC output file(s): .xlsx (Main Report sheet), .csv or '
              '.parquet'))
    parser.add_argument(
        '--prescient', nargs='+', default=[], metavar='PATH',
        help='PRESCIENT QC output file(s)')
    parser.add_argument(
        '--out', default='flag_distribution_dashboard.html',
        help='Path of the HTML dashboard to write')
    parser.add_argument(
        '--top-forms', type=int, default=5,
        help='How many forms get a flag-type breakdown panel')
    parser.add_argument(
        '--top-types', type=int, default=8,
        help='How many flag types to show per form before the Other bucket')
    resolution = parser.add_mutually_exclusive_group()
    resolution.add_argument(
        '--open-only', action='store_true', default=True,
        help=('Count only rows with no resolution marker (blank Date Resolved '
              'and no true Manually Resolved or Currently Resolved marker). '
              'This is the default.'))
    resolution.add_argument(
        '--include-resolved', dest='open_only', action='store_false', default=argparse.SUPPRESS,
        help='Count every row, including resolved and manually resolved rows.')
    parser.add_argument(
        '--all-sheets', action='store_true',
        help=('Count every sheet carrying a Form/Flags pair instead of the '
              'Main Report surface alone. Restores the pre-Main-Report-only '
              'behaviour; note that nothing deduplicates across tabs.'))
    parser.add_argument(
        '--title', default='QC flag distribution',
        help='Dashboard heading')
    parser.add_argument(
        '--no-csv', action='store_true',
        help='Skip the three backing CSVs written beside the HTML')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.pronet and not args.prescient:
        raise SystemExit(
            'Give at least one input: --pronet PATH [PATH ...] and/or '
            '--prescient PATH [PATH ...]')
    if args.top_forms < 1 or args.top_types < 1:
        raise SystemExit('--top-forms and --top-types must be at least 1')

    inputs = [('PRONET', args.pronet), ('PRESCIENT', args.prescient)]
    networks = [name for name, paths in inputs if paths]
    try:
        agg = Aggregate(networks)
    except (OSError, ValueError) as exc:
        raise SystemExit(f'Cannot identify SCID outcome inputs: {exc}') from exc
    stats_by_network = {}
    for network, paths in inputs:
        if not paths:
            continue
        stats = LoadStats()
        records = load_entries(paths, network, stats, args.open_only,
                               main_report_only=not args.all_sheets)
        agg.add(network, records)
        stats_by_network[network] = stats
        print(f"{network}: {fmt(stats.rows_used)} rows -> "
              f"{fmt(stats.entries)} flag entries "
              f"across {fmt(len(set(r[0] for r in records)))} forms")

    if not agg.total:
        hint = ('' if args.all_sheets else
                f" that it carries a '{MAIN_REPORT_NAME}' sheet (or a reports"
                ' column routing rows there) or was passed with --all-sheets,')
        resolution_hint = (
            ' Resolved and manually resolved rows are excluded by default; '
            'pass --include-resolved to count them.' if args.open_only else '')
        raise SystemExit(
            'No flag entries found. Check that the input has a Form/Flags '
            f'(or displayed_form/error_message) column pair,{hint} and '
            f'contains flag entries.{resolution_hint}')

    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    markup = render_html(
        agg, networks, stats_by_network, args.top_forms, args.top_types,
        args.open_only, args.title, main_report_only=not args.all_sheets)
    with open(out_path, 'w', encoding='utf-8') as handle:
        handle.write(markup)
    print(f"wrote {out_path}")

    if not args.no_csv:
        for path in write_backing_csvs(
                agg, networks, args.top_forms, args.top_types, out_path):
            print(f"wrote {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
