"""
Standalone analysis script.

Counts how many participants were recruited through each recruitment method
recorded on the `recruitment_source` form, and renders the result as a
horizontal stacked bar chart (PNG) plus Excel workbooks with color-coded rows.
Bars split each method into PRONET and PRESCIENT, with participant counts
for each network shown alongside them.

The method lives in `chrrecruit` ("How was the participant recruited?"), a
15-option dropdown on the screening-timepoint `recruitment_source` form (see
forms_per_timepoint.json -- the form exists for both the CHR and HC cohorts).
The category labels are read straight out of the REDCap data dictionary rather
than hard coded, so a dictionary revision flows through without a code change.

A participant is counted when `chrrecruit` is non-blank. Form completion is
deliberately NOT required: a recruitment source recorded on a form that is
still open is still a known recruitment source. That also keeps the script
clear of the PRESCIENT `_rpms` completion-variable rewrite, since no completion
variable is read on either network.

The `chrrecruit_*` follow-up variables (chrrecruit_physician,
chrrecruit_community, ...) are "please specify" refinements of the method
chosen in `chrrecruit`, not separate methods, so they are not counted here.

    py -3.12 analyze_dataset/graph_recruitment_sources.py
    py -3.12 analyze_dataset/graph_recruitment_sources.py --output-dir some/dir
    py -3.12 analyze_dataset/graph_recruitment_sources.py --networks PRONET

Both networks are counted by default. The QC pipeline's own network scope
(config 'pipeline_networks' / the QC_NETWORKS environment variable) is NOT
inherited: a machine running PRONET-scoped QC would otherwise produce a
half-sized chart that looks complete. Narrow it with --networks on purpose,
never by inheritance.

Outputs (to config paths.output_path unless --output-dir is given):
    recruitment_source_counts.png       the chart
    recruitment_source_counts.xlsx      counts per method, split by network
    recruitment_source_subjects.xlsx    one row per participant, grouped by
                                        method -- carries subject identifiers,
                                        so treat it as a follow-up listing
                                        rather than something to circulate

The per-subject file also lists participants with no recruitment source on
file at all, so it has MORE rows than the chart and the counts workbook:
it covers every screening row, not just the counted ones. Filter counted_in_chart ==
'yes' to get exactly the population behind the bars. That column and
form_marked_missing are written as the text 'yes'/'no', not booleans.
Both workbooks use the same row colors for each recruitment_method.

Every category in the dictionary is reported, including the ones no site has
used yet, so a zero is visibly a zero rather than a missing row. Values that
are neither blank nor a dictionary code are bucketed and printed rather than
dropped -- see the reconciliation block at the end of the run, which is the
check that the counts add up.
"""

import argparse
import colorsys
import os
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# parents[1] is the project root. Spelled with pathlib rather than the
# "/".join(os.path.realpath(__file__).split("/")) idiom in the older scripts
# here, which collapses to an empty path on Windows (realpath returns
# backslashes, so the split never divides) and only appears to work when the
# script happens to be launched from the project root.
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from utils.utils import Utils


# --- chart constants ------------------------------------------------------
# Fixed network colors keep the chart consistent across network scopes.
NETWORK_COLORS = {'PRONET': '#2469b8', 'PRESCIENT': '#b65b14'}
SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'

# The mark specs are given in CSS pixels; a CSS pixel is 1/96 inch, which is
# the unit that survives a change of --dpi.
CSS_PX = 1.0 / 96.0
BAR_MAX_THICKNESS_IN = 24 * CSS_PX   # cap the bar; the band's leftover is air
BAR_CORNER_RADIUS_IN = 4 * CSS_PX    # rounded at the data end, square at the baseline
ROW_PITCH_IN = 0.42                  # vertical space per category
LABEL_WRAP = 40                      # characters per line in the category labels
# Bezier control-point ratio that approximates a quarter circle.
KAPPA = 0.5523

# This is a study-wide description, so both networks are the default and
# the pipeline's own network scope is deliberately not inherited.
DEFAULT_NETWORKS = ('PRONET', 'PRESCIENT')


class GraphRecruitmentSources():

    FORM = 'recruitment_source'
    TIMEPOINT = 'screening'
    SOURCE_VAR = 'chrrecruit'
    MISSING_VAR = 'chrrecruit_missing'
    STATUS_VAR = 'recruitment_status_v2'
    SUBJECT_VAR = 'subjectid'
    PNG_FILENAME = 'recruitment_source_counts.png'
    XLSX_FILENAME = 'recruitment_source_counts.xlsx'
    SUBJECTS_XLSX_FILENAME = 'recruitment_source_subjects.xlsx'
    # Buckets for values that are non-blank but not one of the dictionary's
    # method codes. Kept out of the bars (neither is a recruitment method) and
    # reported separately, so the totals still reconcile.
    MISSING_CODE_KEY = '__missing_code__'
    UNRECOGNIZED_KEY = '__unrecognized__'
    # Only the per-subject file carries this one: a blank chrrecruit is not a
    # category, so it is never counted, but "which subjects have no source on
    # file" is the actionable half of that and belongs in the listing.
    NO_SOURCE_KEY = '__no_source__'
    BUCKET_LABELS = {
        MISSING_CODE_KEY: 'Missing-data code (-9 / -3 / -99 / 999)',
        UNRECOGNIZED_KEY: 'Unrecognized code',
        NO_SOURCE_KEY: 'No recruitment source recorded',
    }

    def __init__(self, output_dir=None, dpi=150, networks=None):
        self.utils = Utils()
        self.config_info = self.utils.config_info
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.output_path = output_dir or self.config_info['paths']['output_path']
        # --output-dir invites a directory that does not exist yet; creating it
        # here keeps the failure out of saving the workbooks/chart at the end.
        os.makedirs(self.output_path, exist_ok=True)
        self.dpi = dpi

        # code -> label, in dictionary order.
        self.choice_labels = self.build_choice_map()
        # Both networks unless explicitly narrowed. utils.pipeline_networks is
        # the QC *pipeline's* operational scope (config 'pipeline_networks' /
        # the QC_NETWORKS variable); a server scoped to one network for its QC
        # run would otherwise silently halve this chart, which is a study-wide
        # description and not a QC run. Use --networks to narrow on purpose.
        self.networks = list(networks) if networks else list(DEFAULT_NETWORKS)
        # code -> per-network participant counts, seeded at zero for every
        # category so an unused method still gets a row and a bar.
        self.counts = {code: {net: 0 for net in self.networks}
                       for code in self.choice_labels}
        self.counts[self.MISSING_CODE_KEY] = {net: 0 for net in self.networks}
        self.counts[self.UNRECOGNIZED_KEY] = {net: 0 for net in self.networks}
        # Counted alongside, so the chart can be qualified by how many of the
        # participants behind it actually reached recruited status.
        self.recruited_counts = {code: 0 for code in self.counts}
        # Diagnostics that make the run auditable without printing any row.
        self.rows_read = {}
        self.nonblank_seen = {}
        self.forms_marked_missing = {}
        self.unrecognized_values = {}
        self.skipped_networks = []
        # One entry per participant row read, for the per-subject listing.
        self.subject_rows = []
        self.png_path = ''
        self.xlsx_path = ''
        self.subjects_xlsx_path = ''

    def run_script(self):
        self.report_inputs()
        self.count_participants()
        self.write_excel()
        self.write_subject_excel()
        self.render_chart()
        self.report()

    def report_inputs(self):
        """
        Echoes what this run is actually reading.

        A chart built from one network looks exactly as finished as a chart
        built from two, so the inputs get stated up front where a wrong
        network scope or a wrong export directory is visible immediately.
        """
        scope_source = ('--networks' if self.networks != list(DEFAULT_NETWORKS)
                        else 'default: both networks')
        print(f'[recruitment_sources] network scope: '
              f'{", ".join(self.networks)}  (from {scope_source})')
        # The pipeline may be scoped to one network on this machine. This
        # script ignores that on purpose, so say so rather than let the two
        # scopes look like one.
        # Only worth saying when the machine's pipeline is actually narrowed:
        # that is the case that would have silently halved this chart, and it
        # explains why this count exceeds what a QC run on the same box sees.
        pipeline_scope = list(self.utils.pipeline_networks)
        if pipeline_scope != list(DEFAULT_NETWORKS):
            print(f'[recruitment_sources]   note: the QC pipeline on this '
                  f'machine is scoped to {", ".join(pipeline_scope)}'
                  + (f" (QC_NETWORKS={os.environ['QC_NETWORKS']})"
                     if os.environ.get('QC_NETWORKS')
                     else " (config.json 'pipeline_networks')"
                     if self.config_info.get('pipeline_networks') else '')
                  + '; this chart does not inherit that scope.')
        print(f'[recruitment_sources] combined CSV directory: '
              f'{self.comb_csv_path}')
        for network in self.networks:
            path = self.combined_csv_path(network)
            print(f'[recruitment_sources]   {network}: {path} '
                  f'{"[found]" if os.path.exists(path) else "[MISSING]"}')

    # --- data dictionary --------------------------------------------------

    def build_choice_map(self):
        """
        Returns {code: label} for chrrecruit, in data dictionary order.

        REDCap stores the options as "code, label | code, label | ...".
        """
        data_dict_df = self.utils.read_data_dictionary()
        matches = data_dict_df[
            (data_dict_df['Variable / Field Name'] == self.SOURCE_VAR)
            & (data_dict_df['Form Name'] == self.FORM)]
        if matches.empty:
            raise RuntimeError(
                f"{self.SOURCE_VAR} is not in the {self.FORM} form of the "
                "current data dictionary; the recruitment-source options "
                "cannot be resolved.")
        raw_choices = str(matches.iloc[0][
            'Choices, Calculations, OR Slider Labels'])

        choice_labels = {}
        for choice in raw_choices.split('|'):
            if ',' not in choice:
                continue
            code, label = choice.split(',', 1)
            choice_labels.setdefault(code.strip(), label.strip())
        if not choice_labels:
            raise RuntimeError(
                f"Could not parse any options out of the {self.SOURCE_VAR} "
                f"choice string: {raw_choices!r}")
        return choice_labels

    # --- counting ---------------------------------------------------------

    def normalize_code(self, value):
        """
        Puts a raw cell into the same shape as the dictionary codes.

        Everything is read as a string, but a value can still reach us as
        '1.0' if an export wrote it that way, and the dictionary codes for
        chrrecruit are all whole numbers.
        """
        text = str(value).strip()
        if re.fullmatch(r'-?\d+\.0+', text):
            text = text.split('.')[0]
        return text

    def combined_csv_path(self, network):
        """
        Resolves one network's screening export, ignoring filename case.

        The canonical name spells PRONET as ProNET (matching qc_forms_main /
        collect_subject_info), but the exports have not always shipped under
        one casing. On Windows that never mattered; on the Linux server it
        decides whether the file is found at all, and a network that is not
        found is a network silently missing from the chart. So the directory
        is matched case-insensitively rather than trusting one spelling.
        """
        canonical = (f'AMPSCZ-combined-redcap_{self.TIMEPOINT}_'
                     f'{network.replace("PRONET", "ProNET")}-day1to1.csv')
        try:
            entries = os.listdir(self.comb_csv_path)
        except OSError:
            return os.path.join(self.comb_csv_path, canonical)
        for name in entries:
            if name.lower() == canonical.lower():
                return os.path.join(self.comb_csv_path, name)
        # Nothing matched; hand back the canonical name so the not-found
        # message names the file that was actually expected.
        return os.path.join(self.comb_csv_path, canonical)

    def read_network_frame(self, network):
        """
        Reads only the columns this script needs out of one screening CSV.

        Returns None (after logging) when the file or the recruitment variable
        is not there, so one absent network cannot abort the whole run.
        """
        csv_path = self.combined_csv_path(network)
        try:
            available = pd.read_csv(csv_path, nrows=0).columns
        except FileNotFoundError:
            print(f"[recruitment_sources] combined CSV not found for "
                  f"{network}: {csv_path}. Skipping this network.")
            # Naming, not absence, is the usual reason a network goes missing,
            # so say what IS in the directory instead of leaving a guess.
            try:
                nearby = sorted(
                    name for name in os.listdir(self.comb_csv_path)
                    if self.TIMEPOINT in name.lower())
            except OSError:
                nearby = []
            print(f"[recruitment_sources]   {self.TIMEPOINT} files present in "
                  f"{self.comb_csv_path}: {nearby if nearby else 'none'}")
            self.skipped_networks.append(network)
            return None

        for required in (self.SUBJECT_VAR, self.SOURCE_VAR):
            if required not in available:
                print(f"[recruitment_sources] {network} {self.TIMEPOINT} CSV "
                      f"has no '{required}' column; skipping this network.")
                self.skipped_networks.append(network)
                return None

        optional = [var for var in (self.MISSING_VAR, self.STATUS_VAR)
                    if var in available]
        for var in (self.MISSING_VAR, self.STATUS_VAR):
            if var not in optional:
                print(f"[recruitment_sources] note: {network} {self.TIMEPOINT} "
                      f"CSV has no '{var}' column; that diagnostic will read 0.")

        # dtype=str keeps a network whose column happens to hold no blanks
        # from arriving as int64 and turning code 1 into 1 rather than '1';
        # keep_default_na=False keeps blanks as '' the way the pipeline reads
        # these files everywhere else.
        return pd.read_csv(
            csv_path, keep_default_na=False, dtype=str,
            usecols=[self.SUBJECT_VAR, self.SOURCE_VAR] + optional)

    def count_participants(self):
        # The pipeline's -9 / -3 / -99 / 999 family, plus the date sentinels.
        missing_codes = {str(code).strip()
                         for code in self.utils.missing_code_set}
        for network in self.networks:
            combined_df = self.read_network_frame(network)
            if combined_df is None:
                continue
            # The chart counts participants, so make that true rather than
            # assume it: screening is a single timepoint and should already be
            # one row per subject.
            combined_df = combined_df.drop_duplicates(subset=[self.SUBJECT_VAR])
            self.rows_read[network] = len(combined_df)
            self.nonblank_seen[network] = 0
            self.forms_marked_missing[network] = 0

            has_missing_var = self.MISSING_VAR in combined_df.columns
            has_status_var = self.STATUS_VAR in combined_df.columns

            for row in combined_df.itertuples(index=False):
                marked_missing = bool(has_missing_var and self.normalize_code(
                    getattr(row, self.MISSING_VAR)) == '1')
                if marked_missing:
                    self.forms_marked_missing[network] += 1
                status = (str(getattr(row, self.STATUS_VAR)).strip()
                          if has_status_var else '')

                # The listing carries the cell as it was exported, not the
                # normalized form -- when a value fails to map, the original
                # is the part worth seeing.
                raw_value = str(getattr(row, self.SOURCE_VAR))
                code = self.normalize_code(raw_value)
                if not code:
                    # Not a category, so it never reaches the counts; the
                    # per-subject listing still records who it was.
                    self.record_subject(row, network, self.NO_SOURCE_KEY,
                                        raw_value, status, marked_missing)
                    continue
                self.nonblank_seen[network] += 1

                if code in self.choice_labels:
                    key = code
                elif code in missing_codes:
                    key = self.MISSING_CODE_KEY
                else:
                    key = self.UNRECOGNIZED_KEY
                    self.unrecognized_values.setdefault(code, 0)
                    self.unrecognized_values[code] += 1

                self.counts[key][network] += 1
                if status.lower() == 'recruited':
                    self.recruited_counts[key] += 1
                self.record_subject(row, network, key, raw_value, status,
                                    marked_missing)

    def record_subject(self, row, network, key, raw_value, status,
                       marked_missing):
        """
        Remembers one participant for the per-subject listing.

        `key` is a dictionary code for a real method, or one of the bucket
        keys; `raw_value` is what the cell actually held, which is the useful
        part when the value did not map.
        """
        self.subject_rows.append({
            # Carried only to order the file; dropped before writing. Ordering
            # on the key rather than the label keeps a dictionary label that
            # ever matched a bucket label from scrambling the grouping.
            '_key': key,
            'code': key if key in self.choice_labels else '',
            # Membership test rather than `.get(key) or ...`: a dictionary
            # revision can produce an empty label (a trailing '|' in the
            # choice string), and `or` would fall through to a KeyError.
            'recruitment_method': (
                self.choice_labels[key] if key in self.choice_labels
                else self.BUCKET_LABELS[key]),
            'subjectid': getattr(row, self.SUBJECT_VAR),
            'network': network,
            'recruitment_status_v2': status,
            'raw_chrrecruit_value': raw_value,
            'form_marked_missing': marked_missing,
            'counted_in_chart': key in self.choice_labels,
        })

    # --- outputs ----------------------------------------------------------

    def total_for(self, code):
        return sum(self.counts[code].values())

    def ordered_codes(self):
        """
        Dictionary codes in chart order: most participants first, dictionary
        order breaking ties so the order never reshuffles between runs.
        """
        order = list(self.choice_labels)
        return sorted(order,
                      key=lambda code: (-self.total_for(code),
                                        order.index(code)))

    def chart_rows(self):
        """
        (label, count) for the bars, largest first.

        Every dictionary category is present even at zero. The non-method
        buckets are left out of the bars and carried in the caption instead.
        """
        return [(self.choice_labels[code], self.total_for(code))
                for code in self.ordered_codes()]

    def write_workbook(self, df, out_path, sheet_name):
        """Write a filterable Excel table with one fill per recruitment method."""
        # Dictionary order keeps colors consistent across both files and when
        # participant counts change. Duplicate labels share the same fill.
        methods = dict.fromkeys(self.choice_labels.values())
        fills = {}
        for index, method in enumerate(methods):
            # Golden-ratio hue spacing supports new dictionary categories
            # without cycling through a fixed palette. Pale fills keep the
            # dark text readable across the full row.
            rgb = colorsys.hls_to_rgb((index * 0.61803398875) % 1, 0.87, 0.65)
            color = ''.join(f'{round(channel * 255):02X}' for channel in rgb)
            fills[method] = PatternFill('solid', fgColor=color)
        for label, color in zip(self.BUCKET_LABELS.values(),
                                ('FFF2CC', 'F4CCCC', 'E7E6E6')):
            fills.setdefault(label, PatternFill('solid', fgColor=color))

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = sheet_name
        sheet.append(list(df.columns))
        sheet.freeze_panes = 'C2'
        sheet.sheet_view.showGridLines = False
        header_fill = PatternFill('solid', fgColor='24476A')
        header_font = Font(name='Arial', size=10, bold=True, color='FFFFFF')
        body_font = Font(name='Arial', size=10, color='0B0B0B')
        body_alignment = Alignment(vertical='center')
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')

        method_index = df.columns.get_loc('recruitment_method')
        for row_index, values in enumerate(
                df.itertuples(index=False, name=None), start=2):
            fill = fills[values[method_index]]
            # REDCap text may contain control characters that CSV accepts
            # but Excel's XML format cannot represent.
            values = tuple(ILLEGAL_CHARACTERS_RE.sub('', value)
                           if isinstance(value, str) else value
                           for value in values)
            sheet.append(values)
            for column_index, (column, value) in enumerate(
                    zip(df.columns, values), start=1):
                cell = sheet.cell(row_index, column_index)
                # Identifiers/raw source values must remain literal text,
                # including leading zeros or a value starting with '='.
                if isinstance(value, str):
                    cell.data_type = 's'
                cell.fill = fill
                cell.font = body_font
                cell.alignment = body_alignment
                if column == 'pct_of_recorded_methods':
                    # Preserve the existing 0..100 percentage values.
                    cell.number_format = '0.00"%"'
                elif isinstance(value, (int, float)):
                    cell.number_format = '#,##0'

        sheet.auto_filter.ref = sheet.dimensions
        for index, column in enumerate(df.columns, start=1):
            width = max(len(column), max(
                (len(str(value)) for value in df[column]), default=0))
            sheet.column_dimensions[get_column_letter(index)].width = width + 3
        workbook.save(out_path)

    def write_excel(self):
        records = []
        method_total = sum(self.total_for(code) for code in self.choice_labels)
        for code in self.choice_labels:
            records.append({
                'code': code,
                'recruitment_method': self.choice_labels[code],
                'participants': self.total_for(code),
                'pct_of_recorded_methods': (
                    round(100 * self.total_for(code) / method_total, 2)
                    if method_total else 0.0),
                'participants_recruited_status': self.recruited_counts[code],
                **{f'participants_{net}': self.counts[code][net]
                   for net in self.networks},
            })
        for key, label in ((self.MISSING_CODE_KEY,
                            'Missing-data code (-9 / -3 / -99 / 999)'),
                           (self.UNRECOGNIZED_KEY, 'Unrecognized code')):
            records.append({
                'code': '',
                'recruitment_method': label,
                'participants': self.total_for(key),
                'pct_of_recorded_methods': '',
                'participants_recruited_status': self.recruited_counts[key],
                **{f'participants_{net}': self.counts[key][net]
                   for net in self.networks},
            })

        columns = (['code', 'recruitment_method', 'participants',
                    'pct_of_recorded_methods',
                    'participants_recruited_status']
                   + [f'participants_{net}' for net in self.networks])
        out_path = os.path.join(self.output_path, self.XLSX_FILENAME)
        self.write_workbook(pd.DataFrame(records, columns=columns), out_path,
                            'Recruitment counts')
        self.xlsx_path = out_path

    def write_subject_excel(self):
        """
        Writes one row per participant: which recruitment method they are
        counted under, and the raw value behind it.

        Grouped in the same order as the chart, so the two read together.
        Unlike the counts file this carries subject identifiers, so it is a
        listing for follow-up rather than something to circulate.
        """
        # Methods first in chart order, then the non-method buckets, so the
        # file opens on the same rows the chart shows.
        ordered = list(self.ordered_codes()) + [self.MISSING_CODE_KEY,
                                                self.UNRECOGNIZED_KEY,
                                                self.NO_SOURCE_KEY]
        rank = {key: index for index, key in enumerate(ordered)}

        columns = ['code', 'recruitment_method', 'subjectid', 'network',
                   'recruitment_status_v2', 'raw_chrrecruit_value',
                   'form_marked_missing', 'counted_in_chart']
        df = pd.DataFrame(self.subject_rows, columns=['_key'] + columns)
        if not df.empty:
            df = df.assign(_rank=df['_key'].map(rank)).sort_values(
                ['_rank', 'network', 'subjectid'])
            # yes/no rather than True/False: these land in Excel and R as
            # text either way, and 'yes' does not invite a boolean test that
            # quietly fails against the string 'True'.
            for flag in ('form_marked_missing', 'counted_in_chart'):
                df[flag] = df[flag].map({True: 'yes', False: 'no'})
        df = df[columns]
        out_path = os.path.join(self.output_path, self.SUBJECTS_XLSX_FILENAME)
        self.write_workbook(df, out_path, 'Recruitment subjects')
        self.subjects_xlsx_path = out_path

    # --- chart ------------------------------------------------------------

    def rounded_bar_path(self, x_end, y_center, bar_height, radius_x, radius_y):
        """
        A bar whose data end is rounded and whose baseline end is square.

        Radii arrive already converted into data units, one per axis, so the
        corner renders circular rather than stretched by the aspect ratio.
        """
        y0, y1 = y_center - bar_height / 2, y_center + bar_height / 2
        # Never let the rounding eat more than the bar has to give.
        rx = min(radius_x, x_end)
        ry = min(radius_y, bar_height / 2)
        vertices = [
            (0, y0),
            (x_end - rx, y0),
            (x_end - rx + rx * KAPPA, y0), (x_end, y0 + ry - ry * KAPPA),
            (x_end, y0 + ry),
            (x_end, y1 - ry),
            (x_end, y1 - ry + ry * KAPPA), (x_end - rx + rx * KAPPA, y1),
            (x_end - rx, y1),
            (0, y1),
            (0, y0),
        ]
        codes = [MplPath.MOVETO, MplPath.LINETO,
                 MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO,
                 MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO, MplPath.CLOSEPOLY]
        return MplPath(vertices, codes)

    def caption(self):
        parts = []
        missing_code_total = self.total_for(self.MISSING_CODE_KEY)
        unrecognized_total = self.total_for(self.UNRECOGNIZED_KEY)
        if missing_code_total:
            parts.append(f'{missing_code_total:,} participant(s) whose '
                         'recruitment source holds a missing-data code')
        if unrecognized_total:
            parts.append(f'{unrecognized_total:,} holding a value that is not '
                         'a dictionary code '
                         f'({", ".join(sorted(self.unrecognized_values))})')
        if self.skipped_networks:
            parts.append('no data read for '
                         f'{", ".join(sorted(set(self.skipped_networks)))}')
        # Name the networks actually read, not the configured scope, so the
        # source line cannot claim data that the "no data read for" clause
        # below then takes away.
        read_networks = list(self.rows_read) or self.networks
        # Spell out what each network contributed, so "are both networks in
        # here?" is answerable from the figure alone rather than the workbook.
        per_network = ' + '.join(
            f'{network} '
            f'{sum(self.counts[code][network] for code in self.choice_labels):,}'
            for network in read_networks)
        source = (f'Source: {self.SOURCE_VAR} on the {self.FORM} form, '
                  f'{self.TIMEPOINT} timepoint, combined REDCap export '
                  f'({per_network}).')
        return source + (f'  Not shown: {"; ".join(parts)}.' if parts else '')

    def render_chart(self):
        codes = self.ordered_codes()
        labels = [textwrap.fill(self.choice_labels[code], LABEL_WRAP)
                  for code in codes]
        values = [self.total_for(code) for code in codes]
        participants = sum(values)
        # Use one fixed stack/column order even if --networks is reversed.
        networks = [network for network in DEFAULT_NETWORKS
                    if network in self.networks]

        fig_height = ROW_PITCH_IN * len(codes) + 2.1
        fig, ax = plt.subplots(figsize=(13.5, fig_height), dpi=self.dpi)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        y_positions = list(range(len(codes) - 1, -1, -1))
        ax.set_xlim(0, max(max(values, default=0), 1) * 1.02)
        ax.set_ylim(-0.55, len(codes) - 0.45)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=9.5, color=TEXT_PRIMARY)
        # Exact per-network counts are aligned beside each stacked bar.
        # This also makes zero and tiny segments readable without overlap.
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='y', length=0, pad=8)

        fig.text(0.30, 1 - 0.22 / fig_height,
                 'How AMP SCZ participants were recruited',
                 fontsize=15, color=TEXT_PRIMARY, va='top',
                 fontweight='semibold')
        subtitle = (f'{participants:,} '
                    f'participant{"" if participants == 1 else "s"} with a '
                    f'recruitment source recorded · all {len(codes)} '
                    'dictionary methods shown')
        fig.text(0.30, 1 - 0.57 / fig_height, subtitle, fontsize=10.5,
                 color=TEXT_SECONDARY, va='top')

        fig.subplots_adjust(left=0.30, right=0.76,
                            top=1 - 1.25 / fig_height,
                            bottom=0.82 / fig_height)
        column_positions = {network: 1.15 + index * 0.26
                            for index, network in enumerate(networks)}
        for network, column_x in column_positions.items():
            network_total = sum(self.counts[code][network] for code in codes)
            detail = ('no data' if network in self.skipped_networks
                      else f'n={network_total:,}')
            # These headers double as the color legend and network totals.
            fig.text(0.30 + 0.46 * column_x, 1 - 0.86 / fig_height,
                     f'{network}\n({detail})', ha='right', va='top',
                     fontsize=9.5, fontweight='semibold',
                     color=NETWORK_COLORS[network], linespacing=1.4)

        # Layout has to be final before data units can be converted to inches.
        fig.canvas.draw()
        axes_box = ax.get_window_extent()
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        x_units_per_inch = (x_hi - x_lo) / (axes_box.width / self.dpi)
        y_units_per_inch = (y_hi - y_lo) / (axes_box.height / self.dpi)

        bar_height = min(BAR_MAX_THICKNESS_IN * y_units_per_inch, 0.72)
        radius_x = BAR_CORNER_RADIUS_IN * x_units_per_inch
        radius_y = BAR_CORNER_RADIUS_IN * y_units_per_inch
        for code, y_position, total in zip(codes, y_positions, values):
            # Clip the whole stack to one rounded end, keeping the internal
            # network boundary square and adjacent segments flush.
            outline = PathPatch(self.rounded_bar_path(
                total, y_position, bar_height, radius_x, radius_y),
                transform=ax.transData)
            left = 0
            for network in networks:
                count = self.counts[code][network]
                if count:
                    bars = ax.barh(y_position, count, left=left,
                                   height=bar_height,
                                   color=NETWORK_COLORS[network],
                                   edgecolor='none')
                    bars[0].set_clip_path(outline)
                left += count
                label = ('n/a' if network in self.skipped_networks
                         else f'{count:,}')
                ax.text(column_positions[network], y_position, label,
                        transform=ax.get_yaxis_transform(),
                        va='center', ha='right', fontsize=9.5,
                        color=TEXT_PRIMARY if count else TEXT_SECONDARY,
                        clip_on=False)

        fig.text(0.30, 0.26 / fig_height, self.caption(), fontsize=8.5,
                 color=TEXT_SECONDARY, va='bottom', wrap=True)

        out_path = os.path.join(self.output_path, self.PNG_FILENAME)
        fig.savefig(out_path, facecolor=SURFACE, dpi=self.dpi)
        plt.close(fig)
        self.png_path = out_path

    # --- reconciliation ---------------------------------------------------

    def report(self):
        method_total = sum(self.total_for(code) for code in self.choice_labels)
        missing_code_total = self.total_for(self.MISSING_CODE_KEY)
        unrecognized_total = self.total_for(self.UNRECOGNIZED_KEY)
        nonblank_total = sum(self.nonblank_seen.values())
        recruited_total = sum(
            self.recruited_counts[code] for code in self.choice_labels)

        print()
        print('[recruitment_sources] participants per recruitment method '
              f'({self.SOURCE_VAR}, {self.TIMEPOINT}):')
        for label, count in self.chart_rows():
            share = (f'{100 * count / method_total:5.1f}%'
                     if method_total else '    -')
            print(f'    {count:>6,}  {share}  {label}')
        print(f'    {method_total:>6,}          TOTAL with a recruitment '
              'method recorded')
        print(f'    {recruited_total:>6,}          of those, '
              f"{self.STATUS_VAR} == 'recruited'")
        print()
        for network in self.networks:
            if network in self.rows_read:
                print(f'[recruitment_sources] {network}: '
                      f'{self.rows_read[network]:,} participant row(s) read, '
                      f'{self.nonblank_seen[network]:,} with a non-blank '
                      f'{self.SOURCE_VAR}, '
                      f'{self.forms_marked_missing[network]:,} with the form '
                      'marked missing')
        # If this does not balance the counting is wrong, and that is worth
        # saying out loud rather than leaving in a column no one adds up.
        reconciled = (method_total + missing_code_total + unrecognized_total
                      == nonblank_total)
        print(f'[recruitment_sources] reconciliation: {method_total:,} mapped '
              f'+ {missing_code_total:,} missing-data code '
              f'+ {unrecognized_total:,} unrecognized '
              f'= {nonblank_total:,} non-blank '
              f'-- {"OK" if reconciled else "MISMATCH"}')
        if unrecognized_total:
            print('[recruitment_sources] WARNING: values that are not a '
                  'dictionary code were found. Raw value -> count: '
                  + ', '.join(f'{value!r}: {count:,}' for value, count
                              in sorted(self.unrecognized_values.items())))
        print(f'[recruitment_sources] wrote {self.png_path}')
        print(f'[recruitment_sources] wrote {self.xlsx_path}')
        # This file is the one artifact whose row count exceeds the chart's,
        # so reconcile it on screen rather than only in the docstring.
        counted = sum(1 for entry in self.subject_rows
                      if entry['counted_in_chart'])
        buckets = []
        for key in (self.NO_SOURCE_KEY, self.MISSING_CODE_KEY,
                    self.UNRECOGNIZED_KEY):
            total = sum(1 for entry in self.subject_rows
                        if entry['_key'] == key)
            if total:
                buckets.append(f'{total:,} {self.BUCKET_LABELS[key].lower()}')
        print(f'[recruitment_sources] wrote {self.subjects_xlsx_path} '
              '-- contains subject identifiers')
        print(f'[recruitment_sources]   {len(self.subject_rows):,} '
              f'participant rows = {counted:,} counted in the chart'
              + (' + ' + ' + '.join(buckets) if buckets else ''))
        # Last thing on screen, because a partial chart is the one failure
        # that looks like a finished one. The early skip message scrolls away
        # above the table; this does not.
        if self.skipped_networks:
            missing = ', '.join(sorted(set(self.skipped_networks)))
            print()
            print('*' * 72)
            print(f'*  PARTIAL RESULT: no data was read for {missing}.')
            print('*  The counts above cover only '
                  f'{", ".join(self.rows_read) or "no network"}.')
            print('*  Fix the export path/filename above and re-run before '
                  'using this chart.')
            print('*' * 72)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Bar chart of participants per recruitment source '
                    '(chrrecruit on the recruitment_source form).')
    parser.add_argument(
        '--output-dir', default=None,
        help='Where to write the PNG and Excel workbooks. Defaults to '
             'the configured paths.output_path.')
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='Resolution of the PNG (default: 150).')
    parser.add_argument(
        '--networks', default=None,
        help='Comma-separated networks to count. Defaults to both '
             f'({", ".join(DEFAULT_NETWORKS)}); the network scope of the QC '
             'pipeline is not inherited.')
    args = parser.parse_args(argv)
    if args.networks:
        requested = [n.strip().upper() for n in args.networks.split(',')
                     if n.strip()]
        unknown = [n for n in requested if n not in DEFAULT_NETWORKS]
        if unknown or not requested:
            parser.error(f'--networks must name one or more of '
                         f'{", ".join(DEFAULT_NETWORKS)}; got {args.networks!r}')
        args.networks = requested
    return args


if __name__ == '__main__':
    args = parse_args()
    GraphRecruitmentSources(output_dir=args.output_dir, dpi=args.dpi,
                            networks=args.networks).run_script()
