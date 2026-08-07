"""Plot 'flags resolved per day' curves from the deduplicated tracker
history written by ``analyze_flags/estimate_resolved.py``.

Two outputs are produced per network:
  * ``resolved_over_time_{network}_all.png`` — full population, the
    primary view of QC throughput over time.
  * ``resolved_over_time_{network}_filtered.png`` — restricted to forms
    matching ``FILTERED_FORM_PATTERNS`` (currently p1p8 / p9ac32 / mood),
    intended as a focused side analysis on those psychiatric assessment
    forms.

Bulk-operation jumps (server issues / mass scripts that add or drop
hundreds of flags in one revision) skew the curve away from normal
day-to-day QC progress. Instead of the old heuristic per-day spike
thresholds, exclusions now come from the operator-reviewed
``jumps_{network}.xlsx`` written by ``estimate_resolved.py``:
  * entries born in an excluded ADDITION jump are dropped entirely
    (the flag itself is considered an artifact);
  * entries that vanished in an excluded REMOVAL jump are kept but
    NOT counted as resolutions.
A jump is excluded unless its ``include`` cell is affirmative, so the
first run (before manual review) behaves like an automatic filter.
Pass ``apply_jump_exclusions=False`` to ``__init__`` to graph the raw,
unfiltered curve.
"""

import json
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
print(parent_dir)
from utils.utils import Utils
from analyze_flags.manual_review import (
    entry_key_from_row,
    load_jump_decisions,
)
from analyze_flags.paths import (
    ensure_analyze_flags_artifact_dir,
    open_tracker_row_history_csv_basename,
    pipeline_output_path_from_config,
)


class ResolvedGrapher():
    NETWORKS = ['PRESCIENT', 'PRONET']

    # Forms used in the focused side analysis. Stored as substrings so a
    # General_Flag of 'p1p8_form' matches 'p1p8'.
    FILTERED_FORM_PATTERNS = ['p1p8', 'p9ac32', 'mood']

    # Used to key per-variant state (total flag counts). Keep in sync
    # with the variant strings passed to _produce_graph below.
    VARIANTS = ('all', 'filtered')

    def __init__(self, apply_jump_exclusions=True):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json', 'r') as file:
            self.config_info = json.load(file)
        self.output_path = pipeline_output_path_from_config(self.config_info)
        self.analyze_flags_dir = ensure_analyze_flags_artifact_dir(self.output_path)
        # When False, jumps_{network}.xlsx is ignored and every entry /
        # resolution goes into the graph raw. Useful when investigating
        # whether a particular jump was actually a bulk operation or a
        # legitimate burst of QC activity.
        self.apply_jump_exclusions = apply_jump_exclusions
        # Per-(network, variant) running totals so the end-of-run print
        # shows both the all-forms count and the filtered-subset count
        # side by side rather than a single conflated number.
        self.total_flags = {
            n: {v: 0 for v in self.VARIANTS} for n in self.NETWORKS
        }
        # Entries excluded per network by the operator's jump decisions
        # (added = dropped entirely; removed = not counted as resolved).
        self.jump_excluded = {
            n: {'added': 0, 'removed': 0} for n in self.NETWORKS
        }

    def run_script(self):
        for network in self.NETWORKS:
            csv_path = os.path.join(
                self.analyze_flags_dir,
                open_tracker_row_history_csv_basename(network),
            )
            if not os.path.exists(csv_path):
                print(
                    f"No merged history CSV for {network} at {csv_path}; "
                    f"run estimate_resolved.py first. Skipping."
                )
                continue

            df = self._load_history(csv_path)
            if df.empty:
                print(f"{network} history CSV is empty; skipping.")
                continue

            if self.apply_jump_exclusions:
                decisions = load_jump_decisions(
                    self.analyze_flags_dir, network
                )
                if decisions['n_jumps'] == 0:
                    print(
                        f"{network}: no jumps workbook found (or no jumps "
                        f"detected); graphing unfiltered curve."
                    )
                elif decisions['n_undecided']:
                    print(
                        f"{network}: {decisions['n_undecided']} of "
                        f"{decisions['n_jumps']} jumps have no include "
                        f"decision yet — they are EXCLUDED by default. "
                        f"Review jumps_{network}.xlsx to change that."
                    )
                df = self._drop_excluded_added(
                    df, decisions['excluded_added'], network
                )
                excluded_removed = decisions['excluded_removed']
            else:
                excluded_removed = set()

            # Main analysis: all entries.
            self._produce_graph(df, network, 'all', excluded_removed)

            # Focused analysis: only entries from forms matching
            # FILTERED_FORM_PATTERNS.
            filtered_df = self._filter_to_priority_forms(df)
            self._produce_graph(
                filtered_df, network, 'filtered', excluded_removed
            )

        print('jump_excluded:', self.jump_excluded)
        print('total_flags:', self.total_flags)

    def _drop_excluded_added(self, df, excluded_added, network):
        """Drop entries born in an operator-excluded ADDITION jump —
        the flag episode itself is considered an artifact (server
        issue / bad pipeline run), so it shouldn't appear anywhere."""
        if df.empty or not excluded_added:
            return df
        mask = [
            entry_key_from_row(r) not in excluded_added
            for r in df.itertuples(index=False)
        ]
        mask = pd.Series(mask, index=df.index)
        dropped = int((~mask).sum())
        if dropped:
            self.jump_excluded[network]['added'] += dropped
            print(
                f"{network}: dropped {dropped} entries from excluded "
                f"addition jumps."
            )
        return df[mask]

    def _load_history(self, csv_path):
        """Read the entry-level merged-history CSV produced by
        estimate_resolved.py and parse Latest_seen to a normalized
        (midnight) Timestamp so per-day bucketing collapses revisions
        taken at different times of day into one entry.

        Each row in the CSV is one specific-flag episode now (one
        entry per (Subject, Timepoint, General_Flag, variable,
        canonical_template)), so the per-day count below counts
        episodes rather than tracker rows."""
        df = pd.read_csv(csv_path, keep_default_na=False)
        df.columns = df.columns.str.replace(' ', '_')
        missing = [c for c in ('Latest_seen', 'Earliest_seen')
                   if c not in df.columns]
        if missing:
            print(
                f"WARNING: {csv_path} missing columns {missing} "
                f"(found {list(df.columns)}); skipping this network."
            )
            return df.iloc[0:0]
        # utc=True coerces mixed-tz / tz-naive values to a single tz-aware
        # series, so .dt.normalize() can't trip an AttributeError on
        # object-dtype output. Upstream writes UTC anyway, so this is a
        # no-op in the common case but defensive against hand-edited CSVs.
        df['Latest_seen'] = pd.to_datetime(
            df['Latest_seen'], errors='coerce', utc=True
        ).dt.normalize()
        # is_currently_open round-trips through CSV as 'True'/'False';
        # coerce back to bool. If the column is missing (older artifact),
        # fall back to inferring still-open from max(Latest_seen) —
        # logged for visibility.
        if 'is_currently_open' in df.columns:
            df['is_currently_open'] = (
                df['is_currently_open'].astype(str).str.strip().eq('True')
            )
        else:
            print(
                f"WARNING: {csv_path} missing is_currently_open column "
                f"(older artifact format); inferring still-open from "
                f"max(Latest_seen)."
            )
            newest_day = df['Latest_seen'].dropna().max()
            if pd.isna(newest_day):
                df['is_currently_open'] = False
            else:
                df['is_currently_open'] = df['Latest_seen'] == newest_day
        df = df.sort_values(by='Latest_seen', ascending=True)
        return df

    def _filter_to_priority_forms(self, df):
        """Subset df to rows whose General_Flag matches any of
        FILTERED_FORM_PATTERNS as a case-insensitive substring."""
        if 'General_Flag' not in df.columns:
            return df.iloc[0:0]
        pattern = '|'.join(re.escape(s) for s in self.FILTERED_FORM_PATTERNS)
        return df[
            df['General_Flag'].astype(str)
            .str.contains(pattern, case=False, na=False)
        ]

    def _produce_graph(self, df, network, variant, excluded_removed):
        """Bucket entries by Latest_seen day (minus operator-excluded
        removal jumps), plot 'flags resolved per day' curve, and
        persist the filtered subset CSV when variant='filtered' so
        users can inspect what's in the focused view."""
        if df.empty:
            print(f"No rows for {network} {variant}; skipping graph.")
            return

        # Persist the filtered subset to disk so the focused analysis is
        # auditable independently of the graph.
        if variant == 'filtered':
            subset_path = os.path.join(
                self.analyze_flags_dir,
                f"{network}_filtered_history.csv",
            )
            df.to_csv(subset_path, index=False)

        per_date = self._bucket_by_day(df, excluded_removed, network, variant)
        flags_per_day = self._build_per_day_series(
            per_date, network, variant
        )

        if not flags_per_day:
            print(f"{network} {variant}: no plottable days.")
            return

        png_path = os.path.join(
            self.analyze_flags_dir,
            f"resolved_over_time_{network}_{variant}.png",
        )
        self._render_plot(flags_per_day, network, variant, png_path)
        print(
            f"Wrote {png_path}: "
            f"{sum(flags_per_day.values())} flags across "
            f"{len(flags_per_day)} days."
        )

    def _bucket_by_day(self, df, excluded_removed, network, variant):
        """Group entries by Latest_seen day. Entries with
        is_currently_open=True are dropped — those flags are still
        open as of the most recent tracker revision (per the explicit
        column from estimate_resolved.py, NOT inferred from
        max(Latest_seen)), so they're not resolutions of any day.
        Entries whose disappearance the operator excluded (removal
        jump with a non-affirmative include) are real flags but NOT
        real resolutions, so they don't get bucketed either."""
        per_date = {}
        skipped_removed = 0
        for row in df.itertuples(index=False):
            if getattr(row, 'is_currently_open', False):
                continue
            if excluded_removed and \
                    entry_key_from_row(row) in excluded_removed:
                skipped_removed += 1
                continue
            date = getattr(row, 'Latest_seen')
            if pd.isna(date):
                continue
            earliest_date = getattr(row, 'Earliest_seen')
            try:
                days_btwn = self.utils.find_days_between(
                    str(date), str(earliest_date)
                )
            except (ValueError, AttributeError):
                # Malformed dates (NaT, blank) — skip rather than crash
                # the whole network's run.
                continue
            per_date.setdefault(date, []).append({'days_btwn': days_btwn})
        if skipped_removed:
            self.jump_excluded[network]['removed'] += skipped_removed
            print(
                f"{network} {variant}: {skipped_removed} resolutions "
                f"excluded via removal jumps."
            )
        return per_date

    def _build_per_day_series(self, per_date, network, variant):
        """Walk per-date buckets in chronological order and count
        resolutions per day. Jump exclusions were already applied at
        the entry level, so no per-day heuristics remain."""
        flags_per_day = {}
        for date in sorted(per_date.keys()):
            rows = per_date[date]
            if not rows:
                continue
            flags_per_day[date] = len(rows)
            self.total_flags[network][variant] += len(rows)
        return flags_per_day

    def _render_plot(self, flags_per_day, network, variant, png_path):
        if variant == 'all':
            title_suffix = ' (all forms)'
        else:
            patterns = ', '.join(self.FILTERED_FORM_PATTERNS)
            title_suffix = f' (forms matching: {patterns})'

        plt.figure(figsize=(15, 8))
        plt.plot(
            list(flags_per_day.keys()),
            list(flags_per_day.values()),
            marker='o',
        )
        plt.title(f"{network} Flags Resolved Over Time{title_suffix}")
        plt.xlabel('Date')
        plt.ylabel('Flags Resolved')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(png_path, format='png', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    ResolvedGrapher().run_script()
