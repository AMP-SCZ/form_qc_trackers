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
import tempfile

import matplotlib.pyplot as plt
import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)
from utils.utils import Utils
from analyze_flags.manual_review import (
    JumpDecisionIntegrityError,
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

            try:
                df = self._load_history(csv_path, network)
            except (pd.errors.EmptyDataError, pd.errors.ParserError,
                    UnicodeDecodeError, OSError) as e:
                print(
                    f"WARNING: could not read {csv_path} "
                    f"({type(e).__name__}: {str(e)[:200]}); preserving "
                    "existing graph artifacts for this network."
                )
                continue
            if df is None:
                # Invalid/partial input is not evidence that there are no
                # events, so leave last-good artifacts untouched.
                continue

            if df.empty:
                # A readable, schema-valid header-only history is an
                # authoritative successful-empty result. Clear stale plots and
                # persist an empty focused subset rather than preserving old
                # data that no longer exists.
                self._produce_graph(df, network, 'all', set())
                self._produce_graph(
                    self._filter_to_priority_forms(df), network, 'filtered', set()
                )
                continue

            if self.apply_jump_exclusions:
                try:
                    decisions = load_jump_decisions(
                        self.analyze_flags_dir, network
                    )
                except JumpDecisionIntegrityError as e:
                    print(
                        f"WARNING: {network} jump decisions are incomplete "
                        f"({e}); preserving existing graph artifacts for this "
                        "network."
                    )
                    continue
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

    def _load_history(self, csv_path, network=None):
        """Read the entry-level merged-history CSV produced by
        estimate_resolved.py and parse observation timestamps to normalized
        (midnight) UTC Timestamps.

        Each row in the CSV is one specific-flag episode now (one
        entry per (Subject, Timepoint, General_Flag, variable,
        canonical_template)), so the per-day count below counts
        episodes rather than tracker rows."""
        df = pd.read_csv(csv_path, keep_default_na=False)
        df.columns = df.columns.str.replace(' ', '_')
        required = (
            'Subject', 'Timepoint', 'General_Flag', 'variable',
            'canonical_template', 'Latest_seen',
        )
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(
                f"WARNING: {csv_path} missing columns {missing} "
                f"(found {list(df.columns)}); skipping this network."
            )
            return None
        # utc=True coerces mixed-tz / tz-naive values to a single tz-aware
        # series, so .dt.normalize() can't trip an AttributeError on
        # object-dtype output. Upstream writes UTC anyway, so this is a
        # no-op in the common case but defensive against hand-edited CSVs.
        latest_raw = df['Latest_seen'].astype(str).str.strip()
        latest_parsed = pd.to_datetime(
            latest_raw, errors='coerce', utc=True
        ).dt.normalize()
        if latest_parsed.isna().any():
            print(
                f"WARNING: {csv_path} contains "
                f"{int(latest_parsed.isna().sum())} invalid/blank Latest_seen "
                "value(s); preserving the existing graph instead of using a "
                "partial history."
            )
            return None
        df['Latest_seen'] = latest_parsed
        if 'Resolution_observed' in df.columns:
            # Null is intentional for episodes whose resolution has not been
            # observed.  Never fill individual nulls from Latest_seen: doing so
            # would turn an unknown resolution into a fabricated date.
            resolution_raw = df['Resolution_observed'].astype(str).str.strip()
            resolution_parsed = pd.to_datetime(
                resolution_raw, errors='coerce', utc=True
            ).dt.normalize()
            malformed = resolution_raw.ne('') & resolution_parsed.isna()
            if malformed.any():
                print(
                    f"WARNING: {csv_path} contains {int(malformed.sum())} "
                    "malformed non-blank Resolution_observed value(s); "
                    "preserving the existing graph."
                )
                return None
            df['Resolution_observed'] = resolution_parsed
        else:
            # Compatibility only for histories written before the dedicated
            # resolution-observation column was introduced.
            print(
                f"WARNING: {csv_path} missing Resolution_observed column "
                f"(older artifact format); falling back to Latest_seen for "
                f"resolved-date bucketing."
            )
            df['Resolution_observed'] = df['Latest_seen']
        # is_currently_open round-trips through CSV as 'True'/'False';
        # coerce back to bool. If the column is missing (older artifact),
        # fall back to inferring still-open from max(Latest_seen) —
        # logged for visibility.
        if 'is_currently_open' in df.columns:
            open_raw = (
                df['is_currently_open'].astype(str).str.strip().str.casefold()
            )
            invalid_open = ~open_raw.isin(
                {'true', 'false', '1', '0', 'yes', 'no'}
            )
            if invalid_open.any():
                print(
                    f"WARNING: {csv_path} contains "
                    f"{int(invalid_open.sum())} invalid is_currently_open "
                    "value(s); preserving the existing graph."
                )
                return None
            df['is_currently_open'] = (
                open_raw.isin({'true', '1', 'yes'})
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
        if 'resolution_eligible' in df.columns:
            eligible_raw = (
                df['resolution_eligible'].astype(str).str.strip().str.casefold()
            )
            invalid_eligible = ~eligible_raw.isin(
                {'true', 'false', '1', '0', 'yes', 'no'}
            )
            if invalid_eligible.any():
                print(
                    f"WARNING: {csv_path} contains "
                    f"{int(invalid_eligible.sum())} invalid resolution_eligible "
                    "value(s); preserving the existing graph."
                )
                return None
            df['resolution_eligible'] = eligible_raw.isin(
                {'true', '1', 'yes'})
        elif network == 'PRONET' and 'source' in df.columns:
            # Backward compatibility for histories written before the explicit
            # eligibility field. PRONET V2 remains a testing-only source.
            df['resolution_eligible'] = df['source'].map(
                lambda value: 'V1' in str(value).split('+'))
        else:
            # PRESCIENT V1 was formerly production, and callers that do not
            # provide a network retain the historical all-rows behavior.
            df['resolution_eligible'] = True
        df = df[df['resolution_eligible']].copy()
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
        """Bucket entries by Resolution_observed day (minus operator-excluded
        removal jumps), plot 'flags resolved per day' curve, and
        persist the filtered subset CSV when variant='filtered' so
        users can inspect what's in the focused view."""
        # Persist the filtered subset to disk so the focused analysis is
        # auditable independently of the graph. A valid empty subset must also
        # be written so an older non-empty subset cannot remain stale.
        if variant == 'filtered':
            subset_path = os.path.join(
                self.analyze_flags_dir,
                f"{network}_filtered_history.csv",
            )
            self._write_csv_atomic(df, subset_path)

        png_path = os.path.join(
            self.analyze_flags_dir,
            f"resolved_over_time_{network}_{variant}.png",
        )

        if df.empty:
            self._remove_stale_plot(png_path, network, variant)
            print(f"No rows for {network} {variant}; cleared stale graph.")
            return

        per_date = self._bucket_by_day(df, excluded_removed, network, variant)
        flags_per_day = self._build_per_day_series(
            per_date, network, variant
        )

        if not flags_per_day:
            self._remove_stale_plot(png_path, network, variant)
            print(
                f"{network} {variant}: no plottable resolution days; "
                "cleared stale graph."
            )
            return
        self._render_plot(flags_per_day, network, variant, png_path)
        print(
            f"Wrote {png_path}: "
            f"{sum(flags_per_day.values())} flags across "
            f"{len(flags_per_day)} days."
        )

    def _remove_stale_plot(self, png_path, network, variant):
        if not os.path.isfile(png_path):
            return
        try:
            os.remove(png_path)
        except OSError as e:
            print(
                f"WARNING: could not remove stale {network} {variant} graph "
                f"at {png_path} ({type(e).__name__}: {e})."
            )

    @staticmethod
    def _write_csv_atomic(df, path):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.", suffix=".csv",
            dir=directory,
        )
        os.close(fd)
        try:
            df.to_csv(temporary_path, index=False)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def _bucket_by_day(self, df, excluded_removed, network, variant):
        """Group entries by the date resolution was observed. Entries with
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
            date = getattr(row, 'Resolution_observed', pd.NaT)
            if pd.isna(date):
                continue
            per_date.setdefault(date, []).append({})
        if skipped_removed:
            # The filtered variant is a subset of ``all``. Count the excluded
            # entry once, not once per graph view.
            if variant == 'all':
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
        if not per_date:
            self.total_flags[network][variant] = 0
            return {}
        first_day = min(per_date)
        last_day = max(per_date)
        flags_per_day = {
            date: len(per_date.get(date, []))
            for date in pd.date_range(first_day, last_day, freq='D')
        }
        self.total_flags[network][variant] = sum(flags_per_day.values())
        return flags_per_day

    def _render_plot(self, flags_per_day, network, variant, png_path):
        if variant == 'all':
            title_suffix = ' (all forms)'
        else:
            patterns = ', '.join(self.FILTERED_FORM_PATTERNS)
            title_suffix = f' (forms matching: {patterns})'

        directory = os.path.dirname(os.path.abspath(png_path))
        os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(png_path)}.", suffix=".png",
            dir=directory,
        )
        os.close(fd)
        figure, axis = plt.subplots(figsize=(15, 8))
        try:
            axis.plot(
                list(flags_per_day.keys()),
                list(flags_per_day.values()),
                marker='o',
            )
            axis.set_title(f"{network} Flags Resolved Over Time{title_suffix}")
            axis.set_xlabel('Date')
            axis.set_ylabel('Flags Resolved')
            axis.tick_params(axis='x', rotation=45)
            axis.grid(True)
            figure.tight_layout()
            figure.savefig(
                temporary_path, format='png', bbox_inches='tight'
            )
            os.replace(temporary_path, png_path)
        finally:
            plt.close(figure)
            if os.path.exists(temporary_path):
                os.remove(temporary_path)


if __name__ == '__main__':
    ResolvedGrapher().run_script()
