import os
import sys

import os
import sys

# Ensure the project root is on sys.path so `from utils.utils import Utils`
# resolves regardless of the cwd Python was launched from.
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

import pandas as pd
import matplotlib.pyplot as plt
import dropbox
from io import BytesIO
from xml.etree.ElementTree import ParseError as XMLParseError
from utils.utils import Utils

from datetime import datetime, timedelta, date
import json
import numpy as np
import openpyxl

# Truthy values for the v2 `Priority Item` column. After Excel round-trip
# via `keep_default_na=False`, Python booleans come back as the strings
# 'True'/'False'; comparing with `== True` would silently drop everything.
PRIORITY_TRUTHY = (True, 'True', 'true', 'TRUE', 1, '1')


class GraphErrors():
    """
    Class to calculate and
    visualize the number of unresolved
    errors from each site for both networks.
    Operates on the v2 trackers (`{network}_Output_V2.xlsx`).
    """

    def __init__(self):
        self.utils = Utils()
        # Source paths come from config (same loader Utils already uses, so
        # we don't re-open config.json). The v2 trackers live under
        # `formatted_outputs/dropbox_files/` and are named with the V2 suffix.
        self.config_info = self.utils.config_info
        self.output_path = self.config_info['paths']['output_path']
        self.tracker_abs_path = f'{self.output_path}formatted_outputs/dropbox_files'
        self.absolute_path = self.utils.absolute_path
        # Honor testing_enabled the same way create_trackers does, otherwise
        # graphs from a test run would overwrite the production dashboards.
        if self.config_info.get("testing_enabled") == "True":
            self.dropbox_base = '/Apps/Automated QC Trackers/refactoring_tests'
        else:
            self.dropbox_base = '/Apps/Automated QC Trackers'

        self.site_errors = {}
        # Mirror analyze_flags/graph_recovered_flags behavior: drop
        # revisions where a Latest_seen date looks like a bulk operation
        # (lots of flags resolving with very short or near-identical
        # lifespans). Toggle off to plot raw history.
        self.filter_large_spikes = True
        # Bulk-operation thresholds copied verbatim from
        # analyze_flags/graph_recovered_flags.ResolvedGrapher.SPIKE_THRESHOLDS
        # so the curves match what that script would have produced from
        # the same revision history.
        self.spike_thresholds = {
            'count_only': 500,           # any day with >= 300 resolutions
            'count_short_mean': 300,     # >100 resolutions AND mean<7 days
            'short_mean_days': 7,
            'count_low_std': 150,         # >50 resolutions AND stdev<4 days
            'low_std': 4,
            'per_day_cap': 300,          # cap kept rows at this per day
        }
        # Populated by _filter_large_spikes for visibility into which
        # days were dropped (analogous to ResolvedGrapher.large_jumps).
        self.large_jumps = {}

    def run_script(self) -> None:
        """
        Loops through both networks
        and calls functions to visualize
        the number of errors per site.
        """

        for network in ['PRESCIENT']:
            tracker_path = (
                f'{self.tracker_abs_path}/{network}/combined/'
                f'{network}_Output_V2.xlsx'
            )
            tracker_df = pd.read_excel(tracker_path, keep_default_na=False)
            # Line/stacked graphs read the tracker's `Date Resolved`
            # column directly, so the outstanding curve at each date
            # reflects the total unresolved remaining as of that date.
            # Order matters: line/stacked read the space-named columns;
            # collect_errors_per_site mutates df.columns to underscored form.
            self.create_line_graph(tracker_df, network)
            self.create_stacked_line_graph(tracker_df, network)
            self.collect_errors_per_site(tracker_df, network)
            self.visualize_errors(network)

    def _find_dropbox_requester(self, dbx):
        """
        Locate an object that exposes `request_json_object` on the SDK
        version we have. SDK 11+: the Dropbox client itself; older SDKs:
        an internal transport/client attribute.
        """
        if hasattr(dbx, 'request_json_object'):
            return dbx
        for attr in ('_transport', '_client', '_request', '_session',
                     '_Dropbox__transport', '_Dropbox__client'):
            obj = getattr(dbx, attr, None)
            if obj is not None and hasattr(obj, 'request_json_object'):
                return obj
        for name in dir(dbx):
            if name.startswith('__'):
                continue
            try:
                obj = getattr(dbx, name)
            except Exception:
                continue
            if obj is not None and hasattr(obj, 'request_json_object'):
                return obj
        return None

    def _fetch_revision_history(self, network, sample_every_days=4,
                                page_limit=100):
        """
        Walk every revision of `{network}_Output_V2.xlsx` on Dropbox and
        collapse them into one row per unique unresolved flag, keyed on
        (Participant, Timepoint, Form, Flags). Returns a DataFrame with:

          - the four key columns
          - Priority Item (value carried forward from the most recent
            revision the flag appeared in)
          - Earliest_seen: first revision timestamp the flag appeared
            without a Date Resolved / Manually Resolved value
          - Latest_seen: last revision timestamp the flag appeared in
            that same unresolved state — treated downstream as the
            resolution date for flags no longer in the newest revision

        Mirrors `analyze_flags/estimate_resolved.ResolvedEstimator._walk_revisions`
        but filtered to the columns the line/stacked graphs need.
        """
        dbx = self.utils.collect_dropbox_credentials()
        # request_json_object refuses to run with a stale token when the
        # SDK is using a refresh token, so force a refresh first.
        if hasattr(dbx, 'check_and_refresh_access_token'):
            dbx.check_and_refresh_access_token()

        requester = self._find_dropbox_requester(dbx)
        if requester is None:
            raise RuntimeError(
                "No request_json_object found on the Dropbox client. "
                "Upgrade the `dropbox` package, or fall back to plain "
                "requests against /2/files/list_revisions."
            )

        USER_AUTH = getattr(dropbox.dropbox_client, 'USER_AUTH', None)
        if USER_AUTH is None:
            auth_type = getattr(dropbox.dropbox_client, 'AuthType', None)
            USER_AUTH = getattr(auth_type, 'USER', None) if auth_type else None
        if USER_AUTH is None:
            USER_AUTH = 'user'

        path = (
            f'{self.dropbox_base}/{network}/combined/'
            f'{network}_Output_V2.xlsx'
        )

        # Identifying tuple. Comments columns are intentionally excluded
        # so a comment edit doesn't fork the same flag into a "new" row.
        key_cols = ['Participant', 'Timepoint', 'Form', 'Flags']
        resolved_cols = ['Date Resolved', 'Manually Resolved']
        # Some older revisions used the analyze_flags-style schema;
        # normalize them back to the current Output_V2 column names.
        rename_to_current = {
            'Subject': 'Participant',
            'General Flag': 'Form',
            'Specific Flags': 'Flags',
            'Manually Marked as Resolved': 'Manually Resolved',
        }

        unique_rows = {}
        newest_revision = None
        before_rev = None
        has_more = True
        last_kept_when = None
        kept = 0

        while has_more:
            req = {'path': path, 'mode': 'path', 'limit': page_limit}
            if before_rev:
                req['before_rev'] = before_rev
            data = requester.request_json_object(
                'api', 'files/list_revisions', 'rpc', req, USER_AUTH, None,
            )
            entries = data.get('entries', [])
            has_more = bool(data.get('has_more', False))
            if not entries:
                break

            for entry in entries:
                rev = entry['rev']
                when = datetime.fromisoformat(
                    entry['server_modified'].replace('Z', '+00:00')
                )
                if newest_revision is None or when > newest_revision:
                    newest_revision = when
                # Sample to avoid pulling every minor edit. The newest
                # revision is always processed because last_kept_when
                # starts as None.
                if (sample_every_days and last_kept_when is not None
                        and (last_kept_when - when).days < sample_every_days):
                    continue

                try:
                    _, resp = dbx.files_download(path=path, rev=rev)
                    rev_df = pd.read_excel(
                        BytesIO(resp.content), keep_default_na=False
                    )
                except (ValueError, XMLParseError, Exception) as e:
                    print(
                        f'WARNING: skipping revision {rev} from {when} '
                        f'- {type(e).__name__}: {str(e)[:100]}'
                    )
                    continue

                rename_map = {
                    old: new for old, new in rename_to_current.items()
                    if old in rev_df.columns and new not in rev_df.columns
                }
                if rename_map:
                    rev_df = rev_df.rename(columns=rename_map)

                missing = [c for c in key_cols + resolved_cols
                           if c not in rev_df.columns]
                if missing:
                    print(
                        f'WARNING: skipping revision {rev} from {when} '
                        f'- missing columns {missing}'
                    )
                    continue

                date_blank = (
                    rev_df['Date Resolved'].fillna('').astype(str)
                    .str.strip() == ''
                )
                manual_blank = (
                    rev_df['Manually Resolved'].fillna('').astype(str)
                    .str.strip() == ''
                )
                rev_open = rev_df[date_blank & manual_blank]
                if 'Priority Item' in rev_open.columns:
                    priority_series = rev_open['Priority Item']
                else:
                    priority_series = pd.Series(
                        [''] * len(rev_open), index=rev_open.index
                    )

                for idx, row in rev_open[key_cols].iterrows():
                    key = tuple(row[c] for c in key_cols)
                    pri = priority_series.loc[idx]
                    rec = unique_rows.get(key)
                    if rec is None:
                        unique_rows[key] = {
                            'earliest': when,
                            'latest': when,
                            'priority': pri,
                        }
                    else:
                        if when < rec['earliest']:
                            rec['earliest'] = when
                        if when >= rec['latest']:
                            # Carry the priority value forward from the
                            # most recent revision we've seen the flag in.
                            rec['latest'] = when
                            rec['priority'] = pri

                last_kept_when = when
                kept += 1
                print(f'{network} - Revision {kept}: {rev} from {when}')

            before_rev = entries[-1]['rev']

        if not unique_rows:
            return pd.DataFrame(
                columns=key_cols
                + ['Priority Item', 'Earliest_seen', 'Latest_seen']
            )

        rows = []
        for key, rec in unique_rows.items():
            rows.append({
                **dict(zip(key_cols, key)),
                'Priority Item': rec['priority'],
                'Earliest_seen': rec['earliest'],
                'Latest_seen': rec['latest'],
            })
        history_df = pd.DataFrame(rows)
        # Stash the newest revision timestamp so the graph methods can
        # tell which flags are still open (Latest_seen == newest revision)
        # without having to recompute it from the dataframe alone.
        self._newest_revision = newest_revision
        return history_df

    def _detection_date_series(self, df, today):
        """
        v2 trackers store `Days Since Detected` (number of days from the
        most recent detection event to today) instead of an explicit
        detection-date column. Reconstruct the per-row detection date as
        `today - Days Since Detected`. Empty / non-numeric values become
        NaT; `_get_outstanding_as_of` treats NaT as "unknown detection
        date, assume the row is outstanding" so legacy rows with a blank
        `Days Since Detected` still show up in today's count.
        """
        days = pd.to_numeric(df['Days Since Detected'], errors='coerce')
        return today - pd.to_timedelta(days, unit='D')

    def _get_outstanding_as_of(self, df, cutoff_date, today=None):
        """
        Canonical "outstanding as of cutoff_date" filter shared by the
        line graph and the stacked graph so their totals stay aligned.

        A row is outstanding when:
          1. The most recent detection date is on or before cutoff_date
             (or unknown — see note below),
          2. Date Resolved is blank/invalid or strictly after cutoff_date,
          3. Manually Resolved is blank.

        NaT detection dates (blank / non-numeric `Days Since Detected`)
        are treated as outstanding regardless of cutoff_date. Without
        this, today's "Outstanding Queries" undercounts the tracker:
        the canonical tracker definition of unresolved is just
        (Date Resolved blank AND Manually Resolved blank), and legacy
        rows with a blank `Days Since Detected` are unresolved by that
        rule. Historically these rows are assumed always-outstanding,
        which is a slight overestimate for early cutoffs but keeps the
        current rightmost point honest.
        """
        cutoff_date = pd.to_datetime(cutoff_date)
        if today is None:
            today = pd.to_datetime(datetime.today().date())
        date_resolved = pd.to_datetime(df['Date Resolved'], errors='coerce')
        date_detected = self._detection_date_series(df, today)
        return df[
            (date_detected.isna() | (date_detected <= cutoff_date))
            & (date_resolved.isna() | (date_resolved > cutoff_date))
            & (df['Manually Resolved'] == '')
        ]

    def create_stacked_line_graph(self, df, network):
        """
        Per-timepoint outstanding curves stacked, sharing the same
        revision-history source as `create_line_graph`. Reuses the
        cached `_history_df` from the line-graph pass so we don't walk
        Dropbox revisions twice per network. Falls back to fetching if
        called standalone.

        `df` is unused — see `create_line_graph` for the rationale.
        """
        start_date = datetime(2026, 4, 15)
        end_date = datetime.today()
        print('creating stacked graph')

        history_df = getattr(self, '_history_df', None)
        if history_df is None:
            history_df = self._fetch_revision_history(network, sample_every_days=1)
            if self.filter_large_spikes:
                history_df = self._filter_large_spikes(history_df, network)

        if history_df.empty:
            print('empty')
            return

        date_index = pd.date_range(start_date, end_date, freq='D')
        per_tp_curves = {}
        for tp, tp_df in history_df.groupby('Timepoint'):
            tp_per_day = self._per_day_counts_from_history(
                tp_df, start_date, end_date
            )
            per_tp_curves[tp] = self._outstanding_curve(
                tp_per_day['earliest'],
                tp_per_day['effective_latest'],
                date_index,
            )

        if not per_tp_curves:
            print('empty')
            return
        print('not empty')

        df_plot = (
            pd.DataFrame(per_tp_curves, index=date_index)
            .fillna(0)
            .astype(int)
        )

        # Cross-check against create_line_graph's outstanding totals.
        stacked_totals = df_plot.sum(axis=1)
        line_outstanding = getattr(self, '_line_outstanding', {})
        mismatches = []
        for ts, total in stacked_totals.items():
            expected = line_outstanding.get(ts)
            if expected is None and hasattr(ts, 'to_pydatetime'):
                expected = line_outstanding.get(ts.to_pydatetime())
            if expected != int(total):
                mismatches.append((ts, expected, int(total)))
        if mismatches:
            print(f'WARNING: {network} stacked totals do not match line outstanding for '
                  f'{len(mismatches)} dates; first few: {mismatches[:5]}')
        else:
            print(f'{network} stacked totals match line outstanding across '
                  f'{len(stacked_totals)} dates')

        color_map = plt.cm.get_cmap('tab20')
        plt.figure(figsize=(16, 10))
        timepoints = df_plot.columns.tolist()
        values = [df_plot[tp].values for tp in timepoints]

        plt.stackplot(df_plot.index, values, labels=timepoints,
                      colors=color_map(range(len(timepoints))))
        plt.title(f'{network} Outstanding Queries Over Time by Timepoint')
        plt.xlabel('Date')
        plt.ylabel('Query Count')
        plt.xticks(rotation=45)
        plt.legend(title='Timepoint', loc='upper left')
        plt.grid(True)

        graph_path = (
            f'{self.tracker_abs_path}/{network}/combined/'
            f'{network}_outstanding_stacked_graph.png'
        )
        print(graph_path)
        print('saving')
        plt.savefig(graph_path, dpi=800)
        print('saved')
        plt.close()

        self.upload_to_dropbox(graph_path, network)

    def _filter_large_spikes(self, history_df, network):
        """
        Drop bulk-operation spikes from the revision history before it
        reaches the graph methods. Mirrors
        analyze_flags/graph_recovered_flags.ResolvedGrapher._build_per_day_series:
        bucket flags by Latest_seen date, compute count / mean / stdev
        of the per-flag lifespan (days between Earliest_seen and
        Latest_seen), drop the entire date when it trips the spike
        thresholds, and otherwise cap at `per_day_cap` flags per date.

        Skipped dates land in `self.large_jumps[network]` for visibility,
        same shape as ResolvedGrapher tracks them.

        Set `self.filter_large_spikes = False` to bypass entirely.
        """
        if history_df.empty:
            return history_df

        thr = self.spike_thresholds
        df = history_df.copy()
        df['_latest_day'] = pd.to_datetime(df['Latest_seen']).dt.normalize()
        df['_lifespan_days'] = (
            df['_latest_day']
            - pd.to_datetime(df['Earliest_seen']).dt.normalize()
        ).dt.days

        # Still-open flags all share Latest_seen == newest revision timestamp
        # (see `_fetch_revision_history`), so they pile up on one `_latest_day`
        # bucket. That bucket reliably trips the bulk-operation thresholds
        # even though no resolution actually happened — dropping it would
        # remove every currently-open flag from the graph. Skip spike checks
        # on the newest-revision day entirely.
        newest_rev = getattr(self, '_newest_revision', None)
        if newest_rev is not None:
            newest_day = pd.to_datetime(newest_rev)
            # Match the tz state of `_latest_day` so equality holds.
            latest_tz = getattr(df['_latest_day'].dt, 'tz', None)
            new_tz = getattr(newest_day, 'tz', None)
            if latest_tz is None and new_tz is not None:
                newest_day = newest_day.tz_localize(None)
            elif latest_tz is not None and new_tz is None:
                newest_day = newest_day.tz_localize(latest_tz)
            newest_day = newest_day.normalize()
        else:
            newest_day = None

        kept_indices = []
        skipped_dates = []
        for day, day_df in df.groupby('_latest_day'):
            if newest_day is not None and day == newest_day:
                kept_indices.extend(day_df.index.tolist())
                continue
            lifespans = day_df['_lifespan_days'].dropna().to_numpy()
            if lifespans.size == 0:
                # No measurable lifespan on this day; keep as-is.
                kept_indices.extend(day_df.index.tolist())
                continue
            day_count = lifespans.size
            day_mean = float(lifespans.mean())
            day_std = float(lifespans.std(ddof=1)) if day_count > 1 else 0.0
            is_spike = (
                (day_count > thr['count_short_mean']
                 and day_mean < thr['short_mean_days'])
                or day_count >= thr['count_only']
                or (day_count > thr['count_low_std']
                    and day_std < thr['low_std'])
            )
            if is_spike:
                skipped_dates.append(day)
                continue
            # Cap rows per day in original (insertion) order, the same
            # way analyze_flags caps `rows[:300]`.
            kept_indices.extend(day_df.index[: thr['per_day_cap']].tolist())

        if skipped_dates:
            self.large_jumps.setdefault(network, [])
            for d in skipped_dates:
                if d not in self.large_jumps[network]:
                    self.large_jumps[network].append(d)
            print(
                f'{network}: dropped {len(skipped_dates)} spike day(s) '
                f'from line graph; first few: '
                f'{[str(d.date()) for d in skipped_dates[:5]]}'
            )

        # Drop helper cols before returning so downstream methods see
        # the same columns _fetch_revision_history produced.
        return (
            df.loc[kept_indices]
            .drop(columns=['_latest_day', '_lifespan_days'])
            .reset_index(drop=True)
        )

    def _per_day_counts_from_history(self, df, start_date, end_date):
        """
        Translate one-row-per-flag revision history into the per-day
        series the line and stacked graphs both consume.

        A flag with Earliest_seen = E and Latest_seen = L is treated as
        outstanding on every date d where E <= d < effective_L. For
        flags whose Latest_seen matches the newest revision processed
        (`self._newest_revision`), effective_L is pushed past today so
        they stay outstanding through the current date — analyze_flags's
        rule that Latest_seen IS the resolution date only applies once
        the flag has actually dropped out of the tracker.
        """
        date_index = pd.date_range(start_date, end_date, freq='D')
        if df.empty:
            zero = pd.Series(0, index=date_index, dtype='int64')
            return {
                'date_index': date_index,
                'earliest': pd.Series([], dtype='datetime64[ns]'),
                'effective_latest': pd.Series([], dtype='datetime64[ns]'),
                'still_open': pd.Series([], dtype=bool),
                'new_per_day': zero.copy(),
                'resolved_per_day': zero.copy(),
            }

        earliest = pd.to_datetime(df['Earliest_seen'])
        latest = pd.to_datetime(df['Latest_seen'])
        # Dropbox revision timestamps are tz-aware UTC, but date_index /
        # open_sentinel are tz-naive. Strip tz before searchsorted /
        # reindex / .where so the dtypes line up.
        if getattr(earliest.dt, 'tz', None) is not None:
            earliest = earliest.dt.tz_localize(None)
        if getattr(latest.dt, 'tz', None) is not None:
            latest = latest.dt.tz_localize(None)
        earliest = earliest.dt.normalize()
        latest = latest.dt.normalize()
        newest_rev = getattr(self, '_newest_revision', None)
        if newest_rev is None:
            still_open_cutoff = latest.max()
        else:
            cutoff_ts = pd.to_datetime(newest_rev)
            if getattr(cutoff_ts, 'tz', None) is not None:
                cutoff_ts = cutoff_ts.tz_localize(None)
            still_open_cutoff = cutoff_ts.normalize()

        still_open = latest >= still_open_cutoff
        # +1 day so a flag whose effective_latest is today is still
        # counted as outstanding today (cumsum-diff treats end_date as
        # the first non-outstanding day).
        open_sentinel = pd.to_datetime(end_date).normalize() + pd.Timedelta(days=1)
        effective_latest = latest.where(~still_open, open_sentinel)

        # New-on-day-d / resolved-on-day-d series, clipped to the plot
        # window. Resolved excludes still-open flags so we don't draw a
        # spike on the day of the newest revision.
        new_per_day = (
            earliest.value_counts()
            .reindex(date_index, fill_value=0)
            .sort_index()
            .astype('int64')
        )
        resolved_per_day = (
            latest[~still_open]
            .value_counts()
            .reindex(date_index, fill_value=0)
            .sort_index()
            .astype('int64')
        )

        return {
            'date_index': date_index,
            'earliest': earliest,
            'effective_latest': effective_latest,
            'still_open': still_open,
            'new_per_day': new_per_day,
            'resolved_per_day': resolved_per_day,
        }

    def _outstanding_curve(self, earliest, effective_latest, date_index):
        """
        Outstanding-per-day count from per-flag (E, effective_L) bounds.
        Uses searchsorted (not value_counts + reindex) so flags whose
        Earliest_seen falls before the plot window still contribute to
        the curve from start_date forward.

        Convention: a flag is outstanding on day d iff E <= d <= L
        (inclusive on both ends). Latest_seen is the LAST revision the
        flag was open, so it WAS outstanding on that date. That requires
        `side='left'` on the "ended" searchsorted; using `side='right'`
        on both would silently drop each resolved flag from its
        Latest_seen day.
        """
        if len(earliest) == 0:
            return np.zeros(len(date_index), dtype='int64')
        e_sorted = np.sort(earliest.to_numpy())
        l_sorted = np.sort(effective_latest.to_numpy())
        idx_np = date_index.to_numpy()
        started_le_d = np.searchsorted(e_sorted, idx_np, side='right')
        ended_lt_d = np.searchsorted(l_sorted, idx_np, side='left')
        return (started_le_d - ended_lt_d).astype('int64')

    def create_line_graph(self, df, network):
        """
        Daily curves of new / outstanding / resolved / priority queries.

        Detection and resolution dates come from the Dropbox revision
        history of `{network}_Output_V2.xlsx`, not from the in-tracker
        `Days Since Detected` column. `Days Since Detected` is not
        reliable as a per-flag detection date: continuously-open flags
        carry the original detection date forward without recomputation,
        legacy rows may have a blank value entirely, and re-opens reset
        it to today. Walking revisions and keying on
        (Participant, Timepoint, Form, Flags) gives a per-flag
        Earliest_seen (first revision the flag appeared unresolved) and
        Latest_seen (last revision the flag appeared unresolved) that
        actually reflect the tracker's lifecycle.

        `df` (the current tracker) is unused here — only `network` is
        needed to locate the Dropbox file. The signature is kept so
        `run_script`'s caller signature doesn't have to change.
        """
        start_date = datetime(2026, 4, 15)
        end_date = datetime.today()

        history_df = self._fetch_revision_history(network, sample_every_days=1)
        if self.filter_large_spikes:
            history_df = self._filter_large_spikes(history_df, network)
        # Cache so create_stacked_line_graph doesn't re-walk every
        # revision (the walk is the slow part of the run).
        self._history_df = history_df

        per_day = self._per_day_counts_from_history(
            history_df, start_date, end_date
        )
        date_index = per_day['date_index']
        outstanding = self._outstanding_curve(
            per_day['earliest'], per_day['effective_latest'], date_index
        )

        # Priority subset: filter history rows by Priority Item, then
        # recompute the outstanding curve on that subset alone. Priority
        # Item is carried forward from the most recent revision the flag
        # appeared in (see `_fetch_revision_history`).
        if history_df.empty:
            priority_history = history_df
        else:
            priority_history = history_df[
                history_df['Priority Item'].isin(PRIORITY_TRUTHY)
            ]
        pri_per_day = self._per_day_counts_from_history(
            priority_history, start_date, end_date
        )
        priority = self._outstanding_curve(
            pri_per_day['earliest'],
            pri_per_day['effective_latest'],
            date_index,
        )

        new_vals = per_day['new_per_day'].to_numpy()
        resolved_vals = per_day['resolved_per_day'].to_numpy()

        # Cross-check handoff for create_stacked_line_graph: it compares
        # its per-date totals against this dict and warns on mismatch.
        # Key by Timestamp (date_index entries) so the stacked graph,
        # which iterates `df_plot.index` (also Timestamps), can look up
        # directly without conversion.
        self._line_outstanding = {
            ts: int(v) for ts, v in zip(date_index, outstanding.tolist())
        }

        plt.figure(figsize=(15, 10))
        color_match = {'new': 'orange', 'outstanding': 'red',
                       'resolved': 'green', 'priority': 'pink'}
        curves = {
            'new': new_vals,
            'outstanding': outstanding,
            'resolved': resolved_vals,
            'priority': priority,
        }
        for categ, color in color_match.items():
            plt.plot(
                date_index, curves[categ], marker='o', color=color,
                label=(
                    f'{categ.title().replace("Priority", "Outstanding Priority")}'
                    f' Queries'
                ),
            )

        plt.title(f'{network} Queries Over Time')
        plt.xlabel('Date')
        plt.ylabel('Query Count')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()

        graph_path = (
            f'{self.tracker_abs_path}/{network}/combined/'
            f'{network}_queries_line_graph.png'
        )
        plt.savefig(graph_path, dpi=800)
        plt.close()
        self.upload_to_dropbox(graph_path, network)

    def collect_errors_per_site(
        self, df: pd.DataFrame, network: str
    ) -> None:
        """
        Collects number of unresolved errors
        from each site and categorizes them
        by errors that were detected less than
        or more than 30 days ago, and by priority.

        Parameters
        --------------
        df : pd.DataFrame
            Dataframe with Errors
        network : str
            Current network (PRONET or PRESCIENT)
        """

        df.columns = df.columns.str.replace(' ', '_')
        for row in df.itertuples():
            if "SH" in row.Participant:
                continue
            priority = False
            self.site_errors.setdefault(network, {})
            self.site_errors[network].setdefault(
                row.Participant[:2],
                {'under_thirty': 0, 'over_thirty': 0,
                 'priority': 0, 'non_priority': 0})

            if (getattr(row, 'Date_Resolved') != ''
                    or getattr(row, 'Manually_Resolved') != ''):
                continue

            # v2 stores days-since-detection directly. Empty / non-numeric
            # values fall through to under_thirty — picked over skipping
            # the row so a flag with a blank `Days Since Detected` still
            # contributes to its site's count (visibility > precision for
            # the bar graph).
            dsd_raw = getattr(row, 'Days_Since_Detected')
            try:
                diff = int(float(dsd_raw))
            except (TypeError, ValueError):
                diff = 0

            if getattr(row, 'Priority_Item', '') in PRIORITY_TRUTHY:
                priority = True

            if diff > 30:
                self.site_errors[network][row.Participant[:2]]['over_thirty'] += 1
            else:
                self.site_errors[network][row.Participant[:2]]['under_thirty'] += 1
            if priority is True:
                self.site_errors[network][row.Participant[:2]]['priority'] += 1
            else:
                self.site_errors[network][row.Participant[:2]]['non_priority'] += 1

    def visualize_errors(self, network: str) -> None:
        """
        Creates stacked bar graph of
        errors for each site of the
        current network. Saves graph
        as a PNG.

        Parameters
        ----------
        network : str
            Current network (PRONET or PRESCIENT)
        """

        categories = list(self.site_errors[network].keys())
        graph_data = {'over_thirty': [], 'under_thirty': [],
                      'priority': [], 'non_priority': []}

        raw_data_list = []

        for site, errors in self.site_errors[network].items():
            for label in graph_data.keys():
                graph_data[label].append(errors[label])
            errors['site'] = site
            raw_data_list.append(errors)

        raw_data_df = pd.DataFrame(raw_data_list)
        raw_data_df = raw_data_df[['site', 'priority', 'non_priority']]
        csv_path = (
            f'{self.tracker_abs_path}/{network}/combined/'
            f'{network}_errors_per_site.csv'
        )
        excel_path = (
            f'{self.tracker_abs_path}/{network}/combined/'
            f'{network}_errors_per_site_daily_updates.xlsx'
        )
        self.save_to_excel(raw_data_df, excel_path, str(date.today()))
        raw_data_df.to_csv(csv_path, index=False)
        self.upload_to_dropbox(csv_path, network)
        self.upload_to_dropbox(excel_path, network)

        plt.figure(figsize=(15, 9))

        bar_width = 0.35
        indices = np.arange(len(categories))

        plt.bar(indices, graph_data['over_thirty'], width=bar_width,
                label='Detected Over 30 Days Ago', color='#DE3C20')

        plt.bar(indices, graph_data['under_thirty'],
                bottom=graph_data['over_thirty'], width=bar_width,
                label='Detected Under 30 Days Ago', color='#EA9A4A')

        offset = bar_width
        plt.bar(indices + offset, graph_data['priority'], width=bar_width,
                label='High Priority', color='#e188da')

        plt.bar(indices + offset, graph_data['non_priority'],
                width=bar_width, bottom=graph_data['priority'],
                label='Lower Priority', color='#888fe1')

        plt.xticks(indices + offset / 2, categories)

        plt.legend()
        plt.title((f'{network.replace("PRONET","ProNET")} Queries'
                   ' Per Site (Any Forms That Have Unresolved'
                   ' Queries Will Not Be Uploaded to the NDA)'))
        plt.xlabel('Site')
        plt.ylabel('Unresolved Queries')
        graph_path = (
            f'{self.tracker_abs_path}/{network}/combined/'
            f'{network}_queries_graph.png'
        )
        plt.savefig(graph_path, dpi=600)
        plt.close()
        self.upload_to_dropbox(graph_path, network)

    def upload_to_dropbox(self, local_path: str, network: str) -> None:
        """
        Reads the graph from the local file
        and uploads it to dropbox.

        Parameters
        -----------------
        local_path : str
            Path to local graph file
        network : str
            current network (PRONET or PRESCIENT)
        """

        dbx = self.utils.collect_dropbox_credentials()
        with open(local_path, 'rb') as f:
            print(local_path)
            # Preserve the local filename in the Dropbox destination so the
            # line graph, stacked graph, bar graph, csv, and xlsx all land
            # under distinct names. The previous version uploaded
            # everything to `{network}_queries_graph.png`, clobbering itself.
            dbx.files_upload(
                f.read(),
                f'{self.dropbox_base}/{network}/combined/{local_path.split("/")[-1]}',
                mode=dropbox.files.WriteMode.overwrite,
            )

    def save_to_excel(self, df, file_path, sheet_name):
        # Append today's snapshot to the daily-updates xlsx (one sheet per
        # day). If the file is brand new, create it; otherwise open in
        # append mode and create a new sheet for today.
        if os.path.exists(file_path):
            with pd.ExcelWriter(file_path, engine="openpyxl",
                                mode="a", if_sheet_exists="new") as writer:
                df.to_excel(writer, index=False, sheet_name=sheet_name)
        else:
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name=sheet_name)


if __name__ == '__main__':
    GraphErrors().run_script()
