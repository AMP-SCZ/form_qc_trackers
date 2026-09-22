"""
Data preparation for the interactive_graphs dashboard.

Applies the data-shaping logic of `visualize_data/graph_errors.py` to
produce three long-format dataframes per network (PRONET / PRESCIENT)
that the Dash app renders as interactive charts:

  - `stacked_long`: per-day outstanding queries by Timepoint
    (mirrors GraphErrors.create_stacked_line_graph)
  - `line_long`:    per-day new / outstanding / resolved / priority
    (mirrors GraphErrors.create_line_graph)
  - `site_long`:    per-site over_thirty / under_thirty / priority
    (mirrors GraphErrors.collect_errors_per_site / visualize_errors)

Source of truth is the Dropbox revision history of
`{network}_Output_V2.xlsx` — `Days Since Detected` in the current
tracker is unreliable as a per-flag detection date (carries forward
without recompute on continuously-open flags, blank on legacy rows,
resets on re-open), so the revision walk gives a per-flag
(Earliest_seen, Latest_seen) tuple keyed on
(Participant, Timepoint, Form, Flags) that actually reflects the
tracker's lifecycle.

Falls back gracefully:
  1. Revision walk via GraphErrors (most accurate, needs Dropbox creds)
  2. Current-tracker-only stacked/line + per-site bar (no creds needed)
  3. Synthetic data (no tracker available — dashboard still renders)

Caching: shaped frames are saved to `cache/` next to this module so
the dashboard can reload without re-walking revisions on every start.
Set `force_refresh=True` on `build()` to bypass.
"""
import os
import sys
import json
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import dropbox
except Exception:
    dropbox = None

# Project root on sys.path so `from utils.utils import Utils` and
# `from visualize_data.graph_errors import GraphErrors` resolve.
# `interactive_graphs/` may live as a sibling of `visualize_data/`
# under the project root, OR inside `visualize_data/`. Walk up from
# this file looking for the directory that contains BOTH
# `visualize_data/` and `utils/` as siblings — that's the project
# root regardless of where interactive_graphs/ is nested.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent  # fallback
_candidate = _HERE
for _ in range(6):  # cap the walk so symlinked / detached layouts don't loop
    _candidate = _candidate.parent
    if (_candidate / 'visualize_data').is_dir() and (_candidate / 'utils').is_dir():
        _PROJECT_ROOT = _candidate
        break
    if _candidate == _candidate.parent:  # hit filesystem root
        break
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Imports below are deferred-by-try so a missing optional dep (e.g. the
# `dropbox` package, or a missing utils module in a thin test env) still
# lets ProcessData fall back to synthetic data instead of erroring at
# import time.
try:
    from visualize_data.graph_errors import GraphErrors, PRIORITY_TRUTHY
    _GRAPHER_AVAILABLE = True
except Exception as _imp_err:
    GraphErrors = None
    PRIORITY_TRUTHY = (True, 'True', 'true', 'TRUE', 1, '1')
    _GRAPHER_AVAILABLE = False
    print(f'[ProcessData] GraphErrors import failed ({_imp_err}); '
          'will use tracker-only or synthetic data.')


CACHE_VERSION = '3'  # bump when cached column schema or window changes
REVISION_CACHE_VERSION = '1'  # bump when revision-cache schema changes
# Must match the `start_date` hardcoded in
# `visualize_data/graph_errors.py:create_line_graph` /
# `create_stacked_line_graph` — that file is the source of truth for
# the static PNG window and the dashboard must use the same window so
# the interactive curves match the published PNGs.
DEFAULT_START_DATE = pd.Timestamp(2026, 4, 15)


class _IncrementalRevisionCache:
    """
    Per-network parquet cache of the per-flag history that
    GraphErrors._fetch_revision_history would otherwise rebuild from
    scratch on every walk.

    On first run, the FULL walk runs (same cost as `graph_errors.py`)
    and the result is persisted. On subsequent runs, the Dropbox walk
    stops as soon as it reaches a revision older than the cache's
    `newest_processed_ts` — usually a handful of new revisions vs.
    the hundreds a fresh walk would do. At 5 years of study history
    this drops refresh from ~10h to seconds.

    Storage:
      - `{cache_dir}/revisions/{network}_flags.parquet`: one row per
        unique (Participant, Timepoint, Form, Flags). Columns match
        `_fetch_revision_history`'s return value exactly so the
        downstream code paths don't care whether the data came from
        a full walk or an incremental merge.
      - `{cache_dir}/revisions/{network}_meta.json`: newest_processed_ts
        (ISO 8601 with tz), last_built_at, kept_new_revisions counter,
        version. Bumping `REVISION_CACHE_VERSION` invalidates without
        touching the parquet file.
    """

    KEY_COLS = ['Participant', 'Timepoint', 'Form', 'Flags']

    def __init__(self, cache_dir, network):
        rev_dir = Path(cache_dir) / 'revisions'
        self.network = network
        self.flags_path = rev_dir / f'{network}_flags.parquet'
        self.meta_path = rev_dir / f'{network}_meta.json'

    def load(self):
        """Return (flags_df, meta) or (None, None) on miss/bad cache."""
        if not self.flags_path.exists() or not self.meta_path.exists():
            return None, None
        try:
            with open(self.meta_path) as f:
                meta = json.load(f)
            if meta.get('version') != REVISION_CACHE_VERSION:
                print(f'[IncrementalCache] {self.network}: version '
                      f'{meta.get("version")} != '
                      f'{REVISION_CACHE_VERSION}; will rebuild.')
                return None, None
            flags = pd.read_parquet(self.flags_path)
            return flags, meta
        except Exception as e:
            print(f'[IncrementalCache] {self.network}: load failed ({e}).')
            return None, None

    def save(self, flags_df, meta):
        try:
            self.flags_path.parent.mkdir(
                parents=True, exist_ok=True, mode=0o700
            )
            tmp_flags = self.flags_path.with_suffix('.tmp')
            flags_df.to_parquet(tmp_flags, index=False)
            try:
                os.chmod(tmp_flags, 0o600)
            except OSError:
                pass
            tmp_flags.replace(self.flags_path)

            meta = {**meta, 'version': REVISION_CACHE_VERSION}
            tmp_meta = self.meta_path.with_suffix('.tmp')
            with open(tmp_meta, 'w') as f:
                json.dump(meta, f, indent=2, default=str)
            try:
                os.chmod(tmp_meta, 0o600)
            except OSError:
                pass
            tmp_meta.replace(self.meta_path)
            print(f'[IncrementalCache] {self.network}: saved '
                  f'{len(flags_df):,} flags, '
                  f'newest_ts={meta.get("newest_processed_ts")}')
        except Exception as e:
            print(f'[IncrementalCache] {self.network}: save failed ({e}).')


class ProcessData():
    """
    Builds the three long-format datasets the dashboard renders.

    Public attributes after `build()`:
      - `stacked_long`: date, network, timepoint, outstanding
      - `line_long`:    date, network, series, count
      - `site_long`:    network, site, over_thirty, under_thirty,
                        priority, non_priority

    All three use the same revision-history source so daily totals are
    consistent: `stacked_long.outstanding.sum()` for a given (network,
    date) equals `line_long[series=outstanding].count` for that date
    (within the cross-check tolerance `graph_errors.py` already prints).
    """

    def __init__(self, networks=('PRONET', 'PRESCIENT'),
                 use_cache=True, cache_dir=None,
                 start_date=None):
        self.networks = tuple(networks)
        self.use_cache = use_cache

        # Try to instantiate GraphErrors. Failures here (missing config,
        # missing Utils, etc.) collapse into the synthetic-data path.
        self._grapher = None
        self._init_error = None
        if _GRAPHER_AVAILABLE:
            try:
                self._grapher = GraphErrors()
            except Exception as e:
                self._init_error = e
                print(f'[ProcessData] GraphErrors() init failed ({e}); '
                      'will use synthetic data only.')

        self.start_date = (
            pd.to_datetime(start_date) if start_date is not None
            else DEFAULT_START_DATE
        )
        self.end_date = pd.Timestamp(datetime.today().date())

        if cache_dir is None:
            cache_dir = _HERE / 'cache'
        self.cache_dir = Path(cache_dir)

        self.stacked_long = pd.DataFrame()
        self.line_long = pd.DataFrame()
        self.site_long = pd.DataFrame()
        self.build_info = {
            'built_at': None,
            'sources': {},   # {network: 'history' | 'tracker' | 'synthetic'}
            'cache_version': CACHE_VERSION,
        }

    # Backward-compat: the previous version returned a single DataFrame
    # (stacked-only). New callers should use named methods; the tuple
    # return signals to old code that the shape has changed.
    def __call__(self):
        return self.build()

    # -------- public API --------

    def build(self, force_refresh=False):
        """
        Build all three datasets across the configured networks.

        Returns (stacked_long, line_long, site_long). Use `force_refresh`
        to bypass the on-disk cache and re-walk Dropbox revisions.
        """
        if self.use_cache and not force_refresh:
            cached = self._load_cache()
            if cached is not None:
                self.stacked_long, self.line_long, self.site_long, info = cached
                self.build_info = info
                print(f'[ProcessData] loaded cache from {self.cache_dir}')
                return self.stacked_long, self.line_long, self.site_long

        stacked_parts, line_parts, site_parts = [], [], []
        for network in self.networks:
            built = self._build_one_network(network)
            if built is None:
                continue
            stacked_parts.append(built['stacked'])
            line_parts.append(built['line'])
            site_parts.append(built['site'])

        self.stacked_long = (
            pd.concat(stacked_parts, ignore_index=True)
            if stacked_parts else pd.DataFrame()
        )
        self.line_long = (
            pd.concat(line_parts, ignore_index=True)
            if line_parts else pd.DataFrame()
        )
        self.site_long = (
            pd.concat(site_parts, ignore_index=True)
            if site_parts else pd.DataFrame()
        )

        if (self.stacked_long.empty and self.line_long.empty
                and self.site_long.empty):
            print('[ProcessData] no real data available; '
                  'generating synthetic.')
            self._populate_synthetic()

        self.build_info['built_at'] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + 'Z'
        self._cross_check_invariants()
        self._save_cache()
        return self.stacked_long, self.line_long, self.site_long

    def _cross_check_invariants(self):
        """
        Mirror of GraphErrors.create_stacked_line_graph's safety net:
        for each (network, date), the sum of per-timepoint outstanding
        should equal the line graph's `outstanding` series. A mismatch
        means the stacked and line frames drifted; print a warning so
        a future regression surfaces in logs.

        Synthetic data is generated independently for stacked vs line
        (per-timepoint random walks vs per-series poisson/walks) so
        the invariant doesn't hold by construction. Skip the check
        when any source is synthetic — otherwise every startup on a
        fresh clone prints a WARNING that means nothing.
        """
        if self.stacked_long.empty or self.line_long.empty:
            return
        sources = (self.build_info or {}).get('sources', {})
        if any(s == 'synthetic' for s in sources.values()):
            print('[ProcessData] cross-check skipped - synthetic data '
                  'mismatches stacked vs line by construction.')
            return
        try:
            stacked_totals = (
                self.stacked_long
                .groupby(['network', 'date'])['outstanding']
                .sum()
                .rename('stacked_total')
            )
            line_outstanding = (
                self.line_long[self.line_long['series'] == 'outstanding']
                .set_index(['network', 'date'])['count']
                .rename('line_outstanding')
            )
            joined = stacked_totals.to_frame().join(line_outstanding, how='outer')
            mismatches = joined[
                joined['stacked_total'].fillna(-1)
                != joined['line_outstanding'].fillna(-1)
            ]
            if len(mismatches):
                head = mismatches.head(5).to_dict(orient='index')
                print(f'[ProcessData] WARNING: stacked vs line outstanding '
                      f'differ on {len(mismatches)} (network, date) pairs; '
                      f'first few: {head}')
            else:
                print(f'[ProcessData] cross-check ok: stacked totals match '
                      f'line outstanding across {len(joined)} (network, date) pairs')
        except Exception as e:
            print(f'[ProcessData] cross-check skipped ({e}).')

    def get_stacked(self):
        return self.stacked_long

    def get_line(self):
        return self.line_long

    def get_site(self):
        return self.site_long

    # -------- per-network build --------

    def _build_one_network(self, network):
        """
        Pull tracker + revision history for `network`, then produce the
        three long-format frames. Returns None when neither the tracker
        nor the revision walk is reachable, so other networks still
        build.
        """
        if self._grapher is None:
            return None

        tracker_df = self._read_tracker(network)
        history_df = self._safe_fetch_history(network)

        if history_df is not None and not history_df.empty:
            stacked = self._stacked_long_from_history(history_df, network)
            line = self._line_long_from_history(history_df, network)
            self.build_info['sources'][network] = 'history'
        elif tracker_df is not None and not tracker_df.empty:
            stacked = self._stacked_long_from_tracker(tracker_df, network)
            line = self._line_long_from_tracker(tracker_df, network)
            self.build_info['sources'][network] = 'tracker'
        else:
            return None

        if tracker_df is not None and not tracker_df.empty:
            site = self._site_long_from_tracker(tracker_df, network)
        else:
            site = pd.DataFrame()

        return {'stacked': stacked, 'line': line, 'site': site}

    def _tracker_base_path(self):
        """
        Local directory the trackers live in. Mirrors what
        `generate_reports/create_trackers.py:41-42` does for
        testing_enabled: when testing, the path nests under a
        `testing/` subdir of output_path. `GraphErrors.tracker_abs_path`
        is built before that suffix is applied, so splice it in here.
        """
        g = self._grapher
        output_path = g.output_path
        if g.config_info.get("testing_enabled") == "True":
            return f'{output_path}testing/formatted_outputs/dropbox_files'
        return f'{output_path}formatted_outputs/dropbox_files'

    def _read_tracker(self, network):
        """
        Read `{network}_Output_V2.xlsx` from the local output dir.
        Returns None on failure so the caller can fall back to
        revision-history-only or synthetic.
        """
        try:
            tracker_path = (
                f'{self._tracker_base_path()}/{network}/combined/'
                f'{network}_Output_V2.xlsx'
            )
            if not os.path.exists(tracker_path):
                print(f'[ProcessData] {network}: tracker not found at '
                      f'{tracker_path}')
                return None
            return pd.read_excel(tracker_path, keep_default_na=False)
        except Exception as e:
            print(f'[ProcessData] {network}: tracker read failed ({e}).')
            return None

    def _safe_fetch_history(self, network):
        """
        Wrap the revision history walk + spike filter in a try/except
        so missing creds / SDK errors don't crash the whole build.

        Uses an on-disk incremental cache (per-flag parquet, see
        `_IncrementalRevisionCache`) so subsequent runs walk only the
        Dropbox revisions added since the last walk — first walk is
        full cost, subsequent walks are seconds-to-minutes.
        """
        g = self._grapher
        try:
            history_df = self._fetch_history_incremental(network)
        except Exception as e:
            print(f'[ProcessData] {network}: revision walk failed '
                  f'({type(e).__name__}: {str(e)[:200]}); falling back '
                  'to current tracker.')
            return None

        if history_df is None or history_df.empty:
            return history_df

        if g.filter_large_spikes:
            try:
                # Spike filter drops bulk-operation days from the
                # graphs entirely — per user direction, large_jumps
                # are not surfaced in the dashboard or logged.
                history_df = g._filter_large_spikes(history_df, network)
            except Exception as e:
                print(f'[ProcessData] {network}: spike filter failed '
                      f'({e}); keeping raw history.')
        return history_df

    def _fetch_history_incremental(self, network, sample_every_days=1,
                                   page_limit=100):
        """
        Walk Dropbox revisions newer than the cache's
        `newest_processed_ts` and merge the new open-flag set with the
        cached per-flag history. Falls back to the full
        GraphErrors._fetch_revision_history walk on first run or any
        cache miss.

        Returns a DataFrame in the same shape
        `GraphErrors._fetch_revision_history` returns (key cols +
        Priority Item + Earliest_seen + Latest_seen). Also updates
        `self._grapher._newest_revision` so the downstream
        `_per_day_counts_from_history` / `_filter_large_spikes` calls
        see the actual newest revision in this walk — not the cached
        one.

        The walk semantics mirror GraphErrors:
          - Paginate `files/list_revisions` newest-first via
            `before_rev`.
          - Apply the same `sample_every_days` rule.
          - Same rename map for older-schema revisions.
          - Same open-flag mask (Date Resolved blank AND
            Manually Resolved blank).
          - Same priority-carry-forward (when >= rec['latest']).
        The only difference is the stop condition: break out of the
        page loop as soon as a revision older than
        `cached_newest_ts` appears.
        """
        if dropbox is None:
            raise RuntimeError(
                '`dropbox` package not importable; cannot walk revisions.'
            )

        g = self._grapher
        cache = _IncrementalRevisionCache(self.cache_dir, network)
        cached_flags, cached_meta = cache.load()

        if cached_flags is None or cached_flags.empty:
            # Cache miss — full walk, then persist for next time.
            history_df = g._fetch_revision_history(
                network, sample_every_days=sample_every_days,
                page_limit=page_limit,
            )
            if history_df is not None and not history_df.empty:
                newest = g._newest_revision
                cache.save(history_df, {
                    'newest_processed_ts': (
                        newest.isoformat() if newest else None
                    ),
                    'last_built_at': datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + 'Z',
                    'kept_new_revisions': len(history_df),
                    'walk_kind': 'full',
                })
            return history_df

        cached_newest_ts = datetime.fromisoformat(
            cached_meta['newest_processed_ts']
        )
        print(f'[Incremental] {network}: cached_newest_ts='
              f'{cached_newest_ts}; walking only newer revisions.')

        # Reconstruct the unique_rows dict from the cached frame so
        # the walk's update loop can mutate it in place — same shape
        # GraphErrors uses internally.
        unique_rows = {}
        for _, row in cached_flags.iterrows():
            key = tuple(row[c] for c in _IncrementalRevisionCache.KEY_COLS)
            unique_rows[key] = {
                'earliest': pd.to_datetime(row['Earliest_seen']),
                'latest': pd.to_datetime(row['Latest_seen']),
                'priority': row['Priority Item'],
            }

        dbx = g.utils.collect_dropbox_credentials()
        if hasattr(dbx, 'check_and_refresh_access_token'):
            dbx.check_and_refresh_access_token()
        requester = g._find_dropbox_requester(dbx)
        if requester is None:
            raise RuntimeError(
                'No request_json_object found on the Dropbox client; '
                'cannot do incremental walk.'
            )

        USER_AUTH = getattr(dropbox.dropbox_client, 'USER_AUTH', None)
        if USER_AUTH is None:
            auth_type = getattr(dropbox.dropbox_client, 'AuthType', None)
            USER_AUTH = (getattr(auth_type, 'USER', None)
                         if auth_type else None)
        if USER_AUTH is None:
            USER_AUTH = 'user'

        path = (
            f'{g.dropbox_base}/{network}/combined/'
            f'{network}_Output_V2.xlsx'
        )

        key_cols = _IncrementalRevisionCache.KEY_COLS
        resolved_cols = ['Date Resolved', 'Manually Resolved']
        rename_to_current = {
            'Subject': 'Participant',
            'General Flag': 'Form',
            'Specific Flags': 'Flags',
            'Manually Marked as Resolved': 'Manually Resolved',
        }

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

            stop_requested = False
            for entry in entries:
                rev = entry['rev']
                when = datetime.fromisoformat(
                    entry['server_modified'].replace('Z', '+00:00')
                )

                # Incremental stop condition. Note: we tz-align before
                # comparing — server_modified is tz-aware UTC, cached
                # newest_ts was serialized from the same source.
                # The newest_revision update must come AFTER the stop
                # check: if Dropbox ever returns an entry at-or-older
                # than cached_newest_ts first (unexpected order, or
                # the cache is already at the newest), accepting its
                # `when` would wind newest_processed_ts BACKWARD on
                # save, and subsequent incremental walks would start
                # from that older cutoff each time. Skipping the update
                # here lets the `newest_revision is None` fallback
                # below keep the cache at cached_newest_ts.
                if when <= cached_newest_ts:
                    stop_requested = True
                    has_more = False
                    break
                if newest_revision is None or when > newest_revision:
                    newest_revision = when

                if (sample_every_days and last_kept_when is not None
                        and (last_kept_when - when).days
                        < sample_every_days):
                    continue

                try:
                    _, resp = dbx.files_download(path=path, rev=rev)
                    rev_df = pd.read_excel(
                        BytesIO(resp.content), keep_default_na=False
                    )
                except Exception as e:
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
                            rec['latest'] = when
                            rec['priority'] = pri

                last_kept_when = when
                kept += 1
                print(f'[Incremental] {network} - Revision {kept}: '
                      f'{rev} from {when}')

            if stop_requested or not entries:
                break
            before_rev = entries[-1]['rev']

        # Update _newest_revision on the GraphErrors instance so the
        # downstream `_per_day_counts_from_history` / `_outstanding_curve`
        # / `_filter_large_spikes` calls use the actual latest revision
        # in THIS walk, not the cached one. If no new revisions were
        # found, fall back to the cached newest_ts so still-open flags
        # still resolve correctly.
        if newest_revision is None:
            newest_revision = cached_newest_ts
        g._newest_revision = newest_revision

        rows = []
        for key, rec in unique_rows.items():
            rows.append({
                **dict(zip(key_cols, key)),
                'Priority Item': rec['priority'],
                'Earliest_seen': rec['earliest'],
                'Latest_seen': rec['latest'],
            })
        history_df = pd.DataFrame(rows)

        cache.save(history_df, {
            'newest_processed_ts': newest_revision.isoformat(),
            'last_built_at': datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + 'Z',
            'kept_new_revisions': kept,
            'walk_kind': 'incremental',
        })
        return history_df

    # -------- history → long format --------

    def _stacked_long_from_history(self, history_df, network):
        """
        Per-timepoint outstanding curves in long format. Matches the
        per-day series GraphErrors.create_stacked_line_graph would
        plot — same _per_day_counts_from_history / _outstanding_curve
        calls applied per-Timepoint subgroup.
        """
        g = self._grapher
        date_index = pd.date_range(self.start_date, self.end_date, freq='D')
        rows = []
        for tp, tp_df in history_df.groupby('Timepoint'):
            per_day = g._per_day_counts_from_history(
                tp_df, self.start_date, self.end_date
            )
            outstanding = g._outstanding_curve(
                per_day['earliest'],
                per_day['effective_latest'],
                date_index,
            )
            # Keep '' as-is so per-network sums match the all-rows
            # outstanding curve and the dashboard, not the data layer,
            # decides how to display the empty label.
            for d, v in zip(date_index, outstanding):
                rows.append({
                    'date': d.strftime('%Y-%m-%d'),
                    'network': network,
                    'timepoint': str(tp),
                    'outstanding': int(v),
                })
        return pd.DataFrame(rows)

    def _line_long_from_history(self, history_df, network):
        """
        Long-format frame of the four lines GraphErrors.create_line_graph
        plots (new / outstanding / resolved / priority). Outstanding is
        computed from the (Earliest_seen, Latest_seen) bounds; priority
        is the same curve restricted to rows where Priority Item is
        truthy.
        """
        g = self._grapher
        date_index = pd.date_range(self.start_date, self.end_date, freq='D')
        per_day = g._per_day_counts_from_history(
            history_df, self.start_date, self.end_date
        )
        outstanding = g._outstanding_curve(
            per_day['earliest'], per_day['effective_latest'], date_index
        )

        if 'Priority Item' in history_df.columns and not history_df.empty:
            priority_history = history_df[
                history_df['Priority Item'].isin(PRIORITY_TRUTHY)
            ]
        else:
            priority_history = history_df.iloc[0:0]
        pri_per_day = g._per_day_counts_from_history(
            priority_history, self.start_date, self.end_date
        )
        priority = g._outstanding_curve(
            pri_per_day['earliest'],
            pri_per_day['effective_latest'],
            date_index,
        )

        series_map = {
            'new': per_day['new_per_day'].to_numpy(),
            'outstanding': outstanding,
            'resolved': per_day['resolved_per_day'].to_numpy(),
            'priority': priority,
        }
        rows = []
        for series, values in series_map.items():
            for d, v in zip(date_index, values):
                rows.append({
                    'date': d.strftime('%Y-%m-%d'),
                    'network': network,
                    'series': series,
                    'count': int(v),
                })
        return pd.DataFrame(rows)

    # -------- tracker fallback → long format --------

    def _stacked_long_from_tracker(self, tracker_df, network):
        """
        Best-effort stacked-area data when no revision history is
        available. Walks day-by-day from start_date to today, treating
        a flag as outstanding on day `d` if Date Resolved is blank or
        > d AND Manually Resolved is blank AND its reconstructed
        detection date is <= d.

        Detection date is `today - Days Since Detected`, which is what
        GraphErrors._detection_date_series uses. Less accurate than the
        revision walk (continuously-open flags carry their original
        detection date forward, blank rows are treated as always-
        outstanding) but stays internally consistent with how the V2
        tracker stores time.
        """
        df = tracker_df.copy()
        today = pd.to_datetime(datetime.today().date())
        rows = []
        # Coerce columns once outside the loop. Manually Resolved is
        # compared against literal '' to mirror graph_errors._get_outstanding_as_of
        # (graph_errors.py:341) — a `.str.strip()` here would treat
        # whitespace-only entries as resolved and silently diverge.
        date_resolved = pd.to_datetime(df['Date Resolved'], errors='coerce')
        manually_resolved = df['Manually Resolved']
        days_since = pd.to_numeric(df['Days Since Detected'], errors='coerce')
        detection = today - pd.to_timedelta(days_since, unit='D')
        timepoint = df['Timepoint'].astype(str)

        for d in pd.date_range(self.start_date, self.end_date, freq='D'):
            mask = (
                (detection.isna() | (detection <= d))
                & (date_resolved.isna() | (date_resolved > d))
                & (manually_resolved == '')
            )
            counts = timepoint[mask].value_counts()
            for tp, n in counts.items():
                rows.append({
                    'date': d.strftime('%Y-%m-%d'),
                    'network': network,
                    'timepoint': tp if tp else '',
                    'outstanding': int(n),
                })
        return pd.DataFrame(rows)

    def _line_long_from_tracker(self, tracker_df, network):
        """
        Best-effort line-graph data when no revision history is
        available. Only `outstanding` and `priority` can be computed
        with reasonable accuracy from a single tracker snapshot —
        `new` / `resolved` per-day require knowing the lifecycle. Those
        two series come through as zeros so the chart still draws.
        """
        df = tracker_df.copy()
        today = pd.to_datetime(datetime.today().date())
        date_resolved = pd.to_datetime(df['Date Resolved'], errors='coerce')
        # Literal '' to mirror graph_errors._get_outstanding_as_of.
        manually_resolved = df['Manually Resolved']
        days_since = pd.to_numeric(df['Days Since Detected'], errors='coerce')
        detection = today - pd.to_timedelta(days_since, unit='D')
        if 'Priority Item' in df.columns:
            priority_mask = df['Priority Item'].isin(PRIORITY_TRUTHY)
        else:
            priority_mask = pd.Series(False, index=df.index)

        rows = []
        for d in pd.date_range(self.start_date, self.end_date, freq='D'):
            outstanding_mask = (
                (detection.isna() | (detection <= d))
                & (date_resolved.isna() | (date_resolved > d))
                & (manually_resolved == '')
            )
            outstanding = int(outstanding_mask.sum())
            priority = int((outstanding_mask & priority_mask).sum())
            for series, val in (
                ('new', 0),
                ('outstanding', outstanding),
                ('resolved', 0),
                ('priority', priority),
            ):
                rows.append({
                    'date': d.strftime('%Y-%m-%d'),
                    'network': network,
                    'series': series,
                    'count': val,
                })
        return pd.DataFrame(rows)

    def _site_long_from_tracker(self, tracker_df, network):
        """
        Per-site snapshot in long format. Reuses
        GraphErrors.collect_errors_per_site but resets `site_errors`
        for the network first so repeated builds don't accumulate.
        """
        g = self._grapher
        df_copy = tracker_df.copy()
        g.site_errors[network] = {}
        try:
            g.collect_errors_per_site(df_copy, network)
        except Exception as e:
            print(f'[ProcessData] {network}: collect_errors_per_site '
                  f'failed ({e}).')
            return pd.DataFrame()

        rows = []
        for site, counts in g.site_errors.get(network, {}).items():
            rows.append({
                'network': network,
                'site': site,
                'over_thirty': int(counts.get('over_thirty', 0)),
                'under_thirty': int(counts.get('under_thirty', 0)),
                'priority': int(counts.get('priority', 0)),
                'non_priority': int(counts.get('non_priority', 0)),
            })
        return pd.DataFrame(rows)

    # -------- cache --------

    def _cache_paths(self):
        # Parquet over CSV to preserve dtypes — the `date` column round-
        # trips as datetime64 instead of string, which kills the
        # `pd.to_datetime` cost the dashboard pays on every callback.
        return {
            'stacked': self.cache_dir / 'stacked_long.parquet',
            'line': self.cache_dir / 'line_long.parquet',
            'site': self.cache_dir / 'site_long.parquet',
            'meta': self.cache_dir / 'build_info.json',
        }

    def _load_cache(self):
        p = self._cache_paths()
        if not all(p[k].exists() for k in ('stacked', 'line', 'site', 'meta')):
            return None
        try:
            with open(p['meta'], 'r') as f:
                info = json.load(f)
            if info.get('cache_version') != CACHE_VERSION:
                print('[ProcessData] cache version mismatch; rebuilding.')
                return None
            stacked = pd.read_parquet(p['stacked'])
            line = pd.read_parquet(p['line'])
            site = pd.read_parquet(p['site'])
            return stacked, line, site, info
        except Exception as e:
            print(f'[ProcessData] cache load failed ({e}); rebuilding.')
            return None

    def _save_cache(self):
        """
        Write the three parquet frames + meta JSON. Each file lands at
        `*.tmp` first and is then atomically renamed into place; the
        meta JSON is written LAST because `_load_cache` gates on its
        presence. Under SIGTERM mid-write, the worst case is a stale
        `*.tmp` left on disk — `_load_cache` will still load the
        previous complete cache.

        Cache files are chmod'd 0o600 so they aren't world-readable on
        a shared host (these are aggregate counts, not PHI, but they
        carry the same operational sensitivity as the dashboard view).
        """
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            p = self._cache_paths()
            writes = [
                (self.stacked_long, p['stacked']),
                (self.line_long, p['line']),
                (self.site_long, p['site']),
            ]
            for df, path in writes:
                out = df.copy()
                if not out.empty and 'date' in out.columns:
                    out['date'] = pd.to_datetime(out['date'], errors='coerce')
                tmp = path.with_suffix(path.suffix + '.tmp')
                out.to_parquet(tmp, index=False)
                try:
                    os.chmod(tmp, 0o600)
                except OSError:
                    pass  # Windows / fs without chmod — best effort
                tmp.replace(path)

            meta_tmp = p['meta'].with_suffix(p['meta'].suffix + '.tmp')
            with open(meta_tmp, 'w') as f:
                json.dump(self.build_info, f, indent=2)
            try:
                os.chmod(meta_tmp, 0o600)
            except OSError:
                pass
            meta_tmp.replace(p['meta'])

            print(f'[ProcessData] cache written to {self.cache_dir}')
        except Exception as e:
            print(f'[ProcessData] cache save failed ({e}).')

    # -------- synthetic fallback --------

    def _populate_synthetic(self):
        """
        Generate plausible-looking synthetic versions of the three
        datasets so the dashboard can render on a fresh clone with no
        tracker / Dropbox access. Numbers are not meaningful — they're
        just smooth-enough for visual sanity.
        """
        rng = np.random.default_rng(42)
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        timepoints = ['screening', 'baseline', 'month1', 'month2',
                      'month3', 'month6', 'month12']
        series_list = ['new', 'outstanding', 'resolved', 'priority']
        sites_per_network = {
            'PRONET': ['BI', 'KC', 'YA', 'PA', 'SD', 'IR', 'NL',
                       'WU', 'LA', 'MU'],
            'PRESCIENT': ['BM', 'CG', 'ME', 'JE', 'SG', 'HK', 'LS', 'CP'],
        }

        stacked_rows, line_rows, site_rows = [], [], []
        for n in self.networks:
            for tp in timepoints:
                base = int(rng.integers(20, 200))
                walk = rng.normal(0, 5, len(dates)).cumsum()
                vals = np.clip(base + walk, 0, None).astype(int)
                for d, v in zip(dates, vals):
                    stacked_rows.append({
                        'date': d.strftime('%Y-%m-%d'),
                        'network': n, 'timepoint': tp,
                        'outstanding': int(v),
                    })
            for s in series_list:
                if s == 'new':
                    vals = np.clip(rng.poisson(30, len(dates)), 0, None)
                elif s == 'resolved':
                    vals = np.clip(rng.poisson(28, len(dates)), 0, None)
                else:
                    base = int(rng.integers(200, 1500))
                    vals = np.clip(
                        base + rng.normal(0, 10, len(dates)).cumsum(),
                        0, None
                    ).astype(int)
                for d, v in zip(dates, vals):
                    line_rows.append({
                        'date': d.strftime('%Y-%m-%d'),
                        'network': n, 'series': s, 'count': int(v),
                    })
            for site in sites_per_network.get(n, []):
                site_rows.append({
                    'network': n, 'site': site,
                    'over_thirty': int(rng.integers(5, 80)),
                    'under_thirty': int(rng.integers(5, 50)),
                    'priority': int(rng.integers(2, 30)),
                    'non_priority': int(rng.integers(20, 100)),
                })
            self.build_info['sources'][n] = 'synthetic'

        self.stacked_long = pd.DataFrame(stacked_rows)
        self.line_long = pd.DataFrame(line_rows)
        self.site_long = pd.DataFrame(site_rows)


if __name__ == '__main__':
    pd_inst = ProcessData(use_cache=False)
    stacked, line, site = pd_inst.build()
    print('--- stacked_long ---')
    print(stacked.head())
    print(stacked.shape)
    print('--- line_long ---')
    print(line.head())
    print(line.shape)
    print('--- site_long ---')
    print(site.head())
    print(site.shape)
    print('sources:', pd_inst.build_info['sources'])
