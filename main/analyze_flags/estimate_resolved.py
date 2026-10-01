"""Collect the newest daily Dropbox snapshots of the combined trackers and
emit a deduplicated **entry-level** history per network — one row per
contiguous occurrence of a unique
(Subject, Timepoint, General_Flag, variable, canonical_template) flag,
with the date bounds and source tracker(s) that contributed to it.

Both the V1 tracker (``{network}_Output.xlsx``) and the V2 tracker
(``{network}_Output_V2.xlsx``) are walked for both PRESCIENT and
PRONET. Per-revision dataframes are normalized to the V2 column
schema, Timepoint is collapsed to V2 vocabulary, the General_Flag
suffix is stripped and the form name collapsed to its comparison key
(lowercase, spaces/underscores/hyphens removed — V1 capitalized and
spaced form names differently than V2), and each row's Specific_Flags
is exploded into its constituent ``{variable} : {message}`` entries
before keying.

Why entry-level: the prior row-level keying could not represent
"variable A's flag was resolved last week but variable B's flag in
the same row is still open" — the row's Latest_seen reflected
whichever entry persisted longest, so downstream
``currently_outstanding`` and resolved-value stats over-counted.
Keying at the entry level fixes this at the source.

Only the ``Main Report`` worksheet is read from each selected revision. By
default, one snapshot per source per America/New_York calendar day is processed:
the newest available revision that day. Intraday changes between those snapshots
are intentionally omitted. Unreadable historical workbooks are skipped and
recorded in ``history_gaps_{network}.csv``; the newest snapshot must be readable.
Skipped snapshots are unknown observations, not empty reports. Episodes describe
the readable observations: flags or disappearance/reappearance events visible
only inside a gap cannot be recovered. ``skip_unreadable_history=False`` restores
strict rejection of any unreadable selected workbook. ``sample_every_days=0``
retains every revision. A
companion tracker metadata file records the newest successfully processed
revision per source. Openness, Latest_seen, and the raw message used for
before-value recovery are aligned to the network's production source
(PRESCIENT V2; PRONET V1). PRONET V2-only testing entries remain in history but
are explicitly ineligible for resolved-value reporting; PRESCIENT V1-only
history remains eligible because V1 was the production tracker before
promotion.

Manual-review artifacts (see ``manual_review.py``):
  * ``template_mapping.xlsx`` is loaded at the start of the run and
    applied BEFORE entry keying, so operator-declared renames
    (template -> template, variable -> variable) merge into a single
    episode instead of a fake resolve + fake new flag pair. The
    workbook is re-seeded at the end of the run with every template /
    variable observed, preserving existing ``maps_to`` entries.
  * ``jumps_{network}.xlsx`` records every mass addition/removal
    (>= ``jump_threshold`` entries changing between two consecutive
    processed revisions of one source) with the exact affected entry
    keys, for the operator to include/exclude downstream.
  * ``revision_counts_{network}.csv`` is the per-revision audit trail
    (open-entry count + delta per processed revision).
"""

import copy
import hashlib
import heapq
from functools import lru_cache

import pandas as pd
import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
import dropbox
from io import BytesIO
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from zipfile import BadZipFile
from xml.etree.ElementTree import ParseError

# Splitting a Windows path on ``/`` yields an empty repository root because the
# resolved path uses backslashes.  ``dirname`` is platform-aware on both Windows
# and POSIX and keeps direct script execution able to import project packages.
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)
from utils.utils import Utils
from analyze_flags.canonicalize import (
    canonicalize_message,
    explode_specific_flags,
    normalize_general_flag,
    normalize_timepoint,
    split_entry,
    split_specific_flag_entries,
)
from analyze_flags.paths import (
    ensure_analyze_flags_artifact_dir,
    open_tracker_row_history_csv_basename,
    pipeline_output_path_from_config,
    revision_counts_csv_basename,
    tracker_metadata_json_basename,
)
from analyze_flags.value_extraction import infer_before_value
from analyze_flags.history_coverage import validate_revision_coverage
from analyze_flags.history_publication import (
    atomic_write_csv, atomic_write_json, file_fingerprint, publish_staged_files,
)
from analyze_flags.paths import jumps_workbook_basename, template_mapping_basename
from analyze_flags.manual_review import (
    load_template_mapping,
    update_template_mapping,
    write_jumps_workbook,
)


class ResolvedEstimator():
    NETWORKS = ['PRESCIENT', 'PRONET']

    # PRONET's V2 file lives only under refactoring_tests/ until it is
    # promoted. Mirrors extract_blank_flags.py:PRONET_V2_BASE_OVERRIDE so
    # the two scripts walk the same set of paths.
    PRONET_V2_BASE_OVERRIDE = '/Apps/Automated QC Trackers/refactoring_tests/'

    # Tracker source labels stored per entry so consumers can determine
    # "still open in V2 specifically" vs "last seen in V1 only".
    SOURCE_V1 = 'V1'
    SOURCE_V2 = 'V2'

    # Per-network LIVE tracker. For PRESCIENT, V2 has been promoted
    # to production (the V2 tracker is at the production base path).
    # For PRONET, V2 still lives only in /refactoring_tests/ and is in
    # testing — V1 (PRONET_Output.xlsx at the production base) remains
    # the live tracker that QC analysts use day to day. Its concrete path is
    # PRONET/combined/PRONET_Output.xlsx under that base.
    #
    # `is_currently_open` and the entry's reported Earliest_seen /
    # Latest_seen are determined against the live source's per-source
    # dates when available. Entries that ONLY appear in the non-live
    # source (e.g. a flag that only ever existed in PRONET V2 testing)
    # still appear in the merged history but with is_currently_open=False
    # since they're not in the live tracker.
    LIVE_SOURCE_PER_NETWORK = {
        'PRESCIENT': SOURCE_V2,
        'PRONET': SOURCE_V1,
    }

    # History represents the Main Report only. Other report and backup tabs
    # must not contribute flags or affect whether a revision is usable.
    HISTORY_REPORT_SHEET = 'Main Report'
    HISTORY_DAY_TIMEZONE = 'America/New_York'

    # Bound temporary parsing work to one revision. Flag entries frequently
    # repeat verbatim across subjects; mapping and episode state stay uncached.
    _PARSED_FLAGS_CACHE_SIZE = 1024

    def __init__(self, sample_every_days=1, jump_threshold=50,
                 day_timezone=HISTORY_DAY_TIMEZONE, skip_unreadable_history=True):
        # Validate before accessing configuration or creating output folders.
        ZoneInfo(day_timezone)
        self.day_timezone = day_timezone
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json', 'r') as file:
            self.config_info = json.load(file)
        self.output_path = pipeline_output_path_from_config(self.config_info)
        if self.config_info["testing_enabled"] == "True":
            self.dropbox_base = '/Apps/Automated QC Trackers/refactoring_tests/'
        else:
            self.dropbox_base = '/Apps/Automated QC Trackers/'
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.depend_path = self.config_info['paths']['dependencies_path']
        self.analyze_flags_dir = ensure_analyze_flags_artifact_dir(self.output_path)
        # Default 1 selects the newest snapshot per local calendar day. Zero
        # processes every revision; values >1 retain the legacy elapsed-day gap.
        self.sample_every_days = sample_every_days
        self.skip_unreadable_history = skip_unreadable_history
        self._history_gaps_by_network = {}
        # Minimum entries added OR removed between two consecutive
        # processed revisions of one source for the change to be
        # recorded as a reviewable jump in jumps_{network}.xlsx.
        self.jump_threshold = jump_threshold
        # Operator-maintained rename mappings, applied before keying.
        self.template_map, self.variable_map = load_template_mapping(
            self.analyze_flags_dir
        )
        # Everything observed this run, for re-seeding the mapping
        # workbook. Counts are unique episodes (not sightings) so the
        # numbers match the history CSV row counts.
        self._seen_templates = {}
        self._seen_variables = {}
        self._current_network = None

    def run_script(self):
        """Stage history and gap audits together; preserve prior outputs on failure."""
        artifact_dir = Path(self.analyze_flags_dir).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if (artifact_dir / '.history-publication.json').exists():
            raise RuntimeError(
                'Interrupted history publication requires recovery from '
                f"{artifact_dir / '.history-publication.json'}")
        lock_path = artifact_dir / '.history-collection.lock'
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(
                f'Another history collection is active, or an interrupted run '
                f'left its lock: {lock_path}') from error
        os.close(lock_fd)
        original_dir = self.analyze_flags_dir
        failures = []
        seen_before = (copy.deepcopy(self._seen_templates),
                       copy.deepcopy(self._seen_variables))
        published = False
        try:
            with tempfile.TemporaryDirectory(
                    prefix='.history-stage-', dir=artifact_dir) as stage_dir:
                manual_names = [template_mapping_basename()] + [
                    jumps_workbook_basename(network) for network in self.NETWORKS]
                manual_fingerprints = {}
                for name in manual_names:
                    source = artifact_dir / name
                    manual_fingerprints[name] = file_fingerprint(source)
                    if source.exists():
                        shutil.copy2(source, Path(stage_dir) / name)
                # Use precisely the mapping snapshot that will be reseeded.
                if manual_fingerprints[template_mapping_basename()] is not None:
                    self.template_map, self.variable_map = load_template_mapping(stage_dir)
                self._seen_templates, self._seen_variables = {}, {}
                staged_names = []
                for network in self.NETWORKS:
                    network_seen = (copy.deepcopy(self._seen_templates),
                                    copy.deepcopy(self._seen_variables))
                    self._current_network = network
                    try:
                        result = self._collect_network(network)
                        rows, newest, jumps, affected, revisions = result
                        validate_revision_coverage(
                            artifact_dir, network, revisions,
                            sample_every_days=getattr(self, 'sample_every_days', 1),
                            day_timezone=getattr(
                                self, 'day_timezone', self.HISTORY_DAY_TIMEZONE))
                        self._seen_templates, self._seen_variables = (
                            copy.deepcopy(network_seen[0]),
                            copy.deepcopy(network_seen[1]))
                        self._track_reconciled_episodes(rows)
                        self.analyze_flags_dir = stage_dir
                        self._write_history_csv(network, rows, newest)
                        self._write_history_gaps(network)
                        self._write_metadata_json(network, newest)
                        write_jumps_workbook(stage_dir, network, jumps, affected)
                        self._write_revision_counts(network, revisions)
                        staged_names.extend([
                            open_tracker_row_history_csv_basename(network),
                            tracker_metadata_json_basename(network),
                            jumps_workbook_basename(network),
                            revision_counts_csv_basename(network),
                            self._history_gaps_basename(network),
                        ])
                    except Exception as error:
                        self._seen_templates, self._seen_variables = network_seen
                        failures.append(f'{network}: {type(error).__name__}: {error}')
                        print(f'WARNING: {failures[-1]}; prior artifacts preserved.')
                    finally:
                        self.analyze_flags_dir = original_dir
                if failures:
                    # Mapping counts span both networks. Publishing only one
                    # network would lose the failed network's contribution and
                    # produce a mixed generation, so retain the whole bundle.
                    raise RuntimeError(
                        'History collection failed for ' + '; '.join(failures))
                if staged_names:
                    update_template_mapping(
                        stage_dir, self._seen_templates, self._seen_variables)
                    staged_names.append(template_mapping_basename())
                    publish_staged_files(
                        artifact_dir, stage_dir, staged_names, manual_fingerprints)
                    published = True
                    gap_count = sum(len(gaps) for network, gaps in
                                    getattr(self, '_history_gaps_by_network', {}).items()
                                    if network in self.NETWORKS)
                    if gap_count:
                        print('Published history bundle for ' + ', '.join(self.NETWORKS)
                              + f' with {gap_count} unreadable historical snapshots '
                              'skipped. See history_gaps_<network>.csv; flags or '
                              'changes confined to those gaps may be missing.')
                    else:
                        print('Published complete history bundle for '
                              + ', '.join(self.NETWORKS))
        finally:
            self.analyze_flags_dir = original_dir
            if not published:
                self._seen_templates, self._seen_variables = seen_before
            lock_path.unlink(missing_ok=True)

    def _collect_network(self, network):
        """Collect in memory, allowing only explicitly audited historical read gaps."""
        if not hasattr(self, '_history_gaps_by_network'):
            self._history_gaps_by_network = {}
        self._history_gaps_by_network.pop(network, None)
        previous_sources = self._previously_recorded_sources(network)
        unique_rows = {}
        newest_per_source = {self.SOURCE_V1: None, self.SOURCE_V2: None}
        jump_rows, affected_rows, revision_rows = [], [], []
        errors = []
        network_gaps = []
        failed_selected_sources = set()
        for base, basename, source_label in self._tracker_paths(network):
            full_path = f'{base}{network}/combined/{basename}'
            candidate_rows = copy.deepcopy(unique_rows)
            candidate_jumps = list(jump_rows)
            candidate_affected = list(affected_rows)
            candidate_revisions = list(revision_rows)
            seen_before = (copy.deepcopy(self._seen_templates),
                           copy.deepcopy(self._seen_variables))
            revision_failures = []
            source_gaps = []
            walk_progress = {}
            try:
                rejected, processed, listed = self._walk_revisions(
                    full_path, source_label, candidate_rows, candidate_jumps,
                    candidate_affected, candidate_revisions, days_back=None,
                    revision_failures=revision_failures, revision_gaps=source_gaps,
                    walk_progress=walk_progress)
                if (rejected or processed is None
                        or (listed is not None and processed < listed)):
                    raise RuntimeError(
                        f'incomplete {source_label} history '
                        f'(processed={processed}, listed={listed}, rejected={rejected})')
            except Exception as error:
                self._seen_templates, self._seen_variables = seen_before
                # An absent optional tracker can remain absent, but a known
                # listed history with a fatal error must not be hidden by
                # dropping that source from a first-time collection.
                if revision_failures or walk_progress.get('listed_revisions', 0):
                    failed_selected_sources.add(source_label)
                details = ''
                if revision_failures:
                    details = (
                        f'; failed selected revisions={len(revision_failures)}; '
                        + '; '.join(revision_failures[:5]))
                    if len(revision_failures) > 5:
                        details += (
                            f'; {len(revision_failures) - 5} additional failures '
                            '(see WARNING lines above)')
                errors.append(f'{source_label}: {type(error).__name__}: {error}' + details)
                print(f'WARNING: failed walking {full_path}: {errors[-1]}')
                continue
            unique_rows = candidate_rows
            jump_rows, affected_rows = candidate_jumps, candidate_affected
            revision_rows = candidate_revisions
            newest_per_source[source_label] = processed
            network_gaps.extend({**gap, 'network': network} for gap in source_gaps)
        fresh_sources = {source for source, newest in newest_per_source.items()
                         if newest is not None}
        required = (previous_sources | failed_selected_sources
                    | {self.LIVE_SOURCE_PER_NETWORK[network]})
        missing = required - fresh_sources
        if missing:
            raise RuntimeError(
                f'No complete, fresh history for required sources {sorted(missing)}; '
                + '; '.join(errors))
        self._history_gaps_by_network[network] = network_gaps
        return unique_rows, newest_per_source, jump_rows, affected_rows, revision_rows

    def _previously_recorded_sources(self, network):
        """Sources that must remain represented in a replacement history.

        Metadata is authoritative when present. The history CSV fallback keeps
        pre-metadata artifacts safe as well. A transient failure of PRESCIENT
        V1 must not erase formerly-production V1-only resolved episodes merely
        because the current V2 production walk succeeded.
        """

        sources = set()
        metadata_path = os.path.join(
            self.analyze_flags_dir,
            tracker_metadata_json_basename(network),
        )
        try:
            with open(metadata_path, 'r') as metadata_file:
                metadata = json.load(metadata_file)
            for source in (self.SOURCE_V1, self.SOURCE_V2):
                if metadata.get(f'newest_revision_{source}'):
                    sources.add(source)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError(
                f'Cannot validate prior metadata for {network}: {error}') from error

        history_path = os.path.join(
            self.analyze_flags_dir,
            open_tracker_row_history_csv_basename(network),
        )
        try:
            history_sources = pd.read_csv(
                history_path,
                usecols=['source'],
                keep_default_na=False,
                dtype=str,
            )['source']
            for rendered in history_sources:
                sources.update(str(rendered).split('+'))
        except FileNotFoundError:
            pass
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise RuntimeError(
                f'Cannot validate prior history for {network}: {error}') from error
        return sources.intersection({self.SOURCE_V1, self.SOURCE_V2})

    def _tracker_paths(self, network):
        """(base, basename, source_label) triples to walk per network.
        Source label is stored per-entry in the merged history so
        consumers can compute is_currently_open against the right
        per-source newest revision."""
        base = self.dropbox_base
        pairs = [(base, f'{network}_Output.xlsx', self.SOURCE_V1)]
        v2_base = base
        if network == 'PRONET' and self.config_info["testing_enabled"] != "True":
            v2_base = self.PRONET_V2_BASE_OVERRIDE
        pairs.append((v2_base, f'{network}_Output_V2.xlsx', self.SOURCE_V2))
        return pairs

    @staticmethod
    def _coalesce_tracker_columns(frame, rename_v1_to_v2):
        """Normalize mixed V1/V2 sheets without discarding populated aliases."""

        normalized = frame.copy()
        for legacy, current in rename_v1_to_v2.items():
            if legacy not in normalized.columns:
                continue
            if current not in normalized.columns:
                normalized = normalized.rename(columns={legacy: current})
                continue
            current_values = normalized[current]
            current_is_blank = (
                current_values.isna()
                | current_values.astype(str).str.strip().eq('')
            )
            normalized[current] = current_values.where(
                ~current_is_blank, normalized[legacy])
            normalized = normalized.drop(columns=[legacy])
        return normalized

    def _walk_revisions(self, path, source_label, unique_rows,
                        jump_rows, affected_rows, revision_rows,
                        days_back=None, page_limit=100, revision_failures=None,
                        revision_gaps=None, walk_progress=None):
        """List revisions of ``path`` (paging beyond 100 via before_rev),
        select the newest per calendar day by default, and merge Main Report entries into
        ``unique_rows`` in place. Schema is normalized V1 -> V2 before
        any exploding so the rest of the loop only sees one shape.

        Also diffs each processed revision's entry-key set against the
        previous (newer) processed revision of the same source: every
        revision gets a row in ``revision_rows`` (count + delta), and
        deltas >= ``self.jump_threshold`` additionally get a reviewable
        row in ``jump_rows`` with their exact keys in ``affected_rows``.

        Returns the rejected-entry count, newest processed revision, and newest
        listed revision. The processed value is None when any in-scope revision
        failed outside the allowed historical-read-gap policy, so the caller
        rejects the entire staged source.
        Rebuilding history from a partial walk can erase episodes that existed
        only in the skipped revision, even when the newest snapshot was readable.
        When supplied, ``revision_failures`` receives the failed revision IDs,
        timestamps, stages, and reasons so the caller can preserve the actual
        cause in its final error instead of reporting only ``processed=None``.
        ``revision_gaps`` receives structured records for allowed historical
        workbook read failures. These do not synthesize absence or alter the
        selected day's snapshot. The first listed revision is always required,
        including when other revisions share its timestamp.
        ``walk_progress`` identifies a source whose revision history is known
        to exist, even if a later pagination error interrupts the walk.
        """
        if revision_failures is None:
            revision_failures = []
        if revision_gaps is None:
            revision_gaps = []
        if walk_progress is None:
            walk_progress = {}
        walk_progress['listed_revisions'] = 0
        failure_indices = {}

        def record_failure(rev, when, stage, reason):
            detail = f'revision {rev} from {when} during {stage}: {reason}'
            if rev in failure_indices:
                revision_failures[failure_indices[rev]] += f'; during {stage}: {reason}'
            else:
                failure_indices[rev] = len(revision_failures)
                revision_failures.append(detail)
            print(f'WARNING: Skipping {detail}')

        dbx = self.utils.collect_dropbox_credentials()
        if hasattr(dbx, "check_and_refresh_access_token"):
            dbx.check_and_refresh_access_token()

        # SDK 11+: Dropbox client has request_json_object itself. Older
        # SDKs hide it behind an internal transport/client attribute;
        # this helper finds whichever exists.
        def find_requester(client):
            if hasattr(client, "request_json_object"):
                return client
            for attr in ("_transport", "_client", "_request", "_session",
                         "_Dropbox__transport", "_Dropbox__client"):
                obj = getattr(client, attr, None)
                if obj is not None and hasattr(obj, "request_json_object"):
                    return obj
            for name in dir(client):
                if name.startswith("__"):
                    continue
                try:
                    obj = getattr(client, name)
                except Exception:
                    continue
                if obj is not None and hasattr(obj, "request_json_object"):
                    return obj
            return None

        requester = find_requester(dbx)
        if requester is None:
            raise RuntimeError(
                "Couldn't find an internal requester on the Dropbox client with request_json_object(). "
                "Upgrade the `dropbox` package, OR pass an access token and use requests directly."
            )

        USER_AUTH = getattr(dropbox.dropbox_client, "USER_AUTH", None)
        if USER_AUTH is None:
            USER_AUTH = getattr(dropbox.dropbox_client, "AuthType", None)
            USER_AUTH = getattr(USER_AUTH, "USER", None) if USER_AUTH else None
        if USER_AUTH is None:
            USER_AUTH = "user"

        cutoff = None
        if days_back is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
            print(f"Retrieving revisions from the last {days_back} days (cutoff: {cutoff})")
        else:
            print(f"Retrieving all available revisions of {path}")

        # V2 column schema. V1 gets renamed into this before any
        # exploding so the rest of the loop only knows one shape.
        v2_cols = [
            "Subject", "Timepoint", "Additional Comments",
            "Manually Marked as Resolved", "Date Resolved",
            "General Flag", "Specific Flags",
        ]
        rename_v1_to_v2 = {
            "Participant": "Subject",
            "Site Comments": "Additional Comments",
            "Manually Resolved": "Manually Marked as Resolved",
            "Form": "General Flag",
            "Flags": "Specific Flags",
        }

        before_rev = None
        has_more = True
        last_kept_when = None
        last_selected_day = None
        skipped_same_day = 0
        kept = 0
        oldest_revision = None
        newest_revision = None
        newest_revision_id = None
        newest_processed = None
        rejected_entries = 0
        walk_complete = True
        # Previous (newer) successfully processed revision, used for
        # the revision-delta / jump detection. The walk goes newest ->
        # oldest, so deltas are attributed to pending_when (the newer
        # revision, i.e. when the change became visible).
        pending_when = None
        pending_keys = None
        pending_revision = None
        pending_active = None
        listed_revisions = set()
        previous_listed_when = None
        affected_start = len(affected_rows)
        # The Dropbox walk is newest -> oldest.  ``active`` maps each logical
        # flag key in the immediately newer processed snapshot to the concrete
        # occurrence record it belongs to.  If the same key is absent for one
        # processed revision and then appears farther back, that older sighting
        # receives a new occurrence instead of being folded into the newer one.
        episode_state = {
            'active': {},
            'next_ordinal': {},
        }

        while has_more:
            req = {"path": path, "mode": "path", "limit": page_limit}
            if before_rev:
                req["before_rev"] = before_rev

            data = requester.request_json_object(
                "api", "files/list_revisions", "rpc", req, USER_AUTH, None,
            )
            if not isinstance(data, dict):
                raise RuntimeError('Invalid Dropbox revision response')
            if data.get('is_deleted', False):
                raise RuntimeError(f'Tracker is deleted: {path}')
            if not isinstance(data.get('has_more'), bool):
                raise RuntimeError('Missing or invalid has_more in Dropbox revision response')
            entries = data.get("entries", [])
            if not isinstance(entries, list):
                raise RuntimeError('Invalid entries in Dropbox revision response')
            has_more = data["has_more"]
            if not entries:
                if has_more:
                    raise RuntimeError('Empty revision page claims more revisions')
                break

            for entry in entries:
                walk_progress['listed_revisions'] += 1
                rev = entry["rev"]
                when = datetime.fromisoformat(entry["server_modified"].replace("Z", "+00:00"))
                if not isinstance(rev, str) or not rev or rev in listed_revisions:
                    raise RuntimeError(f'Duplicate or invalid revision ID: {rev!r}')
                if when.tzinfo is None or when.utcoffset() is None:
                    raise RuntimeError(f'Revision {rev} has no timestamp timezone')
                if previous_listed_when is not None and when > previous_listed_when:
                    raise RuntimeError('Revisions are not ordered newest to oldest')
                listed_revisions.add(rev)
                if newest_revision_id is None:
                    newest_revision_id = rev
                previous_listed_when = when

                if oldest_revision is None or when < oldest_revision:
                    oldest_revision = when
                if newest_revision is None or when > newest_revision:
                    newest_revision = when

                if cutoff is not None and when < cutoff:
                    has_more = False
                    print(f"Reached cutoff date ({cutoff}), stopping retrieval")
                    break

                # Responses are newest-first, so the first revision encountered
                # on a calendar date is that day's selected snapshot. Reserve
                # the date BEFORE reading it: even an allowed read gap must
                # not silently substitute an older same-day version. This
                # selection state survives pagination boundaries.
                if self.sample_every_days == 1:
                    revision_day = self._revision_day(when)
                    if revision_day == last_selected_day:
                        skipped_same_day += 1
                        continue
                    last_selected_day = revision_day
                elif self.sample_every_days and last_kept_when is not None:
                    if (last_kept_when - when).days < self.sample_every_days:
                        continue

                stage = 'download'
                try:
                    download_metadata, resp = dbx.files_download(path=path, rev=rev)
                    downloaded_rev = getattr(download_metadata, 'rev', rev)
                    if downloaded_rev != rev:
                        raise RuntimeError(
                            f'Requested revision {rev}, received {downloaded_rev}')
                    stage = f'read {self.HISTORY_REPORT_SHEET!r}'
                    df = pd.read_excel(
                        BytesIO(resp.content),
                        keep_default_na=False,
                        sheet_name=self.HISTORY_REPORT_SHEET,
                    )
                    stage = 'normalize tracker columns'
                    df = self._coalesce_tracker_columns(df, rename_v1_to_v2)
                    stage = 'validate tracker columns'
                    missing = [c for c in v2_cols if c not in df.columns]
                    if missing:
                        raise ValueError(
                            f"{self.HISTORY_REPORT_SHEET!r} missing columns "
                            f"{missing} (have {list(df.columns)})"
                        )

                    # Unresolved-only filter — both resolved-status cols
                    # must be blank.
                    stage = 'filter unresolved rows'
                    date_blank = df["Date Resolved"].fillna("").astype(str).str.strip() == ""
                    manual_blank = df["Manually Marked as Resolved"].fillna("").astype(str).str.strip() == ""
                    df_open = df[date_blank & manual_blank]

                    stage = 'merge flag entries'
                    rev_keys, added, dropped = self._merge_revision_rows(
                        df_open, when, source_label, unique_rows,
                        episode_state=episode_state,
                        resolution_observed=pending_when,
                        revision_id=rev,
                    )
                    rejected_entries += dropped
                    if dropped:
                        walk_complete = False
                        record_failure(
                            rev, when, stage,
                            f'{dropped} entries rejected because subject, '
                            'general flag, or variable identity was missing')

                    stage = 'record revision delta'
                    if newest_processed is None or when > newest_processed:
                        newest_processed = when
                    if pending_keys is not None:
                        self._record_revision_delta(
                            source_label, pending_when, pending_keys,
                            when, rev_keys, jump_rows, affected_rows,
                            revision_rows,
                            newer_revision_id=pending_revision,
                            older_revision_id=rev,
                            newer_active=pending_active,
                            older_active=episode_state['active'],
                            source_path=path,
                        )
                    pending_when, pending_keys = when, rev_keys
                    pending_revision = rev
                    pending_active = dict(episode_state['active'])

                    last_kept_when = when
                    kept += 1
                    print(
                        f"{path} - Revision {kept}: {rev} from {when} "
                        f"(+{added} new keys; total {len(unique_rows)})"
                    )
                except Exception as e:
                    if (stage == f'read {self.HISTORY_REPORT_SHEET!r}'
                            and rev != newest_revision_id
                            and getattr(self, 'skip_unreadable_history', True)
                            and self._is_unreadable_workbook_error(e)):
                        gap = {
                            'network': getattr(self, '_current_network', None),
                            'source': source_label,
                            'source_path': path,
                            'revision_id': rev,
                            'revision_when': when.isoformat(),
                            'revision_day': self._revision_day(when).isoformat(),
                            'day_timezone': getattr(
                                self, 'day_timezone', self.HISTORY_DAY_TIMEZONE),
                            'error_type': type(e).__name__,
                            'error_message': str(e),
                        }
                        revision_gaps.append(gap)
                        print(f'WARNING: Historical snapshot gap for {path}: '
                              f'revision {rev} from {when}: {type(e).__name__}: {e}')
                        # Keep the last readable snapshot as the observation
                        # boundary. No empty report or observed disappearance
                        # can be inferred from an unreadable workbook.
                        continue
                    walk_complete = False
                    record_failure(rev, when, stage, f'{type(e).__name__}: {e}')
                    continue

            before_rev = entries[-1]["rev"]

        # Oldest processed revision has no older baseline to diff
        # against — record its count with blank deltas.
        if pending_keys is not None:
            revision_rows.append({
                'source': source_label,
                'source_path': path,
                'revision_id': pending_revision,
                'revision_when': pending_when.isoformat(),
                'open_entries': len(pending_keys),
                'added_since_prev': '',
                'removed_since_prev': '',
            })

        # The oldest sighting (and therefore stable source episode ID) is only
        # known after the complete source walk, particularly for removal jumps.
        for affected in affected_rows[affected_start:]:
            storage_key = affected.pop('_storage_key', None)
            if storage_key is not None:
                affected['source_episode_id'] = self._source_episode_id(
                    storage_key[:5], source_label, unique_rows[storage_key])
                affected['lifecycle_source'] = source_label

        if oldest_revision and newest_revision:
            days_span = (newest_revision - oldest_revision).days
            print(
                f"Done with {path}. Kept {kept} revisions over {days_span} days "
                f"({oldest_revision} -> {newest_revision})."
            )
        else:
            print(f"Done with {path}. Kept {kept} revisions.")
        if self.sample_every_days == 1:
            print(
                f'Newest-per-day selection '
                f'({getattr(self, "day_timezone", self.HISTORY_DAY_TIMEZONE)}): '
                f'skipped {skipped_same_day} older same-day revisions '
                'without downloading them.')
        if not walk_complete:
            print(
                f"WARNING: at least one in-scope revision of {path} was not "
                "fully processed; rejecting the staged source to preserve "
                "last-good history."
            )
            return rejected_entries, None, newest_revision
        if (newest_revision is not None and newest_processed is not None
                and newest_processed < newest_revision):
            print(
                f"WARNING: newest listed revision of {path} "
                f"({newest_revision}) failed to process; staged source will "
                "be rejected instead of regressing to an older snapshot."
            )

        return rejected_entries, newest_processed, newest_revision

    @classmethod
    def _is_unreadable_workbook_error(cls, error):
        """Only known input-file failures; dependency and code errors still fail."""
        if isinstance(error, (BadZipFile, ParseError)):
            return True
        if isinstance(error, ValueError):
            return str(error) in {
                f"Worksheet named '{cls.HISTORY_REPORT_SHEET}' not found",
                f'Worksheet named {cls.HISTORY_REPORT_SHEET} not found',
                'Excel file format cannot be determined, you must specify '
                'an engine manually.',
            }
        return False

    def _revision_day(self, when):
        """Calendar date used for daily selection, independent of machine zone."""
        return when.astimezone(ZoneInfo(getattr(
            self, 'day_timezone', self.HISTORY_DAY_TIMEZONE))).date()

    def _record_revision_delta(self, source_label, newer_when, newer_keys,
                               older_when, older_keys, jump_rows,
                               affected_rows, revision_rows, *,
                               newer_revision_id=None, older_revision_id=None,
                               newer_active=None, older_active=None,
                               source_path=None):
        """Diff two consecutive processed revisions of one source.
        ``newer_keys - older_keys`` = entries that APPEARED at
        ``newer_when``; ``older_keys - newer_keys`` = entries that
        DISAPPEARED at ``newer_when`` (their Latest_seen stays at the
        older revision where they were last present). Counts always go
        to ``revision_rows``; deltas >= jump_threshold also become a
        reviewable jump."""
        added_keys = newer_keys - older_keys
        removed_keys = older_keys - newer_keys
        revision_rows.append({
            'source': source_label,
            'source_path': source_path or '',
            'revision_id': newer_revision_id or '',
            'revision_when': newer_when.isoformat(),
            'open_entries': len(newer_keys),
            'added_since_prev': len(added_keys),
            'removed_since_prev': len(removed_keys),
        })
        for direction, keys in (('added', added_keys),
                                ('removed', removed_keys)):
            if not keys or len(keys) < self.jump_threshold:
                continue
            legacy_jump_id = (
                f"{source_label}_{newer_when.strftime('%Y%m%dT%H%M%SZ')}"
                f"_{direction}"
            )
            jump_id = legacy_jump_id
            if newer_revision_id is not None:
                boundary = json.dumps([newer_revision_id, older_revision_id])
                token = hashlib.sha256(boundary.encode('utf-8')).hexdigest()[:16]
                jump_id = f'{legacy_jump_id}_{token}'
            sorted_keys = sorted(keys)
            sample = '; '.join(
                f"{k[0]}/{k[1]}/{k[2]}/{k[3]}" for k in sorted_keys[:10]
            )
            jump_rows.append({
                'jump_id': jump_id,
                'legacy_jump_id': legacy_jump_id,
                'revision_id': newer_revision_id or '',
                'previous_revision_id': older_revision_id or '',
                'source': source_label,
                'direction': direction,
                'observed_at': newer_when.isoformat(),
                # For removals, affected entries were last seen at the
                # previous (older) revision — that's the day their
                # Latest_seen carries in the history CSV.
                'previous_revision': older_when.isoformat(),
                'n_entries': len(keys),
                'open_entries_after': len(newer_keys),
                'sample_entries': sample,
            })
            for k in sorted_keys:
                affected = {
                    'jump_id': jump_id,
                    'Subject': k[0],
                    'Timepoint': k[1],
                    'General_Flag': k[2],
                    'variable': k[3],
                    'canonical_template': k[4],
                }
                active = newer_active if direction == 'added' else older_active
                if active is not None:
                    affected['_storage_key'] = active[k]
                affected_rows.append(affected)

    @staticmethod
    def _observation_before(observation):
        """Recover a value from one raw-message sighting unless it conflicts."""

        if not observation or observation.get('message_value_conflict', False):
            return None
        return infer_before_value(
            observation.get('message_variable', ''),
            observation.get('message', ''),
        )

    @classmethod
    def _observation_tie_key(cls, observation):
        """Deterministic preference for duplicate sightings at one timestamp."""

        before = cls._observation_before(observation)
        source_rank = {
            'proposed_evidence': 3,
            'blank_flag': 2,
            'check_contract': 2,
            'message_pattern': 2,
        }.get(before.source if before is not None else '', 0)
        return (
            source_rank,
            str(observation.get('message', '')),
            str(observation.get('message_variable', '')),
        )

    @classmethod
    def _merge_message_observation(cls, current, candidate):
        """Merge two sightings for one source while failing closed on values."""

        if current is None or candidate['when'] > current['when']:
            return candidate
        if candidate['when'] < current['when']:
            return current

        current_before = cls._observation_before(current)
        candidate_before = cls._observation_before(candidate)
        chosen = max((current, candidate), key=cls._observation_tie_key)

        if current.get('message_value_conflict', False):
            chosen = dict(chosen)
            chosen['message_value_conflict'] = True
            return chosen
        if (current_before is not None and candidate_before is not None
                and current_before.value != candidate_before.value):
            chosen = dict(chosen)
            chosen['message_value_conflict'] = True
            return chosen
        if candidate_before is not None and current_before is None:
            return candidate
        if current_before is not None and candidate_before is None:
            return current
        return chosen

    @classmethod
    def _select_message_observation(cls, rec, preferred_source=None):
        """Select the lifecycle-aligned raw message and its source."""

        observations = rec.get('messages_per_source', {})
        if preferred_source in observations:
            return preferred_source, observations[preferred_source]
        if observations:
            return max(
                observations.items(),
                key=lambda item: (
                    item[1]['when'],
                    cls._observation_tie_key(item[1]),
                    item[0],
                ),
            )

        # Compatibility for hand-built/legacy in-memory records.
        if 'message' in rec:
            latest_values = list(rec.get('latest_per_source', {}).values())
            return '', {
                'when': max(latest_values) if latest_values else pd.NaT,
                'message': rec.get('message', ''),
                'message_variable': rec.get('message_variable', ''),
                'message_value_conflict': rec.get(
                    'message_value_conflict', False),
            }
        return '', None

    def _refresh_representative_message(self, rec):
        """Keep legacy top-level message keys aligned with the live source."""

        preferred_source = self.LIVE_SOURCE_PER_NETWORK.get(
            getattr(self, '_current_network', None))
        source, observation = self._select_message_observation(
            rec, preferred_source)
        if observation is None:
            return
        rec['message'] = observation['message']
        rec['message_variable'] = observation['message_variable']
        rec['message_value_conflict'] = observation.get(
            'message_value_conflict', False)
        rec['message_source'] = source

    def _merge_revision_rows(self, df_open, when, source_label, unique_rows,
                             episode_state=None, resolution_observed=None,
                             revision_id=None):
        """Merge one revision's unresolved rows into unique_rows.

        Each Specific Flags entry is keyed independently. Raw messages are kept
        per tracker source so the value used as the pre-resolution observation
        comes from the same production lifecycle as Latest_seen. At a duplicate
        timestamp an inferable message wins over a prose-only copy; conflicting
        inferable values are retained for audit but marked unusable.

        ``episode_state`` is supplied by the newest-to-oldest revision walk.  A
        logical key is reused only while it is present in consecutive processed
        snapshots.  Reappearance after an absent snapshot allocates a distinct
        occurrence record within that source and records the newer absent
        snapshot as the first observed resolution. Cross-source matching is
        deferred until both complete lifecycles are known. Omitting the state
        preserves the legacy
        helper behavior used by direct callers that merge isolated frames.
        """

        before = len(unique_rows)
        rejected = 0
        rev_keys = set()
        lifecycle_mode = episode_state is not None
        revision_index = (episode_state.get('processed_count', 0)
                          if lifecycle_mode else None)
        if lifecycle_mode:
            newer_active = dict(episode_state.get('active', {}))
            next_ordinal = episode_state.setdefault('next_ordinal', {})
            current_active = {}
        else:
            newer_active = {}
            next_ordinal = {}
            current_active = {}

        @lru_cache(maxsize=self._PARSED_FLAGS_CACHE_SIZE)
        def parsed_entry(raw):
            var, msg = split_entry(raw)
            canonical_raw = canonicalize_message(var, msg) if var else None
            return var, msg, canonical_raw

        row_columns = ('Subject', 'Timepoint', 'General Flag', 'Specific Flags')
        row_values = None
        if (df_open.columns.nlevels == 1 and df_open.columns.is_unique
                and all(column in df_open.columns for column in row_columns)):
            row_values = df_open.to_numpy()
        if row_values is not None and row_values.dtype.kind not in {'M', 'm'}:
            # iterrows() builds a Series for every row from this same common-
            # dtype array. Rows with string flags cannot be inferred as a
            # datetime/timedelta Series, so their scalars can be read directly.
            subject_col, timepoint_col, general_col, specific_col = (
                df_open.columns.get_loc(column) for column in row_columns)
            rows = (
                values if isinstance(values[specific_col], str)
                else pd.Series(values).array
                for values in row_values
            )
        else:
            # Preserve the existing lazy access/error behavior for malformed
            # frames and direct callers (including an empty, columnless frame).
            # Homogeneous datetime/timedelta arrays also need Series' scalar
            # boxing to retain the original Timestamp/Timedelta string values.
            rows = (row for _, row in df_open.iterrows())
            subject_col, timepoint_col, general_col, specific_col = row_columns

        for row in rows:
            raw_subject = row[subject_col]
            subject = str(raw_subject).strip()
            subject_missing = (
                not subject or (lifecycle_mode
                    and not isinstance(raw_subject, str)
                    and pd.api.types.is_scalar(raw_subject)
                    and pd.isna(raw_subject)))
            if subject_missing:
                if lifecycle_mode:
                    raw_specific = row[specific_col]
                    if (raw_specific is not None
                            and not (pd.api.types.is_scalar(raw_specific)
                                     and pd.isna(raw_specific))):
                        rejected += len(split_specific_flag_entries(str(raw_specific)))
                continue
            tp = normalize_timepoint(row[timepoint_col])
            raw_general = row[general_col]
            general_flag = normalize_general_flag(raw_general)
            general_missing = (
                not general_flag or (lifecycle_mode
                    and not isinstance(raw_general, str)
                    and pd.api.types.is_scalar(raw_general)
                    and pd.isna(raw_general)))
            if general_missing:
                if lifecycle_mode:
                    raw_specific = row[specific_col]
                    if (raw_specific is not None
                            and not (pd.api.types.is_scalar(raw_specific)
                                     and pd.isna(raw_specific))):
                        rejected += len(split_specific_flag_entries(str(raw_specific)))
                continue
            specific = (
                str(row[specific_col])
                if row[specific_col] is not None else '')
            if not specific.strip():
                continue
            for raw in split_specific_flag_entries(specific):
                var, msg, canonical_raw = parsed_entry(raw)
                if not var:
                    rejected += 1
                    continue
                canonical = self.template_map.get(canonical_raw, canonical_raw)
                var_mapped = self.variable_map.get(var, var)
                logical_key = (
                    subject, tp, general_flag, var_mapped, canonical)
                rev_keys.add(logical_key)

                new_episode = False
                if lifecycle_mode:
                    key = current_active.get(logical_key)
                    if key is None:
                        key = newer_active.get(logical_key)
                    if key is None:
                        ordinal = next_ordinal.get(logical_key, 0)
                        next_ordinal[logical_key] = ordinal + 1
                        key = (*logical_key, source_label, ordinal)
                        new_episode = True
                    current_active[logical_key] = key
                else:
                    key = logical_key

                candidate = {
                    'when': when,
                    'message': msg,
                    'message_variable': var,
                    'message_value_conflict': False,
                }
                rec = unique_rows.get(key)
                if rec is None:
                    self._track_seen(canonical_raw, var, general_flag)
                    rec = {
                        'earliest_per_source': {source_label: when},
                        'latest_per_source': {source_label: when},
                        'sources': {source_label},
                        'messages_per_source': {source_label: candidate},
                        'resolution_per_source': {},
                    }
                    unique_rows[key] = rec
                else:
                    src_earliest = rec['earliest_per_source'].get(source_label)
                    if src_earliest is None or when < src_earliest:
                        rec['earliest_per_source'][source_label] = when
                    src_latest = rec['latest_per_source'].get(source_label)
                    if src_latest is None or when > src_latest:
                        rec['latest_per_source'][source_label] = when
                    observations = rec.setdefault('messages_per_source', {})
                    current_observation = observations.get(source_label)
                    # Distinct revisions can share a server timestamp. The
                    # newest-to-oldest walk supplies order; only duplicates
                    # within one revision participate in a value-conflict tie.
                    same_revision = (
                        not lifecycle_mode
                        or rec.get('latest_revision_index_per_source', {}).get(
                            source_label) == revision_index)
                    if (same_revision or current_observation is None
                            or current_observation['when'] != when):
                        observations[source_label] = self._merge_message_observation(
                            current_observation, candidate)
                    rec['sources'].add(source_label)

                rec.setdefault('raw_identity_per_source', {}).setdefault(
                    source_label, {
                        'canonical_raw': canonical_raw,
                        'variable_raw': var,
                        'general_flag': general_flag,
                    })
                if lifecycle_mode:
                    rec.setdefault('episode_ordinal_per_source', {}).setdefault(
                        source_label, key[-1])
                    rec.setdefault('open_in_latest_per_source', {}).setdefault(
                        source_label, revision_index == 0)
                    rec.setdefault('latest_revision_index_per_source', {}).setdefault(
                        source_label, revision_index)
                    rec.setdefault('latest_revision_per_source', {}).setdefault(
                        source_label, revision_id)
                    # Each next processed snapshot is older, including when
                    # its timestamp equals the immediately newer snapshot.
                    rec.setdefault('earliest_revision_per_source', {})[
                        source_label] = revision_id

                if new_episode and resolution_observed is not None:
                    rec.setdefault('resolution_per_source', {})[
                        source_label] = resolution_observed
                    rec.setdefault('resolution_revision_per_source', {})[
                        source_label] = episode_state.get('newer_revision_id')

                self._refresh_representative_message(rec)
        if lifecycle_mode:
            # This older absence bounds the beginning of the newer episode.
            # Retain it so reconciliation cannot join across an observed gap.
            for logical_key, storage_key in newer_active.items():
                if logical_key not in current_active:
                    rec = unique_rows.get(storage_key)
                    if rec is None:
                        continue
                    rec.setdefault('absence_before_per_source', {})[
                        source_label] = when
            episode_state['active'] = current_active
            episode_state['processed_count'] = revision_index + 1
            episode_state['newer_revision_id'] = revision_id
        return rev_keys, len(unique_rows) - before, rejected

    def _track_seen(self, canonical_raw, variable_raw, general_flag):
        """Record a newly observed episode's RAW (pre-mapping) template
        and variable, for re-seeding template_mapping.xlsx. Counting
        only new episodes keeps n_entries comparable to history-CSV row
        counts rather than revision-sighting counts."""
        t = self._seen_templates.get(canonical_raw)
        if t is None:
            t = {
                'n_entries': 0,
                'networks': set(),
                'example_form': general_flag,
                'example_variable': variable_raw,
            }
            self._seen_templates[canonical_raw] = t
        t['n_entries'] += 1
        t['networks'].add(self._current_network or '')

        v = self._seen_variables.get(variable_raw)
        if v is None:
            v = {
                'n_entries': 0,
                'networks': set(),
                'example_form': general_flag,
            }
            self._seen_variables[variable_raw] = v
        v['n_entries'] += 1
        v['networks'].add(self._current_network or '')

    @staticmethod
    def _stable_episode_id(logical_key, earliest_seen):
        """Return a deterministic ID that does not shift on later recurrences."""

        timestamp = pd.Timestamp(earliest_seen)
        earliest_token = (
            '' if pd.isna(timestamp) else timestamp.isoformat())
        identity = '\x1f'.join(
            [*(str(part) for part in logical_key), earliest_token]
        )
        digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]
        return f'ep_{digest}'

    @classmethod
    def _source_episode_id(cls, logical_key, source_label, record):
        """Stable source occurrence ID, independent of newer recurrences.

        Dropbox revision tokens identify the earliest positive snapshot even
        when several revisions have an identical server timestamp. Direct
        helper callers without revision tokens retain a deterministic fallback.
        """
        earliest = record['earliest_per_source'][source_label]
        revision = record.get('earliest_revision_per_source', {}).get(source_label)
        if revision is not None:
            anchor = f'revision:{revision}'
        else:
            timestamp = pd.Timestamp(earliest).isoformat()
            ordinal = record.get('episode_ordinal_per_source', {}).get(source_label)
            anchor = f'timestamp:{timestamp}'
            if ordinal is not None:
                anchor += f':occurrence:{ordinal}'
        identity = json.dumps(
            [*map(str, logical_key), str(source_label), anchor],
            ensure_ascii=False, separators=(',', ':'))
        digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]
        return f'ep_{digest}'

    @staticmethod
    def _source_episodes_overlap(first, second):
        """Require overlapping positive spans without a known separating gap."""
        if len(first['sources']) != 1 or len(second['sources']) != 1:
            return False
        first_source = next(iter(first['sources']))
        second_source = next(iter(second['sources']))
        if first_source == second_source:
            return False
        first_start = first['earliest_per_source'][first_source]
        first_end = first['latest_per_source'][first_source]
        second_start = second['earliest_per_source'][second_source]
        second_end = second['latest_per_source'][second_source]
        if max(first_start, second_start) > min(first_end, second_end):
            return False
        for record, source, other_start, other_end in (
                (first, first_source, second_start, second_end),
                (second, second_source, first_start, first_end)):
            resolved = record.get('resolution_per_source', {}).get(source)
            absent_before = record.get('absence_before_per_source', {}).get(source)
            if resolved is not None and resolved <= other_start:
                return False
            if absent_before is not None and absent_before >= other_end:
                return False
        return True

    def _history_episode_records(self, unique_rows):
        """Yield source episodes reconciled only by unambiguous overlap.

        All source lifecycles are complete before this runs. A stale episode
        spanning two recurrences therefore has two candidate matches and is
        kept separate, rather than joining either recurrence or joining both.
        The collector's source-qualified records are never mutated here.
        """
        groups = {}
        for storage_key, record in unique_rows.items():
            groups.setdefault(tuple(storage_key[:5]), []).append(record)
        map_fields = (
            'earliest_per_source', 'latest_per_source', 'messages_per_source',
            'resolution_per_source', 'earliest_revision_per_source',
            'latest_revision_per_source', 'resolution_revision_per_source',
            'latest_revision_index_per_source', 'open_in_latest_per_source',
            'episode_ordinal_per_source', 'absence_before_per_source',
            'raw_identity_per_source',
        )
        for logical_key, records in groups.items():
            candidates = {index: [] for index in range(len(records))}
            # Sweep positive spans instead of comparing every recurrence pair.
            # Disjoint spans and same-source recurrences need no pair checks.
            spans = []
            for index, record in enumerate(records):
                if ('episode_ordinal_per_source' in record
                        and len(record['sources']) == 1):
                    source = next(iter(record['sources']))
                    spans.append((record['earliest_per_source'][source],
                                  record['latest_per_source'][source], index, source))
            active = {}
            endings = []
            for start, end, index, source in sorted(spans):
                while endings and endings[0][0] < start:
                    _, expired_index, expired_source = heapq.heappop(endings)
                    active[expired_source].pop(expired_index, None)
                for other_source, source_active in active.items():
                    if other_source == source:
                        continue
                    for other_index in source_active:
                        # Two matches already establish ambiguity. Further
                        # edges between two ambiguous records cannot change it.
                        if (len(candidates[index]) >= 2
                                and len(candidates[other_index]) >= 2):
                            continue
                        if self._source_episodes_overlap(
                                records[index], records[other_index]):
                            if len(candidates[index]) < 2:
                                candidates[index].append(other_index)
                            if len(candidates[other_index]) < 2:
                                candidates[other_index].append(index)
                active.setdefault(source, {})[index] = True
                heapq.heappush(endings, (end, index, source))
            emitted = set()
            for index, record in enumerate(records):
                if index in emitted:
                    continue
                emitted.add(index)
                matches = candidates[index]
                if len(matches) == 1 and candidates[matches[0]] == [index]:
                    other_index = matches[0]
                    other = records[other_index]
                    combined = copy.deepcopy(record)
                    combined['sources'].update(other['sources'])
                    for field in map_fields:
                        if field in other:
                            combined.setdefault(field, {}).update(other[field])
                    self._refresh_representative_message(combined)
                    emitted.add(other_index)
                    yield logical_key, combined
                else:
                    yield logical_key, record

    def _track_reconciled_episodes(self, unique_rows):
        """Add this network's output episodes to restored template counters.

        The caller restores counters from before this network's source walks
        first. Use the earliest constituent's original identity so aliases are
        represented deterministically without counting a reconciled pair twice.
        """
        for logical_key, record in self._history_episode_records(unique_rows):
            source = min(record['sources'], key=lambda label: (
                record['earliest_per_source'][label], label))
            raw = record.get('raw_identity_per_source', {}).get(source)
            if raw is None:
                self._track_seen(logical_key[4], logical_key[3], logical_key[2])
            else:
                self._track_seen(
                    raw['canonical_raw'], raw['variable_raw'], raw['general_flag'])

    def _write_history_csv(self, network, unique_rows, newest_per_source):
        """Materialize ``unique_rows`` for one network as an entry-level
        CSV.

        Earliest_seen is the minimum across ALL sources — an entry that
        lived in V1 long before the V2 promotion keeps its full V1
        history (anchoring earliest to the live source truncated
        PRESCIENT lifespans at the promotion date, which contradicted
        the "as far back as Dropbox goes" requirement). Latest_seen
        still comes from the LIVE source when the entry was seen there,
        so openness/resolution tracks the tracker QC analysts actually
        use. For entries that ONLY appear in the non-live source
        (e.g. PRONET V2 testing-only entries), we fall back to the
        non-live source's dates so they still get plotted, just
        within their non-live lifespan and with is_currently_open=False.

        is_currently_open comes from membership in the live source's newest
        processed snapshot, independent of timestamp precision. Historical
        source records are joined only when their positive observation spans
        overlap unambiguously, with no known separating absence. A non-live
        episode without an observed disappearance cannot establish resolution.
        """
        cols = [
            "Subject", "Timepoint", "General_Flag", "variable",
            "message", "canonical_template", "episode_id",
            "Earliest_seen", "Latest_seen", "Resolution_observed",
            "source", "is_currently_open",
            "message_variable", "message_source",
            "message_value_conflict", "resolution_eligible",
            "lifecycle_source", "source_episode_ids",
        ]
        live_source = self.LIVE_SOURCE_PER_NETWORK.get(network)
        live_cutoff = newest_per_source.get(live_source) if live_source else None

        if not unique_rows:
            print(f"No entries collected for {network}; writing empty CSV.")
            out_df = pd.DataFrame(columns=cols)
        else:
            rows = []
            for logical_key, rec in self._history_episode_records(unique_rows):
                subject, tp, general_flag, var, canonical = logical_key
                sources = rec['sources']
                source_str = '+'.join(sorted(sources))
                message_source, message_observation = (
                    self._select_message_observation(rec, live_source))
                if message_observation is None:
                    message_observation = {
                        'message': '',
                        'message_variable': var,
                        'message_value_conflict': True,
                    }

                # PRESCIENT V1 was itself a production tracker before V2
                # promotion, so its historical episodes remain valid.
                # PRONET V2 is testing-only until promotion and cannot prove a
                # live resolution by itself.
                resolution_eligible = (
                    network != 'PRONET' or self.SOURCE_V1 in sources)

                # Earliest across all sources (full history); latest
                # anchored to the live source's view when present,
                # otherwise the overall max across whatever sources we
                # DID see it in.
                earliest_csv = min(rec['earliest_per_source'].values())
                if live_source in rec['latest_per_source']:
                    latest_source = live_source
                    latest_csv = rec['latest_per_source'][live_source]
                else:
                    latest_source, latest_csv = max(
                        rec['latest_per_source'].items(),
                        key=lambda item: (item[1], item[0]),
                    )

                # is_currently_open: still in live source's latest
                # snapshot. Entries not in the live source are by
                # definition not currently open.
                if live_source and live_cutoff is not None:
                    live_latest = rec['latest_per_source'].get(live_source)
                    membership = rec.get('open_in_latest_per_source', {})
                    if live_source in membership:
                        is_open = bool(membership[live_source])
                    else:
                        # Compatibility for hand-built/legacy helper records.
                        is_open = bool(
                            live_latest is not None and live_latest >= live_cutoff)
                else:
                    is_open = False

                resolution_observed = rec.get(
                    'resolution_per_source', {}).get(latest_source)
                if is_open:
                    # A currently open occurrence has not resolved even if a
                    # non-live source recorded an earlier disappearance.
                    resolution_observed = None
                if ('episode_ordinal_per_source' in rec
                        and latest_source != live_source
                        and resolution_observed is None):
                    # An unmatched stale/test tracker cannot establish a live
                    # resolution merely because it is not the live source.
                    resolution_eligible = False

                source_ids = {
                    source: self._source_episode_id(logical_key, source, rec)
                    for source in sorted(sources)
                }
                if 'episode_ordinal_per_source' in rec:
                    # A non-live partner can later become an ambiguous stale
                    # span. Its attachment must never rename the live episode.
                    episode_id = source_ids[latest_source]
                else:
                    episode_id = self._stable_episode_id(logical_key, earliest_csv)

                rows.append({
                    "Subject": subject,
                    "Timepoint": tp,
                    "General_Flag": general_flag,
                    "variable": var,
                    "message": message_observation['message'],
                    "canonical_template": canonical,
                    "episode_id": episode_id,
                    "Earliest_seen": earliest_csv,
                    "Latest_seen": latest_csv,
                    "Resolution_observed": resolution_observed,
                    "source": source_str,
                    "is_currently_open": is_open,
                    "message_variable": message_observation.get(
                        'message_variable', var),
                    "message_source": message_source,
                    "message_value_conflict": message_observation.get(
                        'message_value_conflict', False),
                    "resolution_eligible": resolution_eligible,
                    "lifecycle_source": latest_source,
                    "source_episode_ids": json.dumps(source_ids, sort_keys=True),
                })
            out_df = pd.DataFrame(rows, columns=cols)

        out_csv = os.path.join(
            self.analyze_flags_dir,
            open_tracker_row_history_csv_basename(network),
        )
        atomic_write_csv(out_df, out_csv)
        print(
            f"Wrote {len(out_df)} entries for {network}: {out_csv} "
            f"(live source: {live_source or 'none'})"
        )

    def _write_revision_counts(self, network, revision_rows):
        """Audit CSV: one row per processed revision per source, with
        the open-entry count and the delta vs. the previous (older)
        processed revision. This is the raw material behind the jump
        workbook — useful for eyeballing slow drifts that never trip
        the jump threshold."""
        path = os.path.join(
            self.analyze_flags_dir,
            revision_counts_csv_basename(network),
        )
        cols = ['source', 'source_path', 'revision_id', 'revision_when', 'open_entries',
                'added_since_prev', 'removed_since_prev']
        df = pd.DataFrame(revision_rows, columns=cols)
        if not df.empty:
            df = df.sort_values(['source', 'revision_when']).reset_index(drop=True)
        atomic_write_csv(df, path)
        print(f"Wrote {path}: {len(df)} revision rows.")

    def _write_metadata_json(self, network, newest_per_source):
        """Persist per-source newest successfully PROCESSED revision
        timestamps so consumers don't have to infer ``newest_day`` from
        max(Latest_seen) (which can lag the real newest revision if no
        flag was open at it)."""
        path = os.path.join(
            self.analyze_flags_dir,
            tracker_metadata_json_basename(network),
        )
        v1 = newest_per_source.get(self.SOURCE_V1)
        v2 = newest_per_source.get(self.SOURCE_V2)
        overall = None
        for v in (v1, v2):
            if v is None:
                continue
            if overall is None or v > overall:
                overall = v
        gaps = getattr(self, '_history_gaps_by_network', {}).get(network, [])
        metadata = {
            'history_schema_version': 2,
            'sample_every_days': getattr(self, 'sample_every_days', 1),
            'revision_sampling': (
                'newest_per_calendar_day' if getattr(self, 'sample_every_days', 1) == 1
                else 'all_revisions' if getattr(self, 'sample_every_days', 1) == 0
                else 'elapsed_day_interval'),
            'day_timezone': getattr(self, 'day_timezone', self.HISTORY_DAY_TIMEZONE),
            'network': network,
            'live_source': self.LIVE_SOURCE_PER_NETWORK.get(network),
            'newest_revision_V1': v1.isoformat() if v1 else None,
            'newest_revision_V2': v2.isoformat() if v2 else None,
            'newest_revision_overall': overall.isoformat() if overall else None,
            'skip_unreadable_history': getattr(self, 'skip_unreadable_history', True),
            'history_complete': not gaps,
            'skipped_unreadable_revisions': len(gaps),
            'skipped_unreadable_revisions_per_source': {
                source: sum(gap['source'] == source for gap in gaps)
                for source in (self.SOURCE_V1, self.SOURCE_V2)},
            'history_gaps_file': self._history_gaps_basename(network),
        }
        atomic_write_json(metadata, path)
        print(f"Wrote metadata: {path}")

    @staticmethod
    def _history_gaps_basename(network):
        return f'history_gaps_{network}.csv'

    def _write_history_gaps(self, network):
        """Publish the gap audit with the matching history, even when empty."""
        columns = ['network', 'source', 'source_path', 'revision_id',
                   'revision_when', 'revision_day', 'day_timezone',
                   'error_type', 'error_message']
        gaps = getattr(self, '_history_gaps_by_network', {}).get(network, [])
        path = os.path.join(self.analyze_flags_dir, self._history_gaps_basename(network))
        frame = pd.DataFrame(gaps, columns=columns)
        if not frame.empty:
            frame = frame.sort_values(['source', 'revision_when', 'revision_id'])
        atomic_write_csv(frame, path)
        print(f'Wrote {path}: {len(frame)} unreadable historical snapshots.')


if __name__ == '__main__':
    ResolvedEstimator().run_script()
