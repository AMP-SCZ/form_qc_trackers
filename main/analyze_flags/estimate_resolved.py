"""Walk every Dropbox revision of the combined network trackers and
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

Every recognized report worksheet is read from each revision, and the default
walk processes every revision so short-lived flags remain observable. A
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

import pandas as pd
import os
import sys
import json
import dropbox
from io import BytesIO
from datetime import datetime, timedelta, timezone

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

    # Only sheets generated as tracker review surfaces are authoritative.
    # Schema matching alone is unsafe: an operator-created backup tab can carry
    # the same columns and would otherwise keep obsolete flags open forever.
    AUTHORITATIVE_REPORT_SHEETS = frozenset({
        'Main Report', 'Secondary Report', 'Date Report', 'Cross Checks',
        'Proposed Checks', 'Medication Flags', 'Non Team Forms',
        'Cognition Report', 'Blood Report', 'Fluids Report', 'Scid Report',
        'MRI Report', 'EEG Report', 'Digital Report',
        'Incomplete Forms', 'Missingness Report', 'Conversion Report',
    })

    def __init__(self, sample_every_days=0, jump_threshold=50):
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
        # Default 0 processes every revision so a short-lived, correctable flag
        # cannot disappear between daily samples. Raise this to speed runs at
        # the explicit cost of incomplete episode/value coverage.
        self.sample_every_days = sample_every_days
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
        for network in self.NETWORKS:
            previously_recorded_sources = self._previously_recorded_sources(
                network)
            network_seen_templates_before = copy.deepcopy(self._seen_templates)
            network_seen_variables_before = copy.deepcopy(self._seen_variables)
            # unique_rows is now keyed at the specific-flag entry level.
            # newest_per_source tracks the latest revision timestamp
            # SUCCESSFULLY PROCESSED per (V1, V2) so we can write the
            # metadata JSON and compute is_currently_open against the
            # right source. (Using the newest *seen* revision here was
            # a bug: one corrupt/undownloadable latest snapshot marked
            # every entry resolved.)
            unique_rows = {}
            newest_per_source = {self.SOURCE_V1: None, self.SOURCE_V2: None}
            rejected_entries = 0
            any_fresh_walk_completed = False
            # Per-network manual-review accumulators, filled by the
            # revision-delta tracking inside _walk_revisions.
            jump_rows = []
            affected_rows = []
            revision_rows = []
            self._current_network = network
            live_source = self.LIVE_SOURCE_PER_NETWORK[network]
            for base, basename, source_label in self._tracker_paths(network):
                full_path = f'{base}{network}/combined/{basename}'
                # Stage each source transactionally. A partial walk can mutate
                # existing cross-source records, jump rows, and mapping counts;
                # none of those changes are valid if its newest snapshot failed.
                candidate_rows = copy.deepcopy(unique_rows)
                candidate_jumps = list(jump_rows)
                candidate_affected = list(affected_rows)
                candidate_revisions = list(revision_rows)
                seen_templates_before = copy.deepcopy(self._seen_templates)
                seen_variables_before = copy.deepcopy(self._seen_variables)
                try:
                    rejected, newest_processed, newest_listed = (
                        self._walk_revisions(
                            full_path, source_label, candidate_rows,
                            candidate_jumps, candidate_affected,
                            candidate_revisions, days_back=None,
                        )
                    )
                    is_fresh = (
                        newest_processed is not None
                        and (
                            newest_listed is None
                            or newest_processed >= newest_listed
                        )
                    )
                    if not is_fresh:
                        self._seen_templates = seen_templates_before
                        self._seen_variables = seen_variables_before
                        print(
                            f"WARNING: rejecting stale/incomplete {source_label} "
                            f"history for {network}; newest listed revision "
                            f"{newest_listed} was not successfully processed."
                        )
                        continue
                    unique_rows = candidate_rows
                    jump_rows = candidate_jumps
                    affected_rows = candidate_affected
                    revision_rows = candidate_revisions
                    rejected_entries += rejected
                    newest_per_source[source_label] = newest_processed
                    any_fresh_walk_completed = True
                except Exception as e:
                    self._seen_templates = seen_templates_before
                    self._seen_variables = seen_variables_before
                    # Missing file / transient API error shouldn't kill the
                    # other source or network, but this source contributes
                    # nothing unless its complete newest snapshot is usable.
                    print(
                        f"WARNING: failed walking revisions for {full_path}: "
                        f"{type(e).__name__}: {str(e)[:200]}"
                    )
                    continue
            fresh_sources = {
                source for source, newest in newest_per_source.items()
                if newest is not None
            }
            missing_previous_sources = (
                previously_recorded_sources - fresh_sources)
            if missing_previous_sources:
                self._seen_templates = network_seen_templates_before
                self._seen_variables = network_seen_variables_before
                print(
                    f"Previously recorded tracker source(s) "
                    f"{sorted(missing_previous_sources)} were not freshly "
                    f"processed for {network}; preserving the complete prior "
                    "history / metadata (no write)."
                )
                continue
            if not any_fresh_walk_completed:
                self._seen_templates = network_seen_templates_before
                self._seen_variables = network_seen_variables_before
                print(
                    f"No complete, fresh tracker walk for {network}; "
                    f"preserving existing CSV / metadata (no write)."
                )
                continue
            if newest_per_source[live_source] is None:
                self._seen_templates = network_seen_templates_before
                self._seen_variables = network_seen_variables_before
                # Resolution state is defined by the production tracker. A
                # successful walk of only the non-live/testing source cannot
                # prove that a production flag disappeared. Preserve the last
                # good artifacts instead of manufacturing resolved episodes.
                print(
                    f"No {live_source} production revision was successfully "
                    f"processed for {network}; preserving existing history / "
                    "metadata and derived revision artifacts (no write)."
                )
                continue
            if rejected_entries:
                print(
                    f"{network}: dropped {rejected_entries} malformed "
                    f"Specific_Flags entries (no colon found)."
                )
            self._write_history_csv(network, unique_rows, newest_per_source)
            self._write_metadata_json(network, newest_per_source)
            write_jumps_workbook(
                self.analyze_flags_dir, network, jump_rows, affected_rows,
            )
            self._write_revision_counts(network, revision_rows)
        # One cross-network mapping workbook, re-seeded with everything
        # observed this run (operator maps_to entries preserved).
        update_template_mapping(
            self.analyze_flags_dir,
            self._seen_templates, self._seen_variables,
        )

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
        except (OSError, ValueError, TypeError):
            pass

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
        except (OSError, ValueError, pd.errors.ParserError):
            pass
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
                        days_back=None, page_limit=100):
        """Walk every revision of ``path`` (paging beyond 100 via
        before_rev) and merge unresolved-row entries into
        ``unique_rows`` in place. Schema is normalized V1 -> V2 before
        any exploding so the rest of the loop only sees one shape.

        Also diffs each processed revision's entry-key set against the
        previous (newer) processed revision of the same source: every
        revision gets a row in ``revision_rows`` (count + delta), and
        deltas >= ``self.jump_threshold`` additionally get a reviewable
        row in ``jump_rows`` with their exact keys in ``affected_rows``.

        Returns the rejected-entry count, newest processed revision, and newest
        listed revision. The processed value is None when any in-scope revision
        was incomplete, so the caller rejects the entire staged source.
        Rebuilding history from a partial walk can erase episodes that existed
        only in the skipped revision, even when the newest snapshot was readable.
        """
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
        kept = 0
        oldest_revision = None
        newest_revision = None
        newest_processed = None
        rejected_entries = 0
        walk_complete = True
        # Previous (newer) successfully processed revision, used for
        # the revision-delta / jump detection. The walk goes newest ->
        # oldest, so deltas are attributed to pending_when (the newer
        # revision, i.e. when the change became visible).
        pending_when = None
        pending_keys = None
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

            entries = data.get("entries", [])
            has_more = bool(data.get("has_more", False))
            if not entries:
                break

            for entry in entries:
                rev = entry["rev"]
                when = datetime.fromisoformat(entry["server_modified"].replace("Z", "+00:00"))

                if oldest_revision is None or when < oldest_revision:
                    oldest_revision = when
                if newest_revision is None or when > newest_revision:
                    newest_revision = when

                if cutoff is not None and when < cutoff:
                    has_more = False
                    print(f"Reached cutoff date ({cutoff}), stopping retrieval")
                    break

                # Sampling skip — newest revision always processes because
                # last_kept_when starts at None.
                if self.sample_every_days and last_kept_when is not None:
                    if (last_kept_when - when).days < self.sample_every_days:
                        continue

                try:
                    _, resp = dbx.files_download(path=path, rev=rev)
                    workbook_sheets = pd.read_excel(
                        BytesIO(resp.content),
                        keep_default_na=False,
                        sheet_name=None,
                    )
                    normalized_sheets = []
                    malformed_sheets = []
                    for sheet_name, sheet_df in workbook_sheets.items():
                        if sheet_name not in self.AUTHORITATIVE_REPORT_SHEETS:
                            print(
                                f"WARNING: Skipping non-authoritative sheet "
                                f"{sheet_name!r} in revision {rev}.")
                            continue
                        normalized = self._coalesce_tracker_columns(
                            sheet_df, rename_v1_to_v2)
                        if all(col in normalized.columns for col in v2_cols):
                            normalized_sheets.append(normalized)
                        else:
                            missing_sheet_columns = [
                                col for col in v2_cols
                                if col not in normalized.columns
                            ]
                            malformed_sheets.append(
                                (sheet_name, missing_sheet_columns))
                    if malformed_sheets:
                        walk_complete = False
                        details = '; '.join(
                            f"{name!r} missing {missing}"
                            for name, missing in malformed_sheets
                        )
                        print(
                            f"WARNING: Rejecting revision {rev} from {when}; "
                            f"present authoritative report sheet(s) were "
                            f"malformed: {details}."
                        )
                        continue
                    if not normalized_sheets:
                        walk_complete = False
                        print(f"WARNING: No recognized report sheets in revision {rev}.")
                        continue
                    df = pd.concat(normalized_sheets, ignore_index=True, sort=False)


                    missing = [c for c in v2_cols if c not in df.columns]
                    if missing:
                        walk_complete = False
                        print(
                            f"WARNING: Skipping revision {rev} from {when} "
                            f"- missing columns {missing} (have {list(df.columns)})"
                        )
                        continue

                    # Unresolved-only filter — both resolved-status cols
                    # must be blank.
                    date_blank = df["Date Resolved"].fillna("").astype(str).str.strip() == ""
                    manual_blank = df["Manually Marked as Resolved"].fillna("").astype(str).str.strip() == ""
                    df_open = df[date_blank & manual_blank]

                    rev_keys, added, dropped = self._merge_revision_rows(
                        df_open, when, source_label, unique_rows,
                        episode_state=episode_state,
                        resolution_observed=pending_when,
                    )
                    rejected_entries += dropped
                    if dropped:
                        walk_complete = False

                    if newest_processed is None or when > newest_processed:
                        newest_processed = when
                    if pending_keys is not None:
                        self._record_revision_delta(
                            source_label, pending_when, pending_keys,
                            when, rev_keys, jump_rows, affected_rows,
                            revision_rows,
                        )
                    pending_when, pending_keys = when, rev_keys

                    last_kept_when = when
                    kept += 1
                    print(
                        f"{path} - Revision {kept}: {rev} from {when} "
                        f"(+{added} new keys; total {len(unique_rows)})"
                    )
                except Exception as e:
                    walk_complete = False
                    print(
                        f"WARNING: Skipping revision {rev} from {when} "
                        f"- {type(e).__name__}: {str(e)[:100]}"
                    )
                    continue

            before_rev = entries[-1]["rev"]

        # Oldest processed revision has no older baseline to diff
        # against — record its count with blank deltas.
        if pending_keys is not None:
            revision_rows.append({
                'source': source_label,
                'revision_when': pending_when.isoformat(),
                'open_entries': len(pending_keys),
                'added_since_prev': '',
                'removed_since_prev': '',
            })

        if oldest_revision and newest_revision:
            days_span = (newest_revision - oldest_revision).days
            print(
                f"Done with {path}. Kept {kept} revisions over {days_span} days "
                f"({oldest_revision} -> {newest_revision})."
            )
        else:
            print(f"Done with {path}. Kept {kept} revisions.")
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

    def _record_revision_delta(self, source_label, newer_when, newer_keys,
                               older_when, older_keys, jump_rows,
                               affected_rows, revision_rows):
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
            'revision_when': newer_when.isoformat(),
            'open_entries': len(newer_keys),
            'added_since_prev': len(added_keys),
            'removed_since_prev': len(removed_keys),
        })
        for direction, keys in (('added', added_keys),
                                ('removed', removed_keys)):
            if len(keys) < self.jump_threshold:
                continue
            jump_id = (
                f"{source_label}_{newer_when.strftime('%Y%m%dT%H%M%SZ')}"
                f"_{direction}"
            )
            sorted_keys = sorted(keys)
            sample = '; '.join(
                f"{k[0]}/{k[1]}/{k[2]}/{k[3]}" for k in sorted_keys[:10]
            )
            jump_rows.append({
                'jump_id': jump_id,
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
                affected_rows.append({
                    'jump_id': jump_id,
                    'Subject': k[0],
                    'Timepoint': k[1],
                    'General_Flag': k[2],
                    'variable': k[3],
                    'canonical_template': k[4],
                })

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
                             episode_state=None, resolution_observed=None):
        """Merge one revision's unresolved rows into unique_rows.

        Each Specific Flags entry is keyed independently. Raw messages are kept
        per tracker source so the value used as the pre-resolution observation
        comes from the same production lifecycle as Latest_seen. At a duplicate
        timestamp an inferable message wins over a prose-only copy; conflicting
        inferable values are retained for audit but marked unusable.

        ``episode_state`` is supplied by the newest-to-oldest revision walk.  A
        logical key is reused only while it is present in consecutive processed
        snapshots.  Reappearance after an absent snapshot allocates a distinct
        occurrence record and records that newer absent snapshot as the first
        time resolution was observed.  Omitting the state preserves the legacy
        helper behavior used by direct callers that merge isolated frames.
        """

        before = len(unique_rows)
        rejected = 0
        rev_keys = set()
        lifecycle_mode = episode_state is not None
        if lifecycle_mode:
            newer_active = dict(episode_state.get('active', {}))
            next_ordinal = episode_state.setdefault('next_ordinal', {})
            current_active = {}
        else:
            newer_active = {}
            next_ordinal = {}
            current_active = {}
        for _, row in df_open.iterrows():
            subject = str(row["Subject"]).strip()
            if not subject:
                continue
            tp = normalize_timepoint(row["Timepoint"])
            general_flag = normalize_general_flag(row["General Flag"])
            if not general_flag:
                continue
            specific = (
                str(row["Specific Flags"])
                if row["Specific Flags"] is not None else '')
            if not specific.strip():
                continue
            raw_entries = split_specific_flag_entries(specific)
            for raw in raw_entries:
                var, msg = split_entry(raw)
                if not var:
                    rejected += 1
                    continue
                canonical_raw = canonicalize_message(var, msg)
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
                        key = (*logical_key, ordinal)
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
                    observations[source_label] = self._merge_message_observation(
                        observations.get(source_label), candidate)
                    rec['sources'].add(source_label)

                if new_episode and resolution_observed is not None:
                    rec.setdefault('resolution_per_source', {})[
                        source_label] = resolution_observed

                self._refresh_representative_message(rec)
        if lifecycle_mode:
            episode_state['active'] = current_active
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

        is_currently_open = entry's per-source latest in the live
        source is >= that source's newest revision. An entry that
        lingered in a non-live (testing) tracker but isn't in the
        live tracker isn't really "currently open" from the QC team's
        perspective — which was the previous logic's failure mode.
        """
        cols = [
            "Subject", "Timepoint", "General_Flag", "variable",
            "message", "canonical_template", "episode_id",
            "Earliest_seen", "Latest_seen", "Resolution_observed",
            "source", "is_currently_open",
            "message_variable", "message_source",
            "message_value_conflict", "resolution_eligible",
        ]
        live_source = self.LIVE_SOURCE_PER_NETWORK.get(network)
        live_cutoff = newest_per_source.get(live_source) if live_source else None

        if not unique_rows:
            print(f"No entries collected for {network}; writing empty CSV.")
            out_df = pd.DataFrame(columns=cols)
        else:
            rows = []
            for storage_key, rec in unique_rows.items():
                logical_key = tuple(storage_key[:5])
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
                    is_open = bool(
                        live_latest is not None and live_latest >= live_cutoff
                    )
                else:
                    is_open = False

                resolution_observed = rec.get(
                    'resolution_per_source', {}).get(latest_source)
                if is_open:
                    # A currently open occurrence has not resolved even if a
                    # non-live source recorded an earlier disappearance.
                    resolution_observed = None

                rows.append({
                    "Subject": subject,
                    "Timepoint": tp,
                    "General_Flag": general_flag,
                    "variable": var,
                    "message": message_observation['message'],
                    "canonical_template": canonical,
                    "episode_id": self._stable_episode_id(
                        logical_key, earliest_csv),
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
                })
            out_df = pd.DataFrame(rows, columns=cols)

        out_csv = os.path.join(
            self.analyze_flags_dir,
            open_tracker_row_history_csv_basename(network),
        )
        out_df.to_csv(out_csv, index=False)
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
        cols = ['source', 'revision_when', 'open_entries',
                'added_since_prev', 'removed_since_prev']
        df = pd.DataFrame(revision_rows, columns=cols)
        if not df.empty:
            df = df.sort_values(['source', 'revision_when']).reset_index(drop=True)
        df.to_csv(path, index=False)
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
        metadata = {
            'network': network,
            'live_source': self.LIVE_SOURCE_PER_NETWORK.get(network),
            'newest_revision_V1': v1.isoformat() if v1 else None,
            'newest_revision_V2': v2.isoformat() if v2 else None,
            'newest_revision_overall': overall.isoformat() if overall else None,
        }
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Wrote metadata: {path}")


if __name__ == '__main__':
    ResolvedEstimator().run_script()
