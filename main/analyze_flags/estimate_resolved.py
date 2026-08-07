"""Walk every Dropbox revision of the combined network trackers and
emit a deduplicated **entry-level** history per network — one row per
unique (Subject, Timepoint, General_Flag, variable, canonical_template)
specific-flag episode, with the date bounds and source tracker(s)
that contributed to it.

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

A companion ``tracker_metadata_{network}.json`` records the newest
revision timestamp per source so consumers can determine
``is_currently_open`` against the right tracker (V1-only entries
get compared to V1's newest; entries seen in V2 get compared to
V2's newest). Newest is the newest SUCCESSFULLY PROCESSED revision —
a corrupt/undownloadable latest snapshot must not silently mark every
entry resolved.

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

import pandas as pd
import os
import sys
import json
import dropbox
from io import BytesIO
from datetime import datetime, timedelta, timezone

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
print(parent_dir)
from utils.utils import Utils
from analyze_flags.canonicalize import (
    canonicalize_message,
    explode_specific_flags,
    normalize_general_flag,
    normalize_timepoint,
    split_entry,
)
from analyze_flags.paths import (
    ensure_analyze_flags_artifact_dir,
    open_tracker_row_history_csv_basename,
    pipeline_output_path_from_config,
    revision_counts_csv_basename,
    tracker_metadata_json_basename,
)
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
    # the live tracker that QC analysts use day to day.
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

    def __init__(self, sample_every_days=1, jump_threshold=50):
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
        # Default 1 = daily sampling, matching visualize_data/graph_errors.py.
        # Raise to speed runs at the cost of granularity.
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
            any_walk_completed = False
            # Per-network manual-review accumulators, filled by the
            # revision-delta tracking inside _walk_revisions.
            jump_rows = []
            affected_rows = []
            revision_rows = []
            self._current_network = network
            for base, basename, source_label in self._tracker_paths(network):
                full_path = f'{base}{network}/combined/{basename}'
                try:
                    rejected, newest_processed = self._walk_revisions(
                        full_path, source_label, unique_rows,
                        jump_rows, affected_rows, revision_rows,
                        days_back=None,
                    )
                    rejected_entries += rejected
                    if newest_processed is not None:
                        prev = newest_per_source[source_label]
                        if prev is None or newest_processed > prev:
                            newest_per_source[source_label] = newest_processed
                    any_walk_completed = True
                except Exception as e:
                    # Missing V1 file / transient API error shouldn't kill
                    # the whole network — log and move on so the other
                    # tracker (or the next network) still gets processed.
                    print(
                        f"WARNING: failed walking revisions for {full_path}: "
                        f"{type(e).__name__}: {str(e)[:200]}"
                    )
                    continue
            if not any_walk_completed:
                # Both tracker walks raised (expired token, network down).
                # Writing the empty unique_rows would silently overwrite
                # the last good CSV; skip so the operator can re-run
                # once the underlying issue is fixed.
                print(
                    f"All tracker walks for {network} failed with exceptions; "
                    f"preserving any existing CSV / metadata (no write)."
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

        Returns (rejected_entry_count, newest_revision_processed) so
        the caller can roll up stats and write metadata.
        ``newest_revision_processed`` is the newest revision that was
        successfully downloaded AND merged — NOT just listed — so a
        corrupt latest snapshot can't silently mark everything resolved.
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
        # Previous (newer) successfully processed revision, used for
        # the revision-delta / jump detection. The walk goes newest ->
        # oldest, so deltas are attributed to pending_when (the newer
        # revision, i.e. when the change became visible).
        pending_when = None
        pending_keys = None

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
                    df = pd.read_excel(BytesIO(resp.content), keep_default_na=False)

                    # V1 -> V2 rename, but only for columns that exist and
                    # whose V2 target isn't already present (so a
                    # transitional revision carrying both names doesn't
                    # lose data).
                    rename_map = {
                        old: new for old, new in rename_v1_to_v2.items()
                        if old in df.columns and new not in df.columns
                    }
                    if rename_map:
                        df = df.rename(columns=rename_map)

                    missing = [c for c in v2_cols if c not in df.columns]
                    if missing:
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
                    )
                    rejected_entries += dropped

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
        if (newest_revision is not None and newest_processed is not None
                and newest_processed < newest_revision):
            print(
                f"WARNING: newest listed revision of {path} "
                f"({newest_revision}) failed to process; openness will "
                f"be judged against {newest_processed} instead."
            )

        return rejected_entries, newest_processed

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

    def _merge_revision_rows(self, df_open, when, source_label, unique_rows):
        """Merge one revision's unresolved rows into ``unique_rows`` at
        the entry level: each row's Specific_Flags is exploded into
        (variable, message) pairs, each becomes its own dedup key.

        The operator's manual mapping is applied BEFORE keying: the
        canonical template goes through ``self.template_map`` and the
        variable through ``self.variable_map``, so declared renames
        merge into one episode. (Canonicalization itself still runs on
        the RAW variable, since the message text mentions the raw name.)

        Tracks per-source earliest/latest so the consumer can determine
        is_currently_open against the LIVE source's snapshot (rather
        than the overall max, which would falsely mark entries
        resolved if they were dropped from a non-live testing tracker).

        Returns (revision_key_set, new_keys_added,
        rejected_entry_count) — the key set feeds the revision-delta /
        jump detection in the caller.
        """
        before = len(unique_rows)
        rejected = 0
        rev_keys = set()
        for _, row in df_open.iterrows():
            subject = str(row["Subject"]).strip()
            if not subject:
                continue
            tp = normalize_timepoint(row["Timepoint"])
            general_flag = normalize_general_flag(row["General Flag"])
            if not general_flag:
                continue
            specific = str(row["Specific Flags"]) if row["Specific Flags"] is not None else ''
            if not specific.strip():
                continue
            raw_entries = [e.strip() for e in specific.split('|') if e.strip()]
            for raw in raw_entries:
                var, msg = split_entry(raw)
                if not var:
                    rejected += 1
                    continue
                canonical_raw = canonicalize_message(var, msg)
                canonical = self.template_map.get(canonical_raw, canonical_raw)
                var_mapped = self.variable_map.get(var, var)
                key = (subject, tp, general_flag, var_mapped, canonical)
                rev_keys.add(key)
                rec = unique_rows.get(key)
                if rec is None:
                    self._track_seen(canonical_raw, var, general_flag)
                    unique_rows[key] = {
                        'earliest_per_source': {source_label: when},
                        'latest_per_source': {source_label: when},
                        'sources': {source_label},
                        'message': msg,
                    }
                else:
                    # Per-source earliest / latest. We don't track an
                    # "overall" min/max here — the writer computes them
                    # from per-source dicts when emitting the CSV.
                    src_earliest = rec['earliest_per_source'].get(source_label)
                    if src_earliest is None or when < src_earliest:
                        rec['earliest_per_source'][source_label] = when
                    src_latest = rec['latest_per_source'].get(source_label)
                    if src_latest is None or when > src_latest:
                        rec['latest_per_source'][source_label] = when
                    # Keep the most recent revision's raw message as
                    # the 'example' so the stored example reflects
                    # what's in the tracker today (or last did). The
                    # comparison uses overall latest across all sources
                    # because msg is purely informational.
                    prev_overall_latest = max(rec['latest_per_source'].values())
                    if when >= prev_overall_latest:
                        rec['message'] = msg
                    rec['sources'].add(source_label)
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
            "message", "canonical_template",
            "Earliest_seen", "Latest_seen", "source", "is_currently_open",
        ]
        live_source = self.LIVE_SOURCE_PER_NETWORK.get(network)
        live_cutoff = newest_per_source.get(live_source) if live_source else None

        if not unique_rows:
            print(f"No entries collected for {network}; writing empty CSV.")
            out_df = pd.DataFrame(columns=cols)
        else:
            rows = []
            for (subject, tp, general_flag, var, canonical), rec in unique_rows.items():
                sources = rec['sources']
                source_str = '+'.join(sorted(sources))

                # Earliest across all sources (full history); latest
                # anchored to the live source's view when present,
                # otherwise the overall max across whatever sources we
                # DID see it in.
                earliest_csv = min(rec['earliest_per_source'].values())
                if live_source in rec['latest_per_source']:
                    latest_csv = rec['latest_per_source'][live_source]
                else:
                    latest_csv = max(rec['latest_per_source'].values())

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

                rows.append({
                    "Subject": subject,
                    "Timepoint": tp,
                    "General_Flag": general_flag,
                    "variable": var,
                    "message": rec['message'],
                    "canonical_template": canonical,
                    "Earliest_seen": earliest_csv,
                    "Latest_seen": latest_csv,
                    "source": source_str,
                    "is_currently_open": is_open,
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
