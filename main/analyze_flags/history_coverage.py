"""Reject history rebuilds that have lost previously collected revisions.

This guard preserves the last complete output; it does not reconstruct expired
Dropbox revisions from aggregate history rows. Full-resolution coverage uses
revision IDs or legacy timestamp multiplicity. Daily coverage retains each
source/calendar day's latest snapshot. Pre-audit history offers only a
conservative overall lower bound on the coverage that must remain available.
"""

from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from analyze_flags.paths import (
    open_tracker_row_history_csv_basename,
    revision_counts_csv_basename,
)


_SOURCES = frozenset({"V1", "V2"})


def _read_prior_csv(path, required, network):
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as error:
        raise RuntimeError(
            f"{network}: cannot validate prior history coverage: "
            f"cannot read {path.name}: {error}"
        ) from error
    missing = set(required) - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"{network}: cannot validate prior history coverage: "
            f"{path.name} is missing required columns {sorted(missing)}"
        )
    return frame


def _timestamp(value, network, label):
    try:
        if value is None or not str(value).strip():
            raise ValueError("blank timestamp")
        timestamp = pd.to_datetime(value, utc=True, errors="raise")
        if not isinstance(timestamp, pd.Timestamp) or pd.isna(timestamp):
            raise ValueError("invalid timestamp")
        return timestamp
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            f"{network}: cannot validate history coverage: "
            f"invalid {label} timestamp {value!r}"
        ) from error


def _revision_records(rows, network, label):
    records = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"{network}: malformed {label} revision coverage row")
        source = str(row.get("source", "")).strip()
        if source not in _SOURCES:
            raise RuntimeError(
                f"{network}: invalid source {source!r} in {label} revision coverage"
            )
        records.append({
            "source": source,
            "when": _timestamp(row.get("revision_when"), network, label),
            "id": str(row.get("revision_id") or "").strip(),
            "path": str(row.get("source_path") or "").strip(),
        })
    return records


def _validate_audit_coverage(prior, current, network):
    old_ids = Counter((row["source"], row["id"]) for row in prior if row["id"])
    new_ids = Counter((row["source"], row["id"]) for row in current if row["id"])
    old_times = Counter((row["source"], row["when"]) for row in prior if not row["id"])
    lost_ids = old_ids - new_ids
    # A current revision that satisfies an ID-backed prior row cannot also
    # satisfy a separate legacy row with the same timestamp.
    pending_ids = old_ids.copy()
    new_times = Counter()
    for row in current:
        identity = (row["source"], row["id"])
        if row["id"] and pending_ids[identity]:
            pending_ids[identity] -= 1
        else:
            new_times[(row["source"], row["when"])] += 1
    lost_times = old_times - new_times
    if lost_ids or lost_times:
        missing = sum(lost_ids.values()) + sum(lost_times.values())
        raise RuntimeError(
            f"{network}: revision coverage lost {missing} previously collected "
            "snapshot(s), possibly because Dropbox retention expired; "
            "preserving prior history instead of replacing it with a shorter history"
        )



def _validate_source_paths(prior, current, network):
    # Paths are optional for compatibility with older audit files. When both
    # generations record them, a source must not silently switch environments.
    for source in _SOURCES:
        old_paths = {row["path"] for row in prior
                     if row["source"] == source and row["path"]}
        new_paths = {row["path"] for row in current
                     if row["source"] == source and row["path"]}
        if old_paths and new_paths and old_paths != new_paths:
            raise RuntimeError(
                f"{network}: {source} history coverage source path changed; "
                "preserving prior history"
            )


def _calendar_day(when, day_timezone, network):
    try:
        # Selection uses datetime.astimezone(ZoneInfo), so validation must use
        # that same conversion. Older pandas versions can mishandle ZoneInfo
        # inside Timestamp.tz_convert and group distinct local days together.
        # Convert the UTC Timestamp to a plain datetime before changing zones;
        # keep the original Timestamp for exact snapshot-order comparisons.
        return when.to_pydatetime(warn=False).astimezone(day_timezone).date()
    except (TypeError, ValueError, KeyError) as error:
        raise RuntimeError(
            f"{network}: cannot validate history coverage with calendar-day "
            f"timezone {day_timezone!r}"
        ) from error


def _latest_by_day(records, network, day_timezone, *, require_unique=False):
    days = {}
    for row in records:
        key = (row["source"], _calendar_day(row["when"], day_timezone, network))
        if require_unique and key in days:
            raise RuntimeError(
                f"{network}: invalid daily revision coverage: multiple staged "
                f"snapshots for {key[0]}/{key[1].isoformat()}"
            )
        days[key] = max(days.get(key, row["when"]), row["when"])
    return days


def _validate_daily_coverage(prior, new_days, network, day_timezone):
    old_days = _latest_by_day(prior, network, day_timezone)
    missing = old_days.keys() - new_days.keys()
    if missing:
        rendered = ", ".join(f"{source}/{day.isoformat()}" for source, day in sorted(missing))
        raise RuntimeError(
            f"{network}: revision coverage lost {len(missing)} previously "
            f"collected calendar day(s): {rendered}; preserving prior history"
        )
    older = [key for key, when in old_days.items() if new_days[key] < when]
    if older:
        rendered = ", ".join(f"{source}/{day.isoformat()}" for source, day in sorted(older))
        raise RuntimeError(
            f"{network}: revision coverage would replace the latest previously "
            f"collected snapshot with an older snapshot for calendar day(s): "
            f"{rendered}; preserving prior history"
        )


def validate_revision_coverage(artifact_dir, network, revision_rows, *,
                               sample_every_days=0, day_timezone="America/New_York"):
    """Raise ``RuntimeError`` if staged revisions lose known prior coverage.

    Missing prior artifacts are valid for a first run. Present but unreadable or
    malformed prior artifacts fail closed. The check is independent of flag
    counts, mappings, and resolution state, all of which may legitimately change.
    Daily sampling retains each prior source/day's latest observation rather
    than every intraday revision ID. An even newer same-day revision is valid.
    Modes other than daily sampling retain exact revision coverage as before.
    """
    current = _revision_records(revision_rows, network, "staged")
    if sample_every_days == 1:
        try:
            if not isinstance(day_timezone, ZoneInfo):
                day_timezone = ZoneInfo(day_timezone)
        except (TypeError, ValueError, ZoneInfoNotFoundError) as error:
            raise RuntimeError(
                f"{network}: cannot validate history coverage with calendar-day "
                f"timezone {day_timezone!r}"
            ) from error
        current_days = _latest_by_day(
            current, network, day_timezone, require_unique=True)
    artifact_dir = Path(artifact_dir)
    audit_path = artifact_dir / revision_counts_csv_basename(network)
    if audit_path.exists():
        frame = _read_prior_csv(audit_path, {"source", "revision_when"}, network)
        prior = _revision_records(frame.to_dict("records"), network, "prior")
        if sample_every_days == 1:
            _validate_daily_coverage(prior, current_days, network, day_timezone)
        else:
            _validate_audit_coverage(prior, current, network)
        _validate_source_paths(prior, current, network)
        return

    history_path = artifact_dir / open_tracker_row_history_csv_basename(network)
    if not history_path.exists():
        return
    history = _read_prior_csv(history_path, {"source", "Earliest_seen"}, network)
    if history.empty:
        return

    represented_sources = set()
    previous_times = []
    for row in history.to_dict("records"):
        sources = {source.strip() for source in row["source"].split("+")}
        if not sources or not sources.issubset(_SOURCES):
            raise RuntimeError(
                f"{network}: invalid prior history coverage source {row['source']!r}"
            )
        represented_sources.update(sources)
        previous_times.append(_timestamp(row["Earliest_seen"], network, "Earliest_seen"))

    staged_sources = {row["source"] for row in current}
    if not represented_sources.issubset(staged_sources):
        raise RuntimeError(
            f"{network}: history coverage lost previously represented source(s) "
            f"{sorted(represented_sources - staged_sources)}"
        )
    staged_times = [row["when"] for row in current
                    if row["source"] in represented_sources]
    prior_oldest, staged_oldest = min(previous_times), min(staged_times)
    if sample_every_days == 1:
        coverage_lost = (
            _calendar_day(prior_oldest, day_timezone, network)
            < _calendar_day(staged_oldest, day_timezone, network)
        )
    else:
        coverage_lost = prior_oldest < staged_oldest
    if coverage_lost:
        raise RuntimeError(
            f"{network}: history coverage no longer reaches the prior "
            f"Earliest_seen ({min(previous_times).isoformat()}); "
            "preserving prior history because older Dropbox revisions are missing"
        )
