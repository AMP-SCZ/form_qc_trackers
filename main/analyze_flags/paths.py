"""Shared output locations for analyze_flags (prototype tooling)."""

import os


def pipeline_output_path_from_config(config_info: dict) -> str:
    """Match qc_forms_main / pipeline: base output_path plus testing/ when enabled."""
    out = config_info["paths"]["output_path"]
    if config_info.get("testing_enabled") == "True":
        out = f"{out}testing/"
    return out


def analyze_flags_artifact_dir(output_path: str) -> str:
    """Dedicated artifact root under combined_outputs; paths are never cwd-relative."""
    return os.path.normpath(
        os.path.join(output_path, "combined_outputs", "analyze_flags")
    )


def ensure_analyze_flags_artifact_dir(output_path: str) -> str:
    d = analyze_flags_artifact_dir(output_path)
    os.makedirs(d, exist_ok=True)
    return d


def open_tracker_row_history_csv_basename(network_name: str) -> str:
    """Stable CSV name for Dropbox-revision open-row history (one file per network).

    The file is now keyed at the specific-flag entry level
    (Subject, Timepoint, General_Flag, variable, canonical_template),
    not the row level — so the basename is historical, but the content
    is per-entry. Consumers should also read the companion metadata
    JSON for cross-source newest-revision timestamps.
    """
    return f"open_tracker_row_history_{network_name}.csv"


def tracker_metadata_json_basename(network_name: str) -> str:
    """Stable JSON name for per-network tracker walk metadata.

    Contains newest_revision_V1, newest_revision_V2, and
    newest_revision_overall so consumers can determine whether an
    entry is still open against the right source's latest snapshot
    rather than inferring it from max(Latest_seen) over the history.
    """
    return f"tracker_metadata_{network_name}.json"


def jumps_workbook_basename(network_name: str) -> str:
    """Per-network manual-review workbook of detected count jumps.

    Sheet 'jumps' has one row per detected mass addition/removal with
    an ``include`` column for the operator to fill in; sheet
    'affected_entries' lists the exact entry keys each jump touched so
    consumers can apply the decision precisely (not just by date).
    """
    return f"jumps_{network_name}.xlsx"


def revision_counts_csv_basename(network_name: str) -> str:
    """Per-network audit CSV: one row per processed Dropbox revision
    with its open-entry count and the +added/-removed delta vs. the
    previous (older) processed revision of the same source."""
    return f"revision_counts_{network_name}.csv"


def template_mapping_basename() -> str:
    """Single cross-network manual-mapping workbook. Sheet 'templates'
    maps canonical_template -> maps_to; sheet 'variables' maps
    variable -> maps_to. Blank maps_to = identity (no merge)."""
    return "template_mapping.xlsx"
