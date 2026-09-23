"""Shared path-component rules for generated QC tracker workbooks."""

import re


def sanitize_ra_folder_name(value) -> str:
    """Return the canonical safe folder component for a Melbourne RA."""
    return re.sub(r"[^\w\-]", "_", str(value))[:64] or "unknown_ra"


__all__ = ["sanitize_ra_folder_name"]
