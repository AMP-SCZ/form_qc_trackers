"""
Operator entry point for missingness anomaly summary (default-off).

Reads config.json from repo root (parent of run_discovery/).

Writes:
  combined_outputs/discovery/missingness_anomaly_summary.parquet

Does not touch combined_qc_flags or raw data.
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_ancestor_with(*relpaths):
    candidate = _HERE
    for _ in range(6):
        if all(os.path.exists(os.path.join(candidate, r))
               for r in relpaths):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            return None
        candidate = parent
    return None


_QC_TYPES_PARENT = _find_ancestor_with(os.path.join("qc_types", "discovery"))
_UTILS_PARENT = _find_ancestor_with(os.path.join("utils", "utils.py"))
for _p in (_UTILS_PARENT, _QC_TYPES_PARENT):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

_CONFIG_PATH = None
_cfg_anc = _find_ancestor_with("config.json")
if _cfg_anc is not None:
    _CONFIG_PATH = os.path.join(_cfg_anc, "config.json")

from qc_types.discovery.missingness_summary_checks import (
    MissingnessSummaryChecks,
)


def main():
    if _CONFIG_PATH is None:
        print(
            "[run_missingness_summary] FATAL: config.json not found in any "
            f"ancestor of {_HERE}"
        )
        return 1
    config_path = _CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    enabled = bool(
        cfg.get("discovery", {})
        .get("missingness_summary", {})
        .get("enabled", False)
    )
    if not enabled:
        print(
            "[run_missingness_summary] discovery.missingness_summary "
            "disabled in config.json (default). Set "
            "discovery.missingness_summary.enabled=true to run."
        )
        return 0

    MissingnessSummaryChecks(config_dict=cfg)()
    return 0


if __name__ == "__main__":
    sys.exit(main())
