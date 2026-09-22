"""
Operator entry point for the mixed-effects site trajectory slope
outlier detector (Layer 2 / discovery, default-off).

Reads config.json from the project root.

Discovery-tier semantics:
  - writes only combined_outputs/discovery/
    site_trajectory_slope.parquet
  - DOES NOT touch combined_qc_flags.parquet
  - DOES NOT upload to Dropbox
  - requires statsmodels for the MixedLM fit
"""

import json
import os
import sys

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


_QC_TYPES_PARENT = _find_ancestor_with(
    os.path.join("qc_types", "discovery"))
_UTILS_PARENT = _find_ancestor_with(
    os.path.join("utils", "utils.py"))
for _p in (_UTILS_PARENT, _QC_TYPES_PARENT):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

_CONFIG_PATH = None
_cfg_anc = _find_ancestor_with("config.json")
if _cfg_anc is not None:
    _CONFIG_PATH = os.path.join(_cfg_anc, "config.json")

from qc_types.discovery.site_trajectory_slope_checks import (  # noqa: E402
    SiteTrajectorySlopeChecks)


def main():
    if _CONFIG_PATH is None:
        print(
            "[run_site_trajectory_slope] FATAL: config.json not "
            f"found in any ancestor of {_HERE}"
        )
        return 1
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    try:
        from qc_types.discovery._common import is_truthy_enabled
        enabled = is_truthy_enabled(
            cfg.get('discovery', {})
               .get('site_trajectory_slope', {})
               .get('enabled', False))
    except ImportError:
        # Defensive fallback: keep the old loose semantics if
        # _common isn't importable for any reason.
        enabled = bool(
            cfg.get('discovery', {})
               .get('site_trajectory_slope', {})
               .get('enabled', False))
    if not enabled:
        print(
            "[run_site_trajectory_slope] "
            "discovery.site_trajectory_slope disabled in "
            "config.json (default). Set "
            "discovery.site_trajectory_slope.enabled=true to run.")
        return 0

    detector = SiteTrajectorySlopeChecks(config_dict=cfg)
    detector.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
