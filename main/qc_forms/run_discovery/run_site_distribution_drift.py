"""
Operator entry point for the per-site distribution-shift detector
(Layer 2 / discovery, default-off).

Reads config.json from the project root. Exits with a single
one-line message if the feature is disabled in config.

Discovery-tier semantics:
  - reads dependencies/multi_tp_{network}_numeric_long.parquet
    and (optionally) dependencies/variable_ranges.json
  - writes only combined_outputs/discovery/
    site_distribution_drift.parquet
  - DOES NOT touch combined_qc_flags.parquet
  - DOES NOT upload to Dropbox
  - exit code 0 on success (including the disabled-by-config
    no-op case); non-zero only on a true error
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

from qc_types.discovery.site_distribution_drift_checks import (  # noqa: E402
    SiteDistributionDriftChecks)


def main():
    if _CONFIG_PATH is None:
        print(
            "[run_site_distribution_drift] FATAL: config.json not "
            f"found in any ancestor of {_HERE}"
        )
        return 1
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    try:
        from qc_types.discovery._common import is_truthy_enabled
        enabled = is_truthy_enabled(
            cfg.get('discovery', {})
               .get('site_distribution_drift', {})
               .get('enabled', False))
    except ImportError:
        # Defensive fallback: keep the old loose semantics if
        # _common isn't importable for any reason.
        enabled = bool(
            cfg.get('discovery', {})
               .get('site_distribution_drift', {})
               .get('enabled', False))
    if not enabled:
        print(
            "[run_site_distribution_drift] "
            "discovery.site_distribution_drift disabled in "
            "config.json (default). Set "
            "discovery.site_distribution_drift.enabled=true to run.")
        return 0

    detector = SiteDistributionDriftChecks(config_dict=cfg)
    detector.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
