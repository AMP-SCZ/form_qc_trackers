"""
Operator entry point for copy-forward candidate detection
(Layer 2 / discovery, default-off).

Reads config.json from the project root. Exits with a single
one-line message if the feature is disabled in config.

Discovery-tier semantics:
  - reads only the long-format multi-tp parquet (does not modify
    any source file)
  - writes only combined_outputs/discovery/copy_forward_candidates
    .parquet
  - DOES NOT touch combined_qc_flags.parquet
  - DOES NOT upload to Dropbox
  - exit code 0 on success (including the disabled-by-config
    no-op case); non-zero only on a true error
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

from qc_types.discovery.copy_forward_checks import (
    CopyForwardChecks)


def main():
    if _CONFIG_PATH is None:
        print(
            "[run_copy_forward] FATAL: config.json not found in any "
            f"ancestor of {_HERE}"
        )
        return 1
    config_path = _CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    enabled = bool(
        cfg.get('discovery', {})
           .get('copy_forward', {})
           .get('enabled', False))
    if not enabled:
        print(
            "[run_copy_forward] discovery.copy_forward disabled "
            "in config.json (default). Set "
            "discovery.copy_forward.enabled=true to run."
        )
        return 0

    detector = CopyForwardChecks(config_dict=cfg)
    detector.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
