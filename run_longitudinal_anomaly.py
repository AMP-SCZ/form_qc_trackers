#!/usr/bin/env python3
"""
Top-level CLI entry point for the longitudinal anomaly detection
feature (N1 v1).

Behavior:
  1. Read config.json from the project root (next to this file).
  2. If config['longitudinal_anomaly']['enabled'] is not exactly
     the boolean True, print a clear "feature disabled" message
     and exit 0 — the feature is opt-in by design.
  3. If config['longitudinal_anomaly']['merge_into_main_tracker']
     is true, refuse and exit 1: tracker integration is a future
     N2 deliverable, not part of v1's separate-artifact-only flow.
  4. Otherwise: run the Step 1 producer (writes per-network long
     parquets under dependencies/), then the Step 2 detector
     (writes a single anomaly-flags parquet under
     combined_outputs/longitudinal_anomalies/).

What this runner is NOT:
  - No CLI argument parsing in v1. Everything is driven by
    config.json so that there's a single source of truth for
    operators to tune. Adding --threshold-sweep, --dry-run, etc.
    is deferred to a later version per the N1 scope.
  - No Dropbox interaction.
  - No appending into combined_qc_flags.parquet.
  - No deletions or overwrites of any artifact owned by other
    pipeline stages.

Exceptions from the producer or detector propagate. Their FATAL:
messages are already operator-friendly, and propagating preserves
the Python traceback for debugging. Non-zero process exit code
distinguishes real errors from the "feature disabled" no-op
(which exits 0).
"""

import os
import sys
import json

# Layout-aware sys.path. The repo ships with three known layouts:
#   - flat (local workspace): subpackages directly under {repo}/.
#   - main-nested (production): subpackages under {repo}/main/.
#   - qc_forms-nested: subpackages under {repo}/qc_forms/.
# The producer and detector may also live in DIFFERENT parents on
# the same host (e.g. producer under main/, detector under
# qc_forms/). Probe each subpackage independently and add every
# parent that contains one of our files.
_HERE = os.path.dirname(os.path.realpath(__file__))
_CANDIDATES = [
    os.path.join(_HERE, 'main'),
    os.path.join(_HERE, 'main', 'qc_forms'),
    os.path.join(_HERE, 'qc_forms'),
    _HERE,
]


def _find_subpackage_root(subpkg: str, marker_file: str) -> str:
    for candidate in _CANDIDATES:
        if os.path.isfile(os.path.join(candidate, subpkg, marker_file)):
            return candidate
    raise ImportError(
        f"FATAL: cannot locate {subpkg}/{marker_file}. Looked in "
        f"{_CANDIDATES}. Either the longitudinal feature files were "
        f"not deployed to this host, or they live in a directory "
        f"this runner doesn't know about. Place the file in one of "
        f"the candidate parent directories."
    )


_pv_root = _find_subpackage_root(
    'process_variables', 'produce_multi_tp_long.py')
_qt_root = _find_subpackage_root(
    'qc_types', 'longitudinal_anomaly_checks.py')
for root in (_pv_root, _qt_root):
    if root not in sys.path:
        sys.path.insert(1, root)

from process_variables.produce_multi_tp_long import (
    MultiTPLongFormatProducer)
from qc_types.longitudinal_anomaly_checks import (
    LongitudinalAnomalyChecks)


def main():
    config_path = os.path.join(_HERE, 'config.json')
    if not os.path.exists(config_path):
        print(
            f"[run_longitudinal_anomaly] FATAL: config.json not "
            f"found at {config_path}. Cannot continue."
        )
        sys.exit(1)
    with open(config_path, 'r') as f:
        config = json.load(f)

    lon_cfg = config.get('longitudinal_anomaly', {})
    # Strict identity check against True so that string forms like
    # "true" or numeric 1 do NOT enable the feature accidentally.
    # The disabled message tells the operator how to fix that.
    enabled = lon_cfg.get('enabled', False)
    if enabled is not True:
        print(
            "[run_longitudinal_anomaly] feature is disabled "
            "(config['longitudinal_anomaly']['enabled'] is not "
            f"the boolean true; current value: {enabled!r}). "
            "Set it to a JSON boolean true to run. Exiting "
            "cleanly without touching any artifact."
        )
        sys.exit(0)

    if lon_cfg.get('merge_into_main_tracker', False) is True:
        print(
            "[run_longitudinal_anomaly] FATAL: "
            "config['longitudinal_anomaly']"
            "['merge_into_main_tracker'] is true, but N1 v1 does "
            "not support merging into combined_qc_flags.parquet. "
            "Tracker integration is a future N2 deliverable. Set "
            "merge_into_main_tracker back to false to use the v1 "
            "separate-artifact-only flow."
        )
        sys.exit(1)

    print(
        "[run_longitudinal_anomaly] step 1/2: producing "
        "long-format multi-TP parquet ..."
    )
    MultiTPLongFormatProducer(config_dict=config)()

    print(
        "[run_longitudinal_anomaly] step 2/2: running "
        "longitudinal anomaly detector ..."
    )
    LongitudinalAnomalyChecks(config_dict=config)()

    print("[run_longitudinal_anomaly] done.")


if __name__ == '__main__':
    main()
