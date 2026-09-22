"""Simple anomaly detection for the study's Clinical-measures variables.

A suite of detectors, one combined ranked Excel report plus per-type sheets.
The runner does not read or modify config.json (paths live in
SimpleAnomalyDetector.__init__). Production runs read the repository's
missingness-domain and variable/form dependency JSONs to enforce a fail-closed
clinical scope. The rater_effect detector additionally reads the REDCap data
dictionary (read-only, best-effort, via utils) when it is reachable, uses it to
scope rater variables to their forms, and falls back to
column-name heuristics when it can't (e.g. on Windows, or when the
dictionary is absent).

Entry points:
    python -m simple_anomaly_detection
    python -m simple_anomaly_detection --demo
    from simple_anomaly_detection import SimpleAnomalyDetector
"""

from .runner import SimpleAnomalyDetector
from .demo import build_demo_dataset

__all__ = ["SimpleAnomalyDetector", "build_demo_dataset"]
