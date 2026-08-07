"""Simple, standalone anomaly detection.

A suite of detectors, one combined ranked Excel report plus per-type sheets.
Pure numpy/pandas at its core; the runner does not read or modify config.json
(paths live in SimpleAnomalyDetector.__init__). The one exception is the
rater_effect detector: when the project's utils / data dictionary are
reachable it reads the REDCap data dictionary (read-only, best-effort, via
utils) to scope rater variables to their forms, and falls back to
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
