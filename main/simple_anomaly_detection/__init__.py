"""Simple, standalone anomaly detection.

Seven detectors, one combined ranked Excel report plus per-type sheets.
Does not read or modify config.json; paths live in
SimpleAnomalyDetector.__init__.

Entry points:
    python -m simple_anomaly_detection
    python -m simple_anomaly_detection --demo
    from simple_anomaly_detection import SimpleAnomalyDetector
"""

from .runner import SimpleAnomalyDetector
from .demo import build_demo_dataset

__all__ = ["SimpleAnomalyDetector", "build_demo_dataset"]
