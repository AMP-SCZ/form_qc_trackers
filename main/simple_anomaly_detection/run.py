"""Run all configured anomaly detectors.

Point it at your data via the ``FORMQC_ANOMALY_INPUT`` environment variable
(and optionally ``FORMQC_ANOMALY_OUTPUT``); with neither set, the constructor
raises a clear error rather than guessing a path. No CLI args -- just run
this file. For synthetic data instead, use ``python -m simple_anomaly_detection
--demo``.

    FORMQC_ANOMALY_INPUT=/path/to/combined_csvs python simple_anomaly_detection/run.py
"""

import os
import sys

# This script lives inside the package, so for
# ``from simple_anomaly_detection import SimpleAnomalyDetector`` to
# resolve when invoked as a plain script (e.g.
# ``python simple_anomaly_detection/run.py``), the package's PARENT
# directory needs to be on sys.path -- Python only adds the script's
# own directory by default, which would shadow the package.
_HERE = os.path.dirname(os.path.realpath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from simple_anomaly_detection import SimpleAnomalyDetector


if __name__ == "__main__":
    SimpleAnomalyDetector().run()
