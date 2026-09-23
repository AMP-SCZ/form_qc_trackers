"""Compatibility logger for legacy modules and tests.

Applications remain responsible for configuring handlers and verbosity.  This
module deliberately performs no global logging configuration on import.
"""

import logging


logger = logging.getLogger("formqc")
