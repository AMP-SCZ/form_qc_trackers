"""Compatibility entry point for the shared date-report summary utility.

Keep the deployed module command while using the transferred implementation
at the repository root, including resolution filtering and form exclusions.
"""
from pathlib import Path
import importlib
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_impl = importlib.import_module("date_report_form_summary")

if __name__ == "__main__":
    raise SystemExit(_impl.main())
else:
    sys.modules[__name__] = _impl
