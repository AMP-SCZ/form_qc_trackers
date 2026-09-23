"""Production package and compatibility paths for the deployed source layout.

Legacy modules import sibling packages without the ``main.`` prefix. Keep those
imports available while retaining the deployed ``main/qc_forms`` directory.
"""

from pathlib import Path
import sys

_PACKAGE_ROOT = Path(__file__).resolve().parent
_QC_ROOT = _PACKAGE_ROOT / "qc_forms"
__path__ = [str(_PACKAGE_ROOT), str(_QC_ROOT)]
for _directory in (_PACKAGE_ROOT, _QC_ROOT):
    if str(_directory) not in sys.path:
        sys.path.append(str(_directory))
