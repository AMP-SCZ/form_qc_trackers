"""
Sprint 1 P0-2: tests for qc_types.discovery._common.is_truthy_enabled.

The legacy `bool(cfg.get('enabled', False))` pattern is dangerously
permissive: `bool('false') == True`. An operator who types
`"enabled": "false"` in config.json would silently turn the detector
ON. is_truthy_enabled is strict.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc_types.discovery._common import is_truthy_enabled


# -------------------------- accepts ------------------------------

@pytest.mark.parametrize("value", [
    True,
    1,
    "true",
    "True",
    "TRUE",
    "  true  ",   # whitespace tolerated
    "tRuE",
])
def test_accepts(value):
    assert is_truthy_enabled(value) is True


# -------------------------- rejects ------------------------------

@pytest.mark.parametrize("value", [
    # The critical case: the legacy parser accepted these silently.
    "false",
    "False",
    "FALSE",
    "  false  ",
    "no",
    "yes",
    # JSON-null / Python-None.
    None,
    # Python False / numeric zero / other ints.
    False,
    0,
    2,
    -1,
    # Misc types.
    [],
    [True],
    {},
    {"enabled": True},
    "",
    " ",
    "1",  # string "1" is NOT accepted; only literal int 1
    "0",
])
def test_rejects(value):
    assert is_truthy_enabled(value) is False


def test_bool_subclass_int_distinguished():
    """
    Defensive: bool is an int subclass in Python. is_truthy_enabled
    should still treat True/False correctly and not double-handle
    them via the int branch.
    """
    assert is_truthy_enabled(True) is True
    assert is_truthy_enabled(False) is False
