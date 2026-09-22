"""
Unit tests for the central config.testing_enabled validation.

The valid space is exactly {"True", "False"}. A typo such as "true",
"True " (trailing space), JSON boolean True, integer 1, or "yes"
would silently route a test run to production output paths because
every existing call site compares against the literal string "True"
with == . _validate_testing_enabled raises RuntimeError on anything
else; missing key defaults to "False" (the established existing
behavior at every call site).

These tests exercise _validate_testing_enabled directly and also the
load path via _load_config so a corrupt production config halts at
the first Utils() construction.
"""

import json
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.utils import (
    _CONFIG_CACHE,
    _load_config,
    _validate_testing_enabled,
)


def test_accepts_string_true():
    cfg = {"testing_enabled": "True"}
    _validate_testing_enabled(cfg, "<test>")
    assert cfg["testing_enabled"] == "True"


def test_accepts_string_false():
    cfg = {"testing_enabled": "False"}
    _validate_testing_enabled(cfg, "<test>")
    assert cfg["testing_enabled"] == "False"


def test_missing_key_defaults_to_false():
    cfg = {}
    _validate_testing_enabled(cfg, "<test>")
    assert cfg["testing_enabled"] == "False"


@pytest.mark.parametrize("bad_value", [
    "true",
    "TRUE",
    "True ",
    " True",
    "false",
    "FALSE",
    True,
    False,
    1,
    0,
    "1",
    "0",
    "yes",
    "no",
    None,
])
def test_rejects_typos_and_wrong_types(bad_value):
    cfg = {"testing_enabled": bad_value}
    with pytest.raises(RuntimeError) as ei:
        _validate_testing_enabled(cfg, "/tmp/config.json")
    msg = str(ei.value)
    assert "FATAL" in msg
    assert "testing_enabled" in msg
    assert "/tmp/config.json" in msg


def test_load_config_propagates_validation(tmp_path):
    """End-to-end: a bad value on disk halts _load_config."""
    # Clear any cached entry for this absolute_path so the read fires.
    abs_path = str(tmp_path)
    _CONFIG_CACHE.pop(abs_path, None)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "paths": {
            "dependencies_path": str(tmp_path) + "/",
            "combined_csv_path": str(tmp_path) + "/",
            "output_path": str(tmp_path) + "/",
        },
        "testing_enabled": "true",  # lowercase — the canonical typo
    }))
    with pytest.raises(RuntimeError) as ei:
        _load_config(abs_path)
    assert "FATAL" in str(ei.value)
    # Defensive: ensure the corrupt config did not land in the cache.
    assert abs_path not in _CONFIG_CACHE


def test_load_config_caches_valid(tmp_path):
    abs_path = str(tmp_path)
    _CONFIG_CACHE.pop(abs_path, None)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "paths": {
            "dependencies_path": str(tmp_path) + "/",
            "combined_csv_path": str(tmp_path) + "/",
            "output_path": str(tmp_path) + "/",
        },
        "testing_enabled": "False",
    }))
    cfg = _load_config(abs_path)
    assert cfg["testing_enabled"] == "False"
    # Cached on second call.
    assert _CONFIG_CACHE[abs_path] is cfg
