"""
Unit tests for Utils.load_dependency_json (release-hardening RH-1).

Corrupt JSON must raise RuntimeError — never silently return {}.
"""

import json
import os
import sys
import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.utils import Utils, _project_root


def test_project_root_is_platform_native():
    expected = os.path.dirname(os.path.dirname(os.path.realpath(
        os.path.join(_REPO_ROOT, "utils", "utils.py")
    )))
    assert os.path.normcase(_project_root()) == os.path.normcase(expected)
    assert os.path.isfile(os.path.join(_project_root(), "config.json"))


@pytest.fixture
def utils_with_tmp_dep(tmp_path, monkeypatch):
    """Utils instance whose dependencies_path is tmp_path."""
    dep = tmp_path / "dependencies"
    dep.mkdir()
    cfg = {
        "paths": {
            "dependencies_path": str(dep) + os.sep,
            "combined_csv_path": str(tmp_path / "csv") + os.sep,
            "output_path": str(tmp_path / "out") + os.sep,
        }
    }
    # Bypass Utils.__init__ disk reads — only need config_info + load_dependency_json
    u = object.__new__(Utils)
    u.config_info = cfg
    u.absolute_path = str(tmp_path)
    return u, dep


def test_load_dependency_json_corrupt_raises(utils_with_tmp_dep):
    utils, dep = utils_with_tmp_dep
    bad = dep / "broken.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(RuntimeError) as ei:
        utils.load_dependency_json("broken.json")
    assert "FATAL" in str(ei.value)
    assert "broken.json" in str(ei.value)


def test_load_dependency_json_valid_roundtrip(utils_with_tmp_dep):
    utils, dep = utils_with_tmp_dep
    good = dep / "ok.json"
    payload = {"a": 1, "b": [2, 3]}
    good.write_text(json.dumps(payload), encoding="utf-8")
    out = utils.load_dependency_json("ok.json")
    assert out == payload
