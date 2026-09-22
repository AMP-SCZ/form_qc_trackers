"""Network scope comes from configuration, with an optional per-run override."""

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import utils as utils_module


@pytest.fixture
def make_utils(monkeypatch):
    """Construct Utils without reading config, dependency, or subject files."""
    monkeypatch.delenv("QC_NETWORKS", raising=False)
    monkeypatch.setattr(
        utils_module.Utils, "load_dependency_json", lambda self, filename: {})

    def make(config):
        config_info = {"paths": {"output_path": "unused"}, **config}
        monkeypatch.setattr(utils_module, "_load_config", lambda path: config_info)
        return utils_module.Utils()

    return make


@pytest.mark.parametrize("networks", [
    ["PRONET"],
    ["PRESCIENT"],
    ["PRONET", "PRESCIENT"],
])
def test_config_selects_networks(make_utils, networks):
    utils = make_utils({"pipeline_networks": networks})
    assert utils.pipeline_networks == tuple(networks)


def test_config_normalizes_names_and_preserves_unique_order(make_utils):
    networks = [" prescient ", "ProNET", "PRESCIENT", "pronet"]
    utils = make_utils({"pipeline_networks": networks})
    assert utils.pipeline_networks == ("PRESCIENT", "PRONET")
    assert networks == [" prescient ", "ProNET", "PRESCIENT", "pronet"]


def test_missing_config_networks_requires_explicit_scope(make_utils):
    with pytest.raises(RuntimeError):
        make_utils({})


@pytest.mark.parametrize("networks", [
    [],
    None,
    "PRONET",
    {"PRONET": True},
    ["UNKNOWN"],
])
def test_invalid_config_networks_are_rejected(make_utils, networks):
    with pytest.raises(RuntimeError):
        make_utils({"pipeline_networks": networks})


def test_environment_overrides_config_networks(make_utils, monkeypatch):
    monkeypatch.setenv("QC_NETWORKS", " prescient , PRONET, prescient ")
    utils = make_utils({"pipeline_networks": ["PRONET"]})
    assert utils.pipeline_networks == ("PRESCIENT", "PRONET")


def test_environment_can_supply_scope_without_config_key(make_utils, monkeypatch):
    monkeypatch.setenv("QC_NETWORKS", "PRESCIENT")
    utils = make_utils({})
    assert utils.pipeline_networks == ("PRESCIENT",)


@pytest.mark.parametrize("override", ["", "UNKNOWN", "PRONET,"])
def test_invalid_environment_scope_is_rejected(make_utils, monkeypatch, override):
    monkeypatch.setenv("QC_NETWORKS", override)
    with pytest.raises(RuntimeError):
        make_utils({"pipeline_networks": ["PRONET", "PRESCIENT"]})
