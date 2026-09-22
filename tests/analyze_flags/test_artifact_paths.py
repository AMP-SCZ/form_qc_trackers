"""Contract tests for analyze_flags output locations (Patch 1)."""

import ast
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from analyze_flags.paths import (
    analyze_flags_artifact_dir,
    ensure_analyze_flags_artifact_dir,
    open_tracker_row_history_csv_basename,
    pipeline_output_path_from_config,
)


def test_paths_module_parses_with_python_310_grammar():
    path = Path(_REPO_ROOT) / "main" / "analyze_flags" / "paths.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path), feature_version=10)


def test_testing_enabled_preserves_testing_suffix():
    cfg = {
        "paths": {"output_path": "/data/qc_outputs/"},
        "testing_enabled": "True",
    }
    assert pipeline_output_path_from_config(cfg) == "/data/qc_outputs/testing/"


def test_testing_enabled_adds_missing_posix_separator():
    cfg = {
        "paths": {"output_path": "/data/qc_outputs"},
        "testing_enabled": "True",
    }
    assert pipeline_output_path_from_config(cfg) == "/data/qc_outputs/testing/"


def test_testing_enabled_adds_missing_windows_separator():
    cfg = {
        "paths": {"output_path": r"C:\qc_outputs"},
        "testing_enabled": "True",
    }
    assert pipeline_output_path_from_config(cfg) == "C:\\qc_outputs\\testing\\"


def test_testing_disabled_no_suffix():
    cfg = {
        "paths": {"output_path": "/data/qc_outputs/"},
        "testing_enabled": "False",
    }
    assert pipeline_output_path_from_config(cfg) == "/data/qc_outputs/"


def test_artifact_dir_under_combined_outputs_analyze_flags(tmp_path):
    base = tmp_path / "qc_root"
    base.mkdir()
    art = analyze_flags_artifact_dir(str(base))
    assert os.path.basename(art) == "analyze_flags"
    assert os.path.basename(os.path.dirname(art)) == "combined_outputs"


def test_ensure_creates_directory(tmp_path):
    base = tmp_path / "qc_root"
    base.mkdir()
    art = ensure_analyze_flags_artifact_dir(str(base))
    assert os.path.isdir(art)


def test_artifact_path_not_cwd_relative(tmp_path):
    base = tmp_path / "qc_root"
    base.mkdir()
    art = analyze_flags_artifact_dir(str(base.resolve()))
    assert os.path.isabs(art)
    assert not art.startswith("." + os.sep)


def test_open_history_csv_basename_matches_network():
    assert open_tracker_row_history_csv_basename("PRESCIENT") == (
        "open_tracker_row_history_PRESCIENT.csv"
    )
    assert open_tracker_row_history_csv_basename("PRONET") == (
        "open_tracker_row_history_PRONET.csv"
    )


def test_artifact_paths_do_not_target_production_parquet_trees(tmp_path):
    """Writes must not live under current/old/new/formatted_outputs trees."""
    base = tmp_path / "qc_root"
    base.mkdir()
    art = analyze_flags_artifact_dir(str(base.resolve()))
    norm = art.replace("\\", "/").lower()
    forbidden = (
        "/combined_outputs/current_output/",
        "/combined_outputs/old_output/",
        "/combined_outputs/new_output/",
        "/formatted_outputs/",
    )
    for frag in forbidden:
        assert frag not in norm + "/", frag


def test_artifact_with_testing_parent_still_under_analyze_flags_only(tmp_path):
    root = tmp_path / "qc_outputs" / "testing"
    root.mkdir(parents=True)
    art = analyze_flags_artifact_dir(str(root))
    assert art.endswith(os.path.join("combined_outputs", "analyze_flags"))
