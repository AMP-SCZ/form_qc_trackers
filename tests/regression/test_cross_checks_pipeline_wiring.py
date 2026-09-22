"""Regression tests for one-pass CrossChecks pipeline integration."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _tree():
    return ast.parse(
        (ROOT / "main" / "qc_forms" / "qc_forms_main.py").read_text(encoding="utf-8"))


def _iterate_function():
    for node in ast.walk(_tree()):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "iterate_combined_dfs"):
            return node
    raise AssertionError("QCFormsMain.iterate_combined_dfs was not found")


def test_cross_checks_is_imported_from_qc_types():
    imports = [
        node for node in ast.walk(_tree())
        if isinstance(node, ast.ImportFrom)
    ]

    assert any(
        node.module == "qc_forms.qc_types.cross_checks"
        and any(alias.name == "CrossChecks" for alias in node.names)
        for node in imports
    )


def test_cross_checks_runs_once_per_network_timepoint_before_row_loop():
    function = _iterate_function()
    constructor_calls = [
        node for node in ast.walk(function)
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "CrossChecks")
    ]
    assert len(constructor_calls) == 1
    constructor = constructor_calls[0]

    row_loops = [
        node for node in ast.walk(function)
        if (isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "row")
    ]
    assert len(row_loops) == 1
    row_loop = row_loops[0]

    assert constructor.lineno < row_loop.lineno
    assert not (
        row_loop.lineno <= constructor.lineno <= row_loop.end_lineno), (
            "CrossChecks must be constructed outside the per-row loop")

    extensions = [
        node for node in ast.walk(function)
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "extend"
            and node.args
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Name)
            and node.args[0].func.id == "cross_checks")
    ]
    assert len(extensions) == 1
    assert constructor.lineno < extensions[0].lineno < row_loop.lineno


def test_empty_combined_export_fails_before_cross_checks():
    """A header-only expected CSV must not resolve all prior findings."""
    function = _iterate_function()
    constructor = next(
        node for node in ast.walk(function)
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "CrossChecks"))
    empty_guards = [
        node for node in ast.walk(function)
        if (isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute)
            and isinstance(node.test.value, ast.Name)
            and node.test.value.id == "combined_df"
            and node.test.attr == "empty"
            and any(isinstance(child, ast.Raise)
                    for child in ast.walk(node)))
    ]

    assert len(empty_guards) == 1
    assert empty_guards[0].lineno < constructor.lineno


def test_cross_check_metrics_are_optional_for_legacy_deployments():
    """Older CrossChecks versions must not abort an otherwise valid run."""
    function = _iterate_function()
    guarded_reads = [
        node for node in ast.walk(function)
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 3
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "cross_checks"
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "rule_metrics"
            and isinstance(node.args[2], ast.Dict))
    ]

    assert len(guarded_reads) == 1
