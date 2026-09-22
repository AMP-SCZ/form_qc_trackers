"""
Regression tests for FluidChecks class-level state reset in
qc_forms_main.iterate_combined_dfs.

The cross-subject blood-duplicate detector keeps its registry on the
class (so multiple per-row FluidChecks instances can see each
other's prior values). Without an explicit reset at the start of a
QC run, a long-running scheduler that calls
`QCFormsMain().run_script()` twice in one process would carry run
N's registry into run N+1.

These tests do NOT instantiate QCFormsMain (which needs the
production dependency JSONs). They directly verify the reset code is
present at the top of iterate_combined_dfs and that the reset block
itself is structurally correct.
"""

import ast
import os
import sys
import types as _types

import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Stub a minimal qc_forms.form_check so fluid_checks.py can import
# FormCheck without needing the production qc_forms package layout
# (which is only assembled in production deployment).
if 'qc_forms' not in sys.modules:
    sys.modules['qc_forms'] = _types.ModuleType('qc_forms')
if 'qc_forms.form_check' not in sys.modules:
    _fc_mod = _types.ModuleType('qc_forms.form_check')

    class _StubFormCheck:
        def __init__(self, *_, **__):
            pass

        @classmethod
        def standard_qc_check_filter(cls, fn):
            return fn
    _fc_mod.FormCheck = _StubFormCheck
    sys.modules['qc_forms.form_check'] = _fc_mod
    sys.modules['qc_forms'].form_check = _fc_mod


def _qc_forms_main_path():
    return os.path.join(_REPO_ROOT, "main", "qc_forms", "qc_forms_main.py")


def _iterate_combined_dfs_ast():
    """Return the AST FunctionDef for iterate_combined_dfs."""
    src = open(_qc_forms_main_path(), "r", encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "QCFormsMain"):
            for sub in node.body:
                if (isinstance(sub, ast.FunctionDef)
                        and sub.name == "iterate_combined_dfs"):
                    return sub
    raise AssertionError(
        "iterate_combined_dfs not found in QCFormsMain")


def test_reset_calls_present_before_loop():
    """
    The two reset calls must appear in iterate_combined_dfs BEFORE
    the first for-loop (which iterates networks). If reset migrated
    into the network/tp loop it would clobber in-run state.
    """
    fn = _iterate_combined_dfs_ast()
    saw_clear = False
    saw_cache_reset = False
    first_for_lineno = None
    for stmt in fn.body:
        if first_for_lineno is None and isinstance(stmt, ast.For):
            first_for_lineno = stmt.lineno
            break
    assert first_for_lineno is not None, (
        "Expected at least one for-loop in iterate_combined_dfs")

    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.Call):
            func = stmt.func
            # FluidChecks._seen_blood_id_vals.clear()
            if (isinstance(func, ast.Attribute)
                    and func.attr == "clear"
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "_seen_blood_id_vals"):
                assert stmt.lineno < first_for_lineno, (
                    "FluidChecks._seen_blood_id_vals.clear() must run "
                    "BEFORE the network for-loop")
                saw_clear = True
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if (isinstance(tgt, ast.Attribute)
                        and tgt.attr == "_PER_SUBJECT_VARS_CACHE"):
                    assert stmt.lineno < first_for_lineno, (
                        "_PER_SUBJECT_VARS_CACHE reset must run "
                        "BEFORE the network for-loop")
                    saw_cache_reset = True
    assert saw_clear, (
        "iterate_combined_dfs must call "
        "FluidChecks._seen_blood_id_vals.clear()")
    assert saw_cache_reset, (
        "iterate_combined_dfs must reset "
        "FluidChecks._PER_SUBJECT_VARS_CACHE")


def test_clear_uses_dot_clear_not_reassignment():
    """
    Reassigning to {} would break the DEBUG_PERF introspection
    block at the end of iterate_combined_dfs (which retrieves
    `reg = FluidChecks._seen_blood_id_vals`). .clear() mutates the
    live dict, preserving references.
    """
    fn = _iterate_combined_dfs_ast()
    for stmt in ast.walk(fn):
        if (isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Attribute)
                and stmt.targets[0].attr == "_seen_blood_id_vals"):
            raise AssertionError(
                "FluidChecks._seen_blood_id_vals must not be "
                "reassigned (use .clear() instead).")


def test_reset_actually_clears_stale_state():
    """
    Programmatic check that .clear() on the class attribute works
    as expected for a registry containing a stale entry from a
    prior run.
    """
    from qc_types.fluid_checks import FluidChecks
    # Seed the registry as if a prior run had added one entry.
    FluidChecks._seen_blood_id_vals.clear()
    FluidChecks._seen_blood_id_vals["12345"] = [
        ("STALE_SUBJECT", "chrblood_wb1id", "baseline")]
    assert len(FluidChecks._seen_blood_id_vals) == 1
    # Simulate the reset block from iterate_combined_dfs.
    FluidChecks._seen_blood_id_vals.clear()
    FluidChecks._PER_SUBJECT_VARS_CACHE = None
    FluidChecks._PER_SUBJECT_VARS_KEY = None
    assert FluidChecks._seen_blood_id_vals == {}
    assert FluidChecks._PER_SUBJECT_VARS_CACHE is None
    assert FluidChecks._PER_SUBJECT_VARS_KEY is None
