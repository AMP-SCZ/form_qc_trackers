"""Regression coverage for production-style imports in the deployed checkout."""

from pathlib import Path


def test_main_namespace_resolves_flat_utils_package():
    from main.utils.utils import Utils

    expected = Path(__file__).resolve().parents[2] / "main" / "utils" / "utils.py"
    module = __import__(Utils.__module__, fromlist=["__file__"])
    assert Utils.__module__ == "main.utils.utils"
    assert Path(module.__file__).resolve() == expected.resolve()
