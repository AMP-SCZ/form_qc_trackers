"""Shared harness for synthetic integration tests of the qc_forms pipeline.

This sliced-out folder is normally not runnable: production modules import
via `from qc_forms.X import ...` / `from main.X import ...`, which only
resolve inside the full repo. This harness installs lightweight package
ALIASES so `qc_forms.*` and `main.*` resolve to the real modules at the
repo root, letting the synthetic tests exercise the REAL integrated code
(not stubs). It also seeds Utils' config cache so these synthetic tests never
depend on a developer's local configuration.

No PHI: tests build their own fake DataFrames / dependency dicts. The real
*schema* dependency JSONs in dependencies/ (variable/form definitions) are
non-PHI and may be reused; subject data is always synthetic.
"""
import os
import sys
import types


def project_root():
    # tests/synthetic/_qc_harness.py -> repo root is two dirs up.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def install_shim():
    """Make `qc_forms.*` and `main.*` import the real repo-root modules."""
    root = project_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    package_paths = {
        "qc_forms": [os.path.join(root, "main", "qc_forms")],
        "main": [os.path.join(root, "main"), os.path.join(root, "main", "qc_forms")],
    }
    for path in package_paths["main"]:
        if path not in sys.path:
            sys.path.append(path)
    for alias, paths in package_paths.items():
        mod = sys.modules.get(alias)
        if mod is None:
            mod = types.ModuleType(alias)
            sys.modules[alias] = mod
        # Some earlier test modules install a plain ModuleType stub under the
        # same name. Merely finding the alias in sys.modules is not enough:
        # without __path__ Python correctly reports that qc_forms is not a
        # package. Upgrade the existing object in place so references held by
        # those already-collected tests remain valid.
        mod.__path__ = paths
        mod.__package__ = alias

    # Stub-based unit tests can also leave a synthetic qc_forms.form_check in
    # sys.modules. It has no module origin (and no real config cache), so evict
    # only that incompatible child and let the next normal import load the
    # repository's form_check.py through the package path above.
    qualified_name = "qc_forms.form_check"
    form_check_module = sys.modules.get(qualified_name)
    expected_origin = os.path.normcase(os.path.realpath(
        os.path.join(root, "main", "qc_forms", "form_check.py")))
    actual_origin = getattr(form_check_module, "__file__", "")
    actual_origin = (
        os.path.normcase(os.path.realpath(actual_origin))
        if actual_origin else "")
    if form_check_module is not None and actual_origin != expected_origin:
        del sys.modules[qualified_name]
        package = sys.modules["qc_forms"]
        if getattr(package, "form_check", None) is form_check_module:
            delattr(package, "form_check")
    return root


def seed_config(dependencies_path=None, combined_csv_path=None, output_path=None,
                testing_enabled="False"):
    """Seed Utils' config cache so Utils() constructs without reading a real
    config.json. Defaults dependencies_path to the real (non-PHI schema) dir."""
    root = install_shim()
    from utils import utils as U
    cfg = {
        "paths": {
            "dependencies_path": (dependencies_path or (os.path.join(os.path.dirname(root), "dependencies") + os.sep)),
            "combined_csv_path": (combined_csv_path or (os.path.join(os.path.dirname(root), "dependencies") + os.sep)),
            "output_path": (output_path or (os.path.join(os.path.dirname(root), "dependencies") + os.sep)),
        },
        "testing_enabled": testing_enabled,
        "pipeline_networks": ["PRONET", "PRESCIENT"],
        "withdrawn_enabled": False,
        "recruited_only": False,
    }
    U._CONFIG_CACHE[U._project_root()] = cfg
    return cfg
