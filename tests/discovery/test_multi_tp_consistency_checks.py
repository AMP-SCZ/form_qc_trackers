"""
Tests for qc_types/discovery/multi_tp_consistency_checks.py.

Discovery-tier wrapper around the previously-disabled MultiTPChecks
cross-form/cross-tp checks. Default-off; produces a separate
`multi_tp_consistency_candidates.parquet` under
`combined_outputs/discovery/`. Does NOT append to
`combined_qc_flags.parquet`.
"""

import json
import os
import sys

import pandas as pd
import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc_types.discovery.multi_tp_consistency_checks import (
    MultiTPConsistencyChecks,
    OUTPUT_COLUMNS,
)


@pytest.fixture
def detector_factory(tmp_path, monkeypatch):
    """
    Build a MultiTPConsistencyChecks instance against a tmp deps
    dir. Real Utils is bypassed via monkeypatching the Utils symbol
    inside the detector module's namespace so we never read the
    real project config.json.
    """
    deps = tmp_path / "dependencies"
    out = tmp_path / "output"
    deps.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)

    class FakeUtils:
        absolute_path = str(tmp_path)

    import qc_types.discovery.multi_tp_consistency_checks as mod
    monkeypatch.setattr(mod, 'Utils', FakeUtils)

    def _make(per_subject_vars=None, networks=('TEST',)):
        grouped = {
            'blood_vars': {
                'id_variables': list(per_subject_vars or [
                    'chrblood_wb1id',
                    'chrblood_se1id',
                    'chrblood_pl1id',
                    'chrblood_freezerid',  # excluded by detector
                ]),
                'barcode_variables': list(per_subject_vars or [
                    'chrblood_wb1id_bc',
                    'chrblood_rack_barcode',  # excluded by detector
                ]),
            }
        }
        (deps / "grouped_variables.json").write_text(
            json.dumps(grouped), encoding="utf-8")
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "multi_tp_consistency": {
                    "enabled": True,
                    "networks": list(networks),
                }
            },
        }
        return MultiTPConsistencyChecks(config_dict=cfg)

    return _make


def _write_combined(deps, network, rows):
    """
    Write a multi_tp_{network}_combined.csv whose row dicts come
    from `rows`. Every dict must share the same keys.
    """
    df = pd.DataFrame(rows)
    df.to_csv(
        deps / f"multi_tp_{network}_combined.csv", index=False)


# ------------------------------------------------ blood duplicate within-row

def test_blood_id_within_row_duplicate_flags(detector_factory, tmp_path):
    detector = detector_factory()
    deps = tmp_path / "dependencies"
    _write_combined(deps, 'TEST', [
        {
            'subjectid': 'S001',
            'chrblood_wb1id_baseline': '12345',
            'chrblood_se1id_baseline': '12345',  # same as wb1id
            'chrblood_pl1id_baseline': '67890',
            'chrblood_freezerid_baseline': '99999',
        },
        {
            'subjectid': 'S002',
            'chrblood_wb1id_baseline': '11111',
            'chrblood_se1id_baseline': '22222',
            'chrblood_pl1id_baseline': '33333',
            'chrblood_freezerid_baseline': '99999',
        },
    ])
    result = detector._compute()
    # Only S001 fires (wb1id == se1id). S002 has all distinct.
    assert len(result) == 1
    flag = result.iloc[0]
    assert flag['subjectid'] == 'S001'
    assert flag['algorithm'] == 'within_row_blood_id_duplicate'
    assert flag['value'] == '12345'
    assert (
        'chrblood_wb1id_baseline' in {
            flag['variable'], flag['paired_variable']})
    assert (
        'chrblood_se1id_baseline' in {
            flag['variable'], flag['paired_variable']})


def test_shared_location_vars_excluded(detector_factory, tmp_path):
    """
    Two subjects share the same `chrblood_freezerid` (storage box).
    This is intentional and must NOT fire a within-row duplicate
    flag. Within-row: same subject must have two different per-vial
    vars with the same value. Across-subject: not in scope for this
    check at all. So the only way `chrblood_freezerid` would appear
    as a duplicate is if a single row had two freezer-id columns
    sharing a value — and our detector excludes those vars entirely.
    """
    detector = detector_factory()
    deps = tmp_path / "dependencies"
    _write_combined(deps, 'TEST', [
        {
            'subjectid': 'S001',
            'chrblood_freezerid_baseline': '99999',
            'chrblood_freezerid_month_6': '99999',  # repeated tp
            'chrblood_wb1id_baseline': '12345',
        },
    ])
    result = detector._compute()
    # Zero candidates: the only repeated value is on a shared-
    # location var (excluded) — wb1id is unique.
    assert len(result) == 0


def test_missing_codes_do_not_match(detector_factory, tmp_path):
    detector = detector_factory()
    deps = tmp_path / "dependencies"
    _write_combined(deps, 'TEST', [
        {
            'subjectid': 'S001',
            'chrblood_wb1id_baseline': '-3',     # missing code
            'chrblood_se1id_baseline': '-3',     # missing code
            'chrblood_pl1id_baseline': '',       # blank
            'chrblood_freezerid_baseline': '',
        },
    ])
    result = detector._compute()
    assert len(result) == 0


# ------------------------------------------------ FIGS/PPS age

def test_figs_pps_age_mismatch_mother(detector_factory, tmp_path):
    detector = detector_factory()
    deps = tmp_path / "dependencies"
    _write_combined(deps, 'TEST', [
        {
            'subjectid': 'S001',
            'chrpps_mage_baseline': '45',
            'chrfigs_mother_age_screening': '44',
            'chrpps_fage_baseline': '50',
            'chrfigs_father_age_screening': '50',
        },
    ])
    result = detector._compute()
    assert len(result) == 1
    flag = result.iloc[0]
    assert flag['algorithm'] == 'figs_pps_age_mismatch'
    assert flag['variable'] == 'chrpps_mage_baseline'
    assert flag['paired_variable'] == 'chrfigs_mother_age_screening'
    assert flag['metric_value'] == 1.0


def test_figs_pps_age_match_no_flag(detector_factory, tmp_path):
    detector = detector_factory()
    deps = tmp_path / "dependencies"
    _write_combined(deps, 'TEST', [
        {
            'subjectid': 'S001',
            'chrpps_mage_baseline': '45',
            'chrfigs_mother_age_screening': '45',
            'chrpps_fage_baseline': '50',
            'chrfigs_father_age_screening': '50',
        },
    ])
    result = detector._compute()
    assert len(result) == 0


def test_figs_pps_age_missing_or_nonnumeric_no_flag(
    detector_factory, tmp_path
):
    detector = detector_factory()
    deps = tmp_path / "dependencies"
    _write_combined(deps, 'TEST', [
        {
            'subjectid': 'S001',
            'chrpps_mage_baseline': '999',  # missing code
            'chrfigs_mother_age_screening': '44',
            'chrpps_fage_baseline': '',
            'chrfigs_father_age_screening': '50',
        },
        {
            'subjectid': 'S002',
            'chrpps_mage_baseline': 'unknown',  # non-numeric
            'chrfigs_mother_age_screening': '44',
            'chrpps_fage_baseline': '50',
            'chrfigs_father_age_screening': '50',
        },
    ])
    result = detector._compute()
    assert len(result) == 0


# ---------------------------------- clean row + missing combined CSV

def test_clean_row_zero_candidates(detector_factory, tmp_path):
    detector = detector_factory()
    deps = tmp_path / "dependencies"
    _write_combined(deps, 'TEST', [
        {
            'subjectid': 'S001',
            'chrblood_wb1id_baseline': '11111',
            'chrblood_se1id_baseline': '22222',
            'chrblood_pl1id_baseline': '33333',
            'chrblood_freezerid_baseline': '99999',
            'chrpps_mage_baseline': '45',
            'chrfigs_mother_age_screening': '45',
            'chrpps_fage_baseline': '50',
            'chrfigs_father_age_screening': '50',
        },
    ])
    result = detector._compute()
    assert len(result) == 0


def test_missing_per_network_csv_skipped(detector_factory):
    """
    If a configured network's multi_tp_*_combined.csv is missing,
    the detector should print a WARNING and continue rather than
    abort the discovery run. This matches the discovery-tier
    convention of being usable from incomplete state.
    """
    detector = detector_factory()
    # No CSV written → graceful skip.
    result = detector._compute()
    assert len(result) == 0


# ------------------------------------------------------- output path

def test_output_path_separated_from_main(detector_factory, tmp_path):
    """
    The output parquet must live under combined_outputs/discovery/
    and must NOT touch new_output/old_output/current_output.
    """
    detector = detector_factory()
    deps = tmp_path / "dependencies"
    _write_combined(deps, 'TEST', [
        {
            'subjectid': 'S001',
            'chrblood_wb1id_baseline': '12345',
            'chrblood_se1id_baseline': '12345',
            'chrblood_pl1id_baseline': '67890',
            'chrblood_freezerid_baseline': '99999',
            'chrpps_mage_baseline': '45',
            'chrfigs_mother_age_screening': '45',
            'chrpps_fage_baseline': '50',
            'chrfigs_father_age_screening': '50',
        },
    ])
    detector.run()
    out_dir = (
        tmp_path / "output" / "combined_outputs" / "discovery")
    out_path = (
        out_dir / "multi_tp_consistency_candidates.parquet")
    assert out_path.exists()
    # Atomic write left no .tmp.
    assert list(out_dir.glob("*.tmp")) == []
    # The main-tracker paths were NOT created.
    main_out = (
        tmp_path / "output" / "combined_outputs" / "new_output")
    current_out = (
        tmp_path / "output" / "combined_outputs" / "current_output")
    old_out = (
        tmp_path / "output" / "combined_outputs" / "old_output")
    assert not main_out.exists()
    assert not current_out.exists()
    assert not old_out.exists()


def test_output_schema_columns(detector_factory, tmp_path):
    detector = detector_factory()
    deps = tmp_path / "dependencies"
    _write_combined(deps, 'TEST', [
        {
            'subjectid': 'S001',
            'chrblood_wb1id_baseline': '12345',
            'chrblood_se1id_baseline': '12345',
        },
    ])
    result = detector._compute()
    assert list(result.columns) == OUTPUT_COLUMNS


def test_empty_output_schema(detector_factory, tmp_path):
    detector = detector_factory()
    deps = tmp_path / "dependencies"
    _write_combined(deps, 'TEST', [
        {
            'subjectid': 'S001',
            'chrblood_wb1id_baseline': '11111',
        },
    ])
    result = detector._compute()
    assert len(result) == 0
    assert list(result.columns) == OUTPUT_COLUMNS


# ----------------------------------------------- runner / default-off

def test_runner_default_off(tmp_path, monkeypatch):
    """
    With no `discovery.multi_tp_consistency` key (i.e. enabled
    defaults to False), the runner must exit 0 without
    instantiating the detector.
    """
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "paths": {
            "dependencies_path": str(tmp_path) + "/",
            "output_path": str(tmp_path) + "/",
            "combined_csv_path": str(tmp_path) + "/",
        },
        "testing_enabled": "False",
        # discovery key intentionally absent.
    }))
    import importlib.util
    runner_path = os.path.join(
        _REPO_ROOT, "main", "qc_forms", "run_discovery", "run_multi_tp_consistency.py")
    spec = importlib.util.spec_from_file_location(
        "run_multi_tp_consistency_test", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    # Override the config-path discovery to point at our tmp config.
    monkeypatch.setattr(runner, "_CONFIG_PATH", str(cfg_path))
    rc = runner.main()
    assert rc == 0


def test_runner_enabled_runs_detector(tmp_path, monkeypatch):
    """
    With enabled=True, the runner must invoke MultiTPConsistencyChecks
    and exit 0. We monkeypatch the detector to a no-op stub to avoid
    needing a real grouped_variables.json on disk.
    """
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "paths": {
            "dependencies_path": str(tmp_path) + "/",
            "output_path": str(tmp_path) + "/",
            "combined_csv_path": str(tmp_path) + "/",
        },
        "testing_enabled": "False",
        "discovery": {
            "multi_tp_consistency": {
                "enabled": True,
                "networks": ["TEST"],
            }
        }
    }))
    import importlib.util
    runner_path = os.path.join(
        _REPO_ROOT, "main", "qc_forms", "run_discovery", "run_multi_tp_consistency.py")
    spec = importlib.util.spec_from_file_location(
        "run_multi_tp_consistency_test2", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "_CONFIG_PATH", str(cfg_path))

    invocations = []

    class StubDetector:
        def __init__(self, config_dict=None):
            invocations.append(('init', config_dict))

        def run(self):
            invocations.append(('run',))
    monkeypatch.setattr(
        runner, "MultiTPConsistencyChecks", StubDetector)
    rc = runner.main()
    assert rc == 0
    assert ('run',) in invocations
