"""
Tests for qc_types/discovery/internal_consistency_checks.py.

Cases:
  IC-1: BMI matches weight/height² within tolerance → 0 flags
  IC-2: BMI mismatch beyond tolerance → 1 flag with related vars
  IC-3: numeric_range — age within window → 0 flags; out-of-window → 1
  IC-4: sum_equals — total matches sum → 0; off by > tolerance → 1
  IC-5: ratio_range — ratio within range → 0; out → 1
  IC-6: missing inputs at a (subj, tp) silently skip the rule for
        that subject (no flag, no error)
  IC-7: missing-code rows are not counted as inputs (the producer's
        is_missing_code filter is honored)
  Schema: missing required col → RuntimeError
  Rules: unknown rule kind → RuntimeError at load time
  Empty: 0 flags writes valid empty parquet
  File:  end-to-end run() produces a readable parquet, no .tmp residue
"""

import os
import sys
import json
import pytest
import pandas as pd

_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc_types.discovery.internal_consistency_checks import (
    InternalConsistencyChecks)


LONG_FORMAT_COLUMNS = [
    'subjectid', 'network', 'timepoint', 'cohort', 'event_date',
    'source_form', 'variable', 'value', 'value_numeric',
    'is_missing_code',
]

TEST_TP_ORDER = ['screening', 'baseline', 'month1', 'month2']

_RULES_BMI = {
    "_meta": {"doc": "test rules"},
    "rules": [
        {
            "name": "bmi_check",
            "kind": "bmi_formula",
            "bmi_var": "bmi",
            "height_cm_var": "height_cm",
            "weight_kg_var": "weight_kg",
            "rel_tolerance": 0.05,
        }
    ]
}

_RULES_RANGE = {
    "_meta": {"doc": "test rules"},
    "rules": [
        {
            "name": "age_in_range",
            "kind": "numeric_range",
            "variable": "age",
            "min_value": 12,
            "max_value": 35,
        }
    ]
}

_RULES_SUM = {
    "_meta": {"doc": "test rules"},
    "rules": [
        {
            "name": "sum_check",
            "kind": "sum_equals",
            "total_var": "total",
            "component_vars": ["a", "b", "c"],
            "abs_tolerance": 0.5,
        }
    ]
}

_RULES_RATIO = {
    "_meta": {"doc": "test rules"},
    "rules": [
        {
            "name": "ratio_check",
            "kind": "ratio_range",
            "numerator_var": "num",
            "denominator_var": "den",
            "min_ratio": 0.5,
            "max_ratio": 2.0,
        }
    ]
}

_RULES_BAD = {
    "rules": [{"name": "bogus", "kind": "nonsense_kind"}]
}


def _make_long_df(rows):
    full = []
    for r in rows:
        full.append({
            'subjectid': r['subjectid'],
            'network': r.get('network', 'TEST'),
            'timepoint': r['timepoint'],
            'cohort': r.get('cohort', ''),
            'event_date': r.get('event_date', ''),
            'source_form': r.get('source_form', 'form_a'),
            'variable': r['variable'],
            'value': r['value'],
            'value_numeric': r['value_numeric'],
            'is_missing_code': r['is_missing_code'],
        })
    return pd.DataFrame(full, columns=LONG_FORMAT_COLUMNS)


@pytest.fixture
def detector_factory(tmp_path, monkeypatch):
    deps = tmp_path / "dependencies"
    out = tmp_path / "output"
    deps.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)

    class FakeUtils:
        absolute_path = str(tmp_path)

        def create_timepoint_list(self):
            return list(TEST_TP_ORDER)

    import qc_types.discovery.internal_consistency_checks \
        as detector_module
    monkeypatch.setattr(detector_module, 'Utils', FakeUtils)

    def _make(rules_dict, **overrides):
        rules_path = deps / "test_rules.json"
        rules_path.write_text(json.dumps(rules_dict), encoding='utf-8')
        cfg = {
            "paths": {
                "dependencies_path": str(deps) + "/",
                "output_path": str(out) + "/",
            },
            "testing_enabled": "False",
            "discovery": {
                "internal_consistency": {
                    "networks": ["TEST"],
                    "rules_path": str(rules_path),
                    **overrides,
                }
            },
        }
        return InternalConsistencyChecks(config_dict=cfg)

    return _make


# ---------------------------------------------------------------- IC-1

def test_ic1_bmi_consistent_no_flag(detector_factory):
    detector = detector_factory(_RULES_BMI)
    # height=170 cm, weight=70 kg → BMI = 70 / 1.7^2 ≈ 24.22.
    # Declared 24.2 → relative error <0.1%.
    rows = [
        {'subjectid': 'S1', 'timepoint': 'baseline',
         'variable': 'bmi', 'value': '24.2', 'value_numeric': 24.2,
         'is_missing_code': False},
        {'subjectid': 'S1', 'timepoint': 'baseline',
         'variable': 'height_cm', 'value': '170', 'value_numeric': 170.0,
         'is_missing_code': False},
        {'subjectid': 'S1', 'timepoint': 'baseline',
         'variable': 'weight_kg', 'value': '70', 'value_numeric': 70.0,
         'is_missing_code': False},
    ]
    result = detector._evaluate(_make_long_df(rows), detector._load_rules())
    assert len(result) == 0


# ---------------------------------------------------------------- IC-2

def test_ic2_bmi_mismatch_flags(detector_factory):
    detector = detector_factory(_RULES_BMI)
    # Declared BMI=35 but actual ≈ 24.2 → big discrepancy.
    rows = [
        {'subjectid': 'S1', 'timepoint': 'baseline',
         'variable': 'bmi', 'value': '35', 'value_numeric': 35.0,
         'is_missing_code': False},
        {'subjectid': 'S1', 'timepoint': 'baseline',
         'variable': 'height_cm', 'value': '170', 'value_numeric': 170.0,
         'is_missing_code': False},
        {'subjectid': 'S1', 'timepoint': 'baseline',
         'variable': 'weight_kg', 'value': '70', 'value_numeric': 70.0,
         'is_missing_code': False},
    ]
    result = detector._evaluate(_make_long_df(rows), detector._load_rules())
    assert len(result) == 1
    flag = result.iloc[0]
    assert flag['rule_name'] == 'bmi_check'
    assert flag['rule_kind'] == 'bmi_formula'
    assert 'bmi' in flag['related_variables']
    assert 'height_cm' in flag['related_variables']
    assert 'weight_kg' in flag['related_variables']


# ---------------------------------------------------------------- IC-3

def test_ic3_numeric_range(detector_factory):
    detector = detector_factory(_RULES_RANGE)
    rows = [
        {'subjectid': 'S1', 'timepoint': 'baseline',
         'variable': 'age', 'value': '20', 'value_numeric': 20.0,
         'is_missing_code': False},
        {'subjectid': 'S2', 'timepoint': 'baseline',
         'variable': 'age', 'value': '99', 'value_numeric': 99.0,
         'is_missing_code': False},
        {'subjectid': 'S3', 'timepoint': 'baseline',
         'variable': 'age', 'value': '5', 'value_numeric': 5.0,
         'is_missing_code': False},
    ]
    result = detector._evaluate(_make_long_df(rows), detector._load_rules())
    flagged_subs = set(result['subjectid'])
    assert flagged_subs == {'S2', 'S3'}


# ---------------------------------------------------------------- IC-4

def test_ic4_sum_equals(detector_factory):
    detector = detector_factory(_RULES_SUM)
    rows = []
    # Subject S_OK has total = 1+2+3 = 6, matches.
    for v, n in (('total', 6.0), ('a', 1.0), ('b', 2.0), ('c', 3.0)):
        rows.append({
            'subjectid': 'S_OK', 'timepoint': 'baseline',
            'variable': v, 'value': str(n), 'value_numeric': n,
            'is_missing_code': False,
        })
    # Subject S_BAD has total = 10 but sum = 6 → off by 4 > 0.5.
    for v, n in (('total', 10.0), ('a', 1.0), ('b', 2.0), ('c', 3.0)):
        rows.append({
            'subjectid': 'S_BAD', 'timepoint': 'baseline',
            'variable': v, 'value': str(n), 'value_numeric': n,
            'is_missing_code': False,
        })
    result = detector._evaluate(_make_long_df(rows), detector._load_rules())
    assert (result['subjectid'] == 'S_BAD').sum() == 1
    assert (result['subjectid'] == 'S_OK').sum() == 0


# ---------------------------------------------------------------- IC-5

def test_ic5_ratio_range(detector_factory):
    detector = detector_factory(_RULES_RATIO)
    rows = []
    # OK: ratio = 10/8 = 1.25 in [0.5, 2.0]
    rows += [
        {'subjectid': 'S_OK', 'timepoint': 'baseline',
         'variable': 'num', 'value': '10', 'value_numeric': 10.0,
         'is_missing_code': False},
        {'subjectid': 'S_OK', 'timepoint': 'baseline',
         'variable': 'den', 'value': '8', 'value_numeric': 8.0,
         'is_missing_code': False},
    ]
    # BAD: ratio = 50/10 = 5.0 > 2.0
    rows += [
        {'subjectid': 'S_BAD', 'timepoint': 'baseline',
         'variable': 'num', 'value': '50', 'value_numeric': 50.0,
         'is_missing_code': False},
        {'subjectid': 'S_BAD', 'timepoint': 'baseline',
         'variable': 'den', 'value': '10', 'value_numeric': 10.0,
         'is_missing_code': False},
    ]
    result = detector._evaluate(_make_long_df(rows), detector._load_rules())
    flagged_subs = set(result['subjectid'])
    assert flagged_subs == {'S_BAD'}


# ---------------------------------------------------------------- IC-6

def test_ic6_missing_inputs_skip_silently(detector_factory):
    detector = detector_factory(_RULES_BMI)
    # Only weight provided — BMI rule cannot be evaluated for this
    # subject; no flag, no error.
    rows = [
        {'subjectid': 'S_PARTIAL', 'timepoint': 'baseline',
         'variable': 'weight_kg', 'value': '70', 'value_numeric': 70.0,
         'is_missing_code': False},
    ]
    result = detector._evaluate(_make_long_df(rows), detector._load_rules())
    assert len(result) == 0


# ---------------------------------------------------------------- IC-7

def test_ic7_missing_code_rows_excluded(detector_factory):
    detector = detector_factory(_RULES_BMI)
    # Subject has BMI/h/w but BMI row is is_missing_code=True with
    # value_numeric=999 — the producer's missing-code flag must
    # prevent that 999 from feeding the rule.
    rows = [
        {'subjectid': 'S1', 'timepoint': 'baseline',
         'variable': 'bmi', 'value': '999', 'value_numeric': 999.0,
         'is_missing_code': True},
        {'subjectid': 'S1', 'timepoint': 'baseline',
         'variable': 'height_cm', 'value': '170', 'value_numeric': 170.0,
         'is_missing_code': False},
        {'subjectid': 'S1', 'timepoint': 'baseline',
         'variable': 'weight_kg', 'value': '70', 'value_numeric': 70.0,
         'is_missing_code': False},
    ]
    result = detector._evaluate(_make_long_df(rows), detector._load_rules())
    assert len(result) == 0


# ---------------------------------------------------------------- Schema

def test_schema_validation_missing_column_raises(
    detector_factory, tmp_path
):
    detector = detector_factory(_RULES_BMI)
    bogus = pd.DataFrame({
        'subjectid': ['S1'], 'network': ['TEST'],
        'timepoint': ['baseline'], 'variable': ['v'],
        'source_form': [''], 'value': ['50'],
        'value_numeric': [50.0],
    })
    bogus_path = (
        tmp_path / "dependencies"
        / "multi_tp_TEST_numeric_long.parquet")
    bogus.to_parquet(bogus_path, index=False)
    with pytest.raises(RuntimeError) as excinfo:
        detector._load_inputs()
    assert 'is_missing_code' in str(excinfo.value)


# ---------------------------------------------------------------- Rules

def test_unknown_rule_kind_raises(detector_factory):
    detector = detector_factory(_RULES_BAD)
    with pytest.raises(RuntimeError) as excinfo:
        detector._load_rules()
    assert 'nonsense_kind' in str(excinfo.value)


# ---------------------------------------------------------------- Empty

def test_empty_output_schema(detector_factory):
    detector = detector_factory(_RULES_BMI)
    long_df = _make_long_df([])
    result = detector._evaluate(long_df, detector._load_rules())
    assert len(result) == 0
    expected_cols = list(InternalConsistencyChecks.OUTPUT_COLUMNS)
    assert list(result.columns) == expected_cols


# ---------------------------------------------------------------- File

def test_file_output_atomic(detector_factory, tmp_path):
    detector = detector_factory(_RULES_BMI)
    rows = [
        {'subjectid': 'S1', 'timepoint': 'baseline',
         'variable': 'bmi', 'value': '35', 'value_numeric': 35.0,
         'is_missing_code': False},
        {'subjectid': 'S1', 'timepoint': 'baseline',
         'variable': 'height_cm', 'value': '170', 'value_numeric': 170.0,
         'is_missing_code': False},
        {'subjectid': 'S1', 'timepoint': 'baseline',
         'variable': 'weight_kg', 'value': '70', 'value_numeric': 70.0,
         'is_missing_code': False},
    ]
    long_path = (
        tmp_path / "dependencies"
        / "multi_tp_TEST_numeric_long.parquet")
    _make_long_df(rows).to_parquet(long_path, index=False)
    detector.run()
    out_path = (
        tmp_path / "output" / "combined_outputs"
        / "discovery" / "internal_consistency_candidates.parquet")
    assert out_path.exists()
    out_df = pd.read_parquet(out_path)
    assert len(out_df) == 1
    tmps = list((out_path.parent).glob("*.tmp"))
    assert len(tmps) == 0
