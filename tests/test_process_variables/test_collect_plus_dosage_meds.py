"""
Synthetic tests for PlusDosageMedCollector (no real combined CSVs or
data dictionary needed).

Uses the same stub layout as tests/clinical/test_pharm_checks_med_rules.py
so the module imports without the full pipeline tree.
"""

import os
import sys
import types

import pandas as pd

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class _StubUtils:
    pass


if "utils" not in sys.modules:
    sys.modules["utils"] = types.ModuleType("utils")
if "utils.utils" not in sys.modules:
    _utils_mod = types.ModuleType("utils.utils")
    _utils_mod.Utils = _StubUtils
    sys.modules["utils.utils"] = _utils_mod

from process_variables.collect_plus_dosage_meds import (  # noqa: E402
    PlusDosageMedCollector,
)

MISSING_CODE_SET = frozenset(
    {
        "-3", "-9", "-3.0", "-9.0", "-99", "-99.0",
        "999", "999.0", "1909-09-09", "1903-03-03",
    }
)


class FakeUtils:
    missing_code_set = MISSING_CODE_SET
    pipeline_networks = ("PRONET", "PRESCIENT")

    def __init__(self, timepoints=None):
        self.timepoints = timepoints or []

    def create_timepoint_list(self):
        return list(self.timepoints)


def make_collector(med_code_translations=None):
    collector = PlusDosageMedCollector.__new__(PlusDosageMedCollector)
    collector.utils = FakeUtils()
    collector.med_code_translations = med_code_translations or {}
    collector.plus_dosage_meds = {}
    collector.n_csvs_read = 0
    collector.n_csvs_with_pharm_cols = 0
    collector.n_rows_scanned = 0
    collector.n_plus_dosages = 0
    return collector


def test_variable_format_regexes():
    name_re = PlusDosageMedCollector.name_var_format
    dosage_re = PlusDosageMedCollector.dosage_var_format

    assert name_re.match("chrpharm_med1_name")
    assert name_re.match("chrpharm_med12_name_past")
    assert not name_re.match("chrpharm_med1_name_extra")
    assert not name_re.match("chrpharm_firstdose_med1")

    assert dosage_re.match("chrpharm_med1_dosage")
    assert dosage_re.match("chrpharm_med1_dosage_past")
    assert dosage_re.match("chrpharm_med10_dosage_2")
    assert dosage_re.match("chrpharm_med10_dosage_2_past")
    assert not dosage_re.match("chrpharm_med1_dosagex")
    assert not dosage_re.match("chrpharm_med1_dose")

    match = dosage_re.match("chrpharm_med10_dosage_2_past")
    assert match.group(1) == "10"
    assert match.group(2) == "_past"


def test_collect_med_code_translations():
    data_dict_df = pd.DataFrame(
        {
            "Variable / Field Name": [
                "chrpharm_med1_name",
                "chrpharm_med1_name_past",
                "chrpharm_med1_dosage",
                "chrdemo_age_mos2",
            ],
            "Choices, Calculations, OR Slider Labels": [
                "146, Amitriptyline + Perphenazine | 300, Risperidone"
                " | 555, Label, with comma",
                "146, Duplicate Name Ignored | 280, Combo Med",
                "1, Should Not Be Collected",
                "",
            ],
            "Form Name": ["a", "b", "a", "c"],
        }
    )
    collector = make_collector()
    translations = collector.collect_med_code_translations(data_dict_df)

    assert translations == {
        "146": "Amitriptyline + Perphenazine",
        "300": "Risperidone",
        "555": "Label, with comma",
        "280": "Combo Med",
    }


def test_collect_csv_plus_dosages_records_plus_courses():
    collector = make_collector(
        {"146": "Amitriptyline + Perphenazine", "300": "Risperidone"}
    )
    combined_df = pd.DataFrame(
        {
            "chrpharm_med1_name": ["146", "146.0"],
            "chrpharm_med1_dosage": ["25+4", "25 + 4"],
            "chrpharm_med2_name_past": ["300", "555"],
            "chrpharm_med2_dosage_past": ["8", "5+5"],
            "chrpharm_med3_name": ["", "300"],
            "chrpharm_med3_dosage_2": ["10+10", "160"],
        }
    )
    collector.collect_csv_plus_dosages(combined_df)
    output = collector.format_output()

    assert sorted(output) == [
        "Amitriptyline + Perphenazine",
        "no medication name recorded",
        "unrecognized medication code (555)",
    ]

    combo_entry = output["Amitriptyline + Perphenazine"]
    assert combo_entry["med_codes"] == ["146"]
    assert combo_entry["dosage_vars"] == ["chrpharm_med1_dosage"]
    assert combo_entry["dosage_vals"] == ["25 + 4", "25+4"]
    assert combo_entry["n_observations"] == 2

    unknown_entry = output["unrecognized medication code (555)"]
    assert unknown_entry["med_codes"] == ["555"]
    assert unknown_entry["dosage_vars"] == ["chrpharm_med2_dosage_past"]
    assert unknown_entry["dosage_vals"] == ["5+5"]
    assert unknown_entry["n_observations"] == 1

    no_name_entry = output["no medication name recorded"]
    assert no_name_entry["med_codes"] == []
    assert no_name_entry["dosage_vars"] == ["chrpharm_med3_dosage_2"]
    assert no_name_entry["dosage_vals"] == ["10+10"]
    assert no_name_entry["n_observations"] == 1

    assert collector.n_csvs_with_pharm_cols == 1
    assert collector.n_rows_scanned == 2
    assert collector.n_plus_dosages == 4


def test_sentinel_and_missing_code_names_not_treated_as_medications():
    # 999/888 are in the dropdown choices, so they translate -- but they
    # are sentinels, not medications, and must land in explicit buckets.
    collector = make_collector(
        {"999": "None", "888": "No information", "777": "Unknown"}
    )
    combined_df = pd.DataFrame(
        {
            "chrpharm_med1_name": ["999", "888", "777", "-9", "555"],
            "chrpharm_med1_dosage": ["1+1", "2+2", "3+3", "4+4", "5+5"],
        }
    )
    collector.collect_csv_plus_dosages(combined_df)
    output = collector.format_output()

    assert sorted(output) == [
        "medication name recorded as missing code (-9)",
        "medication unknown (777)",
        "no information (888)",
        "no medication recorded (999)",
        "unrecognized medication code (555)",
    ]


def test_non_plus_dosages_ignored():
    collector = make_collector({"300": "Risperidone"})
    combined_df = pd.DataFrame(
        {
            "chrpharm_med1_name": ["300", "300", "300", "300"],
            "chrpharm_med1_dosage": ["", "-3", "20", "1909-09-09"],
        }
    )
    collector.collect_csv_plus_dosages(combined_df)

    assert collector.format_output() == {}


def test_dosage_without_name_column_still_recorded():
    collector = make_collector()
    combined_df = pd.DataFrame({"chrpharm_med4_dosage": ["1+1"]})
    collector.collect_csv_plus_dosages(combined_df)
    output = collector.format_output()

    assert sorted(output) == ["no medication name recorded"]
    assert output["no medication name recorded"]["n_observations"] == 1


def test_normalize_med_code():
    collector = make_collector()

    assert collector.normalize_med_code("146.0") == "146"
    assert collector.normalize_med_code("146") == "146"
    assert collector.normalize_med_code("") == ""
    assert collector.normalize_med_code("25.5") == "25.5"
    assert collector.normalize_med_code("abc.0") == "abc.0"


def test_sweep_reads_csv_preserving_literal_plus(tmp_path):
    # Goes through the real pd.read_csv path: pins the dtype=str
    # behavior (the default parser would coerce '+10' to the number 10
    # and drop the '+'), the filename template, the usecols filtering
    # of non-pharm columns, and that missing CSVs are skipped without
    # raising.
    collector = make_collector({"146": "Amitriptyline + Perphenazine"})
    collector.utils = FakeUtils(["screening"])
    collector.comb_csv_path = f"{tmp_path}/"
    csv_path = tmp_path / "AMPSCZ-combined-redcap_screening_ProNET-day1to1.csv"
    pd.DataFrame(
        {
            "subjectid": ["TEST0001"],
            "chrdemo_age_mos2": ["240"],
            "chrpharm_med1_name": ["146"],
            "chrpharm_med1_dosage": ["+10"],
        }
    ).to_csv(csv_path, index=False)

    collector.collect_plus_dosage_meds()
    output = collector.format_output()

    assert sorted(output) == ["Amitriptyline + Perphenazine"]
    assert output["Amitriptyline + Perphenazine"]["dosage_vals"] == ["+10"]
    assert collector.n_csvs_read == 1
    assert collector.n_csvs_with_pharm_cols == 1
    assert collector.n_rows_scanned == 1
    assert collector.n_plus_dosages == 1
