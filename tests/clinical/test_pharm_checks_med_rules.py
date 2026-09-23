"""
Additional synthetic medication QC scenarios (see also test_pharm_checks_acute_injection.py).

Uses the same stub layout as test_pharm_checks_acute_injection.py (no full qc_forms tree).
"""

import os
import sys
import types

import pandas as pd
import pytest

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class _StubFormCheck:
    pass


class _StubUtils:
    pass


if "qc_forms" not in sys.modules:
    sys.modules["qc_forms"] = types.ModuleType("qc_forms")
if "qc_forms.form_check" not in sys.modules:
    _form_check_mod = types.ModuleType("qc_forms.form_check")
    _form_check_mod.FormCheck = _StubFormCheck
    sys.modules["qc_forms.form_check"] = _form_check_mod
if "utils" not in sys.modules:
    sys.modules["utils"] = types.ModuleType("utils")
if "utils.utils" not in sys.modules:
    _utils_mod = types.ModuleType("utils.utils")
    _utils_mod.Utils = _StubUtils
    sys.modules["utils.utils"] = _utils_mod

from qc_types.clinical_checks.pharm_checks import PharmChecks  # noqa: E402


class Row:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeUtils:
    def __init__(self):
        self.missing_code_list = [
            "-3", "-9", -3, -9, -3.0, -9.0, "-3.0", "-9.0",
            "1909-09-09", "1903-03-03", "1901-01-01", "-99", -99, -99.0,
            "-99.0", 999, 999.0, "999", "999.0",
        ]

    def can_be_float(self, val):
        try:
            float(val)
            return True
        except (TypeError, ValueError):
            return False

    def check_if_val_date_format(self, string, date_format="%Y-%m-%d"):
        try:
            from datetime import datetime

            datetime.strptime(string, date_format)
            return True
        except (TypeError, ValueError):
            return False

    def all_dtype(self, inp_list):
        out = []
        for n in inp_list:
            out.append(n)
            out.append(str(n))
            out.append(float(n))
            out.append(str(float(n)))
        return out


def _capture_create_row_output(row, forms, vars_list, message, reports):
    return {
        "subject": getattr(row, "subjectid", ""),
        "forms": list(forms),
        "affected_variables": list(vars_list),
        "error_message": message,
        "reports": reports,
    }


def make_pharm_checks():
    p = PharmChecks.__new__(PharmChecks)
    p.utils = FakeUtils()
    p.missing_code_list = [
        "-3", "-9", -3, -9, -3.0, -9.0, "-3.0", "-9.0",
        "1909-09-09", "1903-03-03", "-99", -99, -99.0, "-99.0",
        999, 999.0, "999", "999.0",
    ]
    p.ONGOING_OFFSET_CODES = set(PharmChecks.ONGOING_OFFSET_CODES)
    p.final_output_list = []
    p.create_row_output = _capture_create_row_output
    # Network defaults to PRONET with deployed Main / Non Team Forms routing.
    # PRESCIENT retains its existing Secondary Report routing.
    p.network = "PRONET"
    p.timepoint = "baseline"
    # Bypass the past-form completion gate in `_append_qc` so tests
    # exercise the per-check flag logic in isolation.
    p._past_pharm_marked_complete = lambda row: True
    p.important_form_vars = {
        "past_pharmaceutical_treatment": {
            "completion_var": "past_pharmaceutical_treatment_complete",
        }
    }
    return p


CURR = [
    "current_pharmaceutical_treatment_floating_med_125",
    "current_pharmaceutical_treatment_floating_med_2650",
]
PAST = ["past_pharmaceutical_treatment"]


def test_no_medication_strict_overlap_with_real_med_flags():
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_med1_name=300,
        chrpharm_med1_onset="2022-02-01",
        chrpharm_med1_offset="2022-02-05",
        chrpharm_med2_name=999,
        chrpharm_med2_onset="2022-02-02",
        chrpharm_med2_offset="2022-02-04",
    )
    p.check_no_medication_overlap(row, past=False, forms=CURR)
    assert len(p.final_output_list) == 1


def test_no_medication_touching_boundary_no_flag():
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_med1_name=300,
        chrpharm_med1_onset="2022-02-01",
        chrpharm_med1_offset="2022-02-05",
        chrpharm_med2_name=999,
        chrpharm_med2_onset="2022-02-05",
        chrpharm_med2_offset="2022-02-25",
    )
    p.check_no_medication_overlap(row, past=False, forms=CURR)
    assert p.final_output_list == []


def test_no_medication_missing_offset_later_real_med_flags():
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_med1_name=999,
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="1901-01-01",
        chrpharm_med2_name=300,
        chrpharm_med2_onset="2026-02-01",
        chrpharm_med2_offset="2026-03-01",
    )
    p.check_no_medication_missing_offset(row, past=False, forms=CURR)
    assert len(p.final_output_list) == 1


def test_overlap_skips_non999_missing_name():
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_med1_name=-9,
        chrpharm_med1_onset="2022-02-01",
        chrpharm_med1_offset="2022-02-05",
        chrpharm_med2_name=999,
        chrpharm_med2_onset="2022-02-02",
        chrpharm_med2_offset="2022-02-04",
    )
    p.check_no_medication_overlap(row, past=False, forms=CURR)
    assert p.final_output_list == []


def test_onset_after_mod_date_flags():
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_date_mod="2026-01-10",
        chrpharm_med1_onset="2026-01-15",
        chrpharm_med1_offset="2026-01-20",
    )
    p.check_current_med_dates_not_after_mod_date(row, forms=CURR)
    assert len(p.final_output_list) == 2
    assert all(
        flag["reports"]["reports"] == ["Main Report", "Non Team Forms"]
        for flag in p.final_output_list)
    assert all(
        flag["reports"]["check_id"] == "MED-QC-03"
        for flag in p.final_output_list)


def test_onset_after_mod_date_prescient_routing_is_unchanged():
    p = make_pharm_checks()
    p.network = "PRESCIENT"
    row = Row(
        subjectid="S1",
        chrpharm_date_mod="2026-01-10",
        chrpharm_med1_onset="2026-01-15",
    )

    p.check_current_med_dates_not_after_mod_date(row, forms=CURR)

    assert len(p.final_output_list) == 1
    assert p.final_output_list[0]["reports"]["reports"] == ["Secondary Report"]
    assert p.final_output_list[0]["reports"]["check_id"] == "MED-QC-03"


def test_firstdose_skips_when_missing_onset():
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_med1_onset=-9,
        chrpharm_firstdose_med1="2026-01-01",
    )
    p.pharm_firstdose_check(row, CURR)
    assert p.final_output_list == []


def test_normalize_date_str_timestamp_placeholder_invalid():
    p = make_pharm_checks()
    assert p._normalize_date_str(pd.Timestamp("1903-03-03")) is None


def test_med_qc_01_uses_legacy_latest_assessment_range():
    """A populated top-level latest range must produce the MED-QC-01 flag.

    This locks the consumer side of the uppercase cohort-schedule regression:
    the collector test verifies that production-style ``CHR``/``HC`` keys
    populate this range, and this test verifies that PharmChecks turns the
    resulting stale modification date into the expected output row.
    """
    p = make_pharm_checks()
    p.tp_date_ranges = {
        "S1": {
            "screening": {
                "latest": "2026-03-15",
                "latest_form": "scheduled_form",
            }
        }
    }
    p.important_form_vars["scheduled_form"] = {
        "interview_date_var": "scheduled_date",
    }
    row = Row(
        subjectid="S1",
        chrpharm_date_mod="2026-03-01",
        scheduled_date="2026-03-15",
    )

    p.check_pharm_interview_dates(row, CURR, {})

    assert len(p.final_output_list) == 1
    flag = p.final_output_list[0]
    assert flag["affected_variables"] == [
        "scheduled_date", "chrpharm_date_mod"]
    assert "Please update the pharmaceutical treatment modification date" in (
        flag["error_message"])


def test_acute_injection_still_uses_code_set_not_keywords():
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_med1_name="acute injection ketamine",
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="2026-02-01",
    )
    p.check_acute_injection_longer_than_two_weeks(row, past=False, forms=CURR)
    assert p.final_output_list == []


# --- Sentinel constant + ongoing-offset shape locks ---


def test_ongoing_offset_codes_constant_value():
    """Lock ONGOING_OFFSET_CODES against silent re-broadening. Per
    dependencies/med_info.txt:33, only 1901-01-01 is the ongoing sentinel;
    1909-09-09 and 1903-03-03 are missing/placeholder dates, not ongoing."""
    assert PharmChecks.ONGOING_OFFSET_CODES == frozenset({"1901-01-01"})


def test_past_form_offset_1901_timestamp_flags():
    """MED-QC-08: Timestamp-shaped 1901-01-01 offset still triggers the
    past-form ongoing-medication flag. Pins _is_ongoing_offset_code's
    raw_date_str path so a Timestamp doesn't slip past it."""
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_med1_name_past="300",
        chrpharm_med1_offset_past=pd.Timestamp("1901-01-01"),
    )
    p.check_no_ongoing_med_in_past_form(row, forms=PAST)
    assert len(p.final_output_list) == 1
    assert "chrpharm_med1_offset_past" in p.final_output_list[0]["affected_variables"]


def test_1909_09_09_string_not_treated_as_ongoing():
    """1909-09-09 is missing/placeholder per spec — must NOT be ongoing."""
    p = make_pharm_checks()
    assert p._is_ongoing_offset_code("1909-09-09") is False


def test_1909_09_09_timestamp_not_treated_as_ongoing():
    """Same as above, Timestamp-shaped — covers the raw_date_str split path."""
    p = make_pharm_checks()
    assert p._is_ongoing_offset_code(pd.Timestamp("1909-09-09")) is False


def test_past_form_offset_1909_09_09_no_flag():
    """MED-QC-08 must not fire for 1909-09-09 — that's missing, not ongoing."""
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_med1_name_past="300",
        chrpharm_med1_offset_past="1909-09-09",
    )
    p.check_no_ongoing_med_in_past_form(row, forms=PAST)
    assert p.final_output_list == []


# --- MED-QC-13: blinded medication codes 573 / 542 / 538 / 539 ---


@pytest.mark.parametrize("blinded_code", [573, 542, 538, 539, "573", "573.0"])
def test_blinded_medication_name_flags(blinded_code):
    """MED-QC-13: any blinded medication code on the name field should flag."""
    p = make_pharm_checks()
    row = Row(subjectid="S1", chrpharm_med1_name=blinded_code)
    p.det_med_name_improper_value(row, past=False, forms=CURR)
    assert len(p.final_output_list) == 1
    assert "chrpharm_med1_name" in p.final_output_list[0]["affected_variables"]


def test_blinded_medication_past_form_flags():
    """MED-QC-13: blinded codes on past-form names must also flag."""
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_med1_name_past=542,
        past_pharmaceutical_treatment_complete=2,
    )
    p.det_med_name_improper_value(row, past=True, forms=PAST)
    assert len(p.final_output_list) == 1


# --- MED-QC-11: 888 (no info) + course info ---


def test_med_name_888_with_info_flags():
    """MED-QC-11: name=888 with any non-missing course-level field should flag."""
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_med1_name=888,
        chrpharm_med1_dosage="2",
    )
    p.med_name_missing_check(row, past=False, forms=CURR)
    assert len(p.final_output_list) == 1
    flag = p.final_output_list[0]
    assert "chrpharm_med1_name" in flag["affected_variables"]
    assert "chrpharm_med1_dosage" in flag["affected_variables"]


def test_med_name_888_alone_no_flag():
    """MED-QC-11 must not fire when 888 sits with no follow-up info."""
    p = make_pharm_checks()
    row = Row(subjectid="S1", chrpharm_med1_name=888)
    p.med_name_missing_check(row, past=False, forms=CURR)
    assert p.final_output_list == []


# --- MED-QC-09: intermittent flag = 2 but offset is a real date ---


def test_current_med_intermittent_with_real_offset_flags():
    """MED-QC-09: chrpharm_interm_meds = 2 (course not ongoing) but
    chrpharm_med1_offset has a real date — must flag (offset-by-one mapping
    means chrpharm_interm_meds controls med1_offset)."""
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_interm_meds="2",
        chrpharm_med1_offset="2026-02-01",
    )
    p.check_current_med_not_ongoing_but_has_offset(row, forms=CURR)
    assert len(p.final_output_list) == 1
    flag = p.final_output_list[0]
    assert "chrpharm_interm_meds" in flag["affected_variables"]
    assert "chrpharm_med1_offset" in flag["affected_variables"]


def test_current_med_intermittent_with_ongoing_offset_no_flag():
    """MED-QC-09: when offset is the ongoing sentinel 1901-01-01, a
    chrpharm_interm_meds = 2 marker is internally consistent and must NOT flag."""
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_interm_meds="2",
        chrpharm_med1_offset="1901-01-01",
    )
    p.check_current_med_not_ongoing_but_has_offset(row, forms=CURR)
    assert p.final_output_list == []


def test_current_med_intermittent_med2_uses_interm_meds_1():
    """MED-QC-09 offset-by-one: chrpharm_interm_meds_1 controls
    chrpharm_med2_offset (not chrpharm_med2_offset paired with
    chrpharm_interm_meds_2)."""
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_interm_meds_1="2",
        chrpharm_med2_offset="2026-02-01",
    )
    p.check_current_med_not_ongoing_but_has_offset(row, forms=CURR)
    assert len(p.final_output_list) == 1
    flag = p.final_output_list[0]
    assert "chrpharm_interm_meds_1" in flag["affected_variables"]
    assert "chrpharm_med2_offset" in flag["affected_variables"]


# --- MED-QC-14: compliance boundary values ---


@pytest.mark.parametrize("comp_val", ["0", "100", "50"])
def test_compliance_in_range_no_flag(comp_val):
    """MED-QC-14: 0 and 100 are inclusive boundaries and must not flag."""
    p = make_pharm_checks()
    row = Row(subjectid="S1", chrpharm_med1_comp=comp_val)
    p.check_compliance_range(row, past=False, forms=CURR)
    assert p.final_output_list == []


# --- MED-QC-17: combination medication dose format ---


def _combo_pharm_checks(codes=None):
    """Helper: returns PharmChecks with the production COMBINATION_MED_CODES
    by default. Override `codes` to test custom membership / empty-set logic."""
    p = make_pharm_checks()
    if codes is not None:
        p.COMBINATION_MED_CODES = frozenset(codes)
    return p


def test_combo_med_empty_codes_no_flag():
    """Empty COMBINATION_MED_CODES must short-circuit to no flag."""
    p = _combo_pharm_checks(codes=())
    row = Row(subjectid="S1", chrpharm_med1_name="146", chrpharm_med1_dosage="10")
    p.check_combination_med_dose_format(row, past=False, forms=CURR)
    assert p.final_output_list == []


def test_combo_med_dose_with_plus_no_flag():
    """'25+4' is the canonical combination-med format and must not flag."""
    p = _combo_pharm_checks()
    row = Row(subjectid="S1", chrpharm_med1_name="146", chrpharm_med1_dosage="25+4")
    p.check_combination_med_dose_format(row, past=False, forms=CURR)
    assert p.final_output_list == []


def test_combo_med_dose_decimal_plus_no_flag():
    """Decimals on either side of '+' should also satisfy the format."""
    p = _combo_pharm_checks()
    row = Row(subjectid="S1", chrpharm_med1_name="146", chrpharm_med1_dosage="2.5+0.5")
    p.check_combination_med_dose_format(row, past=False, forms=CURR)
    assert p.final_output_list == []


@pytest.mark.parametrize("bad_dose", ["10", "10+", "+10", "abc+def", "10+abc"])
def test_combo_med_dose_missing_plus_flags(bad_dose):
    """Without a '<num>+<num>' substring the combination-dose rule must flag."""
    p = _combo_pharm_checks()
    row = Row(subjectid="S1", chrpharm_med1_name="146", chrpharm_med1_dosage=bad_dose)
    p.check_combination_med_dose_format(row, past=False, forms=CURR)
    assert len(p.final_output_list) == 1
    flag = p.final_output_list[0]
    assert "chrpharm_med1_name" in flag["affected_variables"]
    assert "chrpharm_med1_dosage" in flag["affected_variables"]


def test_combo_med_missing_dosage_no_flag():
    """Missing dosage is a different rule's concern — combo-format must skip."""
    p = _combo_pharm_checks()
    row = Row(subjectid="S1", chrpharm_med1_name="146", chrpharm_med1_dosage=-9)
    p.check_combination_med_dose_format(row, past=False, forms=CURR)
    assert p.final_output_list == []


def test_combo_med_non_combo_code_skipped():
    """A non-combination med code (e.g. 300/Risperidone) must not trigger the rule."""
    p = _combo_pharm_checks()
    row = Row(subjectid="S1", chrpharm_med1_name="300", chrpharm_med1_dosage="10")
    p.check_combination_med_dose_format(row, past=False, forms=CURR)
    assert p.final_output_list == []


def test_combo_med_past_form_flags():
    """Combination-dose rule must also fire on past_pharmaceutical_treatment."""
    p = _combo_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_med1_name_past="146",
        chrpharm_med1_dosage_past="25",
        past_pharmaceutical_treatment_complete=2,
    )
    p.check_combination_med_dose_format(row, past=True, forms=PAST)
    assert len(p.final_output_list) == 1


@pytest.mark.parametrize("combo_code", [110, 146, 280, "110", "146", "280", 110.0])
def test_combo_med_production_codes_flag(combo_code):
    """All three production combination med codes (110, 146, 280) must fire
    the rule, in int / string / float forms (REDCap shape varies)."""
    p = _combo_pharm_checks()
    row = Row(subjectid="S1", chrpharm_med1_name=combo_code, chrpharm_med1_dosage="25")
    p.check_combination_med_dose_format(row, past=False, forms=CURR)
    assert len(p.final_output_list) == 1


def test_combination_med_codes_production_value():
    """Lock the production set against silent edits."""
    assert PharmChecks.COMBINATION_MED_CODES == frozenset({110, 146, 280})


# --- MED-QC-18: medication daily/total dose cutoffs ---


@pytest.mark.parametrize(
    "code,cutoff",
    [
        ("300", 8),     # Risperidone
        ("305", 30),    # Aripiprazole
        ("201", 20),    # Olanzapine
        ("238", 1200),  # Amisulpride
        ("235", 800),   # Quetiapine
        ("530", 20),    # Asenapine
        ("574", 4),     # Brexpiprazole
        ("721", 6),     # Cariprazine
        ("256", 500),   # Chlorpromazine
        ("69", 900),    # Clozapine
        ("122", 20),    # Haloperidol
        ("728", 42),    # Lumateperone
        ("540", 160),   # Lurasidone
        ("531", 6),     # Paliperidone
        ("75", 150),    # Prochlorperazine
        ("242", 1000),  # Promazine
        ("304", 160),   # Ziprasidone
        ("452", 120),   # Risperidone SC LAI
        ("547", 400),   # Aripiprazole LAI
    ],
)
def test_dose_above_cutoff_flags(code, cutoff):
    """A dose strictly above each cutoff must flag."""
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_med1_name=code,
        chrpharm_med1_dosage=cutoff + 0.5,
    )
    p.check_med_dose_cutoffs(row, past=False, forms=CURR)
    assert len(p.final_output_list) == 1
    flag = p.final_output_list[0]
    assert "chrpharm_med1_dosage" in flag["affected_variables"]


@pytest.mark.parametrize("code,cutoff", [("300", 8), ("69", 900), ("452", 120)])
def test_dose_at_cutoff_no_flag(code, cutoff):
    """Spec says 'higher than' — dose exactly at the cutoff must not flag."""
    p = make_pharm_checks()
    row = Row(subjectid="S1", chrpharm_med1_name=code, chrpharm_med1_dosage=cutoff)
    p.check_med_dose_cutoffs(row, past=False, forms=CURR)
    assert p.final_output_list == []


def test_dose_cutoff_with_combo_string_dose_no_crash():
    """A non-numeric '10+5' dose must be skipped without crashing — combo format
    is handled by check_combination_med_dose_format, not this rule."""
    p = make_pharm_checks()
    row = Row(subjectid="S1", chrpharm_med1_name="300", chrpharm_med1_dosage="10+5")
    p.check_med_dose_cutoffs(row, past=False, forms=CURR)
    assert p.final_output_list == []


def test_dose_cutoff_uncovered_med_code_no_flag():
    """Med codes outside DOSE_CUTOFFS_MG (e.g. 888 'no info') must not flag."""
    p = make_pharm_checks()
    row = Row(subjectid="S1", chrpharm_med1_name="888", chrpharm_med1_dosage=10000)
    p.check_med_dose_cutoffs(row, past=False, forms=CURR)
    assert p.final_output_list == []


def test_dose_cutoff_missing_dosage_no_flag():
    """Missing/sentinel dosage values must be skipped silently."""
    p = make_pharm_checks()
    row = Row(subjectid="S1", chrpharm_med1_name="300", chrpharm_med1_dosage=-9)
    p.check_med_dose_cutoffs(row, past=False, forms=CURR)
    assert p.final_output_list == []


def test_dose_cutoff_normalizes_float_med_code():
    """REDCap codes can arrive as floats (300.0) — must still resolve to '300'."""
    p = make_pharm_checks()
    row = Row(subjectid="S1", chrpharm_med1_name=300.0, chrpharm_med1_dosage=10)
    p.check_med_dose_cutoffs(row, past=False, forms=CURR)
    assert len(p.final_output_list) == 1


def test_dose_cutoff_past_form_flags():
    """Cutoff rule must also fire on past_pharmaceutical_treatment."""
    p = make_pharm_checks()
    row = Row(
        subjectid="S1",
        chrpharm_med1_name_past="300",
        chrpharm_med1_dosage_past=10,
        past_pharmaceutical_treatment_complete=2,
    )
    p.check_med_dose_cutoffs(row, past=True, forms=PAST)
    assert len(p.final_output_list) == 1


# --- MED-QC-19: cross-form AP consistency (lifetime AP vs past pharm) ---


CROSS_FORMS = ["lifetime_ap_exposure_screen", "past_pharmaceutical_treatment"]


def test_constructor_dispatches_ap_pharm_check_at_screening_only(monkeypatch):
    """The production constructor must keep MED-QC-19 attached to dispatch.

    The individual-rule tests below exercise ``check_ap_pharm_mismatch``
    directly.  This regression instead constructs ``PharmChecks`` normally
    and follows ``__init__ -> call_checks -> call_pharm_checks`` so a future
    accidental commenting-out of the production call is caught.
    """

    dependency_loads = []

    def fake_form_check_init(instance, timepoint, network, form_check_info):
        instance.timepoint = timepoint
        instance.network = network
        instance.utils = FakeUtils()
        instance.utils.load_dependency_json = lambda filename: (
            dependency_loads.append((timepoint, filename)) or {})
        instance.final_output_list = []

    # Use the class actually present in PharmChecks' MRO.  This remains
    # reliable whether this isolated test module installed its lightweight
    # FormCheck stub or another test imported the real FormCheck first.
    monkeypatch.setattr(PharmChecks.__mro__[1], "__init__", fake_form_check_init)
    monkeypatch.setattr(
        PharmChecks,
        "_load_chrap_med_names",
        staticmethod(lambda utils: {}),
    )

    # Isolate dispatch from the implementations of all other medication
    # checks.  MED-QC-19 itself gets a spy below.
    other_checks = [
        "med_name_missing_check",
        "det_med_name_improper_value",
        "pharm_firstdose_check",
        "check_pharm_date_chronologies",
        "check_pharm_interview_dates",
        "check_current_med_dates_not_after_mod_date",
        "check_onset_after_offset",
        "check_duplicate_medication_courses",
        "check_no_medication_overlap",
        "check_no_medication_missing_offset",
        "check_no_ongoing_med_in_past_form",
        "check_current_med_not_ongoing_but_has_offset",
        "check_current_med_not_before_past_form",
        "check_no_medication_code_with_info",
        "check_compliance_range",
        "check_frequency_missing_code",
        "check_acute_injection_longer_than_two_weeks",
    ]
    for method_name in other_checks:
        monkeypatch.setattr(
            PharmChecks,
            method_name,
            lambda self, row, *args, **kwargs: None,
        )

    def fail_if_disabled_check_runs(instance, row, *args, **kwargs):
        pytest.fail("an intentionally disabled medication check was dispatched")

    for method_name in (
        "check_combination_med_dose_format",
        "check_med_dose_cutoffs",
    ):
        monkeypatch.setattr(
            PharmChecks, method_name, fail_if_disabled_check_runs)

    calls = []

    def capture_ap_check(instance, row, forms):
        calls.append((instance.timepoint, row, list(forms)))

    monkeypatch.setattr(PharmChecks, "check_ap_pharm_mismatch", capture_ap_check)

    screening_row = Row(subjectid="SCREEN")
    followup_row = Row(subjectid="MONTH1")
    PharmChecks(screening_row, "screening", "PRONET", {})
    PharmChecks(followup_row, "month1", "PRONET", {})

    assert calls == [("screening", screening_row, CROSS_FORMS)]
    assert dependency_loads == [("screening", "ap_med_mappings.json")]


def _ap_pharm_checks():
    """PharmChecks fixture with pre-built AP lookups (small canned subset of
    ap_med_mappings.json) and both form completion gates bypassed.

    Mapping: chrap_ris → {300, 425, 452}, chrap_olz → {201, 726, 720}.
    """
    p = make_pharm_checks()
    p._chrap_to_pharm_codes = {
        "chrap_ris": {"300", "425", "452"},
        "chrap_olz": {"201", "726", "720"},
    }
    p._pharm_code_to_chrap_vars = {
        "300": {"chrap_ris"}, "425": {"chrap_ris"}, "452": {"chrap_ris"},
        "201": {"chrap_olz"}, "726": {"chrap_olz"}, "720": {"chrap_olz"},
    }
    # Canned subset of the data-dictionary-derived name lookup built by
    # PharmChecks._load_chrap_med_names in production.
    p._chrap_to_med_name = {
        "chrap_ris": "Risperidone (Risperdal)",
        "chrap_olz": "Olanzapine (Zyprexa)",
    }
    p._lifetime_ap_marked_complete = lambda row: True
    p.important_form_vars["lifetime_ap_exposure_screen"] = {
        "completion_var": "lifetime_ap_exposure_screen_complete",
    }
    return p


def test_ap_pharm_consistent_no_flag():
    """chrap_ris=1 AND past pharm has 300 (Risperidone) → no flag."""
    p = _ap_pharm_checks()
    row = Row(subjectid="S1", chrap_ris=1, chrpharm_med1_name_past="300")
    p.check_ap_pharm_mismatch(row, forms=CROSS_FORMS)
    assert p.final_output_list == []


def test_ap_pharm_lifetime_yes_past_missing_flags():
    """Direction 1: chrap_ris=1 but no matching past pharm code → flag."""
    p = _ap_pharm_checks()
    row = Row(subjectid="S1", chrap_ris=1, chrpharm_med1_name_past="201")
    p.check_ap_pharm_mismatch(row, forms=CROSS_FORMS)
    assert len(p.final_output_list) == 1
    flag = p.final_output_list[0]
    assert "chrap_ris" in flag["affected_variables"]
    assert "lifetime_ap_exposure_screen" in flag["error_message"]


def test_ap_pharm_past_has_ap_lifetime_no_flags():
    """Direction 2: past pharm has 300 but chrap_ris=0 → flag, with both
    chrap and matching name_var on affected_variables."""
    p = _ap_pharm_checks()
    row = Row(subjectid="S1", chrap_ris=0, chrpharm_med1_name_past="300")
    p.check_ap_pharm_mismatch(row, forms=CROSS_FORMS)
    assert len(p.final_output_list) == 1
    flag = p.final_output_list[0]
    assert "chrap_ris" in flag["affected_variables"]
    assert "chrpharm_med1_name_past" in flag["affected_variables"]


def test_ap_pharm_past_has_ap_lifetime_two_no_flag():
    """Direction 2 fires for any non-1 explicit value, not just 0 (REDCap
    Y/N can be coded 1/2). Locks 'not 1 and not missing' branch."""
    p = _ap_pharm_checks()
    row = Row(subjectid="S1", chrap_ris=2, chrpharm_med1_name_past="300")
    p.check_ap_pharm_mismatch(row, forms=CROSS_FORMS)
    assert len(p.final_output_list) == 1


def test_ap_pharm_lifetime_missing_no_flag():
    """chrap value is missing/sentinel → skip (incomplete-data, not contradiction)."""
    p = _ap_pharm_checks()
    row = Row(subjectid="S1", chrap_ris=-9, chrpharm_med1_name_past="300")
    p.check_ap_pharm_mismatch(row, forms=CROSS_FORMS)
    assert p.final_output_list == []


def test_ap_pharm_chrap_missing_button_skips_all():
    """chrap_missing=1 (RA marked form not collected) → bail entirely."""
    p = _ap_pharm_checks()
    row = Row(
        subjectid="S1",
        chrap_missing=1,
        chrap_ris=1,
        chrpharm_med1_name_past="201",
    )
    p.check_ap_pharm_mismatch(row, forms=CROSS_FORMS)
    assert p.final_output_list == []


def test_ap_pharm_past_pharm_incomplete_skips_all():
    """Bail when past_pharmaceutical_treatment is not marked complete."""
    p = _ap_pharm_checks()
    p._past_pharm_marked_complete = lambda row: False
    row = Row(subjectid="S1", chrap_ris=1, chrpharm_med1_name_past="201")
    p.check_ap_pharm_mismatch(row, forms=CROSS_FORMS)
    assert p.final_output_list == []


def test_ap_pharm_lifetime_form_incomplete_skips_all():
    """Bail when lifetime_ap_exposure_screen is not marked complete."""
    p = _ap_pharm_checks()
    p._lifetime_ap_marked_complete = lambda row: False
    row = Row(subjectid="S1", chrap_ris=1, chrpharm_med1_name_past="201")
    p.check_ap_pharm_mismatch(row, forms=CROSS_FORMS)
    assert p.final_output_list == []


def test_ap_pharm_brand_name_code_also_matches():
    """A second pharm code mapped to the same chrap_var (e.g. Risperidone
    code 425) must also satisfy the 'has matching course' condition."""
    p = _ap_pharm_checks()
    row = Row(subjectid="S1", chrap_ris=1, chrpharm_med1_name_past="425")
    p.check_ap_pharm_mismatch(row, forms=CROSS_FORMS)
    assert p.final_output_list == []


def test_ap_pharm_normalizes_float_past_code():
    """REDCap can deliver pharm codes as floats (300.0); must normalize and match."""
    p = _ap_pharm_checks()
    row = Row(subjectid="S1", chrap_ris=1, chrpharm_med1_name_past=300.0)
    p.check_ap_pharm_mismatch(row, forms=CROSS_FORMS)
    assert p.final_output_list == []


def test_ap_pharm_multiple_drugs_one_pass():
    """Two independent AP mismatches → two flags in a single pass."""
    p = _ap_pharm_checks()
    row = Row(
        subjectid="S1",
        chrap_ris=1,           # lifetime yes, past has no risperidone → flag
        chrap_olz=0,           # lifetime no, past has olanzapine → flag
        chrpharm_med1_name_past="201",  # olanzapine
    )
    p.check_ap_pharm_mismatch(row, forms=CROSS_FORMS)
    assert len(p.final_output_list) == 2


def test_ap_pharm_uncovered_past_code_no_flag():
    """Past pharm code not in the AP mapping (e.g. 999, 122 if not in fixture)
    must not trigger Direction 2."""
    p = _ap_pharm_checks()
    row = Row(subjectid="S1", chrap_ris=0, chrpharm_med1_name_past="9999")
    p.check_ap_pharm_mismatch(row, forms=CROSS_FORMS)
    assert p.final_output_list == []


def test_build_ap_lookups_handles_empty_entries():
    """Brand-name-only entries with empty chrap_var or empty pharm_vals are
    skipped, not crashed."""
    mappings = {
        "Risperidone": {"pharm_vals": ["300", "425"], "chrap_var": "chrap_ris"},
        "Risperdal":   {"pharm_vals": ["300"], "chrap_var": "chrap_ris"},
        "Largactil":   {"pharm_vals": [], "chrap_var": ""},
        "Abilitat":    {"pharm_vals": [], "chrap_var": ""},
    }
    forward, reverse = PharmChecks._build_ap_lookups(mappings)
    assert forward == {"chrap_ris": {"300", "425"}}
    assert reverse == {"300": {"chrap_ris"}, "425": {"chrap_ris"}}
