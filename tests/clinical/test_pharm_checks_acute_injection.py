"""
Tests for PharmChecks.check_acute_injection_longer_than_two_weeks (MED-QC-16),
post-A5: code-based detection via ACUTE_INJECTION_CODES restored from
dependencies/pharm_checks_original.py.

These tests bypass FormCheck.__init__ via PharmChecks.__new__ so they don't
require the full project directory layout (qc_forms/) to be present on disk.
The base FormCheck class and utils.utils.Utils are stubbed in sys.modules
before pharm_checks is imported, and the few attributes the function under
test reads are populated manually on each fresh instance.
"""

import os
import sys
import types
from datetime import datetime

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Test environment setup: stub qc_forms.form_check.FormCheck and
# utils.utils.Utils so PharmChecks (which subclasses FormCheck and imports
# Utils) can be imported in the slim local-test layout.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class _StubFormCheck:
    """Minimal stand-in for qc_forms.form_check.FormCheck. Tests use
    PharmChecks.__new__ to bypass __init__, so the stub only needs to be
    inheritable."""
    pass


class _StubUtils:
    """Stand-in for utils.utils.Utils. Tests construct their own FakeUtils
    per-instance; this stub only needs to satisfy the module-level import."""
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


# ---------------------------------------------------------------------------
# Test scaffolding.
# ---------------------------------------------------------------------------

class Row:
    """Attribute-access shim. hasattr(row, var) and getattr(row, var) behave
    naturally."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeUtils:
    """Stand-in for utils.utils.Utils. Mirrors the production missing list at
    utils/utils.py:34-37 so pharm_checks._is_missing_val routes through the
    same fallback path (lines 816-818) as in production."""

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
            datetime.strptime(string, date_format)
            return True
        except (TypeError, ValueError):
            return False

    def all_dtype(self, inp_list):
        """Mirror production utils.all_dtype: return int, str, float, and
        str-of-float variants of each input value."""
        out = []
        for n in inp_list:
            out.append(n)
            out.append(str(n))
            out.append(float(n))
            out.append(str(float(n)))
        return out


def _captured_create_row_output(row, forms, vars_list, message, reports):
    """Stub for FormCheck.create_row_output. Returns a flag-shaped dict that
    tests can introspect directly."""
    return {
        "subject": getattr(row, "subjectid", ""),
        "forms": list(forms),
        "affected_variables": list(vars_list),
        "error_message": message,
        "reports": reports,
    }


def make_pharm_checks():
    """Construct a PharmChecks instance via __new__, populating only the
    attributes that check_acute_injection_longer_than_two_weeks (and its
    helpers) read."""
    p = PharmChecks.__new__(PharmChecks)
    p.utils = FakeUtils()
    # Local class missing list — same as pharm_checks.py:41-60.
    p.missing_code_list = [
        "-3", "-9", -3, -9, -3.0, -9.0, "-3.0", "-9.0",
        "1909-09-09", "1903-03-03", "-99", -99, -99.0, "-99.0",
        999, 999.0, "999", "999.0",
    ]
    p.ONGOING_OFFSET_CODES = set(PharmChecks.ONGOING_OFFSET_CODES)
    p.final_output_list = []
    p.create_row_output = _captured_create_row_output
    # Network defaults to PRONET so MED-QC-01/03 use the default
    # Main Report / Non Team Forms routing; PRESCIENT routes those flags
    # to Secondary Report instead but is out of scope here.
    p.network = "PRONET"
    p.timepoint = "baseline"
    # Bypass the past-form completion gate in `_append_qc` so tests
    # exercise the per-check flag logic in isolation. Production behavior
    # is covered by integration runs.
    p._past_pharm_marked_complete = lambda row: True
    p.important_form_vars = {
        "past_pharmaceutical_treatment": {
            "completion_var": "past_pharmaceutical_treatment_complete",
        }
    }
    return p


CURR_FORMS = [
    "current_pharmaceutical_treatment_floating_med_125",
    "current_pharmaceutical_treatment_floating_med_2650",
]
PAST_FORMS = ["past_pharmaceutical_treatment"]

ALL_INJECTION_CODES = [11, 13, 76, 123, 151, 245, 257, 271, 724, 725, 726, 727]


# ---------------------------------------------------------------------------
# Class-level constant sanity check.
# ---------------------------------------------------------------------------

def test_acute_injection_codes_constant_value():
    """Lock the canonical set against silent edits."""
    assert PharmChecks.ACUTE_INJECTION_CODES == set(ALL_INJECTION_CODES)


# ---------------------------------------------------------------------------
# Core flag/no-flag behavior.
# ---------------------------------------------------------------------------

def test_acute_injection_code_724_long_course_flags():
    """Test 1: code 724 with duration > 14 days flags."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name=724,
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="2026-01-29",  # 28 days
    )
    p.check_acute_injection_longer_than_two_weeks(row, past=False, forms=CURR_FORMS)

    assert len(p.final_output_list) == 1
    flag = p.final_output_list[0]
    assert flag["affected_variables"] == [
        "chrpharm_med1_name", "chrpharm_med1_onset", "chrpharm_med1_offset",
    ]
    assert "chrpharm_med1_name = 724" in flag["error_message"]
    assert "lasts 28 days" in flag["error_message"]
    assert flag["forms"] == CURR_FORMS


def test_acute_injection_code_724_short_course_no_flag():
    """Test 2: code 724 with duration <= 14 days does not flag (9 days)."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name=724,
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="2026-01-10",  # 9 days
    )
    p.check_acute_injection_longer_than_two_weeks(row, past=False, forms=CURR_FORMS)
    assert p.final_output_list == []


def test_acute_injection_exactly_14_days_no_flag():
    """Test 3: exactly 14 days does not flag (boundary: > 14 excludes 14)."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name=724,
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="2026-01-15",  # exactly 14 days
    )
    p.check_acute_injection_longer_than_two_weeks(row, past=False, forms=CURR_FORMS)
    assert p.final_output_list == []


def test_acute_injection_exactly_15_days_flags():
    """Test 4: exactly 15 days flags (smallest flagged duration)."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name=724,
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="2026-01-16",  # exactly 15 days
    )
    p.check_acute_injection_longer_than_two_weeks(row, past=False, forms=CURR_FORMS)
    assert len(p.final_output_list) == 1


def test_acute_injection_non_acute_code_long_course_no_flag():
    """Test 5: non-acute code (300) with long duration does not flag."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name=300,  # not in ACUTE_INJECTION_CODES
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="2026-01-29",  # 28 days
    )
    p.check_acute_injection_longer_than_two_weeks(row, past=False, forms=CURR_FORMS)
    assert p.final_output_list == []


# ---------------------------------------------------------------------------
# Dtype tolerance.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name_value", [724, 724.0, "724", "724.0", " 724 "])
def test_acute_injection_dtype_variants_flag(name_value):
    """Test 6: dtype variants 724, 724.0, '724', '724.0', ' 724 ' all flag."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name=name_value,
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="2026-01-29",
    )
    p.check_acute_injection_longer_than_two_weeks(row, past=False, forms=CURR_FORMS)
    assert len(p.final_output_list) == 1, (
        f"Expected one flag for name_value={name_value!r}"
    )


# ---------------------------------------------------------------------------
# Full code-set coverage.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code", ALL_INJECTION_CODES)
def test_acute_injection_all_canonical_codes_flag(code):
    """Test 7: every code in ACUTE_INJECTION_CODES flags."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name=code,
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="2026-01-29",
    )
    p.check_acute_injection_longer_than_two_weeks(row, past=False, forms=CURR_FORMS)
    assert len(p.final_output_list) == 1, f"Expected flag for code={code}"
    assert f"= {code}" in p.final_output_list[0]["error_message"]


# ---------------------------------------------------------------------------
# Missing-code skipping.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_val", ["-9", "-3", "-99", "999", "", "nan"])
def test_acute_injection_missing_name_skipped(missing_val):
    """Test 8: missing-code name values do not flag."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name=missing_val,
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="2026-01-29",
    )
    p.check_acute_injection_longer_than_two_weeks(row, past=False, forms=CURR_FORMS)
    assert p.final_output_list == []


# ---------------------------------------------------------------------------
# Past-form parity.
# ---------------------------------------------------------------------------

def test_acute_injection_past_form_flags():
    """Test 9: past-form acute injection flags."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name_past=727,
        chrpharm_med1_onset_past="2026-01-01",
        chrpharm_med1_offset_past="2026-01-29",
    )
    p.check_acute_injection_longer_than_two_weeks(row, past=True, forms=PAST_FORMS)

    assert len(p.final_output_list) == 1
    flag = p.final_output_list[0]
    assert "chrpharm_med1_name_past" in flag["affected_variables"]
    assert "chrpharm_med1_onset_past" in flag["affected_variables"]
    assert "chrpharm_med1_offset_past" in flag["affected_variables"]
    assert flag["forms"] == PAST_FORMS
    assert "chrpharm_med1_name_past = 727" in flag["error_message"]


# ---------------------------------------------------------------------------
# Date-edge skipping.
# ---------------------------------------------------------------------------

def test_acute_injection_offset_before_onset_skipped():
    """Test 10: offset before onset skips."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name=724,
        chrpharm_med1_onset="2026-01-15",
        chrpharm_med1_offset="2026-01-01",
    )
    p.check_acute_injection_longer_than_two_weeks(row, past=False, forms=CURR_FORMS)
    assert p.final_output_list == []


def test_acute_injection_invalid_date_skipped():
    """Garbage date strings silently skip via _valid_date."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name=724,
        chrpharm_med1_onset="not-a-date",
        chrpharm_med1_offset="2026-01-29",
    )
    p.check_acute_injection_longer_than_two_weeks(row, past=False, forms=CURR_FORMS)
    assert p.final_output_list == []


def test_acute_injection_ongoing_offset_skipped():
    """Per A5 scope: keep ongoing-offset behavior unchanged. 1901-01-01 is
    currently filtered upstream by _valid_date because it is in
    utils.missing_code_list. After A2/F3 lands, the operator should revisit
    whether ongoing acute-injection courses should flag."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name=724,
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="1901-01-01",
    )
    p.check_acute_injection_longer_than_two_weeks(row, past=False, forms=CURR_FORMS)
    assert p.final_output_list == []


# ---------------------------------------------------------------------------
# 724.5 must NOT match (float-truncation regression guard).
# ---------------------------------------------------------------------------

def test_course_ongoing_var_matches_grouped_variables_redcap_names():
    """MED-QC-09 must read chrpharm_interm_meds / chrpharm_interm_meds_{n-1}."""
    p = make_pharm_checks()
    assert p._course_ongoing_var(1) == "chrpharm_interm_meds"
    assert p._course_ongoing_var(2) == "chrpharm_interm_meds_1"
    assert p._course_ongoing_var(3) == "chrpharm_interm_meds_2"


def test_normalize_date_str_rejects_missing_code_timestamp():
    """Parity with dependencies/pharm_checks_original: Timestamp-shaped
    study sentinels must not count as valid medication dates."""
    p = make_pharm_checks()
    assert p._normalize_date_str(pd.Timestamp("1903-03-03")) is None
    assert p._normalize_date_str(pd.Timestamp("1909-09-09")) is None


def test_no_medication_999_overlap_with_real_medication_flags():
    """MED-QC-06: 999 in a medication-name field is an explicit no-med period,
    not generic missingness."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name="999",
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="2026-02-01",
        chrpharm_med2_name="300",
        chrpharm_med2_onset="2026-01-15",
        chrpharm_med2_offset="2026-01-20",
    )
    p.check_no_medication_overlap(row, past=False, forms=CURR_FORMS)
    assert len(p.final_output_list) == 1
    flag = p.final_output_list[0]
    assert "chrpharm_med1_name" in flag["affected_variables"]
    assert "chrpharm_med2_name" in flag["affected_variables"]


def test_no_medication_999_missing_offset_with_later_medication_flags():
    """MED-QC-07: open-ended no-medication interval conflicts with later meds."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name=999,
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="",
        chrpharm_med2_name="300",
        chrpharm_med2_onset="2026-01-15",
    )
    p.check_no_medication_missing_offset(row, past=False, forms=CURR_FORMS)
    assert len(p.final_output_list) == 1
    flag = p.final_output_list[0]
    assert "chrpharm_med1_name" in flag["affected_variables"]
    assert "chrpharm_med2_name" in flag["affected_variables"]


def test_no_medication_999_with_course_info_flags():
    """MED-QC-12: 999/no-medication cannot carry dosage/frequency/etc."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name="999.0",
        chrpharm_med1_dosage="2",
        chrpharm_med1_frequency="1",
    )
    p.check_no_medication_code_with_info(row, past=False, forms=CURR_FORMS)
    assert len(p.final_output_list) == 1
    flag = p.final_output_list[0]
    assert "chrpharm_med1_name" in flag["affected_variables"]
    assert "chrpharm_med1_dosage" in flag["affected_variables"]


def test_no_medication_999_alone_no_flag():
    """MED-QC-12: explicit no-medication alone is valid."""
    p = make_pharm_checks()
    row = Row(subjectid="SUB001", chrpharm_med1_name="999")
    p.check_no_medication_code_with_info(row, past=False, forms=CURR_FORMS)
    assert p.final_output_list == []


def test_non_999_missing_med_name_still_skipped():
    """MED-QC-12 must not convert normal missing codes into no-medication."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name="-9",
        chrpharm_med1_dosage="2",
    )
    p.check_no_medication_code_with_info(row, past=False, forms=CURR_FORMS)
    assert p.final_output_list == []


def test_no_medication_999_with_zero_values_routes_to_secondary_report():
    """MED-QC-12: zero-like values (0, 0.0, '0') alongside a 999 no-medication
    code raise a Secondary Report flag rather than firing on the main report."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name="999",
        chrpharm_med1_dosage="0",
        chrpharm_med1_dosage_2=0,
        chrpharm_med1_frequency="0.0",
        chrpharm_med1_comp=0,
        chrpharm_med1_indication="0",
    )
    p.check_no_medication_code_with_info(row, past=False, forms=CURR_FORMS)
    assert len(p.final_output_list) == 1
    flag = p.final_output_list[0]
    assert flag["reports"]["reports"] == ["Secondary Report"]
    assert "chrpharm_med1_dosage" in flag["affected_variables"]
    assert "chrpharm_med1_frequency" in flag["affected_variables"]


def test_no_medication_999_with_zero_and_real_value_flags_both_reports():
    """MED-QC-12: zeros raise a Secondary Report flag while a real non-zero
    value raises the main flag — they are now two separate flags."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name="999",
        chrpharm_med1_dosage="0",
        chrpharm_med1_frequency="2",
    )
    p.check_no_medication_code_with_info(row, past=False, forms=CURR_FORMS)
    assert len(p.final_output_list) == 2

    main_flags = [
        f for f in p.final_output_list
        if f["reports"]["reports"] != ["Secondary Report"]
    ]
    secondary_flags = [
        f for f in p.final_output_list
        if f["reports"]["reports"] == ["Secondary Report"]
    ]
    assert len(main_flags) == 1
    assert len(secondary_flags) == 1
    assert "chrpharm_med1_frequency" in main_flags[0]["affected_variables"]
    assert "chrpharm_med1_dosage" not in main_flags[0]["affected_variables"]
    assert "chrpharm_med1_dosage" in secondary_flags[0]["affected_variables"]


def test_past_form_ongoing_1901_offset_flags():
    """MED-QC-08: 1901-01-01 is the ongoing offset code and is invalid in
    the past pharmaceutical form."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name_past="300",
        chrpharm_med1_offset_past="1901-01-01",
    )
    p.check_no_ongoing_med_in_past_form(row, forms=PAST_FORMS)
    assert len(p.final_output_list) == 1
    assert p.final_output_list[0]["affected_variables"] == ["chrpharm_med1_offset_past"]


def test_onset_after_offset_flags():
    """MED-QC-02: medication course cannot end before it starts."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name="300",
        chrpharm_med1_onset="2026-02-01",
        chrpharm_med1_offset="2026-01-01",
    )
    p.check_onset_after_offset(row, past=False, forms=CURR_FORMS)
    assert len(p.final_output_list) == 1


def test_current_onset_after_mod_date_flags():
    """MED-QC-03: current medication dates cannot be after form modification."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_date_mod="2026-01-01",
        chrpharm_med1_name="300",
        chrpharm_med1_onset="2026-01-02",
        chrpharm_med1_offset="2026-01-03",
    )
    p.check_current_med_dates_not_after_mod_date(row, forms=CURR_FORMS)
    assert len(p.final_output_list) == 2


def test_firstdose_mismatch_med1_flags():
    """MED-QC-04: first-dose datetime date must match med1 onset date."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_onset="2026-01-01",
        chrpharm_firstdose_med1="2026-01-02 09:30:00",
    )
    p.pharm_firstdose_check(row, CURR_FORMS)
    assert len(p.final_output_list) == 1
    assert p.final_output_list[0]["affected_variables"] == [
        "chrpharm_med1_onset",
        "chrpharm_firstdose_med1",
    ]


def test_firstdose_mismatch_med2_flags():
    """MED-QC-04: med_info specifies med{n}, so med2+ must be checked."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med2_name="300",
        chrpharm_med2_onset="2026-01-01",
        chrpharm_firstdose_med2="2026-01-02 09:30:00",
    )
    p.pharm_firstdose_check(row, CURR_FORMS)
    assert len(p.final_output_list) == 1
    assert p.final_output_list[0]["affected_variables"] == [
        "chrpharm_med2_onset",
        "chrpharm_firstdose_med2",
    ]


def test_duplicate_medication_course_flags():
    """MED-QC-05: same exact med name, overlapping dates, both use=1 flags."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name="300",
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="2026-02-01",
        chrpharm_med1_use="1",
        chrpharm_med2_name="300",
        chrpharm_med2_onset="2026-01-15",
        chrpharm_med2_offset="2026-01-20",
        chrpharm_med2_use="1",
    )
    p.check_duplicate_medication_courses(row, past=False, forms=CURR_FORMS)
    assert len(p.final_output_list) == 1


def test_compliance_outside_0_to_100_flags():
    """MED-QC-14: compliance must be 0-100 inclusive."""
    p = make_pharm_checks()
    row = Row(subjectid="SUB001", chrpharm_med1_name="300", chrpharm_med1_comp="101")
    p.check_compliance_range(row, past=False, forms=CURR_FORMS)
    assert len(p.final_output_list) == 1


def test_frequency_negative_except_missing_code_flags():
    """MED-QC-15: negative frequency is only allowed for -3."""
    p = make_pharm_checks()
    row = Row(subjectid="SUB001", chrpharm_med1_name="300", chrpharm_med1_frequency="-4")
    p.check_frequency_missing_code(row, past=False, forms=CURR_FORMS)
    assert len(p.final_output_list) == 1


def test_frequency_negative_missing_code_no_flag():
    """MED-QC-15: -3 remains an allowed missingness code."""
    p = make_pharm_checks()
    row = Row(subjectid="SUB001", chrpharm_med1_name="300", chrpharm_med1_frequency="-3")
    p.check_frequency_missing_code(row, past=False, forms=CURR_FORMS)
    assert p.final_output_list == []


def test_acute_injection_non_integer_float_no_match():
    """Test 11: '724.5' must NOT match. The set-membership pattern
    (str-on-strip vs all_dtype-expanded set) correctly rejects non-integer
    float-shaped values that an int(float(...)) coercion would have
    truncated to a canonical code."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name="724.5",
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="2026-01-29",
    )
    p.check_acute_injection_longer_than_two_weeks(row, past=False, forms=CURR_FORMS)
    assert p.final_output_list == []


def test_no_medication_touching_boundary_same_day_allowed():
    """MED-QC-06: adjacent intervals may share an endpoint (med_info example)."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name=300,
        chrpharm_med1_onset="2022-02-01",
        chrpharm_med1_offset="2022-02-05",
        chrpharm_med2_name=999,
        chrpharm_med2_onset="2022-02-05",
        chrpharm_med2_offset="2022-02-25",
    )
    p.check_no_medication_overlap(row, past=False, forms=CURR_FORMS)
    assert p.final_output_list == []


def test_acute_injection_ignores_free_text_not_rx_codes():
    """MED-QC-16 must use ACUTE_INJECTION_CODES only, not substring matching."""
    p = make_pharm_checks()
    row = Row(
        subjectid="SUB001",
        chrpharm_med1_name="acute injection ketamine course",
        chrpharm_med1_onset="2026-01-01",
        chrpharm_med1_offset="2026-02-01",
    )
    p.check_acute_injection_longer_than_two_weeks(row, past=False, forms=CURR_FORMS)
    assert p.final_output_list == []
