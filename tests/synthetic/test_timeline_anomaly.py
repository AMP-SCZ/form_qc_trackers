"""Precision and integration regressions for timeline_anomaly."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import json

import pandas as pd

from simple_anomaly_detection import timeline_anomaly
from simple_anomaly_detection.common import clean_dates_series
from simple_anomaly_detection.runner import _DETECTORS


def _timeline_slices(
    gaps,
    *,
    earlier="month1",
    later="month2",
    variable="visit_date",
    ids=None,
):
    base = date(2026, 1, 15)
    gaps = list(gaps)
    ids = list(ids or [f"AA{i:04d}" for i in range(len(gaps))])
    left = pd.DataFrame({
        "subjectid": ids,
        variable: [base.isoformat()] * len(gaps),
    })
    right = pd.DataFrame({
        "subjectid": [value.strip() for value in ids],
        variable: [(base + timedelta(days=int(gap))).isoformat()
                   for gap in gaps],
    })
    per_slice = {
        ("PRONET", earlier): left,
        ("PRONET", later): right,
    }
    classify = {
        ("PRONET", earlier): {"date": [variable]},
        ("PRONET", later): {"date": [variable]},
    }
    return per_slice, classify


def test_expected_month_gaps_are_not_off_by_one():
    expected = timeline_anomaly._expected_gap_days
    assert expected("baseline", "month1") == 30
    assert expected("month1", "month2") == 30
    assert expected("month3", "month6") == 90
    assert expected("month12", "month18") == 180
    assert expected("month18", "month24") == 180
    assert expected("month2", "month2") is None
    assert expected("month2", "month1") is None
    assert expected("screening", "baseline") is None


def test_routine_visit_window_variation_does_not_flag_and_order_is_invariant():
    gaps = [28 + (i % 5) for i in range(45)]
    per_slice, classify = _timeline_slices(gaps)
    assert timeline_anomaly.detect_all(per_slice, classify=classify) == []

    reversed_slices = dict(reversed(list(per_slice.items())))
    reversed_classify = dict(reversed(list(classify.items())))
    assert timeline_anomaly.detect_all(
        reversed_slices, classify=reversed_classify) == []


def test_same_timepoint_date_fields_are_never_compared_as_a_timeline():
    frame = pd.DataFrame({
        "subjectid": [f"AA{i:04d}" for i in range(45)],
        "visit_date": ["2026-02-15"] * 45,
        "assessment_date": ["2026-02-15"] * 45,
    })
    slices = {("PRONET", "month1"): frame}
    classify = {("PRONET", "month1"): {
        "date": ["visit_date", "assessment_date"]}}
    assert timeline_anomaly.detect_all(slices, classify=classify) == []


def test_static_consent_dates_are_not_treated_as_visit_timelines():
    slices, classify = _timeline_slices(
        [-30, 30, 30], earlier="baseline", later="month1",
        variable="consent_date")
    assert timeline_anomaly.detect_all(slices, classify=classify) == []


def test_main_date_report_administrative_fields_stay_excluded():
    for variable in timeline_anomaly.EXCLUDED_TIMELINE_VARIABLES:
        assert not timeline_anomaly._is_timeline_date_variable(
            variable, {variable: ({"interview_date_var": variable},)})


def test_populated_contract_does_not_admit_unmapped_visit_substrings():
    contracts = {
        "mapped_visit_date": ({
            "interview_date_var": "mapped_visit_date",
        },),
    }
    assert timeline_anomaly._is_timeline_date_variable(
        "mapped_visit_date", contracts)
    assert timeline_anomaly._is_timeline_date_variable("visit_date", contracts)
    for unrelated in (
            "previous_visit_date", "hospital_visit_date",
            "last_interview_date", "interview_training_date"):
        assert not timeline_anomaly._is_timeline_date_variable(
            unrelated, contracts)


def test_mapped_interview_date_bypasses_shared_variable_exclusions(tmp_path):
    # common.classify_columns excludes the chrdig prefix globally, but the
    # important-form contract explicitly identifies this as an interview date
    # and the dedicated timeline detector must still evaluate it.
    variable = "chrdig_interview_date"
    contract_path = tmp_path / "important_form_vars.json"
    contract_path.write_text(json.dumps({
        "digital_assessment": {
            "interview_date_var": variable,
            "missing_var": "chrdig_missing",
            "completion_var": "chrdig_complete",
        },
    }), encoding="utf-8")
    slices, _classify = _timeline_slices(
        [-1], earlier="month1", later="month2", variable=variable)
    rows = timeline_anomaly.detect_all(
        slices, important_form_vars_path=str(contract_path))
    assert len(rows) == 1
    assert rows[0]["variable"] == variable
    assert rows[0]["finding_kinds"] == "order_reversal"


def test_conversion_and_floating_are_excluded_from_the_timeline():
    base = pd.DataFrame({"subjectid": ["AA001"],
                         "visit_date": ["2026-01-15"]})
    bad = pd.DataFrame({"subjectid": ["AA001"],
                        "visit_date": ["1900-01-01"]})
    slices = {
        ("PRONET", "baseline"): base,
        ("PRONET", "conversion"): bad,
        ("PRONET", "floating"): bad,
    }
    classify = {key: {"date": ["visit_date"]} for key in slices}
    assert timeline_anomaly.detect_all(slices, classify=classify) == []


def test_dates_on_forms_marked_missing_are_excluded(tmp_path):
    contract_path = tmp_path / "important_form_vars.json"
    contract_path.write_text(json.dumps({
        "visit_form": {
            "interview_date_var": "visit_date",
            "missing_var": "visit_missing",
            "completion_var": "visit_complete",
        },
    }), encoding="utf-8")
    slices, classify = _timeline_slices(
        [-30, 30, 30], earlier="baseline", later="month1")
    for frame in slices.values():
        frame["visit_missing"] = [1, 0, 0]

    rows = timeline_anomaly.detect_all(
        slices, classify=classify,
        important_form_vars_path=str(contract_path))
    assert rows == []


def test_prescient_missing_completion_codes_exclude_stale_dates(tmp_path):
    contract_path = tmp_path / "important_form_vars.json"
    contract_path.write_text(json.dumps({
        "visit_form": {
            "interview_date_var": "visit_date",
            "missing_var": "visit_missing",
            "completion_var": "visit_complete",
        },
    }), encoding="utf-8")
    slices, classify = _timeline_slices(
        [-30, 30, 30], earlier="baseline", later="month1")
    prescient_slices = {
        ("PRESCIENT", tp): frame.assign(
            visit_missing=[0, 0, 0],
            visit_complete_rpms=[3, 0, 0],
        )
        for (_network, tp), frame in slices.items()
    }
    prescient_classify = {
        ("PRESCIENT", tp): value
        for (_network, tp), value in classify.items()
    }
    rows = timeline_anomaly.detect_all(
        prescient_slices, classify=prescient_classify,
        important_form_vars_path=str(contract_path))
    assert rows == []


def test_sparse_later_before_earlier_is_still_caught_and_ids_are_trimmed():
    slices, classify = _timeline_slices(
        [-1, 30, 30], earlier="baseline", later="month1",
        ids=[" AA001 ", "AA002", "AA003"])
    rows = timeline_anomaly.detect_all(slices, classify=classify)
    assert len(rows) == 1
    assert rows[0]["subjectid"] == "AA001"
    assert rows[0]["site_id"] == "AA"
    assert rows[0]["finding_kinds"] == "order_reversal"
    assert rows[0]["variable"] == "visit_date"


def test_one_zero_mad_cadence_outlier_is_caught_without_flagging_the_cohort():
    gaps = [30] * 45
    gaps[7] = 60
    slices, classify = _timeline_slices(gaps)
    rows = timeline_anomaly.detect_all(slices, classify=classify)
    assert len(rows) == 1
    assert rows[0]["subjectid"] == "AA0007"
    assert rows[0]["finding_kinds"] == "cadence_outlier"


def test_zero_mad_with_harmless_calendar_variation_still_catches_gross_typo():
    # More than half the cohort is exactly 30 days, so MAD is zero, but the
    # 29/31-day visits mean fewer than 80% are bit-for-bit equal to the median.
    # They are all inside the relevance floor and should form one dominant
    # cadence against which the single 500-day entry is actionable.
    gaps = [30] * 30 + [29] * 7 + [31] * 7 + [500]
    slices, classify = _timeline_slices(gaps)
    rows = timeline_anomaly.detect_all(slices, classify=classify)
    assert len(rows) == 1
    assert rows[0]["subjectid"] == "AA0044"
    assert rows[0]["finding_kinds"] == "cadence_outlier"


def test_order_and_pair_fanout_collapses_to_one_subject_variable_issue():
    ids = [f"AA{i:04d}" for i in range(45)]
    base = date(2026, 1, 15)
    frames = {}
    classify = {}
    for tp, offset in (("baseline", 0), ("month1", 30), ("month2", 60)):
        values = [(base + timedelta(days=offset)).isoformat()] * len(ids)
        frames[("PRONET", tp)] = pd.DataFrame({
            "subjectid": ids, "visit_date": values})
        classify[("PRONET", tp)] = {"date": ["visit_date"]}

    # One month1 typo is both a cadence outlier from baseline and later than
    # month2.  Pairwise evidence must collapse to one actionable review row.
    frames[("PRONET", "month1")].loc[0, "visit_date"] = "2030-01-01"
    rows = timeline_anomaly.detect_all(frames, classify=classify)
    target = [row for row in rows if row["subjectid"] == "AA0000"]
    assert len(target) == 1
    assert target[0]["variable"] == "visit_date"
    assert target[0]["corroborating_rows"] >= 2
    assert target[0]["corroborating_pairs"] >= 2
    assert "order_reversal" in target[0]["finding_kinds"]
    assert "cadence_outlier" in target[0]["finding_kinds"]


def test_dedup_displays_definitive_order_reversal_over_stronger_cadence_row():
    ids = [f"AA{i:04d}" for i in range(45)]
    base = date(2026, 1, 15)
    frames = {}
    classify = {}
    for tp, offset in (("baseline", 0), ("month1", 30), ("month2", 60)):
        frames[("PRONET", tp)] = pd.DataFrame({
            "subjectid": ids,
            "visit_date": [
                (base + timedelta(days=offset)).isoformat()
            ] * len(ids),
        })
        classify[("PRONET", tp)] = {"date": ["visit_date"]}

    # Baseline -> month1 is a 61-day cadence anomaly, while month1 -> month2
    # is a definitive one-day reversal. The order row must be the displayed
    # representative even when the cadence severity is numerically higher.
    frames[("PRONET", "month1")].loc[0, "visit_date"] = (
        base + timedelta(days=61)).isoformat()
    rows = timeline_anomaly.detect_all(frames, classify=classify)
    target = [row for row in rows if row["subjectid"] == "AA0000"]
    assert len(target) == 1
    assert target[0]["method"] == "deterministic timeline ordering"
    assert target[0]["timepoint"] == "month1 -> month2"
    assert "month1=" in target[0]["observed_value"]
    assert "month2=" in target[0]["observed_value"]
    assert target[0]["finding_kinds"] == "cadence_outlier; order_reversal"


def test_all_configured_numeric_and_date_missing_codes_clean_to_nat():
    values = [
        "-3", "-9", -3, -9, -3.0, -9.0, "-3.0", "-9.0",
        "1909-09-09", "1903-03-03", "1901-01-01",
        "-99", -99, -99.0, "-99.0",
        999, 999.0, "999", "999.0",
        datetime(1909, 9, 9, 12, 30),
        "1903-03-03T12:00:00Z",
    ]
    cleaned = clean_dates_series(pd.Series(values, dtype=object))
    assert cleaned.isna().all(), cleaned.tolist()

    real = clean_dates_series(pd.Series(["2026-02-03", date(2026, 2, 4)]))
    assert real.notna().all()


def test_date_cleaning_preserves_source_calendar_day_across_offsets():
    cleaned = clean_dates_series(pd.Series([
        "2026-01-01T23:30:00-05:00",
        "2026-01-02T00:30:00+05:00",
        # This sentinel crosses to the prior day in UTC but must remain missing
        # according to the source calendar date.
        "1903-03-03T00:30:00+14:00",
    ]))
    assert str(cleaned.iloc[0].date()) == "2026-01-01"
    assert str(cleaned.iloc[1].date()) == "2026-01-02"
    assert pd.isna(cleaned.iloc[2])


def test_pandas_one_compatibility_path_parses_mixed_shapes(monkeypatch):
    monkeypatch.setattr(pd, "__version__", "1.4.2")
    cleaned = clean_dates_series(pd.Series([
        "2026-01-01",
        "02/01/2026 12:30:00",
        "2026-03-01T05:00:00Z",
    ]))
    assert cleaned.notna().all(), cleaned.tolist()


def test_runner_registers_timeline_with_a_unique_anomaly_type():
    registry = dict(_DETECTORS)
    assert registry["timeline_anomaly"] is timeline_anomaly
    assert timeline_anomaly.ANOMALY_TYPE != "date_anomaly_simple"
