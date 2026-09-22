"""Focused regression tests for reviewer-relevant time-series findings."""

import numpy as np
import pandas as pd

from simple_anomaly_detection.time_series import (
    _collapse_adjacent_reversals,
    _detect_variable_by_tp,
    _detect_variable_rate,
)


def _two_visit_rows(subject: str, tp_a: str, tp_b: str,
                    date_a: pd.Timestamp, date_b: pd.Timestamp,
                    value_a: float, value_b: float):
    return [
        {"subjectid": subject, "_tp": tp_a, "_tp_rank": 0,
         "visit_date": date_a, "score": value_a},
        {"subjectid": subject, "_tp": tp_b, "_tp_rank": 1,
         "visit_date": date_b, "score": value_b},
    ]


def test_rate_scoring_does_not_compare_incompatible_study_phases():
    """Ordinary early change must not be judged against a flat late phase.

    In the former pooled implementation the 30 early rates were 3% of the
    1,000-rate pool, so every early row cleared z=4 while the global rarity
    gate still allowed them through.
    """
    rows = []
    start = pd.Timestamp("2025-01-01")
    for i in range(30):
        rate = 1.0 + (i - 14.5) * 0.002
        rows.extend(_two_visit_rows(
            f"EA{i:04d}", "baseline", "month1", start,
            start + pd.Timedelta(days=30), 0.0, 30.0 * rate))
    for i in range(970):
        rate = (i - 484.5) * 0.000002
        rows.extend(_two_visit_rows(
            f"LA{i:04d}", "month11", "month12", start,
            start + pd.Timedelta(days=30), 0.0, 30.0 * rate))

    findings = _detect_variable_rate(
        pd.DataFrame(rows), "score", "visit_date", "PRONET")
    assert findings == []


def test_rate_floored_denominator_is_explicit_in_evidence():
    rows = []
    start = pd.Timestamp("2025-01-01")
    for i in range(30):
        if i == 0:
            days, delta = 1, 70.0  # scored as 70 / 7, not 70 / 1
        else:
            days = 30
            delta = 30.0 * (1.0 + (i - 15) * 0.003)
        rows.extend(_two_visit_rows(
            f"AA{i:03d}", "baseline", "month1", start,
            start + pd.Timedelta(days=days), 0.0, delta))

    findings = _detect_variable_rate(
        pd.DataFrame(rows), "score", "visit_date", "PRONET")
    target = [r for r in findings if r["subjectid"] == "AA000"]
    assert len(target) == 1
    row = target[0]
    assert row["elapsed_days"] == 1.0
    assert row["effective_elapsed_days"] == 7.0
    assert row["denominator_was_floored"] is True
    assert "1 actual d" in row["observed_value"]
    assert "effective denominator 7 d" in row["observed_value"]


def test_timepoint_fallback_bridges_a_missing_intermediate_visit():
    rows = []
    for i in range(30):
        month2 = 50.0 if i == 0 else 2.0 + (i - 15) * 0.01
        rows.extend([
            {"subjectid": f"AA{i:03d}", "_tp": "baseline", "_tp_rank": 0,
             "score": 0.0},
            {"subjectid": f"AA{i:03d}", "_tp": "month1", "_tp_rank": 1,
             "score": np.nan},
            {"subjectid": f"AA{i:03d}", "_tp": "month2", "_tp_rank": 2,
             "score": month2},
        ])

    findings = _detect_variable_by_tp(pd.DataFrame(rows), "score", "PRONET")
    target = [r for r in findings if r["subjectid"] == "AA000"]
    assert len(target) == 1
    assert target[0]["timepoint"] == "baseline->month2"
    assert target[0]["transition_from"] == "baseline"
    assert target[0]["transition_to"] == "month2"


def test_middle_visit_spike_collapses_two_opposite_transitions():
    rows = []
    start = pd.Timestamp("2025-01-01")
    for i in range(30):
        if i == 0:
            values = (0.0, 100.0, 0.0)
        else:
            m1 = (i - 15) * 0.02
            values = (0.0, m1, m1 + (i - 15) * 0.015)
        for rank, (tp, value) in enumerate(zip(
                ("baseline", "month1", "month2"), values)):
            rows.append({
                "subjectid": f"AA{i:03d}", "_tp": tp, "_tp_rank": rank,
                "visit_date": start + pd.Timedelta(days=30 * rank),
                "score": value,
            })

    raw = _detect_variable_rate(
        pd.DataFrame(rows), "score", "visit_date", "PRONET")
    target_raw = [r for r in raw if r["subjectid"] == "AA000"]
    assert {r["timepoint"] for r in target_raw} == {
        "baseline->month1", "month1->month2"}

    collapsed = _collapse_adjacent_reversals(target_raw)
    assert len(collapsed) == 1
    row = collapsed[0]
    assert row["timepoint"] == "baseline->month1->month2"
    assert row["episode_middle_timepoint"] == "month1"
    assert row["episode_direction"] == "spike"
    assert row["n_corroborating_transitions"] == 2
    assert "baseline->month1" in row["corroborating_transitions"]
    assert "month1->month2" in row["corroborating_transitions"]

