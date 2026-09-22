"""Focused regressions for correlative false-attribution/Simpson fixes."""

import numpy as np
import pandas as pd


def _pair_row(subj, a, b, sev, tp="baseline", residual_target=""):
    row = {
        "subjectid": subj,
        "network": "PRONET",
        "anomaly_type": "correlative_outlier",
        "severity_score": sev,
        "raw_score": sev / 10.0,
        "timepoint": tp,
        "variable": f"{a} | {b}",
        "variables_involved": f"{a}, {b}",
    }
    if residual_target:
        row["residual_target"] = residual_target
    return row


def test_binary_binary_single_timepoint_behavior_is_preserved():
    from simple_anomaly_detection.correlative import _binary_binary_pairs

    n = 400
    a = np.zeros(n, dtype=int)
    b = np.zeros(n, dtype=int)
    a[:100] = 1
    b[99:199] = 1  # one observed joint row; 25 expected under independence
    df = pd.DataFrame({
        "subjectid": [f"AA{i:04d}" for i in range(n)],
        "_tp": "baseline",
        "a": a,
        "b": b,
    })

    rows = _binary_binary_pairs(df, ["a", "b"], "PRONET")
    hits = [r for r in rows if r["variable"] == "a=1 & b=1"]
    assert len(hits) == 1
    assert hits[0]["subjectid"] == "AA0099"
    assert hits[0]["timepoint"] == "baseline"


def test_binary_binary_conditions_expectation_by_timepoint():
    from simple_anomaly_detection.correlative import (
        _binary_binary_pairs,
        _binary_binary_pairs_one_timepoint,
    )

    # At baseline A is almost universal and B rare; at month1 the marginals
    # reverse. Within each visit A/B co-occur at the ordinary conditional rate.
    # Pooled marginals are both 50%, however, and make the 0.5% joint cell look
    # 50x rarer than the pooled 25% independence expectation (Simpson artifact).
    n = 4000
    frames = []
    for tp, reverse in (("baseline", False), ("month1", True)):
        a = np.ones(n, dtype=int)
        b = np.zeros(n, dtype=int)
        if reverse:
            a[:] = 0
            b[:] = 1
            a[:20] = 1
        else:
            b[:20] = 1
        frames.append(pd.DataFrame({
            "subjectid": [f"{tp[:2].upper()}{i:05d}" for i in range(n)],
            "_tp": tp,
            "a": a,
            "b": b,
        }))
    stacked = pd.concat(frames, ignore_index=True)

    # Pin the discriminator: ignoring timepoint creates the exact false rows
    # this regression guards against; conditioning removes them.
    pooled = _binary_binary_pairs_one_timepoint(
        stacked.drop(columns=["_tp"]), ["a", "b"], "PRONET"
    )
    assert len([r for r in pooled if r["variable"] == "a=1 & b=1"]) == 40
    assert _binary_binary_pairs(stacked, ["a", "b"], "PRONET") == []


def test_degree_without_direction_never_attributes_shared_leg():
    from simple_anomaly_detection.correlative import _dedup_correlative

    rows = [
        _pair_row("S1", "A", "B", 90, residual_target="B"),
        _pair_row("S1", "A", "C", 80, residual_target="C"),
    ]
    out = _dedup_correlative(rows)

    assert len(out) == 2
    assert {r["variable"] for r in out} == {"A | B", "A | C"}
    assert all("likely error" not in str(r.get("explanation", "")) for r in out)


def test_explicit_residual_target_can_collapse_safely():
    from simple_anomaly_detection.correlative import _dedup_correlative

    rows = [
        _pair_row("S1", "X1", "Y", 70, residual_target="Y"),
        _pair_row("S1", "X2", "Y", 90, residual_target="Y"),
        _pair_row("S1", "X3", "Y", 60, residual_target="Y"),
    ]
    out = _dedup_correlative(rows)

    assert len(out) == 1
    assert out[0]["variable"] == "Y"
    assert out[0]["corroborating_pairs"] == 3
    assert "does not prove" in out[0]["explanation"]
