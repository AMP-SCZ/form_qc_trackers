"""Regression tests for reviewer-facing relevance and diversity controls."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def test_exclusion_rules_are_precise_and_drop_administrative_fields():
    from simple_anomaly_detection.common import (
        is_excluded_variable, looks_like_date_column)

    # Historical literal-substring collisions remain clinical variables.
    assert not is_excluded_variable("chrdemo_current_country")
    assert not is_excluded_variable("chrpps_crackle")
    assert not is_excluded_variable("chrap_app")

    for name in ("_tp", "chrpanss_interview_ampscz_id", "chric_record_id",
                 "chrpharm_medication", "chrpsychs_scr_gen_inst2",
                 "chrpsychs_scr_1a3_err", "chrpsychs_fu_1c6_conv_app",
                 "chrcssrsfu_redacp_user"):
        assert is_excluded_variable(name), name

    update = pd.Series(["0", "1"] * 20, dtype=object)
    assert not looks_like_date_column("chrscid_scid5_update_yespanic", update)


def test_format_detector_ignores_natural_free_text():
    from simple_anomaly_detection import format_string

    n = 120
    df = pd.DataFrame({
        "subjectid": [f"AA{i:04d}" for i in range(n)],
        "chrmri_dental_explain": [
            f"Participant explanation number {i} with natural prose." for i in range(n)
        ],
    })
    assert format_string.detect_all({("PRONET", "baseline"): df}) == []


def test_diverse_selection_caps_each_pair_leg_and_duplicate_rows():
    from simple_anomaly_detection.selection import (
        select_diverse_findings, variable_components)

    rows = []
    for i in range(20):
        rows.append({
            "anomaly_type": "pair", "severity_score": 100 - i,
            "raw_score": 20 - i / 10, "network": "PRONET",
            "timepoint": "baseline", "site_id": "AA", "subjectid": f"A{i}",
            "variable": f"dominant | partner{i}", "observed_value": str(i),
            "method": "pair method",
        })
    for i in range(6):
        rows.append({
            "anomaly_type": "single", "severity_score": 70 - i,
            "raw_score": 7 - i / 10, "network": "PRONET",
            "timepoint": "baseline", "site_id": "BB", "subjectid": f"B{i}",
            "variable": f"independent{i}", "observed_value": str(i),
            "method": "single method",
        })
    rows.append(dict(rows[0]))  # exact physical duplicate
    out = select_diverse_findings(pd.DataFrame(rows), row_cap=100,
                                  max_per_variable=3, max_per_subject=10)
    dominant = sum("dominant" in variable_components(r)
                   for _, r in out.iterrows())
    assert dominant == 3
    assert len(out) == 9  # 3 dominant-leg rows + all 6 independent rows
    assert out.duplicated(["anomaly_type", "network", "timepoint", "subjectid",
                           "variable", "observed_value", "method"]).sum() == 0
    assert out["severity_score"].is_monotonic_decreasing


def test_combined_selection_caps_composite_pseudo_variable():
    from simple_anomaly_detection.selection import (
        select_diverse_findings, variable_components)

    rows = [{"anomaly_type": "whole_subject_outlier", "severity_score": 90 - i,
             "raw_score": 9 - i / 10, "network": "PRONET",
             "subjectid": f"AA{i:03d}", "site_id": "AA",
             "variable": "(subject composite)", "method": "composite"}
            for i in range(12)]
    out = select_diverse_findings(
        pd.DataFrame(rows), row_cap=100, max_per_variable=4,
        max_per_subject=10, cap_pseudo_variables=True)
    assert len(out) == 4
    assert variable_components(out.iloc[0], cap_pseudo=True) == (
        "(subject composite)",)


def test_standard_common_tail_becomes_one_distribution_issue():
    from simple_anomaly_detection import standard_outlier

    # Median/MAD are defined by the 26 zeros/ones; four huge values are >5% of
    # n=30, so subject-detail would flood. Keep one batch/distribution summary.
    vals = [0, 1] * 13 + [100, 101, 102, 103]
    df = pd.DataFrame({"subjectid": [f"AA{i:03d}" for i in range(30)], "x": vals})
    cats = {"numeric": ["x"], "binary": [], "low_card": [], "text": [], "date": []}
    out = standard_outlier.detect_per_slice(df, "PRONET", "baseline", cats)
    assert len(out) == 1
    assert out[0]["subjectid"] == "(4 rows)"
    assert out[0]["n_flagged_rows"] == 4
    assert "distribution/batch" in out[0]["explanation"]


def test_date_order_ties_are_neutral_and_fanout_collapses():
    from simple_anomaly_detection import date_anomaly

    n = 60
    base = date(2026, 1, 1)
    subjects = [f"AA{i:03d}" for i in range(n)]
    # 54 ties and balanced 3/3 tails: neither direction is dominant.
    a = [base] * n
    b = [base] * 54 + [base + timedelta(days=400)] * 3 + [
        base - timedelta(days=400)] * 3
    left = ("baseline", "a_date", "baseline::a_date",
            pd.Series(pd.to_datetime(a), index=subjects))
    right = ("baseline", "b_date", "baseline::b_date",
             pd.Series(pd.to_datetime(b), index=subjects))
    assert date_anomaly._pair("PRONET", left, right) == []

    # One subject's bad date linked to several fields collapses to one connected
    # issue without guessing which symmetric leg is wrong.
    rows = []
    for i, other in enumerate(("b", "c", "d")):
        rows.append({"network": "PRONET", "subjectid": "AA999",
                     "variable": f"baseline::a | month{i+1}::{other}",
                     "variables_involved": f"baseline::a, month{i+1}::{other}",
                     "severity_score": 90 - i, "method": "paired date ordering",
                     "explanation": "x"})
    out = date_anomaly._deduplicate_findings(rows)
    assert len(out) == 1
    assert out[0]["corroborating_pairs"] == 3
    assert "primary_variable" not in out[0]


def test_site_detector_is_invariant_to_site_visit_mix():
    from simple_anomaly_detection import site_network

    rng = np.random.default_rng(42)
    per_slice = {}
    sites = ["AA", "BB", "CC", "DD", "EE"]
    for tp, center in (("baseline", 0.0), ("month12", 20.0)):
        rows = []
        for site_idx, site in enumerate(sites):
            # Deliberately different subject counts by site/visit. Within each
            # visit all sites share the same distribution.
            n = 40 + ((site_idx * 11 + (0 if tp == "baseline" else 17)) % 35)
            rows.extend({"subjectid": f"{site}{tp}{i:03d}",
                         "score": center + rng.normal()} for i in range(n))
        per_slice[("PRONET", tp)] = pd.DataFrame(rows)
    out = site_network.detect_all(per_slice)
    median_rows = [r for r in out if r.get("variable") == "score"
                   and "site medians" in str(r.get("method", ""))]
    assert median_rows == []


def test_loader_excludes_blank_and_duplicate_subject_ids(tmp_path):
    from simple_anomaly_detection.runner import SimpleAnomalyDetector

    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    df = pd.DataFrame({
        "subjectid": ["AA001", "AA001", "", "AA002", "AA002", " AA003 "],
        "x": [1, 1, 9999, 2, 2000, 3],
    })
    df.to_csv(inp / "AMPSCZ-combined-redcap_baseline_PRONET.csv", index=False)
    detector = SimpleAnomalyDetector(input_path=str(inp), output_path=str(out),
                                     networks=("PRONET",), timepoints=("baseline",))
    slices = detector._load_slices()
    loaded = slices[("PRONET", "baseline")]
    assert loaded["subjectid"].tolist() == ["AA001", "AA003"]
    log = detector._load_log[0]
    assert log["exact_duplicate_rows_collapsed"] == 1
    assert log["blank_id_rows_excluded"] == 1
    assert log["duplicate_subjectids_excluded"] == 1
    assert log["duplicate_rows_excluded"] == 2
