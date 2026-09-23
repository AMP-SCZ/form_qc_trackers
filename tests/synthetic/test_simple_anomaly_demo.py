"""Regression test for simple_anomaly_detection/ using demo.py's seeded
anomalies as a deterministic oracle.

simple_anomaly_detection is self-contained (numpy/pandas only), so this
builds the synthetic demo dataset into a tmp dir, runs the full
SimpleAnomalyDetector, and asserts:
  (1) every one of the configured detectors produces at least one flag, and
  (2) each deterministically-seeded anomaly surfaces in the expected
      detector's anomaly_type (combined headline or its quota-preserved detail
      tab when a dominant variable is deliberately deferred).

This is the safety net for the methodology/perf changes to the subsystem:
if a refactor stops a seeded anomaly from surfacing, this test fails.
"""
import importlib.util
import os
import sys

import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simple_anomaly_detection.demo import build_demo_dataset  # noqa: E402
from simple_anomaly_detection.runner import SimpleAnomalyDetector  # noqa: E402

# cluster_3d needs scikit-learn; it skips gracefully without it, so the
# cluster_3d assertions are guarded rather than hard-required.
_HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

DETECTORS = (
    "standard_outlier", "time_series", "correlative", "site_network",
    "format_string", "whole_subject_site", "date_anomaly",
    "timeline_anomaly", "rater_effect",
) + (("cluster_3d", "cluster_3d_longitudinal") if _HAS_SKLEARN else ())


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("sad_demo")
    in_dir = os.path.join(str(base), "in")
    out_dir = os.path.join(str(base), "out")  # sibling of in_dir (guard_paths)
    _, _, seeds = build_demo_dataset(in_dir)
    # Demo variables are deliberately generic synthetic names rather than
    # production REDCap fields, so the detector-method oracle runs unrestricted.
    det = SimpleAnomalyDetector(input_path=in_dir, output_path=out_dir,
                                clinical_only=False)
    results = det.run()
    combined = results["combined_ranked"]
    return results, combined, seeds


def test_every_detector_produces_flags(demo_run):
    results, _, _ = demo_run
    empty = [d for d in DETECTORS if results.get(d) is None or results[d].empty]
    assert not empty, f"detectors produced zero flags on the seeded demo: {empty}"


def test_seeded_anomalies_surface(demo_run):
    results, combined, seeds = demo_run
    assert not combined.empty
    sub = combined["subjectid"].astype(str)
    typ = combined["anomaly_type"].astype(str)
    misses = []
    _CLUSTER_TYPES = ("cluster_3d_outlier", "cluster_3d_longitudinal_outlier")
    for s in seeds:
        atype = s["anomaly_type"]
        if atype in _CLUSTER_TYPES and not _HAS_SKLEARN:
            continue  # detectors skip without scikit-learn
        if atype in _CLUSTER_TYPES:
            # combined_ranked now carries ONE combo-summary row per (network,
            # combo) for the cluster detectors, so the seeded subject is listed
            # in that row's `top_subjects`, not as the row's `subjectid`.
            tops = (combined["top_subjects"].astype(str)
                    if "top_subjects" in combined.columns else None)
            hit = tops is not None and ((typ == atype) & tops.str.contains(
                str(s["subjectid"]), na=False, regex=False)).any()
            # The combined headline now enforces a strict underlying-variable
            # quota across detector types. A lower-ranked but valid seeded combo
            # may be deferred to its detector detail tab; it must still surface
            # there (and in audit_candidates), even if not in the headline.
            if not hit:
                detail_name = ("cluster_3d_longitudinal"
                               if atype == "cluster_3d_longitudinal_outlier"
                               else "cluster_3d")
                detail = results.get(detail_name, pd.DataFrame())
                hit = (not detail.empty and
                       detail["subjectid"].astype(str).eq(str(s["subjectid"])).any())
        elif "subjectid" in s:
            hit = ((sub == str(s["subjectid"])) & (typ == atype)).any()
            if not hit:
                detail_name = next((name for name, frame in results.items()
                                    if name != "combined_ranked" and not frame.empty
                                    and "anomaly_type" in frame.columns
                                    and (frame["anomaly_type"].astype(str) == atype).any()),
                                   None)
                if detail_name is not None:
                    detail = results[detail_name]
                    hit = ((detail["subjectid"].astype(str) == str(s["subjectid"]))
                           & (detail["anomaly_type"].astype(str) == atype)).any()
        else:  # site-level seed
            site = combined["site_id"].astype(str)
            var = combined["variable"].astype(str)
            # site_network collapses multi-site rows: the concrete site may now
            # live in the `sites` column with site_id == "(N sites)".
            site_match = site == str(s["site_id"])
            if "sites" in combined.columns:
                site_match = site_match | combined["sites"].astype(str).str.contains(
                    rf"\b{s['site_id']}\b", na=False, regex=True)
            hit = ((typ == atype) & site_match
                   & var.str.contains(s.get("variable_contains", ""), na=False)).any()
        if not hit:
            misses.append(s["kind"])
    assert not misses, f"seeded anomalies that failed to surface as {atype!r}: {misses}"


def test_combined_ranked_is_sorted_desc(demo_run):
    _, combined, _ = demo_run
    sev = pd.to_numeric(combined["severity_score"], errors="coerce").dropna().to_numpy()
    assert (sev[:-1] >= sev[1:]).all(), "combined_ranked is not sorted by severity descending"


# --- correlative de-dup (the multi-partner flooding case the demo can't hit) ---

def _cand(subj, a, b, sev, tp="baseline", residual_target=""):
    row = {"subjectid": subj, "network": "PRONET", "anomaly_type": "correlative_outlier",
           "severity_score": sev, "raw_score": sev / 10.0, "timepoint": tp,
           "variable": f"{a} | {b}", "variables_involved": f"{a}, {b}"}
    if residual_target:
        row["residual_target"] = residual_target
    return row


def test_dedup_collapses_explicit_residual_target():
    from simple_anomaly_detection.correlative import _dedup_correlative
    # Y is explicitly the fitted residual target against X1/X2/X3. Directional
    # corroboration may collapse; graph degree without this metadata may not.
    rows = [_cand("S1", "X1", "Y", 70, residual_target="Y"),
            _cand("S1", "X2", "Y", 90, residual_target="Y"),
            _cand("S1", "X3", "Y", 60, residual_target="Y")]
    out = _dedup_correlative(rows)
    assert len(out) == 1, f"dominant-leg case should collapse to 1 row, got {len(out)}"
    r = out[0]
    assert r["variable"] == "Y"
    assert r["corroborating_pairs"] == 3
    assert float(r["severity_score"]) == 90  # max-severity representative kept


def test_dedup_keeps_isolated_pair_and_collapses_timepoints():
    from simple_anomaly_detection.correlative import _dedup_correlative
    # one isolated pair seen at two timepoints -> one row (max severity), per-pair
    rows = [_cand("S2", "A", "B", 55, tp="baseline"), _cand("S2", "A", "B", 80, tp="month1")]
    out = _dedup_correlative(rows)
    assert len(out) == 1
    assert float(out[0]["severity_score"]) == 80
    # an unrelated subject's single pair survives independently
    rows2 = rows + [_cand("S3", "C", "D", 50)]
    out2 = _dedup_correlative(rows2)
    assert {r["subjectid"] for r in out2} == {"S2", "S3"}


def test_dedup_emits_one_summary_per_explicit_residual_target():
    from simple_anomaly_detection.correlative import _dedup_correlative
    # Y and W are each explicit residual targets in >=2 distinct fits -> one
    # safe directional summary per target.
    rows = [_cand("S", "X1", "Y", 80, residual_target="Y"),
            _cand("S", "X2", "Y", 70, residual_target="Y"),
            _cand("S", "X3", "W", 65, residual_target="W"),
            _cand("S", "X4", "W", 60, residual_target="W")]
    out = _dedup_correlative(rows)
    assert {r["variable"] for r in out} == {"Y", "W"}


def test_dedup_passes_binary_binary_through():
    from simple_anomaly_detection.correlative import _dedup_correlative
    # BB joint-cell rows ("a=x & b=y") are pair-level; not leg-collapsed.
    bb = {"subjectid": "S9", "network": "PRONET", "anomaly_type": "correlative_outlier",
          "severity_score": 60, "raw_score": 1.0, "timepoint": "baseline",
          "variable": "smoker=1 & education=4", "variables_involved": "smoker, education"}
    out = _dedup_correlative([bb])
    assert len(out) == 1 and out[0]["variable"] == "smoker=1 & education=4"


# --- cluster_3d: lock in the spec-defining behaviour the demo can't exercise
#     (its single ~reference-size cluster makes size_factor a no-op). ---

@pytest.mark.skipif(not _HAS_SKLEARN, reason="cluster_3d needs scikit-learn")
def test_cluster3d_size_weight_is_sublinear_and_clamped():
    from simple_anomaly_detection.cluster_3d import _size_factor, SIZE_FACTOR_LO, SIZE_FACTOR_HI
    f10, f100, f1000 = _size_factor(10), _size_factor(100), _size_factor(1000)
    assert f10 < f100 < f1000                      # bigger cluster -> more weight
    assert SIZE_FACTOR_LO <= f10 and f1000 <= SIZE_FACTOR_HI   # clamped
    # sub-linear: a 100x size jump must NOT give anywhere near a 100x weight jump
    assert (f1000 / f10) < 4.0


@pytest.mark.skipif(not _HAS_SKLEARN, reason="cluster_3d needs scikit-learn")
def test_cluster3d_finds_two_clusters_and_flags_between_point():
    import numpy as np
    from sklearn.cluster import DBSCAN
    from simple_anomaly_detection.cluster_3d import _cluster_triple
    rng = np.random.default_rng(0)
    b1 = rng.normal(0.0, 1.0, (75, 3))
    b2 = rng.normal(20.0, 1.0, (75, 3))
    outlier = np.array([[10.0, 10.0, 10.0]])           # between the two clusters
    X = np.vstack([b1, b2, outlier])
    subj = np.array([f"s{i}" for i in range(X.shape[0])])
    arrs = [X[:, 0], X[:, 1], X[:, 2]]
    rows = _cluster_triple("PRONET", "baseline", ("va", "vb", "vc"), arrs, subj, DBSCAN)
    flagged = {r["subjectid"] for r in rows}
    assert subj[-1] in flagged, "the point between the two clusters should be flagged"
    assert all(r["n_clusters"] == 2 for r in rows), "should have found exactly two clusters"


@pytest.mark.skipif(not _HAS_SKLEARN, reason="cluster_3d needs scikit-learn")
def test_cluster3d_canonicalizes_leg_order_detection_neutral():
    # The emitted leg order must be canonical (legs sorted), regardless of the
    # axis/column order passed in -- otherwise the same trio reads "A | B | C"
    # in one network and "B | A | C" in the other (each sorts its numerics by
    # its own completeness), duplicating the combo in combined_ranked. The sort
    # must be detection-neutral (same points flagged) and keep each value paired
    # with its own variable name in observed_value.
    import numpy as np
    from sklearn.cluster import DBSCAN
    from simple_anomaly_detection.cluster_3d import _cluster_triple
    rng = np.random.default_rng(0)
    b1 = rng.normal(0.0, 1.0, (75, 3))
    b2 = rng.normal(20.0, 1.0, (75, 3))
    X = np.vstack([b1, b2, np.array([[10.0, 10.0, 10.0]])])
    subj = np.array([f"s{i}" for i in range(X.shape[0])])
    # Same data, two different incoming column orders (as two networks would).
    fwd = _cluster_triple("PRONET", "baseline", ("weight", "bmi", "age"),
                          [X[:, 0], X[:, 1], X[:, 2]], subj, DBSCAN)
    perm = _cluster_triple("PRESCIENT", "baseline", ("age", "weight", "bmi"),
                           [X[:, 2], X[:, 0], X[:, 1]], subj, DBSCAN)
    assert fwd and perm
    # Legs come out sorted, identical across the two orderings.
    assert fwd[0]["variable"] == "age | bmi | weight"
    assert perm[0]["variable"] == "age | bmi | weight"
    # Detection-neutral: the same subject(s) are flagged either way.
    assert {r["subjectid"] for r in fwd} == {r["subjectid"] for r in perm}
    # value<->name pairing survives the reorder (and matches across orderings).
    assert fwd[0]["observed_value"] == perm[0]["observed_value"]
    assert fwd[0]["observed_value"].startswith("age=")


@pytest.mark.skipif(not _HAS_SKLEARN, reason="cluster_3d needs scikit-learn")
def test_cluster3d_dedup_collapses_triples_per_subject():
    from simple_anomaly_detection.cluster_3d import _dedup
    def row(sev, tri):
        return {"subjectid": "X", "network": "PRONET", "anomaly_type": "cluster_3d_outlier",
                "severity_score": sev, "raw_score": sev / 10.0, "variable": tri,
                "variables_involved": tri.replace(" | ", ", ")}
    out = _dedup([row(60, "a | b | c"), row(90, "a | b | d"), row(55, "a | c | d")])
    assert len(out) == 1
    assert out[0]["corroborating_triples"] == 3
    assert float(out[0]["severity_score"]) == 90


# --- triple-selection diversification: a dominant correlated block must not
#     consume the whole TOP_K budget and crowd out smaller blocks (pure logic,
#     no sklearn). ---

def test_diversify_triples_admits_small_block_and_preserves_coverage():
    from simple_anomaly_detection.cluster_3d import _diversify_triples
    # Big block: variables 0..11 (C(12,3)=220 triples), all at high corr 0.9.
    # Small block: variables 100,101,102 -> one triple, weaker corr 0.5, so in a
    # raw top-K sort it ranks dead last and (with a tight budget) never appears.
    import itertools
    big = [(0.9, i, j, k) for i, j, k in itertools.combinations(range(12), 3)]
    small = [(0.5, 100, 101, 102)]
    triples = sorted(big + small, reverse=True)
    top_k = 50

    raw = triples[:top_k]
    assert (0.5, 100, 101, 102) not in raw          # crowded out by the big block

    div = _diversify_triples(triples, top_k, max_per_var=15)
    assert len(div) == top_k                         # pass 2 refills -> no coverage loss
    assert (0.5, 100, 101, 102) in div               # small block now examined
    # No variable exceeds the cap among the diversify-pass picks is not asserted
    # directly (pass 2 ignores the cap by design); the contract is "small block
    # admitted AND budget fully used", which the two asserts above pin down.

    # When candidates already fit the budget the cap can't bite (the demo case).
    few = [(0.9, 0, 1, 2), (0.8, 1, 2, 3)]
    assert _diversify_triples(few, top_k, max_per_var=1) == few


# --- combos_for_combined: per-subject findings -> Finding-schema combo rows
#     for the combined headline. ---

def test_combos_for_combined_shapes_one_row_per_combo():
    from simple_anomaly_detection.cluster_3d import combos_for_combined

    def row(sev, var, subj, net="PRONET", tp="baseline"):
        return {"subjectid": subj, "network": net, "timepoint": tp,
                "anomaly_type": "cluster_3d_outlier", "severity_score": sev,
                "variable": var, "variables_involved": var.replace(" | ", ", ")}

    rows = [row(90.0, "a | b | c", "S1"), row(95.0, "a | b | c", "S2"),
            row(60.0, "x | y | z", "S3")]
    out = combos_for_combined(rows, "cluster_3d_outlier")
    by_combo = {r["variable"]: r for r in out}
    assert set(by_combo) == {"a | b | c", "x | y | z"}
    abc = by_combo["a | b | c"]
    # severity = the combo's MAX per-subject severity (so it ranks comparably).
    assert float(abc["severity_score"]) == 95.0
    assert abc["anomaly_type"] == "cluster_3d_outlier"
    assert abc["n_subjects"] == 2
    assert abc["subjectid"] == "(2 subjects)"        # count, not a single subject
    # both flagged subjects are traceable via top_subjects (the oracle's hook).
    assert "S2" in abc["top_subjects"] and "S1" in abc["top_subjects"]
    # singular noun for a one-subject combo.
    assert by_combo["x | y | z"]["subjectid"] == "(1 subject)"


# --- summarize_combos: the per-combo summary index. Pure dict logic (no
#     sklearn), and the demo's single correlated block can't exercise the
#     multi-combo grouping/ranking, so build the rows by hand. ---

def test_summarize_combos_groups_ranks_and_lists_subjects():
    from simple_anomaly_detection.cluster_3d import summarize_combos

    def row(sev, var, subj, net="PRONET", tp="baseline"):
        return {"subjectid": subj, "network": net, "timepoint": tp,
                "anomaly_type": "cluster_3d_outlier", "severity_score": sev,
                "variable": var, "variables_involved": var.replace(" | ", ", ")}

    rows = [
        row(90.0, "a | b | c", "S1"),
        row(70.0, "c | a | b", "S2"),                 # same combo, legs reordered
        row(95.0, "a | b | c", "S3", tp="month1"),    # + a second timepoint
        row(60.0, "x | y | z", "S4"),
        row(50.0, "a | b | c", "P1", net="PRESCIENT"),  # same combo, other network
    ]
    summary = summarize_combos(rows)

    # The 3-subject PRONET combo outranks the 1-subject combos (n_subjects desc).
    top = summary[0]
    assert top["network"] == "PRONET"
    assert top["variables"] == "a | b | c"            # legs normalized + sorted
    assert top["n_subjects"] == 3
    assert top["max_severity"] == 95.0
    # subjects listed severity-descending, as "subjectid(sev)".
    assert top["top_subjects"] == "S3(95.0); S1(90.0); S2(70.0)"
    assert set(top["timepoints"].split(", ")) == {"baseline", "month1"}

    # Network is part of the grouping key: the PRESCIENT "a | b | c" is its own row.
    pres = [s for s in summary if s["network"] == "PRESCIENT"]
    assert len(pres) == 1 and pres[0]["n_subjects"] == 1


def test_summarize_combos_counts_and_ranks_by_high_severity():
    # The headline "n_subjects_sev_ge_<thr>" column counts how often a combo
    # shows up AT OR ABOVE the severity bar, and that count is the primary rank
    # key -- so a combo that recurs at high severity outranks one that recurs
    # only weakly, even with fewer total subjects.
    from simple_anomaly_detection.cluster_3d import summarize_combos

    def row(sev, var, subj, net="PRONET", tp="baseline"):
        return {"subjectid": subj, "network": net, "timepoint": tp,
                "severity_score": sev, "variable": var}

    rows = [
        # "a | b | c": 2 subjects, both >= 80.
        row(95.0, "a | b | c", "S1"),
        row(85.0, "a | b | c", "S2"),
        # "x | y | z": 3 subjects but all below 80.
        row(70.0, "x | y | z", "S3"),
        row(65.0, "x | y | z", "S4"),
        row(60.0, "x | y | z", "S5"),
    ]
    summary = summarize_combos(rows, sev_threshold=80.0)
    col = "n_subjects_sev_ge_80"

    by_combo = {s["variables"]: s for s in summary}
    assert by_combo["a | b | c"][col] == 2     # both subjects clear the bar
    assert by_combo["x | y | z"][col] == 0     # none clear it
    # 80.0 itself counts (>= is inclusive): a lone boundary finding -> 1.
    boundary = summarize_combos([row(80.0, "p | q | r", "B")], sev_threshold=80.0)
    assert boundary[0][col] == 1
    # High-severity recurrence ranks first despite x|y|z having more subjects.
    assert summary[0]["variables"] == "a | b | c"
    # Threshold is configurable and surfaced in the column name.
    relaxed = summarize_combos(rows, sev_threshold=60.0)
    assert relaxed[0]["variables"] == "x | y | z"   # now 3 subjects clear 60
    assert relaxed[0]["n_subjects_sev_ge_60"] == 3


def test_diversify_combined_caps_cluster_combos_not_other_detectors():
    # The combined headline must not be dominated by one repeating cluster
    # combo: cap it per (network, combo). Distinct-subject rows from other
    # detectors that merely share a variable label must pass through untouched.
    import pandas as pd
    from simple_anomaly_detection.runner import (
        SimpleAnomalyDetector, _COMBO_ANOMALY_TYPES)

    ct = _COMBO_ANOMALY_TYPES[0]  # "cluster_3d_outlier"
    rows = []
    for i, sev in enumerate([99.0, 98.0, 97.0, 96.0, 95.0, 94.0]):
        rows.append({"anomaly_type": ct, "network": "PRONET",
                     "variable": "a | b | c", "severity_score": sev,
                     "subjectid": f"S{i}"})
    # whole_subject rows share a variable label but are distinct subjects.
    for i, sev in enumerate([80.0, 79.0, 78.0, 77.0, 76.0, 75.0]):
        rows.append({"anomaly_type": "whole_subject_outlier", "network": "PRONET",
                     "variable": "(subject composite)", "severity_score": sev,
                     "subjectid": f"W{i}"})
    combined = pd.DataFrame(rows)

    out = SimpleAnomalyDetector._diversify_combined(combined, 3, _COMBO_ANOMALY_TYPES)
    cl = out[out["anomaly_type"] == ct]
    assert len(cl) == 3                                  # capped to top-3 by severity
    assert set(cl["subjectid"]) == {"S0", "S1", "S2"}    # the 3 most severe kept
    ws = out[out["anomaly_type"] == "whole_subject_outlier"]
    assert len(ws) == 6                                  # other detector untouched

    # A second combo (same vars, different network) is its own group.
    rows.append({"anomaly_type": ct, "network": "PRESCIENT",
                 "variable": "a | b | c", "severity_score": 90.0, "subjectid": "P0"})
    out2 = SimpleAnomalyDetector._diversify_combined(
        pd.DataFrame(rows), 3, _COMBO_ANOMALY_TYPES)
    assert len(out2[out2["network"] == "PRESCIENT"]) == 1

    # cap <= 0 disables; the frame is returned unchanged.
    assert len(SimpleAnomalyDetector._diversify_combined(
        combined, 0, _COMBO_ANOMALY_TYPES)) == len(combined)


def test_diversify_combined_caps_correlative_dominant_variable():
    # correlative is now diversified too: after _dedup_correlative, one bad
    # value re-flagged for many subjects collapses to rows that share a single
    # dominant-variable label, which would otherwise flood the headline. Cap it
    # per (network, variable) like the cluster combos -- but leave distinct-
    # subject rows from non-diversified detectors untouched.
    import pandas as pd
    from simple_anomaly_detection.runner import (
        SimpleAnomalyDetector, _COMBO_ANOMALY_TYPES)
    from simple_anomaly_detection.correlative import ANOMALY_TYPE as CORR

    assert CORR in _COMBO_ANOMALY_TYPES
    rows = []
    # "weight" flagged for 6 distinct subjects (one recurring dominant variable).
    for i, sev in enumerate([99.0, 98.0, 97.0, 96.0, 95.0, 94.0]):
        rows.append({"anomaly_type": CORR, "network": "PRONET",
                     "variable": "weight", "severity_score": sev,
                     "subjectid": f"S{i}"})
    # standard_outlier rows share a variable label but are distinct subjects and
    # not diversified -- they must all survive.
    for i, sev in enumerate([80.0, 79.0, 78.0]):
        rows.append({"anomaly_type": "standard_outlier", "network": "PRONET",
                     "variable": "height", "severity_score": sev,
                     "subjectid": f"O{i}"})
    combined = pd.DataFrame(rows)

    out = SimpleAnomalyDetector._diversify_combined(combined, 3, _COMBO_ANOMALY_TYPES)
    corr = out[out["anomaly_type"] == CORR]
    assert len(corr) == 3                                 # capped to top-3 by severity
    assert set(corr["subjectid"]) == {"S0", "S1", "S2"}   # the 3 most severe kept
    so = out[out["anomaly_type"] == "standard_outlier"]
    assert len(so) == 3                                   # other detector untouched


# --- site_network: collapse same-variable-across-sites into one row ---

def _sn_row(site, variable, method, sev, obs="x"):
    return {"anomaly_type": "site_network_outlier", "severity_score": sev,
            "raw_score": sev / 10.0, "network": "PRONET", "timepoint": "",
            "site_id": site, "subjectid": "(site-level)", "variable": variable,
            "variables_involved": "", "observed_value": obs, "expected_value": "e",
            "explanation": "ex", "method": method}


def test_collapse_multisite_folds_sites_per_variable_and_drift_type():
    from simple_anomaly_detection.site_network import _collapse_multisite
    MED = "cross-site MAD on site medians"
    MISS = "cross-site MAD on site missingness"
    rows = [
        # weight median-shift at 3 sites -> 1 row.
        _sn_row("AA", "weight", MED, 78.0, "site median 78"),
        _sn_row("BB", "weight", MED, 70.0, "site median 70"),
        _sn_row("CC", "weight", MED, 60.0, "site median 60"),
        # weight missingness-shift at 2 sites -> a SEPARATE row (different test).
        _sn_row("AA", "weight", MISS, 61.0),
        _sn_row("FF", "weight", MISS, 55.0),
        # height median-shift at a single site -> passes through as one row.
        _sn_row("DD", "height", MED, 40.0),
    ]
    out = _collapse_multisite(rows)
    # weight: one median row + one missingness row; height: one row -> 3 total.
    assert len(out) == 3
    weight_med = [r for r in out if r["variable"] == "weight" and r["method"] == MED]
    assert len(weight_med) == 1
    r = weight_med[0]
    assert r["n_sites"] == 3
    assert r["site_id"] == "(3 sites)"
    assert set(r["sites"].split(", ")) == {"AA", "BB", "CC"}
    assert float(r["severity_score"]) == 78.0          # strongest site kept
    assert "AA: site median 78" in r["observed_value"]  # per-site values listed
    weight_miss = [r for r in out if r["variable"] == "weight" and r["method"] == MISS]
    assert len(weight_miss) == 1 and weight_miss[0]["n_sites"] == 2
    height = [r for r in out if r["variable"] == "height"]
    assert len(height) == 1 and height[0]["n_sites"] == 1
    assert height[0]["site_id"] == "DD"                # single site not renamed


def test_collapse_multisite_passes_network_level_rows_through():
    from simple_anomaly_detection.site_network import _collapse_multisite
    nl = {"anomaly_type": "site_network_outlier", "severity_score": 50.0,
          "raw_score": 5.0, "network": "PRONET", "timepoint": "",
          "site_id": "(network-level)", "subjectid": "(network-level)",
          "variable": "weight", "variables_involved": "", "observed_value": "o",
          "expected_value": "e", "explanation": "ex", "method": "cross-network median comparison"}
    out = _collapse_multisite([nl])
    assert len(out) == 1 and out[0]["site_id"] == "(network-level)"


def test_suppress_redundant_outliers_expands_collapsed_sites():
    # After collapse a site row's site_id is "(N sites)" with the concrete sites
    # in a `sites` column. Suppression must expand that list so it still matches
    # standard_outlier rows' concrete site_id.
    import pandas as pd
    from simple_anomaly_detection.runner import SimpleAnomalyDetector
    site_df = pd.DataFrame([{
        "anomaly_type": "site_network_outlier", "severity_score": 80.0,
        "network": "PRONET", "site_id": "(2 sites)", "variable": "weight",
        "sites": "AA, BB"}])
    std_df = pd.DataFrame([
        # AA weight at sev 70 (<= 80) -> suppressed by the collapsed site row.
        {"anomaly_type": "standard_outlier", "severity_score": 70.0,
         "network": "PRONET", "site_id": "AA", "variable": "weight", "subjectid": "S1"},
        # BB weight at sev 95 (> 80) -> kept (more extreme than the site shift).
        {"anomaly_type": "standard_outlier", "severity_score": 95.0,
         "network": "PRONET", "site_id": "BB", "variable": "weight", "subjectid": "S2"},
        # CC not in the collapsed list -> untouched.
        {"anomaly_type": "standard_outlier", "severity_score": 50.0,
         "network": "PRONET", "site_id": "CC", "variable": "weight", "subjectid": "S3"},
    ])
    out = SimpleAnomalyDetector._suppress_redundant_outliers(std_df, site_df)
    kept = set(out["subjectid"])
    assert kept == {"S2", "S3"}, kept   # S1 suppressed; S2 (more extreme) & S3 kept


def test_collapse_multisite_records_aligned_per_site_severities():
    # sites and site_severities must be parallel and in the SAME (severity-desc)
    # order, so suppression can cap each site by its own drift severity.
    from simple_anomaly_detection.site_network import _collapse_multisite
    MED = "cross-site MAD on site medians"
    rows = [_sn_row("AB", "weight", MED, 80.0), _sn_row("CD", "weight", MED, 20.0)]
    out = _collapse_multisite(rows)
    assert len(out) == 1
    r = out[0]
    assert r["sites"] == "AB, CD"                  # severity-descending
    assert r["site_severities"] == "80, 20"        # aligned, same order
    assert r["site_id"] == "(2 sites)" and r["n_sites"] == 2


def test_suppress_uses_per_site_severity_not_group_max():
    # A variable drifts strongly at AB (80) and weakly at CD (20); collapsed to
    # one row with group-max severity 80. A standard_outlier at CD with sev 50 is
    # MORE extreme than CD's actual 20-point drift and must be KEPT -- using the
    # group max (80) would wrongly suppress it.
    import pandas as pd
    from simple_anomaly_detection.runner import SimpleAnomalyDetector
    site_df = pd.DataFrame([{
        "anomaly_type": "site_network_outlier", "severity_score": 80.0,
        "network": "PRONET", "site_id": "(2 sites)", "variable": "weight",
        "sites": "AB, CD", "site_severities": "80, 20"}])
    std_df = pd.DataFrame([
        {"anomaly_type": "standard_outlier", "severity_score": 50.0,
         "network": "PRONET", "site_id": "CD", "variable": "weight", "subjectid": "C1"},
        {"anomaly_type": "standard_outlier", "severity_score": 50.0,
         "network": "PRONET", "site_id": "AB", "variable": "weight", "subjectid": "A1"},
    ])
    out = SimpleAnomalyDetector._suppress_redundant_outliers(std_df, site_df)
    kept = set(out["subjectid"])
    # C1 kept (50 > CD's 20); A1 suppressed (50 <= AB's 80).
    assert kept == {"C1"}, kept


def test_diversify_exempts_binary_binary_correlative_cells():
    # BB joint-cell rows (variable "a=x & b=y") are distinct subjects in one rare
    # cell, NOT one variable flooding the list. They must NOT be capped.
    import pandas as pd
    from simple_anomaly_detection.runner import (
        SimpleAnomalyDetector, _COMBO_ANOMALY_TYPES)
    from simple_anomaly_detection.correlative import ANOMALY_TYPE as CORR
    rows = []
    # 6 distinct subjects share one rare cell (identical variable label + sev).
    for i in range(6):
        rows.append({"anomaly_type": CORR, "network": "PRONET",
                     "variable": "sex=2 & rare_flag=1", "severity_score": 70.0,
                     "subjectid": f"B{i}"})
    # ...while a leg-collapsed dominant variable IS still capped.
    for i, sev in enumerate([99.0, 98.0, 97.0, 96.0, 95.0]):
        rows.append({"anomaly_type": CORR, "network": "PRONET",
                     "variable": "weight", "severity_score": sev,
                     "subjectid": f"W{i}"})
    out = SimpleAnomalyDetector._diversify_combined(
        pd.DataFrame(rows), 3, _COMBO_ANOMALY_TYPES)
    bb = out[out["variable"] == "sex=2 & rare_flag=1"]
    assert len(bb) == 6                              # BB cells untouched
    wt = out[out["variable"] == "weight"]
    assert len(wt) == 3                              # dominant variable still capped


def test_summarize_combos_caps_subject_list_with_overflow_note():
    from simple_anomaly_detection.cluster_3d import summarize_combos
    rows = [{"subjectid": f"M{i}", "network": "PRONET", "timepoint": "baseline",
             "variable": "p | q | r", "severity_score": 50.0 + i} for i in range(5)]
    summary = summarize_combos(rows, top_k=2)
    assert len(summary) == 1
    assert summary[0]["n_subjects"] == 5
    assert summary[0]["top_subjects"].endswith("(+3 more)")
    # only the 2 highest-severity subjects are listed before the overflow note.
    assert summary[0]["top_subjects"].startswith("M4(54.0); M3(53.0); (+3 more)")


def test_summarize_combos_tolerates_nan_and_missing_severity():
    # NaN is truthy, so a naive `nan or 0` would slip a NaN into mean/sort/
    # top_subjects; _sev must coerce non-finite/blank severities to 0.0.
    import math
    from simple_anomaly_detection.cluster_3d import summarize_combos
    rows = [
        {"subjectid": "G", "network": "PRONET", "variable": "a | b | c", "severity_score": 80.0},
        {"subjectid": "N", "network": "PRONET", "variable": "a | b | c", "severity_score": float("nan")},
        {"subjectid": "M", "network": "PRONET", "variable": "a | b | c"},  # missing severity
    ]
    summary = summarize_combos(rows)
    assert len(summary) == 1
    s = summary[0]
    assert s["n_subjects"] == 3
    assert math.isfinite(s["mean_severity"]) and math.isfinite(s["max_severity"])
    assert s["max_severity"] == 80.0
    assert "(nan)" not in s["top_subjects"]            # no NaN leaks into the listing
    assert s["top_subjects"].startswith("G(80.0)")     # the real finding ranks first


# --- date_anomaly: cross-timepoint pairing + missing-code exclusion ---

def _date_slices(baseline_col, month1_col, base_iso, gaps, n=45, network="PRONET"):
    """Two single-date-column slices with n subjects; month1 date = base + gap.
    Returns (per_slice, classify, subjectids)."""
    import pandas as pd
    from datetime import date, timedelta
    base = date.fromisoformat(base_iso)
    subj = [f"DT{i:05d}" for i in range(n)]
    bl_vals = [base.isoformat()] * n
    m1_vals = [(base + timedelta(days=int(gaps[i]))).isoformat() for i in range(n)]
    bl = pd.DataFrame({"subjectid": subj, baseline_col: bl_vals})
    m1 = pd.DataFrame({"subjectid": subj, month1_col: m1_vals})
    per_slice = {(network, "baseline"): bl, (network, "month1"): m1}
    classify = {(network, "baseline"): {"date": [baseline_col]},
                (network, "month1"): {"date": [month1_col]}}
    return per_slice, classify, subj, bl, m1


def test_date_anomaly_pairs_different_name_across_timepoints():
    # The cross-timepoint pairing now includes DIFFERENT-named date columns
    # (baseline consent_date vs month_1 visit_date), not just the same column
    # across visits. Seed one subject with an off cross-pair gap and confirm
    # that exact (different-name, different-tp) pair is flagged -- the old
    # same-name-only pairing would never have produced this finding.
    from simple_anomaly_detection import date_anomaly
    gaps = [40 + (i % 5) - 2 for i in range(45)]   # ~40d, small noise so MAD > 0
    gaps[7] = 405                                  # one subject a year late
    per_slice, classify, subj, _, _ = _date_slices(
        "consent_date", "visit_date", "2026-01-10", gaps)

    findings = date_anomaly.detect_all(per_slice, classify=classify)
    hits = [f for f in findings
            if f.get("subjectid") == "DT00007"
            and "consent_date" in str(f.get("variable", ""))
            and "visit_date" in str(f.get("variable", ""))]
    assert hits, f"expected a cross-name cross-tp flag; got {findings}"


def test_date_anomaly_excludes_missing_code_dates():
    # Missing codes must never reach the gap math. Three subjects carry a date
    # SENTINEL in the baseline date -- if it leaked through it would parse as
    # year 1909 and fabricate a ~117-year gap (a screaming false outlier). With
    # the real gaps all ~30d (inside the 14-day floor), a correct run flags
    # nothing.
    from simple_anomaly_detection import date_anomaly
    gaps = [30 + (i % 5) - 2 for i in range(45)]   # 28..32d, no real outlier
    per_slice, classify, subj, bl, m1 = _date_slices(
        "visit_date", "visit_date", "2026-01-10", gaps)
    for i in (0, 1, 2):
        bl.loc[i, "visit_date"] = "1909-09-09"     # date sentinel / missing code
    bl.loc[3, "visit_date"] = "-9"                 # numeric missing code
    bl.loc[4, "visit_date"] = ""                   # blank

    findings = date_anomaly.detect_all(per_slice, classify=classify)
    assert findings == [], findings                          # no fabricated gap
    assert not any("1909" in str(f.get("observed_value", "")) for f in findings)


def test_clean_dates_strips_noncanonical_sentinel_spellings():
    # The string strip only catches the canonical '1909-09-09'. A datetime-
    # suffixed, 'T'-separated, slash, or unpadded spelling of the SAME sentinel
    # date must also be nulled (post-parse, by calendar date) or it parses to a
    # real year-1909 visit and fabricates a ~117-year gap.
    import pandas as pd
    from simple_anomaly_detection.common import clean_dates_series
    spellings = ["1909-09-09", "1909-09-09 00:00:00", "1909-09-09T00:00:00",
                 "09/09/1909", "1909-9-9", "1903-03-03 12:00:00"]
    out = clean_dates_series(pd.Series(spellings))
    assert out.isna().all(), out.tolist()
    # A genuine datetime on a non-sentinel day is preserved (not over-nulled).
    real = clean_dates_series(pd.Series(["2026-01-10 09:30:00"]))
    assert real.notna().all()


# --- rater_effect: the per-value over-use screen, gates, scoping, and the
#     two-tier (within-site arbiter vs single-rater cohort) behaviour the demo
#     alone can't fully exercise. ---

def _ct(rows):
    """Build a rater x value count table from {rater: {value: count}}."""
    import pandas as pd
    return pd.DataFrame(rows).T.fillna(0)


def test_rater_screen_flags_clear_overuser():
    from simple_anomaly_detection.rater_effect import _screen
    ct = _ct({"R1": {"hi": 20, "lo": 5}, "R2": {"hi": 2, "lo": 23}})
    hits = {(h["rater"], h["value"]): h for h in _screen(ct, z_gate=3.0)}
    # R1 over-uses "hi"; the screen is symmetric, so it also surfaces R2
    # over-using the complementary "lo" -- both are true, opposite-direction
    # rater divergences.
    assert ("R1", "hi") in hits and ("R2", "lo") in hits
    h = hits[("R1", "hi")]
    assert abs(h["p_rater"] - 0.8) < 1e-9 and h["z"] > 3.0


def test_rater_screen_effect_size_gate_blocks_big_n_tiny_diff():
    # A 2-point rate gap that is statistically significant only because n is
    # huge (z>=3) must NOT flag: the rate-difference gate is what stops
    # big-n significance from drowning the report.
    from simple_anomaly_detection.rater_effect import _screen
    ct = _ct({"R1": {"hi": 6120, "lo": 5880}, "R2": {"hi": 5880, "lo": 6120}})
    assert _screen(ct, z_gate=3.0) == []


def test_rater_screen_rate_gate_requires_consistent_use():
    # The rater's own rate of the value must clear MIN_RATER_RATE (0.30): a
    # value used <30% of the time is not "consistent" scoring even if it's far
    # above the comparison group. A 4-value spread keeps every other cell below
    # the diff gate, so the "v" cell is what flips on the rate gate alone.
    from simple_anomaly_detection.rater_effect import _screen
    blocked = _ct({"R1": {"v": 7, "a": 7, "b": 7, "c": 7},     # v-rate 0.25
                   "R2": {"v": 0, "a": 10, "b": 9, "c": 9}})
    assert _screen(blocked, z_gate=3.0) == []                  # v at 0.25 -> blocked
    # Bump only v so its rate clears 0.30, holding the comparison at 0 -> flags.
    flips = _ct({"R1": {"v": 12, "a": 6, "b": 5, "c": 5},      # v-rate 0.43
                 "R2": {"v": 0, "a": 10, "b": 9, "c": 9}})
    hits = _screen(flips, z_gate=3.0)
    assert any(h["rater"] == "R1" and h["value"] == "v" for h in hits)


def test_is_rater_variable_excludes_subject_record_id():
    from simple_anomaly_detection.rater_effect import is_rater_variable
    assert is_rater_variable("chrpanss_interview_ampscz_id")
    assert is_rater_variable("foo_redcap_id")
    assert not is_rater_variable("chric_ampscz_id")          # subject Record ID
    assert not is_rater_variable("chric_reconsent_ampscz_id")
    assert not is_rater_variable("subjectid")
    assert not is_rater_variable("chrpanss_p1")


def test_valid_rater_col_cardinality_guard():
    import pandas as pd
    from simple_anomaly_detection.rater_effect import _valid_rater_col
    # A per-subject id column (value == subjectid, one subject each) is NOT a
    # rater field.
    subj = pd.Series([f"S{i:05d}" for i in range(200)])
    assert not _valid_rater_col(subj.copy(), subj)
    # Two raters each covering >= MIN_SUBJECTS_PER_RATER distinct subjects -> valid.
    raters = pd.Series(["RA"] * 30 + ["RB"] * 30)
    subj2 = pd.Series([f"S{i:05d}" for i in range(60)])
    assert _valid_rater_col(raters, subj2)
    # Pooling stress (the bug the subject-aware guard fixes): a subject id
    # repeated across 12 timepoints still fails -- one distinct subject per
    # value, no matter how many forms are stacked.
    pooled = pd.Series([f"S{i:05d}" for i in range(50)] * 12)
    assert not _valid_rater_col(pooled.copy(), pooled)


def test_scored_vars_scoping_form_then_prefix():
    from simple_anomaly_detection.rater_effect import _scored_vars_for_rater
    # Form-name scoping when the map is available.
    v2f = {"r_ampscz_id": "formA", "a1": "formA", "a2": "formA", "b1": "formB"}
    assert _scored_vars_for_rater("r_ampscz_id", ["a1", "a2", "b1"], v2f) == ["a1", "a2"]
    # Prefix scoping when no form map: only same-prefix items.
    got = _scored_vars_for_rater(
        "formx_rater_ampscz_id", ["formx_p1", "formx_p2", "other_v"], {})
    assert got == ["formx_p1", "formx_p2"]


def _rater_slice():
    """One PRONET/baseline slice with three sites exercising both tiers:
      AA: two raters BOTH scoring '3' high -> a SITE effect; within-site
          contrast is ~0, so NEITHER rater should flag.
      BB: BB1 scores '4' high, co-located BB2 does not -> within-site flag.
      CC: single rater CC1 scores '5' high -> cohort / site-confounded flag.
    """
    import pandas as pd
    rows = []

    def add(site, rater, values):
        for k, v in enumerate(values):
            rows.append({"subjectid": f"{site}{len(rows):05d}",
                         "formx_rater_ampscz_id": rater,
                         "formx_item": str(v)})

    add("AA", "AA1", [3] * 21 + [1, 2, 4] * 3)     # rate(3)=0.70
    add("AA", "AA2", [3] * 21 + [1, 2, 4] * 3)     # rate(3)=0.70 (same -> site effect)
    add("BB", "BB1", [4] * 27 + [1, 2, 3])         # rate(4)=0.90
    add("BB", "BB2", [1, 2, 3, 4] * 7 + [1, 2])    # ~uniform
    add("CC", "CC1", [5] * 36 + [1, 2, 3, 4])      # rate(5)=0.90, single rater
    df = pd.DataFrame(rows)
    return {("PRONET", "baseline"): df}


def test_rater_effect_two_tier_behaviour():
    from simple_anomaly_detection import rater_effect
    findings = rater_effect.detect_all(_rater_slice())
    by_rater = {}
    for f in findings:
        by_rater.setdefault(f["rater_id"], []).append(f)

    # Within-site flag for BB1 on value 4, labelled high-confidence.
    bb1 = [f for f in by_rater.get("BB1", []) if f["scored_value"] == "4"]
    assert bb1, f"expected BB1 within-site flag; got {findings}"
    assert bb1[0]["confidence"] == "within-site"
    assert bb1[0]["site_id"] == "BB"

    # Single-rater-site flag for CC1 on value 5, labelled site-confounded.
    cc1 = [f for f in by_rater.get("CC1", []) if f["scored_value"] == "5"]
    assert cc1 and cc1[0]["confidence"] == "site-confounded"

    # The site-wide AA effect (both raters high on '3') must NOT flag either
    # AA rater: the within-site contrast is the arbiter and it is ~0 there.
    assert "AA1" not in by_rater and "AA2" not in by_rater, (
        f"site effect wrongly attributed to a rater: {by_rater.keys()}")


def test_rater_effect_no_pseudoreplication_single_subject():
    # A single subject a rater always scores '9' across many visits is a
    # SUBJECT trait, not a scoring tendency. Per-form counting would see 12
    # nines and flag; the per-(rater, subject) collapse counts it as ONE
    # subject, which fails MIN_VALUE_SUBJECTS -> no finding.
    import pandas as pd
    from simple_anomaly_detection import rater_effect
    base = []
    # Both raters clear MIN_SUBJECTS_PER_RATER so they QUALIFY -- the point is
    # that the per-subject collapse (not the size gate) is what stops the single
    # repeated subject from flagging.
    for i in range(24):   # Z1's ordinary subjects (no 9s)
        base.append({"subjectid": f"ZZ1{i:04d}", "formx_rater_ampscz_id": "Z1",
                     "formx_item": str(1 + (i % 4))})
    for i in range(25):   # co-located Z2's subjects
        base.append({"subjectid": f"ZZ2{i:04d}", "formx_rater_ampscz_id": "Z2",
                     "formx_item": str(1 + (i % 4))})
    base.append({"subjectid": "ZZ_special", "formx_rater_ampscz_id": "Z1",
                 "formx_item": "9"})
    per = {("PRONET", "baseline"): pd.DataFrame(base)}
    for k in range(1, 12):  # the same subject, scored 9 again every visit
        per[("PRONET", f"month{k}")] = pd.DataFrame(
            [{"subjectid": "ZZ_special", "formx_rater_ampscz_id": "Z1",
              "formx_item": "9"}])
    findings = rater_effect.detect_all(per)
    nines = [f for f in findings
             if f["rater_id"] == "Z1" and f["scored_value"] == "9"]
    assert not nines, f"a single repeated subject must not drive a finding: {nines}"


def test_rater_effect_within_site_excludes_off_site_subjects():
    # A home rater's OFF-site subjects must not leak into its home-site
    # contrast. BB1's home is BB (30 BB subjects, spread like co-located BB2),
    # but it also rates 22 AA subjects all '4'. Before the site-match fix that
    # off-site '4' signal inflated BB1's BB within-site row and produced a
    # confident false positive; after it, value 4 is absent at BB -> no flag.
    import pandas as pd
    from simple_anomaly_detection import rater_effect
    rows = []
    for i in range(30):  # BB1 home subjects at BB, unremarkable (1..3)
        rows.append({"subjectid": f"BB1H{i:04d}", "formx_rater_ampscz_id": "BB1",
                     "formx_item": str(1 + (i % 3))})
    for i in range(22):  # BB1 OFF-site subjects at AA, all '4'
        rows.append({"subjectid": f"AA9{i:04d}", "formx_rater_ampscz_id": "BB1",
                     "formx_item": "4"})
    for i in range(30):  # BB2 co-located at BB, same unremarkable spread
        rows.append({"subjectid": f"BB2H{i:04d}", "formx_rater_ampscz_id": "BB2",
                     "formx_item": str(1 + (i % 3))})
    findings = rater_effect.detect_all({("PRONET", "baseline"): pd.DataFrame(rows)})
    leaked = [f for f in findings
              if f["rater_id"] == "BB1" and f["scored_value"] == "4"]
    assert not leaked, f"off-site subjects leaked into the within-site contrast: {leaked}"


def test_rater_effect_canonicalizes_numeric_value_spellings():
    # '4' at one timepoint and '4.0' at another are the SAME value and must not
    # fragment the count (which would dilute the rate below the gates).
    from simple_anomaly_detection.rater_effect import _canonicalize_values
    import pandas as pd
    out = list(_canonicalize_values(pd.Series(["4", "4.0", " 4", "4.5", "abc"])))
    assert out == ["4", "4", "4", "4.5", "abc"]


def test_rater_diagnostic_surfaces_seeds_on_demo(tmp_path):
    # The ungated diagnostic should rank the two seeded rater effects at the top
    # (both have a per-subject rate diff of 1.0), under the correct tiers.
    import os
    from simple_anomaly_detection.demo import build_demo_dataset
    from simple_anomaly_detection.runner import SimpleAnomalyDetector
    from simple_anomaly_detection.rater_effect_diagnostic import diagnose, summarize
    indir = os.path.join(str(tmp_path), "in")
    outdir = os.path.join(str(tmp_path), "out")
    build_demo_dataset(indir)
    per = SimpleAnomalyDetector(input_path=indir, output_path=outdir)._load_slices()
    df, _ = diagnose(per)
    assert not df.empty
    top = set(zip(df.head(2)["variable"], df.head(2)["scored_value"], df.head(2)["tier"]))
    assert ("panss_p2", "5", "cohort") in top
    assert ("panss_p1", "7", "within_site") in top
    assert isinstance(summarize(df), str) and "candidates:" in summarize(df)


def test_rater_effect_site_confounded_severity_is_lower():
    # A site-confounded (cohort) finding is dialled down relative to a
    # within-site one of comparable raw z, so the headline favours the
    # confound-controlled findings.
    from simple_anomaly_detection.rater_effect import (
        _make_finding, SITE_CONFOUNDED_SEVERITY_FACTOR)
    d = {"rater": "R", "value": "5", "a": 36, "n_rater": 40, "b": 0,
         "n_rest": 120, "p_rater": 0.9, "p_rest": 0.0, "z": 6.0}
    within = _make_finding("PRONET", "CC", "f_ampscz_id", "f_item", "", d,
                           tier="within_site", cohort_rate=0.0)
    cohort = _make_finding("PRONET", "CC", "f_ampscz_id", "f_item", "", d,
                           tier="cohort", cohort_rate=0.0)
    sev_within = float(within["severity_score"])
    sev_cohort = float(cohort["severity_score"])
    assert sev_cohort < sev_within
    # within 0.02 of the exact factor (severity is rounded to 2 dp on each row).
    assert abs(sev_cohort - SITE_CONFOUNDED_SEVERITY_FACTOR * sev_within) <= 0.02


def test_rater_effect_degenerate_within_site_falls_back_to_cohort():
    # Regression for the "well-sampled rater silently dropped from BOTH tiers"
    # P1 bug. BB1 is a well-sampled home rater that over-uses value '4', but its
    # sole co-rater BB2 is under-sampled AT HOME (18 < MIN_SUBJECTS_PER_RATER),
    # so BB1's within-site contrast is degenerate (nrest < gate). Before the fix
    # BB1 was in `multi_raters` and the cohort loop skipped it -> 0 findings.
    # After the fix BB1 is NOT `within_tested` (its nrest failed) so it falls
    # through to the cohort/site-confounded tier.
    import pandas as pd
    from simple_anomaly_detection import rater_effect

    rows = []

    def add(rater, prefix, n, vals):
        for i in range(n):
            rows.append({"subjectid": f"{prefix}{i:05d}",
                         "panss_interview_ampscz_id": rater,
                         "panss_p1": str(vals[i % len(vals)])})

    add("BB1", "BB1", 25, [4])          # well-sampled bystander, over-uses '4'
    add("BB2", "BB2", 18, [1, 2, 3])    # co-rater under-sampled AT HOME (BB)
    add("BB2", "AA9", 4, [1, 2, 3])     # + off-site -> BB2 qualifies (total 22)
    add("CC1", "CC", 30, [1, 2, 3])     # single-rater site (background cohort)
    findings = rater_effect.detect_all({("PRONET", "baseline"): pd.DataFrame(rows)})

    bb1 = [f for f in findings
           if f["rater_id"] == "BB1" and f["scored_value"] == "4"]
    assert bb1, f"BB1 must not be dropped from both tiers; got {findings}"
    # It cleared no valid within-site contrast, so it surfaces as site-confounded.
    assert bb1[0]["confidence"] == "site-confounded"
    assert bb1[0]["site_id"] == "BB"


def test_site_network_corr_drift_collapses_dominant_variable():
    # Regression for the correlation-drift fan-out P1. One variable (v00)
    # decorrelated at a single site trips a Fisher-z row against EVERY partner
    # it globally correlates with. Before the fix those emitted as N separate
    # "v00 | vNN" rows (up to ~29); after the fix _collapse_corr_drift folds
    # them into ONE dominant-variable row per site.
    import numpy as np
    import pandas as pd
    from simple_anomaly_detection import site_network

    rng = np.random.default_rng(3)
    nsite, per_n, nvar = 8, 120, 6
    frames = []
    for s in range(nsite):
        latent = rng.normal(size=per_n)
        d = {"subjectid": [f"S{s}{i:04d}" for i in range(per_n)]}
        for v in range(nvar):
            col = latent + rng.normal(scale=0.45, size=per_n)
            if v == 0 and s == 3:           # v00 decorrelated at exactly one site
                col = rng.normal(size=per_n)
            d[f"v{v:02d}"] = col
        frames.append(pd.DataFrame(d))
    big = pd.concat(frames, ignore_index=True)

    findings = site_network.detect_all({("PRONET", "baseline"): big})
    corr = [f for f in findings if "Fisher-z" in str(f.get("method", ""))]
    v0 = [f for f in corr
          if str(f.get("variable", "")).startswith("v00 ")
          or "v00" in str(f.get("variables_involved", ""))]
    # Folded to a single dominant-variable row (not one row per partner pair).
    assert len(v0) == 1, f"corr-drift fan-out not collapsed: {[f['variable'] for f in v0]}"
    assert "(corr-drift)" in v0[0]["variable"]
    assert int(v0[0].get("corroborating_pairs", 0)) >= 2


def test_cluster3d_longitudinal_survives_baseline_heavy_filler(tmp_path):
    # Regression for the "top-N_COLS collapses to one timepoint" silent-disable
    # P1. With >= N_COLS numeric columns concentrated at baseline (routine at
    # real scale), the old global completeness sort selected all-baseline columns
    # -> no triple could span >= MIN_TIMEPOINTS -> the whole cross-timepoint
    # detector returned [] and the seeded longitudinal anomaly was lost. The
    # per-timepoint round-robin keeps later timepoints represented.
    if not _HAS_SKLEARN:
        pytest.skip("cluster_3d_longitudinal needs scikit-learn")
    import os
    import numpy as np
    from simple_anomaly_detection.demo import build_demo_dataset
    from simple_anomaly_detection.runner import SimpleAnomalyDetector
    from simple_anomaly_detection import cluster_3d_longitudinal as L

    indir = os.path.join(str(tmp_path), "in")
    outdir = os.path.join(str(tmp_path), "out")
    _, _, seeds = build_demo_dataset(indir)
    slices = SimpleAnomalyDetector(input_path=indir, output_path=outdir)._load_slices()
    seed45 = next(s["subjectid"] for s in seeds
                  if s["anomaly_type"] == "cluster_3d_longitudinal_outlier")

    # Sanity: the seed surfaces on the clean demo.
    assert any(str(r["subjectid"]) == seed45 for r in L.detect_all(slices))

    # Flood the PRONET baseline slice with > N_COLS high-completeness filler
    # numeric columns (uncorrelated noise, so they add no real triples).
    bkey = next(k for k in slices if k == ("PRONET", "baseline"))
    rng = np.random.default_rng(9)
    bdf = slices[bkey].copy()
    for c in range(L.N_COLS + 5):
        bdf[f"filler{c:02d}"] = rng.normal(size=len(bdf))
    flooded = dict(slices)
    flooded[bkey] = bdf

    after = L.detect_all(flooded)
    assert any(str(r["subjectid"]) == seed45 for r in after), (
        "baseline-heavy filler emptied the cross-timepoint detector "
        "(top-N_COLS collapsed to one timepoint)")
