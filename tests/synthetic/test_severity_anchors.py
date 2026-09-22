"""Anchor-point regression test for the shared severity calibration.

``common.calibrate`` maps a detector's raw score onto the shared 0..100
severity where the detector's flag THRESHOLD must land at ~50 and its EXTREME
anchor at ~95. ``combined_ranked``'s cross-detector comparability depends on
every detector honoring those anchors -- yet nothing pinned them before:
``test_combined_ranked_is_sorted_desc`` only checks that an already-sorted frame
is monotonic (true for any values). This locks the anchors so a drift in
``calibrate()`` (or in a detector's (threshold, extreme) pair) is caught.

NOTE: honoring the anchors makes each detector's scale internally consistent; it
does NOT make a "50" equally surprising across detectors (a 1%-rare string, a
5%-rare date reversal and a 4-sigma value all map near 50). That cross-detector
comparability gap is a documented open decision -- see the 2026-07-09 review
(statistical-methodology-1 / test-coverage-deadcode-6). This test guards the
weaker-but-testable property: no anchor has silently drifted.
"""
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simple_anomaly_detection.common import calibrate, severity_from_z  # noqa: E402

# Every (threshold, extreme) pair the package actually feeds calibrate(), tagged
# with the detector / sub-test that uses it. Collecting them here in one place is
# itself the audit surface the review asks for: the anchors are NOT equally
# surprising across detectors, but each must at least honor its own stated anchor.
CALIBRATION_ANCHORS = [
    ("standard_outlier |z|",              4.0, 10.0),
    ("time_series |z|",                   4.0, 10.0),
    ("date_anomaly gap |z|",              4.0, 10.0),
    ("date_anomaly order -log10(freq)",   1.3,  3.0),
    ("site_network median z",             3.0,  6.0),
    ("site_network missingness z",        3.0,  6.0),
    ("site_network spread log2-IQR",      1.0,  3.0),
    ("site_network corr Fisher-z delta",  0.5,  1.5),
    ("site_network cross-network z",      3.0,  6.0),
    ("format_string -log10(freq)",        2.0,  4.0),
    ("cluster_3d off-relationship dev",   4.0, 10.0),
]


@pytest.mark.parametrize("name,threshold,extreme", CALIBRATION_ANCHORS)
def test_flag_threshold_maps_to_50(name, threshold, extreme):
    assert calibrate(threshold, threshold, extreme) == pytest.approx(50.0, abs=1e-6), name


@pytest.mark.parametrize("name,threshold,extreme", CALIBRATION_ANCHORS)
def test_extreme_maps_to_95(name, threshold, extreme):
    assert calibrate(extreme, threshold, extreme) == pytest.approx(95.0, abs=1e-6), name


def test_calibrate_is_bounded_and_monotone():
    prev = -1.0
    raw = 0.0
    while raw <= 20.0:
        v = calibrate(raw, 4.0, 10.0)
        assert 0.0 <= v <= 100.0, f"calibrate out of [0,100] at raw={raw}: {v}"
        assert v >= prev - 1e-9, f"calibrate not monotone at raw={raw}"
        prev = v
        raw += 0.1


def test_calibrate_handles_nonfinite_and_subthreshold():
    # non-finite / None -> 0 (never a spurious mid-scale severity)
    assert calibrate(float("nan"), 4.0, 10.0) == 0.0
    assert calibrate(float("inf"), 4.0, 10.0) == 0.0
    assert calibrate(None, 4.0, 10.0) == 0.0
    # halfway to the threshold -> ~25 (linear 0->50 below threshold)
    assert calibrate(2.0, 4.0, 10.0) == pytest.approx(25.0, abs=1e-6)


def test_severity_from_z_matches_calibrate():
    for z in (0.0, 2.0, 4.0, 7.0, 10.0, 25.0, -8.0):
        assert severity_from_z(z, 4.0, 10.0) == pytest.approx(
            calibrate(abs(z), 4.0, 10.0))
    assert severity_from_z(float("nan"), 4.0, 10.0) == 0.0


def test_detector_threshold_constants_unchanged():
    """Pin the module-level threshold constants the anchors above depend on, so
    a silent edit to a detector's flag bar is caught here. (Inline calibrate()
    literals -- date-order 1.3/3.0, format 2.0/4.0, cluster 4/10 -- are covered
    by the parametrized anchor tests above.)"""
    from simple_anomaly_detection import (  # noqa: E402
        standard_outlier, time_series, site_network, date_anomaly,
    )
    assert standard_outlier.Z_THRESHOLD == 4.0
    assert time_series.Z_THRESHOLD == 4.0
    assert date_anomaly.GAP_Z_THRESHOLD == 4.0
    assert site_network.SITE_MEDIAN_Z_THRESHOLD == 3.0
    assert site_network.SITE_MISS_Z_THRESHOLD == 3.0
    assert site_network.SITE_IQR_LOG2_THRESHOLD == 1.0
    assert site_network.SITE_CORR_FISHER_DELTA == 0.6
    assert site_network.SITE_CORR_Z_THRESHOLD == 3.5
