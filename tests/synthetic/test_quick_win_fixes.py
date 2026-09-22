"""Regression tests for the 2026-07-09 quick-win fixes.

Locks two behavior-changing fixes so a later edit cannot silently undo them:
  (1) runner._suppress_redundant_outliers restricts its suppression cap to
      site_network MEDIAN-shift rows, so a site's missingness/spread drift can
      no longer delete a genuine individual value outlier (review NEW-P1-a).
  (2) common.clean_dates_series parses mixed-tz-offset date columns without
      raising and returns tz-naive datetimes, so date_anomaly no longer loses a
      whole network's date findings to the mixed-offset crash (KNOWN-OPEN
      DATE_ANOMALY-1).
"""
import os
import sys

import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simple_anomaly_detection import common  # noqa: E402
from simple_anomaly_detection.runner import SimpleAnomalyDetector  # noqa: E402

_MEDIAN = "cross-site MAD on site medians"
_MISS = "cross-site MAD on site missingness"
_SPREAD = "log2 IQR ratio vs cross-site median IQR"


def _std_row(sev):
    return {"network": "PRONET", "variable": "weight_kg", "site_id": "KC",
            "subjectid": "KC00001", "severity_score": sev}


def _site_row(sev, method):
    return {"network": "PRONET", "variable": "weight_kg", "site_id": "KC",
            "severity_score": sev, "method": method}


def _suppress(std_rows, site_rows):
    std_df = pd.DataFrame(std_rows)
    site_df = pd.DataFrame(site_rows)
    return SimpleAnomalyDetector._suppress_redundant_outliers(std_df, site_df)


def test_missingness_shift_does_not_suppress_value_outlier():
    # Individual value outlier (sev 80) at a site whose ONLY site_network signal
    # is a high-severity missingness shift (sev 95). Pre-fix this deleted the
    # individual (80 < 95); post-fix the missingness row can't build the cap.
    kept = _suppress([_std_row(80.0)], [_site_row(95.0, _MISS)])
    assert len(kept) == 1


def test_spread_shift_does_not_suppress_value_outlier():
    kept = _suppress([_std_row(80.0)], [_site_row(95.0, _SPREAD)])
    assert len(kept) == 1


def test_median_shift_still_suppresses_weaker_value_outlier():
    # A median shift (sev 95) DOES legitimately explain a less-extreme
    # individual (sev 80) at the same site -> suppressed.
    kept = _suppress([_std_row(80.0)], [_site_row(95.0, _MEDIAN)])
    assert len(kept) == 0


def test_median_shift_keeps_more_extreme_individual():
    # An individual more extreme (sev 90) than the median shift (sev 70) is
    # kept -- the "keep individuals beyond the site shift" guarantee is intact.
    kept = _suppress([_std_row(90.0)], [_site_row(70.0, _MEDIAN)])
    assert len(kept) == 1


def test_clean_dates_series_mixed_tz_offsets_no_crash():
    # Mixed offset + naive + sentinel + blank. Pre-fix pd.to_datetime without
    # utc=True returned object dtype whose .dt raised; here it must parse.
    s = pd.Series([
        "2026-01-15",                    # naive
        "2026-01-15T05:00:00+05:00",     # offset-bearing (== 00:00 UTC, same day)
        "2026-03-01T22:00:00-05:00",     # different offset
        "1909-09-09",                    # date sentinel -> NaT
        "",                              # blank -> NaT
    ])
    out = common.clean_dates_series(s)

    # tz-naive datetime64 result.
    assert pd.api.types.is_datetime64_ns_dtype(out.dtype)
    assert getattr(out.dt, "tz", None) is None

    assert out.iloc[0] == pd.Timestamp("2026-01-15")
    assert pd.notna(out.iloc[1]) and out.iloc[1].normalize() == pd.Timestamp("2026-01-15")
    assert pd.notna(out.iloc[2])          # parsed, not dropped
    assert pd.isna(out.iloc[3])           # sentinel stripped
    assert pd.isna(out.iloc[4])           # blank


def test_clean_dates_series_all_naive_unchanged():
    # The common case must be byte-for-byte the same calendar dates.
    s = pd.Series(["2026-01-15", "2025-12-31", "2026-02-28"])
    out = common.clean_dates_series(s)
    assert list(out) == [pd.Timestamp("2026-01-15"),
                         pd.Timestamp("2025-12-31"),
                         pd.Timestamp("2026-02-28")]


@pytest.mark.parametrize("vals", [
    [None, "", "1909-09-09"],       # all sentinel / blank -> all NaT
    [None, None],                    # all real NaN
    [],                              # empty
])
def test_clean_dates_series_all_missing_returns_all_nat(vals):
    # format="mixed" can be finicky on all-NA input across pandas releases, and
    # the OTHER caller (time_series) has no per-column guard, so this path must
    # never raise. It is common on real data (a date field all-sentinel for a
    # slice).
    out = common.clean_dates_series(pd.Series(vals, dtype=object))
    assert pd.api.types.is_datetime64_ns_dtype(out.dtype)
    assert int(out.isna().sum()) == len(vals)


def test_suppression_median_label_still_matches_site_network():
    # runner._suppress_redundant_outliers filters on the literal method string
    # site_network emits for the median-shift sub-test. If that label is renamed
    # in site_network without updating the runner, suppression silently no-ops
    # (fail-safe: keeps more rows) -- pin the two together so a rename fails here.
    import inspect

    from simple_anomaly_detection import site_network
    assert _MEDIAN in inspect.getsource(site_network), (
        f"site_network no longer emits method {_MEDIAN!r}; update "
        "runner._suppress_redundant_outliers' filter to match")
