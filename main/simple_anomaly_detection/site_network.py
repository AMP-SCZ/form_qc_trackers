"""Detector 4 - Network / site outlier detection.

For each numeric variable (and binary-as-fraction), compare per-site
distribution stats against the cross-site distribution:

    * median shift     : |site_median - global_median| / cross-site MAD of medians
    * spread shift     : log2(site_IQR / global_IQR)
    * missingness shift: |site_missing_rate - global_rate| / cross-site MAD of rates
    * pair-correlation : within numeric pairs that correlate globally, flag sites
                         whose correlation differs sharply (Fisher z delta)

The cross-network comparison only runs when 3+ networks are present.
With exactly two networks the median equals the mean and both networks
land equidistant from the "global" median by construction -- z is the
same for both, so the test is non-informative.

A site / network needs at least MIN_N_PER_SITE rows to be eligible.

Discovery peer: ``qc_types/discovery/site_distribution_drift_checks.py``
plus ``site_correlation_drift_checks.py`` and ``site_trajectory_slope_checks.py``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .common import (
    Finding, calibrate, classify_columns, matches_excluded_substrings,
    robust_center_scale, site_from_subjectid, stack_by_network,
    to_numeric_clean, short,
)

ANOMALY_TYPE = "site_network_outlier"

# Per-detector exclusion list (case-insensitive substring match against
# variable name). Add strings here to suppress this detector's findings
# on those variables without touching the shared EXCLUDED_VARIABLE_SUBSTRINGS
# in common.py.
EXTRA_EXCLUDED_SUBSTRINGS: tuple = ()

MIN_N_PER_SITE = 30
SITE_MEDIAN_Z_THRESHOLD = 3.0
SITE_IQR_LOG2_THRESHOLD = 1.0   # IQR off by >2x or <0.5x
SITE_MISS_Z_THRESHOLD   = 3.0
SITE_CORR_FISHER_DELTA  = 0.5
N_NUMERIC_FOR_CORR      = 30
# The median-shift and missingness-shift sub-tests compute a robust scale
# ACROSS the per-site medians / missingness rates via robust_center_scale,
# which returns NaN sigma for fewer than 5 inputs (common.py). So those two
# sub-tests need at least this many eligible sites; with 3-4 only the
# spread-shift sub-test (which uses np.nanmedian directly) can run. Below we
# gate explicitly and LOG, rather than letting the NaN silently disable them.
# (A stable small-group estimator could restore them for 3-4 sites, but the
# cross-site MAD of so few medians is itself unstable -- deferred.)
MIN_SITES_FOR_CROSS_SCALE = 5


def detect_all(per_slice: Dict, classify: Dict = None, **_) -> List[dict]:
    """``classify`` is unused here -- this detector classifies the
    network stack once, which is bounded already."""
    by_network: Dict[str, List] = {}
    for (network, tp), df in per_slice.items():
        if df is None or df.empty or "subjectid" not in df.columns:
            continue
        by_network.setdefault(network, []).append((tp, df))

    rows: List[dict] = []
    network_summaries = {}

    for network, items in by_network.items():
        try:
            stacked = stack_by_network(items)
            cats = classify_columns(stacked)
            site_col = "_site"
            stacked[site_col] = stacked["subjectid"].astype(str).str[:2]
            numerics = [c for c in cats["numeric"]
                        if not matches_excluded_substrings(c, EXTRA_EXCLUDED_SUBSTRINGS)]
            rows.extend(_per_site(stacked, numerics, network, site_col))
            rows.extend(_per_site_pairs(stacked, numerics, network, site_col))
            network_summaries[network] = stacked
        except Exception as e:
            print(f"  [site_network] WARN network {network}: {e}")

    # Cross-network only runs with 3+ networks. With 2 it is degenerate:
    # median(A, B) = mean(A, B), so |A - med| == |B - med| and both
    # networks always score the same z.
    if len(network_summaries) >= 3:
        try:
            rows.extend(_per_network(network_summaries))
        except Exception as e:
            print(f"  [site_network] WARN cross-network: {e}")

    return rows


def _per_site(stacked: pd.DataFrame, numerics: List[str], network: str, site_col: str) -> List[dict]:
    out: List[dict] = []
    if not numerics:
        return out
    site_counts = stacked[site_col].value_counts()
    eligible_sites = site_counts[site_counts >= MIN_N_PER_SITE].index.tolist()
    if len(eligible_sites) < 3:
        return out

    n_total = len(stacked)
    gated_cols = 0   # columns where <5 sites disabled median/missingness shift
    for col in numerics:
        num = to_numeric_clean(stacked[col])
        if num.notna().sum() < MIN_N_PER_SITE * 3:
            continue

        per_site = {}
        for site in eligible_sites:
            mask = stacked[site_col] == site
            sub = num[mask]
            n = int(sub.notna().sum())
            if n < MIN_N_PER_SITE:
                continue
            med, _ = robust_center_scale(sub.dropna().to_numpy(dtype=float))
            q1, q3 = np.nanpercentile(sub.dropna(), [25, 75])
            iqr = float(q3 - q1)
            miss_rate = float(mask.sum() - n) / max(int(mask.sum()), 1)
            per_site[site] = (med, iqr, miss_rate, n)
        if len(per_site) < 3:
            continue

        meds = np.array([v[0] for v in per_site.values()], dtype=float)
        iqrs = np.array([v[1] for v in per_site.values()], dtype=float)
        misses = np.array([v[2] for v in per_site.values()], dtype=float)

        # Cross-site robust center/scale for the median- and missingness-shift
        # sub-tests needs >=5 sites (robust_center_scale NaNs sigma below that).
        # Spread-shift uses np.nanmedian directly and runs with >=3.
        i_med = float(np.nanmedian(iqrs)) if np.isfinite(iqrs).any() else float("nan")
        if len(per_site) >= MIN_SITES_FOR_CROSS_SCALE:
            m_med, m_sigma = robust_center_scale(meds)
            ms_med, ms_sigma = robust_center_scale(misses)
        else:
            m_med = m_sigma = ms_med = ms_sigma = float("nan")
            gated_cols += 1

        for site, (med, iqr, miss, n) in per_site.items():
            # Median shift
            if np.isfinite(m_sigma) and m_sigma > 0:
                z = abs(med - m_med) / m_sigma
                if z >= SITE_MEDIAN_Z_THRESHOLD:
                    out.append(Finding(
                        anomaly_type=ANOMALY_TYPE,
                        severity_score=calibrate(z, SITE_MEDIAN_Z_THRESHOLD, 6.0),
                        raw_score=float(z),
                        network=network, site_id=site, subjectid="(site-level)", variable=col,
                        observed_value=f"site median {med:.4g} (n={n})",
                        expected_value=f"cross-site median {m_med:.4g}, sigma {m_sigma:.4g}",
                        explanation=(
                            f"Site {site} median is {z:.2f} cross-site sigmas from the "
                            f"network median for {col}. Suggests systematic shift at site."
                        ),
                        method="cross-site MAD on site medians",
                    ).to_row())

            # Spread shift
            if i_med and np.isfinite(i_med) and i_med > 0 and iqr > 0:
                r = float(np.log2(iqr / i_med))
                if abs(r) >= SITE_IQR_LOG2_THRESHOLD:
                    out.append(Finding(
                        anomaly_type=ANOMALY_TYPE,
                        severity_score=calibrate(abs(r), SITE_IQR_LOG2_THRESHOLD, 3.0),
                        raw_score=abs(r),
                        network=network, site_id=site, subjectid="(site-level)", variable=col,
                        observed_value=f"site IQR {iqr:.4g}",
                        expected_value=f"cross-site median IQR {i_med:.4g}",
                        explanation=(
                            f"Site {site} IQR for {col} is {2 ** abs(r):.2f}x the "
                            f"network IQR ({'wider' if r > 0 else 'narrower'}); "
                            f"possible scale or data-entry-precision issue."
                        ),
                        method="log2 IQR ratio vs cross-site median IQR",
                    ).to_row())

            # Missingness shift
            if np.isfinite(ms_sigma) and ms_sigma > 0:
                mz = abs(miss - ms_med) / ms_sigma
                if mz >= SITE_MISS_Z_THRESHOLD and abs(miss - ms_med) > 0.05:
                    out.append(Finding(
                        anomaly_type=ANOMALY_TYPE,
                        severity_score=calibrate(mz, SITE_MISS_Z_THRESHOLD, 6.0),
                        raw_score=float(mz),
                        network=network, site_id=site, subjectid="(site-level)", variable=col,
                        observed_value=f"site missing rate {miss:.2%}",
                        expected_value=f"cross-site missing-rate median {ms_med:.2%}",
                        explanation=(
                            f"Site {site} missingness for {col} is {mz:.2f} cross-site "
                            f"sigmas from the network median. Possible collection gap."
                        ),
                        method="cross-site MAD on site missingness",
                    ).to_row())
    if gated_cols:
        print(f"  [site_network] {network}: median/missingness-shift disabled "
              f"for {gated_cols} column(s) with <{MIN_SITES_FOR_CROSS_SCALE} "
              f"eligible sites (only spread-shift ran); cross-site MAD needs "
              f">={MIN_SITES_FOR_CROSS_SCALE} sites.")
    return out


def _per_site_pairs(stacked: pd.DataFrame, numerics: List[str], network: str, site_col: str) -> List[dict]:
    """Find site-level correlation drift on the strongest globally-correlated pairs."""
    out: List[dict] = []
    if len(numerics) < 4:
        return out

    # Rank numerics by completeness, then pick the strongest global pairs.
    completeness = {c: int(to_numeric_clean(stacked[c]).notna().sum()) for c in numerics}
    top = sorted(numerics, key=lambda c: completeness[c], reverse=True)[:N_NUMERIC_FOR_CORR]
    series = {c: to_numeric_clean(stacked[c]) for c in top}

    site_counts = stacked[site_col].value_counts()
    eligible_sites = site_counts[site_counts >= MIN_N_PER_SITE].index.tolist()
    if len(eligible_sites) < 3:
        return out

    pairs = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            mask = series[a].notna() & series[b].notna()
            if mask.sum() < MIN_N_PER_SITE * 3:
                continue
            x = series[a][mask].to_numpy(dtype=float)
            y = series[b][mask].to_numpy(dtype=float)
            if np.std(x) <= 0 or np.std(y) <= 0:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            if abs(r) < 0.5:
                continue
            pairs.append((a, b, r))
    pairs.sort(key=lambda t: -abs(t[2]))
    pairs = pairs[:50]   # cap pair budget

    for a, b, r_global in pairs:
        rz_global = _fisher_z(r_global)
        for site in eligible_sites:
            mask = (stacked[site_col] == site) & series[a].notna() & series[b].notna()
            if mask.sum() < MIN_N_PER_SITE:
                continue
            x = series[a][mask].to_numpy(dtype=float)
            y = series[b][mask].to_numpy(dtype=float)
            if np.std(x) <= 0 or np.std(y) <= 0:
                continue
            r_site = float(np.corrcoef(x, y)[0, 1])
            rz_site = _fisher_z(r_site)
            delta = abs(rz_site - rz_global)
            if delta < SITE_CORR_FISHER_DELTA:
                continue
            out.append(Finding(
                anomaly_type=ANOMALY_TYPE,
                severity_score=calibrate(delta, SITE_CORR_FISHER_DELTA, 1.5),
                raw_score=delta,
                network=network, site_id=site, subjectid="(site-level)",
                variable=f"{a} | {b}",
                variables_involved=f"{a}, {b}",
                observed_value=f"site rho={r_site:+.2f} (n={int(mask.sum())})",
                expected_value=f"network rho={r_global:+.2f}",
                explanation=(
                    f"Site {site} correlation for {a} vs {b} differs from the network "
                    f"by Fisher-z delta {delta:.2f}. Possible site-specific protocol "
                    f"or scale shift."
                ),
                method="Fisher-z delta of site vs network correlation",
            ).to_row())
    return out


def _fisher_z(r: float) -> float:
    r = max(-0.999999, min(0.999999, float(r)))
    return 0.5 * np.log((1 + r) / (1 - r))


# ---------------------------------------------------------------------------
# Cross-network comparison.
# ---------------------------------------------------------------------------

def _per_network(network_summaries: Dict[str, pd.DataFrame]) -> List[dict]:
    out: List[dict] = []
    networks = list(network_summaries.keys())
    if len(networks) < 2:
        return out
    # We compare network A to the union of others on a small shared core of
    # numeric variables.
    union_cols = set()
    for df in network_summaries.values():
        union_cols.update(classify_columns(df)["numeric"])
    if EXTRA_EXCLUDED_SUBSTRINGS:
        union_cols = {c for c in union_cols
                      if not matches_excluded_substrings(c, EXTRA_EXCLUDED_SUBSTRINGS)}
    if not union_cols:
        return out
    shared = sorted(union_cols)[:80]

    per_network = {n: {} for n in networks}
    for n, df in network_summaries.items():
        for col in shared:
            if col not in df.columns:
                continue
            num = to_numeric_clean(df[col]).dropna()
            if num.size < 60:
                continue
            med = float(np.median(num))
            mad = float(np.median(np.abs(num - med)))
            sigma = mad * 1.4826 if mad > 0 else float(np.std(num))
            miss = 1.0 - float(num.size) / max(len(df), 1)
            per_network[n][col] = (med, sigma, miss, num.size)

    # Pairwise compare medians.
    for col in shared:
        present = [(n, *per_network[n][col]) for n in networks if col in per_network[n]]
        if len(present) < 2:
            continue
        meds = np.array([t[1] for t in present], dtype=float)
        sigmas = np.array([t[2] for t in present], dtype=float)
        # Use the pooled-ish sigma to keep this comparable to single-site z.
        pooled = float(np.nanmedian(sigmas[np.isfinite(sigmas)]))
        if not np.isfinite(pooled) or pooled <= 0:
            continue
        global_med = float(np.nanmedian(meds))
        for (n, med, sigma, miss, count) in present:
            z = abs(med - global_med) / pooled
            if z >= SITE_MEDIAN_Z_THRESHOLD:
                out.append(Finding(
                    anomaly_type=ANOMALY_TYPE,
                    severity_score=calibrate(z, SITE_MEDIAN_Z_THRESHOLD, 6.0),
                    raw_score=float(z),
                    network=n, site_id="(network-level)", subjectid="(network-level)",
                    variable=col,
                    observed_value=f"network median {med:.4g} (n={count})",
                    expected_value=f"cross-network median {global_med:.4g}",
                    explanation=(
                        f"Network {n} median is {z:.2f} cross-network sigmas from the "
                        f"pooled median for {col}."
                    ),
                    method="cross-network median comparison",
                ).to_row())
    return out
