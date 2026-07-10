"""Detector 7 - Date anomalies."""
 
from __future__ import annotations
 
from typing import Dict, List, Tuple
import re
 
import numpy as np
import pandas as pd
 
from .common import (
    Finding, calibrate, classify_columns, clean_dates_series,
    matches_excluded_substrings, robust_center_scale,
    site_from_subjectid, short,
)
 
ANOMALY_TYPE = "date_anomaly_simple"
 
MIN_PAIR_N = 40
ORDER_DOMINANT_FRAC = 0.95
GAP_Z_THRESHOLD = 4.0
MIN_GAP_ABS_DAYS = 14.0
 
# Exact-day fixed gap rule for month-based timepoints.
# 0 means any difference from expected days will be flagged.
FIXED_GAP_THRESHOLD_DAYS = 0.0
 
EXTRA_EXCLUDED_SUBSTRINGS: tuple = ()
 
 
def detect_all(per_slice: Dict, classify: Dict = None, **_) -> List[dict]:
    classify = classify or {}
    by_network: Dict[str, List] = {}
 
    for (network, tp), df in per_slice.items():
        if df is None or df.empty or "subjectid" not in df.columns:
            continue
 
        by_network.setdefault(network, []).append(
            (tp, df, classify.get((network, tp)))
        )
 
    rows: List[dict] = []
 
    for network, items in by_network.items():
        try:
            rows.extend(_detect_network(network, items))
        except Exception as e:
            print(f"  [date_anomaly] WARN network {network}: {e}")
 
    return rows
 
 
def _detect_network(network: str, items: List) -> List[dict]:
    date_frames: Dict[str, pd.DataFrame] = {}
 
    for tp, df, cats in items:
        if cats is None:
            cats = classify_columns(df)
 
        dcols = [
            c for c in cats["date"]
            if not matches_excluded_substrings(c, EXTRA_EXCLUDED_SUBSTRINGS)
        ]
 
        if not dcols:
            continue
 
        sub = df[["subjectid"] + dcols].copy()
 
        for c in dcols:
            sub[c] = clean_dates_series(sub[c])
 
        date_frames[tp] = sub
 
    if not date_frames:
        return []
 
    by_id: List[Tuple[str, str, str, pd.Series]] = []
 
    for tp, df in date_frames.items():
        ids = df["subjectid"].astype(str)
 
        for c in df.columns:
            if c == "subjectid":
                continue
 
            s = pd.Series(df[c].values, index=ids.values, name=c)
            s = s.groupby(level=0).first()
 
            if s.dropna().size < MIN_PAIR_N:
                continue
 
            label = f"{tp}::{c}"
            by_id.append((tp, c, label, s))
 
    out: List[dict] = []
 
    by_tp: Dict[str, List[Tuple[str, str, str, pd.Series]]] = {}
 
    for e in by_id:
        by_tp.setdefault(e[0], []).append(e)
 
    for tp, ents in by_tp.items():
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                out.extend(_pair(network, ents[i], ents[j]))
 
    by_col: Dict[str, List[Tuple[str, str, str, pd.Series]]] = {}
 
    for e in by_id:
        by_col.setdefault(e[1], []).append(e)
 
    for col, ents in by_col.items():
        if len(ents) < 2:
            continue
 
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                if ents[i][0] == ents[j][0]:
                    continue
 
                out.extend(_pair(network, ents[i], ents[j]))
 
    return out
 
 
def _expected_gap_days(tp_a: str, tp_b: str):
    tp_a = str(tp_a).lower()
    tp_b = str(tp_b).lower()
 
    # Skip baseline, screening, floating, etc.
    # Only month-based timepoints 
    if not tp_a.startswith("month") or not tp_b.startswith("month"):
        return None
 
    m1_match = re.search(r"\d+", tp_a)
    m2_match = re.search(r"\d+", tp_b)
 
    if m1_match is None or m2_match is None:
        return None
 
    m1 = int(m1_match.group())
    m2 = int(m2_match.group())
 
    # same month = 30 days
    # month3 to month4 = 60 days
    # month3 to month6 = 120 days
    return (abs(m2 - m1) + 1) * 30
 
 
def _pair(
    network: str,
    left: Tuple[str, str, str, pd.Series],
    right: Tuple[str, str, str, pd.Series],
) -> List[dict]:
 
    tp_a, col_a, lbl_a, sa = left
    tp_b, col_b, lbl_b, sb = right
 
    joined = pd.concat([sa, sb], axis=1, join="inner")
    joined.columns = ["a", "b"]
    joined = joined.dropna()
 
    if len(joined) < MIN_PAIR_N:
        return []
 
    gap_days = (joined["b"] - joined["a"]).dt.total_seconds() / 86400.0
 
    if gap_days.size < MIN_PAIR_N:
        return []
 
    n = len(joined)
    n_le = int((gap_days >= 0).sum())
    n_ge = int((gap_days <= 0).sum())
 
    out: List[dict] = []
 
    dominant_dir = None
 
    if n_le / n >= ORDER_DOMINANT_FRAC:
        dominant_dir = "a<=b"
        violators = joined[gap_days < 0]
        violation_freq = float(1 - n_le / n)
 
    elif n_ge / n >= ORDER_DOMINANT_FRAC:
        dominant_dir = "a>=b"
        violators = joined[gap_days > 0]
        violation_freq = float(1 - n_ge / n)
 
    if dominant_dir is not None and not violators.empty:
        raw = float(-np.log10(max(violation_freq, 1e-6)))
        severity = calibrate(raw, threshold=1.3, extreme=3.0)
 
        for subj, row in violators.iterrows():
            out.append(Finding(
                anomaly_type=ANOMALY_TYPE,
                severity_score=severity,
                raw_score=raw,
                network=network,
                timepoint=f"{tp_a} <-> {tp_b}",
                site_id=site_from_subjectid(str(subj)),
                subjectid=str(subj),
                variable=f"{lbl_a} | {lbl_b}",
                variables_involved=f"{lbl_a}, {lbl_b}",
                observed_value=(
                    f"a={short(row['a'].date())}, "
                    f"b={short(row['b'].date())}"
                ),
                expected_value=(
                    f"usually {dominant_dir} "
                    f"(~{1 - violation_freq:.1%} of subjects)"
                ),
                explanation=(
                    f"Date order violated. Across {n} subjects, "
                    f"{dominant_dir} holds {1 - violation_freq:.1%} "
                    f"of the time."
                ),
                method="paired date ordering",
            ).to_row())
 
    # Fixed expected-gap check.
    expected_gap = _expected_gap_days(tp_a, tp_b)
 
    if expected_gap is None:
        return out
 
    g = gap_days.to_numpy(dtype=float)
    diff = g - expected_gap
 
    bad = np.where(np.abs(diff) > FIXED_GAP_THRESHOLD_DAYS)[0]
 
    for k in bad:
        observed_gap = float(g[k])
        gap_diff = float(diff[k])
 
        if not np.isfinite(observed_gap) or not np.isfinite(gap_diff):
            continue
 
        subj = str(joined.index[k])
        row_k = joined.iloc[k]
 
        date_a_str = str(row_k["a"].date())
        date_b_str = str(row_k["b"].date())
 
        direction = "longer" if gap_diff > 0 else "shorter"
 
        out.append(Finding(
            anomaly_type=ANOMALY_TYPE,
            severity_score=calibrate(abs(gap_diff), 30.0, 120.0),
            raw_score=abs(gap_diff),
            network=network,
            timepoint=f"{tp_a} <-> {tp_b}",
            site_id=site_from_subjectid(subj),
            subjectid=subj,
            variable=f"{lbl_a} | {lbl_b}",
            variables_involved=f"{lbl_a}, {lbl_b}",
            observed_value=(
                f"a={date_a_str}, b={date_b_str} "
                f"(gap={observed_gap:.1f} days)"
            ),
            expected_value=f"expected gap = {expected_gap:.1f} days",
            explanation=(
                f"Observed gap between {lbl_a} and {lbl_b} is "
                f"{abs(gap_diff):.1f} days {direction} than expected. "
                f"Please verify the dates."
            ),
            method="fixed expected date gap",
            extra={
                "date_a": date_a_str,
                "date_b": date_b_str,
                "observed_gap_days": observed_gap,
                "expected_gap_days": expected_gap,
                "difference_from_expected_days": gap_diff,
            },
        ).to_row())
 
    return out