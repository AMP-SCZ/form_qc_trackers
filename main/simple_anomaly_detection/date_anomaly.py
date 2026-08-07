"""Detector 7 - Date anomalies.

Two checks:

    order  : for each pair of date columns that *usually* land in a
             specific order (>=95% of subjects have date_A <= date_B),
             flag the rare reversals.
    gap    : for each pair, compute the per-subject gap (days). Flag
             gaps whose |robust z| against the cohort gap distribution
             exceeds the threshold. Runs even when there is no dominant
             order direction, because a year-off gap is anomalous
             regardless of which direction is "normal".

Both within-slice (two date columns at the same timepoint) and
cross-slice (any two date columns from DIFFERENT timepoints -- e.g.
baseline consent_date vs month_1 visit_date, not just the same column
across visits) pairs are scanned.

Discovery peer: ``qc_types/discovery/date_anomaly_checks.py``. That
detector follows a different algorithm; this one emits anomaly_type
``date_anomaly_simple`` so concatenated trackers do not collide.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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
# Minimum absolute gap deviation (days) to flag, on top of the z gate. For
# tightly-scheduled visits the gap MAD is tiny, so a benign few-days slack
# (e.g. a 44-day gap vs a 30-day target) trips |z|>=4 even though it is
# clinically meaningless. Requiring at least a 2-week absolute deviation
# removes that cadence-noise false-positive class; genuine multi-month gaps
# clear it easily.
MIN_GAP_ABS_DAYS = 14.0

# Per-detector exclusion list (case-insensitive substring match against
# date-column name). Add strings here to suppress this detector's
# findings on those date columns without touching the shared
# EXCLUDED_VARIABLE_SUBSTRINGS in common.py.
EXTRA_EXCLUDED_SUBSTRINGS: tuple = ()


def detect_all(per_slice: Dict, classify: Dict = None, **_) -> List[dict]:
    classify = classify or {}
    by_network: Dict[str, List] = {}
    for (network, tp), df in per_slice.items():
        if df is None or df.empty or "subjectid" not in df.columns:
            continue
        # Carry the runner's pre-computed classify_columns result so this
        # detector reuses it instead of re-classifying every slice.
        by_network.setdefault(network, []).append((tp, df, classify.get((network, tp))))

    rows: List[dict] = []
    for network, items in by_network.items():
        try:
            rows.extend(_detect_network(network, items))
        except Exception as e:
            print(f"  [date_anomaly] WARN network {network}: {e}")
    return rows


def _detect_network(network: str, items: List) -> List[dict]:
    # Find date-like columns per slice.
    date_frames: Dict[str, pd.DataFrame] = {}  # tp -> DF[subjectid, ...date cols]
    for tp, df, cats in items:
        if cats is None:
            cats = classify_columns(df)
        dcols = [c for c in cats["date"]
                 if not matches_excluded_substrings(c, EXTRA_EXCLUDED_SUBSTRINGS)]
        if not dcols:
            continue
        sub = df[["subjectid"] + dcols].copy()
        for c in dcols:
            sub[c] = clean_dates_series(sub[c])
        date_frames[tp] = sub

    if not date_frames:
        return []

    # Build (col_id, series_indexed_by_subject) entries for ALL date columns.
    # col_id is "<tp>::<colname>" for traceability.
    by_id: List[Tuple[str, str, str, pd.Series]] = []  # (tp, colname, label, series_by_subject)
    for tp, df in date_frames.items():
        ids = df["subjectid"].astype(str)
        for c in df.columns:
            if c == "subjectid":
                continue
            s = pd.Series(df[c].values, index=ids.values, name=c)
            # drop duplicates within subject (keep first non-missing)
            s = s.groupby(level=0).first()
            if s.dropna().size < MIN_PAIR_N:
                continue
            label = f"{tp}::{c}"
            by_id.append((tp, c, label, s))

    out: List[dict] = []

    # Within-slice pairs (same tp, ordered by name to avoid both directions).
    by_tp: Dict[str, List[Tuple[str, str, str, pd.Series]]] = {}
    for e in by_id:
        by_tp.setdefault(e[0], []).append(e)
    for tp, ents in by_tp.items():
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                out.extend(_pair(network, ents[i], ents[j]))

    # Cross-tp pairs: ANY two date columns from DIFFERENT timepoints, whether or
    # not they share a name -- e.g. visit_date across tps, OR baseline
    # consent_date vs month_1 visit_date. (Same-tp pairs are the within-slice
    # loop above.) Missing codes are already stripped to NaT up front by
    # clean_dates_series, and _pair drops any pair with < MIN_PAIR_N subjects
    # carrying both real dates, so unrelated cross-form columns fall out cheaply.
    for i in range(len(by_id)):
        for j in range(i + 1, len(by_id)):
            if by_id[i][0] == by_id[j][0]:
                continue  # same timepoint -> already handled within-slice
            out.extend(_pair(network, by_id[i], by_id[j]))

    # All-pairs over the date columns is O(n^2); log the column count so the
    # real-scale pair work stays observable (mirrors the cluster detectors).
    print(f"  [date_anomaly] {network}: {len(by_id)} date columns scanned "
          f"pairwise (within- and cross-timepoint) -> {len(out)} flags")
    return out


def _pair(network: str,
          left: Tuple[str, str, str, pd.Series],
          right: Tuple[str, str, str, pd.Series]) -> List[dict]:
    tp_a, col_a, lbl_a, sa = left
    tp_b, col_b, lbl_b, sb = right

    # Align on subject id (set intersection).
    joined = pd.concat([sa, sb], axis=1, join="inner")
    joined.columns = ["a", "b"]
    joined = joined.dropna()
    if len(joined) < MIN_PAIR_N:
        return []

    gap_days = (joined["b"] - joined["a"]).dt.total_seconds() / 86400.0
    if gap_days.size < MIN_PAIR_N:
        return []
    # Order check
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
        severity = calibrate(raw, threshold=1.3, extreme=3.0)  # 5% -> 1.3, 0.1% -> 3
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
                observed_value=f"a={short(row['a'].date())}, b={short(row['b'].date())}",
                expected_value=f"usually {dominant_dir} (~{1-violation_freq:.1%} of subjects)",
                explanation=(
                    f"Date order violated. Across {n} subjects, {dominant_dir} "
                    f"holds {1-violation_freq:.1%} of the time."
                ),
                method="paired date ordering",
            ).to_row())

    # Gap z runs unconditionally. Previous version returned early when
    # there was no dominant order; that hid huge gap anomalies in pairs
    # where the order ratio narrowly missed the 95% threshold. If the gap
    # distribution genuinely is bimodal MAD will be inflated and few
    # gaps will exceed the threshold -- benign. The reverse failure
    # (missing a 5-year-off date) was much worse.
    g = gap_days.to_numpy(dtype=float)
    med, sigma = robust_center_scale(g)
    if not np.isfinite(sigma) or sigma <= 0:
        return out
    z = (g - med) / sigma
    bad = np.where((np.abs(z) >= GAP_Z_THRESHOLD)
                   & (np.abs(g - med) >= MIN_GAP_ABS_DAYS))[0]
    for k in bad:
        zi = float(z[k])
        if not np.isfinite(zi):
            continue
        subj = str(joined.index[k])
        row_k = joined.iloc[k]
        date_a_str = str(row_k["a"].date())
        date_b_str = str(row_k["b"].date())
        out.append(Finding(
            anomaly_type=ANOMALY_TYPE,
            severity_score=calibrate(abs(zi), GAP_Z_THRESHOLD, 10.0),
            raw_score=abs(zi),
            network=network,
            timepoint=f"{tp_a} <-> {tp_b}",
            site_id=site_from_subjectid(subj),
            subjectid=subj,
            variable=f"{lbl_a} | {lbl_b}",
            variables_involved=f"{lbl_a}, {lbl_b}",
            observed_value=f"a={date_a_str}, b={date_b_str} (gap={g[k]:.1f} days)",
            expected_value=f"typical gap ~ {med:.1f} +/- {sigma:.1f} days (n={n})",
            explanation=(
                f"Gap between {lbl_a} and {lbl_b} is {zi:+.2f} sigmas from the "
                f"typical interval; very unusual collection cadence."
            ),
            method="MAD on paired date gaps",
            extra={"date_a": date_a_str, "date_b": date_b_str},
        ).to_row())
    return out
