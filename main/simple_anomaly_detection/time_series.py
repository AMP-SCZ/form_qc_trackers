"""Detector 2 - Time series outlier detection.

Scans within-subject trajectories for unusual peaks/drops. The trick
the prompt asks for: only flag peaks of a magnitude that is *unusual
for this variable*. A variable that bounces every visit shouldn't
flood the report.

Approach:
    * Build a long table per (subject, variable, timepoint).
    * For each variable, compute consecutive within-subject deltas.
    * Compute MAD-scaled sigma of the deltas *within each (tp_prev,
      tp_now) pair* -- pooling across all visit gaps would inflate
      sigma with the natural drift that grows with gap length.
    * For each subject, compute |delta - median_delta| / sigma_delta.
    * Compute peak_rarity per (variable, tp-pair). Variables with
      peak_rarity > 0.05 are "noisy" within that gap -- skip them.
    * Flag subjects whose |delta_z| >= Z_THRESHOLD.

Severity comes from the magnitude alone (calibrated z). An earlier
version added a "rarity boost" but it double-counted the gating
signal -- a bare-threshold flag on a calm variable could outrank a
much larger jump on a moderately calm one. The peak_rarity filter
already screens noisy variables out; that is enough.

Discovery peer: ``qc_types/discovery/longitudinal_delta_checks.py``
plus ``trajectory_shape_checks.py``. Same family of question, simpler
algorithm here.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .common import (
    Finding, calibrate, classify_columns, matches_excluded_substrings,
    robust_center_scale, severity_from_z, site_from_subjectid,
    to_numeric_clean, short,
)

ANOMALY_TYPE = "time_series_outlier"
Z_THRESHOLD = 4.0
MIN_DELTAS = 30
MAX_PEAK_RARITY = 0.05    # variables noisier than this are skipped
MIN_TP_PER_SUBJECT = 2

# Per-detector exclusion list (case-insensitive substring match against
# variable name). Add strings here to suppress this detector's findings
# on those variables without touching the shared EXCLUDED_VARIABLE_SUBSTRINGS
# in common.py.
EXTRA_EXCLUDED_SUBSTRINGS: tuple = ()


# Chronologically ordered AMPSCZ timepoints. floating / conversion are NOT
# chronological -- they're event-based forms -- so the delta detector skips
# them entirely. Unknown timepoints sort alphabetically AFTER the known
# canonical order so the result is deterministic across Python runs (no
# hash-based key) even when new tps appear in the data.
_TP_ORDER = (
    "screening", "baseline",
    "month1", "month2", "month3", "month4", "month5", "month6",
    "month7", "month8", "month9", "month10", "month11", "month12",
    "month18", "month24",
)
_TP_RANK = {tp: i for i, tp in enumerate(_TP_ORDER)}
_TP_NONCHRON: Tuple[str, ...] = ("floating", "conversion")


def _tp_rank(tp: str) -> Tuple[int, str]:
    """Sort key: known canonical TPs first by their canonical index, then
    unknown TPs alphabetically. Returning a tuple keeps the key fully
    deterministic across processes."""
    if tp in _TP_RANK:
        return (0, f"{_TP_RANK[tp]:04d}")
    return (1, str(tp))


def _is_chronological(tp: str) -> bool:
    return tp not in _TP_NONCHRON


def detect_all(per_slice: Dict, classify: Dict = None, **_) -> List[dict]:
    """Detect from the wide per-(network, timepoint) frames in ``per_slice``.

    ``classify``: optional dict keyed by (network, tp) holding the
    pre-computed classify_columns() result. When supplied, we skip
    the re-classification per slice (a big win at scale)."""
    rows: List[dict] = []
    by_network: Dict[str, List] = {}
    classify = classify or {}
    for (network, tp), df in per_slice.items():
        if df is None or df.empty or "subjectid" not in df.columns:
            continue
        if not _is_chronological(tp):
            continue
        by_network.setdefault(network, []).append((tp, df, classify.get((network, tp))))

    for network, items in by_network.items():
        items.sort(key=lambda t: _tp_rank(t[0]))
        try:
            rows.extend(_detect_network(network, items))
        except Exception as e:
            print(f"  [time_series] WARN network {network}: {e}")
    return rows


def _detect_network(network: str, ordered_items: List) -> List[dict]:
    if not ordered_items:
        return []
    numeric_cols: set = set()
    for _, df, cats in ordered_items:
        if cats is None:
            cats = classify_columns(df)
        numeric_cols.update(cats["numeric"])
    if EXTRA_EXCLUDED_SUBSTRINGS:
        numeric_cols = {c for c in numeric_cols
                        if not matches_excluded_substrings(c, EXTRA_EXCLUDED_SUBSTRINGS)}
    if not numeric_cols:
        return []

    # Derive a stable integer rank from the already-sorted item order so
    # the DataFrame column assignment stays scalar. _tp_rank() returns a
    # tuple for sorting, which broadcasts wrong when assigned as a column.
    tp_to_int_rank = {tp: i for i, (tp, _, _) in enumerate(ordered_items)}

    stacked = []
    for tp, df, _ in ordered_items:
        keep = ["subjectid"] + [c for c in df.columns if c in numeric_cols]
        sub = df[keep].copy()
        sub["_tp"] = tp
        sub["_tp_rank"] = tp_to_int_rank[tp]
        stacked.append(sub)
    long_df = pd.concat(stacked, ignore_index=True, sort=False)
    long_df = long_df.sort_values(["subjectid", "_tp_rank"]).reset_index(drop=True)

    out: List[dict] = []
    for col in numeric_cols:
        try:
            out.extend(_detect_variable(long_df, col, network))
        except Exception as e:
            print(f"  [time_series] WARN var {network}/{col}: {e}")
    return out


def _detect_variable(long_df: pd.DataFrame, col: str, network: str) -> List[dict]:
    if col not in long_df.columns:
        return []
    num = to_numeric_clean(long_df[col])
    if num.notna().sum() < MIN_DELTAS:
        return []

    work = long_df[["subjectid", "_tp", "_tp_rank"]].copy()
    work["v"] = num
    grp = work.groupby("subjectid", sort=False)
    work["v_prev"] = grp["v"].shift(1)
    work["tp_prev"] = grp["_tp"].shift(1)
    work["delta"] = work["v"] - work["v_prev"]
    work["pair"] = work["tp_prev"].astype(str) + "->" + work["_tp"].astype(str)

    has_delta = work["delta"].notna() & work["tp_prev"].notna()
    if not has_delta.any():
        return []

    # MAD per (tp_prev, tp_now) pair -- otherwise baseline->month1 pools
    # with month12->month18 and the gap-driven natural drift inflates sigma.
    rows: List[dict] = []
    for pair_name, sub in work.loc[has_delta].groupby("pair", sort=False):
        d_vals = sub["delta"].dropna().to_numpy(dtype=float)
        if d_vals.size < MIN_DELTAS:
            continue
        med, sigma = robust_center_scale(d_vals)
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        z = (sub["delta"] - med) / sigma
        peak_rarity = float((z.abs() >= Z_THRESHOLD).sum() / max(d_vals.size, 1))
        if peak_rarity > MAX_PEAK_RARITY:
            continue
        bad_mask = z.abs() >= Z_THRESHOLD
        if not bad_mask.any():
            continue

        # Pre-extract arrays once so we don't pay .loc per finding.
        bad_sub = sub.loc[bad_mask, ["subjectid", "_tp", "tp_prev", "v", "v_prev", "delta"]]
        z_vals = z[bad_mask].to_numpy(dtype=float)

        for k, (_, row) in enumerate(bad_sub.iterrows()):
            zi = float(z_vals[k])
            if not np.isfinite(zi):
                continue
            subj = str(row["subjectid"])
            tp_now = str(row["_tp"])
            tp_prev = str(row["tp_prev"])
            v_now = row["v"]
            v_prev = row["v_prev"]
            d = float(row["delta"])
            severity = severity_from_z(zi, Z_THRESHOLD, 10.0)
            rows.append(Finding(
                anomaly_type=ANOMALY_TYPE,
                severity_score=severity,
                raw_score=abs(zi),
                network=network,
                timepoint=f"{tp_prev}->{tp_now}",
                site_id=site_from_subjectid(subj),
                subjectid=subj,
                variable=col,
                observed_value=f"{short(v_prev)} -> {short(v_now)} (delta {d:+.4g})",
                expected_value=f"typical {tp_prev}->{tp_now} delta median {med:+.4g}, sigma {sigma:.4g}",
                explanation=(
                    f"Within-subject jump z = {zi:+.2f} (vs the typical "
                    f"{tp_prev}->{tp_now} change). Peak rarity for this "
                    f"variable over the same gap is {peak_rarity:.2%}, so "
                    f"the spike is not just a noisy variable bouncing."
                ),
                method="MAD on within-subject deltas, stratified by tp pair",
                extra={"peak_rarity": round(peak_rarity, 4),
                       "n_in_gap": int(d_vals.size)},
            ).to_row())
    return rows
