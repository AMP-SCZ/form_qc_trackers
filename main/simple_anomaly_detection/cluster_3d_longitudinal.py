"""Detector 9 - Longitudinal (cross-timepoint) 3D subspace cluster deviation.

The sibling of `cluster_3d`, but the three axes of each triple are
(variable, TIMEPOINT) columns that may come from DIFFERENT timepoints --
e.g. (weight@baseline, weight@month6, weight@month12) or the cross-variable
(score_total@baseline, anxiety@month6, function@month12). It catches a
subject whose *cross-time pattern* is anomalous even though each individual
value is normal.

Approach: pivot the per-(network, timepoint) slices into ONE wide frame with
one row per subject and columns keyed `{variable}@{timepoint}`. Each axis is a
fixed (variable, timepoint), so its distribution is stable (no visit-drift
smearing -- which is exactly why the within-timepoint detector clusters per
slice and this one can safely pivot). Then it forms triples of those columns
that are (a) mutually correlated and (b) span at least MIN_TIMEPOINTS distinct
timepoints (all-same-timepoint trios are `cluster_3d`'s job, not duplicated
here), and clusters/scores each triple with the SAME machinery as `cluster_3d`
(DBSCAN -> structure gate -> Mahalanobis deviation x sub-linear size weight ->
one finding per subject).

Overlap notes: same-variable-across-time trios partially overlap `time_series`
(which scores consecutive deltas); the unique value here is the cross-VARIABLE
cross-time combinations. scikit-learn (DBSCAN) is optional; the detector skips
gracefully if it is absent.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import pandas as pd

from . import cluster_3d
from .common import (
    classify_columns, matches_excluded_substrings, pairwise_nan_spearman,
    to_numeric_clean,
)

ANOMALY_TYPE = "cluster_3d_longitudinal_outlier"

EXTRA_EXCLUDED_SUBSTRINGS: tuple = ()

N_COLS = 40                 # top-N most-complete (variable@timepoint) columns
MIN_TRIPLE_CORR = 0.4       # all three pairwise |Spearman| must clear this
TOP_K_TRIPLES = 300
MIN_POINTS = 50             # joint-complete SUBJECTS needed per triple
MIN_TIMEPOINTS = 2          # a triple must span >= this many distinct timepoints
                            # (all-same-tp trios belong to cluster_3d, not here)


def detect_all(per_slice: Dict, classify: Dict = None, **_) -> List[dict]:
    try:
        from sklearn.cluster import DBSCAN  # noqa: F401
    except ImportError:
        print("  [cluster_3d_longitudinal] scikit-learn not installed; skipping.")
        return []
    classify = classify or {}
    by_network: Dict[str, List] = {}
    for (network, tp), df in per_slice.items():
        if df is None or df.empty or "subjectid" not in df.columns:
            continue
        by_network.setdefault(network, []).append((tp, df, classify.get((network, tp))))

    rows: List[dict] = []
    for network, items in by_network.items():
        try:
            rows.extend(_detect_network(network, items))
        except Exception as e:
            print(f"  [cluster_3d_longitudinal] WARN network {network}: {e}")
    return cluster_3d._dedup(rows)


def _build_wide(items: List) -> pd.DataFrame:
    """One row per subject; columns `{variable}@{timepoint}` for numeric vars."""
    wide = None
    for tp, df, cats in items:
        c = cats if cats is not None else classify_columns(df)
        nums = [x for x in c["numeric"]
                if x in df.columns
                and not matches_excluded_substrings(x, EXTRA_EXCLUDED_SUBSTRINGS)]
        if not nums:
            continue
        sub = df[["subjectid"] + nums].drop_duplicates("subjectid")
        sub = sub.set_index(sub["subjectid"].astype(str))[nums]
        sub = sub.rename(columns={x: f"{x}@{tp}" for x in nums})
        wide = sub if wide is None else wide.join(sub, how="outer")
    return wide


def _detect_network(network: str, items: List) -> List[dict]:
    from sklearn.cluster import DBSCAN

    wide = _build_wide(items)
    if wide is None or wide.shape[1] < 3:
        return []

    cols_all = list(wide.columns)
    cleaned = {col: to_numeric_clean(wide[col]) for col in cols_all}
    completeness = {col: int(cleaned[col].notna().sum()) for col in cols_all}
    cols = sorted(cols_all, key=lambda c: completeness[c], reverse=True)[:N_COLS]
    if len(cols) < 3:
        return []

    N, V = len(wide), len(cols)
    data = np.empty((N, V), dtype=np.float64)
    for k, col in enumerate(cols):
        data[:, k] = cleaned[col].to_numpy(dtype=np.float64)
    corr, _ = pairwise_nan_spearman(data, min_joint_n=MIN_POINTS)
    absr = np.abs(np.nan_to_num(corr, nan=0.0))
    col_tp = [c.rsplit("@", 1)[1] if "@" in c else "" for c in cols]

    # Correlated triples that span >= MIN_TIMEPOINTS distinct timepoints.
    triples = []
    for i in range(V):
        for j in range(i + 1, V):
            if absr[i, j] < MIN_TRIPLE_CORR:
                continue
            for k in range(j + 1, V):
                if absr[i, k] >= MIN_TRIPLE_CORR and absr[j, k] >= MIN_TRIPLE_CORR \
                        and len({col_tp[i], col_tp[j], col_tp[k]}) >= MIN_TIMEPOINTS:
                    triples.append((min(absr[i, j], absr[i, k], absr[j, k]), i, j, k))
    if not triples:
        return []
    triples.sort(reverse=True)
    triples = triples[:TOP_K_TRIPLES]

    subj_arr = wide.index.to_numpy()
    t0 = time.time()
    out: List[dict] = []
    for (_, i, j, k) in triples:
        triple_cols = (cols[i], cols[j], cols[k])
        arrs = [data[:, i], data[:, j], data[:, k]]
        tp_label = " / ".join(sorted({col_tp[i], col_tp[j], col_tp[k]}))
        out.extend(cluster_3d._cluster_triple(
            network, tp_label, triple_cols, arrs, subj_arr, DBSCAN,
            anomaly_type=ANOMALY_TYPE,
            # A point off the cross-time diagonal solely because of a single
            # extreme value (e.g. weight=320 at one visit) is already owned by
            # standard_outlier; surface only genuine cross-time JOINT patterns.
            exclude_marginal_outliers=True))
    print(f"  [cluster_3d_longitudinal] {network}: {len(triples)} cross-timepoint "
          f"triples in {time.time() - t0:.1f}s")
    return out
