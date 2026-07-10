"""Detector 8 - 3D subspace cluster-deviation detection.

Simulates a reviewer paging through 3D scatter plots of every combination of
three MUTUALLY-CORRELATED numeric variables, looking for the triples where a
small number of large, dense clusters form, and flagging the few points that
sit apart from all of them. A point can look normal on every single axis (and
even every pair) yet sit in empty space for a specific (v1, v2, v3) trio --
this detector is the interpretable middle ground between `correlative` (2D
pairwise) and the full-dimensional Mahalanobis/PCA detectors in
`qc_types/discovery`: a flag names three plottable variables.

ALGORITHM, per network:
  1. Pick the most-complete numeric variables and compute their pairwise
     (Spearman) correlation on the network stack.
  2. Form candidate TRIPLES: only trios whose three pairwise correlations all
     clear MIN_TRIPLE_CORR (so a 3D cluster actually means something), ranked
     by the weakest of the three correlations, capped at TOP_K_TRIPLES.
  3. For each triple, within each (network, timepoint) slice (clustering per
     timepoint so natural visit-to-visit drift doesn't smear the cloud):
       a. take joint-complete points, robust-standardize each axis (median/MAD)
          so the three variables share a scale;
       b. DBSCAN with an eps adapted from the cloud's own k-distances, so it
          self-tunes to the data's density;
       c. STRUCTURE GATE -- only proceed when the picture actually holds: a
          small number (1..MAX_CLUSTERS) of clusters each >= MIN_CLUSTER_SIZE
          hold the BULK of points (noise fraction <= MAX_NOISE_FRACTION).
          A shapeless cloud (no clusters, or half the points "noise") is
          skipped, exactly like a reviewer shrugging at a messy scatter.
       d. SCORE each noise point by how far it sits OFF the nearest cluster's
          SHAPE -- the deviation along the cluster's tight (low-variance)
          directions, i.e. the relationship the cluster encodes. Deviation
          merely ALONG a cluster's high-variance axis is a distribution tail
          and is ignored (so an unusually old subject on a clean age
          trajectory isn't flagged); an isotropic blob with no tight direction
          falls back to full distance. That deviation is then multiplied by a
          SUB-LINEAR weight on the cluster's SIZE (deviating from a 1000-point
          cluster outranks the same deviation from a 10-point one, but only
          log-scaled -- not 100x). Only points >= DEV_THRESHOLD sigmas out are
          emitted.
  4. De-dup to one finding per subject (the most severe triple + a count of
     how many triples flagged them), so one off subject can't flood the top
     of the report.

scikit-learn (DBSCAN) is OPTIONAL: if it is not installed, the detector logs
a one-line skip and returns nothing rather than crashing the run (same pattern
as isolation_forest_checks).
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import pandas as pd

from .common import (
    Finding, calibrate, classify_columns, matches_excluded_substrings,
    pairwise_nan_spearman, robust_center_scale, site_from_subjectid,
    to_numeric_clean, short,
)

ANOMALY_TYPE = "cluster_3d_outlier"

EXTRA_EXCLUDED_SUBSTRINGS: tuple = ()

# --- triple selection -------------------------------------------------------
N_NUMERIC = 40              # most-complete numerics considered per network
MIN_TRIPLE_CORR = 0.4       # all three pairwise |Spearman| must clear this
TOP_K_TRIPLES = 300         # cap; strongest (by weakest-of-three |r|) kept
MIN_DISTINCT_PER_AXIS = 10  # discreteness guard: skip lattice-like ordinal axes

# --- per-(network, timepoint) clustering ------------------------------------
MIN_POINTS = 50             # joint-complete points needed to cluster a triple
DBSCAN_MIN_SAMPLES = 5      # "dense" = at least this many neighbours within eps
EPS_PERCENTILE = 80         # eps from this percentile of the k-distance curve
                            # (generous enough to absorb each cluster's natural
                            # halo, so only genuinely-isolated points are noise)
EPS_SAMPLE_CAP = 1000       # subsample size for the O(n^2) eps estimate
MIN_CLUSTERS = 1
MAX_CLUSTERS = 8
MIN_CLUSTER_SIZE = 10       # clusters smaller than this don't count as structure
MAX_NOISE_FRACTION = 0.30   # "a few points deviate", not half the data

# --- scoring ----------------------------------------------------------------
# Deviation is the Mahalanobis distance to the nearest cluster, in that
# cluster's own covariance metric -- so a point off a TIGHT relationship's
# axis scores high even when the cluster is elongated (Euclidean distance to
# the centroid would be swamped by the cluster's long axis). COV_FLOOR is a
# ridge added to each cluster covariance: it stops a numerically-flat
# direction from producing an infinite distance and keeps the detector
# conservative (a direction is never treated as tighter than this, in the
# globally-robust-standardized units the axes are already in).
COV_FLOOR = 0.15
# A cluster direction is a "tight constraint" if its variance is <= this
# fraction of the cluster's largest-variance direction. Deviation is scored
# ONLY along the tight directions (off the relationship the cluster encodes);
# deviation ALONG a high-variance direction is just a distribution tail and is
# ignored, so e.g. an unusually old subject on a clean age trajectory is not
# flagged. If a cluster is isotropic (no tight direction), we fall back to the
# full distance so points far from a round blob are still caught.
DOMINANT_REL = 0.2
SIZE_REF_N = 100            # cluster size giving a size weight of ~1.0
SIZE_FACTOR_LO, SIZE_FACTOR_HI = 0.6, 1.4   # sub-linear, clamped (10-pt vs 1000-pt
                                            # cluster differ, but not proportionally)
# Conservative flag bar so only points clearly outside all clusters reach the
# report (a cluster-edge point at ~2-3 radii must NOT crowd the top). 4 radii
# mirrors the |z|>=4 bar the other detectors use.
DEV_THRESHOLD = 4.0         # normalized deviation (cluster radii) at the flag bar -> sev 50
DEV_EXTREME = 10.0          # -> sev 95
# When exclude_marginal_outliers=True (used by the longitudinal detector), skip
# a point whose value on ANY axis is itself a >= this-many-sigma 1D outlier --
# its off-cluster position is then "explained" by a single extreme value
# (already owned by standard_outlier), not a genuine JOINT pattern. Because the
# axes are robust-standardized, the standardized coordinate IS the marginal z.
MARGINAL_Z = 4.0


def detect_all(per_slice: Dict, classify: Dict = None, **_) -> List[dict]:
    try:
        from sklearn.cluster import DBSCAN  # noqa: F401
    except ImportError:
        print("  [cluster_3d] scikit-learn not installed; skipping "
              "(pip install scikit-learn to enable the 3D cluster detector).")
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
            print(f"  [cluster_3d] WARN network {network}: {e}")
    return _dedup(rows)


def _detect_network(network: str, items: List) -> List[dict]:
    stacked = pd.concat([df for _, df, _ in items], ignore_index=True, sort=False)

    numset: set = set()
    for _, df, cats in items:
        c = cats if cats is not None else classify_columns(df)
        numset.update(c["numeric"])
    numerics = [c for c in numset
                if c in stacked.columns
                and not matches_excluded_substrings(c, EXTRA_EXCLUDED_SUBSTRINGS)]
    if len(numerics) < 3:
        return []
    completeness = {c: int(to_numeric_clean(stacked[c]).notna().sum()) for c in numerics}
    numerics = sorted(numerics, key=lambda c: completeness[c], reverse=True)[:N_NUMERIC]
    if len(numerics) < 3:
        return []

    # Pairwise Spearman on the stack -> correlated triples.
    N, V = len(stacked), len(numerics)
    data = np.empty((N, V), dtype=np.float64)
    for k, c in enumerate(numerics):
        data[:, k] = to_numeric_clean(stacked[c]).to_numpy(dtype=np.float64)
    corr, _ = pairwise_nan_spearman(data, min_joint_n=MIN_POINTS)
    absr = np.abs(np.nan_to_num(corr, nan=0.0))

    triples = []
    for i in range(V):
        for j in range(i + 1, V):
            if absr[i, j] < MIN_TRIPLE_CORR:
                continue
            for k in range(j + 1, V):
                if absr[i, k] >= MIN_TRIPLE_CORR and absr[j, k] >= MIN_TRIPLE_CORR:
                    triples.append((min(absr[i, j], absr[i, k], absr[j, k]), i, j, k))
    if not triples:
        return []
    triples.sort(reverse=True)
    triples = triples[:TOP_K_TRIPLES]

    # The dominant cost is (triples x slices x DBSCAN); TOP_K_TRIPLES is the
    # hard cap on the first factor. Log the work so real-scale runtime is
    # observable (the demo's 1 triple x small N is not representative).
    t0 = time.time()
    from sklearn.cluster import DBSCAN
    out: List[dict] = []
    for tp, df, _ in items:
        subj_arr = df["subjectid"].astype(str).to_numpy()
        colcache: Dict[str, np.ndarray] = {}
        for (_, i, j, k) in triples:
            for idx in (i, j, k):
                c = numerics[idx]
                if c not in colcache:
                    colcache[c] = (to_numeric_clean(df[c]).to_numpy(dtype=np.float64)
                                   if c in df.columns else None)
        for (_, i, j, k) in triples:
            cols = (numerics[i], numerics[j], numerics[k])
            arrs = [colcache.get(c) for c in cols]
            if any(a is None for a in arrs):
                continue
            out.extend(_cluster_triple(network, tp, cols, arrs, subj_arr, DBSCAN))
    print(f"  [cluster_3d] {network}: {len(triples)} correlated triples x "
          f"{len(items)} slices in {time.time() - t0:.1f}s")
    return out


def _size_factor(size: int) -> float:
    """Sub-linear, clamped weight on cluster size: log10(size)/log10(SIZE_REF_N),
    clamped to [LO, HI]. A 1000-point cluster outweighs a 10-point one, but
    only ~2-3x across two orders of magnitude -- not proportionally."""
    return float(np.clip(np.log10(max(size, 2)) / np.log10(SIZE_REF_N),
                         SIZE_FACTOR_LO, SIZE_FACTOR_HI))


def _adaptive_eps(Xs: np.ndarray, k: int) -> float:
    """eps = a percentile of each point's distance to its k-th neighbour
    (the standard DBSCAN k-distance heuristic), estimated on a subsample to
    keep the O(n^2) distance matrix bounded."""
    n = Xs.shape[0]
    if n <= k + 1:
        return float("nan")
    if n > EPS_SAMPLE_CAP:
        rng = np.random.default_rng(0)
        sample = Xs[rng.choice(n, EPS_SAMPLE_CAP, replace=False)]
    else:
        sample = Xs
    diff = sample[:, None, :] - sample[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    dist.sort(axis=1)
    kth = dist[:, k]  # column 0 is the self-distance (0)
    return float(np.percentile(kth, EPS_PERCENTILE))


def _cluster_triple(network, tp, cols, arrs, subj_arr, DBSCAN,
                    anomaly_type: str = ANOMALY_TYPE,
                    exclude_marginal_outliers: bool = False) -> List[dict]:
    """Cluster one 3-column triple and score the off-cluster points. Shared by
    the within-timepoint detector and the longitudinal (cross-timepoint)
    detector -- the latter passes (variable@timepoint) column names in ``cols``
    and its own ``anomaly_type``."""
    a, b, c = arrs
    joint = ~(np.isnan(a) | np.isnan(b) | np.isnan(c))
    nj = int(joint.sum())
    if nj < MIN_POINTS:
        return []
    idx = np.where(joint)[0]
    X = np.column_stack([a[joint], b[joint], c[joint]])

    # Discreteness guard: lattice-like ordinal axes form grid "clusters".
    for d in range(3):
        if np.unique(X[:, d]).size < MIN_DISTINCT_PER_AXIS:
            return []

    # Robust-standardize each axis so the three share a scale.
    Xs = np.empty_like(X)
    for d in range(3):
        med, sigma = robust_center_scale(X[:, d])
        if not np.isfinite(sigma) or sigma <= 0:
            return []
        Xs[:, d] = (X[:, d] - med) / sigma

    eps = _adaptive_eps(Xs, DBSCAN_MIN_SAMPLES)
    if not np.isfinite(eps) or eps <= 0:
        return []
    labels = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(Xs)

    # Structure gate -- only score where the "few large dense clusters" picture holds.
    big = [cid for cid in set(labels) if cid != -1 and int(np.sum(labels == cid)) >= MIN_CLUSTER_SIZE]
    if not (MIN_CLUSTERS <= len(big) <= MAX_CLUSTERS):
        return []
    noise = labels == -1
    n_noise = int(noise.sum())
    if n_noise == 0 or (1.0 - n_noise / nj) < (1.0 - MAX_NOISE_FRACTION):
        return []

    # Per-cluster centroid + eigendecomposition (for tight-direction scoring).
    cstats = {}
    for cid in big:
        pts = Xs[labels == cid]
        centroid = pts.mean(axis=0)
        cov = np.cov(pts.T) if pts.shape[0] >= 4 else np.eye(3)
        evals, evecs = np.linalg.eigh(np.atleast_2d(cov))  # ascending eigenvalues
        cstats[cid] = (centroid, np.maximum(evals, 0.0), evecs, int(pts.shape[0]))

    out: List[dict] = []
    for li in np.where(noise)[0]:
        p = Xs[li]
        # Skip points that are a 1D outlier on any axis (their off-cluster
        # position is explained by a single extreme value -> standard_outlier's
        # job). Leaves only the genuinely JOINT anomalies. Xs is robust-
        # standardized, so |coordinate| is the marginal z.
        if exclude_marginal_outliers and float(np.abs(p).max()) >= MARGINAL_Z:
            continue
        best = None  # (deviation, cluster_size)
        for centroid, evals, evecs, size in cstats.values():
            lam_max = float(evals.max()) if evals.size else 0.0
            if lam_max <= 0:
                continue
            d = p - centroid
            tight = evals <= DOMINANT_REL * lam_max
            use = tight if tight.any() else np.ones(evals.shape[0], dtype=bool)
            dev2 = 0.0
            for di in range(evals.shape[0]):
                if not use[di]:
                    continue
                proj = float(d @ evecs[:, di])
                dev2 += proj * proj / max(float(evals[di]), COV_FLOOR ** 2)
            dev = float(np.sqrt(dev2))
            if best is None or dev < best[0]:
                best = (dev, size)
        if best is None:
            continue
        dev, size = best
        # Emission gate: only points clearly outside ALL clusters (>= the
        # flag bar in cluster-radii) are reported, mirroring the |z|>=4 gate
        # the other detectors use -- so cluster-edge "noise" points don't
        # clutter the sheet. The size weight below only modulates SEVERITY.
        if dev < DEV_THRESHOLD:
            continue
        size_factor = _size_factor(size)
        raw = dev * size_factor
        sev = calibrate(raw, DEV_THRESHOLD, DEV_EXTREME)
        if sev <= 0:
            continue
        gi = idx[li]
        subj = str(subj_arr[gi])
        rv = X[li]
        out.append(Finding(
            anomaly_type=anomaly_type,
            severity_score=sev,
            raw_score=raw,
            network=network,
            timepoint=tp,
            site_id=site_from_subjectid(subj),
            subjectid=subj,
            variable=" | ".join(cols),
            variables_involved=", ".join(cols),
            observed_value=f"{cols[0]}={short(rv[0])}, {cols[1]}={short(rv[1])}, {cols[2]}={short(rv[2])}",
            expected_value=(f"nearest of {len(cstats)} dense cluster(s): n={size}, "
                            f"{dev:.1f} sigmas off its shape"),
            explanation=(
                f"In the 3D space of {cols[0]}, {cols[1]}, {cols[2]} (mutually correlated), "
                f"this subject sits {dev:.1f} sigmas OFF the shape of the nearest of "
                f"{len(cstats)} dense cluster(s) (nearest n={size}) -- it violates the "
                f"relationship the cluster encodes (not merely far along its main axis)."
            ),
            method="DBSCAN on robust-standardized 3D triple; off-relationship deviation (cluster tight directions) x sub-linear size weight",
            extra={"n_clusters": len(cstats), "nearest_cluster_n": size,
                   "deviation_radii": round(dev, 2), "size_factor": round(size_factor, 2)},
        ).to_row())
    return out


def _dedup(rows: List[dict]) -> List[dict]:
    """One finding per (subject, network): the most severe triple, plus a
    count of how many triples flagged the subject (a point off in many
    triples is usually one bad VALUE, and shouldn't occupy many top slots)."""
    if not rows:
        return rows
    from collections import defaultdict
    groups: Dict = defaultdict(list)
    for r in rows:
        groups[(str(r.get("subjectid", "")), str(r.get("network", "")))].append(r)
    out: List[dict] = []
    for grp in groups.values():
        rep = dict(max(grp, key=lambda r: float(r.get("severity_score", 0) or 0)))
        rep["corroborating_triples"] = len(grp)
        allvars = sorted({v.strip() for r in grp
                          for v in str(r.get("variables_involved", "")).split(",") if v.strip()})
        rep["all_triple_variables"] = ", ".join(allvars[:20])
        out.append(rep)
    return out
