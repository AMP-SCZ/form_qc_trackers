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
# Diversity cap on the triple selection (see _diversify_triples). A single
# tightly-correlated block of M variables yields C(M,3) triples that all clear
# the correlation gate, so a raw top-TOP_K_TRIPLES is consumed entirely by the
# few densest blocks -- a 30-item symptom scale alone gives C(30,3)=4060 triples
# and crowds every other correlated block out of the budget, so the output keeps
# naming the same handful of variables. The selection therefore caps how many
# SELECTED triples any one variable may appear in to this value, so the budget
# spreads across blocks; any leftover budget is then refilled with the strongest
# remaining triples, so total coverage never drops below the old top-K. Below the
# ~TOP_K_TRIPLES*3/N_NUMERIC "fair share" so the cap actively diversifies.
MAX_TRIPLES_PER_VAR = 15

# --- per-(network, timepoint) clustering ------------------------------------
MIN_POINTS = 50             # joint-complete points needed to cluster a triple
DBSCAN_MIN_SAMPLES = 5      # "dense" = at least this many neighbours within eps
EPS_PERCENTILE = 80         # eps from this percentile of the k-distance curve
                            # (generous enough to absorb each cluster's natural
                            # halo, so only genuinely-isolated points are noise)
EPS_SAMPLE_CAP = 1000       # subsample size for the O(n^2) eps estimate
MIN_CLUSTERS = 1
MAX_CLUSTERS = 4         # >4 dense clusters -> too busy a picture; the triple is
                        # skipped and contributes nothing to the output
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

# --- per-combo summary index (summarize_combos) -----------------------------
# A correlated block of M variables yields C(M,3) triples, so the same handful
# of variables recurs across many per-subject findings. summarize_combos folds
# those into ONE row per (network, variable-combo). COMBO_SEV_THRESHOLD is the
# severity bar (shared 0..100 scale) for that index's headline count -- "how
# many of this combo's flagged subjects are at least this severe" -- so a combo
# that recurs at HIGH severity sorts above one that recurs only at the flag bar.
COMBO_SEV_THRESHOLD = 80.0


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
    triples = _diversify_triples(triples, TOP_K_TRIPLES, MAX_TRIPLES_PER_VAR)

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
    # Report the distinct-variable count the selected triples span, not just the
    # triple count: a tight correlated block of M vars yields C(M,3) triples all
    # from the SAME M vars, so "300 triples spanning 9 variables" is the signal
    # that the output will keep naming the same handful (see summarize_combos).
    n_triple_vars = len({idx for (_, i, j, k) in triples for idx in (i, j, k)})
    print(f"  [cluster_3d] {network}: {len(triples)} correlated triples "
          f"spanning {n_triple_vars} distinct variables x "
          f"{len(items)} slices in {time.time() - t0:.1f}s")
    return out


def _diversify_triples(triples: List, top_k: int, max_per_var: int) -> List:
    """Pick up to ``top_k`` triples, spreading them across variables so one
    tightly-correlated block can't monopolise the budget.

    ``triples`` is the full candidate list, ALREADY sorted strongest-first
    (each item ``(weakest_corr, i, j, k)``). Two passes:

      1. DIVERSIFY -- walk strongest-first and keep a triple only while none of
         its three variables has already been selected ``max_per_var`` times.
         This reserves budget for weaker-but-distinct blocks: the densest block
         saturates its variables at the cap and then yields to the next block,
         so a small correlated group that the raw top-K never reached now gets
         examined.
      2. FILL -- top the selection back up to ``top_k`` with the strongest
         not-yet-selected triples, cap ignored. With ``max_per_var``=15 and the
         <= ``N_NUMERIC``=40 variables in play, pass 1 selects at most
         floor(40*15/3)=200 < ``top_k``, so pass 2 ALWAYS runs and the final
         count is always exactly ``top_k`` (when that many candidates exist).

    Net effect: the SAME number of triples is selected as the old
    ``triples[:top_k]``, but redundant triples from an over-represented block
    are reallocated to under-represented blocks (in the single-block degenerate
    case pass 2 simply refills the same block, so the result is identical to the
    old behaviour). A subject sitting off an over-represented block is still
    caught -- the per-subject dedup keeps its most-severe surviving triple -- so
    the reallocation broadens variable coverage without dropping subjects. With
    <= ``top_k`` candidate triples the cap can't bite, so the list is returned
    unchanged (e.g. the demo, whose single correlated trio yields one triple)."""
    if len(triples) <= top_k:
        return triples
    selected: List = []
    chosen: set = set()
    counts: Dict[int, int] = {}
    for pos, t in enumerate(triples):
        if len(selected) >= top_k:
            break
        _, i, j, k = t
        if (counts.get(i, 0) >= max_per_var or counts.get(j, 0) >= max_per_var
                or counts.get(k, 0) >= max_per_var):
            continue
        selected.append(t)
        chosen.add(pos)
        for idx in (i, j, k):
            counts[idx] = counts.get(idx, 0) + 1
    if len(selected) < top_k:
        for pos, t in enumerate(triples):
            if len(selected) >= top_k:
                break
            if pos not in chosen:
                selected.append(t)
    return selected


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
    # Canonicalize leg order so a given trio of variables is always LABELLED the
    # same way, regardless of the per-network completeness ranking that produced
    # the (i<j<k) index order. Without it the same {A,B,C} reads "A | B | C" in
    # one network and "B | A | C" in the other (each network sorts its numerics
    # by its own completeness), so combined_ranked shows the same three
    # variables in a different order -- the redundancy this removes. DBSCAN, the
    # k-distance eps estimate, and the Mahalanobis deviation are all invariant
    # to axis permutation (Euclidean distances and the covariance eigen-structure
    # don't depend on column order), so this changes ONLY the emitted labels,
    # never which points are flagged. The sort key matches summarize_combos'
    # ``sorted(legs)``, so a per-subject row's ``variable`` and its combo-summary
    # key now agree.
    order = sorted(range(3), key=lambda t: str(cols[t]))
    cols = tuple(cols[t] for t in order)
    arrs = [arrs[t] for t in order]
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


def summarize_combos(rows: List[dict], top_k: int = 50,
                     sev_threshold: float = COMBO_SEV_THRESHOLD) -> List[dict]:
    """Collapse the per-subject findings into ONE row per variable combo.

    Why: a tightly-correlated block of M variables yields C(M,3) triples that
    all clear the correlation gate, so the same handful of variables recurs
    across many per-subject rows. This is an additive INDEX over those rows --
    one row per (network, sorted-combo). Each row carries an
    ``n_subjects_sev_ge_<sev_threshold>`` column: HOW OFTEN the combo shows up
    above the severity bar, i.e. how many of its flagged subjects are at least
    ``sev_threshold`` severe (shared 0..100 scale). Rows are ranked by that
    high-severity count first, then by total ``n_subjects`` and ``max_severity``,
    so a combo that "shows up a lot" at high severity floats to the top. A
    ``top_subjects`` column lists the high-severity subjects ``subjectid(severity)``
    descending (capped at ``top_k``, with a ``(+N more)`` overflow note). The
    full per-subject detail stays in the main sheet; nothing here changes what
    is detected.

    ``rows`` is the de-duplicated finding rows (each subject's single most-
    severe triple), so a subject appears under exactly one combo and
    ``n_subjects`` never double-counts. Works for both ``cluster_3d`` and
    ``cluster_3d_longitudinal`` rows (the latter's ``variable`` legs are
    ``var@timepoint`` columns -- still grouped by their sorted set)."""
    if not rows:
        return []
    from collections import defaultdict

    def _sev(r) -> float:
        try:
            v = float(r.get("severity_score", 0) or 0)
        except (TypeError, ValueError):
            return 0.0
        # NaN is truthy, so `nan or 0` -> nan slips past the `or 0`; reject it
        # explicitly (a single NaN would otherwise poison mean/sort/top_subjects).
        return v if np.isfinite(v) else 0.0

    groups: Dict = defaultdict(list)
    for r in rows:
        legs = [p.strip() for p in str(r.get("variable", "")).split("|") if p.strip()]
        combo = " | ".join(sorted(legs))
        groups[(str(r.get("network", "")), combo)].append(r)

    # Headline count column, named so the threshold is visible in the sheet
    # (e.g. "n_subjects_sev_ge_80"); ``:g`` keeps an integer threshold integer.
    count_col = f"n_subjects_sev_ge_{sev_threshold:g}"

    out: List[dict] = []
    for (network, combo), grp in groups.items():
        grp_sorted = sorted(grp, key=_sev, reverse=True)
        sevs = [_sev(r) for r in grp_sorted]
        n_above = sum(1 for s in sevs if s >= sev_threshold)
        bits = [f"{r.get('subjectid', '')}({_sev(r):.1f})" for r in grp_sorted[:top_k]]
        listed = "; ".join(bits)
        if len(grp_sorted) > top_k:
            listed += f"; (+{len(grp_sorted) - top_k} more)"
        tps = sorted({str(r.get("timepoint", "")) for r in grp_sorted
                      if str(r.get("timepoint", ""))})
        out.append({
            "network": network,
            "variables": combo,
            "n_subjects": len(grp_sorted),
            count_col: n_above,
            "max_severity": round(max(sevs), 2) if sevs else 0.0,
            "mean_severity": round(sum(sevs) / len(sevs), 2) if sevs else 0.0,
            "timepoints": ", ".join(tps),
            "top_subjects": listed,
        })
    # High-severity recurrence first, then total subjects, then peak severity --
    # so a combo that flags the most subjects ABOVE the bar leads the index.
    # When no combo clears the bar (count all 0), this degrades exactly to the
    # prior (n_subjects, max_severity) ordering.
    out.sort(key=lambda d: (d[count_col], d["n_subjects"], d["max_severity"]),
             reverse=True)
    return out


def combos_for_combined(rows: List[dict], anomaly_type: str,
                        sev_threshold: float = COMBO_SEV_THRESHOLD,
                        top_k: int = 50) -> List[dict]:
    """Shape the per-combo summary as Finding-schema rows for combined_ranked.

    The combined headline shows ONE row per (network, variable-combo) for the
    cluster detectors instead of one row per flagged subject, so a correlated
    block that flags many subjects becomes a handful of distinct-combo rows
    rather than the same few variables repeated down the sheet (the per-subject
    detail stays in the per-type sheet, and the same numbers populate the
    standalone ``*_combos`` index). This reuses ``summarize_combos`` for the
    grouping/ranking/subject-listing, then maps each combo to a row that sits
    next to the other detectors' per-subject rows:
      - ``severity_score`` = the combo's MAX per-subject severity, so it ranks
        on the same 0..100 scale the combined sort uses;
      - ``subjectid`` names the subject count (``"(N subjects)"``), mirroring
        site_network's ``"(N sites)"`` collapse; the concrete subjects are in
        ``top_subjects``;
      - ``variable`` is the combo, so ``_diversify_combined``'s per-(network,
        variable) cap still applies (a no-op here -- already one row per combo).
    """
    summary = summarize_combos(rows, top_k=top_k, sev_threshold=sev_threshold)
    count_col = f"n_subjects_sev_ge_{sev_threshold:g}"
    out: List[dict] = []
    for s in summary:
        n = s["n_subjects"]
        out.append({
            "anomaly_type": anomaly_type,
            "severity_score": s["max_severity"],
            "raw_score": "",
            "network": s["network"],
            "timepoint": s["timepoints"],
            "site_id": "",
            "subjectid": f"({n} subjects)" if n != 1 else f"({n} subject)",
            "variable": s["variables"],
            "variables_involved": s["variables"].replace(" | ", ", "),
            "observed_value": "",
            "expected_value": f"{n} subject(s) off this 3D combo",
            "explanation": (
                f"{n} subject(s) flagged off the 3D combo {s['variables']} "
                f"(max severity {s['max_severity']}, mean {s['mean_severity']}, "
                f"{s[count_col]} at/above severity {sev_threshold:g}). "
                f"Top: {s['top_subjects']}"),
            "method": "per-combo summary of cluster per-subject findings",
            "n_subjects": n,
            count_col: s[count_col],
            "max_severity": s["max_severity"],
            "mean_severity": s["mean_severity"],
            "timepoints": s["timepoints"],
            "top_subjects": s["top_subjects"],
        })
    return out
