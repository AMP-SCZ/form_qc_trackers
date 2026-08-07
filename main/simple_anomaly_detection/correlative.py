"""Detector 3 - Correlative outlier detection.

Looks for variable pairs that usually move together (or together-not),
then flags rows that violate the learned relationship.

Three pair kinds, each with its own scoring rule:
    numeric x numeric : Spearman correlation gates the pair, then a
                        robust Theil-Sen line gives slope/intercept and
                        residual MAD per timepoint. OLS used to do the
                        fit, but it is non-robust to leverage and was
                        gated by a rank correlation -- a monotonic but
                        nonlinear scale gave high Spearman AND patterned
                        residuals, flagging curvature, not anomalies.
                        Theil-Sen's median-of-slopes is unaffected by
                        a few extreme points.
    binary  x numeric : group-mean shift; residual_z within group per tp.
    binary  x binary  : rare joint cell scored by -log10(observed/expected).

Multiple-choice (3..15 levels) is handled in the binary branch by
treating each level as its own one-vs-rest binary slice.

Pair caps are aggressive on purpose. V choose 2 explodes fast: at
N_NUMERIC=60 we evaluate 1,770 numeric-numeric pairs per network,
N_BINARY=40 adds another 780 binary-binary, and the cross product is
2,400. The caps keep total pair work in the low thousands per network.
Columns are ranked by completeness so we spend the budget on the
variables most likely to surface signal.

Discovery peer: ``qc_types/discovery/pairwise_relationship_checks.py``
plus ``pca_reconstruction_checks.py`` and ``mahalanobis_checks.py``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .common import (
    Finding, calibrate, classify_columns, clean_missing_string,
    matches_excluded_substrings,
    pairwise_nan_ols, pairwise_nan_pearson, pairwise_nan_spearman,
    severity_from_z, site_from_subjectid, stack_by_network,
    to_numeric_clean, short,
)

# Per-detector exclusion list (case-insensitive substring match against
# variable name). Add strings here to suppress this detector's findings
# on pairs involving those variables, without touching the shared
# EXCLUDED_VARIABLE_SUBSTRINGS in common.py.
EXTRA_EXCLUDED_SUBSTRINGS: tuple = ()


_TRIM_FRACTION = 0.10   # drop the worst 10% of residuals before refit


def _trimmed_refit(x: np.ndarray, y: np.ndarray,
                   slope: float, intercept: float) -> Tuple[float, float]:
    """One-pass trimmed OLS refit on the high-residual rows of a pair.

    Cheap robustification for survivors of the rank-correlation screen.
    Compute residuals against the initial OLS line, drop the top
    ``_TRIM_FRACTION`` by |residual|, refit on the remainder. With the
    Spearman gate (|r| >= 0.5 on ranks) already ruling out
    leverage-driven survivors, one trim is plenty -- iterating buys
    nothing measurable on synthetic and real data.
    """
    if x.size < 30:
        return slope, intercept
    resid = y - (slope * x + intercept)
    cutoff = np.quantile(np.abs(resid), 1.0 - _TRIM_FRACTION)
    keep = np.abs(resid) <= cutoff
    n_keep = int(keep.sum())
    if n_keep < 20:
        return slope, intercept
    xk = x[keep]; yk = y[keep]
    var_x = xk.var()
    if var_x <= 0:
        return slope, intercept
    mean_x = xk.mean(); mean_y = yk.mean()
    new_slope = ((xk - mean_x) * (yk - mean_y)).sum() / (var_x * n_keep)
    new_intercept = mean_y - new_slope * mean_x
    return float(new_slope), float(new_intercept)

ANOMALY_TYPE = "correlative_outlier"

N_NUMERIC = 1000      # numeric vars per network (top-N by completeness)
N_BINARY  = 200       # binary/MC vars per network (top-N by completeness)

# Per-branch caps on the number of surviving pairs we'll do the
# expensive per-tp robust work for. Filters happen on the V x V pre-screen
# matrix so unsurvived pairs cost ~nothing.
TOP_K_PAIRS_NN = 5000     # numeric x numeric
TOP_K_PAIRS_BN = 3000     # binary x numeric
TOP_K_PAIRS_BB = 2000     # binary x binary

MIN_JOINT_N      = 80
MIN_CORR_PEARSON = 0.5
# Binary x numeric pool gate, was MIN_CORR_PEARSON / 2 = 0.25. 0.25 lets
# through point-biserial correlations easily produced by site / timepoint
# confounding at N >= 200, which produced the "technical yes/no vs CBC"
# class of false flag.
MIN_CORR_BN      = 0.4
MIN_GROUP_DELTA  = 0.5    # Cohen's d for binary-numeric to count as a relationship
MIN_RARE_CELL_FREQ = 0.005  # cells rarer than 0.5% in a pair are "rare-combo"-eligible
RESIDUAL_Z_THRESHOLD = 4.0

# Within-stratum reproducibility gate for NN and BN survivors. A pooled
# correlation that only exists when timepoints are stacked is almost
# always Simpson-style confounding -- the relationship has to also show
# up within at least STRATUM_REQUIRED timepoints, with the same sign as
# the pooled correlation and |r_tp| >= STRATUM_MIN_ABS_R on a joint set
# of at least STRATUM_MIN_N rows.
STRATUM_MIN_N      = 30
STRATUM_MIN_ABS_R  = 0.30
STRATUM_REQUIRED   = 2

# Domain denylist. Variables matching these are excluded from BOTH sides
# of every correlative pair branch -- they are operational / procedural
# fields with no plausible biology, so any apparent correlation is
# confounded noise.
_DENYLIST_EXACT = frozenset({
    "chrpsychs_av_method",
    "removed",
    "chrpsychs_av_english",
})
_DENYLIST_CONTAINS = ("chrmiss", "chrscid_remote_agree")


def _eff_residual_threshold(n_tests: int) -> float:
    """Effective |z| gate for a block of ``n_tests`` residual comparisons.

    A fixed |z| >= 4 gate fires by chance across the very large (pairs x tps x
    rows) test grid. Scale the gate by the block's test count via the standard
    max-of-Gaussians bound sqrt(2*ln(n)) (a Bonferroni-flavoured floor), never
    dropping below the original RESIDUAL_Z_THRESHOLD. SEVERITY is still
    computed against the original fixed threshold so a flagged row's severity
    stays interpretable -- only the emission GATE tightens with test count.
    Deliberately NOT a BH-FDR/p-value path: the residual-z null here is
    heavy-tailed, so a Gaussian p-value would be mis-specified.
    """
    if n_tests <= 1:
        return RESIDUAL_Z_THRESHOLD
    return max(RESIDUAL_Z_THRESHOLD, float(np.sqrt(2.0 * np.log(float(n_tests)))))


def _dedup_correlative(rows: List[dict]) -> List[dict]:
    """Collapse the redundant rows the pair branches emit for one anomaly.

    Two reductions:
      1. Over timepoints: one row per (subject, network, unordered variable
         pair), keeping the max-severity row.
      2. Across pairs (the big one): when a single variable for a subject
         violates its relationship with >=2 OTHER variables, that is one
         anomalous value re-flagged against every correlated partner. Emit a
         single finding attributed to that dominant variable, recording the
         corroborating-pair count (a value flagged by many partners is MORE
         trustworthy, not 10 separate rows). Pairs not subsumed by a dominant
         leg are emitted per-pair as before.
    """
    if not rows:
        return rows
    from collections import defaultdict

    pair_groups: dict = defaultdict(list)
    singletons: List[dict] = []
    for r in rows:
        # Binary x binary joint-cell rows ("a=x & b=y") describe a PAIR-level
        # rarity, not one variable's value -- leg-collapsing them would
        # mis-describe the property and could merge two distinct rare cells
        # that share a level. Pass them through unchanged.
        if " & " in str(r.get("variable", "")):
            singletons.append(r)
            continue
        parts = [p.strip() for p in str(r.get("variables_involved", "") or "").split(",") if p.strip()]
        if len(parts) != 2:
            singletons.append(r)
            continue
        a, b = sorted(parts)
        pair_groups[(str(r.get("subjectid", "")), str(r.get("network", "")), a, b)].append(r)

    # 1. timepoint collapse -> one rep row per (subject, network, pair)
    collapsed = []
    for (subj, net, a, b), grp in pair_groups.items():
        rep = max(grp, key=lambda r: float(r.get("severity_score", 0) or 0))
        collapsed.append((subj, net, a, b, dict(rep)))

    leg_partners: dict = defaultdict(set)
    leg_rep: dict = {}
    pairs_by_subj: dict = defaultdict(list)
    for (subj, net, a, b, rep) in collapsed:
        pairs_by_subj[(subj, net)].append((a, b, rep))
        for leg, partner in ((a, b), (b, a)):
            leg_partners[(subj, net, leg)].add(partner)
            cur = leg_rep.get((subj, net, leg))
            if cur is None or float(rep.get("severity_score", 0) or 0) > float(cur.get("severity_score", 0) or 0):
                leg_rep[(subj, net, leg)] = rep

    out: List[dict] = []
    for (subj, net), pairs in pairs_by_subj.items():
        dom_legs = {leg for leg in {a for a, _, _ in pairs} | {b for _, b, _ in pairs}
                    if len(leg_partners[(subj, net, leg)]) >= 2}
        covered = set()
        for leg in sorted(dom_legs):
            parts = sorted(leg_partners[(subj, net, leg)])
            rep = dict(leg_rep[(subj, net, leg)])
            rep["variable"] = leg
            rep["variables_involved"] = leg + " + " + ", ".join(parts[:12])
            rep["corroborating_pairs"] = len(parts)
            rep["explanation"] = (
                f"{leg} violates its learned relationship with {len(parts)} other "
                f"variable(s) ({', '.join(parts[:6])}{'...' if len(parts) > 6 else ''}) "
                f"for this subject; strongest residual z {rep.get('raw_score')}. "
                f"Flagged by many partners -> the value of {leg} itself is the likely error."
            )
            out.append(rep)
            for p in parts:
                covered.add(frozenset((leg, p)))
        for (a, b, rep) in pairs:
            if frozenset((a, b)) in covered:
                continue
            r = dict(rep)
            r.setdefault("corroborating_pairs", 1)
            out.append(r)
    out.extend(singletons)
    return out


def _is_denylisted(col: str) -> bool:
    cl = col.lower()
    if cl in _DENYLIST_EXACT:
        return True
    return any(token in cl for token in _DENYLIST_CONTAINS)


def _stratum_stable(x: np.ndarray, y: np.ndarray, tp: np.ndarray,
                    r_pooled: float) -> bool:
    """True if the (x, y) relationship reproduces across timepoints.

    Counts a timepoint as supporting when joint n >= STRATUM_MIN_N,
    |r_tp| >= STRATUM_MIN_ABS_R, and sign(r_tp) matches the pooled sign.
    Returns True as soon as STRATUM_REQUIRED supporters appear, so the
    cost is amortised on stable pairs and only the unstable tail pays
    the full scan.
    """
    if not np.isfinite(r_pooled) or r_pooled == 0.0:
        return False
    target_sign = np.sign(r_pooled)
    supporters = 0
    for tp_val in np.unique(tp):
        sel = tp == tp_val
        if int(sel.sum()) < STRATUM_MIN_N:
            continue
        xs = x[sel]
        ys = y[sel]
        if xs.std() == 0.0 or ys.std() == 0.0:
            continue
        r_tp = float(np.corrcoef(xs, ys)[0, 1])
        if not np.isfinite(r_tp):
            continue
        if abs(r_tp) < STRATUM_MIN_ABS_R:
            continue
        if np.sign(r_tp) != target_sign:
            continue
        supporters += 1
        if supporters >= STRATUM_REQUIRED:
            return True
    return False


def detect_all(per_slice: Dict, classify: Dict = None, **_) -> List[dict]:
    """``classify`` is accepted for API parity with per-slice detectors;
    the stacked-stats branches here recompute on their network stack,
    which is built once per network so the cost is bounded already."""
    rows: List[dict] = []
    by_network: Dict[str, List] = {}
    for (network, tp), df in per_slice.items():
        if df is None or df.empty or "subjectid" not in df.columns:
            continue
        by_network.setdefault(network, []).append((tp, df))

    for network, items in by_network.items():
        try:
            rows.extend(_detect_network(network, items))
        except Exception as e:
            print(f"  [correlative] WARN network {network}: {e}")
    return rows


def _select_variables(stacked: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cats = classify_columns(stacked)
    # Drop denylisted variables before the completeness sort so they
    # don't take a budget slot we'd then have to spend on the next best
    # variable.
    numerics = [c for c in cats["numeric"]
                if not _is_denylisted(c)
                and not matches_excluded_substrings(c, EXTRA_EXCLUDED_SUBSTRINGS)]
    bin_like = [c for c in (cats["binary"] + cats["low_card"])
                if not _is_denylisted(c)
                and not matches_excluded_substrings(c, EXTRA_EXCLUDED_SUBSTRINGS)]

    # rank numerics by completeness
    def completeness(col: str) -> int:
        return int(to_numeric_clean(stacked[col]).notna().sum())
    numerics = sorted(numerics, key=completeness, reverse=True)[:N_NUMERIC]
    bin_like = sorted(bin_like, key=completeness, reverse=True)[:N_BINARY]
    return numerics, bin_like


def _detect_network(network: str, items: List) -> List[dict]:
    if not items:
        return []
    stacked = stack_by_network(items)
    if stacked.empty or "subjectid" not in stacked.columns:
        return []

    numerics, bin_like = _select_variables(stacked)
    rows: List[dict] = []

    rows.extend(_numeric_numeric_pairs(stacked, numerics, network))
    rows.extend(_binary_numeric_pairs(stacked, bin_like, numerics, network))
    rows.extend(_binary_binary_pairs(stacked, bin_like, network))
    # Collapse the per-pair / per-timepoint redundancy so one anomalous value
    # is one finding (with a corroboration count), not dozens of rows that
    # flood the per-type cap. NN/BN rows leg-collapse; BB joint-cell rows are
    # passed through unchanged (pair-level rarity, not a single-value anomaly).
    return _dedup_correlative(rows)


# ---------------------------------------------------------------------------
# numeric x numeric
# ---------------------------------------------------------------------------

def _numeric_numeric_pairs(stacked: pd.DataFrame, numerics: List[str], network: str) -> List[dict]:
    """Numeric x numeric, vectorised end-to-end (V=1000+).

    Phase 1 - matrix screen. Build an (N, V) matrix and run a single
    BLAS-backed Spearman correlation across every pair (ranks computed
    once per column, then NaN-aware Pearson on the rank matrix).

    Phase 2 - vectorised OLS slopes. The same matrix multiplications
    that gave us the Pearson correlation also produce the per-pair OLS
    slope/intercept (cov / var on the joint set). So we get every
    (slope, intercept) for free, in BLAS, across all V x V pairs.
    No per-pair Python loop fitting required. Theil-Sen via scipy used
    to dominate at ~110 ms/pair -- dropping it is the largest single
    speedup in this rewrite.

    Phase 3 - filter to pairs with |r| >= MIN_CORR_PEARSON AND
    joint_n >= MIN_JOINT_N. Cap to TOP_K_PAIRS_NN by |r|.

    Phase 4 - per surviving pair: a one-pass trimmed OLS refit
    (drop the top 10% of residuals, refit on the rest) for
    belt-and-suspenders leverage robustness; then per-tp MAD on the
    refit residuals and outlier enumeration. The trim+refit is O(N)
    per pair so it stays cheap.

    Phase 5 - argpartition the candidate flags to keep top-K by
    severity, build Finding objects only for those.
    """
    if len(numerics) < 2:
        return []

    N = len(stacked)
    V = len(numerics)
    data = np.empty((N, V), dtype=np.float64)
    for k, c in enumerate(numerics):
        data[:, k] = to_numeric_clean(stacked[c]).to_numpy(dtype=np.float64)

    corr, _ = pairwise_nan_spearman(data, min_joint_n=MIN_JOINT_N)
    slopes_m, intercepts_m, joint_n_m = pairwise_nan_ols(data, min_joint_n=MIN_JOINT_N)

    iu, ju = np.triu_indices(V, k=1)
    r_flat = corr[iu, ju]
    n_flat = joint_n_m[iu, ju]
    surv_mask = (
        (np.abs(r_flat) >= MIN_CORR_PEARSON)
        & (n_flat >= MIN_JOINT_N)
        & np.isfinite(r_flat)
        & np.isfinite(slopes_m[iu, ju])
    )
    surv_idx = np.where(surv_mask)[0]
    if surv_idx.size == 0:
        return []
    if surv_idx.size > TOP_K_PAIRS_NN:
        keep = np.argpartition(-np.abs(r_flat[surv_idx]), TOP_K_PAIRS_NN)[:TOP_K_PAIRS_NN]
        surv_idx = surv_idx[keep]

    tp_col = stacked.get("_tp", pd.Series(["?"] * N))
    tp_arr_all = tp_col.astype(str).to_numpy()
    unique_tps = pd.unique(tp_arr_all)
    subj_arr = stacked["subjectid"].astype(str).to_numpy()

    cand_sev: List[float] = []
    cand_raw: List[float] = []
    cand_ridx: List[int] = []
    cand_pair: List[int] = []
    cand_tp: List[str] = []
    pair_meta: List[tuple] = []

    valid_mask = ~np.isnan(data)

    for sidx in surv_idx:
        i, j = int(iu[sidx]), int(ju[sidx])
        joint = valid_mask[:, i] & valid_mask[:, j]
        if joint.sum() < MIN_JOINT_N:
            continue
        x_all = data[joint, i]
        y_all = data[joint, j]
        r = float(corr[i, j])
        joint_idx = np.where(joint)[0]
        tp_arr_pair = tp_arr_all[joint_idx]
        # Stratum stability runs before the trimmed refit so confounded
        # pairs short-circuit out of the expensive per-pair work.
        if not _stratum_stable(x_all, y_all, tp_arr_pair, r):
            continue
        # OLS line from the vectorised pre-pass + a cheap trimmed refit.
        slope = float(slopes_m[i, j])
        intercept = float(intercepts_m[i, j])
        slope, intercept = _trimmed_refit(x_all, y_all, slope, intercept)
        resid = y_all - (slope * x_all + intercept)

        pair_local_idx = len(pair_meta)
        pair_meta.append((numerics[i], numerics[j], i, j,
                          slope, intercept, r, int(joint.sum())))

        for tp_val in unique_tps:
            tp_sel = tp_arr_pair == tp_val
            n_tp = int(tp_sel.sum())
            if n_tp < max(20, MIN_JOINT_N // 3):
                continue
            res_tp = resid[tp_sel]
            med = float(np.median(res_tp))
            mad = float(np.median(np.abs(res_tp - med)))
            sigma = mad * 1.4826
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            z_tp = (res_tp - med) / sigma
            eff_z = _eff_residual_threshold(res_tp.size)
            bad_local = np.where(np.abs(z_tp) >= eff_z)[0]
            if bad_local.size == 0:
                continue
            tp_sel_idx = joint_idx[tp_sel]
            # Vectorised flag append per tp.
            zi_arr = z_tp[bad_local]
            abs_zi = np.abs(zi_arr)
            base = np.clip(50.0 + (abs_zi - RESIDUAL_Z_THRESHOLD) * 7.5, 0, 95.0)
            strength = max(0.0, min(1.0, (abs(r) - MIN_CORR_PEARSON) / (1 - MIN_CORR_PEARSON)))
            sev = np.minimum(100.0, base + strength * 5.0)
            cand_sev.extend(sev.tolist())
            cand_raw.extend(abs_zi.tolist())
            cand_ridx.extend(tp_sel_idx[bad_local].tolist())
            cand_pair.extend([pair_local_idx] * bad_local.size)
            cand_tp.extend([str(tp_val)] * bad_local.size)

    if not cand_sev:
        return []

    sev_arr = np.asarray(cand_sev, dtype=np.float64)
    HARD_CAP = 20000
    if sev_arr.size > HARD_CAP:
        keep = np.argpartition(-sev_arr, HARD_CAP)[:HARD_CAP]
    else:
        keep = np.arange(sev_arr.size)

    out: List[dict] = []
    for kk in keep:
        a, b, ci, cj, slope, intercept, r, n_pair = pair_meta[cand_pair[kk]]
        ridx = cand_ridx[kk]
        tp_val = cand_tp[kk]
        zi = cand_raw[kk]
        x_v = float(data[ridx, ci])
        y_v = float(data[ridx, cj])
        subj = str(subj_arr[ridx])
        out.append(Finding(
            anomaly_type=ANOMALY_TYPE,
            severity_score=float(sev_arr[kk]),
            raw_score=float(zi),
            network=network,
            timepoint=tp_val,
            site_id=site_from_subjectid(subj),
            subjectid=subj,
            variable=f"{a} | {b}",
            variables_involved=f"{a}, {b}",
            observed_value=f"{a}={short(x_v)}, {b}={short(y_v)}",
            expected_value=f"{b} ~= {slope:.4g}*{a} + {intercept:.4g} (rho={r:+.2f}, n={n_pair})",
            explanation=(
                f"Residual z = {zi:+.2f} against the typical linear "
                f"relationship between {a} and {b}, scored against "
                f"the {tp_val} residual distribution."
            ),
            method="vectorised Spearman + BLAS OLS + trimmed refit + per-tp MAD",
            extra={"correlation": round(r, 3)},
        ).to_row())
    return out




# ---------------------------------------------------------------------------
# binary x numeric (also handles MC-as-one-vs-rest)
# ---------------------------------------------------------------------------

def _binary_numeric_pairs(stacked: pd.DataFrame, bin_like: List[str], numerics: List[str], network: str) -> List[dict]:
    """Binary / MC x numeric, with a vectorised one-vs-rest Pearson screen.

    Each level of each binary/MC column becomes a 0/1 indicator. We
    stack those into an (N, V_lvl) matrix and run pairwise Pearson
    against the numeric matrix in one BLAS call. The point-biserial
    correlation equals Pearson when one side is 0/1, so this is exact.

    Only (level, numeric) pairs whose |r| clears MIN_CORR_BN survive
    to the expensive per-tp MAD step. The threshold is lower than
    MIN_CORR_PEARSON because effect-size + grouping has lower-r
    relationships than two continuous variables typically do, but not
    as low as the old MIN_CORR_PEARSON/2 = 0.25 -- that floor admitted
    pairs whose pooled |r| was driven entirely by site/timepoint mix.
    Survivors must also pass `_stratum_stable` before being scored.
    """
    if not bin_like or not numerics:
        return []

    # 1. Encode each (column, level) as a one-vs-rest 0/1 indicator with
    #    NaN preserved. Cap total levels so a few high-cardinality MCs
    #    don't blow up the screen.
    level_keys: List[Tuple[str, object]] = []   # (bcol, level)
    level_cols: List[np.ndarray] = []
    N = len(stacked)
    for c in bin_like:
        s_num = to_numeric_clean(stacked[c])
        if s_num.notna().sum() >= MIN_JOINT_N:
            s = s_num
        else:
            s = clean_missing_string(stacked[c].astype(object))
        levels = pd.Series(s).dropna().unique()
        if len(levels) < 2 or len(levels) > 15:
            continue
        for lv in levels:
            ind = np.where(pd.Series(s).isna(), np.nan,
                           (pd.Series(s) == lv).astype(float))
            if np.nansum(ind) < max(20, MIN_JOINT_N // 4):
                continue
            level_keys.append((c, lv))
            level_cols.append(ind.astype(np.float64))
    if not level_keys:
        return []

    Lvl = np.column_stack(level_cols)            # (N, V_lvl)
    V_num = len(numerics)
    num_data = np.empty((N, V_num), dtype=np.float64)
    for k, c in enumerate(numerics):
        num_data[:, k] = to_numeric_clean(stacked[c]).to_numpy(dtype=np.float64)

    # 2. Vectorised Pearson screen: each level vs each numeric.
    corr, joint_n = pairwise_nan_pearson(Lvl, num_data, min_joint_n=MIN_JOINT_N)
    # corr is (V_lvl, V_num).
    bn_threshold = MIN_CORR_BN
    surv_mask = (np.abs(corr) >= bn_threshold) & (joint_n >= MIN_JOINT_N) & np.isfinite(corr)
    if not surv_mask.any():
        return []

    surv_rows, surv_cols = np.where(surv_mask)
    abs_r = np.abs(corr[surv_rows, surv_cols])
    if surv_rows.size > TOP_K_PAIRS_BN:
        keep = np.argpartition(-abs_r, TOP_K_PAIRS_BN)[:TOP_K_PAIRS_BN]
        surv_rows = surv_rows[keep]
        surv_cols = surv_cols[keep]
        abs_r = abs_r[keep]

    # 3. Per surviving (level, numeric) pair: per-tp MAD on the in-level
    #    subset. Same per-tp logic as before but pair-vectorised.
    tp_col = stacked.get("_tp", pd.Series(["?"] * N))
    tp_arr_all = tp_col.astype(str).to_numpy()
    unique_tps = pd.unique(tp_arr_all)
    subj_arr = stacked["subjectid"].astype(str).to_numpy()

    cand_sev: List[float] = []
    cand_raw: List[float] = []
    cand_ridx: List[int] = []
    cand_meta_idx: List[int] = []
    meta: List[tuple] = []     # (bcol, lv, ncol, ncol_idx, best_delta_proxy, r)

    for pair_local_idx, (lvl_idx, num_idx) in enumerate(zip(surv_rows, surv_cols)):
        bcol, lv = level_keys[lvl_idx]
        ncol = numerics[num_idx]
        r = float(corr[lvl_idx, num_idx])
        in_level = Lvl[:, lvl_idx] == 1.0
        num_valid = ~np.isnan(num_data[:, num_idx])
        joint = in_level & num_valid
        if joint.sum() < max(20, MIN_JOINT_N // 4):
            continue
        # Stratum stability is computed on the level-indicator vs the
        # numeric across rows where both are valid (NOT just in-level),
        # so the per-tp correlation is comparable to the pooled r that
        # gated this pair.
        lvl_valid = ~np.isnan(Lvl[:, lvl_idx])
        pair_joint = lvl_valid & num_valid
        if pair_joint.sum() < MIN_JOINT_N:
            continue
        if not _stratum_stable(
                Lvl[pair_joint, lvl_idx],
                num_data[pair_joint, num_idx],
                tp_arr_all[pair_joint],
                r):
            continue
        meta.append((bcol, lv, ncol, num_idx, abs(r)))
        joint_idx = np.where(joint)[0]
        tp_arr_pair = tp_arr_all[joint_idx]
        for tp_val in unique_tps:
            tp_sel = tp_arr_pair == tp_val
            if tp_sel.sum() < max(20, MIN_JOINT_N // 4):
                continue
            sub_vals = num_data[joint_idx[tp_sel], num_idx]
            med = float(np.median(sub_vals))
            mad = float(np.median(np.abs(sub_vals - med)))
            sigma = mad * 1.4826
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            z = (sub_vals - med) / sigma
            eff_z = _eff_residual_threshold(sub_vals.size)
            bad_local = np.where(np.abs(z) >= eff_z)[0]
            if bad_local.size == 0:
                continue
            tp_sel_idx = joint_idx[tp_sel]
            for bk in bad_local:
                zi = float(z[bk])
                if not np.isfinite(zi):
                    continue
                base = severity_from_z(zi, RESIDUAL_Z_THRESHOLD, 10.0)
                strength = max(0.0, min(1.0, (abs(r) - bn_threshold) / max(bn_threshold, 1e-9)))
                sev = float(min(100.0, base + strength * 5.0))
                cand_sev.append(sev)
                cand_raw.append(abs(zi))
                cand_ridx.append(int(tp_sel_idx[bk]))
                cand_meta_idx.append(len(meta) - 1)

    if not cand_sev:
        return []

    sev_arr = np.asarray(cand_sev, dtype=np.float64)
    HARD_CAP = 20000
    if sev_arr.size > HARD_CAP:
        keep_top = np.argpartition(-sev_arr, HARD_CAP)[:HARD_CAP]
    else:
        keep_top = np.arange(sev_arr.size)

    out: List[dict] = []
    for kk in keep_top:
        bcol, lv, ncol, num_idx, abs_r_val = meta[cand_meta_idx[kk]]
        ridx = cand_ridx[kk]
        subj = str(subj_arr[ridx])
        tp_val = tp_arr_all[ridx]
        obs = float(num_data[ridx, num_idx])
        zi = cand_raw[kk]
        out.append(Finding(
            anomaly_type=ANOMALY_TYPE,
            severity_score=float(sev_arr[kk]),
            raw_score=float(zi),
            network=network,
            timepoint=str(tp_val),
            site_id=site_from_subjectid(subj),
            subjectid=subj,
            variable=f"{bcol}={short(lv)} | {ncol}",
            variables_involved=f"{bcol}, {ncol}",
            observed_value=f"{ncol}={short(obs)} when {bcol}={short(lv)}",
            expected_value=f"{ncol} median+MAD at {tp_val} within {bcol}={short(lv)}",
            explanation=(
                f"{ncol} differs across levels of {bcol} (point-biserial r="
                f"{abs_r_val:+.2f}); at {tp_val} this row is {zi:+.2f} sigmas "
                f"from the level-specific median."
            ),
            method="vectorised point-biserial screen + per-tp MAD",
            extra={"point_biserial_r": round(abs_r_val, 3)},
        ).to_row())
    return out


# ---------------------------------------------------------------------------
# binary x binary  (rare joint cells)
# ---------------------------------------------------------------------------

def _binary_binary_pairs(stacked: pd.DataFrame, bin_like: List[str], network: str) -> List[dict]:
    """Binary / MC x binary / MC, with matrix-multiplication contingency cells.

    Each (column, level) becomes a 0/1 indicator (NaN preserved). All
    cell counts for every (level_a, level_b) pair are then four matrix
    multiplications:

        n(0,0) = (1 - A * M_a).T @ (1 - B * M_b)   [not used]
        n(1,1) = (A * M_a).T @ (B * M_b)
        n(1,0) = (A * M_a).T @ M_b minus n(1,1)... [we just need n_ij]

    Practically we compute n_ij joint counts via M_a.T @ M_b (joint
    valid count) and observed (1,1) cell counts via A.T @ B. Combining
    gives the marginal frequencies and independence baseline in one
    vector op instead of V_lvl^2 pandas groupbys.
    """
    if len(bin_like) < 2:
        return []

    N = len(stacked)
    level_keys: List[Tuple[str, object]] = []
    level_cols: List[np.ndarray] = []
    for c in bin_like:
        s_num = to_numeric_clean(stacked[c])
        if s_num.notna().sum() >= MIN_JOINT_N:
            s = s_num
        else:
            s = clean_missing_string(stacked[c].astype(object))
        s_pd = pd.Series(s)
        levels = s_pd.dropna().unique()
        if len(levels) < 2 or len(levels) > 15:
            continue
        for lv in levels:
            ind_present = (~s_pd.isna()).to_numpy(dtype=bool)
            ind_value = (s_pd == lv).to_numpy(dtype=bool) & ind_present
            if ind_value.sum() < max(20, MIN_JOINT_N // 4):
                continue
            level_keys.append((c, lv))
            # store (column_index_into_bin_like, level_value, present_mask, value_mask)
            level_cols.append((ind_present, ind_value, c))

    if len(level_keys) < 2:
        return []

    # Pack into matrices: M_present (N, L) -- valid-mask; M_value (N, L)
    # -- 1 where the row hits that exact level.
    L = len(level_keys)
    M_present = np.empty((N, L), dtype=np.float64)
    M_value = np.empty((N, L), dtype=np.float64)
    cols_of_level = np.empty(L, dtype=object)
    for k, (ind_present, ind_value, ccol) in enumerate(level_cols):
        M_present[:, k] = ind_present.astype(np.float64)
        M_value[:, k] = ind_value.astype(np.float64)
        cols_of_level[k] = ccol

    # Joint valid count per (level_i, level_j).
    n_ij = M_present.T @ M_present                 # (L, L)
    # Observed joint cell counts: rows where both indicators == 1.
    obs_ij = M_value.T @ M_value                   # (L, L)
    # Per-level counts conditioned on the other side being valid.
    count_a_given_b_valid = M_value.T @ M_present  # (L, L); rows: level_a
    count_b_given_a_valid = M_present.T @ M_value  # (L, L); cols: level_b

    with np.errstate(divide="ignore", invalid="ignore"):
        n_safe = np.where(n_ij > 0, n_ij, 1.0)
        p_a = count_a_given_b_valid / n_safe       # P(level_a) over joint set
        p_b = count_b_given_a_valid / n_safe       # P(level_b) over joint set
        obs_freq = obs_ij / n_safe
        exp_freq = p_a * p_b

    # Drop diagonal AND drop cells where both levels come from the same column.
    same_col = (cols_of_level[:, None] == cols_of_level[None, :])
    np.fill_diagonal(same_col, True)
    # Independence pre-screen: rare observed cell AND much smaller than
    # expected. Vectorised across the whole (L, L) grid.
    rare_mask = (
        (obs_freq <= MIN_RARE_CELL_FREQ)
        & (exp_freq > 5 * obs_freq)
        & (n_ij >= MIN_JOINT_N)
        & ~same_col
    )
    if not rare_mask.any():
        return []
    li, lj = np.where(rare_mask)
    # Only retain the upper triangle of inter-column pairs to avoid
    # symmetric duplicates. We keep where cols_of_level[i] < cols_of_level[j]
    # OR (== and i < j); since same-column pairs are masked the equal case
    # never appears.
    keep_pair = cols_of_level[li] < cols_of_level[lj]
    li, lj = li[keep_pair], lj[keep_pair]
    if li.size == 0:
        return []
    # Cap pairs by sharpness.
    ratios = exp_freq[li, lj] / np.maximum(obs_freq[li, lj], 1e-9)
    if li.size > TOP_K_PAIRS_BB:
        keep = np.argpartition(-ratios, TOP_K_PAIRS_BB)[:TOP_K_PAIRS_BB]
        li, lj = li[keep], lj[keep]

    tp_col = stacked.get("_tp", pd.Series(["?"] * N))
    tp_arr_all = tp_col.astype(str).to_numpy()
    subj_arr = stacked["subjectid"].astype(str).to_numpy()

    out: List[dict] = []
    for a_idx, b_idx in zip(li, lj):
        col_a, lv_a = level_keys[a_idx]
        col_b, lv_b = level_keys[b_idx]
        obs = float(obs_freq[a_idx, b_idx])
        exp = float(exp_freq[a_idx, b_idx])
        ratio = max(1.0, exp / max(obs, 1e-9))
        raw = float(np.log10(ratio))
        severity = calibrate(raw, threshold=1.0, extreme=3.0)
        cell_mask = (M_value[:, a_idx] == 1.0) & (M_value[:, b_idx] == 1.0)
        for ridx in np.where(cell_mask)[0]:
            subj = str(subj_arr[ridx])
            tp = str(tp_arr_all[ridx])
            out.append(Finding(
                anomaly_type=ANOMALY_TYPE,
                severity_score=severity,
                raw_score=raw,
                network=network,
                timepoint=tp,
                site_id=site_from_subjectid(subj),
                subjectid=subj,
                variable=f"{col_a}={short(lv_a)} & {col_b}={short(lv_b)}",
                variables_involved=f"{col_a}, {col_b}",
                observed_value=f"{col_a}={short(lv_a)}, {col_b}={short(lv_b)}",
                expected_value=(
                    f"P({col_a}={short(lv_a)}) * P({col_b}={short(lv_b)}) "
                    f"= {exp:.4f}; observed cell freq {obs:.4f}"
                ),
                explanation=(
                    f"Joint {col_a}/{col_b} cell occurs ~{ratio:.1f}x less "
                    f"often than independent levels would predict; the "
                    f"combination is rare."
                ),
                method="matrix-mult contingency vs independence",
            ).to_row())
    return out
