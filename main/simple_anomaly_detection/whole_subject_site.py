"""Detector 6 - Whole-subject / whole-site deviations.

Aggregates several per-variable signals to find subjects and sites
whose *overall* pattern is unusual:

    numeric:    mean of |robust_z| across the subject's numeric variables,
                stratified per timepoint (so a normal-for-baseline value
                is not judged against the month-12 distribution -- this
                matches what standard_outlier does).
    categorical: mean rarity score (-log10 of value frequency) per variable.
    missingness: per-row missingness over the columns that are alive in
                that row's slice, then averaged. Computes one block per
                (network, tp) instead of a per-row Python loop; on a real
                5k-subject run this saved ~30 min/network.

Per-subject and per-site composites are computed within network; only
the upper tail above the conservative threshold is reported. Subjects
without MIN_EVIDENCE_FOR_SUBJECT contributing measurements are excluded
so a subject seen in one sparsely-populated slice cannot top the tail.

Discovery peer: ``qc_types/discovery/subject_summary.py``.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .common import (
    Finding, calibrate, classify_columns, clean_missing_string,
    matches_excluded_substrings, robust_center_scale, site_from_subjectid,
    stack_by_network, to_numeric_clean, short,
)

ANOMALY_TYPE_SUBJECT = "whole_subject_outlier"
ANOMALY_TYPE_SITE = "whole_site_outlier"

# Per-detector exclusion list (case-insensitive substring match against
# variable name). Variables matching these are excluded from the per-row
# evidence (numeric z, categorical rarity, missingness) that feeds the
# subject/site composites in this detector. Layered on top of the shared
# EXCLUDED_VARIABLE_SUBSTRINGS in common.py.
EXTRA_EXCLUDED_SUBSTRINGS: tuple = ()

MIN_VARS_FOR_SUBJECT = 20
MIN_N_PER_SITE = 30
# A subject must contribute this many variable-level measurements (numeric
# z, categorical rarity, or per-slice missingness) before we'll rank them.
# This stops subjects who appear in only one slice / on a sparsely
# populated form from dominating the tail.
MIN_EVIDENCE_FOR_SUBJECT = 15
# A column has to have at least this many non-missing values within a
# (network, tp) slice before we count its absence as "missing" for a
# subject in that slice. Otherwise structural missingness from a baseline-
# only form would penalize every month-12 subject.
MIN_ALIVE_PER_SLICE = 30


def detect_all(per_slice: Dict, classify: Dict = None, **_) -> List[dict]:
    """``classify`` may carry pre-computed classify_columns() results
    keyed by (network, tp). _alive_cols_per_tp will reuse them.
    """
    by_network: Dict[str, List] = {}
    classify = classify or {}
    for (network, tp), df in per_slice.items():
        if df is None or df.empty or "subjectid" not in df.columns:
            continue
        by_network.setdefault(network, []).append((tp, df, classify.get((network, tp))))

    rows: List[dict] = []
    for network, items in by_network.items():
        try:
            stacked = stack_by_network(items)
            alive = _alive_cols_per_tp(items)
            # Classify the network stack ONCE; both composites reused it.
            cats = classify_columns(stacked)
            rows.extend(_subject_composite(stacked, alive, network, cats))
            rows.extend(_site_composite(stacked, alive, network, cats))
        except Exception as e:
            print(f"  [whole_subject_site] WARN network {network}: {e}")
    return rows


def _alive_cols_per_tp(items: List) -> Dict[str, set]:
    """For each tp, the set of columns with at least MIN_ALIVE_PER_SLICE
    non-missing values. Used so we don't count a baseline-only column as
    missing for every month-12 subject (structural missingness)."""
    out: Dict[str, set] = {}
    for tp, df, _cats in items:
        # We don't reuse classify_columns here because that function
        # discards columns that fall outside its categories (dates,
        # too-sparse strings) -- we still want those tracked for the
        # missingness denominator. We just need a notna count.
        non_miss = (~df.isna() & (df.astype(str) != "")).sum(axis=0)
        out[tp] = set(non_miss[non_miss >= MIN_ALIVE_PER_SLICE].index)
    return out


def _subject_composite(stacked: pd.DataFrame, alive: Dict[str, set], network: str, cats: dict) -> List[dict]:
    numerics = [c for c in cats["numeric"]
                if not matches_excluded_substrings(c, EXTRA_EXCLUDED_SUBSTRINGS)]
    cat_vars = [c for c in (cats["binary"] + cats["low_card"])
                if not matches_excluded_substrings(c, EXTRA_EXCLUDED_SUBSTRINGS)]
    if not numerics and not cat_vars:
        return []

    by_subj: Dict[str, dict] = {}
    subj_arr = stacked["subjectid"].astype(str).to_numpy()
    tp_arr = stacked.get("_tp", pd.Series(["?"] * len(stacked))).astype(str).to_numpy()

    # Numeric z, stratified per timepoint so a baseline value is judged
    # against the baseline distribution, not the month-12 one. Matches
    # the per-tp logic used by standard_outlier.
    for col in numerics:
        num = to_numeric_clean(stacked[col]).to_numpy(dtype=float)
        for tp_val in np.unique(tp_arr):
            tp_mask = tp_arr == tp_val
            vals = num[tp_mask & ~np.isnan(num)]
            if vals.size < MIN_VARS_FOR_SUBJECT:
                continue
            med, sigma = robust_center_scale(vals)
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            # broadcast z over tp_mask rows
            z_full = np.where(tp_mask, (num - med) / sigma, np.nan)
            valid_idx = np.where(np.isfinite(z_full))[0]
            for ridx in valid_idx:
                subj = subj_arr[ridx]
                d = by_subj.setdefault(subj, _empty_subj_dict())
                az = abs(float(z_full[ridx]))
                d["z_sum"] += az
                d["z_n"] += 1
                if az > d["z_max"]:
                    d["z_max"] = az

    for col in cat_vars:
        ser_clean = clean_missing_string(stacked[col].astype(object))
        freq = ser_clean.value_counts(normalize=True, dropna=True)
        if freq.empty:
            continue
        rarity_map = (-np.log10(freq.clip(lower=1e-6))).astype(float)
        # Vectorize: map each row's value to its rarity, drop NaN, then aggregate.
        rar = ser_clean.map(rarity_map).astype(float)
        valid_mask = rar.notna() & (rar > 0)
        if not valid_mask.any():
            continue
        valid_idx = np.where(valid_mask.to_numpy())[0]
        rar_vals = rar.to_numpy()
        for ridx in valid_idx:
            subj = subj_arr[ridx]
            d = by_subj.setdefault(subj, _empty_subj_dict())
            d["rar_sum"] += float(rar_vals[ridx])
            d["rar_n"] += 1

    # Per-row missingness vectorized per (tp) block. Replaces the
    # previous per-row Python loop that did .iloc[ridx] N*V times.
    tracked = set(numerics) | set(cat_vars)
    for tp_val in np.unique(tp_arr):
        alive_cols = list(alive.get(str(tp_val), set()) & tracked)
        if not alive_cols:
            continue
        tp_mask = tp_arr == tp_val
        block = stacked.loc[tp_mask, alive_cols]
        # treat empty-string-like cells as missing too
        is_miss = block.isna() | block.astype(str).apply(lambda c: c.str.strip() == "")
        rate = is_miss.mean(axis=1).to_numpy(dtype=float)
        idx_arr = np.where(tp_mask)[0]
        for k, ridx in enumerate(idx_arr):
            subj = subj_arr[ridx]
            d = by_subj.setdefault(subj, _empty_subj_dict())
            d["miss_sum"] += float(rate[k])
            d["miss_n"] += 1

    if not by_subj:
        return []

    subj_rows = []
    for subj, d in by_subj.items():
        evidence = d["z_n"] + d["rar_n"] + d["miss_n"]
        if evidence < MIN_EVIDENCE_FOR_SUBJECT:
            continue
        # Trim the single largest |z| so one spike (often already caught by
        # standard_outlier) doesn't dominate the subject's numeric component.
        if d["z_n"] > 1:
            avg_z = (d["z_sum"] - d["z_max"]) / (d["z_n"] - 1)
        else:
            avg_z = d["z_sum"] / d["z_n"] if d["z_n"] else 0.0
        avg_rar = d["rar_sum"] / d["rar_n"] if d["rar_n"] else 0.0
        avg_miss = d["miss_sum"] / d["miss_n"] if d["miss_n"] else 0.0
        subj_rows.append((subj, avg_z, avg_rar, avg_miss, evidence))

    sf = pd.DataFrame(subj_rows, columns=["subjectid", "avg_abs_z", "avg_rarity", "avg_missing", "evidence_n"])
    if len(sf) < 10:
        return []
    # NOTE: the documented weights (1.0/0.5/1.5) act on RAW component scales,
    # which are non-commensurate (rarity ~0..6 vs missingness 0..1), so the
    # high-spread component dominates regardless of intent. The principled fix
    # is to robust-z each component across the cohort BEFORE weighting
    # -- but on the demo that quadrupled the flagged tail
    # (11 -> 42) because the z>=3.0 cutoff below was calibrated for THIS raw
    # composite. Shipping that shift unvalidated would change flag volume on
    # real data unpredictably, so it is left for a real-data recalibration of
    # the threshold. avg_abs_z here already has its single largest |z| trimmed.
    sf["composite"] = sf["avg_abs_z"] + 0.5 * sf["avg_rarity"] + 1.5 * sf["avg_missing"]
    composite_vals = sf["composite"].to_numpy(dtype=float)
    med, sigma = robust_center_scale(composite_vals)
    if not np.isfinite(sigma) or sigma <= 0:
        return []
    sf["composite_z"] = (sf["composite"] - med) / sigma
    sf = sf.sort_values("composite_z", ascending=False)

    out: List[dict] = []
    for _, r in sf.iterrows():
        z = float(r["composite_z"])
        if z < 3.0:
            break  # rows are sorted; safe to stop
        subj = str(r["subjectid"])
        out.append(Finding(
            anomaly_type=ANOMALY_TYPE_SUBJECT,
            severity_score=calibrate(z, threshold=3.0, extreme=8.0),
            raw_score=z,
            network=network,
            site_id=site_from_subjectid(subj),
            subjectid=subj,
            variable="(subject composite)",
            observed_value=(
                f"avg|z|={r['avg_abs_z']:.2f}, avg_rarity={r['avg_rarity']:.2f}, "
                f"missing={r['avg_missing']:.2%}, evidence_n={int(r['evidence_n'])}"
            ),
            expected_value=f"cohort composite median {med:.2f}, sigma {sigma:.2f}",
            explanation=(
                f"Subject composite z-score {z:.2f} across numeric outlierness, "
                f"categorical rarity, and per-slice missingness. Evidence "
                f"denominator = {int(r['evidence_n'])} variable-level "
                f"measurements; >= {MIN_EVIDENCE_FOR_SUBJECT} required."
            ),
            method="composite z; per-slice missingness; evidence denominator",
        ).to_row())
    return out


def _empty_subj_dict() -> dict:
    return {"z_sum": 0.0, "z_n": 0, "z_max": 0.0, "rar_sum": 0.0, "rar_n": 0, "miss_sum": 0.0, "miss_n": 0}


def _site_composite(stacked: pd.DataFrame, alive: Dict[str, set], network: str, cats: dict) -> List[dict]:
    numerics = [c for c in cats["numeric"]
                if not matches_excluded_substrings(c, EXTRA_EXCLUDED_SUBSTRINGS)]
    cat_vars = [c for c in (cats["binary"] + cats["low_card"])
                if not matches_excluded_substrings(c, EXTRA_EXCLUDED_SUBSTRINGS)]
    if not numerics and not cat_vars:
        return []
    sites = stacked["subjectid"].astype(str).str[:2]
    stacked = stacked.copy()
    stacked["_site"] = sites
    site_counts = stacked["_site"].value_counts()
    eligible_sites = site_counts[site_counts >= MIN_N_PER_SITE].index.tolist()
    if len(eligible_sites) < 3:
        return []

    num_stats = {}
    for col in numerics:
        num = to_numeric_clean(stacked[col]).dropna()
        if len(num) < 60:
            continue
        med, sigma = robust_center_scale(num.to_numpy(dtype=float))
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        num_stats[col] = (med, sigma)
    cat_freqs = {}
    for col in cat_vars:
        ser_clean = clean_missing_string(stacked[col].astype(object))
        freq = ser_clean.value_counts(normalize=True, dropna=True)
        if not freq.empty:
            cat_freqs[col] = freq

    tp_col = stacked.get("_tp", pd.Series(["?"] * len(stacked)))
    tracked = set(numerics) | set(cat_vars)

    site_rows = []
    for site in eligible_sites:
        mask = stacked["_site"] == site
        sub = stacked[mask]
        z_sum = 0.0; z_n = 0
        for col, (med, sigma) in num_stats.items():
            num = to_numeric_clean(sub[col]).dropna()
            if num.empty:
                continue
            zvals = ((num - med) / sigma).abs()
            z_sum += float(zvals.sum())
            z_n += int(zvals.size)
        avg_z = z_sum / z_n if z_n else 0.0

        rar_sum = 0.0; rar_n = 0
        for col, freq in cat_freqs.items():
            ser_clean = clean_missing_string(sub[col].astype(object)).dropna()
            if ser_clean.empty:
                continue
            rar = (-np.log10(ser_clean.map(freq).clip(lower=1e-6))).astype(float)
            rar_sum += float(rar.sum())
            rar_n += int(rar.size)
        avg_rar = rar_sum / rar_n if rar_n else 0.0

        # Missingness vectorized per (site, tp) block; replaces the
        # nested Python loop that materialized a 3000-col Series per row.
        miss_sum = 0.0; miss_rows = 0
        for tp_val in sub["_tp"].astype(str).unique():
            alive_cols = list(alive.get(str(tp_val), set()) & tracked)
            if not alive_cols:
                continue
            block = sub.loc[sub["_tp"].astype(str) == tp_val, alive_cols]
            if block.empty:
                continue
            is_miss = block.isna() | block.astype(str).apply(lambda c: c.str.strip() == "")
            rates = is_miss.mean(axis=1).to_numpy(dtype=float)
            miss_sum += float(np.sum(rates))
            miss_rows += int(rates.size)
        avg_miss = miss_sum / miss_rows if miss_rows else 0.0

        site_rows.append((site, int(mask.sum()), avg_z, avg_rar, avg_miss))

    if not site_rows:
        return []
    sf = pd.DataFrame(site_rows, columns=["site", "n", "avg_abs_z", "avg_rarity", "avg_missing"])
    if len(sf) < 3:
        return []
    # Same raw-composite caveat as the subject composite (see note there):
    # component standardization is the principled fix but needs a real-data
    # threshold recalibration, so we keep the raw weighted sum here.
    sf["composite"] = sf["avg_abs_z"] + 0.5 * sf["avg_rarity"] + 1.5 * sf["avg_missing"]
    composite_vals = sf["composite"].to_numpy(dtype=float)
    med, sigma = robust_center_scale(composite_vals)
    if not np.isfinite(sigma) or sigma <= 0:
        return []
    sf["composite_z"] = (sf["composite"] - med) / sigma
    sf = sf.sort_values("composite_z", ascending=False)

    out: List[dict] = []
    for _, r in sf.iterrows():
        z = float(r["composite_z"])
        if z < 2.5:
            break
        out.append(Finding(
            anomaly_type=ANOMALY_TYPE_SITE,
            severity_score=calibrate(z, threshold=2.5, extreme=6.0),
            raw_score=z,
            network=network,
            site_id=str(r["site"]),
            subjectid="(site-level)",
            variable="(site composite)",
            observed_value=(
                f"avg|z|={r['avg_abs_z']:.2f}, avg_rarity={r['avg_rarity']:.2f}, "
                f"missing={r['avg_missing']:.2%}, n={int(r['n'])}"
            ),
            expected_value=f"network composite median {med:.2f}, sigma {sigma:.2f}",
            explanation=(
                f"Site composite z = {z:.2f} against the rest of the network. "
                f"Combines numeric outlierness, categorical rarity, and missingness."
            ),
            method="composite z of site-aggregated metrics",
        ).to_row())
    return out
