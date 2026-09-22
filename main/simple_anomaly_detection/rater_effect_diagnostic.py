"""Diagnostic for the rater_effect detector -- calibration, not detection.

When ``rater_effect`` returns nothing on real data even though rater fields are
found and (rater, item) pairs are scanned, the cause is almost always that real
rater effects are subtler than the detector's deliberately conservative gates,
or that the per-subject MODAL collapse cannot see "partial leaning" (a rater who
inflates a value without it ever being a subject's modal score). This module
reveals the actual effect-size landscape so the gates / statistic can be
calibrated -- it does NOT change the detector.

For every (rater field, scored item, value) candidate (the SAME scoping and
guards the detector uses), it computes the rater-vs-comparison contrast UNGATED
under two views, side by side:

  modal view  : fraction of the rater's subjects whose MODAL score is v -- what
                the shipping detector tests.
  rate view   : the rater's mean PER-SUBJECT RATE of v (each subject contributes
                the fraction of their visits scored v). This is visit-count
                neutral and DOES see partial leaning, with a Welch t computed on
                per-subject rates (subject = unit, so no repeated-visit
                pseudoreplication).

Comparing the two columns answers the calibration question directly: if the
rate-view diffs are large where the modal-view diffs are ~0, real effects exist
but are partial-leaning that modal collapse hides (-> move to a rate statistic);
if both are moderate but below the gates, just relax the thresholds.

Tiers match the detector: within-site (a home rater vs co-located home raters,
on that site's subjects only) when the home site has >= 2 home raters, else
cohort (vs the whole network, site-confounded).

Outputs (written to the output folder):
  rater_effect_diagnostic.csv          full ranked candidate table, INCLUDING
                                       rater ids -- treat as PHI, keep local.
  rater_effect_diagnostic_summary.txt  effect-size distribution + threshold
                                       counts + top rows with NO rater ids --
                                       safe to share for calibration.

Run:
  python -m simple_anomaly_detection.rater_effect_diagnostic -i INPUT -o OUTPUT
  (or set FORMQC_ANOMALY_INPUT / FORMQC_ANOMALY_OUTPUT, like the main runner)
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .common import (
    classify_columns, clean_missing_string, stack_by_network,
)
from .rater_effect import (
    MAX_DISTINCT_VALUES, MIN_RATE_DIFF, MIN_RATER_RATE, MIN_RATERS_FOR_VAR,
    MIN_SITE_RATERS_FOR_WITHIN, MIN_SUBJECTS_PER_RATER, MIN_VALUE_SUBJECTS,
    _canonicalize_values, _collapse_modal, _load_var_to_form,
    _scored_vars_for_rater, _valid_rater_col, is_rater_variable,
)

# Thresholds the summary reports candidate counts at, so you can see where to
# set the gates. Purely for the report; the detector is untouched.
DIFF_GRID: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


# ---------------------------------------------------------------------------
# Per-(rater, subject) representations and the leave-one-out contrast.
# ---------------------------------------------------------------------------

def _pivots(rater: pd.Series, subj: pd.Series,
            val: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Two (rater, subject) x value tables: per-subject value RATE (fraction of
    the subject's visits scored v) and the per-subject MODAL indicator."""
    base = pd.DataFrame({"rater": rater.to_numpy(), "subj": subj.to_numpy(),
                         "val": val.to_numpy()})
    nf = base.groupby(["rater", "subj"]).size().rename("n").reset_index()
    cnt = base.groupby(["rater", "subj", "val"]).size().rename("c").reset_index()
    merged = cnt.merge(nf, on=["rater", "subj"])
    merged["rate"] = merged["c"] / merged["n"]
    piv_rate = merged.pivot_table(index=["rater", "subj"], columns="val",
                                  values="rate", fill_value=0.0)

    modal = _collapse_modal(rater, subj, val).assign(one=1.0)
    piv_modal = modal.pivot_table(index=["rater", "subj"], columns="val",
                                  values="one", fill_value=0.0)

    cols = piv_rate.columns.union(piv_modal.columns)
    piv_rate = piv_rate.reindex(columns=cols, fill_value=0.0)
    piv_modal = (piv_modal.reindex(columns=cols, fill_value=0.0)
                 .reindex(index=piv_rate.index, fill_value=0.0))
    return piv_rate, piv_modal


def _loo(piv: pd.DataFrame) -> dict:
    """Leave-one-rater-out two-sample stats per (rater, value), treating each
    (rater, subject) row as one observation (subject = unit). Returns matrices
    indexed by rater x value."""
    g = piv.groupby(level="rater")
    s = g.sum()
    ss = (piv ** 2).groupby(level="rater").sum()
    n = g.size().astype(float)
    npos = (piv > 0).groupby(level="rater").sum()
    total_s = s.sum(axis=0)
    total_ss = ss.sum(axis=0)
    total_n = float(n.sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        rater_mean = s.div(n, axis=0)
        rater_var = (ss - rater_mean.pow(2).mul(n, axis=0)).div(
            (n - 1).clip(lower=1), axis=0)
        rest_n = total_n - n
        rest_s = s.rsub(total_s, axis=1)
        rest_mean = rest_s.div(rest_n, axis=0)
        rest_ss = ss.rsub(total_ss, axis=1)
        rest_var = (rest_ss - rest_mean.pow(2).mul(rest_n, axis=0)).div(
            (rest_n - 1).clip(lower=1), axis=0)
        se = (rater_var.div(n, axis=0) + rest_var.div(rest_n, axis=0)).pow(0.5)
        diff = rater_mean - rest_mean
        t = diff / se.replace(0.0, np.nan)
    return {"diff": diff, "t": t, "rater_mean": rater_mean, "rest_mean": rest_mean,
            "n": n, "rest_n": rest_n, "npos": npos}


def _emit(network: str, rater_col: str, item: str, tier: str, site_label,
          emit_raters, site_of_rater: dict,
          rate: dict, modal: dict) -> List[dict]:
    rows: List[dict] = []
    if rate is None:
        return rows
    diff, t = rate["diff"], rate["t"]
    rmean, restmean = rate["rater_mean"], rate["rest_mean"]
    n, restn, npos = rate["n"], rate["rest_n"], rate["npos"]
    mdiff = modal["diff"] if modal is not None else None
    mr = modal["rater_mean"] if modal is not None else None
    mrest = modal["rest_mean"] if modal is not None else None

    def _g(df, r, v):
        try:
            return float(df.at[r, v])
        except Exception:
            return float("nan")

    for r in emit_raters:
        if r not in diff.index:
            continue
        for v in diff.columns:
            d = _g(diff, r, v)
            if not np.isfinite(d) or d <= 0:
                continue
            tv = _g(t, r, v)
            md = _g(mdiff, r, v) if mdiff is not None else float("nan")
            rows.append({
                "network": network,
                "rater_variable": rater_col,
                "variable": item,
                "scored_value": str(v),
                "rater_id": str(r),
                "tier": tier,
                "site_id": (site_label if site_label is not None
                            else str(site_of_rater.get(r, ""))),
                "n_rater_subjects": int(n.at[r]),
                "n_compare_subjects": int(restn.at[r]),
                "n_subjects_value": int(_g(npos, r, v)) if not np.isnan(_g(npos, r, v)) else 0,
                "subj_rate_rater": round(_g(rmean, r, v), 4),
                "subj_rate_compare": round(_g(restmean, r, v), 4),
                "subj_rate_diff": round(d, 4),
                "welch_t": (round(tv, 2) if np.isfinite(tv) else ""),
                "modal_rate_rater": (round(_g(mr, r, v), 4) if mr is not None else ""),
                "modal_rate_compare": (round(_g(mrest, r, v), 4) if mrest is not None else ""),
                "modal_diff": (round(md, 4) if np.isfinite(md) else ""),
            })
    return rows


def _candidates_for_pair(network: str, rater_col: str, item: str,
                         rater_k: pd.Series, subj_k: pd.Series, val_k: pd.Series
                         ) -> List[dict]:
    piv_rate, piv_modal = _pivots(rater_k, subj_k, val_k)
    if piv_rate.empty:
        return []
    raters_idx = piv_rate.index.get_level_values("rater")
    site_idx = pd.Series(piv_rate.index.get_level_values("subj")).str[:2].to_numpy()
    site_df = pd.DataFrame({"rater": raters_idx.to_numpy(), "site": site_idx})
    site_of_rater = (site_df.groupby("rater")["site"]
                     .agg(lambda s: s.value_counts().idxmax()).to_dict())

    n_subj = piv_rate.groupby(level="rater").size()
    qual = [r for r in n_subj.index if int(n_subj[r]) >= MIN_SUBJECTS_PER_RATER]
    if len(qual) < MIN_RATERS_FOR_VAR:
        return []
    raters_by_site: Dict[str, List[str]] = defaultdict(list)
    for r in qual:
        raters_by_site[str(site_of_rater.get(r, ""))].append(r)
    multi_sites = {s for s, rl in raters_by_site.items()
                   if len(rl) >= MIN_SITE_RATERS_FOR_WITHIN}
    multi_raters = {r for s in multi_sites for r in raters_by_site[s]}

    rater_arr = raters_idx.to_numpy()
    rows: List[dict] = []

    # within-site tier (home raters, on that site's subjects only -- matches
    # the detector's corrected, contamination-free contrast).
    for site in multi_sites:
        home = set(raters_by_site[site])
        mask = np.array([r in home for r in rater_arr]) & (site_idx == site)
        if not mask.any():
            continue
        pr = piv_rate[mask]
        pm = piv_modal[mask]
        if pr.index.get_level_values("rater").nunique() < 2:
            continue
        rows.extend(_emit(network, rater_col, item, "within_site", site,
                          raters_by_site[site], site_of_rater, _loo(pr), _loo(pm)))

    # cohort tier: singleton-home raters vs the whole network.
    singletons = [r for r in qual if r not in multi_raters]
    if singletons:
        rows.extend(_emit(network, rater_col, item, "cohort", None,
                          singletons, site_of_rater, _loo(piv_rate),
                          _loo(piv_modal)))
    return rows


def diagnose(per_slice: Dict, classify: Dict = None) -> Tuple[pd.DataFrame, str]:
    """Return (ranked candidate DataFrame, scoping source)."""
    var_to_form, source = _load_var_to_form()
    by_network: Dict[str, List] = defaultdict(list)
    for (network, tp), df in per_slice.items():
        if df is None or df.empty or "subjectid" not in df.columns:
            continue
        by_network[network].append((tp, df))

    rows: List[dict] = []
    for network, items in by_network.items():
        try:
            stacked = stack_by_network(items)
            cats = classify_columns(stacked)
            scored = [c for c in (cats["binary"] + cats["low_card"])
                      if c != "_tp" and not is_rater_variable(c)]
            subj_all = stacked["subjectid"]
            rater_cols = [c for c in stacked.columns
                          if is_rater_variable(c)
                          and _valid_rater_col(stacked[c], subj_all)]
            for rc in rater_cols:
                for it in _scored_vars_for_rater(rc, scored, var_to_form):
                    rs = clean_missing_string(stacked[rc].astype(object))
                    vs = clean_missing_string(stacked[it].astype(object))
                    keep = rs.notna() & vs.notna()
                    if int(keep.sum()) < MIN_SUBJECTS_PER_RATER * MIN_RATERS_FOR_VAR:
                        continue
                    rater_k = _canonicalize_values(rs[keep])
                    val_k = _canonicalize_values(vs[keep])
                    subj_k = subj_all[keep].astype(str)
                    if val_k.nunique() < 2 or val_k.nunique() > MAX_DISTINCT_VALUES:
                        continue
                    rows.extend(_candidates_for_pair(network, rc, it, rater_k,
                                                     subj_k, val_k))
        except Exception as e:
            print(f"  [rater_diagnostic] WARN network {network}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("subj_rate_diff", ascending=False).reset_index(drop=True)
    return df, source


# ---------------------------------------------------------------------------
# Non-PHI summary.
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> str:
    if df.empty:
        return ("No candidates at all: every (rater field, item) pair produced "
                "zero (rater, value) cells with a positive rater-vs-comparison "
                "difference. The rater fields and items are present, but no "
                "rater scores any value even slightly more than peers -- check "
                "the funnel log lines, or the scoping.")

    lines: List[str] = []

    def _stat(s: pd.Series) -> str:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return "n/a"
        return (f"max={s.max():.3f} p99={s.quantile(.99):.3f} "
                f"p95={s.quantile(.95):.3f} p90={s.quantile(.90):.3f} "
                f"median={s.median():.3f}")

    # Effect-size floor so the distribution isn't dominated by tiny-n noise.
    base = df[(pd.to_numeric(df["n_subjects_value"], errors="coerce") >= MIN_VALUE_SUBJECTS)
              & (pd.to_numeric(df["n_rater_subjects"], errors="coerce") >= MIN_SUBJECTS_PER_RATER)]

    lines.append(f"candidates: {len(df)} total; {len(base)} with "
                 f">= {MIN_VALUE_SUBJECTS} value-subjects and "
                 f">= {MIN_SUBJECTS_PER_RATER} rater-subjects")
    for tier in ("within_site", "cohort"):
        lines.append(f"  by tier: {tier} = {int((df['tier'] == tier).sum())}")
    lines.append("")
    lines.append(f"per-subject RATE diff  (catches partial leaning): "
                 f"{_stat(base['subj_rate_diff'])}")
    lines.append(f"per-subject MODAL diff (what the detector uses):  "
                 f"{_stat(base['modal_diff'])}")
    lines.append("")
    lines.append("candidate counts at each difference threshold "
                 "(rate view | modal view):")
    rate_vals = pd.to_numeric(base["subj_rate_diff"], errors="coerce")
    modal_vals = pd.to_numeric(base["modal_diff"], errors="coerce")
    for thr in DIFF_GRID:
        nr = int((rate_vals >= thr).sum())
        nm = int((modal_vals >= thr).sum())
        lines.append(f"  diff >= {thr:.2f}:   rate {nr:>6}   |   modal {nm:>6}")
    lines.append("")
    lines.append("interpretation: if 'rate' counts are much larger than 'modal' "
                 "counts, real effects exist but are partial-leaning that the "
                 "modal collapse hides (-> switch to a rate statistic). If both "
                 "are small at 0.20, relax the gates toward the diff where "
                 "candidates appear.")
    lines.append("")

    show_cols = ["network", "variable", "scored_value", "tier",
                 "n_rater_subjects", "n_compare_subjects", "n_subjects_value",
                 "subj_rate_rater", "subj_rate_compare", "subj_rate_diff",
                 "welch_t", "modal_diff"]
    top = (base if not base.empty else df).head(20)
    lines.append("top 20 candidates by per-subject rate diff (NO rater ids):")
    lines.append(top[show_cols].to_string(index=False))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse

    from .runner import SimpleAnomalyDetector

    p = argparse.ArgumentParser(
        prog="simple_anomaly_detection.rater_effect_diagnostic",
        description="Ungated rater-effect candidate report (calibration aid).")
    p.add_argument("-i", "--input", default=None, help="input folder (combined CSVs)")
    p.add_argument("-o", "--output", default=None, help="output folder")
    p.add_argument("--top", type=int, default=2000,
                   help="rows to keep in the full CSV (default 2000)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    kwargs = {}
    if args.input:
        kwargs["input_path"] = args.input
    if args.output:
        kwargs["output_path"] = args.output
    det = SimpleAnomalyDetector(**kwargs)
    per_slice = det._load_slices()
    if not per_slice:
        print("[rater_diagnostic] no slices loaded; nothing to do.")
        return 0

    df, source = diagnose(per_slice)
    out_dir = det.output_path
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "rater_effect_diagnostic.csv")
    df.head(args.top).to_csv(csv_path, index=False)
    summary = summarize(df)
    txt_path = os.path.join(out_dir, "rater_effect_diagnostic_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"scoping source: {source}\n\n{summary}\n")

    print(f"[rater_diagnostic] scoping source: {source}")
    print(summary)
    print(f"\n[rater_diagnostic] full ranked candidates (HAS rater ids, keep "
          f"local / PHI): {csv_path}")
    print(f"[rater_diagnostic] non-PHI summary (safe to share): {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
