"""Detector 2 - Time series outlier detection.

Scans within-subject trajectories for unusual peaks/drops. The trick
the prompt asks for: only flag peaks of a magnitude that is *unusual
for this variable*. A variable that bounces every visit shouldn't
flood the report.

The change unit is a *rate*, anchored to interview DATES not timepoints
------------------------------------------------------------------------
A change of +5 over a 30-day gap is ordinary; the same +5 over 3 days is
not -- yet the nominal timepoint label ("baseline->month1") hides that,
and a subject whose real visit cadence is irregular (or whose visits are
mislabeled) gets compared against the wrong gap. So each variable is
scored on its per-day RATE of change, using the actual elapsed time
between the governing form's **interview dates**:

    * Each numeric variable is mapped to its form's interview-date column
      (``chr<form>_interview_date``); variables with no per-form date fall
      back to a single shared visit/interview date, and only failing that
      to the legacy timepoint-ordered delta (so coverage never regresses).
    * Within each subject, observations are ordered by ACTUAL interview
      date (not the nominal timepoint), and consecutive pairs give a
      change ``delta`` over an elapsed ``days`` gap -> ``rate = delta/days``.
    * Rate distributions are estimated within the same nominal transition
      (baseline->month1, month1->month2, ...). Dividing by elapsed days does
      not make early and late study phases exchangeable: a variable can change
      quickly during one phase and remain flat during another. Transition
      stratification prevents an ordinary phase from looking anomalous merely
      because a more common phase has a different center or spread.
    * peak_rarity (fraction of deltas at |z|>=threshold) still screens
      out variables that are merely noisy; flag subjects whose
      |rate_z| >= Z_THRESHOLD.

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
    Finding, classify_columns, clean_dates_series,
    matches_excluded_substrings, robust_center_scale, severity_from_z,
    site_from_subjectid, to_numeric_clean, short,
)

ANOMALY_TYPE = "time_series_outlier"
Z_THRESHOLD = 4.0
MIN_DELTAS = 30
MAX_PEAK_RARITY = 0.05    # variables noisier than this are skipped
# Floor (days) on the interview-date gap used as the rate denominator. A
# genuine longitudinal change measured a handful of days apart (a re-entry,
# an off-schedule visit) would otherwise divide by a tiny gap and manufacture
# a huge rate from ordinary measurement noise. Deltas with a non-positive gap
# (same-day or out-of-order interview dates) carry no rate signal and are
# dropped entirely; deltas with a positive but very short gap are kept with
# the denominator floored here. Judgment call -- tune to your visit cadence.
MIN_ELAPSED_DAYS = 7.0

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


def _interview_date_columns(date_cols) -> Dict[str, str]:
    """Map ``form-prefix -> '<prefix>_interview_date' column`` over the date
    columns present. The governing date for a variable is then the longest
    such prefix the variable starts with (resolved in ``_form_date_for_var``)."""
    out: Dict[str, str] = {}
    for d in date_cols:
        dl = str(d).lower()
        if dl.endswith("_interview_date"):
            out[dl[: -len("_interview_date")]] = d
    return out


def _form_date_for_var(var: str, interview_map: Dict[str, str],
                       shared_date) -> str:
    """The interview-date column governing ``var``: the longest form prefix in
    ``interview_map`` that ``var`` starts with (``chrbprs_total`` ->
    ``chrbprs_interview_date``; the more specific ``chrpsychs_scr`` wins over
    ``chrpsychs`` when both exist), else the single shared visit date, else
    ``None`` (caller uses the timepoint fallback)."""
    vl = str(var).lower()
    best = None
    for pref, dcol in interview_map.items():
        if (vl == pref or vl.startswith(pref + "_")) and \
                (best is None or len(pref) > len(best[0])):
            best = (pref, dcol)
    return best[1] if best else shared_date


def _pick_shared_date(date_cols):
    """A single non-form visit/interview date for variables that have no
    per-form ``<prefix>_interview_date`` (e.g. the synthetic demo's
    ``visit_date``). Prefers a 'visit'/'interview'-named column; never a
    consent / dob / screening / data-entry date."""
    excl = ("consent", "dob", "birth", "screen", "entry")
    cands = [d for d in date_cols
             if not str(d).lower().endswith("_interview_date")
             and ("visit" in str(d).lower() or "interview" in str(d).lower())
             and not any(x in str(d).lower() for x in excl)]
    return cands[0] if cands else None


def _detect_network(network: str, ordered_items: List) -> List[dict]:
    if not ordered_items:
        return []
    numeric_cols: set = set()
    date_cols: set = set()
    for _, df, cats in ordered_items:
        if cats is None:
            cats = classify_columns(df)
        numeric_cols.update(cats["numeric"])
        date_cols.update(cats["date"])
    if EXTRA_EXCLUDED_SUBSTRINGS:
        numeric_cols = {c for c in numeric_cols
                        if not matches_excluded_substrings(c, EXTRA_EXCLUDED_SUBSTRINGS)}
    if not numeric_cols:
        return []

    # Map each numeric variable to the interview-date column that governs it.
    interview_map = _interview_date_columns(date_cols)
    shared_date = _pick_shared_date(date_cols)
    var_date = {col: _form_date_for_var(col, interview_map, shared_date)
                for col in numeric_cols}
    needed_dates = {d for d in var_date.values() if d}

    # Derive a stable integer rank from the already-sorted item order so
    # the DataFrame column assignment stays scalar. _tp_rank() returns a
    # tuple for sorting, which broadcasts wrong when assigned as a column.
    tp_to_int_rank = {tp: i for i, (tp, _, _) in enumerate(ordered_items)}

    stacked = []
    for tp, df, _ in ordered_items:
        cols = [c for c in df.columns if c in numeric_cols or c in needed_dates]
        keep = ["subjectid"] + cols
        sub = df[list(dict.fromkeys(keep))].copy()
        sub["_tp"] = tp
        sub["_tp_rank"] = tp_to_int_rank[tp]
        stacked.append(sub)
    long_df = pd.concat(stacked, ignore_index=True, sort=False)
    # Parse the needed date columns once (strips AMPSCZ date sentinels).
    for d in needed_dates:
        if d in long_df.columns:
            long_df[d] = clean_dates_series(long_df[d])
    long_df = long_df.sort_values(["subjectid", "_tp_rank"]).reset_index(drop=True)

    out: List[dict] = []
    n_rate = n_tp = 0
    for col in numeric_cols:
        dcol = var_date.get(col)
        try:
            if dcol and dcol in long_df.columns:
                out.extend(_detect_variable_rate(long_df, col, dcol, network))
                n_rate += 1
            else:
                out.extend(_detect_variable_by_tp(long_df, col, network))
                n_tp += 1
        except Exception as e:
            print(f"  [time_series] WARN var {network}/{col}: {e}")
    collapsed = _collapse_adjacent_reversals(out)
    print(f"  [time_series] {network}: {len(numeric_cols)} numeric var(s) "
          f"({n_rate} scored by interview-date rate, {n_tp} by timepoint "
          f"fallback) -> {len(collapsed)} findings after collapsing "
          f"{len(out) - len(collapsed)} adjacent reversal row(s)")
    return collapsed


def _detect_variable_rate(long_df: pd.DataFrame, col: str, dcol: str,
                          network: str) -> List[dict]:
    """Score variable ``col`` on its per-day change RATE, using the elapsed
    time between consecutive ``dcol`` (interview-date) observations. Rates are
    scored within nominal-transition strata: elapsed-time normalization does
    not make biologically different study phases exchangeable."""
    if col not in long_df.columns or dcol not in long_df.columns:
        return []
    num = to_numeric_clean(long_df[col])
    if int(num.notna().sum()) < MIN_DELTAS:
        return []

    work = long_df[["subjectid", "_tp", "_tp_rank"]].copy()
    work["v"] = num.to_numpy()
    work["d"] = long_df[dcol].to_numpy()  # datetime64 (already cleaned/parsed)
    work = work[work["v"].notna() & work["d"].notna()]
    if len(work) < MIN_DELTAS:
        return []

    # Order each subject's observations by ACTUAL interview date (tp_rank only
    # breaks exact-date ties), so "consecutive" is chronological, not nominal.
    work = work.sort_values(["subjectid", "d", "_tp_rank"])
    grp = work.groupby("subjectid", sort=False)
    work["v_prev"] = grp["v"].shift(1)
    work["d_prev"] = grp["d"].shift(1)
    work["tp_prev"] = grp["_tp"].shift(1)
    work["pair"] = work["tp_prev"].astype(str) + "->" + work["_tp"].astype(str)

    delta = work["v"] - work["v_prev"]
    elapsed = (work["d"] - work["d_prev"]).dt.total_seconds() / 86400.0
    # Drop deltas with a non-positive gap (same-day / out-of-order interview
    # dates); floor the positive denominator so a few-day gap can't blow the
    # rate up out of measurement noise.
    valid = work["v_prev"].notna() & work["d_prev"].notna() & (elapsed > 0)
    if not valid.any():
        return []
    denom = elapsed.clip(lower=MIN_ELAPSED_DAYS)
    rate = (delta / denom).where(valid)

    rows: List[dict] = []
    scored = work.loc[valid].copy()
    scored["delta"] = delta[valid].to_numpy(dtype=float)
    scored["elapsed"] = elapsed[valid].to_numpy(dtype=float)
    scored["effective_elapsed"] = denom[valid].to_numpy(dtype=float)
    scored["rate"] = rate[valid].to_numpy(dtype=float)

    for pair_name, sub in scored.groupby("pair", sort=False):
        r_vals = sub["rate"].dropna().to_numpy(dtype=float)
        if r_vals.size < MIN_DELTAS:
            continue
        med, sigma = robust_center_scale(r_vals)
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        z = (sub["rate"] - med) / sigma
        peak_rarity = float((z.abs() >= Z_THRESHOLD).sum() / max(r_vals.size, 1))
        if peak_rarity > MAX_PEAK_RARITY:
            continue
        bad = z.abs() >= Z_THRESHOLD
        if not bad.any():
            continue

        for idx, row in sub.loc[bad].iterrows():
            zi = float(z.loc[idx])
            if not np.isfinite(zi):
                continue
            subj = str(row["subjectid"])
            tp_now = str(row["_tp"])
            tp_prev = str(row["tp_prev"])
            d = float(row["delta"])
            days = float(row["elapsed"])
            effective_days = float(row["effective_elapsed"])
            ratei = float(row["rate"])
            severity = severity_from_z(zi, Z_THRESHOLD, 10.0)
            floor_note = (
                f"; rate denominator floored to {effective_days:.0f} d"
                if effective_days > days else ""
            )
            rows.append(Finding(
                anomaly_type=ANOMALY_TYPE,
                severity_score=severity,
                raw_score=abs(zi),
                network=network,
                timepoint=f"{tp_prev}->{tp_now}",
                site_id=site_from_subjectid(subj),
                subjectid=subj,
                variable=col,
                observed_value=(f"{short(row['v_prev'])} -> {short(row['v'])} "
                                f"(delta {d:+.4g} over {days:.0f} actual d; "
                                f"effective denominator {effective_days:.0f} d; "
                                f"rate {ratei:+.4g}/d)"),
                expected_value=(f"typical {pair_name} change-rate median "
                                f"{med:+.4g}/d, sigma {sigma:.4g}/d "
                                f"(n={r_vals.size})"),
                explanation=(
                    f"Within-subject change of {d:+.4g} over {days:.0f} actual "
                    f"days between interview dates ({pair_name}) gives a "
                    f"scoring rate of {ratei:+.4g}/day{floor_note}; z = "
                    f"{zi:+.2f} within the same nominal transition. Peak rarity "
                    f"{peak_rarity:.2%}, so not just a noisy variable bouncing."
                ),
                method=("MAD on within-subject change RATE, stratified by "
                        "nominal transition"),
                extra={"peak_rarity": round(peak_rarity, 4),
                       "n_deltas": int(r_vals.size),
                       "elapsed_days": round(days, 1),
                       "effective_elapsed_days": round(effective_days, 1),
                       "denominator_was_floored": bool(effective_days > days),
                       "date_column": dcol,
                       "transition_from": tp_prev,
                       "transition_to": tp_now,
                       "transition_stratum": str(pair_name),
                       "delta": d,
                       "signed_z": zi},
            ).to_row())
    return rows


def _detect_variable_by_tp(long_df: pd.DataFrame, col: str, network: str) -> List[dict]:
    """Legacy fallback for variables with NO interview-date column: consecutive
    within-subject deltas ordered by nominal timepoint, MAD stratified per
    (tp_prev, tp_now) pair (pooling all gaps would inflate sigma with the
    gap-driven natural drift). Used only when neither a per-form interview date
    nor a shared visit date is available for the variable."""
    if col not in long_df.columns:
        return []
    num = to_numeric_clean(long_df[col])
    if num.notna().sum() < MIN_DELTAS:
        return []

    work = long_df[["subjectid", "_tp", "_tp_rank"]].copy()
    work["v"] = num
    # Consecutive means consecutive OBSERVED values. Keeping an empty month1
    # row in the shift chain would lose the legitimate baseline->month2 delta.
    work = work[work["v"].notna()].copy()
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
                method="MAD on within-subject deltas, stratified by tp pair (no interview date)",
                extra={"peak_rarity": round(peak_rarity, 4),
                       "n_in_gap": int(d_vals.size),
                       "transition_from": tp_prev,
                       "transition_to": tp_now,
                       "transition_stratum": str(pair_name),
                       "delta": d,
                       "signed_z": zi},
            ).to_row())
    return rows


def _collapse_adjacent_reversals(rows: List[dict]) -> List[dict]:
    """Collapse the two edges produced by one anomalous middle visit.

    A single bad middle value normally creates an extreme jump into the visit
    and an opposite extreme jump out. Those are corroborating evidence for one
    subject-variable episode, not two independent review rows. Only adjacent,
    opposite-direction transitions are folded; monotonic changes and unrelated
    transitions remain separate.
    """
    if not rows:
        return rows

    groups: Dict[Tuple[str, str, str], List[dict]] = {}
    for row in rows:
        key = (str(row.get("network", "")), str(row.get("subjectid", "")),
               str(row.get("variable", "")))
        groups.setdefault(key, []).append(row)

    out: List[dict] = []
    for grp in groups.values():
        ordered = sorted(
            grp,
            key=lambda r: (
                _tp_rank(str(r.get("transition_from", ""))),
                _tp_rank(str(r.get("transition_to", ""))),
                -float(r.get("severity_score", 0) or 0),
            ),
        )
        used: set = set()
        for i, first in enumerate(ordered):
            if i in used:
                continue
            first_to = str(first.get("transition_to", ""))
            try:
                d1 = float(first.get("delta"))
            except (TypeError, ValueError):
                d1 = float("nan")
            partner_idx = None
            for j in range(i + 1, len(ordered)):
                if j in used:
                    continue
                second = ordered[j]
                if first_to != str(second.get("transition_from", "")):
                    continue
                try:
                    d2 = float(second.get("delta"))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(d1) and np.isfinite(d2) and d1 * d2 < 0:
                    partner_idx = j
                    break

            if partner_idx is None:
                out.append(first)
                used.add(i)
                continue

            second = ordered[partner_idx]
            used.update((i, partner_idx))
            pair = [first, second]
            rep = dict(max(pair, key=lambda r: float(r.get("severity_score", 0) or 0)))
            start = str(first.get("transition_from", ""))
            middle = first_to
            end = str(second.get("transition_to", ""))
            direction = "spike" if d1 > 0 else "drop"
            rep["timepoint"] = f"{start}->{middle}->{end}"
            rep["transition_from"] = start
            rep["transition_to"] = end
            rep["episode_middle_timepoint"] = middle
            rep["episode_direction"] = direction
            rep["n_corroborating_transitions"] = 2
            rep["corroborating_transitions"] = "; ".join(
                f"{r.get('timepoint', '')}: {r.get('observed_value', '')}" for r in pair
            )
            rep["observed_value"] = rep["corroborating_transitions"]
            rep["explanation"] = (
                f"Adjacent opposite-direction anomalies into and out of {middle} "
                f"are consistent with one anomalous middle-visit {direction}. "
                f"Collapsed two transitions; see corroborating_transitions."
            )
            out.append(rep)

    return sorted(
        out,
        key=lambda r: (
            -float(r.get("severity_score", 0) or 0),
            str(r.get("network", "")), str(r.get("subjectid", "")),
            str(r.get("variable", "")), str(r.get("timepoint", "")),
        ),
    )
