"""Timeline-date anomalies across scheduled study timepoints.

This detector is deliberately narrower than :mod:`date_anomaly`:

* only the *same* date variable is compared across distinct timepoints;
* conversion, floating, and unknown event labels are excluded;
* a later scheduled visit dated before an earlier visit is a deterministic
  ordering error and does not require a large reference cohort;
* unusual cadence must be both far from the nominal schedule and unusual for
  the cohort.  Routine visit-window variation is therefore not reported.

Candidate pair rows are collapsed to one reviewer-facing issue per
``(network, subject, date variable)`` so a single bad date cannot dominate the
tab merely because it disagrees with several other visits.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import json
import os
import re
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .common import (
    Finding,
    calibrate,
    classify_columns,
    clean_dates_series,
    matches_excluded_substrings,
    robust_center_scale,
    severity_from_z,
    site_from_subjectid,
)


ANOMALY_TYPE = "timeline_anomaly"

# Distributional cadence checks need enough paired participants to learn a
# stable reference.  Deterministic later-before-earlier checks do not use this
# gate and remain active in sparse late timepoints.
MIN_PAIR_N = 40
GAP_Z_THRESHOLD = 4.0

# A cadence must miss both the nominal schedule and the cohort center by at
# least two weeks.  This absorbs ordinary visit windows and the 28/29/30/31-day
# difference between calendar months and the nominal 30-day schedule.
MIN_GAP_ABS_DAYS = 14.0

# When the cohort MAD is zero, only use the exact center as a reference if it
# overwhelmingly dominates.  This catches one 500-day typo in an otherwise
# exact schedule without treating a genuinely bimodal cadence as anomalous.
ZERO_SCALE_DOMINANT_FRAC = 0.80

# Backward-compatible name for callers that imported the unfinished module's
# tuning constant.  It is no longer an exact-equality threshold.
FIXED_GAP_THRESHOLD_DAYS = MIN_GAP_ABS_DAYS

EXTRA_EXCLUDED_SUBSTRINGS: tuple = ()

# Match the production Date Report's administrative/non-target date fields.
# These are not longitudinal assessment timelines and must not reappear here
# under a different detector name.
EXCLUDED_TIMELINE_VARIABLES = frozenset({
    "chrmiss_interview_date",
    "chrpharm_interview_date",
    "chrpsychs_av_interview_date",
    "chrif_interview_date",
    "chrgpc_date",
    "chrcbc_interview_date",
    "chrcbccs_review_date",
})

_MONTH_RE = re.compile(r"^month[_-]?(\d+)$", re.IGNORECASE)
_NONCHRONOLOGICAL = frozenset({"conversion", "floating"})


def _default_form_contract_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
        "dependencies", "important_form_vars.json")


@lru_cache(maxsize=8)
def _load_date_form_contracts(path: str) -> Mapping[str, Tuple[dict, ...]]:
    """Map each interview-date field to its form missingness contract."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        # Keep the package's unrestricted/demo mode usable when copied without
        # repository dependencies. Production paths normally have this file;
        # the warning makes the resulting loss of missing-form gating visible.
        print(
            f"  [timeline_anomaly] WARN form contract not found at {path}; "
            "missing-form dates cannot be excluded"
        )
        return {}
    if not isinstance(payload, dict):
        raise ValueError(
            f"Timeline form contract must be a JSON object: {path}")

    by_date: Dict[str, List[dict]] = defaultdict(list)
    for form, info in payload.items():
        if not isinstance(info, dict):
            continue
        date_variable = str(info.get("interview_date_var", "")).strip()
        if not date_variable:
            continue
        by_date[date_variable].append({**info, "form": str(form)})
    return {key: tuple(value) for key, value in by_date.items()}


def _prescient_completion_variable(info: Mapping) -> str:
    completion = str(info.get("completion_var", "")).strip()
    if not completion:
        return ""
    completion = (completion + "_rpms").replace("_hc", "")
    completion = completion.replace("onboarding", "checkin")
    completion = completion.replace(
        "end_of_12month_study_pe", "checkin")
    return completion.replace("end_of_12month_study_p", "checkin")


def _form_marked_missing_mask(
    frame: pd.DataFrame,
    date_variable: str,
    contracts: Mapping[str, Tuple[dict, ...]],
    network: str,
) -> pd.Series:
    """Rows where the form owning ``date_variable`` is explicitly missing."""
    mask = pd.Series(False, index=frame.index, dtype=bool)
    for info in contracts.get(date_variable, ()):
        missing_variable = str(info.get("missing_var", "")).strip()
        if missing_variable and missing_variable in frame.columns:
            missing_values = pd.to_numeric(
                frame[missing_variable], errors="coerce")
            mask |= missing_values.eq(1)

        if str(network).strip().upper() == "PRESCIENT":
            completion_variable = _prescient_completion_variable(info)
            if completion_variable and completion_variable in frame.columns:
                completion_values = pd.to_numeric(
                    frame[completion_variable], errors="coerce")
                mask |= completion_values.isin([3, 4])
    return mask


def _is_timeline_date_variable(
    variable: object,
    contracts: Mapping[str, Tuple[dict, ...]],
) -> bool:
    """Whether a date field represents a repeatable visit assessment date.

    The form contract is authoritative for production REDCap interview dates.
    A narrow visit/interview-name fallback keeps synthetic and standalone data
    usable without admitting static consent, birth, entry, or modification
    dates whose values are not expected to advance at every timepoint.
    """
    name = str(variable).strip()
    lowered = name.casefold()
    if lowered in EXCLUDED_TIMELINE_VARIABLES:
        return False
    if name in contracts:
        return True
    # Only the two exact generic names are safe outside the REDCap contract.
    # Substring matching admitted unrelated/static fields such as
    # previous_visit_date and interview_training_date, recreating the same
    # all-participant noise this detector is designed to avoid.
    return lowered in {"visit_date", "interview_date"}


def _canonical_timepoint(value: object) -> str:
    """Return the canonical timeline label or ``''`` when unsupported."""
    text = str(value or "").strip().lower()
    if text in {"screening", "baseline", "conversion", "floating"}:
        return text
    match = _MONTH_RE.fullmatch(text)
    if match:
        return f"month{int(match.group(1))}"
    return ""


def _timepoint_position(value: object) -> Optional[int]:
    """Sortable chronological position; unsupported events return None."""
    tp = _canonical_timepoint(value)
    if not tp or tp in _NONCHRONOLOGICAL:
        return None
    if tp == "screening":
        return -1
    if tp == "baseline":
        return 0
    match = re.fullmatch(r"month(\d+)", tp)
    return int(match.group(1)) if match else None


def _scheduled_day(value: object) -> Optional[float]:
    """Nominal days from baseline for fixed-cadence timepoints.

    Screening has chronological meaning but no single nominal distance from
    baseline, so it participates in ordering checks only.
    """
    tp = _canonical_timepoint(value)
    if tp == "baseline":
        return 0.0
    match = re.fullmatch(r"month(\d+)", tp)
    return float(int(match.group(1)) * 30) if match else None


def _expected_gap_days(tp_a: str, tp_b: str) -> Optional[float]:
    """Signed nominal gap from earlier ``tp_a`` to later ``tp_b``.

    Unlike the unfinished implementation, month1 -> month2 is 30 days (not
    60), month3 -> month6 is 90 days (not 120), and same-timepoint pairs have
    no scheduled-gap check.
    """
    day_a = _scheduled_day(tp_a)
    day_b = _scheduled_day(tp_b)
    if day_a is None or day_b is None or day_b <= day_a:
        return None
    return day_b - day_a


def detect_all(
    per_slice: Dict,
    classify: Dict = None,
    important_form_vars_path: str = None,
    **_,
) -> List[dict]:
    """Detect timeline anomalies from wide ``(network, timepoint)`` slices."""
    classify = classify or {}
    contract_path = os.path.abspath(
        important_form_vars_path or _default_form_contract_path())
    form_contracts = _load_date_form_contracts(contract_path)
    by_network: Dict[str, List[Tuple[str, pd.DataFrame, object]]] = defaultdict(list)

    for (network, raw_tp), df in per_slice.items():
        tp = _canonical_timepoint(raw_tp)
        if (_timepoint_position(tp) is None or df is None or df.empty
                or "subjectid" not in df.columns):
            continue
        by_network[str(network)].append(
            (tp, df, classify.get((network, raw_tp))))

    rows: List[dict] = []
    # Unexpected failures are intentionally allowed to reach runner.py, which
    # records the detector as crashed.  Known per-column parse failures are
    # isolated below so one malformed date field does not erase a network.
    for network in sorted(by_network):
        rows.extend(_detect_network(
            network, by_network[network], form_contracts=form_contracts))
    return rows


def _detect_network(
    network: str,
    items: List,
    *,
    form_contracts: Mapping[str, Tuple[dict, ...]] = None,
) -> List[dict]:
    # Build same-variable series by timepoint.  There is intentionally no
    # within-timepoint all-pairs loop: unrelated dates on one form do not form
    # a longitudinal timeline and must never receive a 30-day expectation.
    by_column: Dict[str, List[Tuple[str, str, pd.Series]]] = defaultdict(list)
    form_contracts = form_contracts or {}

    for tp, df, cats in sorted(
            items, key=lambda item: (_timepoint_position(item[0]), item[0])):
        cats = classify_columns(df) if cats is None else cats
        # Contract-mapped interview dates are authoritative and must bypass
        # the shared anomaly-variable exclusions. Some real form prefixes
        # (notably chrax*/chrdig*) are intentionally excluded from generic
        # anomaly classification, which otherwise made their mapped interview
        # dates silently disappear from this dedicated timeline check.
        mapped_dates = [
            column for column in df.columns
            if str(column) in form_contracts
        ]
        generic_dates = [
            column for column in df.columns
            if str(column).strip().casefold()
            in {"visit_date", "interview_date"}
        ]
        candidate_dates = dict.fromkeys([
            *mapped_dates, *generic_dates, *cats.get("date", ())])
        date_columns = [
            c for c in candidate_dates
            if c in df.columns
            and _is_timeline_date_variable(c, form_contracts)
            and not matches_excluded_substrings(
                c, EXTRA_EXCLUDED_SUBSTRINGS)
        ]
        if not date_columns:
            continue

        ids = df["subjectid"].astype(str).str.strip()
        valid_id = ids.ne("") & ~ids.str.lower().isin(
            {"nan", "none", "na", "n/a"})

        for column in date_columns:
            # A duplicate label makes df[column] a DataFrame rather than a
            # Series.  It is ambiguous which physical field owns the value, so
            # fail this column closed while retaining every other date field.
            raw = df[column]
            if isinstance(raw, pd.DataFrame):
                print(
                    f"  [timeline_anomaly] WARN {network}/{tp} date col "
                    f"{column}: duplicate column label; dropping this column"
                )
                continue
            try:
                cleaned = clean_dates_series(raw)
            except Exception as exc:
                print(
                    f"  [timeline_anomaly] WARN {network}/{tp} date col "
                    f"{column}: {type(exc).__name__}: {exc}; dropping this "
                    "column"
                )
                continue

            marked_missing = _form_marked_missing_mask(
                df, str(column), form_contracts, network)
            if marked_missing.any():
                cleaned = cleaned.mask(marked_missing)

            series = pd.Series(
                cleaned.loc[valid_id].to_numpy(),
                index=ids.loc[valid_id].to_numpy(),
                name=column,
            )
            # The runner already excludes conflicting duplicate subject rows.
            # This defensive collapse also makes direct detector calls stable;
            # groupby.first chooses the first non-null value.
            series = series.groupby(level=0, sort=False).first()
            if series.notna().any():
                by_column[str(column)].append((tp, str(column), series))

    candidates: List[dict] = []
    for column in sorted(by_column):
        entries = sorted(
            by_column[column],
            key=lambda entry: (_timepoint_position(entry[0]), entry[0]),
        )
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                # Alias normalization can theoretically place duplicate slices
                # at the same canonical visit; those are not a timeline pair.
                if entries[i][0] == entries[j][0]:
                    continue
                candidates.extend(_pair(network, entries[i], entries[j]))

    collapsed = _deduplicate_findings(candidates)
    print(
        f"  [timeline_anomaly] {network}: {len(by_column)} repeated date "
        f"variable(s), {len(candidates)} pair finding(s) -> "
        f"{len(collapsed)} review issue(s)"
    )
    return collapsed


def _pair(
    network: str,
    left: Tuple[str, str, pd.Series],
    right: Tuple[str, str, pd.Series],
) -> List[dict]:
    """Compare one date variable at two canonically ordered timepoints."""
    tp_a, column, series_a = left
    tp_b, right_column, series_b = right
    if column != right_column:
        return []

    joined = pd.concat([series_a, series_b], axis=1, join="inner")
    joined.columns = ["a", "b"]
    joined = joined.dropna()
    if joined.empty:
        return []

    gap_days = (joined["b"] - joined["a"]).dt.total_seconds() / 86400.0
    label_a = f"{tp_a}::{column}"
    label_b = f"{tp_b}::{column}"
    pair_label = f"{label_a} | {label_b}"
    out: List[dict] = []

    # The canonical event order supplies the expected direction.  This is a
    # deterministic integrity check, not a learned 95%-majority relationship,
    # so sparse late visits and tied dates do not disable or bias it.
    for subject in gap_days.index[gap_days < 0]:
        gap = float(gap_days.loc[subject])
        row = joined.loc[subject]
        magnitude = abs(gap)
        out.append(Finding(
            anomaly_type=ANOMALY_TYPE,
            severity_score=calibrate(magnitude, threshold=1.0, extreme=90.0),
            raw_score=magnitude,
            network=network,
            timepoint=f"{tp_a} -> {tp_b}",
            site_id=site_from_subjectid(str(subject)),
            subjectid=str(subject),
            variable=pair_label,
            variables_involved=column,
            observed_value=(
                f"{tp_a}={row['a'].date()}, {tp_b}={row['b'].date()} "
                f"(gap={gap:.1f} days)"
            ),
            expected_value=(
                f"{tp_b} {column} must be on or after {tp_a} {column}"
            ),
            explanation=(
                f"The same date field at the later scheduled timepoint "
                f"{tp_b} is {magnitude:.1f} day(s) before its value at "
                f"{tp_a}."
            ),
            method="deterministic timeline ordering",
            extra={
                "timeline_variable": column,
                "timeline_pair": pair_label,
                "finding_kind": "order_reversal",
                "date_a": str(row["a"].date()),
                "date_b": str(row["b"].date()),
                "observed_gap_days": gap,
            },
        ).to_row())

    expected_gap = _expected_gap_days(tp_a, tp_b)
    nonnegative = gap_days[gap_days >= 0].dropna()
    if expected_gap is None or len(nonnegative) < MIN_PAIR_N:
        return out

    values = nonnegative.to_numpy(dtype=float)
    center, sigma = robust_center_scale(values)
    if not np.isfinite(center):
        return out

    schedule_deviation = np.abs(values - expected_gap)
    cohort_deviation = np.abs(values - center)
    if np.isfinite(sigma) and sigma > 0:
        raw_scores = cohort_deviation / sigma
        bad = (
            (schedule_deviation >= MIN_GAP_ABS_DAYS)
            & (cohort_deviation >= MIN_GAP_ABS_DAYS)
            & (raw_scores >= GAP_Z_THRESHOLD)
        )
        score_kind = "z"
    else:
        # A rounded cadence can have MAD=0 even with harmless 29/30/31-day
        # variation.  Treat the entire non-actionable window around the center
        # as the dominant schedule, rather than requiring 80% bit-for-bit
        # equality with the median; otherwise one 500-day typo is missed when
        # only (say) 70% of visits are exactly 30 days.
        center_share = float(
            (cohort_deviation < MIN_GAP_ABS_DAYS).mean())
        if center_share < ZERO_SCALE_DOMINANT_FRAC:
            return out
        raw_scores = cohort_deviation
        bad = (
            (schedule_deviation >= MIN_GAP_ABS_DAYS)
            & (cohort_deviation >= MIN_GAP_ABS_DAYS)
        )
        score_kind = "days"

    bad_positions = np.flatnonzero(bad)
    for position in bad_positions:
        subject = str(nonnegative.index[position])
        observed_gap = float(values[position])
        expected_deviation = observed_gap - expected_gap
        cohort_delta = observed_gap - center
        raw_score = float(raw_scores[position])
        row = joined.loc[subject]
        severity = (
            severity_from_z(raw_score, GAP_Z_THRESHOLD, 10.0)
            if score_kind == "z"
            else calibrate(raw_score, MIN_GAP_ABS_DAYS, 120.0)
        )
        out.append(Finding(
            anomaly_type=ANOMALY_TYPE,
            severity_score=severity,
            raw_score=raw_score,
            network=network,
            timepoint=f"{tp_a} -> {tp_b}",
            site_id=site_from_subjectid(subject),
            subjectid=subject,
            variable=pair_label,
            variables_involved=column,
            observed_value=(
                f"{tp_a}={row['a'].date()}, {tp_b}={row['b'].date()} "
                f"(gap={observed_gap:.1f} days)"
            ),
            expected_value=(
                f"nominal gap {expected_gap:.1f} days; cohort center "
                f"{center:.1f} days (n={len(nonnegative)})"
            ),
            explanation=(
                f"The {observed_gap:.1f}-day interval is "
                f"{abs(expected_deviation):.1f} day(s) from the nominal "
                f"schedule and {abs(cohort_delta):.1f} day(s) from the "
                f"cohort center. Both exceed the {MIN_GAP_ABS_DAYS:.0f}-day "
                "relevance floor."
            ),
            method="robust scheduled timeline cadence",
            extra={
                "timeline_variable": column,
                "timeline_pair": pair_label,
                "finding_kind": "cadence_outlier",
                "date_a": str(row["a"].date()),
                "date_b": str(row["b"].date()),
                "observed_gap_days": observed_gap,
                "expected_gap_days": expected_gap,
                "cohort_center_gap_days": float(center),
                "difference_from_expected_days": expected_deviation,
                "difference_from_cohort_days": cohort_delta,
                "gap_score_kind": score_kind,
            },
        ).to_row())

    return out


def _deduplicate_findings(rows: List[dict]) -> List[dict]:
    """Collapse pair fan-out to one issue per subject and date variable."""
    if not rows:
        return []

    groups: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("network", "")),
            str(row.get("subjectid", "")),
            str(row.get("timeline_variable", "")),
        )
        groups[key].append(row)

    collapsed: List[dict] = []
    for (_network, _subject, variable), group in groups.items():
        def severity(row: dict) -> float:
            try:
                return float(row.get("severity_score", 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        # A known-timepoint order reversal is deterministic and more
        # actionable than a statistical cadence deviation.  Always display an
        # order row when one exists, even if a cadence row has a numerically
        # higher severity; the latter remains in the corroborating metadata.
        order_rows = [
            row for row in group
            if str(row.get("finding_kind", "")) == "order_reversal"
        ]
        representative = dict(max(order_rows or group, key=severity))
        pairs = sorted({
            str(row.get("timeline_pair", "")) for row in group
            if str(row.get("timeline_pair", ""))
        })
        methods = sorted({
            str(row.get("method", "")) for row in group
            if str(row.get("method", ""))
        })
        kinds = sorted({
            str(row.get("finding_kind", "")) for row in group
            if str(row.get("finding_kind", ""))
        })
        representative["variable"] = variable
        representative["variables_involved"] = variable
        representative["corroborating_rows"] = len(group)
        representative["corroborating_pairs"] = len(pairs)
        representative["corroborating_methods"] = "; ".join(methods)
        representative["finding_kinds"] = "; ".join(kinds)
        representative["related_timeline_pairs"] = "; ".join(pairs[:12]) + (
            f"; (+{len(pairs) - 12} more)" if len(pairs) > 12 else ""
        )
        if len(group) > 1:
            representative["explanation"] = (
                str(representative.get("explanation", ""))
                + f" Collapsed {len(group)} corroborating pair findings "
                  f"across {len(pairs)} timeline pair(s)."
            )
        collapsed.append(representative)

    return sorted(
        collapsed,
        key=lambda row: (
            -float(row.get("severity_score", 0) or 0),
            str(row.get("network", "")),
            str(row.get("subjectid", "")),
            str(row.get("variable", "")),
        ),
    )
