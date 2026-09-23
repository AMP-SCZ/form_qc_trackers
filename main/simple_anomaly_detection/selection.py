"""Deterministic relevance and diversity selection for anomaly reports.

Detectors intentionally produce candidate evidence. This module turns those
candidates into reviewer-facing tabs without allowing one variable, pair leg,
subject, or exact issue to consume the row budget. Candidate rows omitted only
by presentation quotas can still be written to the audit-candidate CSVs.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


_SPLIT_RE = re.compile(r"\s+\|\s+|\s+&\s+|\s*,\s*")
_BLANK_IDS = {"", "nan", "none", "na", "n/a"}


def _normalise_leg(value: object) -> str:
    leg = str(value or "").strip()
    if not leg or leg.startswith("("):
        return ""
    # Date ids are ``timepoint::field``; longitudinal cluster axes are
    # ``field@timepoint``. Quotas apply to the underlying field across visits.
    if "::" in leg:
        leg = leg.split("::", 1)[1]
    if "@" in leg:
        leg = leg.split("@", 1)[0]
    # Binary-cell labels are ``field=value``. The value is evidence, but the
    # field is the resource whose repeated rows need bounding.
    if "=" in leg:
        leg = leg.split("=", 1)[0]
    leg = leg.strip().lower()
    return "" if leg in {"corr-drift", "(corr-drift)"} else leg


def variable_components(row: pd.Series, cap_pseudo: bool = False) -> Tuple[str, ...]:
    """Canonical underlying variables represented by one finding row."""
    anomaly_type = str(row.get("anomaly_type", ""))
    primary = str(row.get("primary_variable", "") or "").strip()
    if primary:
        source = primary
    elif anomaly_type == "date_anomaly_simple" and str(
            row.get("variables_involved", "") or "").strip():
        # Date fan-out summaries retain every implicated field here.
        source = str(row.get("variables_involved", ""))
    else:
        source = str(row.get("variable", "") or "")

    if source.strip().startswith("("):
        return (source.strip().lower(),) if cap_pseudo else tuple()
    parts = _SPLIT_RE.split(source)
    legs = sorted({_normalise_leg(p) for p in parts if _normalise_leg(p)})
    return tuple(legs)


def _valid_entity(row: pd.Series) -> bool:
    """A finding needs a variable and either a subject or site entity."""
    variable = str(row.get("variable", "") or "").strip().lower()
    if variable in _BLANK_IDS:
        return False
    subject = str(row.get("subjectid", "") or "").strip().lower()
    site = str(row.get("site_id", "") or "").strip().lower()
    return subject not in _BLANK_IDS or site not in _BLANK_IDS


def select_diverse_findings(
    frame: pd.DataFrame,
    row_cap: int,
    max_per_variable: int,
    max_per_subject: int,
    *,
    cap_pseudo_variables: bool = False,
) -> pd.DataFrame:
    """Return a stable, severity-prioritized, diversified finding frame.

    Exact duplicate evidence is removed first. Candidates are then ordered in
    fair rounds by rank within their exact issue, so every distinct issue gets a
    chance before a repeated issue consumes the global row cap. A greedy
    component quota bounds *each leg* of pairs/triples, not merely their exact
    label. The retained rows are finally sorted by severity for the established
    report contract; ``rank_within_issue`` makes the selection auditable.
    """
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()

    work = frame.copy()
    for col in ("anomaly_type", "network", "timepoint", "site_id", "subjectid",
                "variable", "observed_value", "method"):
        if col not in work.columns:
            work[col] = ""
    work = work[work.apply(_valid_entity, axis=1)].copy()
    if work.empty:
        return work.reset_index(drop=True)

    identity = ["anomaly_type", "network", "timepoint", "site_id", "subjectid",
                "variable", "observed_value", "method"]
    work = work.drop_duplicates(identity, keep="first").copy()
    work["_severity"] = pd.to_numeric(
        work.get("severity_score"), errors="coerce").replace(
            [np.inf, -np.inf], np.nan).fillna(float("-inf"))
    work["_raw"] = pd.to_numeric(work.get("raw_score"), errors="coerce").replace(
        [np.inf, -np.inf], np.nan).fillna(float("-inf"))
    work["_components"] = work.apply(
        lambda r: variable_components(r, cap_pseudo=cap_pseudo_variables), axis=1)
    work["issue_key"] = work.apply(
        lambda r: "|".join((str(r["network"]), str(r["anomaly_type"]),
                            ",".join(r["_components"]) or str(r["variable"]).lower())),
        axis=1,
    )
    work = work.sort_values(
        ["_severity", "_raw", "network", "issue_key", "subjectid", "timepoint"],
        ascending=[False, False, True, True, True, True], kind="stable")
    work["rank_within_issue"] = work.groupby("issue_key", sort=False).cumcount() + 1

    # Fair-round order controls which rows survive the global cap. Severity is
    # still the tie-breaker inside a round.
    candidates = work.sort_values(
        ["rank_within_issue", "_severity", "_raw", "network", "issue_key",
         "subjectid", "timepoint"],
        ascending=[True, False, False, True, True, True, True], kind="stable")

    variable_counts = {}
    subject_counts = {}
    keep: List[object] = []
    limit = int(row_cap) if row_cap and int(row_cap) > 0 else len(candidates)
    for idx, row in candidates.iterrows():
        components: Iterable[str] = row["_components"]
        net = str(row.get("network", ""))
        # The quota is per TAB, not per network: otherwise the same variable
        # can consume K rows in each network and still dominate the combined
        # workbook view. Fair issue rounds include network in issue_key, so both
        # networks get considered before repeated rows consume the quota.
        var_keys = [(leg,) for leg in components]
        if max_per_variable > 0 and any(
                variable_counts.get(k, 0) >= max_per_variable for k in var_keys):
            continue

        subject = str(row.get("subjectid", "") or "").strip()
        subject_key = None
        if subject and not subject.startswith("(") and subject.lower() not in _BLANK_IDS:
            subject_key = (net, subject)
            if (max_per_subject > 0
                    and subject_counts.get(subject_key, 0) >= max_per_subject):
                continue

        keep.append(idx)
        for key in var_keys:
            variable_counts[key] = variable_counts.get(key, 0) + 1
        if subject_key is not None:
            subject_counts[subject_key] = subject_counts.get(subject_key, 0) + 1
        if len(keep) >= limit:
            break

    selected = work.loc[keep].copy()
    selected = selected.sort_values(
        ["_severity", "_raw", "network", "issue_key", "subjectid", "timepoint"],
        ascending=[False, False, True, True, True, True], kind="stable")
    return selected.drop(columns=["_severity", "_raw", "_components"],
                         errors="ignore").reset_index(drop=True)
