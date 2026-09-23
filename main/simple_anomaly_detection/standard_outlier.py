"""Detector 1 - Standard outlier detection per (network, timepoint, variable).

Numeric variables only. Uses median + MAD-scaled sigma so a handful of
extreme values don't define their own neighborhood. Anything with at
least 30 non-missing observations is eligible; we flag |robust_z| >= 4.

Discovery peer: ``qc_types/discovery/point_spike_checks.py`` and
``local_outlier_checks.py``. This is the simpler standalone variant.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .common import (
    Finding, classify_columns, matches_excluded_substrings,
    robust_center_scale, severity_from_z, site_from_subjectid,
    to_numeric_clean, short,
)

ANOMALY_TYPE = "standard_outlier"
Z_THRESHOLD = 4.0
MIN_N = 30
# An individual-cell detector is only meaningful when the flagged tail is
# actually rare. On discrete, rounded, or mixture-distributed variables a tiny
# MAD can put a sizeable group above |z|=4; those rows describe a mismatched
# model/distribution, not dozens of independent entry errors. Allow at least
# three cells for small slices, otherwise no more than 5% of observed values.
MAX_FLAG_FRACTION = 0.05
MIN_FLAG_COUNT_ALLOWANCE = 3

# Per-detector exclusion list (case-insensitive substring match against
# variable name). Add strings here to suppress this detector's findings
# on those variables without touching the shared EXCLUDED_VARIABLE_SUBSTRINGS
# in common.py.
EXTRA_EXCLUDED_SUBSTRINGS: tuple = ()


def detect_per_slice(df: pd.DataFrame, network: str, timepoint: str,
                     cats: dict = None) -> List[dict]:
    """One pass per (network, timepoint) wide DataFrame. ``subjectid``
    column is assumed; rows are subjects. ``cats`` may carry a pre-built
    classify_columns result so the runner can compute it once per slice."""
    if df is None or df.empty or "subjectid" not in df.columns:
        return []

    if cats is None:
        cats = classify_columns(df)
    out: List[dict] = []

    for col in cats["numeric"]:
        if matches_excluded_substrings(col, EXTRA_EXCLUDED_SUBSTRINGS):
            continue
        num = to_numeric_clean(df[col])
        valid = num.dropna()
        if len(valid) < MIN_N:
            continue
        med, sigma = robust_center_scale(valid.to_numpy(dtype=float))
        if not np.isfinite(sigma):
            continue
        z = (num - med) / sigma
        bad_mask = z.abs() >= Z_THRESHOLD
        if not bad_mask.any():
            continue

        n_bad = int(bad_mask.sum())
        max_rare = max(MIN_FLAG_COUNT_ALLOWANCE,
                       int(np.ceil(MAX_FLAG_FRACTION * len(valid))))
        if n_bad > max_rare:
            # This tail is too common to call its members independent isolated
            # errors. Emit ONE distribution-level issue instead of either
            # flooding the sheet or silently dropping a possible batch error.
            idxs = np.where(bad_mask.values)[0]
            z_bad = z.iloc[idxs].abs().sort_values(ascending=False)
            top_idxs = list(z_bad.index[:5])
            top_bits = []
            for i in top_idxs:
                top_bits.append(
                    f"{df['subjectid'].loc[i]}={short(num.loc[i])} "
                    f"(z={abs(float(z.loc[i])):.2f})")
            out.append(Finding(
                anomaly_type=ANOMALY_TYPE,
                severity_score=severity_from_z(float(z_bad.iloc[0]),
                                                Z_THRESHOLD, 10.0),
                raw_score=float(z_bad.iloc[0]),
                network=network,
                timepoint=timepoint,
                site_id="(multiple)",
                subjectid=f"({n_bad} rows)",
                variable=col,
                observed_value="; ".join(top_bits),
                expected_value=f"~{med:.4g} (median); sigma={sigma:.4g}",
                explanation=(
                    f"{n_bad}/{len(valid)} observed rows ({n_bad/len(valid):.1%}) "
                    f"exceed |robust z| >= {Z_THRESHOLD}. This is too common "
                    f"for isolated-cell reporting (limit {max_rare}); review as "
                    f"a distribution/batch issue. Top examples are shown."
                ),
                method="median + MAD; common-tail distribution summary",
                extra={"n_flagged_rows": n_bad,
                       "flag_fraction": round(n_bad / len(valid), 4),
                       "detail_suppressed": n_bad},
            ).to_row())
            continue

        idxs = np.where(bad_mask.values)[0]
        for i in idxs:
            zi = float(z.iat[i])
            if not np.isfinite(zi):
                continue
            obs = float(num.iat[i])
            subj = str(df["subjectid"].iat[i])
            f = Finding(
                anomaly_type=ANOMALY_TYPE,
                severity_score=severity_from_z(zi, Z_THRESHOLD, 10.0),
                raw_score=abs(zi),
                network=network,
                timepoint=timepoint,
                site_id=site_from_subjectid(subj),
                subjectid=subj,
                variable=col,
                observed_value=short(obs),
                expected_value=f"~{med:.4g} (median); sigma={sigma:.4g}",
                explanation=(
                    f"Robust z = {zi:+.2f} vs cohort median {med:.4g} "
                    f"(n={len(valid)}). Threshold |z| >= {Z_THRESHOLD}."
                ),
                method="median + MAD-scaled sigma",
            )
            out.append(f.to_row())
    return out


def detect_all(per_slice: Dict, classify: Dict = None, **_) -> List[dict]:
    classify = classify or {}
    rows: List[dict] = []
    for (network, tp), df in per_slice.items():
        try:
            rows.extend(detect_per_slice(df, network, tp,
                                         cats=classify.get((network, tp))))
        except Exception as e:
            print(f"  [standard_outlier] WARN slice ({network},{tp}): {e}")
    return rows
