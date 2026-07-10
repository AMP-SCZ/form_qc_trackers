"""Detector 5 - Format / string deviations.

For each text-like column we learn the dominant value-shape pattern,
then flag rows whose pattern is rare.

Two signals:
    length     - flag values whose character length is in a rare bucket
                 (frequency <= 1%, also off the median by >= 4 MADs).
    signature  - char-class signature using char_class_signature(). Letters
                 collapse to 'a'/'A', digits to 'D', punctuation is kept
                 literal. Flag signatures occurring in fewer than 1% of the
                 column.

Free-text columns (comment/note/description) are excluded by the shared
EXCLUDED_VARIABLE_SUBSTRINGS list -- the format detector is *not*
useful on comments and would flood the report if allowed.

No direct discovery peer: format/string pattern auditing is not
covered by ``qc_types/discovery/*``.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List  # Dict re-used for the per-row aggregator

import numpy as np
import pandas as pd

from .common import (
    Finding, calibrate, char_class_signature, classify_columns,
    matches_excluded_substrings, site_from_subjectid, short,
    clean_missing_string,
)

ANOMALY_TYPE = "format_deviation"

# Needs >=100 values: the rare gates below are frequency <= 1%, and for
# N in 60..99 a single occurrence is 1/N > 1%, so the gate could never fire
# -- a silent dead-zone. 100 is the smallest N at which one rare value
# (1/100 = 1%) can trip it.
MIN_VALUES = 100
MAX_SIG_FREQ_RARE = 0.01
MAX_LEN_FREQ_RARE = 0.01
MIN_DOMINANT_FREQ = 0.5   # only score columns where some pattern is at least 50% dominant

# Per-detector exclusion list (case-insensitive substring match against
# variable name). Add strings here to suppress this detector's findings
# on those variables without touching the shared EXCLUDED_VARIABLE_SUBSTRINGS
# in common.py.
EXTRA_EXCLUDED_SUBSTRINGS: tuple = ()


def detect_all(per_slice: Dict, classify: Dict = None, **_) -> List[dict]:
    classify = classify or {}
    rows: List[dict] = []
    for (network, tp), df in per_slice.items():
        if df is None or df.empty or "subjectid" not in df.columns:
            continue
        try:
            rows.extend(_detect_slice(df, network, tp,
                                      cats=classify.get((network, tp))))
        except Exception as e:
            print(f"  [format_deviation] WARN slice ({network},{tp}): {e}")
    return rows


def _detect_slice(df: pd.DataFrame, network: str, tp: str,
                  cats: dict = None) -> List[dict]:
    if cats is None:
        cats = classify_columns(df)
    out: List[dict] = []
    for col in cats["text"]:
        if matches_excluded_substrings(col, EXTRA_EXCLUDED_SUBSTRINGS):
            continue
        ser = clean_missing_string(df[col].astype(object))
        non_missing = ser.dropna().astype(str)
        if len(non_missing) < MIN_VALUES:
            continue

        # Build a per-row finding dict so the same row gets ONE record
        # carrying length AND signature reasons -- not two flags at the
        # same severity for the same cell.
        findings: Dict[int, dict] = {}

        lengths = non_missing.str.len()
        len_counts = lengths.value_counts(normalize=True)
        if len_counts.iloc[0] >= MIN_DOMINANT_FREQ:
            lvals = lengths.to_numpy(dtype=float)
            med = float(np.median(lvals))
            mad = float(np.median(np.abs(lvals - med)))
            sigma = mad * 1.4826 if mad > 0 else float("nan")
            # Map each row's length to its frequency in C.
            freq_per_row = lengths.map(len_counts).astype(float)
            rare_len = freq_per_row <= MAX_LEN_FREQ_RARE
            if rare_len.any():
                # MAD-based z; if sigma is NaN (modal dominance too strong),
                # use the rarity gate alone.
                if np.isfinite(sigma) and sigma > 0:
                    z_len = (lengths - med).abs() / sigma
                else:
                    z_len = pd.Series(np.full(len(lengths), np.inf), index=lengths.index)
                gate = rare_len & ((z_len >= 3.0) | (freq_per_row <= MAX_LEN_FREQ_RARE / 2))
                for ridx in lengths.index[gate]:
                    L = int(lengths.loc[ridx])
                    fr = float(freq_per_row.loc[ridx])
                    raw = float(-np.log10(max(fr, 1e-6)))
                    findings[ridx] = {
                        "raw_score": raw,
                        "severity": calibrate(raw, 2.0, 4.0),
                        "reasons": [
                            f"length {L} occurs in only {fr:.2%} of values "
                            f"(typical length {int(med)})"
                        ],
                        "expected_parts": [f"length ~{int(med)}"],
                    }

        sigs = non_missing.map(char_class_signature)
        sig_counts = sigs.value_counts(normalize=True)
        if not sig_counts.empty and sig_counts.iloc[0] >= MIN_DOMINANT_FREQ:
            rare_sigs = sig_counts[sig_counts <= MAX_SIG_FREQ_RARE]
            if not rare_sigs.empty:
                freq_per_row = sigs.map(rare_sigs).astype(float)
                for ridx in sigs.index[freq_per_row.notna()]:
                    fr = float(freq_per_row.loc[ridx])
                    raw = float(-np.log10(max(fr, 1e-6)))
                    sev = calibrate(raw, 2.0, 4.0)
                    rec = findings.get(ridx)
                    sig_reason = (
                        f"signature '{sigs.loc[ridx]}' occurs in only "
                        f"{fr:.2%} of values"
                    )
                    sig_expected = (
                        f"most common signature '{sig_counts.idxmax()}' "
                        f"({sig_counts.iloc[0]:.0%})"
                    )
                    if rec is None:
                        findings[ridx] = {
                            "raw_score": raw,
                            "severity": sev,
                            "reasons": [sig_reason],
                            "expected_parts": [sig_expected],
                        }
                    else:
                        # Combine into a single record; take the max
                        # severity and merge reasons.
                        rec["raw_score"] = max(rec["raw_score"], raw)
                        rec["severity"] = max(rec["severity"], sev)
                        rec["reasons"].append(sig_reason)
                        rec["expected_parts"].append(sig_expected)

        for ridx, rec in findings.items():
            val = non_missing.loc[ridx]
            # ridx is an index LABEL (from non_missing's index, which is a
            # gappy subset of df's index once missing rows are dropped). Use
            # label-based .loc, NOT positional .iat -- .iat[label] returns the
            # wrong subject (or IndexError) on any non-default/ gappy index.
            subj = str(df["subjectid"].loc[ridx])
            out.append(Finding(
                anomaly_type=ANOMALY_TYPE,
                severity_score=rec["severity"],
                raw_score=rec["raw_score"],
                network=network, timepoint=tp,
                site_id=site_from_subjectid(subj),
                subjectid=subj,
                variable=col,
                observed_value=short(val),
                expected_value="; ".join(rec["expected_parts"]),
                explanation=(
                    f"Format deviation in {col}: " + "; ".join(rec["reasons"]) + "."
                ),
                method="value-length and/or char-class signature frequency",
            ).to_row())
    return out
