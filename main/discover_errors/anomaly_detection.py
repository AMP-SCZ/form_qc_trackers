"""
Discover NEW QC anomalies using ML and statistical methods (no duplicate
of existing rule-based QC). Designed for large longitudinal multisite studies.

This script is intentionally high-sensitivity and explainable: every flag includes
variables involved, values, what's wrong, suggested action, and a severity ranking.

What it includes:
- Row missingness anomalies
- Robust univariate numeric outliers (median/MAD robust z)
- Longitudinal abrupt change (within-subject delta robust z)
- Subject "too-constant" (copy-forward suspicion)
- Rare categorical values
- Rare categorical pairs
- Rare categorical triples (low-cardinality)
- Rare overall categorical patterns (pattern surprisal)
- Exact duplicates (row fingerprint)
- Near-duplicates (nearest-neighbor similarity relative to typical)
- Isolation Forest anomalies (numeric multivariate)
- LOF anomalies (numeric density)
- PCA reconstruction anomalies
- KMeans distance-to-cluster anomalies
- Mahalanobis distance anomalies (LedoitWolf covariance)
- Site categorical distribution shift (chi-square if available)
- Site numeric shift (robust median shift + low variance)
- Digit preference/heaping anomalies (overall + per-site)
- Site correlation-profile anomalies (site vs global correlation structure)
- Site drift over time (if date/timestamp column exists)
- Candidate rule generation (candidate_rules.csv)
- Robust file finder (templates + glob) to avoid missing-file skips
- Variable exclusion by name pattern
- NaN-safe statistics to reduce runtime warnings

Run from repo root:
  python main/discover_errors/discover_new_errors.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from main.utils.utils import Utils

# Optional ML backends (required for most methods)
try:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.covariance import LedoitWolf

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# Optional stats backend
try:
    from scipy.stats import chi2_contingency

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


MISSING_CODES = [
    "-3", "-9", -3, -9, -3.0, -9.0, "-3.0", "-9.0",
    "1909-09-09", "1903-03-03", "1901-01-01",
    "-99", -99, -99.0, "-99.0",
    999, 999.0, "999", "999.0",
    "", "nan", "NaN", "NA", "n/a", "N/A",
]

# If a variable name contains any of these substrings (case-insensitive), it is excluded
EXCLUDED_VARIABLE_NAME_PATTERNS = [
    # Add any substrings here; if a column name contains one, it will be excluded.
    # Keep this fairly specific so you don't accidentally exclude too much.
    "comment",
    "note",
    "description",
    "chrdig",
    "chrax",
    "_missing",
    "_complete",
    "chrblood_wb3pos",
    "chrblood_se1pos",
    "chrblood_se2pos",
    "chrblood_se3pos",
    "chrblood_pl1pos",
    "chrblood_pl2pos",
    "chrblood_pl3pos",
    "chrblood_pl4pos",
    "chrblood_pl5pos",
    "chrblood_pl6pos",
    "chrblood_bc1pos",
    "chrblood_wb1id",
    "chrblood_wb2id",
    "chrblood_wb3id",
    "chrblood_se1id",
    "chrblood_se2id",
    "chrblood_se3id",
    "chrblood_pl1id",
    "chrblood_pl1id_2",
    "chrblood_pl2id",
    "chrblood_pl3id",
    "chrblood_pl4id",
    "chrblood_pl5id",
    "chrblood_pl6id",
    "chrblood_rack_barcode",
    "chrblood_bc1id",
    "chrblood_bc1box"
]
data_dict = pd.read_csv(
'/home/ob001/refactored_qc/dependencies/data_dictionary/current_data_dictionary.csv',
keep_default_na = False)
filtered_data_dict = data_dict[data_dict['Field Type'].isin(['notes','descriptive'])]

additional_excl_vars = filtered_data_dict['Variable / Field Name'].tolist()
print(additional_excl_vars)
EXCLUDED_VARIABLE_NAME_PATTERNS.extend(additional_excl_vars)




# ---------------------------
# Helpers
# ---------------------------

def _is_excluded_variable(col_name: str) -> bool:
    c = str(col_name).lower()
    return any(p.lower() in c for p in EXCLUDED_VARIABLE_NAME_PATTERNS)

def _get_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Safely return a single Series for a column name.
    If duplicate column names exist, use the first occurrence.
    """
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


def _clean_categorical_series(ser: pd.Series) -> pd.Series:
    return (
        ser.astype(str)
        .str.strip()
        .replace("", np.nan)
        .replace([str(x) for x in MISSING_CODES], np.nan)
    )

def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    s = str(val).strip()
    return s == "" or s in set(str(x) for x in MISSING_CODES)


def _safe_str(val: Any, max_len: int = 160) -> str:
    if val is None:
        return ""
    s = str(val).replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _to_numeric_series(ser: pd.Series) -> pd.Series:
    return pd.to_numeric(ser.replace("", np.nan), errors="coerce")


def _robust_zscore(x: pd.Series) -> pd.Series:
    """Robust z-score using median and MAD (scaled). Safe for all-NaN input."""
    v = pd.to_numeric(x, errors="coerce")
    non_missing = v.dropna()
    if non_missing.empty:
        return pd.Series(np.nan, index=x.index)
    med = np.nanmedian(non_missing.values)
    mad = np.nanmedian(np.abs(non_missing.values - med))
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.nan, index=x.index)
    return 0.6745 * (v - med) / mad


def _select_numeric_columns(df: pd.DataFrame, max_cols: int = 80) -> list[str]:
    out: List[str] = []
    seen = set()
    for col in df.columns:
        if col in seen:
            continue
        seen.add(col)

        if col in ("subjectid",) or "__" in col:
            continue
        if "date" in str(col).lower() or "time" in str(col).lower():
            continue
        if _is_excluded_variable(col):
            continue

        ser = _get_series(df, col).replace("", np.nan)
        num = pd.to_numeric(ser, errors="coerce")
        valid = num.dropna()
        if len(valid) < 40 or valid.nunique() < 3:
            continue
        if valid.notna().sum() / max(len(df), 1) < 0.25:
            continue
        out.append(col)
        if len(out) >= max_cols:
            break
    return out


def _select_categorical_columns(df: pd.DataFrame, max_cols: int = 120) -> list[str]:
    out: List[str] = []
    seen = set()
    for col in df.columns:
        if col in seen:
            continue
        seen.add(col)

        if col in ("subjectid",) or "__" in col:
            continue
        if "date" in str(col).lower() or "time" in str(col).lower():
            continue
        if _is_excluded_variable(col):
            continue

        ser = _clean_categorical_series(_get_series(df, col))
        n_filled = ser.notna().sum()
        if n_filled < 60:
            continue
        uniq = ser.dropna().nunique()
        if uniq < 2 or uniq > 250:
            continue

        num = pd.to_numeric(_get_series(df, col), errors="coerce")
        if num.notna().sum() / max(len(df), 1) > 0.75:
            continue
        out.append(col)
        if len(out) >= max_cols:
            break
    return out


def _infer_site_col(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if "siteid" in df.columns:
        return df, "siteid"
    if "site" in df.columns:
        return df, "site"
    df = df.copy()
    df["_site"] = df["subjectid"].astype(str).str[:2] if "subjectid" in df.columns else ""
    return df, "_site"


def _fingerprint_row(row: pd.Series, cols: List[str]) -> str:
    parts = []
    for c in cols:
        v = row.get(c, "")
        parts.append(f"{c}={'<MISSING>' if _is_missing(v) else _safe_str(v, 80)}")
    blob = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.md5(blob).hexdigest()


def _digit_preference_chi2(x: pd.Series) -> float:
    """
    Simple heaping / digit preference score:
    - convert numeric values to ints via floor(abs(x))
    - check last digit distribution 0-9
    - return chi-square-like deviation from uniform
    """
    v = pd.to_numeric(x, errors="coerce").dropna()
    if v.shape[0] < 80:
        return 0.0
    ints = np.floor(np.abs(v.values)).astype(int)
    last = ints % 10
    counts = np.bincount(last, minlength=10).astype(float)
    expected = counts.sum() / 10.0
    if expected <= 0:
        return 0.0
    chi = float(np.sum((counts - expected) ** 2 / expected))
    return chi


def _detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Best-effort: find a likely date/timestamp column.
    Prefers columns whose name includes date/time and parses successfully.
    """
    candidates = []
    for c in df.columns:
        cl = c.lower()
        if "date" in cl or "timestamp" in cl or "time" in cl:
            if _is_excluded_variable(c):
                continue
            candidates.append(c)

    # Try a few common names first
    priority = []
    for p in ["visit_date", "assessment_date", "entry_date", "timestamp", "created", "updated", "date"]:
        for c in candidates:
            if p in c.lower():
                priority.append(c)
    ordered = priority + [c for c in candidates if c not in priority]

    for c in ordered[:12]:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            if parsed.notna().sum() >= max(60, int(0.2 * len(df))):
                return c
        except Exception:
            continue
    return None


@dataclass
class Finding:
    network: str
    timepoint: str
    subjectid: str
    error_type: str
    method: str
    severity_score: float
    severity_label: str
    message: str
    what_is_wrong: str
    suggested_action: str
    form: str = ""
    variables_involved: str = ""
    variable_values: str = ""
    score_or_pct: str = ""
    row_index: Any = ""
    site_id: str = ""


def _severity_label(score: float) -> str:
    if score >= 90:
        return "CRITICAL"
    if score >= 75:
        return "HIGH"
    if score >= 55:
        return "MEDIUM"
    return "LOW"


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _record(f: Finding) -> dict:
    return {
        "network": f.network,
        "timepoint": f.timepoint,
        "subjectid": f.subjectid,
        "site_id": f.site_id,
        "error_type": f.error_type,
        "method": f.method,
        "severity_score": f"{f.severity_score:.2f}",
        "severity_label": f.severity_label,
        "form": f.form or "",
        "variables_involved": f.variables_involved or "",
        "variable_values": f.variable_values or "",
        "message": f.message,
        "what_is_wrong": f.what_is_wrong,
        "suggested_action": f.suggested_action,
        "score_or_pct": f.score_or_pct or "",
        "row_index": f.row_index if f.row_index is not None else "",
    }


# ---------------------------
# Main class
# ---------------------------

class DiscoverNewErrors:
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path

        with open(os.path.join(self.absolute_path, "config.json"), "r", encoding="utf-8") as f:
            self.config_info = json.load(f)

        self.comb_csv_path = self.config_info["paths"]["combined_csv_path"]
        self.output_path = self.config_info["paths"]["output_path"]
        self.results: List[dict] = []

        self.tp_list = self.utils.create_timepoint_list()
        self.tp_list.extend(["floating", "conversion"])

        # Core sensitivity knobs
        self.contamination = 0.03
        self.rarity_pct_threshold = 0.01
        self.rarity_pair_pct_threshold = 0.005
        self.rarity_triple_pct_threshold = 0.003
        self.reconstruction_quantile = 0.985
        self.numeric_rz_threshold = 4.5
        self.longitudinal_rz_threshold = 4.0

        # Site thresholds
        self.site_chi2_p_threshold = 1e-4
        self.site_shift_rz_threshold = 3.2
        self.site_corr_profile_threshold = 0.35  # L1 norm threshold on correlation-profile difference

        # Heaping thresholds
        self.heaping_chi2_threshold_overall = 35.0
        self.heaping_chi2_threshold_site = 45.0

        # Copy-forward suspicion
        self.constant_min_repeats = 4  # subject must have >=4 rows in file to assess constancy
        self.constant_unique_max = 1   # unique numeric values <=1 across repeats triggers

        # Drift over time settings
        self.min_period_rows = 60
        self.drift_step_rz_threshold = 3.8

        # Caps
        self.max_findings_per_tp = 1800
        self.max_ml_findings_per_tp = 450

        # Feature selection caps
        self.max_numeric_cols = 80
        self.max_cat_cols = 110

        # Duplicate settings
        self.dup_cols_cap = 35

        self.random_state = 42

    # ---------------------------
    # File finding
    # ---------------------------

    def _csv_path_candidates(self, timepoint: str, network: str) -> List[str]:
        tp_file = timepoint.replace("month", "month_").replace("floating", "floating_forms")

        templates = [
            os.path.join(self.comb_csv_path, f"AMPSCZ-combined-redcap_{tp_file}_{network}-day1to1.csv"),
            os.path.join(self.comb_csv_path, f"AMPSCZ-combined-redcap_{tp_file}_{network}.csv"),
            os.path.join(self.comb_csv_path, f"AMPSCZ-combined-redcap_{tp_file}_{network}-day1to1_combined.csv"),
        ]

        net_variants = {
            network, network.lower(), network.upper(), network.title(),
            network.replace("_", "-"), network.replace("-", "_"),
        }
        tp_variants = {
            tp_file, tp_file.lower(), tp_file.upper(), tp_file.title(),
            tp_file.replace("_", "-"), tp_file.replace("-", "_"),
        }

        glob_hits: List[str] = []
        for nv in net_variants:
            for tv in tp_variants:
                glob_hits.extend(glob.glob(os.path.join(self.comb_csv_path, f"*{tv}*{nv}*.csv")))
                glob_hits.extend(glob.glob(os.path.join(self.comb_csv_path, f"*{nv}*{tv}*.csv")))

        seen = set()
        ordered = []
        for p in templates + glob_hits:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        return ordered

    # ---------------------------
    # Pipeline
    # ---------------------------

    def run_script(self) -> None:
        if not _HAS_SKLEARN:
            print("scikit-learn is required. Install with: pip install scikit-learn")
            return

        for network in ["ProNET", "PRESCIENT"]:
            for tp in self.tp_list:
                path = None
                cands = self._csv_path_candidates(tp, network)
                for cand in cands:
                    if os.path.exists(cand):
                        path = cand
                        break

                if not path:
                    print(f"Skip (missing): {network} / {tp}")
                    print("  Tried:")
                    for cand in cands[:8]:
                        print("   -", cand)
                    if len(cands) > 8:
                        print(f"   ... ({len(cands)} total candidates)")
                    continue

                print(f"Processing: {network} / {tp} -> {os.path.basename(path)}")

                try:
                    df = pd.read_csv(path, keep_default_na=False, on_bad_lines="skip")
                except Exception as e:
                    print(f"  Error reading CSV: {e}")
                    continue

                if len(df) < 60:
                    print("  Skip: too few rows")
                    continue

                if "subjectid" not in df.columns:
                    print("  Skip: missing subjectid column")
                    continue

                excluded_cols = [c for c in df.columns if _is_excluded_variable(c)]
                if excluded_cols:
                    print(f"  Excluding {len(excluded_cols)} columns by name pattern (examples: {excluded_cols[:8]})")

                df, site_col = _infer_site_col(df)
                dt_col = _detect_datetime_column(df)
                if dt_col:
                    print(f"  Detected datetime column for drift checks: {dt_col}")

                self._run_discovery(df, network, tp, site_col, dt_col)

        self._write_output()

    def _run_discovery(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str, dt_col: Optional[str]) -> None:
        before = len(self.results)

        # High interpretability first
        self._discover_row_missingness(df, network, timepoint, site_col)
        self._discover_univariate_numeric_outliers(df, network, timepoint, site_col)
        self._discover_longitudinal_abrupt_change(df, network, timepoint, site_col)
        self._discover_subject_too_constant(df, network, timepoint, site_col)
        self._discover_digit_preference_heaping(df, network, timepoint, site_col)

        self._discover_categorical_rarity(df, network, timepoint, site_col)
        self._discover_categorical_pair_rarity(df, network, timepoint, site_col)
        self._discover_categorical_triple_rarity(df, network, timepoint, site_col)
        self._discover_categorical_pattern_surprisal(df, network, timepoint, site_col)

        self._discover_duplicates(df, network, timepoint, site_col)

        # Multivariate ML
        self._discover_isolation_forest(df, network, timepoint, site_col)
        self._discover_lof(df, network, timepoint, site_col)
        self._discover_pca_reconstruction(df, network, timepoint, site_col)
        self._discover_kmeans_distance(df, network, timepoint, site_col)
        self._discover_mahalanobis(df, network, timepoint, site_col)

        # Site monitoring
        self._discover_site_distribution_shift(df, network, timepoint, site_col)
        self._discover_site_numeric_shift(df, network, timepoint, site_col)
        self._discover_site_corr_profile(df, network, timepoint, site_col)

        # Drift over time if possible
        if dt_col:
            self._discover_site_drift_over_time(df, network, timepoint, site_col, dt_col)

        # Cap per file
        added = len(self.results) - before
        if added > self.max_findings_per_tp:
            batch = pd.DataFrame(self.results[before:])
            old = self.results[:before]
            if "severity_score" in batch.columns:
                batch["sev"] = pd.to_numeric(batch["severity_score"], errors="coerce").fillna(0.0)
                batch = batch.sort_values("sev", ascending=False).head(self.max_findings_per_tp).drop(columns=["sev"])
                self.results = old + batch.to_dict(orient="records")
            else:
                self.results = old + self.results[before: before + self.max_findings_per_tp]

    # ---------------------------
    # Numeric / longitudinal / missingness
    # ---------------------------

    def _discover_row_missingness(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        cols = [c for c in df.columns if c not in ("subjectid", site_col) and not _is_excluded_variable(c)]
        if len(cols) < 10:
            return

        work = df[cols].copy()
        for c in cols:
            work[c] = work[c].apply(lambda v: np.nan if _is_missing(v) else v)
        miss_frac = work.isna().mean(axis=1)

        rz = _robust_zscore(miss_frac)
        idxs = rz[rz >= 3.5].sort_values(ascending=False).index.tolist()[:250]
        for idx in idxs:
            subj = _safe_str(df.at[idx, "subjectid"])
            site = _safe_str(df.at[idx, site_col])
            frac = float(miss_frac.loc[idx])
            sev = _clamp(40 + 60 * min(frac, 1.0), 0, 100)
            self.results.append(_record(Finding(
                network=network, timepoint=timepoint, subjectid=subj, site_id=site,
                error_type="row_missingness_anomaly", method="missingness",
                severity_score=sev, severity_label=_severity_label(sev),
                message="Row has unusually high missingness compared to other rows.",
                what_is_wrong=f"Missingness fraction is {frac:.2f}, unusually high for this file/timepoint.",
                suggested_action="Check whether this record is partial, has skip-logic failure, or ingest/sync issue; confirm expected fields were captured.",
                variables_involved="(many fields)",
                variable_values=f"missing_frac={frac:.2f}",
                score_or_pct=f"robust_z={float(rz.loc[idx]):.2f}",
                row_index=int(idx),
            )))

    def _discover_univariate_numeric_outliers(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        cols = _select_numeric_columns(df, max_cols=self.max_numeric_cols)
        if not cols:
            return

        cap = 500
        added = 0
        for col in cols:
            if added >= cap:
                break
            x = _to_numeric_series(df[col].replace([str(x) for x in MISSING_CODES], np.nan))
            if x.notna().sum() < 60:
                continue

            rz = _robust_zscore(x)
            extreme = rz.abs().sort_values(ascending=False)
            idxs = extreme[extreme >= self.numeric_rz_threshold].index.tolist()[:60]
            if not idxs:
                continue

            valid = x.dropna()
            if valid.empty:
                continue
            q01, q99 = np.nanquantile(valid.values, 0.01), np.nanquantile(valid.values, 0.99)

            for idx in idxs:
                if added >= cap:
                    break
                subj = _safe_str(df.at[idx, "subjectid"])
                site = _safe_str(df.at[idx, site_col])
                val = x.loc[idx]
                rzv = float(rz.loc[idx])
                sev = _clamp(50 + 10 * abs(rzv), 0, 100)
                direction = "high" if rzv > 0 else "low"

                self.results.append(_record(Finding(
                    network=network, timepoint=timepoint, subjectid=subj, site_id=site,
                    error_type="numeric_outlier", method="robust_z",
                    severity_score=sev, severity_label=_severity_label(sev),
                    message=f"Extreme numeric outlier detected for '{col}'.",
                    what_is_wrong=(
                        f"Value is unusually {direction}: {val} with robust_z={rzv:.2f}. "
                        f"Typical range (1st-99th pct) is approx [{q01:.4g}, {q99:.4g}]."
                    ),
                    suggested_action="Verify source entry, units, decimal placement, and whether the value belongs to another field/visit; confirm missing codes weren’t mis-mapped.",
                    variables_involved=col,
                    variable_values=f"{col}={_safe_str(val)}",
                    score_or_pct=f"robust_z={rzv:.2f}",
                    row_index=int(idx),
                )))
                added += 1

    def _discover_longitudinal_abrupt_change(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        cols = _select_numeric_columns(df, max_cols=min(40, self.max_numeric_cols))
        if not cols:
            return

        work = df.copy()
        work["_row_i"] = np.arange(len(work))
        work = work.sort_values(["subjectid", "_row_i"])

        cap = 400
        added = 0
        for col in cols:
            if added >= cap:
                break
            x = _to_numeric_series(work[col].replace([str(x) for x in MISSING_CODES], np.nan))
            if x.notna().sum() < 80:
                continue

            prev = x.groupby(work["subjectid"]).shift(1)
            delta = x - prev
            drz = _robust_zscore(delta)

            extreme = drz.abs().sort_values(ascending=False)
            idxs = extreme[extreme >= self.longitudinal_rz_threshold].index.tolist()[:60]
            if not idxs:
                continue

            for idx in idxs:
                if added >= cap:
                    break
                subj = _safe_str(work.at[idx, "subjectid"])
                site = _safe_str(work.at[idx, site_col])
                cur = x.loc[idx]
                pv = prev.loc[idx]
                dv = delta.loc[idx]
                drzv = float(drz.loc[idx])
                sev = _clamp(55 + 9 * abs(drzv), 0, 100)

                self.results.append(_record(Finding(
                    network=network, timepoint=timepoint, subjectid=subj, site_id=site,
                    error_type="longitudinal_abrupt_change", method="delta_robust_z",
                    severity_score=sev, severity_label=_severity_label(sev),
                    message=f"Within-subject abrupt change in '{col}' vs prior record for this subject.",
                    what_is_wrong=(
                        f"Previous={_safe_str(pv)}, Current={_safe_str(cur)}, Delta={_safe_str(dv)}. "
                        f"Delta is extreme relative to cohort deltas (robust_z={drzv:.2f})."
                    ),
                    suggested_action="Confirm visit mapping/order; verify whether unit/scale changed, value was mis-entered, or visit was mis-assigned.",
                    variables_involved=col,
                    variable_values=f"prev_{col}={_safe_str(pv)}; {col}={_safe_str(cur)}; delta={_safe_str(dv)}",
                    score_or_pct=f"delta_robust_z={drzv:.2f}",
                    row_index=int(work.at[idx, "_row_i"]),
                )))
                added += 1

    def _discover_subject_too_constant(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        """
        Flag subjects whose numeric values are suspiciously constant across multiple rows.
        This often indicates copy-forward, duplicated visit rows, or un-updated repeated instruments.
        """
        cols = _select_numeric_columns(df, max_cols=min(35, self.max_numeric_cols))
        if not cols:
            return

        cap = 300
        added = 0

        # Pre-coerce numeric once
        num_df = pd.DataFrame(index=df.index)
        for c in cols:
            num_df[c] = _to_numeric_series(df[c].replace([str(x) for x in MISSING_CODES], np.nan))

        # For each subject, check constancy
        for subj, idxs in df.groupby(df["subjectid"].astype(str), dropna=False).groups.items():
            idxs = list(idxs)
            if len(idxs) < self.constant_min_repeats:
                continue
            if added >= cap:
                break

            g = num_df.loc[idxs, cols]
            # Compute per-column unique non-missing counts
            uniq_counts = g.nunique(dropna=True)
            constant_cols = uniq_counts[uniq_counts <= self.constant_unique_max].index.tolist()

            # Only flag if enough columns are constant (avoid false positives)
            if len(constant_cols) < max(3, int(0.35 * len(cols))):
                continue

            # Show up to 8 columns and their constant value
            show_cols = constant_cols[:8]
            vals = []
            for c in show_cols:
                v = g[c].dropna()
                vals.append(f"{c}={_safe_str(v.iloc[0]) if not v.empty else '<ALL_MISSING>'}")
            vals_str = "; ".join(vals)

            # Severity grows with fraction constant
            frac_const = len(constant_cols) / max(len(cols), 1)
            sev = _clamp(55 + 45 * frac_const, 0, 98)

            site = _safe_str(df.loc[idxs[0], site_col])

            self.results.append(_record(Finding(
                network=network, timepoint=timepoint, subjectid=_safe_str(subj), site_id=site,
                error_type="subject_too_constant", method="within_subject_constancy",
                severity_score=sev, severity_label=_severity_label(sev),
                message="Subject shows suspiciously constant numeric values across repeated rows (possible copy-forward).",
                what_is_wrong=(
                    f"Subject has {len(idxs)} rows; {len(constant_cols)}/{len(cols)} numeric fields "
                    f"do not change (or have <= {self.constant_unique_max} unique non-missing values)."
                ),
                suggested_action="Check whether repeated records are duplicates/copy-forward; verify whether later visits should differ or whether values were not updated.",
                variables_involved=";".join(show_cols) + (";..." if len(constant_cols) > len(show_cols) else ""),
                variable_values=vals_str,
                score_or_pct=f"constant_frac={frac_const:.2f}",
                row_index=str(idxs[:6]) + ("..." if len(idxs) > 6 else ""),
            )))
            added += 1

    def _discover_digit_preference_heaping(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        """
        Detect digit preference / heaping (rounding) overall and per site.
        """
        cols = _select_numeric_columns(df, max_cols=min(45, self.max_numeric_cols))
        if not cols:
            return

        # Overall heaping
        for c in cols[:30]:
            x = _to_numeric_series(df[c].replace([str(x) for x in MISSING_CODES], np.nan))
            chi = _digit_preference_chi2(x)
            if chi >= self.heaping_chi2_threshold_overall:
                sev = _clamp(55 + 1.2 * chi, 0, 95)
                self.results.append(_record(Finding(
                    network=network, timepoint=timepoint, subjectid="", site_id="",
                    error_type="digit_preference_heaping_overall", method="last_digit_chi2",
                    severity_score=sev, severity_label=_severity_label(sev),
                    message=f"Digit preference/heaping detected overall for '{c}'.",
                    what_is_wrong=f"Last-digit distribution deviates from uniform (chi2-like={chi:.2f}). This suggests rounding/heaping or default values.",
                    suggested_action="Inspect whether values are being rounded (e.g., always ending in 0/5), device calibration, default entry behavior, or data transformations.",
                    variables_involved=c,
                    variable_values="",
                    score_or_pct=f"chi2={chi:.2f}",
                    row_index="",
                )))

        # Per-site heaping
        # Only evaluate sites with enough rows
        for site, g in df.groupby(site_col, dropna=False):
            if len(g) < 120:
                continue
            for c in cols[:25]:
                x = _to_numeric_series(g[c].replace([str(x) for x in MISSING_CODES], np.nan))
                chi = _digit_preference_chi2(x)
                if chi >= self.heaping_chi2_threshold_site:
                    sev = _clamp(60 + 1.2 * chi, 0, 98)
                    self.results.append(_record(Finding(
                        network=network, timepoint=timepoint, subjectid="", site_id=_safe_str(site),
                        error_type="digit_preference_heaping_site", method="last_digit_chi2_site",
                        severity_score=sev, severity_label=_severity_label(sev),
                        message=f"Digit preference/heaping detected at site '{site}' for '{c}'.",
                        what_is_wrong=f"Site last-digit distribution deviates strongly from uniform (chi2-like={chi:.2f}).",
                        suggested_action="Check for site-specific rounding defaults, device settings, or training issues; compare to other sites.",
                        variables_involved=c,
                        variable_values=f"site={_safe_str(site)}",
                        score_or_pct=f"chi2={chi:.2f}",
                        row_index="",
                    )))

    # ---------------------------
    # Categorical anomalies
    # ---------------------------

    def _discover_categorical_rarity(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        cols = _select_categorical_columns(df, max_cols=self.max_cat_cols)
        if not cols:
            return

        cap = 500
        added = 0

        for col in cols:
            if added >= cap:
                break
            ser = df[col].astype(str).str.strip()
            ser = ser.replace("", np.nan).replace([str(x) for x in MISSING_CODES], np.nan)
            filled = ser.dropna()
            if len(filled) < 80:
                continue
            counts = filled.value_counts()
            total = int(len(filled))
            rare_threshold = max(2, int(total * self.rarity_pct_threshold))
            rare_vals = counts[counts <= rare_threshold].index.tolist()
            if not rare_vals:
                continue

            mask = ser.isin(set(rare_vals))
            idxs = df.index[mask].tolist()[: min(120, cap - added)]
            for idx in idxs:
                if added >= cap:
                    break
                val = _safe_str(df.at[idx, col])
                subj = _safe_str(df.at[idx, "subjectid"])
                site = _safe_str(df.at[idx, site_col])
                pct = 100.0 * float(counts.get(val, 0)) / max(total, 1)
                sev = _clamp(35 + 55 * (1.0 - pct / max(self.rarity_pct_threshold * 100, 1e-6)), 0, 95)

                self.results.append(_record(Finding(
                    network=network, timepoint=timepoint, subjectid=subj, site_id=site,
                    error_type="categorical_rarity", method="value_frequency",
                    severity_score=sev, severity_label=_severity_label(sev),
                    message=f"Rare categorical value for '{col}'.",
                    what_is_wrong=f"Value '{val}' appears in only {pct:.3f}% of non-missing responses.",
                    suggested_action="Verify value allowed by codebook; check whitespace/casing; ensure mapping/export did not introduce a new code.",
                    variables_involved=col,
                    variable_values=f"{col}={val}",
                    score_or_pct=f"{pct:.3f}%",
                    row_index=int(idx),
                )))
                added += 1

    def _discover_categorical_pair_rarity(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        cols = _select_categorical_columns(df, max_cols=min(45, self.max_cat_cols))
        if len(cols) < 6:
            return

        # Choose columns with most filled data
        filled_counts = []
        for c in cols:
            ser = df[c].astype(str).str.strip()
            ser = ser.replace("", np.nan).replace([str(x) for x in MISSING_CODES], np.nan)
            filled_counts.append((c, int(ser.notna().sum())))
        filled_counts.sort(key=lambda x: x[1], reverse=True)
        cols = [c for c, _ in filled_counts[:18]]  # keep runtime reasonable

        cleaned: Dict[str, pd.Series] = {}
        for c in cols:
            ser = df[c].astype(str).str.strip()
            ser = ser.replace("", np.nan).replace([str(x) for x in MISSING_CODES], np.nan)
            cleaned[c] = ser

        cap = 350
        added = 0

        for i in range(len(cols)):
            if added >= cap:
                break
            for j in range(i + 1, len(cols)):
                if added >= cap:
                    break
                a, b = cols[i], cols[j]
                sa, sb = cleaned[a], cleaned[b]
                both = sa.notna() & sb.notna()
                if both.sum() < 140:
                    continue

                pair = sa[both].astype(str) + " || " + sb[both].astype(str)
                counts = pair.value_counts()
                total = int(pair.shape[0])
                rare_threshold = max(2, int(total * self.rarity_pair_pct_threshold))
                rare_pairs = counts[counts <= rare_threshold].index.tolist()
                if not rare_pairs:
                    continue

                for rp in rare_pairs[:5]:
                    va, vb = rp.split(" || ", 1)
                    mask = (sa.astype(str) == va) & (sb.astype(str) == vb)
                    idxs = df.index[mask].tolist()[: min(25, cap - added)]
                    pct = 100.0 * float(counts.get(rp, 0)) / max(total, 1)

                    for idx in idxs:
                        if added >= cap:
                            break
                        subj = _safe_str(df.at[idx, "subjectid"])
                        site = _safe_str(df.at[idx, site_col])
                        sev = _clamp(45 + 50 * (1.0 - pct / max(self.rarity_pair_pct_threshold * 100, 1e-6)), 0, 98)

                        self.results.append(_record(Finding(
                            network=network, timepoint=timepoint, subjectid=subj, site_id=site,
                            error_type="categorical_pair_rarity", method="pair_frequency",
                            severity_score=sev, severity_label=_severity_label(sev),
                            message=f"Rare categorical combination: '{a}={va}' AND '{b}={vb}'.",
                            what_is_wrong=f"This pair occurs in only {pct:.3f}% of rows where both fields are present.",
                            suggested_action="Check for skip-logic failure, inconsistent coding, or a combination that should be impossible under the instrument.",
                            variables_involved=f"{a};{b}",
                            variable_values=f"{a}={_safe_str(va)}; {b}={_safe_str(vb)}",
                            score_or_pct=f"{pct:.3f}%",
                            row_index=int(idx),
                        )))
                        added += 1

    def _discover_categorical_triple_rarity(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        """
        Rare categorical triples for low-cardinality columns only.
        This catches deeper skip-logic patterns without blowing up runtime.
        """
        cols = _select_categorical_columns(df, max_cols=min(50, self.max_cat_cols))
        if len(cols) < 6:
            return

        # keep only low-cardinality columns
        low_cols = []
        for c in cols:
            ser = df[c].astype(str).str.strip()
            ser = ser.replace("", np.nan).replace([str(x) for x in MISSING_CODES], np.nan)
            uniq = ser.dropna().nunique()
            if 2 <= uniq <= 10 and ser.notna().sum() >= 120:
                low_cols.append(c)

        low_cols = low_cols[:12]
        if len(low_cols) < 5:
            return

        # Pre-clean
        cleaned = {}
        for c in low_cols:
            ser = df[c].astype(str).str.strip()
            cleaned[c] = ser.replace("", np.nan).replace([str(x) for x in MISSING_CODES], np.nan)

        cap = 250
        added = 0

        # Evaluate a limited set of triples
        for i in range(len(low_cols)):
            for j in range(i + 1, len(low_cols)):
                for k in range(j + 1, len(low_cols)):
                    if added >= cap:
                        return
                    a, b, c = low_cols[i], low_cols[j], low_cols[k]
                    mask = cleaned[a].notna() & cleaned[b].notna() & cleaned[c].notna()
                    if mask.sum() < 200:
                        continue

                    triples = (
                        cleaned[a][mask].astype(str) + " || " +
                        cleaned[b][mask].astype(str) + " || " +
                        cleaned[c][mask].astype(str)
                    )
                    counts = triples.value_counts()
                    total = int(triples.shape[0])
                    rare_threshold = max(2, int(total * self.rarity_triple_pct_threshold))
                    rare_triples = counts[counts <= rare_threshold].index.tolist()
                    if not rare_triples:
                        continue

                    for rt in rare_triples[:3]:
                        va, vb, vc = rt.split(" || ", 2)
                        m2 = (cleaned[a].astype(str) == va) & (cleaned[b].astype(str) == vb) & (cleaned[c].astype(str) == vc)
                        idxs = df.index[m2].tolist()[: min(15, cap - added)]
                        pct = 100.0 * float(counts.get(rt, 0)) / max(total, 1)

                        for idx in idxs:
                            if added >= cap:
                                return
                            subj = _safe_str(df.at[idx, "subjectid"])
                            site = _safe_str(df.at[idx, site_col])
                            sev = _clamp(55 + 40 * (1.0 - pct / max(self.rarity_triple_pct_threshold * 100, 1e-6)), 0, 99)

                            self.results.append(_record(Finding(
                                network=network, timepoint=timepoint, subjectid=subj, site_id=site,
                                error_type="categorical_triple_rarity", method="triple_frequency",
                                severity_score=sev, severity_label=_severity_label(sev),
                                message=f"Rare categorical triple: '{a}={va}', '{b}={vb}', '{c}={vc}'.",
                                what_is_wrong=f"This triple occurs in only {pct:.3f}% of rows where all three fields are present.",
                                suggested_action="Check whether this combination violates skip logic or represents a miscoding of one of the fields.",
                                variables_involved=f"{a};{b};{c}",
                                variable_values=f"{a}={_safe_str(va)}; {b}={_safe_str(vb)}; {c}={_safe_str(vc)}",
                                score_or_pct=f"{pct:.3f}%",
                                row_index=int(idx),
                            )))
                            added += 1

    def _discover_categorical_pattern_surprisal(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        """
        Rare overall categorical patterns: each row gets a pattern string across selected categorical cols.
        Rows with very rare patterns are flagged. Great for CRF-style forms.
        """
        cols = _select_categorical_columns(df, max_cols=min(40, self.max_cat_cols))
        if len(cols) < 8:
            return

        # pick top columns by filledness
        filled = []
        for c in cols:
            ser = df[c].astype(str).str.strip()
            ser = ser.replace("", np.nan).replace([str(x) for x in MISSING_CODES], np.nan)
            filled.append((c, int(ser.notna().sum())))
        filled.sort(key=lambda x: x[1], reverse=True)
        cols = [c for c, _ in filled[:14]]

        # Build pattern per row
        pat_df = pd.DataFrame(index=df.index)
        for c in cols:
            ser = df[c].astype(str).str.strip()
            ser = ser.replace("", np.nan).replace([str(x) for x in MISSING_CODES], np.nan)
            pat_df[c] = ser.fillna("<MISSING>").astype(str)

        pattern = pat_df.apply(lambda r: " | ".join([f"{c}={r[c]}" for c in cols]), axis=1)
        counts = pattern.value_counts()
        total = float(len(pattern))
        # "surprisal" = -log(p)
        p = pattern.map(lambda s: max(counts.get(s, 1) / total, 1e-12))
        surprisal = -np.log(p)

        # flag top surprisal
        thresh = np.quantile(surprisal.values, 0.99)
        idxs = surprisal[surprisal >= thresh].sort_values(ascending=False).index.tolist()[:200]
        for idx in idxs:
            subj = _safe_str(df.at[idx, "subjectid"])
            site = _safe_str(df.at[idx, site_col])
            sev = _clamp(60 + 10 * (surprisal.loc[idx] - thresh), 0, 98)

            # show only the differing pattern components (top columns)
            vals = "; ".join([f"{c}={_safe_str(df.at[idx, c])}" for c in cols[:10]])
            self.results.append(_record(Finding(
                network=network, timepoint=timepoint, subjectid=subj, site_id=site,
                error_type="categorical_pattern_surprisal", method="pattern_frequency",
                severity_score=sev, severity_label=_severity_label(sev),
                message="Row has a very rare overall categorical response pattern.",
                what_is_wrong=f"This row’s categorical pattern is rare (surprisal={float(surprisal.loc[idx]):.2f}).",
                suggested_action="Review whether skip logic or coding was applied correctly; rare patterns often reflect inconsistent combinations or miscoded categories.",
                variables_involved=";".join(cols),
                variable_values=vals,
                score_or_pct=f"surprisal={float(surprisal.loc[idx]):.2f}",
                row_index=int(idx),
            )))

    # ---------------------------
    # Duplicates
    # ---------------------------

    def _discover_duplicates(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        """
        Exact duplicates (fingerprint) + near duplicates (nearest-neighbor Hamming distance).
        Near duplicates are flagged if they are much more similar than typical nearest-neighbor similarity.
        """
        cols = [c for c in df.columns if c not in ("subjectid",)]
        cols = [c for c in cols if "__" not in c and not _is_excluded_variable(c)]
        if len(cols) < 10:
            return

        # Choose best columns for fingerprints: highest non-missing rates
        rates = []
        for c in cols:
            ser = df[c].apply(lambda v: np.nan if _is_missing(v) else v)
            rates.append((c, float(ser.notna().mean())))
        rates.sort(key=lambda x: x[1], reverse=True)
        cols_use = [c for c, _ in rates[: self.dup_cols_cap]]
        if len(cols_use) < 8:
            return

        # ---- Exact duplicates ----
        fps = []
        for _, row in df[cols_use].iterrows():
            fps.append(_fingerprint_row(row, cols_use))
        fp_ser = pd.Series(fps, index=df.index, name="_fp")
        dup_groups = fp_ser[fp_ser.duplicated(keep=False)].groupby(fp_ser).groups

        for _, idxs in list(dup_groups.items())[:80]:
            if len(idxs) < 2:
                continue
            idx_list = list(idxs)[:6]
            sample_row = df.loc[idx_list[0], cols_use]
            vals_str = "; ".join([f"{c}={_safe_str(sample_row.get(c))}" for c in cols_use[:10]])

            subj_ids = df.loc[idx_list, "subjectid"].astype(str).unique().tolist()
            site_ids = df.loc[idx_list, site_col].astype(str).unique().tolist()

            sev = 70.0
            if len(subj_ids) > 1:
                sev = 95.0
            elif len(site_ids) > 1:
                sev = 88.0

            self.results.append(_record(Finding(
                network=network, timepoint=timepoint, subjectid=_safe_str(subj_ids[0]) if subj_ids else "",
                site_id=_safe_str(site_ids[0]) if site_ids else "",
                error_type="exact_duplicate_rows", method="row_fingerprint",
                severity_score=sev, severity_label=_severity_label(sev),
                message="Exact duplicate rows detected (same fingerprint across key columns).",
                what_is_wrong=f"Rows {idx_list} share identical values across selected fields; subj_ids={subj_ids}; site_ids={site_ids}.",
                suggested_action="Confirm expected repeat instruments vs unintended duplicate import/merge. If unintended, reconcile/remove duplicates at source.",
                variables_involved=";".join(cols_use[:15]) + (";..." if len(cols_use) > 15 else ""),
                variable_values=vals_str,
                score_or_pct=f"dup_count={len(list(idxs))}",
                row_index=str(idx_list),
            )))

        # ---- Near duplicates: similarity relative to typical ----
        cols_cmp = cols_use[:20]
        if len(cols_cmp) < 6:
            return

        norm = df[cols_cmp].copy()
        for c in cols_cmp:
            norm[c] = norm[c].apply(lambda v: "<MISSING>" if _is_missing(v) else _safe_str(v, 60))

        codes = np.zeros((len(norm), len(cols_cmp)), dtype=np.int32)
        for j, c in enumerate(cols_cmp):
            codes[:, j] = pd.factorize(norm[c], sort=False)[0].astype(np.int32)

        nearest = []  # (row_i, row_j, dist)
        subj_groups = df.groupby(df["subjectid"].astype(str), dropna=False).groups
        for subj, idxs in subj_groups.items():
            idxs = list(idxs)
            if len(idxs) < 2:
                continue
            Xs = codes[np.array(idxs), :]
            try:
                nn = NearestNeighbors(n_neighbors=2, metric="hamming")
                nn.fit(Xs)
                dists, nbrs = nn.kneighbors(Xs, return_distance=True)
            except Exception:
                continue
            for local_i, global_i in enumerate(idxs):
                nn_local = int(nbrs[local_i, 1])
                nn_global = int(idxs[nn_local])
                dist = float(dists[local_i, 1])
                nearest.append((int(global_i), int(nn_global), dist))

        if not nearest:
            return

        nd_df = pd.DataFrame(nearest, columns=["row_i", "row_j", "nn_dist"])
        q25 = float(nd_df["nn_dist"].quantile(0.25))
        q50 = float(nd_df["nn_dist"].quantile(0.50))
        q75 = float(nd_df["nn_dist"].quantile(0.75))
        iqr = max(q75 - q25, 1e-9)
        low_outlier_thresh = max(0.0, q25 - 1.5 * iqr)
        p01 = float(nd_df["nn_dist"].quantile(0.01))
        flag_thresh = min(low_outlier_thresh, p01, 0.10)

        cands = nd_df[nd_df["nn_dist"] <= flag_thresh].copy()
        if cands.empty:
            return

        cands["a"] = cands[["row_i", "row_j"]].min(axis=1)
        cands["b"] = cands[["row_i", "row_j"]].max(axis=1)
        cands = cands.drop_duplicates(subset=["a", "b"]).sort_values("nn_dist", ascending=True)

        for _, r in cands.head(250).iterrows():
            i1 = int(r["a"])
            i2 = int(r["b"])
            dist = float(r["nn_dist"])
            similarity = 1.0 - dist

            subj = _safe_str(df.at[i1, "subjectid"])
            site = _safe_str(df.at[i1, site_col])
            site2 = _safe_str(df.at[i2, site_col])

            diffs = []
            diff_vals = []
            for c in cols_cmp:
                if str(norm.at[i1, c]) != str(norm.at[i2, c]):
                    diffs.append(c)
                    diff_vals.append(f"{c}: '{_safe_str(df.at[i1, c])}' vs '{_safe_str(df.at[i2, c])}'")

            sev = _clamp(60 + 40 * (1.0 - min(dist / max(flag_thresh, 1e-9), 1.0)), 0, 100)
            if site and site2 and site != site2:
                sev = _clamp(sev + 15, 0, 100)

            self.results.append(_record(Finding(
                network=network, timepoint=timepoint, subjectid=subj, site_id=site,
                error_type="near_duplicate_rows", method="nearest_neighbor_hamming",
                severity_score=sev, severity_label=_severity_label(sev),
                message="Rows are unusually similar relative to typical nearest-neighbor similarity (possible near-duplicate).",
                what_is_wrong=(
                    f"Row {i1} and Row {i2} differ in only {dist*100:.1f}% of compared fields "
                    f"(similarity={similarity*100:.1f}%). Typical NN distance median={q50*100:.1f}%. "
                    f"Flag threshold={flag_thresh*100:.1f}%."
                ),
                suggested_action="Check whether one row is a duplicate import/re-entry. If both should exist, verify why only a few fields differ (correction vs error).",
                variables_involved=";".join(diffs) if diffs else "(identical on compared fields)",
                variable_values="; ".join(diff_vals) if diff_vals else "(no differences found in compared fields)",
                score_or_pct=f"nn_dist={dist:.4f}",
                row_index=str([i1, i2]),
            )))

    # ---------------------------
    # ML numeric matrix preparation
    # ---------------------------

    def _prepare_numeric_matrix(self, df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        X = df[cols].copy()
        for c in cols:
            X[c] = _to_numeric_series(X[c].replace([str(x) for x in MISSING_CODES], np.nan))
        medians = X.median(numeric_only=True)
        X = X.fillna(medians)
        all_nan_cols = [c for c in X.columns if X[c].isna().all()]
        if all_nan_cols:
            X = X.drop(columns=all_nan_cols)
        return X, list(X.columns)

    def _top_contributors_by_robust_z(self, df: pd.DataFrame, cols: List[str], row_idx: int, topk: int = 6) -> Tuple[str, str]:
        rz_row = {}
        for c in cols:
            z = _robust_zscore(_to_numeric_series(df[c].replace([str(x) for x in MISSING_CODES], np.nan)))
            zv = z.iloc[row_idx] if row_idx < len(z) else np.nan
            rz_row[c] = float(abs(zv)) if pd.notna(zv) else 0.0
        top = sorted(rz_row.items(), key=lambda x: x[1], reverse=True)[:topk]
        vars_involved = ";".join([v for v, _ in top])
        vals = "; ".join([f"{v}={_safe_str(df.iloc[row_idx].get(v))}" for v, _ in top])
        return vars_involved, vals

    # ---------------------------
    # ML methods
    # ---------------------------

    def _discover_isolation_forest(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        cols = _select_numeric_columns(df, max_cols=min(70, self.max_numeric_cols))
        if len(cols) < 10:
            return
        X, cols = self._prepare_numeric_matrix(df, cols)
        if X.shape[1] < 6 or len(X) < 80:
            return

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        clf = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=350,
            max_samples=min(512, len(X)),
            n_jobs=-1,
        )
        pred = clf.fit_predict(Xs)
        scores = clf.score_samples(Xs)  # more negative = more anomalous

        anom_idx = np.where(pred == -1)[0]
        if anom_idx.size == 0:
            return
        anom_idx = anom_idx[np.argsort(scores[anom_idx])]

        cap = self.max_ml_findings_per_tp
        s_min = float(np.min(scores[anom_idx]))
        s_max = float(np.max(scores[anom_idx]))

        for ii in anom_idx[:cap]:
            row = df.iloc[int(ii)]
            subj = _safe_str(row.get("subjectid"))
            site = _safe_str(row.get(site_col))
            s = float(scores[int(ii)])
            norm = 0.0 if s_max == s_min else (s_max - s) / (s_max - s_min)
            sev = _clamp(55 + 40 * norm, 0, 100)
            vars_involved, vals = self._top_contributors_by_robust_z(df, cols, int(ii))

            self.results.append(_record(Finding(
                network=network, timepoint=timepoint, subjectid=subj, site_id=site,
                error_type="isolation_forest_anomaly", method="isolation_forest",
                severity_score=sev, severity_label=_severity_label(sev),
                message="Row has unusual multivariate numeric pattern (Isolation Forest).",
                what_is_wrong="This combination of numeric values is rare compared to the cohort pattern across many fields.",
                suggested_action="Inspect listed fields for unit/scale errors, swapped fields, missing-code leakage, or visit mis-assignment.",
                variables_involved=vars_involved,
                variable_values=vals,
                score_or_pct=f"IF_score={s:.4f}",
                row_index=int(ii),
            )))

    def _discover_lof(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        cols = _select_numeric_columns(df, max_cols=min(70, self.max_numeric_cols))
        if len(cols) < 10:
            return
        X, cols = self._prepare_numeric_matrix(df, cols)
        if X.shape[1] < 6 or len(X) < 120:
            return

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        n_neighbors = min(35, max(10, len(X) // 10))
        if n_neighbors < 10:
            return

        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            n_jobs=-1,
        )
        pred = lof.fit_predict(Xs)
        lof_scores = getattr(lof, "negative_outlier_factor_", None)
        if lof_scores is None:
            return

        anom_idx = np.where(pred == -1)[0]
        if anom_idx.size == 0:
            return
        anom_idx = anom_idx[np.argsort(lof_scores[anom_idx])]  # more negative first

        cap = min(self.max_ml_findings_per_tp, 250)
        s_min = float(np.min(lof_scores[anom_idx]))
        s_max = float(np.max(lof_scores[anom_idx]))

        for ii in anom_idx[:cap]:
            row = df.iloc[int(ii)]
            subj = _safe_str(row.get("subjectid"))
            site = _safe_str(row.get(site_col))

            s = float(lof_scores[int(ii)])
            norm = 0.0 if s_max == s_min else (s_max - s) / (s_max - s_min)
            sev = _clamp(50 + 40 * norm, 0, 100)

            vars_involved, vals = self._top_contributors_by_robust_z(df, cols, int(ii))
            self.results.append(_record(Finding(
                network=network, timepoint=timepoint, subjectid=subj, site_id=site,
                error_type="lof_anomaly", method="local_outlier_factor",
                severity_score=sev, severity_label=_severity_label(sev),
                message="Row is a local density outlier (LOF).",
                what_is_wrong="This row lies in a sparse region of numeric feature space compared to its neighbors.",
                suggested_action="Inspect listed fields for mis-entry, unit errors, swapped fields, or unusual combinations.",
                variables_involved=vars_involved,
                variable_values=vals,
                score_or_pct=f"LOF_neg_factor={float(s):.4f}",
                row_index=int(ii),
            )))

    def _discover_pca_reconstruction(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        cols = _select_numeric_columns(df, max_cols=min(75, self.max_numeric_cols))
        if len(cols) < 12:
            return
        X, cols = self._prepare_numeric_matrix(df, cols)
        if X.shape[1] < 8 or len(X) < 120:
            return

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        n_components = min(18, X.shape[1] - 1, len(X) - 1)
        if n_components < 4:
            return

        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_trans = pca.fit_transform(Xs)
        X_recon = pca.inverse_transform(X_trans)
        resid = (Xs - X_recon) ** 2
        mse = np.mean(resid, axis=1)

        thresh = float(np.quantile(mse, self.reconstruction_quantile))
        anom_idx = np.where(mse >= thresh)[0]
        if anom_idx.size == 0:
            return
        anom_idx = anom_idx[np.argsort(-mse[anom_idx])]

        cap = min(self.max_ml_findings_per_tp, 250)
        for ii in anom_idx[:cap]:
            row = df.iloc[int(ii)]
            subj = _safe_str(row.get("subjectid"))
            site = _safe_str(row.get(site_col))

            # top residual contributors
            r = resid[int(ii), :]
            topk = np.argsort(-r)[:6]
            vars_involved = ";".join([cols[k] for k in topk])
            vals = "; ".join([f"{cols[k]}={_safe_str(row.get(cols[k]))}" for k in topk])

            sev = _clamp(55 + 45 * ((float(mse[int(ii)]) - thresh) / max(thresh, 1e-9)), 0, 100)

            self.results.append(_record(Finding(
                network=network, timepoint=timepoint, subjectid=subj, site_id=site,
                error_type="pca_reconstruction_anomaly", method="pca_reconstruction",
                severity_score=sev, severity_label=_severity_label(sev),
                message="Row has high PCA reconstruction error (unusual numeric pattern).",
                what_is_wrong=f"Reconstruction MSE={float(mse[int(ii)]):.4f} exceeds threshold={thresh:.4f}.",
                suggested_action="Inspect listed fields for scale/unit issues, swapped fields, or coding inconsistencies.",
                variables_involved=vars_involved,
                variable_values=vals,
                score_or_pct=f"MSE={float(mse[int(ii)]):.4f}",
                row_index=int(ii),
            )))

    def _discover_kmeans_distance(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        cols = _select_numeric_columns(df, max_cols=min(60, self.max_numeric_cols))
        if len(cols) < 10:
            return
        X, cols = self._prepare_numeric_matrix(df, cols)
        if X.shape[1] < 8 or len(X) < 200:
            return

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        k = int(np.clip(np.sqrt(len(X) / 2), 4, 12))
        km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
        labels = km.fit_predict(Xs)
        centers = km.cluster_centers_

        # distance to assigned center
        d = np.linalg.norm(Xs - centers[labels], axis=1)
        thresh = float(np.quantile(d, 0.99))
        idxs = np.where(d >= thresh)[0]
        idxs = idxs[np.argsort(-d[idxs])]

        for ii in idxs[:200]:
            row = df.iloc[int(ii)]
            subj = _safe_str(row.get("subjectid"))
            site = _safe_str(row.get(site_col))
            sev = _clamp(55 + 45 * ((float(d[int(ii)]) - thresh) / max(thresh, 1e-9)), 0, 100)
            vars_involved, vals = self._top_contributors_by_robust_z(df, cols, int(ii))

            self.results.append(_record(Finding(
                network=network, timepoint=timepoint, subjectid=subj, site_id=site,
                error_type="kmeans_distance_anomaly", method="kmeans_distance",
                severity_score=sev, severity_label=_severity_label(sev),
                message="Row is far from its nearest numeric cluster center (KMeans distance).",
                what_is_wrong=f"Distance to cluster center={float(d[int(ii)]):.4f} exceeds threshold={thresh:.4f}.",
                suggested_action="Inspect listed fields; cluster-distance anomalies often indicate unusual combinations, unit issues, or swapped values.",
                variables_involved=vars_involved,
                variable_values=vals,
                score_or_pct=f"dist={float(d[int(ii)]):.4f}",
                row_index=int(ii),
            )))

    def _discover_mahalanobis(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        cols = _select_numeric_columns(df, max_cols=min(55, self.max_numeric_cols))
        if len(cols) < 10:
            return
        X, cols = self._prepare_numeric_matrix(df, cols)
        if X.shape[1] < 8 or len(X) < 250:
            return

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        try:
            lw = LedoitWolf().fit(Xs)
            cov_inv = np.linalg.inv(lw.covariance_)
            mu = lw.location_
        except Exception:
            return

        # squared mahalanobis distance
        diff = Xs - mu
        md2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)

        thresh = float(np.quantile(md2, 0.99))
        idxs = np.where(md2 >= thresh)[0]
        idxs = idxs[np.argsort(-md2[idxs])]

        for ii in idxs[:200]:
            row = df.iloc[int(ii)]
            subj = _safe_str(row.get("subjectid"))
            site = _safe_str(row.get(site_col))
            sev = _clamp(60 + 40 * ((float(md2[int(ii)]) - thresh) / max(thresh, 1e-9)), 0, 100)
            vars_involved, vals = self._top_contributors_by_robust_z(df, cols, int(ii))

            self.results.append(_record(Finding(
                network=network, timepoint=timepoint, subjectid=subj, site_id=site,
                error_type="mahalanobis_anomaly", method="mahalanobis_ledoitwolf",
                severity_score=sev, severity_label=_severity_label(sev),
                message="Row is a multivariate outlier (Mahalanobis distance).",
                what_is_wrong=f"Mahalanobis^2={float(md2[int(ii)]):.4f} exceeds threshold={thresh:.4f}.",
                suggested_action="Inspect listed fields; Mahalanobis outliers often indicate scale/unit problems or inconsistent multivariate combinations.",
                variables_involved=vars_involved,
                variable_values=vals,
                score_or_pct=f"MD2={float(md2[int(ii)]):.4f}",
                row_index=int(ii),
            )))

    # ---------------------------
    # Site monitoring
    # ---------------------------

    def _discover_site_distribution_shift(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        cols = _select_categorical_columns(df, max_cols=min(70, self.max_cat_cols))
        if not cols or site_col not in df.columns:
            return
        cols = [c for c in cols if c != site_col][:45]

        cap = 160
        added = 0
        for col in cols:
            if added >= cap:
                break
            tab = pd.crosstab(df[site_col], df[col])
            if tab.size < 8 or tab.shape[0] < 2 or tab.shape[1] < 2:
                continue

            if _HAS_SCIPY:
                try:
                    _, p, _, _ = chi2_contingency(tab)
                except Exception:
                    continue
                if p >= self.site_chi2_p_threshold:
                    continue
            else:
                p = 1.0

            marginal = tab.sum(axis=0) / max(tab.sum().sum(), 1)
            worst_site, worst_diff, worst_over, worst_over_amt = None, -1.0, None, 0.0
            for s in tab.index.tolist():
                row_counts = tab.loc[s]
                if row_counts.sum() < 40:
                    continue
                row_dist = row_counts / row_counts.sum()
                diff = float((row_dist - marginal).abs().sum())
                if diff > worst_diff:
                    worst_diff = diff
                    worst_site = s
                    over = (row_dist - marginal).sort_values(ascending=False)
                    worst_over = str(over.index[0])
                    worst_over_amt = float(over.iloc[0])

            if worst_site is None or worst_diff < 0.18:
                continue

            sev = 70.0
            if _HAS_SCIPY:
                sev = _clamp(60 + 15 * (-np.log10(max(p, 1e-12))), 0, 100)

            self.results.append(_record(Finding(
                network=network, timepoint=timepoint, subjectid="", site_id=_safe_str(worst_site),
                error_type="site_distribution_shift", method="chi2_crosstab" if _HAS_SCIPY else "crosstab_shift_proxy",
                severity_score=sev, severity_label=_severity_label(sev),
                message=f"Variable '{col}' distribution differs by site.",
                what_is_wrong=(
                    f"Site '{worst_site}' differs from overall distribution (L1 diff={worst_diff:.3f}). "
                    + (f"Chi-square p={p:.2e}. " if _HAS_SCIPY else "")
                    + f"Most overrepresented category appears to be '{_safe_str(worst_over)}' (over by ~{worst_over_amt:.3f})."
                ),
                suggested_action="Investigate site-specific coding/training/defaults/translations or mapping/export differences.",
                variables_involved=col,
                variable_values=f"site={_safe_str(worst_site)}; overrepresented='{_safe_str(worst_over)}'",
                score_or_pct=(f"p={p:.2e}" if _HAS_SCIPY else f"L1={worst_diff:.3f}"),
                row_index="",
            )))
            added += 1

    def _discover_site_numeric_shift(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        cols = _select_numeric_columns(df, max_cols=min(60, self.max_numeric_cols))
        if not cols or site_col not in df.columns:
            return

        cap = 200
        added = 0
        for col in cols[:45]:
            if added >= cap:
                break
            x = _to_numeric_series(df[col].replace([str(x) for x in MISSING_CODES], np.nan))
            valid = x.dropna()
            if valid.shape[0] < 120:
                continue

            global_med = float(np.nanmedian(valid.values))
            global_mad = float(np.nanmedian(np.abs(valid.values - global_med)))
            if global_mad <= 0 or np.isnan(global_mad):
                continue
            global_std = float(np.nanstd(valid.values))

            for site, g in df.groupby(site_col, dropna=False):
                if added >= cap:
                    break
                gx = _to_numeric_series(g[col].replace([str(x) for x in MISSING_CODES], np.nan))
                gv = gx.dropna()
                if gv.shape[0] < 40:
                    continue
                site_med = float(np.nanmedian(gv.values))
                rz = 0.6745 * (site_med - global_med) / global_mad

                site_std = float(np.nanstd(gv.values))
                low_var = (site_std < max(1e-8, 0.10 * global_std)) if np.isfinite(global_std) else False

                if abs(rz) >= self.site_shift_rz_threshold or low_var:
                    sev = _clamp(65 + 8 * abs(rz) + (12 if low_var else 0), 0, 100)
                    what = f"Site median differs from global: site_med={site_med:.4g}, global_med={global_med:.4g} (robust_z={rz:.2f})."
                    if low_var:
                        what += f" Also unusually low variance (site_std={site_std:.4g} vs global_std={global_std:.4g})."

                    self.results.append(_record(Finding(
                        network=network, timepoint=timepoint, subjectid="", site_id=_safe_str(site),
                        error_type="site_numeric_shift", method="robust_site_median_shift",
                        severity_score=sev, severity_label=_severity_label(sev),
                        message=f"Numeric distribution shift for '{col}' at site '{site}'.",
                        what_is_wrong=what,
                        suggested_action="Check units/device calibration/site processing; verify no site-specific scaling/coding differences.",
                        variables_involved=col,
                        variable_values=f"site={_safe_str(site)}; site_med={site_med:.4g}; global_med={global_med:.4g}",
                        score_or_pct=f"robust_z={rz:.2f}" + (";low_var=1" if low_var else ""),
                        row_index="",
                    )))
                    added += 1

    def _discover_site_corr_profile(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str) -> None:
        """
        Compare correlation profiles: if a site's correlation structure differs sharply from global,
        it may indicate reversed coding, swapped columns, unit changes, etc.
        """
        cols = _select_numeric_columns(df, max_cols=min(35, self.max_numeric_cols))
        if len(cols) < 10 or site_col not in df.columns:
            return

        # Build global numeric matrix (coerce)
        Xg = pd.DataFrame(index=df.index)
        for c in cols:
            Xg[c] = _to_numeric_series(df[c].replace([str(x) for x in MISSING_CODES], np.nan))
        # Require enough data
        if Xg.dropna().shape[0] < 200:
            return

        global_corr = Xg.corr().fillna(0.0)

        cap = 120
        added = 0
        for site, g in df.groupby(site_col, dropna=False):
            if added >= cap:
                break
            if len(g) < 180:
                continue

            Xs = pd.DataFrame(index=g.index)
            for c in cols:
                Xs[c] = _to_numeric_series(g[c].replace([str(x) for x in MISSING_CODES], np.nan))
            if Xs.dropna().shape[0] < 120:
                continue

            site_corr = Xs.corr().fillna(0.0)

            # Compare correlation profiles per variable
            for v in cols:
                if added >= cap:
                    break
                # L1 difference between correlation vectors (excluding self)
                gv = global_corr[v].drop(index=v, errors="ignore")
                sv = site_corr[v].drop(index=v, errors="ignore")
                # Align
                sv = sv.reindex(gv.index).fillna(0.0)
                diff = float((sv - gv).abs().sum()) / max(len(gv), 1)

                if diff >= self.site_corr_profile_threshold:
                    sev = _clamp(70 + 60 * (diff - self.site_corr_profile_threshold), 0, 100)

                    # Identify top changed correlations
                    delta = (sv - gv).abs().sort_values(ascending=False)
                    top_other = delta.head(5).index.tolist()
                    vals = "; ".join([f"corr({v},{o}) site={sv[o]:.2f} vs global={gv[o]:.2f}" for o in top_other])

                    self.results.append(_record(Finding(
                        network=network, timepoint=timepoint, subjectid="", site_id=_safe_str(site),
                        error_type="site_correlation_profile_shift", method="corr_profile_l1",
                        severity_score=sev, severity_label=_severity_label(sev),
                        message=f"Correlation profile shift at site '{site}' for variable '{v}'.",
                        what_is_wrong=f"Site correlation profile differs from global (avg L1 diff={diff:.3f}).",
                        suggested_action="Check for reversed coding, swapped columns, unit changes, or site-specific preprocessing affecting this variable.",
                        variables_involved=v,
                        variable_values=vals,
                        score_or_pct=f"profile_diff={diff:.3f}",
                        row_index="",
                    )))
                    added += 1
    def _discover_site_drift_over_time(self, df: pd.DataFrame, network: str, timepoint: str, site_col: str, dt_col: str) -> None:
        """
        Detect site drift over time by looking at monthly medians (numeric) and category proportions (categorical).
        Flags abrupt step changes in site behavior.
        """
        try:
            t = pd.to_datetime(_get_series(df, dt_col), errors="coerce", infer_datetime_format=True)
        except Exception:
            return
        if t.notna().sum() < max(80, int(0.2 * len(df))):
            return

        work = df.copy()
        work["_dt"] = t
        work = work[work["_dt"].notna()].copy()
        if len(work) < 150:
            return

        work["_period"] = work["_dt"].dt.to_period("M").astype(str)

        # Numeric drift
        num_cols = _select_numeric_columns(work, max_cols=min(25, self.max_numeric_cols))
        if num_cols:
            for c in num_cols[:20]:
                try:
                    ser = _get_series(work, c)
                except Exception:
                    continue

                tmp = pd.DataFrame({
                    site_col: _get_series(work, site_col),
                    "_period": work["_period"],
                    c: _to_numeric_series(ser.replace([str(x) for x in MISSING_CODES], np.nan)),
                })

                tmp = tmp.dropna(subset=[c])
                if tmp.shape[0] < 200:
                    continue

                grp = tmp.groupby([site_col, "_period"])[c].median().reset_index(name="med")

                for site, g in grp.groupby(site_col):
                    if g.shape[0] < 4:
                        continue
                    g = g.sort_values("_period")
                    med = g["med"].values
                    step = np.diff(med)
                    if len(step) < 3:
                        continue

                    rz = _robust_zscore(pd.Series(step)).abs().fillna(0.0)
                    if rz.max() >= self.drift_step_rz_threshold:
                        k = int(rz.idxmax())
                        sev = _clamp(70 + 7 * float(rz.max()), 0, 100)
                        before_p = g.iloc[k]["_period"]
                        after_p = g.iloc[k + 1]["_period"]
                        before_med = float(g.iloc[k]["med"])
                        after_med = float(g.iloc[k + 1]["med"])

                        self.results.append(_record(Finding(
                            network=network, timepoint=timepoint, subjectid="", site_id=_safe_str(site),
                            error_type="site_drift_over_time_numeric", method="monthly_median_step",
                            severity_score=sev, severity_label=_severity_label(sev),
                            message=f"Site drift over time detected for '{c}' at site '{site}'.",
                            what_is_wrong=(
                                f"Monthly median step change from {before_p} to {after_p}: "
                                f"{before_med:.4g} -> {after_med:.4g} (step robust_z~{float(rz.max()):.2f})."
                            ),
                            suggested_action="Investigate site workflow change, device calibration, staff changes, or pipeline transformation changes around that time.",
                            variables_involved=c,
                            variable_values=f"{dt_col} period change {before_p}->{after_p}",
                            score_or_pct=f"step_rz={float(rz.max()):.2f}",
                            row_index="",
                        )))

        # Categorical drift
        cat_cols = _select_categorical_columns(work, max_cols=min(20, self.max_cat_cols))
        if cat_cols:
            for c in cat_cols[:12]:
                try:
                    ser = _get_series(work, c)
                except Exception:
                    continue

                tmp = pd.DataFrame({
                    site_col: _get_series(work, site_col),
                    "_period": work["_period"],
                    c: _clean_categorical_series(ser),
                })

                tmp = tmp.dropna(subset=[c])
                if tmp.shape[0] < 250:
                    continue

                top_vals = tmp[c].value_counts().head(3).index.tolist()
                for v in top_vals:
                    tmp["_isv"] = (tmp[c].astype(str) == str(v)).astype(int)
                    grp = tmp.groupby([site_col, "_period"])["_isv"].mean().reset_index(name="prop")

                    for site, g in grp.groupby(site_col):
                        if g.shape[0] < 4:
                            continue
                        g = g.sort_values("_period")
                        prop = g["prop"].values
                        step = np.diff(prop)
                        if len(step) < 3:
                            continue

                        rz = _robust_zscore(pd.Series(step)).abs().fillna(0.0)
                        if rz.max() >= self.drift_step_rz_threshold:
                            k = int(rz.idxmax())
                            sev = _clamp(70 + 7 * float(rz.max()), 0, 100)
                            before_p = g.iloc[k]["_period"]
                            after_p = g.iloc[k + 1]["_period"]
                            before_prop = float(g.iloc[k]["prop"])
                            after_prop = float(g.iloc[k + 1]["prop"])

                            self.results.append(_record(Finding(
                                network=network, timepoint=timepoint, subjectid="", site_id=_safe_str(site),
                                error_type="site_drift_over_time_categorical", method="monthly_prop_step",
                                severity_score=sev, severity_label=_severity_label(sev),
                                message=f"Site drift over time detected for '{c}' category '{v}' at site '{site}'.",
                                what_is_wrong=(
                                    f"Monthly proportion step change from {before_p} to {after_p}: "
                                    f"{before_prop:.3f} -> {after_prop:.3f} (step robust_z~{float(rz.max()):.2f})."
                                ),
                                suggested_action="Investigate site coding/training changes, default values, translation differences, or export/mapping changes around that time.",
                                variables_involved=c,
                                variable_values=f"category='{_safe_str(v)}'; {dt_col} period change {before_p}->{after_p}",
                                score_or_pct=f"step_rz={float(rz.max()):.2f}",
                                row_index="",
                            )))    # ---------------------------
        # Candidate rules + severity fusion
        # ---------------------------

    def _write_candidate_rules(self, out_df: pd.DataFrame) -> None:
        if out_df.empty:
            return
        work = out_df.copy()
        for col in ["error_type", "method", "variables_involved",
        "severity_score", "subjectid", "site_id", "timepoint", "row_index", "variable_values"]:
            if col not in work.columns:
                work[col] = ""
        work["severity_score_num"] = pd.to_numeric(work["severity_score"], errors="coerce").fillna(0.0)

        candidate_rows = []

        def examples(df_sub: pd.DataFrame, n: int = 5) -> str:
            ex = []
            for _, r in df_sub.head(n).iterrows():
                ex.append(f"subj={_safe_str(r.get('subjectid',''))}, site={_safe_str(r.get('site_id',''))}, tp={_safe_str(r.get('timepoint',''))}, row={_safe_str(r.get('row_index',''))}")
            return " | ".join(ex)

        def example_values(df_sub: pd.DataFrame, n: int = 3) -> str:
            vals = []
            for _, r in df_sub.head(n).iterrows():
                vals.append(_safe_str(r.get("variable_values", ""), 220))
            return " || ".join(vals)

        # Group by error_type + variables_involved as a default "pattern"
        var_based = work[work["variables_involved"].astype(str).str.strip() != ""].copy()
        if not var_based.empty:
            grouped = (
                var_based.groupby(["error_type", "variables_involved"], dropna=False)
                .agg(
                    n_occurrences=("error_type", "size"),
                    max_severity=("severity_score_num", "max"),
                    mean_severity=("severity_score_num", "mean"),
                    n_subjects=("subjectid", pd.Series.nunique),
                    n_sites=("site_id", pd.Series.nunique),
                )
                .reset_index()
                .sort_values(["n_occurrences", "max_severity", "n_subjects"], ascending=False)
            )
            for _, g in grouped.iterrows():
                if int(g["n_occurrences"]) < 3:
                    continue
                err = _safe_str(g["error_type"])
                vars_involved = _safe_str(g["variables_involved"])
                sub = var_based[(var_based["error_type"] == g["error_type"]) & (var_based["variables_involved"] == g["variables_involved"])].sort_values("severity_score_num", ascending=False)

                candidate_rows.append({
                    "candidate_rule_type": "variable_pattern",
                    "error_type": err,
                    "variables_involved": vars_involved,
                    "n_occurrences": int(g["n_occurrences"]),
                    "n_subjects": int(g["n_subjects"]),
                    "n_sites": int(g["n_sites"]),
                    "max_severity": float(g["max_severity"]),
                    "mean_severity": round(float(g["mean_severity"]), 2),
                    "proposed_rule": f"Consider promoting repeated anomaly pattern '{err}' for variable(s) '{vars_involved}' into a deterministic QC rule.",
                    "rationale": "Pattern repeats across records; likely systematic enough for a deterministic check (after reviewer confirmation).",
                    "example_rows": examples(sub),
                    "example_values": example_values(sub),
                })

        # Site-specific patterns
        site_based = work[work["site_id"].astype(str).str.strip() != ""].copy()
        if not site_based.empty:
            grouped = (
                site_based.groupby(["error_type", "site_id", "variables_involved"], dropna=False)
                .agg(
                    n_occurrences=("error_type", "size"),
                    max_severity=("severity_score_num", "max"),
                    mean_severity=("severity_score_num", "mean"),
                    n_subjects=("subjectid", pd.Series.nunique),
                )
                .reset_index()
                .sort_values(["n_occurrences", "max_severity"], ascending=False)
            )
            for _, g in grouped.iterrows():
                if int(g["n_occurrences"]) < 3:
                    continue
                err = _safe_str(g["error_type"])
                site = _safe_str(g["site_id"])
                vars_involved = _safe_str(g["variables_involved"])
                sub = site_based[(site_based["error_type"] == g["error_type"]) & (site_based["site_id"] == g["site_id"]) & (site_based["variables_involved"] == g["variables_involved"])].sort_values("severity_score_num", ascending=False)

                candidate_rows.append({
                    "candidate_rule_type": "site_pattern",
                    "error_type": err,
                    "site_id": site,
                    "variables_involved": vars_involved,
                    "n_occurrences": int(g["n_occurrences"]),
                    "n_subjects": int(g["n_subjects"]),
                    "n_sites": 1,
                    "max_severity": float(g["max_severity"]),
                    "mean_severity": round(float(g["mean_severity"]), 2),
                    "proposed_rule": f"Add site-targeted QC monitoring for site '{site}' on variable(s) '{vars_involved}' for '{err}'.",
                    "rationale": "Pattern recurs within one site, suggesting site-specific workflow/coding/device/pipeline issues.",
                    "example_rows": examples(sub),
                    "example_values": example_values(sub),
                })

        if not candidate_rows:
            print("No candidate rules identified.")
            return

        cand_df = pd.DataFrame(candidate_rows)
        cand_df["priority_score"] = (
            0.50 * pd.to_numeric(cand_df["max_severity"], errors="coerce").fillna(0.0) +
            0.30 * np.log1p(pd.to_numeric(cand_df["n_occurrences"], errors="coerce").fillna(0.0)) * 20 +
            0.20 * np.log1p(pd.to_numeric(cand_df["n_subjects"], errors="coerce").fillna(0.0)) * 15
        )
        cand_df = cand_df.sort_values(["priority_score", "n_occurrences", "max_severity"], ascending=False).reset_index(drop=True)
        cand_df["candidate_rule_rank"] = np.arange(1, len(cand_df) + 1)

        out_path = os.path.join(self.output_path, "candidate_rules.csv")
        cand_df.to_csv(out_path, index=False)
        print(f"Wrote candidate rules to {out_path}")

    def _severity_fusion(self, out_df: pd.DataFrame) -> pd.DataFrame:
        """
        Improve ranking by:
        - converting each finding's severity to a percentile (calibration)
        - adding a bonus for multi-method agreement on same row/subject/timepoint
        """
        df = out_df.copy()
        df["severity_num"] = pd.to_numeric(df.get("severity_score", 0), errors="coerce").fillna(0.0)

        # percentile calibration within each error_type (avoids one method dominating)
        df["sev_pct"] = 0.0
        for et, g in df.groupby("error_type", dropna=False):
            if len(g) < 10:
                df.loc[g.index, "sev_pct"] = g["severity_num"].rank(pct=True)
            else:
                df.loc[g.index, "sev_pct"] = g["severity_num"].rank(pct=True, method="average")

        # multi-method agreement: same (network,timepoint,subjectid,row_index) with multiple error_types
        key_cols = ["network", "timepoint", "subjectid", "row_index"]
        for c in key_cols:
            if c not in df.columns:
                df[c] = ""
        grp = df.groupby(key_cols, dropna=False)["error_type"].nunique().rename("n_methods").reset_index()
        df = df.merge(grp, on=key_cols, how="left")
        df["n_methods"] = df["n_methods"].fillna(1).astype(int)

        bonus = (df["n_methods"] - 1).clip(lower=0, upper=4) * 0.06  # up to +0.24
        df["priority_score"] = df["sev_pct"] + bonus

        return df.sort_values(["priority_score", "severity_num"], ascending=False).reset_index(drop=True)

    def _write_output(self) -> None:
        if not self.results:
            print("No discoveries.")
            return

        out_df = pd.DataFrame(self.results)

        # Severity fusion and ranking
        out_df = self._severity_fusion(out_df)
        out_df["severity_rank"] = np.arange(1, len(out_df) + 1)

        os.makedirs(self.output_path, exist_ok=True)

        out_path = os.path.join(self.output_path, "discovered_errors_report.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {len(out_df)} discoveries to {out_path}")

        # Summary
        summary_cols = ["network", "timepoint", "severity_label", "error_type", "method"]
        summ = out_df.groupby(summary_cols).size().reset_index(name="n")
        summ_path = os.path.join(self.output_path, "discovered_errors_summary.csv")
        summ.to_csv(summ_path, index=False)
        print(f"Wrote summary to {summ_path}")

        # Candidate rules
        self._write_candidate_rules(out_df)

        # Top 100 readable
        top_txt = os.path.join(self.output_path, "discovered_errors_top100.txt")
        with open(top_txt, "w", encoding="utf-8") as f:
            f.write("Top 100 anomalies (highest priority first)\n\n")
            for _, r in out_df.head(100).iterrows():
                f.write(f"Rank {int(r['severity_rank'])} | {r.get('severity_label','')} | score={r.get('severity_score','')}\n")
                f.write(f"Network/TP: {r.get('network','')} / {r.get('timepoint','')}\n")
                f.write(f"Subject: {r.get('subjectid','')} | Site: {r.get('site_id','')} | Row: {r.get('row_index','')}\n")
                f.write(f"Type/Method: {r.get('error_type','')} / {r.get('method','')}\n")
                f.write(f"Vars: {r.get('variables_involved','')}\n")
                f.write(f"Vals: {r.get('variable_values','')}\n")
                f.write(f"What: {r.get('what_is_wrong','')}\n")
                f.write(f"Action: {r.get('suggested_action','')}\n")
                f.write(f"Score detail: {r.get('score_or_pct','')}\n")
                f.write("-" * 80 + "\n")
        print(f"Wrote top100 view to {top_txt}")

        if "error_type" in out_df.columns:
            print("Counts by error_type:")
            print(out_df["error_type"].value_counts().head(30))


if __name__ == "__main__":
    DiscoverNewErrors().run_script()