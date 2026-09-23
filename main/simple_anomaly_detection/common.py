"""Shared helpers for simple_anomaly_detection.

Goals:
  - One ``Finding`` shape so every detector produces interchangeable rows.
  - One severity calibration so the combined sheet can rank cross-detector.
  - Conservative defaults so the first run is reviewable, not overwhelming.

The severity convention used everywhere here:
    severity_score = 0 ... 100
        ~50  : at the conservative flag threshold
        ~95  : obviously extreme; the kind of case you definitely want to see
        100  : ceiling; we never push past it

Each detector also reports its native ``raw_score`` so the user can compare
within a type without losing fidelity.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Missing / sentinel codes used across the AMPSCZ combined CSVs.
# ---------------------------------------------------------------------------

MISSING_NUMERIC_CODES: Tuple[float, ...] = (
    -3.0, -9.0, -99.0, 999.0,
)
MISSING_STRING_CODES: Tuple[str, ...] = (
    "", "nan", "NaN", "NA", "N/A", "n/a", "None", "none",
    "-3", "-9", "-99", "999",
    "-3.0", "-9.0", "-99.0", "999.0",
    "1909-09-09", "1903-03-03", "1901-01-01",
)

# Date sentinels that pd.to_datetime would happily parse as real dates.
# Stripped before date parsing so 1909-09-09 doesn't become "year 1909".
DATE_SENTINEL_STRINGS: Tuple[str, ...] = (
    "1909-09-09", "1903-03-03", "1901-01-01",
    "9999-09-09", "9999-12-31",
)


# Substrings (case-insensitive) that mark a column as not interesting for
# anomaly detection. Kept aggressive on free-text fields so the format
# detector does not drown the report; the rest are bookkeeping columns
# that produce noisy outliers.
EXCLUDED_VARIABLE_SUBSTRINGS: Tuple[str, ...] = (
    "_complete",
    "_missing",
    "_count",
    "_other",
    "_pharm_",
    "comment",
    "note",
    "description",
    "redcap_event_name",
    "redcap_repeat",
    "chrdig",
    "chrax",
    # Blood-sample bookkeeping: sample IDs, well/aliquot positions,
    # rack barcodes, storage box numbers. They're not analytic data
    # and produce spurious flags. Patterns cover every chrblood_* sample
    # ID/position/barcode/box variable in the AMPSCZ data dictionary
    # (whole-blood wb1..wb3, serum se1..se3, plasma pl1..pl6, buffy
    # coat bc1, plus the rack and barcode fields).
    "chrblood_wb",
    "chrblood_se",
    "chrblood_pl",
    "chrblood_bc",
    "chrblood_rack",
    "barcode",
    "rack",
    "_box",
    # Synthetic / administrative identifiers. Rater fields are consumed
    # directly by rater_effect.py before classification; they must not also be
    # treated as numeric, categorical, or free-text clinical measurements.
    "redcap_",
    "_ampscz_id",
    "record_id",
    "_tp",
    # Pharmaceutical fields use the ``chrpharm_*`` prefix in the production
    # dictionary (not the historical ``_pharm_`` spelling above). Medication
    # course/dose changes are intentionally outside this generic anomaly model.
    "chrpharm",
    # Known rater/user fields with legacy spelling variants. They remain
    # available to the dedicated rater detector but are not clinical values for
    # the general anomaly models.
    "chrcssrsfu_redacp_user",
    "chrpps_redcao_user",
    "chrpred_username",
)

# REDCap descriptive/calculated display fields. These suffixes cover thousands
# of instruction, table, validation-message, and piping-helper variables in the
# AMPSCZ dictionary. Their generated text is not participant data and otherwise
# produces large, non-actionable format/categorical tails.
_EXCLUDED_VARIABLE_SUFFIX_RE = re.compile(
    r"(?:_inst\d*|_table|_list|_err|^chrpsychs_.*_app)$", re.IGNORECASE)

# Several legacy entries above were documented as "substrings", but literal
# substring matching creates real collisions (``_count`` in
# ``current_country`` and ``rack`` in ``crackle``). Treat these as variable-name
# tokens. The remaining entries are safe prefixes/substrings.
_TOKEN_EXCLUSION_RE = re.compile(
    r"(?:^|_)(?:complete|missing|count|other|comment|note|description|rack|box)(?:_|$)",
    re.IGNORECASE,
)

# Identity columns that every detector should leave alone.
ID_COLUMNS: Tuple[str, ...] = (
    "subjectid", "subject_id", "src_subject_id", "study_id",
)


# ---------------------------------------------------------------------------
# Finding row shape.
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    anomaly_type: str
    severity_score: float
    raw_score: float
    network: str = ""
    timepoint: str = ""
    site_id: str = ""
    subjectid: str = ""
    variable: str = ""
    variables_involved: str = ""
    observed_value: str = ""
    expected_value: str = ""
    explanation: str = ""
    method: str = ""
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        d = asdict(self)
        d["severity_score"] = round(float(self.severity_score), 2)
        d["raw_score"] = (
            round(float(self.raw_score), 4)
            if not (isinstance(self.raw_score, float) and math.isnan(self.raw_score))
            else ""
        )
        extra = d.pop("extra", {}) or {}
        for k, v in extra.items():
            if k not in d:
                d[k] = v
        return d


FINDING_COLUMNS: Tuple[str, ...] = (
    "anomaly_type", "severity_score", "raw_score",
    "network", "timepoint", "site_id", "subjectid",
    "variable", "variables_involved",
    "observed_value", "expected_value",
    "explanation", "method",
)


# ---------------------------------------------------------------------------
# Variable / value classification.
# ---------------------------------------------------------------------------

def is_excluded_variable(name: str) -> bool:
    if not isinstance(name, str):
        return True
    s = name.lower().strip()
    if s in {c.lower() for c in ID_COLUMNS}:
        return True
    if s == "_tp" or _EXCLUDED_VARIABLE_SUFFIX_RE.search(s):
        return True
    if _TOKEN_EXCLUSION_RE.search(s):
        return True
    # Token-like entries are handled by _TOKEN_EXCLUSION_RE so they cannot
    # collide with meaningful names such as current_country / crackle.
    token_entries = {
        "_complete", "_missing", "_count", "_other", "comment", "note",
        "description", "rack", "_box", "_tp",
    }
    return any(p in s for p in EXCLUDED_VARIABLE_SUBSTRINGS
               if p not in token_entries)


def matches_excluded_substrings(name: str, substrings: Sequence[str]) -> bool:
    """Case-insensitive substring match. True if any of ``substrings``
    appears in ``name``. Used by each detector module to apply its own
    EXTRA_EXCLUDED_SUBSTRINGS list on top of the shared exclusions above."""
    if not isinstance(name, str) or not substrings:
        return False
    s = name.lower()
    return any(p.lower() in s for p in substrings if p)


_MISSING_STRING_SET = frozenset(MISSING_STRING_CODES)


def clean_missing_string(s: pd.Series) -> pd.Series:
    """Replace missing-code strings with NaN; vectorized.

    The naive .map(lambda) version that lived here before was O(N) Python
    calls and dominated runtime on real AMPSCZ-scale data. The vectorized
    form below uses pandas' C-implemented string ops.
    """
    if s.dtype.kind == "O":
        str_s = s.astype(str).str.strip()
        miss_mask = s.isna() | str_s.isin(_MISSING_STRING_SET)
    else:
        # Numeric -> stringify the codes, compare. Cast back via where.
        str_s = s.astype(object).astype(str).str.strip()
        miss_mask = s.isna() | str_s.isin(_MISSING_STRING_SET)
    return s.where(~miss_mask, np.nan)


def to_numeric_clean(s: pd.Series) -> pd.Series:
    """Coerce to numeric and null out the AMPSCZ missing-code sentinels."""
    num = pd.to_numeric(s, errors="coerce")
    num = num.where(~num.isin(list(MISSING_NUMERIC_CODES)), np.nan)
    return num


# Sentinel CALENDAR dates (normalized to midnight) used for a post-parse
# sweep. The string strip below only catches the canonical form
# ('1909-09-09'); a datetime-suffixed ('1909-09-09 00:00:00'), 'T'-separated,
# slash, or unpadded ('1909-9-9') spelling of the SAME sentinel date would
# otherwise parse to a real year-1909 visit and fabricate a ~117-year gap.
# Matching on the parsed date instead of the source string catches every
# representation. Out-of-range sentinels (9999-*) coerce to NaT and simply
# drop out of the set -- they parse to NaT in real data too, so nothing leaks.
_SENTINEL_DATES = frozenset(
    ts.normalize()
    for ts in pd.to_datetime(list(DATE_SENTINEL_STRINGS), errors="coerce")
    if pd.notna(ts)
)


def clean_dates_series(s: pd.Series) -> pd.Series:
    """Return a tz-naive datetime64 Series with AMPSCZ date sentinels stripped.

    pd.to_datetime would happily parse '1909-09-09' as a real date and feed it
    into gap / order calculations as a 100-year-old visit. Replace the known
    sentinels with NaN, then parse. A second pass after parsing nulls any value
    that lands on a sentinel calendar date regardless of how it was spelled in
    the source (e.g. '1909-09-09 00:00:00', '09/09/1909', '1909-9-9').

    ISO date/datetime inputs are first reduced to their source calendar date;
    interview dates are calendar observations, not instants to be shifted
    across time zones. Parsing then uses ``utc=True, format="mixed"`` on pandas
    2.x and a scalar compatibility path on pandas 1.x. Two failure modes this
    guards, both of which otherwise make
    date_anomaly's coarse per-network try/except swallow EVERY date finding for
    the whole network: (a) a column that mixes UTC offsets parses to an
    OBJECT-dtype Series whose ``.dt`` accessor raises -- ``utc=True`` forces one
    tz-aware dtype instead; (b) a column that mixes naive and offset-bearing
    (or otherwise differently-formatted) strings, where a single inferred
    format silently coerces the non-matching rows to NaT -- ``format="mixed"``
    parses each element independently so no real date is dropped.
    The result is normalized to tz-naive midnight so time-of-day and offset
    differences cannot manufacture a fractional-day reversal.
    """
    if s is None:
        return pd.Series(dtype="datetime64[ns]")
    obj = s.astype(object).where(s.notna(), np.nan)
    sentinel_set = set(DATE_SENTINEL_STRINGS) | set(MISSING_STRING_CODES)
    # Compare the textual scalar representation for every dtype, not just
    # strings.  The runner reads CSVs as strings, but direct/programmatic
    # detector calls can supply numeric -3/-9/-99/999; pandas would otherwise
    # interpret those numbers as nanoseconds after 1970 and fabricate a date.
    iso_prefix = re.compile(
        r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})(?:\s|T|$)")

    def _calendar_input(value):
        text = str(value).strip()
        if text in sentinel_set:
            return np.nan
        match = iso_prefix.match(text)
        if match:
            try:
                calendar_date = pd.Timestamp(
                    year=int(match.group(1)), month=int(match.group(2)),
                    day=int(match.group(3))).date().isoformat()
            except (TypeError, ValueError):
                return value
            # This pre-parse sweep also catches sentinel timestamps whose UTC
            # conversion would cross midnight (e.g. 1903-03-03T00:30+14:00).
            return (np.nan if calendar_date in DATE_SENTINEL_STRINGS
                    else calendar_date)
        return value

    cleaned = obj.map(_calendar_input)

    # ``format='mixed'`` was introduced in pandas 2.0.  Under the older pandas
    # used by some pipeline deployments it can coerce every valid value to NaT
    # when errors='coerce', rather than raising.  Select the supported parser
    # explicitly so timeline coverage cannot silently disappear by version.
    try:
        pandas_major = int(str(pd.__version__).split(".", 1)[0])
    except (TypeError, ValueError):
        pandas_major = 2
    if pandas_major >= 2:
        parsed = pd.to_datetime(
            cleaned, errors="coerce", utc=True, format="mixed")
    else:  # pragma: no cover - exercised in the production pandas-1.x job
        # pandas 1.x lacks format='mixed' and can lock onto the first inferred
        # shape, silently coercing other valid shapes to NaT. Parse each
        # remaining scalar independently, then coerce the mapped series back to
        # one UTC-aware datetime dtype. This branch favors correctness over the
        # vectorized pandas-2.x fast path used on current environments.
        parsed = cleaned.map(
            lambda value: pd.to_datetime(
                value, errors="coerce", utc=True))
        parsed = pd.to_datetime(parsed, errors="coerce", utc=True)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    parsed = parsed.dt.normalize()
    if _SENTINEL_DATES and parsed.notna().any():
        parsed = parsed.mask(parsed.dt.normalize().isin(_SENTINEL_DATES))
    return parsed


def looks_like_date_column(name: str, sample: pd.Series) -> bool:
    """Cheap detector. Used for excluding dates from numeric checks AND for
    feeding the date anomaly detector. Errs on the side of accepting."""
    n = name.lower()
    # Date-like TOKENS only. A raw substring check classified every
    # ``*_update_*`` variable as a date because "update" contains "date";
    # those categorical fields then disappeared from all applicable checks.
    if re.search(r"(?:^|_)(?:date|dob|dt)(?:_|$)", n):
        return True
    if sample.dtype == object:
        # Exclude sentinel-shaped values from the heuristic so a column
        # whose only "dates" are 1909-09-09 sentinels doesn't get
        # promoted to date.
        sentinel_set = set(DATE_SENTINEL_STRINGS) | set(MISSING_STRING_CODES)
        raw = sample.dropna().astype(str).head(80)
        if raw.empty:
            return False
        s = raw[~raw.str.strip().isin(sentinel_set)]
        if s.empty:
            return False
        hits = s.str.match(r"^\s*\d{4}-\d{2}-\d{2}").sum()
        return hits >= max(5, int(0.5 * len(s)))
    return False


def classify_columns(df: pd.DataFrame) -> dict:
    """Return a dict of column lists by inferred type.

    Categories:
        numeric  : numeric, > 10 unique values, > 30 non-missing
        binary   : numeric or string with exactly 2 distinct non-missing values
        low_card : numeric or string with 3..15 distinct values (multiple-choice)
        text     : object dtype, > 15 distinct values, mostly non-numeric
        date     : looks like a date column (excluded from numeric/text checks)

    A column appears in at most one of {numeric, binary, low_card, text}; date
    is reported separately and overlaps are pruned.
    """
    out = {"numeric": [], "binary": [], "low_card": [], "text": [], "date": []}
    for col in df.columns:
        if is_excluded_variable(col):
            continue
        ser = df[col]
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        if looks_like_date_column(col, ser):
            out["date"].append(col)
            continue

        cleaned_str = clean_missing_string(ser.astype(object))
        non_missing = cleaned_str.dropna()
        if len(non_missing) < 30:
            continue

        num = to_numeric_clean(ser)
        n_num = int(num.notna().sum())
        n_str = int(non_missing.shape[0])
        # Numeric-dominant column?
        num_frac = n_num / max(n_str, 1)

        if num_frac >= 0.9:
            num_valid = num.dropna()
            uniq = int(num_valid.nunique())
            if uniq == 2:
                out["binary"].append(col)
            elif uniq <= 15:
                # 11..15-distinct columns can be ordinal scales (e.g. AUDIT)
                # OR true low-cardinality MC. Promote to "numeric" when the
                # value spread looks continuous: range > unique count, AND
                # values cover a non-trivial range. Keep small-range
                # (e.g. 0..3, 1..5) in low_card.
                rng = float(num_valid.max() - num_valid.min())
                if uniq >= 11 and rng >= uniq * 1.5:
                    out["numeric"].append(col)
                else:
                    out["low_card"].append(col)
            elif uniq > 10:
                out["numeric"].append(col)
        else:
            uniq = int(non_missing.nunique())
            if uniq == 2:
                out["binary"].append(col)
            elif uniq <= 15:
                out["low_card"].append(col)
            else:
                out["text"].append(col)
    return out


# ---------------------------------------------------------------------------
# Robust stats helpers.
# ---------------------------------------------------------------------------

_MAD_SCALE = 1.4826  # MAD * scale ≈ sigma for Gaussian-like data


def robust_center_scale(values: np.ndarray) -> Tuple[float, float]:
    """Return (median, MAD-scaled sigma). NaN sigma when MAD is zero.

    Earlier versions fell back to np.std when MAD was zero, which silently
    re-introduced non-robust dispersion exactly on the columns (rounded
    integers, modal-dominant ordinals) where robustness is most needed.
    We now report NaN so callers skip the column instead. Columns that
    are mostly a single value cannot be cleanly scored for outliers; that
    is the right answer, not a small fudged sigma.
    """
    v = values[~np.isnan(values)]
    if v.size < 5:
        return (float("nan"), float("nan"))
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sigma = mad * _MAD_SCALE
    if sigma <= 0:
        return (med, float("nan"))
    return (med, sigma)


# ---------------------------------------------------------------------------
# Vectorised NaN-aware pairwise Pearson correlation. Used by correlative.py
# to pre-screen every (variable, variable) pair in one BLAS pass instead of
# a Python loop with O(N log N) per-pair ranking.
# ---------------------------------------------------------------------------

def pairwise_nan_pearson(
    X: np.ndarray,
    Y: np.ndarray = None,
    min_joint_n: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pairwise Pearson correlation on data with missing values.

    ``X`` is (N, V_x); optional ``Y`` is (N, V_y). With Y=None we treat it
    as the all-pairs problem on X (returns V_x x V_x).

    For each pair (i, j) we use only the rows where BOTH columns are
    non-missing. The formula is the standard one applied to the joint set:

        n_ij      = sum_r M_x[r,i] * M_y[r,j]
        sum_x_ij  = sum over joint set of x[r,i]     = (X*M_x).T @ M_y
        sum_y_ij  = sum over joint set of y[r,j]     = M_x.T @ (Y*M_y)
        sum_xx_ij = sum over joint set of x[r,i]^2   = (X^2 * M_x).T @ M_y
        sum_yy_ij = sum over joint set of y[r,j]^2   = M_x.T @ (Y^2 * M_y)
        sum_xy_ij = sum over joint set of x*y        = (X*M_x).T @ (Y*M_y)

    Since the missing entries are pre-filled with 0 (and the mask is 0
    there), each "* M" multiplication is a no-op on the valid data and
    correctly zeroes out invalid entries. The five matrix multiplications
    above replace V_x * V_y Python loop iterations.

    Returns (correlation, joint_n). Entries where joint_n < min_joint_n
    are set to NaN. Pass ``X`` containing NaN; we'll mask + fill once.
    """
    if Y is None:
        Y = X
        symmetric = True
    else:
        symmetric = False

    Xm = np.where(np.isnan(X), 0.0, X)
    Mx = (~np.isnan(X)).astype(X.dtype, copy=False)
    Ym = np.where(np.isnan(Y), 0.0, Y)
    My = (~np.isnan(Y)).astype(Y.dtype, copy=False)

    n_ij = Mx.T @ My
    sum_x = Xm.T @ My
    sum_y = Mx.T @ Ym
    sum_xx = (Xm * Xm).T @ My
    sum_yy = Mx.T @ (Ym * Ym)
    sum_xy = Xm.T @ Ym

    with np.errstate(divide="ignore", invalid="ignore"):
        n_safe = np.where(n_ij > 0, n_ij, 1.0)
        mean_x = sum_x / n_safe
        mean_y = sum_y / n_safe
        cov = (sum_xy / n_safe) - mean_x * mean_y
        var_x = (sum_xx / n_safe) - mean_x * mean_x
        var_y = (sum_yy / n_safe) - mean_y * mean_y
        denom = np.sqrt(var_x * var_y)
        corr = cov / denom

    too_small = n_ij < min_joint_n
    corr[too_small | (denom <= 0) | ~np.isfinite(corr)] = np.nan
    if symmetric:
        # Symmetrize numerically and zero the diagonal.
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, np.nan)
    return corr, n_ij


def pairwise_nan_spearman(
    X: np.ndarray,
    Y: np.ndarray = None,
    min_joint_n: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spearman = Pearson on ranks. Ranks are computed per column over
    non-missing entries; rows with NaN stay NaN."""
    def _ranks(arr: np.ndarray) -> np.ndarray:
        ranked = pd.DataFrame(arr).rank(method="average").to_numpy(dtype=arr.dtype)
        return ranked
    Xr = _ranks(X)
    if Y is None:
        return pairwise_nan_pearson(Xr, None, min_joint_n=min_joint_n)
    Yr = _ranks(Y)
    return pairwise_nan_pearson(Xr, Yr, min_joint_n=min_joint_n)


def pairwise_nan_ols(
    X: np.ndarray,
    min_joint_n: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised OLS slopes / intercepts / joint counts for every (i, j)
    pair of columns in ``X`` (NaN-aware).

    Uses the same five matrix multiplications as ``pairwise_nan_pearson``
    so an entire V x V slope grid is computed in one BLAS pass instead
    of V^2 / 2 Python-level fits. This is the change that took our
    correlative branch from ~110 ms/pair (subsampled Theil-Sen via
    scipy) to ~microseconds/pair.

    Returns (slopes, intercepts, joint_n). slopes[i, j] is the OLS slope
    of column j regressed on column i (y = j, x = i). Diagonal entries
    are NaN, as are entries where joint_n < min_joint_n.

    Robustness note: with a strong rank-correlation pre-screen (Spearman
    |r| >= 0.5), high-leverage points cannot pass the gate, so OLS on
    the survivors is acceptable. For belt-and-suspenders robustness the
    caller can run a trimmed-OLS refit on each survivor; that costs O(N)
    per pair and is far cheaper than a per-pair Theil-Sen.
    """
    Xm = np.where(np.isnan(X), 0.0, X)
    M = (~np.isnan(X)).astype(X.dtype, copy=False)

    n_ij = M.T @ M
    sum_x = Xm.T @ M
    sum_y = M.T @ Xm
    sum_xy = Xm.T @ Xm
    sum_xx = (Xm * Xm).T @ M

    with np.errstate(divide="ignore", invalid="ignore"):
        n_safe = np.where(n_ij > 0, n_ij, 1.0)
        mean_x = sum_x / n_safe
        mean_y = sum_y / n_safe
        cov_xy = (sum_xy / n_safe) - mean_x * mean_y
        var_x = (sum_xx / n_safe) - mean_x * mean_x
        slope = cov_xy / var_x
        intercept = mean_y - slope * mean_x

    bad = (n_ij < min_joint_n) | (var_x <= 0) | ~np.isfinite(slope)
    slope[bad] = np.nan
    intercept[bad] = np.nan
    np.fill_diagonal(slope, np.nan)
    np.fill_diagonal(intercept, np.nan)
    return slope, intercept, n_ij


# ---------------------------------------------------------------------------
# Severity calibration. One mapping family for the whole package.
# ---------------------------------------------------------------------------

def calibrate(raw: float, threshold: float, extreme: float) -> float:
    """Map ``raw`` to a 0..100 severity:
        raw <= threshold  ->  0..50 (linear)
        raw == threshold  ->  50
        raw == extreme    ->  95
        raw >  extreme    ->  95..100 (asymptotic)
    """
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return 0.0
    if raw < threshold:
        # 0 at raw=0, 50 at raw=threshold.
        if threshold <= 0:
            return 50.0
        return float(max(0.0, min(50.0, 50.0 * raw / threshold)))
    if extreme <= threshold:
        return 95.0
    if raw <= extreme:
        return float(50.0 + 45.0 * (raw - threshold) / (extreme - threshold))
    # Beyond "extreme": squash the tail.
    over = raw - extreme
    return float(min(100.0, 95.0 + 5.0 * (1 - math.exp(-over / max(extreme, 1e-9)))))


def severity_from_z(z: float, threshold: float = 4.0, extreme: float = 10.0) -> float:
    return calibrate(abs(float(z)) if z is not None and np.isfinite(z) else 0.0,
                     threshold, extreme)


# ---------------------------------------------------------------------------
# Output utilities.
# ---------------------------------------------------------------------------

def short(val: Any, maxlen: int = 80) -> str:
    """Compact, reviewer-friendly string form of a value.

    Floats use ``{:.6g}`` so the report no longer shows
    "169.89999999999998"; integers and strings pass through as-is.
    """
    if val is None:
        return ""
    try:
        if isinstance(val, float):
            if np.isnan(val):
                return ""
            return f"{val:.6g}"
    except Exception:
        pass
    s = str(val).replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= maxlen else s[: maxlen - 3] + "..."


def site_from_subjectid(subjectid: str) -> str:
    if not isinstance(subjectid, str) or len(subjectid) < 2:
        return ""
    return subjectid[:2]


def coerce_to_finding_frame(rows: Iterable[dict]) -> pd.DataFrame:
    """Build a DataFrame with the canonical column order in front, others appended."""
    rows = list(rows)
    if not rows:
        return pd.DataFrame(columns=list(FINDING_COLUMNS))
    df = pd.DataFrame(rows)
    front = [c for c in FINDING_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


# ---------------------------------------------------------------------------
# Misc small helpers some detectors need.
# ---------------------------------------------------------------------------

def stack_by_network(items: Iterable) -> pd.DataFrame:
    """Concatenate a network's per-timepoint slices into one frame, tagging
    each row with its timepoint in a ``_tp`` column.

    Each item is ``(timepoint, DataFrame[, ...])``; any extra tuple elements
    (e.g. a cached ``classify_columns`` result) are ignored. Shared by the
    detectors that score on a per-network stack (``correlative``,
    ``site_network``, ``whole_subject_site``).
    """
    parts = []
    for item in items:
        tp, df = item[0], item[1]
        sub = df.copy()
        sub["_tp"] = tp
        parts.append(sub)
    return pd.concat(parts, ignore_index=True, sort=False)


_PATTERN_DIGIT_RE = re.compile(r"\d")
_PATTERN_LOWER_RE = re.compile(r"[a-z]")
_PATTERN_UPPER_RE = re.compile(r"[A-Z]")


def char_class_signature(s: str) -> str:
    """Map characters to a class signature, collapsing runs.

    Examples:
        'AB123' -> 'AD'      (one run of upper letters, one run of digits)
        'KC09001' -> 'AD'
        '2024-01-15' -> 'D-D-D'
        'pronet-baseline_v1' -> 'a-a_aD'
    """
    if not isinstance(s, str):
        return ""
    out = []
    prev = None
    for ch in s:
        if _PATTERN_DIGIT_RE.match(ch):
            cls = "D"
        elif _PATTERN_LOWER_RE.match(ch):
            cls = "a"
        elif _PATTERN_UPPER_RE.match(ch):
            cls = "A"
        else:
            cls = ch  # punctuation kept literally
        if cls != prev:
            out.append(cls)
            prev = cls
    return "".join(out)
