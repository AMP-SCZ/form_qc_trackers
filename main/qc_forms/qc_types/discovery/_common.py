"""
Shared helpers for discovery-tier detectors.

Lives in qc_types/discovery/_common.py. Importers:
  - copy_forward_checks.py
  - longitudinal_delta_checks.py
  - point_spike_checks.py
  - duplicate_record_checks.py
  - pairwise_relationship_checks.py
  - internal_consistency_checks.py
  - missingness_summary_checks.py
  - date_anomaly_checks.py
  - mahalanobis_checks.py
  - subject_summary.py
  - (and qc_types/longitudinal_anomaly_checks.py for the exclusion
    helpers; it lives outside qc_types/discovery/ but imports the
    same constants for consistency.)

NOTHING in this module reads config or instantiates Utils — it is a
pure helpers module. Keeping it side-effect-free is what lets it sit
inside the qc_types/discovery/ package without breaking the existing
sys.path acrobatics that each detector does at import time.
"""

import os
import math

import numpy as np
import pandas as pd


def is_finite_numeric_mask(series):
    """
    Boolean mask: True iff the cell holds a finite number (no NaN,
    no inf, no non-numeric). Safe on empty Series, object-dtype
    Series, and Series with mixed types — coerces non-numeric to
    NaN first so the pipeline never crashes on dtype drift upstream.
    Use in place of `Series.notna() & np.isfinite(Series)` which
    raises on object-dtype.
    """
    coerced = pd.to_numeric(series, errors='coerce')
    return np.isfinite(coerced)


def _is_form_blank(form) -> bool:
    """
    Return True for empty / NaN / whitespace-only source_form values.
    Plain `if not form:` is wrong for `np.nan` (bool(NaN) == True),
    so detectors that gate their per-form loops with `if not form:`
    silently let NaN-form rows through and produce literal 'nan'
    strings in downstream output.
    """
    if form is None:
        return True
    if isinstance(form, float) and math.isnan(form):
        return True
    return not str(form).strip()


# Forms whose variables are device-generated / passive monitoring
# rather than RA-entered. Their natural between- and within-subject
# variance is several orders of magnitude wider than RA-entered
# scales and they dominate every cohort-z output otherwise.
# Substring match on source_form.
DEFAULT_EXCLUDED_FORM_PATTERNS = ('digital_biomarkers_',)

# Variable-name substrings that don't carry meaningful anomaly signal:
# completion flags, date/text-derived integers, count totals, free-text
# "other" fields, and pharm dose fields whose changes are legitimately
# spiky (prescription changes).
DEFAULT_EXCLUDED_VARIABLE_PATTERNS = (
    '_complete', '_date', '_count', '_other', '_pharm_')


def is_excluded_form(form, patterns) -> bool:
    """Substring match: True if `form` contains any pattern in
    `patterns`. Empty form or empty patterns → False."""
    if not form or not patterns:
        return False
    s = str(form)
    return any(p and p in s for p in patterns)


def is_excluded_variable(variable, patterns) -> bool:
    """Substring match: True if `variable` contains any pattern in
    `patterns`. Empty variable or empty patterns → False."""
    if not variable or not patterns:
        return False
    s = str(variable)
    return any(p and p in s for p in patterns)


def atomic_write_parquet(df, final_path: str):
    """
    Write `df` to `final_path` atomically via a PID-stamped tmp +
    os.replace. The tmp suffix is PID-stamped so two concurrent
    pipeline processes cannot collide on the same tmp filename.
    On any exception, the partial tmp is removed before re-raising.
    """
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    tmp_path = f"{final_path}.{os.getpid()}.tmp"
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


# ---------------------------------------------------------------------
# severity_normalized helpers
#
# Each detector publishes its severity on its own scale. The aggregator
# (T2.1) needs a common 0-100 scale to combine signals across detectors.
# These functions implement the per-detector normalization the detectors
# use to populate the `severity_normalized` output column.
#
# Calibration is rough: a normalized score of 100 marks "clearly worth
# investigating" while 50 marks "above threshold but plausibly noise."
# These are NOT statistically calibrated probabilities — they're a
# common sort key. Adjust the constants below if a detector dominates
# the cross-detector workbook for boring reasons.
# ---------------------------------------------------------------------

def normalize_zscore_severity(z: float) -> float:
    """For detectors whose severity is |robust-z|: cap at 100 with
    z=10 mapping to 100 and the threshold (z=3-4) mapping to ~30-40.
    """
    try:
        return float(min(100.0, max(0.0, abs(float(z)) * 10.0)))
    except (TypeError, ValueError):
        return 0.0


def is_truthy_enabled(value) -> bool:
    """
    Strict parser for `discovery.<name>.enabled` flags.

    The previous pattern `bool(cfg.get('enabled', False))` was
    permissive in a dangerous direction: `bool('false') == True`,
    so an operator who hand-edits `config.json` and types
    `"enabled": "false"` silently turns the detector ON.

    Accept only:
      - JSON boolean true
      - integer 1
      - string "true" / "True" / "TRUE" (case-insensitive after
        whitespace strip)

    Everything else (including the JSON string "false", any other
    string, JSON null, JSON object, etc.) returns False. This
    mirrors the strict `testing_enabled` validator in utils.utils.

    Centralized here so every discovery runner and the orchestrator
    use the same parser and a future tightening only touches one
    place.
    """
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def normalize_fraction_severity(frac: float) -> float:
    """For detectors whose severity is a [0, 1] match fraction (or a
    weighted version thereof on 0-100): pass-through capped at 100.
    """
    try:
        v = float(frac)
    except (TypeError, ValueError):
        return 0.0
    return float(min(100.0, max(0.0, v)))


def normalize_passthrough_severity(severity: float) -> float:
    """For detectors whose severity is already explicitly on a 0-100
    scale (copy_forward, duplicate_record): clip and return.
    """
    try:
        return float(min(100.0, max(0.0, float(severity))))
    except (TypeError, ValueError):
        return 0.0


def normalize_relative_to_threshold(
    severity: float, threshold: float,
) -> float:
    """Cross-rule normalization when raw severity is in problem
    units (BMI deviation, range deviation, etc.) and a threshold is
    declared. severity = threshold → 50; 2*threshold → 100. Used by
    internal_consistency where rule kinds disagree on scale.
    """
    try:
        s = float(severity)
        t = float(threshold)
    except (TypeError, ValueError):
        return 0.0
    if t <= 0:
        return normalize_passthrough_severity(s)
    return float(min(100.0, max(0.0, (s / t) * 50.0)))


def normalize_days_severity(days: float) -> float:
    """For date_anomaly monotonicity: convert |backward days| to a
    0-100 scale. 30 days → 10, 300 days → 100; anything beyond a
    year saturates."""
    try:
        d = abs(float(days))
    except (TypeError, ValueError):
        return 0.0
    return float(min(100.0, d / 3.0))
