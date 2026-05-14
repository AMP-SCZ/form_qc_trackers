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
