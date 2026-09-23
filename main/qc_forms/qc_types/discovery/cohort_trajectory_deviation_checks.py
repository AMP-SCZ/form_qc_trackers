"""
Cohort-trajectory deviation detector (Layer 2 / discovery,
default-off).

PROBLEM:
  Imagine overlaying every subject's longitudinal line graph for a
  variable. The cohort consensus trajectory is the per-tp median.
  Points that sit far above or below the consensus at their tp are
  scientifically interesting regardless of whether the same subject
  was "in line" at other visits.

  Existing detectors that are NOT this:
  - point_spike: per-tp cohort robust z + adjacent-quiet gate. A
    subject who's consistently elevated never fires because the
    neighbor-quiet gate filters them out. This is by design for
    point_spike; it intentionally targets isolated spikes.
  - longitudinal_anomaly (N1): residual from WITHIN-subject median,
    not the cohort consensus. Catches "this point is unusual for
    this subject"; says nothing about whether the subject is at the
    cohort's typical level.
  - trajectory_shape: per-subject scalar residual std. Does not
    point at individual deviant timepoints.

ALGORITHM (deterministic, no ML):
  For each (network, variable, cohort, timepoint):
    1. Pool the subjects at this cell. Skip cells with n <
       min_observations_per_tp (default 30) — MAD is unreliable
       below that.
    2. Compute cohort_median and cohort_mad_scale = MAD * 1.4826
       at this cell. Skip if cohort_mad_scale == 0 or NaN (every
       subject has the same value here; no per-point deviation is
       definable).
    3. Per subject at this cell:
         cohort_z = (value_numeric - cohort_median)
                    / cohort_mad_scale
       severity_score = |cohort_z|.
    4. Emit one record per (subject, variable, tp) where
       severity_score >= z_threshold (default 4.0).

  Sort by severity_score desc; apply max_flags_per_variable cap and
  global max_flags_to_write cap to keep the artifact reviewer-sized.

  Cohort stratification is on by default. A CHR-typical baseline
  value would otherwise look "anomalous" against an HC-dominated
  centroid in a pooled cohort; cohort stratification avoids that
  confound. Disable via config when intentionally comparing across
  cohorts.

  Floating / conversion timepoints are excluded (discovery
  convention — the conversion-event measurement is the extreme value
  the study is built to detect, not a data-quality concern).

OUTPUT (atomic, tmp + os.replace):
  combined_outputs/discovery/cohort_trajectory_deviation.parquet

CONSTRAINTS HONORED:
  - reads only the long-format parquet
  - writes only the single artifact under combined_outputs/discovery/
  - does NOT append to combined_qc_flags.parquet
  - does NOT inherit from FormCheck
  - atomic write via PID-stamped tmp + os.replace
  - default-off via config.discovery.cohort_trajectory_deviation
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

parent_dir = "/".join(
    os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_types.discovery._common import (
    DEFAULT_EXCLUDED_FORM_PATTERNS,
    DEFAULT_EXCLUDED_VARIABLE_PATTERNS,
    is_excluded_form,
    is_excluded_variable,
    is_finite_numeric_mask,
    atomic_write_parquet,
    normalize_zscore_severity,
)


_NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})


OUTPUT_COLUMNS = [
    # Audit columns.
    'subjectid', 'network', 'timepoint', 'variable',
    'source_form', 'value', 'value_numeric',
    'expected_value', 'observed_value',
    'algorithm', 'metric_name', 'metric_value',
    'severity_score', 'severity_normalized', 'threshold',
    'error_message', 'dates_detected',
    # Detector-specific (5).
    'cohort', 'cohort_median', 'cohort_mad_scale',
    'cohort_n', 'cohort_z',
    # Evidence timepoints (per-tp comparison — single tp is the
    # only evidence used to produce this flag).
    'evidence_timepoints', 'evidence_max_timepoint',
    # Tracker-compatible aliases.
    'subject', 'displayed_variable', 'displayed_timepoint',
    'affected_variables', 'affected_timepoints',
    'affected_forms', 'displayed_form',
]


REQUIRED_INPUT_COLUMNS = [
    'subjectid', 'network', 'timepoint', 'cohort', 'variable',
    'source_form', 'value', 'value_numeric', 'is_missing_code',
]


class CohortTrajectoryDeviationChecks:
    """
    Per-(subject, tp) cohort-curve deviation detector. Default-off
    via config.discovery.cohort_trajectory_deviation.enabled.
    """

    def __init__(self, config_dict=None):
        self.utils = Utils()
        if config_dict is None:
            with open(
                f'{self.utils.absolute_path}/config.json',
                'r',
            ) as f:
                config_dict = json.load(f)
        self.config_info = config_dict
        self.depen_path = self.config_info['paths']['dependencies_path']
        self.output_path = self.config_info['paths']['output_path']
        if str(self.config_info.get('testing_enabled', '')) == "True":
            self.output_path = self.output_path + "testing/"

        cfg = self.config_info.get(
            'discovery', {}).get(
            'cohort_trajectory_deviation', {})
        self.networks = list(
            cfg.get('networks', ['PRONET', 'PRESCIENT']))
        self.min_observations_per_tp = int(
            cfg.get('min_observations_per_tp', 30))
        self.z_threshold = float(cfg.get('z_threshold', 4.0))
        self.stratify_by_cohort = bool(
            cfg.get('stratify_by_cohort', True))
        self.excluded_source_form_patterns = tuple(
            cfg.get(
                'excluded_source_form_patterns',
                list(DEFAULT_EXCLUDED_FORM_PATTERNS)))
        self.excluded_variable_patterns = tuple(
            cfg.get(
                'excluded_variable_patterns',
                list(DEFAULT_EXCLUDED_VARIABLE_PATTERNS)))
        cap = cfg.get('max_flags_to_write', None)
        self.max_flags_to_write = (
            int(cap) if cap is not None else None)
        pv_cap = cfg.get('max_flags_per_variable', None)
        self.max_flags_per_variable = (
            int(pv_cap) if pv_cap is not None else None)

        # Canonical tp order for evidence_max_timepoint metadata.
        if hasattr(self.utils, 'create_timepoint_list'):
            try:
                self._tp_order = list(
                    self.utils.create_timepoint_list())
            except Exception:
                self._tp_order = []
        else:
            self._tp_order = []
        self._tp_index = {
            tp: i for i, tp in enumerate(self._tp_order)
        }

        self._counters = self._fresh_counters()

    def _fresh_counters(self) -> dict:
        return {
            'input_rows': 0,
            'rows_excluded_missing_code': 0,
            'rows_excluded_non_longitudinal_tp': 0,
            'rows_excluded_by_form_pattern': 0,
            'rows_excluded_by_variable_pattern': 0,
            'valid_scoring_rows': 0,
            'cells_considered': 0,
            'cells_skipped_insufficient_n': 0,
            'cells_skipped_degenerate_mad': 0,
            'candidates_above_threshold': 0,
            'flags_after_caps': 0,
        }

    def __call__(self):
        return self.run()

    # ---------------------------------------------------- inputs

    def _load_inputs(self) -> pd.DataFrame:
        pieces = []
        for network in self.networks:
            path = (
                f"{self.depen_path}"
                f"multi_tp_{network}_numeric_long.parquet")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"FATAL: long-format input missing for "
                    f"network {network}: {path}. Run "
                    f"process_variables.produce_multi_tp_long first."
                )
            df = pd.read_parquet(path)
            missing = [
                c for c in REQUIRED_INPUT_COLUMNS
                if c not in df.columns
            ]
            if missing:
                raise RuntimeError(
                    f"FATAL: long-format parquet at {path} is "
                    f"missing required columns: {missing}.")
            pieces.append(df)
        if not pieces:
            return pd.DataFrame(columns=REQUIRED_INPUT_COLUMNS)
        return pd.concat(pieces, ignore_index=True)

    # ------------------------------------------------ algorithm

    def _compute_candidates(
        self, long_df: pd.DataFrame
    ) -> pd.DataFrame:
        # Standard discovery-tier filters.
        before = len(long_df)
        scoring = long_df[
            long_df['value_numeric'].notna()
            & is_finite_numeric_mask(long_df['value_numeric'])
            & (~long_df['is_missing_code'])
        ].copy()
        self._counters['rows_excluded_missing_code'] = (
            before - len(scoring))

        before = len(scoring)
        scoring = scoring[
            ~scoring['timepoint'].isin(_NON_LONGITUDINAL_TIMEPOINTS)
        ].copy()
        self._counters['rows_excluded_non_longitudinal_tp'] = (
            before - len(scoring))

        if self.excluded_source_form_patterns and len(scoring) > 0:
            before = len(scoring)
            mask = scoring['source_form'].apply(
                lambda f: is_excluded_form(
                    f, self.excluded_source_form_patterns))
            scoring = scoring[~mask].copy()
            self._counters['rows_excluded_by_form_pattern'] = (
                before - len(scoring))
        if self.excluded_variable_patterns and len(scoring) > 0:
            before = len(scoring)
            mask = scoring['variable'].apply(
                lambda v: is_excluded_variable(
                    v, self.excluded_variable_patterns))
            scoring = scoring[~mask].copy()
            self._counters['rows_excluded_by_variable_pattern'] = (
                before - len(scoring))

        self._counters['valid_scoring_rows'] = len(scoring)
        if len(scoring) == 0:
            return self._empty_output_df()

        scoring['cohort'] = (
            scoring['cohort'].fillna('').astype(str).str.lower())

        # If cohort stratification is OFF, treat every subject as in
        # one bucket so the groupby key is harmless.
        if not self.stratify_by_cohort:
            scoring['cohort'] = '__all__'

        form_map = (
            scoring.dropna(subset=['source_form'])
            .groupby('variable', sort=False)['source_form']
            .first().to_dict())

        records = []
        today = str(datetime.today().date())
        groupby_cols = ['network', 'variable', 'cohort', 'timepoint']
        for (network, variable, cohort, timepoint), grp in (
                scoring.groupby(groupby_cols, sort=False)):
            self._counters['cells_considered'] += 1
            # Collapse producer dupes deterministically.
            grp = grp.drop_duplicates(
                subset=['subjectid'], keep='first')
            n = int(len(grp))
            if n < self.min_observations_per_tp:
                self._counters[
                    'cells_skipped_insufficient_n'] += 1
                continue
            vals = grp['value_numeric'].to_numpy(dtype=float)
            cohort_median = float(np.median(vals))
            mad = float(np.median(np.abs(vals - cohort_median)))
            scale = mad * 1.4826
            if not np.isfinite(scale) or scale == 0:
                self._counters[
                    'cells_skipped_degenerate_mad'] += 1
                continue

            # Per-subject cohort_z. NaN cohort_z would short-circuit
            # the threshold compare incorrectly (NaN >= 4.0 is False
            # but we want to be explicit).
            z = (vals - cohort_median) / scale
            severity = np.abs(z)
            mask = severity >= self.z_threshold
            if not mask.any():
                continue

            source_form = form_map.get(variable, '')
            evidence_tp = str(timepoint)
            grp_arr = grp.reset_index(drop=True)
            for i in np.where(mask)[0]:
                self._counters['candidates_above_threshold'] += 1
                row = grp_arr.iloc[int(i)]
                z_signed = float(z[i])
                sev = float(severity[i])
                error_message = (
                    f"{variable} at {timepoint} "
                    f"({cohort or '(any cohort)'}): "
                    f"value={float(row['value_numeric']):.6g}, "
                    f"cohort median={cohort_median:.6g}, "
                    f"cohort_z={z_signed:+.2f}, "
                    f"|z| >= threshold {self.z_threshold:.2g}")
                records.append({
                    'subjectid': str(row['subjectid']),
                    'network': str(network),
                    'timepoint': str(timepoint),
                    'variable': str(variable),
                    'source_form': str(
                        row.get('source_form', '') or source_form
                        or ''),
                    'value': str(row.get('value', '')),
                    'value_numeric': float(row['value_numeric']),
                    'expected_value': cohort_median,
                    'observed_value': float(row['value_numeric']),
                    'algorithm': 'cohort_trajectory_deviation',
                    'metric_name': 'cohort_robust_z',
                    'metric_value': z_signed,
                    'severity_score': sev,
                    'threshold': float(self.z_threshold),
                    'error_message': error_message,
                    'dates_detected': today,
                    'cohort': str(
                        cohort if cohort != '__all__' else ''),
                    'cohort_median': cohort_median,
                    'cohort_mad_scale': scale,
                    'cohort_n': n,
                    'cohort_z': z_signed,
                    'evidence_timepoints': evidence_tp,
                    'evidence_max_timepoint': evidence_tp,
                })

        if not records:
            return self._empty_output_df()
        df = pd.DataFrame(records)
        df = df.sort_values(
            by=['severity_score', 'subjectid', 'variable',
                'timepoint'],
            ascending=[False, True, True, True]).reset_index(
            drop=True)

        if self.max_flags_per_variable is not None:
            before = len(df)
            df = (
                df.groupby('variable', sort=False)
                .head(self.max_flags_per_variable)
                .reset_index(drop=True))
            # Re-sort after the per-variable cap so the global cap
            # below takes the best by overall severity.
            df = df.sort_values(
                by=['severity_score', 'subjectid', 'variable',
                    'timepoint'],
                ascending=[False, True, True, True]).reset_index(
                drop=True)
        if (self.max_flags_to_write is not None
                and len(df) > self.max_flags_to_write):
            df = df.head(self.max_flags_to_write).copy()
        self._counters['flags_after_caps'] = len(df)
        return self._format_output(df)

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out['severity_normalized'] = out['severity_score'].apply(
            normalize_zscore_severity).astype(float)
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = out['variable']
        out['affected_timepoints'] = out['timepoint']
        out['affected_forms'] = out['source_form']
        out['displayed_form'] = out['source_form']
        return out[OUTPUT_COLUMNS]

    def _empty_output_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            'subjectid': pd.Series(dtype='object'),
            'network': pd.Series(dtype='object'),
            'timepoint': pd.Series(dtype='object'),
            'variable': pd.Series(dtype='object'),
            'source_form': pd.Series(dtype='object'),
            'value': pd.Series(dtype='object'),
            'value_numeric': pd.Series(dtype='float64'),
            'expected_value': pd.Series(dtype='float64'),
            'observed_value': pd.Series(dtype='float64'),
            'algorithm': pd.Series(dtype='object'),
            'metric_name': pd.Series(dtype='object'),
            'metric_value': pd.Series(dtype='float64'),
            'severity_score': pd.Series(dtype='float64'),
            'severity_normalized': pd.Series(dtype='float64'),
            'threshold': pd.Series(dtype='float64'),
            'error_message': pd.Series(dtype='object'),
            'dates_detected': pd.Series(dtype='object'),
            'cohort': pd.Series(dtype='object'),
            'cohort_median': pd.Series(dtype='float64'),
            'cohort_mad_scale': pd.Series(dtype='float64'),
            'cohort_n': pd.Series(dtype='int64'),
            'cohort_z': pd.Series(dtype='float64'),
            'evidence_timepoints': pd.Series(dtype='object'),
            'evidence_max_timepoint': pd.Series(dtype='object'),
            'subject': pd.Series(dtype='object'),
            'displayed_variable': pd.Series(dtype='object'),
            'displayed_timepoint': pd.Series(dtype='object'),
            'affected_variables': pd.Series(dtype='object'),
            'affected_timepoints': pd.Series(dtype='object'),
            'affected_forms': pd.Series(dtype='object'),
            'displayed_form': pd.Series(dtype='object'),
        })

    # ----------------------------------------------------- write

    def _write_output(self, df: pd.DataFrame):
        out_dir = (
            f"{self.output_path}combined_outputs/discovery")
        os.makedirs(out_dir, exist_ok=True)
        final_path = (
            f"{out_dir}/cohort_trajectory_deviation.parquet")
        atomic_write_parquet(df, final_path)
        if len(df) == 0:
            print(
                f"[cohort_trajectory_deviation] no candidates "
                f"above |z|>={self.z_threshold} → {final_path}")
        else:
            top = df.head(5)[
                ['subjectid', 'variable', 'timepoint',
                 'severity_score']]
            print(
                f"[cohort_trajectory_deviation] {len(df)} "
                f"candidates → {final_path}")
            print(f"Top 5 by severity:\n{top.to_string(index=False)}")

    def _print_summary(self):
        c = self._counters
        rows = [
            ("input rows", c['input_rows']),
            ("rows excluded missing-code/NaN",
             c['rows_excluded_missing_code']),
            ("rows excluded non-longitudinal tp",
             c['rows_excluded_non_longitudinal_tp']),
            ("rows excluded by form pattern",
             c['rows_excluded_by_form_pattern']),
            ("rows excluded by variable pattern",
             c['rows_excluded_by_variable_pattern']),
            ("valid scoring rows", c['valid_scoring_rows']),
            ("(net,var,cohort,tp) cells considered",
             c['cells_considered']),
            ("cells skipped — insufficient n",
             c['cells_skipped_insufficient_n']),
            ("cells skipped — degenerate MAD",
             c['cells_skipped_degenerate_mad']),
            ("candidates above threshold (pre-cap)",
             c['candidates_above_threshold']),
            ("flags after caps", c['flags_after_caps']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[cohort_trajectory_deviation] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")

    def run(self):
        self._counters = self._fresh_counters()
        long_df = self._load_inputs()
        self._counters['input_rows'] = len(long_df)
        df = self._compute_candidates(long_df)
        self._write_output(df)
        self._print_summary()
