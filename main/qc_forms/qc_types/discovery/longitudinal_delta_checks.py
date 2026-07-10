import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_types.discovery._common import (
    DEFAULT_EXCLUDED_FORM_PATTERNS,
    DEFAULT_EXCLUDED_VARIABLE_PATTERNS,
    is_excluded_form,
    is_excluded_variable,
    atomic_write_parquet,
    normalize_zscore_severity,
)

"""
Longitudinal step-to-step delta robust-z detector
(Layer 2 / discovery, default-off).

Catches a class of error N1 (within-subject median-residual
robust-z) cannot see: a subject who drifts smoothly in one
direction across many visits then jumps abruptly at one
visit. N1 would see the late point as close to the subject's
median (it is the median once you include the drift) and
miss it; this detector ranks the visit-to-visit STEP and
flags the abrupt step.

INPUT (does NOT modify):
  dependencies/multi_tp_{network}_numeric_long.parquet
    — same input as N1 and CopyForwardChecks. Required schema:
    subjectid, network, timepoint, variable, source_form, value,
    value_numeric, is_missing_code.

OUTPUT (atomic, tmp + os.replace):
  combined_outputs/discovery/longitudinal_delta_candidates.parquet
    — single combined parquet (all networks). Tracker-compatible
    alias columns populated. v1 does NOT append to the main
    tracker.

ALGORITHM (deterministic, no ML):
  1. Filter to scoring-eligible: value_numeric.notna() &
     ~is_missing_code.
  2. Restrict to canonical longitudinal timepoints
     (Utils.create_timepoint_list()). Floating and conversion are
     EXCLUDED — event-driven, not regularly spaced; treating them
     as part of an ordered sequence would manufacture spurious
     adjacencies.
  3. For each (network, variable):
     a. Skip if post-filter rows < min_observations_per_variable
        (default 30).
     b. For each subject, sort by canonical tp index, compute
        consecutive deltas (current - previous). Subjects with
        fewer than 2 valid points contribute zero deltas.
     c. Pool all deltas for the (network, variable). Compute
        pooled-delta-MAD center & scale. Skip variable if
        scale is 0 or NaN (degenerate — no variability in
        deltas).
     d. robust_z = (delta - delta_median) / (1.4826 * MAD).
     e. severity_score = |robust_z|.
     f. Flag if severity_score >= robust_z_threshold (default 4.0).
  4. Sort by severity_score descending. Apply max_flags_to_write
     cap (mirrors N1).

OUTPUT SCHEMA (26 columns, see OUTPUT_COLUMNS):
  Audit columns (16, mirror N1):
    subjectid, network, timepoint (=current_tp), variable,
    source_form, value (=current value), value_numeric, expected
    _value (=previous value), observed_value (=current value),
    algorithm, metric_name, metric_value (signed delta robust_z),
    severity_score (|z|), threshold, error_message, dates_detected.
  Delta-specific (3):
    prev_timepoint, prev_value_numeric, delta_value (signed).
  Tracker-compatible aliases (7):
    subject, displayed_variable, displayed_timepoint,
    affected_variables, affected_timepoints (=prev_tp,current_tp),
    affected_forms, displayed_form.

FAILURE MODES (no silent fallbacks):
  - any configured network's long parquet missing → FileNotFoundError
  - long parquet exists but lacks any required column → RuntimeError
  - data filters to zero scoring rows or zero candidates → STILL
    write a valid empty parquet with declared dtypes

CONSTRAINTS HONORED:
  - Reads only the long-format parquets produced by Step 1.
  - Writes only the single candidates parquet under
    combined_outputs/discovery/.
  - Does NOT append to combined_qc_flags.parquet.
  - Does NOT inherit from FormCheck. Does NOT import
    qc_forms_main, form_check, or generate_reports.
  - Atomic write via PID-stamped tmp + os.replace.
"""


_NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})

# Module-level re-exports preserved for backward compatibility.
_DEFAULT_EXCLUDED_FORM_PATTERNS = DEFAULT_EXCLUDED_FORM_PATTERNS
_DEFAULT_EXCLUDED_VARIABLE_PATTERNS = DEFAULT_EXCLUDED_VARIABLE_PATTERNS
_is_excluded_form = is_excluded_form
_is_excluded_variable = is_excluded_variable


class LongitudinalDeltaChecks:
    """
    Discovery-tier adjacent-timepoint delta robust-z detector.
    Complements N1 by ranking visit-to-visit STEPS rather than
    deviations from the subject median.
    """

    OUTPUT_COLUMNS = [
        # Audit columns (16, mirror N1).
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric',
        'expected_value', 'observed_value',
        'algorithm', 'metric_name', 'metric_value',
        'severity_score', 'severity_normalized', 'threshold',
        'error_message', 'dates_detected',
        # Delta-specific (3).
        'prev_timepoint', 'prev_value_numeric', 'delta_value',
        # Tracker-compatible aliases (7).
        'subject', 'displayed_variable', 'displayed_timepoint',
        'affected_variables', 'affected_timepoints',
        'affected_forms', 'displayed_form',
    ]

    REQUIRED_INPUT_COLUMNS = [
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric', 'is_missing_code',
    ]

    def __init__(self, config_dict=None):
        self.utils = Utils()
        if config_dict is None:
            with open(f'{self.utils.absolute_path}/config.json',
                      'r') as f:
                config_dict = json.load(f)
        self.config_info = config_dict
        self.depen_path = self.config_info['paths']['dependencies_path']
        self.output_path = self.config_info['paths']['output_path']
        if str(self.config_info.get('testing_enabled', '')) == "True":
            self.output_path = self.output_path + "testing/"

        ld_cfg = self.config_info.get(
            'discovery', {}).get('longitudinal_delta', {})
        self.networks = list(
            ld_cfg.get('networks', ['PRONET', 'PRESCIENT']))
        self.min_observations_per_variable = int(
            ld_cfg.get('min_observations_per_variable', 30))
        # Per-(prev_tp, curr_tp) pools are smaller than per-variable pools;
        # below ~10 deltas the MAD is too noisy for a robust-z to be
        # meaningful. Tune downward if your network has sparse adjacencies.
        self.min_deltas_per_tp_pair = int(
            ld_cfg.get('min_deltas_per_tp_pair', 10))
        self.robust_z_threshold = float(
            ld_cfg.get('robust_z_threshold', 4.0))
        cap = ld_cfg.get('max_flags_to_write', None)
        self.max_flags_to_write = (
            int(cap) if cap is not None else None)
        # Default excludes digital biomarker forms (device-generated, see
        # module constant). Override with [] to include everything, or pass
        # additional substrings to extend the exclusion list.
        self.excluded_source_form_patterns = tuple(
            ld_cfg.get(
                'excluded_source_form_patterns',
                list(_DEFAULT_EXCLUDED_FORM_PATTERNS)))
        self.excluded_variable_patterns = tuple(
            ld_cfg.get(
                'excluded_variable_patterns',
                list(_DEFAULT_EXCLUDED_VARIABLE_PATTERNS)))

        self._tp_order = list(self.utils.create_timepoint_list())
        self._tp_index = {
            tp: i for i, tp in enumerate(self._tp_order)
        }

        self._counters = self._fresh_counters()

    def _fresh_counters(self) -> dict:
        return {
            'input_rows': 0,
            'rows_excluded_by_form_pattern': 0,
            'rows_excluded_by_variable_pattern': 0,
            'valid_scoring_rows': 0,
            'variables_considered': 0,
            'variables_skipped_insufficient_obs': 0,
            'tp_pair_groups_considered': 0,
            'tp_pair_groups_skipped_insufficient_obs': 0,
            'tp_pair_groups_skipped_degenerate_mad': 0,
            'deltas_above_threshold': 0,
            'deltas_written': 0,
        }

    def __call__(self):
        return self.run()

    def run(self):
        self._counters = self._fresh_counters()
        long_df = self._load_inputs()
        self._counters['input_rows'] = len(long_df)
        flagged = self._compute_candidates(long_df)
        self._counters['deltas_written'] = len(flagged)
        self._write_output(flagged)
        self._print_summary()

    def _load_inputs(self) -> pd.DataFrame:
        pieces = []
        for network in self.networks:
            path = (f"{self.depen_path}"
                    f"multi_tp_{network}_numeric_long.parquet")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"FATAL: long-format input missing for network "
                    f"{network}: {path}. Run "
                    f"process_variables.produce_multi_tp_long "
                    f"first."
                )
            df = pd.read_parquet(path)
            missing_cols = [
                c for c in self.REQUIRED_INPUT_COLUMNS
                if c not in df.columns
            ]
            if missing_cols:
                raise RuntimeError(
                    f"FATAL: long-format parquet at {path} is "
                    f"missing required columns: {missing_cols}. "
                    f"Schema mismatch with the producer — "
                    f"regenerate the long parquet."
                )
            pieces.append(df)
        if not pieces:
            return pd.DataFrame(
                columns=list(self.REQUIRED_INPUT_COLUMNS))
        return pd.concat(pieces, ignore_index=True)

    def _compute_candidates(
        self, long_df: pd.DataFrame
    ) -> pd.DataFrame:
        scoring = long_df[
            long_df['value_numeric'].notna()
            & (~long_df['is_missing_code'])
        ].copy()
        scoring = scoring[
            scoring['timepoint'].isin(self._tp_index)
            & ~scoring['timepoint'].isin(_NON_LONGITUDINAL_TIMEPOINTS)
        ].copy()

        # Drop device-generated / passive-monitoring forms (default:
        # digital_biomarkers_*). Their natural variance is several
        # orders of magnitude wider than RA-entered scales and they
        # dominate the output otherwise.
        # Skip the apply-based filters when scoring is empty: an
        # empty Series.apply returns a float64 (empty) series, which
        # pandas then treats as a column selector — collapsing the
        # frame to zero columns instead of zero rows.
        if self.excluded_source_form_patterns and len(scoring) > 0:
            before = len(scoring)
            excluded_mask = scoring['source_form'].apply(
                lambda f: _is_excluded_form(
                    f, self.excluded_source_form_patterns))
            scoring = scoring[~excluded_mask].copy()
            self._counters['rows_excluded_by_form_pattern'] = (
                before - len(scoring))

        if self.excluded_variable_patterns and len(scoring) > 0:
            before = len(scoring)
            excluded_var_mask = scoring['variable'].apply(
                lambda v: _is_excluded_variable(
                    v, self.excluded_variable_patterns))
            scoring = scoring[~excluded_var_mask].copy()
            self._counters['rows_excluded_by_variable_pattern'] = (
                before - len(scoring))

        self._counters['valid_scoring_rows'] = len(scoring)
        if len(scoring) == 0:
            return self._empty_output_df()

        scored_pieces = []
        for (_net, _var), grp in scoring.groupby(
                ['network', 'variable'], sort=False):
            self._counters['variables_considered'] += 1

            if len(grp) < self.min_observations_per_variable:
                self._counters[
                    'variables_skipped_insufficient_obs'] += 1
                continue

            deltas_df = self._build_deltas(grp)
            if len(deltas_df) == 0:
                # Variable has no within-subject adjacencies (every
                # subject has ≤1 valid point). Nothing to score.
                continue

            # Pool per (prev_tp, current_tp) so scheduled visit gaps
            # don't inflate the MAD: a screening→baseline pair has very
            # different typical delta magnitude than a m_12→m_24 pair.
            for (prev_tp, curr_tp), pair_df in deltas_df.groupby(
                    ['prev_timepoint', 'timepoint'], sort=False):
                self._counters['tp_pair_groups_considered'] += 1
                if len(pair_df) < self.min_deltas_per_tp_pair:
                    self._counters[
                        'tp_pair_groups_skipped_insufficient_obs'] += 1
                    continue
                center = float(np.median(pair_df['delta_value']))
                mad = float(np.median(
                    np.abs(pair_df['delta_value'] - center)))
                scale = mad * 1.4826
                if pd.isna(scale) or scale == 0:
                    self._counters[
                        'tp_pair_groups_skipped_degenerate_mad'] += 1
                    continue
                pair_df = pair_df.copy()
                pair_df['metric_value'] = (
                    (pair_df['delta_value'] - center) / scale)
                pair_df['severity_score'] = (
                    pair_df['metric_value'].abs())
                scored_pieces.append(pair_df)

        if not scored_pieces:
            return self._empty_output_df()
        scored = pd.concat(scored_pieces, ignore_index=True)

        flagged = scored[
            scored['severity_score'] >= self.robust_z_threshold
        ].copy()
        self._counters['deltas_above_threshold'] = len(flagged)
        if len(flagged) == 0:
            return self._empty_output_df()
        flagged = flagged.sort_values(
            'severity_score', ascending=False).reset_index(drop=True)

        if (self.max_flags_to_write is not None
                and len(flagged) > self.max_flags_to_write):
            print(
                f"[longitudinal_delta_checks] WARNING: "
                f"{len(flagged)} deltas exceeded threshold "
                f"{self.robust_z_threshold}; capping output to top "
                f"{self.max_flags_to_write} by severity_score "
                f"(config max_flags_to_write). Increase the cap or "
                f"set to null to see all candidates."
            )
            flagged = flagged.head(self.max_flags_to_write).copy()

        return self._format_output(flagged)

    def _build_deltas(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        For each subject in `group`, sort by canonical tp index
        and emit one row per consecutive pair of valid points.
        Each row represents a single step from prev_tp to
        current_tp with delta_value = current - previous.
        """
        out_rows = []
        for subj, subj_grp in group.groupby('subjectid', sort=False):
            if len(subj_grp) < 2:
                continue
            s = subj_grp.assign(
                _tp_idx=subj_grp['timepoint'].map(self._tp_index)
            ).sort_values(['_tp_idx']).reset_index(drop=True)
            tps = s['timepoint'].tolist()
            tp_idxs = s['_tp_idx'].tolist()
            vals = s['value_numeric'].tolist()
            forms = s['source_form'].tolist()
            raw = s['value'].tolist()
            net = s['network'].iloc[0]
            variable = s['variable'].iloc[0]
            for i in range(1, len(s)):
                if tp_idxs[i] != tp_idxs[i - 1] + 1:
                    continue
                out_rows.append({
                    'subjectid': subj,
                    'network': net,
                    'variable': variable,
                    'source_form': forms[i] or '',
                    'timepoint': tps[i],
                    'value': raw[i],
                    'value_numeric': float(vals[i]),
                    'prev_timepoint': tps[i - 1],
                    'prev_value_numeric': float(vals[i - 1]),
                    'delta_value': float(vals[i] - vals[i - 1]),
                })
        if not out_rows:
            return pd.DataFrame()
        return pd.DataFrame(out_rows)

    def _format_output(
        self, flagged: pd.DataFrame
    ) -> pd.DataFrame:
        today = str(datetime.today().date())
        out = pd.DataFrame()
        out['subjectid'] = flagged['subjectid'].astype(str)
        out['network'] = flagged['network'].astype(str)
        out['timepoint'] = flagged['timepoint'].astype(str)
        out['variable'] = flagged['variable'].astype(str)
        out['source_form'] = (
            flagged['source_form'].fillna('').astype(str))
        out['value'] = flagged['value'].astype(str)
        out['value_numeric'] = (
            flagged['value_numeric'].astype(float))
        out['expected_value'] = (
            flagged['prev_value_numeric'].astype(float))
        out['observed_value'] = (
            flagged['value_numeric'].astype(float))
        out['algorithm'] = 'delta_robust_z'
        out['metric_name'] = 'delta_robust_z'
        out['metric_value'] = (
            flagged['metric_value'].astype(float))
        out['severity_score'] = (
            flagged['severity_score'].astype(float))
        out['severity_normalized'] = (
            flagged['severity_score'].apply(
                normalize_zscore_severity).astype(float))
        out['threshold'] = float(self.robust_z_threshold)
        out['error_message'] = flagged.apply(
            self._build_error_message, axis=1)
        out['dates_detected'] = today
        out['prev_timepoint'] = (
            flagged['prev_timepoint'].astype(str))
        out['prev_value_numeric'] = (
            flagged['prev_value_numeric'].astype(float))
        out['delta_value'] = flagged['delta_value'].astype(float)
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = out['variable']
        out['affected_timepoints'] = (
            flagged['prev_timepoint'].astype(str)
            + ',' + flagged['timepoint'].astype(str))
        out['affected_forms'] = out['source_form']
        out['displayed_form'] = out['source_form']
        return out[self.OUTPUT_COLUMNS]

    def _build_error_message(self, row) -> str:
        return (
            f"{row['variable']}: visit-to-visit step "
            f"{row['prev_timepoint']}({row['prev_value_numeric']:.6g})"
            f" → {row['timepoint']}({row['value_numeric']:.6g}); "
            f"delta={row['delta_value']:.6g}, robust_z="
            f"{row['metric_value']:.3g} "
            f"(|z| {row['severity_score']:.3g} ≥ threshold "
            f"{self.robust_z_threshold:.3g})"
        )

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
            'prev_timepoint': pd.Series(dtype='object'),
            'prev_value_numeric': pd.Series(dtype='float64'),
            'delta_value': pd.Series(dtype='float64'),
            'subject': pd.Series(dtype='object'),
            'displayed_variable': pd.Series(dtype='object'),
            'displayed_timepoint': pd.Series(dtype='object'),
            'affected_variables': pd.Series(dtype='object'),
            'affected_timepoints': pd.Series(dtype='object'),
            'affected_forms': pd.Series(dtype='object'),
            'displayed_form': pd.Series(dtype='object'),
        })

    def _write_output(self, df: pd.DataFrame):
        out_dir = (f"{self.output_path}"
                   f"combined_outputs/discovery")
        os.makedirs(out_dir, exist_ok=True)
        final_path = (
            f"{out_dir}/longitudinal_delta_candidates.parquet")
        self._atomic_write_parquet(df, final_path)

        if len(df) == 0:
            print(
                f"[longitudinal_delta_checks] no deltas above "
                f"threshold {self.robust_z_threshold}; wrote empty "
                f"parquet (valid result) → {final_path}"
            )
        else:
            top = df.head(5)[
                ['subjectid', 'variable', 'prev_timepoint',
                 'timepoint', 'severity_score']]
            print(
                f"[longitudinal_delta_checks] {len(df)} delta "
                f"candidates above threshold "
                f"{self.robust_z_threshold} → {final_path}"
            )
            print(
                f"Top 5 by |z|:\n"
                f"{top.to_string(index=False)}"
            )

    def _print_summary(self):
        c = self._counters
        rows = [
            ("input rows", c['input_rows']),
            ("rows excluded by form pattern",
             c['rows_excluded_by_form_pattern']),
            ("rows excluded by variable pattern",
             c['rows_excluded_by_variable_pattern']),
            ("valid scoring rows",
             c['valid_scoring_rows']),
            ("variables considered (per-network)",
             c['variables_considered']),
            ("variables skipped — insufficient obs",
             c['variables_skipped_insufficient_obs']),
            ("tp-pair groups considered",
             c['tp_pair_groups_considered']),
            ("tp-pair groups skipped — insufficient deltas",
             c['tp_pair_groups_skipped_insufficient_obs']),
            ("tp-pair groups skipped — zero/NaN delta-MAD",
             c['tp_pair_groups_skipped_degenerate_mad']),
            ("deltas above threshold",
             c['deltas_above_threshold']),
            ("deltas written (post-cap)",
             c['deltas_written']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[longitudinal_delta_checks] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")

    def _atomic_write_parquet(
        self, df: pd.DataFrame, final_path: str
    ):
        atomic_write_parquet(df, final_path)


if __name__ == '__main__':
    LongitudinalDeltaChecks()()
