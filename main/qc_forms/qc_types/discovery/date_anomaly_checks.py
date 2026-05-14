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
    normalize_days_severity,
)

"""
Date anomaly detector (Layer 2 / discovery, default-off).

Two semantics, one parquet:
  1. date_monotonicity — for a date variable that recurs across
     canonical timepoints, a within-subject backward step from
     prev_tp to current_tp violates the expected non-decreasing
     ordering of visit dates. Severity = |interval_days|.
  2. date_interval_z — pooled adjacent-tp inter-visit intervals
     have a typical magnitude per (network, variable). An interval
     whose robust-z (MAD-scaled) is extreme is flagged. Severity =
     |robust_z|.

INPUT (does NOT modify):
  dependencies/multi_tp_{network}_dates_long.parquet
    — produced by process_variables.produce_multi_tp_dates_long.
    Required schema: subjectid, network, timepoint, variable,
    source_form, value, value_date, is_missing_code.

OUTPUT (atomic, tmp + os.replace):
  combined_outputs/discovery/date_anomaly_candidates.parquet

ALGORITHM (deterministic, no ML):
  1. Filter scoring-eligible: value_date.notna() & ~is_missing_code.
  2. Restrict to canonical timepoints (Utils.create_timepoint_list()).
     Floating and conversion are EXCLUDED — event-driven, not
     regularly spaced; treating them as part of an ordered visit
     sequence would manufacture spurious adjacencies.
  3. For each (network, variable):
     a. Skip if post-filter rows < min_observations_per_variable
        (default 30).
     b. For each subject, sort by canonical tp index, walk
        consecutive observations. For each pair where
        tp_idx[i] == tp_idx[i-1] + 1, compute interval_days.
        If interval_days <= -min_monotonicity_days_back (default 1,
        i.e. any backward step of at least 1 day), emit a
        date_monotonicity candidate. Else, contribute the interval
        to the pooled (network, variable) distribution.
     c. After collecting pooled intervals, compute median + MAD;
        skip variable if 1.4826*MAD is 0 or NaN.
     d. For each pooled interval, robust_z = (interval - median) /
        (1.4826 * MAD). Emit date_interval_z if |z| >=
        robust_z_threshold (default 4.0).
  4. Sort by severity_score descending. Apply max_flags_to_write
     cap.

SEVERITY SCALE NOTE:
  date_monotonicity severity is |days_back| (tens-to-hundreds is
  common); date_interval_z severity is |z| (typically 4-20). The
  two algorithms share one sheet because the user wants one row
  per anomaly regardless of subtype; within-algorithm ranking is
  exact, cross-algorithm ranking is informally "more backward
  days ≈ more severe than a marginal z." Filter by `algorithm` in
  downstream review if a single-scale view is needed.

OUT OF SCOPE (v2 hooks):
  Cross-variable date ordering (e.g., consent_date should precede
  interview_date) is not handled in v1.

FAILURE MODES (no silent fallbacks):
  - any configured network's dates long parquet missing →
    FileNotFoundError
  - long parquet exists but lacks any required column → RuntimeError
  - data filters to zero scoring rows or zero candidates → STILL
    write a valid empty parquet with declared dtypes

CONSTRAINTS HONORED:
  - Reads only the dates long parquets produced by the date
    producer.
  - Writes only the single candidates parquet under
    combined_outputs/discovery/.
  - Does NOT append to combined_qc_flags.parquet.
  - Does NOT inherit from FormCheck. Does NOT import
    qc_forms_main, form_check, or generate_reports.
  - Atomic write via PID-stamped tmp + os.replace.
"""


_NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})

# date_anomaly OPERATES ON DATE VARIABLES — variable names are
# expected to contain '_date'. The shared DEFAULT_EXCLUDED_VARIABLE
# _PATTERNS includes '_date' (correct for numeric detectors), which
# would silently empty this detector's input. Override with a list
# that drops the same junk patterns EXCEPT '_date'.
_DATE_DETECTOR_DEFAULT_EXCLUDED_VARS = (
    '_complete', '_count', '_other', '_pharm_')


class DateAnomalyChecks:
    """
    Discovery-tier date monotonicity + inter-visit interval
    robust-z detector for multi-tp date variables.
    """

    OUTPUT_COLUMNS = [
        # Audit columns (16).
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric',
        'expected_value', 'observed_value',
        'algorithm', 'metric_name', 'metric_value',
        'severity_score', 'severity_normalized', 'threshold',
        'error_message', 'dates_detected',
        # Date-specific (4).
        'prev_timepoint', 'prev_date', 'current_date', 'interval_days',
        # Tracker-compatible aliases (7).
        'subject', 'displayed_variable', 'displayed_timepoint',
        'affected_variables', 'affected_timepoints',
        'affected_forms', 'displayed_form',
    ]

    REQUIRED_INPUT_COLUMNS = [
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_date', 'is_missing_code',
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

        da_cfg = self.config_info.get(
            'discovery', {}).get('date_anomaly', {})
        self.networks = list(
            da_cfg.get('networks', ['PRONET', 'PRESCIENT']))
        self.min_observations_per_variable = int(
            da_cfg.get('min_observations_per_variable', 30))
        self.robust_z_threshold = float(
            da_cfg.get('robust_z_threshold', 4.0))
        self.min_monotonicity_days_back = int(
            da_cfg.get('min_monotonicity_days_back', 1))
        # Mirror the longitudinal_delta fix: pool intervals per
        # (prev_tp, current_tp). The visit calendar varies a lot
        # across the protocol (screening→baseline is days; m12→m24
        # is a year) so a variable-wide MAD inflates the scale and
        # hides real schedule violations within one window.
        self.min_intervals_per_tp_pair = int(
            da_cfg.get('min_intervals_per_tp_pair', 10))
        cap = da_cfg.get('max_flags_to_write', None)
        self.max_flags_to_write = (
            int(cap) if cap is not None else None)
        self.excluded_source_form_patterns = tuple(
            da_cfg.get(
                'excluded_source_form_patterns',
                list(DEFAULT_EXCLUDED_FORM_PATTERNS)))
        self.excluded_variable_patterns = tuple(
            da_cfg.get(
                'excluded_variable_patterns',
                list(_DATE_DETECTOR_DEFAULT_EXCLUDED_VARS)))

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
            'monotonicity_flags': 0,
            'interval_flags': 0,
            'rows_written': 0,
        }

    def __call__(self):
        return self.run()

    def run(self):
        self._counters = self._fresh_counters()
        long_df = self._load_inputs()
        self._counters['input_rows'] = len(long_df)
        flagged = self._compute_candidates(long_df)
        self._counters['rows_written'] = len(flagged)
        self._write_output(flagged)
        self._print_summary()

    def _load_inputs(self) -> pd.DataFrame:
        pieces = []
        for network in self.networks:
            path = (f"{self.depen_path}"
                    f"multi_tp_{network}_dates_long.parquet")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"FATAL: dates long-format input missing for "
                    f"network {network}: {path}. Run "
                    f"process_variables.produce_multi_tp_dates_long "
                    f"first."
                )
            df = pd.read_parquet(path)
            missing_cols = [
                c for c in self.REQUIRED_INPUT_COLUMNS
                if c not in df.columns
            ]
            if missing_cols:
                raise RuntimeError(
                    f"FATAL: dates long-format parquet at {path} is "
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
            long_df['value_date'].notna()
            & (~long_df['is_missing_code'])
        ].copy()
        scoring = scoring[
            scoring['timepoint'].isin(self._tp_index)
            & ~scoring['timepoint'].isin(_NON_LONGITUDINAL_TIMEPOINTS)
        ].copy()

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

        mono_pieces = []
        interval_pieces = []
        for (_net, _var), grp in scoring.groupby(
                ['network', 'variable'], sort=False):
            self._counters['variables_considered'] += 1

            if len(grp) < self.min_observations_per_variable:
                self._counters[
                    'variables_skipped_insufficient_obs'] += 1
                continue

            mono_df, intervals_df = self._build_intervals(grp)
            if len(mono_df) > 0:
                mono_pieces.append(mono_df)

            if len(intervals_df) == 0:
                continue

            # Pool intervals per (prev_tp, current_tp). Without this,
            # screening→baseline (~weeks) and m12→m24 (~year) intervals
            # share one MAD; the schedule effect dwarfs real anomalies.
            for (prev_tp, curr_tp), pair_df in intervals_df.groupby(
                    ['prev_timepoint', 'timepoint'], sort=False):
                self._counters['tp_pair_groups_considered'] += 1
                if len(pair_df) < self.min_intervals_per_tp_pair:
                    self._counters[
                        'tp_pair_groups_skipped_insufficient_obs'] += 1
                    continue
                center = float(np.median(pair_df['interval_days']))
                mad = float(np.median(
                    np.abs(pair_df['interval_days'] - center)))
                scale = mad * 1.4826
                if pd.isna(scale) or scale == 0:
                    self._counters[
                        'tp_pair_groups_skipped_degenerate_mad'] += 1
                    continue
                pair_df = pair_df.copy()
                pair_df['metric_value'] = (
                    (pair_df['interval_days'] - center) / scale)
                pair_df['severity_score'] = (
                    pair_df['metric_value'].abs())
                kept = pair_df[
                    pair_df['severity_score']
                    >= self.robust_z_threshold
                ].copy()
                if len(kept) > 0:
                    interval_pieces.append(kept)

        mono_all = (
            pd.concat(mono_pieces, ignore_index=True)
            if mono_pieces else pd.DataFrame())
        interval_all = (
            pd.concat(interval_pieces, ignore_index=True)
            if interval_pieces else pd.DataFrame())
        self._counters['monotonicity_flags'] = len(mono_all)
        self._counters['interval_flags'] = len(interval_all)

        if len(mono_all) == 0 and len(interval_all) == 0:
            return self._empty_output_df()

        mono_out = (
            self._format_monotonicity(mono_all)
            if len(mono_all) > 0 else self._empty_output_df())
        interval_out = (
            self._format_interval_z(interval_all)
            if len(interval_all) > 0 else self._empty_output_df())
        flagged = pd.concat(
            [mono_out, interval_out], ignore_index=True)

        flagged = flagged.sort_values(
            'severity_score', ascending=False).reset_index(drop=True)

        if (self.max_flags_to_write is not None
                and len(flagged) > self.max_flags_to_write):
            print(
                f"[date_anomaly_checks] WARNING: "
                f"{len(flagged)} date anomalies exceeded thresholds; "
                f"capping output to top "
                f"{self.max_flags_to_write} by severity_score "
                f"(config max_flags_to_write). Increase the cap or "
                f"set to null to see all candidates."
            )
            flagged = flagged.head(self.max_flags_to_write).copy()

        return flagged[self.OUTPUT_COLUMNS]

    def _build_intervals(
        self, group: pd.DataFrame
    ) -> tuple:
        """
        For each subject in `group`, sort by canonical tp index
        and walk consecutive observations. Pairs whose tp indices
        are exactly adjacent contribute either a monotonicity
        violation (backward step > threshold) or one pooled
        interval (forward / small-negative step).
        """
        mono_rows = []
        interval_rows = []
        for subj, subj_grp in group.groupby('subjectid', sort=False):
            if len(subj_grp) < 2:
                continue
            s = subj_grp.assign(
                _tp_idx=subj_grp['timepoint'].map(self._tp_index)
            ).sort_values(['_tp_idx']).reset_index(drop=True)
            tps = s['timepoint'].tolist()
            tp_idxs = s['_tp_idx'].tolist()
            dates = s['value_date'].tolist()
            forms = s['source_form'].tolist()
            raw = s['value'].tolist()
            net = s['network'].iloc[0]
            variable = s['variable'].iloc[0]
            for i in range(1, len(s)):
                if tp_idxs[i] != tp_idxs[i - 1] + 1:
                    continue
                prev_date = pd.Timestamp(dates[i - 1])
                curr_date = pd.Timestamp(dates[i])
                interval_days = float(
                    (curr_date - prev_date).days)
                row = {
                    'subjectid': subj,
                    'network': net,
                    'variable': variable,
                    'source_form': forms[i] or '',
                    'timepoint': tps[i],
                    'value': raw[i],
                    'prev_timepoint': tps[i - 1],
                    'prev_date': prev_date.date().isoformat(),
                    'current_date': curr_date.date().isoformat(),
                    'interval_days': interval_days,
                }
                if interval_days <= -self.min_monotonicity_days_back:
                    mono_rows.append(row)
                else:
                    interval_rows.append(row)
        mono_df = (
            pd.DataFrame(mono_rows)
            if mono_rows else pd.DataFrame())
        interval_df = (
            pd.DataFrame(interval_rows)
            if interval_rows else pd.DataFrame())
        return mono_df, interval_df

    def _format_monotonicity(
        self, mono: pd.DataFrame
    ) -> pd.DataFrame:
        today = str(datetime.today().date())
        out = pd.DataFrame()
        out['subjectid'] = mono['subjectid'].astype(str)
        out['network'] = mono['network'].astype(str)
        out['timepoint'] = mono['timepoint'].astype(str)
        out['variable'] = mono['variable'].astype(str)
        out['source_form'] = (
            mono['source_form'].fillna('').astype(str))
        out['value'] = mono['value'].astype(str)
        out['value_numeric'] = float('nan')
        out['expected_value'] = float('nan')
        out['observed_value'] = float('nan')
        out['algorithm'] = 'date_monotonicity'
        out['metric_name'] = 'interval_days_signed'
        out['metric_value'] = mono['interval_days'].astype(float)
        out['severity_score'] = (
            mono['interval_days'].abs().astype(float))
        out['severity_normalized'] = (
            mono['interval_days'].apply(
                normalize_days_severity).astype(float))
        out['threshold'] = float(self.min_monotonicity_days_back)
        out['error_message'] = mono.apply(
            self._build_monotonicity_message, axis=1)
        out['dates_detected'] = today
        out['prev_timepoint'] = mono['prev_timepoint'].astype(str)
        out['prev_date'] = mono['prev_date'].astype(str)
        out['current_date'] = mono['current_date'].astype(str)
        out['interval_days'] = mono['interval_days'].astype(float)
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = out['variable']
        out['affected_timepoints'] = (
            mono['prev_timepoint'].astype(str)
            + ',' + mono['timepoint'].astype(str))
        out['affected_forms'] = out['source_form']
        out['displayed_form'] = out['source_form']
        return out[self.OUTPUT_COLUMNS]

    def _format_interval_z(
        self, intervals: pd.DataFrame
    ) -> pd.DataFrame:
        today = str(datetime.today().date())
        out = pd.DataFrame()
        out['subjectid'] = intervals['subjectid'].astype(str)
        out['network'] = intervals['network'].astype(str)
        out['timepoint'] = intervals['timepoint'].astype(str)
        out['variable'] = intervals['variable'].astype(str)
        out['source_form'] = (
            intervals['source_form'].fillna('').astype(str))
        out['value'] = intervals['value'].astype(str)
        out['value_numeric'] = float('nan')
        out['expected_value'] = float('nan')
        out['observed_value'] = float('nan')
        out['algorithm'] = 'date_interval_z'
        out['metric_name'] = 'interval_robust_z'
        out['metric_value'] = (
            intervals['metric_value'].astype(float))
        out['severity_score'] = (
            intervals['severity_score'].astype(float))
        out['severity_normalized'] = (
            intervals['severity_score'].apply(
                normalize_zscore_severity).astype(float))
        out['threshold'] = float(self.robust_z_threshold)
        out['error_message'] = intervals.apply(
            self._build_interval_message, axis=1)
        out['dates_detected'] = today
        out['prev_timepoint'] = (
            intervals['prev_timepoint'].astype(str))
        out['prev_date'] = intervals['prev_date'].astype(str)
        out['current_date'] = intervals['current_date'].astype(str)
        out['interval_days'] = (
            intervals['interval_days'].astype(float))
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = out['variable']
        out['affected_timepoints'] = (
            intervals['prev_timepoint'].astype(str)
            + ',' + intervals['timepoint'].astype(str))
        out['affected_forms'] = out['source_form']
        out['displayed_form'] = out['source_form']
        return out[self.OUTPUT_COLUMNS]

    def _build_monotonicity_message(self, row) -> str:
        return (
            f"{row['variable']}: date moved backward "
            f"{row['prev_timepoint']}({row['prev_date']}) → "
            f"{row['timepoint']}({row['current_date']}); "
            f"interval={row['interval_days']:.0f} days "
            f"(<= -{self.min_monotonicity_days_back})"
        )

    def _build_interval_message(self, row) -> str:
        return (
            f"{row['variable']}: inter-visit interval "
            f"{row['prev_timepoint']}({row['prev_date']}) → "
            f"{row['timepoint']}({row['current_date']}) = "
            f"{row['interval_days']:.0f} days, robust_z="
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
            'prev_date': pd.Series(dtype='object'),
            'current_date': pd.Series(dtype='object'),
            'interval_days': pd.Series(dtype='float64'),
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
            f"{out_dir}/date_anomaly_candidates.parquet")
        self._atomic_write_parquet(df, final_path)

        if len(df) == 0:
            print(
                f"[date_anomaly_checks] no date anomalies above "
                f"thresholds; wrote empty parquet (valid result) → "
                f"{final_path}"
            )
        else:
            top = df.head(5)[
                ['subjectid', 'variable', 'algorithm',
                 'prev_timepoint', 'timepoint', 'severity_score']]
            print(
                f"[date_anomaly_checks] {len(df)} date anomaly "
                f"candidates → {final_path}"
            )
            print(
                f"Top 5 by severity_score:\n"
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
            ("tp-pair groups skipped — insufficient intervals",
             c['tp_pair_groups_skipped_insufficient_obs']),
            ("tp-pair groups skipped — zero/NaN interval-MAD",
             c['tp_pair_groups_skipped_degenerate_mad']),
            ("date_monotonicity flags",
             c['monotonicity_flags']),
            ("date_interval_z flags",
             c['interval_flags']),
            ("rows written (post-cap)",
             c['rows_written']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[date_anomaly_checks] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")

    def _atomic_write_parquet(
        self, df: pd.DataFrame, final_path: str
    ):
        atomic_write_parquet(df, final_path)


if __name__ == '__main__':
    DateAnomalyChecks()()
