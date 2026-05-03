import os
import sys
import json
import pandas as pd
from datetime import datetime
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

"""
Longitudinal anomaly detection (v1).

INPUT (does NOT modify):
  dependencies/multi_tp_{network}_numeric_long.parquet
    — produced by process_variables.produce_multi_tp_long.
    Required schema: subjectid, network, timepoint, variable,
    source_form, value, value_numeric, is_missing_code.
    Schema is validated explicitly at load; a missing column raises
    RuntimeError naming the file path and the missing columns.

OUTPUT (atomic, tmp + os.replace):
  combined_outputs/longitudinal_anomalies/longitudinal_anomaly_flags.parquet
    — single combined parquet (all networks). Per requirement 16
    of the Step 2 spec, the output also carries tracker-compatible
    alias columns (subject, displayed_variable, displayed_timepoint,
    affected_variables, affected_timepoints, affected_forms,
    displayed_form) — same values, different column names — so
    that a future N2 merge into combined_qc_flags.parquet can read
    either set without renaming. v1 still does NOT append to the
    main tracker.

ALGORITHM (v1 only):
  Within-subject median residual + pooled-per-variable MAD robust z.
  No delta-z, no copy-forward suspicion, no cross-form correlation,
  no IsolationForest / LOF / PCA / sklearn — those are deferred to
  later versions.

  For each (network, variable) group:
    1. Filter to scoring-eligible rows ONLY:
         value_numeric.notna() & ~is_missing_code
       Missing-code rows and non-numeric audit rows from the long
       parquet never enter residual / MAD computations.
    2. Drop subjects with fewer than `min_points_per_subject` valid
       points (default 3 — 2 points always have median exactly
       between them, so all residuals are ±delta/2 and the metric
       collapses).
    3. Skip the variable entirely if the post-filter group has
       fewer than `min_observations_per_variable` rows
       (default 30 — MAD is too noisy below that).
    4. subject_median = median of subject's valid points.
    5. residual = value_numeric − subject_median.
    6. pooled_center = median of pooled residuals (per variable).
       scale = MAD(pooled residuals) × 1.4826.
    7. If scale == 0 or NaN: variable is degenerate (all residuals
       identical). Skip — don't produce any flags. v1 explicitly
       does NOT include an abs-residual fallback (would mix data-
       scale and z-scale thresholds confusingly).
    8. robust_z = (residual − pooled_center) / scale.
    9. severity_score = |robust_z|.
   10. Flag if severity_score ≥ `robust_z_threshold` (default 3.5).

OUTPUT SCHEMA (23 columns, see OUTPUT_COLUMNS):
  Audit columns (16):
    subjectid, network, timepoint, variable, source_form, value,
    value_numeric, expected_value (= subject_median),
    observed_value (= value_numeric), algorithm, metric_name,
    metric_value (signed robust_z), severity_score, threshold,
    error_message, dates_detected.
  Tracker-compatible aliases (7, same values, different names):
    subject (= subjectid),
    displayed_variable (= variable),
    displayed_timepoint (= timepoint),
    affected_variables (= variable),
    affected_timepoints (= timepoint),
    affected_forms (= source_form),
    displayed_form (= source_form).

FAILURE MODES (no silent fallbacks):
  - any configured network's long parquet missing → FileNotFoundError
  - long parquet exists but lacks any required column → RuntimeError
    (with file path and missing column list)
  - data filters to zero scoring rows or zero anomalies → STILL
    write a valid empty parquet with declared dtypes (per spec —
    "no anomalies found" is a valid result for this artifact)

DIAGNOSTIC SUMMARY (stdout only, no extra files):
  At end of every run, print the input-row count, the valid scoring-
  row count after filtering, the number of (network, variable)
  groups considered, the number skipped for insufficient post-filter
  observations, the number skipped for zero/NaN MAD, the number of
  anomalies above threshold, and the number actually written
  (post-cap). Operator-facing accounting only.

CONSTRAINTS HONORED:
  - Reads only the long-format parquets produced by Step 1.
  - Writes only the single anomaly-flags parquet under
    combined_outputs/longitudinal_anomalies/.
  - Does NOT append to combined_qc_flags.parquet.
  - Does NOT import qc_forms_main, form_check, or generate_reports.
  - Does NOT inherit from FormCheck.
  - Atomic write via PID-stamped tmp + os.replace.
"""


class LongitudinalAnomalyChecks:
    """
    v1 within-subject longitudinal anomaly detector. Reads long-format
    multi-TP parquets, computes within-subject robust z, writes the
    flagged anomalies to a separate per-feature artifact.
    """

    OUTPUT_COLUMNS = [
        # Audit columns (per Step 2 requirement 15).
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric',
        'expected_value', 'observed_value',
        'algorithm', 'metric_name', 'metric_value',
        'severity_score', 'threshold', 'error_message',
        'dates_detected',
        # Tracker-compatible aliases (per Step 2 requirement 2 in
        # the revisions). Same values, different names — keeps the
        # eventual N2 merge into combined_qc_flags lossless without
        # renames.
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

        lon_cfg = self.config_info.get('longitudinal_anomaly', {})
        self.networks = list(
            lon_cfg.get('networks', ['PRONET', 'PRESCIENT']))
        self.min_points_per_subject = int(
            lon_cfg.get('min_points_per_subject', 3))
        self.min_observations_per_variable = int(
            lon_cfg.get('min_observations_per_variable', 30))
        self.robust_z_threshold = float(
            lon_cfg.get('robust_z_threshold', 3.5))
        cap = lon_cfg.get('max_flags_to_write', None)
        self.max_flags_to_write = (
            int(cap) if cap is not None else None)

        self._counters = self._fresh_counters()

    def _fresh_counters(self) -> dict:
        return {
            'input_rows': 0,
            'valid_scoring_rows': 0,
            'variables_considered': 0,
            'variables_skipped_insufficient_obs': 0,
            'variables_skipped_degenerate_mad': 0,
            'anomalies_above_threshold': 0,
            'anomalies_written': 0,
        }

    def __call__(self):
        return self.run()

    def run(self):
        self._counters = self._fresh_counters()
        long_df = self._load_inputs()
        self._counters['input_rows'] = len(long_df)
        flagged = self._compute_anomalies(long_df)
        self._counters['anomalies_written'] = len(flagged)
        self._write_output(flagged)
        self._print_summary()

    def _load_inputs(self) -> pd.DataFrame:
        """
        Read the per-network long parquets produced by Step 1 and
        concatenate. Validate required columns explicitly per file —
        missing columns is a fatal schema mismatch, not something
        to paper over.
        """
        pieces = []
        for network in self.networks:
            path = (f"{self.depen_path}"
                    f"multi_tp_{network}_numeric_long.parquet")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"FATAL: long-format input missing for network "
                    f"{network}: {path}. Run "
                    f"process_variables.produce_multi_tp_long "
                    f"(Step 1 of the longitudinal pipeline) first."
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
            # No networks configured at all. Treat as empty input.
            return pd.DataFrame(
                columns=list(self.REQUIRED_INPUT_COLUMNS))
        return pd.concat(pieces, ignore_index=True)

    def _compute_anomalies(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the v1 algorithm and return final flagged rows in the
        OUTPUT_COLUMNS schema. May return an empty DataFrame (with
        declared dtypes) — _write_output handles both cases.
        """
        # Filter to scoring-eligible rows. This is the SINGLE place
        # missing-code and non-numeric audit rows are excluded from
        # the math; everything below this line is on real
        # measurements only.
        scoring = long_df[
            long_df['value_numeric'].notna()
            & (~long_df['is_missing_code'])
        ].copy()
        self._counters['valid_scoring_rows'] = len(scoring)
        if len(scoring) == 0:
            return self._empty_output_df()

        scored_pieces = []
        for (_net, _var), grp in scoring.groupby(
                ['network', 'variable'], sort=False):
            self._counters['variables_considered'] += 1
            scored, skip_reason = self._score_one_variable(grp)
            if skip_reason == 'insufficient_obs':
                self._counters[
                    'variables_skipped_insufficient_obs'] += 1
            elif skip_reason == 'degenerate_mad':
                self._counters[
                    'variables_skipped_degenerate_mad'] += 1
            elif scored is not None and len(scored) > 0:
                scored_pieces.append(scored)
        if not scored_pieces:
            return self._empty_output_df()
        scored = pd.concat(scored_pieces, ignore_index=True)

        flagged = scored[
            scored['severity_score'] >= self.robust_z_threshold
        ].copy()
        self._counters['anomalies_above_threshold'] = len(flagged)
        if len(flagged) == 0:
            return self._empty_output_df()
        flagged = flagged.sort_values(
            'severity_score', ascending=False).reset_index(drop=True)

        if (self.max_flags_to_write is not None
                and len(flagged) > self.max_flags_to_write):
            print(
                f"[longitudinal_anomaly_checks] WARNING: "
                f"{len(flagged)} anomalies exceeded threshold "
                f"{self.robust_z_threshold}; capping output to top "
                f"{self.max_flags_to_write} by severity_score "
                f"(config max_flags_to_write). Increase the cap or "
                f"set to null to see all flags."
            )
            flagged = flagged.head(self.max_flags_to_write).copy()

        return self._format_output(flagged)

    def _score_one_variable(self, group: pd.DataFrame):
        """
        Score one (network, variable) sub-group. Returns a tuple
        (df_or_none, skip_reason). skip_reason is one of:
          'insufficient_obs' — group dropped below
              min_observations_per_variable after the per-subject
              minimum filter
          'degenerate_mad'   — pooled MAD scale is 0 or NaN
          None               — successfully scored
        """
        group = group.copy()
        subject_n = group.groupby(
            'subjectid')['value_numeric'].transform('count')
        group = group[
            subject_n >= self.min_points_per_subject].copy()
        if len(group) < self.min_observations_per_variable:
            return None, 'insufficient_obs'

        group['subject_median'] = group.groupby(
            'subjectid')['value_numeric'].transform('median')
        group['residual'] = (
            group['value_numeric'] - group['subject_median'])

        center = group['residual'].median()
        mad = (group['residual'] - center).abs().median()
        scale = mad * 1.4826

        if pd.isna(scale) or scale == 0:
            return None, 'degenerate_mad'

        group['algorithm'] = 'within_subject_robust_z'
        group['metric_name'] = 'within_subject_robust_z'
        group['metric_value'] = (group['residual'] - center) / scale
        group['severity_score'] = group['metric_value'].abs()
        group['expected_value'] = group['subject_median']
        group['observed_value'] = group['value_numeric']
        return group, None

    def _format_output(self, flagged: pd.DataFrame) -> pd.DataFrame:
        """
        Project the scored rows down to the 23-column output schema:
        16 audit columns + 7 tracker-compatible alias columns.
        """
        today = str(datetime.today().date())
        out = pd.DataFrame()
        out['subjectid'] = flagged['subjectid'].astype(str)
        out['network'] = flagged['network'].astype(str)
        out['timepoint'] = flagged['timepoint'].astype(str)
        out['variable'] = flagged['variable'].astype(str)
        out['source_form'] = (
            flagged['source_form'].fillna('').astype(str))
        out['value'] = flagged['value'].astype(str)
        out['value_numeric'] = flagged['value_numeric'].astype(float)
        out['expected_value'] = (
            flagged['expected_value'].astype(float))
        out['observed_value'] = (
            flagged['observed_value'].astype(float))
        out['algorithm'] = flagged['algorithm'].astype(str)
        out['metric_name'] = flagged['metric_name'].astype(str)
        out['metric_value'] = flagged['metric_value'].astype(float)
        out['severity_score'] = (
            flagged['severity_score'].astype(float))
        out['threshold'] = float(self.robust_z_threshold)
        out['error_message'] = out.apply(
            self._build_error_message, axis=1)
        out['dates_detected'] = today
        # Tracker-compatible aliases — pure column-rename pass-throughs.
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = out['variable']
        out['affected_timepoints'] = out['timepoint']
        out['affected_forms'] = out['source_form']
        out['displayed_form'] = out['source_form']
        return out[self.OUTPUT_COLUMNS]

    def _build_error_message(self, row) -> str:
        return (
            f"{row['variable']}: within-subject robust z = "
            f"{row['metric_value']:.3g} at {row['timepoint']} "
            f"(subject median {row['expected_value']:.6g}, "
            f"observed {row['observed_value']:.6g}; "
            f"|z| {row['severity_score']:.3g} ≥ threshold "
            f"{self.robust_z_threshold:.3g})"
        )

    def _empty_output_df(self) -> pd.DataFrame:
        """
        Empty DataFrame with declared dtypes — needed because
        `pd.DataFrame(columns=[...])` produces all-object dtype,
        which breaks parquet schema consistency with the populated
        path. Same column order as OUTPUT_COLUMNS.
        """
        return pd.DataFrame({
            # Audit columns.
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
            'threshold': pd.Series(dtype='float64'),
            'error_message': pd.Series(dtype='object'),
            'dates_detected': pd.Series(dtype='object'),
            # Tracker-compatible aliases.
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
                   f"combined_outputs/longitudinal_anomalies")
        os.makedirs(out_dir, exist_ok=True)
        final_path = f"{out_dir}/longitudinal_anomaly_flags.parquet"
        self._atomic_write_parquet(df, final_path)

        if len(df) == 0:
            print(
                f"[longitudinal_anomaly_checks] no anomalies above "
                f"threshold {self.robust_z_threshold}; wrote empty "
                f"parquet (valid result) → {final_path}"
            )
        else:
            top = df.head(5)[
                ['subjectid', 'variable', 'timepoint',
                 'severity_score']]
            print(
                f"[longitudinal_anomaly_checks] {len(df)} anomalies "
                f"above threshold {self.robust_z_threshold} → "
                f"{final_path}"
            )
            print(
                f"Top 5 by severity:\n"
                f"{top.to_string(index=False)}"
            )

    def _print_summary(self):
        c = self._counters
        rows = [
            ("input rows", c['input_rows']),
            ("valid scoring rows",
             c['valid_scoring_rows']),
            ("variables considered (per-network)",
             c['variables_considered']),
            ("skipped — insufficient obs",
             c['variables_skipped_insufficient_obs']),
            ("skipped — zero/NaN MAD",
             c['variables_skipped_degenerate_mad']),
            ("anomalies above threshold",
             c['anomalies_above_threshold']),
            ("anomalies written (post-cap)",
             c['anomalies_written']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[longitudinal_anomaly_checks] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")

    def _atomic_write_parquet(
        self, df: pd.DataFrame, final_path: str
    ):
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


if __name__ == '__main__':
    LongitudinalAnomalyChecks()()
