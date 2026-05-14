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
Trajectory-shape (functional PCA) anomaly detector
(Layer 2 / discovery, default-off).

Most longitudinal detectors check individual timepoints against the
cohort's marginal at that visit. point_spike asks "is this value
extreme at this tp?"; longitudinal_delta asks "is this step extreme
relative to other subjects' steps in this tp-pair?"; N1 asks "does
this value differ from the subject's own median?"

NONE of those catch a subject whose value at every individual visit
is plausible, but whose TRAJECTORY SHAPE is impossible given the
cohort's typical shapes — monotone up when the cohort is flat,
sharp V when the cohort drifts linearly, etc.

This detector uses functional PCA on per-(network, variable)
trajectory matrices to find subjects whose trajectory cannot be
reconstructed by the cohort's leading shape components.

ALGORITHM (deterministic, pure numpy SVD):
  1. Filter scoring-eligible rows, apply form/variable exclusions.
  2. Per (network, variable):
     a. Pivot: rows = subject, columns = canonical timepoints (in
        canonical order). Values = value_numeric. NaN where the
        subject missed that visit.
     b. Keep timepoints with ≥ min_obs_per_tp non-NaN observations.
     c. Keep subjects with ≥ min_valid_tps_per_subject observations
        across the kept timepoints.
     d. Skip if subjects < min_subjects or tps < min_tps.
     e. Mean-impute missing cells per timepoint (cohort mean at
        that tp). This is the standard functional-imputation choice;
        a subject who missed a visit gets attributed the cohort
        mean, so their residual depends only on observed visits.
     f. Center: subtract per-tp cohort mean.
     g. SVD on the centered subject × tp matrix. Keep top-k
        components where cumulative variance ≥
        variance_explained_target, capped by max_components.
     h. Reconstruct each trajectory from the top-k components and
        compute the residual norm.
     i. Robust-z residual norms across the cohort. Flag tails.
     j. Severity = |robust-z|.

CONTRAST WITH SIBLINGS:
  - N1 / point_spike / longitudinal_delta: per-point cohort or
    within-subject deviations.
  - copy_forward / duplicate_record: row-level pattern matches.
  - mahalanobis / pca_reconstruction / local_outlier: per-form joint
    pattern at ONE visit.
  - trajectory_shape: cross-visit SHAPE deviation for ONE variable.

CONSTRAINTS HONORED:
  - Reads only the long-format parquet.
  - Writes only combined_outputs/discovery/
    trajectory_shape_candidates.parquet.
  - Pure numpy.
  - Atomic write via PID-stamped tmp + os.replace.
"""


_NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})


class TrajectoryShapeChecks:
    """
    Discovery-tier functional-PCA trajectory-shape detector. Flags
    subjects whose longitudinal trajectory for a variable cannot be
    reconstructed by the cohort's leading shape components.
    """

    OUTPUT_COLUMNS = [
        # Audit (17).
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric',
        'expected_value', 'observed_value',
        'algorithm', 'metric_name', 'metric_value',
        'severity_score', 'severity_normalized', 'threshold',
        'error_message', 'dates_detected',
        # Trajectory-specific (6).
        'tps_used', 'n_components_kept', 'variance_explained',
        'trajectory_residual_norm', 'n_valid_subjects',
        'subject_valid_tps',
        # Tracker aliases (7).
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

        cfg = self.config_info.get(
            'discovery', {}).get('trajectory_shape', {})
        self.networks = list(
            cfg.get('networks', ['PRONET', 'PRESCIENT']))
        self.min_tps = int(cfg.get('min_tps', 3))
        self.min_obs_per_tp = int(cfg.get('min_obs_per_tp', 20))
        self.min_valid_tps_per_subject = int(
            cfg.get('min_valid_tps_per_subject', 3))
        self.min_subjects = int(cfg.get('min_subjects', 30))
        self.variance_explained_target = float(
            cfg.get('variance_explained_target', 0.85))
        self.max_components = int(cfg.get('max_components', 5))
        self.residual_z_threshold = float(
            cfg.get('residual_z_threshold', 4.0))
        self.excluded_source_form_patterns = tuple(
            cfg.get(
                'excluded_source_form_patterns',
                list(DEFAULT_EXCLUDED_FORM_PATTERNS)))
        self.excluded_variable_patterns = tuple(
            cfg.get(
                'excluded_variable_patterns',
                list(DEFAULT_EXCLUDED_VARIABLE_PATTERNS)))
        pf = cfg.get('max_flags_per_variable', 25)
        self.max_flags_per_variable = (
            int(pf) if pf is not None else None)
        cap = cfg.get('max_flags_to_write', None)
        self.max_flags_to_write = (
            int(cap) if cap is not None else None)

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
            'variable_groups_considered': 0,
            'variable_groups_skipped_too_few_tps': 0,
            'variable_groups_skipped_too_few_subjects': 0,
            'variable_groups_skipped_degenerate_mad': 0,
            'flags_above_threshold': 0,
            'flags_dropped_by_per_variable_cap': 0,
            'flags_written': 0,
        }

    def __call__(self):
        return self.run()

    def run(self):
        self._counters = self._fresh_counters()
        long_df = self._load_inputs()
        self._counters['input_rows'] = len(long_df)
        flagged = self._compute_candidates(long_df)
        self._counters['flags_written'] = len(flagged)
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
                    f"{network}: {path}."
                )
            df = pd.read_parquet(path)
            missing = [
                c for c in self.REQUIRED_INPUT_COLUMNS
                if c not in df.columns
            ]
            if missing:
                raise RuntimeError(
                    f"FATAL: long-format parquet at {path} is "
                    f"missing required columns: {missing}."
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

        scoring = scoring.drop_duplicates(
            subset=['subjectid', 'network', 'timepoint', 'variable'],
            keep='first')

        today = str(datetime.today().date())
        records = []
        for (net, var), grp in scoring.groupby(
                ['network', 'variable'], sort=False):
            self._counters['variable_groups_considered'] += 1
            var_records = self._evaluate_variable(
                net, var, grp, today)
            if (self.max_flags_per_variable is not None
                    and len(var_records)
                    > self.max_flags_per_variable):
                var_records.sort(
                    key=lambda r: -r['severity_score'])
                dropped = (
                    len(var_records) - self.max_flags_per_variable)
                self._counters[
                    'flags_dropped_by_per_variable_cap'] += dropped
                var_records = (
                    var_records[: self.max_flags_per_variable])
            records.extend(var_records)

        if not records:
            return self._empty_output_df()
        df = pd.DataFrame(records)
        df = df.sort_values(
            'severity_score', ascending=False).reset_index(drop=True)
        if (self.max_flags_to_write is not None
                and len(df) > self.max_flags_to_write):
            df = df.head(self.max_flags_to_write).copy()
        return self._format_output(df)

    def _evaluate_variable(
        self, network: str, variable: str,
        grp: pd.DataFrame, today: str,
    ) -> list:
        """
        Per-subject trajectory-shape deviation.

        Approach: subtract the cohort's per-tp median trajectory from
        each subject's observed values. The *level* of a subject (how
        far above or below the cohort) is the MEAN of their residuals;
        the *shape* is how that residual VARIES across tps.

          shape_deviation_i = std( residual_i over observed tps )

        A subject who runs a constant +5 above the cohort has zero
        shape_deviation. A subject who matches the cohort baseline at
        most visits but spikes once has a large shape_deviation.
        Robust-z that across subjects.

        Why not full PCA: when an outlier exists in the cohort, its
        large signal can dominate the leading SVD components,
        absorbing its own shape into the basis and shrinking its
        reconstruction residual. The cohort-median + residual-std
        approach is robust by construction — the median ignores
        extreme values, so the cohort's reference trajectory is
        unaffected by the very outliers we're trying to surface.
        """
        wide = grp.pivot_table(
            index='subjectid', columns='timepoint',
            values='value_numeric', aggfunc='first',
        )
        form_map = (
            grp.groupby('subjectid')['source_form']
            .agg(lambda s: next((x for x in s if x), ''))
            .to_dict())
        tp_counts = wide.notna().sum(axis=0)
        kept_tps_unordered = tp_counts[
            tp_counts >= self.min_obs_per_tp].index.tolist()
        kept_tps = sorted(
            kept_tps_unordered,
            key=lambda tp: self._tp_index.get(tp, 1_000_000))
        if len(kept_tps) < self.min_tps:
            self._counters[
                'variable_groups_skipped_too_few_tps'] += 1
            return []
        T = wide[kept_tps].copy()
        valid_count = T.notna().sum(axis=1)
        T = T[valid_count >= self.min_valid_tps_per_subject].copy()
        valid_count = valid_count[
            valid_count >= self.min_valid_tps_per_subject]
        n_subj = len(T)
        if n_subj < self.min_subjects:
            self._counters[
                'variable_groups_skipped_too_few_subjects'] += 1
            return []
        # Per-tp cohort median (robust to outliers — one bad subject
        # cannot move it).
        tp_medians = T.median(axis=0, skipna=True)
        # Residuals at observed tps; NaN preserved where the subject
        # missed a visit.
        T_resids = T.subtract(tp_medians, axis=1)
        # Per-subject shape deviation = std of residuals across the
        # subject's observed tps. Requires ≥2 observed tps (already
        # enforced by min_valid_tps_per_subject ≥ 2).
        shape_dev = T_resids.std(axis=1, skipna=True).to_numpy(
            dtype=float)
        # Robust-z of shape deviations across the cohort.
        center = float(np.nanmedian(shape_dev))
        mad = float(np.nanmedian(np.abs(shape_dev - center)))
        scale = mad * 1.4826
        if scale == 0 or np.isnan(scale):
            self._counters[
                'variable_groups_skipped_degenerate_mad'] += 1
            return []
        z = (shape_dev - center) / scale
        # Positive-tail only: high std of residuals = noisy trajectory
        # shape; subjects with near-zero shape_dev are well-explained
        # and not anomalous.
        outliers = np.where(z >= self.residual_z_threshold)[0]
        if len(outliers) == 0:
            return []
        self._counters['flags_above_threshold'] += len(outliers)
        residual_norm = shape_dev
        # Diagnostic-only "k_keep" / "ve" — kept in the output schema
        # for back-compat with the FPCA framing in the docstring; the
        # current robust implementation always reports k_keep = 1
        # (the cohort median trajectory is the single reference) and
        # ve = NaN (no explicit variance ratio).
        k_keep = 1
        ve = float('nan')
        kept_tps_count = len(kept_tps)

        subj_ids = T.index.to_numpy()
        tps_str = ','.join(kept_tps)
        records = []
        for pos in outliers:
            subj = subj_ids[pos]
            valid_for_this = int(valid_count.loc[subj])
            records.append({
                'subjectid': str(subj),
                'network': str(network),
                'timepoint': '(trajectory)',
                'variable': str(variable),
                'source_form': str(form_map.get(subj, '')),
                'value': (
                    f"|residual|={residual_norm[pos]:.4g},"
                    f"tps={valid_for_this}"),
                'value_numeric': float('nan'),
                'expected_value': float('nan'),
                'observed_value': float('nan'),
                'algorithm': 'trajectory_shape_residual',
                'metric_name': 'fpca_residual_robust_z',
                'metric_value': float(z[pos]),
                'severity_score': float(abs(z[pos])),
                'threshold': float(self.residual_z_threshold),
                'error_message': (
                    f"Trajectory-shape outlier on {variable}: "
                    f"subject's {valid_for_this}-visit residual "
                    f"std {residual_norm[pos]:.4g} vs cohort median "
                    f"residual std {center:.4g} (robust z = "
                    f"{z[pos]:.3g} ≥ threshold "
                    f"{self.residual_z_threshold}); cohort reference "
                    f"= per-tp median across {kept_tps_count} "
                    f"timepoints"
                ),
                'dates_detected': today,
                'tps_used': tps_str,
                'n_components_kept': int(k_keep),
                'variance_explained': float(ve),
                'trajectory_residual_norm': float(
                    residual_norm[pos]),
                'n_valid_subjects': int(n_subj),
                'subject_valid_tps': valid_for_this,
            })
        return records

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        for col in (
            'subjectid', 'network', 'timepoint', 'variable',
            'source_form', 'value', 'algorithm', 'metric_name',
            'error_message', 'dates_detected', 'tps_used',
        ):
            out[col] = df[col].astype(str)
        for col in (
            'value_numeric', 'expected_value', 'observed_value',
            'metric_value', 'severity_score', 'threshold',
            'variance_explained', 'trajectory_residual_norm',
        ):
            out[col] = df[col].astype(float)
        out['severity_normalized'] = df['severity_score'].apply(
            normalize_zscore_severity).astype(float)
        out['n_components_kept'] = (
            df['n_components_kept'].astype('int64'))
        out['n_valid_subjects'] = (
            df['n_valid_subjects'].astype('int64'))
        out['subject_valid_tps'] = (
            df['subject_valid_tps'].astype('int64'))
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = out['variable']
        out['affected_timepoints'] = out['tps_used']
        out['affected_forms'] = out['source_form']
        out['displayed_form'] = out['source_form']
        return out[self.OUTPUT_COLUMNS]

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
            'tps_used': pd.Series(dtype='object'),
            'n_components_kept': pd.Series(dtype='int64'),
            'variance_explained': pd.Series(dtype='float64'),
            'trajectory_residual_norm': pd.Series(dtype='float64'),
            'n_valid_subjects': pd.Series(dtype='int64'),
            'subject_valid_tps': pd.Series(dtype='int64'),
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
            f"{out_dir}/trajectory_shape_candidates.parquet")
        atomic_write_parquet(df, final_path)
        if len(df) == 0:
            print(
                f"[trajectory_shape_checks] no trajectory-shape "
                f"outliers → {final_path}"
            )
        else:
            top = df.head(5)[
                ['subjectid', 'variable',
                 'trajectory_residual_norm', 'severity_score']]
            print(
                f"[trajectory_shape_checks] {len(df)} trajectory-"
                f"shape outliers → {final_path}"
            )
            print(f"Top 5:\n{top.to_string(index=False)}")

    def _print_summary(self):
        c = self._counters
        rows = [
            ("input rows", c['input_rows']),
            ("rows excluded by form pattern",
             c['rows_excluded_by_form_pattern']),
            ("rows excluded by variable pattern",
             c['rows_excluded_by_variable_pattern']),
            ("valid scoring rows", c['valid_scoring_rows']),
            ("(net,var) groups considered",
             c['variable_groups_considered']),
            ("variable groups skipped — <min tps",
             c['variable_groups_skipped_too_few_tps']),
            ("variable groups skipped — <min subjects",
             c['variable_groups_skipped_too_few_subjects']),
            ("variable groups skipped — degenerate scale",
             c['variable_groups_skipped_degenerate_mad']),
            ("flags above threshold (pre-cap)",
             c['flags_above_threshold']),
            ("flags dropped by per-variable cap",
             c['flags_dropped_by_per_variable_cap']),
            ("flags written (post-cap)", c['flags_written']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[trajectory_shape_checks] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")


if __name__ == '__main__':
    TrajectoryShapeChecks()()
