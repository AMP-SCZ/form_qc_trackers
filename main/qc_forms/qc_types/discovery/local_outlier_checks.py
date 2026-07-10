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
Local outlier (k-NN distance) anomaly detector
(Layer 2 / discovery, default-off).

Mahalanobis and PCA reconstruction both assume the cohort forms a
single elliptical / linear cluster. Real clinical cohorts often
have multiple sub-populations (cohort × diagnosis × site) so the
"normal" region is multimodal and shaped. A subject can be far
from the GLOBAL centroid yet sit firmly inside a sub-cluster — or
sit between sub-clusters where no other subjects are.

This detector uses **k-NN distance** as a local density proxy.
For each row in the joint subset of a form, compute its distance
to its k-th nearest neighbor; cohort-wide robust-z those distances;
flag rows whose local density is unusually low (k-th NN distance
is unusually large).

CONTRAST WITH SIBLINGS:
  - Mahalanobis: global cluster centroid + scale.
  - PCA reconstruction: global linear subspace.
  - local_outlier: local density. Catches multimodal cohorts and
    "between-cluster" outliers, complementary to the global views.

ALGORITHM (deterministic, pure numpy):
  1. Filter scoring-eligible rows, apply form/variable exclusions.
  2. Per (network, source_form):
     a. Pivot to wide on (subjectid, timepoint) × variable.
     b. Keep variables with ≥ min_obs_per_variable non-NaN values
        and non-zero std. Cap at max_vars (highest-std taken if more).
     c. Joint subset: rows with all kept vars non-NaN.
     d. Skip if rows < min_joint_obs (default 30) or vars < min_vars.
     e. Standardize per-variable.
     f. Compute pairwise Euclidean distance matrix.
     g. For each row, take the k-th smallest distance (k = config
        n_neighbors, default 5). This is the row's k-NN distance.
     h. Robust-z those distances across the cohort: median + MAD.
     i. Flag rows with z ≥ residual_z_threshold AND row's k-NN
        distance ABOVE cohort median (sparse-region outliers only;
        we don't care about "subject is in the middle of the
        cluster", which gets small z).

NOTE: pairwise distance is O(n²) in memory; we cap form size to
keep this tractable. For n=5000 subjects × 50 vars the matrix is
200 MB float64 — manageable but bounds the practical limit.

CONSTRAINTS HONORED:
  - Reads only the long-format parquet.
  - Writes only combined_outputs/discovery/
    local_outlier_candidates.parquet.
  - Pure numpy.
  - Atomic write via PID-stamped tmp + os.replace.
"""


_NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})


class LocalOutlierChecks:
    """
    Discovery-tier k-NN distance local-outlier detector. Flags
    subjects whose joint pattern sits in a sparse region of the
    cohort's joint feature space.
    """

    OUTPUT_COLUMNS = [
        # Audit (17).
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric',
        'expected_value', 'observed_value',
        'algorithm', 'metric_name', 'metric_value',
        'severity_score', 'severity_normalized', 'threshold',
        'error_message', 'dates_detected',
        # Local-outlier-specific (5).
        'variables_used', 'n_neighbors', 'kth_nn_distance',
        'cohort_median_kth_nn', 'n_joint_observed',
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
            'discovery', {}).get('local_outlier', {})
        self.networks = list(
            cfg.get('networks', ['PRONET', 'PRESCIENT']))
        self.n_neighbors = int(cfg.get('n_neighbors', 5))
        self.min_vars = int(cfg.get('min_vars', 3))
        self.max_vars = int(cfg.get('max_vars', 30))
        self.min_obs_per_variable = int(
            cfg.get('min_obs_per_variable', 30))
        self.min_joint_obs = int(cfg.get('min_joint_obs', 30))
        self.max_form_rows = int(cfg.get('max_form_rows', 10000))
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
        pf = cfg.get('max_flags_per_form', 50)
        self.max_flags_per_form = (
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
            'form_groups_considered': 0,
            'form_groups_skipped_too_few_vars': 0,
            'form_groups_skipped_too_few_joint': 0,
            'form_groups_skipped_too_many_rows': 0,
            'form_groups_skipped_degenerate_mad': 0,
            'flags_above_threshold': 0,
            'flags_dropped_by_per_form_cap': 0,
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
        for (net, form), grp in scoring.groupby(
                ['network', 'source_form'], sort=False):
            if not form:
                continue
            self._counters['form_groups_considered'] += 1
            form_records = self._evaluate_form(
                net, form, grp, today)
            if (self.max_flags_per_form is not None
                    and len(form_records)
                    > self.max_flags_per_form):
                form_records.sort(
                    key=lambda r: -r['severity_score'])
                dropped = (
                    len(form_records) - self.max_flags_per_form)
                self._counters[
                    'flags_dropped_by_per_form_cap'] += dropped
                form_records = (
                    form_records[: self.max_flags_per_form])
            records.extend(form_records)

        if not records:
            return self._empty_output_df()
        df = pd.DataFrame(records)
        df = df.sort_values(
            'severity_score', ascending=False).reset_index(drop=True)
        if (self.max_flags_to_write is not None
                and len(df) > self.max_flags_to_write):
            df = df.head(self.max_flags_to_write).copy()
        return self._format_output(df)

    def _evaluate_form(
        self, network: str, source_form: str,
        grp: pd.DataFrame, today: str,
    ) -> list:
        wide = grp.pivot_table(
            index=['subjectid', 'timepoint'],
            columns='variable',
            values='value_numeric',
            aggfunc='first',
        )
        eligible = []
        for var in wide.columns:
            col = wide[var]
            if col.notna().sum() < self.min_obs_per_variable:
                continue
            if float(col.std(skipna=True)) == 0.0:
                continue
            eligible.append(var)
        if len(eligible) < self.min_vars:
            self._counters['form_groups_skipped_too_few_vars'] += 1
            return []
        if len(eligible) > self.max_vars:
            stds = wide[eligible].std(skipna=True).sort_values(
                ascending=False)
            eligible = stds.head(self.max_vars).index.tolist()
        joint = wide[eligible].dropna(how='any')
        n_joint = len(joint)
        if n_joint < self.min_joint_obs:
            self._counters['form_groups_skipped_too_few_joint'] += 1
            return []
        if n_joint > self.max_form_rows:
            # Bound O(n²) memory. If the cohort is too large for a
            # full pairwise matrix, the operator can lift this with
            # max_form_rows or implement a kd-tree variant.
            self._counters['form_groups_skipped_too_many_rows'] += 1
            return []
        if n_joint <= self.n_neighbors:
            self._counters['form_groups_skipped_too_few_joint'] += 1
            return []

        X = joint.to_numpy(dtype=float)
        # Standardize per-variable so equal weight in Euclidean
        # distance regardless of native scale.
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=1)
        sigma = np.where(sigma > 0, sigma, 1.0)
        Z = (X - mu) / sigma
        # Pairwise squared Euclidean distance via the ‖a−b‖² =
        # ‖a‖² + ‖b‖² − 2·a·b identity (faster + lower memory than
        # broadcasting subtraction for these matrix sizes).
        sq_norms = (Z ** 2).sum(axis=1)
        # gram = Z @ Z.T → (n, n); sq_dist[i,j] = sq_norms[i] +
        # sq_norms[j] − 2·gram[i,j]. Clip negatives from floating
        # error.
        gram = Z @ Z.T
        sq_dist = (
            sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram)
        np.fill_diagonal(sq_dist, np.inf)  # exclude self-distance
        sq_dist = np.maximum(sq_dist, 0.0)
        dist = np.sqrt(sq_dist)
        # k-th NN distance per row (k = n_neighbors). np.partition
        # places the kth-smallest at index (k-1) without sorting.
        kth = np.partition(dist, self.n_neighbors - 1, axis=1)[
            :, self.n_neighbors - 1]
        center = float(np.median(kth))
        mad = float(np.median(np.abs(kth - center)))
        scale = mad * 1.4826
        if scale == 0 or np.isnan(scale):
            self._counters[
                'form_groups_skipped_degenerate_mad'] += 1
            return []
        z = (kth - center) / scale
        # Flag only the upper tail — sparse-region outliers. A
        # subject sitting in a dense pocket has small kth-NN
        # distance (z<<0) but is not anomalous; it's typical.
        outliers = np.where(z >= self.residual_z_threshold)[0]
        if len(outliers) == 0:
            return []
        self._counters['flags_above_threshold'] += len(outliers)

        idx_arr = joint.index.to_numpy()
        used_vars = ','.join(eligible[:15]) + (
            ',...' if len(eligible) > 15 else '')
        records = []
        for pos in outliers:
            subj, tp = idx_arr[pos]
            records.append({
                'subjectid': str(subj),
                'network': str(network),
                'timepoint': str(tp),
                'variable': '(form-local-density)',
                'source_form': str(source_form),
                'value': (
                    f"vars={len(eligible)},k={self.n_neighbors},"
                    f"kth_d={kth[pos]:.3g}"),
                'value_numeric': float('nan'),
                'expected_value': float('nan'),
                'observed_value': float('nan'),
                'algorithm': 'knn_distance_local_outlier',
                'metric_name': 'kth_nn_distance_robust_z',
                'metric_value': float(z[pos]),
                'severity_score': float(abs(z[pos])),
                'threshold': float(self.residual_z_threshold),
                'error_message': (
                    f"Local-density outlier on form "
                    f"{source_form}: subject's k={self.n_neighbors} "
                    f"NN distance {kth[pos]:.4g} (cohort median "
                    f"{center:.4g}, robust z = {z[pos]:.3g}, "
                    f"|z| {abs(z[pos]):.3g} ≥ threshold "
                    f"{self.residual_z_threshold}) — sits in a "
                    f"sparse region of the joint feature space "
                    f"defined by {len(eligible)} variables"
                ),
                'dates_detected': today,
                'variables_used': used_vars,
                'n_neighbors': int(self.n_neighbors),
                'kth_nn_distance': float(kth[pos]),
                'cohort_median_kth_nn': float(center),
                'n_joint_observed': int(n_joint),
            })
        return records

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        for col in (
            'subjectid', 'network', 'timepoint', 'variable',
            'source_form', 'value', 'algorithm', 'metric_name',
            'error_message', 'dates_detected', 'variables_used',
        ):
            out[col] = df[col].astype(str)
        for col in (
            'value_numeric', 'expected_value', 'observed_value',
            'metric_value', 'severity_score', 'threshold',
            'kth_nn_distance', 'cohort_median_kth_nn',
        ):
            out[col] = df[col].astype(float)
        out['severity_normalized'] = df['severity_score'].apply(
            normalize_zscore_severity).astype(float)
        out['n_neighbors'] = df['n_neighbors'].astype('int64')
        out['n_joint_observed'] = (
            df['n_joint_observed'].astype('int64'))
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = out['variables_used']
        out['affected_timepoints'] = out['timepoint']
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
            'variables_used': pd.Series(dtype='object'),
            'n_neighbors': pd.Series(dtype='int64'),
            'kth_nn_distance': pd.Series(dtype='float64'),
            'cohort_median_kth_nn': pd.Series(dtype='float64'),
            'n_joint_observed': pd.Series(dtype='int64'),
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
            f"{out_dir}/local_outlier_candidates.parquet")
        atomic_write_parquet(df, final_path)
        if len(df) == 0:
            print(
                f"[local_outlier_checks] no local-density outliers "
                f"→ {final_path}"
            )
        else:
            top = df.head(5)[
                ['subjectid', 'source_form',
                 'kth_nn_distance', 'severity_score']]
            print(
                f"[local_outlier_checks] {len(df)} local-density "
                f"outliers → {final_path}"
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
            ("(net,form) groups considered",
             c['form_groups_considered']),
            ("form groups skipped — <min eligible vars",
             c['form_groups_skipped_too_few_vars']),
            ("form groups skipped — <min joint obs",
             c['form_groups_skipped_too_few_joint']),
            ("form groups skipped — >max_form_rows",
             c['form_groups_skipped_too_many_rows']),
            ("form groups skipped — degenerate scale",
             c['form_groups_skipped_degenerate_mad']),
            ("flags above threshold (pre-cap)",
             c['flags_above_threshold']),
            ("flags dropped by per-form cap",
             c['flags_dropped_by_per_form_cap']),
            ("flags written (post-cap)", c['flags_written']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[local_outlier_checks] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")


if __name__ == '__main__':
    LocalOutlierChecks()()
