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
Mahalanobis k-tuple anomaly detector
(Layer 2 / discovery, default-off).

Catches multivariate outliers that no pairwise or univariate check can
see: a subject whose values are individually plausible AND whose every
pairwise plot is plausible, but whose JOINT k-variable pattern is
inconsistent with the cohort's joint distribution.

Concrete AMPSCZ examples:
  - SIPS positive subscale items P1..P5: a subject scores 1 on every
    item except P3 where they score 6. Pairwise outliers may not fire
    because each item alone is plausible. The 5-tuple distance from
    the cohort cloud's centroid is large.
  - Demographic + cognition tuple where each value is in range but
    their joint cluster doesn't include this combination.

ALGORITHM (deterministic, no scikit-learn):
  1. Filter scoring-eligible rows: value_numeric.notna() &
     ~is_missing_code. Restrict to canonical longitudinal timepoints.
     Apply form/variable pattern exclusion.
  2. Per (network, source_form):
     a. Pivot to wide on (subjectid, timepoint) × variable; values are
        value_numeric. NaN where the variable wasn't observed.
     b. Filter to variables that have:
          - at least min_obs_per_variable non-NaN values
          - non-zero cohort std
     c. Pick the variable subset using greedy-by-mean-abs-correlation
        until we have at most max_tuple_size variables. The variables
        most correlated with the rest of the form are the tightest
        joint cluster; subjects deviating from THAT cluster carry the
        clearest multivariate signal.
     d. Restrict rows to those with ALL chosen variables non-NaN.
     e. Skip if joint sample size < min_joint_obs (default 30).
     f. Compute sample mean μ and covariance Σ. Regularize via
        ridge: Σ_reg = Σ + λ * tr(Σ)/k * I (Ledoit-Wolf-style
        small constant). Default λ = 0.01 keeps Σ invertible for
        moderately collinear variables without distorting genuine
        outliers.
     g. Σ_inv = numpy.linalg.pinv(Σ_reg) (pinv defends against
        residual near-singularity).
     h. For each subject row, D² = (x − μ)ᵀ Σ_inv (x − μ).
     i. Cohort robust-z of D² values: median + MAD * 1.4826.
     j. severity_score = |robust_z|. Flag if ≥ residual_z_threshold
        (default 4.0).
  3. Apply per-form and global caps. Sort by severity descending.

OUTPUT SCHEMA (audit + mahalanobis-specific):
  Audit (17): subjectid, network, timepoint, variable
  ('(form-multivariate)'), source_form, value (compact summary),
  value_numeric (NaN — multi-variable), expected_value (NaN),
  observed_value (NaN), algorithm ('mahalanobis_form_tuple'),
  metric_name ('mahalanobis_d_squared_robust_z'), metric_value (z),
  severity_score (|z|), severity_normalized, threshold,
  error_message, dates_detected.
  Mahalanobis-specific (4): tuple_variables (comma-joined),
  mahalanobis_d_squared, tuple_size, n_joint_observed.
  Tracker aliases (7).

CONSTRAINTS HONORED:
  - Reads only the long-format parquet.
  - Writes only combined_outputs/discovery/
    mahalanobis_candidates.parquet.
  - Does NOT append to combined_qc_flags.parquet.
  - Atomic write via PID-stamped tmp + os.replace.
"""


_NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})


class MahalanobisChecks:
    """
    Discovery-tier Mahalanobis multivariate outlier detector. Picks
    the most-correlated variable subset within each form, computes
    Mahalanobis D² of every joint observation against the cohort
    centroid, and flags subjects whose D² is a robust-z outlier.
    """

    OUTPUT_COLUMNS = [
        # Audit (17).
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric',
        'expected_value', 'observed_value',
        'algorithm', 'metric_name', 'metric_value',
        'severity_score', 'severity_normalized', 'threshold',
        'error_message', 'dates_detected',
        # Mahalanobis-specific (4).
        'tuple_variables', 'mahalanobis_d_squared',
        'tuple_size', 'n_joint_observed',
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

        mh = self.config_info.get(
            'discovery', {}).get('mahalanobis', {})
        self.networks = list(
            mh.get('networks', ['PRONET', 'PRESCIENT']))
        # The tightest correlated cluster of this many variables per
        # form is what we score. Larger k catches more joint patterns
        # but needs more joint observations and a stabler Σ.
        self.max_tuple_size = int(mh.get('max_tuple_size', 5))
        self.min_tuple_size = int(mh.get('min_tuple_size', 3))
        self.min_obs_per_variable = int(
            mh.get('min_obs_per_variable', 30))
        self.min_joint_obs = int(mh.get('min_joint_obs', 30))
        # Ridge regularization fraction of trace(Σ)/k. Small values
        # preserve genuine outliers; larger values defend against
        # near-singular covariance matrices.
        self.shrinkage_lambda = float(
            mh.get('shrinkage_lambda', 0.01))
        self.residual_z_threshold = float(
            mh.get('residual_z_threshold', 4.0))
        self.excluded_source_form_patterns = tuple(
            mh.get(
                'excluded_source_form_patterns',
                list(DEFAULT_EXCLUDED_FORM_PATTERNS)))
        self.excluded_variable_patterns = tuple(
            mh.get(
                'excluded_variable_patterns',
                list(DEFAULT_EXCLUDED_VARIABLE_PATTERNS)))
        pv_cap = mh.get('max_flags_per_form', 50)
        self.max_flags_per_form = (
            int(pv_cap) if pv_cap is not None else None)
        cap = mh.get('max_flags_to_write', None)
        self.max_flags_to_write = (
            int(cap) if cap is not None else None)
        # 'sample' (default): classical sample mean / covariance.
        # 'mcd': deterministic fast-MCD — pick the subset of size
        # h_frac*n that minimizes the covariance determinant. Robust
        # to ~25% cohort contamination, at the cost of one extra
        # iteration loop. Use 'mcd' when the cohort itself is
        # suspected of containing many of the anomalies we want to
        # detect (the sample mean / cov is dragged toward them).
        self.covariance_estimator = str(
            mh.get('covariance_estimator', 'sample')).lower()
        if self.covariance_estimator not in ('sample', 'mcd'):
            raise RuntimeError(
                f"FATAL: discovery.mahalanobis."
                f"covariance_estimator must be 'sample' or 'mcd'; "
                f"got {self.covariance_estimator!r}"
            )
        self.mcd_subset_fraction = float(
            mh.get('mcd_subset_fraction', 0.75))
        self.mcd_max_iter = int(mh.get('mcd_max_iter', 20))

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
            'form_groups_skipped_singular': 0,
            'mcd_concentration_iterations_total': 0,
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
            form_records = self._evaluate_form(net, form, grp, today)
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
        """
        Pick the most-correlated variable cluster up to max_tuple_size,
        compute Mahalanobis D² per joint observation, flag tails.
        """
        wide = grp.pivot_table(
            index=['subjectid', 'timepoint'],
            columns='variable',
            values='value_numeric',
            aggfunc='first',
        )
        # Eligible variables: enough non-NaN values, non-zero std.
        eligible = []
        for var in wide.columns:
            col = wide[var]
            if col.notna().sum() < self.min_obs_per_variable:
                continue
            if float(col.std(skipna=True)) == 0.0:
                continue
            eligible.append(var)
        if len(eligible) < self.min_tuple_size:
            self._counters['form_groups_skipped_too_few_vars'] += 1
            return []

        # Pick the tightest correlated cluster greedily. Start from
        # the variable with the highest mean absolute correlation to
        # all other eligible vars; add the next-most-correlated each
        # step until we hit max_tuple_size or correlations drop.
        corr_mat = wide[eligible].corr(method='spearman').fillna(0.0)
        # mean |r| across the rest for each candidate var
        mean_abs_r = corr_mat.abs().mean(axis=1)
        seed = mean_abs_r.idxmax()
        chosen = [seed]
        while len(chosen) < self.max_tuple_size:
            remaining = [v for v in eligible if v not in chosen]
            if not remaining:
                break
            # Pick the var with highest mean |r| to the already-chosen
            mean_to_chosen = (
                corr_mat.loc[remaining, chosen].abs().mean(axis=1))
            best = mean_to_chosen.idxmax()
            chosen.append(best)
        if len(chosen) < self.min_tuple_size:
            self._counters['form_groups_skipped_too_few_vars'] += 1
            return []

        joint = wide[chosen].dropna(how='any')
        n_joint = len(joint)
        if n_joint < self.min_joint_obs:
            self._counters['form_groups_skipped_too_few_joint'] += 1
            return []

        X = joint.to_numpy(dtype=float)
        k = X.shape[1]
        if n_joint < 2:
            return []
        if self.covariance_estimator == 'mcd':
            mu, cov = self._mcd_mean_cov(X)
        else:
            mu = X.mean(axis=0)
            Xc_init = X - mu
            cov = (Xc_init.T @ Xc_init) / (n_joint - 1)
        Xc = X - mu
        # Ridge-shrunk Σ. Defends against numerical singularity from
        # nearly-collinear pairs without losing the multivariate
        # structure that defines the cohort cloud.
        trace_avg = float(np.trace(cov)) / max(k, 1)
        cov_reg = cov + self.shrinkage_lambda * trace_avg * np.eye(k)
        try:
            cov_inv = np.linalg.pinv(cov_reg)
        except np.linalg.LinAlgError:
            self._counters['form_groups_skipped_singular'] += 1
            return []
        d_squared = np.einsum('ij,jk,ik->i', Xc, cov_inv, Xc)
        # Robust-z of D² across cohort: a subject whose joint pattern
        # is far from the cluster's centroid stands out.
        center = float(np.median(d_squared))
        mad = float(np.median(np.abs(d_squared - center)))
        scale = mad * 1.4826
        if scale == 0 or np.isnan(scale):
            self._counters['form_groups_skipped_singular'] += 1
            return []
        z = (d_squared - center) / scale
        abs_z = np.abs(z)
        outlier_idx = np.where(abs_z >= self.residual_z_threshold)[0]
        if len(outlier_idx) == 0:
            return []
        self._counters['flags_above_threshold'] += len(outlier_idx)

        idx_arr = joint.index.to_numpy()
        tuple_str = ','.join(chosen)
        records = []
        for pos in outlier_idx:
            subj, tp = idx_arr[pos]
            records.append({
                'subjectid': str(subj),
                'network': str(network),
                'timepoint': str(tp),
                'variable': '(form-multivariate)',
                'source_form': str(source_form),
                'value': (
                    f"tuple={tuple_str},D2={d_squared[pos]:.3g}"),
                'value_numeric': float('nan'),
                'expected_value': float('nan'),
                'observed_value': float('nan'),
                'algorithm': 'mahalanobis_form_tuple',
                'metric_name': 'mahalanobis_d_squared_robust_z',
                'metric_value': float(z[pos]),
                'severity_score': float(abs_z[pos]),
                'threshold': float(self.residual_z_threshold),
                'error_message': (
                    f"Multivariate outlier on form {source_form}: "
                    f"subject's {len(chosen)}-tuple "
                    f"({tuple_str}) has Mahalanobis D² = "
                    f"{d_squared[pos]:.4g} vs cohort median "
                    f"{center:.4g} (robust z = {z[pos]:.3g}, "
                    f"|z| {abs_z[pos]:.3g} ≥ threshold "
                    f"{self.residual_z_threshold})"
                ),
                'dates_detected': today,
                'tuple_variables': tuple_str,
                'mahalanobis_d_squared': float(d_squared[pos]),
                'tuple_size': int(k),
                'n_joint_observed': int(n_joint),
            })
        return records

    def _mcd_mean_cov(self, X: np.ndarray) -> tuple:
        """
        Deterministic fast-MCD (Minimum Covariance Determinant)
        location & scatter estimator.

        Concentration step: from any subset S, compute (μ_S, Σ_S),
        then form a new subset by taking the h rows with smallest
        Mahalanobis distance to μ_S. Iterate until the subset stops
        changing (or hits mcd_max_iter). The minimum-determinant
        subset is the iteration-stable fixed point.

        Initial subset (deterministic): the h rows nearest the
        coordinate-wise median by Euclidean distance. This avoids
        the classical Rousseeuw–Van Driessen random-init step so
        repeated runs over the same data produce identical results
        (important for reproducible flags).

        h_frac defaults to 0.75 — robust to ~25% contamination,
        which roughly matches the worst-case outlier rate we expect
        in a cohort that contains the very anomalies we're trying
        to surface.
        """
        n, k = X.shape
        h = max(int(round(n * self.mcd_subset_fraction)), k + 1)
        h = min(h, n)
        # Deterministic init: rows closest to coordinate-wise median.
        coord_med = np.median(X, axis=0)
        init_dist = np.sqrt(((X - coord_med) ** 2).sum(axis=1))
        idx = np.argsort(init_dist)[:h]
        prev_idx_set = None
        for _ in range(self.mcd_max_iter):
            self._counters[
                'mcd_concentration_iterations_total'] += 1
            Xs = X[idx]
            mu = Xs.mean(axis=0)
            Xc = Xs - mu
            cov = (Xc.T @ Xc) / max(h - 1, 1)
            trace_avg = float(np.trace(cov)) / max(k, 1)
            cov_reg = (
                cov + self.shrinkage_lambda * trace_avg * np.eye(k))
            try:
                cov_inv = np.linalg.pinv(cov_reg)
            except np.linalg.LinAlgError:
                # Fall back to sample estimates; the caller's ridge
                # regularization will then make the system solvable.
                mu = X.mean(axis=0)
                Xc_all = X - mu
                cov = (Xc_all.T @ Xc_all) / max(n - 1, 1)
                return mu, cov
            Xc_all = X - mu
            d2 = np.einsum('ij,jk,ik->i', Xc_all, cov_inv, Xc_all)
            new_idx = np.argsort(d2)[:h]
            new_set = frozenset(int(x) for x in new_idx)
            if new_set == prev_idx_set:
                idx = new_idx
                break
            prev_idx_set = new_set
            idx = new_idx
        Xs = X[idx]
        mu = Xs.mean(axis=0)
        Xc = Xs - mu
        cov = (Xc.T @ Xc) / max(h - 1, 1)
        return mu, cov

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        for col in (
            'subjectid', 'network', 'timepoint', 'variable',
            'source_form', 'value', 'algorithm', 'metric_name',
            'error_message', 'dates_detected', 'tuple_variables',
        ):
            out[col] = df[col].astype(str)
        for col in (
            'value_numeric', 'expected_value', 'observed_value',
            'metric_value', 'severity_score', 'threshold',
            'mahalanobis_d_squared',
        ):
            out[col] = df[col].astype(float)
        out['severity_normalized'] = df['severity_score'].apply(
            normalize_zscore_severity).astype(float)
        out['tuple_size'] = df['tuple_size'].astype('int64')
        out['n_joint_observed'] = (
            df['n_joint_observed'].astype('int64'))
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = out['tuple_variables']
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
            'tuple_variables': pd.Series(dtype='object'),
            'mahalanobis_d_squared': pd.Series(dtype='float64'),
            'tuple_size': pd.Series(dtype='int64'),
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
        final_path = f"{out_dir}/mahalanobis_candidates.parquet"
        atomic_write_parquet(df, final_path)
        if len(df) == 0:
            print(
                f"[mahalanobis_checks] no multivariate outliers "
                f"found → {final_path}"
            )
        else:
            top = df.head(5)[
                ['subjectid', 'source_form', 'tuple_variables',
                 'severity_score']]
            print(
                f"[mahalanobis_checks] {len(df)} multivariate "
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
            ("form groups skipped — singular covariance",
             c['form_groups_skipped_singular']),
            ("MCD concentration iterations (total)",
             c['mcd_concentration_iterations_total']),
            ("flags above threshold (pre-cap)",
             c['flags_above_threshold']),
            ("flags dropped by per-form cap",
             c['flags_dropped_by_per_form_cap']),
            ("flags written (post-cap)", c['flags_written']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[mahalanobis_checks] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")


if __name__ == '__main__':
    MahalanobisChecks()()
