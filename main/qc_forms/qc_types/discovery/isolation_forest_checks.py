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
Isolation Forest anomaly detector
(Layer 2 / discovery, default-off).

Tree-based, model-free, distribution-free joint anomaly detector.
sklearn's IsolationForest fits an ensemble of random trees that
recursively partition the feature space. Outliers — points that sit
in low-density regions — get isolated in fewer splits than typical
points, and their isolation depth becomes their anomaly score.

WHY ADD THIS WHEN WE ALREADY HAVE Mahalanobis / PCA / LOF / TS?
  - Mahalanobis assumes a single elliptical cluster (one Σ).
  - PCA reconstruction assumes a single linear subspace.
  - LOF / k-NN distance is local but uses Euclidean distance —
    can be misled by axis-scale or many irrelevant dimensions.
  - Trajectory shape is per-variable longitudinal.
  - Isolation Forest is parametric-free: makes no assumption about
    cluster shape, scales gracefully to many variables, and works
    when "anomalous" means "anywhere unusual" without specifying
    along which axis.

Cost: optional sklearn dependency (already optional elsewhere in
the project). If sklearn isn't installed, the detector skips with a
single warning rather than failing the pipeline.

ALGORITHM:
  1. Filter scoring-eligible rows, apply form/variable exclusions.
  2. Per (network, source_form):
     a. Pivot to wide on (subjectid, timepoint) × variable.
     b. Keep variables with ≥ min_obs_per_variable non-NaN values
        and non-zero std. Cap at max_vars (highest-std taken).
     c. Joint subset: rows with all kept vars non-NaN.
     d. Skip if rows < min_joint_obs or vars < min_vars.
     e. Standardize per-variable.
     f. Fit sklearn IsolationForest with n_estimators=100,
        contamination='auto', max_samples='auto',
        random_state=42 (deterministic).
     g. Compute score_samples per row. Lower = more anomalous in
        sklearn's convention.
     h. Robust-z the *negative* of the scores so a positive z
        means "more anomalous"; matches the convention of every
        other detector in this layer.
     i. Flag rows with z ≥ residual_z_threshold.

  3. Per-row contributing variables: pick the top 5 features with
     largest absolute z-score against the cohort median — these
     are the variables most likely to be driving the IF's
     decision. Surface them in the error message so the flag is
     actionable, not just "sklearn says weird."

CONSTRAINTS HONORED:
  - Reads only the long-format parquet.
  - Writes only combined_outputs/discovery/
    isolation_forest_candidates.parquet.
  - Sklearn is OPTIONAL: missing sklearn → warning + empty parquet.
  - Atomic write via PID-stamped tmp + os.replace.
"""


_NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})


class IsolationForestChecks:
    """
    Discovery-tier Isolation-Forest detector. Fits one IF per
    (network, source_form) on the joint subset of cohort
    observations, computes anomaly scores, and flags subjects in
    the tail of the score distribution.
    """

    OUTPUT_COLUMNS = [
        # Audit (17).
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric',
        'expected_value', 'observed_value',
        'algorithm', 'metric_name', 'metric_value',
        'severity_score', 'severity_normalized', 'threshold',
        'error_message', 'dates_detected',
        # IF-specific (5).
        'variables_used', 'isolation_forest_score',
        'top_contributing_variables', 'n_estimators_used',
        'n_joint_observed',
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
            'discovery', {}).get('isolation_forest', {})
        self.networks = list(
            cfg.get('networks', ['PRONET', 'PRESCIENT']))
        self.min_vars = int(cfg.get('min_vars', 3))
        self.max_vars = int(cfg.get('max_vars', 40))
        self.min_obs_per_variable = int(
            cfg.get('min_obs_per_variable', 30))
        self.min_joint_obs = int(cfg.get('min_joint_obs', 30))
        self.n_estimators = int(cfg.get('n_estimators', 100))
        self.max_samples = cfg.get('max_samples', 'auto')
        self.contamination = cfg.get('contamination', 'auto')
        self.residual_z_threshold = float(
            cfg.get('residual_z_threshold', 4.0))
        self.random_state = int(cfg.get('random_state', 42))
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
        self._sklearn_warned = False
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
            'form_groups_skipped_no_sklearn': 0,
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
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            if not self._sklearn_warned:
                print(
                    "[isolation_forest_checks] WARNING: sklearn "
                    "not installed; this detector is being skipped. "
                    "Install scikit-learn to enable, or set "
                    "discovery.isolation_forest.enabled=false to "
                    "silence."
                )
                self._sklearn_warned = True
            self._counters['form_groups_skipped_no_sklearn'] += 1
            return []

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

        X = joint.to_numpy(dtype=float)
        # Standardize per-variable. IsolationForest is scale-invariant
        # per-feature in principle (each split is on one feature) but
        # standardization keeps the contamination='auto' threshold
        # behaving the same way across variable scales.
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=1)
        sigma = np.where(sigma > 0, sigma, 1.0)
        Z = (X - mu) / sigma
        # Per-variable cohort medians + MADs for the
        # top-contributing-variables explanation later.
        cohort_medians = np.median(X, axis=0)
        cohort_mads = np.median(np.abs(X - cohort_medians), axis=0)
        cohort_scales = cohort_mads * 1.4826
        cohort_scales = np.where(
            cohort_scales > 0, cohort_scales, 1.0)

        clf = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        try:
            clf.fit(Z)
            scores = clf.score_samples(Z)
        except Exception as e:
            print(
                f"[isolation_forest_checks] WARNING: sklearn IF "
                f"failed on (network={network}, form={source_form}): "
                f"{e}; skipping this form."
            )
            return []
        # sklearn convention: more-negative scores are more anomalous.
        # Flip sign so "more positive" = "more anomalous", matching
        # this layer's robust-z convention.
        anomaly_score = -scores
        center = float(np.median(anomaly_score))
        mad = float(np.median(np.abs(anomaly_score - center)))
        scale = mad * 1.4826
        if scale == 0 or np.isnan(scale):
            self._counters[
                'form_groups_skipped_degenerate_mad'] += 1
            return []
        z = (anomaly_score - center) / scale
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
            # Per-feature deviation from cohort median, in robust-z
            # units. The top features by |z| are the most likely
            # drivers of the IF's decision.
            feat_dev = np.abs(
                (X[pos] - cohort_medians) / cohort_scales)
            top_feat_idx = np.argsort(-feat_dev)[:5]
            top_feats = [
                f"{eligible[j]}={X[pos, j]:.4g}(z={feat_dev[j]:.2g})"
                for j in top_feat_idx
            ]
            top_feats_str = '; '.join(top_feats)
            records.append({
                'subjectid': str(subj),
                'network': str(network),
                'timepoint': str(tp),
                'variable': '(form-isolation-forest)',
                'source_form': str(source_form),
                'value': (
                    f"vars={len(eligible)},"
                    f"if_score={anomaly_score[pos]:.4g}"),
                'value_numeric': float('nan'),
                'expected_value': float('nan'),
                'observed_value': float('nan'),
                'algorithm': 'isolation_forest',
                'metric_name': 'isolation_forest_anomaly_z',
                'metric_value': float(z[pos]),
                'severity_score': float(abs(z[pos])),
                'threshold': float(self.residual_z_threshold),
                'error_message': (
                    f"Isolation Forest outlier on form "
                    f"{source_form}: anomaly z = {z[pos]:.3g} "
                    f"(IF score {anomaly_score[pos]:.4g} vs cohort "
                    f"median {center:.4g}); top contributing "
                    f"variables: {top_feats_str}"
                ),
                'dates_detected': today,
                'variables_used': used_vars,
                'isolation_forest_score': float(
                    anomaly_score[pos]),
                'top_contributing_variables': top_feats_str,
                'n_estimators_used': int(self.n_estimators),
                'n_joint_observed': int(n_joint),
            })
        return records

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        for col in (
            'subjectid', 'network', 'timepoint', 'variable',
            'source_form', 'value', 'algorithm', 'metric_name',
            'error_message', 'dates_detected', 'variables_used',
            'top_contributing_variables',
        ):
            out[col] = df[col].astype(str)
        for col in (
            'value_numeric', 'expected_value', 'observed_value',
            'metric_value', 'severity_score', 'threshold',
            'isolation_forest_score',
        ):
            out[col] = df[col].astype(float)
        out['severity_normalized'] = df['severity_score'].apply(
            normalize_zscore_severity).astype(float)
        out['n_estimators_used'] = (
            df['n_estimators_used'].astype('int64'))
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
            'isolation_forest_score': pd.Series(dtype='float64'),
            'top_contributing_variables': pd.Series(dtype='object'),
            'n_estimators_used': pd.Series(dtype='int64'),
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
            f"{out_dir}/isolation_forest_candidates.parquet")
        atomic_write_parquet(df, final_path)
        if len(df) == 0:
            print(
                f"[isolation_forest_checks] no isolation-forest "
                f"outliers → {final_path}"
            )
        else:
            top = df.head(5)[
                ['subjectid', 'source_form',
                 'isolation_forest_score', 'severity_score']]
            print(
                f"[isolation_forest_checks] {len(df)} "
                f"isolation-forest outliers → {final_path}"
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
            ("form groups skipped — sklearn unavailable",
             c['form_groups_skipped_no_sklearn']),
            ("form groups skipped — degenerate scale",
             c['form_groups_skipped_degenerate_mad']),
            ("flags above threshold (pre-cap)",
             c['flags_above_threshold']),
            ("flags dropped by per-form cap",
             c['flags_dropped_by_per_form_cap']),
            ("flags written (post-cap)", c['flags_written']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[isolation_forest_checks] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")


if __name__ == '__main__':
    IsolationForestChecks()()
