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
Pairwise-relationship anomaly detector (Layer 2 / discovery,
default-off).

Catches a class of error that NO single-variable detector can see:
when two variables are strongly correlated across the cohort, a
subject whose (var_a, var_b) value pair lies far off the cohort's
linear-fit line is anomalous — even if neither var_a nor var_b
alone is an outlier. Typical AMPSCZ examples:
  - height_cm and weight_kg → a subject with tall height and very
    low weight (or vice versa) is suspicious even if both values
    are individually within physiological range.
  - SIPS positive symptom subscale items → items on the same
    construct correlate; a subject scoring 6 on P1 but 0 on every
    other positive item is unusual.
  - Cognition battery subscales correlate; a subject scoring
    average on most but extreme on one stands out only when you
    look at the cross-variable cloud.

ALGORITHM (deterministic; no scikit-learn):
  1. Filter scoring-eligible rows: value_numeric.notna() &
     ~is_missing_code. Restrict to canonical longitudinal tps
     (floating/conversion excluded). Drop excluded source_form
     patterns (default digital_biomarkers_*).
  2. For each (network, source_form):
     a. Pivot to wide on (subjectid, timepoint) × variable; values
        are value_numeric. (Cell can be NaN if that variable wasn't
        observed at that subject/tp.)
     b. For each variable pair (va, vb) with va < vb alphabetically:
          - joint = wide[[va, vb]].dropna()
          - If len(joint) < min_obs_per_pair (default 30): skip.
          - If either column is constant (std=0): skip.
          - r = pearson(x, y). If |r| < min_correlation
            (default 0.6): skip — relationship too weak to define
            an off-the-line outlier.
          - OLS fit y = a + b*x via numpy.polyfit.
          - residuals = y − (a + b*x).
          - center = median(residuals); scale = MAD * 1.4826.
            If scale == 0 or NaN: skip.
          - z_i = (residual_i − center) / scale for each joint
            observation.
          - If |z_i| ≥ residual_z_threshold (default 4.0): emit
            one flag (per pair × per observation) noting both
            variables, their values, the predicted value of vb
            given va, and the residual.
  3. Sort flags by severity = |z| descending; apply
     max_flags_per_variable cap (default 25) so one over-firing
     pair can't monopolize the sheet; then global cap.

OUTPUT SCHEMA (matches sibling-detector audit columns; adds
pairwise-specific 6):
  ... audit columns (16) ...
  variable_b, value_b_numeric, predicted_b_given_a, residual_value,
  correlation_r, n_pairs_observed
  ... tracker-compatible aliases (7) ...

EVALUATION SCOPE:
  Pairs are constrained to WITHIN the same source_form. This is the
  natural correlation cluster (variables on the same scale / form
  are most likely to be related) and keeps the cost manageable —
  O(F × V_form²) instead of O(V²) globally.

CONSTRAINTS HONORED:
  - Reads only the long-format parquets produced by Step 1.
  - Writes only combined_outputs/discovery/
    pairwise_relationship_candidates.parquet.
  - Does NOT append to combined_qc_flags.parquet.
  - Does NOT inherit from FormCheck.
  - Atomic write via PID-stamped tmp + os.replace.
"""


_NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})
_DEFAULT_EXCLUDED_FORM_PATTERNS = DEFAULT_EXCLUDED_FORM_PATTERNS
_DEFAULT_EXCLUDED_VARIABLE_PATTERNS = DEFAULT_EXCLUDED_VARIABLE_PATTERNS
_is_excluded_form = is_excluded_form
_is_excluded_variable = is_excluded_variable


class PairwiseRelationshipChecks:
    """
    Discovery-tier pairwise-correlation outlier detector. For each
    strongly-correlated pair of variables within a form, flag
    observations whose residual from the cohort's linear fit is an
    outlier.
    """

    OUTPUT_COLUMNS = [
        # Audit columns (16).
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric',
        'expected_value', 'observed_value',
        'algorithm', 'metric_name', 'metric_value',
        'severity_score', 'severity_normalized', 'threshold',
        'error_message', 'dates_detected',
        # Pairwise-specific (6).
        'variable_b', 'value_b_numeric', 'predicted_b_given_a',
        'residual_value', 'correlation_r', 'n_pairs_observed',
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

        pr_cfg = self.config_info.get(
            'discovery', {}).get('pairwise_relationship', {})
        self.networks = list(
            pr_cfg.get('networks', ['PRONET', 'PRESCIENT']))
        self.min_obs_per_pair = int(
            pr_cfg.get('min_obs_per_pair', 30))
        self.min_correlation = float(
            pr_cfg.get('min_correlation', 0.6))
        self.residual_z_threshold = float(
            pr_cfg.get('residual_z_threshold', 4.0))
        # Per-pair cap: keep at most this many flags from a single
        # (variable_a, variable_b) pair so one over-firing pair can't
        # monopolize the sheet.
        per_pair = pr_cfg.get('max_flags_per_pair', 50)
        self.max_flags_per_pair = (
            int(per_pair) if per_pair is not None else None)
        # Per-variable cap: applied after sort, before global cap.
        pv_cap = pr_cfg.get('max_flags_per_variable', 25)
        self.max_flags_per_variable = (
            int(pv_cap) if pv_cap is not None else None)
        cap = pr_cfg.get('max_flags_to_write', None)
        self.max_flags_to_write = (
            int(cap) if cap is not None else None)
        self.excluded_source_form_patterns = tuple(
            pr_cfg.get(
                'excluded_source_form_patterns',
                list(_DEFAULT_EXCLUDED_FORM_PATTERNS)))
        self.excluded_variable_patterns = tuple(
            pr_cfg.get(
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
            'form_groups_considered': 0,
            'pairs_considered': 0,
            'pairs_skipped_insufficient_joint': 0,
            'pairs_skipped_constant_variable': 0,
            'pairs_skipped_low_correlation': 0,
            'pairs_skipped_degenerate_mad': 0,
            'pairs_yielding_flags': 0,
            'flags_above_threshold': 0,
            'flags_dropped_by_per_pair_cap': 0,
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
            missing_cols = [
                c for c in self.REQUIRED_INPUT_COLUMNS
                if c not in df.columns
            ]
            if missing_cols:
                raise RuntimeError(
                    f"FATAL: long-format parquet at {path} is "
                    f"missing required columns: {missing_cols}."
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
            excl_mask = scoring['source_form'].apply(
                lambda f: _is_excluded_form(
                    f, self.excluded_source_form_patterns))
            scoring = scoring[~excl_mask].copy()
            self._counters['rows_excluded_by_form_pattern'] = (
                before - len(scoring))

        if self.excluded_variable_patterns and len(scoring) > 0:
            before = len(scoring)
            excl_var = scoring['variable'].apply(
                lambda v: _is_excluded_variable(
                    v, self.excluded_variable_patterns))
            scoring = scoring[~excl_var].copy()
            self._counters['rows_excluded_by_variable_pattern'] = (
                before - len(scoring))

        self._counters['valid_scoring_rows'] = len(scoring)
        if len(scoring) == 0:
            return self._empty_output_df()

        records = []
        for (network, source_form), grp in scoring.groupby(
                ['network', 'source_form'], sort=False):
            if not source_form:
                continue
            self._counters['form_groups_considered'] += 1
            records.extend(
                self._evaluate_form(network, source_form, grp))

        if not records:
            return self._empty_output_df()

        df = pd.DataFrame(records)
        df = df.sort_values(
            'severity_score', ascending=False).reset_index(drop=True)

        # Per-variable cap: limit how many flags any single variable
        # can produce after counting both (a, b) and (a's appearance
        # as b-of-other-pair) occurrences.
        if self.max_flags_per_variable is not None:
            before = len(df)
            df = (
                df.groupby('variable', sort=False)
                .head(self.max_flags_per_variable)
                .reset_index(drop=True))
            self._counters['flags_dropped_by_per_variable_cap'] = (
                before - len(df))

        if (self.max_flags_to_write is not None
                and len(df) > self.max_flags_to_write):
            print(
                f"[pairwise_relationship_checks] WARNING: capping "
                f"output to top {self.max_flags_to_write} by "
                f"severity_score."
            )
            df = df.head(self.max_flags_to_write).copy()
        return self._format_output(df)

    def _evaluate_form(
        self, network: str, source_form: str, grp: pd.DataFrame
    ) -> list:
        """
        Pivot rows for one (network, source_form) to wide form,
        iterate variable pairs, and emit per-observation flag dicts.
        """
        # Drop dupes deterministically: keep first observation per
        # (subjectid, timepoint, variable). Producer should not emit
        # dupes, but defending the pivot against an unexpected
        # MultiIndex collision is cheap.
        grp = grp.drop_duplicates(
            subset=['subjectid', 'timepoint', 'variable'],
            keep='first')
        try:
            wide = grp.pivot_table(
                index=['subjectid', 'timepoint'],
                columns='variable',
                values='value_numeric',
                aggfunc='first',
            )
        except Exception:
            return []
        var_list = sorted(wide.columns.tolist())
        if len(var_list) < 2:
            return []

        records = []
        today = str(datetime.today().date())
        for i, va in enumerate(var_list):
            col_a = wide[va]
            for vb in var_list[i + 1:]:
                self._counters['pairs_considered'] += 1
                col_b = wide[vb]
                pair = pd.concat(
                    [col_a, col_b], axis=1, keys=[va, vb]).dropna()
                if len(pair) < self.min_obs_per_pair:
                    self._counters[
                        'pairs_skipped_insufficient_joint'] += 1
                    continue
                x = pair[va].to_numpy(dtype=float)
                y = pair[vb].to_numpy(dtype=float)
                std_x = float(np.std(x))
                std_y = float(np.std(y))
                if std_x == 0 or std_y == 0:
                    self._counters[
                        'pairs_skipped_constant_variable'] += 1
                    continue
                # Rank-based (Spearman) correlation for the gate.
                # Pearson is too easily dragged below threshold by a
                # single extreme outlier — exactly the pattern we
                # want to detect. Spearman ranks first, so the line
                # stays "visible" even when a few subjects sit far
                # off it.
                rank_x = pd.Series(x).rank().to_numpy()
                rank_y = pd.Series(y).rank().to_numpy()
                r = float(np.corrcoef(rank_x, rank_y)[0, 1])
                if np.isnan(r) or abs(r) < self.min_correlation:
                    self._counters[
                        'pairs_skipped_low_correlation'] += 1
                    continue
                slope, intercept = np.polyfit(x, y, 1)
                predicted_y = slope * x + intercept
                residuals = y - predicted_y
                center = float(np.median(residuals))
                mad = float(
                    np.median(np.abs(residuals - center)))
                scale = mad * 1.4826
                if scale == 0 or np.isnan(scale):
                    self._counters[
                        'pairs_skipped_degenerate_mad'] += 1
                    continue
                z = (residuals - center) / scale
                abs_z = np.abs(z)
                outlier_pos = np.where(
                    abs_z >= self.residual_z_threshold)[0]
                if len(outlier_pos) == 0:
                    continue
                self._counters['pairs_yielding_flags'] += 1
                # Cap per-pair flag count: a noisy pair (say due to a
                # heavy-tail distribution) shouldn't drown the sheet.
                if (self.max_flags_per_pair is not None
                        and len(outlier_pos)
                        > self.max_flags_per_pair):
                    # Take the most extreme ones.
                    order = np.argsort(-abs_z[outlier_pos])
                    kept = outlier_pos[
                        order[: self.max_flags_per_pair]]
                    self._counters[
                        'flags_dropped_by_per_pair_cap'] += (
                            len(outlier_pos) - len(kept))
                    outlier_pos = kept
                self._counters['flags_above_threshold'] += len(
                    outlier_pos)
                idx_arr = pair.index.to_numpy()
                for pos in outlier_pos:
                    subj, tp = idx_arr[pos]
                    records.append({
                        'subjectid': str(subj),
                        'network': str(network),
                        'timepoint': str(tp),
                        'variable': va,
                        'source_form': str(source_form),
                        'value': str(x[pos]),
                        'value_numeric': float(x[pos]),
                        'expected_value': float(
                            (y[pos] - intercept) / slope
                            if slope != 0 else x[pos]),
                        'observed_value': float(x[pos]),
                        'algorithm': 'pairwise_relationship',
                        'metric_name': 'residual_robust_z',
                        'metric_value': float(z[pos]),
                        'severity_score': float(abs_z[pos]),
                        'threshold': float(
                            self.residual_z_threshold),
                        'error_message': (
                            f"{va} vs {vb}: subject's pair "
                            f"({x[pos]:.4g}, {y[pos]:.4g}) deviates "
                            f"from cohort fit (r={r:.3g}, "
                            f"slope={slope:.4g}); "
                            f"predicted {vb}|{va}="
                            f"{predicted_y[pos]:.4g}, observed "
                            f"{vb}={y[pos]:.4g}, residual="
                            f"{residuals[pos]:.4g}, robust z="
                            f"{z[pos]:.3g}"
                        ),
                        'dates_detected': today,
                        'variable_b': vb,
                        'value_b_numeric': float(y[pos]),
                        'predicted_b_given_a': float(
                            predicted_y[pos]),
                        'residual_value': float(residuals[pos]),
                        'correlation_r': float(r),
                        'n_pairs_observed': int(len(pair)),
                    })
        return records

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        for col in (
            'subjectid', 'network', 'timepoint', 'variable',
            'source_form', 'value',
        ):
            out[col] = df[col].astype(str)
        for col in (
            'value_numeric', 'expected_value', 'observed_value',
            'metric_value', 'severity_score', 'threshold',
            'value_b_numeric', 'predicted_b_given_a',
            'residual_value', 'correlation_r',
        ):
            out[col] = df[col].astype(float)
        out['severity_normalized'] = df['severity_score'].apply(
            normalize_zscore_severity).astype(float)
        out['algorithm'] = df['algorithm'].astype(str)
        out['metric_name'] = df['metric_name'].astype(str)
        out['error_message'] = df['error_message'].astype(str)
        out['dates_detected'] = df['dates_detected'].astype(str)
        out['variable_b'] = df['variable_b'].astype(str)
        out['n_pairs_observed'] = df['n_pairs_observed'].astype('int64')
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = (
            df['variable'].astype(str) + ',' + df['variable_b'].astype(str))
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
            'variable_b': pd.Series(dtype='object'),
            'value_b_numeric': pd.Series(dtype='float64'),
            'predicted_b_given_a': pd.Series(dtype='float64'),
            'residual_value': pd.Series(dtype='float64'),
            'correlation_r': pd.Series(dtype='float64'),
            'n_pairs_observed': pd.Series(dtype='int64'),
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
            f"{out_dir}/pairwise_relationship_candidates.parquet")
        self._atomic_write_parquet(df, final_path)
        if len(df) == 0:
            print(
                f"[pairwise_relationship_checks] no pairwise "
                f"residual outliers found → {final_path}"
            )
        else:
            top = df.head(5)[
                ['subjectid', 'variable', 'variable_b', 'timepoint',
                 'severity_score']]
            print(
                f"[pairwise_relationship_checks] {len(df)} "
                f"pairwise residual outliers → {final_path}"
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
            ("variable pairs considered",
             c['pairs_considered']),
            ("pairs skipped — insufficient joint obs",
             c['pairs_skipped_insufficient_joint']),
            ("pairs skipped — constant variable",
             c['pairs_skipped_constant_variable']),
            ("pairs skipped — |r| below threshold",
             c['pairs_skipped_low_correlation']),
            ("pairs skipped — degenerate MAD",
             c['pairs_skipped_degenerate_mad']),
            ("pairs yielding any flags",
             c['pairs_yielding_flags']),
            ("flags above threshold (pre-cap)",
             c['flags_above_threshold']),
            ("flags dropped by per-pair cap",
             c['flags_dropped_by_per_pair_cap']),
            ("flags dropped by per-variable cap",
             c['flags_dropped_by_per_variable_cap']),
            ("flags written (post-cap)",
             c['flags_written']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[pairwise_relationship_checks] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")

    def _atomic_write_parquet(
        self, df: pd.DataFrame, final_path: str
    ):
        atomic_write_parquet(df, final_path)


if __name__ == '__main__':
    PairwiseRelationshipChecks()()
