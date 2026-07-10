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
    normalize_passthrough_severity,
)

"""
Copy-forward detector (Layer 2 / discovery, default-off).

REWRITTEN to fire on FORM-LEVEL copy patterns rather than single-
variable identical runs. The prior single-variable run check fired
mostly on legitimate stable-state behavior (a subject scoring 0 on
a symptom Likert for 4 consecutive visits is expected, not suspect).

The smoking gun for RA copy-paste between visits is when a LARGE
FRACTION of a form's numeric variables are bit-identical from one
visit to the next — that pattern is essentially impossible from real
clinical stability alone.

ALGORITHM (deterministic, no ML):
  1. Filter scoring-eligible rows: value_numeric.notna() &
     ~is_missing_code. Restrict to canonical timepoints (floating/
     conversion excluded). Drop excluded form / variable patterns.
  2. Per (network, subjectid, source_form):
     a. Pivot to wide: rows = timepoint, columns = variable.
     b. Sort timepoints by canonical index.
     c. For each consecutive pair (tp_a, tp_b):
          - joint = both rows non-NaN per variable.
          - Skip if len(joint) < min_form_vars (default 8).
          - match_fraction = bit-identical count / len(joint).
          - flag when match_fraction ≥ min_match_fraction
            (default 0.85).
  3. Weight severity by cohort variability: the more variable the
     form's values are across the cohort at this tp-pair window,
     the higher the severity. Specifically:
       severity = 100 * match_fraction
                  * (cohort_form_variability_factor)
     where cohort_form_variability_factor is the average across the
     form's variables of (mad_scale > 0 ? 1.0 : 0.0). Pure-stable
     forms (everyone has the same value) get severity discounted to
     0 — they cannot be evidence of copy-paste.
  4. Sort by severity desc; apply max_flags_per_subject_form cap and
     global cap.

INPUT (read-only):
  dependencies/multi_tp_{network}_numeric_long.parquet

OUTPUT (atomic, tmp + os.replace):
  combined_outputs/discovery/copy_forward_candidates.parquet
    — single combined parquet, audit + form-copy-specific columns.

CONSTRAINTS HONORED:
  - Reads only the long-format parquet.
  - Writes only combined_outputs/discovery/
    copy_forward_candidates.parquet.
  - Does NOT append to combined_qc_flags.parquet.
  - Does NOT inherit from FormCheck.
  - Atomic write via PID-stamped tmp + os.replace.
"""


_NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})
_DEFAULT_EXCLUDED_FORM_PATTERNS = DEFAULT_EXCLUDED_FORM_PATTERNS
_DEFAULT_EXCLUDED_VARIABLE_PATTERNS = DEFAULT_EXCLUDED_VARIABLE_PATTERNS
_is_excluded_form = is_excluded_form
_is_excluded_variable = is_excluded_variable


class CopyForwardChecks:
    """
    Form-level copy-forward detector. Flags subject × form ×
    consecutive-tp-pair combinations where a large fraction of the
    form's numeric variables are bit-identical between visits.
    """

    OUTPUT_COLUMNS = [
        # Audit (16).
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric',
        'expected_value', 'observed_value',
        'algorithm', 'metric_name', 'metric_value',
        'severity_score', 'severity_normalized', 'threshold',
        'error_message', 'dates_detected',
        # Copy-forward-specific (5).
        'prev_timepoint', 'match_fraction',
        'compared_variables_count',
        'identical_variables', 'cohort_variability_factor',
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

        cf_cfg = self.config_info.get(
            'discovery', {}).get('copy_forward', {})
        self.networks = list(
            cf_cfg.get('networks', ['PRONET', 'PRESCIENT']))
        self.min_match_fraction = float(
            cf_cfg.get('min_match_fraction', 0.85))
        # Lowered from 8 → 4: after the variable-pattern exclusions
        # (which drop _complete/_date/_count/_other/_pharm_) most
        # AMPSCZ forms retain only a handful of numeric vars per
        # visit. 4 keeps statistical meaning while letting smaller
        # forms participate.
        self.min_form_vars = int(cf_cfg.get('min_form_vars', 4))
        self.min_cohort_obs_per_variable = int(
            cf_cfg.get('min_cohort_obs_per_variable', 20))
        self.excluded_source_form_patterns = tuple(
            cf_cfg.get(
                'excluded_source_form_patterns',
                list(_DEFAULT_EXCLUDED_FORM_PATTERNS)))
        self.excluded_variable_patterns = tuple(
            cf_cfg.get(
                'excluded_variable_patterns',
                list(_DEFAULT_EXCLUDED_VARIABLE_PATTERNS)))
        cap = cf_cfg.get('max_flags_to_write', None)
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
            'subject_form_groups_considered': 0,
            'tp_pairs_considered': 0,
            'tp_pairs_skipped_too_few_joint_vars': 0,
            'tp_pairs_above_threshold': 0,
            'flags_zero_after_variability_weighting': 0,
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
            mask = scoring['source_form'].apply(
                lambda f: _is_excluded_form(
                    f, self.excluded_source_form_patterns))
            scoring = scoring[~mask].copy()
            self._counters['rows_excluded_by_form_pattern'] = (
                before - len(scoring))

        if self.excluded_variable_patterns and len(scoring) > 0:
            before = len(scoring)
            mask = scoring['variable'].apply(
                lambda v: _is_excluded_variable(
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

        # Pre-compute cohort variability per (network, source_form,
        # variable): fraction of the form's variables whose MAD scale
        # is non-zero is the "interesting form-pair" weight. A form
        # where everyone has the same value across the cohort gets
        # a 0 weight and never produces a flag — bit-identical
        # repetition is the cohort's normal behavior, not a copy.
        variability = self._cohort_variability(scoring)

        records = []
        today = str(datetime.today().date())
        for (net, subj, form), grp in scoring.groupby(
                ['network', 'subjectid', 'source_form'], sort=False):
            if not form:
                continue
            self._counters['subject_form_groups_considered'] += 1
            wide = grp.pivot_table(
                index='timepoint', columns='variable',
                values='value_numeric', aggfunc='first',
            )
            wide = wide.assign(
                _tp_idx=wide.index.map(self._tp_index)
            ).sort_values('_tp_idx')
            tps = wide.index.tolist()
            wide = wide.drop(columns=['_tp_idx'])
            if len(tps) < 2:
                continue
            for i in range(len(tps) - 1):
                tp_a = tps[i]
                tp_b = tps[i + 1]
                self._counters['tp_pairs_considered'] += 1
                row_a = wide.loc[tp_a]
                row_b = wide.loc[tp_b]
                joint = pd.concat(
                    [row_a.rename('a'), row_b.rename('b')], axis=1
                ).dropna()
                if len(joint) < self.min_form_vars:
                    self._counters[
                        'tp_pairs_skipped_too_few_joint_vars'] += 1
                    continue
                identical_mask = joint['a'] == joint['b']
                matches = int(identical_mask.sum())
                frac = float(matches) / float(len(joint))
                if frac < self.min_match_fraction:
                    continue
                self._counters['tp_pairs_above_threshold'] += 1
                identical_vars = joint[identical_mask].index.tolist()
                # Weight by cohort variability of the form's vars.
                var_weights = [
                    variability.get((net, form, v), 0.0)
                    for v in joint.index.tolist()
                ]
                cohort_factor = (
                    float(np.mean(var_weights))
                    if var_weights else 0.0)
                severity = 100.0 * frac * cohort_factor
                if severity <= 0:
                    self._counters[
                        'flags_zero_after_variability_weighting'] += 1
                    continue
                records.append({
                    'subjectid': str(subj),
                    'network': str(net),
                    'timepoint': str(tp_b),
                    'variable': '(form-level)',
                    'source_form': str(form),
                    'value': (
                        f"form={form},tp_pair={tp_a}->{tp_b},"
                        f"matches={matches}/{len(joint)}"
                    ),
                    'value_numeric': float('nan'),
                    'expected_value': float('nan'),
                    'observed_value': float('nan'),
                    'algorithm': 'form_level_copy_forward',
                    'metric_name': 'form_match_fraction',
                    'metric_value': frac,
                    'severity_score': severity,
                    'threshold': float(self.min_match_fraction),
                    'error_message': (
                        f"Form-level copy-forward: {matches} of "
                        f"{len(joint)} numeric variables on "
                        f"form {form} are bit-identical "
                        f"between {tp_a} and {tp_b} "
                        f"({frac:.1%} ≥ "
                        f"{self.min_match_fraction:.0%}); cohort "
                        f"variability factor "
                        f"{cohort_factor:.2f}"
                    ),
                    'dates_detected': today,
                    'prev_timepoint': str(tp_a),
                    'match_fraction': frac,
                    'compared_variables_count': int(len(joint)),
                    'identical_variables': ','.join(
                        identical_vars[:15]
                        + (['...']
                           if len(identical_vars) > 15 else [])),
                    'cohort_variability_factor': cohort_factor,
                })

        if not records:
            return self._empty_output_df()
        df = pd.DataFrame(records)
        df = df.sort_values(
            'severity_score', ascending=False).reset_index(drop=True)
        if (self.max_flags_to_write is not None
                and len(df) > self.max_flags_to_write):
            df = df.head(self.max_flags_to_write).copy()
        return self._format_output(df)

    def _cohort_variability(
        self, scoring: pd.DataFrame
    ) -> dict:
        """
        Map (network, source_form, variable) -> 1.0 if cohort MAD
        > 0 and the variable has at least
        min_cohort_obs_per_variable observations, else 0.0. Used as
        a per-flag weighting factor.
        """
        result = {}
        for (net, form, var), grp in scoring.groupby(
                ['network', 'source_form', 'variable'], sort=False):
            if (len(grp)
                    < self.min_cohort_obs_per_variable):
                result[(net, form, var)] = 0.0
                continue
            vals = grp['value_numeric'].to_numpy(dtype=float)
            center = float(np.median(vals))
            mad = float(np.median(np.abs(vals - center)))
            result[(net, form, var)] = 1.0 if mad > 0 else 0.0
        return result

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        for col in (
            'subjectid', 'network', 'timepoint', 'variable',
            'source_form', 'value', 'algorithm', 'metric_name',
            'error_message', 'dates_detected', 'prev_timepoint',
            'identical_variables',
        ):
            out[col] = df[col].astype(str)
        for col in (
            'value_numeric', 'expected_value', 'observed_value',
            'metric_value', 'severity_score', 'threshold',
            'match_fraction', 'cohort_variability_factor',
        ):
            out[col] = df[col].astype(float)
        out['severity_normalized'] = df['severity_score'].apply(
            normalize_passthrough_severity).astype(float)
        out['compared_variables_count'] = (
            df['compared_variables_count'].astype('int64'))
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = out['identical_variables']
        out['affected_timepoints'] = (
            df['prev_timepoint'].astype(str) + ',' + df['timepoint'].astype(str))
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
            'prev_timepoint': pd.Series(dtype='object'),
            'match_fraction': pd.Series(dtype='float64'),
            'compared_variables_count': pd.Series(dtype='int64'),
            'identical_variables': pd.Series(dtype='object'),
            'cohort_variability_factor': pd.Series(dtype='float64'),
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
            f"{out_dir}/copy_forward_candidates.parquet")
        self._atomic_write_parquet(df, final_path)
        if len(df) == 0:
            print(
                f"[copy_forward_checks] no form-level copy-forward "
                f"candidates → {final_path}"
            )
        else:
            top = df.head(5)[
                ['subjectid', 'source_form', 'prev_timepoint',
                 'timepoint', 'severity_score']]
            print(
                f"[copy_forward_checks] {len(df)} form-level "
                f"copy-forward candidates → {final_path}"
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
            ("(net,subj,form) groups considered",
             c['subject_form_groups_considered']),
            ("tp-pairs considered",
             c['tp_pairs_considered']),
            ("tp-pairs skipped — <min joint form vars",
             c['tp_pairs_skipped_too_few_joint_vars']),
            ("tp-pairs above match threshold",
             c['tp_pairs_above_threshold']),
            ("flags zeroed by cohort-variability weighting",
             c['flags_zero_after_variability_weighting']),
            ("flags written (post-cap)", c['flags_written']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[copy_forward_checks] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")

    def _atomic_write_parquet(
        self, df: pd.DataFrame, final_path: str
    ):
        atomic_write_parquet(df, final_path)


if __name__ == '__main__':
    CopyForwardChecks()()
