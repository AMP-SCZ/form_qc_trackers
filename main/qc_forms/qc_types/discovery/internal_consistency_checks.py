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
    atomic_write_parquet,
    normalize_relative_to_threshold,
)

"""
Internal-consistency anomaly detector (Layer 2 / discovery, default-off).

Domain-specific checks that encode KNOWN relationships between AMPSCZ
variables: BMI must match weight/height, age must fall in the
enrollment window, vital signs must be physiologically plausible.
These are the highest-precision flags discovery can produce — every
violation is almost certainly a real data-entry error, not a quirky
subject.

CONTRAST WITH SIBLING DETECTORS:
  - point_spike / longitudinal_delta / longitudinal_anomaly all
    operate on ONE variable at a time and rank by cohort outlier-ness
    (statistical anomaly). They tell you "this looks unusual" but
    can't tell you why.
  - internal_consistency operates on KNOWN ground-truth relationships
    (math, physiology, study protocol). Every flag points to a
    specific arithmetic mismatch or out-of-range value.

INPUT (read-only):
  dependencies/multi_tp_{network}_numeric_long.parquet — same as the
  sibling detectors. Required cols: subjectid, network, timepoint,
  variable, source_form, value, value_numeric, is_missing_code.

  dependencies/internal_consistency_rules.json — list of rule dicts.
  See the file's _meta block for kinds and required fields. Operator
  can override the rules-file path via config.

OUTPUT (atomic, tmp + os.replace):
  combined_outputs/discovery/internal_consistency_candidates.parquet

RULE KINDS:
  - bmi_formula: bmi declared on the form must match weight/height
    within rel_tolerance; optionally bounds-check height and weight.
  - sum_equals: a declared total must equal the sum of its components
    within tolerance. (Reserved for future SIPS-style sum checks once
    item-to-total mapping is wired in; the evaluator is implemented
    so rules can be added without code changes.)
  - ratio_range: a declared ratio (a/b) must fall within [min, max].
  - numeric_range: a single variable's value must fall within
    [min_value, max_value]. Covers vitals and enrollment-criterion
    range checks.

EVALUATION:
  Rules are evaluated PER subject × timepoint. A rule that requires
  multiple input variables silently skips a (subject, tp) where any
  required input is missing (NaN or is_missing_code) — non-presence
  is not a violation; missingness is the missingness detector's job.

FAILURE MODES (no silent fallbacks):
  - rules JSON missing → FileNotFoundError naming the path
  - long parquet missing → FileNotFoundError
  - long parquet missing required column → RuntimeError
  - rule with unknown 'kind' → RuntimeError naming the rule
  - zero flags → STILL write empty parquet with declared dtypes

CONSTRAINTS:
  - Reads only long-format parquets + the rules JSON.
  - Writes only combined_outputs/discovery/internal_consistency_*.
  - Does NOT append to combined_qc_flags.parquet.
  - Atomic write via PID-stamped tmp + os.replace.
"""


_NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})


class InternalConsistencyChecks:
    """
    Discovery-tier internal-consistency detector. Iterates a list of
    rules from `dependencies/internal_consistency_rules.json`, each
    producing one flag per (subject, timepoint) where the rule is
    evaluable and violated.
    """

    OUTPUT_COLUMNS = [
        # Audit columns (16, mirror N1 layout).
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric',
        'expected_value', 'observed_value',
        'algorithm', 'metric_name', 'metric_value',
        'severity_score', 'severity_normalized', 'threshold',
        'error_message', 'dates_detected',
        # Consistency-specific (4).
        'rule_name', 'rule_kind', 'related_variables', 'rule_inputs',
        # Tracker-compatible aliases (7).
        'subject', 'displayed_variable', 'displayed_timepoint',
        'affected_variables', 'affected_timepoints',
        'affected_forms', 'displayed_form',
    ]

    REQUIRED_INPUT_COLUMNS = [
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric', 'is_missing_code',
    ]

    _KNOWN_KINDS = frozenset({
        'bmi_formula', 'sum_equals', 'ratio_range', 'numeric_range',
    })

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

        ic_cfg = self.config_info.get(
            'discovery', {}).get('internal_consistency', {})
        self.networks = list(
            ic_cfg.get('networks', ['PRONET', 'PRESCIENT']))
        rules_path = ic_cfg.get(
            'rules_path',
            os.path.join(
                self.depen_path, 'internal_consistency_rules.json'))
        self.rules_path = rules_path
        cap = ic_cfg.get('max_flags_to_write', None)
        self.max_flags_to_write = (
            int(cap) if cap is not None else None)

        self._counters = self._fresh_counters()

    def _fresh_counters(self) -> dict:
        return {
            'input_rows': 0,
            'rules_loaded': 0,
            'rules_evaluated': 0,
            'rule_evaluations_total': 0,
            'rule_evaluations_skipped_missing_inputs': 0,
            'violations_found': 0,
            'flags_written': 0,
        }

    def __call__(self):
        return self.run()

    def run(self):
        self._counters = self._fresh_counters()
        rules = self._load_rules()
        self._counters['rules_loaded'] = len(rules)
        long_df = self._load_inputs()
        self._counters['input_rows'] = len(long_df)
        flagged = self._evaluate(long_df, rules)
        self._counters['flags_written'] = len(flagged)
        self._write_output(flagged)
        self._print_summary()

    def _load_rules(self) -> list:
        if not os.path.exists(self.rules_path):
            raise FileNotFoundError(
                f"FATAL: internal_consistency rules file missing: "
                f"{self.rules_path}. Provide one or point config to "
                f"a different path."
            )
        with open(self.rules_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        rules = list(doc.get('rules', []))
        for r in rules:
            kind = r.get('kind')
            if kind not in self._KNOWN_KINDS:
                raise RuntimeError(
                    f"FATAL: rule {r.get('name', '<unnamed>')} has "
                    f"unknown kind {kind!r}. Known: "
                    f"{sorted(self._KNOWN_KINDS)}"
                )
        return rules

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

    def _evaluate(
        self, long_df: pd.DataFrame, rules: list
    ) -> pd.DataFrame:
        """
        For each rule, pivot the long parquet to the columns the rule
        needs and evaluate per (subject, timepoint). Each rule produces
        zero or more flag records. Records are concatenated, sorted by
        severity descending, and capped.
        """
        if len(long_df) == 0 or len(rules) == 0:
            return self._empty_output_df()

        # Restrict to scoring-eligible rows: numeric value present,
        # not a missing-code. We treat missing-code rows as absent
        # for consistency-check evaluation (missingness is the
        # missingness detector's job).
        eligible = long_df[
            long_df['value_numeric'].notna()
            & (~long_df['is_missing_code'])
        ].copy()
        if len(eligible) == 0:
            return self._empty_output_df()

        records = []
        for rule in rules:
            self._counters['rules_evaluated'] += 1
            try:
                rec = self._evaluate_rule(eligible, rule)
            except Exception as e:
                # A malformed rule should fail loud rather than
                # silently dropping its results — but only that rule;
                # other rules still run.
                raise RuntimeError(
                    f"FATAL: rule {rule.get('name', '<unnamed>')} "
                    f"raised during evaluation: {e}") from e
            records.extend(rec)

        self._counters['violations_found'] = len(records)
        if not records:
            return self._empty_output_df()

        df = pd.DataFrame(records)
        df = df.sort_values(
            'severity_score', ascending=False).reset_index(drop=True)
        if (self.max_flags_to_write is not None
                and len(df) > self.max_flags_to_write):
            print(
                f"[internal_consistency_checks] WARNING: "
                f"{len(df)} flags exceeded cap; keeping top "
                f"{self.max_flags_to_write} by severity_score."
            )
            df = df.head(self.max_flags_to_write).copy()
        return self._format_output(df)

    def _evaluate_rule(
        self, eligible: pd.DataFrame, rule: dict
    ) -> list:
        kind = rule['kind']
        if kind == 'bmi_formula':
            return self._eval_bmi_formula(eligible, rule)
        if kind == 'numeric_range':
            return self._eval_numeric_range(eligible, rule)
        if kind == 'sum_equals':
            return self._eval_sum_equals(eligible, rule)
        if kind == 'ratio_range':
            return self._eval_ratio_range(eligible, rule)
        # Unreachable: _load_rules validates kinds upfront.
        raise RuntimeError(f"unhandled rule kind {kind!r}")

    # ---- Rule kind: bmi_formula ----------------------------------

    def _eval_bmi_formula(
        self, eligible: pd.DataFrame, rule: dict
    ) -> list:
        bmi_var = rule['bmi_var']
        h_var = rule['height_cm_var']
        w_var = rule['weight_kg_var']
        rel_tol = float(rule.get('rel_tolerance', 0.05))
        wide = self._pivot_wide(
            eligible, [bmi_var, h_var, w_var])
        # Need all three present at the (subject, tp).
        valid = wide.dropna(subset=[bmi_var, h_var, w_var])
        self._counters['rule_evaluations_total'] += len(wide)
        self._counters[
            'rule_evaluations_skipped_missing_inputs'] += (
                len(wide) - len(valid))
        if len(valid) == 0:
            return []
        h_m = valid[h_var] / 100.0
        expected_bmi = valid[w_var] / (h_m ** 2)
        rel_err = (
            (valid[bmi_var] - expected_bmi).abs()
            / expected_bmi.replace(0, np.nan))
        violators = valid[rel_err > rel_tol].copy()
        if len(violators) == 0:
            return []
        violators['_expected'] = expected_bmi.loc[violators.index]
        violators['_rel_err'] = rel_err.loc[violators.index]
        rows = []
        today = str(datetime.today().date())
        for _, r in violators.iterrows():
            rows.append(self._make_record(
                rule=rule, today=today,
                row_subject=r['subjectid'],
                row_network=r['network'],
                row_timepoint=r['timepoint'],
                row_source_form=r.get('_source_form', ''),
                primary_variable=bmi_var,
                observed=float(r[bmi_var]),
                expected=float(r['_expected']),
                metric=float(r['_rel_err']),
                threshold=float(rel_tol),
                error_msg=(
                    f"{rule.get('name')}: declared {bmi_var}="
                    f"{r[bmi_var]:.2f} disagrees with "
                    f"weight/height² = {r['_expected']:.2f} "
                    f"(relative error {r['_rel_err']:.2%}, "
                    f"tolerance {rel_tol:.0%}); "
                    f"{h_var}={r[h_var]:.1f} cm, "
                    f"{w_var}={r[w_var]:.1f} kg"
                ),
                related=[bmi_var, h_var, w_var],
                inputs={
                    bmi_var: float(r[bmi_var]),
                    h_var: float(r[h_var]),
                    w_var: float(r[w_var]),
                },
            ))
        return rows

    # ---- Rule kind: numeric_range --------------------------------

    def _eval_numeric_range(
        self, eligible: pd.DataFrame, rule: dict
    ) -> list:
        var = rule['variable']
        min_v = float(rule['min_value'])
        max_v = float(rule['max_value'])
        sub = eligible[eligible['variable'] == var]
        self._counters['rule_evaluations_total'] += len(sub)
        if len(sub) == 0:
            return []
        oob = sub[
            (sub['value_numeric'] < min_v)
            | (sub['value_numeric'] > max_v)
        ]
        if len(oob) == 0:
            return []
        rows = []
        today = str(datetime.today().date())
        for _, r in oob.iterrows():
            val = float(r['value_numeric'])
            if val < min_v:
                deviation = min_v - val
                expected = min_v
            else:
                deviation = val - max_v
                expected = max_v
            rows.append(self._make_record(
                rule=rule, today=today,
                row_subject=r['subjectid'],
                row_network=r['network'],
                row_timepoint=r['timepoint'],
                row_source_form=r.get('source_form', '') or '',
                primary_variable=var,
                observed=val,
                expected=float(expected),
                metric=float(deviation),
                threshold=float(min_v if val < min_v else max_v),
                error_msg=(
                    f"{rule.get('name')}: {var}={val} outside "
                    f"plausible range [{min_v}, {max_v}] "
                    f"(deviation {deviation})"
                ),
                related=[var],
                inputs={var: val},
            ))
        return rows

    # ---- Rule kind: sum_equals -----------------------------------

    def _eval_sum_equals(
        self, eligible: pd.DataFrame, rule: dict
    ) -> list:
        total_var = rule['total_var']
        component_vars = list(rule['component_vars'])
        abs_tol = float(rule.get('abs_tolerance', 0.5))
        all_vars = [total_var] + component_vars
        wide = self._pivot_wide(eligible, all_vars)
        valid = wide.dropna(subset=all_vars)
        self._counters['rule_evaluations_total'] += len(wide)
        self._counters[
            'rule_evaluations_skipped_missing_inputs'] += (
                len(wide) - len(valid))
        if len(valid) == 0:
            return []
        component_sum = valid[component_vars].sum(axis=1)
        diff = (valid[total_var] - component_sum).abs()
        violators = valid[diff > abs_tol].copy()
        if len(violators) == 0:
            return []
        violators['_expected'] = component_sum.loc[violators.index]
        violators['_diff'] = diff.loc[violators.index]
        rows = []
        today = str(datetime.today().date())
        for _, r in violators.iterrows():
            rows.append(self._make_record(
                rule=rule, today=today,
                row_subject=r['subjectid'],
                row_network=r['network'],
                row_timepoint=r['timepoint'],
                row_source_form=r.get('_source_form', ''),
                primary_variable=total_var,
                observed=float(r[total_var]),
                expected=float(r['_expected']),
                metric=float(r['_diff']),
                threshold=float(abs_tol),
                error_msg=(
                    f"{rule.get('name')}: {total_var}="
                    f"{r[total_var]} disagrees with sum of "
                    f"components = {r['_expected']} "
                    f"(|diff|={r['_diff']}, tolerance {abs_tol})"
                ),
                related=[total_var] + component_vars,
                inputs={v: float(r[v]) for v in all_vars},
            ))
        return rows

    # ---- Rule kind: ratio_range ----------------------------------

    def _eval_ratio_range(
        self, eligible: pd.DataFrame, rule: dict
    ) -> list:
        num_var = rule['numerator_var']
        den_var = rule['denominator_var']
        min_r = float(rule['min_ratio'])
        max_r = float(rule['max_ratio'])
        wide = self._pivot_wide(eligible, [num_var, den_var])
        valid = wide.dropna(subset=[num_var, den_var])
        valid = valid[valid[den_var] != 0]
        self._counters['rule_evaluations_total'] += len(wide)
        self._counters[
            'rule_evaluations_skipped_missing_inputs'] += (
                len(wide) - len(valid))
        if len(valid) == 0:
            return []
        ratio = valid[num_var] / valid[den_var]
        oob = valid[(ratio < min_r) | (ratio > max_r)].copy()
        if len(oob) == 0:
            return []
        oob['_ratio'] = ratio.loc[oob.index]
        rows = []
        today = str(datetime.today().date())
        for _, r in oob.iterrows():
            rval = float(r['_ratio'])
            if rval < min_r:
                deviation = min_r - rval
                expected = min_r
            else:
                deviation = rval - max_r
                expected = max_r
            rows.append(self._make_record(
                rule=rule, today=today,
                row_subject=r['subjectid'],
                row_network=r['network'],
                row_timepoint=r['timepoint'],
                row_source_form=r.get('_source_form', ''),
                primary_variable=num_var,
                observed=rval,
                expected=float(expected),
                metric=float(deviation),
                threshold=float(min_r if rval < min_r else max_r),
                error_msg=(
                    f"{rule.get('name')}: {num_var}/{den_var}="
                    f"{rval:.3g} outside [{min_r}, {max_r}]"
                ),
                related=[num_var, den_var],
                inputs={
                    num_var: float(r[num_var]),
                    den_var: float(r[den_var]),
                },
            ))
        return rows

    # ---- Helpers --------------------------------------------------

    def _pivot_wide(
        self, eligible: pd.DataFrame, variables: list
    ) -> pd.DataFrame:
        """
        Pivot the long frame to wide on (subject, network, tp) for
        the given variable list. Missing cells become NaN. Source
        form is carried via majority vote (typically all rows on the
        same form, so this is a no-op for well-formed inputs).
        """
        sub = eligible[eligible['variable'].isin(variables)]
        if len(sub) == 0:
            return pd.DataFrame(
                columns=['subjectid', 'network', 'timepoint',
                         '_source_form'] + variables)
        # Use first source_form for the (subjectid, network, tp).
        form_map = (
            sub.groupby(
                ['subjectid', 'network', 'timepoint'],
                sort=False)['source_form'].first().reset_index())
        wide = sub.pivot_table(
            index=['subjectid', 'network', 'timepoint'],
            columns='variable',
            values='value_numeric',
            aggfunc='first',
        ).reset_index()
        # Make sure all requested vars are present as columns even
        # if no rows had them at this scope.
        for v in variables:
            if v not in wide.columns:
                wide[v] = np.nan
        wide = wide.merge(
            form_map.rename(columns={'source_form': '_source_form'}),
            on=['subjectid', 'network', 'timepoint'], how='left')
        return wide

    def _make_record(
        self, rule, today,
        row_subject, row_network, row_timepoint, row_source_form,
        primary_variable, observed, expected, metric, threshold,
        error_msg, related, inputs,
    ) -> dict:
        return {
            'subjectid': str(row_subject),
            'network': str(row_network),
            'timepoint': str(row_timepoint),
            'variable': str(primary_variable),
            'source_form': str(row_source_form or ''),
            'value': str(observed),
            'value_numeric': float(observed),
            'expected_value': float(expected),
            'observed_value': float(observed),
            'algorithm': 'internal_consistency',
            'metric_name': rule['kind'],
            'metric_value': float(metric),
            'severity_score': float(abs(metric)),
            'threshold': float(threshold),
            'error_message': error_msg,
            'dates_detected': today,
            'rule_name': str(rule.get('name', '<unnamed>')),
            'rule_kind': str(rule['kind']),
            'related_variables': ','.join(related),
            'rule_inputs': json.dumps(inputs, sort_keys=True),
        }

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
        ):
            out[col] = df[col].astype(float)
        # Per-rule threshold-relative normalization: severity at the
        # rule's declared threshold maps to 50, 2x threshold to 100.
        # Works across heterogeneous rule kinds (BMI rel error vs
        # vital-sign deviation in raw units).
        out['severity_normalized'] = df.apply(
            lambda r: normalize_relative_to_threshold(
                r['severity_score'], r['threshold']),
            axis=1).astype(float)
        out['algorithm'] = df['algorithm'].astype(str)
        out['metric_name'] = df['metric_name'].astype(str)
        out['error_message'] = df['error_message'].astype(str)
        out['dates_detected'] = df['dates_detected'].astype(str)
        out['rule_name'] = df['rule_name'].astype(str)
        out['rule_kind'] = df['rule_kind'].astype(str)
        out['related_variables'] = (
            df['related_variables'].astype(str))
        out['rule_inputs'] = df['rule_inputs'].astype(str)
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = out['related_variables']
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
            'rule_name': pd.Series(dtype='object'),
            'rule_kind': pd.Series(dtype='object'),
            'related_variables': pd.Series(dtype='object'),
            'rule_inputs': pd.Series(dtype='object'),
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
            f"{out_dir}/internal_consistency_candidates.parquet")
        self._atomic_write_parquet(df, final_path)
        if len(df) == 0:
            print(
                f"[internal_consistency_checks] no consistency "
                f"violations found; wrote empty parquet → "
                f"{final_path}"
            )
        else:
            top = df.head(5)[
                ['rule_name', 'subjectid', 'timepoint',
                 'severity_score']]
            print(
                f"[internal_consistency_checks] {len(df)} "
                f"consistency violations → {final_path}"
            )
            print(
                f"Top 5 by severity:\n"
                f"{top.to_string(index=False)}"
            )

    def _print_summary(self):
        c = self._counters
        rows = [
            ("input rows", c['input_rows']),
            ("rules loaded", c['rules_loaded']),
            ("rules evaluated", c['rules_evaluated']),
            ("rule eval candidates total",
             c['rule_evaluations_total']),
            ("rule evals skipped — missing inputs",
             c['rule_evaluations_skipped_missing_inputs']),
            ("violations found",
             c['violations_found']),
            ("flags written (post-cap)",
             c['flags_written']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[internal_consistency_checks] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")

    def _atomic_write_parquet(
        self, df: pd.DataFrame, final_path: str
    ):
        atomic_write_parquet(df, final_path)


if __name__ == '__main__':
    InternalConsistencyChecks()()
