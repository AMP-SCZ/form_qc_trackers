import os
import sys
import json
import hashlib
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
Duplicate-record detector (Layer 2 / discovery, default-off).

REWRITTEN to target the two duplicate patterns that actually occur in
AMPSCZ data; the prior CSV-fingerprint + within-subject NN pass was
structurally aimed at the wrong axis (one row per subject per file).

Two complementary passes:

  1. CROSS-SUBJECT, SAME-TP, SAME-SITE form-level fingerprint match.
     The "operator entered the same patient twice with different IDs"
     or "copy/pasted an entire form between subjects at the same
     site" pattern. For each (network, site_prefix, source_form, tp),
     build a per-subject fingerprint over the form's numeric values;
     flag groups of ≥ cross_subject_min_group_size with identical
     fingerprints, requiring at least min_cols_for_fingerprint columns
     and a match fraction ≥ cross_subject_min_match_fraction
     (within-group exact match by construction; the fraction is
     informational here).

  2. WITHIN-SUBJECT, CROSS-TP form-level near-duplicate.
     The "RA literally copied the previous visit's row" pattern. For
     each (subject, source_form), for each pair of canonically-
     adjacent timepoints (tp_idx, tp_idx+1), compute the fraction of
     joint-numeric variables whose value_numeric is bit-equal between
     the two visits. Flag when the fraction is ≥
     within_subject_min_match_fraction (default 0.85) over at least
     within_subject_min_form_vars compared columns.

INPUT (read-only):
  dependencies/multi_tp_{network}_numeric_long.parquet — the same
  long-format data the other discovery detectors use. We no longer
  read per-tp CSVs; this is intentional. The long parquet is already
  numeric-validated and missing-code-flagged.

OUTPUT (atomic, tmp + os.replace):
  combined_outputs/discovery/duplicate_record_candidates.parquet

OUTPUT SCHEMA (single combined sheet across both passes):
  Audit (16):
    subjectid, network, timepoint, variable ('(row-level)'),
    source_form, value (compact summary), value_numeric (NaN),
    expected_value (NaN), observed_value (NaN), algorithm
    (=duplicate_method), metric_name, metric_value, severity_score,
    threshold, error_message, dates_detected.
  Duplicate-specific (6):
    duplicate_method, group_id, group_size, match_fraction,
    partner_subjectids, compared_variables_count.
  Tracker-compatible aliases (7).

FAILURE MODES:
  - any configured network's long parquet missing → FileNotFoundError
  - long parquet missing required column → RuntimeError
  - zero flags → STILL write empty parquet with declared dtypes

CONSTRAINTS HONORED:
  - Reads only the long-format parquets.
  - Writes only combined_outputs/discovery/
    duplicate_record_candidates.parquet.
  - Does NOT append to combined_qc_flags.parquet.
  - Does NOT inherit from FormCheck.
  - Atomic write via PID-stamped tmp + os.replace.
"""


_NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})
_DEFAULT_EXCLUDED_FORM_PATTERNS = DEFAULT_EXCLUDED_FORM_PATTERNS
_DEFAULT_EXCLUDED_VARIABLE_PATTERNS = DEFAULT_EXCLUDED_VARIABLE_PATTERNS
_is_excluded_form = is_excluded_form
_is_excluded_variable = is_excluded_variable


def _fingerprint_values(vals: np.ndarray) -> str:
    """
    Deterministic fingerprint over an aligned numeric array. We
    round to 6 sig figs to absorb float-formatting noise across
    REDCap exports without merging genuinely different values.
    """
    repr_str = '|'.join(f"{v:.6g}" for v in vals)
    return hashlib.md5(
        repr_str.encode('utf-8', errors='ignore')).hexdigest()


class DuplicateRecordChecks:
    """
    Rewritten duplicate detector. Two passes, one combined output:
    cross-subject same-tp same-site fingerprint match and
    within-subject cross-tp form-level near-duplicate.
    """

    OUTPUT_COLUMNS = [
        # Audit (16).
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric',
        'expected_value', 'observed_value',
        'algorithm', 'metric_name', 'metric_value',
        'severity_score', 'severity_normalized', 'threshold',
        'error_message', 'dates_detected',
        # Duplicate-specific (6).
        'duplicate_method', 'group_id', 'group_size',
        'match_fraction', 'partner_subjectids',
        'compared_variables_count',
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

        dup_cfg = self.config_info.get(
            'discovery', {}).get('duplicate_record', {})
        self.networks = list(
            dup_cfg.get('networks', ['PRONET', 'PRESCIENT']))
        self.min_cols_for_fingerprint = int(
            dup_cfg.get('min_cols_for_fingerprint', 12))
        self.dup_cols_cap = int(dup_cfg.get('dup_cols_cap', 30))
        self.cross_subject_min_group_size = int(
            dup_cfg.get('cross_subject_min_group_size', 2))
        self.cross_subject_min_match_fraction = float(
            dup_cfg.get('cross_subject_min_match_fraction', 0.85))
        self.within_subject_min_match_fraction = float(
            dup_cfg.get('within_subject_min_match_fraction', 0.85))
        self.within_subject_min_form_vars = int(
            dup_cfg.get('within_subject_min_form_vars', 8))
        self.excluded_source_form_patterns = tuple(
            dup_cfg.get(
                'excluded_source_form_patterns',
                list(_DEFAULT_EXCLUDED_FORM_PATTERNS)))
        self.excluded_variable_patterns = tuple(
            dup_cfg.get(
                'excluded_variable_patterns',
                list(_DEFAULT_EXCLUDED_VARIABLE_PATTERNS)))
        cap = dup_cfg.get('max_flags_to_write', None)
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
            'cross_subject_groups_considered': 0,
            'cross_subject_groups_skipped_too_few_cols': 0,
            'cross_subject_duplicate_groups': 0,
            'cross_subject_records_emitted': 0,
            'within_subject_pairs_considered': 0,
            'within_subject_pairs_skipped_too_few_cols': 0,
            'within_subject_duplicate_pairs': 0,
            'within_subject_records_emitted': 0,
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

        records = []
        records.extend(self._cross_subject_pass(scoring))
        records.extend(self._within_subject_pass(scoring))

        if not records:
            return self._empty_output_df()
        df = pd.DataFrame(records)
        df = df.sort_values(
            'severity_score', ascending=False).reset_index(drop=True)
        if (self.max_flags_to_write is not None
                and len(df) > self.max_flags_to_write):
            df = df.head(self.max_flags_to_write).copy()
        return self._format_output(df)

    # ---- Pass 1: cross-subject, same-tp, same-site fingerprint ----

    def _cross_subject_pass(self, scoring: pd.DataFrame) -> list:
        """
        For each (network, site_prefix, source_form, timepoint),
        build a fingerprint per subject from up to dup_cols_cap of
        the form's variables (ordered alphabetically for
        determinism). Subjects whose fingerprints collide form a
        duplicate group.
        """
        today = str(datetime.today().date())
        records = []
        sc = scoring.copy()
        sc['site_prefix'] = (
            sc['subjectid'].astype(str).str[:2])
        for (net, site, form, tp), grp in sc.groupby(
                ['network', 'site_prefix', 'source_form',
                 'timepoint'], sort=False):
            self._counters['cross_subject_groups_considered'] += 1
            if not form:
                continue
            var_list = sorted(grp['variable'].unique().tolist())
            if len(var_list) < self.min_cols_for_fingerprint:
                self._counters[
                    'cross_subject_groups_skipped_too_few_cols'] += 1
                continue
            cols_use = var_list[: self.dup_cols_cap]
            wide = grp.pivot_table(
                index='subjectid', columns='variable',
                values='value_numeric', aggfunc='first',
            ).reindex(columns=cols_use)
            # Require subject to have ALL cols_use present —
            # otherwise the fingerprint isn't comparable.
            complete = wide.dropna(how='any')
            if len(complete) < self.cross_subject_min_group_size:
                continue
            fps = {}
            for subj in complete.index:
                vals = complete.loc[subj, cols_use].to_numpy(
                    dtype=float)
                fp = _fingerprint_values(vals)
                fps.setdefault(fp, []).append(str(subj))
            for fp, subjects in fps.items():
                if len(subjects) < self.cross_subject_min_group_size:
                    continue
                self._counters[
                    'cross_subject_duplicate_groups'] += 1
                for subj in subjects:
                    partners = [s for s in subjects if s != subj]
                    self._counters[
                        'cross_subject_records_emitted'] += 1
                    records.append({
                        'subjectid': subj,
                        'network': str(net),
                        'timepoint': str(tp),
                        'variable': '(row-level)',
                        'source_form': str(form),
                        'value': (
                            f"site={site},form={form},cols={len(cols_use)}"
                        ),
                        'value_numeric': float('nan'),
                        'expected_value': float('nan'),
                        'observed_value': float('nan'),
                        'algorithm': 'cross_subject_fingerprint',
                        'metric_name': 'duplicate_group_size',
                        'metric_value': float(len(subjects)),
                        'severity_score': float(
                            50.0 + 25.0 * min(len(subjects) - 1, 2)),
                        'threshold': float(
                            self.cross_subject_min_group_size),
                        'error_message': (
                            f"Cross-subject duplicate: {len(subjects)} "
                            f"subjects at site {site}, form {form}, "
                            f"timepoint {tp} have identical numeric "
                            f"values across {len(cols_use)} compared "
                            f"variables (partners: "
                            f"{','.join(partners[:5])}"
                            f"{'...' if len(partners) > 5 else ''})"
                        ),
                        'dates_detected': today,
                        'duplicate_method': 'cross_subject_fingerprint',
                        'group_id': str(fp),
                        'group_size': float(len(subjects)),
                        'match_fraction': 1.0,
                        'partner_subjectids': ','.join(partners),
                        'compared_variables_count': int(len(cols_use)),
                    })
        return records

    # ---- Pass 2: within-subject, cross-tp form-level near-dup -----

    def _within_subject_pass(self, scoring: pd.DataFrame) -> list:
        """
        For each (network, source_form, subject), walk consecutive
        canonical timepoints. For each adjacent (tp_a, tp_b) pair,
        compute fraction of joint variables with identical numeric
        values. Flag pairs at or above
        within_subject_min_match_fraction.

        Note: we use ALL canonical timepoint adjacencies, not just
        observed ones, because copy-forward between non-adjacent
        observed visits (e.g. baseline → m_2 when m_1 wasn't done)
        is the same RA behavior. The check still requires both
        timepoints to have observations on this form.
        """
        today = str(datetime.today().date())
        records = []
        for (net, form, subj), grp in scoring.groupby(
                ['network', 'source_form', 'subjectid'], sort=False):
            if not form:
                continue
            wide = grp.pivot_table(
                index='timepoint', columns='variable',
                values='value_numeric', aggfunc='first',
            )
            # Sort timepoints by canonical index. Unrecognized tps
            # were already filtered upstream.
            wide = wide.assign(
                _tp_idx=wide.index.map(self._tp_index)
            ).sort_values('_tp_idx')
            tps = wide.index.tolist()
            tp_idxs = wide['_tp_idx'].tolist()
            wide = wide.drop(columns=['_tp_idx'])
            if len(tps) < 2:
                continue
            for i in range(len(tps) - 1):
                tp_a = tps[i]
                tp_b = tps[i + 1]
                self._counters[
                    'within_subject_pairs_considered'] += 1
                # Optional: only canonically-adjacent visits. The
                # producer-side filter has already enforced canonical
                # tps; we allow non-adjacent so a baseline→m_4 copy
                # (with m_1/m_2/m_3 missing) still surfaces.
                row_a = wide.loc[tp_a]
                row_b = wide.loc[tp_b]
                joint = pd.concat(
                    [row_a.rename('a'), row_b.rename('b')], axis=1
                ).dropna()
                if len(joint) < self.within_subject_min_form_vars:
                    self._counters[
                        'within_subject_pairs_skipped_too_few_cols'] += 1
                    continue
                matches = (joint['a'] == joint['b']).sum()
                frac = float(matches) / float(len(joint))
                if frac < self.within_subject_min_match_fraction:
                    continue
                self._counters[
                    'within_subject_duplicate_pairs'] += 1
                self._counters[
                    'within_subject_records_emitted'] += 1
                identical_vars = joint[
                    joint['a'] == joint['b']
                ].index.tolist()
                records.append({
                    'subjectid': str(subj),
                    'network': str(net),
                    'timepoint': str(tp_b),
                    'variable': '(row-level)',
                    'source_form': str(form),
                    'value': (
                        f"form={form},tp_pair={tp_a}->{tp_b},"
                        f"matches={matches}/{len(joint)}"
                    ),
                    'value_numeric': float('nan'),
                    'expected_value': float('nan'),
                    'observed_value': float('nan'),
                    'algorithm': 'within_subject_form_copy',
                    'metric_name': 'within_subject_match_fraction',
                    'metric_value': frac,
                    'severity_score': float(
                        50.0 + 50.0 * frac),
                    'threshold': float(
                        self.within_subject_min_match_fraction),
                    'error_message': (
                        f"Within-subject form copy: {matches} of "
                        f"{len(joint)} joint numeric variables on form "
                        f"{form} are bit-identical between {tp_a} and "
                        f"{tp_b} ({frac:.1%} match — threshold "
                        f"{self.within_subject_min_match_fraction:.0%})"
                    ),
                    'dates_detected': today,
                    'duplicate_method': 'within_subject_form_copy',
                    'group_id': f"{subj}|{form}|{tp_a}->{tp_b}",
                    'group_size': 2.0,
                    'match_fraction': frac,
                    'partner_subjectids': '',
                    'compared_variables_count': int(len(joint)),
                })
        return records

    # ---- Output formatting ----------------------------------------

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        for col in (
            'subjectid', 'network', 'timepoint', 'variable',
            'source_form', 'value', 'algorithm', 'metric_name',
            'error_message', 'dates_detected', 'duplicate_method',
            'group_id', 'partner_subjectids',
        ):
            out[col] = df[col].astype(str)
        for col in (
            'value_numeric', 'expected_value', 'observed_value',
            'metric_value', 'severity_score', 'threshold',
            'group_size', 'match_fraction',
        ):
            out[col] = df[col].astype(float)
        out['severity_normalized'] = df['severity_score'].apply(
            normalize_passthrough_severity).astype(float)
        out['compared_variables_count'] = (
            df['compared_variables_count'].astype('int64'))
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = out['variable']
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
            'duplicate_method': pd.Series(dtype='object'),
            'group_id': pd.Series(dtype='object'),
            'group_size': pd.Series(dtype='float64'),
            'match_fraction': pd.Series(dtype='float64'),
            'partner_subjectids': pd.Series(dtype='object'),
            'compared_variables_count': pd.Series(dtype='int64'),
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
            f"{out_dir}/duplicate_record_candidates.parquet")
        self._atomic_write_parquet(df, final_path)
        if len(df) == 0:
            print(
                f"[duplicate_record_checks] no duplicate candidates "
                f"found → {final_path}"
            )
        else:
            top = df.head(5)[
                ['subjectid', 'duplicate_method', 'source_form',
                 'timepoint', 'severity_score']]
            print(
                f"[duplicate_record_checks] {len(df)} duplicate "
                f"records → {final_path}"
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
            ("cross-subj (net,site,form,tp) groups considered",
             c['cross_subject_groups_considered']),
            ("cross-subj groups skipped — <min cols",
             c['cross_subject_groups_skipped_too_few_cols']),
            ("cross-subj duplicate groups found",
             c['cross_subject_duplicate_groups']),
            ("cross-subj records emitted",
             c['cross_subject_records_emitted']),
            ("within-subj tp-pairs considered",
             c['within_subject_pairs_considered']),
            ("within-subj pairs skipped — <min cols",
             c['within_subject_pairs_skipped_too_few_cols']),
            ("within-subj duplicate pairs found",
             c['within_subject_duplicate_pairs']),
            ("within-subj records emitted",
             c['within_subject_records_emitted']),
            ("flags written (post-cap)", c['flags_written']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[duplicate_record_checks] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")

    def _atomic_write_parquet(
        self, df: pd.DataFrame, final_path: str
    ):
        atomic_write_parquet(df, final_path)


if __name__ == '__main__':
    DuplicateRecordChecks()()
