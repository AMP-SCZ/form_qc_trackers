"""
Discovery-tier multi-timepoint consistency detector (default-off).

This module produces a separate `multi_tp_consistency_candidates.parquet`
under `combined_outputs/discovery/` and does NOT append to the main
`combined_qc_flags.parquet`. It is the discovery-only counterpart of
the previously-commented production dispatch at qc_forms_main.py:87-97.

ALGORITHM (deterministic, no ML):

  Per network, read `dependencies/multi_tp_{network}_combined.csv`
  (one row per subject; columns are timepoint-suffixed copies of the
  per-form variables, e.g. `chrblood_wb1id_baseline`,
  `chrblood_wb1id_month_2`). For each subject:

    1. Within-row blood-ID/barcode duplicate
       For every pair of columns whose underlying variable name is
       in `grouped_variables.blood_vars.id_variables` ∪
       `grouped_variables.blood_vars.barcode_variables`, if both
       columns hold the same non-missing value on the same subject,
       emit one candidate. Storage-location variables (freezer,
       rack, box) are excluded — those are intentionally shared
       across many subjects' vials by design and would all collide.

    2. FIGS/PPS parental-age consistency
       For the two protocol-defined pairs
         chrpps_mage_baseline    ↔ chrfigs_mother_age_screening
         chrpps_fage_baseline    ↔ chrfigs_father_age_screening
       compare the two values across forms. If both are numeric,
       non-missing, and unequal, emit one candidate.

OUTPUT (atomic, tmp + os.replace):
  combined_outputs/discovery/multi_tp_consistency_candidates.parquet

INPUT (read-only):
  dependencies/multi_tp_{network}_combined.csv
  dependencies/grouped_variables.json

CONSTRAINTS HONORED:
  - Writes only under combined_outputs/discovery/.
  - Does NOT append to combined_qc_flags.parquet.
  - Does NOT inherit from FormCheck (avoids loading the full
    production form_check_info — keeps the discovery runner usable
    when only multi_tp_*_combined.csv and grouped_variables.json
    are present).
  - Atomic write via PID-stamped tmp + os.replace.
  - Default-off via config.discovery.multi_tp_consistency.enabled.
"""

import json
import os
import sys
from datetime import datetime

import pandas as pd

parent_dir = "/".join(
    os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_types.discovery._common import (
    atomic_write_parquet,
    normalize_zscore_severity,
)


# Mirrors qc_types/fluid_checks.py:_SHARED_LOCATION_VARS. Storage-
# location IDs are shared across many subjects' vials by design so
# they will always collide; the production blood-duplicate check
# excludes them and so must this discovery wrapper.
_SHARED_LOCATION_VARS = frozenset({
    'chrblood_freezerid',
    'chrblood_rack_barcode',
    'chrblood_bc1box',
})

# Two protocol-defined pairs of cross-form parental-age variables.
# Mirrors the dict literal previously hard-coded in
# qc_types/multi_tp_checks.py:check_checks.
_FIGS_PPS_AGE_PAIRS = (
    ('chrpps_mage_baseline', 'chrfigs_mother_age_screening'),
    ('chrpps_fage_baseline', 'chrfigs_father_age_screening'),
)


_MISSING_CODE_STRINGS = frozenset({
    '', '-3', '-9', '-99', '999', '999.0',
    '1909-09-09', '1903-03-03', '1901-01-01',
    'NA', 'N/A', 'NaN', 'nan',
})


def _is_missing(val) -> bool:
    """
    Discovery-tier missing-code detector. Conservative: treat the
    Utils.missing_code_set members as missing plus NaN/None plus
    empty whitespace. We don't want to import Utils.missing_code_set
    because that ties this module to Utils() instantiation timing.
    """
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if s == '':
        return True
    return s in _MISSING_CODE_STRINGS


def _can_be_float(val) -> bool:
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


OUTPUT_COLUMNS = [
    # Audit columns (mirror discovery convention).
    'subjectid', 'network', 'timepoint', 'variable',
    'source_form', 'value', 'value_numeric',
    'expected_value', 'observed_value',
    'algorithm', 'metric_name', 'metric_value',
    'severity_score', 'severity_normalized', 'threshold',
    'error_message', 'dates_detected',
    # Tracker-compatible aliases.
    'subject', 'displayed_variable', 'displayed_timepoint',
    'affected_variables', 'affected_timepoints',
    'affected_forms', 'displayed_form',
    # Discovery-specific evidence list (paired variable name).
    'paired_variable',
]


class MultiTPConsistencyChecks:
    """
    Discovery-tier multi-timepoint consistency detector. Defaults off
    via config.discovery.multi_tp_consistency.enabled.
    """

    def __init__(self, config_dict=None):
        self.utils = Utils()
        if config_dict is None:
            with open(
                f'{self.utils.absolute_path}/config.json',
                'r',
            ) as f:
                config_dict = json.load(f)
        self.config_info = config_dict
        self.depen_path = self.config_info['paths']['dependencies_path']
        self.output_path = self.config_info['paths']['output_path']
        if str(self.config_info.get('testing_enabled', '')) == "True":
            self.output_path = self.output_path + "testing/"

        cfg = self.config_info.get(
            'discovery', {}).get('multi_tp_consistency', {})
        self.networks = list(
            cfg.get('networks', ['PRONET', 'PRESCIENT']))

        self._counters = self._fresh_counters()

    def _fresh_counters(self) -> dict:
        return {
            'subjects_scanned': 0,
            'blood_dup_candidates': 0,
            'figs_pps_age_candidates': 0,
            'candidates_written': 0,
        }

    def __call__(self):
        return self.run()

    # ------------------------------------------------------- inputs

    def _load_per_subject_vars(self) -> frozenset:
        """
        Read dependencies/grouped_variables.json and return the
        deduplicated set of per-vial id + barcode variables,
        excluding the shared-location vars. Mirrors the FluidChecks
        cross-subject blood-duplicate-check's `per_subject_vars`
        construction.
        """
        path = f"{self.depen_path}grouped_variables.json"
        with open(path, 'r', encoding='utf-8') as f:
            grouped = json.load(f)
        blood = grouped.get('blood_vars', {})
        ids = set(blood.get('id_variables', []))
        barcodes = set(blood.get('barcode_variables', []))
        return frozenset((ids | barcodes) - _SHARED_LOCATION_VARS)

    def _load_network_combined(self, network: str) -> pd.DataFrame:
        """
        Read dependencies/multi_tp_{network}_combined.csv with
        strict parsing — let the producer's data quality issues
        surface loudly rather than silently dropping rows. Empty
        cells stay as empty strings (no NaN coercion) so the
        missing-code logic treats them consistently.
        """
        path = (f"{self.depen_path}"
                f"multi_tp_{network}_combined.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FATAL: multi-tp combined CSV missing for network "
                f"{network}: {path}. Run "
                f"process_variables.collect_multi_timepoint_data "
                f"to produce it.")
        return pd.read_csv(
            path,
            keep_default_na=False,
            encoding='utf-8',
            on_bad_lines='error',
            dtype=str,
        )

    # ----------------------------------------------------- detectors

    def _check_blood_id_within_row(
        self, subjectid: str, network: str,
        row_dict: dict, per_subject_vars: frozenset,
    ) -> list:
        """
        Within-row duplicate detection across blood ID/barcode
        columns. row_dict maps full-suffixed column name (e.g.
        `chrblood_wb1id_baseline`) to the cell value. A column
        contributes if and only if its underlying variable name
        (the longest known prefix in per_subject_vars that matches
        the start of the column name) is in per_subject_vars.

        Emits one candidate per unordered (col_a, col_b) pair where
        both hold the same non-missing value.
        """
        candidates = []
        # Pre-compute (var, full_col, val) tuples for the eligible
        # columns. Skip missing / shared-location vars / wrong vars.
        eligible = []
        for col, val in row_dict.items():
            # The multi_tp_*_combined.csv column names are
            # <base_var>_<tp>; the underlying variable is the
            # longest known prefix. Use a simple linear scan over
            # per_subject_vars to identify the base var (cheap;
            # per_subject_vars has ~14-20 entries).
            base_var = None
            for var in per_subject_vars:
                if col == var or col.startswith(var + '_'):
                    if base_var is None or len(var) > len(base_var):
                        base_var = var
            if base_var is None:
                continue
            if _is_missing(val):
                continue
            eligible.append((base_var, col, str(val).strip()))

        seen_pairs = set()
        for i in range(len(eligible)):
            base_i, col_i, val_i = eligible[i]
            for j in range(i + 1, len(eligible)):
                base_j, col_j, val_j = eligible[j]
                if val_i != val_j:
                    continue
                pair = tuple(sorted((col_i, col_j)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                error_message = (
                    f"Within-row duplicate blood ID/barcode: "
                    f"{pair[0]} = {val_i} and "
                    f"{pair[1]} = {val_j} on subject "
                    f"{subjectid}.")
                candidates.append({
                    'subjectid': subjectid,
                    'network': network,
                    # Multi-tp checks span timepoints; use a
                    # synthetic label like other discovery
                    # detectors that don't pin a single tp.
                    'timepoint': '(multi_tp)',
                    'variable': pair[0],
                    'source_form':
                        'blood_sample_preanalytic_quality_assurance',
                    'value': val_i,
                    'value_numeric': float('nan'),
                    'expected_value': float('nan'),
                    'observed_value': float('nan'),
                    'algorithm': 'within_row_blood_id_duplicate',
                    'metric_name': 'identical_value_match',
                    'metric_value': 1.0,
                    'severity_score': 100.0,
                    'threshold': 1.0,
                    'error_message': error_message,
                    'paired_variable': pair[1],
                })
        return candidates

    def _check_figs_pps_age(
        self, subjectid: str, network: str, row_dict: dict,
    ) -> list:
        """
        Cross-form parental-age consistency. The two pairs live on
        different forms (PPS at baseline, FIGS at screening), so
        the multi_tp_combined CSV is the right place to compare
        them. Both values must be numeric and non-missing; if they
        differ, emit one candidate per pair.
        """
        candidates = []
        for pps_var, figs_var in _FIGS_PPS_AGE_PAIRS:
            pps_val = row_dict.get(pps_var)
            figs_val = row_dict.get(figs_var)
            if pps_val is None or figs_val is None:
                continue
            if _is_missing(pps_val) or _is_missing(figs_val):
                continue
            if not (_can_be_float(pps_val)
                    and _can_be_float(figs_val)):
                continue
            pps_num = float(pps_val)
            figs_num = float(figs_val)
            if pps_num == figs_num:
                continue
            error_message = (
                f"{pps_var} is {pps_val}, but {figs_var} is "
                f"{figs_val}.")
            candidates.append({
                'subjectid': subjectid,
                'network': network,
                'timepoint': '(multi_tp)',
                'variable': pps_var,
                'source_form':
                    'family_interview_for_genetic_studies_figs',
                'value': str(pps_val),
                'value_numeric': pps_num,
                'expected_value': figs_num,
                'observed_value': pps_num,
                'algorithm': 'figs_pps_age_mismatch',
                'metric_name': 'absolute_age_difference_years',
                'metric_value': float(abs(pps_num - figs_num)),
                'severity_score': float(abs(pps_num - figs_num)),
                'threshold': 0.0,
                'error_message': error_message,
                'paired_variable': figs_var,
            })
        return candidates

    # -------------------------------------------------------- driver

    def _compute(self) -> pd.DataFrame:
        per_subject_vars = self._load_per_subject_vars()
        records = []
        for network in self.networks:
            try:
                df = self._load_network_combined(network)
            except FileNotFoundError as e:
                # A discovery runner that can run from incomplete
                # state is more useful than one that aborts on a
                # missing per-network artifact. Print and skip.
                print(
                    f"[multi_tp_consistency_checks] WARNING: "
                    f"skipping network {network} — {e}")
                continue
            if 'subjectid' not in df.columns:
                print(
                    f"[multi_tp_consistency_checks] WARNING: "
                    f"network {network} CSV has no subjectid "
                    f"column; skipping.")
                continue
            for row in df.itertuples(index=False):
                self._counters['subjects_scanned'] += 1
                row_dict = row._asdict()
                subjectid = row_dict.get('subjectid', '')
                blood_cands = self._check_blood_id_within_row(
                    subjectid, network, row_dict, per_subject_vars)
                self._counters['blood_dup_candidates'] += len(
                    blood_cands)
                age_cands = self._check_figs_pps_age(
                    subjectid, network, row_dict)
                self._counters['figs_pps_age_candidates'] += len(
                    age_cands)
                records.extend(blood_cands)
                records.extend(age_cands)
        return self._format_output(records)

    def _format_output(self, records: list) -> pd.DataFrame:
        today = str(datetime.today().date())
        if not records:
            return self._empty_output_df()
        df = pd.DataFrame(records)
        df['dates_detected'] = today
        # severity_normalized — Sprint 1 P0-7. Without this column the
        # cross-detector subject_summary aggregator null-coalesces to
        # 0.0, ranking every multi_tp_consistency flag dead-last.
        # Blood-ID duplicates already publish severity on a 0-100 scale
        # (fixed at 100); FIGS/PPS age mismatches publish absolute
        # years of difference (typically 0-50). Use the same
        # zscore-shaped mapping as the other discovery detectors so a
        # 5-year FIGS/PPS gap (severity=5) maps to ~50 and a blood-ID
        # duplicate (severity=100) caps at 100.
        df['severity_normalized'] = df['severity_score'].apply(
            normalize_zscore_severity).astype(float)
        df['subject'] = df['subjectid']
        df['displayed_variable'] = df['variable']
        df['displayed_timepoint'] = df['timepoint']
        df['affected_variables'] = df['variable']
        df['affected_timepoints'] = df['timepoint']
        df['affected_forms'] = df['source_form']
        df['displayed_form'] = df['source_form']
        return df[OUTPUT_COLUMNS]

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
            'subject': pd.Series(dtype='object'),
            'displayed_variable': pd.Series(dtype='object'),
            'displayed_timepoint': pd.Series(dtype='object'),
            'affected_variables': pd.Series(dtype='object'),
            'affected_timepoints': pd.Series(dtype='object'),
            'affected_forms': pd.Series(dtype='object'),
            'displayed_form': pd.Series(dtype='object'),
            'paired_variable': pd.Series(dtype='object'),
        })

    # --------------------------------------------------------- write

    def _write_output(self, df: pd.DataFrame):
        out_dir = (
            f"{self.output_path}combined_outputs/discovery")
        os.makedirs(out_dir, exist_ok=True)
        final_path = (
            f"{out_dir}/multi_tp_consistency_candidates.parquet")
        atomic_write_parquet(df, final_path)
        if len(df) == 0:
            print(
                f"[multi_tp_consistency_checks] no candidates → "
                f"{final_path}")
        else:
            print(
                f"[multi_tp_consistency_checks] {len(df)} "
                f"candidates → {final_path}")

    def _print_summary(self):
        c = self._counters
        rows = [
            ("subjects scanned", c['subjects_scanned']),
            ("blood-ID/barcode duplicate candidates",
             c['blood_dup_candidates']),
            ("FIGS/PPS age-mismatch candidates",
             c['figs_pps_age_candidates']),
            ("candidates written", c['candidates_written']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[multi_tp_consistency_checks] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")

    def run(self):
        self._counters = self._fresh_counters()
        df = self._compute()
        self._counters['candidates_written'] = len(df)
        self._write_output(df)
        self._print_summary()
