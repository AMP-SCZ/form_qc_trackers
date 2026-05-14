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
Missingness anomaly summary (Layer 2 / discovery, default-off).

Uses the same long-format numeric parquet as N1 / copy-forward / delta.
Respects project missing codes via the producer's `is_missing_code` flag
only — never re-encodes raw cells here.

Two complementary views:
  1) variable_rate — among variables at a (network, timepoint), flag
     variables whose missing-code rate is an outlier vs other variables
     (high missingness on one field vs siblings at the same visit wave).
  2) site_rate — among site prefixes (subjectid first two characters)
     at a (network, timepoint), flag sites whose aggregate missingness
     rate is an outlier vs other sites (operational / training issue).

INPUT (read-only):
  dependencies/multi_tp_{network}_numeric_long.parquet

OUTPUT (atomic tmp + os.replace):
  combined_outputs/discovery/missingness_anomaly_summary.parquet

Does NOT modify source data or combined_qc_flags.parquet.
"""


def _robust_scale(rates: np.ndarray, med: float) -> float:
    """
    Primary scale: MAD × 1.4826. If most rates are identical, MAD
    collapses to ~0 — fall back to IQR/1.349, then std, then epsilon
    so a single outlier rate can still produce a finite z-score.
    """
    mad = float(np.median(np.abs(rates - med)) * 1.4826)
    if mad > 1e-12:
        return mad
    q25, q75 = np.percentile(rates, [25, 75])
    iqr = float((q75 - q25) / 1.349)
    if iqr > 1e-12:
        return iqr
    std = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
    if std > 1e-12:
        return std
    return 1e-12


class MissingnessSummaryChecks:
    """Discovery-tier missingness outlier screen on long-format data."""

    OUTPUT_COLUMNS = [
        'anomaly_type', 'network', 'timepoint', 'variable', 'site_prefix',
        'source_form', 'n_rows', 'n_missing', 'missing_rate',
        'benchmark_median_rate', 'benchmark_mad_scale',
        'metric_value', 'severity_score', 'severity_normalized',
        'threshold', 'algorithm',
        'metric_name', 'error_message', 'dates_detected',
        'subject', 'displayed_variable', 'displayed_timepoint',
        'affected_variables', 'affected_timepoints', 'affected_forms',
        'displayed_form',
    ]

    REQUIRED_INPUT_COLUMNS = [
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric', 'is_missing_code',
    ]

    def __init__(self, config_dict=None):
        self.utils = Utils()
        if config_dict is None:
            with open(f'{self.utils.absolute_path}/config.json', 'r') as f:
                config_dict = json.load(f)
        self.config_info = config_dict
        self.depen_path = self.config_info['paths']['dependencies_path']
        self.output_path = self.config_info['paths']['output_path']
        if str(self.config_info.get('testing_enabled', '')) == "True":
            self.output_path = self.output_path + "testing/"

        ms = self.config_info.get('discovery', {}).get(
            'missingness_summary', {})
        self.networks = list(ms.get('networks', ['PRONET', 'PRESCIENT']))
        self.min_rows_per_slice = int(ms.get('min_rows_per_slice', 20))
        self.robust_z_threshold = float(ms.get('robust_z_threshold', 3.0))
        cap = ms.get('max_rows_to_write', 50000)
        self.max_rows_to_write = int(cap) if cap is not None else None
        self.excluded_source_form_patterns = tuple(
            ms.get(
                'excluded_source_form_patterns',
                list(DEFAULT_EXCLUDED_FORM_PATTERNS)))
        self.excluded_variable_patterns = tuple(
            ms.get(
                'excluded_variable_patterns',
                list(DEFAULT_EXCLUDED_VARIABLE_PATTERNS)))
        self._counters = self._fresh_counters()

    def _fresh_counters(self):
        return {
            'input_rows': 0,
            'rows_excluded_by_form_pattern': 0,
            'rows_excluded_by_variable_pattern': 0,
            'variable_slices': 0,
            'variable_flags': 0,
            'site_slices': 0,
            'site_flags': 0,
            'rows_written': 0,
        }

    def __call__(self):
        return self.run()

    def run(self):
        self._counters = self._fresh_counters()
        long_df = self._load_inputs()
        self._counters['input_rows'] = len(long_df)
        out = self._compute(long_df)
        self._counters['rows_written'] = len(out)
        self._write_output(out)
        self._print_summary()

    def _load_inputs(self) -> pd.DataFrame:
        pieces = []
        for network in self.networks:
            path = (
                f"{self.depen_path}multi_tp_{network}_numeric_long.parquet"
            )
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"FATAL: long-format input missing for network "
                    f"{network}: {path}. Run "
                    f"process_variables.produce_multi_tp_long first."
                )
            df = pd.read_parquet(path)
            miss = [
                c for c in self.REQUIRED_INPUT_COLUMNS
                if c not in df.columns
            ]
            if miss:
                raise RuntimeError(
                    f"FATAL: {path} missing columns: {miss}"
                )
            pieces.append(df)
        if not pieces:
            return pd.DataFrame(columns=self.REQUIRED_INPUT_COLUMNS)
        return pd.concat(pieces, ignore_index=True)

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return self._empty_output()

        # Apply form / variable exclusions before the missing-rate
        # roll-ups. Otherwise digital biomarker forms and known-noisy
        # patterns dominate the variable-rate sheet.
        if self.excluded_source_form_patterns and len(df) > 0:
            before = len(df)
            mask = df['source_form'].apply(
                lambda f: is_excluded_form(
                    f, self.excluded_source_form_patterns))
            df = df[~mask].copy()
            self._counters['rows_excluded_by_form_pattern'] = (
                before - len(df))
        if self.excluded_variable_patterns and len(df) > 0:
            before = len(df)
            mask = df['variable'].apply(
                lambda v: is_excluded_variable(
                    v, self.excluded_variable_patterns))
            df = df[~mask].copy()
            self._counters['rows_excluded_by_variable_pattern'] = (
                before - len(df))
        if len(df) == 0:
            return self._empty_output()
        today = str(datetime.today().date())
        rows = []

        # --- Variable-level: missing rate per (network, tp, variable)
        vg = (
            df.groupby(['network', 'timepoint', 'variable'], sort=False)
            .agg(
                n_rows=('is_missing_code', 'size'),
                n_missing=('is_missing_code', 'sum'),
                source_form=('source_form', 'first'),
            )
            .reset_index()
        )
        vg = vg[vg['n_rows'] >= self.min_rows_per_slice]
        vg['missing_rate'] = vg['n_missing'].astype(float) / vg[
            'n_rows'].astype(float)

        for (net, tp), sub in vg.groupby(['network', 'timepoint'],
                                         sort=False):
            rates = sub['missing_rate'].to_numpy(dtype=float)
            if len(rates) < 3:
                continue
            med = float(np.median(rates))
            scale = _robust_scale(rates, med)
            self._counters['variable_slices'] += len(sub)
            for _, r in sub.iterrows():
                mr = float(r['missing_rate'])
                z = (mr - med) / scale if scale > 1e-15 else 0.0
                # Flag high missingness only (operational concern)
                if scale > 1e-15 and z >= self.robust_z_threshold and mr > med:
                    self._counters['variable_flags'] += 1
                    rows.append(self._row_dict(
                        'variable_rate', net, tp, str(r['variable']),
                        '', str(r['source_form']),
                        int(r['n_rows']), int(r['n_missing']), mr,
                        med, scale, z, abs(z), self.robust_z_threshold,
                        today,
                        f"High missing-code rate vs other variables at "
                        f"same network/timepoint (robust z={z:.2f}).",
                    ))

        # --- Site-level: aggregate by subjectid prefix
        site_df = df.copy()
        site_df['site_prefix'] = (
            site_df['subjectid'].astype(str).str[:2]
        )
        sg = (
            site_df.groupby(
                ['network', 'timepoint', 'site_prefix'], sort=False
            )
            .agg(
                n_rows=('is_missing_code', 'size'),
                n_missing=('is_missing_code', 'sum'),
            )
            .reset_index()
        )
        sg = sg[sg['n_rows'] >= self.min_rows_per_slice]
        sg['missing_rate'] = sg['n_missing'].astype(float) / sg[
            'n_rows'].astype(float)

        for (net, tp), sub in sg.groupby(['network', 'timepoint'],
                                         sort=False):
            rates = sub['missing_rate'].to_numpy(dtype=float)
            if len(rates) < 4:
                continue
            med = float(np.median(rates))
            scale = _robust_scale(rates, med)
            self._counters['site_slices'] += len(sub)
            for _, r in sub.iterrows():
                mr = float(r['missing_rate'])
                z = (mr - med) / scale if scale > 1e-15 else 0.0
                if scale > 1e-15 and z >= self.robust_z_threshold and mr > med:
                    self._counters['site_flags'] += 1
                    rows.append(self._row_dict(
                        'site_rate', net, tp, '(site_aggregate)',
                        str(r['site_prefix']), '',
                        int(r['n_rows']), int(r['n_missing']), mr,
                        med, scale, z, abs(z), self.robust_z_threshold,
                        today,
                        f"High aggregate missing-code rate vs other sites "
                        f"at same network/timepoint (robust z={z:.2f}).",
                    ))

        if not rows:
            return self._empty_output()

        out = pd.DataFrame(rows)
        out = out.sort_values(
            'severity_score', ascending=False
        ).reset_index(drop=True)
        if self.max_rows_to_write is not None:
            out = out.head(self.max_rows_to_write)
        return out

    def _row_dict(
        self, atype, net, tp, var, site_p, sform,
        n_rows, n_miss, mr, bmed, bmad, z, sev, thresh, today, err,
    ):
        return {
            'anomaly_type': atype,
            'network': net,
            'timepoint': tp,
            'variable': var,
            'site_prefix': site_p,
            'source_form': sform,
            'n_rows': n_rows,
            'n_missing': n_miss,
            'missing_rate': mr,
            'benchmark_median_rate': bmed,
            'benchmark_mad_scale': bmad,
            'metric_value': z,
            'severity_score': sev,
            'severity_normalized': normalize_zscore_severity(sev),
            'threshold': thresh,
            'algorithm': 'missingness_summary_v1',
            'metric_name': 'robust_z_excess_missing_rate',
            'error_message': err,
            'dates_detected': today,
            'subject': '',
            'displayed_variable': var,
            'displayed_timepoint': tp,
            'affected_variables': var,
            'affected_timepoints': tp,
            'affected_forms': sform,
            'displayed_form': sform,
        }

    def _empty_output(self) -> pd.DataFrame:
        """Typed empty frame for consistent parquet schema."""
        return pd.DataFrame({
            'anomaly_type': pd.Series(dtype='object'),
            'network': pd.Series(dtype='object'),
            'timepoint': pd.Series(dtype='object'),
            'variable': pd.Series(dtype='object'),
            'site_prefix': pd.Series(dtype='object'),
            'source_form': pd.Series(dtype='object'),
            'n_rows': pd.Series(dtype='int64'),
            'n_missing': pd.Series(dtype='int64'),
            'missing_rate': pd.Series(dtype='float64'),
            'benchmark_median_rate': pd.Series(dtype='float64'),
            'benchmark_mad_scale': pd.Series(dtype='float64'),
            'metric_value': pd.Series(dtype='float64'),
            'severity_score': pd.Series(dtype='float64'),
            'severity_normalized': pd.Series(dtype='float64'),
            'threshold': pd.Series(dtype='float64'),
            'algorithm': pd.Series(dtype='object'),
            'metric_name': pd.Series(dtype='object'),
            'error_message': pd.Series(dtype='object'),
            'dates_detected': pd.Series(dtype='object'),
            'subject': pd.Series(dtype='object'),
            'displayed_variable': pd.Series(dtype='object'),
            'displayed_timepoint': pd.Series(dtype='object'),
            'affected_variables': pd.Series(dtype='object'),
            'affected_timepoints': pd.Series(dtype='object'),
            'affected_forms': pd.Series(dtype='object'),
            'displayed_form': pd.Series(dtype='object'),
        })

    def _write_output(self, df: pd.DataFrame):
        out_dir = os.path.join(
            self.output_path, 'combined_outputs', 'discovery')
        final_path = os.path.join(
            out_dir, 'missingness_anomaly_summary.parquet')
        df = df.reindex(columns=self.OUTPUT_COLUMNS)
        self._atomic_write_parquet(df, final_path)
        print(
            f"[missingness_summary_checks] wrote {len(df)} rows → "
            f"{final_path}"
        )

    def _atomic_write_parquet(self, df: pd.DataFrame, final_path: str):
        atomic_write_parquet(df, final_path)

    def _print_summary(self):
        c = self._counters
        print("[missingness_summary_checks] summary:")
        print(f"  input rows                              {c['input_rows']}")
        print(f"  rows excluded by form pattern           {c['rows_excluded_by_form_pattern']}")
        print(f"  rows excluded by variable pattern       {c['rows_excluded_by_variable_pattern']}")
        print(f"  variable (net,tp,var) slices            {c['variable_slices']}")
        print(f"  variable_rate flags                     {c['variable_flags']}")
        print(f"  site (net,tp,site) slices               {c['site_slices']}")
        print(f"  site_rate flags                         {c['site_flags']}")
        print(f"  rows written (post-cap)                 {c['rows_written']}")


if __name__ == '__main__':
    MissingnessSummaryChecks()()
