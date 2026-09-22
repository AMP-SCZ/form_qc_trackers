"""
Per-site scale-relationship correlation drift detector
(Layer 2 / discovery, default-off).

PROBLEM:
  A site can produce in-range, distribution-normal values that still
  contain a subtle scoring error: the relationship between two
  conceptually-linked variables is broken. Example: at site X, BPRS
  positive subscale and BPRS negative subscale are uncorrelated,
  whereas the rest of the network shows the expected positive
  correlation. This pattern is invisible to per-row range checks AND
  invisible to Feature 1 (which only looks at marginal distributions).

ALGORITHM (deterministic, no ML):
  Site is derived from `subjectid[:2]`.

  Input: dependencies/expected_scale_relationships.json containing
  a "pairs" array. Each entry:
    {
      "var_a": "...",
      "var_b": "...",
      "direction": "positive" | "negative",   (informational)
      "min_ref_rho": 0.2,                      (optional override)
      "cohorts": ["chr"],                      (optional restriction)
      "tps": ["baseline", "month_6"],          (optional restriction)
      "label": "BPRS positive vs negative subscale"  (optional)
    }

  For each pair × (cohort, tp):
    1. Pivot the long parquet to wide on (subjectid, var_a, var_b)
       so each row holds both values for one subject.
    2. Per site, compute Spearman ρ (pair-complete cases only) and
       sample size n.
    3. Restrict to "reference-eligible" sites where n >= min_n_per_cell
       AND the per-site Spearman is computable (>=4 distinct values
       per variable).
    4. Require at least 3 reference-eligible sites; otherwise emit
       audit rows only.
    5. For each eligible site, compute the leave-one-out peer
       reference: median of other eligible sites' ρ values, plus an
       expected-sampling-SD of Fisher-z under H0:
           fisher_z_sd ≈ 1.0 / sqrt(min_neighbor_n - 3)
       where min_neighbor_n = min of (this site's n, peer median n).
       This is the canonical Fisher-z SE that makes |Δfisher_z|
       comparable across sites with different sample sizes.
    6. Convert site_rho and reference_rho to Fisher z; report Δz =
       site_z − reference_z and standardized z = Δz / fisher_z_sd.
    7. Flag when |Δfisher_z| > delta_fisher_z_threshold (default 0.4)
       AND |reference_rho| > min_ref_rho (default 0.2). The
       reference-rho floor protects against flagging when the peer
       relationship is too weak to be reliable.

WHAT THIS CATCHES:
  - Sites where subscale ↔ total relationships are broken (e.g.
    BPRS positive subscale doesn't predict BPRS total).
  - Sites where two related symptom dimensions are de-coupled
    (e.g. GAF ↔ BPRS at one site shows ρ ≈ 0 while peers ρ ≈ -0.5).
  - Mid-study rater changes that affect one subscale but not others.

FALSE-POSITIVE CONTROLS:
  1. The curated-pair list is the gate — we never test all-pairs
     (which would be NHST flood).
  2. The reference-rho floor (default 0.2) protects against flagging
     when peer ρ is too weak to call the relationship "real".
  3. Fisher-z standardization makes the test scale-free across cells
     with different sample sizes.
  4. Stratification by (cohort, tp) keeps HC and CHR populations
     separate.
  5. Minimum site-n guard same as Feature 1.

OUTPUT (atomic, tmp + os.replace):
  combined_outputs/discovery/site_correlation_drift.parquet
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

parent_dir = "/".join(
    os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_types.discovery._common import (
    is_finite_numeric_mask,
    atomic_write_parquet,
    normalize_zscore_severity,
)


_NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})


OUTPUT_COLUMNS = [
    # Audit columns.
    'subjectid', 'network', 'timepoint', 'variable',
    'source_form', 'value', 'value_numeric',
    'expected_value', 'observed_value',
    'algorithm', 'metric_name', 'metric_value',
    'severity_score', 'severity_normalized', 'threshold',
    'error_message', 'dates_detected',
    # Detector-specific (10).
    'site', 'cohort', 'pair_label', 'var_a', 'var_b',
    'site_rho', 'site_n', 'ref_rho', 'ref_n',
    'site_fisher_z', 'ref_fisher_z', 'delta_fisher_z',
    'z_delta_fisher', 'is_audit_only',
    # Evidence timepoints metadata.
    'evidence_timepoints', 'evidence_max_timepoint',
    # Tracker-compatible aliases.
    'subject', 'displayed_variable', 'displayed_timepoint',
    'affected_variables', 'affected_timepoints',
    'affected_forms', 'displayed_form',
]


REQUIRED_INPUT_COLUMNS = [
    'subjectid', 'network', 'timepoint', 'cohort', 'variable',
    'source_form', 'value', 'value_numeric', 'is_missing_code',
]


def _fisher_z(rho: float) -> float:
    """Fisher z-transform. Clips rho to (-0.999, 0.999) for stability."""
    if rho is None or np.isnan(rho):
        return float('nan')
    r = float(rho)
    r = max(min(r, 0.999), -0.999)
    return 0.5 * float(np.log((1.0 + r) / (1.0 - r)))


class SiteCorrelationDriftChecks:
    """
    Per-site scale-relationship correlation drift detector.
    Default-off via config.discovery.site_correlation_drift.enabled.
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
            'discovery', {}).get('site_correlation_drift', {})
        self.networks = list(
            cfg.get('networks', ['PRONET', 'PRESCIENT']))
        self.min_n_per_cell = int(cfg.get('min_n_per_cell', 30))
        self.delta_fisher_z_threshold = float(
            cfg.get('delta_fisher_z_threshold', 0.4))
        self.default_min_ref_rho = float(
            cfg.get('default_min_ref_rho', 0.2))
        cap = cfg.get('max_flags_to_write', None)
        self.max_flags_to_write = (
            int(cap) if cap is not None else None)
        self.max_audit_rows = int(
            cfg.get('max_audit_rows', 5000))
        # Pairs may come from config or from a JSON dependency file.
        self._pairs = self._load_pairs(cfg)

        # Canonical timepoint order (Patch 6 convention).
        if hasattr(self.utils, 'create_timepoint_list'):
            try:
                self._tp_order = list(
                    self.utils.create_timepoint_list())
            except Exception:
                self._tp_order = []
        else:
            self._tp_order = []
        self._tp_index = {
            tp: i for i, tp in enumerate(self._tp_order)
        }

        self._counters = self._fresh_counters()

    def _fresh_counters(self) -> dict:
        return {
            'input_rows': 0,
            'valid_scoring_rows': 0,
            'pairs_loaded': len(self._pairs),
            'cells_considered': 0,
            'cells_skipped_too_few_eligible_sites': 0,
            'cells_skipped_weak_reference_rho': 0,
            'flags_emitted': 0,
            'audit_rows_emitted': 0,
        }

    def _load_pairs(self, cfg) -> list:
        # Pairs can live in config.discovery.site_correlation_drift.pairs
        # OR (the preferred place for curated clinical content) in a
        # JSON dependency file. Config-supplied pairs override the
        # JSON file when both are present.
        pairs = cfg.get('pairs')
        if pairs is not None:
            return list(pairs)
        path = (
            f"{self.depen_path}expected_scale_relationships.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return list(data.get('pairs', []))
        except (json.JSONDecodeError, OSError):
            return []

    def __call__(self):
        return self.run()

    # ---------------------------------------------------- inputs

    def _load_inputs(self) -> pd.DataFrame:
        pieces = []
        for network in self.networks:
            path = (
                f"{self.depen_path}"
                f"multi_tp_{network}_numeric_long.parquet")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"FATAL: long-format input missing for "
                    f"network {network}: {path}.")
            df = pd.read_parquet(path)
            missing = [
                c for c in REQUIRED_INPUT_COLUMNS
                if c not in df.columns
            ]
            if missing:
                raise RuntimeError(
                    f"FATAL: long-format parquet at {path} missing "
                    f"required columns: {missing}.")
            pieces.append(df)
        if not pieces:
            return pd.DataFrame(columns=REQUIRED_INPUT_COLUMNS)
        return pd.concat(pieces, ignore_index=True)

    # ------------------------------------------------ algorithm

    def _compute_candidates(
        self, long_df: pd.DataFrame
    ) -> pd.DataFrame:
        scoring = long_df[
            long_df['value_numeric'].notna()
            & is_finite_numeric_mask(long_df['value_numeric'])
            & (~long_df['is_missing_code'])
        ].copy()
        scoring = scoring[
            ~scoring['timepoint'].isin(_NON_LONGITUDINAL_TIMEPOINTS)
        ].copy()
        self._counters['valid_scoring_rows'] = len(scoring)
        if len(scoring) == 0 or not self._pairs:
            return self._empty_output_df()

        scoring['site'] = (
            scoring['subjectid'].astype(str).str[:2])
        scoring = scoring[scoring['site'].str.len() == 2].copy()
        scoring['cohort'] = (
            scoring['cohort'].fillna('').astype(str).str.lower())

        form_map = (
            scoring.dropna(subset=['source_form'])
            .groupby('variable', sort=False)['source_form']
            .first().to_dict())

        flag_records = []
        audit_records = []
        today = str(datetime.today().date())

        for pair_spec in self._pairs:
            var_a = pair_spec.get('var_a')
            var_b = pair_spec.get('var_b')
            if not var_a or not var_b:
                continue
            pair_label = pair_spec.get(
                'label', f"{var_a} vs {var_b}")
            min_ref_rho = float(
                pair_spec.get(
                    'min_ref_rho', self.default_min_ref_rho))
            cohort_filter = pair_spec.get('cohorts')
            tp_filter = pair_spec.get('tps')

            df_a = scoring[scoring['variable'] == var_a][
                ['subjectid', 'network', 'site', 'cohort',
                 'timepoint', 'value_numeric']
            ].rename(columns={'value_numeric': 'val_a'})
            df_b = scoring[scoring['variable'] == var_b][
                ['subjectid', 'network', 'site', 'cohort',
                 'timepoint', 'value_numeric']
            ].rename(columns={'value_numeric': 'val_b'})
            joined = df_a.merge(
                df_b,
                on=['subjectid', 'network', 'site', 'cohort',
                    'timepoint'],
                how='inner')
            if len(joined) == 0:
                continue
            if cohort_filter:
                joined = joined[
                    joined['cohort'].isin(
                        [c.lower() for c in cohort_filter])]
            if tp_filter:
                joined = joined[
                    joined['timepoint'].isin(tp_filter)]
            if len(joined) == 0:
                continue

            groupby_cols = ['network', 'cohort', 'timepoint']
            for (network, cohort, timepoint), grp in (
                    joined.groupby(groupby_cols, sort=False)):
                self._counters['cells_considered'] += 1
                sites_in_cell = grp['site'].unique()
                if len(sites_in_cell) < 2:
                    continue

                # Per-site Spearman ρ and n.
                per_site = {}
                for site in sites_in_cell:
                    g = grp[grp['site'] == site]
                    n = int(len(g))
                    if n < 4:
                        per_site[site] = {
                            'n': n, 'rho': float('nan'),
                            'unique_a': 0, 'unique_b': 0,
                        }
                        continue
                    unique_a = int(g['val_a'].nunique())
                    unique_b = int(g['val_b'].nunique())
                    if unique_a < 4 or unique_b < 4:
                        # Spearman degenerates on near-constant
                        # variables — skip computation.
                        per_site[site] = {
                            'n': n, 'rho': float('nan'),
                            'unique_a': unique_a,
                            'unique_b': unique_b,
                        }
                        continue
                    rho_result = spearmanr(
                        g['val_a'].to_numpy(),
                        g['val_b'].to_numpy())
                    rho = (
                        float(rho_result.correlation)
                        if hasattr(rho_result, 'correlation')
                        else float(rho_result[0]))
                    per_site[site] = {
                        'n': n,
                        'rho': rho,
                        'unique_a': unique_a,
                        'unique_b': unique_b,
                    }

                eligible_sites = [
                    s for s, st in per_site.items()
                    if st['n'] >= self.min_n_per_cell
                    and not np.isnan(st['rho'])
                ]
                if len(eligible_sites) < 3:
                    self._counters[
                        'cells_skipped_too_few_eligible_sites'] += 1
                    # Emit audit rows for each site that had data.
                    for site, st in per_site.items():
                        if (len(audit_records)
                                < self.max_audit_rows):
                            audit_records.append(
                                self._build_record(
                                    site, network, cohort, timepoint,
                                    var_a, var_b, pair_label,
                                    st['n'], st['rho'],
                                    ref_n=0, ref_rho=float('nan'),
                                    delta_z=float('nan'),
                                    z_delta=float('nan'),
                                    severity=float('nan'),
                                    is_audit_only=True,
                                    error_message=(
                                        f"AUDIT: only "
                                        f"{len(eligible_sites)} "
                                        f"eligible peer site(s) in "
                                        f"this cell."),
                                    form_map=form_map,
                                    today=today,
                                ))
                    continue

                # Need a strong-enough peer reference (median of
                # other-site rhos must exceed min_ref_rho in magnitude).
                # If not, peer relationship is too weak; skip flag
                # emission for this cell.
                peer_rhos_for_floor = np.array([
                    per_site[s]['rho'] for s in eligible_sites
                ], dtype=float)
                global_ref_rho = float(np.median(peer_rhos_for_floor))
                if abs(global_ref_rho) < min_ref_rho:
                    self._counters[
                        'cells_skipped_weak_reference_rho'] += 1
                    continue

                # Per-site flag/audit emission.
                for site, st in per_site.items():
                    if (st['n'] < self.min_n_per_cell
                            or np.isnan(st['rho'])):
                        if (len(audit_records)
                                < self.max_audit_rows):
                            audit_records.append(
                                self._build_record(
                                    site, network, cohort, timepoint,
                                    var_a, var_b, pair_label,
                                    st['n'], st['rho'],
                                    ref_n=int(len(eligible_sites)
                                              - 1),
                                    ref_rho=global_ref_rho,
                                    delta_z=float('nan'),
                                    z_delta=float('nan'),
                                    severity=float('nan'),
                                    is_audit_only=True,
                                    error_message=(
                                        f"AUDIT: site n={st['n']} "
                                        f"< min_n_per_cell or rho "
                                        f"non-computable "
                                        f"(unique_a={st['unique_a']},"
                                        f" unique_b={st['unique_b']})."),
                                    form_map=form_map,
                                    today=today,
                                ))
                        continue

                    # LOO peer reference: median of OTHER eligible
                    # sites' rhos.
                    peer_rhos = np.array([
                        per_site[s]['rho']
                        for s in eligible_sites if s != site
                    ], dtype=float)
                    peer_ns = np.array([
                        per_site[s]['n']
                        for s in eligible_sites if s != site
                    ], dtype=int)
                    if len(peer_rhos) < 2:
                        continue
                    ref_rho = float(np.median(peer_rhos))
                    ref_n_median = int(np.median(peer_ns))

                    site_z = _fisher_z(st['rho'])
                    ref_z = _fisher_z(ref_rho)
                    delta_z = site_z - ref_z

                    # Fisher-z SE depends on the smaller of the two
                    # sample sizes.
                    min_n = max(min(st['n'], ref_n_median), 4)
                    fisher_z_sd = (
                        float(np.sqrt(2.0 / (min_n - 3.0)))
                        if min_n > 3 else float('inf'))
                    z_standardized = (
                        delta_z / fisher_z_sd
                        if fisher_z_sd > 0 else 0.0)

                    if abs(delta_z) <= self.delta_fisher_z_threshold:
                        continue

                    severity = float(abs(z_standardized))
                    error_message = (
                        f"Site {site} {pair_label} "
                        f"{cohort or '(any)'}/{timepoint}: "
                        f"rho={st['rho']:+.2f} (n={st['n']}) "
                        f"vs peer-site rho={ref_rho:+.2f} "
                        f"(n_peer median={ref_n_median}); "
                        f"Fisher Δz={delta_z:+.2f}, "
                        f"|z|={abs(z_standardized):.2f}.")
                    flag_records.append(
                        self._build_record(
                            site, network, cohort, timepoint,
                            var_a, var_b, pair_label,
                            st['n'], st['rho'],
                            ref_n=ref_n_median,
                            ref_rho=ref_rho,
                            delta_z=delta_z,
                            z_delta=z_standardized,
                            severity=severity,
                            is_audit_only=False,
                            error_message=error_message,
                            form_map=form_map,
                            today=today,
                        ))
                    self._counters['flags_emitted'] += 1

        all_records = flag_records + audit_records
        self._counters['audit_rows_emitted'] = len(audit_records)
        if not all_records:
            return self._empty_output_df()
        df = pd.DataFrame(all_records)
        df = df.sort_values(
            by=['severity_score', 'site', 'var_a', 'var_b',
                'cohort', 'timepoint'],
            ascending=[False, True, True, True, True, True],
            na_position='last').reset_index(drop=True)
        if (self.max_flags_to_write is not None
                and (df['is_audit_only'] == False).sum()
                > self.max_flags_to_write):
            keep_flags = df[df['is_audit_only'] == False].head(
                self.max_flags_to_write)
            keep_audit = df[df['is_audit_only']]
            df = pd.concat([keep_flags, keep_audit],
                           ignore_index=True)
        return self._format_output(df)

    def _build_record(
        self, site, network, cohort, timepoint, var_a, var_b,
        pair_label, site_n, site_rho, ref_n, ref_rho,
        delta_z, z_delta, severity, is_audit_only,
        error_message, form_map, today,
    ) -> dict:
        source_form = (
            form_map.get(var_a) or form_map.get(var_b) or '')
        evidence_tp = str(timepoint)
        return {
            'subjectid': '',
            'network': str(network),
            'timepoint': str(timepoint),
            'variable': str(var_a),
            'source_form': str(source_form or ''),
            'value': f"site_n={site_n}",
            'value_numeric': float('nan'),
            'expected_value': float(ref_rho)
                if not np.isnan(ref_rho) else float('nan'),
            'observed_value': float(site_rho)
                if not np.isnan(site_rho) else float('nan'),
            'algorithm': 'site_correlation_drift',
            'metric_name': 'fisher_z_delta',
            'metric_value': float(delta_z)
                if not np.isnan(delta_z) else float('nan'),
            'severity_score': severity,
            'threshold': float(self.delta_fisher_z_threshold),
            'error_message': error_message,
            'dates_detected': today,
            'site': str(site),
            'cohort': str(cohort),
            'pair_label': str(pair_label),
            'var_a': str(var_a),
            'var_b': str(var_b),
            'site_rho': float(site_rho)
                if not np.isnan(site_rho) else float('nan'),
            'site_n': int(site_n),
            'ref_rho': float(ref_rho)
                if not np.isnan(ref_rho) else float('nan'),
            'ref_n': int(ref_n),
            'site_fisher_z': _fisher_z(site_rho)
                if not np.isnan(site_rho) else float('nan'),
            'ref_fisher_z': _fisher_z(ref_rho)
                if not np.isnan(ref_rho) else float('nan'),
            'delta_fisher_z': float(delta_z)
                if not np.isnan(delta_z) else float('nan'),
            'z_delta_fisher': float(z_delta)
                if not np.isnan(z_delta) else float('nan'),
            'is_audit_only': bool(is_audit_only),
            'evidence_timepoints': evidence_tp,
            'evidence_max_timepoint': evidence_tp,
        }

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out['severity_normalized'] = out['severity_score'].apply(
            normalize_zscore_severity).astype(float)
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = (
            out['var_a'] + ',' + out['var_b'])
        out['affected_timepoints'] = out['timepoint']
        out['affected_forms'] = out['source_form']
        out['displayed_form'] = out['source_form']
        return out[OUTPUT_COLUMNS]

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
            'site': pd.Series(dtype='object'),
            'cohort': pd.Series(dtype='object'),
            'pair_label': pd.Series(dtype='object'),
            'var_a': pd.Series(dtype='object'),
            'var_b': pd.Series(dtype='object'),
            'site_rho': pd.Series(dtype='float64'),
            'site_n': pd.Series(dtype='int64'),
            'ref_rho': pd.Series(dtype='float64'),
            'ref_n': pd.Series(dtype='int64'),
            'site_fisher_z': pd.Series(dtype='float64'),
            'ref_fisher_z': pd.Series(dtype='float64'),
            'delta_fisher_z': pd.Series(dtype='float64'),
            'z_delta_fisher': pd.Series(dtype='float64'),
            'is_audit_only': pd.Series(dtype='bool'),
            'evidence_timepoints': pd.Series(dtype='object'),
            'evidence_max_timepoint': pd.Series(dtype='object'),
            'subject': pd.Series(dtype='object'),
            'displayed_variable': pd.Series(dtype='object'),
            'displayed_timepoint': pd.Series(dtype='object'),
            'affected_variables': pd.Series(dtype='object'),
            'affected_timepoints': pd.Series(dtype='object'),
            'affected_forms': pd.Series(dtype='object'),
            'displayed_form': pd.Series(dtype='object'),
        })

    # ----------------------------------------------------- write

    def _write_output(self, df: pd.DataFrame):
        out_dir = (
            f"{self.output_path}combined_outputs/discovery")
        os.makedirs(out_dir, exist_ok=True)
        final_path = (
            f"{out_dir}/site_correlation_drift.parquet")
        atomic_write_parquet(df, final_path)
        if len(df) == 0:
            print(
                f"[site_correlation_drift] no candidates → "
                f"{final_path}")
        else:
            print(
                f"[site_correlation_drift] {len(df)} rows "
                f"({(df['is_audit_only'] == False).sum()} flags, "
                f"{(df['is_audit_only'] == True).sum()} audit) → "
                f"{final_path}")

    def _print_summary(self):
        c = self._counters
        rows = [
            ("input rows", c['input_rows']),
            ("valid scoring rows", c['valid_scoring_rows']),
            ("pairs loaded", c['pairs_loaded']),
            ("cells considered", c['cells_considered']),
            ("cells skipped — too few eligible sites",
             c['cells_skipped_too_few_eligible_sites']),
            ("cells skipped — weak reference rho",
             c['cells_skipped_weak_reference_rho']),
            ("flags emitted", c['flags_emitted']),
            ("audit rows emitted", c['audit_rows_emitted']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[site_correlation_drift] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")

    def run(self):
        self._counters = self._fresh_counters()
        long_df = self._load_inputs()
        self._counters['input_rows'] = len(long_df)
        df = self._compute_candidates(long_df)
        self._write_output(df)
        self._print_summary()
