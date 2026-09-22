"""
Site-level distribution-shift detector (Layer 2 / discovery, default-off).

PROBLEM:
  Every existing discovery detector groups by `(network, variable, ...)` or
  `(network, source_form, ...)` — none stratify by SITE (the 2-letter
  subjectid prefix). A site whose entire scoring distribution is shifted
  relative to its peers is therefore statistically INVISIBLE to per-subject
  detectors — every subject at that site is similarly "off" and the shift
  drowns into the per-subject median or the per-form cohort centroid.

ALGORITHM (deterministic, no ML):
  Site is derived from `subjectid[:2]`.

  For each (network, variable, cohort, timepoint) cell:
    1. Per site in the cell, compute (n, median, IQR, MAD,
       p_at_floor, p_at_ceiling) on that site's subjects.
    2. Restrict to "reference-eligible" sites where n >= min_n_per_cell.
       Sites with n < min_n_per_cell emit AUDIT rows and never
       contribute to the reference.
    3. Require at least 3 reference-eligible sites in the cell;
       otherwise skip (cell silently dropped, no flags, no audit
       rows — there's nothing comparative to say).
    4. For each reference-eligible site (LOO at site-median level):
       reference_median  = median of OTHER sites' medians
       reference_mad     = 1.4826 * median(|other_medians - reference_median|)
       reference_iqr     = median of OTHER sites' IQRs
       reference_iqr_mad = 1.4826 * median(|other_iqrs - reference_iqr|)
       z_median = (site_median - reference_median) / reference_mad
       z_iqr    = (site_iqr - reference_iqr) / reference_iqr_mad
       delta_p_at_floor   = site_p_at_floor - median(other_p_at_floors)
       delta_p_at_ceiling = site_p_at_ceiling - median(other_p_at_ceilings)
    5. Flag when any of:
         |z_median| > z_threshold_median (default 3.0)
         |z_iqr|    > z_threshold_iqr    (default 3.0)
         |delta_p_at_floor|   > delta_floor_ceiling_threshold (default 0.15)
         |delta_p_at_ceiling| > delta_floor_ceiling_threshold (default 0.15)

  Each site contributes ONE point to the reference (not its raw
  subject values), which means an outlier site cannot drag the
  reference and produce spurious flags at peer sites.

  Cells below the n threshold emit an AUDIT row (severity_score = NaN,
  no error_message) so the operator can see what was skipped without
  treating it as a flag.

  Floor / ceiling are read from dependencies/variable_ranges.json when
  available (variable: {"min": x, "max": y} per entry). Variables not
  present in variable_ranges are skipped for the floor/ceiling check
  but still receive median/IQR comparison.

WHAT THIS CATCHES:
  - Site rater systematically scoring higher or lower (median shift)
  - Site rater compressing the scale (IQR shrunk)
  - Site clustering at the maximum / minimum (ceiling/floor effects)
  - Mid-study scoring drift at a site (when run cross-tp on stable
    variables — flags persist across tps for the affected site)
  - Form-version mismatches that shift the distribution at one site
  - Onboarding-wave artifacts (new site looks unlike peers)

FALSE-POSITIVE CONTROLS (in priority order):
  1. Stratification: compares within (cohort, tp) — HC vs CHR vs HSC
     populations are NEVER pooled together. Different baselines do not
     produce flags.
  2. LOO peer-site reference is within-network (PRONET sites compared
     to PRONET sites only). Cross-network demographic differences
     (Melbourne vs USA recruitment) do not create cross-network flags.
  3. Effect-size threshold (not just p-value): z >= 3 corresponds to
     roughly one-in-thousand magnitude effects, conservative.
  4. Minimum-n guard: small sites at rare tps emit audit rows, not flag
     rows; n < 30 is opted-out by default.
  5. Robust statistics (median / IQR / MAD) are insensitive to single
     outlier subjects so the site flag really reflects systematic shift.

OUTPUT (atomic, tmp + os.replace):
  combined_outputs/discovery/site_distribution_drift.parquet

CONSTRAINTS HONORED:
  - Reads only the long-format parquet and (optionally) variable_ranges.json.
  - Writes only the single candidates parquet under
    combined_outputs/discovery/.
  - Does NOT append to combined_qc_flags.parquet.
  - Does NOT inherit from FormCheck.
  - Atomic write via PID-stamped tmp + os.replace.
  - Default-off via config.discovery.site_distribution_drift.enabled.
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

parent_dir = "/".join(
    os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_types.discovery._common import (
    DEFAULT_EXCLUDED_FORM_PATTERNS,
    DEFAULT_EXCLUDED_VARIABLE_PATTERNS,
    is_excluded_form,
    is_excluded_variable,
    is_finite_numeric_mask,
    atomic_write_parquet,
    normalize_zscore_severity,
)


# Mirrors the discovery-tier convention to keep N1/discovery aligned —
# the conversion-event measurement at floating/conversion is, by
# definition, the extreme value of interest; folding it into a site
# median would create artificial cross-site shifts.
_NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})


OUTPUT_COLUMNS = [
    # Audit columns (mirror N1 schema).
    'subjectid', 'network', 'timepoint', 'variable',
    'source_form', 'value', 'value_numeric',
    'expected_value', 'observed_value',
    'algorithm', 'metric_name', 'metric_value',
    'severity_score', 'severity_normalized', 'threshold',
    'error_message', 'dates_detected',
    # Detector-specific (12).
    'site', 'cohort', 'n_subjects',
    'site_median', 'site_iqr', 'site_mad',
    'ref_median', 'ref_iqr', 'ref_mad',
    'site_p_at_floor', 'site_p_at_ceiling',
    'ref_p_at_floor', 'ref_p_at_ceiling',
    'z_median', 'z_iqr',
    'delta_p_at_floor', 'delta_p_at_ceiling',
    'is_audit_only',
    # Evidence timepoints metadata (Patch 6 convention).
    'evidence_timepoints', 'evidence_max_timepoint',
    # Tracker-compatible aliases (7).
    'subject', 'displayed_variable', 'displayed_timepoint',
    'affected_variables', 'affected_timepoints',
    'affected_forms', 'displayed_form',
]


REQUIRED_INPUT_COLUMNS = [
    'subjectid', 'network', 'timepoint', 'cohort', 'variable',
    'source_form', 'value', 'value_numeric', 'is_missing_code',
]


class SiteDistributionDriftChecks:
    """
    Per-site distribution-shift detector. Default-off.
    Writes combined_outputs/discovery/site_distribution_drift.parquet.
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
            'discovery', {}).get('site_distribution_drift', {})
        self.networks = list(
            cfg.get('networks', ['PRONET', 'PRESCIENT']))
        self.min_n_per_cell = int(cfg.get('min_n_per_cell', 30))
        self.z_threshold_median = float(
            cfg.get('z_threshold_median', 3.0))
        self.z_threshold_iqr = float(
            cfg.get('z_threshold_iqr', 3.0))
        self.delta_floor_ceiling_threshold = float(
            cfg.get('delta_floor_ceiling_threshold', 0.15))
        self.excluded_source_form_patterns = tuple(
            cfg.get(
                'excluded_source_form_patterns',
                list(DEFAULT_EXCLUDED_FORM_PATTERNS)))
        self.excluded_variable_patterns = tuple(
            cfg.get(
                'excluded_variable_patterns',
                list(DEFAULT_EXCLUDED_VARIABLE_PATTERNS)))
        cap = cfg.get('max_flags_to_write', None)
        self.max_flags_to_write = (
            int(cap) if cap is not None else None)
        # Audit rows (sample-size guard misses) — keep them written so
        # the operator can see which (site,variable,cohort,tp) cells
        # were silently skipped. Bounded by a separate cap so the
        # audit doesn't overwhelm the flagged output.
        self.max_audit_rows = int(
            cfg.get('max_audit_rows', 5000))

        # Variable ranges for floor/ceiling detection. Optional.
        self._variable_ranges = self._load_variable_ranges()

        # Canonical timepoint order for evidence_max_timepoint
        # (Patch 6 convention).
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
            'rows_excluded_by_form_pattern': 0,
            'rows_excluded_by_variable_pattern': 0,
            'rows_excluded_non_longitudinal_tp': 0,
            'rows_excluded_missing_code': 0,
            'valid_scoring_rows': 0,
            'cells_considered': 0,
            'cells_skipped_insufficient_n': 0,
            'cells_skipped_degenerate_ref': 0,
            'flags_emitted': 0,
            'audit_rows_emitted': 0,
        }

    def _load_variable_ranges(self) -> dict:
        path = f"{self.depen_path}variable_ranges.json"
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f) or {}
        except (json.JSONDecodeError, OSError):
            # Optional file; degrade gracefully (skip floor/ceiling).
            return {}

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
                    f"network {network}: {path}. Run "
                    f"process_variables.produce_multi_tp_long first."
                )
            df = pd.read_parquet(path)
            missing = [
                c for c in REQUIRED_INPUT_COLUMNS
                if c not in df.columns
            ]
            if missing:
                raise RuntimeError(
                    f"FATAL: long-format parquet at {path} is "
                    f"missing required columns: {missing}.")
            pieces.append(df)
        if not pieces:
            return pd.DataFrame(columns=REQUIRED_INPUT_COLUMNS)
        return pd.concat(pieces, ignore_index=True)

    # ------------------------------------------------ algorithm

    def _site_from_subjectid(self, sid):
        s = str(sid)
        return s[:2] if len(s) >= 2 else ''

    def _per_site_stats(self, vals: np.ndarray,
                        floor: float, ceil: float) -> dict:
        """
        Robust stats on a 1-D numeric array. NaNs already excluded
        upstream. floor/ceil may be None when the variable has no
        range defined; in that case the floor/ceiling proportions
        are NaN.
        """
        n = int(len(vals))
        if n == 0:
            return {
                'n': 0, 'median': float('nan'),
                'iqr': float('nan'), 'mad': float('nan'),
                'p_at_floor': float('nan'),
                'p_at_ceiling': float('nan'),
            }
        med = float(np.median(vals))
        q1 = float(np.quantile(vals, 0.25))
        q3 = float(np.quantile(vals, 0.75))
        iqr = q3 - q1
        mad = float(np.median(np.abs(vals - med)))
        p_floor = (
            float(np.mean(vals <= floor))
            if floor is not None else float('nan'))
        p_ceil = (
            float(np.mean(vals >= ceil))
            if ceil is not None else float('nan'))
        return {
            'n': n, 'median': med, 'iqr': iqr, 'mad': mad,
            'p_at_floor': p_floor, 'p_at_ceiling': p_ceil,
        }

    def _loo_reference(self, other_site_stats: list,
                       site_n: int) -> dict:
        """
        Leave-one-out reference at site-median level.

        Center: median of the K-1 peer sites' medians (LOO). Robust
        to one outlier site since that site is removed from the pool
        before the reference is computed (no contamination).

        Scale: the EXPECTED sampling SD of a median computed from
        site_n samples, derived from the pooled within-site MAD.
        Specifically:
          pooled_within_sd = (median of peer-site MADs) * 1.4826
          expected_median_sd =
              pooled_within_sd * sqrt(pi / (2 * site_n))
        This is the asymptotic SD of a sample median under H0 (all
        sites draw from the same distribution). Using this rather
        than the MAD-of-medians fixes a key failure mode: with
        K=4-6 peer sites the MAD-of-medians estimate is very noisy
        and routinely flags non-shifted sites at threshold |z|>3.
        The pooled-within-site noise floor is stable across runs
        and gives proper Type-I control.

        Same idea for IQR — peer-site IQR median plus a noise floor
        derived from the pooled within-site MAD of IQRs.
        """
        if not other_site_stats:
            return None
        medians = np.array(
            [s['median'] for s in other_site_stats], dtype=float)
        iqrs = np.array(
            [s['iqr'] for s in other_site_stats], dtype=float)
        mads = np.array(
            [s['mad'] for s in other_site_stats], dtype=float)
        floors = np.array(
            [s['p_at_floor'] for s in other_site_stats], dtype=float)
        ceils = np.array(
            [s['p_at_ceiling'] for s in other_site_stats], dtype=float)

        ref_median = float(np.median(medians))
        ref_iqr = float(np.median(iqrs))
        # Pooled within-site SD = (median of peer MADs) * 1.4826.
        pooled_within_mad = float(np.median(mads))
        pooled_within_sd = pooled_within_mad * 1.4826
        # Asymptotic SD of a sample median, sqrt(pi/(2n))*sigma.
        if site_n > 0 and pooled_within_sd > 0:
            median_sd = (
                pooled_within_sd
                * float(np.sqrt(np.pi / (2.0 * site_n))))
        else:
            median_sd = 0.0
        # IQR sampling SD — approximate by 1.5 * median_sd
        # (sampling variability of IQR is typically ~1.5x that of
        # the median for normal-ish data; a fixed multiplier
        # avoids reliance on K-1 IQR points).
        iqr_sd = 1.5 * median_sd

        ref_floor = (
            float(np.nanmedian(floors))
            if not np.all(np.isnan(floors)) else float('nan'))
        ref_ceil = (
            float(np.nanmedian(ceils))
            if not np.all(np.isnan(ceils)) else float('nan'))
        return {
            'median': ref_median,
            'median_sd': median_sd,
            'iqr': ref_iqr,
            'iqr_sd': iqr_sd,
            'p_at_floor': ref_floor,
            'p_at_ceiling': ref_ceil,
            'mad': pooled_within_mad,
        }

    def _floor_ceiling(self, variable: str):
        rng = self._variable_ranges.get(variable, None)
        if not isinstance(rng, dict):
            return None, None
        # Accept either explicit {'min','max'} or REDCap-style
        # {'text_validation_min','text_validation_max'}.
        floor = rng.get('min')
        ceil = rng.get('max')
        if floor is None:
            floor = rng.get('text_validation_min')
        if ceil is None:
            ceil = rng.get('text_validation_max')
        try:
            floor = float(floor) if floor is not None else None
        except (TypeError, ValueError):
            floor = None
        try:
            ceil = float(ceil) if ceil is not None else None
        except (TypeError, ValueError):
            ceil = None
        return floor, ceil

    def _compute_candidates(
        self, long_df: pd.DataFrame
    ) -> pd.DataFrame:
        # Standard discovery-tier filters.
        scoring = long_df[
            long_df['value_numeric'].notna()
            & is_finite_numeric_mask(long_df['value_numeric'])
            & (~long_df['is_missing_code'])
        ].copy()
        self._counters['rows_excluded_missing_code'] = (
            len(long_df) - len(scoring))

        before = len(scoring)
        scoring = scoring[
            ~scoring['timepoint'].isin(_NON_LONGITUDINAL_TIMEPOINTS)
        ].copy()
        self._counters['rows_excluded_non_longitudinal_tp'] = (
            before - len(scoring))

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
            return self._empty_output_df(), pd.DataFrame()

        # Derive site from subjectid prefix. Subjects with a <2-char
        # id (shouldn't happen on real data) are dropped from the
        # site analysis.
        scoring['site'] = (
            scoring['subjectid'].astype(str).str[:2])
        scoring = scoring[scoring['site'].str.len() == 2].copy()

        # Normalize cohort to a clean string for groupby stability.
        scoring['cohort'] = (
            scoring['cohort'].fillna('').astype(str).str.lower())

        # Pick a single source_form per variable for the output
        # display (the first observed). Cohort-pooled.
        form_map = (
            scoring.dropna(subset=['source_form'])
            .groupby('variable', sort=False)['source_form']
            .first().to_dict())

        flag_records = []
        audit_records = []
        today = str(datetime.today().date())

        # Iterate cells: (network, variable, cohort, timepoint).
        groupby_cols = ['network', 'variable', 'cohort', 'timepoint']
        for (network, variable, cohort, timepoint), grp in (
                scoring.groupby(groupby_cols, sort=False)):
            self._counters['cells_considered'] += 1
            floor, ceil = self._floor_ceiling(variable)

            # One row per (subjectid, site) per cell — collapse
            # producer dupes deterministically.
            grp = grp.drop_duplicates(
                subset=['subjectid'], keep='first')

            sites_in_cell = grp['site'].unique()
            if len(sites_in_cell) < 2:
                continue

            # Step 1: per-site stats for every site in the cell.
            per_site = {}
            for site in sites_in_cell:
                site_vals = grp.loc[
                    grp['site'] == site, 'value_numeric'
                ].to_numpy(dtype=float)
                per_site[site] = self._per_site_stats(
                    site_vals, floor, ceil)

            # Step 2: eligibility split — reference-eligible sites
            # have n >= min_n_per_cell; small sites (< min) emit
            # audit rows only.
            eligible_sites = [
                s for s, st in per_site.items()
                if st['n'] >= self.min_n_per_cell
            ]
            # Need at least 3 eligible peer sites for any single
            # site to have a 2+ -peer LOO reference. With K=3 eligible
            # total, each gets a 2-peer reference; with K=2 the LOO
            # reference is one point (no MAD).
            if len(eligible_sites) < 3:
                # All sites in this cell get audit-only rows when
                # they had any data. There's nothing comparative we
                # can stand behind.
                for site, site_stats in per_site.items():
                    if (len(audit_records)
                            < self.max_audit_rows):
                        audit_records.append(
                            self._build_base_record(
                                grp, site, network, variable,
                                cohort, timepoint, form_map,
                                site_stats,
                                ref_stats={
                                    'median': float('nan'),
                                    'iqr': float('nan'),
                                    'mad': float('nan'),
                                    'p_at_floor': float('nan'),
                                    'p_at_ceiling': float('nan'),
                                    'n': 0,
                                },
                                z_median=float('nan'),
                                z_iqr=float('nan'),
                                delta_floor=float('nan'),
                                delta_ceil=float('nan'),
                                severity=float('nan'),
                                is_audit_only=True,
                                error_message=(
                                    f"AUDIT: only "
                                    f"{len(eligible_sites)} "
                                    f"eligible peer site(s) in cell "
                                    f"(min 3 required for LOO "
                                    f"reference). No flags emitted."),
                                today=today,
                            ))
                continue

            # Step 3: for each site, compute LOO reference excluding
            # itself (only when this site itself is eligible).
            for site, site_stats in per_site.items():
                if site_stats['n'] < self.min_n_per_cell:
                    self._counters[
                        'cells_skipped_insufficient_n'] += 1
                    if (len(audit_records)
                            < self.max_audit_rows):
                        audit_records.append(
                            self._build_base_record(
                                grp, site, network, variable,
                                cohort, timepoint, form_map,
                                site_stats,
                                ref_stats={
                                    'median': float('nan'),
                                    'iqr': float('nan'),
                                    'mad': float('nan'),
                                    'p_at_floor': float('nan'),
                                    'p_at_ceiling': float('nan'),
                                    'n': 0,
                                },
                                z_median=float('nan'),
                                z_iqr=float('nan'),
                                delta_floor=float('nan'),
                                delta_ceil=float('nan'),
                                severity=float('nan'),
                                is_audit_only=True,
                                error_message=(
                                    f"AUDIT: cell n={site_stats['n']}"
                                    f" < min_n_per_cell="
                                    f"{self.min_n_per_cell}; site "
                                    "skipped from flag computation."),
                                today=today,
                            ))
                    continue

                # LOO at site-median level: peer set excludes self.
                other_stats = [
                    per_site[s] for s in eligible_sites
                    if s != site
                ]
                ref = self._loo_reference(
                    other_stats, site_n=site_stats['n'])
                if ref is None or ref['median_sd'] == 0:
                    self._counters[
                        'cells_skipped_degenerate_ref'] += 1
                    continue
                ref_stats = {
                    'median': ref['median'],
                    'iqr': ref['iqr'],
                    'mad': ref['mad'],
                    'p_at_floor': ref['p_at_floor'],
                    'p_at_ceiling': ref['p_at_ceiling'],
                    'n': len(other_stats),
                }

                z_median = (
                    (site_stats['median'] - ref['median'])
                    / ref['median_sd']
                    if ref['median_sd'] > 0 else 0.0)
                z_iqr = (
                    (site_stats['iqr'] - ref['iqr'])
                    / ref['iqr_sd']
                    if ref['iqr_sd'] > 0 else 0.0)
                delta_floor = (
                    site_stats['p_at_floor']
                    - ref['p_at_floor']
                    if not (np.isnan(site_stats['p_at_floor'])
                            or np.isnan(ref['p_at_floor']))
                    else float('nan'))
                delta_ceil = (
                    site_stats['p_at_ceiling']
                    - ref['p_at_ceiling']
                    if not (np.isnan(site_stats['p_at_ceiling'])
                            or np.isnan(ref['p_at_ceiling']))
                    else float('nan'))

                triggers = []
                if abs(z_median) > self.z_threshold_median:
                    triggers.append(
                        f"median_z={z_median:.2f}")
                if abs(z_iqr) > self.z_threshold_iqr:
                    triggers.append(f"iqr_z={z_iqr:.2f}")
                if (not np.isnan(delta_floor)
                        and abs(delta_floor)
                        > self.delta_floor_ceiling_threshold):
                    triggers.append(
                        f"delta_p_floor={delta_floor:+.2f}")
                if (not np.isnan(delta_ceil)
                        and abs(delta_ceil)
                        > self.delta_floor_ceiling_threshold):
                    triggers.append(
                        f"delta_p_ceil={delta_ceil:+.2f}")
                if not triggers:
                    continue

                severity = float(max(
                    abs(z_median), abs(z_iqr),
                    abs(delta_floor) * 10
                    if not np.isnan(delta_floor) else 0,
                    abs(delta_ceil) * 10
                    if not np.isnan(delta_ceil) else 0,
                ))
                error_message = (
                    f"Site {site} {variable} {cohort or '(any)'}/"
                    f"{timepoint}: median={site_stats['median']:.4g} "
                    f"vs peer-site median={ref_stats['median']:.4g} "
                    f"(n_site={site_stats['n']}, "
                    f"n_peer={ref_stats['n']}); triggers: "
                    f"{', '.join(triggers)}")

                flag_records.append(
                    self._build_base_record(
                        grp, site, network, variable, cohort,
                        timepoint, form_map, site_stats, ref_stats,
                        z_median=z_median, z_iqr=z_iqr,
                        delta_floor=delta_floor,
                        delta_ceil=delta_ceil,
                        severity=severity,
                        is_audit_only=False,
                        error_message=error_message,
                        today=today,
                    ))
                self._counters['flags_emitted'] += 1

        all_records = flag_records + audit_records
        self._counters['audit_rows_emitted'] = len(audit_records)
        if not all_records:
            return self._empty_output_df(), pd.DataFrame()

        df = pd.DataFrame(all_records)
        # Stable sort: severity desc (NaN audit rows at the bottom),
        # then site, variable, cohort, timepoint for determinism.
        df = df.sort_values(
            by=['severity_score', 'site',
                'variable', 'cohort', 'timepoint'],
            ascending=[False, True, True, True, True],
            na_position='last').reset_index(drop=True)
        if (self.max_flags_to_write is not None
                and (df['is_audit_only'] == False).sum()
                > self.max_flags_to_write):
            keep_flags = df[df['is_audit_only'] == False].head(
                self.max_flags_to_write)
            keep_audit = df[df['is_audit_only']]
            df = pd.concat([keep_flags, keep_audit],
                           ignore_index=True)
        return self._format_output(df), df

    def _build_base_record(
        self, grp, site, network, variable, cohort, timepoint,
        form_map, site_stats, ref_stats,
        z_median, z_iqr, delta_floor, delta_ceil,
        severity, is_audit_only, error_message, today,
    ) -> dict:
        source_form = form_map.get(variable, '')
        # Evidence_timepoints metadata — site-distribution drift is
        # a per-tp comparison, so evidence is the single tp being
        # tested.
        evidence_tp = str(timepoint)
        return {
            'subjectid': '',  # site-level flag, no single subject
            'network': str(network),
            'timepoint': str(timepoint),
            'variable': str(variable),
            'source_form': str(source_form or ''),
            'value': f"site_n={site_stats['n']}",
            'value_numeric': float('nan'),
            'expected_value': ref_stats['median'],
            'observed_value': site_stats['median'],
            'algorithm': 'site_distribution_drift',
            'metric_name': 'robust_z_median',
            'metric_value': float(z_median),
            'severity_score': severity,
            'threshold': float(self.z_threshold_median),
            'error_message': error_message,
            'dates_detected': today,
            'site': str(site),
            'cohort': str(cohort),
            'n_subjects': int(site_stats['n']),
            'site_median': float(site_stats['median']),
            'site_iqr': float(site_stats['iqr']),
            'site_mad': float(site_stats['mad']),
            'ref_median': float(ref_stats['median']),
            'ref_iqr': float(ref_stats['iqr']),
            'ref_mad': float(ref_stats['mad']),
            'site_p_at_floor': float(site_stats['p_at_floor']),
            'site_p_at_ceiling': float(site_stats['p_at_ceiling']),
            'ref_p_at_floor': float(ref_stats['p_at_floor']),
            'ref_p_at_ceiling': float(ref_stats['p_at_ceiling']),
            'z_median': float(z_median),
            'z_iqr': float(z_iqr),
            'delta_p_at_floor': float(delta_floor),
            'delta_p_at_ceiling': float(delta_ceil),
            'is_audit_only': bool(is_audit_only),
            'evidence_timepoints': evidence_tp,
            'evidence_max_timepoint': evidence_tp,
        }

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Severity_normalized via discovery z-score convention.
        out['severity_normalized'] = out['severity_score'].apply(
            normalize_zscore_severity).astype(float)
        # Tracker-compatible aliases — same values, alternate names.
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = out['variable']
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
            'n_subjects': pd.Series(dtype='int64'),
            'site_median': pd.Series(dtype='float64'),
            'site_iqr': pd.Series(dtype='float64'),
            'site_mad': pd.Series(dtype='float64'),
            'ref_median': pd.Series(dtype='float64'),
            'ref_iqr': pd.Series(dtype='float64'),
            'ref_mad': pd.Series(dtype='float64'),
            'site_p_at_floor': pd.Series(dtype='float64'),
            'site_p_at_ceiling': pd.Series(dtype='float64'),
            'ref_p_at_floor': pd.Series(dtype='float64'),
            'ref_p_at_ceiling': pd.Series(dtype='float64'),
            'z_median': pd.Series(dtype='float64'),
            'z_iqr': pd.Series(dtype='float64'),
            'delta_p_at_floor': pd.Series(dtype='float64'),
            'delta_p_at_ceiling': pd.Series(dtype='float64'),
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
            f"{out_dir}/site_distribution_drift.parquet")
        atomic_write_parquet(df, final_path)
        if len(df) == 0:
            print(
                f"[site_distribution_drift] no candidates → "
                f"{final_path}")
        else:
            print(
                f"[site_distribution_drift] {len(df)} rows "
                f"({(df['is_audit_only'] == False).sum()} flags, "
                f"{(df['is_audit_only'] == True).sum()} audit) → "
                f"{final_path}")

    def _print_summary(self):
        c = self._counters
        rows = [
            ("input rows", c['input_rows']),
            ("rows excluded missing-code/NaN",
             c['rows_excluded_missing_code']),
            ("rows excluded non-longitudinal tp",
             c['rows_excluded_non_longitudinal_tp']),
            ("rows excluded by form pattern",
             c['rows_excluded_by_form_pattern']),
            ("rows excluded by variable pattern",
             c['rows_excluded_by_variable_pattern']),
            ("valid scoring rows", c['valid_scoring_rows']),
            ("(net,var,cohort,tp) cells considered",
             c['cells_considered']),
            ("cells skipped — insufficient n at site",
             c['cells_skipped_insufficient_n']),
            ("cells skipped — degenerate reference",
             c['cells_skipped_degenerate_ref']),
            ("flags emitted", c['flags_emitted']),
            ("audit rows emitted", c['audit_rows_emitted']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[site_distribution_drift] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")

    def run(self):
        self._counters = self._fresh_counters()
        long_df = self._load_inputs()
        self._counters['input_rows'] = len(long_df)
        formatted, _raw = self._compute_candidates(long_df)
        self._write_output(formatted)
        self._print_summary()
