"""
Mixed-effects site trajectory slope outlier detector
(Layer 2 / discovery, default-off).

PROBLEM:
  Detect sites whose longitudinal slope on a variable is systematically
  different from the network — e.g. a site whose mean BPRS is improving
  faster than peers, or whose GAF deteriorates more steeply over time.
  Captures rater drift over the study, mid-study protocol amendments,
  and onboarding-wave stabilization. Distinct from:
    - Feature 1 (per-tp cross-sectional shift), which catches a static
      offset but not a trajectory difference;
    - the existing longitudinal_anomaly N1 detector, which catches
      per-subject outliers and ignores site structure entirely.

ALGORITHM (deterministic, no ML; uses statsmodels MixedLM):
  Site is derived from `subjectid[:2]`.

  For each (network, variable):
    1. Build long-format frame with columns:
         subjectid, site, cohort, canonical_tp_index, value_numeric
       canonical_tp_index is read from Utils.create_timepoint_list()
       (screening=0, baseline=1, month1=2, ...). Floating / conversion
       rows are excluded.
    2. Restrict to subjects with >= min_obs_per_subject (default 3)
       valid observations on this variable.
    3. Restrict to sites with >= min_subjects_per_site (default 20)
       eligible subjects.
    4. Require at least 3 eligible sites; otherwise skip the variable.
    5. Fit MixedLM:
         value_numeric ~ canonical_tp_index + C(cohort) +
             (canonical_tp_index | site) + (1 | subjectid)
       Cohort fixed effect absorbs HC/CHR/HSC differences in trajectory.
       Subject random intercept absorbs subject-level baseline variance.
       Site random slope on canonical_tp_index is the BLUP of interest.
    6. Extract per-site BLUP of the slope and its standard error;
       standardize: z_site = slope_blup / slope_blup_se. Flag sites
       where |z_site| > z_threshold (default 2.5).

  Bayesian-EB shrinkage in the BLUP protects small sites: their BLUPs
  are pulled toward zero so they don't false-flag from sampling noise.

WHAT THIS CATCHES:
  - Site whose mean trajectory has a systematically different slope
    (e.g. one rater's improvement detection runs ahead of cohort).
  - Mid-study scoring drift at a site (when run on stable variables).
  - Onboarding-wave artifacts (new site's slope is anomalous early).
  - Form-version transitions that affect one site's trajectory.

WHAT IT WILL NOT CATCH:
  - Step changes within a site (a sudden shift at one tp) — fit by
    Feature 1 or longitudinal_delta instead.
  - Non-linear trajectories — the linear-slope formulation only sees
    the linear component.

FALSE-POSITIVE CONTROLS:
  1. Fix cohort as a fixed effect — HC vs CHR vs HSC don't pollute
     the site slope.
  2. Subject random intercept — subject-level outliers do not pull
     the site slope.
  3. Minimum per-site n — small sites are silently skipped (audit
     row only), not flagged.
  4. Minimum eligible site count — variables with <3 contributing
     sites cannot produce comparative flags.
  5. EB shrinkage in MixedLM BLUPs is naturally conservative for
     small sites.

OUTPUT (atomic, tmp + os.replace):
  combined_outputs/discovery/site_trajectory_slope.parquet
"""

import json
import os
import sys
import warnings
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


_NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})


OUTPUT_COLUMNS = [
    # Audit columns.
    'subjectid', 'network', 'timepoint', 'variable',
    'source_form', 'value', 'value_numeric',
    'expected_value', 'observed_value',
    'algorithm', 'metric_name', 'metric_value',
    'severity_score', 'severity_normalized', 'threshold',
    'error_message', 'dates_detected',
    # Detector-specific.
    'site', 'n_subjects', 'n_observations',
    'slope_blup', 'slope_blup_se', 'z_slope',
    'fixed_slope', 'fixed_slope_se',
    'n_eligible_sites', 'is_audit_only',
    # Evidence timepoints.
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


class SiteTrajectorySlopeChecks:
    """
    Mixed-effects site random-slope outlier detector.
    Default-off via config.discovery.site_trajectory_slope.enabled.
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
            'discovery', {}).get('site_trajectory_slope', {})
        self.networks = list(
            cfg.get('networks', ['PRONET', 'PRESCIENT']))
        self.min_obs_per_subject = int(
            cfg.get('min_obs_per_subject', 3))
        self.min_subjects_per_site = int(
            cfg.get('min_subjects_per_site', 20))
        self.min_eligible_sites = int(
            cfg.get('min_eligible_sites', 3))
        self.z_threshold = float(cfg.get('z_threshold', 2.5))
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
        self.max_audit_rows = int(
            cfg.get('max_audit_rows', 5000))
        # Sprint 1 P0-5: variables_allowlist is REQUIRED unless
        # `allow_unbounded` is explicitly true.
        #
        # MixedLM fits ~seconds-to-minutes per (network, variable);
        # at AMPSCZ V=2000 × 2 networks = 4000 fits per run, which
        # is hours-to-days wall-time. The previous default
        # (allowlist=[] meaning "run on every variable") behaved
        # opposite to a safe default. Fail loudly so an operator
        # cannot accidentally launch a multi-day job.
        self.variables_allowlist = list(
            cfg.get('variables_allowlist', []))
        self.allow_unbounded = bool(
            cfg.get('allow_unbounded', False))
        if not self.variables_allowlist and not self.allow_unbounded:
            raise RuntimeError(
                "[site_trajectory_slope] FATAL: "
                "discovery.site_trajectory_slope.variables_allowlist "
                "is empty and `allow_unbounded` is not true. The "
                "detector fits a MixedLM per (network, variable) "
                "— running on every numeric variable in the long "
                "parquet is hours-to-days of wall-time at AMPSCZ "
                "scale. Either populate `variables_allowlist` "
                "with the clinical-team-blessed scales (recommended "
                "starter: BPRS/PANSS/GAF/SIPS totals) or set "
                "`allow_unbounded: true` explicitly to confirm an "
                "unbounded run is intended."
            )

        # Canonical timepoint index — required for the model's
        # linear-time covariate.
        try:
            self._tp_order = list(
                self.utils.create_timepoint_list())
        except (AttributeError, TypeError):
            self._tp_order = []
        self._tp_index = {
            tp: i for i, tp in enumerate(self._tp_order)
        }

        self._counters = self._fresh_counters()

    def _fresh_counters(self) -> dict:
        return {
            'input_rows': 0,
            'valid_scoring_rows': 0,
            'variables_considered': 0,
            'variables_skipped_too_few_sites': 0,
            'variables_skipped_mixedlm_failed': 0,
            'sites_audited_small_n': 0,
            'flags_emitted': 0,
            'audit_rows_emitted': 0,
        }

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

    def _fit_mixedlm(self, df: pd.DataFrame):
        """
        Fit value_numeric ~ canonical_tp_index + C(cohort)
            with random slope on canonical_tp_index by site
            and random intercept by subject.

        Returns (fitted_model_or_None, error_or_None). Failures are
        caught so the whole detector doesn't abort on one
        ill-conditioned variable group.
        """
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
        except ImportError:
            return None, (
                "statsmodels not available; install statsmodels to "
                "use site_trajectory_slope detector")
        # Drop near-constant variables (MixedLM will choke). Use the
        # within-site std spread as a proxy.
        if df['value_numeric'].nunique() < 4:
            return None, "value has fewer than 4 unique values"
        # Drop cohort levels with zero observations.
        cohort_counts = df['cohort'].value_counts()
        df = df[df['cohort'].isin(
            cohort_counts[cohort_counts > 0].index)].copy()

        formula = (
            "value_numeric ~ canonical_tp_index + C(cohort)")
        re_formula = "~canonical_tp_index"
        vc_formula = {"subject": "0 + C(subjectid)"}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = smf.mixedlm(
                    formula,
                    data=df,
                    groups=df['site'],
                    re_formula=re_formula,
                    vc_formula=vc_formula,
                )
                # method='lbfgs' is more robust on small samples
                # than the default 'powell' fallback path.
                result = model.fit(
                    method='lbfgs', maxiter=200, disp=False)
                if not result.converged:
                    return None, "mixedlm did not converge"
                return result, None
            except Exception as exc:  # noqa: BLE001
                return None, (
                    f"mixedlm fit failed: "
                    f"{type(exc).__name__}: {exc}")

    def _extract_site_slope_blups(self, result, fit_grp):
        """
        Return a dict site → {slope_blup, slope_blup_se, z_slope}
        from a fitted MixedLM result, plus the BLUPs array.

        Center: median of all site BLUPs (robust to one outlier).

        Scale: the WITHIN-SITE sampling SE of a slope under H0
        (all sites share the same true slope), computed from the
        model's residual variance:
            sampling_se ≈ sqrt(result.scale / median(SS_x_per_site))
        where result.scale is the residual variance after fixed and
        random effects are removed (so it's the noise-only variance,
        not contaminated by between-site slope variability), and
        SS_x_per_site = sum_over_obs_in_site (tp_idx_i - tp_idx̄)².

        Why this rather than MAD-of-BLUPs or sqrt(K)*bse_fixed?
        - MAD-of-BLUPs collapses near zero when K is small (~5) and
          peer BLUPs cluster tightly around the same true slope; any
          peer then false-flags from sampling noise.
        - sqrt(K)*bse_fixed grows when one site is a true outlier
          (because the random-slope variance inflates bse_fixed),
          so the test loses power exactly when there IS a real
          shifted site.
        The within-site sampling SE is stable: it depends only on
        the noise variance (which is well-estimated from K*n_per_site
        observations) and per-site sample size (a design constant
        per run). Outlier sites do not contaminate it. With EB
        shrinkage this is a conservative noise floor.
        """
        out = {}
        re = getattr(result, 'random_effects', {})
        for group, vec in re.items():
            slope = None
            if hasattr(vec, 'index'):
                if 'canonical_tp_index' in vec.index:
                    slope = float(vec['canonical_tp_index'])
                elif len(vec) >= 2:
                    slope = float(vec.iloc[1])
            elif hasattr(vec, '__len__') and len(vec) >= 2:
                slope = float(vec[1])
            if slope is None:
                continue
            out[group] = {'slope_blup': slope}
        blups = np.array(
            [v['slope_blup'] for v in out.values()], dtype=float)
        if len(blups) < 2:
            for site in out:
                out[site]['slope_blup_se'] = 1.0
                out[site]['z_slope'] = 0.0
            return out, blups

        med = float(np.median(blups))

        # Within-site SS_x for each eligible site.
        # SS_x_per_site = sum_over_observations_in_site
        #                   (tp_idx_i - tp_idx̄_site)²
        # which equals n_obs_per_site * variance(tp_idx) under the
        # design.
        site_ss_x = (
            fit_grp.groupby('site')['canonical_tp_index']
            .apply(lambda s: float(np.sum(
                (s - s.mean()) ** 2)))
            .to_numpy())
        median_ss_x = (
            float(np.median(site_ss_x))
            if len(site_ss_x) > 0 else 1.0)
        residual_var = float(getattr(result, 'scale', 1.0))
        if median_ss_x > 0 and residual_var > 0:
            sampling_se = float(
                np.sqrt(residual_var / median_ss_x))
        else:
            sampling_se = 0.0
        # Belt-and-braces: also include the MAD-based SE so a
        # genuinely-noisy cohort of BLUPs doesn't get over-flagged
        # against a too-tight sampling-only noise floor.
        mad = float(np.median(np.abs(blups - med)))
        mad_se = mad * 1.4826
        se = max(sampling_se, mad_se, 1e-12)
        for site in out:
            out[site]['slope_blup_se'] = se
            out[site]['z_slope'] = (
                (out[site]['slope_blup'] - med) / se)
        return out, blups

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
        # Restrict to tps with a known canonical index.
        scoring['canonical_tp_index'] = (
            scoring['timepoint'].map(self._tp_index))
        scoring = scoring.dropna(
            subset=['canonical_tp_index']).copy()
        scoring['canonical_tp_index'] = (
            scoring['canonical_tp_index'].astype(int))

        if self.excluded_source_form_patterns and len(scoring) > 0:
            mask = scoring['source_form'].apply(
                lambda f: is_excluded_form(
                    f, self.excluded_source_form_patterns))
            scoring = scoring[~mask].copy()
        if self.excluded_variable_patterns and len(scoring) > 0:
            mask = scoring['variable'].apply(
                lambda v: is_excluded_variable(
                    v, self.excluded_variable_patterns))
            scoring = scoring[~mask].copy()
        if self.variables_allowlist:
            scoring = scoring[
                scoring['variable'].isin(self.variables_allowlist)
            ].copy()

        self._counters['valid_scoring_rows'] = len(scoring)
        if len(scoring) == 0:
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

        for (network, variable), grp in scoring.groupby(
                ['network', 'variable'], sort=False):
            self._counters['variables_considered'] += 1

            # Subject-level eligibility.
            subj_obs = grp.groupby(
                'subjectid')['value_numeric'].transform('count')
            grp = grp[subj_obs >= self.min_obs_per_subject].copy()
            if len(grp) == 0:
                continue

            # Site-level eligibility.
            site_subj_counts = (
                grp.groupby('site')['subjectid']
                .nunique())
            eligible_sites = site_subj_counts[
                site_subj_counts >= self.min_subjects_per_site
            ].index.tolist()
            small_sites = site_subj_counts[
                site_subj_counts < self.min_subjects_per_site
            ].index.tolist()

            # Audit rows for small sites (emit one per variable so
            # the operator can see which sites were silently
            # excluded).
            for site in small_sites:
                self._counters['sites_audited_small_n'] += 1
                n_subj_at_site = int(site_subj_counts[site])
                if (len(audit_records)
                        < self.max_audit_rows):
                    audit_records.append(
                        self._build_record(
                            site=site, network=network,
                            variable=variable,
                            n_subjects=n_subj_at_site,
                            n_observations=int(
                                len(grp[grp['site'] == site])),
                            slope_blup=float('nan'),
                            slope_blup_se=float('nan'),
                            z_slope=float('nan'),
                            fixed_slope=float('nan'),
                            fixed_slope_se=float('nan'),
                            n_eligible_sites=len(eligible_sites),
                            severity=float('nan'),
                            is_audit_only=True,
                            error_message=(
                                f"AUDIT: site n_subjects="
                                f"{n_subj_at_site} < "
                                f"min_subjects_per_site="
                                f"{self.min_subjects_per_site}; "
                                "site excluded from mixed-model "
                                "fit."),
                            form_map=form_map,
                            today=today,
                        ))

            if len(eligible_sites) < self.min_eligible_sites:
                self._counters[
                    'variables_skipped_too_few_sites'] += 1
                continue

            fit_grp = grp[grp['site'].isin(eligible_sites)].copy()
            result, err = self._fit_mixedlm(fit_grp)
            if result is None:
                self._counters[
                    'variables_skipped_mixedlm_failed'] += 1
                continue

            site_blups, blups_array = (
                self._extract_site_slope_blups(result, fit_grp))
            fixed_slope = float(
                result.fe_params.get('canonical_tp_index', 0.0))
            fixed_slope_se = float(
                result.bse.get('canonical_tp_index', 0.0))

            # Canonical-tp evidence list — every tp the variable was
            # observed at within this network.
            evidence_tps = sorted(
                set(fit_grp['timepoint'].astype(str)),
                key=lambda t: self._tp_index.get(t, 1_000_000))
            evidence_str = ','.join(evidence_tps)
            evidence_max = evidence_tps[-1] if evidence_tps else ''

            for site, info in site_blups.items():
                z = info['z_slope']
                if abs(z) <= self.z_threshold:
                    continue
                n_subj_at_site = int(site_subj_counts.get(site, 0))
                n_obs_at_site = int(
                    len(fit_grp[fit_grp['site'] == site]))
                severity = float(abs(z))
                error_message = (
                    f"Site {site} variable {variable}: slope BLUP "
                    f"{info['slope_blup']:+.4g} "
                    f"(z={z:+.2f}, threshold |z|>"
                    f"{self.z_threshold}); peer median slope BLUP "
                    f"≈ {float(np.median(blups_array)):+.4g}; "
                    f"fixed-effect cohort slope "
                    f"{fixed_slope:+.4g}/tp.")
                flag_records.append(
                    self._build_record(
                        site=site, network=network,
                        variable=variable,
                        n_subjects=n_subj_at_site,
                        n_observations=n_obs_at_site,
                        slope_blup=info['slope_blup'],
                        slope_blup_se=info['slope_blup_se'],
                        z_slope=z,
                        fixed_slope=fixed_slope,
                        fixed_slope_se=fixed_slope_se,
                        n_eligible_sites=len(eligible_sites),
                        severity=severity,
                        is_audit_only=False,
                        error_message=error_message,
                        form_map=form_map,
                        today=today,
                        evidence_timepoints=evidence_str,
                        evidence_max_timepoint=evidence_max,
                    ))
                self._counters['flags_emitted'] += 1

        all_records = flag_records + audit_records
        self._counters['audit_rows_emitted'] = len(audit_records)
        if not all_records:
            return self._empty_output_df()
        df = pd.DataFrame(all_records)
        df = df.sort_values(
            by=['severity_score', 'site', 'variable'],
            ascending=[False, True, True],
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
        self, site, network, variable,
        n_subjects, n_observations,
        slope_blup, slope_blup_se, z_slope,
        fixed_slope, fixed_slope_se,
        n_eligible_sites,
        severity, is_audit_only, error_message,
        form_map, today,
        evidence_timepoints='', evidence_max_timepoint='',
    ) -> dict:
        source_form = form_map.get(variable, '')
        return {
            'subjectid': '',
            'network': str(network),
            'timepoint': '(trajectory)',
            'variable': str(variable),
            'source_form': str(source_form or ''),
            'value': f"site_n_subjects={n_subjects}",
            'value_numeric': float('nan'),
            'expected_value': float(fixed_slope)
                if not np.isnan(fixed_slope) else float('nan'),
            'observed_value': float(slope_blup)
                if not np.isnan(slope_blup) else float('nan'),
            'algorithm': 'site_trajectory_slope_mixedlm',
            'metric_name': 'site_slope_blup_robust_z',
            'metric_value': float(z_slope)
                if not np.isnan(z_slope) else float('nan'),
            'severity_score': severity,
            'threshold': float(self.z_threshold),
            'error_message': error_message,
            'dates_detected': today,
            'site': str(site),
            'n_subjects': int(n_subjects),
            'n_observations': int(n_observations),
            'slope_blup': float(slope_blup)
                if not np.isnan(slope_blup) else float('nan'),
            'slope_blup_se': float(slope_blup_se)
                if not np.isnan(slope_blup_se) else float('nan'),
            'z_slope': float(z_slope)
                if not np.isnan(z_slope) else float('nan'),
            'fixed_slope': float(fixed_slope)
                if not np.isnan(fixed_slope) else float('nan'),
            'fixed_slope_se': float(fixed_slope_se)
                if not np.isnan(fixed_slope_se) else float('nan'),
            'n_eligible_sites': int(n_eligible_sites),
            'is_audit_only': bool(is_audit_only),
            'evidence_timepoints': str(evidence_timepoints),
            'evidence_max_timepoint': str(evidence_max_timepoint),
        }

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out['severity_normalized'] = out['severity_score'].apply(
            normalize_zscore_severity).astype(float)
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = out['variable']
        out['affected_timepoints'] = out['evidence_timepoints']
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
            'n_subjects': pd.Series(dtype='int64'),
            'n_observations': pd.Series(dtype='int64'),
            'slope_blup': pd.Series(dtype='float64'),
            'slope_blup_se': pd.Series(dtype='float64'),
            'z_slope': pd.Series(dtype='float64'),
            'fixed_slope': pd.Series(dtype='float64'),
            'fixed_slope_se': pd.Series(dtype='float64'),
            'n_eligible_sites': pd.Series(dtype='int64'),
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
            f"{out_dir}/site_trajectory_slope.parquet")
        atomic_write_parquet(df, final_path)
        if len(df) == 0:
            print(
                f"[site_trajectory_slope] no candidates → "
                f"{final_path}")
        else:
            print(
                f"[site_trajectory_slope] {len(df)} rows "
                f"({(df['is_audit_only'] == False).sum()} flags, "
                f"{(df['is_audit_only'] == True).sum()} audit) → "
                f"{final_path}")

    def _print_summary(self):
        c = self._counters
        rows = [
            ("input rows", c['input_rows']),
            ("valid scoring rows", c['valid_scoring_rows']),
            ("variables considered (per-network)",
             c['variables_considered']),
            ("variables skipped — <min eligible sites",
             c['variables_skipped_too_few_sites']),
            ("variables skipped — mixedlm fit failed",
             c['variables_skipped_mixedlm_failed']),
            ("sites audited — small n",
             c['sites_audited_small_n']),
            ("flags emitted", c['flags_emitted']),
            ("audit rows emitted", c['audit_rows_emitted']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[site_trajectory_slope] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")

    def run(self):
        self._counters = self._fresh_counters()
        long_df = self._load_inputs()
        self._counters['input_rows'] = len(long_df)
        df = self._compute_candidates(long_df)
        self._write_output(df)
        self._print_summary()
