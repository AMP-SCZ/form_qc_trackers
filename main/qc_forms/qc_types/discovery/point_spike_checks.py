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
Point-spike candidate detector (Layer 2 / discovery, default-off).

Targets the visual "line plot outlier" intuition: at a single visit a
subject sits far from the rest of the cohort, then returns to the
cohort baseline at the adjacent visit(s). Existing N1 (within-subject
median residual) catches subjects who deviate from their OWN history
but cannot tell whether the deviating value is also far from the
cohort at that visit. longitudinal_delta catches a single big STEP
but treats the up-step and the down-step that bracket a spike as two
unrelated rows. This detector composes both views: cohort-at-tp
z-score AND neighbor-tp quiescence.

INPUT (does NOT modify):
  dependencies/multi_tp_{network}_numeric_long.parquet
    — same long parquet as N1 / copy-forward / delta. Required
    columns: subjectid, network, timepoint, variable, source_form,
    value, value_numeric, is_missing_code.

OUTPUT (atomic, tmp + os.replace):
  combined_outputs/discovery/point_spike_candidates.parquet
    — single combined parquet (all networks). Same audit + tracker
    alias columns as the other discovery detectors. v1 does NOT
    append to the main tracker.

ALGORITHM (deterministic, no ML):
  1. Filter scoring-eligible rows: value_numeric.notna() &
     ~is_missing_code.
  2. Restrict to canonical longitudinal timepoints
     (Utils.create_timepoint_list()); floating / conversion excluded.
  3. Drop rows whose source_form matches any
     excluded_source_form_patterns (default: digital_biomarkers_*).
     These passive-monitoring forms have device-driven variance an
     order of magnitude above RA-entered scales and otherwise
     dominate the output.
  4. For each (network, variable, timepoint):
     a. Skip if n < min_observations_per_tp (default 20).
     b. Compute cohort_median + 1.4826*MAD across all subjects'
        values at that tp.
     c. Skip if scale is 0 or NaN.
     d. Assign each observation a cohort_z =
        (value − cohort_median) / scale.
  5. For each (network, variable, subjectid) trajectory, sort by
     canonical tp index. For each observation with
     |cohort_z| ≥ high_threshold:
       - Look up the same subject's observations at the
         canonically-adjacent prior and next timepoints (tp_idx−1 and
         tp_idx+1). Use only observations that were themselves
         scored (i.e. their (network, variable, tp) cohort pool was
         non-degenerate and met min_observations_per_tp).
       - "Available neighbors" = the subset of {prev, next} that
         exists and is scored.
       - Skip if 0 available neighbors AND require_at_least_one_neighbor
         is true (default).
       - Otherwise: ALL available neighbors must have |cohort_z|
         ≤ low_threshold (default 2.0). Otherwise skip.
       - severity = |current_z| − max(available neighbor |z|, or 0).
         Higher score = sharper spike against quieter neighbors.
  6. Sort by severity descending. Apply max_flags_to_write cap.

CONFIG:
  discovery.point_spike.{
    enabled: false,
    networks: ["PRONET","PRESCIENT"],
    min_observations_per_tp: 20,
    high_threshold: 4.0,
    low_threshold: 2.0,
    require_at_least_one_neighbor: true,
    excluded_source_form_patterns: ["digital_biomarkers_"],
    max_flags_to_write: null
  }

FAILURE MODES (no silent fallbacks):
  - any configured network's long parquet missing → FileNotFoundError
  - long parquet exists but lacks any required column → RuntimeError
  - data filters to zero scoring rows or zero spikes → STILL write a
    valid empty parquet with declared dtypes

CONSTRAINTS HONORED:
  - Reads only the long-format parquets produced by Step 1.
  - Writes only the single candidates parquet under
    combined_outputs/discovery/.
  - Does NOT append to combined_qc_flags.parquet.
  - Does NOT import qc_forms_main, form_check, or generate_reports.
  - Does NOT inherit from FormCheck.
  - Atomic write via PID-stamped tmp + os.replace.
"""


_NON_LONGITUDINAL_TIMEPOINTS = frozenset({'floating', 'conversion'})

# Module-level re-exports preserved for backward compatibility with
# any caller that imported these symbols from this module.
_DEFAULT_EXCLUDED_FORM_PATTERNS = DEFAULT_EXCLUDED_FORM_PATTERNS
_DEFAULT_EXCLUDED_VARIABLE_PATTERNS = DEFAULT_EXCLUDED_VARIABLE_PATTERNS
_is_excluded_form = is_excluded_form
_is_excluded_variable = is_excluded_variable


class PointSpikeChecks:
    """
    Discovery-tier point-spike detector. Flags observations whose
    cohort robust-z at the tp is large while the same subject's
    adjacent-tp observations are quiet.
    """

    OUTPUT_COLUMNS = [
        # Audit columns (16, mirror N1).
        'subjectid', 'network', 'timepoint', 'variable',
        'source_form', 'value', 'value_numeric',
        'expected_value', 'observed_value',
        'algorithm', 'metric_name', 'metric_value',
        'severity_score', 'severity_normalized', 'threshold',
        'error_message', 'dates_detected',
        # Spike-specific (5).
        'cohort_median', 'cohort_mad_scale',
        'prev_neighbor_z', 'next_neighbor_z',
        'num_neighbors_considered',
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

        ps_cfg = self.config_info.get(
            'discovery', {}).get('point_spike', {})
        self.networks = list(
            ps_cfg.get('networks', ['PRONET', 'PRESCIENT']))
        self.min_observations_per_tp = int(
            ps_cfg.get('min_observations_per_tp', 20))
        self.high_threshold = float(
            ps_cfg.get('high_threshold', 4.0))
        self.low_threshold = float(
            ps_cfg.get('low_threshold', 2.0))
        self.require_at_least_one_neighbor = bool(
            ps_cfg.get('require_at_least_one_neighbor', True))
        self.excluded_source_form_patterns = tuple(
            ps_cfg.get(
                'excluded_source_form_patterns',
                list(_DEFAULT_EXCLUDED_FORM_PATTERNS)))
        self.excluded_variable_patterns = tuple(
            ps_cfg.get(
                'excluded_variable_patterns',
                list(_DEFAULT_EXCLUDED_VARIABLE_PATTERNS)))
        cap = ps_cfg.get('max_flags_to_write', None)
        self.max_flags_to_write = (
            int(cap) if cap is not None else None)
        # Per-variable cap applied AFTER severity sort, BEFORE the
        # global max_flags_to_write cap. Stops one noisy variable from
        # monopolizing the top of the sheet. Set to null to disable.
        pv_cap = ps_cfg.get('max_flags_per_variable', 25)
        self.max_flags_per_variable = (
            int(pv_cap) if pv_cap is not None else None)

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
            'tp_groups_considered': 0,
            'tp_groups_skipped_insufficient_obs': 0,
            'tp_groups_skipped_degenerate_mad': 0,
            'scored_observations': 0,
            'candidate_high_z_points': 0,
            'candidates_dropped_unscored_neighbor': 0,
            'candidates_dropped_nan_z': 0,
            'spikes_passing_neighbor_filter': 0,
            'spikes_dropped_by_per_variable_cap': 0,
            'spikes_written': 0,
        }

    def __call__(self):
        return self.run()

    def run(self):
        self._counters = self._fresh_counters()
        long_df = self._load_inputs()
        self._counters['input_rows'] = len(long_df)
        flagged = self._compute_candidates(long_df)
        self._counters['spikes_written'] = len(flagged)
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
                    f"{network}: {path}. Run "
                    f"process_variables.produce_multi_tp_long first."
                )
            df = pd.read_parquet(path)
            missing_cols = [
                c for c in self.REQUIRED_INPUT_COLUMNS
                if c not in df.columns
            ]
            if missing_cols:
                raise RuntimeError(
                    f"FATAL: long-format parquet at {path} is "
                    f"missing required columns: {missing_cols}. "
                    f"Schema mismatch with the producer — "
                    f"regenerate the long parquet."
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

        # Skip the apply-based filters when scoring is empty: an
        # empty Series.apply returns a float64 (empty) series, which
        # pandas then treats as a column selector — collapsing the
        # frame to zero columns instead of zero rows.
        if self.excluded_source_form_patterns and len(scoring) > 0:
            before = len(scoring)
            excluded_mask = scoring['source_form'].apply(
                lambda f: _is_excluded_form(
                    f, self.excluded_source_form_patterns))
            scoring = scoring[~excluded_mask].copy()
            self._counters['rows_excluded_by_form_pattern'] = (
                before - len(scoring))

        if self.excluded_variable_patterns and len(scoring) > 0:
            before = len(scoring)
            excluded_var_mask = scoring['variable'].apply(
                lambda v: _is_excluded_variable(
                    v, self.excluded_variable_patterns))
            scoring = scoring[~excluded_var_mask].copy()
            self._counters['rows_excluded_by_variable_pattern'] = (
                before - len(scoring))

        self._counters['valid_scoring_rows'] = len(scoring)
        if len(scoring) == 0:
            return self._empty_output_df()

        # Step 1: compute cohort z per (network, variable, tp).
        scored = self._assign_cohort_z(scoring)
        if len(scored) == 0:
            return self._empty_output_df()
        self._counters['scored_observations'] = len(scored)

        # Step 2: per (network, variable, subjectid), find spike points
        # whose neighbors are quiet. `scoring` is forwarded so we can
        # distinguish "no observation at adjacent tp" from "observation
        # exists but cohort pool was filtered out" — the latter is
        # treated as 'unknown loud' rather than admitted as 'absent'.
        flagged = self._select_spikes(scored, scoring)
        if len(flagged) == 0:
            return self._empty_output_df()

        flagged = flagged.sort_values(
            'severity_score', ascending=False).reset_index(drop=True)

        # Per-variable cap stops a single noisy variable from
        # monopolizing the sheet. Applied BEFORE the global cap so
        # other variables get representation.
        if self.max_flags_per_variable is not None:
            before = len(flagged)
            flagged = (
                flagged.groupby('variable', sort=False)
                .head(self.max_flags_per_variable)
                .reset_index(drop=True))
            self._counters['spikes_dropped_by_per_variable_cap'] = (
                before - len(flagged))

        if (self.max_flags_to_write is not None
                and len(flagged) > self.max_flags_to_write):
            print(
                f"[point_spike_checks] WARNING: "
                f"{len(flagged)} spikes exceeded threshold "
                f"{self.high_threshold}; capping output to top "
                f"{self.max_flags_to_write} by severity_score "
                f"(config max_flags_to_write). Increase the cap or "
                f"set to null to see all candidates."
            )
            flagged = flagged.head(self.max_flags_to_write).copy()

        return self._format_output(flagged)

    def _assign_cohort_z(
        self, scoring: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute per-(network, variable, timepoint) robust-z for each
        observation. Returns scoring rows enriched with cohort_z,
        cohort_median, cohort_mad_scale; drops rows whose pool was
        skipped for insufficient obs or degenerate MAD.
        """
        pieces = []
        for (_net, _var, _tp), grp in scoring.groupby(
                ['network', 'variable', 'timepoint'], sort=False):
            self._counters['tp_groups_considered'] += 1
            if len(grp) < self.min_observations_per_tp:
                self._counters[
                    'tp_groups_skipped_insufficient_obs'] += 1
                continue
            vals = grp['value_numeric'].to_numpy(dtype=float)
            center = float(np.median(vals))
            mad = float(np.median(np.abs(vals - center)))
            scale = mad * 1.4826
            if pd.isna(scale) or scale == 0:
                self._counters[
                    'tp_groups_skipped_degenerate_mad'] += 1
                continue
            g = grp.copy()
            g['cohort_median'] = center
            g['cohort_mad_scale'] = scale
            g['cohort_z'] = (g['value_numeric'] - center) / scale
            pieces.append(g)
        if not pieces:
            return pd.DataFrame()
        return pd.concat(pieces, ignore_index=True)

    def _select_spikes(
        self, scored: pd.DataFrame, scoring: pd.DataFrame
    ) -> pd.DataFrame:
        """
        For each (network, variable, subjectid) trajectory: identify
        candidate spikes (|z| >= high_threshold), then require
        canonically-adjacent neighbors (when present in raw data) to
        have |z| <= low_threshold.

        Neighbor semantics:
          - "available scored neighbor" = an obs at tp_idx±1 whose
            (network, variable, tp) cohort pool was scored. Its |z|
            must be <= low_threshold.
          - "raw-but-unscored neighbor" = an obs exists at tp_idx±1
            in `scoring` (i.e. it passed value/missing/form/variable
            filters) but its cohort pool was dropped for insufficient
            obs or degenerate MAD. We can't verify quietness, so the
            candidate is REJECTED — "unknown is loud".
          - "absent neighbor" = the subject has no observation at
            tp_idx±1 at all. Counted toward num_neighbors_considered=0;
            the spike fails when require_at_least_one_neighbor is True.

        Severity = |cohort_z|. The neighbor filter is a GATE only;
        once passed, ranking is by absolute cohort outlier-ness so
        the top-of-sheet shows the visually most extreme points.
        """
        # Producer invariant: one row per (network, variable, subjectid,
        # timepoint) in the long parquet. If violated, the neighbor
        # lookup is nondeterministic — fail loud rather than silently
        # dedupe so the upstream producer bug surfaces.
        dup_mask = scored.duplicated(
            subset=['network', 'variable', 'subjectid', 'timepoint'],
            keep=False)
        if dup_mask.any():
            dup_sample = (
                scored.loc[
                    dup_mask,
                    ['network', 'variable', 'subjectid', 'timepoint']
                ].head(5).to_dict('records'))
            raise RuntimeError(
                "FATAL: duplicate (network, variable, subjectid, "
                f"timepoint) rows in scored long-format data "
                f"({int(dup_mask.sum())} duplicate rows). This "
                "indicates a producer bug in "
                "multi_tp_*_numeric_long.parquet — neighbor lookup "
                "would be nondeterministic. Sample dupes: "
                f"{dup_sample}"
            )

        # Build raw-observation tp_idx set per (network, variable,
        # subject) BEFORE the cohort-z pool filter. Used to distinguish
        # "no observation at adjacent visit" from "obs exists but its
        # pool was dropped" — see semantics in docstring above.
        scoring_idx = scoring.assign(
            tp_idx=scoring['timepoint'].map(self._tp_index))
        raw_tp_by_subject = {}
        for (net, var, subj), grp in scoring_idx.groupby(
                ['network', 'variable', 'subjectid'], sort=False):
            raw_tp_by_subject[(net, var, subj)] = set(
                int(x) for x in grp['tp_idx'].tolist()
                if not pd.isna(x))

        scored = scored.assign(
            tp_idx=scored['timepoint'].map(self._tp_index))
        records = []
        for (net, var, subj), grp in scored.groupby(
                ['network', 'variable', 'subjectid'], sort=False):
            if len(grp) == 0:
                continue
            g = grp.sort_values('tp_idx').reset_index(drop=True)
            idx_to_pos = {
                int(g.at[i, 'tp_idx']): i for i in range(len(g))
            }
            raw_idxs = raw_tp_by_subject.get((net, var, subj), set())
            abs_z = g['cohort_z'].abs().to_numpy()
            for i, row in g.iterrows():
                # NaN cohort_z would silently fall into the spike
                # branch because NaN<threshold evaluates False.
                # Drop explicitly with a counter so the diagnostic
                # summary surfaces any producer drift.
                if pd.isna(abs_z[i]):
                    self._counters['candidates_dropped_nan_z'] += 1
                    continue
                if abs_z[i] < self.high_threshold:
                    continue
                self._counters['candidate_high_z_points'] += 1
                cur_idx = int(row['tp_idx'])
                neighbor_zs = []
                prev_z = next_z = float('nan')
                unscored_neighbor = False
                for offset, label in ((-1, 'prev'), (1, 'next')):
                    n_idx = cur_idx + offset
                    pos = idx_to_pos.get(n_idx)
                    if pos is not None:
                        z = float(g.iloc[pos]['cohort_z'])
                        if pd.isna(z):
                            # A scored neighbor with NaN cohort_z is
                            # uninterpretable — same semantic as a raw
                            # obs whose pool was dropped: "unknown is
                            # loud", reject the candidate.
                            unscored_neighbor = True
                            continue
                        neighbor_zs.append(abs(z))
                        if label == 'prev':
                            prev_z = z
                        else:
                            next_z = z
                    elif n_idx in raw_idxs:
                        # Obs exists at adjacent visit but its cohort
                        # pool was dropped — we cannot prove it's
                        # quiet, so the candidate fails the gate.
                        unscored_neighbor = True

                if unscored_neighbor:
                    self._counters[
                        'candidates_dropped_unscored_neighbor'] += 1
                    continue

                if (len(neighbor_zs) == 0
                        and self.require_at_least_one_neighbor):
                    continue
                if any(z > self.low_threshold for z in neighbor_zs):
                    continue
                self._counters[
                    'spikes_passing_neighbor_filter'] += 1
                # Severity = |z|. The neighbor filter above is a gate;
                # ranking is by raw cohort outlier-ness so the user
                # sees the most extreme points first.
                severity = float(abs_z[i])
                records.append({
                    'subjectid': str(row['subjectid']),
                    'network': str(row['network']),
                    'timepoint': str(row['timepoint']),
                    'variable': str(row['variable']),
                    'source_form': str(row['source_form'] or ''),
                    'value': row['value'],
                    'value_numeric': float(row['value_numeric']),
                    'cohort_median': float(row['cohort_median']),
                    'cohort_mad_scale': float(row['cohort_mad_scale']),
                    'cohort_z': float(row['cohort_z']),
                    'prev_neighbor_z': prev_z,
                    'next_neighbor_z': next_z,
                    'num_neighbors_considered': len(neighbor_zs),
                    'severity_score': severity,
                })
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def _format_output(
        self, flagged: pd.DataFrame
    ) -> pd.DataFrame:
        today = str(datetime.today().date())
        out = pd.DataFrame()
        out['subjectid'] = flagged['subjectid'].astype(str)
        out['network'] = flagged['network'].astype(str)
        out['timepoint'] = flagged['timepoint'].astype(str)
        out['variable'] = flagged['variable'].astype(str)
        out['source_form'] = flagged['source_form'].astype(str)
        out['value'] = flagged['value'].astype(str)
        out['value_numeric'] = flagged['value_numeric'].astype(float)
        out['expected_value'] = flagged['cohort_median'].astype(float)
        out['observed_value'] = flagged['value_numeric'].astype(float)
        out['algorithm'] = 'point_spike'
        out['metric_name'] = 'cohort_robust_z'
        out['metric_value'] = flagged['cohort_z'].astype(float)
        out['severity_score'] = flagged['severity_score'].astype(float)
        out['severity_normalized'] = (
            flagged['severity_score'].apply(
                normalize_zscore_severity).astype(float))
        out['threshold'] = float(self.high_threshold)
        out['error_message'] = flagged.apply(
            self._build_error_message, axis=1)
        out['dates_detected'] = today
        out['cohort_median'] = flagged['cohort_median'].astype(float)
        out['cohort_mad_scale'] = (
            flagged['cohort_mad_scale'].astype(float))
        out['prev_neighbor_z'] = (
            flagged['prev_neighbor_z'].astype(float))
        out['next_neighbor_z'] = (
            flagged['next_neighbor_z'].astype(float))
        out['num_neighbors_considered'] = (
            flagged['num_neighbors_considered'].astype('int64'))
        out['subject'] = out['subjectid']
        out['displayed_variable'] = out['variable']
        out['displayed_timepoint'] = out['timepoint']
        out['affected_variables'] = out['variable']
        out['affected_timepoints'] = out['timepoint']
        out['affected_forms'] = out['source_form']
        out['displayed_form'] = out['source_form']
        return out[self.OUTPUT_COLUMNS]

    def _build_error_message(self, row) -> str:
        nz = []
        if not pd.isna(row['prev_neighbor_z']):
            nz.append(f"prev_z={row['prev_neighbor_z']:.2g}")
        if not pd.isna(row['next_neighbor_z']):
            nz.append(f"next_z={row['next_neighbor_z']:.2g}")
        nz_str = ', '.join(nz) if nz else 'no neighbors'
        return (
            f"{row['variable']}: cohort outlier at {row['timepoint']} "
            f"(value={row['value_numeric']:.6g}, cohort median="
            f"{row['cohort_median']:.6g}, z={row['cohort_z']:.2g}) "
            f"while adjacent visits quiet ({nz_str}); spike severity="
            f"{row['severity_score']:.2g} "
            f"(threshold |z|>={self.high_threshold}, neighbor "
            f"|z|<={self.low_threshold})"
        )

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
            'cohort_median': pd.Series(dtype='float64'),
            'cohort_mad_scale': pd.Series(dtype='float64'),
            'prev_neighbor_z': pd.Series(dtype='float64'),
            'next_neighbor_z': pd.Series(dtype='float64'),
            'num_neighbors_considered': pd.Series(dtype='int64'),
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
            f"{out_dir}/point_spike_candidates.parquet")
        self._atomic_write_parquet(df, final_path)

        if len(df) == 0:
            print(
                f"[point_spike_checks] no point-spike candidates "
                f"above |z|>={self.high_threshold} with quiet "
                f"neighbors (|z|<={self.low_threshold}); wrote empty "
                f"parquet (valid result) → {final_path}"
            )
        else:
            top = df.head(5)[
                ['subjectid', 'variable', 'timepoint',
                 'severity_score']]
            print(
                f"[point_spike_checks] {len(df)} point-spike "
                f"candidates → {final_path}"
            )
            print(
                f"Top 5 by severity:\n"
                f"{top.to_string(index=False)}"
            )

    def _print_summary(self):
        c = self._counters
        rows = [
            ("input rows", c['input_rows']),
            ("rows excluded by form pattern",
             c['rows_excluded_by_form_pattern']),
            ("rows excluded by variable pattern",
             c['rows_excluded_by_variable_pattern']),
            ("valid scoring rows", c['valid_scoring_rows']),
            ("(net,var,tp) groups considered",
             c['tp_groups_considered']),
            ("(net,var,tp) groups skipped — insufficient obs",
             c['tp_groups_skipped_insufficient_obs']),
            ("(net,var,tp) groups skipped — zero/NaN MAD",
             c['tp_groups_skipped_degenerate_mad']),
            ("scored observations", c['scored_observations']),
            ("candidate high-|z| points",
             c['candidate_high_z_points']),
            ("candidates dropped — NaN cohort_z",
             c['candidates_dropped_nan_z']),
            ("candidates dropped — raw neighbor was unscored",
             c['candidates_dropped_unscored_neighbor']),
            ("spikes passing neighbor filter",
             c['spikes_passing_neighbor_filter']),
            ("spikes dropped by per-variable cap",
             c['spikes_dropped_by_per_variable_cap']),
            ("spikes written (post-cap)", c['spikes_written']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[point_spike_checks] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")

    def _atomic_write_parquet(
        self, df: pd.DataFrame, final_path: str
    ):
        atomic_write_parquet(df, final_path)


if __name__ == '__main__':
    PointSpikeChecks()()
